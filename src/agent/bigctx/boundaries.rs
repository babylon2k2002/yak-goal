//! Resolving what the model cited into a concrete range.
//!
//! The model names a range as `m00012..m00030` (message refs) or `b3..b7`
//! (block refs, for distilling already-compressed material). This module turns
//! either into a pair of indices into the *visible* list, plus the ids that
//! range actually covers.
//!
//! The subtlety is that a cited ref may no longer be visible: its message was
//! folded into a block. Rather than fail, the resolver *snaps* to whatever
//! represents that material now — the block covering it, or the active ancestor
//! that consumed that block — and reports the snap so the model learns what
//! happened instead of silently compressing the wrong thing.

use std::collections::BTreeMap;

use crate::agent::bigctx::prune::{is_rendered_summary, summary_message_id};
use crate::agent::bigctx::refs::{BLOCKED_REF, ref_to_index};
use crate::agent::bigctx::state::{active_blocks, block_by_id};
use crate::agent::bigctx::types::{CompressionBlock, CompressionState, CoreMessage};

/// What a boundary ref points at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryKind {
    Message,
    Block,
}

/// A parsed boundary ref.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedBoundary {
    pub kind: BoundaryKind,
    pub numeric_id: u64,
    pub raw: String,
}

/// Parse `mNNNNN` or `bN`. Case- and padding-insensitive.
pub fn parse_boundary(reference: &str) -> Option<ParsedBoundary> {
    let normalized = reference.trim().to_ascii_lowercase();
    if let Some(index) = ref_to_index(&normalized) {
        return Some(ParsedBoundary {
            kind: BoundaryKind::Message,
            numeric_id: index,
            raw: normalized,
        });
    }
    let digits = normalized.strip_prefix('b')?;
    if digits.is_empty() || digits.len() > 9 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let numeric_id: u64 = digits.parse().ok()?;
    if numeric_id < 1 {
        return None;
    }
    Some(ParsedBoundary {
        kind: BoundaryKind::Block,
        numeric_id,
        raw: normalized,
    })
}

/// Why a range could not be resolved. The message is model-facing: it names the
/// failing endpoint and says what to do next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundaryError {
    pub kind: BoundaryErrorKind,
    pub endpoint: Endpoint,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryErrorKind {
    /// The ref never existed here: a typo, or a ref from another session.
    Unknown,
    /// The ref existed but its material is folded into a block.
    Consumed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endpoint {
    Start,
    End,
}

impl std::fmt::Display for BoundaryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for BoundaryError {}

/// A resolved range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedRange {
    pub start_index: usize,
    pub end_index: usize,
    /// Raw ids of the compressible messages in the span, in order. Excludes
    /// rendered summaries: they are view artefacts, not content.
    pub message_ids: Vec<String>,
    /// Active blocks whose visible representation falls inside the span. These
    /// are consumed by the new fold (this is how tier-2 distillation works).
    pub nested_block_ids: Vec<String>,
    pub boundary_kind: BoundaryKind,
    /// Human-readable notes about boundaries that were rewritten, so the model
    /// can see its input was adjusted.
    pub notes: Vec<String>,
}

/// Resolve `start..end` against the visible list.
///
/// Reversed pairs are normalised rather than rejected: direction is
/// notational, and both orders select the same region.
pub fn resolve_boundaries(
    start_ref: &str,
    end_ref: &str,
    messages: &[CoreMessage],
    state: &CompressionState,
) -> Result<ResolvedRange, BoundaryError> {
    let start = parse_boundary(start_ref).ok_or_else(|| BoundaryError {
        kind: BoundaryErrorKind::Unknown,
        endpoint: Endpoint::Start,
        message: format!(
            "startId=\"{start_ref}\" is not a valid ref. Use mNNNNN (a message) or bN (a block)."
        ),
    })?;
    let end = parse_boundary(end_ref).ok_or_else(|| BoundaryError {
        kind: BoundaryErrorKind::Unknown,
        endpoint: Endpoint::End,
        message: format!(
            "endId=\"{end_ref}\" is not a valid ref. Use mNNNNN (a message) or bN (a block)."
        ),
    })?;

    let mut index_by_id: BTreeMap<&str, usize> = BTreeMap::new();
    for (index, message) in messages.iter().enumerate() {
        index_by_id.entry(message.id.as_str()).or_insert(index);
    }

    let mut notes = Vec::new();
    let start_res = resolve_anchor(&start, state, &index_by_id, Endpoint::Start)?;
    if let Some(note) = start_res.snapped {
        notes.push(note);
    }
    let end_res = resolve_anchor(&end, state, &index_by_id, Endpoint::End)?;
    if let Some(note) = end_res.snapped {
        notes.push(note);
    }

    let (start_index, end_index) = if start_res.index > end_res.index {
        notes.push(format!(
            "note: refs were given reversed ({}→{}), normalised to {}..{}",
            start.raw, end.raw, end.raw, start.raw
        ));
        (end_res.index, start_res.index)
    } else {
        (start_res.index, end_res.index)
    };

    let mut message_ids = Vec::new();
    for message in messages.iter().take(end_index + 1).skip(start_index) {
        // Synthetic summaries are view representations, not compressible
        // content: letting them into a block would make the block claim to
        // cover itself.
        if is_rendered_summary(message) {
            continue;
        }
        message_ids.push(message.id.clone());
    }

    let boundary_kind = if start.kind == BoundaryKind::Block || end.kind == BoundaryKind::Block {
        BoundaryKind::Block
    } else {
        BoundaryKind::Message
    };

    let mut nested_block_ids = Vec::new();
    for block in active_blocks(state) {
        if block_visible_in_range(block, &index_by_id, start_index, end_index) {
            nested_block_ids.push(block.block_id.clone());
        }
    }

    Ok(ResolvedRange {
        start_index,
        end_index,
        message_ids,
        nested_block_ids,
        boundary_kind,
        notes,
    })
}

struct AnchorResolution {
    index: usize,
    snapped: Option<String>,
}

fn resolve_anchor(
    boundary: &ParsedBoundary,
    state: &CompressionState,
    index_by_id: &BTreeMap<&str, usize>,
    endpoint: Endpoint,
) -> Result<AnchorResolution, BoundaryError> {
    let label = match endpoint {
        Endpoint::Start => "startId",
        Endpoint::End => "endId",
    };

    if boundary.kind == BoundaryKind::Message {
        let raw_id = state
            .message_refs
            .raw_for(&boundary.raw)
            .map(str::to_string)
            .or_else(|| {
                let padded = format!("m{:05}", boundary.numeric_id);
                state.message_refs.raw_for(&padded).map(str::to_string)
            });

        let Some(raw_id) = raw_id else {
            return Err(BoundaryError {
                kind: BoundaryErrorKind::Unknown,
                endpoint,
                message: format!(
                    "{label}=\"{}\" does not exist in this session (typo, or a ref from \
                     another session) — run acp_status for the current refs.",
                    boundary.raw
                ),
            });
        };

        if let Some(index) = index_by_id.get(raw_id.as_str()) {
            return Ok(AnchorResolution {
                index: *index,
                snapped: None,
            });
        }

        // Folded away: snap to whatever represents it now.
        if let Some(index) = active_owner_anchor(state, &[raw_id.clone()], index_by_id) {
            return Ok(AnchorResolution {
                index,
                snapped: Some(format!(
                    "{label}=\"{}\" refers to a message already compressed into an active \
                     block — anchored to the active block covering it instead.",
                    boundary.raw
                )),
            });
        }

        // No ancestor owns it — but a block may have compressed it DIRECTLY.
        // Snapping there would turn a message-range retry into a duplicate
        // same-tier block, so name the block and let the model cite it.
        if let Some(owner) = direct_owner_block(state, &raw_id) {
            return Err(BoundaryError {
                kind: BoundaryErrorKind::Consumed,
                endpoint,
                message: format!(
                    "{label}=\"{}\" is already compressed into block {owner} — cite that \
                     block instead (e.g. startId=\"{owner}\", endId=\"{owner}\") to rewrite or \
                     merge it, or run acp_status to see what is still compressible.",
                    boundary.raw
                ),
            });
        }

        return Err(BoundaryError {
            kind: BoundaryErrorKind::Consumed,
            endpoint,
            message: format!(
                "{label}=\"{}\" is not in the visible context (it was consumed by a block). \
                 Run acp_status for the current refs.",
                boundary.raw
            ),
        });
    }

    let block_id = format!("b{}", boundary.numeric_id);
    let Some(block) = block_by_id(state, &block_id) else {
        return Err(BoundaryError {
            kind: BoundaryErrorKind::Unknown,
            endpoint,
            message: format!(
                "{label}=\"{block_id}\" does not exist in this session (typo, or a block from \
                 another session) — run acp_status for the current blocks."
            ),
        });
    };

    if block.active
        && let Some(index) = visible_block_anchor(block, index_by_id)
    {
        return Ok(AnchorResolution {
            index,
            snapped: None,
        });
    }

    if let Some(index) = active_owner_anchor(state, &block.effective_message_ids, index_by_id) {
        return Ok(AnchorResolution {
            index,
            snapped: Some(format!(
                "{label}=\"{block_id}\" was consumed by a higher-tier block — anchored to the \
                 active block covering its content instead."
            )),
        });
    }

    Err(BoundaryError {
        kind: BoundaryErrorKind::Consumed,
        endpoint,
        message: format!(
            "{label}=\"{block_id}\" is not in the visible context (distilled into a higher-tier \
             block) — run acp_status to verify."
        ),
    })
}

/// Snap a consumed anchor to the active block that inherited its content.
///
/// Only true ancestors qualify — a block that got the content by consuming
/// another block. A block that compressed the message directly is NOT a snap
/// target: the caller must be told "already compressed, cite the block instead"
/// rather than silently turning a message-range retry into a duplicate block.
fn active_owner_anchor(
    state: &CompressionState,
    owned_ids: &[String],
    index_by_id: &BTreeMap<&str, usize>,
) -> Option<usize> {
    if owned_ids.is_empty() {
        return None;
    }
    let mut best: Option<usize> = None;
    for block in active_blocks(state) {
        let inherited = inherited_content_ids(state, block);
        if !owned_ids.iter().any(|id| inherited.contains(id.as_str())) {
            continue;
        }
        if let Some(anchor) = visible_block_anchor(block, index_by_id) {
            best = Some(match best {
                Some(current) => current.min(anchor),
                None => anchor,
            });
        }
    }
    best
}

/// The active block that compressed `raw_id` directly, if any.
fn direct_owner_block(state: &CompressionState, raw_id: &str) -> Option<String> {
    let base = crate::agent::bigctx::types::base_id_of(raw_id);
    active_blocks(state)
        .find(|b| {
            b.direct_message_ids
                .iter()
                .any(|id| crate::agent::bigctx::types::base_id_of(id) == base)
        })
        .map(|b| b.block_id.clone())
}

/// Content a block INHERITED by consuming other blocks: the union of its
/// children's effective coverage. Derived from `direct_block_ids` rather than
/// inferred as `effective − direct`, so an imported or rebuilt state cannot
/// flip the consumed-vs-snap decision.
fn inherited_content_ids<'a>(
    state: &'a CompressionState,
    block: &'a CompressionBlock,
) -> std::collections::BTreeSet<&'a str> {
    let mut ids = std::collections::BTreeSet::new();
    for child_id in &block.direct_block_ids {
        if let Some(child) = block_by_id(state, child_id) {
            for id in &child.effective_message_ids {
                ids.insert(id.as_str());
            }
        }
    }
    ids
}

/// Where a block is visible: its rendered summary if present, else the earliest
/// visible raw message it covers. Without the summary fallback, an active block
/// whose raws were folded becomes unresolvable and cannot be promoted.
pub fn visible_block_anchor(
    block: &CompressionBlock,
    index_by_id: &BTreeMap<&str, usize>,
) -> Option<usize> {
    if let Some(index) = index_by_id.get(summary_message_id(&block.block_id).as_str()) {
        return Some(*index);
    }
    earliest_index_of_ids(&block.effective_message_ids, index_by_id)
}

/// True when any visible representation of the block falls inside the span.
///
/// The summary alone is not sufficient: when a block covers the session's first
/// user message, folding keeps that raw and puts the summary before it, so the
/// summary can sit outside a range that still contains the raw.
pub fn block_visible_in_range(
    block: &CompressionBlock,
    index_by_id: &BTreeMap<&str, usize>,
    start_index: usize,
    end_index: usize,
) -> bool {
    if let Some(index) = index_by_id.get(summary_message_id(&block.block_id).as_str())
        && (start_index..=end_index).contains(index)
    {
        return true;
    }
    earliest_index_of_ids(&block.effective_message_ids, index_by_id)
        .is_some_and(|index| (start_index..=end_index).contains(&index))
}

/// Earliest visible index among `ids`.
pub fn earliest_index_of_ids(
    ids: &[String],
    index_by_id: &BTreeMap<&str, usize>,
) -> Option<usize> {
    ids.iter()
        .filter_map(|id| index_by_id.get(id.as_str()).copied())
        .min()
}

/// True when a ref is the protected placeholder and can never be an endpoint.
pub fn is_blocked_ref(reference: &str) -> bool {
    reference.trim().eq_ignore_ascii_case(BLOCKED_REF)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::bigctx::state::create_initial_state;
    use crate::agent::bigctx::types::{
        BlockGeneration, MessageContentType, MessageRole, Tier,
    };

    fn user(id: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::User, MessageContentType::Text).with_text("t")
    }

    fn block(id: &str, covers: &[&str]) -> CompressionBlock {
        CompressionBlock {
            block_id: id.to_string(),
            run_id: "r1".to_string(),
            tier: Tier::One,
            topic: None,
            start_ref: None,
            end_ref: None,
            summary: "s".to_string(),
            direct_message_ids: covers.iter().map(|s| s.to_string()).collect(),
            effective_message_ids: covers.iter().map(|s| s.to_string()).collect(),
            direct_block_ids: Vec::new(),
            compressed_tokens: 10,
            created_at: 0,
            survived_count: 0,
            generation: BlockGeneration::Young,
            active: true,
            expanded: false,
            duration_ms: None,
            compress_call_id: None,
        }
    }

    fn state_with_refs(ids: &[&str]) -> CompressionState {
        let mut state = create_initial_state();
        for (offset, id) in ids.iter().enumerate() {
            let reference = format!("m{:05}", offset + 1);
            state.message_refs.by_raw.insert(id.to_string(), reference.clone());
            state.message_refs.by_ref.insert(reference, id.to_string());
        }
        state
    }

    #[test]
    fn parsing_accepts_both_ref_kinds() {
        assert_eq!(
            parse_boundary("m00012"),
            Some(ParsedBoundary {
                kind: BoundaryKind::Message,
                numeric_id: 12,
                raw: "m00012".into()
            })
        );
        assert_eq!(
            parse_boundary("b7"),
            Some(ParsedBoundary {
                kind: BoundaryKind::Block,
                numeric_id: 7,
                raw: "b7".into()
            })
        );
    }

    #[test]
    fn parsing_rejects_junk() {
        for junk in ["", "m", "b", "b0", "x1", "m00000000", "m0"] {
            assert_eq!(parse_boundary(junk), None, "{junk}");
        }
    }

    #[test]
    fn a_plain_message_range_resolves() {
        let state = state_with_refs(&["t1.1", "t1.2", "t1.3"]);
        let msgs = vec![user("t1.1"), user("t1.2"), user("t1.3")];
        let range = resolve_boundaries("m00001", "m00002", &msgs, &state).unwrap();
        assert_eq!(range.start_index, 0);
        assert_eq!(range.end_index, 1);
        assert_eq!(range.message_ids, vec!["t1.1", "t1.2"]);
        assert!(range.notes.is_empty());
    }

    #[test]
    fn a_reversed_pair_is_normalised_with_a_note() {
        let state = state_with_refs(&["t1.1", "t1.2", "t1.3"]);
        let msgs = vec![user("t1.1"), user("t1.2"), user("t1.3")];
        let range = resolve_boundaries("m00003", "m00001", &msgs, &state).unwrap();
        assert_eq!(range.start_index, 0);
        assert_eq!(range.end_index, 2);
        assert!(range.notes.iter().any(|n| n.contains("reversed")));
    }

    #[test]
    fn an_unknown_ref_names_the_failing_endpoint() {
        let state = state_with_refs(&["t1.1"]);
        let msgs = vec![user("t1.1")];
        let err = resolve_boundaries("m00099", "m00001", &msgs, &state).unwrap_err();
        assert_eq!(err.kind, BoundaryErrorKind::Unknown);
        assert_eq!(err.endpoint, Endpoint::Start);
        assert!(err.message.contains("startId"), "{}", err.message);
    }

    #[test]
    fn a_directly_compressed_message_tells_the_model_to_cite_the_block() {
        // Snapping here would turn a message-range retry into a duplicate
        // same-tier block, so the resolver names the block instead.
        let mut state = state_with_refs(&["t1.1", "t1.2", "t1.3"]);
        state.blocks.push(block("b1", &["t1.2"]));
        let msgs = vec![
            user("t1.1"),
            CoreMessage::new(
                summary_message_id("b1"),
                MessageRole::System,
                MessageContentType::Text,
            )
            .with_text("s"),
            user("t1.3"),
        ];
        let err = resolve_boundaries("m00002", "m00003", &msgs, &state).unwrap_err();
        assert_eq!(err.kind, BoundaryErrorKind::Consumed);
        assert!(
            err.message.contains("b1") && err.message.contains("cite that block"),
            "the model must be told to cite b1: {}",
            err.message
        );
    }

    #[test]
    fn a_message_consumed_by_a_higher_tier_block_snaps_to_it() {
        let mut state = state_with_refs(&["t1.1", "t1.2", "t1.3"]);
        // b1 compressed t1.2; b2 distilled b1, so t1.2 is INHERITED by b2
        state.blocks.push(block("b1", &["t1.2"]));
        let mut b2 = block("b2", &[]);
        b2.direct_block_ids = vec!["b1".to_string()];
        b2.effective_message_ids = vec!["t1.2".to_string()];
        state.blocks.push(b2);

        let msgs = vec![
            user("t1.1"),
            CoreMessage::new(
                summary_message_id("b2"),
                MessageRole::System,
                MessageContentType::Text,
            )
            .with_text("s2"),
            user("t1.3"),
        ];
        let range = resolve_boundaries("m00002", "m00003", &msgs, &state).unwrap();
        assert!(
            range.notes.iter().any(|n| n.contains("already compressed")),
            "the ancestor snap must be reported: {:?}",
            range.notes
        );
    }

    #[test]
    fn a_block_range_resolves_to_its_summary() {
        let mut state = state_with_refs(&["t1.1", "t1.2"]);
        state.blocks.push(block("b1", &["t1.2"]));
        let msgs = vec![
            user("t1.1"),
            CoreMessage::new(
                summary_message_id("b1"),
                MessageRole::System,
                MessageContentType::Text,
            )
            .with_text("s"),
        ];
        let range = resolve_boundaries("b1", "b1", &msgs, &state).unwrap();
        assert_eq!(range.boundary_kind, BoundaryKind::Block);
        assert_eq!(range.start_index, 1);
        assert_eq!(range.nested_block_ids, vec!["b1"]);
    }

    #[test]
    fn an_unknown_block_is_reported_as_unknown() {
        let state = state_with_refs(&["t1.1"]);
        let msgs = vec![user("t1.1")];
        let err = resolve_boundaries("b9", "b9", &msgs, &state).unwrap_err();
        assert_eq!(err.kind, BoundaryErrorKind::Unknown);
        assert!(err.message.contains("b9"));
    }

    #[test]
    fn rendered_summaries_are_not_compressible_content() {
        let mut state = state_with_refs(&["t1.1", "t1.2"]);
        state.blocks.push(block("b1", &["t1.2"]));
        let msgs = vec![
            user("t1.1"),
            CoreMessage::new(
                summary_message_id("b1"),
                MessageRole::System,
                MessageContentType::Text,
            )
            .with_text("s"),
        ];
        let range = resolve_boundaries("m00001", "b1", &msgs, &state).unwrap();
        assert_eq!(range.message_ids, vec!["t1.1"]);
        assert!(!range.message_ids.iter().any(|id| id.starts_with("acp_summary_")));
    }

    #[test]
    fn nested_blocks_inside_the_span_are_collected() {
        let mut state = state_with_refs(&["t1.1", "t1.2", "t1.3"]);
        state.blocks.push(block("b1", &["t1.2"]));
        state.blocks.push(block("b2", &["t1.3"]));
        let msgs = vec![
            user("t1.1"),
            CoreMessage::new(
                summary_message_id("b1"),
                MessageRole::System,
                MessageContentType::Text,
            )
            .with_text("s1"),
            CoreMessage::new(
                summary_message_id("b2"),
                MessageRole::System,
                MessageContentType::Text,
            )
            .with_text("s2"),
        ];
        let range = resolve_boundaries("b1", "b2", &msgs, &state).unwrap();
        assert_eq!(range.nested_block_ids, vec!["b1", "b2"]);
    }
}
