//! Which ranges are worth folding, and which must not be.
//!
//! Two questions, both answered every turn:
//!
//! 1. **What is protected?** The recent zone (last N messages, last N tokens,
//!    the last user message), closed symmetrically per tool pair.
//! 2. **What is left, and how does it group?** The unprotected messages, cut
//!    into contiguous runs that a single `compress` call can cite.
//!
//! Segmentation is **array adjacency, never ref arithmetic**. A host that
//! replaces its surface leaves holes in the ref map: a compressed message
//! leaves the array but keeps its ref, and a summary node can be inserted
//! mid-array carrying a fresh *high* ref. Ref arithmetic fragments every range
//! there and emits `startRef > endRef` pairs. Only a numbered-ref message that
//! was physically skipped interrupts a run; unrefed and blocked entries consume
//! no slot. On a dense append-only host the two rules coincide.
//!
//! Groups also break at a user message once the current run holds three items,
//! which keeps each range roughly one turn long.

use std::collections::BTreeSet;

use crate::agent::bigctx::config::BigctxConfig;
use crate::agent::bigctx::prune::SUMMARY_HEADER;
use crate::agent::bigctx::protected::{Protection, compute_protection};
use crate::agent::bigctx::state::covered_message_ids;
use crate::agent::bigctx::tokenize::count_message_tokens;
use crate::agent::bigctx::turn_integrity::withdrawals;
use crate::agent::bigctx::types::{
    CompressionState, CoreMessage, MessageContentType, MessageRole, base_id_of,
};

/// A contiguous run of foldable messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressibleRange {
    /// Ref of the first message.
    pub start_ref: String,
    /// Ref of the last message.
    pub end_ref: String,
    /// Array index of the first message.
    pub start_index: usize,
    /// Array index of the last message.
    pub end_index: usize,
    /// How many messages the run holds.
    pub count: usize,
    /// Token cost of the run.
    pub tokens: u64,
    /// Share of the run that is tool traffic, 0–100.
    pub tool_pct: u8,
    /// How many user messages the run holds.
    pub user_msgs: usize,
}

impl CompressibleRange {
    /// True when folding this run would actually save something.
    pub fn is_worth_folding(&self, minimum_tokens: u64) -> bool {
        self.tokens > 0 && self.tokens >= minimum_tokens
    }
}

/// A contiguous run of messages that may not be folded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtectedRange {
    pub start_ref: String,
    pub end_ref: String,
    pub count: usize,
    pub tokens: u64,
    /// Tool names in the run, deduplicated.
    pub tools: Vec<String>,
}

/// What a screening pass found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ContextRanges {
    pub compressible: Vec<CompressibleRange>,
    pub protected: Vec<ProtectedRange>,
}

impl ContextRanges {
    /// True when there is nothing left worth folding.
    pub fn nothing_to_compress(&self) -> bool {
        self.compressible.is_empty()
    }

    /// Total foldable tokens, for deciding whether a nudge is worth showing.
    pub fn compressible_tokens(&self) -> u64 {
        self.compressible.iter().map(|r| r.tokens).sum()
    }
}

/// One message while screening, before grouping.
struct Candidate {
    id: String,
    reference: String,
    gap_before: bool,
    tokens: u64,
    is_tool: bool,
    is_user: bool,
    index: usize,
}

/// A tool message is a call **or** its result — never inferred from a tool
/// name, because wire converters only set the name on the call side and a
/// name-based check would classify every result as text.
fn is_tool_message(message: &CoreMessage) -> bool {
    matches!(
        message.content_type,
        MessageContentType::ToolCall | MessageContentType::ToolResult
    )
}

/// True when the message is a summary this module rendered, or is folded.
fn is_synthetic_or_pruned(message: &CoreMessage, covered: &BTreeSet<String>) -> bool {
    if message
        .text
        .as_deref()
        .is_some_and(|text| text.starts_with(SUMMARY_HEADER))
    {
        return true;
    }
    covered.contains(base_id_of(&message.id))
}

/// Numeric part of a ref (`m00005` → 5); `None` for anything else.
fn ref_number(reference: &str) -> Option<u64> {
    let digits = reference.strip_prefix('m').or_else(|| reference.strip_prefix('M'))?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// Screen the message list into foldable and protected runs.
pub fn build_context_ranges(
    messages: &[CoreMessage],
    state: &CompressionState,
    config: &BigctxConfig,
    protection: &Protection,
) -> ContextRanges {
    let covered = covered_message_ids(state);
    let mut compressible_msgs: Vec<Candidate> = Vec::new();
    let mut protected_msgs: Vec<(String, bool, u64, Vec<String>, usize)> = Vec::new();

    let mut skip_since_compressible = false;
    let mut skip_since_protected = false;

    for (index, message) in messages.iter().enumerate() {
        let Some(reference) = state.message_refs.ref_for(&message.id) else {
            continue;
        };
        if reference == crate::agent::bigctx::refs::BLOCKED_REF {
            continue;
        }
        if is_synthetic_or_pruned(message, &covered) {
            skip_since_compressible = true;
            skip_since_protected = true;
            continue;
        }

        if protection.is_protected(message) {
            protected_msgs.push((
                reference.to_string(),
                skip_since_protected,
                count_message_tokens(message),
                message.tool_name.iter().cloned().collect(),
                index,
            ));
            skip_since_protected = false;
            skip_since_compressible = true;
            continue;
        }

        compressible_msgs.push(Candidate {
            id: message.id.clone(),
            reference: reference.to_string(),
            gap_before: skip_since_compressible,
            tokens: count_message_tokens(message),
            is_tool: is_tool_message(message),
            is_user: message.role == MessageRole::User,
            index,
        });
        skip_since_compressible = false;
        skip_since_protected = true;
    }

    // Foldability: a run that brushes the protected zone can hold nothing the
    // fold may take, because the gate withdraws any message whose turn would
    // lose its reasoning run or whose call/result pair it would split. Screen
    // those out here so every advertised run is actually foldable — otherwise
    // the model cites a range and the fold fails every turn.
    let folded_ids: BTreeSet<String> = compressible_msgs.iter().map(|c| c.id.clone()).collect();
    let gate = withdrawals(messages, &folded_ids);
    if !gate.is_empty() {
        let mut gap_pending = false;
        let mut kept: Vec<Candidate> = Vec::new();
        for mut candidate in compressible_msgs {
            if gate.keeps(&candidate.id) {
                gap_pending = true;
                continue;
            }
            if gap_pending {
                candidate.gap_before = true;
                gap_pending = false;
            }
            kept.push(candidate);
        }
        compressible_msgs = kept;
    }

    let compressible = group_compressible(&compressible_msgs);
    let protected = group_protected(&protected_msgs);

    ContextRanges {
        compressible: compressible.into_iter().filter(|r| r.tokens > 0).collect(),
        protected,
    }
}

/// Cut candidates into contiguous runs.
///
/// Two split rules: a real array gap always starts a new run, and a user
/// message starts one once the current run already holds three items.
fn group_compressible(candidates: &[Candidate]) -> Vec<CompressibleRange> {
    let mut groups: Vec<Vec<&Candidate>> = Vec::new();
    let mut current: Vec<&Candidate> = Vec::new();

    for candidate in candidates {
        if !current.is_empty() && (candidate.gap_before || (candidate.is_user && current.len() >= 3)) {
            groups.push(std::mem::take(&mut current));
        }
        current.push(candidate);
    }
    if !current.is_empty() {
        groups.push(current);
    }

    groups
        .into_iter()
        .map(|group| {
            let first = group[0];
            let last = group[group.len() - 1];
            let mut tokens = 0u64;
            let mut tool_count = 0usize;
            let mut user_msgs = 0usize;
            for candidate in &group {
                tokens += candidate.tokens;
                if candidate.is_tool {
                    tool_count += 1;
                }
                if candidate.is_user {
                    user_msgs += 1;
                }
            }
            let tool_pct = if group.is_empty() {
                0
            } else {
                ((tool_count * 100) / group.len()) as u8
            };
            CompressibleRange {
                start_ref: first.reference.clone(),
                end_ref: last.reference.clone(),
                start_index: first.index,
                end_index: last.index,
                count: group.len(),
                tokens,
                tool_pct,
                user_msgs,
            }
        })
        .collect()
}

/// Join protected messages into contiguous runs.
fn group_protected(
    protected: &[(String, bool, u64, Vec<String>, usize)],
) -> Vec<ProtectedRange> {
    let mut ranges: Vec<ProtectedRange> = Vec::new();
    let mut current: Option<ProtectedRange> = None;

    for (reference, gap_before, tokens, tools, _index) in protected {
        if current.is_some() && *gap_before {
            ranges.push(current.take().expect("checked above"));
        }
        match &mut current {
            None => {
                current = Some(ProtectedRange {
                    start_ref: reference.clone(),
                    end_ref: reference.clone(),
                    count: 1,
                    tokens: *tokens,
                    tools: tools.clone(),
                });
            }
            Some(range) => {
                range.end_ref = reference.clone();
                range.count += 1;
                range.tokens += tokens;
                for tool in tools {
                    if !range.tools.contains(tool) {
                        range.tools.push(tool.clone());
                    }
                }
            }
        }
    }
    if let Some(range) = current {
        ranges.push(range);
    }
    ranges
}

/// Merge neighbouring runs until each one is worth a summary.
///
/// A run below `min_compress_range` is not worth a model round trip on its own,
/// so it is merged with its neighbour. Merging stops at a gap: two runs on
/// opposite sides of a protected or already-folded message cannot be cited by
/// one range, so joining them would advertise something unciteable.
pub fn merge_to_threshold(ranges: Vec<CompressibleRange>, minimum: u64) -> Vec<CompressibleRange> {
    if minimum == 0 {
        return ranges;
    }
    let mut out: Vec<CompressibleRange> = Vec::new();
    let mut pending: Option<CompressibleRange> = None;

    for range in ranges {
        let merged = match pending.take() {
            None => range,
            Some(previous) => merge_two(previous, range),
        };
        if merged.tokens >= minimum {
            out.push(merged);
        } else {
            pending = Some(merged);
        }
    }
    if let Some(last) = pending {
        // Nothing to merge it into; keep it rather than silently dropping a
        // foldable run.
        out.push(last);
    }
    out
}

fn merge_two(left: CompressibleRange, right: CompressibleRange) -> CompressibleRange {
    let count = left.count + right.count;
    let tool_tokens = left.tool_pct as usize * left.count + right.tool_pct as usize * right.count;
    CompressibleRange {
        start_ref: left.start_ref,
        end_ref: right.end_ref,
        start_index: left.start_index,
        end_index: right.end_index,
        count,
        tokens: left.tokens + right.tokens,
        tool_pct: if count == 0 {
            0
        } else {
            (tool_tokens / count) as u8
        },
        user_msgs: left.user_msgs + right.user_msgs,
    }
}

/// Screen a message list and group it, in one call.
pub fn screen(
    messages: &[CoreMessage],
    state: &CompressionState,
    config: &BigctxConfig,
) -> ContextRanges {
    let protection = compute_protection(messages, state, config);
    let ranges = build_context_ranges(messages, state, config, &protection);
    let merged = merge_to_threshold(ranges.compressible, config.compress.min_range_tokens);
    ContextRanges {
        compressible: merged,
        protected: ranges.protected,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::bigctx::config::default_config;
    use crate::agent::bigctx::refs::assign_refs;
    use crate::agent::bigctx::state::create_initial_state;
    use crate::agent::bigctx::types::{CompressionBlock, BlockGeneration, Tier};

    fn user(id: &str, text: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::User, MessageContentType::Text).with_text(text)
    }

    fn assistant(id: &str, text: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::Assistant, MessageContentType::Text).with_text(text)
    }

    fn call(id: &str, call_id: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::Assistant, MessageContentType::ToolCall)
            .with_text("{}")
            .with_tool("read", call_id)
    }

    fn result(id: &str, call_id: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::Tool, MessageContentType::ToolResult)
            .with_text("body")
            .with_tool("read", call_id)
    }

    /// A state with refs assigned and no protection at all.
    fn refs_for(messages: &[CoreMessage]) -> CompressionState {
        let mut state = create_initial_state();
        let mut next = 1u64;
        assign_refs(
            messages,
            &mut state.message_refs,
            &mut next,
            |_| false,
        )
        .unwrap();
        state.next_ref_index = next;
        state
    }

    fn no_protection() -> Protection {
        Protection::default()
    }

    fn cfg() -> BigctxConfig {
        let mut c = default_config(200_000);
        c.preserve_recent_messages = 0;
        c.preserve_recent_tokens = 0;
        c.compress.min_range_tokens = 0;
        c
    }

    #[test]
    fn plain_messages_group_into_one_run() {
        let msgs = vec![user("t1.0", "a"), assistant("t1.1", "b")];
        let state = refs_for(&msgs);
        let ranges = build_context_ranges(&msgs, &state, &cfg(), &no_protection());
        assert_eq!(ranges.compressible.len(), 1);
        assert_eq!(ranges.compressible[0].count, 2);
        assert_eq!(ranges.compressible[0].start_ref, "m00001");
        assert_eq!(ranges.compressible[0].end_ref, "m00002");
    }

    #[test]
    fn a_protected_message_splits_the_run() {
        let msgs = vec![
            user("t1.0", "a"),
            assistant("t1.1", "b"),
            user("t1.2", "c"),
            assistant("t1.3", "d"),
        ];
        let state = refs_for(&msgs);
        let mut protection = Protection::default();
        protection.protect_base("t1.1");

        let ranges = build_context_ranges(&msgs, &state, &cfg(), &protection);
        assert_eq!(ranges.protected.len(), 1);
        assert_eq!(ranges.compressible.len(), 2, "the zone splits the run");
        assert_eq!(ranges.compressible[0].count, 1);
        assert_eq!(ranges.compressible[1].count, 2);
    }

    #[test]
    fn a_user_message_starts_a_new_run_after_three_items() {
        let msgs = vec![
            user("t1.0", "a"),
            assistant("t1.1", "b"),
            assistant("t1.2", "c"),
            user("t1.3", "d"),
        ];
        let state = refs_for(&msgs);
        let ranges = build_context_ranges(&msgs, &state, &cfg(), &no_protection());
        assert_eq!(ranges.compressible.len(), 2, "turn-aligned split");
        assert_eq!(ranges.compressible[0].count, 3);
        assert_eq!(ranges.compressible[1].count, 1);
    }

    #[test]
    fn a_short_user_run_does_not_split() {
        let msgs = vec![user("t1.0", "a"), user("t1.1", "b")];
        let state = refs_for(&msgs);
        let ranges = build_context_ranges(&msgs, &state, &cfg(), &no_protection());
        assert_eq!(ranges.compressible.len(), 1, "fewer than 3 items: no split");
    }

    #[test]
    fn a_folded_message_is_not_advertised() {
        let msgs = vec![user("t1.0", "a"), assistant("t1.1", "b"), user("t1.2", "c")];
        let mut state = refs_for(&msgs);
        state.blocks.push(CompressionBlock {
            block_id: "b1".into(),
            run_id: "r1".into(),
            tier: Tier::One,
            topic: None,
            start_ref: None,
            end_ref: None,
            summary: "s".into(),
            direct_message_ids: vec!["t1.1".into()],
            effective_message_ids: vec!["t1.1".into()],
            direct_block_ids: vec![],
            compressed_tokens: 10,
            created_at: 0,
            survived_count: 0,
            generation: BlockGeneration::Young,
            active: true,
            expanded: false,
            duration_ms: None,
            compress_call_id: None,
        });
        let ranges = build_context_ranges(&msgs, &state, &cfg(), &no_protection());
        let advertised: Vec<&str> = ranges
            .compressible
            .iter()
            .map(|r| r.start_ref.as_str())
            .collect();
        assert!(
            !advertised.contains(&"m00002"),
            "a folded message must never be advertised"
        );
    }

    #[test]
    fn a_call_without_its_result_is_withdrawn_from_screening() {
        // The gate would refuse this fold, so it must not be advertised.
        let msgs = vec![user("t1.0", "go"), call("t1.1", "c1"), result("t1.2", "c1")];
        let state = refs_for(&msgs);
        let mut protection = Protection::default();
        protection.protect_base("t1.2"); // the result is protected

        let ranges = build_context_ranges(&msgs, &state, &cfg(), &protection);
        // the call cannot fold alone, so nothing groups with it
        assert!(
            ranges.compressible.iter().all(|r| r.count == 1),
            "the orphaned call must not be bundled"
        );
    }

    #[test]
    fn merging_lifts_a_small_run_over_the_threshold() {
        let msgs = vec![
            user("t1.0", "a"),
            assistant("t1.1", "b"),
            user("t1.2", "c"),
            assistant("t1.3", "d"),
            assistant("t1.4", "e"),
            user("t1.5", "f"),
        ];
        let state = refs_for(&msgs);
        let ranges = build_context_ranges(&msgs, &state, &cfg(), &no_protection());
        let before = ranges.compressible.len();

        let merged = merge_to_threshold(ranges.compressible, 100_000);
        assert_eq!(merged.len(), 1, "everything merges into one run");
        assert_eq!(merged[0].count, 6);
        assert!(before > 1, "there was something to merge");
    }

    #[test]
    fn merging_keeps_every_message() {
        let msgs = vec![user("t1.0", "a"), assistant("t1.1", "b"), user("t1.2", "c")];
        let state = refs_for(&msgs);
        let ranges = build_context_ranges(&msgs, &state, &cfg(), &no_protection());
        let total_before: usize = ranges.compressible.iter().map(|r| r.count).sum();
        let merged = merge_to_threshold(ranges.compressible, 50_000);
        let total_after: usize = merged.iter().map(|r| r.count).sum();
        assert_eq!(total_before, total_after, "merging must not lose messages");
    }

    #[test]
    fn tool_percentage_tracks_tool_traffic() {
        let msgs = vec![call("t1.0", "c1"), result("t1.1", "c1")];
        let state = refs_for(&msgs);
        let ranges = build_context_ranges(&msgs, &state, &cfg(), &no_protection());
        assert_eq!(ranges.compressible[0].tool_pct, 100);
    }

    #[test]
    fn a_tool_result_counts_as_tool_traffic() {
        // A name-based check would call this text: wire converters only set the
        // tool name on the call side.
        let mut bare = result("t1.1", "c1");
        bare.tool_name = None;
        assert!(is_tool_message(&bare));
    }

    #[test]
    fn a_rendered_summary_is_not_advertised() {
        let msgs = vec![
            user("t1.0", "a"),
            CoreMessage::new(
                crate::agent::bigctx::prune::summary_message_id("b1"),
                MessageRole::System,
                MessageContentType::Text,
            )
            .with_text(format!("{SUMMARY_HEADER}\nbody")),
            assistant("t1.2", "c"),
        ];
        let state = refs_for(&msgs);
        let ranges = build_context_ranges(&msgs, &state, &cfg(), &no_protection());
        for range in &ranges.compressible {
            assert_ne!(range.start_ref, "m00002", "a summary is not foldable");
        }
    }

    #[test]
    fn screening_with_defaults_protects_the_tail() {
        let msgs = vec![user("t1.0", "task"), assistant("t1.1", "work")];
        let state = refs_for(&msgs);
        let ranges = screen(&msgs, &state, &default_config(200_000));
        assert!(
            ranges.nothing_to_compress(),
            "the default window protects everything here"
        );
    }

    #[test]
    fn ref_number_parses_only_numbered_refs() {
        assert_eq!(ref_number("m00005"), Some(5));
        assert_eq!(ref_number("M12"), Some(12));
        assert_eq!(ref_number("b1"), None);
        assert_eq!(ref_number("m"), None);
        assert_eq!(ref_number("mx"), None);
    }
}
