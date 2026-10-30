//! Applying a compression: turning a resolved range into a block.
//!
//! The host supplies the summary (it owns the model call); this module owns
//! everything else — what the block records, which blocks it consumes, what
//! happens to the state, and what the model is told.
//!
//! Two invariants:
//!
//! - **A block never covers a protected message.** The resolver refuses such a
//!   range before we get here, but the check is repeated: a block that covered a
//!   protected message would fold the model's own working set on the next pass.
//! - **Ids are never re-issued.** A consumed block keeps its record; only
//!   `active` flips. That is what makes an old citation resolvable forever.

use crate::agent::bigctx::boundaries::ResolvedRange;
use crate::agent::bigctx::prune::is_rendered_summary;
use crate::agent::bigctx::state::{
    active_blocks_at_tier, allocate_block_id, allocate_run_id, block_by_id,
};
use crate::agent::bigctx::tokenize::count_message_tokens;
use crate::agent::bigctx::types::{
    BlockGeneration, CompressionBlock, CompressionState, CoreMessage, Tier, base_id_of,
};

/// Why a compression was refused. These are model-facing: each one says what to
/// do instead of just what failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompressError {
    /// The range holds no messages that could actually be folded.
    NothingToCompress { reason: String },
    /// Every message in the range is protected.
    AllProtected,
    /// The range is too small to be worth a summary.
    RangeTooSmall { tokens: u64, minimum: u64 },
    /// The summary is the wrong size.
    SummarySize { tokens: u64, min: u64, max: u64 },
    /// The block the range cited is not resolvable.
    UnknownBlock(String),
}

impl std::fmt::Display for CompressError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompressError::NothingToCompress { reason } => {
                write!(f, "nothing to compress: {reason}")
            }
            CompressError::AllProtected => write!(
                f,
                "every message in that range is protected (a protected tool, the recent working \
                 set, or the last user message). Compress an older range, or run acp_status to \
                 see what is still compressible."
            ),
            CompressError::RangeTooSmall { tokens, minimum } => write!(
                f,
                "that range is only {tokens} tokens; compressing fewer than {minimum} costs a \
                 summary and buys nothing. Include more messages, or leave it alone."
            ),
            CompressError::SummarySize { tokens, min, max } => write!(
                f,
                "the summary is {tokens} tokens; it must be between {min} and {max}. Keep the \
                 file paths, decisions, signatures and error strings, and drop the rest."
            ),
            CompressError::UnknownBlock(id) => write!(f, "block {id} does not exist"),
        }
    }
}

impl std::error::Error for CompressError {}

/// What a successful compression produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressOutcome {
    pub block_id: String,
    pub tier: Tier,
    /// Original tokens the block replaced.
    pub compressed_tokens: u64,
    /// Blocks consumed by this fold (tier 2/3 only).
    pub consumed_block_ids: Vec<String>,
    /// Messages folded.
    pub message_count: usize,
    /// True when an existing block was rewritten in place rather than a new one
    /// created — the model should be told the difference.
    pub refolded: bool,
}

/// Fold `range` into a new block.
///
/// `protected` is the set of base ids that may not be folded. A message in the
/// range that is protected is silently skipped rather than failing the whole
/// call: a range that spans a protected message is still compressible around it.
pub fn apply_compression(
    state: &mut CompressionState,
    messages: &[CoreMessage],
    range: &ResolvedRange,
    summary: &str,
    protected: &crate::agent::bigctx::protected::Protection,
    config: &crate::agent::bigctx::config::BigctxConfig,
    compress_call_id: Option<String>,
    now: u64,
) -> Result<CompressOutcome, CompressError> {
    let summary_tokens = crate::agent::bigctx::tokenize::default_count_tokens(summary);
    if summary_tokens < config.compress.min_summary_tokens {
        return Err(CompressError::SummarySize {
            tokens: summary_tokens,
            min: config.compress.min_summary_tokens,
            max: config.compress.max_summary_tokens,
        });
    }
    if summary_tokens > config.compress.max_summary_tokens {
        return Err(CompressError::SummarySize {
            tokens: summary_tokens,
            min: config.compress.min_summary_tokens,
            max: config.compress.max_summary_tokens,
        });
    }

    // Which of the cited messages can actually be folded.
    let mut foldable: Vec<&CoreMessage> = Vec::new();
    let mut compressed_tokens = 0u64;
    let mut protected_seen = 0usize;
    for id in &range.message_ids {
        let Some(message) = messages.iter().find(|m| &m.id == id) else {
            continue;
        };
        if is_rendered_summary(message) {
            continue;
        }
        if protected.is_protected(message) {
            protected_seen += 1;
            continue;
        }
        compressed_tokens += count_message_tokens(message);
        foldable.push(message);
    }

    if foldable.is_empty() {
        if protected_seen > 0 {
            return Err(CompressError::AllProtected);
        }
        return Err(CompressError::NothingToCompress {
            reason: "the range resolved to no messages (they may already be folded)".to_string(),
        });
    }

    // A consumed block's summary counts toward what this fold replaces.
    let mut consumed_block_ids = Vec::new();
    for block_id in &range.nested_block_ids {
        if let Some(block) = block_by_id(state, block_id) {
            if block.active {
                compressed_tokens += crate::agent::bigctx::tokenize::default_count_tokens(
                    &block.summary,
                );
                consumed_block_ids.push(block.block_id.clone());
            }
        }
    }

    if compressed_tokens < config.compress.min_range_tokens && consumed_block_ids.is_empty() {
        return Err(CompressError::RangeTooSmall {
            tokens: compressed_tokens,
            minimum: config.compress.min_range_tokens,
        });
    }

    // Tier: folding blocks promotes one level; folding only messages is tier 1.
    let tier = if consumed_block_ids.is_empty() {
        Tier::One
    } else {
        let highest = consumed_block_ids
            .iter()
            .filter_map(|id| block_by_id(state, id).map(|b| b.tier))
            .max()
            .unwrap_or(Tier::One);
        match highest {
            Tier::One => Tier::Two,
            Tier::Two | Tier::Three => Tier::Three,
        }
    };

    let block_id = allocate_block_id(state);
    let run_id = allocate_run_id(state);

    let direct_message_ids: Vec<String> = foldable.iter().map(|m| m.id.clone()).collect();
    // The effective set is this fold's own ids plus everything the consumed
    // blocks covered — that is what keeps a message folded three tiers ago
    // recognised as folded.
    let mut effective: Vec<String> = direct_message_ids.clone();
    for consumed_id in &consumed_block_ids {
        if let Some(block) = block_by_id(state, consumed_id) {
            for id in &block.effective_message_ids {
                if !effective.contains(id) {
                    effective.push(id.clone());
                }
            }
        }
    }

    let block = CompressionBlock {
        block_id: block_id.clone(),
        run_id,
        tier,
        topic: None,
        summary: summary.trim().to_string(),
        direct_message_ids,
        effective_message_ids: effective,
        direct_block_ids: consumed_block_ids.clone(),
        compressed_tokens,
        created_at: now,
        survived_count: 0,
        generation: BlockGeneration::Young,
        active: true,
        expanded: false,
        duration_ms: None,
        compress_call_id,
    };

    // Consuming a block deactivates it: its content now lives in this block.
    for consumed_id in &consumed_block_ids {
        if let Some(block) = state.blocks.iter_mut().find(|b| &b.block_id == consumed_id) {
            block.active = false;
        }
    }

    let message_count = block.direct_message_ids.len();
    state.blocks.push(block);
    state.stats.tokens_compressed += compressed_tokens;
    state.stats.compression_count += 1;

    Ok(CompressOutcome {
        block_id,
        tier,
        compressed_tokens,
        consumed_block_ids,
        message_count,
        refolded: false,
    })
}

/// Restore a block: deactivate it so its material becomes visible again.
///
/// The block record is kept, not deleted: its id stays resolvable, and the
/// original messages come back from the history the host still holds. `expanded`
/// records that this deactivation was deliberate, so a later sync does not
/// re-activate it and re-fold what the user just asked to see.
pub fn decompress_block(
    state: &mut CompressionState,
    block_id: &str,
) -> Result<CompressionBlock, CompressError> {
    let Some(block) = state.blocks.iter_mut().find(|b| b.block_id == block_id) else {
        return Err(CompressError::UnknownBlock(block_id.to_string()));
    };
    block.active = false;
    block.expanded = true;
    Ok(block.clone())
}

/// Whether tier-2 or tier-3 distillation is due.
///
/// Counts active blocks per tier rather than comparing token mass: summaries are
/// roughly 10:1 condensed, so a token comparison would starve the higher tier
/// forever.
pub fn distillation_due(state: &CompressionState, config: &crate::agent::bigctx::config::BigctxConfig) -> Option<Tier> {
    if !config.tiers.enabled {
        return None;
    }
    if active_blocks_at_tier(state, Tier::Two) >= config.tiers.tier3_trigger {
        return Some(Tier::Three);
    }
    if active_blocks_at_tier(state, Tier::One) >= config.tiers.tier2_trigger {
        return Some(Tier::Two);
    }
    None
}

/// Base ids covered by `range`, for the caller to check against protection.
pub fn range_bases(range: &ResolvedRange) -> std::collections::BTreeSet<String> {
    range
        .message_ids
        .iter()
        .map(|id| base_id_of(id).to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::bigctx::boundaries::BoundaryKind;
    use crate::agent::bigctx::config::default_config;
    use crate::agent::bigctx::protected::{Protection, compute_protection};
    use crate::agent::bigctx::state::CompressionStateExt;
    use crate::agent::bigctx::state::create_initial_state;
    use crate::agent::bigctx::types::{MessageContentType, MessageRole};

    fn user(id: &str, text: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::User, MessageContentType::Text).with_text(text)
    }

    fn range(ids: &[&str], nested: &[&str]) -> ResolvedRange {
        ResolvedRange {
            start_index: 0,
            end_index: ids.len().saturating_sub(1),
            message_ids: ids.iter().map(|s| s.to_string()).collect(),
            nested_block_ids: nested.iter().map(|s| s.to_string()).collect(),
            boundary_kind: BoundaryKind::Message,
            notes: Vec::new(),
        }
    }

    fn cfg() -> crate::agent::bigctx::config::BigctxConfig {
        let mut c = default_config(200_000);
        c.compress.min_range_tokens = 0;
        c.preserve_recent_messages = 0;
        c.preserve_recent_tokens = 0;
        c
    }

    fn no_protection() -> Protection {
        Protection::default()
    }

    fn big(n: usize) -> String {
        "x".repeat(n)
    }

    #[test]
    fn a_plain_range_becomes_a_tier_one_block() {
        let mut state = create_initial_state();
        let msgs = vec![user("t1.1", &big(400)), user("t1.2", &big(400))];
        let out = apply_compression(
            &mut state,
            &msgs,
            &range(&["t1.1", "t1.2"], &[]),
            "a summary that is long enough to pass the minimum length check",
            &no_protection(),
            &cfg(),
            None,
            0,
        )
        .unwrap();

        assert_eq!(out.tier, Tier::One);
        assert_eq!(out.message_count, 2);
        assert_eq!(state.blocks.len(), 1);
        assert_eq!(state.blocks[0].block_id, "b1");
        assert!(state.blocks[0].active);
        assert_eq!(state.stats.compression_count, 1);
        assert!(state.stats.tokens_compressed > 0);
    }

    #[test]
    fn a_protected_message_is_skipped_not_fatal() {
        let mut state = create_initial_state();
        let msgs = vec![user("t1.1", &big(400)), user("t1.2", &big(400))];
        let mut prot = Protection::default();
        prot.protect_base("t1.1");

        let out = apply_compression(
            &mut state,
            &msgs,
            &range(&["t1.1", "t1.2"], &[]),
            "a summary that is long enough to pass the minimum length check",
            &prot,
            &cfg(),
            None,
            0,
        )
        .unwrap();

        assert_eq!(out.message_count, 1, "only the unprotected one folded");
        assert_eq!(state.blocks[0].direct_message_ids, vec!["t1.2"]);
    }

    #[test]
    fn an_entirely_protected_range_is_refused_with_advice() {
        let mut state = create_initial_state();
        let msgs = vec![user("t1.1", &big(400))];
        let mut prot = Protection::default();
        prot.protect_base("t1.1");

        let err = apply_compression(
            &mut state,
            &msgs,
            &range(&["t1.1"], &[]),
            "a summary that is long enough to pass the minimum length check",
            &prot,
            &cfg(),
            None,
            0,
        )
        .unwrap_err();
        assert_eq!(err, CompressError::AllProtected);
        assert!(err.to_string().contains("acp_status"));
    }

    #[test]
    fn a_too_short_summary_is_refused() {
        let mut state = create_initial_state();
        let msgs = vec![user("t1.1", &big(400))];
        let err = apply_compression(
            &mut state,
            &msgs,
            &range(&["t1.1"], &[]),
            "too short",
            &no_protection(),
            &cfg(),
            None,
            0,
        )
        .unwrap_err();
        assert!(matches!(err, CompressError::SummarySize { .. }));
    }

    #[test]
    fn an_oversized_summary_is_refused() {
        let mut state = create_initial_state();
        let msgs = vec![user("t1.1", &big(400))];
        let mut c = cfg();
        c.compress.max_summary_tokens = 10;
        let err = apply_compression(
            &mut state,
            &msgs,
            &range(&["t1.1"], &[]),
            &big(400),
            &no_protection(),
            &c,
            None,
            0,
        )
        .unwrap_err();
        assert!(matches!(err, CompressError::SummarySize { .. }));
    }

    #[test]
    fn a_range_below_the_minimum_is_refused() {
        let mut state = create_initial_state();
        let msgs = vec![user("t1.1", "tiny")];
        let mut c = cfg();
        c.compress.min_range_tokens = 5_000;
        let err = apply_compression(
            &mut state,
            &msgs,
            &range(&["t1.1"], &[]),
            "a summary that is long enough to pass the minimum length check",
            &no_protection(),
            &c,
            None,
            0,
        )
        .unwrap_err();
        assert!(matches!(err, CompressError::RangeTooSmall { .. }));
    }

    #[test]
    fn folding_a_block_promotes_the_tier_and_consumes_it() {
        let mut state = create_initial_state();
        let msgs = vec![user("t1.1", &big(400)), user("t1.2", &big(400))];
        let c = cfg();

        // tier 1
        apply_compression(
            &mut state,
            &msgs,
            &range(&["t1.1"], &[]),
            "first summary that is long enough to pass the minimum length check",
            &no_protection(),
            &c,
            None,
            0,
        )
        .unwrap();

        // tier 2, consuming b1
        let out = apply_compression(
            &mut state,
            &msgs,
            &range(&[], &["b1"]),
            "second summary that is long enough to pass the minimum length check",
            &no_protection(),
            &c,
            None,
            0,
        )
        .unwrap();

        assert_eq!(out.tier, Tier::Two);
        assert_eq!(out.consumed_block_ids, vec!["b1"]);
        assert!(!state.blocks[0].active, "b1 was consumed");
        assert!(state.blocks[1].active);
        assert_eq!(state.blocks[1].direct_block_ids, vec!["b1"]);
    }

    #[test]
    fn a_consumed_blocks_coverage_is_inherited() {
        let mut state = create_initial_state();
        let msgs = vec![user("t1.1", &big(400))];
        let c = cfg();

        apply_compression(
            &mut state,
            &msgs,
            &range(&["t1.1"], &[]),
            "first summary that is long enough to pass the minimum length check",
            &no_protection(),
            &c,
            None,
            0,
        )
        .unwrap();
        apply_compression(
            &mut state,
            &msgs,
            &range(&[], &["b1"]),
            "second summary that is long enough to pass the minimum length check",
            &no_protection(),
            &c,
            None,
            0,
        )
        .unwrap();

        // t1.1 is still covered, now via b2
        assert!(state.blocks[1].effective_message_ids.contains(&"t1.1".to_string()));
        assert!(state
            .covered()
            .contains("t1.1"));
    }

    #[test]
    fn decompress_restores_visibility_and_is_marked_deliberate() {
        let mut state = create_initial_state();
        let msgs = vec![user("t1.1", &big(400))];
        apply_compression(
            &mut state,
            &msgs,
            &range(&["t1.1"], &[]),
            "a summary that is long enough to pass the minimum length check",
            &no_protection(),
            &cfg(),
            None,
            0,
        )
        .unwrap();
        assert!(state.covered().contains("t1.1"));

        let restored = decompress_block(&mut state, "b1").unwrap();
        assert!(!restored.active);
        assert!(restored.expanded, "a deliberate restore must be recorded");
        assert!(!state.covered().contains("t1.1"));
    }

    #[test]
    fn decompressing_an_unknown_block_is_an_error() {
        let mut state = create_initial_state();
        let err = decompress_block(&mut state, "b9").unwrap_err();
        assert_eq!(err, CompressError::UnknownBlock("b9".to_string()));
    }

    #[test]
    fn tier_two_distillation_fires_on_the_configured_count() {
        let mut state = create_initial_state();
        let mut c = default_config(200_000);
        c.tiers.tier2_trigger = 2;
        assert_eq!(distillation_due(&state, &c), None);

        let msgs: Vec<CoreMessage> = (1..=2).map(|i| user(&format!("t1.{i}"), &big(400))).collect();
        for i in 1..=2 {
            apply_compression(
                &mut state,
                &msgs,
                &range(&[&format!("t1.{i}")], &[]),
                "a summary that is long enough to pass the minimum length check",
                &no_protection(),
                &c,
                None,
                0,
            )
            .unwrap();
        }
        assert_eq!(distillation_due(&state, &c), Some(Tier::Two));
    }

    #[test]
    fn distillation_is_off_when_tiers_are_disabled() {
        let mut state = create_initial_state();
        let mut c = default_config(200_000);
        c.tiers.enabled = false;
        c.tiers.tier2_trigger = 1;
        let msgs = vec![user("t1.1", &big(400))];
        apply_compression(
            &mut state,
            &msgs,
            &range(&["t1.1"], &[]),
            "a summary that is long enough to pass the minimum length check",
            &no_protection(),
            &c,
            None,
            0,
        )
        .unwrap();
        assert_eq!(distillation_due(&state, &c), None);
    }

    #[test]
    fn protection_from_the_recent_window_blocks_a_fold() {
        let mut state = create_initial_state();
        let mut c = default_config(200_000);
        c.compress.min_range_tokens = 0;
        // default window protects everything here
        let msgs = vec![user("t1.1", &big(400))];
        let prot = compute_protection(&msgs, &state, &c);
        let err = apply_compression(
            &mut state,
            &msgs,
            &range(&["t1.1"], &[]),
            "a summary that is long enough to pass the minimum length check",
            &prot,
            &c,
            None,
            0,
        )
        .unwrap_err();
        assert_eq!(err, CompressError::AllProtected);
    }

    #[test]
    fn range_bases_normalises_projections() {
        let r = range(&["t1.1#r0", "t1.2"], &[]);
        let bases = range_bases(&r);
        assert!(bases.contains("t1.1"));
        assert!(bases.contains("t1.2"));
    }
}
