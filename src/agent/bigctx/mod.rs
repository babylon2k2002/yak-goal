//! Big-context: model-driven, reversible context compression.
//!
//! Ported from the algorithm in `billion-context`'s kernel (MIT). The idea:
//! instead of hard-truncating a conversation when it outgrows the window, the
//! model compresses a *range* of messages into a summary block. The block keeps
//! the original message ids, so the content can be restored later
//! (`decompress`), searched without restoring (`search_context`), and distilled
//! further as the session grows (tier 1 → 2 → 3).
//!
//! The pieces, in dependency order:
//!
//! - [`types`] — the message/state vocabulary. `CoreMessage` is a flattened
//!   view of yak's [`crate::providers::Msg`]: one original message may project
//!   to several cores (an assistant turn with reasoning + text + tool calls),
//!   and every core carries a stable `id`.
//! - [`tokenize`] — CJK-aware token estimation.
//! - [`refs`] — the `m00001` citation refs the model uses to name a range.
//! - [`state`] — block ids, coverage, tier bookkeeping.
//! - [`config`] — thresholds, validated once.
//!
//! Everything here is pure: no I/O, no clock beyond `created_at` stamping, no
//! provider knowledge. The host (see [`super::compact`]) owns the mapping to
//! and from `Msg` and decides when to run.

pub mod boundaries;
pub mod compress;
pub mod config;
pub mod hide_consumed;
pub mod host;
pub mod protected;
pub mod nudge;
pub mod pipeline;
pub mod prune;
pub mod recommend;
pub mod refs;
pub mod state;
pub mod sync;
pub mod tokenize;
pub mod tools;
pub mod turn_integrity;
pub mod types;

pub use boundaries::{ResolvedRange, parse_boundary, resolve_boundaries};
pub use compress::{CompressError, CompressOutcome, apply_compression, decompress_block};
pub use config::{BigctxConfig, default_config};
pub use hide_consumed::{HideResult, hide_consumed_compress_calls};
pub use host::{IdScheme, Projection, project, render};
pub use nudge::{NudgeDecision, decide_nudge, reset_after_compression, stamp_nudge};
pub use pipeline::{TurnResult, process_turn};
pub use protected::{Protection, compute_protection};
pub use prune::{PruneOptions, prune, summary_message_id};
pub use recommend::{ContextRanges, CompressibleRange, ProtectedRange, build_context_ranges, screen};
pub use refs::{assign_refs, empty_ref_map, index_to_ref, ref_to_index};
pub use state::{
    CompressionStateExt, active_blocks, allocate_block_id, allocate_run_id, block_by_id,
    covered_message_ids, create_initial_state, highest_active_tier,
};
pub use sync::{SyncResult, advance_survival, sync_blocks};
pub use tokenize::{count_message_tokens, default_count_tokens};
pub use tools::{BigctxHandle, BigctxSession, bigctx_tools};
pub use turn_integrity::{Withdrawals, turn_groups, withdrawals};
pub use types::{
    BlockGeneration, CompressionBlock, CompressionState, CompressionStats, CoreMessage,
    MessageContentType, MessageRefMap, MessageRole, NudgeState, Tier,
};
