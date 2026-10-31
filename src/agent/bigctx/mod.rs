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
pub mod cache_report;
pub mod compress;
pub mod config;
pub mod crush;
pub mod hide_consumed;
pub mod host;
pub mod protected;
pub mod nudge;
pub mod nudge_text;
pub mod pipeline;
pub mod prune;
pub mod recommend;
pub mod render_refs;
pub mod report;
pub mod rules;
pub mod refs;
pub mod state;
pub mod sync;
pub mod tokenize;
pub mod tools;

#[cfg(test)]
mod e2e_tests;
pub mod turn_integrity;
pub mod truncate;
pub mod types;

// Only the names the host actually reaches for are re-exported. The modules
// are public, so anything else stays one `bigctx::<module>::` away — a
// re-export nobody calls is a name that rots silently and shows up as an
// unused-import warning on every build.
pub use boundaries::resolve_boundaries;
pub use cache_report::{
    CacheLedger, CacheReport, CacheSample, FoldEvent, build_cache_report, render_cache_report,
};
pub use compress::{apply_compression, decompress_block};
pub use config::{BigctxConfig, config_from_json, default_config};
pub use crush::{CrushResult, crush_tool_results};
pub use hide_consumed::hide_consumed_compress_calls;
pub use host::{IdScheme, project, render};
pub use nudge::{NudgeDecision, decide_nudge, reset_after_compression, stamp_nudge};
pub use nudge_text::render_nudge;
pub use pipeline::{process_turn, process_turn_at};
pub use protected::{Protection, compute_protection};
pub use prune::{PruneOptions, prune};
pub use recommend::{ContextRanges, screen};
pub use render_refs::{RenderStrategy, render_refs};
pub use report::{Report, build_report, render_report, status_line, tool_share};
pub use refs::assign_refs;
pub use state::{CompressionStateExt, create_initial_state};
pub use tools::{BigctxHandle, BigctxSession, bigctx_tools};
pub use truncate::{TruncateOptions, TruncateResult, truncate_large_outputs};
pub use types::{
    BlockGeneration, CompressionBlock, CompressionState, CoreMessage, MessageContentType,
    MessageRole, Tier,
};
