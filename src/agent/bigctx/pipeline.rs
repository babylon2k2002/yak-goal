//! One turn, start to finish.
//!
//! The host hands in the visible messages and the state; this module walks the
//! same node sequence the upstream kernel uses and returns the view the model
//! should see. Every node is a pure function of `(messages, state, effects)`,
//! which is what makes the sequence testable one node at a time.
//!
//! The nodes that matter for correctness, in order:
//!
//! 1. **assign-refs** — give every message a citation ref, blocking the ones
//!    that must never be folded.
//! 2. **sync-blocks** — reconcile block activity against what is actually
//!    visible, then age the blocks one pass.
//! 3. **prune** — rebuild the view with folded material replaced by summaries.
//! 4. **recommend** — decide which ranges are worth folding, and which are
//!    protected.
//! 5. **turn-integrity** — withdraw any fold that would split a turn or a
//!    call/result pair.
//!
//! Deliberately absent, for now: the nudge cadence, emergency truncation, the
//! absorb/crush pair, and ref rendering. Those are model-facing UX or
//! independent features; they do not change what a fold *is*. They slot in as
//! further nodes without touching this skeleton.

use std::collections::BTreeSet;

use crate::agent::bigctx::config::BigctxConfig;
use crate::agent::bigctx::prune::{PruneOptions, prune};
use crate::agent::bigctx::protected::{Protection, compute_protection};
use crate::agent::bigctx::sync::{advance_survival, sync_blocks};
use crate::agent::bigctx::turn_integrity::{Withdrawals, withdrawals};
use crate::agent::bigctx::types::{CompressionState, CoreMessage};

/// What one pass produced.
#[derive(Debug, Clone)]
pub struct TurnResult {
    /// The view the model should see.
    pub messages: Vec<CoreMessage>,
    /// The state to carry into the next pass.
    pub state: CompressionState,
    /// What may not be folded this pass.
    pub protection: Protection,
    /// Blocks whose material vanished from the view.
    pub deactivated: Vec<String>,
    /// Ids a requested fold could not take without splitting a turn or pair.
    pub withdrawals: Withdrawals,
    /// Whether to tell the model that compressing would help, and why.
    pub nudge: crate::agent::bigctx::NudgeDecision,
    /// How many tool results were mechanically crushed this pass.
    pub crushed: usize,
    /// How many payloads the last-resort truncation cut, and why it may have
    /// cut none when the session is over its window.
    pub truncated: crate::agent::bigctx::TruncateResult,
    /// Tool results stored out of the context, and placeholders re-projected.
    pub ccr: crate::agent::bigctx::CcrResult,
}

/// Run one pass over `messages`.
///
/// `messages` is the *full* resent history, not a pruned view: turn grouping
/// and call/result pairing are array-dependent, so judging a fold against a
/// pruned view withdraws different messages than applying it does.
pub fn process_turn(
    messages: &[CoreMessage],
    state: &CompressionState,
    config: &BigctxConfig,
) -> TurnResult {
    process_turn_with_store(messages, state, config, 0, None, 0)
}

/// As [`process_turn`], with the current token count so the nudge can judge
/// pressure, and a content store for oversized tool results. A caller that does
/// not know the size passes 0 and gets no nudge rather than a wrong one.
pub fn process_turn_with_store(
    messages: &[CoreMessage],
    state: &CompressionState,
    config: &BigctxConfig,
    token_count: u64,
    store: Option<&mut crate::agent::bigctx::ContentStore>,
    round: u64,
) -> TurnResult {
    // 0. Crush oversized tool results first. It has to happen before pruning:
    //    once a result is folded into a summary its text is gone, and a crush
    //    that only sees the summary saves nothing. Deterministic, so the same
    //    bytes come out on every turn a result is re-sent.
    let crushed = if config.crush.enabled {
        crate::agent::bigctx::crush_tool_results(messages, config.crush.min_reduction)
    } else {
        crate::agent::bigctx::CrushResult::unchanged(messages)
    };
    let messages = &crushed.messages[..];


    // 0a. assign-refs. Protection is computed from the same rules the gate uses
    //     later, so a message blocked here is blocked consistently. This has to
    //     precede CCR: a placeholder names the ref of the message it replaces.
    let mut next = state.clone();
    let protection = compute_protection(messages, &next, config);
    sync_refs(&mut next, messages, &protection);

    // 0b. Content-cached retrieval: keep oversized tool results out of the
    //     context without losing them. It MUST run before prune: once a result
    //     is folded into a summary its text is gone and there is nothing left to
    //     store. It also runs at arrival, so the bytes are frozen the first time
    //     — a placeholder that changed per turn would move the prefix and
    //     invalidate the cache it exists to protect.
    let ccr = if config.ccr.enabled
        && let Some(store) = store
    {
        crate::agent::bigctx::ccr::project(
            messages,
            &mut next,
            store,
            config.ccr.min_tool_tokens,
        )
    } else {
        crate::agent::bigctx::CcrResult {
            messages: messages.to_vec(),
            stored: 0,
            reprojected: 0,
        }
    };
    let messages = &ccr.messages[..];
    if ccr.stored > 0 || ccr.reprojected > 0 {
        sync_refs(&mut next, messages, &protection);
    }

    // 1. sync-blocks, then age. Ageing before the fold decision means a block
    //    that has survived long enough is already eligible for distillation.
    let (synced, sync_result) = sync_blocks(messages, &next);
    next = synced;
    advance_survival(&mut next, config.promotion_threshold);

    // 2. prune. Summaries are injected so the model can see what it folded.
    let view = prune(
        messages,
        &next,
        &PruneOptions {
            inject_summaries: true,
            ..Default::default()
        },
    );


    // 2b. Hide compress calls whose blocks are gone, and stub the summaries a
    //     live call repeats. A long session otherwise pays for the receipts:
    //     the arguments duplicate every summary the block already holds.
    let hidden = crate::agent::bigctx::hide_consumed_compress_calls(&next, &view);
    let view = hidden.messages;

    // 3. The fold gate runs against the full array, not the view, so that a
    //      requested range is judged the way it will be applied.
    let requested = collect_fold_candidates(&view, &next, &protection);
    let gate = withdrawals(messages, &requested);

    // 4. nudge: does the model need telling? The pending mass is measured from
    //    the same screening the status tool reports, so the number in the nudge
    //    and the number in `acp_status` cannot disagree.
    // Screened against the FULL array, not the pruned view. A folded message is
    // gone from the view but still present here, and `screen` is what decides
    // what may be folded next — running it on the view would only ever see what
    // survived, so a session whose whole visible tail is protected reports
    // "nothing compressible" even when most of the history is foldable.
    let ranges = crate::agent::bigctx::screen(messages, &next, config);
    let tier1_pending = ranges.compressible_tokens();
    let mut next = next;
    let nudge = crate::agent::bigctx::decide_nudge(
        &next,
        config,
        token_count,
        [tier1_pending, 0, 0],
    );
    crate::agent::bigctx::stamp_nudge(&mut next, &nudge, token_count, round);

    // 5. emergency truncate: the valve for a session that ignored the nudge.
    //    Runs last because everything above is preferable — a fold keeps the
    //    originals, a truncation does not. Only fires above the threshold.
    let truncated = crate::agent::bigctx::truncate_large_outputs(
        &view,
        token_count,
        config,
        &crate::agent::bigctx::TruncateOptions {
            protect_recent_messages: config.preserve_recent_messages,
            include_text_messages: true,
            ..Default::default()
        },
    );
    let view = truncated.messages.clone();

    // 6. render-refs: write each message's citation ref into its text.
    //
    // Without this the model cannot name a message it wants to fold without
    // first asking acp_status — a round trip on every compression, and the
    // reason the feature read as something you look up rather than something
    // you do. Runs last so it tags the final view, and the token snapshot it
    // writes is frozen at first render so the prefix cache stays stable.
    let view = crate::agent::bigctx::render_refs(
        &view,
        &mut next,
        crate::agent::bigctx::RenderStrategy::All,
    );

    TurnResult {
        messages: view,
        state: next,
        protection,
        deactivated: sync_result.deactivated,
        withdrawals: gate,
        nudge,
        crushed: crushed.crushed,
        truncated,
        ccr,
    }
}

/// The ids currently folded away, for the gate to check.
///
/// This is the set a fold *would* remove: everything an active block covers,
/// restricted to ids that are actually present. It exists so the caller can see
/// whether the current folding is self-consistent — a block that covers half a
/// turn is a latent 400 the moment the provider replays it.
fn collect_fold_candidates(
    _view: &[CoreMessage],
    state: &CompressionState,
    _protection: &Protection,
) -> BTreeSet<String> {
    let mut folded = BTreeSet::new();
    for block in &state.blocks {
        if !block.active {
            continue;
        }
        for id in &block.effective_message_ids {
            folded.insert(id.clone());
        }
    }
    folded
}

/// Assign refs to every message, blocking the protected ones.
///
/// Runs in place on `state` so refs are never re-issued: a message that was
/// folded last turn keeps the ref it had, which is what lets a later restore
/// name it.
fn sync_refs(state: &mut CompressionState, messages: &[CoreMessage], protection: &Protection) {
    let mut next_index = state.next_ref_index;
    let outcome = crate::agent::bigctx::refs::assign_refs(
        messages,
        &mut state.message_refs,
        &mut next_index,
        |message| protection.is_protected(message),
    );
    if outcome.is_ok() {
        state.next_ref_index = next_index;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::bigctx::config::default_config;
    use crate::agent::bigctx::prune::is_rendered_summary;
    use crate::agent::bigctx::state::create_initial_state;
    use crate::agent::bigctx::types::{
        BlockGeneration, CompressionBlock, MessageContentType, MessageRole, Tier,
    };

    fn user(id: &str, text: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::User, MessageContentType::Text).with_text(text)
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

    fn block(id: &str, covers: &[&str]) -> CompressionBlock {
        CompressionBlock {
            block_id: id.to_string(),
            run_id: "r1".to_string(),
            tier: Tier::One,
            topic: None,
            start_ref: None,
            end_ref: None,
            summary: "a summary".to_string(),
            direct_message_ids: covers.iter().map(|s| s.to_string()).collect(),
            effective_message_ids: covers.iter().map(|s| s.to_string()).collect(),
            direct_block_ids: vec![],
            compressed_tokens: 100,
            created_at: 0,
            survived_count: 0,
            generation: BlockGeneration::Young,
            active: true,
            expanded: false,
            duration_ms: None,
            compress_call_id: None,
        }
    }

    #[test]
    fn an_empty_state_leaves_the_view_untouched() {
        let cfg = default_config(200_000);
        let msgs = vec![user("t1.0", "hello"), user("t1.1", "again")];
        let out = process_turn(&msgs, &create_initial_state(), &cfg);
        assert_eq!(out.messages.len(), 2);
        assert!(out.deactivated.is_empty());
        assert!(out.withdrawals.is_empty());
    }

    #[test]
    fn a_fold_replaces_its_material_with_a_summary() {
        let cfg = default_config(200_000);
        let mut state = create_initial_state();
        state.blocks.push(block("b1", &["t1.1"]));
        // t1.0 is the first user message, and prune pins that one: folding it
        // would lose the task statement and move the prefix-cache anchor.
        let msgs = vec![user("t1.0", "the task"), user("t1.1", "folded away")];

        let out = process_turn(&msgs, &state, &cfg);
        assert!(
            out.messages.iter().any(is_rendered_summary),
            "the folded material must be replaced by a summary"
        );
        assert!(
            !out.messages.iter().any(|m| m.id == "t1.1"),
            "the folded message must be gone from the view"
        );
        assert!(
            out.messages.iter().any(|m| m.id == "t1.0"),
            "the first user message is pinned"
        );
    }

    #[test]
    fn refs_are_assigned_on_the_first_pass_and_kept() {
        let cfg = default_config(200_000);
        let msgs = vec![user("t1.0", "a")];
        let first = process_turn(&msgs, &create_initial_state(), &cfg);
        let reference = first.state.message_refs.ref_for("t1.0").map(str::to_string);
        assert!(reference.is_some());

        // a second pass must not re-issue
        let second = process_turn(&msgs, &first.state, &cfg);
        assert_eq!(
            second.state.message_refs.ref_for("t1.0").map(str::to_string),
            reference
        );
    }

    #[test]
    fn survival_advances_each_pass() {
        let mut cfg = default_config(200_000);
        cfg.promotion_threshold = 2;
        let mut state = create_initial_state();
        state.blocks.push(block("b1", &["t1.0"]));
        let msgs = vec![user("t1.0", "x"), user("t1.1", "y")];

        let first = process_turn(&msgs, &state, &cfg);
        assert_eq!(first.state.blocks[0].survived_count, 1);
        let second = process_turn(&msgs, &first.state, &cfg);
        assert_eq!(second.state.blocks[0].survived_count, 2);
        assert_eq!(
            second.state.blocks[0].generation,
            BlockGeneration::Old,
            "a block that keeps surviving is eligible for distillation"
        );
    }

    #[test]
    fn a_vanished_block_is_reported_deactivated() {
        let cfg = default_config(200_000);
        let mut state = create_initial_state();
        state.blocks.push(block("b1", &["t1.9"]));
        let msgs = vec![user("t1.0", "only this")];
        let out = process_turn(&msgs, &state, &cfg);
        assert_eq!(out.deactivated, vec!["b1".to_string()]);
    }

    #[test]
    fn the_gate_flags_a_fold_that_would_split_a_pair() {
        // A block covering only the call of a call/result exchange is a latent
        // 400: the result would be replayed without its call.
        let cfg = default_config(200_000);
        let mut state = create_initial_state();
        state.blocks.push(block("b1", &["t1.1"]));
        let msgs = vec![user("t1.0", "go"), call("t1.1", "c1"), result("t1.2", "c1")];

        let out = process_turn(&msgs, &state, &cfg);
        assert!(
            out.withdrawals.keeps("t1.1"),
            "the call cannot fold without its result"
        );
        assert!(out.withdrawals.keeps("t1.2"));
    }

    #[test]
    fn a_consistent_fold_passes_the_gate_cleanly() {
        let cfg = default_config(200_000);
        let mut state = create_initial_state();
        state.blocks.push(block("b1", &["t1.1", "t1.2"]));
        let msgs = vec![user("t1.0", "go"), call("t1.1", "c1"), result("t1.2", "c1")];

        let out = process_turn(&msgs, &state, &cfg);
        assert!(
            out.withdrawals.is_empty(),
            "folding a whole exchange is legal: {:?}",
            out.withdrawals
        );
    }

    #[test]
    fn protection_is_reported_for_the_caller() {
        let cfg = default_config(200_000);
        let msgs = vec![user("t1.0", "the task")];
        let out = process_turn(&msgs, &create_initial_state(), &cfg);
        assert!(
            out.protection.is_protected(&msgs[0]),
            "the recent window protects the only message"
        );
    }
}
