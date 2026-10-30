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
    // 1. assign-refs. Protection is computed from the same rules the gate uses
    //    later, so a message blocked here is blocked consistently.
    let mut next = state.clone();
    let protection = compute_protection(messages, &next, config);
    sync_refs(&mut next, messages, &protection);

    // 2. sync-blocks, then age. Ageing before the fold decision means a block
    //    that has survived long enough is already eligible for distillation.
    let (synced, sync_result) = sync_blocks(messages, &next);
    next = synced;
    advance_survival(&mut next, config.promotion_threshold);

    // 3. prune. Summaries are injected so the model can see what it folded.
    let view = prune(
        messages,
        &next,
        &PruneOptions {
            inject_summaries: true,
            ..Default::default()
        },
    );

    // 4/5. The fold gate runs against the full array, not the view, so that a
    //      requested range is judged the way it will be applied.
    let requested = collect_fold_candidates(&view, &next, &protection);
    let gate = withdrawals(messages, &requested);

    TurnResult {
        messages: view,
        state: next,
        protection,
        deactivated: sync_result.deactivated,
        withdrawals: gate,
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
        state.blocks.push(block("b1", &["t1.0"]));
        let msgs = vec![user("t1.0", "folded away"), user("t1.1", "still here")];

        let out = process_turn(&msgs, &state, &cfg);
        assert!(
            out.messages.iter().any(is_rendered_summary),
            "the folded material must be replaced by a summary"
        );
        assert!(
            !out.messages.iter().any(|m| m.id == "t1.0"),
            "the original must be gone from the view"
        );
        assert!(out.messages.iter().any(|m| m.id == "t1.1"));
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
