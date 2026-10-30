//! Block activity, recomputed against what is actually visible.
//!
//! A block is `active` when the material it covers is still folded away. That
//! is not a fact the state can know on its own: the host hands us the message
//! list every turn, and the two can disagree. Three disagreements are real and
//! each has a distinct resolution:
//!
//! - **Consumed by a parent.** A tier-2 block swallowed a tier-1 block. The
//!   child is inactive because its content now lives in the parent — not
//!   because anything was restored. Never reactivate it.
//! - **Deliberately restored.** The user asked to see a block's material, so
//!   the block is inactive *on purpose*. Reactivating it would re-fold what the
//!   user just asked for: double cost and lost originals.
//! - **Vanished from the view.** The messages it covered are gone and its
//!   rendered summary is not there either — the host dropped the material
//!   entirely, so the block no longer represents anything.
//!
//! Everything else is active, including a block whose raw messages have been
//! replaced by its own rendered summary: the summary *is* the block's visible
//! representation.

use std::collections::BTreeSet;

use crate::agent::bigctx::prune::summary_message_id;
use crate::agent::bigctx::types::{CompressionState, CoreMessage, base_id_of};

/// What a sync changed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SyncResult {
    /// Blocks that were active and are now not, because their material is gone.
    pub deactivated: Vec<String>,
    /// Blocks that were inactive and are now active again.
    pub reactivated: Vec<String>,
}

/// Recompute block activity against `messages`, returning a new state.
///
/// The input state is cloned rather than mutated: the caller's snapshot must
/// not change under it, and the block sub-arrays must not alias the input.
pub fn sync_blocks(messages: &[CoreMessage], state: &CompressionState) -> (CompressionState, SyncResult) {
    let mut next = state.clone();
    let mut result = SyncResult::default();

    let present_ids: BTreeSet<&str> = messages.iter().map(|m| m.id.as_str()).collect();
    let present_bases: BTreeSet<&str> = messages
        .iter()
        .map(|m| base_id_of(&m.id))
        .filter(|base| !base.is_empty())
        .collect();

    prune_token_snapshot(&mut next, messages);

    // Every block a parent consumed, so a consumed child is never reactivated
    // by the "still present" check below.
    let consumed: BTreeSet<String> = next
        .blocks
        .iter()
        .flat_map(|block| block.direct_block_ids.iter().cloned())
        .collect();

    for block in &mut next.blocks {
        if consumed.contains(&block.block_id) {
            block.active = false;
            continue;
        }
        if block.expanded {
            // Deliberately restored. Keep it inactive; a fresh compress of the
            // same range creates a NEW block.
            block.active = false;
            continue;
        }

        block.active = true;
        // A block whose raw messages were replaced by its rendered summary is
        // still present — the summary IS its visible representation. Without
        // this, a host that passes a pruned view would lose every block's
        // activity on the next turn.
        let still_present = block
            .effective_message_ids
            .iter()
            .any(|id| present_bases.contains(base_id_of(id)))
            || present_ids.contains(summary_message_id(&block.block_id).as_str());
        if !still_present {
            block.active = false;
            result.deactivated.push(block.block_id.clone());
        }
    }

    result.deactivated.sort();
    (next, result)
}

/// Drop token-snapshot entries whose refs no longer name a present message.
///
/// Refs are additive — `assign_refs` never removes them — so without this the
/// snapshot grows without bound across a long session.
fn prune_token_snapshot(next: &mut CompressionState, messages: &[CoreMessage]) {
    if next.token_snapshot.is_empty() {
        return;
    }
    let live_refs: BTreeSet<String> = messages
        .iter()
        .filter_map(|m| next.message_refs.by_raw.get(&m.id).cloned())
        .collect();
    if next.token_snapshot.len() == live_refs.len() {
        return;
    }
    next.token_snapshot
        .retain(|reference, _| live_refs.contains(reference));
}

/// Bump each active block's survival counter and promote it to `Old` once it
/// has survived `threshold` passes.
///
/// This is the signal tier-2 distillation uses: a block that keeps surviving is
/// one the session has not needed to restore, so its detail can be condensed
/// further.
pub fn advance_survival(state: &mut CompressionState, threshold: u32) {
    let threshold = threshold.max(1);
    for block in &mut state.blocks {
        if !block.active {
            continue;
        }
        block.survived_count = block.survived_count.saturating_add(1);
        if block.survived_count >= threshold {
            block.generation = crate::agent::bigctx::types::BlockGeneration::Old;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::bigctx::prune::summary_message_id;
    use crate::agent::bigctx::state::create_initial_state;
    use crate::agent::bigctx::types::{
        BlockGeneration, CompressionBlock, MessageContentType, MessageRole, Tier,
    };

    fn user(id: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::User, MessageContentType::Text).with_text("x")
    }

    fn block(id: &str, covers: &[&str], nested: &[&str]) -> CompressionBlock {
        CompressionBlock {
            block_id: id.to_string(),
            run_id: "r1".to_string(),
            tier: Tier::One,
            topic: None,
            summary: "s".to_string(),
            direct_message_ids: covers.iter().map(|s| s.to_string()).collect(),
            effective_message_ids: covers.iter().map(|s| s.to_string()).collect(),
            direct_block_ids: nested.iter().map(|s| s.to_string()).collect(),
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

    #[test]
    fn a_block_whose_material_is_present_stays_active() {
        let mut state = create_initial_state();
        state.blocks.push(block("b1", &["t1.0"], &[]));
        let msgs = vec![user("t1.0")];
        let (next, result) = sync_blocks(&msgs, &state);
        assert!(next.blocks[0].active);
        assert!(result.deactivated.is_empty());
    }

    #[test]
    fn a_consumed_block_is_never_reactivated() {
        let mut state = create_initial_state();
        state.blocks.push(block("b1", &["t1.0"], &[]));
        state.blocks.push(block("b2", &[], &["b1"]));
        state.blocks[0].active = false;
        let msgs = vec![user("t1.0")];
        let (next, _) = sync_blocks(&msgs, &state);
        assert!(!next.blocks[0].active, "b1 lives inside b2 now");
        assert!(next.blocks[1].active);
    }

    #[test]
    fn a_deliberately_restored_block_stays_down() {
        let mut state = create_initial_state();
        state.blocks.push(block("b1", &["t1.0"], &[]));
        state.blocks[0].active = false;
        state.blocks[0].expanded = true;
        let msgs = vec![user("t1.0")];
        let (next, _) = sync_blocks(&msgs, &state);
        assert!(
            !next.blocks[0].active,
            "reactivating would re-fold what the user asked to see"
        );
    }

    #[test]
    fn a_block_whose_material_vanished_is_deactivated() {
        let mut state = create_initial_state();
        state.blocks.push(block("b1", &["t1.9"], &[]));
        let msgs = vec![user("t1.0")];
        let (next, result) = sync_blocks(&msgs, &state);
        assert!(!next.blocks[0].active);
        assert_eq!(result.deactivated, vec!["b1".to_string()]);
    }

    #[test]
    fn a_rendered_summary_keeps_its_block_alive() {
        let mut state = create_initial_state();
        state.blocks.push(block("b1", &["t1.0"], &[]));
        // the host passes a pruned view: raw message gone, summary present
        let msgs = vec![CoreMessage::new(
            summary_message_id("b1"),
            MessageRole::System,
            MessageContentType::Text,
        )
        .with_text("[Compressed conversation section]\nbody")];
        let (next, result) = sync_blocks(&msgs, &state);
        assert!(
            next.blocks[0].active,
            "the summary IS the block's visible representation"
        );
        assert!(result.deactivated.is_empty());
    }

    #[test]
    fn coverage_is_base_normalised() {
        let mut state = create_initial_state();
        state.blocks.push(block("b1", &["t1.0"], &[]));
        // only a projection of the base is present
        let msgs = vec![CoreMessage::new(
            "t1.0#c0",
            MessageRole::Assistant,
            MessageContentType::ToolCall,
        )
        .with_text("{}")
        .with_tool("read", "c1")];
        let (next, _) = sync_blocks(&msgs, &state);
        assert!(next.blocks[0].active);
    }

    #[test]
    fn survival_promotes_a_block_to_old() {
        let mut state = create_initial_state();
        state.blocks.push(block("b1", &["t1.0"], &[]));
        assert_eq!(state.blocks[0].generation, BlockGeneration::Young);
        for _ in 0..3 {
            advance_survival(&mut state, 3);
        }
        assert_eq!(state.blocks[0].generation, BlockGeneration::Old);
        assert_eq!(state.blocks[0].survived_count, 3);
    }

    #[test]
    fn survival_skips_inactive_blocks() {
        let mut state = create_initial_state();
        state.blocks.push(block("b1", &["t1.0"], &[]));
        state.blocks[0].active = false;
        advance_survival(&mut state, 1);
        assert_eq!(state.blocks[0].survived_count, 0);
    }

    #[test]
    fn the_input_state_is_not_mutated() {
        let mut state = create_initial_state();
        state.blocks.push(block("b1", &["t1.9"], &[]));
        let msgs = vec![user("t1.0")];
        let _ = sync_blocks(&msgs, &state);
        assert!(state.blocks[0].active, "the caller's snapshot is untouched");
    }
}
