//! Block lifecycle: allocation, coverage, tier bookkeeping.
//!
//! A block is the durable record of one compressed range. Two id sets matter
//! and they are deliberately different:
//!
//! - `direct_message_ids` — what *this* fold covered. Used for anchoring, so a
//!   re-fold rewrites only the region it actually rewrote.
//! - `effective_message_ids` — direct ids plus everything inherited from
//!   consumed children. Used for coverage, so a message folded three tiers ago
//!   is still recognised as folded.

use crate::agent::bigctx::types::{
    BlockGeneration, CompressionBlock, CompressionState, MessageRefMap, Tier, base_id_of,
};

/// A fresh, empty state.
pub fn create_initial_state() -> CompressionState {
    CompressionState {
        blocks: Vec::new(),
        message_refs: MessageRefMap::default(),
        token_snapshot: Default::default(),
        nudge: Default::default(),
        stats: Default::default(),
        next_block_id: 1,
        next_run_id: 1,
        next_ref_index: 1,
    }
}

/// `b1`, `b2`, … — monotonic, never reused.
pub fn allocate_block_id(state: &mut CompressionState) -> String {
    let id = state.next_block_id.max(1);
    state.next_block_id = id + 1;
    format!("b{id}")
}

/// `r1`, `r2`, … — monotonic, never reused.
pub fn allocate_run_id(state: &mut CompressionState) -> String {
    let id = state.next_run_id.max(1);
    state.next_run_id = id + 1;
    format!("r{id}")
}

/// Look a block up by id.
pub fn block_by_id<'a>(state: &'a CompressionState, block_id: &str) -> Option<&'a CompressionBlock> {
    state.blocks.iter().find(|b| b.block_id == block_id)
}

/// Every active block, in creation order.
pub fn active_blocks(state: &CompressionState) -> impl Iterator<Item = &CompressionBlock> {
    state.blocks.iter().filter(|b| b.active)
}

/// Count of active blocks.
pub fn active_block_count(state: &CompressionState) -> usize {
    state.blocks.iter().filter(|b| b.active).count()
}

/// Ids covered by any active block, base-normalised.
///
/// Base-normalised because coverage is decided per original message: a range
/// that folded `t12.3#r0` folded `t12.3`.
pub fn covered_message_ids(state: &CompressionState) -> std::collections::BTreeSet<String> {
    let mut covered = std::collections::BTreeSet::new();
    for block in active_blocks(state) {
        for id in &block.effective_message_ids {
            covered.insert(base_id_of(id).to_string());
        }
    }
    covered
}

/// Deepest tier currently live, or `None` when nothing is compressed.
pub fn highest_active_tier(state: &CompressionState) -> Option<Tier> {
    active_blocks(state).map(|b| b.tier).max()
}

/// Count of active blocks at exactly `tier`.
pub fn active_blocks_at_tier(state: &CompressionState, tier: Tier) -> usize {
    state
        .blocks
        .iter()
        .filter(|b| b.active && b.tier == tier)
        .count()
}

/// Bump every active block's survival counter, promoting to `Old` at the
/// threshold. Called once per pass; this is the only writer of `generation`.
pub fn advance_survival(state: &mut CompressionState, promotion_threshold: u32) {
    for block in state.blocks.iter_mut() {
        if !block.active {
            continue;
        }
        block.survived_count = block.survived_count.saturating_add(1);
        if block.survived_count >= promotion_threshold {
            block.generation = BlockGeneration::Old;
        }
    }
}

/// Helpers the host calls on a state; kept as an extension trait so the plain
/// struct stays a serde-shaped record.
pub trait CompressionStateExt {
    fn active_blocks(&self) -> impl Iterator<Item = &CompressionBlock>;
    fn covered(&self) -> std::collections::BTreeSet<String>;
    fn is_covered(&self, id: &str) -> bool;
}

impl CompressionStateExt for CompressionState {
    fn active_blocks(&self) -> impl Iterator<Item = &CompressionBlock> {
        self.blocks.iter().filter(|b| b.active)
    }

    fn covered(&self) -> std::collections::BTreeSet<String> {
        covered_message_ids(self)
    }

    fn is_covered(&self, id: &str) -> bool {
        let base = base_id_of(id);
        self.active_blocks()
            .any(|b| b.effective_message_ids.iter().any(|i| base_id_of(i) == base))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(id: &str, tier: Tier, covers: &[&str]) -> CompressionBlock {
        CompressionBlock {
            block_id: id.to_string(),
            run_id: "r1".to_string(),
            tier,
            topic: None,
            summary: "s".to_string(),
            direct_message_ids: covers.iter().map(|s| s.to_string()).collect(),
            effective_message_ids: covers.iter().map(|s| s.to_string()).collect(),
            direct_block_ids: Vec::new(),
            compressed_tokens: 0,
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
    fn block_ids_are_monotonic_and_never_reused() {
        let mut state = create_initial_state();
        assert_eq!(allocate_block_id(&mut state), "b1");
        assert_eq!(allocate_block_id(&mut state), "b2");
        assert_eq!(state.next_block_id, 3);
    }

    #[test]
    fn run_ids_are_monotonic() {
        let mut state = create_initial_state();
        assert_eq!(allocate_run_id(&mut state), "r1");
        assert_eq!(allocate_run_id(&mut state), "r2");
    }

    #[test]
    fn coverage_ignores_inactive_blocks() {
        let mut state = create_initial_state();
        state.blocks.push(block("b1", Tier::One, &["t1.1"]));
        let mut dead = block("b2", Tier::One, &["t1.2"]);
        dead.active = false;
        state.blocks.push(dead);

        let covered = covered_message_ids(&state);
        assert!(covered.contains("t1.1"));
        assert!(!covered.contains("t1.2"));
    }

    #[test]
    fn coverage_is_base_normalised() {
        let mut state = create_initial_state();
        state.blocks.push(block("b1", Tier::One, &["t1.1#r0"]));
        assert!(state.is_covered("t1.1"));
        assert!(state.is_covered("t1.1#call_9"));
        assert!(!state.is_covered("t1.2"));
    }

    #[test]
    fn highest_active_tier_tracks_depth() {
        let mut state = create_initial_state();
        assert_eq!(highest_active_tier(&state), None);
        state.blocks.push(block("b1", Tier::One, &["a"]));
        assert_eq!(highest_active_tier(&state), Some(Tier::One));
        state.blocks.push(block("b2", Tier::Two, &["b"]));
        assert_eq!(highest_active_tier(&state), Some(Tier::Two));
        state.blocks[1].active = false;
        assert_eq!(highest_active_tier(&state), Some(Tier::One));
    }

    #[test]
    fn survival_promotes_to_old_at_the_threshold() {
        let mut state = create_initial_state();
        state.blocks.push(block("b1", Tier::One, &["a"]));
        for _ in 0..2 {
            advance_survival(&mut state, 3);
            assert_eq!(state.blocks[0].generation, BlockGeneration::Young);
        }
        advance_survival(&mut state, 3);
        assert_eq!(state.blocks[0].survived_count, 3);
        assert_eq!(state.blocks[0].generation, BlockGeneration::Old);
    }

    #[test]
    fn survival_skips_inactive_blocks() {
        let mut state = create_initial_state();
        let mut dead = block("b1", Tier::One, &["a"]);
        dead.active = false;
        state.blocks.push(dead);
        advance_survival(&mut state, 1);
        assert_eq!(state.blocks[0].survived_count, 0);
    }

    #[test]
    fn tier_counts_only_active() {
        let mut state = create_initial_state();
        state.blocks.push(block("b1", Tier::One, &["a"]));
        state.blocks.push(block("b2", Tier::One, &["b"]));
        let mut dead = block("b3", Tier::One, &["c"]);
        dead.active = false;
        state.blocks.push(dead);
        assert_eq!(active_blocks_at_tier(&state, Tier::One), 2);
        assert_eq!(active_block_count(&state), 2);
    }
}
