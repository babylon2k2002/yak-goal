//! Rendering a nudge into text the model reads.
//!
//! [`decide_nudge`] answers *whether* to speak; this answers *what to say*. The
//! two are separate because a decision is cheap to test and prose is not, and
//! because a host may want the decision (for `--json`) without the text.
//!
//! Two voices:
//!
//! - **gentle** — an efficiency suggestion. Explicitly *not* an overflow
//!   warning: saying "you are about to run out" when the context is at 50%
//!   teaches the model to distrust the signal.
//! - **emergency** — the context is actually near its limit and the model must
//!   act now.
//!
//! The rendered text always carries the citation syntax, because the most
//! common failure is a model that understands it should compress but writes the
//! call wrong.

use crate::agent::bigctx::config::BigctxConfig;
use crate::agent::bigctx::nudge::NudgeDecision;
use crate::agent::bigctx::recommend::ContextRanges;
use crate::agent::bigctx::rules::{DEFAULT_PROMPTS, Prompts};
use crate::agent::bigctx::types::{CompressionState, Tier};

/// Which register the nudge speaks in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NudgeVoice {
    /// Room left; compress to stay lean.
    Gentle,
    /// Near the limit; compress now.
    Emergency,
}

impl NudgeVoice {
    pub fn as_str(self) -> &'static str {
        match self {
            NudgeVoice::Gentle => "gentle",
            NudgeVoice::Emergency => "emergency",
        }
    }
}

/// A rendered nudge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedNudge {
    pub voice: NudgeVoice,
    pub text: String,
}

/// How many compressible ranges to list before summarising the rest.
///
/// Listing every range in a long session would make the nudge itself a
/// significant part of the context it is trying to save.
const MAX_RANGES_SHOWN: usize = 8;

/// Render a nudge, or `None` when the decision says not to inject one.
pub fn render_nudge(
    decision: &NudgeDecision,
    ranges: &ContextRanges,
    state: &CompressionState,
    config: &BigctxConfig,
) -> Option<RenderedNudge> {
    if !decision.inject {
        return None;
    }

    let usage = if config.model_context_limit > 0 {
        decision_pressure(decision)
    } else {
        0.0
    };
    let voice = if usage >= config.nudge.emergency_threshold_pct {
        NudgeVoice::Emergency
    } else {
        NudgeVoice::Gentle
    };

    Some(RenderedNudge {
        voice,
        text: render(decision, ranges, state, config, voice, DEFAULT_PROMPTS),
    })
}

/// The pressure the decision was made under.
///
/// Carried on the decision rather than recomputed: the renderer must describe
/// the same situation the decision judged, or a nudge can say "95% used" for a
/// decision that fired at 50%.
fn decision_pressure(decision: &NudgeDecision) -> f64 {
    decision.pressure
}

fn render(
    decision: &NudgeDecision,
    ranges: &ContextRanges,
    state: &CompressionState,
    config: &BigctxConfig,
    voice: NudgeVoice,
    prompts: Prompts,
) -> String {
    let mut out: Vec<String> = Vec::new();

    match voice {
        NudgeVoice::Emergency => {
            out.push(
                "⚠️ Context limit reached — compress now. Prioritize consumed tool outputs."
                    .to_string(),
            );
            out.push(String::new());
            out.push(prompts.compress_philosophy.to_string());
        }
        NudgeVoice::Gentle => {
            out.push(
                "This is an efficiency nudge to compress early and keep context lean — not an \
                 overflow warning. A separate, stronger alert will appear if the context is \
                 actually full."
                    .to_string(),
            );
            out.push(String::new());
            out.push(prompts.compress_philosophy.to_string());
        }
    }

    out.push(String::new());
    out.push(format!(
        "Context: {} of {} used ({:.0}%), {} tokens reclaimable.",
        format_k(decision.token_count),
        format_k(config.model_context_limit),
        decision.pressure * 100.0,
        format_k(decision.pending_tokens)
    ));

    // Tier-2/3 distillation reads differently: the target is existing blocks,
    // not conversation ranges.
    if let Some(tier) = decision.tier
        && tier != Tier::One
    {
        out.push(String::new());
        out.push(render_distillation(tier, state, prompts));
    } else {
        out.push(String::new());
        out.push(render_ranges(ranges));
    }

    out.push(String::new());
    out.push(prompts.rules_for(decision.tier.unwrap_or(Tier::One)).to_string());

    out.push(String::new());
    out.push(call_shape(decision.tier, ranges));

    // Collapse the blank-line runs the assembly above can leave behind.
    let mut cleaned: Vec<&str> = Vec::new();
    let mut previous_blank = false;
    for line in out.iter().map(String::as_str) {
        let blank = line.is_empty();
        if blank && previous_blank {
            continue;
        }
        cleaned.push(line);
        previous_blank = blank;
    }
    cleaned.join("\n").trim_end().to_string()
}

fn render_ranges(ranges: &ContextRanges) -> String {
    if ranges.compressible.is_empty() {
        return "[No specific ranges detected — compress any consumed content.]".to_string();
    }

    let shown = ranges.compressible.len().min(MAX_RANGES_SHOWN);
    let hidden = ranges.compressible.len() - shown;

    let mut out = String::from("Compressible ranges (oldest first):\n");
    for range in ranges.compressible.iter().take(shown) {
        out.push_str(&format!(
            "  {}..{}  {} msgs  {} tok  {}% tool",
            range.start_ref,
            range.end_ref,
            range.count,
            format_k(range.tokens),
            range.tool_pct
        ));
        if range.user_msgs > 0 {
            out.push_str(&format!("  ({} user)", range.user_msgs));
        }
        out.push('\n');
    }
    if hidden > 0 {
        out.push_str(&format!("  …and {hidden} more older range(s)\n"));
    }
    out.trim_end().to_string()
}

fn render_distillation(tier: Tier, state: &CompressionState, prompts: Prompts) -> String {
    let targets: Vec<&crate::agent::bigctx::types::CompressionBlock> = state
        .blocks
        .iter()
        .filter(|b| b.active && b.tier.number() < tier.number())
        .collect();

    let mut out = String::new();
    match tier {
        Tier::Two => out.push_str("[TIER 2 DISTILLATION TRIGGER]\n"),
        Tier::Three => out.push_str("[TIER 3 CONDENSATION TRIGGER]\n"),
        Tier::One => {}
    }
    out.push_str(&format!(
        "Target {} block(s) to distill:\n",
        targets.len()
    ));
    for block in targets.iter().take(MAX_RANGES_SHOWN) {
        let summary_tokens = crate::agent::bigctx::tokenize::default_count_tokens(&block.summary);
        out.push_str(&format!(
            "  {}  {} msgs  {}→{}",
            block.block_id,
            block.effective_message_ids.len(),
            format_k(block.compressed_tokens),
            format_k(summary_tokens)
        ));
        if let Some(topic) = &block.topic {
            out.push_str(&format!("  \"{topic}\""));
        }
        out.push('\n');
    }
    if let Some(first) = targets.first()
        && let Some(last) = targets.last()
    {
        out.push_str(&format!(
            "\nCite them as startId=\"{}\", endId=\"{}\".\n",
            first.block_id, last.block_id
        ));
    }
    out.push_str("\n");
    out.push_str(prompts.rules_for(tier));
    out.trim_end().to_string()
}

/// The literal call shape, with real refs from this session.
///
/// A model that knows it should compress and gets the syntax wrong costs a
/// round trip and learns the tool is finicky.
fn call_shape(tier: Option<Tier>, ranges: &ContextRanges) -> String {
    if let Some(tier) = tier
        && tier != Tier::One
    {
        return format!(
            "Call it as: compress({{\"content\": [{{\"startId\": \"b1\", \"endId\": \"bN\", \
             \"summary\": \"...\"}}]}}) — a tier-{} fold cites block ids."
        );
    }
    match ranges.compressible.first() {
        Some(range) => format!(
            "Call it as: compress({{\"content\": [{{\"startId\": \"{}\", \"endId\": \"{}\", \
             \"summary\": \"...\"}}]}}) — batch every range into ONE call.",
            range.start_ref, range.end_ref
        ),
        None => "Call it as: compress({{\"content\": [{{\"startId\": \"mNNNNN\", \"endId\": \
                 \"mNNNNN\", \"summary\": \"...\"}}]}})."
            .to_string(),
    }
}

/// `1500` → `1.5K`. Keeps the nudge's own token cost small.
fn format_k(n: u64) -> String {
    if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::bigctx::config::default_config;
    use crate::agent::bigctx::recommend::CompressibleRange;
    use crate::agent::bigctx::state::create_initial_state;
    use crate::agent::bigctx::types::{BlockGeneration, CompressionBlock};

    fn cfg(window: u64) -> BigctxConfig {
        let mut c = default_config(window);
        c.model_context_limit = window;
        c
    }

    fn decision(inject: bool, pressure: f64, tier: Option<Tier>) -> NudgeDecision {
        NudgeDecision {
            inject,
            reason: "test".to_string(),
            tier,
            pending_tokens: 30_000,
            token_count: 70_000,
            pressure,
        }
    }

    fn ranges() -> ContextRanges {
        ContextRanges {
            compressible: vec![
                CompressibleRange {
                    start_ref: "m00002".to_string(),
                    end_ref: "m00005".to_string(),
                    start_index: 1,
                    end_index: 4,
                    count: 4,
                    tokens: 12_000,
                    tool_pct: 75,
                    user_msgs: 1,
                },
                CompressibleRange {
                    start_ref: "m00008".to_string(),
                    end_ref: "m00010".to_string(),
                    start_index: 7,
                    end_index: 9,
                    count: 3,
                    tokens: 8_000,
                    tool_pct: 100,
                    user_msgs: 0,
                },
            ],
            protected: Vec::new(),
        }
    }

    fn block(id: &str, tier: Tier) -> CompressionBlock {
        CompressionBlock {
            block_id: id.to_string(),
            run_id: "r1".to_string(),
            tier,
            topic: Some("auth".to_string()),
            start_ref: Some("m00002".to_string()),
            end_ref: Some("m00005".to_string()),
            summary: "a summary".to_string(),
            direct_message_ids: vec!["t1.1".to_string()],
            effective_message_ids: vec!["t1.1".to_string()],
            direct_block_ids: vec![],
            compressed_tokens: 20_000,
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
    fn no_nudge_when_the_decision_says_no() {
        let out = render_nudge(
            &decision(false, 0.5, Some(Tier::One)),
            &ranges(),
            &create_initial_state(),
            &cfg(100_000),
        );
        assert!(out.is_none());
    }

    #[test]
    fn a_gentle_nudge_says_it_is_not_an_overflow_warning() {
        // Saying "you are about to run out" at 50% teaches the model to
        // distrust the signal.
        let out = render_nudge(
            &decision(true, 0.50, Some(Tier::One)),
            &ranges(),
            &create_initial_state(),
            &cfg(100_000),
        )
        .unwrap();
        assert_eq!(out.voice, NudgeVoice::Gentle);
        assert!(out.text.contains("not an overflow warning"));
        assert!(!out.text.contains("⚠️"));
    }

    #[test]
    fn an_emergency_nudge_says_compress_now() {
        let out = render_nudge(
            &decision(true, 0.96, Some(Tier::One)),
            &ranges(),
            &create_initial_state(),
            &cfg(100_000),
        )
        .unwrap();
        assert_eq!(out.voice, NudgeVoice::Emergency);
        assert!(out.text.contains("compress now"));
    }

    #[test]
    fn a_tier_one_nudge_lists_the_ranges_with_refs() {
        let out = render_nudge(
            &decision(true, 0.70, Some(Tier::One)),
            &ranges(),
            &create_initial_state(),
            &cfg(100_000),
        )
        .unwrap();
        assert!(out.text.contains("m00002..m00005"), "{}", out.text);
        assert!(out.text.contains("m00008..m00010"));
    }

    #[test]
    fn a_tier_one_nudge_carries_the_call_shape() {
        // A model that knows to compress but gets the syntax wrong costs a
        // round trip.
        let out = render_nudge(
            &decision(true, 0.70, Some(Tier::One)),
            &ranges(),
            &create_initial_state(),
            &cfg(100_000),
        )
        .unwrap();
        assert!(out.text.contains("compress("), "{}", out.text);
        assert!(out.text.contains("startId"));
        assert!(out.text.contains("ONE call"), "batches are the cheap path");
    }

    #[test]
    fn a_tier_one_nudge_carries_the_verbatim_rules() {
        let out = render_nudge(
            &decision(true, 0.70, Some(Tier::One)),
            &ranges(),
            &create_initial_state(),
            &cfg(100_000),
        )
        .unwrap();
        assert!(out.text.contains("KEEP VERBATIM"));
    }

    #[test]
    fn a_distillation_nudge_names_the_blocks_not_ranges() {
        let mut state = create_initial_state();
        state.blocks.push(block("b1", Tier::One));
        state.blocks.push(block("b2", Tier::One));

        let out = render_nudge(
            &decision(true, 0.70, Some(Tier::Two)),
            &ranges(),
            &state,
            &cfg(100_000),
        )
        .unwrap();
        assert!(out.text.contains("TIER 2"), "{}", out.text);
        assert!(out.text.contains("b1"), "names the blocks to distill");
        assert!(out.text.contains("b2"));
        assert!(
            !out.text.contains("m00002..m00005"),
            "a distillation does not cite message ranges"
        );
    }

    #[test]
    fn a_tier_three_nudge_says_condensation() {
        let mut state = create_initial_state();
        state.blocks.push(block("b1", Tier::One));
        state.blocks.push(block("b2", Tier::Two));

        let out = render_nudge(
            &decision(true, 0.70, Some(Tier::Three)),
            &ranges(),
            &state,
            &cfg(100_000),
        )
        .unwrap();
        assert!(out.text.contains("TIER 3"), "{}", out.text);
        assert!(out.text.contains("CONDENSATION"));
    }

    #[test]
    fn a_distillation_nudge_uses_the_tier_rules() {
        let mut state = create_initial_state();
        state.blocks.push(block("b1", Tier::One));
        let out = render_nudge(
            &decision(true, 0.70, Some(Tier::Two)),
            &ranges(),
            &state,
            &cfg(100_000),
        )
        .unwrap();
        assert!(
            out.text.contains("TIER 2 COMPRESSION — DISTILLATION"),
            "the tier-2 rules must be the ones attached"
        );
    }

    #[test]
    fn the_nudge_states_the_reclaimable_mass() {
        let out = render_nudge(
            &decision(true, 0.70, Some(Tier::One)),
            &ranges(),
            &create_initial_state(),
            &cfg(100_000),
        )
        .unwrap();
        assert!(out.text.contains("30.0K"), "{}", out.text);
        assert!(out.text.contains("70.0K"), "the current size");
    }

    #[test]
    fn a_long_range_list_is_capped() {
        // Listing every range would make the nudge part of the problem.
        let mut many = ranges();
        for i in 0..20 {
            many.compressible.push(CompressibleRange {
                start_ref: format!("m{:05}", 100 + i),
                end_ref: format!("m{:05}", 101 + i),
                start_index: 100 + i as usize,
                end_index: 101 + i as usize,
                count: 2,
                tokens: 1_000,
                tool_pct: 50,
                user_msgs: 0,
            });
        }
        let out = render_nudge(
            &decision(true, 0.70, Some(Tier::One)),
            &many,
            &create_initial_state(),
            &cfg(100_000),
        )
        .unwrap();
        assert!(out.text.contains("more older range"), "{}", out.text);
    }

    #[test]
    fn an_empty_range_list_still_renders() {
        let out = render_nudge(
            &decision(true, 0.70, Some(Tier::One)),
            &ContextRanges::default(),
            &create_initial_state(),
            &cfg(100_000),
        )
        .unwrap();
        assert!(out.text.contains("No specific ranges detected"));
    }

    #[test]
    fn format_k_is_compact() {
        assert_eq!(format_k(999), "999");
        assert_eq!(format_k(1_500), "1.5K");
        assert_eq!(format_k(1_000_000), "1000.0K");
    }

    #[test]
    fn no_blank_line_runs_survive() {
        let out = render_nudge(
            &decision(true, 0.70, Some(Tier::One)),
            &ranges(),
            &create_initial_state(),
            &cfg(100_000),
        )
        .unwrap();
        assert!(
            !out.text.contains("\n\n\n"),
            "the assembly can leave gaps; they must be collapsed"
        );
    }
}
