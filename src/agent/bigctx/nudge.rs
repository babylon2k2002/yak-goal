//! When to tell the model that compressing would help.
//!
//! A model does not compress on its own. Left alone it keeps reading, the
//! window fills, and the provider eventually refuses the request. This module
//! decides when a nudge is worth injecting and what it should say.
//!
//! Two things make this non-trivial:
//!
//! - **The nudge must not repeat every turn.** It lives in the request, so an
//!   unthrottled nudge is a fixed cost paid on every round — and a model that
//!   ignores it will ignore it again. Injection is gated on *growth* since the
//!   last nudge, not on absolute size.
//! - **A nudge that offers nothing is worse than none.** If every compressible
//!   range is below the minimum, the model tries, gets refused, and learns the
//!   tool does not work. The pending mass is checked before offering.
//!
//! Deliberately simpler than upstream: one band, one growth gate, one
//! per-tier cadence map. Upstream's tier arbitration has a dozen interacting
//! gates refined against production incidents; those refinements are about
//! *which tier* to prefer, not about whether to nudge at all. Tier selection
//! here is "the tier with the most pending", which is the behaviour upstream's
//! pressure path converges on.

use crate::agent::bigctx::config::BigctxConfig;
use crate::agent::bigctx::types::{CompressionState, NudgeState, Tier};

/// What the nudge decided.
#[derive(Debug, Clone, PartialEq)]
pub struct NudgeDecision {
    /// True when a nudge should be injected this turn.
    pub inject: bool,
    /// Why, in one line — shown in `--json` and useful in a bug report.
    pub reason: String,
    /// The tier whose pending mass is largest, when one is offered.
    pub tier: Option<Tier>,
    /// Tokens the model could reclaim by folding what is currently offered.
    pub pending_tokens: u64,
    /// The context size the decision was made at, so the renderer describes the
    /// same situation the decision judged.
    pub token_count: u64,
    /// Usage fraction at that size (0.0–1.0+).
    pub pressure: f64,
}

impl NudgeDecision {
    fn no(reason: impl Into<String>) -> NudgeDecision {
        NudgeDecision {
            inject: false,
            reason: reason.into(),
            tier: None,
            pending_tokens: 0,
            token_count: 0,
            pressure: 0.0,
        }
    }
}

/// How far the context has to grow before a nudge re-fires, scaled to the
/// window.
///
/// A fixed step is wrong at both ends: 20K is a rounding error on a 1M window
/// and most of a 32K one. The cap keeps a huge window from going quiet for
/// hundreds of thousands of tokens.
pub fn adaptive_growth(window: Option<u64>, config: &BigctxConfig) -> u64 {
    let Some(window) = window.filter(|w| *w > 0) else {
        return config.nudge.min_growth_floor;
    };
    let scaled = (window as f64 * config.nudge.growth_ratio).round() as u64;
    scaled
        .clamp(config.nudge.min_growth_floor, config.nudge.growth_cap)
        .max(1)
}

/// Decide whether to nudge this turn.
///
/// `pending_by_tier` is the reclaimable token mass at each tier: the
/// compressible ranges for tier 1, the active blocks for tiers 2 and 3.
pub fn decide_nudge(
    state: &CompressionState,
    config: &BigctxConfig,
    token_count: u64,
    pending_by_tier: [u64; 3],
) -> NudgeDecision {
    if !config.nudge.enabled {
        return NudgeDecision::no("nudges are off");
    }

    let window = config.model_context_limit;
    let usage = if window > 0 {
        token_count as f64 / window as f64
    } else {
        0.0
    };
    let over_limit = usage >= config.nudge.max_context_limit_pct;

    // The tier with the most to reclaim. Tier 2/3 only count when tiered
    // distillation is enabled, or the model would be told to distill blocks it
    // cannot distill.
    let mut best = Some(Tier::One);
    let mut best_pending = pending_by_tier[0];
    if config.tiers.enabled {
        for (index, tier) in [Tier::Two, Tier::Three].into_iter().enumerate() {
            let pending = pending_by_tier[index + 1];
            if pending > best_pending {
                best_pending = pending;
                best = Some(tier);
            }
        }
    }

    if best_pending == 0 {
        return NudgeDecision::no("nothing is compressible");
    }

    // Under pressure the growth gate is skipped: waiting for the context to
    // grow when it is already over the limit is waiting for a refusal.
    if over_limit {
        let minimum = min_pressure_benefit(window, config);
        if best_pending < minimum {
            // Offering a sub-threshold pending in a hot band produces a nudge
            // that can never succeed, and every "success" would re-arm it.
            return NudgeDecision::no(format!(
                "over the limit but only {best_pending} tokens are reclaimable (need {minimum})"
            ));
        }
        return NudgeDecision {
            inject: true,
            reason: format!(
                "over the limit: {:.0}% used, {best_pending} tokens reclaimable",
                usage * 100.0
            ),
            tier: best,
            pending_tokens: best_pending,
            token_count,
            pressure: usage,
        };
    }

    // Below the band, a nudge is noise: the model has room and a fold costs it
    // a model call plus a prefix-cache miss.
    if usage < config.nudge.min_context_limit_pct {
        return NudgeDecision::no(format!(
            "only {:.0}% used (nudges start at {:.0}%)",
            usage * 100.0,
            config.nudge.min_context_limit_pct * 100.0
        ));
    }

    // Growth gate: has the context grown enough since the last nudge (or since
    // the session started) to be worth another one?
    let growth = adaptive_growth(Some(window), config);
    let reference = growth_reference(&state.nudge, token_count);
    let grown = token_count.saturating_sub(reference);

    // A session that arrives already over the band — a resumed thread, a
    // restored state — has grown nothing this process has seen. Gating it on
    // growth would leave it sitting on a full window forever, so a first sight
    // with real mass bypasses the gate.
    let first_sight = state.nudge.last_nudge_shown_tokens == 0
        && state.nudge.baseline_tokens == 0
        && best_pending >= growth;
    if !first_sight && grown < growth {
        return NudgeDecision::no(format!(
            "grown {grown} tokens since the last nudge (needs {growth})"
        ));
    }

    // Per-tier cadence: the same tier does not re-fire on the same mass twice.
    if let Some(tier) = best {
        let shown = state
            .nudge
            .last_shown_by_tier
            .get(&tier.number())
            .copied()
            .unwrap_or(0);
        if shown > 0 && token_count.saturating_sub(shown) < growth {
            return NudgeDecision::no(format!(
                "tier {} was nudged {} tokens ago (needs {growth})",
                tier.number(),
                token_count.saturating_sub(shown)
            ));
        }
    }

    NudgeDecision {
        inject: true,
        reason: format!(
            "{:.0}% used, {best_pending} tokens reclaimable",
            usage * 100.0
        ),
        tier: best,
        pending_tokens: best_pending,
        token_count,
        pressure: usage,
    }
}

/// Minimum reclaimable mass for a pressure-band nudge to be worth injecting.
fn min_pressure_benefit(window: u64, config: &BigctxConfig) -> u64 {
    let scaled = (window as f64 * config.nudge.min_pressure_benefit_ratio).round() as u64;
    scaled.max(5_000)
}

/// What "growth" is measured against.
///
/// The last nudge is the right reference when there was one: the model has
/// already been told, so the question is whether anything changed. Before that,
/// the session baseline is the only meaningful anchor.
fn growth_reference(nudge: &NudgeState, token_count: u64) -> u64 {
    if nudge.last_nudge_shown_tokens > 0 {
        nudge.last_nudge_shown_tokens
    } else if nudge.baseline_tokens > 0 {
        nudge.baseline_tokens
    } else {
        token_count
    }
}

/// Record that a nudge was shown, so the next pass can measure growth from it.
pub fn stamp_nudge(state: &mut CompressionState, decision: &NudgeDecision, token_count: u64) {
    if state.nudge.baseline_tokens == 0 {
        state.nudge.baseline_tokens = token_count;
    }
    if !decision.inject {
        return;
    }
    state.nudge.last_nudge_shown_tokens = token_count;
    if let Some(tier) = decision.tier {
        state
            .nudge
            .last_shown_by_tier
            .insert(tier.number(), token_count);
    }
}

/// Clear the cadence after a successful compression.
///
/// The model acted, so the previous nudge is spent: the next one should be
/// measured against the *new* size, not the size that prompted the last one.
/// Without this the session stays "already nudged" forever and goes quiet while
/// the context refills.
pub fn reset_after_compression(state: &mut CompressionState, token_count: u64) {
    state.nudge.last_nudge_shown_tokens = 0;
    state.nudge.last_per_message_nudge_tokens = token_count;
    state.nudge.baseline_tokens = token_count;
    state.nudge.last_shown_by_tier.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::bigctx::config::default_config;
    use crate::agent::bigctx::state::create_initial_state;

    fn cfg(window: u64) -> BigctxConfig {
        let mut c = default_config(window);
        c.model_context_limit = window;
        c
    }

    #[test]
    fn adaptive_growth_scales_with_the_window() {
        // The floor dominates at both ends of the default config (20K), so the
        // scaling only shows once the floor is lowered — which is what a small
        // window needs. Assert the shape with a floor that lets it show.
        let mut c = cfg(200_000);
        c.nudge.min_growth_floor = 1_000;
        c.nudge.growth_cap = 1_000_000;

        let small = adaptive_growth(Some(50_000), &c);
        let large = adaptive_growth(Some(500_000), &c);
        assert_eq!(small, 1_000, "50K * 2% = 1000, at the floor");
        assert_eq!(large, 10_000, "500K * 2% = 10000");
        assert!(large > small, "a bigger window tolerates a bigger step");
    }

    #[test]
    fn the_default_floor_dominates_below_two_megabytes() {
        // Stated as a test because it surprised me: with the shipped defaults
        // the step is a constant 20K for every window under ~1M, and the cap
        // binds only above 2M.
        let c = cfg(200_000);
        assert_eq!(adaptive_growth(Some(32_000), &c), 20_000);
        assert_eq!(adaptive_growth(Some(1_000_000), &c), 20_000);
        assert_eq!(adaptive_growth(Some(2_000_000), &c), 40_000);
    }

    #[test]
    fn adaptive_growth_is_capped() {
        let c = cfg(10_000_000);
        let step = adaptive_growth(Some(10_000_000), &c);
        assert!(
            step <= c.nudge.growth_cap,
            "an enormous window must not go quiet forever: {step}"
        );
    }

    #[test]
    fn adaptive_growth_has_a_floor() {
        let c = cfg(1_000);
        let step = adaptive_growth(Some(1_000), &c);
        assert!(step >= c.nudge.min_growth_floor, "got {step}");
    }

    #[test]
    fn no_nudge_below_the_band() {
        let c = cfg(100_000);
        let state = create_initial_state();
        let decision = decide_nudge(&state, &c, 1_000, [50_000, 0, 0]);
        assert!(!decision.inject, "{}", decision.reason);
    }

    #[test]
    fn no_nudge_when_nothing_is_reclaimable() {
        let c = cfg(100_000);
        let state = create_initial_state();
        let decision = decide_nudge(&state, &c, 90_000, [0, 0, 0]);
        assert!(!decision.inject);
        assert!(decision.reason.contains("nothing is compressible"));
    }

    #[test]
    fn over_the_limit_nudges_without_waiting_for_growth() {
        let c = cfg(100_000);
        let state = create_initial_state();
        // 95% used, plenty reclaimable
        let decision = decide_nudge(&state, &c, 95_000, [40_000, 0, 0]);
        assert!(decision.inject, "{}", decision.reason);
        assert_eq!(decision.tier, Some(Tier::One));
    }

    #[test]
    fn over_the_limit_with_tiny_pending_is_refused() {
        // Offering a sub-threshold range in a hot band produces a nudge that
        // can never succeed.
        let c = cfg(100_000);
        let state = create_initial_state();
        let decision = decide_nudge(&state, &c, 95_000, [100, 0, 0]);
        assert!(!decision.inject, "{}", decision.reason);
    }

    #[test]
    fn a_nudge_does_not_repeat_without_growth() {
        let c = cfg(100_000);
        let mut state = create_initial_state();
        let first = decide_nudge(&state, &c, 70_000, [30_000, 0, 0]);
        assert!(first.inject, "{}", first.reason);
        stamp_nudge(&mut state, &first, 70_000);

        let again = decide_nudge(&state, &c, 70_000, [30_000, 0, 0]);
        assert!(!again.inject, "an unthrottled nudge is paid every round");
    }

    #[test]
    fn a_nudge_refires_after_enough_growth() {
        let c = cfg(100_000);
        let mut state = create_initial_state();
        let first = decide_nudge(&state, &c, 70_000, [30_000, 0, 0]);
        stamp_nudge(&mut state, &first, 70_000);

        let step = adaptive_growth(Some(100_000), &c);
        let again = decide_nudge(&state, &c, 70_000 + step + 1, [30_000, 0, 0]);
        assert!(again.inject, "{}", again.reason);
    }

    #[test]
    fn the_largest_pending_tier_wins() {
        let mut c = cfg(100_000);
        c.tiers.enabled = true;
        let state = create_initial_state();
        let decision = decide_nudge(&state, &c, 95_000, [10_000, 50_000, 0]);
        assert_eq!(decision.tier, Some(Tier::Two));
        assert_eq!(decision.pending_tokens, 50_000);
    }

    #[test]
    fn a_disabled_tier_is_never_offered() {
        let mut c = cfg(100_000);
        c.tiers.enabled = false;
        let state = create_initial_state();
        let decision = decide_nudge(&state, &c, 95_000, [10_000, 50_000, 0]);
        assert_eq!(
            decision.tier,
            Some(Tier::One),
            "the model cannot distill what the config forbids"
        );
    }

    #[test]
    fn a_first_sight_with_real_mass_bypasses_the_growth_gate() {
        // A resumed thread arrives already full; gating it on growth would
        // leave it sitting on a full window forever.
        let c = cfg(100_000);
        let state = create_initial_state();
        let step = adaptive_growth(Some(100_000), &c);
        let decision = decide_nudge(&state, &c, 70_000, [step + 1, 0, 0]);
        assert!(decision.inject, "{}", decision.reason);
    }

    #[test]
    fn compression_resets_the_cadence() {
        let c = cfg(100_000);
        let mut state = create_initial_state();
        let first = decide_nudge(&state, &c, 70_000, [30_000, 0, 0]);
        stamp_nudge(&mut state, &first, 70_000);

        // the model compressed: the context shrank
        reset_after_compression(&mut state, 40_000);
        assert_eq!(state.nudge.last_nudge_shown_tokens, 0);

        // and the next nudge is measured from the new size
        let next = decide_nudge(&state, &c, 70_000, [30_000, 0, 0]);
        assert!(next.inject, "{}", next.reason);
    }

    #[test]
    fn nudges_can_be_switched_off() {
        let mut c = cfg(100_000);
        c.nudge.enabled = false;
        let state = create_initial_state();
        let decision = decide_nudge(&state, &c, 99_000, [50_000, 0, 0]);
        assert!(!decision.inject);
    }

    #[test]
    fn stamping_records_the_baseline_once() {
        let mut state = create_initial_state();
        let decision = NudgeDecision::no("test");
        stamp_nudge(&mut state, &decision, 1_000);
        assert_eq!(state.nudge.baseline_tokens, 1_000);
        stamp_nudge(&mut state, &decision, 9_000);
        assert_eq!(
            state.nudge.baseline_tokens, 1_000,
            "the baseline is the session start, not the latest turn"
        );
    }
}
