//! Thresholds and their validation.
//!
//! Every number here came from the upstream kernel's defaults, which were tuned
//! against production data (their paper reports 174k model calls over 4.5
//! months). Where a yak-specific choice was needed it is called out in a
//! comment — do not "clean up" those numbers without a measurement.

use serde::{Deserialize, Serialize};

/// When to tell the model it is worth compressing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NudgeConfig {
    /// Usage fraction above which a nudge is allowed.
    pub max_context_limit_pct: f64,
    /// Usage fraction below which no nudge is sent at all.
    pub min_context_limit_pct: f64,
    /// Only consider nudging every Nth round.
    pub frequency: u64,
    /// Also require at least this many rounds since the last nudge.
    pub iteration_threshold: u64,
    /// Token growth that must accumulate before repeating a nudge.
    pub growth_tokens: u64,
    /// Hard floor for the adaptive growth threshold.
    pub min_growth_floor: u64,
    /// Ratio applied to the pending token mass for the anti-thrashing floor.
    pub min_growth_ratio: f64,
    /// Usage fraction that always nudges, ignoring cadence.
    pub emergency_threshold_pct: f64,
    /// Master switch: no nudge is injected when false.
    #[serde(default = "yes")]
    pub enabled: bool,
    /// Growth step as a fraction of the window (before the floor/cap clamp).
    #[serde(default = "growth_ratio_default")]
    pub growth_ratio: f64,
    /// Ceiling on the adaptive growth step, so a huge window does not go quiet.
    #[serde(default = "growth_cap_default")]
    pub growth_cap: u64,
    /// Reclaimable mass needed for a pressure-band nudge, as a fraction of the
    /// window (floored at 5000 tokens).
    #[serde(default = "min_pressure_benefit_ratio_default")]
    pub min_pressure_benefit_ratio: f64,
}

fn yes() -> bool {
    true
}

fn growth_ratio_default() -> f64 {
    0.02
}

fn growth_cap_default() -> u64 {
    40_000
}

fn min_pressure_benefit_ratio_default() -> f64 {
    0.01
}

/// Tier distillation triggers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TierConfig {
    pub enabled: bool,
    /// Distill to tier 2 once this many tier-1 blocks are active.
    pub tier2_trigger: usize,
    /// Condense to tier 3 once this many tier-2 blocks are active.
    pub tier3_trigger: usize,
}

/// The emergency safety valve: truncate tool output when nothing else helps.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TruncateConfig {
    /// Usage fraction that arms truncation.
    pub threshold: f64,
    /// Keep the most recent N tool results verbatim.
    pub keep_recent_results: usize,
    /// Per-result character cap once truncation fires.
    pub max_result_chars: usize,
}

/// Range/summary size limits enforced at compression time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompressLimits {
    /// Refuse a range whose originals are smaller than this — compressing a
    /// tiny range costs a summary and buys nothing.
    pub min_range_tokens: u64,
    /// Reject a summary longer than this; a summary that big defeats the point.
    pub max_summary_tokens: u64,
    /// Reject a summary shorter than this; it cannot be carrying the detail the
    /// philosophy asks for.
    pub min_summary_tokens: u64,
}

/// Everything the algorithm is configured by.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BigctxConfig {
    /// The model's window. All fractions are of this.
    pub model_context_limit: u64,
    pub tiers: TierConfig,
    pub nudge: NudgeConfig,
    pub truncate: TruncateConfig,
    pub compress: CompressLimits,
    /// Tool names whose results are never folded. `*` suffix matches a prefix.
    pub protected_tools: Vec<String>,
    /// Tool names whose *latest* result is never folded (e.g. `read`, so the
    /// file the model is looking at stays in context).
    pub protected_latest_tools: Vec<String>,
    /// The most recent N messages are always kept verbatim.
    pub preserve_recent_messages: usize,
    /// The most recent N tokens are always kept verbatim.
    pub preserve_recent_tokens: u64,
    /// Passes a block must survive before it counts as `Old` (tier-2 signal).
    pub promotion_threshold: u32,
    pub crush: CrushConfig,
}

/// Deterministic compression of oversized tool results.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CrushConfig {
    /// Off by default: the JSON strategy reserves a key in the payload, so a
    /// tool result that legitimately contains it cannot be told apart from
    /// crushed output.
    pub enabled: bool,
    /// Fraction of tokens that must be saved for a crush to be used. A crush
    /// that saves almost nothing costs a prefix-cache miss for no gain.
    pub min_reduction: f64,
}

/// Defaults, with the window supplied by the caller.
///
/// The fractions mirror upstream: nudge from 45% usage, insist from 75%,
/// truncate at 95%. `model_context_limit` is the only argument because every
/// other value is a fraction of it or an absolute token count that does not
/// scale with the model.
pub fn default_config(model_context_limit: u64) -> BigctxConfig {
    BigctxConfig {
        model_context_limit,
        tiers: TierConfig {
            enabled: true,
            // Upstream's production values. The `billion-context-pi` fork used
            // 1000/2000, but that is for a 1M window; these are the kernel's
            // own numbers and scale better to the 128k-200k windows yak sees.
            tier2_trigger: 5,
            tier3_trigger: 10,
        },
        nudge: NudgeConfig {
            max_context_limit_pct: 0.75,
            min_context_limit_pct: 0.45,
            frequency: 5,
            iteration_threshold: 15,
            growth_tokens: 50_000,
            min_growth_floor: 20_000,
            min_growth_ratio: 0.45,
            emergency_threshold_pct: 0.95,
            enabled: true,
            growth_ratio: 0.02,
            growth_cap: 40_000,
            min_pressure_benefit_ratio: 0.01,
        },
        truncate: TruncateConfig {
            threshold: 0.95,
            keep_recent_results: 3,
            max_result_chars: 8_000,
        },
        compress: CompressLimits {
            min_range_tokens: 5_000,
            max_summary_tokens: 20_000,
            min_summary_tokens: 50,
        },
        protected_tools: Vec::new(),
        protected_latest_tools: Vec::new(),
        preserve_recent_messages: 3,
        preserve_recent_tokens: 5_000,
        promotion_threshold: 5,
        crush: CrushConfig {
            enabled: false,
            min_reduction: 0.1,
        },
    }
}

impl BigctxConfig {
    /// The usage fraction a count represents, 0.0 when the window is unknown.
    pub fn usage_fraction(&self, tokens: u64) -> f64 {
        if self.model_context_limit == 0 {
            return 0.0;
        }
        tokens as f64 / self.model_context_limit as f64
    }

    /// True when `tool` is never foldable.
    pub fn is_tool_protected(&self, tool: &str) -> bool {
        matches_any(&self.protected_tools, tool)
    }

    /// True when only the latest result of `tool` is protected.
    pub fn is_tool_latest_protected(&self, tool: &str) -> bool {
        matches_any(&self.protected_latest_tools, tool)
    }

    /// Reject a config that would misbehave, naming every problem at once.
    pub fn validate(&self) -> Result<(), String> {
        let mut errors: Vec<String> = Vec::new();

        if self.model_context_limit == 0 {
            errors.push("model_context_limit must be greater than 0".to_string());
        }
        if !(0.0..=1.0).contains(&self.nudge.min_context_limit_pct) {
            errors.push("nudge.min_context_limit_pct must be in [0, 1]".to_string());
        }
        if !(0.0..=1.0).contains(&self.nudge.max_context_limit_pct) {
            errors.push("nudge.max_context_limit_pct must be in [0, 1]".to_string());
        }
        if !(0.0..=1.0).contains(&self.nudge.emergency_threshold_pct) {
            errors.push("nudge.emergency_threshold_pct must be in [0, 1]".to_string());
        }
        if self.nudge.min_context_limit_pct > self.nudge.max_context_limit_pct {
            errors.push(
                "nudge.min_context_limit_pct must not exceed nudge.max_context_limit_pct"
                    .to_string(),
            );
        }
        if self.nudge.max_context_limit_pct > self.nudge.emergency_threshold_pct {
            errors.push(
                "nudge.max_context_limit_pct must not exceed nudge.emergency_threshold_pct"
                    .to_string(),
            );
        }
        if !(0.0..=1.0).contains(&self.truncate.threshold) || self.truncate.threshold == 0.0 {
            errors.push("truncate.threshold must be in (0, 1]".to_string());
        }
        if self.tiers.tier2_trigger < 1 {
            errors.push("tiers.tier2_trigger must be >= 1".to_string());
        }
        if self.tiers.tier3_trigger <= self.tiers.tier2_trigger {
            errors.push("tiers.tier3_trigger must be greater than tiers.tier2_trigger".to_string());
        }
        if self.compress.min_summary_tokens > self.compress.max_summary_tokens {
            errors.push(
                "compress.min_summary_tokens must not exceed compress.max_summary_tokens"
                    .to_string(),
            );
        }
        if self.promotion_threshold < 1 {
            errors.push("promotion_threshold must be >= 1".to_string());
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    }
}

/// Glob-suffix match: `"mcp__*"` matches `"mcp__foo"`, an exact name matches
/// itself, `"*"` matches everything.
fn matches_any(patterns: &[String], tool: &str) -> bool {
    patterns.iter().any(|pattern| {
        if pattern == "*" {
            return true;
        }
        match pattern.strip_suffix('*') {
            Some(prefix) => tool.starts_with(prefix),
            None => tool == pattern,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_validate() {
        default_config(200_000).validate().unwrap();
    }

    #[test]
    fn a_zero_window_is_refused() {
        let err = default_config(0).validate().unwrap_err();
        assert!(err.contains("model_context_limit"), "{err}");
    }

    #[test]
    fn inverted_nudge_bands_are_refused() {
        let mut cfg = default_config(200_000);
        cfg.nudge.min_context_limit_pct = 0.9;
        cfg.nudge.max_context_limit_pct = 0.5;
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("min_context_limit_pct"), "{err}");
    }

    #[test]
    fn inverted_tier_triggers_are_refused() {
        let mut cfg = default_config(200_000);
        cfg.tiers.tier3_trigger = 2;
        cfg.tiers.tier2_trigger = 5;
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("tier3_trigger"), "{err}");
    }

    #[test]
    fn every_problem_is_reported_at_once() {
        let mut cfg = default_config(0);
        cfg.promotion_threshold = 0;
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("model_context_limit"));
        assert!(err.contains("promotion_threshold"));
    }

    #[test]
    fn exact_tool_names_match() {
        let mut cfg = default_config(1000);
        cfg.protected_tools = vec!["read".to_string()];
        assert!(cfg.is_tool_protected("read"));
        assert!(!cfg.is_tool_protected("read_many"));
    }

    #[test]
    fn a_trailing_star_matches_a_prefix() {
        let mut cfg = default_config(1000);
        cfg.protected_tools = vec!["mcp__*".to_string()];
        assert!(cfg.is_tool_protected("mcp__github"));
        assert!(!cfg.is_tool_protected("read"));
    }

    #[test]
    fn a_bare_star_matches_everything() {
        let mut cfg = default_config(1000);
        cfg.protected_tools = vec!["*".to_string()];
        assert!(cfg.is_tool_protected("anything"));
    }

    #[test]
    fn usage_fraction_handles_a_zero_window() {
        let mut cfg = default_config(1000);
        assert_eq!(cfg.usage_fraction(500), 0.5);
        cfg.model_context_limit = 0;
        assert_eq!(cfg.usage_fraction(500), 0.0);
    }

    #[test]
    fn config_round_trips_through_json() {
        let cfg = default_config(200_000);
        let wire = serde_json::to_string(&cfg).unwrap();
        let back: BigctxConfig = serde_json::from_str(&wire).unwrap();
        assert_eq!(back, cfg);
    }
}
