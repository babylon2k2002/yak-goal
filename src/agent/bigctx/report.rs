//! Making the compression visible.
//!
//! A feature nobody can see is a feature nobody can trust. This produces the
//! report behind `/status` and the `acp_status` tool: what is folded, what it
//! saved, what is still foldable, and — the part that catches real problems —
//! whether the numbers are *consistent*.
//!
//! The consistency checks exist because the failure modes here are silent. A
//! block that covers messages nobody can see, a summary larger than the
//! material it replaced, a fold whose coverage overlaps another's: each is a
//! bug that produces a plausible-looking report while the session quietly loses
//! context. [`audit`] names them rather than leaving them to be noticed.

use crate::agent::bigctx::config::BigctxConfig;
use crate::agent::bigctx::prune::SUMMARY_HEADER;
use crate::agent::bigctx::recommend::ContextRanges;
use crate::agent::bigctx::state::covered_message_ids;
use crate::agent::bigctx::tokenize::{count_message_tokens, default_count_tokens};
use crate::agent::bigctx::types::{
    CompressionBlock, CompressionState, CoreMessage, MessageContentType, MessageRole, Tier,
    base_id_of,
};

/// `1500` → `1.5K`. A report that itself costs tokens is a poor trade.
pub fn format_tokens(n: u64) -> String {
    if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn pct(part: u64, total: u64) -> u64 {
    if total == 0 {
        return 0;
    }
    (part as f64 / total as f64 * 100.0).round() as u64
}

/// One block, as the report shows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockLine {
    pub block_id: String,
    pub tier: Tier,
    pub message_count: usize,
    /// Tokens of material this block replaced.
    pub compressed_tokens: u64,
    /// Tokens the summary itself costs.
    pub summary_tokens: u64,
    pub topic: Option<String>,
    /// First line of the summary, for recognition.
    pub preview: String,
}

impl BlockLine {
    /// How many times smaller the summary is than what it replaced.
    ///
    /// Below 1.0 the block is a net LOSS: it costs more to carry than it saved.
    pub fn ratio(&self) -> f64 {
        if self.summary_tokens == 0 {
            return 0.0;
        }
        self.compressed_tokens as f64 / self.summary_tokens as f64
    }

    pub fn is_net_loss(&self) -> bool {
        self.summary_tokens >= self.compressed_tokens
    }
}

/// The whole picture.
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    /// Tokens in the visible view.
    pub visible_tokens: u64,
    /// Messages visible.
    pub visible_messages: usize,
    /// Messages folded away.
    pub folded_messages: usize,
    /// Active blocks.
    pub blocks: Vec<BlockLine>,
    /// Cumulative tokens saved across every compression.
    pub tokens_saved: u64,
    /// How many compressions ran.
    pub compression_count: u64,
    /// The model's window, when known.
    pub window: Option<u64>,
    /// What is still foldable.
    pub compressible: ContextRanges,
    /// Problems found. Empty is the healthy case.
    pub problems: Vec<String>,
}

impl Report {
    /// Usage as a fraction of the window, when the window is known.
    pub fn usage(&self) -> Option<f64> {
        self.window
            .filter(|w| *w > 0)
            .map(|w| self.visible_tokens as f64 / w as f64)
    }

    /// True when nothing is folded.
    pub fn is_idle(&self) -> bool {
        self.blocks.is_empty() && self.folded_messages == 0
    }

    /// True when every problem found is cosmetic.
    pub fn is_healthy(&self) -> bool {
        self.problems.is_empty()
    }
}

/// Build the report.
pub fn build_report(
    view: &[CoreMessage],
    full: &[CoreMessage],
    state: &CompressionState,
    config: &BigctxConfig,
) -> Report {
    let covered = covered_message_ids(state);
    let visible_messages = view.len();
    let folded_messages = full
        .iter()
        .filter(|m| covered.contains(base_id_of(&m.id)))
        .count();

    let visible_tokens: u64 = view.iter().map(count_message_tokens).sum();

    let blocks: Vec<BlockLine> = state
        .blocks
        .iter()
        .filter(|b| b.active)
        .map(|block| BlockLine {
            block_id: block.block_id.clone(),
            tier: block.tier,
            message_count: block.effective_message_ids.len(),
            compressed_tokens: block.compressed_tokens,
            summary_tokens: default_count_tokens(&block.summary),
            topic: block.topic.clone(),
            preview: first_line(&block.summary),
        })
        .collect();

    let compressible = crate::agent::bigctx::screen(full, state, config);

    let window = (config.model_context_limit > 0).then_some(config.model_context_limit);

    let mut report = Report {
        visible_tokens,
        visible_messages,
        folded_messages,
        blocks,
        tokens_saved: state.stats.tokens_compressed,
        compression_count: state.stats.compression_count,
        window,
        compressible,
        problems: Vec::new(),
    };
    report.problems = audit(full, state, &report);
    report
}

/// The plain-text rendering the model and the user both read.
pub fn render_report(report: &Report) -> String {
    let mut out = String::new();

    match report.usage() {
        Some(usage) => out.push_str(&format!(
            "context: {} of {} used ({:.0}%)\n",
            format_tokens(report.visible_tokens),
            format_tokens(report.window.unwrap_or(0)),
            usage * 100.0
        )),
        None => out.push_str(&format!(
            "context: {} used (window unknown)\n",
            format_tokens(report.visible_tokens)
        )),
    }

    out.push_str(&format!(
        "messages: {} visible, {} folded\n",
        report.visible_messages, report.folded_messages
    ));

    if report.compression_count > 0 {
        out.push_str(&format!(
            "saved: {} tokens across {} compression(s)\n",
            format_tokens(report.tokens_saved),
            report.compression_count
        ));
    }

    if report.blocks.is_empty() {
        out.push_str("\nNo blocks — nothing has been compressed this session.\n");
    } else {
        out.push_str(&format!("\nblocks ({}):\n", report.blocks.len()));
        for block in &report.blocks {
            out.push_str(&format!(
                "  {} T{}  {} msgs  {}→{}{}\n",
                block.block_id,
                block.tier.number(),
                block.message_count,
                format_tokens(block.compressed_tokens),
                format_tokens(block.summary_tokens),
                block
                    .topic
                    .as_deref()
                    .map(|t| format!("  \"{t}\""))
                    .unwrap_or_default()
            ));
            out.push_str(&format!("     {}\n", block.preview));
        }

        // Tier mix, only when there is more than one tier to compare.
        let tiers: Vec<u8> = {
            let mut seen: Vec<u8> = report
                .blocks
                .iter()
                .map(|b| b.tier.number())
                .collect();
            seen.sort_unstable();
            seen.dedup();
            seen
        };
        if tiers.len() > 1 {
            let parts: Vec<String> = tiers
                .iter()
                .map(|tier| {
                    let in_tier: Vec<&BlockLine> = report
                        .blocks
                        .iter()
                        .filter(|b| b.tier.number() == *tier)
                        .collect();
                    let tokens: u64 = in_tier.iter().map(|b| b.summary_tokens).sum();
                    format!("T{tier}: {} ({} blocks)", format_tokens(tokens), in_tier.len())
                })
                .collect();
            out.push_str(&format!("\ntiers: {}\n", parts.join(" | ")));
        }
    }

    if report.compressible.compressible.is_empty() {
        out.push_str(
            "\ncompressible: none — everything left is protected (the recent working set) or too \
             small to be worth a summary.\n",
        );
    } else {
        out.push_str(&format!(
            "\ncompressible ({} tokens):\n",
            format_tokens(report.compressible.compressible_tokens())
        ));
        for range in report.compressible.compressible.iter().take(8) {
            out.push_str(&format!(
                "  {}..{}  {} msgs  {} tok  {}% tool\n",
                range.start_ref,
                range.end_ref,
                range.count,
                format_tokens(range.tokens),
                range.tool_pct
            ));
        }
        let extra = report.compressible.compressible.len().saturating_sub(8);
        if extra > 0 {
            out.push_str(&format!("  …and {extra} more\n"));
        }
    }

    if !report.problems.is_empty() {
        out.push_str("\n⚠ problems:\n");
        for problem in &report.problems {
            out.push_str(&format!("  - {problem}\n"));
        }
    }

    out.trim_end().to_string()
}

/// Name the inconsistencies worth knowing about.
///
/// Each of these is a silent failure: the report looks plausible, and the
/// session quietly loses or wastes context.
pub fn audit(
    full: &[CoreMessage],
    state: &CompressionState,
    report: &Report,
) -> Vec<String> {
    let mut problems: Vec<String> = Vec::new();

    let present: std::collections::BTreeSet<&str> =
        full.iter().map(|m| base_id_of(&m.id)).collect();

    // A block covering messages that are not in the array cannot be restored:
    // its record names ids nothing can produce.
    for block in state.blocks.iter().filter(|b| b.active) {
        let missing = block
            .effective_message_ids
            .iter()
            .filter(|id| !present.contains(base_id_of(id)))
            .count();
        if missing > 0 {
            problems.push(format!(
                "{} covers {missing} message(s) that are not in the history — it cannot be restored",
                block.block_id
            ));
        }
    }

    // A block whose summary is bigger than what it replaced costs more than it
    // saves. Not fatal, but it means the fold was pointless and the model
    // should not make more like it.
    for block in &report.blocks {
        if block.is_net_loss() {
            problems.push(format!(
                "{} is a net loss: {} replaced by a {} summary",
                block.block_id,
                format_tokens(block.compressed_tokens),
                format_tokens(block.summary_tokens)
            ));
        }
    }

    // Two active blocks covering the same message means a fold was applied to
    // already-folded material: the coverage arithmetic is wrong somewhere.
    let mut seen: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
    for block in state.blocks.iter().filter(|b| b.active) {
        for id in &block.effective_message_ids {
            let base = base_id_of(id).to_string();
            if let Some(other) = seen.get(&base) {
                if other != &block.block_id {
                    problems.push(format!(
                        "{} and {} both cover {base} — a fold landed on already-folded material",
                        other, block.block_id
                    ));
                }
            } else {
                seen.insert(base, block.block_id.clone());
            }
        }
    }

    // A summary rendered into the view with no block behind it is an orphan:
    // the model sees condensed material it can never restore.
    let active_ids: std::collections::BTreeSet<&str> = state
        .blocks
        .iter()
        .filter(|b| b.active)
        .map(|b| b.block_id.as_str())
        .collect();
    for message in full {
        if message.content_type != MessageContentType::Text
            || message.role != MessageRole::System
        {
            continue;
        }
        let Some(text) = message.text.as_deref() else {
            continue;
        };
        if !text.starts_with(SUMMARY_HEADER) {
            continue;
        }
        if let Some(block_id) = message.id.strip_prefix("acp_summary_")
            && !active_ids.contains(block_id)
        {
            problems.push(format!(
                "the view renders a summary for {block_id}, which is not an active block"
            ));
        }
    }

    // The same message covered by two different tiers means a tier-2 fold
    // consumed a block whose material a tier-1 block still claims.
    for block in &report.blocks {
        if block.tier == Tier::One {
            continue;
        }
        let consumed: std::collections::BTreeSet<&str> = state
            .blocks
            .iter()
            .find(|b| b.block_id == block.block_id)
            .map(|b| b.direct_block_ids.iter().map(String::as_str).collect())
            .unwrap_or_default();
        if consumed.is_empty() {
            problems.push(format!(
                "{} is tier {} but consumed no blocks — its tier is wrong",
                block.block_id,
                block.tier.number()
            ));
        }
    }

    problems.sort();
    problems.dedup();
    problems
}

fn first_line(text: &str) -> String {
    let line = text.lines().next().unwrap_or("").trim();
    if line.chars().count() > 100 {
        let head: String = line.chars().take(97).collect();
        format!("{head}…")
    } else {
        line.to_string()
    }
}

/// A one-line summary for a status bar: `3 blocks · 42.1K saved · 71%`.
pub fn status_line(report: &Report) -> String {
    if report.is_idle() {
        return "no compression yet".to_string();
    }
    let mut parts = vec![format!("{} block(s)", report.blocks.len())];
    parts.push(format!("{} saved", format_tokens(report.tokens_saved)));
    if let Some(usage) = report.usage() {
        parts.push(format!("{:.0}% used", usage * 100.0));
    }
    if !report.is_healthy() {
        parts.push(format!("{} problem(s)", report.problems.len()));
    }
    parts.join(" · ")
}

/// Share of the visible context that is tool traffic.
///
/// The number that tells a user whether compression would even help: a session
/// that is 5% tool output has little to reclaim.
pub fn tool_share(view: &[CoreMessage]) -> u64 {
    let total: u64 = view.iter().map(count_message_tokens).sum();
    if total == 0 {
        return 0;
    }
    let tool: u64 = view
        .iter()
        .filter(|m| {
            matches!(
                m.content_type,
                MessageContentType::ToolCall | MessageContentType::ToolResult
            )
        })
        .map(count_message_tokens)
        .sum();
    pct(tool, total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::bigctx::config::default_config;
    use crate::agent::bigctx::state::create_initial_state;
    use crate::agent::bigctx::types::{BlockGeneration, CompressionBlock};

    fn user(id: &str, text: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::User, MessageContentType::Text).with_text(text)
    }

    fn result(id: &str, text: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::Tool, MessageContentType::ToolResult)
            .with_text(text)
            .with_tool("bash", "c1")
    }

    fn block(id: &str, tier: Tier, covers: &[&str], consumed: &[&str]) -> CompressionBlock {
        CompressionBlock {
            block_id: id.to_string(),
            run_id: "r1".to_string(),
            tier,
            topic: None,
            start_ref: None,
            end_ref: None,
            summary: "a summary of what happened".to_string(),
            direct_message_ids: covers.iter().map(|s| s.to_string()).collect(),
            effective_message_ids: covers.iter().map(|s| s.to_string()).collect(),
            direct_block_ids: consumed.iter().map(|s| s.to_string()).collect(),
            compressed_tokens: 10_000,
            created_at: 0,
            survived_count: 0,
            generation: BlockGeneration::Young,
            active: true,
            expanded: false,
            duration_ms: None,
            compress_call_id: None,
        }
    }

    fn cfg(window: u64) -> BigctxConfig {
        let mut c = default_config(window);
        c.model_context_limit = window;
        c
    }

    #[test]
    fn an_empty_session_reports_idle() {
        let messages = vec![user("t1.0", "hello")];
        let report = build_report(&messages, &messages, &create_initial_state(), &cfg(100_000));
        assert!(report.is_idle());
        assert!(report.is_healthy());
        assert_eq!(report.blocks.len(), 0);
    }

    #[test]
    fn a_fold_shows_up_with_its_savings() {
        let full = vec![user("t1.0", "go"), result("t1.1", "a long body")];
        let view = vec![user("t1.0", "go")];
        let mut state = create_initial_state();
        state.blocks.push(block("b1", Tier::One, &["t1.1"], &[]));
        state.stats.tokens_compressed = 9_000;
        state.stats.compression_count = 1;

        let report = build_report(&view, &full, &state, &cfg(100_000));
        assert_eq!(report.blocks.len(), 1);
        assert_eq!(report.folded_messages, 1);
        assert_eq!(report.tokens_saved, 9_000);
        assert!(!report.is_idle());
    }

    #[test]
    fn the_report_renders_without_panicking_on_an_empty_session() {
        let messages = vec![user("t1.0", "hello")];
        let report = build_report(&messages, &messages, &create_initial_state(), &cfg(100_000));
        let text = render_report(&report);
        assert!(text.contains("No blocks"));
        assert!(text.contains("compressible"));
    }

    #[test]
    fn the_report_shows_the_savings_and_the_ratio() {
        let full = vec![user("t1.0", "go"), result("t1.1", "body")];
        let view = vec![user("t1.0", "go")];
        let mut state = create_initial_state();
        state.blocks.push(block("b1", Tier::One, &["t1.1"], &[]));
        let report = build_report(&view, &full, &state, &cfg(100_000));
        let text = render_report(&report);
        assert!(text.contains("b1"), "{}", text);
        assert!(text.contains("10.0K"), "the compressed size: {}", text);
    }

    #[test]
    fn usage_is_reported_against_the_window() {
        let messages = vec![user("t1.0", "x".repeat(40_000).as_str())];
        let report = build_report(&messages, &messages, &create_initial_state(), &cfg(100_000));
        let usage = report.usage().expect("a window is known");
        assert!(usage > 0.05 && usage < 0.2, "got {usage}");
    }

    #[test]
    fn a_zero_window_reports_unknown_usage() {
        let messages = vec![user("t1.0", "x")];
        let report = build_report(&messages, &messages, &create_initial_state(), &cfg(0));
        assert_eq!(report.usage(), None);
        assert!(render_report(&report).contains("window unknown"));
    }

    // ── audit ──

    #[test]
    fn a_block_covering_missing_messages_is_flagged() {
        // Its record names ids nothing can produce, so it cannot be restored.
        let full = vec![user("t1.0", "go")];
        let mut state = create_initial_state();
        state.blocks.push(block("b1", Tier::One, &["t9.9"], &[]));
        let report = build_report(&full, &full, &state, &cfg(100_000));
        assert!(
            report.problems.iter().any(|p| p.contains("cannot be restored")),
            "{:?}",
            report.problems
        );
    }

    #[test]
    fn a_net_loss_block_is_flagged() {
        let full = vec![user("t1.0", "go"), result("t1.1", "body")];
        let mut state = create_initial_state();
        let mut b = block("b1", Tier::One, &["t1.1"], &[]);
        b.compressed_tokens = 5;
        b.summary = "a summary that is much longer than the five tokens it replaced".to_string();
        state.blocks.push(b);
        let report = build_report(&full, &full, &state, &cfg(100_000));
        assert!(
            report.problems.iter().any(|p| p.contains("net loss")),
            "{:?}",
            report.problems
        );
    }

    #[test]
    fn two_blocks_covering_one_message_are_flagged() {
        let full = vec![user("t1.0", "go"), result("t1.1", "body")];
        let mut state = create_initial_state();
        state.blocks.push(block("b1", Tier::One, &["t1.1"], &[]));
        state.blocks.push(block("b2", Tier::One, &["t1.1"], &[]));
        let report = build_report(&full, &full, &state, &cfg(100_000));
        assert!(
            report
                .problems
                .iter()
                .any(|p| p.contains("both cover")),
            "{:?}",
            report.problems
        );
    }

    #[test]
    fn a_tier_two_block_with_no_consumed_children_is_flagged() {
        // A distillation that consumed nothing is a tier-1 fold mislabelled.
        let full = vec![user("t1.0", "go"), result("t1.1", "body")];
        let mut state = create_initial_state();
        state.blocks.push(block("b2", Tier::Two, &["t1.1"], &[]));
        let report = build_report(&full, &full, &state, &cfg(100_000));
        assert!(
            report.problems.iter().any(|p| p.contains("tier is wrong")),
            "{:?}",
            report.problems
        );
    }

    #[test]
    fn a_healthy_session_has_no_problems() {
        let full = vec![user("t1.0", "go"), result("t1.1", "body")];
        let mut state = create_initial_state();
        state.blocks.push(block("b1", Tier::One, &["t1.1"], &[]));
        let report = build_report(&full, &full, &state, &cfg(100_000));
        assert!(report.is_healthy(), "{:?}", report.problems);
    }

    #[test]
    fn a_tier_two_block_with_children_is_healthy() {
        let full = vec![user("t1.0", "go"), result("t1.1", "body")];
        let mut state = create_initial_state();
        let mut child = block("b1", Tier::One, &["t1.1"], &[]);
        child.active = false;
        state.blocks.push(child);
        state.blocks.push(block("b2", Tier::Two, &["t1.1"], &["b1"]));
        let report = build_report(&full, &full, &state, &cfg(100_000));
        assert!(report.is_healthy(), "{:?}", report.problems);
    }

    // ── the one-liner ──

    #[test]
    fn an_idle_session_says_so() {
        let messages = vec![user("t1.0", "hello")];
        let report = build_report(&messages, &messages, &create_initial_state(), &cfg(100_000));
        assert_eq!(status_line(&report), "no compression yet");
    }

    #[test]
    fn the_status_line_carries_the_essentials() {
        let full = vec![user("t1.0", "go"), result("t1.1", "body")];
        let mut state = create_initial_state();
        state.blocks.push(block("b1", Tier::One, &["t1.1"], &[]));
        state.stats.tokens_compressed = 9_000;
        let report = build_report(&full, &full, &state, &cfg(100_000));
        let line = status_line(&report);
        assert!(line.contains("1 block"), "{line}");
        assert!(line.contains("9.0K saved"), "{line}");
        assert!(line.contains("used"), "{line}");
    }

    #[test]
    fn the_status_line_surfaces_problems() {
        let full = vec![user("t1.0", "go")];
        let mut state = create_initial_state();
        state.blocks.push(block("b1", Tier::One, &["t9.9"], &[]));
        let report = build_report(&full, &full, &state, &cfg(100_000));
        assert!(status_line(&report).contains("problem"));
    }

    // ── tool share ──

    #[test]
    fn tool_share_measures_the_reclaimable_part() {
        // A session that is 5% tool output has little compression to offer.
        let all_tool = vec![result("t1.0", &"x".repeat(4_000))];
        assert_eq!(tool_share(&all_tool), 100);

        let all_text = vec![user("t1.0", &"x".repeat(4_000))];
        assert_eq!(tool_share(&all_text), 0);

        let mixed = vec![
            user("t1.0", &"x".repeat(2_000)),
            result("t1.1", &"x".repeat(2_000)),
        ];
        let share = tool_share(&mixed);
        assert!((40..=60).contains(&share), "got {share}");
    }

    #[test]
    fn an_empty_view_has_no_tool_share() {
        assert_eq!(tool_share(&[]), 0);
    }

    #[test]
    fn format_tokens_is_compact() {
        assert_eq!(format_tokens(0), "0");
        assert_eq!(format_tokens(999), "999");
        assert_eq!(format_tokens(1_500), "1.5K");
    }
}
