//! Output-side steering: make the model say less, without losing correctness.
//!
//! Bigctx shrinks the *input*. This module is the other lever: it decides how
//! much prose the model should produce. Two independent knobs, both decided
//! here and landed by the host:
//!
//! - **verbosity** — a directive appended to the system prompt, at one of five
//!   levels (L0 = nothing, L4 = fragments only).
//! - **effort routing** — on a *mechanical* turn (the model is continuing after
//!   a tool result, not answering a person), lower the reasoning effort the
//!   client already asked for, toward a floor. Never raises.
//!
//! This module DECIDES; it never touches a wire. That split is why it is pure
//! functions over a structural description of the final turn: the host maps its
//! own message type in (see [`classify`]'s contract) and writes the decision out.
//!
//! Three rules are load-bearing and easy to break accidentally:
//!
//! 1. **Clamp-only.** [`clamp_effort_to_floor`] returns `None` for an absent,
//!    unrecognized, or already-low value. Coercing an unknown value that
//!    happens to sit below the floor would be a *raise*, and a model without
//!    effort support 400s on an injected field.
//! 2. **Tail-only injection.** [`apply_to_prompt`] appends or replaces in
//!    place; it never prepends. The prefix cache keys on stable head bytes, so
//!    prepending would cost a full re-read every turn.
//! 3. **Byte-stable directives.** The L1–L4 texts are frozen. Editing one
//!    character busts the prefix cache of every session pinned at that level,
//!    so a change here is a deliberate, announced cache bust — not a wording
//!    tweak.

use serde_json::Value;

/// What kind of turn the model is about to take.
///
/// Classified structurally — block composition only, no pattern-matching on
/// content. Conservative by design: anything unrecognized is
/// [`TurnKind::Unknown`] rather than a guess, because a composition we cannot
/// read may carry user intent we cannot see.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnKind {
    /// The final message carries user signal: answer it.
    NewUserAsk,
    /// The final message is only tool results: continue the work.
    MechanicalContinuation,
    /// At least one tool result in the final message reported an error.
    ErrorContinuation,
    /// Unrecognized composition, or no final user turn.
    Unknown,
}

impl TurnKind {
    pub fn as_str(self) -> &'static str {
        match self {
            TurnKind::NewUserAsk => "new_user_ask",
            TurnKind::MechanicalContinuation => "mechanical_continuation",
            TurnKind::ErrorContinuation => "error_continuation",
            TurnKind::Unknown => "unknown",
        }
    }
}

/// One content block of the final turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKind {
    /// User signal: text, an image, or a document.
    UserSignal,
    /// Machine-produced result. `is_error` is the wire's error flag.
    ToolResult { is_error: bool },
}

/// A minimal view of the final turn. The host projects its own messages in.
///
/// `text` is `Some` when the wire sent plain-string content; `blocks` is used
/// otherwise. When both are present `text` wins, matching the wires that allow
/// a string as the whole body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuralMessage {
    pub role: String,
    pub text: Option<String>,
    pub blocks: Vec<BlockKind>,
}

impl StructuralMessage {
    pub fn user_text(text: impl Into<String>) -> StructuralMessage {
        StructuralMessage {
            role: "user".to_string(),
            text: Some(text.into()),
            blocks: Vec::new(),
        }
    }

    pub fn user_blocks(blocks: Vec<BlockKind>) -> StructuralMessage {
        StructuralMessage {
            role: "user".to_string(),
            text: None,
            blocks,
        }
    }
}

/// Classify the FINAL turn. Only the last message matters.
///
/// Projection contract (the host's job, stated here because correctness
/// depends on it):
///
/// - A plain user message → [`StructuralMessage::user_text`]. Empty text is
///   `Unknown`, never `NewUserAsk`.
/// - A user message carrying attachments → text wins if non-empty, else the
///   attachments become [`BlockKind::UserSignal`].
/// - Consecutive trailing tool results fold into ONE synthetic final user
///   message of [`BlockKind::ToolResult`] blocks, each carrying its own error
///   flag. This is what makes a tool round read as a continuation rather than a
///   new ask.
/// - Anything else (an assistant tail, a summary) is not a user turn →
///   `Unknown`.
pub fn classify(messages: &[StructuralMessage]) -> TurnKind {
    let Some(last) = messages.last() else {
        return TurnKind::Unknown;
    };
    if last.role != "user" {
        return TurnKind::Unknown;
    }
    if let Some(text) = last.text.as_deref() {
        return if text.trim().is_empty() {
            TurnKind::Unknown
        } else {
            TurnKind::NewUserAsk
        };
    }
    if last.blocks.is_empty() {
        return TurnKind::Unknown;
    }

    let mut saw_tool_result = false;
    let mut saw_error = false;
    for block in &last.blocks {
        match *block {
            BlockKind::ToolResult { is_error } => {
                saw_tool_result = true;
                saw_error |= is_error;
            }
            BlockKind::UserSignal => return TurnKind::NewUserAsk,
        }
    }
    if saw_error {
        TurnKind::ErrorContinuation
    } else if saw_tool_result {
        TurnKind::MechanicalContinuation
    } else {
        TurnKind::Unknown
    }
}

pub const MIN_VERBOSITY_LEVEL: u8 = 0;
pub const MAX_VERBOSITY_LEVEL: u8 = 4;
pub const DEFAULT_VERBOSITY_LEVEL: u8 = 2;

/// The conciseness directives. `0` is no directive; `None` means the same.
///
/// FROZEN. These strings are cache-sensitive: every session pinned at a level
/// sends these exact bytes as part of its system prompt. Rewording one is a
/// deliberate cache bust, not an edit.
pub fn verbosity_directive(level: u8) -> Option<&'static str> {
    match level {
        1 => Some(
            "Skip preamble and postamble. Do not announce what you are about to do or recap what you just did; start with the substance.",
        ),
        2 => Some(
            "Skip preamble and postamble; start with the substance. Never restate code, file contents, diffs, or tool output that already appear in this conversation — reference them by path and line instead. After a tool call succeeds, continue without narrating the result.",
        ),
        3 => Some(
            "Skip preamble and postamble. Never restate code, file contents, diffs, or tool output already in this conversation — cite the exact file path and line or symbol instead, always; a reference that omits the location is not a reference. Give conclusions only; omit rationale unless the user asks why. Prefer the smallest edit over rewriting whole files. Keep prose to the minimum needed to be unambiguous. Never drop anything the turn or task needs to be correct, including negations (not, never, no, only, except) — shorten how you say it, not what you say. Use full prose for destructive or irreversible actions, security warnings, and any multi-step sequence where brevity would create ambiguity.",
        ),
        4 => Some(
            "Minimum tokens. Fragments fine. No preamble, no postamble, no restating context, no rationale. Answer, smallest-possible edits, nothing else. Never drop anything the turn or task needs to be correct, including negations (not, never, no, only, except). Use full prose for destructive or irreversible actions, security warnings, and any multi-step sequence where brevity would create ambiguity.",
        ),
        _ => None,
    }
}

/// Default wrapper for the directive. A host should pass its own so a body
/// crossing several proxies never collides on the same tag.
///
/// Must be a full opening XML-style tag `<name>`; the closing tag is derived
/// from it.
pub const DEFAULT_STEERING_SENTINEL: &str = "<acp_output_steering>";

/// The sentinel-wrapped directive, or `None` at L0.
pub fn render_steering_block(level: u8, sentinel: &str) -> Option<String> {
    let text = verbosity_directive(level)?;
    let closing = closing_tag(sentinel);
    Some(format!("{sentinel}\n{text}\n{closing}"))
}

/// `</name` derived from `<name>`.
///
/// Returns the opening half of the closing tag without its `>`, matching the
/// upstream's slicing: the caller searches for it and consumes through the
/// `>`, which also finds a malformed unclosed block's end.
fn closing_tag(sentinel: &str) -> String {
    let name = sentinel.strip_prefix('<').unwrap_or(sentinel);
    format!("</{name}")
}

/// The result of placing a steering block into a prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    pub updated: String,
    /// False when the prompt already held exactly this block — which is what
    /// makes a retry a byte-stable no-op.
    pub changed: bool,
}

/// Idempotently place `block` in `existing`.
///
/// When a block wrapped by `sentinel` is already present it is replaced in
/// place, keeping the surrounding text; otherwise the block is appended at the
/// tail. Never prepends — the prefix cache needs stable head bytes.
///
/// A malformed (unclosed) sentinel block is consumed to the end of the string,
/// so re-applying neither duplicates the closing tag nor grows the block.
pub fn apply_to_prompt(existing: &str, block: &str, sentinel: &str) -> Placement {
    let closing = closing_tag(sentinel);
    match existing.find(sentinel) {
        Some(start) => {
            // Consume through the closing tag's `>`, or to the end when the
            // block was never closed.
            let end = match existing[start..].find(&closing) {
                Some(rel) => {
                    let after = start + rel + closing.len();
                    match existing[after..].find('>') {
                        Some(gt) => after + gt + 1,
                        None => existing.len(),
                    }
                }
                None => existing.len(),
            };
            let prefix = existing[..start].trim_end();
            let tail = existing[end..].trim_start_matches('\n');
            let mut parts: Vec<&str> = Vec::with_capacity(3);
            if !prefix.is_empty() {
                parts.push(prefix);
            }
            parts.push(block);
            if !tail.is_empty() {
                parts.push(tail);
            }
            let updated = parts.join("\n\n");
            Placement {
                changed: updated != existing,
                updated,
            }
        }
        None => {
            let trimmed = existing.trim();
            let updated = if trimmed.is_empty() {
                block.to_string()
            } else {
                format!("{}\n\n{block}", existing.trim_end())
            };
            Placement {
                changed: updated != existing,
                updated,
            }
        }
    }
}

/// Ascending shared effort ladder. A host maps these onto its wire field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Effort {
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
}

impl Effort {
    pub fn as_str(self) -> &'static str {
        match self {
            Effort::Minimal => "minimal",
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
            Effort::Xhigh => "xhigh",
        }
    }

    pub fn parse(value: &str) -> Option<Effort> {
        match value {
            "minimal" => Some(Effort::Minimal),
            "low" => Some(Effort::Low),
            "medium" => Some(Effort::Medium),
            "high" => Some(Effort::High),
            "xhigh" => Some(Effort::Xhigh),
            _ => None,
        }
    }
}

/// Clamp-only floor: the value to write, or `None` when nothing should change.
///
/// Returns `None` for an absent field, an unrecognized value, or a value
/// already at/below `floor`. It never raises — a client that explicitly asked
/// for `minimal` keeps it, and an unrecognized value is left alone rather than
/// coerced (coercing something that sits below the floor would be a raise).
pub fn clamp_effort_to_floor(current: Option<&str>, floor: Effort) -> Option<Effort> {
    let current = Effort::parse(current?)?;
    if current > floor {
        Some(floor)
    } else {
        None
    }
}

/// Resolved steering configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputSteeringConfig {
    /// Master switch. Off by default — both levers are inert until asked for.
    pub enabled: bool,
    /// 0–4; 0 injects nothing.
    pub verbosity_level: u8,
    /// Lower effort on mechanical continuations. On whenever `enabled` is.
    pub effort_routing: bool,
}

impl Default for OutputSteeringConfig {
    fn default() -> OutputSteeringConfig {
        OutputSteeringConfig {
            enabled: false,
            verbosity_level: DEFAULT_VERBOSITY_LEVEL,
            effort_routing: true,
        }
    }
}

/// Resolve a raw level, keeping the distinction the upstream makes: an
/// *omitted* field silently takes the default, while a *present* but
/// out-of-range value falls back **with a warning**. A valid partial config
/// must not be accused of being malformed.
pub fn resolve_verbosity_level(value: Option<&Value>) -> (u8, Option<String>) {
    let default = DEFAULT_VERBOSITY_LEVEL;
    let Some(value) = value else {
        return (default, None);
    };
    if let Some(n) = value.as_u64() {
        if (MIN_VERBOSITY_LEVEL as u64..=MAX_VERBOSITY_LEVEL as u64).contains(&n) {
            return (n as u8, None);
        }
    }
    (
        default,
        Some(format!(
            "[config] output_steering.verbosity_level must be an integer {MIN_VERBOSITY_LEVEL}-{MAX_VERBOSITY_LEVEL}; got {value} — falling back to {default}"
        )),
    )
}

/// Validate a raw `output_steering` config object.
///
/// Malformed fields fall back to defaults rather than erroring; warnings are
/// returned as data so the host can log them. Non-object input yields pure
/// defaults.
pub fn resolve_config(value: Option<&Value>) -> (OutputSteeringConfig, Vec<String>) {
    let Some(object) = value.and_then(Value::as_object) else {
        return (OutputSteeringConfig::default(), Vec::new());
    };
    let (level, warning) = resolve_verbosity_level(object.get("verbosity_level"));
    let config = OutputSteeringConfig {
        // Strictly `true`: a stray truthy value must not switch this on.
        enabled: object.get("enabled").and_then(Value::as_bool) == Some(true),
        verbosity_level: level,
        // Off only when explicitly false.
        effort_routing: object.get("effort_routing").and_then(Value::as_bool) != Some(false),
    };
    (config, warning.into_iter().collect())
}

/// The decision for one turn. Landing it is the host's job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decision {
    /// Echoed for logging: what the final turn looked like.
    pub turn_kind: TurnKind,
    /// 0 means inject nothing. Always 0 while disabled.
    pub verbosity_level: u8,
    /// True iff enabled && effort routing && a mechanical continuation. The
    /// host lowers each effort field the client already sent toward its wire
    /// floor — never injecting an absent field, never raising a value.
    pub lower_effort: bool,
}

/// Single entry point: structural view of the final message list + config →
/// decision. Pure and side-effect free.
pub fn decide(messages: &[StructuralMessage], config: OutputSteeringConfig) -> Decision {
    let turn_kind = classify(messages);
    Decision {
        turn_kind,
        verbosity_level: if config.enabled {
            config.verbosity_level
        } else {
            0
        },
        lower_effort: config.enabled
            && config.effort_routing
            && turn_kind == TurnKind::MechanicalContinuation,
    }
}

/// Project yak's own message list onto the structural view this module needs.
///
/// The only judgement call is the tail: a run of trailing tool results must
/// fold into ONE synthetic user message, or a tool round would look like a
/// non-user turn and be classified `Unknown` — losing exactly the case effort
/// routing exists for. Everything before that tail is irrelevant, because only
/// the final turn is classified.
pub fn project(messages: &[crate::providers::Msg]) -> Vec<StructuralMessage> {
    use crate::providers::Msg;

    // Trailing tool results, in order, form the final synthetic turn.
    let mut tail: Vec<BlockKind> = Vec::new();
    for msg in messages.iter().rev() {
        match msg {
            Msg::ToolResult { error, .. } => tail.push(BlockKind::ToolResult {
                is_error: error.is_some(),
            }),
            _ => break,
        }
    }

    if !tail.is_empty() {
        // Reverse walk, so flip back to wire order.
        tail.reverse();
        return vec![StructuralMessage::user_blocks(tail)];
    }

    // No trailing tool results: the final message speaks for itself.
    match messages.last() {
        // A multimodal-only turn: the attachments are the user signal.
        Some(Msg::User {
            text, attachments, ..
        }) if text.trim().is_empty() && !attachments.is_empty() => {
            vec![StructuralMessage::user_blocks(vec![BlockKind::UserSignal])]
        }
        Some(Msg::User { text, .. }) => vec![StructuralMessage::user_text(text.clone())],
        // An assistant tail (a completed reply) or a summary is not a user
        // turn — there is nothing to steer.
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool_result(is_error: bool) -> BlockKind {
        BlockKind::ToolResult { is_error }
    }

    // ── classification ──────────────────────────────────────────────────────

    #[test]
    fn a_user_text_message_is_a_new_ask() {
        let msgs = [StructuralMessage::user_text("fix the bug")];
        assert_eq!(classify(&msgs), TurnKind::NewUserAsk);
    }

    #[test]
    fn an_empty_user_text_is_unknown_not_a_new_ask() {
        // Guessing here would let an empty turn be treated as fresh work.
        let msgs = [StructuralMessage::user_text("   ")];
        assert_eq!(classify(&msgs), TurnKind::Unknown);
    }

    #[test]
    fn no_messages_is_unknown() {
        assert_eq!(classify(&[]), TurnKind::Unknown);
    }

    #[test]
    fn a_trailing_assistant_message_is_unknown() {
        let msgs = [StructuralMessage {
            role: "assistant".into(),
            text: Some("here you go".into()),
            blocks: Vec::new(),
        }];
        assert_eq!(classify(&msgs), TurnKind::Unknown);
    }

    #[test]
    fn tool_results_alone_are_a_mechanical_continuation() {
        let msgs = [StructuralMessage::user_blocks(vec![tool_result(false)])];
        assert_eq!(classify(&msgs), TurnKind::MechanicalContinuation);
    }

    #[test]
    fn a_tool_error_is_an_error_continuation() {
        let msgs = [StructuralMessage::user_blocks(vec![
            tool_result(false),
            tool_result(true),
        ])];
        assert_eq!(classify(&msgs), TurnKind::ErrorContinuation);
    }

    #[test]
    fn an_error_outranks_a_clean_result_regardless_of_order() {
        let msgs = [StructuralMessage::user_blocks(vec![
            tool_result(true),
            tool_result(false),
        ])];
        assert_eq!(classify(&msgs), TurnKind::ErrorContinuation);
    }

    #[test]
    fn a_user_signal_block_alongside_tool_results_is_a_new_ask() {
        let msgs = [StructuralMessage::user_blocks(vec![
            tool_result(false),
            BlockKind::UserSignal,
        ])];
        assert_eq!(classify(&msgs), TurnKind::NewUserAsk);
    }

    #[test]
    fn a_blockless_user_message_is_unknown() {
        let msgs = [StructuralMessage::user_blocks(Vec::new())];
        assert_eq!(classify(&msgs), TurnKind::Unknown);
    }

    #[test]
    fn only_the_final_message_is_classified() {
        let msgs = [
            StructuralMessage::user_text("original ask"),
            StructuralMessage::user_blocks(vec![tool_result(false)]),
        ];
        assert_eq!(classify(&msgs), TurnKind::MechanicalContinuation);
    }

    // ── directives ──────────────────────────────────────────────────────────

    #[test]
    fn level_zero_has_no_directive() {
        assert!(verbosity_directive(0).is_none());
        assert!(verbosity_directive(5).is_none());
        assert!(verbosity_directive(u8::MAX).is_none());
    }

    #[test]
    fn every_level_from_one_to_four_has_a_substantial_directive() {
        for level in 1..=4u8 {
            let text = verbosity_directive(level).expect("level must exist");
            assert!(text.len() > 60, "L{level} is only {} bytes", text.len());
        }
    }

    #[test]
    fn the_higher_levels_keep_the_correctness_guardrail() {
        // L3/L4 trade away prose, not facts. The negation clause is the thing
        // that keeps a shortened answer from inverting its meaning.
        for level in [3u8, 4u8] {
            let text = verbosity_directive(level).unwrap();
            assert!(text.contains("negations"), "L{level} lost the guardrail");
            assert!(
                text.contains("destructive or irreversible"),
                "L{level} lost the safety carve-out"
            );
        }
    }

    #[test]
    fn the_rendered_block_is_wrapped_in_the_sentinel() {
        let block = render_steering_block(2, DEFAULT_STEERING_SENTINEL).unwrap();
        assert!(block.starts_with(DEFAULT_STEERING_SENTINEL));
        assert!(block.ends_with("</acp_output_steering>"));
    }

    #[test]
    fn a_custom_sentinel_produces_a_matching_closing_tag() {
        let block = render_steering_block(1, "<my_steering>").unwrap();
        assert!(block.ends_with("</my_steering>"), "got: {block}");
    }

    // ── placement ───────────────────────────────────────────────────────────

    #[test]
    fn a_block_is_appended_to_a_non_empty_prompt() {
        let block = render_steering_block(1, DEFAULT_STEERING_SENTINEL).unwrap();
        let p = apply_to_prompt("you are a helpful agent", &block, DEFAULT_STEERING_SENTINEL);
        assert!(p.changed);
        assert!(p.updated.starts_with("you are a helpful agent"));
        assert!(p.updated.contains(DEFAULT_STEERING_SENTINEL));
    }

    #[test]
    fn placing_into_an_empty_prompt_yields_just_the_block() {
        let block = render_steering_block(1, DEFAULT_STEERING_SENTINEL).unwrap();
        let p = apply_to_prompt("   ", &block, DEFAULT_STEERING_SENTINEL);
        assert_eq!(p.updated, block);
    }

    #[test]
    fn re_applying_the_same_block_is_a_byte_stable_no_op() {
        // This is what makes a compress retry safe: the second attempt must
        // not change a single byte, or the prefix cache is lost.
        let block = render_steering_block(2, DEFAULT_STEERING_SENTINEL).unwrap();
        let once = apply_to_prompt("system prompt", &block, DEFAULT_STEERING_SENTINEL);
        let twice = apply_to_prompt(&once.updated, &block, DEFAULT_STEERING_SENTINEL);
        assert!(!twice.changed, "re-application must be a no-op");
        assert_eq!(twice.updated, once.updated);
    }

    #[test]
    fn a_different_level_replaces_the_block_in_place() {
        let l1 = render_steering_block(1, DEFAULT_STEERING_SENTINEL).unwrap();
        let l3 = render_steering_block(3, DEFAULT_STEERING_SENTINEL).unwrap();
        let first = apply_to_prompt("head text", &l1, DEFAULT_STEERING_SENTINEL);
        let second = apply_to_prompt(&first.updated, &l3, DEFAULT_STEERING_SENTINEL);
        assert!(second.changed);
        assert!(second.updated.contains("conclusions only"), "L3 must be in");
        assert!(
            !second.updated.contains("Do not announce what you are about"),
            "L1 must be gone"
        );
        assert!(
            second.updated.starts_with("head text"),
            "surrounding text must survive"
        );
    }

    #[test]
    fn text_after_the_block_is_preserved() {
        let block = render_steering_block(1, DEFAULT_STEERING_SENTINEL).unwrap();
        let existing = format!("head\n\n{block}\n\ntail instructions");
        let p = apply_to_prompt(
            &existing,
            &render_steering_block(2, DEFAULT_STEERING_SENTINEL).unwrap(),
            DEFAULT_STEERING_SENTINEL,
        );
        assert!(p.updated.contains("head"), "got: {}", p.updated);
        assert!(
            p.updated.contains("tail instructions"),
            "got: {}",
            p.updated
        );
    }

    #[test]
    fn an_unclosed_sentinel_block_is_consumed_to_the_end() {
        // A malformed block must not accumulate a second closing tag.
        let existing = format!("head\n\n{DEFAULT_STEERING_SENTINEL}\nstale directive");
        let block = render_steering_block(1, DEFAULT_STEERING_SENTINEL).unwrap();
        let p = apply_to_prompt(&existing, &block, DEFAULT_STEERING_SENTINEL);
        assert_eq!(
            p.updated.matches(&closing_tag(DEFAULT_STEERING_SENTINEL)).count(),
            1,
            "exactly one closing tag: {}",
            p.updated
        );
        assert!(p.updated.starts_with("head"));
    }

    #[test]
    fn placement_never_prepends() {
        let block = render_steering_block(1, DEFAULT_STEERING_SENTINEL).unwrap();
        let p = apply_to_prompt("STABLE HEAD", &block, DEFAULT_STEERING_SENTINEL);
        assert!(p.updated.starts_with("STABLE HEAD"));
    }

    // ── effort ──────────────────────────────────────────────────────────────

    #[test]
    fn effort_is_lowered_toward_the_floor() {
        assert_eq!(clamp_effort_to_floor(Some("high"), Effort::Low), Some(Effort::Low));
        assert_eq!(
            clamp_effort_to_floor(Some("xhigh"), Effort::Low),
            Some(Effort::Low)
        );
    }

    #[test]
    fn effort_is_never_raised() {
        // A client that asked for minimal keeps minimal.
        assert_eq!(clamp_effort_to_floor(Some("minimal"), Effort::Low), None);
        assert_eq!(clamp_effort_to_floor(Some("low"), Effort::Low), None);
    }

    #[test]
    fn an_absent_or_unknown_effort_is_left_alone() {
        assert_eq!(clamp_effort_to_floor(None, Effort::Low), None);
        assert_eq!(clamp_effort_to_floor(Some("turbo"), Effort::Low), None);
        assert_eq!(clamp_effort_to_floor(Some(""), Effort::Low), None);
    }

    #[test]
    fn the_floor_can_be_medium() {
        assert_eq!(
            clamp_effort_to_floor(Some("high"), Effort::Medium),
            Some(Effort::Medium)
        );
        assert_eq!(clamp_effort_to_floor(Some("low"), Effort::Medium), None);
    }

    // ── config ──────────────────────────────────────────────────────────────

    #[test]
    fn steering_is_off_by_default() {
        let (config, warnings) = resolve_config(None);
        assert!(!config.enabled);
        assert_eq!(config.verbosity_level, DEFAULT_VERBOSITY_LEVEL);
        assert!(config.effort_routing);
        assert!(warnings.is_empty());
    }

    #[test]
    fn a_valid_partial_config_produces_no_warning() {
        let (config, warnings) = resolve_config(Some(&json!({"enabled": true})));
        assert!(config.enabled);
        assert_eq!(config.verbosity_level, DEFAULT_VERBOSITY_LEVEL);
        assert!(warnings.is_empty(), "a valid partial must not warn");
    }

    #[test]
    fn an_out_of_range_level_warns_and_falls_back() {
        let (config, warnings) = resolve_config(Some(&json!({"verbosity_level": 9})));
        assert_eq!(config.verbosity_level, DEFAULT_VERBOSITY_LEVEL);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("verbosity_level"));
    }

    #[test]
    fn a_negative_or_fractional_level_falls_back_with_a_warning() {
        for bad in [json!(-1), json!(2.5), json!("2")] {
            let (config, warnings) = resolve_config(Some(&json!({"verbosity_level": bad})));
            assert_eq!(config.verbosity_level, DEFAULT_VERBOSITY_LEVEL);
            assert_eq!(warnings.len(), 1, "{bad} must warn");
        }
    }

    #[test]
    fn enabled_requires_a_real_true() {
        let (config, _) = resolve_config(Some(&json!({"enabled": "yes"})));
        assert!(!config.enabled, "a truthy string must not switch this on");
    }

    #[test]
    fn effort_routing_is_on_unless_explicitly_false() {
        let (on, _) = resolve_config(Some(&json!({"enabled": true})));
        assert!(on.effort_routing);
        let (off, _) = resolve_config(Some(&json!({"effort_routing": false})));
        assert!(!off.effort_routing);
    }

    #[test]
    fn a_non_object_config_yields_defaults() {
        let (config, warnings) = resolve_config(Some(&json!("nonsense")));
        assert_eq!(config, OutputSteeringConfig::default());
        assert!(warnings.is_empty());
    }

    // ── decisions ───────────────────────────────────────────────────────────

    #[test]
    fn a_disabled_config_injects_nothing_and_lowers_nothing() {
        let msgs = [StructuralMessage::user_blocks(vec![tool_result(false)])];
        let d = decide(&msgs, OutputSteeringConfig::default());
        assert_eq!(d.verbosity_level, 0);
        assert!(!d.lower_effort);
        // The classification is still reported: it is the observability half.
        assert_eq!(d.turn_kind, TurnKind::MechanicalContinuation);
    }

    #[test]
    fn an_enabled_config_injects_its_level() {
        let msgs = [StructuralMessage::user_text("do the thing")];
        let config = OutputSteeringConfig {
            enabled: true,
            ..Default::default()
        };
        let d = decide(&msgs, config);
        assert_eq!(d.verbosity_level, DEFAULT_VERBOSITY_LEVEL);
        assert!(!d.lower_effort, "a new ask is not mechanical");
    }

    #[test]
    fn effort_is_lowered_only_on_a_mechanical_continuation() {
        let config = OutputSteeringConfig {
            enabled: true,
            ..Default::default()
        };
        let mechanical = [StructuralMessage::user_blocks(vec![tool_result(false)])];
        assert!(decide(&mechanical, config).lower_effort);

        let errored = [StructuralMessage::user_blocks(vec![tool_result(true)])];
        assert!(
            !decide(&errored, config).lower_effort,
            "an error needs reasoning, not less of it"
        );

        let ask = [StructuralMessage::user_text("why?")];
        assert!(!decide(&ask, config).lower_effort);
    }

    #[test]
    fn disabling_effort_routing_keeps_the_verbosity_lever() {
        let config = OutputSteeringConfig {
            enabled: true,
            effort_routing: false,
            ..Default::default()
        };
        let mechanical = [StructuralMessage::user_blocks(vec![tool_result(false)])];
        let d = decide(&mechanical, config);
        assert!(!d.lower_effort);
        assert_eq!(d.verbosity_level, DEFAULT_VERBOSITY_LEVEL);
    }

    // ── projection from yak's own Msg ───────────────────────────────────────

    #[test]
    fn a_plain_user_tail_projects_to_a_new_ask() {
        use crate::providers::Msg;
        let msgs = [Msg::user("please fix it")];
        assert_eq!(classify(&project(&msgs)), TurnKind::NewUserAsk);
    }

    #[test]
    fn trailing_tool_results_fold_into_one_mechanical_turn() {
        use crate::providers::Msg;
        let msgs = [
            Msg::user("run the tests"),
            Msg::ToolResult {
                call_id: "c1".into(),
                name: "bash".into(),
                content: "ok".into(),
                error: None,
                attachments: Vec::new(),
            },
            Msg::ToolResult {
                call_id: "c2".into(),
                name: "bash".into(),
                content: "ok too".into(),
                error: None,
                attachments: Vec::new(),
            },
        ];
        let structural = project(&msgs);
        assert_eq!(structural.len(), 1, "the tail folds into one message");
        assert_eq!(structural[0].blocks.len(), 2, "both results are kept");
        assert_eq!(classify(&structural), TurnKind::MechanicalContinuation);
    }

    #[test]
    fn a_failed_tool_result_projects_to_an_error_continuation() {
        use crate::providers::{Msg, ToolError};
        let msgs = [Msg::ToolResult {
            call_id: "c1".into(),
            name: "bash".into(),
            content: "boom".into(),
            error: Some(ToolError::Failed),
            attachments: Vec::new(),
        }];
        assert_eq!(classify(&project(&msgs)), TurnKind::ErrorContinuation);
    }

    #[test]
    fn an_assistant_tail_projects_to_nothing() {
        use crate::providers::Msg;
        let msgs = [Msg::user("hi"), Msg::Assistant {
            text: "done".into(),
            tool_calls: Vec::new(),
            reasoning: None,
            reasoning_meta: None,
        }];
        assert!(project(&msgs).is_empty());
        assert_eq!(classify(&project(&msgs)), TurnKind::Unknown);
    }

    #[test]
    fn an_empty_history_projects_to_nothing() {
        assert!(project(&[]).is_empty());
    }

    #[test]
    fn an_empty_user_text_projects_to_unknown() {
        use crate::providers::Msg;
        let msgs = [Msg::user("  ")];
        assert_eq!(classify(&project(&msgs)), TurnKind::Unknown);
    }
}
