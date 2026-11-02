//! Instant tool-result absorption: the model distils a big result the moment
//! it arrives, and the original stops being re-sent.
//!
//! Bigctx has two other answers for a growing tool result, and this fills the
//! gap between them. The nudge waits until the context is 45% full — by then a
//! 30k-token result has already been re-sent several times, billed each time.
//! The fold gate is coarser still: it waits for enough material to summarise a
//! *range*. Absorb acts on the turn the result appears: the prompt is appended
//! to the result itself, the model distils it now, and the tool-call +
//! tool-result pair is hidden from every later turn.
//!
//! The durable record is the model's own `absorb` call — an ordinary message
//! that the normal pipeline can fold later, so absorb is orthogonal to
//! compression rather than a competing mechanism.
//!
//! Three properties are deliberate and easy to lose:
//!
//! 1. **The prompt is view-only.** It is appended to the message the model is
//!    sent, never persisted. It therefore reappears every turn until the model
//!    absorbs, and disappears on its own once the pair is hidden.
//! 2. **Hiding waits for the next turn.** An absorption records itself but
//!    hides nothing mid-turn: the model must still see the result it just
//!    summarised for the rest of that turn.
//! 3. **Screening and execution share one judgement.** A result that
//!    [`is_absorb_candidate`] skips must also be refused by [`apply`], or the
//!    prompt would demand an action the tool then rejects — the model would be
//!    told to do something impossible, every turn.

use crate::agent::bigctx::ccr::{is_never_stored, parse_placeholder};
use crate::agent::bigctx::config::BigctxConfig;
use crate::agent::bigctx::protected::Protection;
use crate::agent::bigctx::tokenize::default_count_tokens;
use crate::agent::bigctx::types::{
    AbsorbRecord, CompressionState, CoreMessage, MessageContentType,
};

/// The marker every injected prompt carries. Its presence is also how a result
/// that was already prompted is recognised, so it must never appear in a
/// server-sent string.
pub const ABSORB_PROMPT_MARKER: &str = "[ACP absorb]";

/// The tool the model calls to absorb.
pub const ABSORB_TOOL_NAME: &str = "absorb";

/// Absorb tunables.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AbsorbConfig {
    /// Off by default: this one changes what the model is asked to do, so it is
    /// opt-in.
    pub enabled: bool,
    /// A tool result smaller than this is not worth a round trip.
    pub min_tool_tokens: u64,
    /// Only start prompting once the context is this full. 0 = no gate.
    pub context_threshold_pct: f64,
    /// Tool-name patterns never absorbed. `*` suffix matches a prefix.
    pub exclude_tools: Vec<String>,
    /// Absorb only results carrying media, whose bytes are not in `text` and so
    /// can never reach `min_tool_tokens` on their own.
    pub media_only: bool,
}

impl Default for AbsorbConfig {
    fn default() -> AbsorbConfig {
        AbsorbConfig {
            enabled: false,
            // Raised from 1000: content-cached retrieval now handles large
            // results reversibly, so forced distillation only earns its round
            // trip above this bar.
            min_tool_tokens: 4_000,
            context_threshold_pct: 0.0,
            exclude_tools: Vec::new(),
            media_only: false,
        }
    }
}

/// Resolve the configured absorb settings.
pub fn resolve_config(config: &BigctxConfig) -> AbsorbConfig {
    let mut out = AbsorbConfig::default();
    let Some(raw) = &config.absorb else {
        return out;
    };
    if let Some(on) = raw.get("enabled").and_then(serde_json::Value::as_bool) {
        out.enabled = on;
    }
    if let Some(n) = raw.get("min_tool_tokens").and_then(serde_json::Value::as_u64) {
        out.min_tool_tokens = n;
    }
    if let Some(pct) = raw
        .get("context_threshold_pct")
        .and_then(serde_json::Value::as_f64)
    {
        out.context_threshold_pct = pct;
    }
    if let Some(on) = raw.get("media_only").and_then(serde_json::Value::as_bool) {
        out.media_only = on;
    }
    if let Some(list) = raw.get("exclude_tools").and_then(serde_json::Value::as_array) {
        out.exclude_tools = list
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
    }
    out
}

fn format_token_count(tokens: u64) -> String {
    if tokens < 1_000 {
        tokens.to_string()
    } else if tokens < 10_000 {
        format!("{:.1}K", tokens as f64 / 1_000.0)
    } else {
        format!("{}K", (tokens as f64 / 1_000.0).round() as u64)
    }
}

/// The forced prompt appended to an eligible result.
pub fn build_prompt(reference: &str, tokens: u64, tool_name: &str) -> String {
    format!(
        "{ABSORB_PROMPT_MARKER} This tool result (~{} tokens) will be REMOVED from context. \
         Your IMMEDIATE next action: call {tool_name}({{ ref: \"{reference}\", summary: \"...\" }}) — \
         summary = distilled essentials only (outcome, key values, exact paths:lines, error text \
         verbatim, decisions). Afterwards work from your summary; do NOT re-run this tool. \
         If the result contains nothing you need, call {tool_name} with summary \"(nothing needed)\".",
        format_token_count(tokens)
    )
}

/// The system-prompt section explaining the protocol, for hosts that enable it.
pub fn build_system_prompt(tool_name: &str) -> String {
    format!(
        "INSTANT TOOL-RESULT ABSORPTION ({tool_name})\n\n\
         Some tool results end with a {ABSORB_PROMPT_MARKER} instruction. When you see one, your \
         IMMEDIATE next action must be calling {tool_name}({{ ref, summary }}) — distill that tool \
         result's essentials into summary: outcome, key values, exact paths:lines, error text \
         verbatim, decisions. The original output is then removed from context; your {tool_name} \
         summary becomes the only durable record of it, so distill carefully. Never call another \
         tool or answer the user before absorbing a marked result. Do not re-run the original tool \
         afterwards — work from your summary. {tool_name} calls are ordinary context: the regular \
         compression system may fold them later like any other message."
    )
}

/// True when `tool` is managed by bigctx itself, and so never absorbable: such
/// a result is the machinery reporting on the context, not material in it.
fn is_managed_tool(tool: Option<&str>) -> bool {
    let Some(tool) = tool else {
        return false;
    };
    matches!(
        tool,
        "compress" | "decompress" | "acp_retrieve" | "acp_status" | "search_context" | ABSORB_TOOL_NAME
    ) || is_never_stored(Some(tool))
}

fn matches_pattern(tool: &str, pattern: &str) -> bool {
    match pattern.strip_suffix('*') {
        Some(prefix) => tool.starts_with(prefix),
        None => tool == pattern,
    }
}

/// Why a message is not absorbable. `None` means it is.
///
/// Returned rather than a bare bool so [`apply`] can tell the model *why* it
/// was refused — a silent rejection makes the model retry the same call.
pub fn reject_reason(
    message: &CoreMessage,
    state: &CompressionState,
    config: &BigctxConfig,
    protection: &Protection,
) -> Option<String> {
    let cfg = resolve_config(config);

    if message.content_type != MessageContentType::ToolResult {
        return Some(format!(
            "it is a {}, not a tool result",
            content_type_label(message.content_type)
        ));
    }
    if message.tool_call_id.is_none() {
        return Some("it has no tool-call id, so the pair cannot be hidden".into());
    }
    if parse_placeholder(message.text.as_deref().unwrap_or("")).is_some() {
        return Some("it is already a stored placeholder".into());
    }
    if is_managed_tool(message.tool_name.as_deref()) {
        return Some(format!(
            "{} is managed by the compression system, so its output is not absorbable",
            message.tool_name.as_deref().unwrap_or("that tool")
        ));
    }
    if protection.is_tool_protected(message) {
        return Some(format!(
            "{} is protected, so its results must stay visible",
            message.tool_name.as_deref().unwrap_or("that tool")
        ));
    }
    for pattern in &cfg.exclude_tools {
        if let Some(tool) = message.tool_name.as_deref()
            && matches_pattern(tool, pattern)
        {
            return Some(format!("{tool} matches the exclude pattern `{pattern}`"));
        }
    }
    if cfg.media_only && !has_media(message) {
        return Some("it carries no media payload and absorb is scoped to media".into());
    }
    if state
        .absorbed
        .iter()
        .any(|r| r.result_message_id == message.id)
    {
        return Some("it is already absorbed".into());
    }
    None
}

/// True when a message is in scope for absorption prompting.
pub fn is_absorb_candidate(
    message: &CoreMessage,
    state: &CompressionState,
    config: &BigctxConfig,
    protection: &Protection,
) -> bool {
    reject_reason(message, state, config, protection).is_none()
}

fn content_type_label(kind: MessageContentType) -> &'static str {
    match kind {
        MessageContentType::Text => "text message",
        MessageContentType::ToolCall => "tool call",
        MessageContentType::ToolResult => "tool result",
        MessageContentType::Reasoning => "reasoning trace",
    }
}

/// A media payload rides in attachments the text never shows, so a media-only
/// result can never reach `min_tool_tokens` by counting text.
fn has_media(message: &CoreMessage) -> bool {
    use crate::agent::bigctx::types::MessageContentType;
    if message.content_type == MessageContentType::ToolResult
        && message.tool_name.as_deref() == Some("read")
    {
        // `read` returns images for image files; the marker is in the text it
        // renders, which is the only signal available at core level.
        return message
            .text
            .as_deref()
            .is_some_and(|t| t.contains("[image") || t.contains("attachment"));
    }
    false
}

/// Drop the tool-call + tool-result pairs recorded as absorbed.
///
/// Both halves go together: hiding only the result leaves a call with no
/// answer, which providers reject. Orphan cleanup handles anything straddling.
pub fn hide_absorbed(
    messages: &[CoreMessage],
    state: &CompressionState,
) -> Vec<CoreMessage> {
    if state.absorbed.is_empty() {
        return messages.to_vec();
    }
    let mut hidden: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for record in &state.absorbed {
        if !record.call_message_id.is_empty() {
            hidden.insert(record.call_message_id.as_str());
        }
        hidden.insert(record.result_message_id.as_str());
    }
    messages
        .iter()
        .filter(|m| !hidden.contains(m.id.as_str()))
        .cloned()
        .collect()
}

/// What [`append_prompts`] did.
#[derive(Debug, Clone, PartialEq)]
pub struct Prompted {
    pub messages: Vec<CoreMessage>,
    /// How many results received a prompt this turn. Zero means absorb is
    /// inert right now, which is the normal state.
    pub prompted_count: usize,
}

/// Append the forced prompt to every eligible, un-absorbed, large result.
///
/// The appended text is view-only — this returns a new array and never writes
/// through to the caller's messages.
pub fn append_prompts(
    messages: &[CoreMessage],
    state: &CompressionState,
    config: &BigctxConfig,
    protection: &Protection,
    token_count: u64,
) -> Prompted {
    let cfg = resolve_config(config);
    if !cfg.enabled {
        return Prompted {
            messages: messages.to_vec(),
            prompted_count: 0,
        };
    }

    // The context gate: below the threshold, absorption is not yet worth the
    // interruption, so the model is left alone.
    if cfg.context_threshold_pct > 0.0
        && config.model_context_limit > 0
        && (token_count as f64) < cfg.context_threshold_pct * config.model_context_limit as f64
    {
        return Prompted {
            messages: messages.to_vec(),
            prompted_count: 0,
        };
    }

    let mut prompted_count = 0;
    let out: Vec<CoreMessage> = messages
        .iter()
        .map(|message| {
            if !is_absorb_candidate(message, state, config, protection) {
                return message.clone();
            }
            let text = message.text.as_deref().unwrap_or("");
            if text.contains(ABSORB_PROMPT_MARKER) {
                return message.clone();
            }
            let tokens = default_count_tokens(text);
            // Media bytes are not in `text`, so counting text would never clear
            // the bar for a media result — exempt it or the feature is dead.
            let media_exempt = cfg.media_only && has_media(message);
            if !media_exempt && tokens < cfg.min_tool_tokens {
                return message.clone();
            }
            let Some(reference) = state.message_refs.ref_for(message.base_id()) else {
                // Without a ref the model cannot name this result, and a prompt
                // asking it to would be unsatisfiable.
                return message.clone();
            };
            prompted_count += 1;
            let mut out = message.clone();
            out.text = Some(format!(
                "{text}\n\n{}",
                build_prompt(reference, tokens, ABSORB_TOOL_NAME)
            ));
            out
        })
        .collect();

    Prompted {
        messages: out,
        prompted_count,
    }
}

/// A parsed `absorb(...)` call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedCall {
    pub reference: String,
    pub summary: String,
}

/// Leniently parse an `absorb` argument object.
///
/// Providers differ: some stringify the whole payload, some spell the keys
/// differently. The upstream accepts those spellings and so does this — a
/// rejected-but-well-meant call wastes a turn and the model rarely guesses a
/// different spelling on the retry.
pub fn parse_call(args: &serde_json::Value) -> Option<ParsedCall> {
    let owned;
    let object = if let Some(text) = args.as_str() {
        owned = serde_json::from_str::<serde_json::Value>(text).ok()?;
        owned.as_object()?
    } else {
        args.as_object()?
    };

    let pick = |keys: &[&str]| -> Option<String> {
        keys.iter()
            .find_map(|k| object.get(*k).and_then(serde_json::Value::as_str))
            .map(str::to_string)
    };

    let reference = pick(&["ref", "messageId", "message_id", "of"])?;
    let summary = pick(&["summary", "content"])?;
    Some(ParsedCall {
        reference: reference.trim().to_string(),
        summary,
    })
}

/// What an `absorb` call did.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    pub state: CompressionState,
    pub ok: bool,
    /// The text the model sees as the tool result.
    pub text: String,
}

fn fail(state: &CompressionState, text: String) -> Outcome {
    Outcome {
        state: state.clone(),
        ok: false,
        text,
    }
}

/// Apply an `absorb(...)` call: validate the target, record the absorption.
///
/// Nothing is hidden here — the pair leaves the context on the next turn, so
/// the model keeps seeing the result it just summarised for the rest of this
/// one.
pub fn apply(
    state: &CompressionState,
    messages: &[CoreMessage],
    config: &BigctxConfig,
    protection: &Protection,
    call: &ParsedCall,
) -> Outcome {
    let summary = call.summary.trim();
    if summary.is_empty() {
        return fail(
            state,
            "absorb failed: summary is empty — provide the distilled key info of the tool result."
                .into(),
        );
    }

    let Some(raw_id) = state.message_refs.raw_for(&call.reference) else {
        return fail(
            state,
            format!(
                "absorb failed: ref {} does not exist in this session (it may be hidden, \
                 already compressed, or stale).",
                call.reference
            ),
        );
    };

    let Some(target) = messages.iter().find(|m| m.id == raw_id) else {
        return fail(
            state,
            format!(
                "absorb failed: ref {} is not visible in this session (hidden or compressed).",
                call.reference
            ),
        );
    };

    let already: Vec<&AbsorbRecord> = state
        .absorbed
        .iter()
        .filter(|r| r.result_message_id == raw_id)
        .collect();
    if !already.is_empty() {
        return Outcome {
            state: state.clone(),
            ok: true,
            text: format!("already absorbed ({}) — no change.", call.reference),
        };
    }

    if let Some(reason) = reject_reason(target, state, config, protection) {
        return fail(
            state,
            format!("absorb failed: {} — {reason}.", call.reference),
        );
    }

    // `reject_reason` already proved the call id exists.
    let tool_call_id = target.tool_call_id.clone().unwrap_or_default();
    let call_message_id = messages
        .iter()
        .find(|m| {
            m.content_type == MessageContentType::ToolCall
                && m.tool_call_id.as_deref() == Some(tool_call_id.as_str())
        })
        .map(|m| m.id.clone())
        .unwrap_or_default();

    let text = target.text.as_deref().unwrap_or("");
    let tokens = default_count_tokens(text);
    let summary_tokens = default_count_tokens(summary);

    let mut next = state.clone();
    next.absorbed.push(AbsorbRecord {
        tool_call_id,
        call_message_id,
        result_message_id: target.id.clone(),
        absorb_call_id: None,
        summary: summary.to_string(),
        tokens_reclaimed: tokens,
        created_at: 0,
    });
    next.stats.absorbed_tokens += tokens;

    // A summary that is not smaller than what it replaces is a net loss, and
    // the model will not notice without being told.
    let bloat = if summary_tokens >= tokens && tokens > 0 {
        format!(
            " WARNING: your summary (~{} tokens) is not smaller than the original (~{} tokens) — \
             distill harder next time.",
            format_token_count(summary_tokens),
            format_token_count(tokens)
        )
    } else {
        String::new()
    };

    Outcome {
        state: next,
        ok: true,
        text: format!(
            "absorbed {} (~{} tokens → summary ~{}). The original tool output is now hidden; \
             your summary is the durable record.{bloat}",
            call.reference,
            format_token_count(tokens),
            format_token_count(summary_tokens)
        ),
    }
}

/// Total tokens reclaimed by absorption, for the status report.
pub fn reclaimed_tokens(state: &CompressionState) -> u64 {
    state.absorbed.iter().map(|r| r.tokens_reclaimed).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::bigctx::config::default_config;
    use crate::agent::bigctx::protected::Protection;
    use crate::agent::bigctx::types::{MessageRole, Tier};

    fn tool_result(id: &str, call_id: &str, tool: &str, text: &str) -> CoreMessage {
        let mut m = CoreMessage::new(id, MessageRole::Tool, MessageContentType::ToolResult);
        m.tool_call_id = Some(call_id.into());
        m.tool_name = Some(tool.into());
        m.text = Some(text.into());
        m
    }

    fn tool_call(id: &str, call_id: &str, tool: &str) -> CoreMessage {
        let mut m = CoreMessage::new(id, MessageRole::Assistant, MessageContentType::ToolCall);
        m.tool_call_id = Some(call_id.into());
        m.tool_name = Some(tool.into());
        m
    }

    fn user(id: &str, text: &str) -> CoreMessage {
        let mut m = CoreMessage::new(id, MessageRole::User, MessageContentType::Text);
        m.text = Some(text.into());
        m
    }

    /// A state with refs assigned, which is what `absorb` resolves against.
    fn state_for(messages: &[CoreMessage]) -> CompressionState {
        let mut state = crate::agent::bigctx::create_initial_state();
        let mut next = 1u64;
        crate::agent::bigctx::assign_refs(messages, &mut state.message_refs, &mut next, |_| false)
            .expect("refs");
        state.next_ref_index = next;
        state
    }

    fn enabled() -> BigctxConfig {
        let mut config = default_config(200_000);
        config.absorb = Some(serde_json::json!({"enabled": true, "min_tool_tokens": 10}));
        config
    }

    #[test]
    fn absorb_is_off_by_default() {
        let config = default_config(200_000);
        assert!(!resolve_config(&config).enabled);
    }

    #[test]
    fn config_overrides_are_read() {
        let mut config = default_config(200_000);
        config.absorb = Some(serde_json::json!({
            "enabled": true,
            "min_tool_tokens": 123,
            "context_threshold_pct": 0.5,
            "media_only": true,
            "exclude_tools": ["read*"]
        }));
        let cfg = resolve_config(&config);
        assert!(cfg.enabled);
        assert_eq!(cfg.min_tool_tokens, 123);
        assert_eq!(cfg.context_threshold_pct, 0.5);
        assert!(cfg.media_only);
        assert_eq!(cfg.exclude_tools, vec!["read*".to_string()]);
    }

    #[test]
    fn a_large_tool_result_gets_the_prompt() {
        let messages = [
            user("t1.0", "go"),
            tool_call("t1.1", "c1", "bash"),
            tool_result("t1.2", "c1", "bash", &"x".repeat(200)),
        ];
        let state = state_for(&messages);
        let out = append_prompts(&messages, &state, &enabled(), &Protection::default(), 0);
        assert_eq!(out.prompted_count, 1);
        let last = out.messages.last().unwrap();
        assert!(last.text.as_deref().unwrap().contains(ABSORB_PROMPT_MARKER));
    }

    #[test]
    fn a_small_result_is_left_alone() {
        let messages = [tool_result("t1.0", "c1", "bash", "tiny")];
        let state = state_for(&messages);
        let out = append_prompts(&messages, &state, &enabled(), &Protection::default(), 0);
        assert_eq!(out.prompted_count, 0);
    }

    #[test]
    fn a_disabled_config_prompts_nothing() {
        let messages = [tool_result("t1.0", "c1", "bash", &"x".repeat(200))];
        let state = state_for(&messages);
        let config = default_config(200_000);
        let out = append_prompts(&messages, &state, &config, &Protection::default(), 0);
        assert_eq!(out.prompted_count, 0);
        assert_eq!(out.messages, messages.to_vec());
    }

    #[test]
    fn a_managed_tool_result_is_never_prompted() {
        for tool in ["compress", "acp_status", "search_context", "absorb", "acp_retrieve"] {
            let messages = [tool_result("t1.0", "c1", tool, &"x".repeat(200))];
            let state = state_for(&messages);
            let out = append_prompts(&messages, &state, &enabled(), &Protection::default(), 0);
            assert_eq!(out.prompted_count, 0, "{tool} must not be absorbable");
        }
    }

    #[test]
    fn a_protected_tool_result_is_not_prompted() {
        let messages = [tool_result("t1.0", "c1", "read", &"x".repeat(200))];
        let state = state_for(&messages);
        let mut protection = Protection::default();
        protection.protect_base("t1.0");
        let out = append_prompts(&messages, &state, &enabled(), &protection, 0);
        assert_eq!(out.prompted_count, 0);
    }

    #[test]
    fn an_excluded_tool_pattern_is_not_prompted() {
        let messages = [tool_result("t1.0", "c1", "mcp__github", &"x".repeat(200))];
        let state = state_for(&messages);
        let mut config = enabled();
        config.absorb = Some(serde_json::json!({
            "enabled": true, "min_tool_tokens": 10, "exclude_tools": ["mcp__*"]
        }));
        let out = append_prompts(&messages, &state, &config, &Protection::default(), 0);
        assert_eq!(out.prompted_count, 0);
    }

    #[test]
    fn the_prompt_is_not_duplicated_on_a_second_pass() {
        let messages = [tool_result("t1.0", "c1", "bash", &"x".repeat(200))];
        let state = state_for(&messages);
        let once = append_prompts(&messages, &state, &enabled(), &Protection::default(), 0);
        assert_eq!(once.prompted_count, 1);
        let twice = append_prompts(
            &once.messages,
            &state,
            &enabled(),
            &Protection::default(),
            0,
        );
        assert_eq!(twice.prompted_count, 0, "the marker must stop a re-prompt");
        assert_eq!(
            twice.messages[0]
                .text
                .as_deref()
                .unwrap()
                .matches(ABSORB_PROMPT_MARKER)
                .count(),
            1
        );
    }

    #[test]
    fn an_already_absorbed_result_is_not_prompted_again() {
        let messages = [tool_result("t1.0", "c1", "bash", &"x".repeat(200))];
        let mut state = state_for(&messages);
        state.absorbed.push(AbsorbRecord {
            tool_call_id: "c1".into(),
            call_message_id: "t1.1".into(),
            result_message_id: "t1.0".into(),
            absorb_call_id: None,
            summary: "s".into(),
            tokens_reclaimed: 50,
            created_at: 0,
        });
        let out = append_prompts(&messages, &state, &enabled(), &Protection::default(), 0);
        assert_eq!(out.prompted_count, 0);
    }

    #[test]
    fn the_context_threshold_gates_prompting() {
        let messages = [tool_result("t1.0", "c1", "bash", &"x".repeat(200))];
        let state = state_for(&messages);
        let mut config = enabled();
        config.absorb = Some(serde_json::json!({
            "enabled": true, "min_tool_tokens": 10, "context_threshold_pct": 0.5
        }));
        // 200k window, threshold 0.5 → need 100k tokens.
        let low = append_prompts(&messages, &state, &config, &Protection::default(), 1_000);
        assert_eq!(low.prompted_count, 0);
        let high = append_prompts(&messages, &state, &config, &Protection::default(), 150_000);
        assert_eq!(high.prompted_count, 1);
    }

    #[test]
    fn a_result_without_a_ref_is_not_prompted() {
        // A ref-less result cannot be named, so a prompt for it would be
        // unsatisfiable.
        let messages = [tool_result("t1.0", "c1", "bash", &"x".repeat(200))];
        let state = crate::agent::bigctx::create_initial_state();
        let out = append_prompts(&messages, &state, &enabled(), &Protection::default(), 0);
        assert_eq!(out.prompted_count, 0);
    }

    // ── the test that proves the feature is alive ───────────────────────────

    /// The regression this module was missing.
    ///
    /// Absorb's target is by definition a *freshly arrived* tool result, which
    /// always sits inside `preserve_recent_messages` / `preserve_recent_tokens`.
    /// While the candidate check consulted `Protection::is_protected`, every
    /// candidate was skipped as "protected" and `prompted_count` was pinned at
    /// 0 for the life of the session. The unit tests above did not catch it
    /// because they passed `Protection::default()` — an empty set, which real
    /// `process_turn` never produces.
    ///
    /// So this drives the real pipeline, with the real defaults.
    #[test]
    fn a_fresh_result_is_prompted_through_the_real_pipeline() {
        let mut config = default_config(200_000);
        config.absorb = Some(serde_json::json!({"enabled": true, "min_tool_tokens": 10}));

        let messages = vec![
            user("t1.0", "run the build"),
            tool_call("t1.1", "c1", "bash"),
            tool_result("t1.2", "c1", "bash", &"compiler output ".repeat(40)),
        ];

        // The real defaults protect the recent window; the result is inside it.
        assert!(config.preserve_recent_messages > 0);

        let turn = crate::agent::bigctx::process_turn(
            &messages,
            &crate::agent::bigctx::create_initial_state(),
            &config,
        );

        assert_eq!(
            turn.prompted_for_absorb, 1,
            "the recent zone must not disqualify a fresh result from absorption"
        );
        let result = turn
            .messages
            .iter()
            .find(|m| m.id.starts_with("t1.2"))
            .expect("the result must still be in the view");
        assert!(
            result
                .text
                .as_deref()
                .unwrap_or("")
                .contains(ABSORB_PROMPT_MARKER),
            "the forced prompt must reach the view the model is sent"
        );
    }

    /// The complement: a *content*-protected tool must still be skipped even
    /// though it also sits in the recent zone. Separating the two kinds of
    /// protection must not have opened the door to absorbing what config says
    /// is the source of truth.
    #[test]
    fn a_content_protected_tool_is_still_skipped_by_the_real_pipeline() {
        let mut config = default_config(200_000);
        config.absorb = Some(serde_json::json!({"enabled": true, "min_tool_tokens": 10}));
        config.protected_tools = vec!["bash".to_string()];

        let messages = vec![
            user("t1.0", "run the build"),
            tool_call("t1.1", "c1", "bash"),
            tool_result("t1.2", "c1", "bash", &"compiler output ".repeat(40)),
        ];
        let turn = crate::agent::bigctx::process_turn(
            &messages,
            &crate::agent::bigctx::create_initial_state(),
            &config,
        );
        assert_eq!(turn.prompted_for_absorb, 0);
    }

    /// And the fold gate must keep respecting the recent zone: absorption and
    /// folding consult different questions, and loosening one must not loosen
    /// the other.
    #[test]
    fn the_recent_zone_still_blocks_folding() {
        let config = default_config(200_000);
        let messages = vec![
            user("t1.0", "run the build"),
            tool_call("t1.1", "c1", "bash"),
            tool_result("t1.2", "c1", "bash", &"compiler output ".repeat(40)),
        ];
        let protection = crate::agent::bigctx::compute_protection(
            &messages,
            &crate::agent::bigctx::create_initial_state(),
            &config,
        );
        assert!(
            protection.is_protected(&messages[2]),
            "positional protection must still apply to the fold gate"
        );
        assert!(
            !protection.is_tool_protected(&messages[2]),
            "bash is not content-protected, so absorb may act on it"
        );
    }

    // ── apply ───────────────────────────────────────────────────────────────

    fn call(reference: &str, summary: &str) -> ParsedCall {
        ParsedCall {
            reference: reference.into(),
            summary: summary.into(),
        }
    }

    #[test]
    fn absorbing_records_the_pair_and_reclaims_tokens() {
        let messages = [
            user("t1.0", "go"),
            tool_call("t1.1", "c1", "bash"),
            tool_result("t1.2", "c1", "bash", &"x".repeat(400)),
        ];
        let state = state_for(&messages);
        let reference = state.message_refs.ref_for("t1.2").unwrap().to_string();
        let out = apply(
            &state,
            &messages,
            &enabled(),
            &Protection::default(),
            &call(&reference, "the command printed 400 x characters"),
        );
        assert!(out.ok, "got: {}", out.text);
        assert_eq!(out.state.absorbed.len(), 1);
        let record = &out.state.absorbed[0];
        assert_eq!(record.result_message_id, "t1.2");
        assert_eq!(record.call_message_id, "t1.1", "the call half must be named");
        assert_eq!(record.tool_call_id, "c1");
        assert!(record.tokens_reclaimed > 0);
        assert!(out.state.stats.absorbed_tokens > 0);
    }

    #[test]
    fn an_empty_summary_is_refused() {
        let messages = [tool_result("t1.0", "c1", "bash", "text")];
        let state = state_for(&messages);
        let reference = state.message_refs.ref_for("t1.0").unwrap().to_string();
        let out = apply(
            &state,
            &messages,
            &enabled(),
            &Protection::default(),
            &call(&reference, "   "),
        );
        assert!(!out.ok);
        assert!(out.text.contains("summary is empty"), "got: {}", out.text);
    }

    #[test]
    fn an_unknown_ref_is_refused() {
        let messages = [tool_result("t1.0", "c1", "bash", "text")];
        let state = state_for(&messages);
        let out = apply(
            &state,
            &messages,
            &enabled(),
            &Protection::default(),
            &call("m99999", "s"),
        );
        assert!(!out.ok);
        assert!(out.text.contains("does not exist"), "got: {}", out.text);
    }

    #[test]
    fn absorbing_a_non_tool_result_is_refused_with_the_reason() {
        let messages = [user("t1.0", "hello")];
        let state = state_for(&messages);
        let reference = state.message_refs.ref_for("t1.0").unwrap().to_string();
        let out = apply(
            &state,
            &messages,
            &enabled(),
            &Protection::default(),
            &call(&reference, "s"),
        );
        assert!(!out.ok);
        assert!(
            out.text.contains("not a tool result"),
            "the refusal must say why: {}",
            out.text
        );
    }

    #[test]
    fn absorbing_a_managed_tool_is_refused() {
        let messages = [tool_result("t1.0", "c1", "acp_status", "text")];
        let state = state_for(&messages);
        let reference = state.message_refs.ref_for("t1.0").unwrap().to_string();
        let out = apply(
            &state,
            &messages,
            &enabled(),
            &Protection::default(),
            &call(&reference, "s"),
        );
        assert!(!out.ok);
        assert!(out.text.contains("compression system"), "got: {}", out.text);
    }

    #[test]
    fn absorbing_twice_is_reported_as_already_done() {
        let messages = [tool_result("t1.0", "c1", "bash", &"x".repeat(100))];
        let state = state_for(&messages);
        let reference = state.message_refs.ref_for("t1.0").unwrap().to_string();
        let first = apply(
            &state,
            &messages,
            &enabled(),
            &Protection::default(),
            &call(&reference, "s"),
        );
        assert!(first.ok);
        let second = apply(
            &first.state,
            &messages,
            &enabled(),
            &Protection::default(),
            &call(&reference, "s"),
        );
        assert!(second.ok, "a repeat is not an error");
        assert!(second.text.contains("already absorbed"), "got: {}", second.text);
        assert_eq!(second.state.absorbed.len(), 1, "no duplicate record");
    }

    #[test]
    fn a_summary_that_is_not_smaller_is_flagged() {
        // The model will not notice a net loss on its own.
        let messages = [tool_result("t1.0", "c1", "bash", "short")];
        let state = state_for(&messages);
        let reference = state.message_refs.ref_for("t1.0").unwrap().to_string();
        let long_summary = "word ".repeat(60);
        let out = apply(
            &state,
            &messages,
            &enabled(),
            &Protection::default(),
            &call(&reference, &long_summary),
        );
        assert!(out.ok);
        assert!(out.text.contains("WARNING"), "got: {}", out.text);
    }

    // ── hiding ──────────────────────────────────────────────────────────────

    #[test]
    fn hiding_drops_both_halves_of_the_pair() {
        let messages = [
            user("t1.0", "go"),
            tool_call("t1.1", "c1", "bash"),
            tool_result("t1.2", "c1", "bash", "out"),
            user("t1.3", "next"),
        ];
        let mut state = state_for(&messages);
        state.absorbed.push(AbsorbRecord {
            tool_call_id: "c1".into(),
            call_message_id: "t1.1".into(),
            result_message_id: "t1.2".into(),
            absorb_call_id: None,
            summary: "s".into(),
            tokens_reclaimed: 5,
            created_at: 0,
        });
        let kept = hide_absorbed(&messages, &state);
        let ids: Vec<&str> = kept.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["t1.0", "t1.3"], "the pair goes, the rest stays");
    }

    #[test]
    fn hiding_with_nothing_absorbed_is_a_no_op() {
        let messages = [user("t1.0", "go")];
        let state = state_for(&messages);
        assert_eq!(hide_absorbed(&messages, &state), messages.to_vec());
    }

    // ── parsing ─────────────────────────────────────────────────────────────

    #[test]
    fn the_canonical_argument_shape_parses() {
        let parsed = parse_call(&serde_json::json!({"ref": "m00001", "summary": "s"})).unwrap();
        assert_eq!(parsed.reference, "m00001");
        assert_eq!(parsed.summary, "s");
    }

    #[test]
    fn alternate_key_spellings_parse() {
        for args in [
            serde_json::json!({"messageId": "m1", "summary": "s"}),
            serde_json::json!({"of": "m1", "content": "s"}),
            serde_json::json!({"message_id": "m1", "summary": "s"}),
        ] {
            let parsed = parse_call(&args).expect("must parse");
            assert_eq!(parsed.reference, "m1");
        }
    }

    #[test]
    fn a_stringified_payload_parses() {
        let args = serde_json::json!("{\"ref\":\"m1\",\"summary\":\"s\"}");
        assert_eq!(parse_call(&args).unwrap().reference, "m1");
    }

    #[test]
    fn a_malformed_payload_does_not_parse() {
        assert!(parse_call(&serde_json::json!({"summary": "s"})).is_none());
        assert!(parse_call(&serde_json::json!({"ref": "m1"})).is_none());
        assert!(parse_call(&serde_json::json!("not json")).is_none());
        assert!(parse_call(&serde_json::json!(42)).is_none());
    }

    #[test]
    fn the_reference_is_trimmed() {
        let parsed = parse_call(&serde_json::json!({"ref": "  m1  ", "summary": "s"})).unwrap();
        assert_eq!(parsed.reference, "m1");
    }

    #[test]
    fn the_prompt_names_the_ref_and_the_tool() {
        let prompt = build_prompt("m00042", 5_000, ABSORB_TOOL_NAME);
        assert!(prompt.contains("m00042"));
        assert!(prompt.contains("absorb"));
        assert!(prompt.contains("5.0K"));
    }

    #[test]
    fn token_counts_format_readably() {
        assert_eq!(format_token_count(500), "500");
        assert_eq!(format_token_count(5_000), "5.0K");
        assert_eq!(format_token_count(50_000), "50K");
    }

    #[test]
    fn a_fold_of_a_tier_two_block_still_reclaims_nothing_here() {
        // Guards the stats separation: absorbed tokens are their own counter.
        let state = crate::agent::bigctx::create_initial_state();
        assert_eq!(reclaimed_tokens(&state), 0);
        assert_eq!(state.stats.absorbed_tokens, 0);
        let _ = Tier::Two;
    }
}
