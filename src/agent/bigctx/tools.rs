//! The `compress` / `decompress` / `acp_retrieve` / `search_context` /
//! `acp_status` tools: the model's handle on its own context.
//! its own context.
//!
//! Everything here is a thin shell. The algorithm lives in the rest of
//! `bigctx`; these tools parse a call, hand it the state, and turn the result
//! back into text the model can act on. That split is why the tools can be
//! tested without a provider and the algorithm can be tested without a tool.
//!
//! The state is shared rather than owned: a tool runs on the agent's tool
//! thread, but the state belongs to the session. [`BigctxHandle`] is that
//! shared cell — one lock, taken for the length of one call.

use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use crate::agent::bigctx::compress::{CompressError, apply_compression_with_refs};
use crate::agent::bigctx::config::BigctxConfig;
use crate::agent::bigctx::protected::compute_protection;
use crate::agent::bigctx::types::{CompressionState, CoreMessage};
use crate::agent::tools::{Tool, ToolOutput, ToolProgress};
use crate::agent::approval::Tier;
use std::path::Path;

/// The shared compression state, plus the messages it applies to.
///
/// Both live behind one lock on purpose: a `compress` call must see the same
/// message array the protection rules were computed against, or it could fold
/// a message that the very next turn considers protected.
#[derive(Clone, Default)]
pub struct BigctxHandle {
    inner: Arc<Mutex<Option<BigctxSession>>>,
}

/// One session's worth of state.
#[derive(Clone)]
pub struct BigctxSession {
    pub state: CompressionState,
    /// The full resent history, projected to cores.
    pub messages: Vec<CoreMessage>,
    /// The same array after the pipeline folded it — what the model is
    /// actually sent. `acp_status` prices this, not `messages`: the full array
    /// still holds every folded message, so pricing it would report the size
    /// the fold was supposed to relieve, not the size that remains.
    pub view: Vec<CoreMessage>,
    pub config: BigctxConfig,
    /// Where a fold records itself for the cache accounting. None when the
    /// caller did not ask for a report.
    pub ledger: Option<crate::agent::bigctx::CacheLedger>,
    /// Originals kept out of the context, retrievable by ref. Shared with the
    /// pipeline: a placeholder the model sees must be retrievable through the
    /// same store, and the store must survive across rounds.
    pub store: Option<std::sync::Arc<std::sync::Mutex<crate::agent::bigctx::ContentStore>>>,
}

impl BigctxHandle {
    pub fn new() -> BigctxHandle {
        BigctxHandle::default()
    }

    /// Install this turn's state and view. Called by the session before a
    /// request goes out.
    pub fn set(&self, session: BigctxSession) {
        if let Ok(mut slot) = self.inner.lock() {
            *slot = Some(session);
        }
    }

    /// Clear it: a session that ended must not leave a tool holding state.
    pub fn clear(&self) {
        if let Ok(mut slot) = self.inner.lock() {
            *slot = None;
        }
    }

    /// Run `f` against the session, if there is one.
    pub fn with<R>(&self, f: impl FnOnce(&mut BigctxSession) -> R) -> Option<R> {
        let mut slot = self.inner.lock().ok()?;
        slot.as_mut().map(f)
    }
}

/// No session attached: the tool was called outside a compressing turn.
const NO_SESSION: &str =
    "No conversation is available to compress. This tool only works during an active turn.";

// ─── compress ────────────────────────────────────────────────────────────────

/// One range from a `compress` call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressRange {
    pub start: String,
    pub end: String,
    pub summary: String,
    pub topic: Option<String>,
}

/// Parse the `content` argument.
///
/// Accepts every shape the upstream tool accepts, because a pre-validating
/// host that rejects something the kernel would honour costs the model a round
/// trip for nothing:
///
/// - a JSON array of objects (`{startId, endId, summary, topic?}`),
/// - a JSON array of line-form strings (`"m00150–m00220 Topic\nsummary…"`),
/// - a single JSON string holding that array (some gateways stringify it),
/// - a plain string holding one or more line-form blocks.
///
/// `startRef`/`endRef` are accepted as alternates for `startId`/`endId`.
pub fn parse_compress_content(content: &Value) -> Result<Vec<CompressRange>, String> {
    match content {
        Value::Array(items) => {
            let mut out = Vec::new();
            for item in items {
                match item {
                    Value::String(text) => out.extend(parse_line_form(text)?),
                    Value::Object(_) => out.push(parse_range_object(item)?),
                    other => {
                        return Err(format!(
                            "each content entry must be an object or a string, got {}",
                            kind_of(other)
                        ));
                    }
                }
            }
            if out.is_empty() {
                return Err("content held no ranges".to_string());
            }
            Ok(out)
        }
        Value::String(text) => {
            // A gateway that stringifies the array: parse it back rather than
            // failing a call the model meant literally.
            let trimmed = text.trim_start();
            if trimmed.starts_with('[') {
                if let Ok(parsed) = serde_json::from_str::<Value>(trimmed) {
                    return parse_compress_content(&parsed);
                }
            }
            let ranges = parse_line_form(text)?;
            if ranges.is_empty() {
                return Err("content held no ranges".to_string());
            }
            Ok(ranges)
        }
        other => Err(format!(
            "content must be an array or a string, got {}",
            kind_of(other)
        )),
    }
}

fn kind_of(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

fn parse_range_object(item: &Value) -> Result<CompressRange, String> {
    let pick = |a: &str, b: &str| -> Option<String> {
        item.get(a)
            .or_else(|| item.get(b))
            .and_then(Value::as_str)
            .map(str::to_string)
    };
    let start = pick("startId", "startRef")
        .ok_or_else(|| "a content object needs startId (or startRef)".to_string())?;
    let end =
        pick("endId", "endRef").ok_or_else(|| "a content object needs endId (or endRef)".to_string())?;
    let summary = item
        .get("summary")
        .and_then(Value::as_str)
        .ok_or_else(|| "a content object needs a summary".to_string())?
        .to_string();
    let topic = item
        .get("topic")
        .and_then(Value::as_str)
        .map(str::to_string);
    Ok(CompressRange {
        start,
        end,
        summary,
        topic,
    })
}

/// Parse the line form: `m00150–m00220 optional topic` then the summary.
///
/// Blocks are separated by a line that *is* a range header. A summary may
/// contain blank lines, so the split is structural (the header shape) rather
/// than blank-line-based.
fn parse_line_form(text: &str) -> Result<Vec<CompressRange>, String> {
    let mut out: Vec<CompressRange> = Vec::new();
    let mut current: Option<(String, String, Option<String>, Vec<String>)> = None;

    for line in text.lines() {
        if let Some((start, end, topic)) = parse_range_header(line) {
            if let Some((s, e, t, body)) = current.take() {
                out.push(CompressRange {
                    start: s,
                    end: e,
                    summary: body.join("\n").trim().to_string(),
                    topic: t,
                });
            }
            current = Some((start, end, topic, Vec::new()));
        } else if let Some((_, _, _, body)) = current.as_mut() {
            body.push(line.to_string());
        }
    }
    if let Some((s, e, t, body)) = current.take() {
        out.push(CompressRange {
            start: s,
            end: e,
            summary: body.join("\n").trim().to_string(),
            topic: t,
        });
    }
    Ok(out)
}

/// Recognise a header line: `<ref><dash><ref>` then an optional topic.
///
/// Accepts every dash a model actually types — the upstream tool does, and a
/// wrong dash would otherwise be read as summary text and silently fold the
/// wrong range.
fn parse_range_header(line: &str) -> Option<(String, String, Option<String>)> {
    let trimmed = line.trim();
    let (left, right) = split_at_dash(trimmed)?;
    let start = left.trim();
    let rest = right.trim_start();
    // the end ref runs to the first space
    let end_len = rest
        .find(|c: char| c.is_whitespace())
        .unwrap_or(rest.len());
    let end = &rest[..end_len];
    if !looks_like_ref(start) || !looks_like_ref(end) {
        return None;
    }
    let topic = rest[end_len..].trim();
    Some((
        start.to_ascii_lowercase(),
        end.to_ascii_lowercase(),
        (!topic.is_empty()).then(|| topic.to_string()),
    ))
}

fn split_at_dash(text: &str) -> Option<(&str, &str)> {
    for dash in ["–", "—", "->", "~", "-"] {
        if let Some(at) = text.find(dash) {
            return Some((&text[..at], &text[at + dash.len()..]));
        }
    }
    None
}

fn looks_like_ref(text: &str) -> bool {
    let lowered = text.to_ascii_lowercase();
    let digits = match lowered.strip_prefix('m') {
        Some(digits) => digits,
        None => match lowered.strip_prefix('b') {
            Some(digits) => digits,
            None => return false,
        },
    };
    !digits.is_empty() && digits.len() <= 9 && digits.bytes().all(|b| b.is_ascii_digit())
}

/// The `compress` tool.
pub struct CompressTool {
    handle: BigctxHandle,
    /// A prompt pack's replacement description, when one is configured.
    /// `None` keeps the built-in text compiled into `description()`.
    description_override: Option<String>,
    /// A prompt pack's per-parameter description overrides, applied to the
    /// schema `parameters()` returns.
    param_overrides: std::collections::BTreeMap<String, String>,
}

impl CompressTool {
    pub fn new(handle: BigctxHandle) -> CompressTool {
        CompressTool {
            handle,
            description_override: None,
            param_overrides: Default::default(),
        }
    }

    /// As [`CompressTool::new`], with a pack-supplied description.
    pub fn with_description(handle: BigctxHandle, description: Option<String>) -> CompressTool {
        CompressTool {
            handle,
            description_override: description,
            param_overrides: Default::default(),
        }
    }
}

impl Tool for CompressTool {
    fn name(&self) -> &str {
        "compress"
    }
    fn tier(&self) -> Tier {
        Tier::Write
    }
    fn description(&self) -> &str {
        if let Some(text) = self.description_override.as_deref() {
            return text;
        }
        "Replace consumed conversation ranges with self-contained summaries you write, identified \
         by their refs. Batch every range into ONE call. Use when the content is genuinely \
         consumed — you have extracted what you need and will not need the originals again. The \
         originals are kept and can be restored with `decompress`."
    }
    fn parameters(&self) -> Value {
        let schema = json!({
            "type": "object",
            "properties": {
                "content": {
                    "description": "One or more ranges to compress. PREFERRED: ONE plain string holding all ranges — each block starts with a 'm00150-m00220 optional topic' header line followed by that block's summary. Also accepted: an array of objects {startId, endId, summary, topic?}.",
                    "anyOf": [
                        {"type": "string"},
                        {"type": "array", "items": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "object", "properties": {
                                    "startId": {"type": "string"},
                                    "endId": {"type": "string"},
                                    "startRef": {"type": "string"},
                                    "endRef": {"type": "string"},
                                    "summary": {"type": "string"},
                                    "topic": {"type": "string"}
                                }}
                            ]
                        }}
                    ]
                }
            },
            "required": ["content"]
        });
        crate::agent::bigctx::packs::apply_param_descriptions(
            &schema,
            &self.param_overrides,
        )
    }
    fn preview(&self, args: &Value) -> String {
        match parse_compress_content(&args["content"]) {
            Ok(ranges) => format!("{} range(s)", ranges.len()),
            Err(_) => "compress".to_string(),
        }
    }
    fn execute(&self, args: &Value, _cwd: &Path, _log: &mut dyn FnMut(ToolProgress)) -> ToolOutput {
        let ranges = match parse_compress_content(&args["content"]) {
            Ok(ranges) => ranges,
            Err(error) => return ToolOutput::err(format!("compress: {error}")),
        };

        let Some(report) = self.handle.with(|session| apply_ranges(session, &ranges)) else {
            return ToolOutput::err(NO_SESSION);
        };
        match report {
            Ok(text) => ToolOutput::ok(text),
            Err(text) => ToolOutput::err(text),
        }
    }
}

/// Apply every range in one call, reporting what happened to each.
///
/// One failing range must not discard the others: a batch that folds three of
/// four ranges is a real saving, and the model is told exactly which one was
/// refused and why.
fn apply_ranges(
    session: &mut BigctxSession,
    ranges: &[CompressRange],
) -> Result<String, String> {
    let protection = compute_protection(&session.messages, &session.state, &session.config);
    let mut folded: Vec<String> = Vec::new();
    let mut refused: Vec<String> = Vec::new();

    for range in ranges {
        let resolved = match crate::agent::bigctx::resolve_boundaries(
            &range.start,
            &range.end,
            &session.messages,
            &session.state,
        ) {
            Ok(resolved) => resolved,
            Err(error) => {
                refused.push(format!(
                    "{}..{}: {error}",
                    range.start, range.end
                ));
                continue;
            }
        };

        match apply_compression_with_refs(
            &mut session.state,
            &session.messages,
            &resolved,
            &range.summary,
            &protection,
            &session.config,
            None,
            Some((range.start.clone(), range.end.clone())),
            0,
        ) {
            Ok(outcome) => {
                // Record the topic the model supplied; the block was created
                // before we knew it.
                if let Some(topic) = &range.topic
                    && let Some(block) = session
                        .state
                        .blocks
                        .iter_mut()
                        .find(|b| b.block_id == outcome.block_id)
                {
                    block.topic = Some(topic.clone());
                }
                // Record the fold so the cache report can charge it for the
                // re-payment it forces. The divergence point is where the fold
                // starts in the post-fold view; without it the report cannot
                // tell a fold's cost from ordinary cache expiry.
                if let Some(ledger) = session.ledger.as_ref() {
                    let view_after = crate::agent::bigctx::tokenize::count_messages_tokens(
                        &session.messages,
                    );
                    ledger.record_fold(crate::agent::bigctx::FoldEvent {
                        tokens_compressed: outcome.compressed_tokens,
                        summary_tokens: crate::agent::bigctx::tokenize::default_count_tokens(
                            &range.summary,
                        ),
                        first_fold_start_tokens: Some(
                            view_after.saturating_sub(outcome.compressed_tokens),
                        ),
                        view_after: Some(view_after),
                    });
                }
                folded.push(format!(
                    "{}..{} → {} (tier {}, {} messages, {} tokens)",
                    range.start,
                    range.end,
                    outcome.block_id,
                    outcome.tier.number(),
                    outcome.message_count,
                    outcome.compressed_tokens
                ));
            }
            Err(error) => refused.push(format!("{}..{}: {}", range.start, range.end, describe(&error))),
        }
    }

    if folded.is_empty() && !refused.is_empty() {
        return Err(format!(
            "nothing was compressed.\n\n{}",
            refused.join("\n")
        ));
    }

    if !folded.is_empty() {
        // The model acted on the last nudge, so that nudge is spent: measure
        // the next one from the new size, not the size that prompted this.
        // Without this the session stays "already nudged" and goes quiet while
        // the context refills.
        let remaining = crate::agent::bigctx::screen(
            &session.messages,
            &session.state,
            &session.config,
        );
        crate::agent::bigctx::reset_after_compression(
            &mut session.state,
            remaining.compressible_tokens(),
        );
    }

    let mut out = String::new();
    if !folded.is_empty() {
        out.push_str("Compressed:\n");
        for line in &folded {
            out.push_str("  ");
            out.push_str(line);
            out.push('\n');
        }
    }
    if !refused.is_empty() {
        out.push_str("\nRefused:\n");
        for line in &refused {
            out.push_str("  ");
            out.push_str(line);
            out.push('\n');
        }
    }
    Ok(out.trim_end().to_string())
}

/// Model-facing wording for a refusal. The error types already carry advice;
/// this only adds the "what to do instead" framing where it is missing.
fn describe(error: &CompressError) -> String {
    error.to_string()
}

// ─── decompress ──────────────────────────────────────────────────────────────

/// The `decompress` tool: make a block's material visible again.
pub struct DecompressTool {
    handle: BigctxHandle,
    /// A prompt pack's replacement description, when one is configured.
    /// `None` keeps the built-in text compiled into `description()`.
    description_override: Option<String>,
    /// A prompt pack's per-parameter description overrides, applied to the
    /// schema `parameters()` returns.
    param_overrides: std::collections::BTreeMap<String, String>,
}

impl DecompressTool {
    pub fn new(handle: BigctxHandle) -> DecompressTool {
        DecompressTool {
            handle,
            description_override: None,
            param_overrides: Default::default(),
        }
    }

    /// As [`DecompressTool::new`], with a pack-supplied description.
    pub fn with_description(handle: BigctxHandle, description: Option<String>) -> DecompressTool {
        DecompressTool {
            handle,
            description_override: description,
            param_overrides: Default::default(),
        }
    }
}

impl Tool for DecompressTool {
    fn name(&self) -> &str {
        "decompress"
    }
    fn tier(&self) -> Tier {
        Tier::Write
    }
    fn description(&self) -> &str {
        if let Some(text) = self.description_override.as_deref() {
            return text;
        }
        "Restore a compressed block so its original messages become visible again. Use when a \
         summary turns out not to carry what you need. The block id comes from acp_status."
    }
    fn parameters(&self) -> Value {
        let schema = json!({
            "type": "object",
            "properties": {
                "blockId": {
                    "type": "string",
                    "description": "Block id from acp_status, e.g. b3"
                }
            },
            "required": ["blockId"]
        });
        crate::agent::bigctx::packs::apply_param_descriptions(
            &schema,
            &self.param_overrides,
        )
    }
    fn preview(&self, args: &Value) -> String {
        format!("restore {}", args["blockId"].as_str().unwrap_or("?"))
    }
    fn execute(&self, args: &Value, _cwd: &Path, _log: &mut dyn FnMut(ToolProgress)) -> ToolOutput {
        let raw = args["blockId"].as_str().unwrap_or("").trim();
        let Some(block_id) = parse_block_id(raw) else {
            return ToolOutput::err(format!(
                "decompress: \"{raw}\" is not a block id. Use the bN ids acp_status lists."
            ));
        };

        let Some(result) = self.handle.with(|session| {
            crate::agent::bigctx::decompress_block(&mut session.state, &block_id)
        }) else {
            return ToolOutput::err(NO_SESSION);
        };

        match result {
            Ok(block) => ToolOutput::ok(format!(
                "Restored {}. Its {} message(s) are visible again — the next request carries them \
                 in full, so expect the context to grow by roughly {} tokens.",
                block.block_id,
                block.effective_message_ids.len(),
                block.compressed_tokens
            )),
            Err(error) => ToolOutput::err(format!("decompress: {error}")),
        }
    }
}

/// Normalise a block id: `b3`, `B3`, `b03`, or a bare `3`.
pub fn parse_block_id(raw: &str) -> Option<String> {
    let lowered = raw.trim().to_ascii_lowercase();
    let digits = lowered.strip_prefix('b').unwrap_or(&lowered);
    if digits.is_empty() || digits.len() > 9 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let number: u64 = digits.parse().ok()?;
    (number > 0).then(|| format!("b{number}"))
}

// ─── acp_status ──────────────────────────────────────────────────────────────

/// The `acp_status` tool: what is folded, what is left, what may be folded.
pub struct StatusTool {
    handle: BigctxHandle,
    /// A prompt pack's replacement description, when one is configured.
    /// `None` keeps the built-in text compiled into `description()`.
    description_override: Option<String>,
    /// A prompt pack's per-parameter description overrides, applied to the
    /// schema `parameters()` returns.
    param_overrides: std::collections::BTreeMap<String, String>,
}

impl StatusTool {
    pub fn new(handle: BigctxHandle) -> StatusTool {
        StatusTool {
            handle,
            description_override: None,
            param_overrides: Default::default(),
        }
    }

    /// As [`StatusTool::new`], with a pack-supplied description.
    pub fn with_description(handle: BigctxHandle, description: Option<String>) -> StatusTool {
        StatusTool {
            handle,
            description_override: description,
            param_overrides: Default::default(),
        }
    }
}

impl Tool for StatusTool {
    fn name(&self) -> &str {
        "acp_status"
    }
    fn tier(&self) -> Tier {
        Tier::Read
    }
    fn description(&self) -> &str {
        if let Some(text) = self.description_override.as_deref() {
            return text;
        }
        "Report the context: which blocks are folded, how many tokens they saved, and which \
         ranges are still compressible. Call this before a compress call to get the current refs — \
         refs from earlier calls may be stale."
    }
    fn parameters(&self) -> Value {
        let schema = json!({"type": "object", "properties": {}});
        crate::agent::bigctx::packs::apply_param_descriptions(
            &schema,
            &self.param_overrides,
        )
    }
    fn preview(&self, _args: &Value) -> String {
        "context status".to_string()
    }
    fn execute(&self, _args: &Value, _cwd: &Path, _log: &mut dyn FnMut(ToolProgress)) -> ToolOutput {
        let Some(text) = self.handle.with(render_status) else {
            return ToolOutput::err(NO_SESSION);
        };
        ToolOutput::ok(text)
    }
}

fn render_status(session: &mut BigctxSession) -> String {
    // The report owns the rendering. Duplicating it here would give the model
    // and the user two descriptions of the same state that can disagree.
    let report = crate::agent::bigctx::build_report(
        &session.view,
        &session.messages,
        &session.state,
        &session.config,
    );
    let mut text = crate::agent::bigctx::render_report(&report);
    if !report.compressible.compressible.is_empty() {
        text.push_str("

Cite a range as startId/endId when you call compress.
");
    }

    // What CCR is holding out of the context. A placeholder the model sees is
    // only useful if the original is still there, so the count belongs in the
    // status the model reads.
    if let Some(store) = session.store.as_ref()
        && let Ok(store) = store.lock()
        && !store.is_empty()
    {
        text.push_str("

");
        text.push_str(&crate::agent::bigctx::ccr::render_store_summary(&store));
    }

    // The cache accounting, when the caller asked for it. This is the part that
    // says whether folding is actually cheaper — a fold that breaks the prompt
    // cache can cost more than the tokens it removed.
    if let Some(ledger) = session.ledger.as_ref()
        && !ledger.is_empty()
    {
        let cache = ledger.report();
        text.push_str("

cache:
");
        text.push_str(&crate::agent::bigctx::render_cache_report(&cache));
    }
    text
}

// ─── acp_cache ───────────────────────────────────────────────────────────────

/// The `acp_cache` tool: what compression has actually cost.
///
/// `acp_status` reports the context; this reports the *price*. A fold removes
/// tokens but rewrites the prompt head, so it can cost more in cache misses than
/// it saves — and nothing else in the toolset can tell the model that. Without
/// it the model optimises for the number it can see (context size) and cannot
/// see the one that decides whether compressing helped.
pub struct CacheTool {
    handle: BigctxHandle,
    description_override: Option<String>,
    param_overrides: std::collections::BTreeMap<String, String>,
}

impl CacheTool {
    pub fn new(handle: BigctxHandle) -> CacheTool {
        CacheTool {
            handle,
            description_override: None,
            param_overrides: Default::default(),
        }
    }

    pub fn with_description(handle: BigctxHandle, description: Option<String>) -> CacheTool {
        CacheTool {
            handle,
            description_override: description,
            param_overrides: Default::default(),
        }
    }
}

impl Tool for CacheTool {
    fn name(&self) -> &str {
        "acp_cache"
    }
    fn tier(&self) -> Tier {
        Tier::Read
    }
    fn description(&self) -> &str {
        if let Some(text) = self.description_override.as_deref() {
            return text;
        }
        "Prompt-cache reconciliation: how many input tokens were billed, how many hit the cache,          and whether the folds paid for themselves. Use when asked what compression costs, or to          decide whether folding more is worth it."
    }
    fn parameters(&self) -> Value {
        let schema = json!({
            "type": "object",
            "properties": {
                "detail": {
                    "type": "string",
                    "enum": ["summary", "full"],
                    "description": "summary (default) reports totals and verdicts; full adds every fold."
                }
            }
        });
        crate::agent::bigctx::packs::apply_param_descriptions(&schema, &self.param_overrides)
    }
    fn preview(&self, _args: &Value) -> String {
        "cache report".to_string()
    }
    fn execute(&self, args: &Value, _cwd: &Path, _log: &mut dyn FnMut(ToolProgress)) -> ToolOutput {
        let full = args
            .get("detail")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|d| d == "full");
        let detail = if full {
            crate::agent::bigctx::CacheDetail::Full
        } else {
            // The default stays compact: the totals and the verdict are what a
            // decision needs. Every fold listed is a wall of numbers the model
            // would read and then not act on.
            crate::agent::bigctx::CacheDetail::Summary
        };
        let text = self.handle.with(|session| {
            let Some(ledger) = session.ledger.as_ref() else {
                return "No cache ledger for this session — it is recorded only when the caller asked for cache accounting."
                    .to_string();
            };
            if ledger.is_empty() {
                return "No requests recorded yet. The ledger fills as turns complete."
                    .to_string();
            }
            crate::agent::bigctx::render_cache_report_at(&ledger.report(), detail)
        });
        match text {
            Some(text) => ToolOutput::ok(text),
            None => ToolOutput::err(NO_SESSION),
        }
    }
}

/// Every bigctx tool, bound to one handle, with the built-in descriptions.
pub fn bigctx_tools(handle: BigctxHandle) -> Vec<Box<dyn Tool>> {
    bigctx_tools_with(handle, &std::collections::BTreeMap::new())
}

/// As [`bigctx_tools`], applying a prompt pack's tool-description overrides.
///
/// Descriptions are baked into the tool objects rather than looked up per call:
/// the registry is built once per task, and a description that changed between
/// two requests would move the head of the prompt and cost a cache miss.
pub fn bigctx_tools_with(
    handle: BigctxHandle,
    overrides: &std::collections::BTreeMap<String, crate::agent::bigctx::packs::ToolPromptOverride>,
) -> Vec<Box<dyn Tool>> {
    let describe = |name: &str| -> Option<String> {
        overrides
            .get(name)
            .and_then(|over| over.description.clone())
    };
    let params = |name: &str| -> std::collections::BTreeMap<String, String> {
        overrides
            .get(name)
            .map(|over| over.param_descriptions.clone())
            .unwrap_or_default()
    };
    let mut compress = CompressTool::with_description(handle.clone(), describe("compress"));
    compress.param_overrides = params("compress");
    let mut decompress = DecompressTool::with_description(handle.clone(), describe("decompress"));
    decompress.param_overrides = params("decompress");
    let mut retrieve = RetrieveTool::with_description(handle.clone(), describe("acp_retrieve"));
    retrieve.param_overrides = params("acp_retrieve");
    let mut search = SearchTool::with_description(handle.clone(), describe("search_context"));
    search.param_overrides = params("search_context");
    let mut absorb = AbsorbTool::with_description(handle.clone(), describe("absorb"));
    absorb.param_overrides = params("absorb");
    let mut status = StatusTool::with_description(handle.clone(), describe("acp_status"));
    status.param_overrides = params("acp_status");
    let mut cache = CacheTool::with_description(handle, describe("acp_cache"));
    cache.param_overrides = params("acp_cache");

    vec![
        Box::new(compress),
        Box::new(decompress),
        Box::new(retrieve),
        Box::new(search),
        Box::new(absorb),
        Box::new(status),
        Box::new(cache),
    ]
}

// ─── absorb ─────────────────────────────────────────────────────────────────

/// The `absorb` tool: distil a large tool result now, so its original stops
/// being re-sent.
///
/// The nudge waits until the context is 45% full; by then a 30k-token result
/// has already been resent several times. This acts on the turn the result
/// appears — the forced prompt appended to the result names the ref, and the
/// model calls this with its distillation. The pair then leaves the context on
/// the next turn and the model's own summary becomes the durable record.
pub struct AbsorbTool {
    handle: BigctxHandle,
    /// A prompt pack's replacement description, when one is configured.
    /// `None` keeps the built-in text compiled into `description()`.
    description_override: Option<String>,
    /// A prompt pack's per-parameter description overrides, applied to the
    /// schema `parameters()` returns.
    param_overrides: std::collections::BTreeMap<String, String>,
}

impl AbsorbTool {
    pub fn new(handle: BigctxHandle) -> AbsorbTool {
        AbsorbTool {
            handle,
            description_override: None,
            param_overrides: Default::default(),
        }
    }

    /// As [`AbsorbTool::new`], with a pack-supplied description.
    pub fn with_description(handle: BigctxHandle, description: Option<String>) -> AbsorbTool {
        AbsorbTool {
            handle,
            description_override: description,
            param_overrides: Default::default(),
        }
    }
}

impl Tool for AbsorbTool {
    fn name(&self) -> &str {
        crate::agent::bigctx::absorb::ABSORB_TOOL_NAME
    }
    fn tier(&self) -> Tier {
        Tier::Read
    }
    fn description(&self) -> &str {
        if let Some(text) = self.description_override.as_deref() {
            return text;
        }
        "Distil a large tool result into its essentials. The original output is then removed \
         from the context and your summary becomes the only durable record of it. Call this \
         immediately when a tool result ends with an [ACP absorb] instruction."
    }
    fn parameters(&self) -> Value {
        let schema = json!({
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "The message ref the [ACP absorb] instruction named."
                },
                "summary": {
                    "type": "string",
                    "description": "Distilled essentials: outcome, key values, exact paths:lines, error text verbatim, decisions."
                }
            },
            "required": ["ref", "summary"]
        });
        crate::agent::bigctx::packs::apply_param_descriptions(
            &schema,
            &self.param_overrides,
        )
    }
    fn preview(&self, args: &Value) -> String {
        format!("absorb {}", args["ref"].as_str().unwrap_or(""))
    }
    fn execute(&self, args: &Value, _cwd: &Path, _log: &mut dyn FnMut(ToolProgress)) -> ToolOutput {
        self.run(args, None)
    }

    /// Absorb overrides `execute_call` because the absorption record stores the
    /// id of the call that produced the distillate, and only the full call
    /// carries it.
    fn execute_call(
        &self,
        call: &crate::providers::ToolCall,
        _cwd: &Path,
        _log: &mut dyn FnMut(ToolProgress),
    ) -> ToolOutput {
        self.run(&call.arguments, Some(call.id.clone()))
    }
}

impl AbsorbTool {
    fn run(&self, args: &Value, call_id: Option<String>) -> ToolOutput {
        let Some(mut parsed) = crate::agent::bigctx::absorb::parse_call(args) else {
            return ToolOutput::err(
                "absorb needs both `ref` (the message ref) and `summary` (the distilled text).",
            );
        };
        parsed.call_id = call_id;

        let result = self.handle.with(|session| {
            // The protection set is recomputed here rather than carried on the
            // handle: it is cheap, derived from the same rules, and a stale
            // copy would let this tool absorb a result protection forbids.
            let protection =
                crate::agent::bigctx::compute_protection(&session.messages, &session.state, &session.config);
            let outcome = crate::agent::bigctx::absorb::apply(
                &session.state,
                &session.messages,
                &session.config,
                &protection,
                &parsed,
            );
            if outcome.ok {
                session.state = outcome.state;
            }
            (outcome.ok, outcome.text)
        });

        match result {
            Some((true, text)) => ToolOutput::ok(text),
            Some((false, text)) => ToolOutput::err(text),
            None => ToolOutput::err(NO_SESSION),
        }
    }
}

// ─── search_context ──────────────────────────────────────────────────────────

/// The `search_context` tool: find folded material by content.
///
/// Without this the model can only reach folded content through a ref it
/// already remembers — `decompress` takes a block id, `acp_retrieve` takes a
/// message ref, and over a long session the model has neither. Search scores
/// every block summary and every folded message against a query and hands back
/// the refs to open, which is what makes a folded history navigable instead of
/// merely small.
///
/// Read-only and cheap: scoring runs over cached per-doc features, so a repeat
/// search costs one pass over the query rather than the corpus.
pub struct SearchTool {
    handle: BigctxHandle,
    /// A prompt pack's replacement description, when one is configured.
    /// `None` keeps the built-in text compiled into `description()`.
    description_override: Option<String>,
    /// A prompt pack's per-parameter description overrides, applied to the
    /// schema `parameters()` returns.
    param_overrides: std::collections::BTreeMap<String, String>,
}

impl SearchTool {
    pub fn new(handle: BigctxHandle) -> SearchTool {
        SearchTool {
            handle,
            description_override: None,
            param_overrides: Default::default(),
        }
    }

    /// As [`SearchTool::new`], with a pack-supplied description.
    pub fn with_description(handle: BigctxHandle, description: Option<String>) -> SearchTool {
        SearchTool {
            handle,
            description_override: description,
            param_overrides: Default::default(),
        }
    }
}

impl Tool for SearchTool {
    fn name(&self) -> &str {
        crate::agent::bigctx::search::tool_name()
    }
    fn tier(&self) -> Tier {
        Tier::Read
    }
    fn description(&self) -> &str {
        if let Some(text) = self.description_override.as_deref() {
            return text;
        }
        "Search the conversation by content, including material that has been folded into \
         summaries. Use this to find which block or message holds a detail before calling \
         decompress (blockId) or acp_retrieve (ref). Searching is cheap and does not restore \
         anything — it returns ranked refs with previews."
    }
    fn parameters(&self) -> Value {
        let schema = json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Words to look for. Mixed Chinese/English is fine."
                },
                "limit": {
                    "type": "number",
                    "description": "Max results (default 10)."
                },
                "algorithm": {
                    "type": "string",
                    "description": "Ranking algorithm: hybrid (default), bm25, fuzzy, substring."
                }
            },
            "required": ["query"]
        });
        crate::agent::bigctx::packs::apply_param_descriptions(
            &schema,
            &self.param_overrides,
        )
    }
    fn preview(&self, args: &Value) -> String {
        format!(
            "search {:?}",
            args["query"].as_str().unwrap_or("")
        )
    }
    fn execute(&self, args: &Value, _cwd: &Path, _log: &mut dyn FnMut(ToolProgress)) -> ToolOutput {
        let query = args["query"].as_str().unwrap_or("").trim().to_string();
        if query.is_empty() {
            return ToolOutput::err("search_context needs a non-empty `query`.");
        }
        let mut options = crate::agent::bigctx::search::SearchOptions::default();
        if let Some(n) = args["limit"].as_u64() {
            options.limit = (n as usize).clamp(1, 50);
        }
        if let Some(a) = args["algorithm"].as_str().filter(|a| !a.trim().is_empty()) {
            options.algorithm = Some(a.to_string());
        }
        let Some(text) = self.handle.with(|session| search_context(session, &query, &options)) else {
            return ToolOutput::err(NO_SESSION);
        };
        ToolOutput::ok(text)
    }
}

fn search_context(
    session: &mut BigctxSession,
    query: &str,
    options: &crate::agent::bigctx::search::SearchOptions,
) -> String {
    use crate::agent::bigctx::search::{block_docs, message_docs, search, SearchDocKind};

    let mut docs = block_docs(&session.state);
    docs.extend(message_docs(&session.messages, &session.state));

    if docs.is_empty() {
        return "Nothing is searchable yet — no blocks have been created and no message refs \
                have been assigned."
            .to_string();
    }

    let hits = search(&docs, query, options);
    if hits.is_empty() {
        return format!(
            "No match for {query:?} across {} searchable documents ({} blocks, {} messages). \
             Try different words, or a shorter phrase.",
            docs.len(),
            docs.iter().filter(|d| d.kind == SearchDocKind::Block).count(),
            docs.iter().filter(|d| d.kind == SearchDocKind::Message).count(),
        );
    }

    let mut out = String::new();
    out.push_str(&format!(
        "{} match(es) for {query:?} (searched {} documents):\n",
        hits.len(),
        docs.len()
    ));
    for hit in &hits {
        out.push_str(&format!(
            "\n  {} {}  score {:.2}  ~{} tok  tier {}\n",
            hit.kind.as_str(),
            hit.ref_id,
            hit.score,
            hit.tokens,
            hit.tier
        ));
        if let (SearchDocKind::Message, Some(block)) = (hit.kind, hit.block_id.as_deref()) {
            out.push_str(&format!("     folded into {block} — decompress({block}) for the range\n"));
        }
        out.push_str(&format!("     {}\n", hit.preview.replace('\n', " ")));
    }
    out.push_str("\nTo open a hit: decompress blockId=bN (restores the range) or acp_retrieve ref=mNNNNN (one message).");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::bigctx::config::default_config;
    use crate::agent::bigctx::create_initial_state;
    use crate::agent::bigctx::types::{MessageContentType, MessageRole};

    fn user(id: &str, text: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::User, MessageContentType::Text).with_text(text)
    }

    fn session_with(messages: Vec<CoreMessage>) -> BigctxSession {
        let mut config = default_config(200_000);
        config.preserve_recent_messages = 0;
        config.preserve_recent_tokens = 0;
        config.compress.min_range_tokens = 0;
        // Refs must exist before a range can be cited: `resolve_boundaries`
        // looks a raw id up through the ref map, so an unassigned session has
        // no mNNNNN to resolve and every call would report "does not exist".
        let mut state = create_initial_state();
        let mut next = 1u64;
        crate::agent::bigctx::assign_refs(messages.as_slice(), &mut state.message_refs, &mut next, |_| false)
            .expect("refs");
        state.next_ref_index = next;
        BigctxSession {
            state,
            view: messages.clone(),
            messages,
            config,
            ledger: None,
            store: None,
        }
    }

    fn summary() -> String {
        "the assistant read the auth module and found the token check in verify.rs line 42; the \
         decision was to keep the 60s connect budget and the open question is whether the retry \
         belongs in the transport or above it"
            .to_string()
    }

    // ── parsing ──

    #[test]
    fn an_object_array_parses() {
        let content = json!([{
            "startId": "m00001",
            "endId": "m00002",
            "summary": "a summary",
            "topic": "auth"
        }]);
        let ranges = parse_compress_content(&content).unwrap();
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, "m00001");
        assert_eq!(ranges[0].end, "m00002");
        assert_eq!(ranges[0].topic.as_deref(), Some("auth"));
    }

    #[test]
    fn start_ref_and_end_ref_are_accepted() {
        let content = json!([{
            "startRef": "m00001",
            "endRef": "m00002",
            "summary": "s"
        }]);
        let ranges = parse_compress_content(&content).unwrap();
        assert_eq!(ranges[0].start, "m00001");
    }

    #[test]
    fn the_line_form_parses_one_block() {
        let content = json!("m00001–m00002 auth\nthe summary body");
        let ranges = parse_compress_content(&content).unwrap();
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, "m00001");
        assert_eq!(ranges[0].end, "m00002");
        assert_eq!(ranges[0].topic.as_deref(), Some("auth"));
        assert_eq!(ranges[0].summary, "the summary body");
    }

    #[test]
    fn the_line_form_parses_several_blocks() {
        let content = json!("m00001–m00002 first\nbody one\n\nm00005–m00007 second\nbody two");
        let ranges = parse_compress_content(&content).unwrap();
        assert_eq!(ranges.len(), 2, "the header shape is the split, not blank lines");
        assert_eq!(ranges[0].topic.as_deref(), Some("first"));
        assert_eq!(ranges[1].start, "m00005");
        assert!(ranges[1].summary.contains("body two"));
    }

    #[test]
    fn a_summary_may_contain_blank_lines() {
        let content = json!("m00001-m00002\npara one\n\npara two");
        let ranges = parse_compress_content(&content).unwrap();
        assert_eq!(ranges.len(), 1);
        assert!(ranges[0].summary.contains("para one"));
        assert!(ranges[0].summary.contains("para two"));
    }

    #[test]
    fn every_dash_a_model_types_is_accepted() {
        for dash in ["-", "–", "—", "->", "~"] {
            let content = json!(format!("m00001{dash}m00002\nthe body"));
            let ranges = parse_compress_content(&content).unwrap_or_else(|e| {
                panic!("dash {dash:?} was rejected: {e}");
            });
            assert_eq!(ranges[0].end, "m00002", "dash {dash:?}");
        }
    }

    #[test]
    fn a_stringified_array_is_parsed_back() {
        // Some gateways hand the model's array over as a string.
        let content = json!("[{\"startId\":\"m00001\",\"endId\":\"m00002\",\"summary\":\"s\"}]");
        let ranges = parse_compress_content(&content).unwrap();
        assert_eq!(ranges.len(), 1);
    }

    #[test]
    fn a_block_ref_header_parses() {
        let content = json!("b1-b2\nthe body");
        let ranges = parse_compress_content(&content).unwrap();
        assert_eq!(ranges[0].start, "b1");
        assert_eq!(ranges[0].end, "b2");
    }

    #[test]
    fn prose_is_not_mistaken_for_a_header() {
        // No header means no range. The parser reports that as an empty list;
        // the caller decides whether "nothing parsed" is an error, and the
        // tool does (a compress call with no range is meaningless).
        let content = json!("this is just a summary with no header");
        let error = parse_compress_content(&content).unwrap_err();
        assert!(error.contains("no ranges"), "got: {error}");
    }

    #[test]
    fn an_object_without_bounds_is_refused() {
        let content = json!([{"summary": "s"}]);
        let error = parse_compress_content(&content).unwrap_err();
        assert!(error.contains("startId"));
    }

    #[test]
    fn block_id_normalisation() {
        assert_eq!(parse_block_id("b3").as_deref(), Some("b3"));
        assert_eq!(parse_block_id("B3").as_deref(), Some("b3"));
        assert_eq!(parse_block_id("b03").as_deref(), Some("b3"));
        assert_eq!(parse_block_id("3").as_deref(), Some("b3"));
        assert_eq!(parse_block_id("b0"), None);
        assert_eq!(parse_block_id("bx"), None);
        assert_eq!(parse_block_id(""), None);
    }

    // ── tools ──

    fn run(tool: &dyn Tool, args: Value) -> ToolOutput {
        let mut log = |_: ToolProgress| {};
        tool.execute(&args, Path::new("."), &mut log)
    }

    #[test]
    fn compress_folds_a_range_and_reports_it() {
        let handle = BigctxHandle::new();
        let messages = vec![user("t1.0", "a"), user("t1.1", "b")];
        handle.set(session_with(messages));

        let tool = CompressTool::new(handle.clone());
        let out = run(
            &tool,
            json!({"content": [{"startId": "m00001", "endId": "m00002", "summary": summary()}]}),
        );
        assert!(out.error.is_none(), "unexpected failure: {}", out.content);
        assert!(out.content.contains("Compressed:"));
        assert!(out.content.contains("b1"));

        let blocks = handle.with(|s| s.state.blocks.len()).unwrap();
        assert_eq!(blocks, 1);
    }

    #[test]
    fn a_failing_range_does_not_discard_the_others() {
        let handle = BigctxHandle::new();
        let messages = vec![user("t1.0", "a"), user("t1.1", "b")];
        handle.set(session_with(messages));

        let tool = CompressTool::new(handle.clone());
        let out = run(
            &tool,
            json!({"content": [
                {"startId": "m00001", "endId": "m00001", "summary": summary()},
                {"startId": "m99999", "endId": "m99999", "summary": summary()}
            ]}),
        );
        assert!(out.content.contains("Compressed:"));
        assert!(out.content.contains("Refused:"));
        assert_eq!(handle.with(|s| s.state.blocks.len()).unwrap(), 1);
    }

    #[test]
    fn an_all_refused_batch_is_an_error() {
        let handle = BigctxHandle::new();
        handle.set(session_with(vec![user("t1.0", "a")]));
        let tool = CompressTool::new(handle.clone());
        let out = run(
            &tool,
            json!({"content": [{"startId": "m99999", "endId": "m99999", "summary": summary()}]}),
        );
        assert!(out.error.is_some());
        assert!(out.content.contains("nothing was compressed"));
    }

    #[test]
    fn compress_without_a_session_says_so() {
        let tool = CompressTool::new(BigctxHandle::new());
        let out = run(
            &tool,
            json!({"content": [{"startId": "m1", "endId": "m1", "summary": "s"}]}),
        );
        assert!(out.error.is_some());
        assert!(out.content.contains("No conversation"));
    }

    #[test]
    fn decompress_restores_a_block() {
        let handle = BigctxHandle::new();
        let messages = vec![user("t1.0", "a"), user("t1.1", "b")];
        handle.set(session_with(messages));

        run(
            &CompressTool::new(handle.clone()),
            json!({"content": [{"startId": "m00001", "endId": "m00002", "summary": summary()}]}),
        );
        let out = run(&DecompressTool::new(handle.clone()), json!({"blockId": "b1"}));
        assert!(out.error.is_none(), "{}", out.content);
        assert!(out.content.contains("Restored b1"));

        let active = handle
            .with(|s| s.state.blocks.iter().filter(|b| b.active).count())
            .unwrap();
        assert_eq!(active, 0);
    }

    #[test]
    fn decompressing_an_unknown_block_is_an_error() {
        let handle = BigctxHandle::new();
        handle.set(session_with(vec![user("t1.0", "a")]));
        let out = run(&DecompressTool::new(handle.clone()), json!({"blockId": "b9"}));
        assert!(out.error.is_some());
    }

    #[test]
    fn a_bad_block_id_is_refused_before_touching_state() {
        let handle = BigctxHandle::new();
        handle.set(session_with(vec![user("t1.0", "a")]));
        let out = run(&DecompressTool::new(handle.clone()), json!({"blockId": "nonsense"}));
        assert!(out.error.is_some());
        assert!(out.content.contains("not a block id"));
    }

    #[test]
    fn status_reports_the_blocks_and_the_foldable_ranges() {
        let handle = BigctxHandle::new();
        let messages = vec![
            user("t1.0", "the task"),
            user("t1.1", "more"),
            user("t1.2", "even more"),
        ];
        handle.set(session_with(messages));

        run(
            &CompressTool::new(handle.clone()),
            json!({"content": [{"startId": "m00002", "endId": "m00003", "summary": summary()}]}),
        );

        let out = run(&StatusTool::new(handle.clone()), json!({}));
        assert!(out.error.is_none(), "{}", out.content);
        assert!(out.content.contains("b1"));
        assert!(out.content.contains("saved"));
    }

    #[test]
    fn status_on_an_empty_session_says_nothing_is_compressed() {
        let handle = BigctxHandle::new();
        handle.set(session_with(vec![user("t1.0", "a")]));
        let out = run(&StatusTool::new(handle.clone()), json!({}));
        assert!(out.content.contains("compressed so far"));
    }

    #[test]
    fn the_registry_binds_every_tool_to_one_handle() {
        let handle = BigctxHandle::new();
        handle.set(session_with(vec![user("t1.0", "a")]));
        let tools = bigctx_tools(handle.clone());
        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert_eq!(
            names,
            vec![
                "compress",
                "decompress",
                "acp_retrieve",
                "search_context",
                "absorb",
                "acp_status",
                "acp_cache"
            ]
        );
    }

    #[test]
    fn clearing_the_handle_detaches_every_tool() {
        let handle = BigctxHandle::new();
        handle.set(session_with(vec![user("t1.0", "a")]));
        handle.clear();
        let out = run(&StatusTool::new(handle.clone()), json!({}));
        assert!(out.error.is_some());
    }

    #[test]
    fn search_context_refuses_an_empty_query() {
        let handle = BigctxHandle::new();
        handle.set(session_with(vec![user("t1.0", "a")]));
        let out = run(&SearchTool::new(handle), json!({"query": "  "}));
        assert!(out.error.is_some());
        assert!(out.content.contains("query"), "got: {}", out.content);
    }

    #[test]
    fn search_context_refuses_without_a_session() {
        let out = run(&SearchTool::new(BigctxHandle::new()), json!({"query": "x"}));
        assert!(out.error.is_some());
        assert!(out.content.contains("No conversation"));
    }

    #[test]
    fn search_context_finds_an_unfolded_message_by_content() {
        // `session_with` assigns refs, so a fresh session is already searchable.
        let handle = BigctxHandle::new();
        handle.set(session_with(vec![
            user("t1.0", "hello there"),
            user("t1.1", "the supervisor will handle it"),
        ]));
        let out = run(&SearchTool::new(handle), json!({"query": "supervisor"}));
        assert!(out.error.is_none(), "got: {}", out.content);
        assert!(
            out.content.contains("match"),
            "expected a hit: {}",
            out.content
        );
    }

    #[test]
    fn search_context_says_so_when_a_query_matches_nothing() {
        let handle = BigctxHandle::new();
        handle.set(session_with(vec![user("t1.0", "hello there")]));
        let out = run(
            &SearchTool::new(handle),
            json!({"query": "zzzznonexistentzzzz"}),
        );
        assert!(out.error.is_none(), "got: {}", out.content);
        assert!(
            out.content.contains("No match"),
            "got: {}",
            out.content
        );
    }

    #[test]
    fn search_context_finds_a_folded_message_and_names_its_block() {
        use crate::agent::bigctx::types::{BlockGeneration, CompressionBlock, Tier};

        let mut session = session_with(vec![
            user("t1.0", "we discussed the database migration plan"),
            user("t1.1", "unrelated chatter"),
        ]);
        session
            .state
            .message_refs
            .by_raw
            .insert("t1.0".into(), "m00001".into());
        session
            .state
            .message_refs
            .by_ref
            .insert("m00001".into(), "t1.0".into());
        session
            .state
            .message_refs
            .by_raw
            .insert("t1.1".into(), "m00002".into());
        session
            .state
            .message_refs
            .by_ref
            .insert("m00002".into(), "t1.1".into());
        session.state.blocks.push(CompressionBlock {
            block_id: "b3".into(),
            run_id: "r1".into(),
            tier: Tier::One,
            topic: Some("schema work".into()),
            start_ref: None,
            end_ref: None,
            summary: "migration plan agreed".into(),
            direct_message_ids: vec!["t1.0".into()],
            effective_message_ids: vec!["t1.0".into()],
            direct_block_ids: vec![],
            compressed_tokens: 50,
            created_at: 0,
            survived_count: 0,
            generation: BlockGeneration::Young,
            active: true,
            expanded: false,
            duration_ms: None,
            compress_call_id: None,
        });

        let handle = BigctxHandle::new();
        handle.set(session);
        let out = run(
            &SearchTool::new(handle),
            json!({"query": "migration", "limit": 5}),
        );
        assert!(out.error.is_none(), "got: {}", out.content);
        assert!(
            out.content.contains("m00001"),
            "the folded message must be findable: {}",
            out.content
        );
        assert!(
            out.content.contains("b3"),
            "a message hit must name its owning block: {}",
            out.content
        );
    }
}

// ─── acp_retrieve ────────────────────────────────────────────────────────────

/// The `acp_retrieve` tool: pull a stored original back by ref.
///
/// This is the other half of the placeholder. Without it a stored result is
/// simply gone from the model's point of view, and it would have to guess.
pub struct RetrieveTool {
    handle: BigctxHandle,
    /// A prompt pack's replacement description, when one is configured.
    /// `None` keeps the built-in text compiled into `description()`.
    description_override: Option<String>,
    /// A prompt pack's per-parameter description overrides, applied to the
    /// schema `parameters()` returns.
    param_overrides: std::collections::BTreeMap<String, String>,
}

impl RetrieveTool {
    pub fn new(handle: BigctxHandle) -> RetrieveTool {
        RetrieveTool {
            handle,
            description_override: None,
            param_overrides: Default::default(),
        }
    }

    /// As [`RetrieveTool::new`], with a pack-supplied description.
    pub fn with_description(handle: BigctxHandle, description: Option<String>) -> RetrieveTool {
        RetrieveTool {
            handle,
            description_override: description,
            param_overrides: Default::default(),
        }
    }
}

impl Tool for RetrieveTool {
    fn name(&self) -> &str {
        crate::agent::bigctx::ccr::RETRIEVE_TOOL_NAME
    }
    fn tier(&self) -> Tier {
        Tier::Read
    }
    fn description(&self) -> &str {
        if let Some(text) = self.description_override.as_deref() {
            return text;
        }
        "Retrieve the full text of a stored tool result. Placeholders like \
         `📦 [acp-stored #m00042 · file read · 40,120 tok]` mark results that were kept out of the \
         context to save space; call this with the ref to read one back. Large originals return \
         a pointer instead of the payload."
    }
    fn parameters(&self) -> Value {
        let schema = json!({
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "The ref from the placeholder, e.g. m00042"
                }
            },
            "required": ["ref"]
        });
        crate::agent::bigctx::packs::apply_param_descriptions(
            &schema,
            &self.param_overrides,
        )
    }
    fn preview(&self, args: &Value) -> String {
        format!("retrieve {}", args["ref"].as_str().unwrap_or("?"))
    }
    fn execute(&self, args: &Value, _cwd: &Path, _log: &mut dyn FnMut(ToolProgress)) -> ToolOutput {
        let raw = args["ref"].as_str().unwrap_or("").trim();
        let Some(reference) = normalize_ref(raw) else {
            return ToolOutput::err(format!(
                "retrieve: \"{raw}\" is not a ref. Use the mNNNNN from the placeholder."
            ));
        };

        let Some(result) = self.handle.with(|session| {
            let store = session.store.as_ref()?.lock().ok()?;
            crate::agent::bigctx::ccr::retrieve(
                &store,
                &reference,
                session.config.ccr.retrieve_inline_tokens,
            )
        }) else {
            return ToolOutput::err(NO_SESSION);
        };

        match result {
            Some(got) => ToolOutput::ok(got.text),
            None => ToolOutput::err(format!(
                "retrieve: nothing is stored under {reference}. It may never have been stored, \
                 or the ref is from another session — the placeholders in the conversation name \
                 the ones that exist."
            )),
        }
    }
}

/// Normalise a ref: `m42`, `M42`, `m00042` all become `m00042`.
fn normalize_ref(raw: &str) -> Option<String> {
    let lowered = raw.trim().to_ascii_lowercase();
    let digits = lowered.strip_prefix('m')?;
    if digits.is_empty() || digits.len() > 9 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let number: u64 = digits.parse().ok()?;
    (number > 0).then(|| format!("m{number:05}"))
}

#[cfg(test)]
mod retrieve_tests {
    use super::*;

    fn session_with_store() -> BigctxSession {
        let messages = vec![CoreMessage::new(
            "t1.0",
            crate::agent::bigctx::types::MessageRole::Tool,
            crate::agent::bigctx::types::MessageContentType::ToolResult,
        )
        .with_text("body")
        .with_tool("read", "c1")];
        let mut config = crate::agent::bigctx::default_config(200_000);
        config.preserve_recent_messages = 0;
        config.preserve_recent_tokens = 0;
        let mut state = crate::agent::bigctx::create_initial_state();
        let mut next = 1u64;
        crate::agent::bigctx::assign_refs(&messages, &mut state.message_refs, &mut next, |_| false)
            .unwrap();
        state.next_ref_index = next;
        BigctxSession {
            state,
            view: messages.clone(),
            messages,
            config,
            ledger: None,
            store: None,
        }
    }

    fn run(tool: &dyn Tool, args: Value) -> ToolOutput {
        let mut log = |_: ToolProgress| {};
        tool.execute(&args, Path::new("."), &mut log)
    }

    #[test]
    fn refs_are_normalised() {
        assert_eq!(normalize_ref("m42").as_deref(), Some("m00042"));
        assert_eq!(normalize_ref("M42").as_deref(), Some("m00042"));
        assert_eq!(normalize_ref("m00042").as_deref(), Some("m00042"));
        assert_eq!(normalize_ref("x42"), None);
        assert_eq!(normalize_ref("m"), None);
        assert_eq!(normalize_ref("m0"), None);
    }

    #[test]
    fn an_unknown_ref_says_so() {
        let handle = BigctxHandle::new();
        handle.set(session_with_store());
        let out = run(&RetrieveTool::new(handle), json!({"ref": "m00001"}));
        assert!(out.error.is_some());
        assert!(out.content.contains("nothing is stored"), "{}", out.content);
    }

    #[test]
    fn a_bad_ref_is_refused() {
        let handle = BigctxHandle::new();
        handle.set(session_with_store());
        let out = run(&RetrieveTool::new(handle), json!({"ref": "nonsense"}));
        assert!(out.error.is_some());
        assert!(out.content.contains("not a ref"));
    }

    #[test]
    fn a_stored_original_comes_back() {
        let handle = BigctxHandle::new();
        let mut session = session_with_store();
        let store = std::sync::Arc::new(std::sync::Mutex::new(
            crate::agent::bigctx::ContentStore::new(),
        ));
        store
            .lock()
            .unwrap()
            .store("m00001", "t1.0", "the full body", "file read", Some("read"), 100, "the", None);
        session.store = Some(store);
        handle.set(session);
        let out = run(&RetrieveTool::new(handle), json!({"ref": "m00001"}));
        assert!(out.error.is_none(), "{}", out.content);
        assert_eq!(out.content, "the full body");
    }

    #[test]
    fn retrieve_without_a_session_says_so() {
        let out = run(&RetrieveTool::new(BigctxHandle::new()), json!({"ref": "m00001"}));
        assert!(out.error.is_some());
        assert!(out.content.contains("No conversation"));
    }
}
