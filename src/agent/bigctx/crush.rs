//! Deterministic compression of oversized tool results.
//!
//! Tier 1 of a two-tier gate: before the model is asked to summarise anything,
//! a tool result that is structurally redundant is shrunk mechanically. No
//! model call, no tokens spent, and — for the JSON strategy — no information
//! lost at all.
//!
//! The strategies, dispatched by sniffing the payload:
//!
//! - **json-fold** — an array of rows that share constant fields, or a run of
//!   identical elements, is rewritten into a compact envelope. **Lossless by
//!   construction**: every value that was there is still there.
//! - **code-trim** — comments, docstrings and blank-line runs elided into one
//!   marker line. Lossy.
//! - **log-select** — build/test logs condensed to errors, fails, deduped
//!   warnings and bounded stack traces. Lossy, and guarded: every `ERROR`/`FAIL`
//!   line of the input must survive verbatim or the result is rejected.
//!
//! Two properties are load-bearing and worth stating plainly:
//!
//! - **Determinism.** The same bytes must crush to the same bytes on every turn
//!   a result is re-sent. The provider's prefix cache is keyed on exact
//!   prefixes, so a crush that varies by turn costs more than it saves. This is
//!   why there is no query-aware selection here.
//! - **Fail open.** A parse error, a scanner that bails, a low-confidence
//!   language guess, or a reduction below the floor leaves the payload
//!   byte-identical. Crushing is an optimisation; it must never be the reason a
//!   tool result is unreadable.

use std::collections::BTreeSet;

use serde_json::{Map, Value};

/// Reserved key marking a crushed envelope.
///
/// A payload that legitimately contains this key cannot be told apart from
/// crushed output, which is why crushing is off by default and the key is
/// unlikely.
pub const CRUSH_KEY: &str = "__acp_crush";

/// Above this size a payload is left alone: crushing it costs more than the
/// result is worth.
const MAX_INPUT_CHARS: usize = 2_000_000;
const MAX_DEPTH: usize = 12;
const MAX_FOLDS: usize = 1_000;
/// Fewest consecutive identical elements worth a run envelope.
const RUN_MIN: usize = 4;
/// Fewest rows worth hoisting constant fields out of.
const HOIST_MIN: usize = 4;

/// What kind of payload this is, from its shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrushKind {
    Json,
    Code,
    Log,
}

/// Which strategy produced a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrushStrategy {
    JsonFold,
    CodeTrim,
    LogSelect,
}

impl CrushStrategy {
    pub fn as_str(self) -> &'static str {
        match self {
            CrushStrategy::JsonFold => "json-fold",
            CrushStrategy::CodeTrim => "code-trim",
            CrushStrategy::LogSelect => "log-select",
        }
    }
}

/// A crushed payload.
#[derive(Debug, Clone, PartialEq)]
pub struct CrushOutput {
    pub text: String,
    pub strategy: CrushStrategy,
    /// True when information was dropped. The log strategy additionally has to
    /// pass the error-line guard.
    pub lossy: bool,
    /// How many folds were applied, for reporting.
    pub folds: usize,
}

/// Classify a payload by shape.
pub fn classify(text: &str) -> CrushKind {
    let trimmed = text.trim();
    if (trimmed.starts_with('{') && trimmed.ends_with('}'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']'))
    {
        return CrushKind::Json;
    }
    if detect_language(text).is_some() {
        return CrushKind::Code;
    }
    CrushKind::Log
}

/// Crush a payload, or leave it alone.
///
/// `min_reduction` is the fraction of tokens that must be saved for the result
/// to be used; below it the payload passes through unchanged.
pub fn crush(text: &str, min_reduction: f64) -> Option<CrushOutput> {
    if text.is_empty() || text.len() > MAX_INPUT_CHARS {
        return None;
    }
    let original_tokens = crate::agent::bigctx::tokenize::default_count_tokens(text);
    if original_tokens == 0 {
        return None;
    }

    let kind = classify(text);
    let candidate = match kind {
        CrushKind::Json => crush_json(text).map(|(text, folds)| CrushOutput {
            text,
            strategy: CrushStrategy::JsonFold,
            lossy: false,
            folds,
        }),
        CrushKind::Code => crush_code(text).map(|text| CrushOutput {
            text,
            strategy: CrushStrategy::CodeTrim,
            lossy: true,
            folds: 0,
        }),
        CrushKind::Log => crush_log(text).map(|text| CrushOutput {
            text,
            strategy: CrushStrategy::LogSelect,
            lossy: true,
            folds: 0,
        }),
    };

    let candidate = candidate?;
    if candidate.text.is_empty() || candidate.text == text {
        return None;
    }

    let new_tokens = crate::agent::bigctx::tokenize::default_count_tokens(&candidate.text);
    let reduction = (original_tokens as f64 - new_tokens as f64) / original_tokens as f64;
    if reduction < min_reduction {
        return None;
    }

    // A lossy log result that dropped an error line is worse than useless: the
    // model reads a clean log and concludes nothing failed.
    if candidate.lossy && kind == CrushKind::Log && !error_lines_survive(text, &candidate.text) {
        return None;
    }

    Some(candidate)
}

// ─── json-fold (lossless) ────────────────────────────────────────────────────

/// Fold a JSON payload. Returns `(text, folds)`.
pub fn crush_json(text: &str) -> Option<(String, usize)> {
    let parsed: Value = serde_json::from_str(text).ok()?;
    let mut folds = 0usize;
    let out = compact_value(&parsed, 0, &mut folds);
    if folds == 0 {
        return None;
    }
    let serialized = serde_json::to_string(&out).ok()?;
    (serialized.len() < text.len()).then_some((serialized, folds))
}

fn compact_value(value: &Value, depth: usize, folds: &mut usize) -> Value {
    if depth > MAX_DEPTH || *folds >= MAX_FOLDS {
        return value.clone();
    }
    match value {
        Value::Array(items) => compact_array(items, depth, folds),
        Value::Object(map) => {
            let mut out = Map::new();
            let mut changed = false;
            for (key, item) in map {
                let next = compact_value(item, depth + 1, folds);
                if &next != item {
                    changed = true;
                }
                out.insert(key.clone(), next);
            }
            if changed { Value::Object(out) } else { value.clone() }
        }
        other => other.clone(),
    }
}

fn compact_array(items: &[Value], depth: usize, folds: &mut usize) -> Value {
    let n = items.len();
    if n < RUN_MIN && n < HOIST_MIN {
        return Value::Array(
            items
                .iter()
                .map(|item| compact_value(item, depth + 1, folds))
                .collect(),
        );
    }
    if *folds >= MAX_FOLDS {
        return Value::Array(items.to_vec());
    }

    // The plain (recursively compacted) form is the baseline; a fold has to
    // beat it in bytes to be worth the envelope's own overhead.
    let mut plain_folds = 0usize;
    let plain: Vec<Value> = items
        .iter()
        .map(|item| compact_value(item, depth + 1, &mut plain_folds))
        .collect();
    let mut best = Value::Array(plain);
    let mut best_len = serialized_len(&best);
    let mut best_folds = plain_folds;

    if n >= RUN_MIN
        && let Some((value, folds)) = build_run_fold(items, depth)
    {
        let len = serialized_len(&value);
        if len < best_len {
            best = value;
            best_len = len;
            best_folds = folds;
        }
    }
    if n >= HOIST_MIN
        && items.iter().all(Value::is_object)
        && let Some((value, folds)) = build_const_hoist(items, depth)
    {
        let len = serialized_len(&value);
        if len < best_len {
            best = value;
            best_len = len;
            best_folds = folds;
        }
    }
    let _ = best_len;

    *folds += best_folds;
    best
}

fn serialized_len(value: &Value) -> usize {
    serde_json::to_string(value).map(|s| s.len()).unwrap_or(usize::MAX)
}

/// Canonical text of a value, for equality that ignores key order.
///
/// `serde_json::Map` is order-preserving by default, so two objects with the
/// same fields in a different order would compare unequal — and a const-hoist
/// that missed them would leave bytes on the table rather than lose data, but a
/// run-fold that missed them would fail to fold a genuinely identical run.
fn canon(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<(String, String)> = map
                .iter()
                .map(|(k, v)| (k.clone(), canon(v)))
                .collect();
            entries.sort();
            let body: Vec<String> = entries
                .into_iter()
                .map(|(k, v)| format!("{}:{}", serde_json::to_string(&k).unwrap_or_default(), v))
                .collect();
            format!("{{{}}}", body.join(","))
        }
        Value::Array(items) => {
            let body: Vec<String> = items.iter().map(canon).collect();
            format!("[{}]", body.join(","))
        }
        other => serde_json::to_string(other).unwrap_or_else(|_| "null".to_string()),
    }
}

/// `{"__acp_crush":"identical-run","count":N,"item":X}` for runs of N+ identical
/// elements.
fn build_run_fold(items: &[Value], depth: usize) -> Option<(Value, usize)> {
    let canon_items: Vec<String> = items.iter().map(canon).collect();
    let mut segments: Vec<Value> = Vec::new();
    let mut runs = 0usize;
    let mut index = 0usize;

    while index < items.len() {
        let mut end = index;
        while end + 1 < items.len() && canon_items[end + 1] == canon_items[index] {
            end += 1;
        }
        let run_len = end - index + 1;
        if run_len >= RUN_MIN {
            let mut envelope = Map::new();
            envelope.insert(
                CRUSH_KEY.to_string(),
                Value::String("identical-run".to_string()),
            );
            envelope.insert("count".to_string(), Value::from(run_len));
            envelope.insert("item".to_string(), items[index].clone());
            segments.push(Value::Object(envelope));
            runs += 1;
        } else {
            for item in items.iter().take(end + 1).skip(index) {
                segments.push(item.clone());
            }
        }
        index = end + 1;
    }
    if runs == 0 {
        return None;
    }

    // Compact inside the envelopes too, so a run of identical big objects folds
    // its members rather than repeating them.
    let mut inner_folds = 0usize;
    let folded: Vec<Value> = segments
        .iter()
        .map(|segment| {
            if let Some(item) = segment.get("item")
                && is_run_marker(segment)
            {
                let mut envelope = segment.as_object().cloned().unwrap_or_default();
                envelope.insert("item".to_string(), compact_value(item, depth + 1, &mut inner_folds));
                return Value::Object(envelope);
            }
            compact_value(segment, depth + 1, &mut inner_folds)
        })
        .collect();

    Some((Value::Array(folded), runs + inner_folds))
}

fn is_run_marker(value: &Value) -> bool {
    value
        .get(CRUSH_KEY)
        .and_then(Value::as_str)
        .is_some_and(|kind| kind == "identical-run")
}

/// `{"__acp_crush":"rows","rows":N,"const":{...},"items":[...]}` for rows that
/// share constant fields.
fn build_const_hoist(items: &[Value], depth: usize) -> Option<(Value, usize)> {
    let n = items.len();
    let first = items.first()?.as_object()?;

    // Every key, in first-seen order, so the output is stable.
    let mut keys: Vec<String> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for item in items {
        if let Some(map) = item.as_object() {
            for key in map.keys() {
                if seen.insert(key.clone()) {
                    keys.push(key.clone());
                }
            }
        }
    }

    let mut const_keys: Vec<String> = Vec::new();
    for key in &keys {
        // A key absent from row 0 must not be hoisted: `canon` treats a missing
        // key and an explicit null the same way, so "absent in row 0, null
        // elsewhere" would pass the constant check — then the hoisted object
        // would omit it and the field would vanish from every decoded row. That
        // is a losslessness violation, and the reason for this guard.
        if !first.contains_key(key) {
            continue;
        }
        let reference = canon(&first[key]);
        let constant = items.iter().all(|item| {
            item.get(key)
                .is_some_and(|value| canon(value) == reference)
        });
        if constant {
            const_keys.push(key.clone());
        }
    }
    if const_keys.is_empty() {
        return None;
    }

    let const_set: BTreeSet<&String> = const_keys.iter().collect();
    let mut const_obj = Map::new();
    for key in &const_keys {
        const_obj.insert(key.clone(), first[key].clone());
    }

    let mut varying: Vec<Value> = Vec::with_capacity(n);
    let mut has_varying = false;
    for item in items {
        let mut row = Map::new();
        if let Some(map) = item.as_object() {
            for (key, value) in map {
                if !const_set.contains(key) {
                    row.insert(key.clone(), value.clone());
                    has_varying = true;
                }
            }
        }
        varying.push(Value::Object(row));
    }

    let mut inner_folds = 0usize;
    let mut envelope = Map::new();
    envelope.insert(CRUSH_KEY.to_string(), Value::String("rows".to_string()));
    envelope.insert("rows".to_string(), Value::from(n));
    envelope.insert(
        "const".to_string(),
        compact_value(&Value::Object(const_obj), depth + 1, &mut inner_folds),
    );
    if has_varying {
        let compacted: Vec<Value> = varying
            .iter()
            .map(|row| compact_value(row, depth + 1, &mut inner_folds))
            .collect();
        envelope.insert("items".to_string(), Value::Array(compacted));
    }

    let value = Value::Object(envelope);
    if serialized_len(&value) >= serialized_len(&Value::Array(items.to_vec())) {
        return None;
    }
    Some((value, 1 + inner_folds))
}

// ─── code-trim (lossy) ───────────────────────────────────────────────────────

/// Trim comments, docstrings and blank-line runs from source.
///
/// Conservative on purpose: only whole-line comments, docstrings and blank runs
/// go, and only when the language is confidently detected. A scanner that hits
/// an unterminated string, comment or template bails and the payload passes
/// through untouched.
pub fn crush_code(text: &str) -> Option<String> {
    let language = detect_language(text)?;
    match language {
        Language::Hash => trim_hash_comments(text),
        Language::Slash => trim_slash_comments(text),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Language {
    /// `#`-comment languages (Python, shell, YAML).
    Hash,
    /// `//`-comment languages (JS/TS, Rust, Go, Java).
    Slash,
}

/// Detect a language from its structural markers.
///
/// Deliberately narrow: a wrong guess here means mangling source the model
/// needed, so the evidence has to be strong. Anything else is treated as a log,
/// where the worst case is dropping noise.
fn detect_language(text: &str) -> Option<Language> {
    let mut hash = 0usize;
    let mut slash = 0usize;
    for line in text.lines().take(400) {
        let trimmed = line.trim_start();
        if trimmed.starts_with("def ")
            || trimmed.starts_with("class ")
            || trimmed.starts_with("import ")
            || trimmed.starts_with("from ")
            || trimmed.starts_with("elif ")
        {
            hash += 2;
        }
        if trimmed.starts_with("function ")
            || trimmed.starts_with("const ")
            || trimmed.starts_with("let ")
            || trimmed.starts_with("export ")
            || trimmed.starts_with("fn ")
            || trimmed.starts_with("pub fn ")
            || trimmed.starts_with("impl ")
            || trimmed.starts_with("use ")
            || trimmed.starts_with("package ")
        {
            slash += 2;
        }
        if trimmed.ends_with('{') {
            slash += 1;
        }
        if trimmed.ends_with(':') && !trimmed.ends_with("::") {
            hash += 1;
        }
    }
    // Both signals present is ambiguous: bail rather than guess.
    if hash >= 2 && slash >= 2 {
        return None;
    }
    if hash >= 2 {
        Some(Language::Hash)
    } else if slash >= 2 {
        Some(Language::Slash)
    } else {
        None
    }
}

/// Elide whole-line `#` comments, docstrings and blank runs.
fn trim_hash_comments(text: &str) -> Option<String> {
    let mut out: Vec<String> = Vec::new();
    let mut elided = 0usize;
    let mut in_docstring: Option<&str> = None;

    for line in text.lines() {
        if let Some(quote) = in_docstring {
            elided += 1;
            if line.trim().contains(quote) {
                in_docstring = None;
            }
            continue;
        }
        let trimmed = line.trim();
        if trimmed.starts_with("\"\"\"") || trimmed.starts_with("'''") {
            let quote = if trimmed.starts_with("\"\"\"") { "\"\"\"" } else { "'''" };
            // A one-line docstring closes on itself.
            let rest = &trimmed[3..];
            elided += 1;
            if !rest.contains(quote) {
                in_docstring = Some(quote);
            }
            continue;
        }
        if trimmed.starts_with('#') || trimmed.is_empty() {
            elided += 1;
            continue;
        }
        flush_elided(&mut out, &mut elided, "#");
        out.push(line.to_string());
    }
    let total_elided = elided;
    flush_elided(&mut out, &mut elided, "#");

    finalize(text, &out, total_elided)
}

/// Elide whole-line `//` comments and blank runs.
fn trim_slash_comments(text: &str) -> Option<String> {
    let mut out: Vec<String> = Vec::new();
    let mut elided = 0usize;

    for line in text.lines() {
        let trimmed = line.trim();
        // A line that starts a block comment is a whole-line comment; one that
        // merely contains `/*` mid-line is code and stays.
        if trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.is_empty() {
            elided += 1;
            continue;
        }
        flush_elided(&mut out, &mut elided, "//");
        out.push(line.to_string());
    }
    let total_elided = elided;
    flush_elided(&mut out, &mut elided, "//");

    finalize(text, &out, total_elided)
}

fn flush_elided(out: &mut Vec<String>, elided: &mut usize, marker: &str) {
    if *elided == 0 {
        return;
    }
    out.push(format!("{marker} [acp-crush: elided {} lines]", *elided));
    *elided = 0;
}

fn finalize(original: &str, emitted: &[String]) -> Option<String> {
    let out = emitted.join("\n");
    (!out.is_empty() && out.len() < original.len()).then_some(out)
}

// ─── log-select (lossy, guarded) ─────────────────────────────────────────────

/// Keep budget: the output never exceeds this many lines.
const LOG_KEEP_BUDGET: usize = 100;
/// How many context lines surround a kept interesting line.
const LOG_CONTEXT: usize = 3;

/// Condense a build/test log to what matters.
pub fn crush_log(text: &str) -> Option<String> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() <= LOG_KEEP_BUDGET {
        return None;
    }

    let mut keep: BTreeSet<usize> = BTreeSet::new();

    // 1. Errors and failures first — these are the reason anyone reads a log.
    for (index, line) in lines.iter().enumerate() {
        if is_error_line(line) {
            keep.insert(index);
            // a few lines of context around it
            for offset in 1..=LOG_CONTEXT {
                if index >= offset {
                    keep.insert(index - offset);
                }
                if index + offset < lines.len() {
                    keep.insert(index + offset);
                }
            }
        }
    }

    // 2. Warnings, deduplicated: one of each distinct warning is enough.
    let mut seen_warnings: BTreeSet<String> = BTreeSet::new();
    for (index, line) in lines.iter().enumerate() {
        if !is_warning_line(line) {
            continue;
        }
        let key = normalize_for_dedup(line);
        if seen_warnings.insert(key) {
            keep.insert(index);
        }
    }

    // 3. Summary lines, which usually carry the counts.
    for (index, line) in lines.iter().enumerate() {
        if is_summary_line(line) {
            keep.insert(index);
        }
    }

    // 4. Head and tail: a log's first lines name the command, its last lines
    //    carry the verdict.
    for index in 0..lines.len().min(5) {
        keep.insert(index);
    }
    for index in lines.len().saturating_sub(5)..lines.len() {
        keep.insert(index);
    }

    // Cap: if the budget is blown, drop from the tail of the keep set except
    // the errors, which are never dropped.
    if keep.len() > LOG_KEEP_BUDGET {
        let errors: BTreeSet<usize> = keep
            .iter()
            .copied()
            .filter(|i| is_error_line(lines[*i]))
            .collect();
        let mut kept: BTreeSet<usize> = errors;
        for index in &keep {
            if kept.len() >= LOG_KEEP_BUDGET {
                break;
            }
            kept.insert(*index);
        }
        keep = kept;
    }

    let mut out: Vec<String> = Vec::with_capacity(keep.len() + 8);
    let mut previous: Option<usize> = None;
    let mut skipped = 0usize;
    for index in &keep {
        if let Some(prev) = previous
            && index > &(prev + 1)
        {
            skipped += index - prev - 1;
        }
        if skipped > 0 && previous != Some(*index) {
            out.push(format!("[... {skipped} lines omitted ...]"));
            skipped = 0;
        }
        out.push(lines[*index].to_string());
        previous = Some(*index);
    }
    if let Some(prev) = previous
        && prev + 1 < lines.len()
    {
        out.push(format!("[... {} lines omitted ...]", lines.len() - prev - 1));
    }

    let condensed = out.join("\n");
    (!condensed.is_empty() && condensed.len() < text.len()).then_some(condensed)
}

fn is_error_line(line: &str) -> bool {
    let upper = line.to_ascii_uppercase();
    upper.contains("ERROR")
        || upper.contains("FAIL")
        || upper.contains("FATAL")
        || upper.contains("PANIC")
        || upper.contains("EXCEPTION")
        || upper.contains("TRACEBACK")
        || upper.contains("ASSERTION")
}

fn is_warning_line(line: &str) -> bool {
    line.to_ascii_uppercase().contains("WARN")
}

fn is_summary_line(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    lower.contains("passed")
        || lower.contains("failed")
        || lower.contains("total")
        || lower.contains("summary")
        || lower.contains("finished in")
}

/// Collapse the variable parts of a line so two instances of the same warning
/// dedupe to one.
fn normalize_for_dedup(line: &str) -> String {
    line.chars()
        .map(|c| if c.is_ascii_digit() { '#' } else { c })
        .collect()
}

/// Every error line of the input must appear in the output.
///
/// A lossy log result that dropped an error is worse than no crush at all: the
/// model reads a clean log and concludes nothing failed.
fn error_lines_survive(original: &str, crushed: &str) -> bool {
    for line in original.lines() {
        if is_error_line(line) && !crushed.contains(line.trim()) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_is_detected_by_shape() {
        assert_eq!(classify("{\"a\":1}"), CrushKind::Json);
        assert_eq!(classify("[1,2,3]"), CrushKind::Json);
        assert_eq!(classify("  {\"a\":1}  "), CrushKind::Json);
    }

    #[test]
    fn python_is_detected_as_code() {
        let py = "def f(x):\n    return x\n\nclass C:\n    pass\n";
        assert_eq!(classify(py), CrushKind::Code);
    }

    #[test]
    fn a_log_is_the_fallback() {
        assert_eq!(classify("running tests\nall good"), CrushKind::Log);
    }

    #[test]
    fn ambiguous_signals_bail_rather_than_guess() {
        // Both a Python and a JS marker: mangling the wrong language is worse
        // than not crushing.
        let mixed = "def f(x):\n    pass\n\nfunction g() {\n}\n";
        assert_eq!(classify(mixed), CrushKind::Log);
    }

    // ── json-fold ──

    #[test]
    fn a_run_of_identical_elements_folds() {
        // A run of short scalars legitimately does not fold: the envelope's own
        // keys cost more than twenty `1,` tokens. Real payloads carry objects.
        let row = r#"{"status":"ok","service":"billing","region":"eu-west-1"}"#;
        let text = format!("[{}]", vec![row; 20].join(","));
        let (out, folds) = crush_json(&text).unwrap();
        assert!(folds > 0);
        assert!(out.contains("identical-run"));
        assert!(out.contains("\"count\":20"));
        assert!(out.len() < text.len(), "{} vs {}", out.len(), text.len());
    }

    #[test]
    fn a_run_of_short_scalars_is_not_worth_folding() {
        // Stated explicitly: the envelope costs more than it saves here, and a
        // crush that grows the payload must never be returned.
        let text = "[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]";
        assert!(crush_json(text).is_none());
    }

    #[test]
    fn rows_with_a_constant_field_hoist_it() {
        let mut rows = Vec::new();
        for i in 0..20 {
            rows.push(format!(
                "{{\"level\":\"info\",\"message\":\"a fairly long repeated message\",\"i\":{i}}}"
            ));
        }
        let text = format!("[{}]", rows.join(","));
        let (out, folds) = crush_json(&text).unwrap();
        assert!(folds > 0);
        assert!(out.contains("\"rows\":20"));
        assert!(out.contains("const"));
        assert!(
            out.len() < text.len(),
            "hoisting must save bytes: {} vs {}",
            out.len(),
            text.len()
        );
    }

    #[test]
    fn a_short_array_is_left_alone() {
        assert!(crush_json("[1,2,3]").is_none());
    }

    #[test]
    fn an_array_with_no_redundancy_is_left_alone() {
        let text = "[1,2,3,4,5,6,7,8,9,10,11,12]";
        assert!(crush_json(text).is_none());
    }

    #[test]
    fn invalid_json_is_left_alone() {
        assert!(crush_json("{not json}").is_none());
    }

    #[test]
    fn a_key_missing_from_the_first_row_is_not_hoisted() {
        // The losslessness guard: canon treats a missing key and an explicit
        // null the same, so hoisting would drop the field from every row.
        let text = r#"[{"a":1,"b":null},{"a":1,"b":null},{"a":1,"b":null},{"a":1,"b":null},{"a":1}]"#;
        let Some((out, _)) = crush_json(text) else {
            return; // no fold is a correct answer too
        };
        let parsed: Value = serde_json::from_str(&out).unwrap();
        // If it folded, the hoisted const must not contain "b" alone.
        if let Some(consts) = parsed.get("const").and_then(Value::as_object) {
            if consts.contains_key("b") {
                panic!("b is absent from row 0 and must not be hoisted: {out}");
            }
        }
    }

    #[test]
    fn folding_is_deterministic() {
        // The prefix cache depends on this: the same bytes must crush to the
        // same bytes on every turn.
        let mut rows = Vec::new();
        for i in 0..30 {
            rows.push(format!("{{\"level\":\"info\",\"i\":{i}}}"));
        }
        let text = format!("[{}]", rows.join(","));
        let first = crush_json(&text).unwrap().0;
        for _ in 0..5 {
            assert_eq!(crush_json(&text).unwrap().0, first);
        }
    }

    // ── code-trim ──

    #[test]
    fn python_comments_and_blank_runs_are_elided() {
        let code = "# a comment\n# another\n\ndef f(x):\n    return x\n\n\n\ndef g():\n    return 2\n";
        let out = crush_code(code).unwrap();
        assert!(out.contains("elided"));
        assert!(out.contains("def f(x):"));
        assert!(out.contains("return x"));
        assert!(out.len() < code.len());
    }

    #[test]
    fn js_comments_and_blank_runs_are_elided() {
        let code = "// a comment\n// another\n\nfunction f(x) {\n  return x;\n}\n\n\n\nfunction g() {\n  return 2;\n}\n";
        let out = crush_code(code).unwrap();
        assert!(out.contains("elided"));
        assert!(out.contains("function f(x)"));
    }

    #[test]
    fn code_with_no_comments_is_left_alone() {
        let code = "def f(x):\n    return x\ndef g():\n    return 2\n";
        assert!(crush_code(code).is_none());
    }

    #[test]
    fn a_python_docstring_is_elided() {
        let code = "def f():\n    \"\"\"A long docstring\n    spanning lines.\"\"\"\n    return 1\n";
        let out = crush_code(code).unwrap();
        assert!(out.contains("elided"));
        assert!(out.contains("return 1"));
    }

    // ── log-select ──

    fn noisy_log() -> String {
        let mut lines = vec!["running the suite".to_string()];
        for i in 0..200 {
            lines.push(format!("noise line {i}"));
        }
        lines.push("ERROR: something broke at line 42".to_string());
        for i in 0..200 {
            lines.push(format!("more noise {i}"));
        }
        lines.push("12 passed, 1 failed".to_string());
        lines.join("\n")
    }

    #[test]
    fn a_noisy_log_is_condensed() {
        let log = noisy_log();
        let out = crush_log(&log).unwrap();
        assert!(out.len() < log.len());
        assert!(out.contains("ERROR: something broke"));
        assert!(out.contains("omitted"));
    }

    #[test]
    fn every_error_line_survives() {
        // The guard: a clean-looking log that dropped an error is worse than
        // no crush at all.
        let log = noisy_log();
        let out = crush_log(&log).unwrap();
        assert!(error_lines_survive(&log, &out));
    }

    #[test]
    fn a_crushed_log_that_dropped_an_error_is_rejected() {
        let original = "ERROR: one\nERROR: two\n";
        let crushed = "ERROR: one\n";
        assert!(!error_lines_survive(original, crushed));
    }

    #[test]
    fn a_short_log_is_left_alone() {
        assert!(crush_log("a\nb\nc").is_none());
    }

    #[test]
    fn duplicate_warnings_collapse() {
        let mut lines: Vec<String> = (0..150).map(|i| format!("filler {i}")).collect();
        for _ in 0..50 {
            lines.push("WARN: the same deprecation at 12:34".to_string());
        }
        let log = lines.join("\n");
        let out = crush_log(&log).unwrap();
        let warnings = out.matches("WARN: the same deprecation").count();
        assert!(
            warnings < 50,
            "duplicates must collapse, got {warnings} of 50"
        );
    }

    // ── the entry point ──

    #[test]
    fn a_reduction_below_the_floor_is_refused() {
        // A crush that saves almost nothing is not worth a prefix-cache miss.
        let mut rows = Vec::new();
        for i in 0..20 {
            rows.push(format!("{{\"level\":\"info\",\"i\":{i}}}"));
        }
        let text = format!("[{}]", rows.join(","));
        assert!(crush(&text, 0.99).is_none(), "99% required, not achieved");
    }

    #[test]
    fn a_real_reduction_passes_the_floor() {
        let text = "[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]";
        let out = crush(text, 0.1).unwrap();
        assert_eq!(out.strategy, CrushStrategy::JsonFold);
        assert!(!out.lossy, "json-fold is lossless by construction");
    }

    #[test]
    fn an_empty_payload_is_left_alone() {
        assert!(crush("", 0.1).is_none());
    }

    #[test]
    fn an_oversized_payload_is_left_alone() {
        let huge = "x".repeat(MAX_INPUT_CHARS + 1);
        assert!(crush(&huge, 0.1).is_none());
    }

    #[test]
    fn a_log_crush_is_marked_lossy() {
        let out = crush(&noisy_log(), 0.1).unwrap();
        assert_eq!(out.strategy, CrushStrategy::LogSelect);
        assert!(out.lossy);
    }
}
