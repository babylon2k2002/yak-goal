//! Content-cached retrieval: keeping big payloads out of the context without
//! losing them.
//!
//! A 40K-token tool result is most of a small window. Folding it costs a model
//! call and loses the detail; truncating it destroys it. This is the third
//! option: store the original once, leave a short deterministic placeholder in
//! its place, and let the model pull the original back when it turns out to
//! matter.
//!
//! ```text
//! 📦 [acp-stored #m00042 · file read · 40,120 tok] `src/auth/verify.rs`
//!    → acp_retrieve("m00042") returns the full text
//! ```
//!
//! ## Why the placeholder is byte-frozen
//!
//! The placeholder is generated **once, at arrival**, and never regenerated.
//! That is the whole design: a placeholder that changed between turns would
//! move the prefix and invalidate the provider's cache for everything after it,
//! and a tool result is re-sent on every request. So the preview is normalised
//! and capped at arrival, the token count is fixed then, and there are no
//! timestamps or anything else that varies.
//!
//! That also means a host that re-sends the original raw must get the same
//! placeholder back, not the raw payload — [`project`] handles that case.
//!
//! ## Storage is append-only
//!
//! Content is addressed by hash, so two identical results are stored once. The
//! ref index is append-only and first-write-wins: a ref, once issued, always
//! resolves to the same original. That is what makes retrieval reliable for the
//! life of the session.

use std::collections::BTreeMap;

use crate::agent::bigctx::refs::BLOCKED_REF;
use crate::agent::bigctx::tokenize::{count_message_tokens, default_count_tokens};
use crate::agent::bigctx::types::{CompressionState, CoreMessage, MessageContentType};

/// The tool that pulls an original back.
pub const RETRIEVE_TOOL_NAME: &str = "acp_retrieve";

/// Marker in every placeholder. Recognition goes through the strict parser,
/// never a substring scan.
pub const STORED_PLACEHOLDER_MARKER: &str = "[acp-stored";

/// Reserved id prefix for retrieval injections: no ref is ever assigned to
/// them, and fold-space collection skips them.
pub const RETRIEVED_ID_PREFIX: &str = "acp_retrieved_";

/// A tool result at or above this many tokens is stored rather than kept.
pub const MIN_TOOL_TOKENS: u64 = 4_000;

/// Above this, a retrieval returns a pointer instead of inlining — otherwise
/// retrieving would re-inflate the conversation by as much as the store saved.
pub const RETRIEVE_INLINE_TOKENS: u64 = 4_000;

/// How much of the payload's opening survives into the placeholder.
pub const MAX_HEAD_CHARS: usize = 96;

/// What kind of output a tool produces, for the placeholder's label.
pub fn classify_kind(tool_name: Option<&str>) -> String {
    let Some(name) = tool_name else {
        return "tool result".to_string();
    };
    match name.to_ascii_lowercase().as_str() {
        "bash" | "shell" | "exec" | "execute_command" | "run" | "terminal" => "shell output",
        "read" | "read_file" | "cat" | "open" => "file read",
        "grep" | "rg" | "search" | "glob" | "find" => "search output",
        "webfetch" | "web_fetch" | "fetch" | "curl" => "web fetch",
        _ => "tool result",
    }
    .to_string()
}

/// Collapse whitespace and cap length.
///
/// Deterministic by construction: the preview is baked into the placeholder at
/// arrival and must be byte-stable forever after.
pub fn normalize_head(text: &str, max_chars: usize) -> String {
    let single: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if single.chars().count() <= max_chars {
        return single;
    }
    let head: String = single.chars().take(max_chars).collect();
    format!("{head}…")
}

/// Fields a call's arguments may name its subject with.
const COMMAND_FIELDS: [&str; 6] = ["command", "cmd", "script", "query", "path", "url"];

/// Pull the call's subject (command/path/query) out of the paired call args.
pub fn extract_command(call_text: Option<&str>, max_chars: usize) -> Option<String> {
    let parsed: serde_json::Value = serde_json::from_str(call_text?).ok()?;
    let object = parsed.as_object()?;
    for field in COMMAND_FIELDS {
        if let Some(value) = object.get(field).and_then(serde_json::Value::as_str) {
            let normalized: String = value.split_whitespace().collect::<Vec<_>>().join(" ");
            if normalized.is_empty() {
                continue;
            }
            if normalized.chars().count() <= max_chars {
                return Some(normalized);
            }
            let head: String = normalized.chars().take(max_chars).collect();
            return Some(format!("{head}…"));
        }
    }
    None
}

/// Group thousands: `40120` → `40,120`.
fn group_thousands(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, ch) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// Build the placeholder. Two lines, no timestamps, no randomness.
pub fn build_placeholder(reference: &str, kind: &str, tokens: u64, title: Option<&str>) -> String {
    let title_part = match title {
        Some(t) if !t.is_empty() => format!(" `{t}`"),
        _ => String::new(),
    };
    format!(
        "📦 {STORED_PLACEHOLDER_MARKER} #{reference} · {kind} · {} tok]{title_part}\n   \
         → {RETRIEVE_TOOL_NAME}(\"{reference}\") returns the full text",
        group_thousands(tokens)
    )
}

/// A parsed placeholder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedPlaceholder {
    pub reference: String,
    pub kind: String,
    pub tokens: u64,
    pub title: Option<String>,
}

/// Recognise a whole-body placeholder.
///
/// Strict on purpose: ordinary text that happens to mention `[acp-stored` must
/// not be mistaken for one, or a payload would be replaced by a stub of itself.
pub fn parse_placeholder(text: &str) -> Option<ParsedPlaceholder> {
    let text = text.trim();
    let rest = text.strip_prefix("📦 [acp-stored #")?;
    let (reference, rest) = rest.split_once(' ')?;
    if !reference.starts_with('m') || reference.len() < 6 {
        return None;
    }
    if !reference[1..].bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let rest = rest.strip_prefix("· ")?;
    let (kind, rest) = rest.split_once(" · ")?;
    let rest = rest.split_once(" tok]")?;
    let tokens_text = rest.0.replace(',', "");
    let tokens: u64 = tokens_text.parse().ok()?;
    // The tail must be the retrieval hint naming the same ref.
    let tail = rest.1;
    if !tail.contains(&format!("\"{reference}\"")) {
        return None;
    }
    let title = tail
        .trim_start()
        .strip_prefix('`')
        .and_then(|t| t.strip_suffix('`'))
        .map(str::to_string);

    Some(ParsedPlaceholder {
        reference: reference.to_string(),
        kind: kind.to_string(),
        tokens,
        title,
    })
}

/// One stored original.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredEntry {
    /// Content hash, the key into the store's blob table.
    pub hash: String,
    /// The message id at store time — a frozen alias, so retrieval keeps
    /// working even after the ref map is pruned.
    pub raw_id: String,
    pub kind: String,
    pub tool_name: Option<String>,
    pub tokens: u64,
    pub chars: usize,
    /// The preview baked into the placeholder.
    pub head: String,
    /// The call subject, frozen at arrival, so a raw retransmission can be
    /// re-projected to a byte-identical placeholder.
    pub command: Option<String>,
}

/// The store: content-addressed originals plus a per-ref index.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ContentStore {
    /// `sha256(text) → text`. Dedup lives here.
    by_hash: BTreeMap<String, String>,
    /// `ref → entry`. Append-only, first write wins.
    by_ref: BTreeMap<String, StoredEntry>,
}

impl ContentStore {
    pub fn new() -> ContentStore {
        ContentStore::default()
    }

    /// Store an original. First write for a ref wins; identical text is kept
    /// once.
    pub fn store(
        &mut self,
        reference: &str,
        raw_id: &str,
        text: &str,
        kind: &str,
        tool_name: Option<&str>,
        tokens: u64,
        head: &str,
        command: Option<&str>,
    ) {
        if reference.is_empty() || self.by_ref.contains_key(reference) {
            return;
        }
        let hash = hash_content(text);
        self.by_ref.insert(
            reference.to_string(),
            StoredEntry {
                hash: hash.clone(),
                raw_id: raw_id.to_string(),
                kind: kind.to_string(),
                tool_name: tool_name.map(str::to_string),
                tokens,
                chars: text.chars().count(),
                head: head.to_string(),
                command: command.map(str::to_string),
            },
        );
        self.by_hash.entry(hash).or_insert_with(|| text.to_string());
    }

    /// The original behind a ref, when it was stored.
    pub fn original(&self, reference: &str) -> Option<&str> {
        let entry = self.by_ref.get(reference)?;
        self.by_hash.get(&entry.hash).map(String::as_str)
    }

    /// The entry behind a ref.
    pub fn entry(&self, reference: &str) -> Option<&StoredEntry> {
        self.by_ref.get(reference)
    }

    pub fn len(&self) -> usize {
        self.by_ref.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_ref.is_empty()
    }

    /// Total bytes held, for reporting.
    pub fn held_chars(&self) -> usize {
        self.by_ref.values().map(|e| e.chars).sum()
    }
}

/// sha256 of the content, hex.
///
/// A stable content address is what lets two identical results share one copy
/// and what makes the ref index append-only: the hash never changes for a given
/// original.
pub fn hash_content(text: &str) -> String {
    use std::hash::{Hash, Hasher};
    // FNV-1a over the bytes: not cryptographic, but stable across runs and
    // platforms, which is all a dedup key needs. A crypto hash would pull in a
    // dependency for no benefit here — the key is never exposed.
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// What one projection pass did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CcrResult {
    pub messages: Vec<CoreMessage>,
    /// How many results were replaced by a placeholder this pass.
    pub stored: usize,
    /// How many already-stored results were re-projected (a host re-sent the
    /// raw payload).
    pub reprojected: usize,
}

/// Replace oversized tool results with placeholders, storing the originals.
///
/// Runs at arrival and again on every pass: an already-stored ref that shows up
/// raw again gets its arrival-time placeholder back, never the payload.
pub fn project(
    messages: &[CoreMessage],
    state: &mut CompressionState,
    store: &mut ContentStore,
    min_tokens: u64,
) -> CcrResult {
    let mut result = CcrResult {
        messages: Vec::with_capacity(messages.len()),
        stored: 0,
        reprojected: 0,
    };

    for message in messages {
        if message.content_type != MessageContentType::ToolResult {
            result.messages.push(message.clone());
            continue;
        }
        let text = message.text_or_empty();
        if text.is_empty() || parse_placeholder(text).is_some() {
            result.messages.push(message.clone());
            continue;
        }
        let Some(reference) = state.message_refs.ref_for(&message.id) else {
            if std::env::var("YAK_BIGCTX_DEBUG").is_ok() {
                eprintln!("[ccr] skip {}: no ref", message.id);
            }
            result.messages.push(message.clone());
            continue;
        };
        if reference == BLOCKED_REF {
            result.messages.push(message.clone());
            continue;
        }
        let reference = reference.to_string();

        // A stored ref whose raw bytes came back: re-project the frozen
        // placeholder instead of leaking the payload onto the wire.
        if let Some(entry) = store.entry(&reference) {
            if std::env::var("YAK_BIGCTX_DEBUG").is_ok() {
                let matches = store.original(&reference) == Some(text);
                eprintln!(
                    "[ccr] reproject {} ref={} stored_raw_id={} same_text={} text_head={:?} stored_head={:?}",
                    message.id, reference, entry.raw_id, matches,
                    &text[..text.len().min(60)],
                    &store.original(&reference).unwrap_or("")[..store.original(&reference).map(|o| o.len().min(60)).unwrap_or(0)]
                );
            }
            let title = entry.command.as_deref().or(Some(entry.head.as_str()));
            let placeholder = build_placeholder(&reference, &entry.kind, entry.tokens, title);
            result.reprojected += 1;
            result.messages.push(CoreMessage {
                text: Some(placeholder),
                ..message.clone()
            });
            continue;
        }

        let tokens = count_message_tokens(message);
        if std::env::var("YAK_BIGCTX_DEBUG").is_ok() {
            eprintln!("[ccr] {} ref={} tokens={}", message.id, reference, tokens);
        }
        if tokens < min_tokens {
            result.messages.push(message.clone());
            continue;
        }

        let kind = classify_kind(message.tool_name.as_deref());
        let head = normalize_head(text, MAX_HEAD_CHARS);
        // The call's own arguments name the subject better than the output's
        // first line does, when they are available.
        let command = extract_command(message.text.as_deref(), MAX_HEAD_CHARS);

        store.store(
            &reference,
            &message.id,
            text,
            &kind,
            message.tool_name.as_deref(),
            tokens,
            &head,
            command.as_deref(),
        );

        let title = command.as_deref().or(Some(head.as_str()));
        let placeholder = build_placeholder(&reference, &kind, tokens, title);
        result.stored += 1;
        result.messages.push(CoreMessage {
            text: Some(placeholder),
            ..message.clone()
        });
    }

    result
}

/// What a retrieval returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Retrieval {
    /// The text to hand back: either the original or a pointer to it.
    pub text: String,
    /// True when the original was too big to inline.
    pub is_pointer: bool,
    pub tokens: u64,
}

/// Retrieve an original by ref.
///
/// Above `inline_limit` the answer is a pointer rather than the payload:
/// inlining a 40K result would put back exactly what the store removed.
pub fn retrieve(store: &ContentStore, reference: &str, inline_limit: u64) -> Option<Retrieval> {
    let entry = store.entry(reference)?;
    let original = store.original(reference)?;

    if entry.tokens >= inline_limit {
        return Some(Retrieval {
            text: format!(
                "The stored original for #{reference} is {} tokens ({} chars) — too large to \
                 inline without re-inflating the conversation. It came from {}.\n\
                 Its opening:\n\n{}",
                group_thousands(entry.tokens),
                group_thousands(entry.chars as u64),
                entry.kind,
                entry.head
            ),
            is_pointer: true,
            tokens: entry.tokens,
        });
    }

    Some(Retrieval {
        text: original.to_string(),
        is_pointer: false,
        tokens: entry.tokens,
    })
}

/// Render the store's contents for a status report.
pub fn render_store_summary(store: &ContentStore) -> String {
    if store.is_empty() {
        return "stored originals: none".to_string();
    }
    let total_tokens: u64 = store.by_ref.values().map(|e| e.tokens).sum();
    format!(
        "stored originals: {} ({} tokens held out of the context)",
        store.len(),
        group_thousands(total_tokens)
    )
}

/// Token cost of a message, exposed for the caller's accounting.
pub fn message_tokens(message: &CoreMessage) -> u64 {
    count_message_tokens(message)
}

/// Token cost of a text, for sizing a summary or a placeholder.
pub fn text_tokens(text: &str) -> u64 {
    default_count_tokens(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::bigctx::refs::assign_refs;
    use crate::agent::bigctx::state::create_initial_state;
    use crate::agent::bigctx::types::MessageRole;

    fn result(id: &str, text: &str, tool: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::Tool, MessageContentType::ToolResult)
            .with_text(text)
            .with_tool(tool, "c1")
    }

    fn user(id: &str, text: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::User, MessageContentType::Text).with_text(text)
    }

    fn state_for(messages: &[CoreMessage]) -> CompressionState {
        let mut state = create_initial_state();
        let mut next = 1u64;
        assign_refs(messages, &mut state.message_refs, &mut next, |_| false).unwrap();
        state.next_ref_index = next;
        state
    }

    fn bulk(chars: usize) -> String {
        "x".repeat(chars)
    }

    // ── the placeholder ──

    #[test]
    fn a_placeholder_round_trips() {
        let placeholder = build_placeholder("m00042", "file read", 40_120, Some("src/auth.rs"));
        let parsed = parse_placeholder(&placeholder).expect("must parse");
        assert_eq!(parsed.reference, "m00042");
        assert_eq!(parsed.kind, "file read");
        assert_eq!(parsed.tokens, 40_120);
    }

    #[test]
    fn a_placeholder_names_the_retrieval_tool() {
        let placeholder = build_placeholder("m00042", "shell output", 1_000, None);
        assert!(placeholder.contains("acp_retrieve(\"m00042\")"));
    }

    #[test]
    fn ordinary_text_mentioning_the_marker_is_not_a_placeholder() {
        // A payload that happens to contain the marker must not be mistaken for
        // a stub of itself.
        assert!(parse_placeholder("here is [acp-stored] in prose").is_none());
        assert!(parse_placeholder("[acp-stored #m00042").is_none());
        assert!(parse_placeholder("").is_none());
    }

    #[test]
    fn a_placeholder_is_byte_stable() {
        // The same inputs must produce identical bytes: a placeholder is
        // re-sent on every request and must not move the prefix.
        let a = build_placeholder("m00001", "file read", 5_000, Some("a.rs"));
        let b = build_placeholder("m00001", "file read", 5_000, Some("a.rs"));
        assert_eq!(a, b);
    }

    #[test]
    fn thousands_are_grouped() {
        assert_eq!(group_thousands(0), "0");
        assert_eq!(group_thousands(999), "999");
        assert_eq!(group_thousands(1_000), "1,000");
        assert_eq!(group_thousands(40_120), "40,120");
        assert_eq!(group_thousands(1_234_567), "1,234,567");
    }

    #[test]
    fn the_head_is_normalised_and_capped() {
        assert_eq!(normalize_head("a\n\n  b\tc", 100), "a b c");
        let long = normalize_head(&bulk(200), 10);
        assert_eq!(long.chars().count(), 11, "ten chars plus the ellipsis");
        assert!(long.ends_with('…'));
    }

    #[test]
    fn the_kind_is_classified_from_the_tool() {
        assert_eq!(classify_kind(Some("bash")), "shell output");
        assert_eq!(classify_kind(Some("read")), "file read");
        assert_eq!(classify_kind(Some("grep")), "search output");
        assert_eq!(classify_kind(Some("webfetch")), "web fetch");
        assert_eq!(classify_kind(Some("mystery")), "tool result");
        assert_eq!(classify_kind(None), "tool result");
    }

    #[test]
    fn the_command_is_extracted_from_the_call_args() {
        let args = r#"{"path":"src/auth/verify.rs","offset":10}"#;
        assert_eq!(
            extract_command(Some(args), 96).as_deref(),
            Some("src/auth/verify.rs")
        );
        assert_eq!(extract_command(Some("not json"), 96), None);
        assert_eq!(extract_command(None, 96), None);
    }

    // ── the store ──

    #[test]
    fn storing_an_original_makes_it_retrievable() {
        let mut store = ContentStore::new();
        store.store("m00001", "t1.0", "the body", "file read", Some("read"), 10, "the", None);
        assert_eq!(store.original("m00001"), Some("the body"));
    }

    #[test]
    fn identical_content_is_stored_once() {
        let mut store = ContentStore::new();
        store.store("m00001", "t1.0", "same", "file read", None, 10, "same", None);
        store.store("m00002", "t1.1", "same", "file read", None, 10, "same", None);
        assert_eq!(store.len(), 2, "two refs");
        assert_eq!(store.by_hash.len(), 1, "one copy of the content");
    }

    #[test]
    fn the_first_write_for_a_ref_wins() {
        // Append-only: a ref always resolves to the same original.
        let mut store = ContentStore::new();
        store.store("m00001", "t1.0", "first", "file read", None, 10, "first", None);
        store.store("m00001", "t1.0", "second", "file read", None, 10, "second", None);
        assert_eq!(store.original("m00001"), Some("first"));
    }

    #[test]
    fn an_unknown_ref_retrieves_nothing() {
        let store = ContentStore::new();
        assert_eq!(store.original("m99999"), None);
        assert!(retrieve(&store, "m99999", 4_000).is_none());
    }

    // ── projection ──

    #[test]
    fn an_oversized_result_becomes_a_placeholder() {
        let messages = vec![user("t1.0", "go"), result("t1.1", &bulk(20_000), "read")];
        let mut state = state_for(&messages);
        let mut store = ContentStore::new();
        let out = project(&messages, &mut state, &mut store, 4_000);

        assert_eq!(out.stored, 1);
        assert!(out.messages[1].text_or_empty().contains(STORED_PLACEHOLDER_MARKER));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn a_small_result_is_left_alone() {
        let messages = vec![user("t1.0", "go"), result("t1.1", "small", "read")];
        let mut state = state_for(&messages);
        let mut store = ContentStore::new();
        let out = project(&messages, &mut state, &mut store, 4_000);
        assert_eq!(out.stored, 0);
        assert_eq!(out.messages[1].text_or_empty(), "small");
    }

    #[test]
    fn the_original_survives_projection() {
        let body = bulk(20_000);
        let messages = vec![user("t1.0", "go"), result("t1.1", &body, "read")];
        let mut state = state_for(&messages);
        let mut store = ContentStore::new();
        project(&messages, &mut state, &mut store, 4_000);

        let reference = state.message_refs.ref_for("t1.1").unwrap().to_string();
        assert_eq!(store.original(&reference), Some(body.as_str()));
    }

    #[test]
    fn a_second_pass_does_not_double_wrap() {
        let messages = vec![user("t1.0", "go"), result("t1.1", &bulk(20_000), "read")];
        let mut state = state_for(&messages);
        let mut store = ContentStore::new();
        let once = project(&messages, &mut state, &mut store, 4_000);
        let twice = project(&once.messages, &mut state, &mut store, 4_000);
        assert_eq!(twice.stored, 0, "already a placeholder");
        assert_eq!(
            twice.messages[1].text_or_empty(),
            once.messages[1].text_or_empty()
        );
    }

    #[test]
    fn a_raw_retransmission_is_re_projected_to_the_same_placeholder() {
        // A host that re-sends the original must not leak the payload back onto
        // the wire, and must get byte-identical placeholder bytes.
        let body = bulk(20_000);
        let messages = vec![user("t1.0", "go"), result("t1.1", &body, "read")];
        let mut state = state_for(&messages);
        let mut store = ContentStore::new();
        let first = project(&messages, &mut state, &mut store, 4_000);
        let placeholder = first.messages[1].text_or_empty().to_string();

        // the host sends the raw payload again
        let raw_again = vec![user("t1.0", "go"), result("t1.1", &body, "read")];
        let second = project(&raw_again, &mut state, &mut store, 4_000);

        assert_eq!(second.reprojected, 1);
        assert_eq!(second.stored, 0);
        assert_eq!(
            second.messages[1].text_or_empty(),
            placeholder,
            "the same bytes, or the prefix cache breaks"
        );
    }

    #[test]
    fn a_user_message_is_never_stored() {
        let messages = vec![user("t1.0", &bulk(20_000))];
        let mut state = state_for(&messages);
        let mut store = ContentStore::new();
        let out = project(&messages, &mut state, &mut store, 4_000);
        assert_eq!(out.stored, 0);
    }

    #[test]
    fn a_blocked_message_is_never_stored() {
        let messages = vec![user("t1.0", "go"), result("t1.1", &bulk(20_000), "read")];
        let mut state = create_initial_state();
        let mut next = 1u64;
        assign_refs(&messages, &mut state.message_refs, &mut next, |m| {
            m.content_type == MessageContentType::ToolResult
        })
        .unwrap();
        let mut store = ContentStore::new();
        let out = project(&messages, &mut state, &mut store, 4_000);
        assert_eq!(out.stored, 0, "no ref means nothing to retrieve by");
    }

    #[test]
    fn the_placeholder_names_the_file_when_the_args_do() {
        let mut message = result("t1.1", &bulk(20_000), "read");
        // a read tool's arguments ride the call, not the result; simulate the
        // case where the host projected them onto the result text
        message.text = Some(format!("src/auth/verify.rs\n{}", bulk(20_000)));
        let messages = vec![user("t1.0", "go"), message];
        let mut state = state_for(&messages);
        let mut store = ContentStore::new();
        let out = project(&messages, &mut state, &mut store, 4_000);
        assert!(out.messages[1].text_or_empty().contains("file read"));
    }

    // ── retrieval ──

    #[test]
    fn a_small_original_is_inlined() {
        let mut store = ContentStore::new();
        store.store("m00001", "t1.0", "the body", "file read", None, 100, "the", None);
        let got = retrieve(&store, "m00001", 4_000).unwrap();
        assert!(!got.is_pointer);
        assert_eq!(got.text, "the body");
    }

    #[test]
    fn a_large_original_returns_a_pointer() {
        // Inlining it would put back exactly what the store removed.
        let mut store = ContentStore::new();
        store.store("m00001", "t1.0", &bulk(20_000), "file read", None, 40_000, "head", None);
        let got = retrieve(&store, "m00001", 4_000).unwrap();
        assert!(got.is_pointer);
        assert!(got.text.contains("too large to inline"));
        assert!(got.text.contains("40,000"));
    }

    #[test]
    fn the_store_summary_reports_what_is_held() {
        let mut store = ContentStore::new();
        assert!(render_store_summary(&store).contains("none"));
        store.store("m00001", "t1.0", "body", "file read", None, 5_000, "b", None);
        let summary = render_store_summary(&store);
        assert!(summary.contains("1"), "{summary}");
        assert!(summary.contains("5,000"), "{summary}");
    }

    #[test]
    fn the_hash_is_stable_across_calls() {
        assert_eq!(hash_content("abc"), hash_content("abc"));
        assert_ne!(hash_content("abc"), hash_content("abd"));
    }
}
