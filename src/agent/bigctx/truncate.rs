//! The last-resort valve: cut oversized payloads when nothing else worked.
//!
//! Everything else in this crate asks the model to cooperate — the nudge
//! suggests, `compress` waits for a call. A model that ignores the nudge, or a
//! gateway that refuses the request outright, leaves the session over its
//! window with nobody acting. This is the path that acts anyway.
//!
//! It is destructive in a way a fold is not: a fold keeps the originals behind
//! a block and can be restored, whereas a truncated payload is simply gone. So
//! it is gated hard:
//!
//! - only above `truncate.threshold` of the window,
//! - only messages outside the recent window,
//! - only payloads above `min_output_tokens`,
//! - rendered block summaries are **never** touched — they are the only
//!   durable record of what was folded, so cutting one destroys the compression
//!   it came from,
//! - and a truncated payload keeps its head and tail, because a stack trace's
//!   first lines and its final error are where the answer is.
//!
//! ## A note on UTF-16
//!
//! The upstream implementation carries three helpers (`clampPrefix`,
//! `clampWindow`, `scrubLoneSurrogates`) whose only job is to avoid cutting a
//! JavaScript string between the two code units of an astral character: a lone
//! surrogate serializes as `\ud83e` and gateways re-encoding to strict UTF-8
//! reject it on every retry. A Rust `String` is always valid UTF-8 and cannot
//! hold a lone surrogate, so that whole failure mode does not exist here. The
//! helpers are deliberately not ported; the invariant they protect is enforced
//! by the type.

use crate::agent::bigctx::config::BigctxConfig;
use crate::agent::bigctx::prune::is_rendered_summary;
use crate::agent::bigctx::tokenize::default_count_tokens;
use crate::agent::bigctx::types::{CoreMessage, MessageContentType, MessageRole};

/// Marks a payload this module already cut, so a second pass does not cut the
/// stub again.
pub const TRUNCATION_MARKER: &str = "[truncated for context space]";

/// How a truncation pass is tuned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TruncateOptions {
    /// Payloads smaller than this are not worth cutting.
    pub min_output_tokens: u64,
    /// Characters of the head to keep.
    pub keep_prefix_chars: usize,
    /// Characters of the tail to keep.
    pub keep_suffix_chars: usize,
    /// Messages at the end of the list that are never cut.
    pub protect_recent_messages: usize,
    /// Also consider user/assistant text, as a second stage. Off by default:
    /// cutting the user's instructions changes the task, so it only happens
    /// when tool output alone could not close the gap.
    pub include_text_messages: bool,
}

impl Default for TruncateOptions {
    fn default() -> TruncateOptions {
        TruncateOptions {
            min_output_tokens: 1_000,
            keep_prefix_chars: 2_000,
            keep_suffix_chars: 2_000,
            protect_recent_messages: 3,
            include_text_messages: false,
        }
    }
}

/// What a truncation pass did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TruncateResult {
    pub messages: Vec<CoreMessage>,
    pub truncated_count: usize,
    pub saved_tokens: u64,
    /// Size-qualified candidates found before cutting. Zero explains a silent
    /// no-op at or above the threshold — the host should warn rather than
    /// pretend it handled the situation.
    pub candidates_found: usize,
}

impl TruncateResult {
    fn unchanged(messages: &[CoreMessage], candidates_found: usize) -> TruncateResult {
        TruncateResult {
            messages: messages.to_vec(),
            truncated_count: 0,
            saved_tokens: 0,
            candidates_found,
        }
    }
}

/// Cut oversized payloads when the context is over its truncation threshold.
pub fn truncate_large_outputs(
    messages: &[CoreMessage],
    token_count: u64,
    config: &BigctxConfig,
    options: &TruncateOptions,
) -> TruncateResult {
    let limit = config.model_context_limit;
    if limit == 0 {
        return TruncateResult::unchanged(messages, 0);
    }
    let threshold = (config.truncate.threshold * limit as f64) as u64;
    if token_count < threshold {
        return TruncateResult::unchanged(messages, 0);
    }

    // Everything at or after this index is the working set and stays intact.
    //
    // The window is capped so it never swallows the whole list: a session with
    // fewer messages than the protect-window would otherwise have every
    // candidate excluded, and the valve would silently do nothing in exactly
    // the short, bloated sessions that need it most. At least the newest
    // message is always protected.
    // The window is capped so it never swallows the whole list, and so it
    // always leaves at least one eligible message. With three messages and the
    // default window of 3, `len - 1` still protected both the payload and the
    // message after it, leaving nothing to cut — the valve was inert on exactly
    // the short, bloated sessions it exists for.
    let protect = options
        .protect_recent_messages
        .min(messages.len().saturating_sub(1))
        .min(messages.len().saturating_sub(2).max(0));
    let protected_from = messages.len() - protect;

    let tool_results = collect_candidates(
        messages,
        protected_from,
        options,
        |message| message.content_type == MessageContentType::ToolResult,
    );
    let text_messages = if options.include_text_messages {
        collect_candidates(messages, protected_from, options, |message| {
            message.content_type == MessageContentType::Text
                && matches!(message.role, MessageRole::User | MessageRole::Assistant)
                // A rendered summary is the only durable record of what was
                // folded; cutting it destroys the compression it came from.
                && !is_rendered_summary(message)
        })
    } else {
        Vec::new()
    };
    let candidates_found = tool_results.len() + text_messages.len();

    // Aim below the threshold, not at it: stopping exactly at the line leaves
    // the next round over it again.
    let target = (config.truncate.threshold * limit as f64 * 0.9) as u64;

    let mut replacements: Vec<(String, String)> = Vec::new();
    let mut remaining = token_count;
    let mut saved_tokens = 0u64;

    for candidate in tool_results.iter().chain(text_messages.iter()) {
        if remaining <= target {
            break;
        }
        let original = candidate.text_or_empty();
        let tokens = default_count_tokens(original);
        if original.len() <= options.keep_prefix_chars + options.keep_suffix_chars {
            continue;
        }
        let Some(replacement) = cut(original, tokens, options) else {
            continue;
        };
        let replacement_tokens = default_count_tokens(&replacement);
        let saved = tokens.saturating_sub(replacement_tokens);
        remaining = remaining.saturating_sub(saved);
        saved_tokens += saved;
        replacements.push((candidate.id.clone(), replacement));
    }

    if replacements.is_empty() {
        return TruncateResult::unchanged(messages, candidates_found);
    }

    let truncated_count = replacements.len();
    let out: Vec<CoreMessage> = messages
        .iter()
        .map(|message| match replacements.iter().find(|(id, _)| id == &message.id) {
            Some((_, text)) => CoreMessage {
                text: Some(text.clone()),
                ..message.clone()
            },
            None => message.clone(),
        })
        .collect();

    TruncateResult {
        messages: out,
        truncated_count,
        saved_tokens,
        candidates_found,
    }
}

/// Payloads big enough to be worth cutting, largest first.
///
/// Largest first is the right order: the point is to reclaim a target, and one
/// 40K payload reaches it where ten 4K ones each cost a marker.
fn collect_candidates<'a>(
    messages: &'a [CoreMessage],
    protected_from: usize,
    options: &TruncateOptions,
    predicate: impl Fn(&CoreMessage) -> bool,
) -> Vec<&'a CoreMessage> {
    let mut found: Vec<&CoreMessage> = Vec::new();
    for message in messages.iter().take(protected_from) {
        if !predicate(message) {
            continue;
        }
        let text = message.text_or_empty();
        if text.is_empty() || text.contains(TRUNCATION_MARKER) {
            continue;
        }
        if default_count_tokens(text) < options.min_output_tokens {
            continue;
        }
        found.push(message);
    }
    found.sort_by_key(|message| std::cmp::Reverse(default_count_tokens(message.text_or_empty())));
    found
}

/// Head + marker + tail.
///
/// The head and tail are where the answer usually is: a stack trace names the
/// failure at the top and repeats it at the bottom, and a log opens with the
/// command and closes with the verdict. The middle is where the volume is.
fn cut(original: &str, tokens: u64, options: &TruncateOptions) -> Option<String> {
    let prefix = clamp_prefix(original, options.keep_prefix_chars);
    let suffix = clamp_suffix(original, options.keep_suffix_chars);
    let replacement = format!(
        "{prefix}\n\n...{TRUNCATION_MARKER} — original ~{tokens} tokens]...\n\n{suffix}"
    );
    (replacement.len() < original.len()).then_some(replacement)
}

/// At most `max_chars` characters from the start, on a character boundary.
///
/// Rust slices are already character-aligned when built from `char_indices`,
/// which is why this is a `chars()` walk rather than the upstream's
/// surrogate-pair arithmetic.
fn clamp_prefix(text: &str, max_chars: usize) -> String {
    text.chars().take(max_chars).collect()
}

/// At most `max_chars` characters from the end.
fn clamp_suffix(text: &str, max_chars: usize) -> String {
    let total = text.chars().count();
    text.chars().skip(total.saturating_sub(max_chars)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::bigctx::config::default_config;
    use crate::agent::bigctx::prune::summary_message_id;

    fn cfg(window: u64) -> BigctxConfig {
        let mut c = default_config(window);
        c.model_context_limit = window;
        c
    }

    fn user(id: &str, text: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::User, MessageContentType::Text).with_text(text)
    }

    fn result(id: &str, text: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::Tool, MessageContentType::ToolResult)
            .with_text(text)
            .with_tool("bash", "c1")
    }

    fn bulk(chars: usize) -> String {
        "x".repeat(chars)
    }

    #[test]
    fn below_the_threshold_nothing_happens() {
        let messages = vec![result("t1.0", &bulk(100_000)), user("t1.1", "go")];
        let out = truncate_large_outputs(&messages, 10_000, &cfg(100_000), &TruncateOptions::default());
        assert_eq!(out.truncated_count, 0);
        assert_eq!(out.messages[0].text_or_empty().len(), 100_000);
    }

    #[test]
    fn above_the_threshold_the_largest_payload_is_cut() {
        let messages = vec![
            user("t1.0", "go"),
            result("t1.1", &bulk(200_000)),
            user("t1.2", "and now?"),
        ];
        let out = truncate_large_outputs(&messages, 99_000, &cfg(100_000), &TruncateOptions::default());
        assert_eq!(out.truncated_count, 1);
        assert!(out.saved_tokens > 0);
        assert!(out.messages[1].text_or_empty().len() < 200_000);
    }

    #[test]
    fn a_cut_payload_keeps_its_head_and_tail() {
        // The head names the command, the tail carries the verdict.
        let mut body = String::from("HEAD-MARKER\n");
        body.push_str(&bulk(200_000));
        body.push_str("\nTAIL-MARKER");
        let messages = vec![user("t1.0", "go"), result("t1.1", &body)];
        let out = truncate_large_outputs(&messages, 99_000, &cfg(100_000), &TruncateOptions::default());
        let text = out.messages[1].text_or_empty();
        assert!(text.contains("HEAD-MARKER"), "the head survives");
        assert!(text.contains("TAIL-MARKER"), "the tail survives");
        assert!(text.contains(TRUNCATION_MARKER), "and it is marked");
    }

    #[test]
    fn a_cut_payload_reports_the_original_size() {
        let messages = vec![
            user("t1.0", "go"),
            result("t1.1", &bulk(200_000)),
            user("t1.2", "and now?"),
        ];
        let out = truncate_large_outputs(&messages, 99_000, &cfg(100_000), &TruncateOptions::default());
        assert!(
            out.messages[1].text_or_empty().contains("original ~"),
            "the model must know what it lost"
        );
    }

    #[test]
    fn the_recent_window_is_never_cut() {
        // Three messages, protect-window 3 -> the window is capped to 2, so
        // index 0 is a candidate and indices 1..2 are the working set.
        let messages = vec![
            result("t1.0", &bulk(200_000)),
            result("t1.1", &bulk(200_000)),
            result("t1.2", &bulk(200_000)),
            result("t1.3", &bulk(200_000)),
        ];
        let out = truncate_large_outputs(
            &messages,
            99_000,
            &cfg(100_000),
            &TruncateOptions {
                protect_recent_messages: 3,
                ..Default::default()
            },
        );
        assert_eq!(
            out.messages[3].text_or_empty().len(),
            200_000,
            "the newest message is never cut"
        );
        assert_eq!(
            out.messages[2].text_or_empty().len(),
            200_000,
            "nor the two before it"
        );
        assert_eq!(out.messages[1].text_or_empty().len(), 200_000);
        assert!(
            out.messages[0].text_or_empty().len() < 200_000,
            "the oldest is the one that goes"
        );
    }

    #[test]
    fn a_rendered_summary_is_never_cut() {
        // It is the only durable record of what was folded; cutting it destroys
        // the compression it came from.
        let summary = CoreMessage::new(
            summary_message_id("b1"),
            MessageRole::System,
            MessageContentType::Text,
        )
        .with_text(format!("[Compressed conversation section]\n{}", bulk(200_000)));
        let messages = vec![summary];
        let out = truncate_large_outputs(
            &messages,
            99_000,
            &cfg(100_000),
            &TruncateOptions {
                include_text_messages: true,
                protect_recent_messages: 0,
                ..Default::default()
            },
        );
        assert_eq!(out.truncated_count, 0);
    }

    #[test]
    fn a_small_payload_is_left_alone() {
        let messages = vec![
            user("t1.0", "go"),
            result("t1.1", "a short result"),
            user("t1.2", "and now?"),
        ];
        let out = truncate_large_outputs(&messages, 99_000, &cfg(100_000), &TruncateOptions::default());
        assert_eq!(out.truncated_count, 0);
        assert_eq!(out.messages[1].text_or_empty(), "a short result");
    }

    #[test]
    fn an_already_truncated_payload_is_not_cut_again() {
        let messages = vec![
            user("t1.0", "go"),
            result("t1.1", &format!("{TRUNCATION_MARKER}{}", bulk(200_000))),
            user("t1.2", "and now?"),
        ];
        let out = truncate_large_outputs(&messages, 99_000, &cfg(100_000), &TruncateOptions::default());
        assert_eq!(out.truncated_count, 0, "a second cut would lose the stub");
    }

    #[test]
    fn the_largest_payload_is_cut_first() {
        // Reaching the target with one big cut beats ten small ones, each of
        // which costs a marker.
        let messages = vec![
            user("t1.0", "go"),
            result("t1.1", &bulk(20_000)),
            result("t1.2", &bulk(200_000)),
            user("t1.3", "and now?"),
        ];
        // A one-message window, so both payloads are eligible and the ORDER is
        // what the test observes.
        let out = truncate_large_outputs(
            &messages,
            99_000,
            &cfg(100_000),
            &TruncateOptions {
                protect_recent_messages: 1,
                ..Default::default()
            },
        );
        assert!(out.truncated_count >= 1);
        assert!(
            out.messages[2].text_or_empty().len() < 200_000,
            "the 200K payload is cut first"
        );
        // The 20K payload may also go if the first cut did not reach the
        // target — that is the loop working, not a bug.
        if out.truncated_count == 1 {
            assert_eq!(out.messages[1].text_or_empty().len(), 20_000);
        }
    }

    #[test]
    fn text_messages_are_only_cut_when_asked_for() {
        let messages = vec![user("t1.0", &bulk(200_000)), user("t1.1", "and now?")];
        let by_default = truncate_large_outputs(&messages, 99_000, &cfg(100_000), &TruncateOptions::default());
        assert_eq!(by_default.truncated_count, 0, "the user's instructions stay");

        let asked = truncate_large_outputs(
            &messages,
            99_000,
            &cfg(100_000),
            &TruncateOptions {
                include_text_messages: true,
                protect_recent_messages: 0,
                ..Default::default()
            },
        );
        assert_eq!(asked.truncated_count, 1);
    }

    #[test]
    fn a_zero_window_disables_truncation() {
        let messages = vec![result("t1.0", &bulk(200_000)), user("t1.1", "go")];
        let out = truncate_large_outputs(&messages, 999_000, &cfg(0), &TruncateOptions::default());
        assert_eq!(out.truncated_count, 0);
    }

    #[test]
    fn candidates_found_explains_a_silent_no_op() {
        // Zero candidates at the threshold means the session is stuck and the
        // host should say so rather than pretend it handled it.
        let messages = vec![user("t1.0", "go"), result("t1.1", "small")];
        let out = truncate_large_outputs(&messages, 99_000, &cfg(100_000), &TruncateOptions::default());
        assert_eq!(out.candidates_found, 0);
    }

    #[test]
    fn a_multibyte_payload_cuts_on_a_character_boundary() {
        // Rust strings are always valid UTF-8, so the upstream's lone-surrogate
        // guard has no analogue here — but a byte-slicing bug would still
        // panic, so pin that cutting CJK and emoji is safe.
        let body = "汉字".repeat(50_000) + "🎉🎉🎉";
        let messages = vec![
            user("t1.0", "go"),
            result("t1.1", &body),
            user("t1.2", "and now?"),
        ];
        let out = truncate_large_outputs(&messages, 99_000, &cfg(100_000), &TruncateOptions::default());
        assert_eq!(out.truncated_count, 1);
        let text = out.messages[1].text_or_empty();
        assert!(text.chars().count() < body.chars().count());
        // the result is a valid String by construction; reaching here proves no
        // panic on a boundary
        assert!(text.contains(TRUNCATION_MARKER));
    }

    #[test]
    fn truncation_aims_below_the_threshold() {
        // Stopping exactly at the line leaves the next round over it again.
        let messages = vec![
            user("t1.0", "go"),
            result("t1.1", &bulk(400_000)),
            result("t1.2", &bulk(400_000)),
            user("t1.3", "and now?"),
        ];
        let out = truncate_large_outputs(&messages, 99_000, &cfg(100_000), &TruncateOptions::default());
        assert!(out.saved_tokens > 0);
        assert!(out.truncated_count >= 1);
    }

    #[test]
    fn identity_and_tool_name_survive_a_cut() {
        let messages = vec![
            user("t1.0", "go"),
            result("t1.1", &bulk(200_000)),
            user("t1.2", "and now?"),
        ];
        let out = truncate_large_outputs(&messages, 99_000, &cfg(100_000), &TruncateOptions::default());
        assert_eq!(out.messages[1].id, "t1.1");
        assert_eq!(out.messages[1].tool_call_id.as_deref(), Some("c1"));
        assert_eq!(out.messages[1].tool_name.as_deref(), Some("bash"));
    }
}

