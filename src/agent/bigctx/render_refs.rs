//! Showing the model its own refs.
//!
//! Without this the model has no way to name a message except by asking
//! `acp_status` first. That is a round trip on every compression, and it makes
//! the whole feature feel like a tool you have to look up rather than a thing
//! you can just do. Each message gets a tag in front of its text:
//!
//! ```text
//! <acp tokens="3.7K" type="tool">m00042</acp>
//! line 0: the quick brown fox …
//! ```
//!
//! The model reads the ref straight off the message it wants to fold.
//!
//! ## Two properties that are load-bearing
//!
//! **The token count is frozen at first render.** The tag sits at the head of
//! its message, so a tag whose number changes on every turn shifts the prefix
//! and invalidates the provider's cache for everything after it. The snapshot
//! is written once per ref and reused forever — a stale-but-stable count beats
//! an accurate one that costs a cache miss.
//!
//! **Re-rendering is idempotent.** A message that already carries its own tag
//! has it stripped before the fresh one goes on. Without that the tags stack
//! up every pass and the text grows without bound.

use crate::agent::bigctx::refs::BLOCKED_REF;
use crate::agent::bigctx::tokenize::default_count_tokens;
use crate::agent::bigctx::types::{CompressionState, CoreMessage, MessageContentType};

/// Which messages get a tag written into their text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderStrategy {
    /// Tag every mapped message.
    All,
    /// Tag only user/assistant text. Structured tool content stays pristine —
    /// a proxy host that forwards it would otherwise pollute the payload.
    TextOnly,
    /// Tag nothing. The host reads the ref map directly.
    None,
}

/// `999` → `999`, `3700` → `3.7K`, `41000` → `41K`.
fn format_tokens(tokens: u64) -> String {
    if tokens < 1_000 {
        return tokens.to_string();
    }
    if tokens < 10_000 {
        return format!("{:.1}K", tokens as f64 / 1_000.0);
    }
    format!("{}K", (tokens as f64 / 1_000.0).round() as u64)
}

/// What a message is, for the tag's `type` attribute.
fn classify(message: &CoreMessage) -> String {
    match message.content_type {
        MessageContentType::ToolCall | MessageContentType::ToolResult => message
            .tool_name
            .clone()
            .unwrap_or_else(|| "tool".to_string()),
        other => format!("{other:?}").to_lowercase(),
    }
}

/// The tag for one message.
fn acp_tag(reference: &str, tokens: u64, kind: &str) -> String {
    format!(
        "<acp tokens=\"{}\" type=\"{}\">{}</acp>",
        format_tokens(tokens),
        kind,
        reference
    )
}

/// Strip a leading tag that names `reference`.
///
/// Only this message's own tag is removed: a message whose text legitimately
/// starts with another ref's tag is content, not a stale render.
fn strip_own_tag<'a>(text: &'a str, reference: &str) -> &'a str {
    let Some(rest) = text.strip_prefix("<acp ") else {
        return text;
    };
    let Some(close) = rest.find('>') else {
        return text;
    };
    let body = &rest[close + 1..];
    let expected = format!("{reference}</acp>");
    let stripped = match body.strip_prefix(&expected) {
        Some(after) => after,
        None => return text,
    };
    stripped.strip_prefix('\n').unwrap_or(stripped)
}

/// Render tags into a message list.
///
/// The state's token snapshot is updated in place: a ref is priced on first
/// render and never re-priced.
pub fn render_refs(
    messages: &[CoreMessage],
    state: &mut CompressionState,
    strategy: RenderStrategy,
) -> Vec<CoreMessage> {
    if strategy == RenderStrategy::None {
        return messages.to_vec();
    }

    messages
        .iter()
        .map(|message| {
            let Some(reference) = state.message_refs.ref_for(&message.id) else {
                return message.clone();
            };
            if reference == BLOCKED_REF {
                return message.clone();
            }
            if strategy == RenderStrategy::TextOnly
                && message.content_type != MessageContentType::Text
            {
                return message.clone();
            }

            let clean = strip_own_tag(message.text_or_empty(), reference);

            // Frozen at first render. The tag is at the head of its message, so
            // a count that changed per turn would shift the prefix and
            // invalidate the provider's cache for everything after it.
            let tokens = match state.token_snapshot.get(reference) {
                Some(frozen) => *frozen,
                None => {
                    let measured = default_count_tokens(clean);
                    state
                        .token_snapshot
                        .insert(reference.to_string(), measured);
                    measured
                }
            };

            let tag = acp_tag(reference, tokens, &classify(message));
            let text = if clean.is_empty() {
                tag
            } else {
                format!("{tag}\n{clean}")
            };

            CoreMessage {
                text: Some(text),
                ..message.clone()
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::bigctx::refs::assign_refs;
    use crate::agent::bigctx::state::create_initial_state;
    use crate::agent::bigctx::types::MessageRole;

    fn user(id: &str, text: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::User, MessageContentType::Text).with_text(text)
    }

    fn result(id: &str, text: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::Tool, MessageContentType::ToolResult)
            .with_text(text)
            .with_tool("read", "c1")
    }

    fn state_for(messages: &[CoreMessage]) -> CompressionState {
        let mut state = create_initial_state();
        let mut next = 1u64;
        assign_refs(messages, &mut state.message_refs, &mut next, |_| false).unwrap();
        state.next_ref_index = next;
        state
    }

    #[test]
    fn every_message_gets_its_ref_in_front() {
        let messages = vec![user("t1.0", "hello"), result("t1.1", "body")];
        let mut state = state_for(&messages);
        let rendered = render_refs(&messages, &mut state, RenderStrategy::All);

        assert!(rendered[0].text_or_empty().starts_with("<acp "), "{}", rendered[0].text_or_empty());
        assert!(rendered[0].text_or_empty().contains("m00001</acp>"));
        assert!(rendered[0].text_or_empty().ends_with("hello"));
        assert!(rendered[1].text_or_empty().contains("m00002</acp>"));
    }

    #[test]
    fn the_tag_carries_the_kind_and_the_size() {
        let messages = vec![result("t1.0", &"x".repeat(4_000))];
        let mut state = state_for(&messages);
        let rendered = render_refs(&messages, &mut state, RenderStrategy::All);
        let text = rendered[0].text_or_empty();
        assert!(text.contains("type=\"read\""), "{text}");
        assert!(text.contains("tokens=\"1.0K\""), "{text}");
    }

    #[test]
    fn the_token_count_is_frozen_at_first_render() {
        // The tag sits at the head of its message; a count that changed per
        // turn would shift the prefix and invalidate the cache.
        let messages = vec![user("t1.0", &"x".repeat(400))];
        let mut state = state_for(&messages);
        let first = render_refs(&messages, &mut state, RenderStrategy::All);
        let frozen = first[0].text_or_empty().to_string();

        // the text grows, but the tag must not
        let grown = vec![user("t1.0", &"x".repeat(4_000))];
        let second = render_refs(&grown, &mut state, RenderStrategy::All);
        let tag_of = |t: &str| t.lines().next().unwrap_or("").to_string();
        assert_eq!(
            tag_of(&frozen),
            tag_of(second[0].text_or_empty()),
            "the frozen tag must not move"
        );
    }

    #[test]
    fn re_rendering_does_not_stack_tags() {
        let messages = vec![user("t1.0", "hello")];
        let mut state = state_for(&messages);

        let once = render_refs(&messages, &mut state, RenderStrategy::All);
        let twice = render_refs(&once, &mut state, RenderStrategy::All);
        let thrice = render_refs(&twice, &mut state, RenderStrategy::All);

        assert_eq!(
            thrice[0].text_or_empty(),
            once[0].text_or_empty(),
            "a second render must replace the tag, not add one"
        );
        assert_eq!(
            thrice[0].text_or_empty().matches("<acp ").count(),
            1,
            "exactly one tag"
        );
    }

    #[test]
    fn a_foreign_tag_at_the_head_survives() {
        // Text that legitimately starts with another ref's tag is content.
        let messages = vec![user("t1.0", "<acp tokens=\"5\" type=\"text\">m99999</acp>\nreal content")];
        let mut state = state_for(&messages);
        let rendered = render_refs(&messages, &mut state, RenderStrategy::All);
        assert!(
            rendered[0].text_or_empty().contains("m99999"),
            "a foreign tag is not this message's stale render"
        );
    }

    #[test]
    fn a_blocked_message_gets_no_tag() {
        let messages = vec![user("t1.0", "hello")];
        let mut state = create_initial_state();
        let mut next = 1u64;
        assign_refs(&messages, &mut state.message_refs, &mut next, |_| true).unwrap();
        let rendered = render_refs(&messages, &mut state, RenderStrategy::All);
        assert_eq!(rendered[0].text_or_empty(), "hello");
    }

    #[test]
    fn text_only_leaves_tool_content_pristine() {
        let messages = vec![user("t1.0", "ask"), result("t1.1", "the body")];
        let mut state = state_for(&messages);
        let rendered = render_refs(&messages, &mut state, RenderStrategy::TextOnly);
        assert!(rendered[0].text_or_empty().contains("<acp "));
        assert_eq!(
            rendered[1].text_or_empty(),
            "the body",
            "structured content must not be polluted"
        );
    }

    #[test]
    fn the_none_strategy_touches_nothing() {
        let messages = vec![user("t1.0", "hello")];
        let mut state = state_for(&messages);
        let rendered = render_refs(&messages, &mut state, RenderStrategy::None);
        assert_eq!(rendered[0].text_or_empty(), "hello");
    }

    #[test]
    fn an_empty_message_still_gets_its_tag() {
        let messages = vec![user("t1.0", "")];
        let mut state = state_for(&messages);
        let rendered = render_refs(&messages, &mut state, RenderStrategy::All);
        assert!(rendered[0].text_or_empty().starts_with("<acp "));
        assert!(rendered[0].text_or_empty().ends_with("</acp>"));
    }

    #[test]
    fn format_tokens_is_compact() {
        assert_eq!(format_tokens(0), "0");
        assert_eq!(format_tokens(999), "999");
        assert_eq!(format_tokens(1_500), "1.5K");
        assert_eq!(format_tokens(41_000), "41K");
    }

    #[test]
    fn the_snapshot_records_every_ref_it_priced() {
        let messages = vec![user("t1.0", "a"), user("t1.1", "b")];
        let mut state = state_for(&messages);
        let _ = render_refs(&messages, &mut state, RenderStrategy::All);
        assert_eq!(state.token_snapshot.len(), 2);
    }
}
