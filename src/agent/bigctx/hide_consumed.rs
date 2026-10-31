//! Keeping the compress calls from becoming the next problem.
//!
//! A `compress` call quotes every range it folded, and its arguments carry the
//! full summary text — which the rendered block summary already holds. On a
//! long session that duplication alone was measured at ~22K tokens upstream.
//! Worse, a call whose block has since been consumed or restored is dead
//! weight: it describes a fold that no longer exists.
//!
//! So consumed calls are removed from the view entirely, live ones are
//! rewritten down to the ranges that still exist, and their summaries are
//! stubbed. The block stays the durable record; the call is only a receipt.
//!
//! **Orphans are kept, but bounded.** A compress call with no block behind it
//! is a failure the model must be able to see — hiding it makes a
//! deterministic model re-issue the same no-op forever, pinned at a fixed
//! point (upstream measured 3,849 identical calls over five hours). The newest
//! [`KEEP_LAST_ORPHANED`] orphans stay visible; older ones are hidden, so the
//! residue is bounded regardless of session length.

use std::collections::{BTreeMap, BTreeSet};

use crate::agent::bigctx::types::{CompressionState, CoreMessage, MessageContentType};

/// How many orphaned compress calls stay visible.
pub const KEEP_LAST_ORPHANED: usize = 2;

/// How much of a live call's summary survives, for recall.
const SUMMARY_STUB_CHARS: usize = 200;

/// The name the compress tool answers to.
pub const COMPRESS_TOOL_NAME: &str = "compress";

/// What hiding changed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HideResult {
    pub messages: Vec<CoreMessage>,
    /// How many messages were removed from the view.
    pub hidden: usize,
    /// Refs of the orphan calls/results this pass hid. Boundary resolution
    /// snaps a range endpoint onto a hidden-but-intact orphan with these.
    pub hidden_orphan_refs: Vec<String>,
}

/// What happened to one compress call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fate {
    /// Its block is still active: the call is a live receipt.
    Active,
    /// Its block was consumed or restored: the call describes a fold that no
    /// longer exists.
    Inactive,
    /// No block ever existed: a failed or no-op call.
    Orphan,
}

fn is_compress_call(message: &CoreMessage) -> bool {
    message.content_type == MessageContentType::ToolCall
        && message.tool_name.as_deref() == Some(COMPRESS_TOOL_NAME)
}

/// Remove consumed compress calls, rewrite live ones, keep recent orphans.
pub fn hide_consumed_compress_calls(
    state: &CompressionState,
    messages: &[CoreMessage],
) -> HideResult {
    // Which call ids still have an active block, and which ever had one.
    let mut all_call_ids: BTreeSet<&str> = BTreeSet::new();
    let mut active_call_ids: BTreeSet<&str> = BTreeSet::new();
    let mut live_ranges: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();

    for block in &state.blocks {
        let Some(call_id) = block.compress_call_id.as_deref() else {
            continue;
        };
        all_call_ids.insert(call_id);
        if !block.active {
            continue;
        }
        active_call_ids.insert(call_id);
        let mut ranges = live_ranges.entry(call_id).or_default();
        for (start, end) in quoted_ranges(block) {
            ranges.insert(range_key(&start, &end));
        }
    }

    let fate_of = |message: &CoreMessage| -> Fate {
        let Some(call_id) = message.tool_call_id.as_deref() else {
            return Fate::Orphan;
        };
        if active_call_ids.contains(call_id) {
            Fate::Active
        } else if all_call_ids.contains(call_id) {
            Fate::Inactive
        } else {
            Fate::Orphan
        }
    };

    // The newest orphans stay visible, so a failing call cannot loop unseen.
    let mut kept_orphans: BTreeSet<String> = BTreeSet::new();
    for message in messages.iter().rev() {
        if kept_orphans.len() >= KEEP_LAST_ORPHANED {
            break;
        }
        if is_compress_call(message)
            && fate_of(message) == Fate::Orphan
            && let Some(call_id) = message.tool_call_id.as_deref()
        {
            kept_orphans.insert(call_id.to_string());
        }
    }

    // Which calls disappear, and which stay.
    let mut hidden_call_ids: BTreeSet<String> = BTreeSet::new();
    let mut fate_by_call_id: BTreeMap<String, Fate> = BTreeMap::new();
    for message in messages {
        if !is_compress_call(message) {
            continue;
        }
        let Some(call_id) = message.tool_call_id.as_deref() else {
            continue;
        };
        let fate = fate_of(message);
        fate_by_call_id.insert(call_id.to_string(), fate);
        let hide = match fate {
            Fate::Inactive => true,
            Fate::Orphan => !kept_orphans.contains(call_id),
            Fate::Active => false,
        };
        if hide {
            hidden_call_ids.insert(call_id.to_string());
        }
    }

    let mut out: Vec<CoreMessage> = Vec::with_capacity(messages.len());
    let mut hidden = 0usize;
    let mut hidden_orphan_refs: Vec<String> = Vec::new();

    let remember = |message: &CoreMessage, hidden_orphan_refs: &mut Vec<String>| {
        let fate = message
            .tool_call_id
            .as_deref()
            .and_then(|id| fate_by_call_id.get(id).copied());
        if fate != Some(Fate::Orphan) {
            return;
        }
        if let Some(reference) = state.message_refs.ref_for(&message.id)
            && reference != crate::agent::bigctx::refs::BLOCKED_REF
            && !hidden_orphan_refs.iter().any(|r| r == reference)
        {
            hidden_orphan_refs.push(reference.to_string());
        }
    };

    for message in messages {
        // The call itself.
        if is_compress_call(message) {
            let hidden_here = message
                .tool_call_id
                .as_deref()
                .is_none_or(|id| hidden_call_ids.contains(id));
            if hidden_here {
                hidden += 1;
                remember(message, &mut hidden_orphan_refs);
                continue;
            }
            // A live call: keep only the ranges that still exist, and stub the
            // summaries the block already carries.
            if let Some(call_id) = message.tool_call_id.as_deref()
                && let Some(ranges) = live_ranges.get(call_id)
            {
                let rewritten = rewrite_call_text(&message.text_or_empty(), ranges);
                out.push(CoreMessage {
                    text: Some(rewritten),
                    ..message.clone()
                });
                continue;
            }
            out.push(message.clone());
            continue;
        }

        // Its result goes with it, or the exchange is left unpaired.
        if message.content_type == MessageContentType::ToolResult
            && let Some(call_id) = message.tool_call_id.as_deref()
            && hidden_call_ids.contains(call_id)
        {
            hidden += 1;
            remember(message, &mut hidden_orphan_refs);
            continue;
        }

        out.push(message.clone());
    }

    HideResult {
        messages: out,
        hidden,
        hidden_orphan_refs,
    }
}

fn range_key(start: &str, end: &str) -> String {
    format!("{start}::{end}")
}

/// The range a block recorded.
///
/// Empty when the block has no recorded span (one created before the refs were
/// stored). The caller then keeps the call's text as written: a wrong rewrite
/// is worse than a stale one.
fn quoted_ranges(block: &crate::agent::bigctx::types::CompressionBlock) -> Vec<(String, String)> {
    match (&block.start_ref, &block.end_ref) {
        (Some(start), Some(end)) => vec![(start.clone(), end.clone())],
        _ => Vec::new(),
    }
}

/// Rewrite a live call's argument text: drop ranges that no longer exist and
/// stub the summaries.
///
/// Returns the original text when nothing changed, so an unchanged call is
/// byte-identical and the provider's prefix cache survives.
fn rewrite_call_text(text: &str, live_ranges: &BTreeSet<String>) -> String {
    let Some(start) = text.find('{') else {
        return text.to_string();
    };
    let prefix = &text[..start];
    let Ok(mut parsed) = serde_json::from_str::<serde_json::Value>(&text[start..]) else {
        return text.to_string();
    };
    let Some(content) = parsed.get_mut("content") else {
        return text.to_string();
    };

    let mut changed = false;
    let rewritten = match content {
        serde_json::Value::Array(entries) => {
            let kept: Vec<serde_json::Value> = entries
                .iter()
                .filter(|entry| {
                    let (Some(s), Some(e)) = (
                        entry.get("startId").and_then(|v| v.as_str()),
                        entry.get("endId").and_then(|v| v.as_str()),
                    ) else {
                        return true;
                    };
                    live_ranges.contains(&range_key(s, e))
                })
                .map(|entry| stub_summary(entry, &mut changed))
                .collect();
            if kept.len() != entries.len() {
                changed = true;
            }
            if kept.is_empty() {
                return text.to_string();
            }
            serde_json::Value::Array(kept)
        }
        serde_json::Value::String(_) => return text.to_string(),
        _ => return text.to_string(),
    };

    if !changed {
        return text.to_string();
    }
    *content = rewritten;
    format!("{prefix}{parsed}")
}

/// Shorten a range entry's summary to a recall stub.
fn stub_summary(entry: &serde_json::Value, changed: &mut bool) -> serde_json::Value {
    let Some(summary) = entry.get("summary").and_then(|v| v.as_str()) else {
        return entry.clone();
    };
    if summary.chars().count() <= SUMMARY_STUB_CHARS {
        return entry.clone();
    }
    *changed = true;
    let stub: String = summary.chars().take(SUMMARY_STUB_CHARS - 1).collect();
    let mut out = entry.clone();
    if let Some(obj) = out.as_object_mut() {
        obj.insert(
            "summary".to_string(),
            serde_json::Value::String(format!("{stub}…")),
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::bigctx::state::create_initial_state;
    use crate::agent::bigctx::types::{
        BlockGeneration, CompressionBlock, MessageRole, Tier,
    };

    fn user(id: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::User, MessageContentType::Text).with_text("go")
    }

    fn compress_call(id: &str, call_id: &str, text: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::Assistant, MessageContentType::ToolCall)
            .with_text(text)
            .with_tool(COMPRESS_TOOL_NAME, call_id)
    }

    fn compress_result(id: &str, call_id: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::Tool, MessageContentType::ToolResult)
            .with_text("Compressed: b1")
            .with_tool(COMPRESS_TOOL_NAME, call_id)
    }

    fn block_with_span(
        id: &str,
        call_id: Option<&str>,
        active: bool,
        span: Option<(&str, &str)>,
    ) -> CompressionBlock {
        let mut block = block(id, call_id, active);
        if let Some((start, end)) = span {
            block.start_ref = Some(start.to_string());
            block.end_ref = Some(end.to_string());
        }
        block
    }

    fn block(id: &str, call_id: Option<&str>, active: bool) -> CompressionBlock {
        CompressionBlock {
            block_id: id.to_string(),
            run_id: "r1".to_string(),
            tier: Tier::One,
            topic: None,
            start_ref: None,
            end_ref: None,
            summary: "s".to_string(),
            direct_message_ids: vec!["t1.0".to_string()],
            effective_message_ids: vec!["t1.0".to_string()],
            direct_block_ids: vec![],
            compressed_tokens: 10,
            created_at: 0,
            survived_count: 0,
            generation: BlockGeneration::Young,
            active,
            expanded: false,
            duration_ms: None,
            compress_call_id: call_id.map(str::to_string),
        }
    }

    fn args(pairs: &[(&str, &str)]) -> String {
        let content: Vec<String> = pairs
            .iter()
            .map(|(s, e)| format!("{{\"startId\":\"{s}\",\"endId\":\"{e}\",\"summary\":\"short\"}}"))
            .collect();
        format!("{{\"content\":[{}]}}", content.join(","))
    }

    #[test]
    fn an_active_calls_exchange_survives() {
        let mut state = create_initial_state();
        state.blocks.push(block("b1", Some("c1"), true));
        let msgs = vec![
            user("t1.0"),
            compress_call("t1.1", "c1", &args(&[("m00001", "m00002")])),
            compress_result("t1.2", "c1"),
        ];
        let out = hide_consumed_compress_calls(&state, &msgs);
        assert_eq!(out.hidden, 0);
        assert_eq!(out.messages.len(), 3);
    }

    #[test]
    fn an_inactive_calls_exchange_is_hidden() {
        // Its block was consumed or restored: the call describes a fold that
        // no longer exists.
        let mut state = create_initial_state();
        state.blocks.push(block("b1", Some("c1"), false));
        let msgs = vec![
            user("t1.0"),
            compress_call("t1.1", "c1", &args(&[("m00001", "m00002")])),
            compress_result("t1.2", "c1"),
        ];
        let out = hide_consumed_compress_calls(&state, &msgs);
        assert_eq!(out.hidden, 2, "call and its result");
        assert_eq!(out.messages.len(), 1);
    }

    #[test]
    fn the_newest_orphans_stay_visible() {
        // A failing call must remain observable, or a deterministic model
        // re-issues the same no-op forever.
        let state = create_initial_state();
        let msgs = vec![
            user("t1.0"),
            compress_call("t1.1", "c1", &args(&[("m00001", "m00001")])),
            compress_result("t1.2", "c1"),
            compress_call("t1.3", "c2", &args(&[("m00002", "m00002")])),
            compress_result("t1.4", "c2"),
        ];
        let out = hide_consumed_compress_calls(&state, &msgs);
        assert_eq!(out.hidden, 0, "both orphans are within the keep window");
        assert_eq!(out.messages.len(), 5);
    }

    #[test]
    fn older_orphans_are_hidden_so_the_residue_is_bounded() {
        let state = create_initial_state();
        let mut msgs = vec![user("t1.0")];
        for i in 0..5 {
            msgs.push(compress_call(&format!("t1.{}", i * 2 + 1), &format!("c{i}"), &args(&[("m00001", "m00001")])));
            msgs.push(compress_result(&format!("t1.{}", i * 2 + 2), &format!("c{i}")));
        }
        let out = hide_consumed_compress_calls(&state, &msgs);
        assert_eq!(out.hidden, 6, "three of five orphan pairs hidden");
        // two pairs plus the user message
        assert_eq!(out.messages.len(), 5);
    }

    #[test]
    fn a_call_without_a_call_id_is_hidden() {
        let state = create_initial_state();
        let mut call = compress_call("t1.1", "c1", &args(&[("m00001", "m00001")]));
        call.tool_call_id = None;
        let msgs = vec![user("t1.0"), call];
        let out = hide_consumed_compress_calls(&state, &msgs);
        assert_eq!(out.hidden, 1);
    }

    #[test]
    fn a_non_compress_call_is_untouched() {
        let state = create_initial_state();
        let read = CoreMessage::new("t1.1", MessageRole::Assistant, MessageContentType::ToolCall)
            .with_text("{}")
            .with_tool("read", "c1");
        let msgs = vec![user("t1.0"), read];
        let out = hide_consumed_compress_calls(&state, &msgs);
        assert_eq!(out.hidden, 0);
        assert_eq!(out.messages.len(), 2);
    }

    #[test]
    fn a_hidden_orphans_ref_is_reported() {
        // Boundary resolution snaps a range endpoint onto a hidden-but-intact
        // orphan through this list.
        let mut state = create_initial_state();
        let mut next = 1u64;
        let msgs = vec![
            user("t1.0"),
            compress_call("t1.1", "c1", &args(&[("m00001", "m00001")])),
            compress_call("t1.2", "c2", &args(&[("m00001", "m00001")])),
            compress_call("t1.3", "c3", &args(&[("m00001", "m00001")])),
        ];
        crate::agent::bigctx::assign_refs(&msgs, &mut state.message_refs, &mut next, |_| false).unwrap();
        let out = hide_consumed_compress_calls(&state, &msgs);
        assert!(!out.hidden_orphan_refs.is_empty(), "the oldest orphan's ref");
    }

    #[test]
    fn a_long_summary_in_a_live_call_is_stubbed() {
        let mut state = create_initial_state();
        state
            .blocks
            .push(block_with_span("b1", Some("c1"), true, Some(("m00001", "m00002"))));
        let long = "x".repeat(500);
        let text = format!(
            "{{\"content\":[{{\"startId\":\"m00001\",\"endId\":\"m00002\",\"summary\":\"{long}\"}}]}}"
        );
        let msgs = vec![user("t1.0"), compress_call("t1.1", "c1", &text)];
        let out = hide_consumed_compress_calls(&state, &msgs);
        let rewritten = out.messages[1].text_or_empty();
        assert!(
            rewritten.chars().count() < text.chars().count(),
            "the summary must be stubbed down"
        );
        assert!(rewritten.contains('…'), "and marked as a stub");
    }

    #[test]
    fn a_short_summary_is_left_alone() {
        let mut state = create_initial_state();
        state
            .blocks
            .push(block_with_span("b1", Some("c1"), true, Some(("m00001", "m00002"))));
        let text = args(&[("m00001", "m00002")]);
        let msgs = vec![user("t1.0"), compress_call("t1.1", "c1", &text)];
        let out = hide_consumed_compress_calls(&state, &msgs);
        assert_eq!(
            out.messages[1].text_or_empty(),
            text,
            "an unchanged call must stay byte-identical for the prefix cache"
        );
    }

    #[test]
    fn a_block_without_a_recorded_span_leaves_its_call_untouched() {
        // Without the refs the block cannot say which ranges are still live,
        // and a wrong rewrite is worse than a stale one.
        let mut state = create_initial_state();
        state.blocks.push(block("b1", Some("c1"), true));
        let text = args(&[("m00001", "m00002")]);
        let msgs = vec![user("t1.0"), compress_call("t1.1", "c1", &text)];
        let out = hide_consumed_compress_calls(&state, &msgs);
        assert_eq!(out.messages[1].text_or_empty(), text);
    }

    #[test]
    fn unparseable_call_text_is_left_alone() {
        let mut state = create_initial_state();
        state.blocks.push(block("b1", Some("c1"), true));
        let msgs = vec![user("t1.0"), compress_call("t1.1", "c1", "not json at all")];
        let out = hide_consumed_compress_calls(&state, &msgs);
        assert_eq!(out.messages[1].text_or_empty(), "not json at all");
    }

    #[test]
    fn a_result_whose_call_was_never_seen_is_untouched() {
        let state = create_initial_state();
        let msgs = vec![user("t1.0"), compress_result("t1.1", "unknown")];
        let out = hide_consumed_compress_calls(&state, &msgs);
        assert_eq!(out.hidden, 0);
    }
}
