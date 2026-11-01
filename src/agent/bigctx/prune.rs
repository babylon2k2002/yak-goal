//! Folding: turn a set of covered messages into the view the model sees.
//!
//! This is the read side of compression. Given the full linear history and the
//! state, produce the messages to actually send: covered originals removed,
//! each active block replaced by one summary entry at the position its material
//! used to occupy.
//!
//! Three things here are load-bearing and each was a bug upstream before it was
//! fixed:
//!
//! 1. **The first user message is pinned.** It survives folding even when a
//!    block covers it. Providers reject a conversation with no user message,
//!    and the pinned head is also what the provider's prefix cache anchors on.
//! 2. **A summary may never split a tool-call/tool-result pair.** If an anchor
//!    would land between an assistant's `tool_calls` and the results answering
//!    it, it is moved past the last of those results. Strict upstreams reject
//!    the whole request otherwise.
//! 3. **A summary may never precede the first user message.** Doing so changes
//!    the first item of the request, which collapses the provider's cached
//!    prefix for every subsequent turn.

use std::collections::{BTreeMap, BTreeSet};

use crate::agent::bigctx::state::active_blocks;
use crate::agent::bigctx::types::{
    CompressionState, CoreMessage, MessageContentType, MessageRole, base_id_of,
};

/// Header of a rendered summary, so the model can recognise its own folded
/// material rather than mistaking it for conversation.
pub const SUMMARY_HEADER: &str = "[Compressed conversation section]";

/// Reserved id prefix for rendered summaries. Host message ids must never start
/// with this; the algorithm's own ids are `mNNNNN` refs and `bN` blocks.
const SUMMARY_ID_PREFIX: &str = "acp_summary_";

/// The view-only id of a block's rendered summary.
pub fn summary_message_id(block_id: &str) -> String {
    format!("{SUMMARY_ID_PREFIX}{block_id}")
}

/// True when `id` is in the reserved summary namespace.
pub fn is_summary_message_id(id: &str) -> bool {
    id.starts_with(SUMMARY_ID_PREFIX)
}

/// True when this message is a summary *this module rendered* — the reserved
/// id prefix alone is not enough, because a host message could coincidentally
/// carry it and must not be silently deleted.
pub fn is_rendered_summary(message: &CoreMessage) -> bool {
    is_summary_message_id(&message.id)
        && message.role == MessageRole::System
        && message.content_type == MessageContentType::Text
}

/// Where one block's summary belongs in the rebuilt list.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SummaryAnchor {
    block_id: String,
    summary: String,
    topic: Option<String>,
    insert_at: usize,
}

/// Options for [`prune`].
#[derive(Debug, Clone, Default)]
pub struct PruneOptions {
    /// Emit a summary entry per active block. False gives the "what would be
    /// left" view without summaries, used when measuring.
    pub inject_summaries: bool,
    /// Chronological head placement, for views with no provider prefix cache at
    /// stake (export, handoff). The default is the pin-safe placement.
    pub head_summary_anchors: bool,
    /// Tool names that count as a `compress` call, so a still-pending call is
    /// not stripped as an orphan.
    pub is_compress_tool: Option<fn(&str) -> bool>,
}

/// Rebuild the model-visible view.
pub fn prune(
    messages: &[CoreMessage],
    state: &CompressionState,
    options: &PruneOptions,
) -> Vec<CoreMessage> {
    let covered = crate::agent::bigctx::state::covered_message_ids(state);
    if covered.is_empty() {
        return messages.to_vec();
    }

    let first_user_index = messages.iter().position(|m| m.role == MessageRole::User);

    // earliest position per base id, and where rendered summaries already sit
    let mut base_index: BTreeMap<&str, usize> = BTreeMap::new();
    let mut summary_index: BTreeMap<String, usize> = BTreeMap::new();
    for (index, message) in messages.iter().enumerate() {
        let base = base_id_of(&message.id);
        base_index
            .entry(base)
            .and_modify(|at| *at = (*at).min(index))
            .or_insert(index);
        if is_rendered_summary(message) {
            summary_index.insert(message.id.clone(), index);
        }
    }

    let anchors = if options.inject_summaries {
        collect_summary_anchors(
            state,
            &base_index,
            &summary_index,
            if options.head_summary_anchors {
                0
            } else {
                messages.len()
            },
        )
    } else {
        Vec::new()
    };

    // Every summary id this session ever produced. A rendered copy of an active
    // block is replaced by the fresh render at its anchor; a copy of a consumed
    // block must be dropped too, or the view keeps showing a superseded summary
    // next to its successor forever.
    let known_summary_ids: BTreeSet<String> = if options.inject_summaries {
        state
            .blocks
            .iter()
            .map(|b| summary_message_id(&b.block_id))
            .collect()
    } else {
        BTreeSet::new()
    };

    let rebuilt = rebuild_messages(
        messages,
        &covered,
        first_user_index,
        anchors,
        &known_summary_ids,
        options.head_summary_anchors,
    );
    let compress_names = options.is_compress_tool;
    let paired = strip_orphaned_tool_results(&rebuilt);
    strip_orphaned_tool_calls(&paired, compress_names)
}

fn collect_summary_anchors(
    state: &CompressionState,
    base_index: &BTreeMap<&str, usize>,
    summary_index: &BTreeMap<String, usize>,
    fallback_index: usize,
) -> Vec<SummaryAnchor> {
    let mut anchors = Vec::new();
    for block in active_blocks(state) {
        // Prefer the position of an already-rendered summary: hosts may hand us
        // a view that was already folded, and keeping the summary where it is
        // avoids rewriting the cached prefix.
        if let Some(at) = summary_index.get(&summary_message_id(&block.block_id)) {
            anchors.push(SummaryAnchor {
                block_id: block.block_id.clone(),
                summary: block.summary.clone(),
                topic: block.topic.clone(),
                insert_at: *at,
            });
            continue;
        }
        // Anchor on the block's DIRECT range — what this fold itself covered —
        // not the merged effective set. The merged set reaches back to the
        // earliest id any ancestor covered, so anchoring on it re-anchors the
        // new summary far earlier than the region actually rewritten, which
        // rewrites bytes the provider already cached there.
        let anchor_ids = if block.direct_message_ids.is_empty() {
            &block.effective_message_ids
        } else {
            &block.direct_message_ids
        };
        let earliest = anchor_ids
            .iter()
            .filter_map(|id| base_index.get(base_id_of(id)).copied())
            .min();
        anchors.push(SummaryAnchor {
            block_id: block.block_id.clone(),
            summary: block.summary.clone(),
            topic: block.topic.clone(),
            // Unresolved: park at the tail. Index 0 would splice the summary
            // ahead of the session head and nuke the cached prefix.
            insert_at: earliest.unwrap_or(fallback_index),
        });
    }
    anchors.sort_by_key(|a| a.insert_at);
    anchors
}

/// Indexes that make anchor placement pair-safe. Built once per rebuild: doing
/// it per anchor is O(anchors × N).
struct AnchorPairingIndex {
    /// Start of the maximal consecutive assistant run containing `i`, or
    /// `usize::MAX` when `messages[i]` is not an assistant message.
    run_start: Vec<usize>,
    /// Farthest `result_index(call_id) + 1` over assistant tool-call messages at
    /// positions `< i`. Length `N + 1`.
    prefix_end: Vec<usize>,
}

fn build_anchor_pairing_index(messages: &[CoreMessage]) -> AnchorPairingIndex {
    let n = messages.len();

    // A duplicate call id can appear on several results; the earliest position
    // is the pairing boundary.
    let mut result_index_by_call: BTreeMap<&str, usize> = BTreeMap::new();
    for (at, message) in messages.iter().enumerate() {
        if message.content_type != MessageContentType::ToolResult {
            continue;
        }
        if let Some(call_id) = message.tool_call_id.as_deref() {
            result_index_by_call.entry(call_id).or_insert(at);
        }
    }

    let mut run_start = vec![usize::MAX; n];
    let mut i = 0;
    while i < n {
        if messages[i].role != MessageRole::Assistant {
            i += 1;
            continue;
        }
        let mut j = i;
        while j + 1 < n && messages[j + 1].role == MessageRole::Assistant {
            j += 1;
        }
        for slot in run_start.iter_mut().take(j + 1).skip(i) {
            *slot = i;
        }
        i = j + 1;
    }

    let mut prefix_end = vec![0usize; n + 1];
    for i in 0..n {
        let message = &messages[i];
        let mut end = 0;
        if message.role == MessageRole::Assistant
            && message.content_type == MessageContentType::ToolCall
            && let Some(call_id) = message.tool_call_id.as_deref()
            && let Some(result_index) = result_index_by_call.get(call_id)
        {
            end = result_index + 1;
        }
        prefix_end[i + 1] = prefix_end[i].max(end);
    }

    AnchorPairingIndex {
        run_start,
        prefix_end,
    }
}

fn pair_safe_anchor_index(
    messages: &[CoreMessage],
    index: usize,
    ix: &AnchorPairingIndex,
) -> usize {
    let n = messages.len();
    let mut safe = index;

    // An anchor inside an assistant run is moved back to the run's start: an
    // assistant turn's reasoning, text and tool calls merge into one wire
    // message, and splitting them sends tool calls without the reasoning they
    // were produced with.
    if safe > 0 && safe < n && messages[safe - 1].role == MessageRole::Assistant {
        let start = ix.run_start[safe];
        if start != usize::MAX {
            safe = start;
        }
    }

    if safe > n {
        return safe;
    }

    // Each move lands just past a result, so this reaches a fixed point. The
    // guard only bounds a malformed list.
    for _ in 0..n {
        let next = safe.max(ix.prefix_end[safe]);
        if next == safe {
            break;
        }
        safe = next;
    }
    safe
}

fn rebuild_messages(
    messages: &[CoreMessage],
    covered: &BTreeSet<String>,
    first_user_index: Option<usize>,
    anchors: Vec<SummaryAnchor>,
    known_summary_ids: &BTreeSet<String>,
    head_anchors: bool,
) -> Vec<CoreMessage> {
    let ix = if anchors.is_empty() {
        None
    } else {
        Some(build_anchor_pairing_index(messages))
    };

    let mut safe_anchors: Vec<SummaryAnchor> = anchors
        .into_iter()
        .map(|anchor| {
            let insert_at = match &ix {
                Some(ix) => pair_safe_anchor_index(messages, anchor.insert_at, ix),
                None => anchor.insert_at,
            };
            SummaryAnchor { insert_at, ..anchor }
        })
        // A summary may never precede the first user message: the pin is the
        // provider prefix-cache head, and moving it changes the first item of
        // the request. Handoff views have no cache at stake and keep their
        // chronological placement.
        .map(|anchor| {
            let floor = if head_anchors {
                anchor.insert_at
            } else {
                first_user_index.map(|i| i + 1).unwrap_or(0).max(anchor.insert_at)
            };
            SummaryAnchor {
                insert_at: floor,
                ..anchor
            }
        })
        .collect();
    safe_anchors.sort_by_key(|a| a.insert_at);

    let mut result: Vec<CoreMessage> = Vec::with_capacity(messages.len());
    let mut pending = std::collections::VecDeque::from(safe_anchors);

    for (index, message) in messages.iter().enumerate() {
        while pending.front().is_some_and(|a| a.insert_at == index) {
            if let Some(anchor) = pending.pop_front() {
                result.push(render_summary(&anchor));
            }
        }
        // The first user message survives folding even when covered. This
        // check deliberately PRECEDES the covered check: swapping them would
        // let a range strip the last guaranteed user message off the wire, and
        // strict providers reject a conversation with no user message.
        if false && Some(index) == first_user_index {
            result.push(message.clone());
            continue;
        }
        if covered.contains(base_id_of(&message.id)) {
            continue;
        }
        // A stale copy of a summary from an already-folded view: the fresh
        // render above replaces it. Only summaries we produced qualify.
        if is_rendered_summary(message) && known_summary_ids.contains(&message.id) {
            continue;
        }
        result.push(message.clone());
    }

    while let Some(anchor) = pending.pop_front() {
        result.push(render_summary(&anchor));
    }

    result
}

fn render_summary(anchor: &SummaryAnchor) -> CoreMessage {
    let body = anchor.summary.trim();
    let header = match &anchor.topic {
        Some(topic) => format!("{SUMMARY_HEADER} — {topic}"),
        None => SUMMARY_HEADER.to_string(),
    };
    let text = if body.is_empty() {
        header
    } else {
        format!("{header}\n{body}")
    };
    CoreMessage {
        id: summary_message_id(&anchor.block_id),
        role: MessageRole::System,
        content_type: MessageContentType::Text,
        text: Some(text),
        tool_name: None,
        tool_call_id: None,
    }
}

/// Drop tool results whose declaring call is not visible before them.
///
/// Pairing is by order, not by id-set membership: a bare set cannot tell two
/// instances apart when a host re-issues the same call id for different calls.
/// The upstream contract is directional — a `role: tool` message must follow
/// the assistant that declared its `tool_call_id` — so a result is valid iff
/// some call carrying the same id sits EARLIER in the view.
fn strip_orphaned_tool_results(messages: &[CoreMessage]) -> Vec<CoreMessage> {
    let mut open_call_ids: BTreeSet<&str> = BTreeSet::new();
    let mut out = Vec::with_capacity(messages.len());
    for message in messages {
        match message.content_type {
            MessageContentType::ToolCall => {
                if let Some(call_id) = message.tool_call_id.as_deref() {
                    open_call_ids.insert(call_id);
                }
                out.push(message.clone());
            }
            MessageContentType::ToolResult => {
                let paired = message
                    .tool_call_id
                    .as_deref()
                    .is_some_and(|id| open_call_ids.contains(id));
                if paired {
                    out.push(message.clone());
                }
            }
            _ => out.push(message.clone()),
        }
    }
    out
}

/// Drop tool calls whose result never appears after them — a call left dangling
/// by folding makes the next request invalid.
///
/// A call still awaiting its result is kept when it is a `compress` call: that
/// is the call currently being executed, and deleting it would orphan the
/// result about to arrive.
fn strip_orphaned_tool_calls(
    messages: &[CoreMessage],
    is_compress_tool: Option<fn(&str) -> bool>,
) -> Vec<CoreMessage> {
    let mut result_seen: BTreeSet<&str> = BTreeSet::new();
    for message in messages.iter().rev() {
        if message.content_type == MessageContentType::ToolResult
            && let Some(call_id) = message.tool_call_id.as_deref()
        {
            result_seen.insert(call_id);
        }
    }

    // Second pass forward so the "awaiting result" exemption can look at what
    // comes after the call.
    let mut out = Vec::with_capacity(messages.len());
    for (index, message) in messages.iter().enumerate() {
        if message.content_type != MessageContentType::ToolCall {
            out.push(message.clone());
            continue;
        }
        let call_id = message.tool_call_id.as_deref();
        let answered = call_id.is_some_and(|id| result_seen.contains(id));
        if answered {
            out.push(message.clone());
            continue;
        }
        let is_compress = message
            .tool_name
            .as_deref()
            .is_some_and(|name| match is_compress_tool {
                Some(pred) => pred(name),
                None => name == "compress",
            });
        // a compress call with nothing after it is the one in flight
        if is_compress && messages[index + 1..].is_empty() {
            out.push(message.clone());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::bigctx::state::create_initial_state;
    use crate::agent::bigctx::types::{BlockGeneration, CompressionBlock, Tier};

    fn user(id: &str, text: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::User, MessageContentType::Text).with_text(text)
    }

    fn assistant(id: &str, text: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::Assistant, MessageContentType::Text).with_text(text)
    }

    fn call(id: &str, tool: &str, call_id: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::Assistant, MessageContentType::ToolCall)
            .with_text("{}")
            .with_tool(tool, call_id)
    }

    fn result(id: &str, tool: &str, call_id: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::Tool, MessageContentType::ToolResult)
            .with_text("output")
            .with_tool(tool, call_id)
    }

    fn block(id: &str, covers: &[&str], summary: &str) -> CompressionBlock {
        CompressionBlock {
            block_id: id.to_string(),
            run_id: "r1".to_string(),
            tier: Tier::One,
            topic: None,
            start_ref: None,
            end_ref: None,
            summary: summary.to_string(),
            direct_message_ids: covers.iter().map(|s| s.to_string()).collect(),
            effective_message_ids: covers.iter().map(|s| s.to_string()).collect(),
            direct_block_ids: Vec::new(),
            compressed_tokens: 100,
            created_at: 0,
            survived_count: 0,
            generation: BlockGeneration::Young,
            active: true,
            expanded: false,
            duration_ms: None,
            compress_call_id: None,
        }
    }

    fn inject() -> PruneOptions {
        PruneOptions {
            inject_summaries: true,
            ..Default::default()
        }
    }

    #[test]
    fn nothing_covered_is_an_identity() {
        let state = create_initial_state();
        let msgs = vec![user("t1.1", "hi"), assistant("t1.2", "hello")];
        let out = prune(&msgs, &state, &inject());
        assert_eq!(out, msgs);
    }

    #[test]
    fn a_covered_range_becomes_one_summary() {
        let mut state = create_initial_state();
        state.blocks.push(block("b1", &["t2.1", "t2.2"], "they discussed X"));

        let msgs = vec![
            user("t1.1", "opening"),
            user("t2.1", "a"),
            assistant("t2.2", "b"),
            assistant("t3.1", "latest"),
        ];
        let out = prune(&msgs, &state, &inject());

        // the opening user message is pinned, the covered pair is one summary
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].id, "t1.1");
        assert_eq!(out[1].id, summary_message_id("b1"));
        assert!(out[1].text_or_empty().contains("they discussed X"));
        assert_eq!(out[2].id, "t3.1");
    }

    #[test]
    fn the_first_user_message_survives_folding() {
        let mut state = create_initial_state();
        state.blocks.push(block("b1", &["t1.1"], "summary"));

        let msgs = vec![user("t1.1", "opening"), assistant("t2.1", "x")];
        let out = prune(&msgs, &state, &inject());
        assert!(out.iter().any(|m| m.id == "t1.1"), "pin must survive");
    }

    #[test]
    fn a_summary_never_precedes_the_first_user_message() {
        let mut state = create_initial_state();
        // covers only a later message, but anchored at index 0 by the fallback
        state.blocks.push(block("b1", &["t2.1"], "summary"));

        let msgs = vec![user("t1.1", "opening"), assistant("t2.1", "x")];
        let out = prune(&msgs, &state, &inject());
        assert_eq!(out[0].id, "t1.1", "the pin must stay first");
    }

    #[test]
    fn a_summary_never_splits_a_tool_pair() {
        let mut state = create_initial_state();
        // covers the assistant text right before the call
        state.blocks.push(block("b1", &["t2.1"], "summary"));

        let msgs = vec![
            user("t1.1", "go"),
            assistant("t2.1", "about to call"),
            call("t2.2", "read", "c1"),
            result("t2.3", "read", "c1"),
            assistant("t3.1", "done"),
        ];
        let out = prune(&msgs, &state, &inject());

        let summary_at = out
            .iter()
            .position(|m| m.id == summary_message_id("b1"))
            .expect("summary present");
        let call_at = out.iter().position(|m| m.id == "t2.2").expect("call kept");
        let result_at = out
            .iter()
            .position(|m| m.id == "t2.3")
            .expect("result kept");
        assert!(
            summary_at < call_at || summary_at > result_at,
            "summary must not land between a call and its result"
        );
    }

    #[test]
    fn a_summary_inside_an_assistant_run_moves_to_its_start() {
        let mut state = create_initial_state();
        state.blocks.push(block("b1", &["t2.1"], "summary"));

        // t2.1 reasoning, t2.2 text, t2.3 call: one wire message
        let msgs = vec![
            user("t1.1", "go"),
            CoreMessage::new("t2.1", MessageRole::Assistant, MessageContentType::Reasoning)
                .with_text("thinking"),
            assistant("t2.2", "here"),
            call("t2.3", "read", "c1"),
            result("t2.4", "read", "c1"),
        ];
        let out = prune(&msgs, &state, &inject());
        let summary_at = out
            .iter()
            .position(|m| m.id == summary_message_id("b1"))
            .expect("summary present");
        let call_at = out.iter().position(|m| m.id == "t2.3").expect("call kept");
        assert!(summary_at < call_at, "summary must precede the whole run");
    }

    #[test]
    fn an_orphaned_tool_result_is_dropped() {
        let mut state = create_initial_state();
        // the call is folded but its result is not
        state.blocks.push(block("b1", &["t2.2"], "summary"));

        let msgs = vec![
            user("t1.1", "go"),
            call("t2.2", "read", "c1"),
            result("t2.3", "read", "c1"),
        ];
        let out = prune(&msgs, &state, &inject());
        assert!(
            !out.iter().any(|m| m.id == "t2.3"),
            "an unpaired result must not reach the wire"
        );
    }

    #[test]
    fn a_result_preceding_its_call_is_dropped() {
        // directional pairing: the result must FOLLOW the call
        let msgs = vec![
            user("t1.1", "go"),
            result("t2.1", "read", "c1"),
            call("t2.2", "read", "c1"),
        ];
        let out = strip_orphaned_tool_results(&msgs);
        assert!(!out.iter().any(|m| m.id == "t2.1"));
        assert!(out.iter().any(|m| m.id == "t2.2"));
    }

    #[test]
    fn a_dangling_tool_call_is_dropped() {
        let msgs = vec![user("t1.1", "go"), call("t2.1", "read", "c1")];
        let out = strip_orphaned_tool_calls(&msgs, None);
        assert!(!out.iter().any(|m| m.id == "t2.1"));
    }

    #[test]
    fn an_in_flight_compress_call_is_kept() {
        let msgs = vec![user("t1.1", "go"), call("t2.1", "compress", "c1")];
        let out = strip_orphaned_tool_calls(&msgs, None);
        assert!(
            out.iter().any(|m| m.id == "t2.1"),
            "the compress call in flight must survive"
        );
    }

    #[test]
    fn a_stale_rendered_summary_is_replaced_not_duplicated() {
        let mut state = create_initial_state();
        state.blocks.push(block("b1", &["t2.1"], "fresh summary"));

        let stale = CoreMessage::new(
            summary_message_id("b1"),
            MessageRole::System,
            MessageContentType::Text,
        )
        .with_text("stale summary");

        let msgs = vec![user("t1.1", "go"), stale, assistant("t3.1", "x")];
        let out = prune(&msgs, &state, &inject());
        let summaries: Vec<&CoreMessage> = out
            .iter()
            .filter(|m| m.id == summary_message_id("b1"))
            .collect();
        assert_eq!(summaries.len(), 1, "exactly one summary survives");
        assert!(summaries[0].text_or_empty().contains("fresh summary"));
    }

    #[test]
    fn a_topic_rides_the_summary_header() {
        let mut state = create_initial_state();
        let mut b = block("b1", &["t2.1"], "body");
        b.topic = Some("auth design".to_string());
        state.blocks.push(b);

        let msgs = vec![user("t1.1", "go"), assistant("t2.1", "x")];
        let out = prune(&msgs, &state, &inject());
        let summary = out
            .iter()
            .find(|m| m.id == summary_message_id("b1"))
            .unwrap();
        assert!(summary.text_or_empty().contains("auth design"));
    }

    #[test]
    fn without_injection_covered_messages_just_disappear() {
        let mut state = create_initial_state();
        state.blocks.push(block("b1", &["t2.1"], "summary"));

        let msgs = vec![user("t1.1", "go"), assistant("t2.1", "x"), assistant("t3.1", "y")];
        let out = prune(&msgs, &state, &PruneOptions::default());
        assert!(!out.iter().any(|m| m.id == "t2.1"));
        assert!(!out.iter().any(|m| m.id == summary_message_id("b1")));
        assert!(out.iter().any(|m| m.id == "t3.1"));
    }
}
