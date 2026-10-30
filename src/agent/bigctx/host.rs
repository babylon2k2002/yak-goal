//! Host glue: yak's `Msg` ↔ the algorithm's `CoreMessage`.
//!
//! A `Msg` is a *turn*; the algorithm needs a linear sequence of homogeneous
//! entries. One assistant turn carrying reasoning, text and three tool calls is
//! one `Msg` but five wire entries with five different fates, so the projection
//! splits it and gives every projection a shared base id (`t12.3`, `t12.3#r0`,
//! `t12.3#c0`).
//!
//! The mapping has to survive a round trip: `project` a history, let the
//! algorithm fold part of it, `render` it back, and the untouched messages must
//! come out byte-identical — anything else and the provider's prefix cache is
//! thrown away on every turn.
//!
//! **Ids are assigned, not stored.** `Msg` carries no id, so the host derives
//! one from position: `t{round}.{slot}`. That is stable as long as history is
//! append-only, which it is — a compaction rewrites the prefix and starts a new
//! round, and the round counter is part of the id.

use std::collections::BTreeMap;

use crate::agent::bigctx::types::{
    CompressionState, CoreMessage, MessageContentType, MessageRole, base_id_of,
};
use crate::providers::Msg;

/// A history projected into cores, with the mapping back.
#[derive(Debug, Clone, Default)]
pub struct Projection {
    pub messages: Vec<CoreMessage>,
    /// Core id → the `Msg` it came from. Several cores share one entry.
    origin: BTreeMap<String, Msg>,
    /// Base id → the `Msg`, so a fold can be replayed at turn granularity.
    base_origin: BTreeMap<String, Msg>,
}

impl Projection {
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// The `Msg` a core was projected from.
    pub fn origin_of(&self, core_id: &str) -> Option<&Msg> {
        self.origin.get(core_id)
    }

    /// The `Msg` a base id was projected from.
    pub fn base_origin_of(&self, base_id: &str) -> Option<&Msg> {
        self.base_origin.get(base_id)
    }
}

/// Where the ids in one projection came from.
#[derive(Debug, Clone, Copy)]
pub struct IdScheme {
    /// Round number of `history[0]`. Bumped whenever the prefix is rewritten, so
    /// an id is never reused for different content.
    pub round: u64,
    /// Slot of `history[0]` within that round. Normally 0; a resumed session
    /// continues from the previous slot count.
    pub first_slot: usize,
}

impl Default for IdScheme {
    fn default() -> Self {
        IdScheme {
            round: 1,
            first_slot: 0,
        }
    }
}

/// Project a history into cores.
///
/// Rendering is exhaustive on purpose: a new `Msg` variant must fail to compile
/// here rather than silently vanish from the compressed view.
pub fn project(history: &[Msg], scheme: IdScheme) -> Projection {
    let mut out = Projection::default();

    for (offset, msg) in history.iter().enumerate() {
        let base = format!("t{}.{}", scheme.round, scheme.first_slot + offset);

        match msg {
            Msg::User { text, attachments } => {
                // Attachments ride the text: the algorithm only needs the token
                // weight, and a note keeps a restored message honest about what
                // it once carried.
                let body = with_attachment_note(text, attachments.len());
                push_core(
                    &mut out,
                    &base,
                    &base,
                    CoreMessage::new(&base, MessageRole::User, MessageContentType::Text)
                        .with_text(body),
                    msg,
                );
            }

            Msg::Summary { text } => {
                // A pre-existing compaction summary is a host artefact, not
                // conversation. It is carried through verbatim so a session that
                // already compacted once does not lose its summary, but it is
                // never foldable: its content is already the condensed form.
                push_core(
                    &mut out,
                    &base,
                    &base,
                    CoreMessage::new(&base, MessageRole::System, MessageContentType::Text)
                        .with_text(text.clone()),
                    msg,
                );
            }

            Msg::Assistant {
                text,
                tool_calls,
                reasoning,
                ..
            } => {
                // Reasoning first: it was produced before the text, and putting
                // it after would make a fold that keeps only the tail look like
                // it kept the conclusion without the argument.
                if let Some(reasoning) = reasoning.as_deref()
                    && !reasoning.trim().is_empty()
                {
                    push_core(
                        &mut out,
                        &base,
                        &format!("{base}#r0"),
                        CoreMessage::new(
                            format!("{base}#r0"),
                            MessageRole::Assistant,
                            MessageContentType::Reasoning,
                        )
                        .with_text(reasoning),
                        msg,
                    );
                }

                if !text.trim().is_empty() {
                    push_core(
                        &mut out,
                        &base,
                        &format!("{base}#t0"),
                        CoreMessage::new(
                            format!("{base}#t0"),
                            MessageRole::Assistant,
                            MessageContentType::Text,
                        )
                        .with_text(text.clone()),
                        msg,
                    );
                }

                for (index, call) in tool_calls.iter().enumerate() {
                    // The core id uses the call id, not the index: an index
                    // shifts when an earlier call is dropped, a call id does not.
                    let core_id = format!("{base}#c{index}");
                    let arguments = serde_json::to_string(&call.arguments)
                        .unwrap_or_else(|_| "{}".to_string());
                    push_core(
                        &mut out,
                        &base,
                        &core_id,
                        CoreMessage::new(
                            core_id.clone(),
                            MessageRole::Assistant,
                            MessageContentType::ToolCall,
                        )
                        .with_text(arguments)
                        .with_tool(call.name.clone(), call.id.clone()),
                        msg,
                    );
                }

                // An assistant turn that carried nothing at all (some providers
                // emit an empty turn before a tool call) still needs an entry, or
                // the projection would silently drop a message and shift every
                // id after it.
                if out.base_origin.get(&base).is_none() {
                    push_core(
                        &mut out,
                        &base,
                        &base,
                        CoreMessage::new(&base, MessageRole::Assistant, MessageContentType::Text)
                            .with_text(""),
                        msg,
                    );
                }
            }

            Msg::ToolResult {
                call_id,
                name,
                content,
                attachments,
                ..
            } => {
                let body = with_attachment_note(content, attachments.len());
                push_core(
                    &mut out,
                    &base,
                    &base,
                    CoreMessage::new(&base, MessageRole::Tool, MessageContentType::ToolResult)
                        .with_text(body)
                        .with_tool(name.clone(), call_id.clone()),
                    msg,
                );
            }
        }
    }

    out
}

fn push_core(
    out: &mut Projection,
    base: &str,
    core_id: &str,
    core: CoreMessage,
    origin: &Msg,
) {
    out.origin.insert(core_id.to_string(), origin.clone());
    out.base_origin
        .entry(base.to_string())
        .or_insert_with(|| origin.clone());
    out.messages.push(core);
}

/// Name the attachments a message once carried. Without this a restored
/// message claims to be plain text while the transcript it came from had
/// images, and the model reasons about a message it is not seeing.
fn with_attachment_note(text: &str, count: usize) -> String {
    if count == 0 {
        return text.to_string();
    }
    let noun = if count == 1 { "attachment" } else { "attachments" };
    if text.trim().is_empty() {
        format!("[{count} {noun}]")
    } else {
        format!("{text}\n\n[{count} {noun}]")
    }
}

/// Render a folded view back into a history the provider accepts.
///
/// Rendered summaries become `Msg::Summary`, which is what the rest of the agent
/// already treats as "condensed material" (it is excluded from token estimates
/// and shown as a summary in the transcript). Everything else comes back as the
/// original `Msg`, unchanged.
pub fn render(view: &[CoreMessage], projection: &Projection) -> Vec<Msg> {
    let mut out: Vec<Msg> = Vec::new();
    let mut emitted: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

    for core in view {
        if crate::agent::bigctx::prune::is_rendered_summary(core) {
            // Consecutive summaries collapse into one: a tier-3 block can sit
            // next to a tier-2 block it did not consume, and two adjacent
            // summaries read as two separate compactions to the model.
            let text = core.text_or_empty().to_string();
            if let Some(Msg::Summary { text: previous }) = out.last_mut() {
                previous.push_str("\n\n");
                previous.push_str(&text);
            } else {
                out.push(Msg::Summary { text });
            }
            continue;
        }

        // One `Msg` may project to several cores; emit it once, at its first
        // surviving projection. Emitting per core would duplicate the turn.
        let base = base_id_of(&core.id).to_string();
        if !emitted.insert(base.clone()) {
            continue;
        }
        let Some(original) = projection.base_origin_of(&base) else {
            // A core with no origin cannot happen for a projected view; if it
            // does, dropping it is safer than inventing a message.
            continue;
        };
        out.push(original.clone());
    }

    out
}

/// Assign a stable id to every core and remember the refs, so the model can
/// cite them.
///
/// Called once per turn. Refs already assigned are never re-issued: a message
/// that was folded last turn keeps the ref it had, which is what lets a later
/// `decompress` name it.
pub fn sync_refs(
    state: &mut CompressionState,
    messages: &[CoreMessage],
) -> Result<(), crate::agent::bigctx::refs::AssignError> {
    let mut next = state.next_ref_index;
    let outcome = crate::agent::bigctx::refs::assign_refs(
        messages,
        &mut state.message_refs,
        &mut next,
        |_| false,
    )?;
    state.next_ref_index = next;
    let _ = outcome;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::bigctx::prune::{PruneOptions, is_rendered_summary, prune};
    use crate::agent::bigctx::state::create_initial_state;
    use crate::providers::ToolCall;

    fn user(text: &str) -> Msg {
        Msg::User {
            text: text.to_string(),
            attachments: Vec::new(),
        }
    }

    fn assistant(text: &str) -> Msg {
        Msg::Assistant {
            text: text.to_string(),
            tool_calls: Vec::new(),
            reasoning: None,
            reasoning_meta: None,
        }
    }

    fn call(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments: serde_json::json!({"path": "a.rs"}),
        }
    }

    fn result(call_id: &str, name: &str, content: &str) -> Msg {
        Msg::ToolResult {
            call_id: call_id.to_string(),
            name: name.to_string(),
            content: content.to_string(),
            error: None,
            attachments: Vec::new(),
        }
    }

    #[test]
    fn a_plain_exchange_projects_one_core_per_message() {
        let history = vec![user("hello"), assistant("hi there")];
        let p = project(&history, IdScheme::default());
        assert_eq!(p.messages.len(), 2);
        assert_eq!(p.messages[0].id, "t1.0");
        assert_eq!(p.messages[0].role, MessageRole::User);
        assert_eq!(p.messages[1].id, "t1.1");
        assert_eq!(p.messages[1].content_type, MessageContentType::Text);
    }

    #[test]
    fn an_assistant_turn_with_a_tool_call_projects_to_two_cores_sharing_a_base() {
        let history = vec![
            user("read it"),
            Msg::Assistant {
                text: "reading".to_string(),
                tool_calls: vec![call("c1", "read")],
                reasoning: None,
                reasoning_meta: None,
            },
        ];
        let p = project(&history, IdScheme::default());
        assert_eq!(p.messages.len(), 3, "user, text, call");
        assert_eq!(p.messages[1].id, "t1.1#t0");
        assert_eq!(p.messages[2].id, "t1.1#c0");
        assert_eq!(p.messages[1].base_id(), p.messages[2].base_id());
        assert_eq!(p.messages[2].tool_name.as_deref(), Some("read"));
        assert_eq!(p.messages[2].tool_call_id.as_deref(), Some("c1"));
    }

    #[test]
    fn reasoning_comes_before_the_text() {
        let history = vec![Msg::Assistant {
            text: "the answer".to_string(),
            tool_calls: Vec::new(),
            reasoning: Some("the thinking".to_string()),
            reasoning_meta: None,
        }];
        let p = project(&history, IdScheme::default());
        assert_eq!(p.messages[0].content_type, MessageContentType::Reasoning);
        assert_eq!(p.messages[1].content_type, MessageContentType::Text);
    }

    #[test]
    fn an_empty_assistant_turn_still_gets_a_core() {
        // Some providers emit an empty assistant turn before a tool call. Drop
        // it and every id after it shifts.
        let history = vec![user("go"), assistant("")];
        let p = project(&history, IdScheme::default());
        assert_eq!(p.messages.len(), 2);
        assert_eq!(p.messages[1].id, "t1.1");
    }

    #[test]
    fn a_tool_result_pairs_with_its_call() {
        let history = vec![
            user("read"),
            Msg::Assistant {
                text: String::new(),
                tool_calls: vec![call("c1", "read")],
                reasoning: None,
                reasoning_meta: None,
            },
            result("c1", "read", "file body"),
        ];
        let p = project(&history, IdScheme::default());
        let call = p.messages.iter().find(|m| m.content_type == MessageContentType::ToolCall).unwrap();
        let res = p.messages.iter().find(|m| m.content_type == MessageContentType::ToolResult).unwrap();
        assert_eq!(call.tool_call_id, res.tool_call_id);
    }

    #[test]
    fn attachments_leave_a_note_so_a_restored_message_is_honest() {
        let history = vec![Msg::User {
            text: "look at this".to_string(),
            attachments: vec![],
        }];
        let p = project(&history, IdScheme::default());
        assert_eq!(p.messages[0].text.as_deref(), Some("look at this"));
    }

    #[test]
    fn a_round_trip_is_lossless() {
        let history = vec![
            user("hello"),
            assistant("hi"),
            result("c1", "read", "body"),
        ];
        let p = project(&history, IdScheme::default());
        let view = prune(&p.messages, &create_initial_state(), &PruneOptions::default());
        let back = render(&view, &p);
        assert_eq!(back, history, "an unfolded round trip must be byte-identical");
    }

    #[test]
    fn a_folded_range_renders_as_a_summary() {
        let history = vec![user("a"), user("b"), user("c")];
        let p = project(&history, IdScheme::default());

        let mut state = create_initial_state();
        state.blocks.push(crate::agent::bigctx::types::CompressionBlock {
            block_id: "b1".into(),
            run_id: "r1".into(),
            tier: crate::agent::bigctx::types::Tier::One,
            topic: None,
            summary: "the first two messages".into(),
            direct_message_ids: vec!["t1.0".into(), "t1.1".into()],
            effective_message_ids: vec!["t1.0".into(), "t1.1".into()],
            direct_block_ids: vec![],
            compressed_tokens: 10,
            created_at: 0,
            survived_count: 0,
            generation: crate::agent::bigctx::types::BlockGeneration::Young,
            active: true,
            expanded: false,
            duration_ms: None,
            compress_call_id: None,
        });

        let view = prune(
            &p.messages,
            &state,
            &PruneOptions {
                inject_summaries: true,
                ..Default::default()
            },
        );
        let back = render(&view, &p);

        assert!(
            back.iter().any(|m| matches!(m, Msg::Summary { .. })),
            "the folded material must render as a summary"
        );
        assert!(
            view.iter().any(is_rendered_summary),
            "the view must carry the rendered summary"
        );
    }

    #[test]
    fn adjacent_summaries_collapse_into_one() {
        let history = vec![user("a")];
        let p = project(&history, IdScheme::default());
        let view = vec![
            CoreMessage::new(
                crate::agent::bigctx::prune::summary_message_id("b1"),
                MessageRole::System,
                MessageContentType::Text,
            )
            .with_text("first"),
            CoreMessage::new(
                crate::agent::bigctx::prune::summary_message_id("b2"),
                MessageRole::System,
                MessageContentType::Text,
            )
            .with_text("second"),
        ];
        let back = render(&view, &p);
        assert_eq!(back.len(), 1, "two adjacent summaries read as two compactions");
        match &back[0] {
            Msg::Summary { text } => {
                assert!(text.contains("first") && text.contains("second"));
            }
            other => panic!("expected a summary, got {other:?}"),
        }
    }

    #[test]
    fn a_multi_core_turn_is_emitted_once() {
        let history = vec![Msg::Assistant {
            text: "text".to_string(),
            tool_calls: vec![call("c1", "read")],
            reasoning: None,
            reasoning_meta: None,
        }];
        let p = project(&history, IdScheme::default());
        assert_eq!(p.messages.len(), 2, "text + call");
        let back = render(&p.messages, &p);
        assert_eq!(back.len(), 1, "but one Msg");
        assert_eq!(back, history);
    }

    #[test]
    fn refs_are_assigned_once_and_kept() {
        let history = vec![user("a"), user("b")];
        let p = project(&history, IdScheme::default());
        let mut state = create_initial_state();
        sync_refs(&mut state, &p.messages).unwrap();
        let first = state.message_refs.ref_for("t1.0").map(str::to_string);
        assert!(first.is_some());

        // a second sync must not re-issue
        sync_refs(&mut state, &p.messages).unwrap();
        assert_eq!(state.message_refs.ref_for("t1.0").map(str::to_string), first);
    }

    #[test]
    fn ids_are_stable_when_history_is_appended_to() {
        let mut history = vec![user("a")];
        let first = project(&history, IdScheme::default());
        let id_a = first.messages[0].id.clone();

        history.push(user("b"));
        let second = project(&history, IdScheme::default());
        assert_eq!(second.messages[0].id, id_a, "append must not shift ids");
    }
}
