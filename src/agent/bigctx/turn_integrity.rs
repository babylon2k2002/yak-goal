//! Atomic turns: what a fold is not allowed to split.
//!
//! A turn is a reasoning run, the assistant text/tool-call burst that follows
//! it, and every tool result paired to that burst's calls. Two rules make the
//! group atomic:
//!
//! - **Strict-echo reasoning providers** (DeepSeek: "The `reasoning_content` in
//!   the thinking mode must be passed back to the API") reject a request whose
//!   assistant tool-call turn survives without its reasoning.
//! - **Every OpenAI-wire provider** rejects a tool result whose call is gone.
//!
//! So a fold either takes the whole turn or none of it. [`withdrawals`] is the
//! gate: give it the ids a fold wants to remove, and it hands back the ids that
//! must stay visible instead.
//!
//! The two rules feed each other, which is why the gate repeats to a fixed
//! point: withdrawing a call to keep its result intact leaves a turn whose
//! reasoning still folds while its call now stays visible — exactly the
//! reasoning split — and withdrawing a turn can strand a result whose call
//! survived outside it.
//!
//! The reverse direction stays allowed: reasoning and text may stay visible
//! while the call and its result fold, because that still leaves a valid
//! stream.

use std::collections::{BTreeMap, BTreeSet};

use crate::agent::bigctx::types::{CoreMessage, MessageContentType, MessageRole};

/// True for the assistant messages that carry the "burst" of one turn.
fn is_assistant_act(message: &CoreMessage) -> bool {
    message.role == MessageRole::Assistant
        && matches!(
            message.content_type,
            MessageContentType::Text | MessageContentType::ToolCall
        )
}

fn is_reasoning(message: &CoreMessage) -> bool {
    message.content_type == MessageContentType::Reasoning
}

fn is_call(message: &CoreMessage) -> bool {
    message.role == MessageRole::Assistant && message.content_type == MessageContentType::ToolCall
}

fn is_result(message: &CoreMessage) -> bool {
    message.content_type == MessageContentType::ToolResult
}

/// Disjoint turn groups, as member-id lists.
///
/// Grouping is adjacency-based: a reasoning run pairs with the assistant burst
/// immediately after it. A burst with no preceding reasoning still groups with
/// its sibling calls and results. Messages belonging to no turn (user, system,
/// orphan reasoning, summaries) get no group — they carry no pairing
/// constraint.
pub fn turn_groups(messages: &[CoreMessage]) -> Vec<Vec<String>> {
    // First result wins: a retried call can produce two results, and only the
    // first is the one the wire pairs with the call.
    let mut result_id_by_call_id: BTreeMap<&str, &str> = BTreeMap::new();
    for message in messages {
        if is_result(message)
            && let Some(call_id) = message.tool_call_id.as_deref()
        {
            result_id_by_call_id.entry(call_id).or_insert(&message.id);
        }
    }

    let mut grouped: BTreeSet<String> = BTreeSet::new();
    let mut groups: Vec<Vec<String>> = Vec::new();

    for (index, message) in messages.iter().enumerate() {
        if message.id.is_empty() || grouped.contains(&message.id) {
            continue;
        }
        if !(is_reasoning(message) || is_assistant_act(message)) {
            continue;
        }

        // Walk back to the head of the reasoning run that introduces this turn.
        let mut reasoning_start = index;
        if is_reasoning(message) {
            while reasoning_start > 0 && is_reasoning(&messages[reasoning_start - 1]) {
                reasoning_start -= 1;
            }
        } else {
            let mut start = index;
            while start > 0 && is_assistant_act(&messages[start - 1]) {
                start -= 1;
            }
            reasoning_start = start;
            while reasoning_start > 0 && is_reasoning(&messages[reasoning_start - 1]) {
                reasoning_start -= 1;
            }
        }

        // The burst begins after the reasoning run.
        let mut burst_start = reasoning_start;
        while burst_start < messages.len() && is_reasoning(&messages[burst_start]) {
            burst_start += 1;
        }
        if burst_start >= messages.len() || !is_assistant_act(&messages[burst_start]) {
            // Orphan reasoning run with no companion burst: no pairing
            // constraint, so it gets no group.
            continue;
        }

        let mut burst_end = burst_start;
        while burst_end + 1 < messages.len() && is_assistant_act(&messages[burst_end + 1]) {
            burst_end += 1;
        }

        let mut members: BTreeSet<String> = BTreeSet::new();
        for message in messages.iter().take(burst_end + 1).skip(reasoning_start) {
            if message.id.is_empty() {
                continue;
            }
            members.insert(message.id.clone());
            if is_call(message)
                && let Some(call_id) = message.tool_call_id.as_deref()
                && let Some(result_id) = result_id_by_call_id.get(call_id)
            {
                members.insert((*result_id).to_string());
            }
        }

        for id in &members {
            grouped.insert(id.clone());
        }
        groups.push(members.into_iter().collect());
    }

    groups
}

/// Which folded ids the rebuild cannot keep folded, and what that cost.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Withdrawals {
    /// Ids that must stay visible.
    pub withdrawn: BTreeSet<String>,
    /// Turns withdrawn to keep a reasoning run with its call.
    pub split_turn_count: usize,
    /// Call/result pairs withdrawn together.
    pub split_pair_count: usize,
}

impl Withdrawals {
    pub fn is_empty(&self) -> bool {
        self.withdrawn.is_empty()
    }

    /// True when this id must stay visible.
    pub fn keeps(&self, id: &str) -> bool {
        self.withdrawn.contains(id)
    }
}

/// The fold-integrity gate.
///
/// Given the ids a fold removes from the visible stream, return the ids that
/// must stay visible instead. Runs to a fixed point; each turn and pair is
/// acted on once and reported once.
pub fn withdrawals(messages: &[CoreMessage], folded_ids: &BTreeSet<String>) -> Withdrawals {
    let mut remaining: BTreeSet<String> = folded_ids.clone();
    let mut withdrawn: BTreeSet<String> = BTreeSet::new();
    let mut handled_turns: BTreeSet<usize> = BTreeSet::new();
    let mut handled_pairs: BTreeSet<String> = BTreeSet::new();

    let mut reasoning_ids: BTreeSet<&str> = BTreeSet::new();
    let mut call_ids: BTreeSet<&str> = BTreeSet::new();
    let mut call_id_by_message_id: BTreeMap<&str, &str> = BTreeMap::new();
    let mut result_id_by_call_id: BTreeMap<&str, &str> = BTreeMap::new();

    for message in messages {
        if message.id.is_empty() {
            continue;
        }
        if is_reasoning(message) {
            reasoning_ids.insert(&message.id);
        }
        if is_call(message) {
            call_ids.insert(&message.id);
            if let Some(call_id) = message.tool_call_id.as_deref() {
                call_id_by_message_id.insert(&message.id, call_id);
            }
        }
        if is_result(message)
            && let Some(call_id) = message.tool_call_id.as_deref()
        {
            result_id_by_call_id.entry(call_id).or_insert(&message.id);
        }
    }

    let groups = turn_groups(messages);
    let mut changed = true;
    while changed {
        changed = false;

        // INV1: a kept assistant call must keep its reasoning run.
        for (index, group) in groups.iter().enumerate() {
            if handled_turns.contains(&index) {
                continue;
            }
            let fold_has_reasoning = group
                .iter()
                .any(|id| remaining.contains(id) && reasoning_ids.contains(id.as_str()));
            if !fold_has_reasoning {
                continue;
            }
            let kept_has_call = group
                .iter()
                .any(|id| !remaining.contains(id) && call_ids.contains(id.as_str()));
            if !kept_has_call {
                continue;
            }
            handled_turns.insert(index);
            for id in group {
                remaining.remove(id);
                withdrawn.insert(id.clone());
            }
            changed = true;
        }

        // INV2: a call and its result are one exchange.
        for message in messages {
            if message.id.is_empty() || !call_ids.contains(message.id.as_str()) {
                continue;
            }
            let Some(call_id) = call_id_by_message_id.get(message.id.as_str()) else {
                continue;
            };
            if handled_pairs.contains(*call_id) {
                continue;
            }
            let Some(result_id) = result_id_by_call_id.get(call_id) else {
                continue;
            };
            if remaining.contains(&message.id) == remaining.contains(*result_id) {
                continue;
            }
            handled_pairs.insert((*call_id).to_string());
            remaining.remove(&message.id);
            remaining.remove(*result_id);
            withdrawn.insert(message.id.clone());
            withdrawn.insert((*result_id).to_string());
            changed = true;
        }
    }

    Withdrawals {
        withdrawn,
        split_turn_count: handled_turns.len(),
        split_pair_count: handled_pairs.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reasoning(id: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::Assistant, MessageContentType::Reasoning).with_text("why")
    }

    fn text(id: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::Assistant, MessageContentType::Text).with_text("answer")
    }

    fn call(id: &str, call_id: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::Assistant, MessageContentType::ToolCall)
            .with_text("{}")
            .with_tool("read", call_id)
    }

    fn result(id: &str, call_id: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::Tool, MessageContentType::ToolResult)
            .with_text("body")
            .with_tool("read", call_id)
    }

    fn user(id: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::User, MessageContentType::Text).with_text("go")
    }

    fn folded(ids: &[&str]) -> BTreeSet<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_reasoning_run_and_its_burst_form_one_group() {
        let msgs = vec![user("t1.0"), reasoning("t1.1"), text("t1.2")];
        let groups = turn_groups(&msgs);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0], vec!["t1.1".to_string(), "t1.2".to_string()]);
    }

    #[test]
    fn a_burst_and_its_result_join_the_group() {
        let msgs = vec![reasoning("t1.1"), call("t1.2", "c1"), result("t1.3", "c1")];
        let groups = turn_groups(&msgs);
        assert_eq!(groups.len(), 1);
        assert!(groups[0].contains(&"t1.3".to_string()));
        assert_eq!(groups[0].len(), 3);
    }

    #[test]
    fn orphan_reasoning_gets_no_group() {
        // No assistant burst follows, so there is no pairing constraint.
        let msgs = vec![user("t1.0"), reasoning("t1.1")];
        assert!(turn_groups(&msgs).is_empty());
    }

    #[test]
    fn a_user_message_gets_no_group() {
        let msgs = vec![user("t1.0"), text("t1.1")];
        let groups = turn_groups(&msgs);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0], vec!["t1.1".to_string()]);
    }

    #[test]
    fn folding_reasoning_while_keeping_the_call_withdraws_the_whole_turn() {
        // INV1. This is the DeepSeek 400: a tool call replayed without its
        // reasoning.
        let msgs = vec![reasoning("t1.1"), call("t1.2", "c1"), result("t1.3", "c1")];
        let w = withdrawals(&msgs, &folded(&["t1.1"]));
        assert!(w.keeps("t1.1"), "the reasoning must stay");
        assert!(w.keeps("t1.2"), "and so must the call it belongs to");
        assert_eq!(w.split_turn_count, 1);
    }

    #[test]
    fn folding_a_call_while_keeping_its_result_withdraws_both() {
        // INV2. This is the OpenAI-wire 400: a result whose call is gone.
        let msgs = vec![call("t1.1", "c1"), result("t1.2", "c1")];
        let w = withdrawals(&msgs, &folded(&["t1.1"]));
        assert!(w.keeps("t1.1"));
        assert!(w.keeps("t1.2"), "the result cannot outlive its call");
        assert_eq!(w.split_pair_count, 1);
    }

    #[test]
    fn folding_both_halves_is_allowed() {
        let msgs = vec![call("t1.1", "c1"), result("t1.2", "c1")];
        let w = withdrawals(&msgs, &folded(&["t1.1", "t1.2"]));
        assert!(w.is_empty(), "a whole exchange may fold");
    }

    #[test]
    fn keeping_reasoning_while_folding_the_call_is_allowed() {
        // The reverse direction leaves a valid stream: reasoning and text stay,
        // the call and result go.
        let msgs = vec![reasoning("t1.1"), call("t1.2", "c1"), result("t1.3", "c1")];
        let w = withdrawals(&msgs, &folded(&["t1.2", "t1.3"]));
        assert!(w.is_empty());
    }

    #[test]
    fn the_gate_repeats_until_it_settles() {
        // Withdrawing the call for INV2 leaves a turn whose reasoning still
        // folds while its call stays visible — the INV1 split. One pass would
        // miss it.
        let msgs = vec![reasoning("t1.1"), call("t1.2", "c1"), result("t1.3", "c1")];
        let w = withdrawals(&msgs, &folded(&["t1.1", "t1.3"]));
        assert!(w.keeps("t1.1"), "reasoning would split from the kept call");
        assert!(w.keeps("t1.2"));
        assert!(w.keeps("t1.3"));
    }

    #[test]
    fn two_turns_are_independent() {
        let msgs = vec![
            reasoning("t1.1"),
            call("t1.2", "c1"),
            result("t1.3", "c1"),
            user("t1.4"),
            reasoning("t1.5"),
            call("t1.6", "c2"),
            result("t1.7", "c2"),
        ];
        // fold the first turn whole, keep the second
        let w = withdrawals(&msgs, &folded(&["t1.1", "t1.2", "t1.3"]));
        assert!(w.is_empty(), "the second turn is untouched");
    }

    #[test]
    fn a_plain_text_turn_folds_freely() {
        let msgs = vec![user("t1.0"), text("t1.1")];
        let w = withdrawals(&msgs, &folded(&["t1.1"]));
        assert!(w.is_empty());
    }
}
