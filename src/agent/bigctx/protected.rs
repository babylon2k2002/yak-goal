//! What may never be folded.
//!
//! Three independent reasons a message is off-limits:
//!
//! 1. **Its tool is protected** — `protected_tools` names tools whose results
//!    are the model's source of truth (a skill definition, a rule list). Folding
//!    them loses information the session cannot rebuild.
//! 2. **It is the latest of its tool** — `protected_latest_tools` names
//!    cumulative-snapshot tools (`plan`, `todo_list`) where only the newest
//!    result matters and every older one is redundant. The newest is protected;
//!    the older ones stay foldable.
//! 3. **It is in the recent zone** — the last N messages, the last N tokens,
//!    and the last user message. This is what keeps the current working set
//!    intact so the model does not lose the thread it is on.
//!
//! A fourth rule is structural rather than a preference: a tool call and its
//! result are protected or foldable *together*. Protecting one half and folding
//! the other leaves an unpaired entry on the wire, which strict upstreams reject
//! outright.

use std::collections::{BTreeMap, BTreeSet};

use crate::agent::bigctx::config::BigctxConfig;
use crate::agent::bigctx::tokenize::count_message_tokens;
use crate::agent::bigctx::types::{CompressionState, CoreMessage, MessageContentType, base_id_of};

/// Which messages must stay visible this pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Protection {
    /// Base ids that may not be folded.
    protected_bases: BTreeSet<String>,
    /// `tool_call_id`s whose exchange is protected, so a result projected
    /// without a tool name is still covered.
    protected_calls: BTreeSet<String>,
}

impl Protection {
    pub fn is_protected(&self, message: &CoreMessage) -> bool {
        if self.protected_bases.contains(base_id_of(&message.id)) {
            return true;
        }
        message
            .tool_call_id
            .as_deref()
            .is_some_and(|id| self.protected_calls.contains(id))
    }

    pub fn is_empty(&self) -> bool {
        self.protected_bases.is_empty() && self.protected_calls.is_empty()
    }

    pub fn len(&self) -> usize {
        self.protected_bases.len()
    }
}

/// Compute protection for this pass.
///
/// `state` is needed because a folded message is already invisible: it must not
/// count toward the "last N messages" window, or the window would silently
/// shrink as the session grows.
pub fn compute_protection(
    messages: &[CoreMessage],
    state: &CompressionState,
    config: &BigctxConfig,
) -> Protection {
    let covered = crate::agent::bigctx::state::covered_message_ids(state);
    let mut out = Protection::default();

    // Tool-level protection first: it is independent of position.
    for message in messages {
        let Some(tool) = message.tool_name.as_deref() else {
            continue;
        };
        if config.is_tool_protected(tool) {
            out.protected_bases.insert(base_id_of(&message.id).to_string());
            if let Some(call_id) = message.tool_call_id.as_deref() {
                out.protected_calls.insert(call_id.to_string());
            }
        }
    }

    // "Latest only" protection: the newest exchange per pattern.
    for pattern_index in 0..config.protected_latest_tools.len() {
        let pattern = &config.protected_latest_tools[pattern_index];
        let newest = messages.iter().rev().find(|m| {
            m.content_type == MessageContentType::ToolCall
                && m.tool_name
                    .as_deref()
                    .is_some_and(|tool| tool_matches(tool, pattern))
        });
        if let Some(message) = newest {
            out.protected_bases
                .insert(base_id_of(&message.id).to_string());
            if let Some(call_id) = message.tool_call_id.as_deref() {
                out.protected_calls.insert(call_id.to_string());
            }
        }
    }

    // The recent zone: last N visible messages, last N visible tokens, and the
    // last visible user message.
    let mut visible: Vec<(&CoreMessage, u64)> = Vec::new();
    for message in messages {
        if covered.contains(base_id_of(&message.id)) {
            continue;
        }
        visible.push((message, count_message_tokens(message)));
    }

    if config.preserve_recent_messages > 0 {
        let start = visible.len().saturating_sub(config.preserve_recent_messages);
        for (message, _) in &visible[start..] {
            out.protected_bases
                .insert(base_id_of(&message.id).to_string());
        }

        // The last user message carries the intent of the current task; losing
        // it makes the model answer a question it can no longer see.
        if let Some((message, _)) = visible.iter().rev().find(|(m, _)| {
            m.role == crate::agent::bigctx::types::MessageRole::User
        }) {
            out.protected_bases
                .insert(base_id_of(&message.id).to_string());
        }
    }

    if config.preserve_recent_tokens > 0 {
        let mut accumulated = 0u64;
        for (message, tokens) in visible.iter().rev() {
            if accumulated >= config.preserve_recent_tokens {
                break;
            }
            out.protected_bases
                .insert(base_id_of(&message.id).to_string());
            accumulated += tokens;
        }
    }

    close_tool_pair_gap(messages, &covered, &mut out);
    out
}

/// Make the recent zone symmetric per tool pair.
///
/// The recent-window rules can protect a call and miss its result (or the other
/// way round) when the two halves have different sizes. Folding one half strands
/// an unpaired entry, which strict validators reject, so both halves are
/// protected whenever either is.
fn close_tool_pair_gap(
    messages: &[CoreMessage],
    covered: &BTreeSet<String>,
    out: &mut Protection,
) {
    let mut call_base_by_call_id: BTreeMap<&str, &str> = BTreeMap::new();
    let mut result_bases: Vec<(&str, &str)> = Vec::new();
    for message in messages {
        if covered.contains(base_id_of(&message.id)) {
            continue;
        }
        let Some(call_id) = message.tool_call_id.as_deref() else {
            continue;
        };
        match message.content_type {
            MessageContentType::ToolCall => {
                call_base_by_call_id
                    .entry(call_id)
                    .or_insert(base_id_of(&message.id));
            }
            MessageContentType::ToolResult => {
                result_bases.push((call_id, base_id_of(&message.id)));
            }
            _ => {}
        }
    }

    for (call_id, result_base) in result_bases {
        let Some(call_base) = call_base_by_call_id.get(call_id) else {
            continue;
        };
        let call_protected = out.protected_bases.contains(*call_base);
        let result_protected = out.protected_bases.contains(result_base);
        if call_protected || result_protected {
            out.protected_bases.insert((*call_base).to_string());
            out.protected_bases.insert(result_base.to_string());
        }
    }
}

/// Glob-suffix tool-name match, same semantics as [`BigctxConfig::is_tool_protected`].
fn tool_matches(tool: &str, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    match pattern.strip_suffix('*') {
        Some(prefix) => tool.starts_with(prefix),
        None => tool == pattern,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::bigctx::config::default_config;
    use crate::agent::bigctx::state::create_initial_state;
    use crate::agent::bigctx::types::{MessageRole, Tier};

    fn user(id: &str, text: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::User, MessageContentType::Text).with_text(text)
    }

    fn call(id: &str, tool: &str, call_id: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::Assistant, MessageContentType::ToolCall)
            .with_text("{}")
            .with_tool(tool, call_id)
    }

    fn result(id: &str, tool: &str, call_id: &str, text: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::Tool, MessageContentType::ToolResult)
            .with_text(text)
            .with_tool(tool, call_id)
    }

    #[test]
    fn the_recent_window_protects_the_tail() {
        let cfg = default_config(200_000);
        let state = create_initial_state();
        let msgs = vec![
            user("t1.1", "old"),
            user("t1.2", "older"),
            user("t1.3", "newest"),
        ];
        let prot = compute_protection(&msgs, &state, &cfg);
        // preserve_recent_messages defaults to 3, so everything is protected
        assert!(prot.is_protected(&msgs[0]));
        assert!(prot.is_protected(&msgs[2]));
    }

    #[test]
    fn the_last_user_message_is_always_protected() {
        let mut cfg = default_config(200_000);
        cfg.preserve_recent_messages = 1;
        cfg.preserve_recent_tokens = 0;
        let state = create_initial_state();
        let msgs = vec![
            user("t1.1", "the task"),
            result("t1.2", "read", "c1", "output"),
        ];
        let prot = compute_protection(&msgs, &state, &cfg);
        assert!(
            prot.is_protected(&msgs[0]),
            "user intent must survive the window"
        );
    }

    #[test]
    fn a_protected_tool_protects_both_halves() {
        let mut cfg = default_config(200_000);
        cfg.preserve_recent_messages = 0;
        cfg.preserve_recent_tokens = 0;
        cfg.protected_tools = vec!["plan".to_string()];
        let state = create_initial_state();

        let msgs = vec![
            user("t1.1", "go"),
            call("t1.2", "plan", "c1"),
            result("t1.3", "plan", "c1", "the plan"),
        ];
        let prot = compute_protection(&msgs, &state, &cfg);
        assert!(prot.is_protected(&msgs[1]), "the call");
        assert!(prot.is_protected(&msgs[2]), "and its result");
    }

    #[test]
    fn a_result_without_a_tool_name_is_still_covered_by_pairing() {
        let mut cfg = default_config(200_000);
        cfg.preserve_recent_messages = 0;
        cfg.preserve_recent_tokens = 0;
        cfg.protected_tools = vec!["plan".to_string()];
        let state = create_initial_state();

        let mut bare = result("t1.3", "plan", "c1", "the plan");
        bare.tool_name = None; // host projected it without a name

        let msgs = vec![user("t1.1", "go"), call("t1.2", "plan", "c1"), bare];
        let prot = compute_protection(&msgs, &state, &cfg);
        assert!(prot.is_protected(&msgs[2]), "pairing covers the bare half");
    }

    #[test]
    fn latest_only_protects_the_newest_instance() {
        let mut cfg = default_config(200_000);
        cfg.preserve_recent_messages = 0;
        cfg.preserve_recent_tokens = 0;
        cfg.protected_latest_tools = vec!["plan".to_string()];
        let state = create_initial_state();

        let msgs = vec![
            user("t1.1", "go"),
            call("t2.1", "plan", "c1"),
            result("t2.2", "plan", "c1", "old plan"),
            call("t3.1", "plan", "c2"),
            result("t3.2", "plan", "c2", "new plan"),
        ];
        let prot = compute_protection(&msgs, &state, &cfg);
        assert!(prot.is_protected(&msgs[3]), "the newest call");
        assert!(prot.is_protected(&msgs[4]), "the newest result");
        assert!(!prot.is_protected(&msgs[1]), "the older call stays foldable");
        assert!(!prot.is_protected(&msgs[2]), "the older result stays foldable");
    }

    #[test]
    fn folded_messages_do_not_count_toward_the_window() {
        let mut cfg = default_config(200_000);
        cfg.preserve_recent_messages = 1;
        cfg.preserve_recent_tokens = 0;

        let mut state = create_initial_state();
        state.blocks.push(crate::agent::bigctx::types::CompressionBlock {
            block_id: "b1".into(),
            run_id: "r1".into(),
            tier: Tier::One,
            topic: None,
            summary: "s".into(),
            direct_message_ids: vec!["t1.2".into()],
            effective_message_ids: vec!["t1.2".into()],
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

        let msgs = vec![
            user("t1.1", "go"),
            user("t1.2", "folded away"),
            user("t1.3", "visible tail"),
        ];
        let prot = compute_protection(&msgs, &state, &cfg);
        // only one visible message, so the window covers exactly t1.3
        assert!(prot.is_protected(&msgs[2]));
    }

    #[test]
    fn a_wildcard_tool_pattern_works() {
        let mut cfg = default_config(200_000);
        cfg.preserve_recent_messages = 0;
        cfg.preserve_recent_tokens = 0;
        cfg.protected_tools = vec!["mcp__*".to_string()];
        let state = create_initial_state();

        let msgs = vec![
            user("t1.1", "go"),
            call("t1.2", "mcp__github", "c1"),
            result("t1.3", "mcp__github", "c1", "x"),
        ];
        let prot = compute_protection(&msgs, &state, &cfg);
        assert!(prot.is_protected(&msgs[1]));
    }

    #[test]
    fn nothing_protected_when_all_knobs_are_off() {
        let mut cfg = default_config(200_000);
        cfg.preserve_recent_messages = 0;
        cfg.preserve_recent_tokens = 0;
        let state = create_initial_state();
        let msgs = vec![user("t1.1", "a"), user("t1.2", "b")];
        let prot = compute_protection(&msgs, &state, &cfg);
        assert!(prot.is_empty());
    }
}
