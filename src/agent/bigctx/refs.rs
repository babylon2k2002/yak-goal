//! Citation refs: the `m00001` names the model uses to point at messages.
//!
//! Refs are *additive and permanent*. Once a message has a ref it keeps it for
//! the life of the session, including after compression removes it from the
//! visible history — that is precisely what lets `decompress` restore it and
//! `search_context` report where a hit came from.
//!
//! Width is a padding floor, not a fixed width: index 100000 renders as
//! `m100000`. Every ref issued below 100000 therefore stays byte-identical
//! when the ceiling lifts.

use crate::agent::bigctx::types::MessageRefMap;

const REF_WIDTH: usize = 5;
const MIN_INDEX: u64 = 1;
const MAX_INDEX: u64 = 9_999_999;

/// A protected message's placeholder ref. It is never a compressible endpoint;
/// protection is recorded so the model can see the message exists and cannot
/// be folded.
pub const BLOCKED_REF: &str = "BLOCKED";

/// `1` → `"m00001"`.
pub fn index_to_ref(index: u64) -> Option<String> {
    if !(MIN_INDEX..=MAX_INDEX).contains(&index) {
        return None;
    }
    Some(format!("m{index:0REF_WIDTH$}"))
}

/// `"m00001"` / `"m1"` → `1`. Returns `None` for anything else.
pub fn ref_to_index(reference: &str) -> Option<u64> {
    let body = reference.trim().to_ascii_lowercase();
    let digits = body.strip_prefix('m')?;
    if digits.is_empty() || digits.len() > 7 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let index: u64 = digits.parse().ok()?;
    (MIN_INDEX..=MAX_INDEX).contains(&index).then_some(index)
}

/// An empty map.
pub fn empty_ref_map() -> MessageRefMap {
    MessageRefMap::default()
}

/// Why a ref could not be handed out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssignError {
    /// Every index up to the ceiling is taken. Only reachable after ~10 million
    /// messages in one session.
    Exhausted,
}

impl std::fmt::Display for AssignError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AssignError::Exhausted => write!(
                f,
                "ref capacity exhausted: cannot allocate beyond {}",
                index_to_ref(MAX_INDEX).unwrap_or_default()
            ),
        }
    }
}

impl std::error::Error for AssignError {}

/// What [`assign_refs`] did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AssignRefsOutcome {
    pub newly_assigned: u64,
}

/// Give every not-yet-referenced core a ref, in order.
///
/// `is_protected` marks a core as unfoldable: it gets [`BLOCKED_REF`] instead
/// of a numeric ref, so the model can cite it but the algorithm refuses to fold
/// it. Protection is not permanent — if it lifts, the blocked entry is dropped
/// and the core receives a fresh numeric ref on the next pass (fresh number
/// only; the id-never-reused contract holds).
pub fn assign_refs(
    messages: &[crate::agent::bigctx::types::CoreMessage],
    map: &mut MessageRefMap,
    next_index: &mut u64,
    is_protected: impl Fn(&crate::agent::bigctx::types::CoreMessage) -> bool,
) -> Result<AssignRefsOutcome, AssignError> {
    let mut outcome = AssignRefsOutcome::default();
    let mut cursor = (*next_index).max(MIN_INDEX);

    for message in messages {
        if message.id.is_empty() {
            continue;
        }
        let protected_now = is_protected(message);
        if let Some(prev) = map.by_raw.get(&message.id) {
            if prev == BLOCKED_REF && !protected_now {
                map.by_raw.remove(&message.id);
            } else {
                continue;
            }
        }
        if protected_now {
            map.by_raw.insert(message.id.clone(), BLOCKED_REF.to_string());
            continue;
        }
        let (text, index) = allocate_free_ref(map, cursor)?;
        cursor = index + 1;
        map.by_raw.insert(message.id.clone(), text.clone());
        map.by_ref.insert(text, message.id.clone());
        outcome.newly_assigned += 1;
    }

    *next_index = cursor;
    Ok(outcome)
}

fn allocate_free_ref(map: &MessageRefMap, start: u64) -> Result<(String, u64), AssignError> {
    let mut candidate = start.max(MIN_INDEX);
    while candidate <= MAX_INDEX {
        let text = index_to_ref(candidate).ok_or(AssignError::Exhausted)?;
        if !map.by_ref.contains_key(&text) {
            return Ok((text, candidate));
        }
        candidate += 1;
    }
    Err(AssignError::Exhausted)
}

/// Highest numeric ref currently in use, 0 when none.
pub fn highest_used_index(map: &MessageRefMap) -> u64 {
    map.by_raw
        .values()
        .filter_map(|r| ref_to_index(r))
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::bigctx::types::{CoreMessage, MessageContentType, MessageRole};

    fn core(id: &str) -> CoreMessage {
        CoreMessage::new(id, MessageRole::User, MessageContentType::Text)
    }

    #[test]
    fn refs_pad_to_five_digits() {
        assert_eq!(index_to_ref(1).as_deref(), Some("m00001"));
        assert_eq!(index_to_ref(42).as_deref(), Some("m00042"));
        assert_eq!(index_to_ref(99999).as_deref(), Some("m99999"));
    }

    #[test]
    fn refs_widen_past_the_padding_floor() {
        assert_eq!(index_to_ref(100000).as_deref(), Some("m100000"));
        assert_eq!(index_to_ref(9_999_999).as_deref(), Some("m9999999"));
    }

    #[test]
    fn index_zero_and_past_the_ceiling_are_refused() {
        assert_eq!(index_to_ref(0), None);
        assert_eq!(index_to_ref(10_000_000), None);
    }

    #[test]
    fn parsing_accepts_loose_padding_and_case() {
        assert_eq!(ref_to_index("m00001"), Some(1));
        assert_eq!(ref_to_index("m1"), Some(1));
        assert_eq!(ref_to_index("M00001"), Some(1));
        assert_eq!(ref_to_index("  m00042 "), Some(42));
    }

    #[test]
    fn parsing_rejects_junk() {
        for junk in ["", "m", "b1", "x00001", "m0000001x", "m0000000"] {
            assert_eq!(ref_to_index(junk), None, "{junk} should not parse");
        }
    }

    #[test]
    fn assignment_is_sequential_and_idempotent() {
        let mut map = empty_ref_map();
        let mut next = 1;
        let msgs = vec![core("t1.1"), core("t1.2")];
        let first = assign_refs(&msgs, &mut map, &mut next, |_| false).unwrap();
        assert_eq!(first.newly_assigned, 2);
        assert_eq!(map.ref_for("t1.1"), Some("m00001"));
        assert_eq!(map.ref_for("t1.2"), Some("m00002"));
        assert_eq!(next, 3);

        // a second pass over the same cores assigns nothing new
        let again = assign_refs(&msgs, &mut map, &mut next, |_| false).unwrap();
        assert_eq!(again.newly_assigned, 0);
        assert_eq!(next, 3);
    }

    #[test]
    fn a_new_core_continues_from_the_cursor() {
        let mut map = empty_ref_map();
        let mut next = 1;
        assign_refs(&[core("t1.1")], &mut map, &mut next, |_| false).unwrap();
        assign_refs(&[core("t1.2")], &mut map, &mut next, |_| false).unwrap();
        assert_eq!(map.ref_for("t1.2"), Some("m00002"));
    }

    #[test]
    fn protected_cores_get_the_blocked_marker() {
        let mut map = empty_ref_map();
        let mut next = 1;
        let msgs = vec![core("t1.1"), core("t1.2")];
        assign_refs(&msgs, &mut map, &mut next, |m| m.id == "t1.1").unwrap();
        assert_eq!(map.ref_for("t1.1"), Some(BLOCKED_REF));
        assert_eq!(map.ref_for("t1.2"), Some("m00001"));
        // the blocked marker consumes no numeric index
        assert_eq!(next, 2);
    }

    #[test]
    fn protection_lifting_issues_a_fresh_numeric_ref() {
        let mut map = empty_ref_map();
        let mut next = 1;
        assign_refs(&[core("t1.1")], &mut map, &mut next, |_| true).unwrap();
        assert_eq!(map.ref_for("t1.1"), Some(BLOCKED_REF));
        assign_refs(&[core("t1.1")], &mut map, &mut next, |_| false).unwrap();
        assert_eq!(map.ref_for("t1.1"), Some("m00001"));
    }

    #[test]
    fn empty_ids_are_skipped() {
        let mut map = empty_ref_map();
        let mut next = 1;
        assign_refs(&[core("")], &mut map, &mut next, |_| false).unwrap();
        assert!(map.is_empty());
        assert_eq!(next, 1);
    }

    #[test]
    fn highest_used_index_ignores_the_blocked_marker() {
        let mut map = empty_ref_map();
        let mut next = 1;
        let msgs = vec![core("a"), core("b"), core("c")];
        assign_refs(&msgs, &mut map, &mut next, |m| m.id == "c").unwrap();
        assert_eq!(highest_used_index(&map), 2);
    }
}
