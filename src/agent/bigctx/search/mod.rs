//! Search — find folded material by content, without remembering a ref.
//!
//! `acp_retrieve` and `decompress` both need the model to already know *which*
//! ref to ask for. Over a long session it cannot: a fold from two hours ago
//! left a summary and the model has no index back into it. Search closes that
//! gap — it scores every block summary and every folded message against a
//! query and hands back refs to open.
//!
//! The two doc sources are built here from the host's own data:
//!
//! - [`block_docs`] — one doc per block, from `state.blocks`. These are the
//!   summaries that are currently standing in for the folded material.
//! - [`message_docs`] — one doc per folded message. `history` still carries
//!   every message the algorithm ever saw (folding is a view transform, not a
//!   deletion), so the original text is searchable. A hit names the block that
//!   owns it, which is exactly what `decompress` wants.
//!
//! Searchable text is otherwise immutable, so the per-doc features are cached
//! process-wide (see `doc_cache`); a repeat search costs one pass over the
//! query, not the corpus.

pub mod bm25;
pub mod doc_cache;
pub mod fuzzy;
pub mod hybrid;
pub mod registry;
pub mod substring;
pub mod tokenizer;
pub mod types;

use crate::agent::bigctx::prune::is_rendered_summary;
use crate::agent::bigctx::search::registry::get_algorithm;
pub use crate::agent::bigctx::search::types::{
    RoleWeights, ScoredBlock, SearchDoc, SearchDocKind, SearchOptions, SearchResult, SearchRole,
    DEFAULT_ALGORITHM,
};
use crate::agent::bigctx::tokenize::default_count_tokens;
use crate::agent::bigctx::types::{base_id_of, CompressionState, CoreMessage, MessageRole};

/// The name the tool is exposed under. Mirrors the kernel's
/// `SEARCH_CONTEXT_TOOL_NAME`, so a prompt written against either names the
/// same tool.
pub fn tool_name() -> &'static str {
    "search_context"
}

/// One document per block, from the state's summaries.
///
/// The block's own id is both the doc ref and the owning block: opening a
/// search hit with `decompress` takes a block id, so this lines up directly.
pub fn block_docs(state: &CompressionState) -> Vec<SearchDoc> {
    let mut out = Vec::with_capacity(state.blocks.len());
    for b in &state.blocks {
        let summary = b.summary.as_str();
        if summary.is_empty() && b.topic.is_none() {
            continue;
        }
        let topic = b.topic.as_deref().unwrap_or(b.block_id.as_str());
        out.push(SearchDoc {
            kind: SearchDocKind::Block,
            ref_id: b.block_id.clone(),
            text: match b.topic.as_deref() {
                Some(t) => format!("{t} {summary}"),
                None => summary.to_string(),
            },
            title: topic.to_string(),
            role: None,
            block_id: Some(b.block_id.clone()),
            tier: b.tier.number(),
            tokens: b.compressed_tokens,
        });
    }
    out
}

/// One document per message the host can still see, annotated with the block
/// that folded it (when one did).
///
/// Rendered summaries are skipped: they are the block's own text, already
/// searchable through [`block_docs`], and including them would let one hit
/// appear twice.
pub fn message_docs(history: &[CoreMessage], state: &CompressionState) -> Vec<SearchDoc> {
    // base id -> owning block, so a folded message points at what to open.
    let mut owner: std::collections::BTreeMap<&str, (&str, u8)> = std::collections::BTreeMap::new();
    for b in &state.blocks {
        for id in &b.effective_message_ids {
            owner
                .entry(base_id_of(id))
                .or_insert((b.block_id.as_str(), b.tier.number()));
        }
    }

    let mut out = Vec::with_capacity(history.len());
    for m in history {
        if is_rendered_summary(m) {
            continue;
        }
        let text = m.text_or_empty();
        if text.trim().is_empty() {
            continue;
        }
        let Some(reference) = state.message_refs.ref_for(m.base_id()) else {
            // No ref means the algorithm never assigned one, which means the
            // host cannot act on a hit. Skip rather than surface a dead ref.
            continue;
        };
        let (block_id, tier) = match owner.get(m.base_id()) {
            Some((b, t)) => (Some(b.to_string()), *t),
            None => (None, 1),
        };
        let role = match m.role {
            MessageRole::User => Some(SearchRole::User),
            MessageRole::Assistant => Some(SearchRole::Assistant),
            MessageRole::Tool => Some(SearchRole::Tool),
            MessageRole::System => None,
        };
        out.push(SearchDoc {
            kind: SearchDocKind::Message,
            ref_id: reference.to_string(),
            text: text.to_string(),
            title: format!("{}: {}", role_label(role), clamp_prefix(text, 60)),
            role,
            block_id,
            tier,
            tokens: default_count_tokens(text),
        });
    }
    out
}

fn role_label(role: Option<SearchRole>) -> &'static str {
    match role {
        Some(SearchRole::User) => "user",
        Some(SearchRole::Assistant) => "assistant",
        Some(SearchRole::Tool) => "tool",
        None => "system",
    }
}

/// Score `docs` against `query` and return the ranked hits.
///
/// Ordering is by weighted score, descending; ties keep doc order. Results
/// below `min_score` are filtered out *after* weighting, so a role multiplier
/// cannot resurrect a zero-scoring doc.
pub fn search(docs: &[SearchDoc], query: &str, options: &SearchOptions) -> Vec<SearchResult> {
    if docs.is_empty() || query.trim().is_empty() {
        return Vec::new();
    }
    let name = options.algorithm.as_deref().unwrap_or(DEFAULT_ALGORITHM);
    let Some(algorithm) = get_algorithm(name) else {
        return Vec::new();
    };

    let raw = algorithm.score(docs, query);
    let weighted = apply_role_weight(raw, docs, options.role_weights);

    let by_ref: std::collections::BTreeMap<&str, &SearchDoc> =
        docs.iter().map(|d| (d.ref_id.as_str(), d)).collect();

    let mut results: Vec<SearchResult> = weighted
        .into_iter()
        .filter(|s| s.score >= options.min_score)
        .filter_map(|s| {
            let doc = by_ref.get(s.ref_id.as_str())?;
            Some(SearchResult {
                kind: doc.kind,
                ref_id: doc.ref_id.clone(),
                block_id: doc.block_id.clone(),
                tier: doc.tier,
                score: s.score,
                title: doc.title.clone(),
                preview: make_preview(&doc.text, query, options.preview_length),
                role: doc.role,
                tokens: doc.tokens,
            })
        })
        .collect();

    // Descending by score. `partial_cmp` cannot fail here (no NaN produces a
    // finite score, and the folds below keep every input finite), but sorting
    // must never panic, so an incomparable pair is treated as equal.
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(options.limit.max(1));
    results
}

fn apply_role_weight(
    scored: Vec<ScoredBlock>,
    docs: &[SearchDoc],
    weights: RoleWeights,
) -> Vec<ScoredBlock> {
    let by_ref: std::collections::BTreeMap<&str, &SearchDoc> =
        docs.iter().map(|d| (d.ref_id.as_str(), d)).collect();
    scored
        .into_iter()
        .map(|s| {
            let weight = by_ref
                .get(s.ref_id.as_str())
                .map(|d| weights.weight_for(d))
                .unwrap_or(1.0);
            ScoredBlock {
                ref_id: s.ref_id,
                score: s.score * weight,
            }
        })
        .collect()
}

/// Preview centred on the first query-term hit, falling back to the head.
///
/// A head-only preview shows the same first line for every doc and tells the
/// model nothing about *where* the match was.
fn make_preview(text: &str, query: &str, len: usize) -> String {
    if text.is_empty() {
        return String::new();
    }
    let terms: Vec<String> = query
        .to_lowercase()
        .split_whitespace()
        .filter(|t| t.chars().count() > 1)
        .map(str::to_string)
        .collect();
    if terms.is_empty() {
        return clamp_prefix(text, len);
    }

    let lower = text.to_lowercase();
    // Lowercasing can change the character count ("İ" folds to two chars), so
    // `lower`'s char indices are not `text`'s. When they disagree, the hit
    // position would slice the wrong place in `text`; fall back to the head
    // rather than guess. In practice this never fires for ASCII or CJK.
    if lower.chars().count() != text.chars().count() {
        return clamp_prefix(text, len);
    }

    let Some(hit_byte) = terms.iter().find_map(|t| lower.find(t.as_str())) else {
        return clamp_prefix(text, len);
    };

    // Byte offset -> char index (safe: `hit_byte` is a boundary in `lower`),
    // then char-indexed slicing of `text` so CJK never splits a codepoint.
    let hit_char = lower[..hit_byte].chars().count();
    let total_chars = text.chars().count();
    let half = len / 2;
    let start = hit_char.saturating_sub(half);
    let end = (start + len).min(total_chars);
    let body: String = text.chars().skip(start).take(end - start).collect();
    let mut out = String::new();
    if start > 0 {
        out.push('…');
    }
    out.push_str(body.trim());
    if end < total_chars {
        out.push('…');
    }
    out
}

fn clamp_prefix(text: &str, len: usize) -> String {
    if text.chars().count() <= len {
        return text.to_string();
    }
    let truncated: String = text.chars().take(len).collect();
    format!("{}…", truncated.trim_end())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::bigctx::prune::summary_message_id;
    use crate::agent::bigctx::types::{MessageContentType, Tier};

    fn doc(id: &str, text: &str) -> SearchDoc {
        SearchDoc {
            kind: SearchDocKind::Message,
            ref_id: id.into(),
            text: text.into(),
            title: id.into(),
            role: Some(SearchRole::Tool),
            block_id: None,
            tier: 1,
            tokens: 10,
        }
    }

    #[test]
    fn results_are_ranked_by_score_descending() {
        let docs = [
            doc("a", "database migration plan"),
            doc("b", "unrelated chatter about nothing"),
        ];
        let hits = search(&docs, "migration", &SearchOptions::default());
        assert!(!hits.is_empty());
        assert_eq!(hits[0].ref_id, "a");
    }

    #[test]
    fn an_empty_query_returns_nothing() {
        let docs = [doc("a", "text")];
        assert!(search(&docs, "   ", &SearchOptions::default()).is_empty());
    }

    #[test]
    fn no_docs_returns_nothing() {
        assert!(search(&[], "q", &SearchOptions::default()).is_empty());
    }

    #[test]
    fn an_unknown_algorithm_returns_nothing_rather_than_panicking() {
        let docs = [doc("a", "text")];
        let opts = SearchOptions {
            algorithm: Some("semantic".into()),
            ..Default::default()
        };
        assert!(search(&docs, "text", &opts).is_empty());
    }

    #[test]
    fn the_limit_is_honoured() {
        let docs: Vec<SearchDoc> = (0..20)
            .map(|i| doc(&format!("r{i}"), "common term here"))
            .collect();
        let opts = SearchOptions {
            limit: 3,
            ..Default::default()
        };
        assert_eq!(search(&docs, "common", &opts).len(), 3);
    }

    #[test]
    fn min_score_filters_the_tail() {
        let docs = [doc("a", "tokenizer"), doc("b", "zzzzz")];
        let opts = SearchOptions {
            min_score: 0.5,
            ..Default::default()
        };
        let hits = search(&docs, "tokenizer", &opts);
        assert!(hits.iter().all(|h| h.ref_id != "b"));
    }

    #[test]
    fn a_preview_centres_on_the_hit() {
        let text = format!("{}NEEDLE{}", "x".repeat(500), "y".repeat(500));
        let docs = [doc("a", &text)];
        let hits = search(&docs, "needle", &SearchOptions::default());
        assert!(!hits.is_empty());
        assert!(
            hits[0].preview.contains("NEEDLE"),
            "preview must show the hit, got: {}",
            hits[0].preview
        );
    }

    #[test]
    fn a_cjk_preview_never_panics_on_a_char_boundary() {
        let text = "身份验证流程".repeat(200);
        let docs = [doc("a", &text)];
        let hits = search(&docs, "验证", &SearchOptions::default());
        // The assertion is that this did not panic; content is secondary.
        assert!(!hits.is_empty());
    }

    #[test]
    fn block_docs_expose_topic_and_summary() {
        let mut state = crate::agent::bigctx::create_initial_state();
        state.blocks.push(crate::agent::bigctx::types::CompressionBlock {
            block_id: "b1".into(),
            run_id: "r1".into(),
            tier: Tier::One,
            topic: Some("auth work".into()),
            start_ref: None,
            end_ref: None,
            summary: "the login flow was rewritten".into(),
            direct_message_ids: vec![],
            effective_message_ids: vec![],
            direct_block_ids: vec![],
            compressed_tokens: 100,
            created_at: 0,
            survived_count: 0,
            generation: crate::agent::bigctx::types::BlockGeneration::Young,
            active: true,
            expanded: false,
            duration_ms: None,
            compress_call_id: None,
        });
        let docs = block_docs(&state);
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].ref_id, "b1");
        assert!(docs[0].text.contains("auth work"));
        assert!(docs[0].text.contains("login flow"));
        assert_eq!(docs[0].tier, 1);
    }

    #[test]
    fn message_docs_skip_rendered_summaries() {
        let mut state = crate::agent::bigctx::create_initial_state();
        state
            .message_refs
            .by_raw
            .insert("t1.0".into(), "m00001".into());
        state
            .message_refs
            .by_ref
            .insert("m00001".into(), "t1.0".into());
        let history = vec![
            CoreMessage::new("t1.0", MessageRole::User, MessageContentType::Text)
                .with_text("hello"),
            CoreMessage::new(
                summary_message_id("b1"),
                MessageRole::System,
                MessageContentType::Text,
            )
            .with_text("a summary"),
        ];
        let docs = message_docs(&history, &state);
        assert_eq!(docs.len(), 1, "the rendered summary must be skipped");
        assert_eq!(docs[0].ref_id, "m00001");
    }

    #[test]
    fn a_folded_message_names_its_owning_block() {
        let mut state = crate::agent::bigctx::create_initial_state();
        state
            .message_refs
            .by_raw
            .insert("t1.0".into(), "m00001".into());
        state
            .message_refs
            .by_ref
            .insert("m00001".into(), "t1.0".into());
        state.blocks.push(crate::agent::bigctx::types::CompressionBlock {
            block_id: "b7".into(),
            run_id: "r1".into(),
            tier: Tier::Two,
            topic: None,
            start_ref: None,
            end_ref: None,
            summary: "s".into(),
            direct_message_ids: vec!["t1.0".into()],
            effective_message_ids: vec!["t1.0".into()],
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
        let history = vec![
            CoreMessage::new("t1.0", MessageRole::User, MessageContentType::Text)
                .with_text("folded away"),
        ];
        let docs = message_docs(&history, &state);
        assert_eq!(docs[0].block_id.as_deref(), Some("b7"));
        assert_eq!(docs[0].tier, 2);
    }

    #[test]
    fn a_message_without_a_ref_is_skipped() {
        let state = crate::agent::bigctx::create_initial_state();
        let history = vec![
            CoreMessage::new("t9.9", MessageRole::User, MessageContentType::Text)
                .with_text("no ref assigned"),
        ];
        assert!(message_docs(&history, &state).is_empty());
    }
}
