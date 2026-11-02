//! Fuzzy character-bigram matching (Jaccard-style overlap).
//!
//! Decomposes the query into character bigrams and measures how much of that
//! set each doc contains. Robust to typos (`tokan` ≈ `token`), partial words,
//! and identical across scripts — CJK gains the most, since bigrams need no
//! word boundary to exist.
//!
//! Query-term gate. CJK gets its own length rule and Latin is frozen:
//!
//! - `len >= 4` for any script — typo-tolerant matching needs a couple of
//!   characters before it means anything.
//! - `len >= 2 && CJK` — Chinese/Japanese/Korean words are mostly
//!   two-character atomic units (登录/缓存), so a Latin-style `>= 4` rule would
//!   lock the entire CJK query space out of this recall channel. Single CJK
//!   characters stay excluded: one character cannot form a bigram.
//!
//! This is the recall channel: weakest precision of any single algorithm, and
//! the 0.3-weighted component that lets the hybrid default survive typos.

use crate::agent::bigctx::search::doc_cache::doc_features;
use crate::agent::bigctx::search::tokenizer::{char_bigrams, is_cjk};
use crate::agent::bigctx::search::types::{SearchAlgorithm, SearchDoc, ScoredBlock};

pub struct Fuzzy;

/// Longest a query term may be before bigram overlap stops being meaningful:
/// a long term's bigram set is so specific that nothing overlaps, and the
/// comparison cost grows with it for no recall gain.
const MAX_TERM_CHARS: usize = 40;

impl SearchAlgorithm for Fuzzy {
    fn name(&self) -> &'static str {
        "fuzzy"
    }

    fn description(&self) -> &'static str {
        "Character bigram overlap. Typo-tolerant, script-agnostic, high recall."
    }

    fn score(&self, docs: &[SearchDoc], query: &str) -> Vec<ScoredBlock> {
        let zeroed = || {
            docs.iter()
                .map(|d| ScoredBlock {
                    ref_id: d.ref_id.clone(),
                    score: 0.0,
                })
                .collect::<Vec<_>>()
        };

        let query_terms: Vec<&str> = query
            .to_lowercase()
            .split([' ', ',', '\t', '\n'])
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .filter(|t| {
                let len = t.chars().count();
                len <= MAX_TERM_CHARS && (len >= 4 || (len >= 2 && t.chars().any(is_cjk)))
            })
            .collect();
        if query_terms.is_empty() {
            return zeroed();
        }

        let mut query_grams: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for term in &query_terms {
            for gram in char_bigrams(term) {
                query_grams.insert(gram);
            }
        }
        if query_grams.is_empty() {
            return zeroed();
        }

        let denom = query_grams.len() as f64;
        docs.iter()
            .map(|doc| {
                let features = doc_features(&doc.text);
                let hits = query_grams
                    .iter()
                    .filter(|g| features.grams.contains(g.as_str()))
                    .count();
                ScoredBlock {
                    ref_id: doc.ref_id.clone(),
                    score: hits as f64 / denom,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::bigctx::search::types::SearchDocKind;

    fn doc(id: &str, text: &str) -> SearchDoc {
        SearchDoc {
            kind: SearchDocKind::Message,
            ref_id: id.into(),
            text: text.into(),
            title: id.into(),
            role: None,
            block_id: None,
            tier: 1,
            tokens: 1,
        }
    }

    fn score_of(docs: &[SearchDoc], query: &str, id: &str) -> f64 {
        Fuzzy
            .score(docs, query)
            .into_iter()
            .find(|s| s.ref_id == id)
            .map(|s| s.score)
            .unwrap_or(0.0)
    }

    #[test]
    fn a_typo_still_matches() {
        let docs = [doc("a", "the tokenizer splits words")];
        // tokanizer: one wrong character, bigram overlap survives
        assert!(score_of(&docs, "tokanizer", "a") > 0.0);
    }

    #[test]
    fn short_latin_terms_are_gated_out() {
        let docs = [doc("a", "to of us")];
        // "to" is 2 chars and not CJK: no signal, so no score at all
        assert_eq!(score_of(&docs, "to", "a"), 0.0);
    }

    #[test]
    fn two_char_cjk_terms_are_admitted() {
        let docs = [doc("a", "缓存失效策略")];
        assert!(
            score_of(&docs, "缓存", "a") > 0.0,
            "a 2-char CJK word must reach the scorer"
        );
    }

    #[test]
    fn a_single_cjk_char_is_gated_out() {
        let docs = [doc("a", "验证")];
        // one char cannot form a bigram, nothing to compare
        assert_eq!(score_of(&docs, "验", "a"), 0.0);
    }

    #[test]
    fn an_unrelated_query_scores_zero() {
        let docs = [doc("a", "database migration")];
        assert_eq!(score_of(&docs, "zzzz", "a"), 0.0);
    }

    #[test]
    fn overlap_is_relative_to_the_query_not_the_doc() {
        // A doc containing every query bigram scores 1.0 regardless of length.
        let docs = [doc("a", "token")];
        let s = score_of(&docs, "token", "a");
        assert!(s > 0.99, "got {s}");
    }
}
