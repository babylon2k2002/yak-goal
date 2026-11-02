//! Substring counting — the original baseline.
//!
//! Exact, lowercased substring occurrence counts. Predictable, blind to
//! morphology, typos and CJK word boundaries. Kept as the deterministic
//! reference: when `hybrid` ranks something unexpectedly, comparing against
//! `substring` says whether the surprise came from stem/bigram normalization or
//! from the corpus itself.

use crate::agent::bigctx::search::doc_cache::doc_features;
use crate::agent::bigctx::search::types::{SearchAlgorithm, SearchDoc, ScoredBlock};

pub struct Substring;

impl SearchAlgorithm for Substring {
    fn name(&self) -> &'static str {
        "substring"
    }

    fn description(&self) -> &'static str {
        "Exact substring counting (original baseline). Predictable, no normalization."
    }

    fn score(&self, docs: &[SearchDoc], query: &str) -> Vec<ScoredBlock> {
        let terms: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .filter(|t| !t.is_empty())
            .map(str::to_string)
            .collect();
        if terms.is_empty() {
            return docs
                .iter()
                .map(|d| ScoredBlock {
                    ref_id: d.ref_id.clone(),
                    score: 0.0,
                })
                .collect();
        }
        docs.iter()
            .map(|doc| {
                let haystack = &doc_features(&doc.text).lower;
                let mut score = 0u32;
                for term in &terms {
                    score += count_occurrences(haystack, term) as u32;
                }
                ScoredBlock {
                    ref_id: doc.ref_id.clone(),
                    score: score as f64,
                }
            })
            .collect()
    }
}

/// Non-overlapping occurrence count. An empty needle matches nothing, so a
/// blank term cannot inflate every doc equally.
fn count_occurrences(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }
    haystack.matches(needle).count()
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
        Substring
            .score(docs, query)
            .into_iter()
            .find(|s| s.ref_id == id)
            .map(|s| s.score)
            .unwrap_or(0.0)
    }

    #[test]
    fn it_counts_every_occurrence() {
        let docs = [doc("a", "cache the cache in cache")];
        assert_eq!(score_of(&docs, "cache", "a"), 3.0);
    }

    #[test]
    fn matching_is_case_insensitive() {
        let docs = [doc("a", "Cache")];
        assert_eq!(score_of(&docs, "cache", "a"), 1.0);
    }

    #[test]
    fn it_is_blind_to_stemming() {
        // The documented limitation: "compressed" is not "compression"
        let docs = [doc("a", "we compressed it")];
        assert_eq!(score_of(&docs, "compression", "a"), 0.0);
    }

    #[test]
    fn an_empty_query_scores_zero_everywhere() {
        let docs = [doc("a", "anything")];
        assert_eq!(score_of(&docs, "   ", "a"), 0.0);
    }

    #[test]
    fn cjk_substrings_match_when_exact() {
        let docs = [doc("a", "身份验证流程")];
        assert_eq!(score_of(&docs, "身份验证", "a"), 1.0);
    }
}
