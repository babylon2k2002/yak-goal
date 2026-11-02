//! BM25 with stemming and CJK bigram tokenization.
//!
//! `k1=1.2`, `b=0.75` — standard IR constants. IDF down-weights terms common
//! across the corpus; length normalization keeps a long summary from
//! outranking a short one by raw term count alone.
//!
//! This is the precision channel. On the upstream's mixed EN/CJK benchmark it
//! scored MRR 0.833 against substring's 0.797, better on every metric in
//! isolation, and it is the 0.7-weighted component of the hybrid default.

use crate::agent::bigctx::search::doc_cache::doc_features;
use crate::agent::bigctx::search::tokenizer::tokenize;
use crate::agent::bigctx::search::types::{SearchAlgorithm, SearchDoc, ScoredBlock};

pub struct Bm25;

impl SearchAlgorithm for Bm25 {
    fn name(&self) -> &'static str {
        "bm25"
    }

    fn description(&self) -> &'static str {
        "BM25 with stemming + CJK bigram tokenization. IR-standard relevance ranking."
    }

    fn score(&self, docs: &[SearchDoc], query: &str) -> Vec<ScoredBlock> {
        let n = docs.len();
        if n == 0 {
            return Vec::new();
        }
        const K1: f64 = 1.2;
        const B: f64 = 0.75;

        let features: Vec<_> = docs.iter().map(|d| doc_features(&d.text)).collect();
        let total_len: f64 = features.iter().map(|f| f.len as f64).sum();
        let avgdl = if n > 0 { total_len / n as f64 } else { 0.0 };

        let query_terms = tokenize(query, true);
        if query_terms.is_empty() {
            return docs
                .iter()
                .map(|d| ScoredBlock {
                    ref_id: d.ref_id.clone(),
                    score: 0.0,
                })
                .collect();
        }

        // Document frequency per distinct query term.
        let mut idf: std::collections::BTreeMap<&str, f64> = std::collections::BTreeMap::new();
        for term in &query_terms {
            if idf.contains_key(term.as_str()) {
                continue;
            }
            let df = features
                .iter()
                .filter(|f| f.tf.contains_key(term.as_str()))
                .count() as f64;
            let value = (1.0 + (n as f64 - df + 0.5) / (df + 0.5)).ln();
            idf.insert(term.as_str(), value);
        }

        docs.iter()
            .zip(features.iter())
            .map(|(doc, f)| {
                let mut score = 0.0;
                for term in &query_terms {
                    let Some(&tf) = f.tf.get(term.as_str()) else {
                        continue;
                    };
                    let tf = tf as f64;
                    let idf_t = idf.get(term.as_str()).copied().unwrap_or(0.0);
                    let norm = 1.0 - B + B * (f.len as f64) / if avgdl > 0.0 { avgdl } else { 1.0 };
                    score += idf_t * (tf * (K1 + 1.0)) / (tf + K1 * norm);
                }
                ScoredBlock {
                    ref_id: doc.ref_id.clone(),
                    score,
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
        Bm25
            .score(docs, query)
            .into_iter()
            .find(|s| s.ref_id == id)
            .map(|s| s.score)
            .unwrap_or(0.0)
    }

    #[test]
    fn a_matching_doc_outranks_a_non_matching_one() {
        let docs = [doc("a", "database migration plan"), doc("b", "unrelated chatter")];
        assert!(score_of(&docs, "migration", "a") > score_of(&docs, "migration", "b"));
    }

    #[test]
    fn stemming_matches_morphology() {
        let docs = [doc("a", "we compressed the context")];
        // "compression" must reach "compressed"
        assert!(score_of(&docs, "compression", "a") > 0.0);
    }

    #[test]
    fn cjk_queries_match_cjk_docs() {
        let docs = [doc("a", "身份验证流程说明")];
        assert!(score_of(&docs, "身份验证", "a") > 0.0);
    }

    #[test]
    fn an_empty_query_scores_everything_zero() {
        let docs = [doc("a", "text")];
        assert_eq!(score_of(&docs, "", "a"), 0.0);
    }

    #[test]
    fn a_rare_term_outweighs_a_common_one() {
        // "cache" appears everywhere; "reentrant" in one doc only.
        let docs = [
            doc("a", "cache cache cache"),
            doc("b", "cache reentrant"),
        ];
        // The rare term must pull b above a for query "reentrant".
        assert!(score_of(&docs, "reentrant", "b") > score_of(&docs, "reentrant", "a"));
    }
}
