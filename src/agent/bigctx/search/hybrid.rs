//! Hybrid: normalized BM25(stem) + fuzzy bigram, weighted 0.7 / 0.3.
//!
//! BM25 supplies precision on real terms (morphology, IDF, length norm); fuzzy
//! supplies recall on typos, partials and cross-script text. Each component is
//! max-normalized to [0,1] before weighting, so their scales stay comparable
//! whatever the corpus size — otherwise BM25's unbounded scores would swamp
//! fuzzy's 0-1 overlap entirely.
//!
//! This is the default. Upstream benchmark over 32 blocks and 48 mixed EN/CJK
//! queries:
//!
//! ```text
//! substring  MRR 0.797  R@1 0.792  R@3 0.792
//! bm25       MRR 0.833  R@1 0.833  R@3 0.833
//! fuzzy      MRR 0.795  R@1 0.708  R@3 0.875
//! hybrid     MRR 0.898  R@1 0.875  R@3 0.917   <- best on every metric
//! ```
//!
//! The weight ratio is robust: anything from 0.6 to 0.8 for BM25 scores within
//! 0.001 MRR of this.

use crate::agent::bigctx::search::bm25::Bm25;
use crate::agent::bigctx::search::fuzzy::Fuzzy;
use crate::agent::bigctx::search::types::{SearchAlgorithm, SearchDoc, ScoredBlock};

pub struct Hybrid;

const W_BM25: f64 = 0.7;
const W_FUZZY: f64 = 0.3;

impl SearchAlgorithm for Hybrid {
    fn name(&self) -> &'static str {
        "hybrid"
    }

    fn description(&self) -> &'static str {
        "Weighted BM25(stem) + fuzzy bigram. Default — best precision + recall."
    }

    fn score(&self, docs: &[SearchDoc], query: &str) -> Vec<ScoredBlock> {
        let bm = Bm25.score(docs, query);
        let fz = Fuzzy.score(docs, query);

        // Fold, not `max()` over an iterator of f64 (which has no Ord).
        let max_bm = bm.iter().fold(1e-9_f64, |m, r| m.max(r.score));
        let max_fz = fz.iter().fold(1e-9_f64, |m, r| m.max(r.score));

        let bm_by_ref: std::collections::BTreeMap<&str, f64> =
            bm.iter().map(|r| (r.ref_id.as_str(), r.score / max_bm)).collect();
        let fz_by_ref: std::collections::BTreeMap<&str, f64> =
            fz.iter().map(|r| (r.ref_id.as_str(), r.score / max_fz)).collect();

        docs.iter()
            .map(|d| ScoredBlock {
                ref_id: d.ref_id.clone(),
                score: W_BM25 * bm_by_ref.get(d.ref_id.as_str()).copied().unwrap_or(0.0)
                    + W_FUZZY * fz_by_ref.get(d.ref_id.as_str()).copied().unwrap_or(0.0),
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
        Hybrid
            .score(docs, query)
            .into_iter()
            .find(|s| s.ref_id == id)
            .map(|s| s.score)
            .unwrap_or(0.0)
    }

    #[test]
    fn the_best_doc_scores_one() {
        // Both channels normalize to 1.0 for the top doc, so hybrid does too.
        let docs = [doc("a", "tokenizer splits words"), doc("b", "nothing here")];
        let s = score_of(&docs, "tokenizer", "a");
        assert!((s - 1.0).abs() < 1e-9, "got {s}");
    }

    #[test]
    fn a_typo_matches_through_the_fuzzy_channel() {
        let docs = [doc("a", "the tokenizer splits words")];
        assert!(score_of(&docs, "tokanizer", "a") > 0.0);
    }

    #[test]
    fn morphology_matches_through_the_bm25_channel() {
        let docs = [doc("a", "we compressed the context")];
        assert!(score_of(&docs, "compression", "a") > 0.0);
    }

    #[test]
    fn an_unrelated_doc_scores_zero() {
        let docs = [doc("a", "database migration"), doc("b", "zzzz")];
        assert_eq!(score_of(&docs, "database", "b"), 0.0);
    }

    #[test]
    fn every_doc_gets_exactly_one_score() {
        let docs = [doc("a", "x y"), doc("b", "p q"), doc("c", "m n")];
        let scored = Hybrid.score(&docs, "x");
        assert_eq!(scored.len(), 3);
        let refs: Vec<&str> = scored.iter().map(|s| s.ref_id.as_str()).collect();
        assert_eq!(refs, vec!["a", "b", "c"], "order must follow the input");
    }

    /// Retrieval quality against a corpus shaped like real folded history:
    /// many near-miss distractors, one true target, mixed EN/CJK.
    ///
    /// The distractors here deliberately do *not* repeat the query's terms — a
    /// doc that shares two of three terms and is shorter than the target will
    /// win on BM25 length normalization, and that is correct behaviour, not a
    /// bug to assert against. (Measured once while writing this test: a doc
    /// reading "we discussed login performance but not tokens" beat the real
    /// target. Retrieval has no notion of negation, and the upstream TS with
    /// the same k1=1.2 / b=0.75 does the same.)
    #[test]
    fn the_target_outranks_every_distractor() {
        let mut docs = vec![
            doc("target", "身份验证流程 was rewritten: the login flow now refreshes tokens"),
            doc("d1", "database migration plan for the billing schema"),
            doc("d2", "performance review of the checkout page"),
            doc("d3", "the tokenizer splits words into bigrams"),
        ];
        for i in 0..30 {
            docs.push(doc(&format!("noise{i}"), "unrelated chatter about nothing"));
        }

        // min_score 0 so the comparison sees every doc, including the ones a
        // real search would filter out.
        let opts = crate::agent::bigctx::search::SearchOptions {
            min_score: 0.0,
            limit: docs.len(),
            ..Default::default()
        };
        let ranked = crate::agent::bigctx::search::search(&docs, "authentication login tokens", &opts);
        assert_eq!(
            ranked.first().map(|r| r.ref_id.as_str()),
            Some("target"),
            "ranked: {:?}",
            ranked
                .iter()
                .take(5)
                .map(|r| (&r.ref_id, r.score))
                .collect::<Vec<_>>()
        );
        let target_score = ranked[0].score;
        for hit in &ranked[1..] {
            assert!(
                hit.score < target_score,
                "{} scored {} which is not below the target's {}",
                hit.ref_id,
                hit.score,
                target_score
            );
        }
    }

    /// The CJK side of the same question: a query with no verbatim overlap
    /// with the target text must still find it via bigrams. `验证实现` is not a
    /// substring of `重构了身份验证的实现方式`, but the bigrams 验证 and 实现 are.
    #[test]
    fn a_cjk_query_finds_its_target_without_verbatim_overlap() {
        let docs = [
            doc("target", "重构了身份验证的实现方式"),
            doc("other", "数据库迁移计划"),
            doc("noise", "今天天气很好"),
        ];
        let ranked = crate::agent::bigctx::search::search(
            &docs,
            "验证实现",
            &crate::agent::bigctx::search::SearchOptions::default(),
        );
        assert_eq!(ranked.first().map(|r| r.ref_id.as_str()), Some("target"));
    }

    /// The bigram fallback has a noise floor: unrelated text still shares
    /// short character bigrams with the query (`nothing` and `authentication`
    /// both contain `th` and `in`). Scores are max-normalized, so an absolute
    /// bound is meaningless — the only claim worth pinning is *relative*: an
    /// unrelated doc must stay far below a genuinely matching one.
    ///
    /// This is the accepted cost of replacing ICU segmentation with bigrams —
    /// recall over precision.
    #[test]
    fn unrelated_text_scores_far_below_a_real_match() {
        let docs = [
            doc("match", "authentication login tokens"),
            doc("noise", "unrelated chatter about nothing"),
        ];
        let matched = score_of(&docs, "authentication login tokens", "match");
        let noise = score_of(&docs, "authentication login tokens", "noise");
        assert!(noise > 0.0, "bigrams do overlap — the floor is not zero");
        assert!(
            noise < matched * 0.5,
            "noise {noise} must stay well below the real match {matched}"
        );
    }
}
