//! What the compression actually cost.
//!
//! A fold looks like a win: 15K tokens of tool output replaced by a 90-token
//! summary. But it also **breaks the provider's prompt cache** — everything
//! from the fold's first divergence point onwards is re-billed at full price on
//! the next request, and the tokens it "saved" were often cached reads at a
//! tenth the price anyway. Without measuring that, "we saved 15K tokens" is a
//! number that can be true and still be a loss.
//!
//! ## The ledger closes exactly
//!
//! Every request reports `input` and `cached`, so the miss is
//! `input - cached`. This module splits that miss into three buckets that **add
//! up to it by construction**, with no estimate participating:
//!
//! ```text
//! missed = newContent + compRepay + ttlRepay
//! ```
//!
//! - **newContent** — tokens genuinely appended since the previous request.
//!   Not an invalidation at all; clamped to the observed growth.
//! - **compRepay** — the re-payment a fold forced, by moving the prefix
//!   divergence point. Clamped to both the remaining miss and the structural
//!   excess the fold implies.
//! - **ttlRepay** — whatever is left: cache expiry, provider eviction, or a
//!   wire-level rewrite this module cannot see. Named honestly rather than
//!   guessed at.
//!
//! Because the last bucket is "the remainder", the sum is exact for any input.
//! That is the point: a report that does not reconcile is a report that lies
//! somewhere, and the lying part is always the part you needed.

use crate::agent::bigctx::tokenize::default_count_tokens;

/// One request's usage, as the provider reported it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheSample {
    /// Total prompt tokens billed. `cached` is INCLUDED (providers differ on
    /// this; normalise before recording).
    pub input: u64,
    /// Cache-hit tokens the provider reported.
    pub cached: u64,
    pub output: u64,
}

impl CacheSample {
    pub fn new(input: u64, cached: u64, output: u64) -> CacheSample {
        CacheSample {
            input,
            cached,
            output,
        }
    }

    /// Tokens billed at full price.
    pub fn missed(&self) -> u64 {
        self.input.saturating_sub(self.cached)
    }

    /// Cache-hit share in whole percent.
    pub fn hit_pct(&self) -> u64 {
        if self.input == 0 {
            return 0;
        }
        (self.cached as f64 / self.input as f64 * 100.0).round() as u64
    }
}

/// A fold, as the cache accounting sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoldEvent {
    /// Tokens the fold removed from the view.
    pub tokens_compressed: u64,
    /// Tokens the summary itself costs.
    pub summary_tokens: u64,
    /// Token offset of the fold's earliest divergence point in the post-fold
    /// view. Used only as a clamp bound, never inside the closure.
    pub first_fold_start_tokens: Option<u64>,
    /// Post-fold view size.
    pub view_after: Option<u64>,
}

impl FoldEvent {
    pub fn new(tokens_compressed: u64, summary_tokens: u64) -> FoldEvent {
        FoldEvent {
            tokens_compressed,
            summary_tokens,
            first_fold_start_tokens: None,
            view_after: None,
        }
    }

    /// The divergence point this fold implies.
    fn divergence(&self) -> u64 {
        self.first_fold_start_tokens
            .or(self.view_after)
            .unwrap_or(0)
    }
}

/// How one request's miss splits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Decomposition {
    pub missed: u64,
    pub new_content: u64,
    pub comp_repay: u64,
    pub ttl_repay: u64,
    /// Which pending fold was credited, by index.
    pub fold_index: Option<usize>,
}

impl Decomposition {
    /// The closure holds: the three buckets sum to the miss.
    pub fn balances(&self) -> bool {
        self.new_content + self.comp_repay + self.ttl_repay == self.missed
    }
}

/// Split one request's miss into the three buckets.
///
/// `pending_folds` are the folds applied after `prev` and no later than `cur`.
pub fn decompose(
    prev: Option<&CacheSample>,
    cur: &CacheSample,
    pending_folds: &[FoldEvent],
) -> Decomposition {
    let missed = cur.missed();
    // Growth over the previous request is the ceiling on how much of the miss
    // is legitimate new content: you cannot have appended more than the request
    // grew.
    let growth = prev.map_or(0, |p| cur.input.saturating_sub(p.input));
    let new_content = missed.min(growth);
    let remainder = missed - new_content;

    let mut comp_repay = 0;
    let mut fold_index = None;

    if remainder > 0 && !pending_folds.is_empty() {
        // A combined rewrite diverges at the EARLIEST fold start point: past
        // that offset nothing in the prefix can still match.
        let mut best = u64::MAX;
        let mut owner = 0usize;
        for (index, fold) in pending_folds.iter().enumerate() {
            let divergence = fold.divergence();
            if divergence < best {
                best = divergence;
                owner = index;
            }
        }
        // Everything after the divergence point must be re-paid, minus the
        // legitimate fresh append that also lives out there.
        let structural_excess = cur.input.saturating_sub(best).saturating_sub(new_content);
        comp_repay = remainder.min(structural_excess);
        if comp_repay > 0 {
            fold_index = Some(owner);
        }
    }

    Decomposition {
        missed,
        new_content,
        comp_repay,
        ttl_repay: remainder - comp_repay,
        fold_index,
    }
}

/// Running totals, reconciled.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CacheTotals {
    pub requests: usize,
    pub input: u64,
    pub cached: u64,
    pub output: u64,
    pub new_content: u64,
    pub comp_repay: u64,
    pub ttl_repay: u64,
}

impl CacheTotals {
    pub fn hit_pct(&self) -> u64 {
        if self.input == 0 {
            return 0;
        }
        (self.cached as f64 / self.input as f64 * 100.0).round() as u64
    }

    /// `input - cached - (new + comp + ttl)`. Zero by construction; a non-zero
    /// value means the accounting has a bug, which is worth asserting rather
    /// than discovering in a bill.
    pub fn residual(&self) -> i64 {
        let accounted = self.new_content + self.comp_repay + self.ttl_repay;
        self.input as i64 - self.cached as i64 - accounted as i64
    }

    pub fn balances(&self) -> bool {
        self.residual() == 0
    }

    /// Share of the miss caused by folding, in whole percent.
    pub fn comp_repay_pct(&self) -> u64 {
        let missed = self.input.saturating_sub(self.cached);
        if missed == 0 {
            return 0;
        }
        (self.comp_repay as f64 / missed as f64 * 100.0).round() as u64
    }
}

/// Whether a fold paid for itself.
#[derive(Debug, Clone, PartialEq)]
pub struct FoldEconomics {
    pub seq: usize,
    /// Tokens the fold removed.
    pub saved: u64,
    /// Tokens the summary costs.
    pub summary: u64,
    /// Requests observed after this fold.
    pub requests_after: usize,
    /// Tokens re-paid because of this fold.
    pub repay: u64,
    /// Raw token delta: `repay + summary - saved`. Negative means the fold
    /// removed more tokens than it forced us to send again.
    pub net_token_delta: i64,
    /// How many requests it takes for the per-request saving to cover the
    /// one-time re-payment. `None` when the fold saves nothing per request.
    pub breakeven_turns: Option<f64>,
    /// True once enough requests have gone by to cover the cost.
    pub paid_back: Option<bool>,
    /// Tokens not billed so far, by this fold's per-request saving.
    pub saved_so_far: u64,
}

/// Cache read is ~1/10 the price of a fresh input token on the providers yak
/// targets; a summary is generated, so it costs like output (~4× input).
const PRICE_WRITE: f64 = 1.0;
const PRICE_READ: f64 = 0.1;
const PRICE_OUTPUT: f64 = 4.0;

/// Work out whether each fold paid for itself.
pub fn fold_economics(folds: &[(FoldEvent, usize, u64)]) -> Vec<FoldEconomics> {
    folds
        .iter()
        .enumerate()
        .map(|(index, (fold, requests_after, repay))| {
            let saved = fold.tokens_compressed;
            let summary = fold.summary_tokens;
            let saving_per_turn = saved.saturating_sub(summary);

            // One-time cost in input-token-equivalent units: the re-payment
            // costs the difference between a cache write and a read, the
            // summary costs output price, and the removed tokens would have
            // been cheap cached reads.
            let one_time_cost = (PRICE_WRITE - PRICE_READ) * *repay as f64
                + PRICE_OUTPUT * summary as f64
                - PRICE_READ * saved as f64;
            let per_turn_saving = saving_per_turn as f64 * PRICE_READ;

            let breakeven = (per_turn_saving > 0.0)
                .then(|| one_time_cost.max(0.0) / per_turn_saving);

            FoldEconomics {
                seq: index + 1,
                saved,
                summary,
                requests_after: *requests_after,
                repay: *repay,
                net_token_delta: *repay as i64 + summary as i64 - saved as i64,
                breakeven_turns: breakeven,
                paid_back: breakeven.map(|n| *requests_after as f64 >= n),
                saved_so_far: saving_per_turn * *requests_after as u64,
            }
        })
        .collect()
}

/// The whole picture, ready to print.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheReport {
    pub totals: CacheTotals,
    pub folds: Vec<FoldEconomics>,
    /// Folds whose re-payment was measured against a following request.
    pub measured_folds: usize,
    /// Folds that have not yet paid back.
    pub unpaid_folds: usize,
}

impl CacheReport {
    /// True when the ledger closes. A false here is a bug, not a warning.
    pub fn balances(&self) -> bool {
        self.totals.balances()
    }

    /// True when folding is costing more than it saves.
    ///
    /// The signal worth acting on: if most folds never pay back, the session
    /// would be cheaper with compression off.
    pub fn compression_is_losing(&self) -> bool {
        self.measured_folds > 0 && self.unpaid_folds * 2 > self.measured_folds
    }
}

/// Build the report from a run's samples and folds.
///
/// `folds` carries each fold with the number of requests observed after it and
/// the tokens attributed to it.
pub fn build_cache_report(
    samples: &[CacheSample],
    folds: &[(FoldEvent, usize, u64)],
) -> CacheReport {
    let mut totals = CacheTotals {
        requests: samples.len(),
        ..Default::default()
    };

    // Folds waiting to be charged. A fold lands in the gap between two
    // requests, so the first request after it pays the re-payment.
    //
    // `folds` carries each fold with the number of requests that followed it,
    // which lets the timeline be reconstructed: fold k happened after
    // `samples.len() - requests_after` requests.
    let mut pending: Vec<FoldEvent> = Vec::new();
    for (index, sample) in samples.iter().enumerate() {
        // Everything folded before this request and not yet charged.
        for (fold, requests_after, _) in folds {
            let happened_before = samples.len().saturating_sub(*requests_after);
            if happened_before == index {
                pending.push(*fold);
            }
        }

        totals.input += sample.input;
        totals.cached += sample.cached;
        totals.output += sample.output;

        let prev = index.checked_sub(1).map(|i| &samples[i]);
        let decomposition = decompose(prev, sample, &pending);
        totals.new_content += decomposition.new_content;
        totals.comp_repay += decomposition.comp_repay;
        totals.ttl_repay += decomposition.ttl_repay;

        // A fold is credited to the first sample after it and no further: the
        // re-payment happens once, on the request whose prefix moved.
        if decomposition.fold_index.is_some() {
            pending.clear();
        }
    }

    let economics = fold_economics(folds);
    let measured_folds = economics
        .iter()
        .filter(|f| f.breakeven_turns.is_some() && f.requests_after > 0)
        .count();
    let unpaid_folds = economics
        .iter()
        .filter(|f| f.paid_back == Some(false))
        .count();

    CacheReport {
        totals,
        folds: economics,
        measured_folds,
        unpaid_folds,
    }
}

/// Render the report.
pub fn render_cache_report(report: &CacheReport) -> String {
    let t = &report.totals;
    let mut out = String::new();

    out.push_str(&format!(
        "requests: {}  ·  input {}  ·  cached {} ({:.0}%)\n",
        t.requests,
        format_tokens(t.input),
        format_tokens(t.cached),
        t.hit_pct() as f64
    ));

    let missed = t.input.saturating_sub(t.cached);
    out.push_str(&format!(
        "miss split: {} new content · {} compression repay ({:.0}% of the miss) · {} unattributed\n",
        format_tokens(t.new_content),
        format_tokens(t.comp_repay),
        t.comp_repay_pct() as f64,
        format_tokens(t.ttl_repay)
    ));

    if report.folds.is_empty() {
        out.push_str("\nno folds — nothing has been compressed.\n");
        return out.trim_end().to_string();
    }

    out.push_str(&format!("\nfolds ({}):\n", report.folds.len()));
    for fold in &report.folds {
        out.push_str(&format!(
            "  #{}  {}→{}  repaid {}  ",
            fold.seq,
            format_tokens(fold.saved),
            format_tokens(fold.summary),
            format_tokens(fold.repay)
        ));
        match (fold.breakeven_turns, fold.paid_back) {
            (Some(n), Some(true)) => out.push_str(&format!(
                "paid back in {:.1} request(s) ({} seen)\n",
                n, fold.requests_after
            )),
            (Some(n), Some(false)) => out.push_str(&format!(
                "NOT paid back: needs {:.1} request(s), {} seen\n",
                n, fold.requests_after
            )),
            _ => out.push_str("no per-request saving to pay back with\n"),
        }
    }

    if report.compression_is_losing() {
        out.push_str(&format!(
            "\n⚠ {} of {} folds have not paid back — this session may be cheaper \
             with compression off.\n",
            report.unpaid_folds, report.measured_folds
        ));
    }

    if !report.balances() {
        out.push_str(&format!(
            "\n⚠ the ledger does not balance (residual {}). This is an accounting \
             bug, not a provider quirk.\n",
            t.residual()
        ));
    }

    out.trim_end().to_string()
}

fn format_tokens(n: u64) -> String {
    if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

/// Token count of a summary, for the fold record.
pub fn summary_tokens(text: &str) -> u64 {
    default_count_tokens(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(input: u64, cached: u64) -> CacheSample {
        CacheSample::new(input, cached, 100)
    }

    // ── the ledger ──

    #[test]
    fn the_ledger_closes_on_a_single_request() {
        let samples = vec![sample(10_000, 8_000)];
        let report = build_cache_report(&samples, &[]);
        assert!(
            report.balances(),
            "residual {}",
            report.totals.residual()
        );
    }

    #[test]
    fn the_ledger_closes_across_growth() {
        let samples = vec![sample(10_000, 8_000), sample(12_000, 9_000)];
        let report = build_cache_report(&samples, &[]);
        assert!(report.balances(), "residual {}", report.totals.residual());
    }

    #[test]
    fn the_ledger_closes_with_a_fold() {
        let samples = vec![sample(20_000, 18_000), sample(6_000, 1_000)];
        let folds = vec![(
            FoldEvent {
                tokens_compressed: 15_000,
                summary_tokens: 90,
                first_fold_start_tokens: Some(4_000),
                view_after: Some(5_000),
            },
            1,
            5_000,
        )];
        let report = build_cache_report(&samples, &folds);
        assert!(report.balances(), "residual {}", report.totals.residual());
    }

    #[test]
    fn the_ledger_closes_with_wild_numbers() {
        // The closure must hold for ANY input, not just plausible ones.
        let cases = [
            vec![sample(0, 0)],
            vec![sample(1_000, 999)],
            vec![sample(1_000, 0), sample(500, 400)],
            vec![sample(100_000, 99_999), sample(1, 0)],
            vec![sample(5_000, 0), sample(5_000, 5_000), sample(9_000, 1)],
        ];
        for (i, samples) in cases.iter().enumerate() {
            let report = build_cache_report(samples, &[]);
            assert!(
                report.balances(),
                "case {i}: residual {}",
                report.totals.residual()
            );
        }
    }

    #[test]
    fn growth_is_attributed_to_new_content_not_to_a_fold() {
        // Appending a message is not an invalidation.
        let samples = vec![sample(10_000, 9_000), sample(12_000, 9_000)];
        let report = build_cache_report(&samples, &[]);
        assert_eq!(report.totals.new_content, 2_000, "the appended tokens");
        assert_eq!(report.totals.comp_repay, 0, "nothing was folded");
    }

    #[test]
    fn a_fold_takes_the_blame_for_the_miss_it_causes() {
        // 20K cached down to 6K with 1K cached: the prefix moved.
        let samples = vec![sample(20_000, 19_000), sample(6_000, 1_000)];
        let folds = vec![(
            FoldEvent {
                tokens_compressed: 15_000,
                summary_tokens: 90,
                first_fold_start_tokens: Some(4_000),
                view_after: Some(5_000),
            },
            1,
            5_000,
        )];
        let report = build_cache_report(&samples, &folds);
        assert!(
            report.totals.comp_repay > 0,
            "the fold must be charged for the re-payment"
        );
    }

    #[test]
    fn a_miss_with_no_fold_pending_is_unattributed() {
        let samples = vec![sample(10_000, 1_000)];
        let report = build_cache_report(&samples, &[]);
        assert_eq!(report.totals.comp_repay, 0);
        assert!(report.totals.ttl_repay > 0, "it has to land somewhere");
    }

    #[test]
    fn the_earliest_divergence_point_wins() {
        let prev = sample(20_000, 19_000);
        let cur = sample(10_000, 1_000);
        let folds = [
            FoldEvent {
                tokens_compressed: 1_000,
                summary_tokens: 10,
                first_fold_start_tokens: Some(8_000),
                view_after: None,
            },
            FoldEvent {
                tokens_compressed: 1_000,
                summary_tokens: 10,
                first_fold_start_tokens: Some(2_000),
                view_after: None,
            },
        ];
        let d = decompose(Some(&prev), &cur, &folds);
        assert_eq!(d.fold_index, Some(1), "the earlier divergence owns it");
    }

    #[test]
    fn decomposition_always_balances() {
        let prev = sample(10_000, 5_000);
        for input in [0u64, 1, 5_000, 10_000, 100_000] {
            for cached in [0u64, 1, 2_500, 5_000, 99_999] {
                let cur = sample(input, cached);
                let d = decompose(Some(&prev), &cur, &[]);
                assert!(d.balances(), "input {input} cached {cached}: {d:?}");
            }
        }
    }

    // ── economics ──

    #[test]
    fn a_fold_that_removes_a_lot_pays_back() {
        // 15K replaced by 90 tokens, repaid once.
        let folds = vec![(
            FoldEvent::new(15_000, 90),
            10,   // ten requests followed
            5_000 // and the re-payment was 5K
        )];
        let economics = fold_economics(&folds);
        assert_eq!(economics[0].paid_back, Some(true));
    }

    #[test]
    fn a_fold_that_saves_nothing_per_request_cannot_pay_back() {
        // A summary as big as what it replaced saves nothing per turn.
        let folds = vec![(FoldEvent::new(100, 100), 50, 0)];
        let economics = fold_economics(&folds);
        assert_eq!(economics[0].breakeven_turns, None);
        assert_eq!(economics[0].paid_back, None, "unobservable, not false");
    }

    #[test]
    fn a_fold_that_was_repaid_enormously_does_not_pay_back() {
        let folds = vec![(
            FoldEvent::new(1_000, 50),
            2,        // only two requests
            50_000,   // but a huge re-payment
        )];
        let economics = fold_economics(&folds);
        assert_eq!(economics[0].paid_back, Some(false));
        assert!(economics[0].net_token_delta > 0, "it cost more than it saved");
    }

    #[test]
    fn net_token_delta_is_negative_for_a_good_fold() {
        let folds = vec![(FoldEvent::new(15_000, 90), 1, 5_000)];
        let economics = fold_economics(&folds);
        assert!(economics[0].net_token_delta < 0);
    }

    #[test]
    fn saved_so_far_grows_with_every_request() {
        let one = fold_economics(&[(FoldEvent::new(10_000, 100), 1, 0)]);
        let ten = fold_economics(&[(FoldEvent::new(10_000, 100), 10, 0)]);
        assert!(ten[0].saved_so_far > one[0].saved_so_far);
    }

    // ── the verdict ──

    #[test]
    fn mostly_unpaid_folds_means_compression_is_losing() {
        let folds = vec![
            (FoldEvent::new(1_000, 50), 1, 50_000),
            (FoldEvent::new(1_000, 50), 1, 50_000),
            (FoldEvent::new(15_000, 90), 20, 0),
        ];
        let report = build_cache_report(&[], &folds);
        assert!(report.compression_is_losing());
    }

    #[test]
    fn mostly_paid_folds_is_healthy() {
        let folds = vec![
            (FoldEvent::new(15_000, 90), 20, 0),
            (FoldEvent::new(15_000, 90), 20, 0),
            (FoldEvent::new(1_000, 50), 1, 50_000),
        ];
        let report = build_cache_report(&[], &folds);
        assert!(!report.compression_is_losing());
    }

    #[test]
    fn no_folds_is_not_a_verdict() {
        let report = build_cache_report(&[sample(1_000, 900)], &[]);
        assert!(!report.compression_is_losing(), "nothing to judge");
    }

    // ── rendering ──

    #[test]
    fn the_report_renders() {
        let samples = vec![sample(10_000, 8_000), sample(12_000, 9_000)];
        let report = build_cache_report(&samples, &[]);
        let text = render_cache_report(&report);
        assert!(text.contains("requests: 2"), "{text}");
        assert!(text.contains("miss split"), "{text}");
        assert!(text.contains("no folds"), "{text}");
    }

    #[test]
    fn the_report_shows_a_fold_row() {
        let folds = vec![(FoldEvent::new(15_000, 90), 10, 5_000)];
        let report = build_cache_report(&[], &folds);
        let text = render_cache_report(&report);
        assert!(text.contains("#1"), "{text}");
        assert!(text.contains("15.0K"), "{text}");
    }

    #[test]
    fn the_report_warns_when_folds_do_not_pay_back() {
        let folds = vec![(FoldEvent::new(1_000, 50), 1, 50_000)];
        let report = build_cache_report(&[], &folds);
        let text = render_cache_report(&report);
        assert!(text.contains("NOT paid back"), "{text}");
    }

    #[test]
    fn sample_helpers() {
        let s = CacheSample::new(1_000, 250, 10);
        assert_eq!(s.missed(), 750);
        assert_eq!(s.hit_pct(), 25);
        assert_eq!(CacheSample::new(0, 0, 0).hit_pct(), 0);
    }
}
