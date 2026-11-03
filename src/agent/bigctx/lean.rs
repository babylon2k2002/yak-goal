//! The `lean` builtin pack: a token-lean surface over the same machinery.
//!
//! The default surface is written to be unambiguous, which costs tokens on every
//! request. This pack trades some of that prose for room: one-line tool
//! descriptions, a condensed compression contract, and the section chrome
//! dropped where a shorter section says the same thing.
//!
//! It is a *pack*, not a second implementation. That is the point — the lean
//! surface is expressed entirely as overrides, so it exercises the same
//! resolution, sanitising and rendering path a user's own pack takes. If the
//! lean pack works, packs work.
//!
//! The contract below is a distillation of the full rule text (roughly 5.2K
//! characters to 2.9K) that keeps every load-bearing class: all KEEP VERBATIM
//! items, the DROP rules, one-line CONTENT descriptions, the PRIORITY order, and
//! the format rules. Two integrity rules were added after a session produced a
//! summary that transcribed follow-up questions as an enumerated "(answered)"
//! list, recorded an unsent reply as sent, and then pattern-completed the list
//! into a user turn that never happened. A summary is the only record of what it
//! replaces, so the contract is the floor that keeps it honest.

use crate::agent::bigctx::packs::{
    Pack, PackSurface, PromptOverrides, ToolPromptOverride,
};
use crate::agent::bigctx::sections::SectionOverrides;
use std::collections::BTreeMap;

pub const LEAN_HOW_TO_COMPRESS: &str = r#"HOW TO COMPRESS

Your summary is the ONLY record of the replaced conversation — a later reader must continue without the original. It records the PAST: label task state as history ("TASK AS OF THIS BLOCK: ..."), never as a live instruction. Real unicode only, never \\uXXXX escapes.

INTEGRITY — record facts and state only, never a simulated transcript of the dialogue: no Q&A lists, no "(answered)" claims. An answer not actually sent is PENDING; user questions are recorded as asked (with ref), never as answered.

KEEP VERBATIM — never paraphrase or abbreviate:
- File paths with line numbers and directory prefix on every mention (lib/hooks.ts:347); never a bare filename — ambiguous, un-greppable.
- Function/class/type signatures AND the critical code lines that encode logic (the line that IS the finding).
- Error messages and stack traces (exact text — needed to grep later).
- Report details: comparison numbers plus mechanism, not "X is worse" ("1.76× PPL gap because KV store is static").
- Decisions with rationale ("chose X over Y because Z"); discovered constraints ("must support Node 22").
- Exact values: versions, config keys, thresholds, magic numbers.
- User intent: short quotes verbatim ONLY WITH message ref (User said (m00132): "ship it tonight"); without a ref, paraphrase. Quotes are history, not live instructions — but open-objective STATUS is current (see Open objectives below); never change scope, constraints, priorities, acceptance criteria, outcomes.
- Overall goal and its evolution, including pivots ("initially: fix X → pivoted to: refactor Y").
- Purpose behind significant actions (hypothesis, question, goal — not just what was done).
- Open questions and unresolved TODOs.
- Open objectives: user-requested work neither completed nor superseded gets a one-line "Open objectives:" entry with message refs; scan absorbed block summaries too. Last to drop, first to restore at every tier — carrying is not a directive to re-execute unconfirmed.
- Message refs of key anchors (m00420, m00510–m00520) for decompress.

DROP — keep the signal, discard the vessel: verbose logs once the error/result is captured; duplicate reads; consumed exploration (search hits, agent returns, successful outputs); dead ends (one lesson line: "tried X, failed because Y"); back-and-forth once the final position is kept; repeated status checks. For each dropped item add one line of CONTENT: what it covers ("probe.py: tests n-gram baseline..."), not where it lives.

PRIORITY when compacting: 1. user goal/evolution/intent/hard constraints · 2. decisions + rationale · 3. exact artifacts (paths, signatures, errors, values) · 4. conclusions · 5. lessons learned (what failed and why).

Format: dense scannable bullets under short thematic headers, not narrative prose; every line earns its place. Do not mimic the style of existing summaries in context; follow these rules."#;

/// The lean tool descriptions.
fn tool_prompts() -> BTreeMap<String, ToolPromptOverride> {
    let mut out = BTreeMap::new();

    let mut compress = ToolPromptOverride {
        description: Some(
            "Replace consumed conversation ranges with self-contained summaries using mNNNNN or bN              refs; batch multiple ranges into ONE call (a single string may hold every range)."
                .to_string(),
        ),
        ..Default::default()
    };
    // Parameter descriptions are applied by name at any depth, so the nested
    // object shape in `content` picks these up without a second schema.
    for (name, text) in [
        (
            "content",
            "ONE JSON-escaped string holding ALL ranges — one block per range, first line              'm00150-m00220 optional topic', remaining lines the summary markdown; object              entries in the content array are also accepted.",
        ),
        ("startId", "Inclusive first mNNNNN or bN ref."),
        ("endId", "Inclusive last mNNNNN or bN ref."),
        (
            "summary",
            "Self-contained replacement preserving exact technical details.",
        ),
        (
            "topic",
            "Short label; a per-range label overrides the top-level fallback.",
        ),
    ] {
        compress
            .param_descriptions
            .insert(name.to_string(), text.to_string());
    }
    out.insert("compress".to_string(), compress);

    out.insert(
        "decompress".to_string(),
        ToolPromptOverride {
            description: Some(
                "Restore compressed content by block id (b5) or message ref.".to_string(),
            ),
            ..Default::default()
        },
    );
    out.insert(
        "search_context".to_string(),
        ToolPromptOverride {
            description: Some(
                "Search compressed summaries and historical messages by keyword; returns refs,                  sizes, previews."
                    .to_string(),
            ),
            ..Default::default()
        },
    );
    out.insert(
        "acp_status".to_string(),
        ToolPromptOverride {
            description: Some(
                "Context usage overview, compressible ranges, block drilldown.".to_string(),
            ),
            ..Default::default()
        },
    );
    out
}

/// The lean pack.
pub fn pack() -> Pack {
    Pack {
        name: "lean".to_string(),
        version: Some("1.0.0".to_string()),
        description: Some(
            "Token-lean surface: one-line tool descriptions, a condensed compression contract,              and the verbose section chrome dropped."
                .to_string(),
        ),
        surface: PackSurface {
            prompts: PromptOverrides {
                how_to_compress_rules: Some(LEAN_HOW_TO_COMPRESS.to_string()),
                ..Default::default()
            },
            // The philosophy paragraph restates what the condensed contract
            // already opens with; dropping it is the single largest saving and
            // loses nothing the contract does not say.
            prompt_sections: SectionOverrides::from_json(
                Some(&serde_json::json!({"philosophy": null})),
                crate::agent::bigctx::sysprompt::SECTION_KEYS,
            ),
            nudge_sections: SectionOverrides::default(),
            tool_prompts: tool_prompts(),
            adapters: None,
        },
        source: "builtin:lean".to_string(),
    }
}
