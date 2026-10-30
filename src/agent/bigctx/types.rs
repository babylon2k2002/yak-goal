//! The vocabulary the compression algorithm speaks.
//!
//! Deliberately NOT `crate::providers::Msg`: a `Msg` is a *turn*, while the
//! algorithm needs a *linear sequence of homogeneous entries*. One assistant
//! turn carrying reasoning, text and three tool calls is one `Msg` but four
//! wire entries with four different fates under compression, so the host
//! projects it into four [`CoreMessage`]s that share a base id.
//!
//! Base ids: a core id is `<base>` or `<base>#<discriminator>`. Coverage, and
//! therefore folding, is decided per base — a range that covers
//! `t12.3#r0` covers every projection of that message.

use serde::{Deserialize, Serialize};

/// Who authored a core.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MessageRole {
    User,
    Assistant,
    System,
    Tool,
}

/// What a core carries. Kept separate from [`MessageRole`] because the wire
/// role and the payload shape are independent: a tool result is `role: tool`
/// with `content_type: tool-result`, while a rendered block summary is
/// `role: system` with `content_type: text`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MessageContentType {
    Text,
    ToolCall,
    ToolResult,
    Reasoning,
}

/// One entry of the linear view the algorithm operates on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoreMessage {
    /// Stable, never re-issued. `<base>` or `<base>#<disc>`; see the module doc.
    pub id: String,
    pub role: MessageRole,
    pub content_type: MessageContentType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Set on `tool-call` (the called tool) and `tool-result` (the answering
    /// tool), so a fold can pair them and a status report can name them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// The `tool_call_id` pairing a call with its result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl CoreMessage {
    pub fn new(id: impl Into<String>, role: MessageRole, content_type: MessageContentType) -> Self {
        Self {
            id: id.into(),
            role,
            content_type,
            text: None,
            tool_name: None,
            tool_call_id: None,
        }
    }

    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    pub fn with_tool(mut self, name: impl Into<String>, call_id: impl Into<String>) -> Self {
        self.tool_name = Some(name.into());
        self.tool_call_id = Some(call_id.into());
        self
    }

    /// Base id: everything before the first `#`.
    pub fn base_id(&self) -> &str {
        base_id_of(&self.id)
    }

    /// The text this core contributes to the request, empty when it has none.
    pub fn text_or_empty(&self) -> &str {
        self.text.as_deref().unwrap_or("")
    }
}

/// Base id of any core id: everything before the first `#`. Sub-projections
/// (`base#callId`) are alternate renderings of one original message, so
/// coverage is decided per original.
pub fn base_id_of(id: &str) -> &str {
    match id.find('#') {
        Some(0) | None => id,
        Some(at) => &id[..at],
    }
}

/// Compression depth. A tier-1 block covers original messages; a tier-2 block
/// covers tier-1 blocks (their summaries); tier 3 covers tier 2.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
pub enum Tier {
    One,
    Two,
    Three,
}

impl Tier {
    pub fn number(self) -> u8 {
        match self {
            Tier::One => 1,
            Tier::Two => 2,
            Tier::Three => 3,
        }
    }

    pub fn from_number(n: u8) -> Option<Tier> {
        match n {
            1 => Some(Tier::One),
            2 => Some(Tier::Two),
            3 => Some(Tier::Three),
            _ => None,
        }
    }
}

impl TryFrom<u8> for Tier {
    type Error = String;

    fn try_from(n: u8) -> Result<Self, Self::Error> {
        Tier::from_number(n).ok_or_else(|| format!("tier must be 1, 2 or 3, got {n}"))
    }
}

impl From<Tier> for u8 {
    fn from(t: Tier) -> u8 {
        t.number()
    }
}

/// Whether a block's material is still recent enough to be worth keeping
/// detailed. Purely bookkeeping: a block becomes `Old` after surviving
/// `promotion_threshold` passes, which is the signal tier-2 distillation uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BlockGeneration {
    Young,
    Old,
}

/// One compressed range: the summary that replaced it, and the ids it owns.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompressionBlock {
    /// `b1`, `b2`, … — never re-issued.
    pub block_id: String,
    /// `r1`, `r2`, … — the compression call that created this block.
    pub run_id: String,
    pub tier: Tier,
    /// Optional label the model supplied, shown in the summary header.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topic: Option<String>,
    pub summary: String,
    /// The message ids this fold itself covered. Anchoring uses these (not the
    /// merged effective set) so a re-fold rewrites as little cached prefix as
    /// possible.
    pub direct_message_ids: Vec<String>,
    /// `direct_message_ids` plus every id inherited from consumed child blocks.
    /// This is what coverage tests against.
    pub effective_message_ids: Vec<String>,
    /// Blocks this fold consumed (tier-2/3 distillation).
    pub direct_block_ids: Vec<String>,
    /// Tokens of the material this block replaced, for reporting.
    pub compressed_tokens: u64,
    /// Unix seconds.
    pub created_at: u64,
    /// How many passes this block has survived; drives `generation`.
    pub survived_count: u32,
    pub generation: BlockGeneration,
    /// False once consumed by a parent fold or explicitly decompressed.
    pub active: bool,
    /// Set when the user explicitly decompressed this block: its inactive state
    /// is intentional and must not be undone by a later sync.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub expanded: bool,
    /// Wall time the summarizer took, for diagnostics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    /// The tool call that produced this block, so the call can be hidden from
    /// the model's view while the block stays.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compress_call_id: Option<String>,
}

impl CompressionBlock {
    /// True when this block still represents content the model should see.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Every id this block covers, base-normalised, for set membership tests.
    pub fn covered_bases(&self) -> impl Iterator<Item = &str> {
        self.effective_message_ids
            .iter()
            .map(|id| base_id_of(id))
    }
}

/// Raw id ↔ citation ref, both directions. Refs are assigned once and never
/// re-issued, so `by_raw` entries for compressed messages stay valid forever —
/// that is what makes `decompress` possible after the messages left the view.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MessageRefMap {
    pub by_raw: std::collections::BTreeMap<String, String>,
    pub by_ref: std::collections::BTreeMap<String, String>,
}

impl MessageRefMap {
    pub fn ref_for(&self, raw_id: &str) -> Option<&str> {
        self.by_raw.get(raw_id).map(String::as_str)
    }

    pub fn raw_for(&self, reference: &str) -> Option<&str> {
        self.by_ref.get(reference).map(String::as_str)
    }

    pub fn is_empty(&self) -> bool {
        self.by_raw.is_empty()
    }
}

/// Nudge cadence bookkeeping: the algorithm tells the model when it is worth
/// compressing, and must not repeat itself every turn.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NudgeState {
    /// Token count at the last injected nudge.
    pub last_nudge_shown_tokens: u64,
    /// Token count at the last per-message nudge.
    pub last_per_message_nudge_tokens: u64,
    /// Session baseline the growth thresholds measure against.
    pub baseline_tokens: u64,
    /// Per-tier token count at that tier's last injection.
    pub last_shown_by_tier: std::collections::BTreeMap<u8, u64>,
}

/// Cumulative counters shown by `acp_status`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CompressionStats {
    pub tokens_compressed: u64,
    pub compression_count: u64,
}

/// Everything the algorithm must remember across turns.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompressionState {
    pub blocks: Vec<CompressionBlock>,
    pub message_refs: MessageRefMap,
    /// First-render token count per ref. Written once, read forever, so the
    /// `<acp tokens=…>` tags do not churn and bust the provider prefix cache.
    pub token_snapshot: std::collections::BTreeMap<String, u64>,
    pub nudge: NudgeState,
    pub stats: CompressionStats,
    /// Next ids to hand out. Monotonic; never reset, so an issued id is never
    /// reused with different content.
    pub next_block_id: u64,
    pub next_run_id: u64,
    /// Next `mNNNNN` index.
    pub next_ref_index: u64,
}

impl Default for CompressionState {
    fn default() -> Self {
        crate::agent::bigctx::state::create_initial_state()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_id_strips_the_discriminator() {
        assert_eq!(base_id_of("t12.3"), "t12.3");
        assert_eq!(base_id_of("t12.3#r0"), "t12.3");
        assert_eq!(base_id_of("t12.3#call_abc"), "t12.3");
        // a leading '#' is not a discriminator: it would make the base empty
        assert_eq!(base_id_of("#x"), "#x");
    }

    #[test]
    fn tier_round_trips_through_serde() {
        for tier in [Tier::One, Tier::Two, Tier::Three] {
            let wire = serde_json::to_string(&tier).unwrap();
            assert_eq!(wire, tier.number().to_string());
            let back: Tier = serde_json::from_str(&wire).unwrap();
            assert_eq!(back, tier);
        }
    }

    #[test]
    fn a_bad_tier_number_is_rejected_not_wrapped() {
        assert!(serde_json::from_str::<Tier>("4").is_err());
        assert!(serde_json::from_str::<Tier>("0").is_err());
    }

    #[test]
    fn tier_ordering_follows_depth() {
        assert!(Tier::One < Tier::Two);
        assert!(Tier::Two < Tier::Three);
    }

    #[test]
    fn covered_bases_normalises_projections() {
        let block = CompressionBlock {
            block_id: "b1".into(),
            run_id: "r1".into(),
            tier: Tier::One,
            topic: None,
            summary: "s".into(),
            direct_message_ids: vec!["t1.1#r0".into()],
            effective_message_ids: vec!["t1.1#r0".into(), "t1.2".into()],
            direct_block_ids: vec![],
            compressed_tokens: 10,
            created_at: 0,
            survived_count: 0,
            generation: BlockGeneration::Young,
            active: true,
            expanded: false,
            duration_ms: None,
            compress_call_id: None,
        };
        let bases: Vec<&str> = block.covered_bases().collect();
        assert_eq!(bases, vec!["t1.1", "t1.2"]);
    }
}
