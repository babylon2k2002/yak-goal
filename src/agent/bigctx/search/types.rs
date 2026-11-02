//! Search types — the vocabulary a search algorithm scores against.
//!
//! Two data sources are searchable:
//!
//! - **Compressed blocks** (tier summaries; ref is the block id, `b1`).
//! - **Folded messages** (original text from the host's history; ref is the
//!   message ref, `m00042`).
//!
//! The second is the important one. Compression turns a 40K-token range into a
//! 400-token summary; without searching the originals, the model can only get
//! detail back by remembering a ref, which it cannot do reliably. Search is
//! what makes a folded history navigable rather than merely small.
//!
//! A `SearchAlgorithm` is a stateless scorer over a unified [`SearchDoc`]
//! slice. Roles carry a configurable weight (user intent outranks tool noise).

use serde::{Deserialize, Serialize};

/// Where a searchable document came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SearchDocKind {
    Block,
    Message,
}

impl SearchDocKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SearchDocKind::Block => "block",
            SearchDocKind::Message => "message",
        }
    }
}

/// The role a message doc came from. Blocks have no role.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SearchRole {
    User,
    Assistant,
    Tool,
}

/// One searchable document: a block summary or a folded message.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchDoc {
    pub kind: SearchDocKind,
    /// What to hand `decompress` / `acp_retrieve`: `b2` for a block,
    /// `m00042` for a message.
    pub ref_id: String,
    /// The text scored against the query (block: topic + summary; message:
    /// its content).
    pub text: String,
    /// Short label for the result list.
    pub title: String,
    /// Message role; `None` for blocks. Drives role weighting.
    pub role: Option<SearchRole>,
    /// The block that owns this doc. For a block that is itself; for a message
    /// it is the block that folded it, so a hit points at what to open.
    pub block_id: Option<String>,
    pub tier: u8,
    /// Approximate size, for the "how big is this" column.
    pub tokens: u64,
}

/// Per-role score multipliers. Defaults favour user intent over tool noise.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoleWeights {
    pub user: f64,
    pub assistant: f64,
    pub tool: f64,
    pub block: f64,
}

impl Default for RoleWeights {
    fn default() -> RoleWeights {
        RoleWeights {
            user: 1.5,
            assistant: 1.0,
            tool: 0.6,
            block: 1.0,
        }
    }
}

impl RoleWeights {
    pub fn weight_for(&self, doc: &SearchDoc) -> f64 {
        match doc.kind {
            SearchDocKind::Block => self.block,
            SearchDocKind::Message => match doc.role {
                Some(SearchRole::User) => self.user,
                Some(SearchRole::Assistant) => self.assistant,
                Some(SearchRole::Tool) => self.tool,
                None => 1.0,
            },
        }
    }
}

/// One document's score from one algorithm.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoredBlock {
    pub ref_id: String,
    pub score: f64,
}

/// A stateless scorer: docs in, scores out, in doc order.
pub trait SearchAlgorithm: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn score(&self, docs: &[SearchDoc], query: &str) -> Vec<ScoredBlock>;
}

/// A ranked hit, after role weighting, filtering and truncation.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    pub kind: SearchDocKind,
    pub ref_id: String,
    pub block_id: Option<String>,
    pub tier: u8,
    pub score: f64,
    pub title: String,
    pub preview: String,
    pub role: Option<SearchRole>,
    pub tokens: u64,
}

/// Knobs a caller may pass.
#[derive(Clone, Debug)]
pub struct SearchOptions {
    pub algorithm: Option<String>,
    pub limit: usize,
    pub preview_length: usize,
    pub min_score: f64,
    pub role_weights: RoleWeights,
}

impl Default for SearchOptions {
    fn default() -> SearchOptions {
        SearchOptions {
            algorithm: None,
            limit: 10,
            preview_length: 200,
            min_score: 0.01,
            role_weights: RoleWeights::default(),
        }
    }
}

/// The default algorithm: best precision and recall together.
pub const DEFAULT_ALGORITHM: &str = "hybrid";

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(kind: SearchDocKind, role: Option<SearchRole>) -> SearchDoc {
        SearchDoc {
            kind,
            ref_id: "r".into(),
            text: "t".into(),
            title: "T".into(),
            role,
            block_id: None,
            tier: 1,
            tokens: 1,
        }
    }

    #[test]
    fn weights_favour_user_intent_over_tool_noise() {
        let w = RoleWeights::default();
        let user = w.weight_for(&doc(SearchDocKind::Message, Some(SearchRole::User)));
        let tool = w.weight_for(&doc(SearchDocKind::Message, Some(SearchRole::Tool)));
        assert!(user > tool, "user {user} must outweigh tool {tool}");
    }

    #[test]
    fn a_block_is_weighted_as_a_block() {
        let w = RoleWeights::default();
        assert_eq!(w.weight_for(&doc(SearchDocKind::Block, None)), w.block);
    }
}
