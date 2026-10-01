//! Cold declarations from the providers selected for an immutable plan.
//!
//! These describe decode-context selection only. They do not prove empirical
//! cost coverage, live resources, or permission to execute a route.

use crate::vnext::{NodeId, ProviderId};
use serde::Serialize;
use std::num::NonZeroU64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeContextBoundaryKind {
    KernelFamily,
    ReplayPartition,
}

/// First sequence frontier on the new side of a selection boundary.
///
/// `first_sequence_tokens` includes the current decode input token. For one
/// decode token it equals existing KV tokens plus one, exactly the source
/// range's exclusive end (`offset + count`). It is not the prior KV length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct DecodeContextBoundary {
    pub first_sequence_tokens: NonZeroU64,
    pub kind: DecodeContextBoundaryKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum DecodeContextCoverage {
    /// No completeness claim. Retain any proven obligations even when another
    /// selection dimension cannot be described by the declared boundary kinds.
    Unknown {
        maximum_sequence_tokens: Option<NonZeroU64>,
        known_boundaries: Vec<DecodeContextBoundary>,
    },
    /// Complete for this node's decode-context kernel-family and replay-
    /// partition selection through the stated compiled sequence frontier.
    /// An empty list explicitly declares no such changes in that domain.
    Declared {
        maximum_sequence_tokens: NonZeroU64,
        boundaries: Vec<DecodeContextBoundary>,
    },
}

impl Default for DecodeContextCoverage {
    fn default() -> Self {
        Self::Unknown {
            maximum_sequence_tokens: None,
            known_boundaries: Vec::new(),
        }
    }
}

impl DecodeContextCoverage {
    pub fn known_boundaries(&self) -> &[DecodeContextBoundary] {
        match self {
            Self::Unknown {
                known_boundaries, ..
            } => known_boundaries,
            Self::Declared { boundaries, .. } => boundaries,
        }
    }

    pub const fn maximum_sequence_tokens(&self) -> Option<NonZeroU64> {
        match self {
            Self::Unknown {
                maximum_sequence_tokens,
                ..
            } => *maximum_sequence_tokens,
            Self::Declared {
                maximum_sequence_tokens,
                ..
            } => Some(*maximum_sequence_tokens),
        }
    }

    pub const fn is_complete(&self) -> bool {
        matches!(self, Self::Declared { .. })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BoundDecodeContextCoverage {
    pub node_id: NodeId,
    pub provider_id: ProviderId,
    pub coverage: DecodeContextCoverage,
}

/// Each row belongs to the actual bound provider for one plan node. Unknown
/// rows never erase known obligations from that node or another node.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ExecutorDecodeContextCoverage {
    pub nodes: Vec<BoundDecodeContextCoverage>,
}

impl ExecutorDecodeContextCoverage {
    /// An absent plan/provider declaration is not a proof of no boundaries.
    pub fn is_complete(&self) -> bool {
        !self.nodes.is_empty() && self.nodes.iter().all(|node| node.coverage.is_complete())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_preserves_known_obligations_and_differs_from_no_boundaries() {
        let maximum = NonZeroU64::new(32).unwrap();
        let boundary = DecodeContextBoundary {
            first_sequence_tokens: NonZeroU64::new(17).unwrap(),
            kind: DecodeContextBoundaryKind::KernelFamily,
        };
        let unknown = DecodeContextCoverage::Unknown {
            maximum_sequence_tokens: Some(maximum),
            known_boundaries: vec![boundary],
        };
        assert!(!unknown.is_complete());
        assert_eq!(unknown.known_boundaries(), [boundary]);
        assert_eq!(unknown.maximum_sequence_tokens(), Some(maximum));
        let declared = DecodeContextCoverage::Declared {
            maximum_sequence_tokens: maximum,
            boundaries: Vec::new(),
        };
        assert!(declared.is_complete());
        assert!(declared.known_boundaries().is_empty());
        assert!(!ExecutorDecodeContextCoverage::default().is_complete());
    }
}
