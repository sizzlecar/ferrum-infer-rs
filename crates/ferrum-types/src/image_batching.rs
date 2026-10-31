//! Image request scheduling policies, independent of text token scheduling.

use serde::{Deserialize, Serialize};

/// Selects how a service forms a batch from compatible queued image requests.
/// Capacity limits remain upper bounds under either policy, never reservations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageBatchPolicy {
    /// Preserve an explicitly configured batch ceiling and gathering window.
    #[default]
    Configured,
    /// Dispatch available work immediately. Adapt compatible backlog using
    /// compiled capacity/residency demand and measured execution costs. Unknown
    /// weight-swapping probes spend earned execution savings; fresh capacity
    /// admission remains authoritative. Never wait for future arrivals.
    Auto,
}
