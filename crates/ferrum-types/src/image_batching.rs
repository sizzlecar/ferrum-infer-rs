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
    /// Dispatch available work immediately. At the next idle boundary, combine
    /// compatible backlog and reduce it only on a typed preparation-capacity
    /// rejection. No future arrivals are awaited to fill a batch.
    Auto,
}
