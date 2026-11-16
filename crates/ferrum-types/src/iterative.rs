//! Payload-independent request identity and iterative collection policy.
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestCorrelation {
    /// Minted by the service, never taken from a caller's request-id header.
    pub request_id: Uuid,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_attempt_id: Option<Uuid>,
}
impl RequestCorrelation {
    pub fn new(client_attempt_id: Option<Uuid>) -> Self {
        Self {
            request_id: Uuid::new_v4(),
            client_attempt_id,
        }
    }
}

/// Selects how a service forms a batch from compatible queued iterative requests.
/// Capacity limits remain upper bounds under either policy, never reservations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IterativeBatchPolicy {
    /// Preserve an explicitly configured batch ceiling and gathering window.
    #[default]
    Configured,
    /// Dispatch available work immediately. Adapt compatible backlog using
    /// compiled capacity/residency demand and measured execution costs. Unknown
    /// weight-swapping probes spend earned execution savings; fresh capacity
    /// admission remains authoritative. Never wait for future arrivals.
    /// A declared auxiliary initializer yields to resident requests when all
    /// have exactly one solver transition left; existing prepared work may join.
    Auto,
}
