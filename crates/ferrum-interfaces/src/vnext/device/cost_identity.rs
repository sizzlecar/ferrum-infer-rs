//! Actual graph-capture capability for complete eager cost-route evidence.

/// Actual backend capability, independent of requested execution/timing policy.
/// Unknown cannot be promoted to Unsupported from an eager submission: it may
/// have captured a graph while executing eager work.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DeviceCostGraphCaptureCapability {
    #[default]
    Unknown,
    /// This runtime implementation cannot execute graph capture at all.
    Unsupported,
    /// The runtime can capture graphs. An eager execution alone does not prove
    /// that the same submission performed no capture or graph preparation.
    Supported,
}
