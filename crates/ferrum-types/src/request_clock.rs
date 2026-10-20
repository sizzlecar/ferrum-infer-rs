//! Process-local request time. Wall samples correlate events; only the
//! monotonic anchor is used for request age, deadlines and elapsed time.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// The first local API boundary that observed this request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestIngressBoundary {
    /// Entry into an HTTP handler, after the framework's JSON extraction.
    HttpHandler,
    EngineApi,
    SchedulerApi,
}

impl RequestIngressBoundary {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HttpHandler => "http_handler",
            Self::EngineApi => "engine_api",
            Self::SchedulerApi => "scheduler_api",
        }
    }
}

/// One original local clock sample, propagated through request execution.
///
/// This value deliberately has no wire representation or wall-time constructor.
/// Cloning preserves the original sample; it does not start another clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestIngressAnchor {
    instant: Instant,
    boundary: RequestIngressBoundary,
    wall_before: SystemTime,
    wall_after: SystemTime,
}

impl RequestIngressAnchor {
    /// Capture once at the first local entrypoint. Wall-clock conversion is
    /// deferred so an unrepresentable wall time cannot reject ordinary inference.
    pub fn capture(boundary: RequestIngressBoundary) -> Self {
        let wall_before = SystemTime::now();
        let instant = Instant::now();
        let wall_after = SystemTime::now();
        Self {
            instant,
            boundary,
            wall_before,
            wall_after,
        }
    }

    pub fn instant(self) -> Instant {
        self.instant
    }

    pub fn boundary(self) -> RequestIngressBoundary {
        self.boundary
    }

    pub fn elapsed_at(self, now: Instant) -> Duration {
        now.saturating_duration_since(self.instant)
    }

    pub fn elapsed(self) -> Duration {
        self.elapsed_at(Instant::now())
    }

    /// Convert the original bracketing wall samples for diagnostic correlation.
    /// Never use this result to reconstruct an Instant or compute a duration.
    pub fn wall_correlation(self) -> Result<RequestWallCorrelation, RequestWallClockError> {
        let before = unix_nanos(self.wall_before)?;
        let after = unix_nanos(self.wall_after)?;
        let lower = before.min(after);
        let upper = before.max(after);
        let span = upper - lower;
        Ok(RequestWallCorrelation {
            unix_nanos: lower + span / 2,
            max_error_nanos: span as u64,
        })
    }
}

/// A wall-time association with the original monotonic sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestWallCorrelation {
    pub unix_nanos: i64,
    /// Full span of the bracketing samples, preserving the conservative event
    /// correlation convention. This is not a bound on clock synchronization
    /// error or on arbitrary wall-clock discontinuities between the samples.
    pub max_error_nanos: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestWallClockError {
    BeforeUnixEpoch,
    OutOfRange,
}

impl std::fmt::Display for RequestWallClockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::BeforeUnixEpoch => "request wall clock predates Unix epoch",
            Self::OutOfRange => "request wall clock Unix nanoseconds exceed i64",
        })
    }
}

impl std::error::Error for RequestWallClockError {}

fn unix_nanos(time: SystemTime) -> Result<i64, RequestWallClockError> {
    let elapsed = time
        .duration_since(UNIX_EPOCH)
        .map_err(|_| RequestWallClockError::BeforeUnixEpoch)?;
    i64::try_from(elapsed.as_nanos()).map_err(|_| RequestWallClockError::OutOfRange)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EngineTokenTimingEvidence, InferenceRequest};

    #[test]
    fn request_clock_preserves_original_anchor_across_entrypoints_and_clones() {
        let anchor = RequestIngressAnchor::capture(RequestIngressBoundary::HttpHandler);
        let mut request = InferenceRequest::new("hello", "test").with_ingress(anchor);
        request.created_at = chrono::DateTime::from_timestamp(0, 0).unwrap();
        let mut cloned = request.clone();
        cloned.created_at = chrono::DateTime::from_timestamp(7_258_118_400, 0).unwrap();
        assert_eq!(
            request.ensure_ingress(RequestIngressBoundary::EngineApi),
            anchor
        );
        assert_eq!(
            cloned.ensure_ingress(RequestIngressBoundary::SchedulerApi),
            anchor
        );
        let now = anchor.instant() + Duration::from_secs(3);
        assert_eq!(
            cloned.ingress.unwrap().elapsed_at(now),
            Duration::from_secs(3)
        );
    }

    #[test]
    fn request_clock_is_not_reconstructed_from_wire_or_request_id() {
        let mut original = InferenceRequest::new("hello", "test");
        original.ensure_ingress(RequestIngressBoundary::HttpHandler);
        let mut wire = serde_json::to_value(&original).unwrap();
        assert!(wire.get("ingress").is_none());
        wire["ingress"] = serde_json::json!({"instant": 1, "boundary": "http_handler"});
        let mut received: InferenceRequest = serde_json::from_value(wire).unwrap();
        assert_eq!(received.id, original.id);
        assert_eq!(received.ingress, None);
        let before = Instant::now();
        let actual = received.ensure_ingress(RequestIngressBoundary::EngineApi);
        assert!(actual.instant() >= before);
        assert!(actual.instant() <= Instant::now());
        assert_eq!(actual.boundary(), RequestIngressBoundary::EngineApi);
    }

    #[test]
    fn request_clock_wall_correlation_uses_original_bracket_in_either_order() {
        let original = RequestIngressAnchor::capture(RequestIngressBoundary::EngineApi);
        for (before, after) in [(10, 21), (21, 10)] {
            let anchor = RequestIngressAnchor {
                wall_before: UNIX_EPOCH + Duration::from_nanos(before),
                wall_after: UNIX_EPOCH + Duration::from_nanos(after),
                ..original
            };
            assert_eq!(
                anchor.wall_correlation().unwrap(),
                RequestWallCorrelation {
                    unix_nanos: 15,
                    max_error_nanos: 11,
                }
            );
            assert_eq!(
                anchor.elapsed_at(original.instant() + Duration::from_secs(2)),
                Duration::from_secs(2)
            );
        }
    }

    #[test]
    fn request_clock_survives_unrepresentable_wall_samples() {
        let original = RequestIngressAnchor::capture(RequestIngressBoundary::EngineApi);
        for (wall, expected) in [
            (
                UNIX_EPOCH - Duration::from_nanos(1),
                RequestWallClockError::BeforeUnixEpoch,
            ),
            (
                UNIX_EPOCH + Duration::from_nanos(i64::MAX as u64 + 1),
                RequestWallClockError::OutOfRange,
            ),
        ] {
            let anchor = RequestIngressAnchor {
                wall_before: wall,
                wall_after: wall,
                ..original
            };
            assert_eq!(anchor.wall_correlation(), Err(expected));
            let mut request = InferenceRequest::new("hello", "test").with_ingress(anchor);
            let retained = request.ensure_ingress(RequestIngressBoundary::SchedulerApi);
            assert_eq!(retained.instant(), original.instant());
            assert_eq!(
                retained.elapsed_at(original.instant() + Duration::from_secs(1)),
                Duration::from_secs(1)
            );
        }
    }

    #[test]
    fn request_clock_elapsed_saturates_for_an_earlier_observation() {
        let anchor = RequestIngressAnchor::capture(RequestIngressBoundary::EngineApi);
        assert_eq!(
            anchor.elapsed_at(anchor.instant() - Duration::from_nanos(1)),
            Duration::ZERO
        );
    }

    #[test]
    fn request_clock_legacy_evidence_does_not_invent_an_ingress_boundary() {
        let old = serde_json::json!({
            "clock_source": "rust_std_instant",
            "wall_anchor_unix_nanos": 1,
            "wall_anchor_max_error_nanos": 0,
            "token_commit_nanos_since_request_start": [12]
        });
        let mut evidence: EngineTokenTimingEvidence = serde_json::from_value(old).unwrap();
        assert_eq!(evidence.request_start_boundary, None);
        evidence.validate(1).unwrap();
        evidence.request_start_boundary = Some(RequestIngressBoundary::HttpHandler);
        let wire = serde_json::to_value(&evidence).unwrap();
        assert_eq!(wire["request_start_boundary"], "http_handler");
        assert_eq!(
            serde_json::from_value::<EngineTokenTimingEvidence>(wire).unwrap(),
            evidence
        );
    }
}
