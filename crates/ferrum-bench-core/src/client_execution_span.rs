//! Request execution intervals on one benchmark phase's monotonic clock.

use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Nanoseconds relative to the measured run's duration origin. Compare spans
/// only within the same benchmark run, cell, repeat and phase.
///
/// Start is the client's existing request/latency origin, after request-body
/// construction and before sending. End is sampled in the request task after
/// stream processing and local cleanup, before releasing its concurrency
/// permit when present. It may be later than the existing E2E/terminal observation. This is
/// neither server execution time nor the full time holding a client permit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientExecutionSpan {
    pub start_ns: u64,
    pub end_ns: u64,
}

impl ClientExecutionSpan {
    /// Missing, reversed or unrepresentable clock observations must not be
    /// replaced with zero or a saturated timestamp.
    pub fn from_instants(origin: Instant, start: Instant, end: Instant) -> Option<Self> {
        end.checked_duration_since(start)?;
        Some(Self {
            start_ns: origin_offset_ns(origin, start)?,
            end_ns: origin_offset_ns(origin, end)?,
        })
    }
}

fn origin_offset_ns(origin: Instant, instant: Instant) -> Option<u64> {
    u64::try_from(instant.checked_duration_since(origin)?.as_nanos()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BenchmarkRequestRecord;
    use std::time::Duration;

    #[test]
    fn client_execution_span_preserves_shared_origin_overlap_and_zero() {
        let origin = Instant::now();
        let first =
            ClientExecutionSpan::from_instants(origin, origin, origin + Duration::from_nanos(30))
                .unwrap();
        let second = ClientExecutionSpan::from_instants(
            origin,
            origin + Duration::from_nanos(20),
            origin + Duration::from_nanos(40),
        )
        .unwrap();
        assert_eq!(
            first,
            ClientExecutionSpan {
                start_ns: 0,
                end_ns: 30
            }
        );
        assert_eq!(
            second,
            ClientExecutionSpan {
                start_ns: 20,
                end_ns: 40
            }
        );
        assert!(second.start_ns < first.end_ns);
        assert_eq!(
            ClientExecutionSpan::from_instants(origin, origin, origin),
            Some(ClientExecutionSpan {
                start_ns: 0,
                end_ns: 0
            })
        );
        assert!(ClientExecutionSpan::from_instants(
            origin + Duration::from_nanos(1),
            origin,
            origin
        )
        .is_none());
        assert!(ClientExecutionSpan::from_instants(
            origin,
            origin + Duration::from_nanos(1),
            origin
        )
        .is_none());
    }

    #[test]
    fn client_execution_span_old_json_remains_missing_and_new_json_roundtrips() {
        let old = serde_json::json!({
            "benchmark_run_id": "bench", "cell_id": "cell", "repeat_index": 0,
            "phase": "measured", "request_index": 0
        });
        let mut record: BenchmarkRequestRecord = serde_json::from_value(old.clone()).unwrap();
        assert_eq!(record.client_execution_span, None);
        assert_eq!(serde_json::to_value(&record).unwrap(), old);
        record.client_execution_span = Some(ClientExecutionSpan {
            start_ns: 0,
            end_ns: 0,
        });
        let roundtrip: BenchmarkRequestRecord =
            serde_json::from_str(&serde_json::to_string(&record).unwrap()).unwrap();
        assert_eq!(roundtrip, record);
    }
}
