use ferrum_interfaces::vnext::DeviceProgramBindingUploadSnapshot;
use parking_lot::Mutex;

/// A measurement baseline, never a reset of the possibly shared runtime.
#[derive(Default)]
pub(super) struct ProgramBindingUploadMetrics {
    baseline: Mutex<Option<DeviceProgramBindingUploadSnapshot>>,
}

impl ProgramBindingUploadMetrics {
    pub(super) fn reset_baseline(&self, snapshot: Option<DeviceProgramBindingUploadSnapshot>) {
        *self.baseline.lock() = snapshot;
    }

    pub(super) fn snapshot(
        &self,
        current: Option<DeviceProgramBindingUploadSnapshot>,
    ) -> serde_json::Value {
        let baseline = *self.baseline.lock();
        let delta = current
            .zip(baseline)
            .and_then(|(now, before)| now.checked_since(before));
        serde_json::json!({
            "supported": current.is_some(),
            "window_available": delta.is_some(),
            "scope": "runtime_typed_binding_preludes_since_executor_startup_baseline_including_partial_failures",
            "counters": delta,
            "accounting": {
                "planned": "live_payload_bytes, planned_upload_bytes, logical_arena_bytes and physical_arena_bytes count plans when a prelude enqueue is attempted",
                "successful": "successful_upload_bytes, successful_1d_copies/2d_copies and successful_scatter_dispatches count only CUDA calls returning success, including calls before a later error; not GPU completion",
                "compact_transport": "compact_scatter_preludes and compact_scatter_sparse_fallback_preludes count actual enqueue attempts on the selected compact route or its pre-enqueue scratch-capacity fallback",
                "attempts": "succeeded_preludes completed all host enqueue calls; failed_preludes include ordinary error or unwind; pre-enqueue rejection has no attempted prelude",
                "reset": "executor startup advances only this baseline; shared runtime cumulative counters are not reset",
                "limitations": "counts include all users sharing this runtime and all timing modes; no latency or completion inference; unsupported, decreasing or saturated counters produce no window"
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn program_binding_upload_baseline_preserves_shared_runtime_and_partial_failure() {
        let metrics = ProgramBindingUploadMetrics::default();
        let startup = DeviceProgramBindingUploadSnapshot {
            attempted_preludes: 4,
            succeeded_preludes: 4,
            live_payload_bytes: 40,
            planned_upload_bytes: 64,
            logical_arena_bytes: 80,
            physical_arena_bytes: 128,
            successful_upload_bytes: 64,
            successful_1d_copies: 4,
            ..Default::default()
        };
        metrics.reset_baseline(Some(startup));
        let current = DeviceProgramBindingUploadSnapshot {
            attempted_preludes: 7,
            succeeded_preludes: 5,
            failed_preludes: 2,
            live_payload_bytes: 64,
            planned_upload_bytes: 112,
            logical_arena_bytes: 128,
            physical_arena_bytes: 224,
            successful_upload_bytes: 88,
            successful_1d_copies: 5,
            successful_2d_copies: 1,
            ..Default::default()
        };
        let delta = metrics.snapshot(Some(current));
        assert_eq!(delta["counters"]["attempted_preludes"], 3);
        assert_eq!(delta["counters"]["failed_preludes"], 2);
        assert_eq!(delta["counters"]["successful_upload_bytes"], 24);
        assert_eq!(delta["counters"]["planned_upload_bytes"], 48);
        metrics.reset_baseline(Some(current));
        assert_eq!(
            metrics.snapshot(Some(current))["counters"]["attempted_preludes"],
            0
        );
        assert_eq!(
            current.attempted_preludes, 7,
            "no runtime mutation at baseline reset"
        );
        let another_executor = ProgramBindingUploadMetrics::default();
        another_executor.reset_baseline(Some(startup));
        assert_eq!(
            another_executor.snapshot(Some(current))["counters"]["attempted_preludes"],
            3
        );
    }

    #[test]
    fn program_binding_upload_unsupported_or_reset_counter_has_no_fabricated_window() {
        let metrics = ProgramBindingUploadMetrics::default();
        assert_eq!(metrics.snapshot(None)["supported"], false);
        assert!(metrics.snapshot(None)["counters"].is_null());
        metrics.reset_baseline(Some(DeviceProgramBindingUploadSnapshot {
            successful_upload_bytes: 10,
            ..Default::default()
        }));
        let value = metrics.snapshot(Some(DeviceProgramBindingUploadSnapshot::default()));
        assert_eq!(value["supported"], true);
        assert_eq!(value["window_available"], false);
        assert!(value["counters"].is_null());
    }

    #[test]
    fn program_binding_upload_saturation_never_reports_a_valid_zero_window() {
        let metrics = ProgramBindingUploadMetrics::default();
        let saturated = DeviceProgramBindingUploadSnapshot {
            successful_upload_bytes: u64::MAX,
            ..Default::default()
        };
        metrics.reset_baseline(Some(saturated));
        assert_eq!(metrics.snapshot(Some(saturated))["window_available"], false);
        metrics.reset_baseline(Some(DeviceProgramBindingUploadSnapshot::default()));
        assert_eq!(metrics.snapshot(Some(saturated))["window_available"], false);
    }
}
