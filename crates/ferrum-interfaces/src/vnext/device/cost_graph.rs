//! Per-stream and per-submission CUDA graph evidence. Capability describes what
//! a runtime can do; this evidence records what the owning stream can do now.
//! It grants no capture, replay, submission or cost-calibration authority.

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceCostGraphConfiguration {
    Unconfigured,
    StartupPreparing,
    StartupReady,
    OnDemand,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DeviceCostGraphStreamState {
    configuration: DeviceCostGraphConfiguration,
    resident_executables: u64,
    resident_programs: u64,
    rejected_executables: u64,
}

impl DeviceCostGraphStreamState {
    pub fn new(
        configuration: DeviceCostGraphConfiguration,
        resident_executables: u64,
        resident_programs: u64,
        rejected_executables: u64,
    ) -> Option<Self> {
        if configuration == DeviceCostGraphConfiguration::Unconfigured
            && (resident_executables != 0 || resident_programs != 0 || rejected_executables != 0)
        {
            return None;
        }
        Some(Self {
            configuration,
            resident_executables,
            resident_programs,
            rejected_executables,
        })
    }

    pub const fn is_unconfigured_empty(self) -> bool {
        matches!(
            self.configuration,
            DeviceCostGraphConfiguration::Unconfigured
        ) && self.resident_executables == 0
            && self.resident_programs == 0
            && self.rejected_executables == 0
    }
}

/// Actual native preparation and execution evidence. An empty replay list by
/// itself cannot establish a graph-free eager path: capture may have run first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DeviceSubmissionGraphEvidence {
    before: DeviceCostGraphStreamState,
    after_preparation: DeviceCostGraphStreamState,
    capture_requested: bool,
    candidate_segments: u64,
    captured_segments: u64,
    capture_rejected_segments: u64,
    uploaded_segments: u64,
    replayed_segments: u64,
}

impl DeviceSubmissionGraphEvidence {
    pub const fn replayed_segments(self) -> u64 {
        self.replayed_segments
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        before: DeviceCostGraphStreamState,
        after_preparation: DeviceCostGraphStreamState,
        capture_requested: bool,
        candidate_segments: u64,
        captured_segments: u64,
        capture_rejected_segments: u64,
        uploaded_segments: u64,
        replayed_segments: u64,
    ) -> Option<Self> {
        if captured_segments.checked_add(capture_rejected_segments)? > candidate_segments
            || uploaded_segments > captured_segments
        {
            return None;
        }
        Some(Self {
            before,
            after_preparation,
            capture_requested,
            candidate_segments,
            captured_segments,
            capture_rejected_segments,
            uploaded_segments,
            replayed_segments,
        })
    }

    pub const fn proves_unconfigured_eager(self) -> bool {
        self.before.is_unconfigured_empty()
            && self.after_preparation.is_unconfigured_empty()
            && !self.capture_requested
            && self.captured_segments == 0
            && self.capture_rejected_segments == 0
            && self.uploaded_segments == 0
            && self.replayed_segments == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(configuration: DeviceCostGraphConfiguration) -> DeviceCostGraphStreamState {
        DeviceCostGraphStreamState::new(configuration, 0, 0, 0).unwrap()
    }

    #[test]
    fn actual_unconfigured_empty_stream_can_certify_eager_without_denying_graph_capability() {
        let empty = state(DeviceCostGraphConfiguration::Unconfigured);
        assert!(
            DeviceSubmissionGraphEvidence::new(empty, empty, false, 3, 0, 0, 0, 0)
                .unwrap()
                .proves_unconfigured_eager()
        );
        assert!(DeviceCostGraphStreamState::new(
            DeviceCostGraphConfiguration::Unconfigured,
            1,
            0,
            0
        )
        .is_none());
        assert!(DeviceCostGraphStreamState::new(
            DeviceCostGraphConfiguration::Unconfigured,
            0,
            1,
            0
        )
        .is_none());
    }

    #[test]
    fn no_replay_does_not_hide_capture_preparation_or_configuration_changes() {
        let empty = state(DeviceCostGraphConfiguration::Unconfigured);
        for configured in [
            DeviceCostGraphConfiguration::StartupPreparing,
            DeviceCostGraphConfiguration::StartupReady,
            DeviceCostGraphConfiguration::OnDemand,
        ] {
            assert!(!DeviceSubmissionGraphEvidence::new(
                empty,
                state(configured),
                false,
                3,
                0,
                0,
                0,
                0
            )
            .unwrap()
            .proves_unconfigured_eager());
        }
        assert!(
            !DeviceSubmissionGraphEvidence::new(empty, empty, true, 3, 0, 0, 0, 0)
                .unwrap()
                .proves_unconfigured_eager()
        );
        assert!(
            !DeviceSubmissionGraphEvidence::new(empty, empty, false, 3, 1, 0, 1, 0)
                .unwrap()
                .proves_unconfigured_eager()
        );
        assert!(
            !DeviceSubmissionGraphEvidence::new(empty, empty, false, 3, 0, 1, 0, 0)
                .unwrap()
                .proves_unconfigured_eager()
        );
        assert!(
            !DeviceSubmissionGraphEvidence::new(empty, empty, false, 3, 0, 0, 0, 1)
                .unwrap()
                .proves_unconfigured_eager()
        );
        assert!(DeviceSubmissionGraphEvidence::new(empty, empty, false, 0, 1, 0, 0, 0).is_none());
    }
}
