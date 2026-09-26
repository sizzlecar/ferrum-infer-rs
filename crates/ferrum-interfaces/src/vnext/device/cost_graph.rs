//! Per-stream and per-submission CUDA graph evidence. Capability describes what
//! a runtime can do; this evidence records what the owning stream can do now.
//! It grants no capture, replay, submission or cost-calibration authority.

use serde::Serialize;
mod catalog;
pub use catalog::*;

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

    pub const fn configuration(self) -> DeviceCostGraphConfiguration {
        self.configuration
    }

    pub const fn is_ready(self) -> bool {
        matches!(
            self.configuration,
            DeviceCostGraphConfiguration::StartupReady | DeviceCostGraphConfiguration::OnDemand
        )
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

    /// A direct resident replay performed no capture, upload or adaptive
    /// candidate preparation. The runtime must also validate every sealed
    /// program/segment and its uploaded executable before giving this evidence
    /// to a submission guard. Counts alone do not authenticate residency.
    pub fn proves_warm_direct_replay(self) -> bool {
        self.before == self.after_preparation
            && self.before.is_ready()
            && !self.capture_requested
            && self.candidate_segments == 0
            && self.captured_segments == 0
            && self.capture_rejected_segments == 0
            && self.uploaded_segments == 0
            && self.replayed_segments > 0
    }

    /// Historical observation only: a configured OnDemand stream executed an
    /// eager wave without candidates, capture, upload, replay, or cache change.
    /// This does not authorize a future eager wave on the configured stream.
    pub fn proves_configured_eager_observation(self) -> bool {
        self.before == self.after_preparation
            && self.before.configuration() == DeviceCostGraphConfiguration::OnDemand
            && !self.capture_requested
            && self.candidate_segments == 0
            && self.captured_segments == 0
            && self.capture_rejected_segments == 0
            && self.uploaded_segments == 0
            && self.replayed_segments == 0
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
    fn configured_eager_observation_excludes_every_adaptive_action_and_future_authority() {
        let ready =
            DeviceCostGraphStreamState::new(DeviceCostGraphConfiguration::OnDemand, 3, 2, 1)
                .unwrap();
        let proof = DeviceSubmissionGraphEvidence::new(ready, ready, false, 0, 0, 0, 0, 0).unwrap();
        assert!(proof.proves_configured_eager_observation());
        assert!(!proof.proves_unconfigured_eager());
        assert!(!proof.proves_warm_direct_replay());
        let other =
            DeviceCostGraphStreamState::new(DeviceCostGraphConfiguration::OnDemand, 4, 2, 1)
                .unwrap();
        for rejected in [
            DeviceSubmissionGraphEvidence::new(ready, other, false, 0, 0, 0, 0, 0),
            DeviceSubmissionGraphEvidence::new(ready, ready, true, 0, 0, 0, 0, 0),
            DeviceSubmissionGraphEvidence::new(ready, ready, false, 1, 0, 0, 0, 0),
            DeviceSubmissionGraphEvidence::new(ready, ready, false, 1, 1, 0, 0, 0),
            DeviceSubmissionGraphEvidence::new(ready, ready, false, 1, 0, 1, 0, 0),
            DeviceSubmissionGraphEvidence::new(ready, ready, false, 1, 1, 0, 1, 0),
            DeviceSubmissionGraphEvidence::new(ready, ready, false, 0, 0, 0, 0, 1),
        ] {
            assert!(!rejected.unwrap().proves_configured_eager_observation());
        }
        for configuration in [
            DeviceCostGraphConfiguration::Unconfigured,
            DeviceCostGraphConfiguration::StartupPreparing,
            DeviceCostGraphConfiguration::StartupReady,
        ] {
            let value = state(configuration);
            assert!(
                !DeviceSubmissionGraphEvidence::new(value, value, false, 0, 0, 0, 0, 0)
                    .unwrap()
                    .proves_configured_eager_observation()
            );
        }
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
    #[test]
    fn warm_replay_needs_unchanged_ready_catalog_and_no_preparation() {
        let ready =
            DeviceCostGraphStreamState::new(DeviceCostGraphConfiguration::OnDemand, 1, 1, 0)
                .unwrap();
        let warm = DeviceSubmissionGraphEvidence::new(ready, ready, false, 0, 0, 0, 0, 1).unwrap();
        assert!(warm.proves_warm_direct_replay());
        assert!(!warm.proves_unconfigured_eager());
        let changed =
            DeviceCostGraphStreamState::new(DeviceCostGraphConfiguration::OnDemand, 2, 1, 0)
                .unwrap();
        for evidence in [
            DeviceSubmissionGraphEvidence::new(ready, changed, false, 0, 0, 0, 0, 1),
            DeviceSubmissionGraphEvidence::new(ready, ready, true, 0, 0, 0, 0, 1),
            DeviceSubmissionGraphEvidence::new(ready, ready, false, 1, 1, 0, 1, 1),
            DeviceSubmissionGraphEvidence::new(ready, ready, false, 1, 0, 0, 0, 1),
            DeviceSubmissionGraphEvidence::new(ready, ready, false, 0, 0, 0, 0, 0),
        ] {
            assert!(!evidence.unwrap().proves_warm_direct_replay());
        }
        let preparing = state(DeviceCostGraphConfiguration::StartupPreparing);
        assert!(
            !DeviceSubmissionGraphEvidence::new(preparing, preparing, false, 0, 0, 0, 0, 1)
                .unwrap()
                .proves_warm_direct_replay()
        );
    }
}
