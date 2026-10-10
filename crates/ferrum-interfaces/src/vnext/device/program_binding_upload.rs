use serde::{Deserialize, Serialize};

/// Runtime-lifetime counters for attempted typed program-binding preludes.
/// Plans are counted when enqueue starts; successful transfers are counted
/// only after their host API returns successfully, including partial failure.
/// No field establishes device completion or resource-release authority.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceProgramBindingUploadSnapshot {
    pub attempted_preludes: u64,
    pub succeeded_preludes: u64,
    pub failed_preludes: u64,
    /// Enqueue attempts that selected compact scatter, including partial failure.
    #[serde(default)]
    pub compact_scatter_preludes: u64,
    /// Enqueue attempts that fell back from compact scatter to sparse copies.
    #[serde(default)]
    pub compact_scatter_sparse_fallback_preludes: u64,
    /// Original provider payload, before explicitly authorized row padding.
    pub live_payload_bytes: u64,
    /// Bytes scheduled by the coalesced transfer plan, including row padding.
    pub planned_upload_bytes: u64,
    /// Sum of this wave's checked logical binding destination lengths.
    pub logical_arena_bytes: u64,
    /// Compiled physical arena capacity, counted once per attempted prelude.
    pub physical_arena_bytes: u64,
    pub successful_upload_bytes: u64,
    pub successful_1d_copies: u64,
    pub successful_2d_copies: u64,
    /// Scatter launches accepted by the driver, not completed device work.
    #[serde(default)]
    pub successful_scatter_dispatches: u64,
}

impl DeviceProgramBindingUploadSnapshot {
    /// Derives a measurement window without resetting a shared runtime.
    /// A decreasing backend counter is an invalid window, not a zero delta.
    pub fn checked_since(self, baseline: Self) -> Option<Self> {
        // Saturating producers cannot distinguish an exact MAX from overflow.
        // Neither may be presented as an observed zero or partial delta.
        for snapshot in [self, baseline] {
            if [
                snapshot.attempted_preludes,
                snapshot.succeeded_preludes,
                snapshot.failed_preludes,
                snapshot.compact_scatter_preludes,
                snapshot.compact_scatter_sparse_fallback_preludes,
                snapshot.live_payload_bytes,
                snapshot.planned_upload_bytes,
                snapshot.logical_arena_bytes,
                snapshot.physical_arena_bytes,
                snapshot.successful_upload_bytes,
                snapshot.successful_1d_copies,
                snapshot.successful_2d_copies,
                snapshot.successful_scatter_dispatches,
            ]
            .contains(&u64::MAX)
            {
                return None;
            }
        }
        Some(Self {
            attempted_preludes: self
                .attempted_preludes
                .checked_sub(baseline.attempted_preludes)?,
            succeeded_preludes: self
                .succeeded_preludes
                .checked_sub(baseline.succeeded_preludes)?,
            failed_preludes: self.failed_preludes.checked_sub(baseline.failed_preludes)?,
            compact_scatter_preludes: self
                .compact_scatter_preludes
                .checked_sub(baseline.compact_scatter_preludes)?,
            compact_scatter_sparse_fallback_preludes: self
                .compact_scatter_sparse_fallback_preludes
                .checked_sub(baseline.compact_scatter_sparse_fallback_preludes)?,
            live_payload_bytes: self
                .live_payload_bytes
                .checked_sub(baseline.live_payload_bytes)?,
            planned_upload_bytes: self
                .planned_upload_bytes
                .checked_sub(baseline.planned_upload_bytes)?,
            logical_arena_bytes: self
                .logical_arena_bytes
                .checked_sub(baseline.logical_arena_bytes)?,
            physical_arena_bytes: self
                .physical_arena_bytes
                .checked_sub(baseline.physical_arena_bytes)?,
            successful_upload_bytes: self
                .successful_upload_bytes
                .checked_sub(baseline.successful_upload_bytes)?,
            successful_1d_copies: self
                .successful_1d_copies
                .checked_sub(baseline.successful_1d_copies)?,
            successful_2d_copies: self
                .successful_2d_copies
                .checked_sub(baseline.successful_2d_copies)?,
            successful_scatter_dispatches: self
                .successful_scatter_dispatches
                .checked_sub(baseline.successful_scatter_dispatches)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_scatter_counters_preserve_legacy_snapshot_records() {
        let legacy = serde_json::json!({
            "attempted_preludes": 3,
            "succeeded_preludes": 2,
            "failed_preludes": 1,
            "live_payload_bytes": 48,
            "planned_upload_bytes": 64,
            "logical_arena_bytes": 256,
            "physical_arena_bytes": 512,
            "successful_upload_bytes": 40,
            "successful_1d_copies": 2,
            "successful_2d_copies": 1
        });
        let decoded: DeviceProgramBindingUploadSnapshot =
            serde_json::from_value(legacy.clone()).unwrap();
        assert_eq!(decoded.compact_scatter_preludes, 0);
        assert_eq!(decoded.compact_scatter_sparse_fallback_preludes, 0);
        assert_eq!(decoded.successful_scatter_dispatches, 0);
        let encoded = serde_json::to_value(decoded).unwrap();
        for (key, value) in legacy.as_object().unwrap() {
            assert_eq!(&encoded[key], value);
        }
    }

    #[test]
    fn compact_scatter_windows_reject_resets_and_saturated_observations() {
        let baseline = DeviceProgramBindingUploadSnapshot {
            compact_scatter_preludes: 7,
            compact_scatter_sparse_fallback_preludes: 2,
            successful_scatter_dispatches: 6,
            ..Default::default()
        };
        let current = DeviceProgramBindingUploadSnapshot {
            compact_scatter_preludes: 10,
            compact_scatter_sparse_fallback_preludes: 4,
            successful_scatter_dispatches: 8,
            ..Default::default()
        };
        let delta = current.checked_since(baseline).unwrap();
        assert_eq!(delta.compact_scatter_preludes, 3);
        assert_eq!(delta.compact_scatter_sparse_fallback_preludes, 2);
        assert_eq!(delta.successful_scatter_dispatches, 2);
        assert_eq!(
            serde_json::from_value::<DeviceProgramBindingUploadSnapshot>(
                serde_json::to_value(current).unwrap()
            )
            .unwrap(),
            current
        );
        let fields: [fn(&mut DeviceProgramBindingUploadSnapshot, u64); 3] = [
            |snapshot: &mut DeviceProgramBindingUploadSnapshot, value| {
                snapshot.compact_scatter_preludes = value
            },
            |snapshot: &mut DeviceProgramBindingUploadSnapshot, value| {
                snapshot.compact_scatter_sparse_fallback_preludes = value
            },
            |snapshot: &mut DeviceProgramBindingUploadSnapshot, value| {
                snapshot.successful_scatter_dispatches = value
            },
        ];
        for field in fields {
            let mut reset = current;
            field(&mut reset, 0);
            assert!(reset.checked_since(baseline).is_none());
            let mut saturated = current;
            field(&mut saturated, u64::MAX);
            assert!(saturated.checked_since(baseline).is_none());
            assert!(saturated.checked_since(saturated).is_none());
        }
    }
}
