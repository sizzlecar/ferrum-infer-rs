use serde::Serialize;

/// Runtime-lifetime counters for attempted typed program-binding preludes.
/// Plans are counted when enqueue starts; successful transfers are counted
/// only after their host API returns successfully, including partial failure.
/// No field establishes device completion or resource-release authority.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct DeviceProgramBindingUploadSnapshot {
    pub attempted_preludes: u64,
    pub succeeded_preludes: u64,
    pub failed_preludes: u64,
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
                snapshot.live_payload_bytes,
                snapshot.planned_upload_bytes,
                snapshot.logical_arena_bytes,
                snapshot.physical_arena_bytes,
                snapshot.successful_upload_bytes,
                snapshot.successful_1d_copies,
                snapshot.successful_2d_copies,
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
        })
    }
}
