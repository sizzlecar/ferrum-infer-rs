//! Small, lane-local input hints. Only successful writes to a still-held exact
//! Step slot may publish; proofs never retain a Step, allocation lease or buffer.
use super::*;
use crate::vnext::{BatchStepId, LaneStableArenaSlotIdentity, StepResourceLease};

const MAX_INPUT_WINDOWS: usize = 1024;
const MAX_INPUT_CONTENT_BYTES: usize = 16 * 1024;
const MAX_INPUT_WINDOW_BYTES: usize = 64;

struct ResidentInputWindow {
    resource: ResourceId,
    participant: u32,
    logical_offset: u64,
    physical_resource: ResourceId,
    physical_offset: u64,
    owner: Weak<dyn Send + Sync>,
    bytes: Vec<u8>,
}

#[derive(Default)]
pub(super) struct InputResidencyCache {
    current: Mutex<InputResidencyState>,
}

#[derive(Default)]
struct InputResidencyState {
    epoch: u64,
    slot: Option<LaneStableArenaSlotIdentity>,
    windows: Vec<ResidentInputWindow>,
}

/// Encoding counts for ordinary input uploads, independent of program-binding
/// transfer counters. These count generated or omitted commands, not GPU time.
#[derive(Debug, Clone, Copy, Default)]
pub struct InputResidencyUploadCounts {
    pub encoded_commands: u64,
    pub encoded_bytes: u64,
    pub reused_commands: u64,
    pub reused_bytes: u64,
}

#[must_use = "publish only with the exact successful completion and held Step"]
pub struct InputResidencyPublication<R: DeviceRuntime> {
    lane: Weak<ExecutionLane<R>>,
    slot: LaneStableArenaSlotIdentity,
    epoch: u64,
    submission: SubmittedOperationReceipt,
    step: BatchStepId,
    previous: Vec<ResidentInputWindow>,
    windows: Vec<ResidentInputWindow>,
    content_bytes: usize,
    pub(super) counts: InputResidencyUploadCounts,
}

impl<R: DeviceRuntime> InputResidencyPublication<R> {
    /// Consuming an unsuccessful, stale or dropped ticket never republishes
    /// prior contents. Borrowing the exact Step prevents retirement during this
    /// check, without retaining its authority in the resulting cache.
    pub fn publish(
        self,
        completion: &OperationCompletionReceipt,
        step: &StepResourceLease<R>,
    ) -> Result<bool, VNextError> {
        let Some(lane) = self.lane.upgrade() else {
            return Ok(false);
        };
        if !matches!(
            completion.disposition(),
            OperationCompletionDisposition::Succeeded
        ) || completion.submission() != &self.submission
            || step.batch_step_id() != self.step
            || !Arc::ptr_eq(step.execution_lane(), &lane)
            || step.claimed_backing().lane_stable_slot_identity().as_ref() != Some(&self.slot)
            || lane.fail_closed.load(Ordering::Acquire)
        {
            return Ok(false);
        }
        let Some(cache) = lane.input_residency.get() else {
            return Ok(false);
        };
        // This is an optional hint. A poisoned cache is permanently unreadable;
        // leave the successful execution unchanged and fall back to uploads.
        let Ok(mut state) = cache.current.lock() else {
            return Ok(false);
        };
        if state.epoch != self.epoch || state.slot.as_ref() != Some(&self.slot) {
            return Ok(false);
        }
        let retired = std::mem::replace(&mut state.windows, self.windows);
        drop(state);
        drop(retired);
        Ok(true)
    }

    pub(crate) fn visit_piece(
        &mut self,
        eligible: bool,
        resource: &ResourceId,
        participant: u32,
        logical_offset: u64,
        physical_resource: &ResourceId,
        physical_offset: u64,
        owner: Weak<dyn Send + Sync>,
        bytes: &[u8],
    ) -> bool {
        let cacheable = eligible
            && bytes.len() <= MAX_INPUT_WINDOW_BYTES
            && self.windows.len() < MAX_INPUT_WINDOWS
            && self
                .content_bytes
                .checked_add(bytes.len())
                .is_some_and(|total| total <= MAX_INPUT_CONTENT_BYTES);
        let reused = cacheable
            && self.previous.iter().any(|window| {
                window.resource == *resource
                    && window.participant == participant
                    && window.logical_offset == logical_offset
                    && window.physical_resource == *physical_resource
                    && window.physical_offset == physical_offset
                    && window.owner.ptr_eq(&owner)
                    && window.owner.strong_count() != 0
                    && window.bytes == bytes
            });
        // Any upload, including an ordinary/non-cache request or another
        // resource alias, revokes overlapping evidence before encoding it.
        let overlaps = |window: &ResidentInputWindow| {
            window.physical_resource == *physical_resource
                && physical_offset
                    < window
                        .physical_offset
                        .saturating_add(window.bytes.len() as u64)
                && window.physical_offset < physical_offset.saturating_add(bytes.len() as u64)
        };
        self.previous.retain(|window| !overlaps(window));
        self.windows.retain(|window| !overlaps(window));
        self.content_bytes = self.windows.iter().map(|window| window.bytes.len()).sum();
        if cacheable {
            self.content_bytes += bytes.len();
            self.windows.push(ResidentInputWindow {
                resource: resource.clone(),
                participant,
                logical_offset,
                physical_resource: physical_resource.clone(),
                physical_offset,
                owner,
                bytes: bytes.to_vec(),
            });
        }
        if reused {
            self.counts.reused_commands = self.counts.reused_commands.saturating_add(1);
            self.counts.reused_bytes = self.counts.reused_bytes.saturating_add(bytes.len() as u64);
        } else {
            self.counts.encoded_commands = self.counts.encoded_commands.saturating_add(1);
            self.counts.encoded_bytes =
                self.counts.encoded_bytes.saturating_add(bytes.len() as u64);
        }
        reused
    }
}

impl<R: DeviceRuntime> ExecutionLane<R> {
    /// Forget all optional input contents, including outstanding publication
    /// tickets. Startup reset and resource-slot replacement require fresh writes.
    pub fn clear_input_residency(&self) -> Result<(), VNextError> {
        let Some(cache) = self.input_residency.get() else {
            return Ok(());
        };
        let Ok(mut state) = cache.current.lock() else {
            return Ok(());
        };
        state.epoch = state.epoch.saturating_add(1);
        state.slot = None;
        let retired = std::mem::take(&mut state.windows);
        drop(state);
        drop(retired);
        Ok(())
    }
}

impl<R: DeviceRuntime> CompletionReservation<R> {
    pub(crate) fn begin_input_residency(
        &self,
        requested: bool,
    ) -> Result<Option<InputResidencyPublication<R>>, VNextError> {
        let lane = self
            .lane
            .as_ref()
            .ok_or_else(|| invalid_completion("input residency reservation has no lane"))?;
        if !requested && lane.input_residency.get().is_none() {
            return Ok(None);
        }
        let slot = self
            .wave()
            .step_resources()
            .claimed_backing()
            .lane_stable_slot_identity();
        let Some(slot) = slot else {
            lane.clear_input_residency()?;
            return Ok(None);
        };
        let cache = lane
            .input_residency
            .get_or_init(InputResidencyCache::default);
        let Ok(mut state) = cache.current.lock() else {
            return Ok(None);
        };
        let previous = std::mem::take(&mut state.windows);
        let same_slot = state.slot.as_ref() == Some(&slot);
        // Exhaustion disables the optimization instead of wrapping identity.
        let Some(epoch) = state.epoch.checked_add(1) else {
            state.slot = None;
            drop(state);
            drop(previous);
            return Ok(None);
        };
        state.epoch = epoch;
        state.slot = Some(slot.clone());
        drop(state);
        let previous = if same_slot {
            previous
        } else {
            drop(previous);
            Vec::new()
        };
        let submission = self
            .receipt
            .as_ref()
            .ok_or_else(|| invalid_completion("input residency reservation has no receipt"))?
            .clone();
        Ok(Some(InputResidencyPublication {
            lane: Arc::downgrade(lane),
            slot,
            epoch,
            submission,
            step: self.wave().batch_step_id(),
            previous,
            windows: Vec::new(),
            content_bytes: 0,
            counts: InputResidencyUploadCounts::default(),
        }))
    }

    pub(crate) fn set_input_residency_publication(
        &mut self,
        publication: Option<InputResidencyPublication<R>>,
    ) {
        self.input_residency_publication = publication;
    }
}

impl<R: DeviceRuntime> CompletionHandle<R> {
    pub fn input_residency_upload_counts(&self) -> Option<InputResidencyUploadCounts> {
        self.input_residency_upload_counts
    }

    /// All cloned handles share one optional ticket; leaving it untaken or
    /// dropping it is a fail-closed cache miss on the next wave.
    pub fn take_input_residency_publication(&self) -> Option<InputResidencyPublication<R>> {
        self.input_residency_publication
            .as_ref()?
            .lock()
            .ok()?
            .take()
    }
}
