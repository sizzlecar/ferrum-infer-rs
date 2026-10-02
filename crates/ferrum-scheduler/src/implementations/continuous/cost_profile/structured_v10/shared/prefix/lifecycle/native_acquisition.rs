//! A declared acknowledged restore establishes the initial preparation
//! frontier. An observed maintenance ordinal advances only its original FIFO;
//! it never advances numerical offers, inference calls or shape/sample counts.
use super::*;
impl Preparation {
    pub(in super::super::super) fn with_native_acquisition(
        plan: StructuredPrefixPlanV5,
        mode: crate::implementations::continuous::cost_profile::structured_v10::lifecycle::LifecycleMode,
        native: Option<StructuredNativePrefixAcquisitionPlanV1>,
    ) -> Self {
        let mut value = Self::with_mode(plan, mode);
        value.native_acquisition = native;
        value
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn native_restored(
        &mut self,
        phase: StructuredProfilePhaseV10,
        cohort: usize,
        slot_index: usize,
        before: Frontier,
        after: Frontier,
        input_tokens_sha256: [u8; 32],
        capture: NativeTransfer,
        restore: NativeTransfer,
        captured_at_ns: u64,
        acknowledged_at_ns: u64,
        expires_at_ns: u64,
        acknowledged: bool,
        maintenance_fifo: Option<u64>,
        progress: &mut Progress<'_>,
    ) -> Result<bool, CostProfileError> {
        progress.lifecycle.active(progress.phase, cohort)?;
        let declared = self
            .native_acquisition
            .as_ref()
            .and_then(|plan| plan.phases.get(progress.phase))
            .and_then(|phase| phase.get(cohort))
            .and_then(Option::as_ref)
            .ok_or_else(fail)?;
        let active = self.active.as_mut().ok_or_else(fail)?;
        if phase.index() != progress.phase
            || active.phase != progress.phase
            || active.cohort != cohort
            || self.pending.is_some()
            || active.slots.iter().any(|slot| slot.id.is_none())
            || !acknowledged
            || maintenance_fifo.is_some_and(|fifo| progress.last_fifo.checked_add(1) != Some(fifo))
            || captured_at_ns == 0
            || captured_at_ns > acknowledged_at_ns
            || acknowledged_at_ns >= expires_at_ns
            || acknowledged_at_ns < progress.source_opened_at_ns
            || acknowledged_at_ns < progress.earliest
            || acknowledged_at_ns < *progress.last_finalized
            || input_tokens_sha256 != declared.input_tokens_sha256
            || capture.kind != NativeTransferKind::Capture
            || restore.kind != NativeTransferKind::Restore
            || capture.slot == restore.slot
            || capture.boundary_tokens != declared.boundary_tokens
            || restore.boundary_tokens != declared.boundary_tokens
            || capture.checkpoint_coordinator != restore.checkpoint_coordinator
            || capture.checkpoint_serial != restore.checkpoint_serial
            || capture.plan_hash != restore.plan_hash
            || capture.layout_fingerprint != restore.layout_fingerprint
            || capture.runtime_implementation_fingerprint
                != restore.runtime_implementation_fingerprint
            || capture.device_id != restore.device_id
            || capture.plan_hash != declared.native_scope.plan_hash
            || capture.layout_fingerprint != declared.native_scope.layout_fingerprint
            || capture.runtime_implementation_fingerprint
                != declared.native_scope.runtime_implementation_fingerprint
            || capture.device_id != declared.native_scope.device_id
            || active
                .native_capture
                .as_ref()
                .is_some_and(|previous| previous != &capture)
            || active
                .slots
                .iter()
                .any(|slot| slot.native_restore_slot == Some(restore.slot))
        {
            return Err(fail());
        }
        let max = progress.limits.max_source_field_bytes.get();
        for transfer in [&capture, &restore] {
            if transfer.slot == 0
                || transfer.checkpoint_coordinator == 0
                || transfer.checkpoint_serial == 0
                || transfer.sequence_generation == 0
                || transfer.request_generation == 0
                || [
                    &transfer.plan_hash,
                    &transfer.layout_fingerprint,
                    &transfer.runtime_implementation_fingerprint,
                    &transfer.device_id,
                ]
                .iter()
                .any(|value| value.is_empty() || value.len() > max)
            {
                return Err(fail());
            }
        }
        let slot = active.slots.get_mut(slot_index).ok_or_else(fail)?;
        if slot.id.as_deref() != Some(before.request_id.as_str())
            || slot.frontier.is_some()
            || slot.native_restored
            || slot.released.is_some()
            || before.request_id.is_empty()
            || before.request_id.len() > max
            || before.owner_incarnation == 0
            || before.work_generation == 0
            || before.kv_tokens != 0
            || before.generated_tokens != 0
            || before.model_cache_id.is_some()
            || !before.pending_utf8.is_empty()
            || before.output_accepted_ordinal != 0
            || after.request_id != before.request_id
            || after.owner_incarnation != before.owner_incarnation
            || after.work_generation < before.work_generation
            || after.generated_tokens != before.generated_tokens
            || after.kv_tokens != declared.boundary_tokens
            || after
                .model_cache_id
                .as_ref()
                .is_none_or(|id| id.is_empty() || id.len() > max)
            || after.pending_utf8 != before.pending_utf8
            || after.output_accepted_ordinal != before.output_accepted_ordinal
        {
            return Err(fail());
        }
        slot.frontier = Some(after);
        slot.prompt = Some(declared.prompt_tokens);
        slot.native_restored = true;
        slot.native_restore_slot = Some(restore.slot);
        active.native_capture.get_or_insert(capture);
        *progress.last_finalized = acknowledged_at_ns;
        if let Some(fifo) = maintenance_fifo {
            *progress.last_fifo = fifo;
        }
        Ok(false)
    }
}
