//! One explicitly passed selected-wave guard. It never owns a global wave slot.
use super::*;
#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod metal_tests;
mod prefills;
mod prepare;
use ferrum_interfaces::execution_cost::{
    ActualRowWork, ActualWaveKind, ExpectedExecutionWave, ExpectedWaveInput,
    GuardedCostObservation, GuardedDispatchOutcome, GuardedNotSubmitted, GuardedNotSubmittedReason,
    NonblockingHostSubmissionGuard, WaveCommitment,
};

pub(super) struct GuardedExecution<'a> {
    pub(super) expected: &'a ExpectedExecutionWave,
    host: &'a dyn NonblockingHostSubmissionGuard,
    reconciled: Mutex<Option<GuardedNotSubmitted>>,
    replan_before_encode: AtomicBool,
    pub(super) maintenance: Mutex<Option<Box<dyn std::any::Any + Send + Sync>>>,
}
impl GuardedExecution<'_> {
    pub(super) fn preserves_request(&self) -> bool {
        self.reconciled.lock().is_some() || self.replan_before_encode.load(Ordering::Acquire)
    }
    /// A failed Ready restoration is terminal, even if the device Step was
    /// rolled back. Do not expose a reusable receipt after its owner vanished.
    fn invalidate_preservation(&self) {
        self.reconciled.lock().take();
        self.replan_before_encode.store(false, Ordering::Release);
        self.maintenance.lock().take();
    }
    pub(super) fn record_reconciled(&self, receipt: GuardedNotSubmitted) {
        *self.reconciled.lock() = Some(receipt);
    }
}

pub(super) struct ActualPreparedGuard<'a, 'b, R: DeviceRuntime> {
    pub executor: &'a VNextModelExecutor<R>,
    pub selected: &'a GuardedExecution<'b>,
    pub participants: &'a [VNextExecutionParticipant<'a, R>],
    pub kind: VNextExecutionWaveKind,
    pub output: VNextProductOutputMode,
    pub masks: &'a [VNextProductTokenMaskSubmissionPlan],
}

impl<R: DeviceRuntime> ActualPreparedGuard<'_, '_, R> {
    fn check_work(&self) -> std::result::Result<(), GuardedNotSubmittedReason> {
        let work = self.selected.expected.work();
        let kind = match self.kind {
            VNextExecutionWaveKind::Prefill => ActualWaveKind::Prefill,
            VNextExecutionWaveKind::Decode => ActualWaveKind::Decode,
            VNextExecutionWaveKind::Mixed => ActualWaveKind::Mixed,
        };
        if work.lane_id() != self.executor.lane.id()
            || work.plan_hash() != self.executor.resolved_plan.execution_plan().plan_hash()
            || work.kind() != kind
            || work.participants().len() != self.participants.len()
        {
            return Err(GuardedNotSubmittedReason::ResourceClaimMismatch);
        }
        for (actual, expected) in self.participants.iter().zip(work.participants()) {
            let selected = expected.selection();
            if actual.sequence.request_id() != &selected.request_id
                || !actual.sequence.active.load(Ordering::Acquire)
                || actual.sequence.session.resources().coordinator_id() != work.coordinator_id()
                || !expected
                    .resource()
                    .matches_session_identity(&actual.sequence.session)
            {
                return Err(GuardedNotSubmittedReason::ResourceClaimMismatch);
            }
            let range = actual.span.immediate_token_range();
            let matches = match (&selected.input, selected.work, actual.output_role) {
                (
                    ExpectedWaveInput::Decode { cache_id },
                    ActualRowWork::Decode { kv_tokens },
                    VNextParticipantOutputRole::Decode(policy),
                ) => {
                    cache_id == &actual.sequence.cache_id
                        && range.start == u64::from(kv_tokens)
                        && range.end.checked_sub(range.start) == Some(1)
                        && selected
                            .decode_policy
                            .as_ref()
                            .is_some_and(|captured| captured.same_captured_input(policy))
                }
                (
                    ExpectedWaveInput::Prefill { chunk },
                    ActualRowWork::Prefill {
                        offset,
                        count,
                        total_prompt_tokens,
                    },
                    role @ (VNextParticipantOutputRole::IntermediatePrefill
                    | VNextParticipantOutputRole::FinalPrefill),
                ) => {
                    range.start == u64::from(offset)
                        && range.end.checked_sub(range.start) == Some(u64::from(count))
                        && actual.span.full_input_tokens() == u64::from(total_prompt_tokens)
                        && chunk.is_final()
                            == matches!(role, VNextParticipantOutputRole::FinalPrefill)
                }
                _ => false,
            };
            if !matches {
                return Err(GuardedNotSubmittedReason::ActualRouteMismatch);
            }
        }
        Ok(())
    }
}

impl<R: DeviceRuntime> PreparedWaveSubmissionGuard for ActualPreparedGuard<'_, '_, R> {
    fn check(
        &self,
        device: &DeviceSubmissionAttribution,
        readback: ferrum_interfaces::execution_cost::CoreReadbackRoute,
    ) -> std::result::Result<(), GuardedNotSubmittedReason> {
        self.check_work()?;
        // Core has already bound this native command to the exact Step/wave.
        // Only the cost commitment additionally requires a predicted route.
        if let WaveCommitment::CostWitness(expected) = self.selected.expected.commitment() {
            let actual = cost_observation::dispatch::actual_shape_from_device(
                self.executor,
                self.participants,
                self.kind,
                self.output,
                self.masks,
                Some(device),
                0,
                readback,
                |id| {
                    expected
                        .participants()
                        .iter()
                        .find(|row| &row.request_id == id)
                        .map(|row| &row.host)
                },
            )
            .map_err(|_| GuardedNotSubmittedReason::AttributionUnavailable)?;
            let shape = expected.canonical();
            if actual.kind != shape.kind
                || actual.path != shape.path
                || actual.graph != shape.graph
                || actual.row_order != shape.row_order
                || actual.provider_signature != shape.provider_signature
                || actual.output_policy_signature != shape.output_policy_signature
                || actual.numeric_features != shape.numeric_features
                || actual.host_content_features != shape.host_content_features
                || actual.row_multiset_features != shape.row_multiset_features
                || actual.recurrent_state_bytes != shape.recurrent_state_bytes
                || actual
                    .rows
                    .iter()
                    .map(|row| row.work)
                    .ne(shape.rows.iter().copied())
                || actual.restore_bytes != 0
                || actual.maintenance_bytes != 0
                || actual.maintenance_units != 0
            {
                #[cfg(all(test, feature = "metal", target_os = "macos"))]
                eprintln!(
                "guarded route mismatch: kind={:?} expected_work={:?} actual_work={:?}; unequal(kind,path,graph,order,provider,output,numeric,recurrent)=({},{},{},{},{},{},{},{}); actual_mask_uploads={:?}; actual_commands={}",
                self.kind,
                &shape.rows[..shape.rows.len().min(4)],
                actual.rows.iter().take(4).map(|row| row.work).collect::<Vec<_>>(),
                actual.kind != shape.kind,
                actual.path != shape.path,
                actual.graph != shape.graph,
                actual.row_order != shape.row_order,
                actual.provider_signature != shape.provider_signature,
                actual.output_policy_signature != shape.output_policy_signature,
                actual.numeric_features != shape.numeric_features,
                actual.recurrent_state_bytes != shape.recurrent_state_bytes,
                self.masks.iter().take(4).map(|mask| mask.upload_required).collect::<Vec<_>>(),
                device.commands().len(),
            );
                return Err(GuardedNotSubmittedReason::ActualRouteMismatch);
            }
        }
        // Last check: no more potentially blocking work belongs between this
        // callback and the backend's irreversible native commit.
        self.selected
            .host
            .check()
            .map_err(GuardedNotSubmittedReason::HostRejected)
    }
}

impl<R: DeviceRuntime> VNextModelExecutor<R> {
    fn supports_guarded_execution_policy(&self) -> bool {
        self.runtime.supports_guarded_submission()
            && self.checkpoint_capture.is_none()
            && self.diagnostic_fault.is_none()
            && self.device_timing_mode() == DeviceTimingMode::Off
            && self
                .resolved_plan
                .execution_plan()
                .payload()
                .memory()
                .reusable_execution()
                .and_then(|plan| plan.program_policy())
                .is_none()
    }

    pub(super) fn supports_slo_execution(&self) -> bool {
        self.supports_guarded_execution_policy()
            && self
                .resolved_plan
                .execution_plan()
                .payload()
                .memory()
                .checkpoint_capacity()
                .is_none()
            && self.future_cost_policy().is_ok()
    }

    fn supports_guarded_work(&self, expected: &ExpectedExecutionWave) -> bool {
        self.supports_guarded_execution_policy()
            && match expected.commitment() {
                WaveCommitment::CostWitness(_) => self.future_cost_policy().is_ok(),
                WaveCommitment::CompleteRequests(_) => true,
            }
    }

    pub(super) async fn execute_guarded_decode(
        &self,
        inputs: &[PlanRuntimeDecodeInput],
        expected: &ExpectedExecutionWave,
        host: &dyn NonblockingHostSubmissionGuard,
        mut observation: GuardedCostObservation<'_, '_>,
    ) -> GuardedDispatchOutcome<Vec<PlanRuntimeDecodeOutput>> {
        if !self.supports_guarded_work(expected)
            || expected.work().kind() != ActualWaveKind::Decode
            || inputs.is_empty()
            || inputs.len() != expected.work().participants().len()
            || inputs
                .iter()
                .zip(expected.work().participants())
                .any(|(input, row)| {
                    let row = row.selection();
                    input.request_id != row.request_id
                        || !matches!(&row.input, ExpectedWaveInput::Decode { cache_id }
                            if cache_id == &input.kv_cache.cache_id())
                        || !row
                            .decode_policy
                            .as_ref()
                            .is_some_and(|policy| policy.same_captured_input(&input.logits_policy))
                })
        {
            return GuardedDispatchOutcome::Unsupported;
        }
        let selected = GuardedExecution {
            expected,
            host,
            reconciled: Mutex::new(None),
            replan_before_encode: AtomicBool::new(false),
            maintenance: Mutex::new(None),
        };
        let result = self
            .execute_plan_runtime_decode_batch_inner(
                inputs,
                observation.as_deref_mut(),
                Some(&selected),
            )
            .await;
        if selected.replan_before_encode.load(Ordering::Acquire) {
            if let Some(observation) = observation {
                observation.finish_call(ObservedCallOutcome::NotSubmitted);
            }
            return GuardedDispatchOutcome::ReplanBeforeEncode;
        }
        if let Some(receipt) = selected.reconciled.lock().take() {
            if let Some(observation) = observation {
                observation.finish_call(ObservedCallOutcome::NotSubmitted);
            }
            return GuardedDispatchOutcome::NotSubmittedAfterPreparation(receipt);
        }
        if let Some(observation) = observation {
            observation.finish_call(match &result {
                Ok(PlanRuntimeBatchDecodeOutcome::Completed(_)) => ObservedCallOutcome::Completed,
                Ok(PlanRuntimeBatchDecodeOutcome::Deferred(_)) => ObservedCallOutcome::Deferred,
                Err(_) => ObservedCallOutcome::Failed,
            });
        }
        let outcome = match result {
            Ok(PlanRuntimeBatchDecodeOutcome::Completed(outputs)) => {
                GuardedDispatchOutcome::Submitted(Ok(outputs))
            }
            Ok(PlanRuntimeBatchDecodeOutcome::Deferred(reason)) => {
                GuardedDispatchOutcome::Deferred(reason)
            }
            Err(error) => GuardedDispatchOutcome::Submitted(Err(error)),
        };
        self.attach_guarded_maintenance(&selected, outcome)
    }
}
