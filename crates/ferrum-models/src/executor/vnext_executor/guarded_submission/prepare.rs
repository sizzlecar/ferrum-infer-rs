//! Selected waves consume resident capacity once. Growth, eviction and prefix
//! maintenance invalidate the original planning evidence and belong to a later
//! controller iteration, never to this guarded preparation.
use super::*;
use crate::executor::vnext_executor::execution_maintenance::MaintenanceSource;

fn deferred(
    deferred: &AdmissionDeferred,
    stage: ExecutorExecutionCapacityStage,
) -> Result<ExecutorExecutionCapacityDeferral> {
    match deferred.action() {
        DeferredAction::WaitForRelease => {
            ExecutorExecutionCapacityDeferral::from_admission(deferred, stage)
        }
        DeferredAction::AwaitBackingGrowth => {
            ExecutorExecutionCapacityDeferral::from_pending_maintenance(deferred, stage)
        }
        _ => Err(FerrumError::resource_exhausted(
            "selected wave requires an unsupported capacity action",
        )),
    }
}

impl<R: DeviceRuntime> VNextModelExecutor<R> {
    pub(in crate::executor::vnext_executor) fn guarded_extend_once(
        &self,
        sequence: &VNextSequence<R>,
        target: ResourceWorkShape,
        guarded: &GuardedExecution<'_>,
    ) -> Result<VNextExecutionCapacityDecision<()>> {
        if !sequence.active.load(Ordering::Acquire) {
            return Err(FerrumError::cancelled(
                "selected sequence is no longer active",
            ));
        }
        let request =
            SequenceResourceExtensionRequest::new(target, AdmissionPressureAction::WaitForRelease)
                .map_err(|error| FerrumError::backend(error.to_string()))?;
        let stage = ExecutorExecutionCapacityStage::SequenceExtension;
        match sequence
            .session
            .try_ensure_backing_covers(request)
            .map_err(|error| FerrumError::backend(error.to_string()))?
        {
            SequenceResourceExtensionDecision::Current(_)
            | SequenceResourceExtensionDecision::Extended(_) => {
                Ok(VNextExecutionCapacityDecision::Ready(()))
            }
            SequenceResourceExtensionDecision::Deferred(value) => {
                let evidence = deferred(&value, stage)?;
                if value.action() == DeferredAction::AwaitBackingGrowth {
                    self.retain_guarded_maintenance(
                        guarded,
                        stage,
                        MaintenanceSource::Logical(value),
                    )?;
                }
                Ok(VNextExecutionCapacityDecision::Deferred(evidence))
            }
            SequenceResourceExtensionDecision::BackingDeferred(value) => {
                let evidence =
                    ExecutorExecutionCapacityDeferral::from_backing(value.evidence(), stage)?;
                self.retain_guarded_maintenance(
                    guarded,
                    stage,
                    MaintenanceSource::Sequence(value),
                )?;
                Ok(VNextExecutionCapacityDecision::Deferred(evidence))
            }
            SequenceResourceExtensionDecision::RetryRequired(_) => {
                guarded.replan_before_encode.store(true, Ordering::Release);
                Err(FerrumError::resource_exhausted(
                    "selected sequence frame changed before encode",
                ))
            }
            SequenceResourceExtensionDecision::PermanentRejected(value) => {
                Err(FerrumError::request_validation(format!(
                    "selected sequence exceeds immutable fit: {value:?}"
                )))
            }
        }
    }

    pub(in crate::executor::vnext_executor) fn guarded_begin_step_once(
        &self,
        batch: &ExecutionBatchParticipants<R>,
        spans: &[TokenSpanWork],
        kind: VNextExecutionWaveKind,
        guarded: &GuardedExecution<'_>,
    ) -> Result<VNextExecutionCapacityDecision<Arc<StepResourceLease<R>>>> {
        let stage = ExecutorExecutionCapacityStage::StepAdmission;
        match self.try_begin_step_for_spans(batch, spans, kind)? {
            StepResourceAdmissionDecision::Admitted(step) => {
                Ok(VNextExecutionCapacityDecision::Ready(step))
            }
            StepResourceAdmissionDecision::Deferred(value) => {
                let evidence = deferred(&value, stage)?;
                if value.action() == DeferredAction::AwaitBackingGrowth {
                    self.retain_guarded_maintenance(
                        guarded,
                        stage,
                        MaintenanceSource::Logical(value),
                    )?;
                }
                Ok(VNextExecutionCapacityDecision::Deferred(evidence))
            }
            StepResourceAdmissionDecision::BackingDeferred(value) => {
                let evidence =
                    ExecutorExecutionCapacityDeferral::from_backing(value.evidence(), stage)?;
                self.retain_guarded_maintenance(guarded, stage, MaintenanceSource::Step(value))?;
                Ok(VNextExecutionCapacityDecision::Deferred(evidence))
            }
            StepResourceAdmissionDecision::PermanentRejected(value) => Err(FerrumError::backend(
                format!("selected Step exceeds immutable plan: {value:?}"),
            )),
        }
    }

    pub(in crate::executor::vnext_executor) fn guarded_prepare_wave_once(
        &self,
        step: &Arc<StepResourceLease<R>>,
        sequences: &[Arc<VNextSequence<R>>],
        spans: &[TokenSpanWork],
        guarded: &GuardedExecution<'_>,
    ) -> Result<VNextExecutionCapacityDecision<PreparedStepSubmissionWave<R>>> {
        Self::validate_step_maintenance_participants(step, sequences)?;
        let stage = ExecutorExecutionCapacityStage::SubmissionWave;
        match self.try_prepare_wave_for_spans(step, spans)? {
            StepSubmissionWaveAdmissionDecision::Prepared(wave) => {
                self.metrics.prepared_wave_topology.record(&wave);
                Ok(VNextExecutionCapacityDecision::Ready(wave))
            }
            StepSubmissionWaveAdmissionDecision::Deferred(value) => {
                let evidence = deferred(&value, stage)?;
                if value.action() == DeferredAction::AwaitBackingGrowth {
                    self.retain_guarded_maintenance(
                        guarded,
                        stage,
                        MaintenanceSource::Logical(value),
                    )?;
                }
                Ok(VNextExecutionCapacityDecision::Deferred(evidence))
            }
            StepSubmissionWaveAdmissionDecision::BackingDeferred(value) => {
                let evidence =
                    ExecutorExecutionCapacityDeferral::from_backing(value.evidence(), stage)?;
                self.retain_guarded_maintenance(
                    guarded,
                    stage,
                    MaintenanceSource::PendingWave(value),
                )?;
                Ok(VNextExecutionCapacityDecision::Deferred(evidence))
            }
            StepSubmissionWaveAdmissionDecision::RequestStateDeferred(value) => {
                let mut requests = Vec::new();
                for sequence in sequences {
                    if value.blockers().iter().any(|blocker| {
                        blocker.request() == sequence.session.resources().request_authority()
                    }) && !requests.contains(sequence.request_id())
                    {
                        requests.push(sequence.request_id().clone());
                    }
                }
                ExecutorRequestStateDeferral::new(stage, requests, value)
                    .map(VNextExecutionCapacityDecision::RequestStateDeferred)
            }
            StepSubmissionWaveAdmissionDecision::PermanentRejected(value) => Err(
                FerrumError::backend(format!("selected wave exceeds immutable plan: {value:?}")),
            ),
            StepSubmissionWaveAdmissionDecision::RequestStateSplitRequired(_) => Err(
                FerrumError::internal("selected wave requires a different sibling-state topology"),
            ),
            StepSubmissionWaveAdmissionDecision::RequestStatePoisoned(value) => Err(
                FerrumError::backend(format!("selected Request-state is poisoned: {value:?}")),
            ),
        }
    }
}
