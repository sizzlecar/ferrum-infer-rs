use super::*;

impl<R: DeviceRuntime> PreparedTensorExecutor<R> {
    pub(super) fn admit_sequence(
        &self,
        work: &ResourceWorkShape,
        run_id: RunId,
        request_id: RequestIdentity,
    ) -> Result<Arc<AdmittedSequenceResources<R>>, TensorExecutionError<R>> {
        let request = RequestResourceAdmissionRequest::new(
            work.clone(),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )?;
        let sequence = SequenceResourceAdmissionRequest::new(
            work.clone(),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )?;
        for pass in 0..=self.config.maximum_backing_maintenance_passes {
            match self.binding.runtime_binding().try_admit_initial_sequence(
                request.clone(),
                sequence.clone(),
                run_id.clone(),
                request_id.clone(),
            )? {
                InitialSequenceResourceAdmissionDecision::Admitted(sequence) => {
                    return Ok(sequence)
                }
                InitialSequenceResourceAdmissionDecision::BackingDeferred(deferred) => {
                    self.check_maintenance_limit(TensorAdmissionStage::Request, pass)?;
                    deferred.maintain()?;
                }
                InitialSequenceResourceAdmissionDecision::Deferred(reason) => {
                    return Err(admission(
                        TensorAdmissionStage::Request,
                        TensorAdmissionFailure::Deferred,
                        format!("{reason:?}"),
                    ))
                }
                InitialSequenceResourceAdmissionDecision::PermanentRejected(reason) => {
                    return Err(admission(
                        TensorAdmissionStage::Request,
                        TensorAdmissionFailure::Rejected,
                        format!("{reason:?}"),
                    ))
                }
            }
        }
        Err(invalid("tensor initial admission exhausted its bounded loop").into())
    }

    pub(super) fn admit_step(
        &self,
        batch: &ExecutionBatchParticipants<R>,
        lane: &Arc<ExecutionLane<R>>,
        work: &ResourceWorkShape,
    ) -> Result<Arc<StepResourceLease<R>>, TensorExecutionError<R>> {
        let request = StepResourceAdmissionRequest::new(
            batch.bind_tensor_work_shape(work)?,
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )?;
        for pass in 0..=self.config.maximum_backing_maintenance_passes {
            let decision = if tracing::enabled!(target: "ferrum_execution::tensor::admission", tracing::Level::DEBUG)
            {
                let mut elapsed = [0u64; 6];
                let result = batch.try_begin_step_profiled(request.clone(), lane, |phase, time| {
                    use StepResourceAdmissionProfilePhase::*;
                    let index = match phase {
                        AuthorityAndPolicyValidate => 0,
                        DemandEvaluate => 1,
                        BackingClaim => 2,
                        LogicalCapacityClaim => 3,
                        TransactionValidateAndFingerprint => 4,
                        FrameCaptureAndLease => 5,
                    };
                    elapsed[index] = elapsed[index]
                        .saturating_add(u64::try_from(time.as_micros()).unwrap_or(u64::MAX));
                });
                // Emit after the original admission returns, outside its locks.
                tracing::debug!(target: "ferrum_execution::tensor::admission",
                    pass, authority_us=elapsed[0], demand_us=elapsed[1],
                    backing_us=elapsed[2], logical_us=elapsed[3],
                    transaction_us=elapsed[4], frame_us=elapsed[5],
                    "tensor step admission host timing");
                result
            } else {
                batch.try_begin_step(request.clone(), lane)
            }?;
            match decision {
                StepResourceAdmissionDecision::Admitted(step) => return Ok(step),
                StepResourceAdmissionDecision::BackingDeferred(deferred) => {
                    self.check_maintenance_limit(TensorAdmissionStage::Step, pass)?;
                    let outcome = deferred.maintain()?;
                    tracing::debug!(pass, ?outcome, "tensor step backing maintenance outcome");
                }
                StepResourceAdmissionDecision::Deferred(reason) => {
                    return Err(admission(
                        TensorAdmissionStage::Step,
                        TensorAdmissionFailure::Deferred,
                        format!("{reason:?}"),
                    ))
                }
                StepResourceAdmissionDecision::PermanentRejected(reason) => {
                    return Err(admission(
                        TensorAdmissionStage::Step,
                        TensorAdmissionFailure::Rejected,
                        format!("{reason:?}"),
                    ))
                }
            }
        }
        Err(invalid("tensor step admission exhausted its bounded loop").into())
    }

    pub(super) fn prepare_wave(
        &self,
        step: &Arc<StepResourceLease<R>>,
    ) -> Result<PreparedStepSubmissionWave<R>, TensorExecutionError<R>> {
        let work = step.shared_tensor_invocation_work_shape()?;
        for pass in 0..=self.config.maximum_backing_maintenance_passes {
            // The executor always submits the complete immutable plan. Keep
            // the shared work authority and ordinary per-step admission while
            // avoiding a cloned request vector for already committed topology.
            match step.try_prepare_full_plan_submission_wave(
                work.clone(),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )? {
                StepSubmissionWaveAdmissionDecision::Prepared(wave) => return Ok(wave),
                StepSubmissionWaveAdmissionDecision::BackingDeferred(deferred) => {
                    self.check_maintenance_limit(TensorAdmissionStage::Wave, pass)?;
                    deferred.maintain()?;
                }
                StepSubmissionWaveAdmissionDecision::Deferred(reason) => {
                    return Err(admission(
                        TensorAdmissionStage::Wave,
                        TensorAdmissionFailure::Deferred,
                        format!("{reason:?}"),
                    ))
                }
                StepSubmissionWaveAdmissionDecision::PermanentRejected(reason) => {
                    return Err(admission(
                        TensorAdmissionStage::Wave,
                        TensorAdmissionFailure::Rejected,
                        format!("{reason:?}"),
                    ))
                }
                StepSubmissionWaveAdmissionDecision::RequestStateDeferred(_)
                | StepSubmissionWaveAdmissionDecision::RequestStateSplitRequired(_)
                | StepSubmissionWaveAdmissionDecision::RequestStatePoisoned(_) => {
                    return Err(admission(
                        TensorAdmissionStage::Wave,
                        TensorAdmissionFailure::RequestState,
                        "fixed tensor work cannot split or bypass a request-state hazard"
                            .to_owned(),
                    ))
                }
            }
        }
        Err(invalid("tensor wave admission exhausted its bounded loop").into())
    }

    fn check_maintenance_limit(
        &self,
        stage: TensorAdmissionStage,
        pass: usize,
    ) -> Result<(), TensorExecutionError<R>> {
        if pass >= self.config.maximum_backing_maintenance_passes {
            return Err(admission(
                stage,
                TensorAdmissionFailure::MaintenanceLimit,
                "physical backing still needs maintenance".to_owned(),
            ));
        }
        Ok(())
    }
}

fn admission<R: DeviceRuntime>(
    stage: TensorAdmissionStage,
    kind: TensorAdmissionFailure,
    reason: String,
) -> TensorExecutionError<R> {
    TensorExecutionError::Admission {
        stage,
        kind,
        reason,
    }
}
