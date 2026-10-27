use super::*;

pub(super) enum AdmissionProgress<T> {
    Ready(T),
    Deferred(CapacityWaitCondition),
}

impl<R: DeviceRuntime> PreparedTensorExecutor<R> {
    pub(super) fn admit_sequence(
        &self,
        work: &ResourceWorkShape,
        run_id: RunId,
        request_id: RequestIdentity,
        control: Option<&TensorAdmissionControl>,
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
        let mut maintenance = 0;
        loop {
            if control.is_some_and(TensorAdmissionControl::is_cancelled) {
                return Err(TensorExecutionError::CancelledBeforeSubmission(
                    TensorAdmissionCancellation::retired(),
                ));
            }
            let wait = match self.binding.runtime_binding().try_admit_initial_sequence(
                request.clone(),
                sequence.clone(),
                run_id.clone(),
                request_id.clone(),
            )? {
                InitialSequenceResourceAdmissionDecision::Admitted(sequence) => {
                    return Ok(sequence)
                }
                InitialSequenceResourceAdmissionDecision::BackingDeferred(deferred) => {
                    self.check_maintenance_limit(TensorAdmissionStage::Request, maintenance)?;
                    maintenance_progress(deferred.maintain()?, &mut maintenance)
                }
                InitialSequenceResourceAdmissionDecision::Deferred(reason) => {
                    Some(reason.wait_condition().clone())
                }
                InitialSequenceResourceAdmissionDecision::PermanentRejected(reason) => {
                    return Err(admission(
                        TensorAdmissionStage::Request,
                        TensorAdmissionFailure::Rejected,
                        format!("{reason:?}"),
                    ))
                }
            };
            if let Some(wait) = wait {
                if !self.wait_for_capacity(TensorAdmissionStage::Request, wait, control)? {
                    // Neither request nor sequence admission committed.
                    return Err(TensorExecutionError::CancelledBeforeSubmission(
                        TensorAdmissionCancellation::retired(),
                    ));
                }
            }
        }
    }

    pub(super) fn admit_step(
        &self,
        batch: &ExecutionBatchParticipants<R>,
        lane: &Arc<ExecutionLane<R>>,
        work: &ResourceWorkShape,
        maintenance: &mut usize,
    ) -> Result<AdmissionProgress<Arc<StepResourceLease<R>>>, TensorExecutionError<R>> {
        let request = StepResourceAdmissionRequest::new(
            batch.bind_tensor_work_shape(work)?,
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )?;
        loop {
            match batch.try_begin_step(request.clone(), lane)? {
                StepResourceAdmissionDecision::Admitted(step) => {
                    return Ok(AdmissionProgress::Ready(step))
                }
                StepResourceAdmissionDecision::BackingDeferred(deferred) => {
                    self.check_maintenance_limit(TensorAdmissionStage::Step, *maintenance)?;
                    if let Some(wait) = maintenance_progress(deferred.maintain()?, maintenance) {
                        return Ok(AdmissionProgress::Deferred(wait));
                    }
                }
                StepResourceAdmissionDecision::Deferred(reason) => {
                    return Ok(AdmissionProgress::Deferred(reason.wait_condition().clone()))
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
    }

    pub(super) fn prepare_wave(
        &self,
        step: &Arc<StepResourceLease<R>>,
        maintenance: &mut usize,
    ) -> Result<AdmissionProgress<PreparedStepSubmissionWave<R>>, TensorExecutionError<R>> {
        let work = step.shared_tensor_invocation_work_shape()?;
        loop {
            match step.try_prepare_full_plan_submission_wave(
                work.clone(),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )? {
                StepSubmissionWaveAdmissionDecision::Prepared(wave) => {
                    return Ok(AdmissionProgress::Ready(wave))
                }
                StepSubmissionWaveAdmissionDecision::BackingDeferred(deferred) => {
                    self.check_maintenance_limit(TensorAdmissionStage::Wave, *maintenance)?;
                    if let Some(wait) = maintenance_progress(deferred.maintain()?, maintenance) {
                        // The deferral and its step clone drop before rollback.
                        return Ok(AdmissionProgress::Deferred(wait));
                    }
                }
                StepSubmissionWaveAdmissionDecision::Deferred(reason) => {
                    return Ok(AdmissionProgress::Deferred(reason.wait_condition().clone()))
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
    }

    pub(super) fn wait_for_capacity(
        &self,
        stage: TensorAdmissionStage,
        condition: CapacityWaitCondition,
        control: Option<&TensorAdmissionControl>,
    ) -> Result<bool, TensorExecutionError<R>> {
        let Some(control) = control else {
            return Err(admission(
                stage,
                TensorAdmissionFailure::Deferred,
                format!("{condition:?}"),
            ));
        };
        if control.is_cancelled() {
            return Ok(false);
        }
        // Subscribe then recheck the original observation: no lost release.
        let registration = self.resources.register_capacity_waiter(&condition)?;
        control.wait(registration).map_err(Into::into)
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

fn maintenance_progress(
    outcome: DynamicDeferredMaintenanceOutcome,
    passes: &mut usize,
) -> Option<CapacityWaitCondition> {
    match outcome {
        DynamicDeferredMaintenanceOutcome::WaitForRelease { wait_condition, .. } => {
            Some(wait_condition)
        }
        DynamicDeferredMaintenanceOutcome::RetryAdmission { .. }
        | DynamicDeferredMaintenanceOutcome::Maintained(_) => {
            *passes += 1;
            None
        }
    }
}

#[cfg(test)]
#[allow(dead_code)]
#[path = "../../tests/capacity_boundary/fixture.rs"]
mod resource_fixture;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn capacity_wait_does_not_spend_the_backing_maintenance_budget() {
        use super::resource_fixture as f;
        let catalog = f::catalog();
        let (plan, bucket) = f::plan(&catalog, true, 3);
        let (competing_plan, _) = f::plan(&catalog, false, 128);
        let runtime = f::runtime(&plan);
        let lane = ExecutionLane::create(runtime.clone()).unwrap();
        let root = f::provision(&plan, runtime.clone(), "tensor-wait-budget");
        let first = f::sequence(&root, "tensor-wait-budget.first");
        let second = f::sequence(&root, "tensor-wait-budget.second");
        let first_session = first.open_session().unwrap();
        let second_session = second.open_session().unwrap();
        let first_batch = ExecutionBatchParticipants::new(vec![first_session.clone()]).unwrap();
        let second_batch = ExecutionBatchParticipants::new(vec![second_session.clone()]).unwrap();
        let first_step = f::step(&first_batch, &lane, bucket.as_ref());
        let first_wave = f::wave(&first_step);
        let second_step = f::step(&second_batch, &lane, bucket.as_ref());
        let StepSubmissionWaveAdmissionDecision::BackingDeferred(deferred) =
            f::try_wave(&second_step)
        else {
            panic!("occupied reusable wave must defer");
        };
        let competing = f::provision_competing(&competing_plan, runtime);
        let mut passes = 0;
        // Cross the configured maintenance bound using actual blocked outcomes;
        // none may spend a growth/retry pass merely because capacity is busy.
        for _ in 0..=TensorExecutorConfig::default().maximum_backing_maintenance_passes {
            let outcome = deferred.maintain().unwrap();
            assert!(matches!(
                &outcome,
                DynamicDeferredMaintenanceOutcome::WaitForRelease { .. }
            ));
            let condition = maintenance_progress(outcome, &mut passes).unwrap();
            assert_eq!(passes, 0);
            assert!(!root
                .register_capacity_waiter(&condition)
                .unwrap()
                .recheck()
                .unwrap()
                .should_retry());
        }
        drop(first_wave);
        drop(deferred);
        let StepSubmissionWaveAdmissionDecision::Prepared(second_wave) = f::try_wave(&second_step)
        else {
            panic!("released slot must admit without more maintenance");
        };
        drop(second_wave);
        first_step.try_retire_normal().unwrap();
        second_step.try_retire_normal().unwrap();
        drop((first_batch, second_batch));
        first_session.try_complete().unwrap();
        second_session.try_complete().unwrap();
        drop((first_session, second_session, first, second, lane));
        f::close(root);
        f::close(competing);
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
