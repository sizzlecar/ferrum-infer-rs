//! Separate, one-use maintenance continuation. No model work is executed here.
use super::*;
use ferrum_interfaces::execution_cost::{
    ExpectedWaveWork, GuardedDispatchOutcome, HostSubmissionRejection,
    NonblockingHostSubmissionGuard,
};
use ferrum_interfaces::model_executor::{
    ExecutorExecutionMaintenanceOutcome, ExecutorExecutionMaintenanceProgress,
    ExecutorExecutionMaintenanceTicket,
};

pub(super) enum MaintenanceSource<R: DeviceRuntime> {
    Logical(AdmissionDeferred),
    Sequence(SequenceExtensionBackingDeferral<R>),
    Step(StepAdmissionBackingDeferral<R>),
    PendingWave(StepSubmissionWaveBackingDeferral<R>),
    ReconciledWave(ReconciledSubmissionWaveMaintenance<R>),
}

struct PendingMaintenance<R: DeviceRuntime> {
    source: MaintenanceSource<R>,
    stage: ExecutorExecutionCapacityStage,
    plan: Weak<PlanRuntimeResources<R>>,
    owners: Vec<Weak<VNextSequence<R>>>,
    work: ExpectedWaveWork,
}

impl<R: DeviceRuntime> VNextModelExecutor<R> {
    pub(super) fn retain_guarded_maintenance(
        &self,
        selected: &GuardedExecution<'_>,
        stage: ExecutorExecutionCapacityStage,
        source: MaintenanceSource<R>,
    ) -> Result<()> {
        let registry = self.sequences.lock();
        let owners = selected
            .expected
            .work()
            .participants()
            .iter()
            .map(|row| {
                let selection = row.selection();
                let sequence = match &selection.input {
                    ferrum_interfaces::execution_cost::ExpectedWaveInput::Decode { cache_id } => {
                        registry.active.get(cache_id).cloned()
                    }
                    ferrum_interfaces::execution_cost::ExpectedWaveInput::Prefill { .. } => {
                        registry
                            .prefills
                            .get(&selection.request_id)
                            .and_then(|slot| {
                                if slot.cancelled.load(Ordering::Acquire) {
                                    return None;
                                }
                                match &*slot.state.lock() {
                                    VNextPrefillSlotState::Ready(sequence)
                                    | VNextPrefillSlotState::Executing(sequence) => {
                                        Some(Arc::clone(sequence))
                                    }
                                    _ => None,
                                }
                            })
                    }
                }
                .filter(|sequence| {
                    sequence.active.load(Ordering::Acquire)
                        && row.resource().matches_session_identity(&sequence.session)
                });
                sequence.map(|value| Arc::downgrade(&value)).ok_or_else(|| {
                    FerrumError::cancelled("maintenance owner changed during failed acquire")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        drop(registry);
        let mut slot = selected.maintenance.lock();
        if slot.is_some() {
            return Err(FerrumError::internal(
                "guarded maintenance slot already occupied",
            ));
        }
        *slot = Some(Box::new(PendingMaintenance {
            source,
            stage,
            plan: Arc::downgrade(&self.plan_resources),
            owners,
            work: selected.expected.work().clone(),
        }));
        Ok(())
    }

    pub(super) fn reconcile_guarded_maintenance_step(
        &self,
        step: Arc<StepResourceLease<R>>,
        selected: &GuardedExecution<'_>,
    ) -> Result<()> {
        let Some(context) = selected.maintenance.lock().take() else {
            return self.rollback_unsubmitted_step(step, "guarded deferred Step");
        };
        let mut context = *context
            .downcast::<PendingMaintenance<R>>()
            .map_err(|_| FerrumError::internal("guarded maintenance runtime mismatch"))?;
        if let MaintenanceSource::PendingWave(source) = context.source {
            let reconciled = source.reconcile_for_maintenance(step).map_err(|failure| {
                let error =
                    FerrumError::backend(format!("maintenance Step rollback: {}", failure.error()));
                self.abort_unsubmitted_step(failure.into_step(), error)
            })?;
            context.source = MaintenanceSource::ReconciledWave(reconciled);
        } else {
            self.rollback_unsubmitted_step(step, "guarded logical-deferred Step")?;
        }
        *selected.maintenance.lock() = Some(Box::new(context));
        Ok(())
    }

    pub(super) fn attach_guarded_maintenance<T>(
        &self,
        selected: &GuardedExecution<'_>,
        outcome: GuardedDispatchOutcome<T>,
    ) -> GuardedDispatchOutcome<T> {
        let GuardedDispatchOutcome::Deferred(deferral) = outcome else {
            return outcome;
        };
        let Some(context) = selected.maintenance.lock().take() else {
            return GuardedDispatchOutcome::Deferred(deferral);
        };
        let context = match context.downcast::<PendingMaintenance<R>>() {
            Ok(value) => *value,
            Err(_) => {
                return GuardedDispatchOutcome::Submitted(Err(FerrumError::internal(
                    "maintenance runtime mismatch",
                )))
            }
        };
        if matches!(context.source, MaintenanceSource::PendingWave(_)) {
            return GuardedDispatchOutcome::Submitted(Err(FerrumError::internal(
                "maintenance Step was not reconciled",
            )));
        }
        let request_ids = context
            .work
            .participants()
            .iter()
            .map(|row| row.selection().request_id.clone())
            .collect();
        match ExecutorExecutionMaintenanceTicket::from_backend(context.stage, request_ids, context)
        {
            Ok(ticket) => GuardedDispatchOutcome::MaintenanceDeferred { deferral, ticket },
            Err(error) => GuardedDispatchOutcome::Submitted(Err(error)),
        }
    }

    pub(super) fn maintain_guarded_execution_once(
        &self,
        ticket: ExecutorExecutionMaintenanceTicket,
        guard: &dyn NonblockingHostSubmissionGuard,
    ) -> Result<ExecutorExecutionMaintenanceOutcome> {
        use ExecutorExecutionMaintenanceOutcome as Outcome;
        let Some(context) = ticket.into_backend::<PendingMaintenance<R>>() else {
            return Ok(Outcome::Unsupported);
        };
        if !context
            .plan
            .upgrade()
            .is_some_and(|plan| Arc::ptr_eq(&plan, &self.plan_resources))
        {
            return Ok(Outcome::Unsupported);
        }
        let Some(owners) = context
            .owners
            .iter()
            .map(Weak::upgrade)
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(Outcome::Rejected(HostSubmissionRejection::Cancelled));
        };
        let check_owners = || {
            for (owner, row) in owners.iter().zip(context.work.participants()) {
                use ferrum_interfaces::execution_cost::ActualRowWork;
                if !owner.active.load(Ordering::Acquire) {
                    return Err(HostSubmissionRejection::Cancelled);
                }
                let matches = row.resource().matches_session_identity(&owner.session)
                    && match row.selection().work {
                        ActualRowWork::Prefill { offset, .. } => {
                            owner.prefill_tokens_processed.load(Ordering::Acquire)
                                == offset as usize
                        }
                        ActualRowWork::Decode { kv_tokens } => {
                            owner.tokens.lock().len() == kv_tokens as usize
                        }
                        ActualRowWork::Restore | ActualRowWork::Maintenance => false,
                    };
                if !matches {
                    return Err(HostSubmissionRejection::FrontierChanged);
                }
            }
            Ok(())
        };
        if let Err(reason) = check_owners() {
            return Ok(Outcome::Rejected(reason));
        }
        let mut guards = Vec::with_capacity(owners.len());
        for owner in &owners {
            match owner.operation.try_lock() {
                Ok(guard) => guards.push(guard),
                Err(_) => return Ok(Outcome::Rejected(HostSubmissionRejection::Busy)),
            }
        }
        if let Err(reason) = guard.check() {
            return Ok(Outcome::Rejected(reason));
        }
        if let Err(reason) = check_owners() {
            return Ok(Outcome::Rejected(reason));
        }
        let source = match &context.source {
            MaintenanceSource::Logical(value) => VNextExecutionMaintenanceSource::Logical(value),
            MaintenanceSource::Sequence(value) => {
                VNextExecutionMaintenanceSource::Backing(value.evidence())
            }
            MaintenanceSource::Step(value) => {
                VNextExecutionMaintenanceSource::Backing(value.evidence())
            }
            MaintenanceSource::ReconciledWave(value) => {
                VNextExecutionMaintenanceSource::Backing(value.evidence())
            }
            MaintenanceSource::PendingWave(_) => return Ok(Outcome::Unsupported),
        };
        // Keep real source evidence for exact wait reconstruction. No serialized
        // numerical deferral can be turned into a growth authority here.
        let outcome = match &context.source {
            MaintenanceSource::Logical(value) => {
                self.plan_resources.maintain_for_admission_deferred(value)
            }
            MaintenanceSource::Sequence(value) => value.maintain(),
            MaintenanceSource::Step(value) => value.maintain(),
            MaintenanceSource::ReconciledWave(value) => value.maintain(),
            MaintenanceSource::PendingWave(_) => unreachable!(),
        }
        .map_err(|error| FerrumError::backend(error.to_string()))?;
        match outcome {
            DynamicDeferredMaintenanceOutcome::RetryAdmission { current_epochs } => {
                Ok(Outcome::Recapture {
                    observed: ExecutorAdmissionEpochs::from_capacity(current_epochs),
                    progress: None,
                })
            }
            DynamicDeferredMaintenanceOutcome::Maintained(receipt) => {
                let status = self
                    .plan_resources
                    .dynamic_pool_status()
                    .map_err(|error| FerrumError::backend(error.to_string()))?;
                let observed = ExecutorAdmissionEpochs::from_capacity(status.epochs());
                let progress = ExecutorExecutionMaintenanceProgress::from_growth_receipts(
                    1,
                    observed,
                    std::slice::from_ref(&receipt),
                    status.pools(),
                )?;
                if let Some(sink) = self
                    .event_sink
                    .read()
                    .clone()
                    .filter(|sink| sink.records_execution_resource_maintenance())
                {
                    let stage = match context.stage {
                        ExecutorExecutionCapacityStage::SequenceExtension => {
                            ExecutionResourceMaintenanceStage::SequenceExtension
                        }
                        ExecutorExecutionCapacityStage::StepAdmission => {
                            ExecutionResourceMaintenanceStage::StepAdmission
                        }
                        ExecutorExecutionCapacityStage::SubmissionWave => {
                            ExecutionResourceMaintenanceStage::SubmissionWave
                        }
                    };
                    let event = BoundExecutionResourceMaintenance::bind(
                        stage,
                        owners.iter().map(|owner| owner.active_binding.as_ref()),
                        receipt,
                    )
                    .map_err(|error| FerrumError::backend(error.to_string()))?;
                    sink.record_execution_resource_maintenance(event)
                        .map_err(|error| FerrumError::backend(error.to_string()))?;
                }
                Ok(Outcome::Recapture {
                    observed,
                    progress: Some(progress),
                })
            }
            DynamicDeferredMaintenanceOutcome::WaitForRelease {
                current_epochs,
                wait_condition,
                pressure,
                maintenance_boundary,
            } => {
                let observed = ExecutorAdmissionEpochs::from_capacity(current_epochs);
                let deferred = match source {
                    VNextExecutionMaintenanceSource::Logical(value) => {
                        ExecutorExecutionCapacityDeferral::from_admission_maintenance(
                            value,
                            observed,
                            wait_condition,
                            pressure,
                            maintenance_boundary,
                            context.stage,
                        )
                    }
                    VNextExecutionMaintenanceSource::Backing(value) => {
                        ExecutorExecutionCapacityDeferral::from_backing_maintenance(
                            value,
                            observed,
                            wait_condition,
                            pressure,
                            maintenance_boundary,
                            context.stage,
                        )
                    }
                }?;
                Ok(Outcome::Wait(deferred))
            }
        }
    }
}
