//! Zero-submission admission using the executor's original plan and pools.
//! A prepared wave is explicitly rolled back before returning. Its physical
//! backing stays with the same pool/device grants; execution still admits a
//! fresh invocation and must not treat the receipt as execution authority.
use super::*;

#[derive(Debug)]
pub enum TensorDynamicPreparationOutcome<R: DeviceRuntime> {
    Prepared(IdleDynamicBackingOutcome),
    /// Every logical authority has retired and all candidate dynamic backing
    /// has been released by its original plan before this result is returned.
    Capacity {
        refusal: TensorExecutionError<R>,
        rollback: DynamicPoolRebalanceReceipt,
    },
}

impl<R: DeviceRuntime> PreparedTensorExecutor<R> {
    /// Prepare the exact supplied input work on the existing actor without
    /// encoding uploads, launching operators or reading outputs. Expected
    /// capacity refusal returns Ok only after real rollback, preserving the
    /// actor; other errors retain ordinary failure/retirement semantics.
    pub fn try_prepare_dynamic_backing_on_actor(
        &mut self,
        actor: &mut TensorSerialActor<R>,
        run_id: RunId,
        request_id: RequestIdentity,
        inputs: Vec<ManagedHostTensor>,
    ) -> Result<TensorDynamicPreparationOutcome<R>, TensorExecutionError<R>> {
        actor.execute(|lane| {
            let work = self.binding.bind_inputs(inputs)?;
            let mut scope = TensorExecutionRetirement {
                step: None,
                session: None,
            };
            let admitted = (|| {
                let sequence = self.admit_sequence(&work, run_id, request_id)?;
                scope.session = Some(sequence.open_session()?);
                let batch = ExecutionBatchParticipants::new(vec![scope.session()?.clone()])?;
                scope.step = Some(self.admit_step(&batch, lane, &work)?);
                let wave = self.prepare_wave(scope.step.as_ref().expect("admitted step"))?;
                // Drop is the existing explicit rollback contract for a wave
                // which has never reached OperationDispatch.
                drop(wave);
                Ok::<_, TensorExecutionError<R>>(())
            })();
            if let Err(source) = scope.try_abort() {
                return Err(TensorExecutionError::Retirement {
                    source,
                    retirement: scope,
                });
            }
            match admitted {
                Ok(()) => {
                    // This owner-aware bound includes real retained chunks.
                    // It is used only to observe/retain the already acquired
                    // backing; it did not replace admission above.
                    let bound = self
                        .resources
                        .minimum_runnable_dynamic_bytes_with_residency()?;
                    let retained = self.resources.try_retain_idle_dynamic_backing(bound)?;
                    match retained {
                        IdleDynamicBackingOutcome::Busy(reason) => Err(invalid(format!(
                            "prepared dynamic backing remained busy: {reason:?}"
                        ))
                        .into()),
                        ready => Ok(TensorDynamicPreparationOutcome::Prepared(ready)),
                    }
                }
                Err(refusal) if refusal.is_capacity_refusal() => {
                    match self.resources.try_release_idle_dynamic_backing()? {
                        IdleDynamicReleaseOutcome::Released(rollback) => {
                            Ok(TensorDynamicPreparationOutcome::Capacity { refusal, rollback })
                        }
                        IdleDynamicReleaseOutcome::Busy(busy) => Err(invalid(format!(
                            "candidate dynamic rollback remained busy: {busy:?}"
                        ))
                        .into()),
                    }
                }
                Err(original) => Err(original),
            }
        })
    }
}

impl<R: DeviceRuntime> TensorExecutionError<R> {
    /// Classify ordinary admission pressure. This is not retirement evidence:
    /// a caller must still close the exact admitted scopes before retrying.
    pub fn is_capacity_refusal(&self) -> bool {
        matches!(
            self,
            TensorExecutionError::Admission {
                kind: TensorAdmissionFailure::Deferred
                    | TensorAdmissionFailure::Rejected
                    | TensorAdmissionFailure::MaintenanceLimit,
                ..
            } | TensorExecutionError::Runtime(
                VNextError::DeviceCapacityUnavailable(_)
                    | VNextError::DeviceAllocationUnavailable(_)
                    | VNextError::DynamicPoolResidentUnavailable(_)
            )
        )
    }
}
