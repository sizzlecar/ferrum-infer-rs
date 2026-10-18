use super::*;

pub(super) struct ResidentFrameOwner<R: DeviceRuntime> {
    pub(super) program_retention: Mutex<Option<DeviceBufferRetention>>,
    pub(super) participants: Vec<AdmittedStepParticipant<R>>,
    pub(super) work: Arc<BatchWorkShape>,
    pub(super) tables: std::sync::OnceLock<Vec<Arc<SequencePhysicalBindingTable<R::Buffer>>>>,
    pub(super) step_id: BatchStepId,
    pub(super) completed_boundary: Mutex<StepCompletedBoundarySlot>,
    pub(super) submitted: AtomicBool,
    pub(super) finalized: bool,
    pub(super) reservation: Arc<ResidentDecodeReservation<R>>,
    // Reopen only after all frame, backing and completion owners have released.
    pub(super) gate: ResidentUseGate,
}

#[must_use = "retire only after the successful completion/readback owner is released"]
pub struct ResidentFrameRetirement<R: DeviceRuntime> {
    pub(super) frame: Arc<ResidentFrameOwner<R>>,
}

#[must_use = "a failed retirement retains the exact fresh frame"]
pub struct ResidentFrameFinalizationFailure<R: DeviceRuntime> {
    retirement: ResidentFrameRetirement<R>,
    pub(super) error: VNextError,
}

impl<R: DeviceRuntime> fmt::Debug for ResidentFrameFinalizationFailure<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResidentFrameFinalizationFailure")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}
impl<R: DeviceRuntime> ResidentFrameFinalizationFailure<R> {
    pub fn error(&self) -> &VNextError {
        &self.error
    }
    pub fn into_retirement(self) -> ResidentFrameRetirement<R> {
        self.retirement
    }
}
impl<R: DeviceRuntime> ResidentFrameRetirement<R> {
    pub fn batch_step_id(&self) -> BatchStepId {
        self.frame.step_id
    }
    pub fn try_retire_normal(
        self,
    ) -> Result<StepRetirementReceipt, ResidentFrameFinalizationFailure<R>> {
        self.finalize(StepFrameFinalization::Commit)
    }
    pub fn try_rollback_unsubmitted(
        self,
    ) -> Result<StepRetirementReceipt, ResidentFrameFinalizationFailure<R>> {
        self.finalize(StepFrameFinalization::RollbackUnsubmitted)
    }
    pub fn try_abort(self) -> Result<StepRetirementReceipt, ResidentFrameFinalizationFailure<R>> {
        self.finalize(StepFrameFinalization::Abort)
    }
    fn finalize(
        self,
        disposition: StepFrameFinalization,
    ) -> Result<StepRetirementReceipt, ResidentFrameFinalizationFailure<R>> {
        let mut frame = match Arc::try_unwrap(self.frame) {
            Ok(frame) => frame,
            Err(frame) => {
                return Err(ResidentFrameFinalizationFailure {
                    retirement: Self { frame },
                    error: invalid_resource(
                        "resident frame still has a submission, packet or quarantine owner",
                    ),
                })
            }
        };
        let result = (|| {
            if frame.finalized
                || disposition == StepFrameFinalization::RollbackUnsubmitted
                    && frame.submitted.load(Ordering::Acquire)
            {
                return Err(invalid_resource(
                    "resident frame cannot roll back submitted work or retire twice",
                ));
            }
            let completed = frame
                .completed_boundary
                .get_mut()
                .map_err(|_| invalid_resource("resident completed-boundary mutex poisoned"))?;
            let mut holds = frame
                .participants
                .iter_mut()
                .map(|p| &mut p.frame)
                .collect::<Vec<_>>();
            let dispositions =
                finalize_session_frames_with_boundary(&mut holds, disposition, completed.proof())?;
            completed.consume();
            frame.finalized = true;
            frame.gate.mark_finalized();
            Ok(StepRetirementReceipt {
                batch_step_id: frame.step_id,
                participants: frame
                    .participants
                    .iter()
                    .zip(dispositions)
                    .map(|(p, disposition)| StepParticipantRetirement {
                        assignment: StepParticipantFrameAssignment::new(
                            p.session.sequence_authority(),
                            p.session.request_authority(),
                            p.frame.frame_id,
                        ),
                        disposition,
                    })
                    .collect(),
            })
        })();
        result.map_err(|error| ResidentFrameFinalizationFailure {
            retirement: Self {
                frame: Arc::new(frame),
            },
            error,
        })
    }
}
