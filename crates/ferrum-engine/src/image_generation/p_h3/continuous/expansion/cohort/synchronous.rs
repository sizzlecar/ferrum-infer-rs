//! `run` batches use the same sealed expansion and post-retirement handoff as
//! `serve`. Singletons keep their existing path; homogeneous visual groups
//! traverse the same sequential-producer/shared-causal state machine.
use super::*;
use crate::image_generation::service::ImagePreparationResources;

pub(in crate::image_generation::p_h3::continuous) enum PreparedRunInitialization {
    Individual(ImageBatchPreparation),
    Expanded(ImageRequestInitialization),
}

struct SynchronousCohortFailure {
    source: ImageBatchError,
    _owner: Option<ImageRequestInitialization>,
    _preparation: Option<ImageBatchPreparation>,
    _completed: Vec<ImageInitializationCohortCompletion>,
}
impl std::fmt::Debug for SynchronousCohortFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}
impl std::fmt::Display for SynchronousCohortFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "H3 synchronous expansion cohort: {}", self.source)
    }
}
impl std::error::Error for SynchronousCohortFailure {}

impl<R: DeviceRuntime> PH3ImagePipeline<R> {
    pub(in crate::image_generation::p_h3::continuous) fn prepare_run_initializations(
        &mut self,
        mut preparations: Vec<ImageBatchPreparation>,
        requests: &[ImageRequestRef<'_>],
    ) -> InitResult<Vec<PreparedRunInitialization>> {
        let mut compatible = preparations.len() > 1;
        if compatible {
            let (first, others) = preparations.split_first_mut().expect("nonempty");
            compatible = self.maximum_initialization_cohort_members(first)? >= requests.len();
            for next in others {
                compatible &= self.initialization_cohort_compatible(first, next)?;
            }
        }
        if !compatible {
            return Ok(preparations
                .into_iter()
                .map(PreparedRunInitialization::Individual)
                .collect());
        }
        let identities = requests.iter().map(|_| Uuid::new_v4()).collect::<Vec<_>>();
        let members = preparations
            .into_iter()
            .zip(&identities)
            .map(
                |(preparation, &request_id)| ImageInitializationCohortMember {
                    request_id,
                    preparation,
                    resources: ImagePreparationResources {
                        output_credit_bytes: 0,
                    },
                },
            )
            .collect();
        let mut prepared = self.prepare_initialization_cohort(members)?;
        if !self.try_admit_initialization_cohort_at_boundary(&mut prepared, None, &mut [])? {
            return Err(ImageBatchError::Overloaded(
                "H3 expansion cohort cannot obtain actual Host capacity".into(),
            ));
        }
        let inputs = requests
            .iter()
            .zip(identities)
            .map(|(&request, request_id)| ImageInitializationCohortRequest {
                request_id,
                request,
                cancelled: false,
            })
            .collect::<Vec<_>>();
        let mut owner = match self.begin_initialization_cohort(&mut prepared, &inputs) {
            Ok(owner) => owner,
            Err(source) => {
                return Err(ImageBatchError::Fatal(Box::new(SynchronousCohortFailure {
                    source,
                    _owner: None,
                    _preparation: Some(prepared),
                    _completed: vec![],
                })))
            }
        };
        struct ThreadWake(std::thread::Thread);
        impl std::task::Wake for ThreadWake {
            fn wake(self: Arc<Self>) {
                self.0.unpark();
            }
            fn wake_by_ref(self: &Arc<Self>) {
                self.0.unpark();
            }
        }
        let wake = Waker::from(Arc::new(ThreadWake(std::thread::current())));
        let _subscription = self.execution.host.subscribe_capacity(wake.clone());
        loop {
            let progress = match self.advance_initialization_cohort(&mut owner, &inputs, &wake) {
                Ok(progress) => progress,
                Err(source) => {
                    return Err(ImageBatchError::Fatal(Box::new(SynchronousCohortFailure {
                        source,
                        _owner: Some(owner),
                        _preparation: None,
                        _completed: vec![],
                    })))
                }
            };
            if !progress.finished {
                if !progress.made_progress {
                    std::thread::park();
                }
                continue;
            }
            if self.initialization_cohort_blocks_denoising(&owner)
                || progress.completed.len() != inputs.len()
                || !progress
                    .completed
                    .iter()
                    .zip(&inputs)
                    .all(|(completed, input)| {
                        completed.request_id == input.request_id
                            && !matches!(
                                completed.outcome,
                                ImageInitializationCohortOutcome::Ready(_)
                            )
                    })
            {
                self.execution.stopped = true;
                return Err(ImageBatchError::Fatal(Box::new(SynchronousCohortFailure {
                    source: ImageBatchError::Rejected(
                        "H3 expansion retirement membership differs".into(),
                    ),
                    _owner: Some(owner),
                    _preparation: None,
                    _completed: progress.completed,
                })));
            }
            return progress
                .completed
                .into_iter()
                .zip(&inputs)
                .map(|(completed, input)| {
                    if completed.request_id != input.request_id {
                        return Err(ImageBatchError::Rejected(
                            "H3 expansion result order differs".into(),
                        ));
                    }
                    match completed.outcome {
                        ImageInitializationCohortOutcome::Pending(initialization) => {
                            Ok(PreparedRunInitialization::Expanded(initialization))
                        }
                        ImageInitializationCohortOutcome::Rejected { diagnostic } => {
                            Err(ImageBatchError::Rejected(diagnostic))
                        }
                        ImageInitializationCohortOutcome::Cancelled => {
                            Err(ImageBatchError::Rejected("H3 expansion cancelled".into()))
                        }
                        ImageInitializationCohortOutcome::Ready(_) => {
                            Err(ImageBatchError::Rejected(
                                "H3 expansion skipped downstream initialization".into(),
                            ))
                        }
                    }
                })
                .collect();
        }
    }
}
