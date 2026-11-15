//! The original worker owns every pending initialization until its terminal
//! boundary. Advancing one wave never publishes a denoising membership.
use super::*;

#[must_use = "pending initialization retains its original execution and retirement owners"]
pub struct RequestInitialization(Box<dyn Any + Send>);
impl fmt::Debug for RequestInitialization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RequestInitialization")
            .finish_non_exhaustive()
    }
}
impl RequestInitialization {
    pub fn new<T: Any + Send>(value: T) -> Self {
        Self(Box::new(value))
    }
    pub fn get<T: Any + Send>(&self) -> Result<&T, BatchError> {
        self.0.downcast_ref().ok_or_else(|| {
            BatchError::Rejected("initialization belongs to another generator".into())
        })
    }
    pub fn get_mut<T: Any + Send>(&mut self) -> Result<&mut T, BatchError> {
        self.0.downcast_mut().ok_or_else(|| {
            BatchError::Fatal(Box::new(service_error(
                ServiceErrorKind::Generation,
                "initialization belongs to another generator; original owner retained",
            )))
        })
    }
    /// Foreign downcasts return the original possible native owner intact.
    pub fn into_inner<T: Any + Send>(self) -> Result<T, Self> {
        match self.0.downcast::<T>() {
            Ok(value) => Ok(*value),
            Err(owner) => Err(Self(owner)),
        }
    }
}

pub enum InitializationOutcome {
    Ready(DenoisingRequest),
    Pending(RequestInitialization),
}
pub enum InitializationProgress {
    Ready(DenoisingRequest),
    Pending {
        made_progress: bool,
    },
    /// The original initialization is quiescent. The worker may now release
    /// its job's preparing/output credit and drop this opaque owner.
    Cancelled,
    /// Request-local rejection after the original native owner is retired,
    /// with no recovery obligation. This permits releasing this owner and its
    /// admission credit without poisoning other requests. An execution error
    /// with a live/unknown owner must instead remain Fatal.
    Rejected {
        diagnostic: String,
    },
}

#[derive(Default)]
pub(crate) struct Tick {
    pub advanced: bool,
    pub made_progress: bool,
}

/// Cancellation is not a Drop operation for a possible native initialization.
/// All ordinary queued/ready cancellations may still retire immediately.
pub(crate) fn advance<R: ServiceRequest, O: Send + 'static>(
    worker: &mut dyn BatchWorker<Request = R, Output = O>,
    shared: &Shared<R, O>,
    batch: &mut Option<DenoisingBatch>,
    running: &mut [Running<R, O>],
    pending: &mut VecDeque<Pending<R, O>>,
) -> Result<Tick, BatchError> {
    let closing = shared.lock().closing;
    let cancelled = |p: &Pending<R, O>| closing || p.job.cancelled.load(Ordering::Acquire);
    let mut index = 0;
    while index < pending.len() {
        if cancelled(&pending[index]) && pending[index].initialization.is_none() {
            let p = pending.remove(index).unwrap();
            let (kind, message) = cancellation(closing);
            reject_pending_step(worker, p, shared, kind, message);
        } else {
            index += 1;
        }
    }
    // Prefer an actual cancellation over ordinary initialization work, while
    // still retiring at most one original wave per worker iteration.
    let index = pending
        .iter()
        .position(|p| p.initialization.is_some() && cancelled(p))
        .or_else(|| {
            // A ready FIFO member must try joining the current cohort before
            // later initialization replaces its resident working set. If that
            // fresh admission cannot fit, the authorized cohort advances and
            // may retry after its next real step. Cancellation above still
            // retires original initialization owners across this boundary.
            // An existing native initialization that blocks denoiser restore
            // must also reach retirement; suppressing it would strand both
            // the suspended cohort and its ready FIFO waiter.
            if !running.is_empty() && pending.front().is_some_and(|p| p.state.is_some()) {
                let generator = worker.denoising_generator().unwrap();
                pending.iter().position(|p| {
                    p.initialization.as_ref().is_some_and(|initialization| {
                        generator.initialization_blocks_denoising(initialization)
                    })
                })
            } else {
                pending.iter().position(|p| p.initialization.is_some())
            }
        });
    let Some(index) = index else {
        return Ok(Tick::default());
    };
    if !running.is_empty() {
        pause(worker, batch, running)?;
    }
    let mut candidate = pending.remove(index).unwrap();
    let cancel_requested = cancelled(&candidate);
    let span = candidate
        .job
        .trace
        .phase_started(observations::LifecyclePhase::InitializationAdvance);
    let result = worker
        .denoising_generator()
        .unwrap()
        .advance_initialization_at_boundary(
            candidate.initialization.as_mut().unwrap(),
            &candidate.job.request,
            cancel_requested,
            &shared.resources.waker(),
            batch.as_mut(),
            &mut running.iter_mut().map(|r| &mut r.state).collect::<Vec<_>>(),
        );
    candidate.job.trace.phase_finished(
        span,
        if result.is_ok() {
            observations::PhaseOutcome::Completed
        } else {
            observations::PhaseOutcome::Failed
        },
    );
    match result {
        Ok(InitializationProgress::Pending { made_progress }) => {
            pending.insert(index, candidate);
            Ok(Tick {
                advanced: true,
                made_progress,
            })
        }
        Ok(InitializationProgress::Ready(request)) => {
            candidate.job.trace.phase_finished(
                candidate.initialization_trace.take(),
                observations::PhaseOutcome::Completed,
            );
            drop(candidate.initialization.take());
            candidate.state = Some(request);
            let withdrawn =
                shared.lock().closing || candidate.job.cancelled.load(Ordering::Acquire);
            if withdrawn {
                let (kind, message) = cancellation(shared.lock().closing);
                reject_pending_step(worker, candidate, shared, kind, message);
            } else {
                pending.insert(index, candidate);
            }
            Ok(Tick {
                // This request is now quiescent and may join at this exact
                // fence. Deferring its real membership until another forward
                // can finish a short survivor before it ever shares work.
                advanced: withdrawn,
                made_progress: true,
            })
        }
        Ok(InitializationProgress::Cancelled) => {
            candidate.job.trace.phase_finished(
                candidate.initialization_trace.take(),
                observations::PhaseOutcome::Cancelled,
            );
            drop(candidate.initialization.take());
            let (kind, message) = cancellation(shared.lock().closing);
            reject_pending_step(worker, candidate, shared, kind, message);
            Ok(Tick {
                advanced: true,
                made_progress: true,
            })
        }
        Ok(InitializationProgress::Rejected { diagnostic }) => {
            candidate.job.trace.phase_finished(
                candidate.initialization_trace.take(),
                observations::PhaseOutcome::Failed,
            );
            drop(candidate.initialization.take());
            reject_pending_step(
                worker,
                candidate,
                shared,
                ServiceErrorKind::Generation,
                &diagnostic,
            );
            Ok(Tick {
                advanced: true,
                made_progress: true,
            })
        }
        Err(error) => {
            candidate.job.trace.phase_finished(
                candidate.initialization_trace.take(),
                observations::PhaseOutcome::Failed,
            );
            // Even a misclassified nonfatal error cannot drop an opaque
            // owner which may hold submitted/unknown initialization work.
            pending.insert(index, candidate);
            Err(BatchError::Fatal(source(error)))
        }
    }
}

fn cancellation(closing: bool) -> (ServiceErrorKind, &'static str) {
    if closing {
        (ServiceErrorKind::Shutdown, "image service is shutting down")
    } else {
        (
            ServiceErrorKind::Cancelled,
            "image request cancelled before joining a step",
        )
    }
}
