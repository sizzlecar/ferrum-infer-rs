//! The original worker owns every pending initialization until its terminal
//! boundary. Advancing one wave never publishes a denoising membership.
use super::*;

#[must_use = "pending initialization retains its original execution and retirement owners"]
pub struct ImageRequestInitialization(Box<dyn Any + Send>);
impl fmt::Debug for ImageRequestInitialization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageRequestInitialization")
            .finish_non_exhaustive()
    }
}
impl ImageRequestInitialization {
    pub fn new<T: Any + Send>(value: T) -> Self {
        Self(Box::new(value))
    }
    pub fn get<T: Any + Send>(&self) -> Result<&T, ImageBatchError> {
        self.0.downcast_ref().ok_or_else(|| {
            ImageBatchError::Rejected("initialization belongs to another generator".into())
        })
    }
    pub fn get_mut<T: Any + Send>(&mut self) -> Result<&mut T, ImageBatchError> {
        self.0.downcast_mut().ok_or_else(|| {
            ImageBatchError::Fatal(Box::new(service_error(
                ImageServiceErrorKind::Generation,
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

pub enum ImageInitializationOutcome {
    Ready(ImageDenoisingRequest),
    Pending(ImageRequestInitialization),
}
pub enum ImageInitializationProgress {
    Ready(ImageDenoisingRequest),
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
pub(super) struct Tick {
    pub advanced: bool,
    pub made_progress: bool,
}

/// Cancellation is not a Drop operation for a possible native initialization.
/// All ordinary queued/ready cancellations may still retire immediately.
pub(super) fn advance<O: Send + 'static>(
    worker: &mut dyn BatchWorker<Output = O>,
    shared: &Shared<O>,
    batch: &mut Option<ImageDenoisingBatch>,
    running: &mut [Running<O>],
    pending: &mut VecDeque<Pending<O>>,
) -> Result<Tick, ImageBatchError> {
    let closing = shared.lock().closing;
    let cancelled = |p: &Pending<O>| closing || p.job.cancelled.load(Ordering::Acquire);
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
    let span = candidate.job.trace.phase_started(
        crate::image_generation::continuous_trace::ImageTracePhase::InitializationAdvance,
    );
    let result = worker
        .denoising_generator()
        .unwrap()
        .advance_image_initialization_at_boundary(
            candidate.initialization.as_mut().unwrap(),
            candidate.job.request.image_request_ref(),
            cancel_requested,
            &shared.resources.waker(),
            batch.as_mut(),
            &mut running.iter_mut().map(|r| &mut r.state).collect::<Vec<_>>(),
        );
    candidate.job.trace.phase_finished(
        span,
        if result.is_ok() {
            crate::image_generation::continuous_trace::ImageTracePhaseOutcome::Completed
        } else {
            crate::image_generation::continuous_trace::ImageTracePhaseOutcome::Failed
        },
    );
    match result {
        Ok(ImageInitializationProgress::Pending { made_progress }) => {
            pending.insert(index, candidate);
            Ok(Tick {
                advanced: true,
                made_progress,
            })
        }
        Ok(ImageInitializationProgress::Ready(request)) => {
            candidate.job.trace.phase_finished(
                candidate.initialization_trace.take(),
                crate::image_generation::continuous_trace::ImageTracePhaseOutcome::Completed,
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
        Ok(ImageInitializationProgress::Cancelled) => {
            candidate.job.trace.phase_finished(
                candidate.initialization_trace.take(),
                crate::image_generation::continuous_trace::ImageTracePhaseOutcome::Cancelled,
            );
            drop(candidate.initialization.take());
            let (kind, message) = cancellation(shared.lock().closing);
            reject_pending_step(worker, candidate, shared, kind, message);
            Ok(Tick {
                advanced: true,
                made_progress: true,
            })
        }
        Ok(ImageInitializationProgress::Rejected { diagnostic }) => {
            candidate.job.trace.phase_finished(
                candidate.initialization_trace.take(),
                crate::image_generation::continuous_trace::ImageTracePhaseOutcome::Failed,
            );
            drop(candidate.initialization.take());
            reject_pending_step(
                worker,
                candidate,
                shared,
                ImageServiceErrorKind::Generation,
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
                crate::image_generation::continuous_trace::ImageTracePhaseOutcome::Failed,
            );
            // Even a misclassified nonfatal error cannot drop an opaque
            // owner which may hold submitted/unknown initialization work.
            pending.insert(index, candidate);
            Err(ImageBatchError::Fatal(source(error)))
        }
    }
}

fn cancellation(closing: bool) -> (ImageServiceErrorKind, &'static str) {
    if closing {
        (
            ImageServiceErrorKind::Shutdown,
            "image service is shutting down",
        )
    } else {
        (
            ImageServiceErrorKind::Cancelled,
            "image request cancelled before joining a step",
        )
    }
}
