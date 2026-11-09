//! Publish a survivor plan before withdrawing its old rows. Completed lanes
//! then decode and deliver one at a time, without retaining a batch of images.
use super::*;

pub(super) fn retire_exits<O: Send + 'static>(
    worker: &mut dyn BatchWorker<Output = O>,
    shared: &Shared<O>,
    batch: &mut Option<ImageDenoisingBatch>,
    running: &mut Vec<Running<O>>,
    finishing: &mut VecDeque<Running<O>>,
) -> Result<(), ImageBatchError> {
    if !running
        .iter()
        .any(|r| r.complete || r.job.cancelled.load(Ordering::Acquire))
    {
        return Ok(());
    }
    pause(worker, batch, running)?;
    // Cancellation racing actual admission is observed at the next boundary;
    // this exact proposal determines the one newly prepared physical cohort.
    let exits = running
        .iter()
        .map(|r| r.complete || r.job.cancelled.load(Ordering::Acquire))
        .collect::<Vec<_>>();
    if exits.iter().any(|exit| !*exit) {
        let next = worker.denoising_generator().unwrap().try_prepare_batch(
            &mut running
                .iter_mut()
                .zip(&exits)
                .filter(|(_, exit)| !**exit)
                .map(|(r, _)| &mut r.state)
                .collect::<Vec<_>>(),
            batch.as_mut(),
        );
        match next {
            Ok(Some(next)) => {
                *batch = Some(next);
                // Actual survivor admission has published no request exit yet.
                // Fence this exact new owner before the first decoder stage.
                worker.denoising_generator().unwrap().pause_batch(
                    batch.as_mut().unwrap(),
                    &mut running
                        .iter_mut()
                        .zip(&exits)
                        .filter(|(_, exit)| !**exit)
                        .map(|(r, _)| &mut r.state)
                        .collect::<Vec<_>>(),
                )?;
            }
            Err(ImageBatchError::Fatal(source)) => return Err(ImageBatchError::Fatal(source)),
            other => {
                // The old cohort contains completed/cancelled rows and cannot
                // advance again. Fail only its un-runnable survivors finitely;
                // real completed lanes keep the old fenced work owner and can
                // still produce their independently valid images.
                let message = match other {
                    Err(source) => source.to_string(),
                    Ok(None) => {
                        "surviving cohort cannot obtain compact native/work capacity".into()
                    }
                    Ok(Some(_)) => unreachable!(),
                };
                for r in running.drain(..) {
                    if r.job.cancelled.load(Ordering::Acquire) {
                        worker
                            .denoising_generator()
                            .unwrap()
                            .record_request_exit(&r.state, ImageDenoisingExitReason::Cancelled);
                        cancel(shared, r);
                    } else if r.complete {
                        worker
                            .denoising_generator()
                            .unwrap()
                            .record_request_exit(&r.state, ImageDenoisingExitReason::Completed);
                        finishing.push_back(r);
                    } else {
                        worker
                            .denoising_generator()
                            .unwrap()
                            .record_request_exit(&r.state, ImageDenoisingExitReason::Failed);
                        drop(r.state);
                        shared.lock().active -= 1;
                        reject_job(r.job, ImageServiceErrorKind::Generation, &message);
                    }
                }
                return Ok(());
            }
        }
    }
    let mut index = 0;
    for exit in exits {
        if !exit {
            index += 1;
            continue;
        }
        let r = running.remove(index);
        if r.job.cancelled.load(Ordering::Acquire) {
            worker
                .denoising_generator()
                .unwrap()
                .record_request_exit(&r.state, ImageDenoisingExitReason::Cancelled);
            cancel(shared, r);
        } else {
            debug_assert!(r.complete);
            worker
                .denoising_generator()
                .unwrap()
                .record_request_exit(&r.state, ImageDenoisingExitReason::Completed);
            finishing.push_back(r);
        }
    }
    Ok(())
}

/// `false` is temporary, observable completion backpressure. The original
/// lane and fenced work guard remain owned, while authorized survivors can
/// consume another real forward. No decoded canvas is buffered in this queue.
pub(super) fn drain_finished<O: Send + 'static>(
    worker: &mut dyn BatchWorker<Output = O>,
    shared: &Shared<O>,
    batch: &mut Option<ImageDenoisingBatch>,
    running: &mut [Running<O>],
    finishing: &mut VecDeque<Running<O>>,
    initializing: bool,
) -> Result<bool, ImageBatchError> {
    if finishing.is_empty() {
        return Ok(true);
    }
    // A blocked front must not retain a later cancelled lane's credit. Scan
    // every completed cancellation before trying the remaining FIFO order.
    let mut index = 0;
    while index < finishing.len() {
        if finishing[index].job.cancelled.load(Ordering::Acquire) {
            cancel(shared, finishing.remove(index).unwrap());
        } else {
            index += 1;
        }
    }
    if finishing.is_empty() {
        return Ok(true);
    }
    if !running.is_empty() {
        pause(worker, batch, running)?;
    }
    while let Some(request) = finishing.front_mut() {
        if request.job.cancelled.load(Ordering::Acquire) {
            cancel(shared, finishing.pop_front().unwrap());
            continue;
        }
        let output = worker.denoising_generator().unwrap().try_finish_request(
            &mut request.state,
            batch
                .as_mut()
                .expect("completed lane retains its fenced work owner"),
        );
        match output {
            Ok(Some(output)) => {
                let r = finishing.pop_front().unwrap();
                drop(r.state);
                // Move this canvas to the caller before allocating the next.
                // A cancelled receiver drops it and immediately refunds its
                // actual charge through the existing Host subscription.
                deliver(shared, r.job, output);
            }
            Ok(None) => return Ok(false),
            Err(ImageBatchError::AwaitDenoisingProgress {
                stage: crate::image_generation::ImageStage::Decoder,
                diagnostic,
            }) => {
                if initializing
                    || running
                        .iter()
                        .any(|r| !r.complete && !r.job.cancelled.load(Ordering::Acquire))
                {
                    // The worker prohibits new joins while finishing is
                    // pending. These actual live cursors therefore reach a
                    // finite fence and can release their own native pins.
                    return Ok(false);
                }
                // No uncompleted owned cursor can change native capacity and
                // this protocol has no external native-capacity notification.
                // Preserve worker usability and finish this request finitely.
                let r = finishing.pop_front().unwrap();
                drop(r.state);
                shared.lock().active -= 1;
                reject_job(r.job, ImageServiceErrorKind::QueueFull, &diagnostic);
            }
            Err(ImageBatchError::Fatal(source)) => return Err(ImageBatchError::Fatal(source)),
            Err(source) => {
                let (kind, message) = super::super::scheduler::rejection(source)
                    .map_err(|failure| ImageBatchError::Fatal(Box::new(failure)))?;
                let r = finishing.pop_front().unwrap();
                drop(r.state);
                shared.lock().active -= 1;
                reject_job(r.job, kind, &message);
            }
        }
    }
    Ok(true)
}

fn cancel<O>(shared: &Shared<O>, r: Running<O>) {
    drop(r.state);
    shared.lock().active -= 1;
    reject_job(
        r.job,
        ImageServiceErrorKind::Cancelled,
        "image request withdrawn at a completed denoising step",
    );
}

pub(super) fn failure_owners<O>(
    mut running: Vec<Running<O>>,
    finishing: VecDeque<Running<O>>,
) -> Vec<Running<O>> {
    running.extend(finishing);
    running
}
