use super::*;

fn take_batch<O: Send + 'static>(
    worker: &dyn BatchWorker<Output = O>,
    shared: &Shared<O>,
) -> Option<Vec<Queued<O>>> {
    let mut state = shared.lock();
    loop {
        prune_cancelled(&mut state.queue);
        if state.closing {
            reject_pending(&mut state, ImageServiceErrorKind::Shutdown);
            return None;
        }
        let Some(first) = state.queue.front() else {
            state = shared
                .changed
                .wait(state)
                .unwrap_or_else(|e| e.into_inner());
            continue;
        };
        let maximum_batch = match &first.request {
            QueuedRequest::Generate(_) => shared.config.max_num_seqs,
            QueuedRequest::Edit(..) => shared
                .config
                .max_num_seqs
                .min(worker.maximum_edit_batch_size().max(1)),
        };
        let mut count = 1;
        let mut incompatible = false;
        for job in state
            .queue
            .iter()
            .skip(1)
            .take(maximum_batch.saturating_sub(1))
        {
            let compatible = job.request.generation().width == first.request.generation().width
                && job.request.generation().height == first.request.generation().height
                && match (&first.request, &job.request) {
                    (QueuedRequest::Generate(a), QueuedRequest::Generate(b)) => {
                        worker.compatible(a, b)
                    }
                    (QueuedRequest::Edit(a, _), QueuedRequest::Edit(b, _)) => {
                        worker.compatible_edits(a, b)
                    }
                    _ => false,
                };
            if !compatible {
                incompatible = true;
                break;
            }
            count += 1;
        }
        let remaining = first
            .submitted
            .checked_add(Duration::from_millis(
                shared.config.request_batch_max_wait_ms,
            ))
            .and_then(|deadline| deadline.checked_duration_since(Instant::now()));
        if shared.config.batching == ImageBatchPolicy::Auto
            || count == maximum_batch
            || incompatible
            || remaining.is_none()
            || remaining == Some(Duration::ZERO)
        {
            let batch: Vec<_> = state.queue.drain(..count).collect();
            state.preparing = count;
            state.preparing_bytes = batch.iter().map(Queued::pending_bytes).sum();
            return Some(batch);
        }
        state = shared
            .changed
            .wait_timeout(state, remaining.unwrap_or(Duration::ZERO))
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }
}

/// Candidate ownership remains charged as waiting until device execution starts.
/// Requeue uses the original owners and arrival times, ahead of newer arrivals.
fn return_tail<O>(
    batch: &mut Vec<Queued<O>>,
    keep: usize,
    shared: &Shared<O>,
    device_capacity: bool,
) {
    let mut state = shared.lock();
    for mut job in batch.drain(keep..).rev() {
        // A waiting job must not hold output credit ahead of the FIFO head.
        if let Some(output) = &mut job.output {
            output.lease.take();
        }
        state.preparing -= 1;
        state.preparing_bytes -= job.pending_bytes();
        state.queue.push_front(job);
    }
    if device_capacity {
        state.capacity_reductions = state.capacity_reductions.saturating_add(1);
    }
}

/// Acquire only the compatible FIFO prefix that currently has output credit.
/// A resource-blocked tail cannot stop an already authorized prefix executing.
fn admit_output_prefix<O: Send + 'static>(
    worker: &dyn BatchWorker<Output = O>,
    batch: &mut Vec<Queued<O>>,
    shared: &Shared<O>,
) -> Result<bool, ImageWorkerFailure> {
    'retry: loop {
        remove_cancelled(batch, &mut shared.lock());
        if batch.is_empty() {
            return Ok(false);
        }
        for index in 0..batch.len() {
            if batch[index]
                .output
                .as_ref()
                .is_some_and(|output| output.lease.is_some())
            {
                continue;
            }
            let gate = batch[index]
                .output
                .as_ref()
                .map(|output| output.gate.clone());
            let _subscription = gate
                .as_ref()
                .map(|gate| gate.subscribe_capacity(shared.resources.waker()));
            let observed = shared.resources.epoch();
            // This is initial candidate construction, before any compiled plan.
            // Current committed bytes already include preceding jobs' grants;
            // demand adds only the growing engine peak and this next grant.
            let outcome =
                output_prefix_available(worker, batch, index, shared).and_then(|available| {
                    if !available {
                        return Ok(false);
                    }
                    match gate.as_ref().map(|gate| gate.try_acquire()).transpose() {
                        Ok(Some(Some(lease))) => {
                            batch[index].output.as_mut().unwrap().lease = Some(lease);
                            Ok(true)
                        }
                        Ok(None) => Ok(true),
                        Ok(Some(None)) => Ok(false),
                        Err(ImageOutputGateError::Capacity(message)) => {
                            Err(ImageBatchError::Overloaded(message))
                        }
                        Err(ImageOutputGateError::Failed(message)) => {
                            Err(ImageBatchError::Rejected(message))
                        }
                    }
                });
            match outcome {
                Ok(true) => (),
                Err(ImageBatchError::Fatal(error)) => {
                    return Err(ImageWorkerFailure::Generation(error))
                }
                Ok(false) | Err(_) if index > 0 => {
                    return_tail(batch, index, shared, false);
                    return Ok(true);
                }
                Ok(false) => {
                    // No prefix has acquired credit yet. Retain the compatible
                    // candidate so a capacity wake can admit all now-ready jobs,
                    // rather than permanently reducing this wait to one image.
                    if shared.lock().closing || batch[0].cancelled.load(Ordering::Acquire) {
                        continue 'retry;
                    }
                    shared.resources.wait(observed);
                    continue 'retry;
                }
                Err(error) => {
                    return_tail(batch, 1, shared, false);
                    let (kind, message) = rejection(error)?;
                    reject_candidate(std::mem::take(batch), shared, kind, &message);
                    return Ok(false);
                }
            }
        }
        return Ok(true);
    }
}

fn output_prefix_available<O: Send + 'static>(
    worker: &dyn BatchWorker<Output = O>,
    batch: &[Queued<O>],
    index: usize,
    shared: &Shared<O>,
) -> Result<bool, ImageBatchError> {
    let Some(host) = &shared.host else {
        return Ok(true);
    };
    let requests: Vec<_> = batch[..=index].iter().map(|job| &job.request).collect();
    let Some(engine_bytes) = worker.execution_output_bytes(&requests)? else {
        return Ok(true);
    };
    let held = batch
        .iter()
        .filter_map(|job| job.output.as_ref())
        .filter(|output| output.lease.is_some())
        .map(|output| output.gate.host_output_bytes(host))
        .try_fold(0usize, |total, bytes| total.checked_add(bytes))
        .ok_or_else(|| ImageBatchError::Overloaded("held output credit overflows".into()))?;
    let next = batch[index]
        .output
        .as_ref()
        .map_or(0, |output| output.gate.host_output_bytes(host));
    let demand = engine_bytes
        .checked_add(next)
        .ok_or_else(|| ImageBatchError::Overloaded("combined output demand overflows".into()))?;
    let claims = [(HostMemoryClass::Output, demand)];
    host.check_scope_after_output_release(&claims, held)
        .map_err(|error| preparation_error(error.into()))?;
    match host.check_scope_admission(&claims) {
        Ok(()) => Ok(true),
        Err(error) if error.is_temporary_capacity() => {
            host.check_scope_after_output_release(&claims, held)
                .map_err(|error| preparation_error(error.into()))?;
            Ok(false)
        }
        Err(error) => Err(preparation_error(error.into())),
    }
}

fn remove_cancelled<O>(batch: &mut Vec<Queued<O>>, state: &mut State<O>) -> bool {
    let before = batch.len();
    let mut index = 0;
    while index < batch.len() {
        if state.closing || batch[index].cancelled.load(Ordering::Acquire) {
            let job = batch.remove(index);
            state.preparing -= 1;
            state.preparing_bytes -= job.pending_bytes();
            let kind = if state.closing {
                ImageServiceErrorKind::Shutdown
            } else {
                ImageServiceErrorKind::Cancelled
            };
            let _ = job.response.send(Err(service_error(
                kind,
                "image request withdrawn before execution",
            )));
        } else {
            index += 1;
        }
    }
    before != batch.len()
}

fn rejection(
    error: ImageBatchError,
) -> Result<(ImageServiceErrorKind, String), ImageWorkerFailure> {
    match error {
        ImageBatchError::InvalidInput(message) => {
            Ok((ImageServiceErrorKind::InvalidRequest, message))
        }
        ImageBatchError::Rejected(message) => Ok((ImageServiceErrorKind::Generation, message)),
        ImageBatchError::Capacity { diagnostic, .. } => {
            Ok((ImageServiceErrorKind::Generation, diagnostic))
        }
        ImageBatchError::Overloaded(message) => Ok((ImageServiceErrorKind::QueueFull, message)),
        ImageBatchError::Fatal(error) => Err(ImageWorkerFailure::Generation(error)),
    }
}

fn reject_candidate<O>(
    batch: Vec<Queued<O>>,
    shared: &Shared<O>,
    kind: ImageServiceErrorKind,
    message: &str,
) {
    {
        let mut state = shared.lock();
        state.preparing = 0;
        state.preparing_bytes = 0;
        state.active = 0;
        if kind == ImageServiceErrorKind::QueueFull {
            state.rejected = state.rejected.saturating_add(batch.len() as u64);
        }
    }
    for job in batch {
        let kind = if job.cancelled.load(Ordering::Acquire) {
            ImageServiceErrorKind::Cancelled
        } else {
            kind
        };
        let _ = job.response.send(Err(service_error(kind, message)));
    }
}

/// Charge the metadata-proven retained input bound before decoding. This is a
/// waiting-queue quota; the shared Host account still authorizes every actual
/// backing (including transient decode/resize and execution allocations).
fn reserve_preprocessing<O: Send + 'static>(
    worker: &dyn BatchWorker<Output = O>,
    batch: &mut [Queued<O>],
    shared: &Shared<O>,
) -> Result<(), ImageBatchError> {
    let bounds = batch
        .iter()
        .map(|job| worker.preparation_bytes(&job.request))
        .collect::<Result<Vec<_>, _>>()?;
    let extra = bounds
        .iter()
        .zip(batch.iter())
        .try_fold(0_usize, |sum, (&bound, job)| {
            sum.checked_add(bound.saturating_sub(job.preprocessing_bytes))
        })
        .ok_or_else(|| {
            ImageBatchError::Overloaded("image prepared-input byte bound overflows".into())
        })?;
    let mut state = shared.lock();
    let queued = state.queue.iter().map(Queued::pending_bytes).sum::<usize>();
    if queued
        .saturating_add(state.preparing_bytes)
        .saturating_add(extra)
        > shared.config.maximum_pending_bytes
    {
        return Err(ImageBatchError::Overloaded(
            "image prepared inputs exceed waiting Host byte budget".into(),
        ));
    }
    for (job, bound) in batch.iter_mut().zip(bounds) {
        job.preprocessing_bytes = job.preprocessing_bytes.max(bound);
    }
    state.preparing_bytes += extra;
    Ok(())
}

pub(super) fn worker_loop<O: Send + 'static>(
    mut worker: Box<dyn BatchWorker<Output = O>>,
    shared: Arc<Shared<O>>,
) -> Retirement<O> {
    let _host_subscription = shared
        .host
        .as_ref()
        .map(|host| host.subscribe_capacity(shared.resources.waker()));
    let run = catch_unwind(AssertUnwindSafe(|| {
        'batches: while let Some(mut batch) = take_batch(worker.as_ref(), &shared) {
            let prepared = 'prepare: loop {
                remove_cancelled(&mut batch, &mut shared.lock());
                if batch.is_empty() {
                    continue 'batches;
                }
                match admit_output_prefix(worker.as_ref(), &mut batch, &shared) {
                    Ok(true) => (),
                    Ok(false) => continue 'batches,
                    Err(failure) => break Err(failure),
                }
                let preparation_started = Instant::now();
                let outcome = catch_unwind(AssertUnwindSafe(|| {
                    reserve_preprocessing(worker.as_ref(), &mut batch, &shared)?;
                    let mut requests = batch
                        .iter_mut()
                        .map(|job| &mut job.request)
                        .collect::<Vec<_>>();
                    worker.prepare(&mut requests)
                }));
                let elapsed = preparation_started.elapsed();
                for job in &mut batch {
                    job.preparation_duration = job.preparation_duration.saturating_add(elapsed);
                }
                match outcome {
                    Ok(Ok(mut prepared)) => {
                        loop {
                            let observed = shared.resources.epoch();
                            if remove_cancelled(&mut batch, &mut shared.lock()) {
                                drop(prepared);
                                continue 'prepare;
                            }
                            let output_credit_bytes = batch
                                .iter()
                                .filter_map(|job| job.output.as_ref())
                                .map(|output| {
                                    shared
                                        .host
                                        .as_ref()
                                        .map_or(0, |host| output.gate.host_output_bytes(host))
                                })
                                .try_fold(0usize, |total, bytes| total.checked_add(bytes));
                            let resources = match output_credit_bytes {
                                Some(output_credit_bytes) => ImagePreparationResources {
                                    output_credit_bytes,
                                },
                                None => {
                                    drop(prepared);
                                    reject_candidate(
                                        batch,
                                        &shared,
                                        ImageServiceErrorKind::QueueFull,
                                        "combined output grants exceed addressable Host memory",
                                    );
                                    continue 'batches;
                                }
                            };
                            let admission = match catch_unwind(AssertUnwindSafe(|| {
                                worker.try_admit_prepared(&mut prepared, resources)
                            })) {
                                Ok(admission) => admission,
                                Err(panic) => break 'prepare Err(ImageWorkerFailure::Panic(panic)),
                            };
                            match admission {
                                Ok(true) => break,
                                Ok(false) => {
                                    // This token has submitted no device work.
                                    // Keep its plans and edit owners while an
                                    // independently held output is consumed.
                                    if shared.lock().closing
                                        || batch
                                            .iter()
                                            .any(|job| job.cancelled.load(Ordering::Acquire))
                                    {
                                        continue;
                                    }
                                    shared.resources.wait(observed);
                                }
                                Err(error) => match rejection(error) {
                                    Ok((kind, message)) => {
                                        drop(prepared);
                                        reject_candidate(batch, &shared, kind, &message);
                                        continue 'batches;
                                    }
                                    Err(failure) => break 'prepare Err(failure),
                                },
                            }
                        }
                        let mut state = shared.lock();
                        // A cancellation invalidates the compiled batch geometry.
                        // Dropping this unused preparation is safe; no work exists.
                        if remove_cancelled(&mut batch, &mut state) {
                            drop(state);
                            drop(prepared);
                            continue;
                        }
                        state.preparing = 0;
                        state.preparing_bytes = 0;
                        state.active = batch.len();
                        state.executed_batches = state.executed_batches.saturating_add(1);
                        state.executed_requests =
                            state.executed_requests.saturating_add(batch.len() as u64);
                        state.last_batch_size = batch.len();
                        break Ok((prepared, state.executed_batches));
                    }
                    Ok(Err(ImageBatchError::Capacity { proof, .. }))
                        if shared.config.batching == ImageBatchPolicy::Auto && batch.len() > 1 =>
                    {
                        let keep = batch.len() - 1;
                        tracing::debug!(
                            candidate_batch = batch.len(),
                            next_batch = keep,
                            required_bytes = proof.report.required_bytes,
                            capacity_bytes = proof.report.capacity_bytes,
                            "image candidate reduced before device execution"
                        );
                        return_tail(&mut batch, keep, &shared, true);
                    }
                    Ok(Err(error)) => match rejection(error) {
                        Ok((kind, message)) => {
                            reject_candidate(batch, &shared, kind, &message);
                            continue 'batches;
                        }
                        Err(failure) => break Err(failure),
                    },
                    Err(panic) => break Err(ImageWorkerFailure::Panic(panic)),
                }
            };
            // Scope all request and nested generator/stage events to one actual
            // dispatch. Failed preparations never receive an execution identity.
            let dispatch_span = prepared.as_ref().ok().map(|(_, dispatch_id)| {
                tracing::debug_span!(
                    target: "ferrum_engine::image_generation::timing",
                    "image_dispatch",
                    dispatch_id = *dispatch_id,
                )
            });
            let _dispatch = dispatch_span.as_ref().map(tracing::Span::enter);
            let mut responses = Vec::with_capacity(batch.len());
            let mut requests = Vec::with_capacity(batch.len());
            let now = Instant::now();
            let batch_size = batch.len();
            for (batch_index, job) in batch.into_iter().enumerate() {
                if prepared.is_ok() {
                    tracing::debug!(target: "ferrum_engine::image_generation::timing",
                    batch_index,
                    batch_size,
                    seed = job.request.generation().seed,
                    width = job.request.generation().width,
                    height = job.request.generation().height,
                    request_kind = if matches!(job.request, QueuedRequest::Edit(..)) { "edit" } else { "generate" },
                    queue_wait_us = u64::try_from(now.duration_since(job.submitted).saturating_sub(job.preparation_duration).as_micros()).unwrap_or(u64::MAX),
                    queue_and_prepare_us = u64::try_from(now.duration_since(job.submitted).as_micros()).unwrap_or(u64::MAX),
                    preparation_us = u64::try_from(job.preparation_duration.as_micros()).unwrap_or(u64::MAX),
                    "image request admitted to execution");
                }
                requests.push(job.request);
                responses.push((
                    job.cancelled,
                    job.response,
                    job._prompt_reservation,
                    job.output,
                ));
            }
            let outcome = prepared.and_then(|(prepared, _)| {
                match catch_unwind(AssertUnwindSafe(|| worker.execute(prepared, requests))) {
                    Ok(Ok(outputs)) if outputs.len() == responses.len() => Ok(outputs),
                    Ok(Ok(_)) => Err(ImageWorkerFailure::Generation(Box::new(service_error(
                        ImageServiceErrorKind::Generation,
                        "generator returned a different number of outputs than batch requests",
                    )))),
                    // Capacity at execution time is never a retry signal.
                    Ok(Err(error @ ImageBatchError::Capacity { .. })) => {
                        Err(ImageWorkerFailure::Generation(Box::new(error)))
                    }
                    Ok(Err(error)) => match rejection(error) {
                        Ok((kind, message)) => {
                            {
                                let mut state = shared.lock();
                                state.active = 0;
                                if kind == ImageServiceErrorKind::QueueFull {
                                    state.rejected =
                                        state.rejected.saturating_add(responses.len() as u64);
                                }
                            }
                            for (cancelled, response, prompt_reservation, output) in
                                responses.drain(..)
                            {
                                let kind = if cancelled.load(Ordering::Acquire) {
                                    ImageServiceErrorKind::Cancelled
                                } else {
                                    kind
                                };
                                // A terminal rejection makes capacity reusable before
                                // the waiter can submit its next request.
                                drop(output);
                                drop(prompt_reservation);
                                let _ = response.send(Err(service_error(kind, &message)));
                            }
                            Ok(Vec::new())
                        }
                        Err(failure) => Err(failure),
                    },
                    Err(panic) => Err(ImageWorkerFailure::Panic(panic)),
                }
            });
            match outcome {
                Ok(outputs) => {
                    for ((cancelled, response, _, gate), output) in
                        responses.into_iter().zip(outputs)
                    {
                        let result = if cancelled.load(Ordering::Acquire) {
                            Err(service_error(
                                ImageServiceErrorKind::Cancelled,
                                "image response cancelled; active execution completed",
                            ))
                        } else {
                            if let Some(QueuedOutput {
                                lease: Some(lease),
                                response,
                                ..
                            }) = gate
                            {
                                let _ = response.send(lease);
                            }
                            Ok(output)
                        };
                        let _ = response.send(result);
                    }
                    shared.lock().active = 0;
                }
                Err(failure) => {
                    let (kind, message) = match &failure {
                        ImageWorkerFailure::Generation(error) => (ImageServiceErrorKind::Generation,
                            catch_unwind(AssertUnwindSafe(|| error.to_string())).unwrap_or_else(|_| "image generation failed; error formatter panicked, original owner retained".to_owned())),
                        ImageWorkerFailure::Panic(_) => (ImageServiceErrorKind::WorkerPanic, "image worker panicked; original owners retained".to_owned()),
                    };
                    let mut state = shared.lock();
                    state.accepting = false;
                    state.closing = true;
                    state.active = 0;
                    state.preparing = 0;
                    state.preparing_bytes = 0;
                    reject_pending(&mut state, ImageServiceErrorKind::Stopped);
                    for (_, response, _, _) in responses {
                        let _ = response.send(Err(service_error(kind, &message)));
                    }
                    return Some(failure);
                }
            }
        }
        None
    }));
    let mut failure = match run {
        Ok(failure) => failure,
        Err(panic) => Some(ImageWorkerFailure::Panic(panic)),
    };
    if failure.is_none() {
        failure = match catch_unwind(AssertUnwindSafe(|| worker.shutdown())) {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(ImageWorkerFailure::Generation(error)),
            Err(panic) => Some(ImageWorkerFailure::Panic(panic)),
        };
    }
    {
        let mut state = shared.lock();
        state.accepting = false;
        state.closing = true;
        state.active = 0;
        state.preparing = 0;
        state.preparing_bytes = 0;
        reject_pending(&mut state, ImageServiceErrorKind::Stopped);
    }
    Retirement {
        worker: Some(worker),
        failure,
    }
}
