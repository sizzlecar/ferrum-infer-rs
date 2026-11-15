//! Opt-in finite initialization cohorts. Queue entries keep their original
//! response/Output owners; exactly one opaque initialization owns shared work.
use super::*;
use std::collections::BTreeSet;

pub struct InitializationCohortMember {
    pub request_id: Uuid,
    pub preparation: BatchPreparation,
    /// Already held by this job, not additional credit for the common phase.
    pub resources: PreparationResources,
}
#[derive(Clone, Copy)]
pub struct InitializationCohortRequest<'a, R: ServiceRequest> {
    pub request_id: Uuid,
    pub request: &'a R,
    /// A fresh whole-cohort snapshot, sticky once true, including closing.
    pub cancelled: bool,
}
pub enum InitializationCohortOutcome {
    Ready(DenoisingRequest),
    /// Hand off independent downstream work only after the shared auxiliary
    /// owner has retired and finished=true. No ready lane is fabricated.
    Pending(RequestInitialization),
    Cancelled,
    Rejected {
        diagnostic: String,
    },
}
pub struct InitializationCohortCompletion {
    pub request_id: Uuid,
    pub outcome: InitializationCohortOutcome,
}
pub struct InitializationCohortProgress {
    pub made_progress: bool,
    /// All original members have completed exactly once and the opaque owner
    /// is quiescent. The worker may now drop it and publish individual results.
    pub finished: bool,
    pub completed: Vec<InitializationCohortCompletion>,
}

/// Finite, sealed initialization is explicit generator behavior. The default
/// denoising-generator method returns None and preserves its singleton path.
/// Each cohort owns one original Host phase and native/recovery owner. No join
/// or member replacement is permitted after preparation.
pub(crate) trait InitializationCohortGenerator: Send {
    type Request: ServiceRequest;
    /// Zero selects the original singleton protocol for this preparation.
    /// The worker additionally bounds this by its existing service lane limit.
    fn maximum_initialization_cohort_members(
        &mut self,
        prepared: &mut BatchPreparation,
    ) -> Result<usize, BatchError>;
    /// Metadata only; false leaves the next exact FIFO preparation untouched.
    fn initialization_cohort_compatible(
        &mut self,
        first: &mut BatchPreparation,
        next: &mut BatchPreparation,
    ) -> Result<bool, BatchError>;
    /// No device submission or hidden admission. Validate all members before
    /// consuming their preparations; preserve exact ingress/source identities.
    fn prepare_initialization_cohort(
        &mut self,
        members: Vec<InitializationCohortMember>,
    ) -> Result<BatchPreparation, BatchError>;
    /// Authorize the aggregate real shared phase once. Downstream work may
    /// instead transfer as Pending after retirement. false preserves preparation
    /// and any old runnable batch. If old
    /// native state is suspended, an ordinary refusal must restore it first.
    fn try_admit_initialization_cohort_at_boundary(
        &mut self,
        prepared: &mut BatchPreparation,
        batch: Option<&mut DenoisingBatch>,
        survivors: &mut [&mut DenoisingRequest],
    ) -> Result<bool, BatchError>;
    /// A nonfatal error proves no possible native work or recovery owner and
    /// leaves the preparation safe to drop. Otherwise retain the original
    /// owner in Fatal. The worker retains this preparation in fatal custody.
    fn begin_initialization_cohort(
        &mut self,
        prepared: &mut BatchPreparation,
        members: &[InitializationCohortRequest<'_, Self::Request>],
    ) -> Result<RequestInitialization, BatchError>;
    /// At most one actual execution/maintenance/close wave per call. A false
    /// made_progress registers wake on the original external wait. Completions
    /// may arrive over multiple calls but are withheld until finished. Pending
    /// handoffs are allowed only with finished=true after shared retirement.
    /// Returning Ready instead requires those real states to fit together.
    fn advance_initialization_cohort(
        &mut self,
        initialization: &mut RequestInitialization,
        members: &[InitializationCohortRequest<'_, Self::Request>],
        wake: &std::task::Waker,
    ) -> Result<InitializationCohortProgress, BatchError>;
    /// false is legal only after the shared auxiliary native owner no longer
    /// blocks real restoration. Per-member EOS alone is insufficient.
    fn initialization_cohort_blocks_denoising(
        &mut self,
        initialization: &RequestInitialization,
    ) -> bool;
}

struct Member<R: ServiceRequest, O> {
    pending: Pending<R, O>,
    cancelled: bool,
    completion: Option<InitializationCohortOutcome>,
}
pub(crate) struct Cohort<R: ServiceRequest, O> {
    // Field order retires native state before releasing the preparation/jobs.
    owner: Option<RequestInitialization>,
    prepared: Option<BatchPreparation>,
    members: Vec<Member<R, O>>,
    // Returned Ready owners also survive a malformed protocol response.
    unvalidated: Vec<InitializationCohortCompletion>,
}

fn protocol(message: &'static str) -> BatchError {
    BatchError::Fatal(Box::new(service_error(
        ServiceErrorKind::Generation,
        message,
    )))
}
fn generator<R: ServiceRequest, O: Send + 'static>(
    worker: &mut dyn BatchWorker<Request = R, Output = O>,
) -> Result<&mut dyn InitializationCohortGenerator<Request = R>, BatchError> {
    worker
        .denoising_generator()
        .and_then(|generator| generator.initialization_cohort_generator())
        .ok_or_else(|| {
            protocol("initialization cohort capability disappeared; original owner retained")
        })
}

pub(crate) fn prepare_member<R: ServiceRequest, O: Send + 'static>(
    worker: &mut dyn BatchWorker<Request = R, Output = O>,
    candidate: &mut Pending<R, O>,
) -> Result<(), BatchError> {
    if candidate.prepared.is_some() {
        return Ok(());
    }
    let started = Instant::now();
    let span = candidate
        .job
        .trace
        .phase_started(observations::LifecyclePhase::MetadataPreparation);
    let prepared = worker
        .denoising_generator()
        .unwrap()
        .prepare_request(&candidate.job.request);
    candidate.job.preparation_duration = candidate
        .job
        .preparation_duration
        .saturating_add(started.elapsed());
    candidate.job.trace.phase_finished(
        span,
        if prepared.is_ok() {
            observations::PhaseOutcome::Completed
        } else {
            observations::PhaseOutcome::Failed
        },
    );
    let mut prepared = prepared?;
    worker
        .denoising_generator()
        .unwrap()
        .bind_request_correlation(&mut prepared, candidate.job.trace.correlation);
    worker
        .denoising_generator()
        .unwrap()
        .bind_request_cancellation(&mut prepared, candidate.job.cancelled.clone());
    candidate.prepared = Some(prepared);
    Ok(())
}

/// Pull only an already-queued compatible prefix. The original scheduler owns
/// request_batch_max_wait_ms; this consumer adds no extra timer or wait window.
fn pull_next<R: ServiceRequest, O: Send + 'static>(
    worker: &dyn BatchWorker<Request = R, Output = O>,
    shared: &Shared<R, O>,
    first: &Queued<R, O>,
) -> Option<Pending<R, O>> {
    let mut state = shared.lock();
    if state.closing {
        return None;
    }
    prune_cancelled(&mut state.queue);
    let next = state.queue.front()?;
    if !super::super::scheduler::compatible_queued(worker, first, next) {
        return None;
    }
    let job = state.queue.pop_front().unwrap();
    job.trace.dequeued();
    state.preparing += 1;
    state.preparing_bytes += job.pending_bytes();
    Some(Pending::new(job))
}

/// Prepare/freeze one finite prefix. No original singleton code runs unless
/// the generator opted in and accepted this first preparation.
pub(crate) fn collect<R: ServiceRequest, O: Send + 'static>(
    worker: &mut dyn BatchWorker<Request = R, Output = O>,
    shared: &Shared<R, O>,
    running: &[Running<R, O>],
    pending: &mut VecDeque<Pending<R, O>>,
    cohort: &mut Option<Cohort<R, O>>,
) -> Result<bool, BatchError> {
    if cohort.is_some()
        || !worker
            .denoising_generator()
            .is_some_and(|g| g.initialization_cohort_generator().is_some())
        || !pull(worker, running, pending, shared)
    {
        return Ok(false);
    }
    let first = pending.front_mut().unwrap();
    if first.state.is_some() || first.initialization.is_some() {
        return Ok(false);
    }
    let eligibility = (|| {
        prepare_member(worker, first)?;
        generator(worker)?.maximum_initialization_cohort_members(first.prepared.as_mut().unwrap())
    })();
    let maximum = match eligibility {
        Ok(maximum) => maximum.min(shared.config.max_num_seqs.saturating_sub(running.len())),
        Err(error @ BatchError::Fatal(_)) => return Err(error),
        Err(error) => {
            let (kind, message) = rejection(error)?;
            reject_pending_step(worker, pending.pop_front().unwrap(), shared, kind, &message);
            return Ok(true);
        }
    };
    if maximum == 0 {
        return Ok(false);
    }
    if running.is_empty() && maximum >= 2 && shared.config.batching == BatchPolicy::Auto {
        wait_for_prefix(worker, shared, pending, maximum);
    }
    let mut members = Vec::<Member<R, O>>::new();
    let mut withdrew_member = false;
    // Withdrawals also consume this finite turn's inspection budget.
    for _ in 0..maximum {
        let candidate = pending.pop_front().or_else(|| {
            members
                .first()
                .and_then(|first| pull_next(worker, shared, &first.pending.job))
        });
        let Some(mut candidate) = candidate else {
            break;
        };
        if candidate.state.is_some()
            || candidate.initialization.is_some()
            || members.first().is_some_and(|first| {
                !super::super::scheduler::compatible_queued(
                    worker,
                    &first.pending.job,
                    &candidate.job,
                )
            })
        {
            pending.push_front(candidate);
            break;
        }
        let closing = shared.lock().closing;
        if closing || candidate.job.cancelled.load(Ordering::Acquire) {
            withdrew_member = true;
            reject_pending_step(
                worker,
                candidate,
                shared,
                if closing {
                    ServiceErrorKind::Shutdown
                } else {
                    ServiceErrorKind::Cancelled
                },
                "image request withdrawn before cohort initialization",
            );
            continue;
        }
        let inspect = (|| {
            prepare_member(worker, &mut candidate)?;
            if let Some(first) = members.first_mut() {
                if !generator(worker)?.initialization_cohort_compatible(
                    first.pending.prepared.as_mut().unwrap(),
                    candidate.prepared.as_mut().unwrap(),
                )? {
                    return Ok(false);
                }
            }
            try_output(worker, &mut candidate.job, shared)
        })();
        match inspect {
            Ok(true) => members.push(Member {
                pending: candidate,
                cancelled: false,
                completion: None,
            }),
            Ok(false) => {
                pending.push_front(candidate);
                break;
            }
            Err(error @ BatchError::Fatal(_)) => {
                pending.push_front(candidate);
                for member in members.into_iter().rev() {
                    pending.push_front(member.pending);
                }
                return Err(error);
            }
            Err(error) => {
                let classification = rejection(error);
                match classification {
                    Ok((kind, message)) => {
                        withdrew_member = true;
                        reject_pending_step(worker, candidate, shared, kind, &message)
                    }
                    Err(error) => {
                        pending.push_front(candidate);
                        for member in members.into_iter().rev() {
                            pending.push_front(member.pending);
                        }
                        return Err(error);
                    }
                }
            }
        }
    }
    if members.is_empty() {
        return Ok(withdrew_member);
    }
    // Install worker custody before invoking any generator-owned group API.
    *cohort = Some(Cohort {
        owner: None,
        prepared: None,
        members,
        unvalidated: Vec::new(),
    });
    let group = cohort.as_mut().unwrap();
    let mut ids = BTreeSet::new();
    let inputs = group
        .members
        .iter_mut()
        .map(|member| {
            let request_id = member.pending.job.trace.correlation.request_id;
            if !ids.insert(request_id) {
                return Err(protocol(
                    "initialization cohort contains duplicate original job identities",
                ));
            }
            let resources = PreparationResources {
                output_credit_bytes: output_credit(shared, &member.pending.job)?,
            };
            Ok(InitializationCohortMember {
                request_id,
                resources,
                preparation: member.pending.prepared.take().unwrap(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    match generator(worker)?.prepare_initialization_cohort(inputs) {
        Ok(prepared) => group.prepared = Some(prepared),
        Err(error @ BatchError::Fatal(_)) => return Err(error),
        Err(error) => {
            let (kind, message) = rejection(error)?;
            reject_before_begin(worker, shared, cohort, kind, &message)?;
        }
    }
    Ok(true)
}

/// Auto's ordinary image scheduler intentionally does not wait. A generator
/// opting into finite initialization may use the already declared batch wait,
/// but only while idle, before any auxiliary owner/grant is acquired. The
/// original submission deadline includes metadata work and is never restarted.
fn wait_for_prefix<R: ServiceRequest, O: Send + 'static>(
    worker: &dyn BatchWorker<Request = R, Output = O>,
    shared: &Shared<R, O>,
    pending: &VecDeque<Pending<R, O>>,
    maximum: usize,
) {
    let first = &pending.front().expect("eligible first member").job;
    let Some(deadline) = first.submitted.checked_add(Duration::from_millis(
        shared.config.request_batch_max_wait_ms,
    )) else {
        return;
    };
    let mut state = shared.lock();
    loop {
        if state.closing || first.cancelled.load(Ordering::Acquire) {
            return;
        }
        prune_cancelled(&mut state.queue);
        let mut count = 0usize;
        let mut incompatible = false;
        for job in pending
            .iter()
            .map(|p| &p.job)
            .chain(state.queue.iter())
            .take(maximum)
        {
            if !super::super::scheduler::compatible_queued(worker, first, job) {
                incompatible = true;
                break;
            }
            count += 1;
        }
        if count == maximum || incompatible {
            return;
        }
        let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
            return;
        };
        if remaining.is_zero() {
            return;
        }
        state = shared
            .changed
            .wait_timeout(state, remaining)
            .unwrap_or_else(|error| error.into_inner())
            .0;
    }
}

fn rejection(error: BatchError) -> Result<(ServiceErrorKind, String), BatchError> {
    super::super::scheduler::rejection(error)
        .map_err(|failure| BatchError::Fatal(Box::new(failure)))
}
fn reject_before_begin<R: ServiceRequest, O: Send + 'static>(
    worker: &mut dyn BatchWorker<Request = R, Output = O>,
    shared: &Shared<R, O>,
    cohort: &mut Option<Cohort<R, O>>,
    kind: ServiceErrorKind,
    message: &str,
) -> Result<(), BatchError> {
    if cohort.as_ref().is_some_and(|group| group.owner.is_some()) {
        return Err(protocol("cannot reject a live shared initialization owner"));
    }
    if let Some(group) = cohort.take() {
        drop(group.prepared);
        for member in group.members {
            reject_pending_step(worker, member.pending, shared, kind, message);
        }
    }
    Ok(())
}

pub(crate) fn blocks_denoising<R: ServiceRequest, O: Send + 'static>(
    worker: &mut dyn BatchWorker<Request = R, Output = O>,
    cohort: Option<&Cohort<R, O>>,
) -> Result<bool, BatchError> {
    let Some(owner) = cohort.and_then(|group| group.owner.as_ref()) else {
        return Ok(false);
    };
    Ok(generator(worker)?.initialization_cohort_blocks_denoising(owner))
}

/// The group remains in worker custody even if the generator returns malformed
/// completions or errors. Results are validated as a whole before job mutation.
pub(crate) fn advance<R: ServiceRequest, O: Send + 'static>(
    worker: &mut dyn BatchWorker<Request = R, Output = O>,
    shared: &Shared<R, O>,
    batch: &mut Option<DenoisingBatch>,
    running: &mut [Running<R, O>],
    pending: &mut VecDeque<Pending<R, O>>,
    cohort: &mut Option<Cohort<R, O>>,
) -> Result<(initialization::Tick, bool), BatchError> {
    let Some(group) = cohort.as_mut() else {
        return Ok((initialization::Tick::default(), false));
    };
    if group.owner.is_none() {
        let closing = shared.lock().closing;
        if group
            .members
            .iter()
            .all(|m| closing || m.pending.job.cancelled.load(Ordering::Acquire))
        {
            reject_before_begin(
                worker,
                shared,
                cohort,
                if closing {
                    ServiceErrorKind::Shutdown
                } else {
                    ServiceErrorKind::Cancelled
                },
                "image initialization withdrawn before native admission",
            )?;
            return Ok((
                initialization::Tick {
                    advanced: true,
                    made_progress: true,
                },
                false,
            ));
        }
        if !running.is_empty() {
            pause(worker, batch, running)?;
        }
        let admission = generator(worker)?.try_admit_initialization_cohort_at_boundary(
            group
                .prepared
                .as_mut()
                .ok_or_else(|| protocol("cohort preparation was not retained"))?,
            batch.as_mut(),
            &mut running.iter_mut().map(|r| &mut r.state).collect::<Vec<_>>(),
        );
        match admission {
            Ok(true) => (),
            Ok(false) => {
                return Ok((
                    initialization::Tick {
                        advanced: true,
                        made_progress: false,
                    },
                    false,
                ))
            }
            Err(
                BatchError::Capacity { .. }
                | BatchError::InitializationAwaitDenoisingProgress { .. },
            ) if !running.is_empty() => {
                return Ok((
                    initialization::Tick {
                        advanced: true,
                        made_progress: false,
                    },
                    false,
                ))
            }
            Err(error @ BatchError::Fatal(_)) => return Err(error),
            Err(error) => {
                let (kind, message) = rejection(error)?;
                reject_before_begin(worker, shared, cohort, kind, &message)?;
                return Ok((
                    initialization::Tick {
                        advanced: true,
                        made_progress: true,
                    },
                    false,
                ));
            }
        }
        for member in &mut group.members {
            member.pending.job.trace.initialization_started();
            member.pending.initialization_trace = member
                .pending
                .job
                .trace
                .phase_started(observations::LifecyclePhase::Initialization);
        }
        // Disjoint field borrows keep views of all ORIGINAL jobs, without
        // copying prompt/PNG buffers or borrowing a job after it is delivered.
        let closing = shared.lock().closing;
        let Cohort {
            owner,
            prepared,
            members,
            ..
        } = group;
        let views = snapshot(members, closing);
        let begun =
            generator(worker)?.begin_initialization_cohort(prepared.as_mut().unwrap(), &views);
        match begun {
            Ok(initialization) => *owner = Some(initialization),
            Err(error @ BatchError::Fatal(_)) => return Err(error),
            Err(error) => {
                let (kind, message) = rejection(error)?;
                reject_before_begin(worker, shared, cohort, kind, &message)?;
            }
        }
        return Ok((
            initialization::Tick {
                advanced: true,
                made_progress: true,
            },
            false,
        ));
    }
    if !running.is_empty() {
        pause(worker, batch, running)?;
    }
    let closing = shared.lock().closing;
    let spans = group
        .members
        .iter()
        .map(|member| {
            member
                .pending
                .job
                .trace
                .phase_started(observations::LifecyclePhase::InitializationAdvance)
        })
        .collect::<Vec<_>>();
    let Cohort { owner, members, .. } = group;
    let views = snapshot(members, closing);
    let result = generator(worker)?.advance_initialization_cohort(
        owner.as_mut().unwrap(),
        &views,
        &shared.resources.waker(),
    );
    for (member, span) in group.members.iter().zip(spans) {
        member.pending.job.trace.phase_finished(
            span,
            if result.is_ok() {
                observations::PhaseOutcome::Completed
            } else {
                observations::PhaseOutcome::Failed
            },
        );
    }
    let progress = result?;
    let completed_progress = !progress.completed.is_empty();
    group.unvalidated = progress.completed;
    validate_completions(group, progress.finished)?;
    if progress.finished
        && generator(worker)?.initialization_cohort_blocks_denoising(group.owner.as_ref().unwrap())
    {
        return Err(protocol(
            "initialization finished while its shared auxiliary owner is still live",
        ));
    }
    for completion in group.unvalidated.drain(..) {
        group
            .members
            .iter_mut()
            .find(|member| member.pending.job.trace.correlation.request_id == completion.request_id)
            .unwrap()
            .completion = Some(completion.outcome);
    }
    if !progress.finished {
        return Ok((
            initialization::Tick {
                advanced: true,
                made_progress: progress.made_progress || completed_progress,
            },
            false,
        ));
    }
    let mut group = cohort.take().unwrap();
    // finished is the explicit quiescent ownership boundary, not member EOS.
    drop(group.owner.take());
    drop(group.prepared.take());
    let closing = shared.lock().closing;
    let mut ready = VecDeque::new();
    for mut member in group.members {
        let outcome = member.completion.take().unwrap();
        let cancelled =
            member.cancelled || closing || member.pending.job.cancelled.load(Ordering::Acquire);
        let trace_outcome = match &outcome {
            InitializationCohortOutcome::Pending(_) => observations::PhaseOutcome::Completed,
            InitializationCohortOutcome::Ready(_) if !cancelled => {
                observations::PhaseOutcome::Completed
            }
            InitializationCohortOutcome::Ready(_) | InitializationCohortOutcome::Cancelled => {
                observations::PhaseOutcome::Cancelled
            }
            InitializationCohortOutcome::Rejected { .. } => observations::PhaseOutcome::Failed,
        };
        if !matches!(&outcome, InitializationCohortOutcome::Pending(_)) {
            member
                .pending
                .job
                .trace
                .phase_finished(member.pending.initialization_trace.take(), trace_outcome);
        }
        match outcome {
            InitializationCohortOutcome::Pending(initialization) => {
                member.pending.initialization = Some(initialization);
                // Even if cancelled concurrently, the independent continuation
                // must itself observe cancellation and prove safe retirement.
                ready.push_back(member.pending);
            }
            InitializationCohortOutcome::Ready(request) => {
                member.pending.state = Some(request);
                if cancelled {
                    reject_pending_step(
                        worker,
                        member.pending,
                        shared,
                        if closing {
                            ServiceErrorKind::Shutdown
                        } else {
                            ServiceErrorKind::Cancelled
                        },
                        "image request withdrawn before joining a step",
                    );
                } else {
                    ready.push_back(member.pending);
                }
            }
            InitializationCohortOutcome::Cancelled => reject_pending_step(
                worker,
                member.pending,
                shared,
                if closing {
                    ServiceErrorKind::Shutdown
                } else {
                    ServiceErrorKind::Cancelled
                },
                "image initialization cancelled at its terminal boundary",
            ),
            InitializationCohortOutcome::Rejected { diagnostic } => reject_pending_step(
                worker,
                member.pending,
                shared,
                ServiceErrorKind::Generation,
                &diagnostic,
            ),
        }
    }
    ready.append(pending);
    *pending = ready;
    Ok((
        initialization::Tick {
            advanced: false,
            made_progress: true,
        },
        true,
    ))
}

fn snapshot<R: ServiceRequest, O>(
    members: &mut [Member<R, O>],
    closing: bool,
) -> Vec<InitializationCohortRequest<'_, R>> {
    members
        .iter_mut()
        .map(|member| {
            member.cancelled |= closing || member.pending.job.cancelled.load(Ordering::Acquire);
            InitializationCohortRequest {
                request_id: member.pending.job.trace.correlation.request_id,
                request: &member.pending.job.request,
                cancelled: member.cancelled,
            }
        })
        .collect()
}
fn validate_completions<R: ServiceRequest, O>(
    group: &Cohort<R, O>,
    finished: bool,
) -> Result<(), BatchError> {
    let mut seen = BTreeSet::new();
    for completed in &group.unvalidated {
        if !finished && matches!(&completed.outcome, InitializationCohortOutcome::Pending(_)) {
            return Err(protocol(
                "independent initialization handoff requires finished shared owner",
            ));
        }
        let member = group
            .members
            .iter()
            .find(|member| member.pending.job.trace.correlation.request_id == completed.request_id)
            .ok_or_else(|| {
                protocol("initialization returned an unknown member; original group retained")
            })?;
        if member.completion.is_some() || !seen.insert(completed.request_id) {
            return Err(protocol(
                "initialization completed a member twice; original group retained",
            ));
        }
    }
    if finished
        && group.members.iter().any(|member| {
            member.completion.is_none()
                && !seen.contains(&member.pending.job.trace.correlation.request_id)
        })
    {
        return Err(protocol(
            "initialization finished without every original member; original group retained",
        ));
    }
    Ok(())
}

struct RetainedJob<R: ServiceRequest> {
    _request: R,
    _output: Option<QueuedOutput>,
    _prompt: Option<HostMemoryReservation>,
    _preparation: Option<BatchPreparation>,
    _request_state: Option<DenoisingRequest>,
    _completion: Option<InitializationCohortOutcome>,
}
pub(crate) struct CohortFailure<R: ServiceRequest> {
    source: WorkerFailure,
    pub(crate) _owner: Option<RequestInitialization>,
    _preparation: Option<BatchPreparation>,
    _unvalidated: Vec<InitializationCohortCompletion>,
    _jobs: Vec<RetainedJob<R>>,
}
impl<R: ServiceRequest> fmt::Debug for CohortFailure<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageInitializationCohortFailure")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}
impl<R: ServiceRequest> fmt::Display for CohortFailure<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.source, f)
    }
}
impl<R: ServiceRequest> Error for CohortFailure<R> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.source)
    }
}

/// Existing fatal() owns the old batch/single initializers. This wrapper adds
/// the ONE shared owner and every original member backing without refunding
/// Output credit while unknown native state still has retirement obligations.
pub(crate) fn retain_failure<R: ServiceRequest, O: Send + 'static>(
    failure: WorkerFailure,
    cohort: Option<Cohort<R, O>>,
) -> WorkerFailure {
    let Some(group) = cohort else { return failure };
    // Formatting only borrows the original error. Even a user formatter panic
    // must leave its recovery authority and every original cohort owner intact.
    let cause = catch_unwind(AssertUnwindSafe(|| failure.to_string()))
        .unwrap_or_else(|_| "error formatter panicked".to_owned());
    let message =
        format!("cohort initialization failed: {cause}; original shared recovery owner retained");
    let mut jobs = Vec::new();
    for member in group.members {
        let Pending {
            job,
            prepared,
            state,
            ..
        } = member.pending;
        let Queued {
            trace,
            request,
            response,
            output,
            _prompt_reservation,
            ..
        } = job;
        let _ = response.send(Err(trace.error(ServiceErrorKind::Generation, &message)));
        jobs.push(RetainedJob {
            _request: request,
            _output: output,
            _prompt: _prompt_reservation,
            _preparation: prepared,
            _request_state: state,
            _completion: member.completion,
        });
    }
    WorkerFailure::Generation(Box::new(CohortFailure {
        source: failure,
        _owner: group.owner,
        _preparation: group.prepared,
        _unvalidated: group.unvalidated,
        _jobs: jobs,
    }))
}
