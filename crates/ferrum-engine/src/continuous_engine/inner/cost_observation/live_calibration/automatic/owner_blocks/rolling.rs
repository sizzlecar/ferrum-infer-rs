//! Worker coordinator for overlapping original source7 sessions.
//! One transport block owns every raw ticket. Source membership is fixed before
//! reserve; a source deadline/failure cannot damage another source's record.
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1;
use ferrum_scheduler::implementations::continuous::cost_profile::StructuredServiceRecordV7;

enum CloseOutput {
    Closed(session::PreparedClose, StructuredServiceRecordV7),
    Retired(session::RetirementReport),
}
type CloseTask = close_task::OwnedTask<BlockSession, CloseOutput>;

struct PendingClose {
    /// None owns a terminal source, which returns only an archive/audit result.
    index: Option<usize>,
    enrollment: tickets::SourceEnrollment,
    window: Arc<tickets::Window>,
    cutoff: u64,
    task: CloseTask,
}

struct Active {
    session: BlockSession,
    failure: Option<String>,
}
struct Successor {
    parent_generation: u64,
    parent_capture: Option<[u8; 32]>,
    seed: Option<DeclaredAlgorithmUniverseV1>,
}

// The source7 protocol already caps failure text at 4096 bytes. Reuse that
// bound for retained coordinator reasons, including multi-byte diagnostics.
fn retained_reason(value: impl std::fmt::Display) -> String {
    let text = value.to_string();
    let mut reason = String::with_capacity(4096);
    for ch in text.chars() {
        if reason.len() + ch.len_utf8() > 4096 {
            break;
        }
        reason.push(ch);
    }
    reason
}

#[derive(Debug, Clone, Serialize)]
pub(in crate::continuous_engine::inner::cost_observation) struct RollingAudit {
    pub original_block: u64,
    pub at_boundary: bool,
    pub active_sources: Vec<StructuredServiceAuditV7>,
    pub active_enrollments: Vec<tickets::SourceEnrollment>,
    pub pending_successor_parents: Vec<u64>,
    pub pending_publication_generation: Option<u64>,
    pub pending_close_enrollment: Option<tickets::SourceEnrollment>,
    pub pending_retirement_generation: Option<u64>,
    pub close_task_stack_bytes: usize,
    pub close_task_retained_bytes: usize,
    pub retirement_handoff_bytes: usize,
    pub resource: budget::BudgetAudit,
}

pub(in super::super) struct Rolling {
    sources: Vec<Active>,
    pending: VecDeque<Successor>,
    budget: budget::Budget,
    original_block: u64,
    at_boundary: bool,
    close_cursor: usize,
    pending_publication: Option<u64>,
    last_admission_block: Option<u64>,
    pending_close: Option<PendingClose>,
    retirement_handoff_bytes: usize,
    #[cfg(test)]
    close_gate: Option<Arc<close_task::TestGate>>,
}
impl Rolling {
    #[cfg(test)]
    pub(in super::super) fn hold_close(&mut self) -> Arc<close_task::TestGate> {
        let gate = Arc::new(close_task::TestGate::default());
        assert!(self.close_gate.replace(gate.clone()).is_none());
        gate
    }
    pub(in super::super) fn computation_pending(&self) -> bool {
        self.pending_close.is_some()
    }
    pub(in super::super) fn finish_pending(&self) -> bool {
        !self.at_boundary
            || !self.sources.is_empty()
            || self.pending_publication.is_some()
            || self.pending_close.is_some()
    }
    fn new(
        controller: &Controller,
        live: &LiveCalibration,
        seed: Option<DeclaredAlgorithmUniverseV1>,
    ) -> Result<Self, FerrumError> {
        let schedule = schedule(&controller.settings, &live.declaration.settings)?;
        let slots = controller
            .settings
            .maximum_retained_generations
            .get()
            .checked_add(1)
            .ok_or_else(|| error("rolling source capacity overflow"))?;
        let sources = Vec::<Active>::with_capacity(slots);
        let mut pending = VecDeque::<Successor>::with_capacity(slots);
        let retirement_handoff_bytes = session::PreparedRetirement::retained_bytes(
            controller.settings.discovery_offered_waves.get(),
        )
        .ok_or_else(|| error("retirement handoff capacity overflow"))?;
        let metadata = sources
            .capacity()
            .checked_mul(std::mem::size_of::<Active>())
            .and_then(|bytes| {
                bytes.checked_add(
                    pending
                        .capacity()
                        .checked_mul(std::mem::size_of::<Successor>())?,
                )
            })
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Self>()))
            // One task for the whole coordinator, partitioned across the
            // existing K+1 slots before collector/catalog shares are derived.
            .and_then(|bytes| bytes.checked_add(close_task::STACK_BYTES))
            .and_then(|bytes| bytes.checked_add(CloseTask::retained_bytes()))
            .and_then(|bytes| bytes.checked_add(retirement_handoff_bytes))
            .map(|bytes| bytes.div_ceil(slots))
            // Published domains and a temporary exact-sized new-domain list.
            .and_then(|bytes| {
                bytes.checked_add(
                    controller
                        .settings
                        .maximum_owners
                        .get()
                        .checked_mul(2 * std::mem::size_of::<[u8; 32]>())?,
                )
            })
            .and_then(|bytes| bytes.checked_add(4096))
            .ok_or_else(|| error("rolling source metadata capacity overflow"))?;
        let budget = budget::Budget::new(
            &controller.settings,
            live.maximum_retained_bytes,
            &schedule,
            metadata,
        )?;
        pending.push_back(Successor {
            parent_generation: 0,
            parent_capture: None,
            seed,
        });
        Ok(Self {
            sources,
            pending,
            budget,
            original_block: 0,
            at_boundary: true,
            close_cursor: 0,
            pending_publication: None,
            last_admission_block: None,
            pending_close: None,
            retirement_handoff_bytes,
            #[cfg(test)]
            close_gate: None,
        })
    }
    pub(in super::super) fn audit(&self) -> RollingAudit {
        RollingAudit {
            original_block: self.original_block,
            at_boundary: self.at_boundary,
            active_enrollments: self
                .sources
                .iter()
                .map(|source| source.session.enrollment())
                .chain(
                    self.pending_close
                        .iter()
                        .map(|pending| pending.enrollment.clone()),
                )
                .collect(),
            active_sources: self
                .sources
                .iter()
                .map(|source| source.session.audit())
                .collect(),
            pending_successor_parents: self.pending.iter().map(|p| p.parent_generation).collect(),
            pending_publication_generation: self.pending_publication,
            pending_close_enrollment: self.pending_close.as_ref().map(|p| p.enrollment.clone()),
            pending_retirement_generation: self
                .pending_close
                .as_ref()
                .filter(|pending| pending.index.is_none())
                .map(|pending| pending.enrollment.generation),
            close_task_stack_bytes: close_task::STACK_BYTES,
            close_task_retained_bytes: CloseTask::retained_bytes(),
            retirement_handoff_bytes: self.retirement_handoff_bytes,
            resource: self.budget.audit(),
        }
    }
    pub(super) fn has_sources(&self) -> bool {
        !self.at_boundary
    }
    pub(in super::super) fn validate_ticket(&self, ticket: &Ticket) -> Result<(), &'static str> {
        if self.pending_close.is_some() {
            // Dispatch requires every original ticket already retired. No
            // later consumption can add a member to this frozen population.
            return Err("automatic_original_block_already_retired");
        }
        if self.at_boundary
            || ticket.window.phase as u64 != self.original_block
            || !ticket.window.is_rolling()
        {
            return Err("automatic_original_rolling_block_mismatch");
        }
        for enrollment in ticket.window.enrolled() {
            let source = self
                .sources
                .iter()
                .find(|source| source.session.enrollment().generation == enrollment.generation)
                .ok_or("automatic_original_rolling_source_missing")?;
            source.session.validate_ticket(ticket)?;
        }
        Ok(())
    }
    pub(super) fn ingest(&mut self, live: &LiveCalibration, cutoff: u64) {
        if self.at_boundary || self.pending_close.is_some() {
            return;
        }
        for source in &mut self.sources {
            if source.failure.is_some() {
                continue;
            }
            let result = source.session.ingest_one(live, cutoff);
            let generation = source.session.enrollment().generation;
            let charge = self.budget.charge(generation, source.session.work());
            if let Err(reason) = result.and(charge) {
                source.failure = Some(retained_reason(reason));
            }
        }
    }
    pub(super) fn has_pending_records(&self, live: &LiveCalibration) -> bool {
        !self.at_boundary
            && self.pending_close.is_none()
            && self
                .sources
                .iter()
                .any(|s| s.failure.is_none() && s.session.has_pending_records(live))
    }
    pub(in super::super) fn check_fresh_catalog(
        &self,
        fresh: &BTreeSet<[u8; 32]>,
        charges: &BTreeMap<[u8; 32], Option<usize>>,
    ) -> Result<(), FerrumError> {
        if fresh.iter().any(|origin| {
            (self
                .sources
                .iter()
                .any(|source| source.session.enrollment().capture_identity == *origin)
                || self
                    .pending_close
                    .as_ref()
                    .is_some_and(|pending| pending.enrollment.capture_identity == *origin))
                && charges
                    .get(origin)
                    .copied()
                    .flatten()
                    .is_none_or(|bytes| bytes > self.budget.catalog_bytes)
        }) {
            return Err(error(
                "rolling catalog exceeds its declared source-slot partition",
            ));
        }
        Ok(())
    }
    pub(in super::super) fn check_catalog_origins(
        &self,
        origins: &VecDeque<[u8; 32]>,
    ) -> Result<(), FerrumError> {
        let mut captures: BTreeSet<_> = origins.iter().copied().collect();
        captures.extend(
            self.sources
                .iter()
                .map(|source| source.session.enrollment().capture_identity),
        );
        captures.extend(
            self.pending_close
                .iter()
                .map(|pending| pending.enrollment.capture_identity),
        );
        captures.extend(
            self.pending
                .iter()
                .filter_map(|pending| pending.parent_capture),
        );
        if captures.len() > self.budget.maximum_sources {
            return Err(error(
                "published catalog and collecting sources exceed shared capture slots",
            ));
        }
        Ok(())
    }
    pub(in super::super) fn note_publication(&mut self, failure: Option<String>) {
        if let Some(generation) = self.pending_publication.take() {
            if let Some(reason) = failure {
                if let Some(source) = self
                    .sources
                    .iter_mut()
                    .find(|s| s.session.enrollment().generation == generation)
                {
                    source.failure = Some(retained_reason(format_args!(
                        "runtime installation failed: {reason}"
                    )));
                }
            }
        }
    }
    fn captures(&self, state: &State, skip_pending: bool) -> BTreeSet<[u8; 32]> {
        let mut captures: BTreeSet<_> = state.origins.iter().copied().collect();
        captures.extend(
            self.sources
                .iter()
                .map(|s| s.session.enrollment().capture_identity),
        );
        captures.extend(
            self.pending_close
                .iter()
                .map(|pending| pending.enrollment.capture_identity),
        );
        captures.extend(
            self.pending
                .iter()
                .skip(usize::from(skip_pending))
                .filter_map(|p| p.parent_capture),
        );
        captures
    }
    fn settle_retirement(
        &mut self,
        controller: &Controller,
        live: &LiveCalibration,
        state: &mut State,
        window: &tickets::Window,
        mut session: BlockSession,
        result: Result<session::RetirementReport, FerrumError>,
    ) {
        let generation = session.enrollment().generation;
        let failure = match result {
            Ok(report) => {
                let (failure, published) = report.apply(live, generation);
                if let Some(source) = published {
                    state.last_diagnostic_publication =
                        Some(DiagnosticPublication { generation, source });
                }
                failure
            }
            Err(reason) => {
                // Panic, cancellation or failed handoff may leave a partially
                // modified collector. Never retry or qualify that owner.
                let reason = retained_reason(reason);
                session.abandon_retirement(live, &reason);
                live.remember_failed_source(generation, None, false, false, &reason);
                Some(reason)
            }
        };
        if let Some(reason) = failure {
            controller.record_history(
                state,
                GenerationAudit {
                    generation,
                    discovery_origin: None,
                    failed: true,
                    installed: false,
                    reason: Some(reason),
                    first_settlement_failure: live.collected.lock().first_settlement_failure,
                    population: window.audit(),
                    phase_populations: std::array::from_fn(|_| None),
                },
            );
            let _ =
                live.failed_generations
                    .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1));
        }
        let observed = session.work();
        if let Err(reason) = self.budget.close(generation, observed) {
            tracing::warn!(generation, observed_work = ?observed, %reason,
                "Rolling source final work rejected; reservation forfeited and closed");
            live.remember_failed_source(generation, None, false, false, &reason.to_string());
        }
    }

    /// false means the unique task owns a retirement. Keep the original Window
    /// and collected evidence until it returns; no new roster can be enrolled.
    fn retire(
        &mut self,
        controller: &Controller,
        live: &LiveCalibration,
        state: &mut State,
        window: &Arc<tickets::Window>,
        clock: &dyn CostObservationClock,
        cutoff: u64,
        stopped: bool,
    ) -> Result<bool, FerrumError> {
        let now = clock
            .now_ns()
            .ok_or_else(|| error("rolling source close clock unavailable"))?;
        let mut index = 0;
        while index < self.sources.len() {
            let source = &mut self.sources[index];
            let audit = source.session.audit();
            let terminal =
                !audit.owners.is_empty() && audit.owners.iter().all(|owner| owner.phase.is_none());
            if now > source.session.enrollment().deadline_ns {
                source
                    .failure
                    .get_or_insert_with(|| "original source collection deadline expired".into());
            }
            if source.failure.is_none()
                && terminal
                && !audit.owners.iter().any(|owner| owner.qualified)
            {
                source.failure = Some(failed_population_reason(&audit));
            }
            if source.failure.is_none() && !terminal && !stopped {
                index += 1;
                continue;
            }
            let source = self.sources.remove(index);
            let enrollment = source.session.enrollment();
            let prepared =
                match source
                    .session
                    .prepare_retirement(live, window, clock, cutoff, source.failure)
                {
                    Ok(prepared) => prepared,
                    Err(reason) => {
                        self.settle_retirement(
                            controller,
                            live,
                            state,
                            window,
                            source.session,
                            Err(reason),
                        );
                        continue;
                    }
                };
            if let Some(notification) = live.notification.get() {
                #[cfg(test)]
                let gate = self.close_gate.take();
                match CloseTask::spawn(
                    source.session,
                    notification.clone(),
                    BlockSession::abandon,
                    move |session| {
                        #[cfg(test)]
                        if let Some(gate) = gate {
                            gate.pause()?;
                        }
                        Ok(CloseOutput::Retired(session.compute_retirement(prepared)))
                    },
                ) {
                    Ok(task) => {
                        self.pending_close = Some(PendingClose {
                            index: None,
                            enrollment,
                            window: window.clone(),
                            cutoff,
                            task,
                        });
                        return Ok(false);
                    }
                    Err((session, reason)) => {
                        self.settle_retirement(
                            controller,
                            live,
                            state,
                            window,
                            session,
                            Err(error(format!("source retirement spawn failed: {reason}"))),
                        );
                    }
                }
            } else {
                // Workerless fixtures use the identical frozen transaction.
                let mut session = source.session;
                let report = session.compute_retirement(prepared);
                self.settle_retirement(controller, live, state, window, session, Ok(report));
            }
        }
        Ok(true)
    }
    fn open_boundary(
        &mut self,
        controller: &Controller,
        live: &LiveCalibration,
        state: &mut State,
        clock: &dyn CostObservationClock,
        cutoff: u64,
        stopped: bool,
    ) -> Result<(), FerrumError> {
        let old = live.active.read().clone();
        if old.audit().issued != old.audit().retired {
            return Ok(());
        }
        if !self.retire(controller, live, state, &old, clock, cutoff, stopped)? {
            return Ok(());
        }
        if stopped {
            self.pending.clear();
            return Ok(());
        }
        if self.sources.is_empty() && self.pending.is_empty() {
            self.pending.push_back(Successor {
                parent_generation: state.generation,
                parent_capture: None,
                seed: state.cold_algorithm_seed.clone(),
            });
        }
        let opening = ExportClockReading::opening(clock).map_err(error)?;
        // At most one attempt per genuine boundary; a repeated worker turn at
        // the same boundary cannot select another seed or mint another credit.
        if !self.pending.is_empty()
            && self.last_admission_block != Some(self.original_block)
            && self.captures(state, true).len() < self.budget.maximum_sources
        {
            self.last_admission_block = Some(self.original_block);
            let generation = state
                .generation
                .checked_add(1)
                .ok_or_else(|| error("rolling generation overflow"))?;
            if self.budget.reserve(generation) {
                // Generation identifies the attempt, including failed
                // construction. A later genuine boundary cannot reuse it.
                state.generation = generation;
                let pending = self.pending.front().expect("checked pending successor");
                let result = BlockSession::open(
                    live,
                    controller,
                    generation,
                    opening,
                    cutoff,
                    clock,
                    pending.seed.as_ref(),
                    Some(self.budget.collector_bytes),
                );
                match result {
                    Ok(mut session) => {
                        if let Err(reason) = self.budget.charge(generation, session.work()) {
                            session.fail(live, &old, clock, cutoff, &reason.to_string());
                            let observed = session.work();
                            let closed = self.budget.close(generation, observed);
                            tracing::warn!(generation, observed_work = ?observed, ?closed, %reason,
                                "Rolling source admission work rejected before original enrollment");
                            live.remember_failed_source(
                                generation,
                                None,
                                false,
                                false,
                                &reason.to_string(),
                            );
                            let _ = live.failed_generations.fetch_update(
                                Ordering::AcqRel,
                                Ordering::Acquire,
                                |n| n.checked_add(1),
                            );
                        } else {
                            state.opening_ns = opening.monotonic_ns;
                            state.declaration_sha256 = Some(session.declaration_sha256);
                            self.pending.pop_front();
                            self.sources.push(Active {
                                session,
                                failure: None,
                            });
                        }
                    }
                    Err(reason) => {
                        // Construction may have encoded its header before
                        // failing. Forfeit the entire reserved work allowance.
                        self.budget.forfeit(generation)?;
                        tracing::warn!(generation, %reason, "Rolling source could not open at original boundary");
                    }
                }
            }
        }
        for source in &mut self.sources {
            if source.session.next_pending {
                if let Err(reason) = source.session.open_next(live, opening.monotonic_ns, cutoff) {
                    source.failure = Some(retained_reason(reason));
                }
                if let Err(reason) = self.budget.charge(
                    source.session.enrollment().generation,
                    source.session.work(),
                ) {
                    source.failure = Some(retained_reason(reason));
                }
            }
        }
        // Local open failure cannot enroll an invalid source. Retire it while
        // the old complete raw block is still available.
        if !self.retire(controller, live, state, &old, clock, cutoff, false)? {
            return Ok(());
        }
        let enrollments: Vec<_> = self
            .sources
            .iter()
            .map(|s| s.session.enrollment())
            .collect();
        let deadline = enrollments
            .iter()
            .map(|s| s.deadline_ns)
            .max()
            .or_else(|| {
                opening
                    .monotonic_ns
                    .checked_add(live.declaration.maximum_window_ns)
            })
            .ok_or_else(|| error("rolling transport deadline overflow"))?;
        let block = self
            .original_block
            .checked_add(1)
            .ok_or_else(|| error("rolling original block overflow"))?;
        let next = tickets::Window::with_enrollments(
            state.generation,
            usize::try_from(block).map_err(error)?,
            controller.settings.discovery_offered_waves.get(),
            deadline,
            enrollments,
        );
        let retained_bytes = next
            .retained_bytes()
            .filter(|bytes| *bytes <= live.maximum_retained_bytes)
            .ok_or_else(|| error("rolling original ticket/roster memory capacity"))?;
        if let Some(worker) = live.notification.get() {
            next.attach_worker(worker.clone());
        }
        *live.collected.lock() = Collected {
            retained_bytes,
            ..Default::default()
        };
        *live.active.write() = next;
        self.original_block = block;
        self.close_cursor = 0;
        self.at_boundary = false;
        state.finished = false;
        Ok(())
    }
    fn advance(
        &mut self,
        controller: &Controller,
        live: &LiveCalibration,
        state: &mut State,
        clock: &dyn CostObservationClock,
        cutoff: u64,
        stopped: bool,
    ) -> Result<Option<publication::Publication>, FerrumError> {
        if let Some(pending) = &self.pending_close {
            let current = live.active.read().clone();
            if pending.index.is_some() && !pending.current(&current, clock.now_ns(), cutoff) {
                pending.task.cancel();
            }
            if !pending.task.ready() {
                return Ok(None);
            }
            let mut pending = self.pending_close.take().expect("owned close");
            let (mut session, result) = pending.task.take().expect("ready owned close");
            if let Some(index) = pending.index {
                let result = if pending.current(&current, clock.now_ns(), cutoff) {
                    result.and_then(|result| match result {
                        CloseOutput::Closed(prepared, close) => session.finish_close(
                            live,
                            &current,
                            clock,
                            &controller.import,
                            prepared,
                            close,
                        ),
                        CloseOutput::Retired(_) => Err(error("owned close result kind changed")),
                    })
                } else {
                    Err(error("completed original block is expired or obsolete"))
                };
                self.sources.insert(
                    index,
                    Active {
                        session,
                        failure: None,
                    },
                );
                self.close_cursor = index + 1;
                if let Some(publication) =
                    self.finish_result(controller, state, &current, index, result)?
                {
                    return Ok(Some(publication));
                }
            } else {
                // Retiring an expired source must finish even after its TTL.
                // Its result has no model/publication authority.
                let report = result.and_then(|result| match result {
                    CloseOutput::Retired(report) => Ok(report),
                    CloseOutput::Closed(..) => Err(error("owned retirement result kind changed")),
                });
                self.settle_retirement(controller, live, state, &pending.window, session, report);
            }
        }
        if self.at_boundary {
            self.open_boundary(controller, live, state, clock, cutoff, stopped)?;
            return Ok(None);
        }
        let window = live.active.read().clone();
        let now = clock.now_ns();
        for source in &mut self.sources {
            if now.is_none_or(|at| at > source.session.enrollment().deadline_ns) {
                source.failure.get_or_insert_with(|| {
                    "original source collection deadline expired or clock unavailable".into()
                });
            }
        }
        let complete = now.is_some_and(|at| window.complete(at));
        if now.is_none()
            || (stopped && !complete)
            || (!self.sources.is_empty() && self.sources.iter().all(|s| s.failure.is_some()))
        {
            window.fail();
        }
        let population = window.audit();
        if population.issued != population.retired {
            return Ok(None);
        }
        if population.failed {
            for source in &mut self.sources {
                source
                    .failure
                    .get_or_insert_with(|| "shared original block failed or incomplete".into());
            }
            if !self.retire(controller, live, state, &window, clock, cutoff, stopped)? {
                return Ok(None);
            }
            state.last_failure = Some(RetainedFailure {
                window,
                collected: std::mem::take(&mut *live.collected.lock()),
            });
            self.at_boundary = true;
            return Ok(None);
        }
        if !complete
            || self
                .sources
                .iter()
                .any(|s| s.failure.is_none() && !s.session.records_complete())
        {
            return Ok(None);
        }
        while self.close_cursor < self.sources.len() {
            let index = self.close_cursor;
            let source = &mut self.sources[index];
            self.close_cursor += 1;
            if source.failure.is_some() {
                continue;
            }
            if let Some(notification) = live.notification.get() {
                let prepared = match source.session.prepare_close(live, &window, clock, cutoff) {
                    Ok(value) => value,
                    Err(reason) => {
                        source.failure = Some(retained_reason(reason));
                        continue;
                    }
                };
                let source = self.sources.remove(index);
                let enrollment = source.session.enrollment();
                #[cfg(test)]
                let gate = self.close_gate.take();
                match CloseTask::spawn(
                    source.session,
                    notification.clone(),
                    BlockSession::abandon,
                    move |session| {
                        #[cfg(test)]
                        if let Some(gate) = gate {
                            gate.pause()?;
                        }
                        let close = session.compute_close(&prepared)?;
                        Ok(CloseOutput::Closed(prepared, close))
                    },
                ) {
                    Ok(task) => {
                        self.pending_close = Some(PendingClose {
                            index: Some(index),
                            enrollment,
                            window,
                            cutoff,
                            task,
                        });
                        return Ok(None);
                    }
                    Err((session, reason)) => {
                        self.sources.insert(
                            index,
                            Active {
                                session,
                                failure: Some(retained_reason(format_args!(
                                    "block close spawn failed: {reason}"
                                ))),
                            },
                        );
                        continue;
                    }
                }
            }
            // Deterministic workerless fixtures retain the identical
            // prepare/compute/finish protocol without starting a hidden worker.
            let result = source
                .session
                .complete(live, &window, clock, cutoff, &controller.import);
            if let Some(publication) =
                self.finish_result(controller, state, &window, index, result)?
            {
                return Ok(Some(publication));
            }
        }
        self.budget.complete_original_block(self.original_block)?;
        self.at_boundary = true;
        Ok(None)
    }

    fn finish_result(
        &mut self,
        controller: &Controller,
        state: &mut State,
        window: &Arc<tickets::Window>,
        index: usize,
        result: Result<Option<publication::Publication>, FerrumError>,
    ) -> Result<Option<publication::Publication>, FerrumError> {
        let source = &mut self.sources[index];
        let generation = source.session.enrollment().generation;
        let charge = self.budget.charge(generation, source.session.work());
        if let Err(reason) = charge {
            source.failure = Some(retained_reason(reason));
            return Ok(None);
        }
        if let Some(seed) = source.session.take_successor_seed() {
            if self.pending.len() >= self.budget.maximum_sources {
                return Err(error("rolling pending successor capacity"));
            }
            self.pending.push_back(Successor {
                parent_generation: generation,
                parent_capture: Some(source.session.enrollment().capture_identity),
                seed,
            });
        }
        match result {
            Ok(Some(publication)) => {
                self.pending_publication = Some(generation);
                state.install_pending = true;
                controller.record_history(
                    state,
                    GenerationAudit {
                        generation,
                        discovery_origin: None,
                        failed: false,
                        installed: false,
                        reason: None,
                        first_settlement_failure: None,
                        population: window.audit(),
                        phase_populations: std::array::from_fn(|_| None),
                    },
                );
                return Ok(Some(publication));
            }
            Ok(None) => {}
            Err(reason) => source.failure = Some(retained_reason(reason)),
        }
        Ok(None)
    }
}

impl PendingClose {
    fn current(&self, current: &Arc<tickets::Window>, now: Option<u64>, cutoff: u64) -> bool {
        Arc::ptr_eq(&self.window, current)
            && cutoff >= self.cutoff
            && current.enrollment(self.enrollment.generation) == Some(&self.enrollment)
            && now.is_some_and(|at| at <= self.enrollment.deadline_ns && current.complete(at))
    }
}

impl Controller {
    pub(super) fn advance_rolling_owner_blocks(
        &self,
        live: &LiveCalibration,
        clock: &dyn CostObservationClock,
        cutoff: u64,
        stopping: bool,
    ) -> Result<Option<publication::Publication>, FerrumError> {
        if stopping {
            self.stop();
        }
        let mut state = self.state.lock();
        if !self.requested.load(Ordering::Acquire) || state.install_pending {
            return Ok(None);
        }
        let mut rolling = match state.rolling.take() {
            Some(value) => value,
            None => Rolling::new(self, live, state.cold_algorithm_seed.clone())?,
        };
        let result = rolling.advance(
            self,
            live,
            &mut state,
            clock,
            cutoff,
            self.stopped.load(Ordering::Acquire),
        );
        state.rolling = Some(rolling);
        result
    }
}
