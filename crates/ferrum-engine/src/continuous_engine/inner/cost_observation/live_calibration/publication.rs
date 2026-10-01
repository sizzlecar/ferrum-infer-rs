//! Sole-worker phase progression. Producers never take this state lock or do IO.
use super::super::{
    profile,
    profile_export::{ExportClockReading, PublishedFile, StagedFile},
};
use super::*;
use ferrum_scheduler::implementations::continuous::{
    cost_model::structured_v2::StructuredPhaseV2,
    cost_profile::{
        self as file, StructuredServiceClockV6, StructuredServiceCollectorV6,
        StructuredServiceDeclarationV6, StructuredServiceHeaderV6, StructuredServiceRecordV6,
        StructuredServiceWaveV6,
    },
};
mod failure;
pub(super) use failure::FailedSourceReceipt;

pub(super) struct Policy {
    pub directory: PathBuf,
    pub maximum_generations: usize,
    pub maximum_source_bytes: u64,
    pub import: ferrum_types::SloCostProfileImportConfig,
    pub opening: ExportClockReading,
}

pub(super) struct State {
    generation: u64,
    opening: Option<ExportClockReading>,
    session: Option<Session>,
    open_attempted: bool,
    published_source: Option<PublishedFile>,
    failure: Option<String>,
    install_pending: bool,
    stopping: bool,
    finished: bool,
}
impl Default for State {
    fn default() -> Self {
        Self {
            generation: 1,
            opening: None,
            session: None,
            open_attempted: false,
            published_source: None,
            failure: None,
            install_pending: false,
            stopping: false,
            finished: false,
        }
    }
}
struct Session {
    collector: StructuredServiceCollectorV6,
    source: Option<StagedFile>,
    profile_path: PathBuf,
    phase: usize,
    written_ticket: u64,
    write_failed: bool,
    footer_written: bool,
}
pub(in crate::continuous_engine::inner::cost_observation) struct Publication {
    pub children: Vec<file::ImportedStructuredModelV2>,
    pub receipt: ferrum_types::SloCostProfileReceipt,
}

fn error(e: impl std::fmt::Display) -> FerrumError {
    FerrumError::config(format!("live structured calibration: {e}"))
}
pub(super) fn phase(index: usize) -> StructuredPhaseV2 {
    match index {
        0 => StructuredPhaseV2::Fit,
        1 => StructuredPhaseV2::Residual,
        _ => StructuredPhaseV2::Qualification,
    }
}
fn paired(v: ExportClockReading) -> StructuredServiceClockV6 {
    StructuredServiceClockV6 {
        wall_unix_ns: v.wall_unix_ns,
        monotonic_ns: v.monotonic_ns,
    }
}

impl Session {
    fn open(
        live: &LiveCalibration,
        policy: &Policy,
        generation: u64,
        opening: ExportClockReading,
        cutoff: u64,
    ) -> Result<Self, FerrumError> {
        use sha2::Digest;
        std::fs::create_dir_all(&policy.directory).map_err(error)?;
        let id = uuid::Uuid::new_v4();
        let header = StructuredServiceHeaderV6::new(
            sha2::Sha256::digest(id.as_bytes()).into(),
            generation,
            file::ProfileFingerprint::from(&live.fingerprint),
            serde_json::to_value(live.producer.current()?).map_err(error)?,
            paired(opening),
            StructuredServiceDeclarationV6 {
                route_population: ferrum_types::SloCalibrationRoutePopulationV1::AllAttempts,
                domain_policy: ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredServiceDomainPolicyV1::AllOffered,
                nonnegative_envelope: None,
                phase_offered_waves: live.declaration.phase_offered_waves,
                maximum_window_ns: live.declaration.maximum_window_ns,
                settings: live.declaration.settings.clone(),
                scopes: live.declaration.scopes.clone(),
                maximum_retained_numeric_bytes: live.maximum_retained_bytes,
            },
            policy.maximum_source_bytes,
        )
        .map_err(error)?;
        let source_path = policy
            .directory
            .join(format!("service-{id}-{generation}.jsonl"));
        let profile_path = policy
            .directory
            .join(format!("service-{id}-{generation}.profile.json"));
        let mut source =
            StagedFile::create(&source_path, policy.maximum_source_bytes).map_err(error)?;
        source.preserve_unpublished();
        let initialized = source.json_line(&header).map_err(error).and_then(|_| {
            StructuredServiceCollectorV6::new(header, profile::load_limits(&policy.import))
                .map_err(error)
        });
        let collector = match initialized {
            Ok(collector) => collector,
            Err(reason) => {
                live.remember_failed_source(
                    generation,
                    Some(source.unpublished_receipt()),
                    false,
                    true,
                    &reason.to_string(),
                );
                return Err(reason);
            }
        };
        let mut session = Self {
            collector,
            source: Some(source),
            profile_path,
            phase: 0,
            written_ticket: 0,
            write_failed: false,
            footer_written: false,
        };
        let result = session.record(&StructuredServiceRecordV6::PhaseOpen {
            phase: phase(0),
            opened_at_ns: opening.monotonic_ns,
            fifo_cutoff: cutoff,
        });
        if let Err(reason) = result {
            live.remember_failed_source(
                generation,
                session.source.as_ref().map(StagedFile::unpublished_receipt),
                false,
                true,
                &reason.to_string(),
            );
            return Err(reason);
        }
        Ok(session)
    }
    fn record(&mut self, record: &StructuredServiceRecordV6) -> Result<(), FerrumError> {
        self.collector.push(record).map_err(error)?;
        self.write_raw(record)
    }
    fn write_raw(&mut self, record: &impl serde::Serialize) -> Result<(), FerrumError> {
        if self.write_failed {
            return Err(error("source already has an incomplete write"));
        }
        let result = self
            .source
            .as_mut()
            .ok_or_else(|| error("source already sealed"))?
            .json_line(record)
            .map_err(error);
        self.write_failed |= result.is_err();
        result
    }
}

pub(super) fn completed_record(
    wave: &Wave,
    phase_index: usize,
    offset: u64,
) -> Result<StructuredServiceRecordV6, FerrumError> {
    let stages = match &wave.evidence {
        WaveEvidence::Settled(stages) => stages,
        WaveEvidence::NoSubmission(receipt) => {
            return Ok(StructuredServiceRecordV6::NotSubmitted {
                attempt: receipt.wire.clone().with_source_position(
                    offset
                        .checked_add(wave.ticket)
                        .ok_or_else(|| error("ticket overflow"))?,
                    phase(phase_index),
                ),
            });
        }
    };
    if let Some(route) = stages.route_evidence.as_ref().filter(|r| r.is_outside()) {
        return Ok(StructuredServiceRecordV6::OutsideDeclaredRoute {
            wave: route.outside_record(
                stages,
                offset
                    .checked_add(wave.ticket)
                    .ok_or_else(|| error("ticket overflow"))?,
                phase(phase_index),
                wave.fifo,
            )?,
        });
    }
    let independent = stages
        .statistical_evidence
        .as_ref()
        .and_then(|v| v.independent_attention_v2())
        .map(|v| v.to_wire_v2());
    let diagnostic = serde_json::to_value(stages.structured_source_view()).map_err(error)?;
    let wire = StructuredServiceWaveV6::from_diagnostic(
        offset
            .checked_add(wave.ticket)
            .ok_or_else(|| error("ticket overflow"))?,
        phase(phase_index),
        stages
            .prepare_started_at_ns
            .ok_or_else(|| error("ticket clock missing"))?,
        wave.fifo,
        diagnostic,
        independent,
    )
    .map_err(error)?;
    let wire = if let Some(route) = &stages.route_evidence {
        wire.with_prepared_route(serde_json::to_value(route.eligible_diagnostic()).map_err(error)?)
            .map_err(error)?
    } else {
        wire
    };
    Ok(StructuredServiceRecordV6::Completed { wave: wire })
}

impl LiveCalibration {
    #[cfg(test)]
    pub(in crate::continuous_engine::inner::cost_observation) fn pending_profile_path(
        &self,
    ) -> Option<PathBuf> {
        self.worker
            .lock()
            .session
            .as_ref()
            .map(|s| s.profile_path.clone())
    }

    /// Called only after the original worker releases its drain/feedback locks.
    /// Every new phase opens after the prior complete population was frozen.
    pub(in crate::continuous_engine::inner::cost_observation) fn advance(
        &self,
        clock: &dyn CostObservationClock,
        cutoff: u64,
    ) -> Result<Option<Publication>, FerrumError> {
        if let Some(automatic) = &self.automatic {
            return automatic.advance(self, clock, cutoff, false);
        }
        self.advance_or_finish(clock, cutoff, false)
    }

    /// Stop issuing before the final FIFO drain; existing private tickets keep
    /// their original retirement obligation and may still deliver evidence.
    pub(in crate::continuous_engine::inner::cost_observation) fn stop_capture(&self) {
        if let Some(automatic) = &self.automatic {
            automatic.stop();
        }
        self.active.read().close();
    }

    pub(in crate::continuous_engine::inner::cost_observation) fn finish(
        &self,
        clock: &dyn CostObservationClock,
        cutoff: u64,
    ) -> Result<Option<Publication>, FerrumError> {
        if let Some(automatic) = &self.automatic {
            return automatic.advance(self, clock, cutoff, true);
        }
        self.advance_or_finish(clock, cutoff, true)
    }

    fn advance_or_finish(
        &self,
        clock: &dyn CostObservationClock,
        cutoff: u64,
        stopping: bool,
    ) -> Result<Option<Publication>, FerrumError> {
        let Some(policy) = &self.publication else {
            return Ok(None);
        };
        let mut state = self.worker.lock();
        let mut window = self.active.read().clone();
        state.stopping |= stopping;
        if state.stopping {
            if state.finished {
                return Ok(None);
            }
            window.fail();
            state
                .failure
                .get_or_insert_with(|| "shutdown before generation publication".into());
        }
        if state.finished {
            let audit = window.audit();
            if state.generation >= policy.maximum_generations as u64
                || audit.retired != audit.issued
            {
                return Ok(None);
            }
            let opening = ExportClockReading::opening(clock).map_err(error)?;
            let deadline = opening
                .monotonic_ns
                .checked_add(self.declaration.maximum_window_ns)
                .ok_or_else(|| error("window clock overflow"))?;
            state.generation += 1;
            state.opening = Some(opening);
            state.session = None;
            state.open_attempted = false;
            state.published_source = None;
            state.failure = None;
            state.install_pending = false;
            state.finished = false;
            *self.collected.lock() = Collected::default();
            window = tickets::Window::new(
                state.generation,
                0,
                self.declaration.phase_offered_waves[0],
                deadline,
            );
            if let Some(worker) = self.notification.get() {
                window.attach_worker(worker.clone());
            }
            *self.active.write() = window.clone();
        }
        if window.audit().failed {
            state
                .failure
                .get_or_insert_with(|| "offered window failed or expired".into());
        }
        if state.failure.is_some() {
            return self.finish_failed_generation(&mut state, &window, policy, clock, cutoff);
        }
        let result = self.advance_generation(&mut state, &window, policy, clock, cutoff);
        if let Err(reason) = result {
            window.fail();
            state.failure.get_or_insert_with(|| reason.to_string());
            return self.finish_failed_generation(&mut state, &window, policy, clock, cutoff);
        }
        result
    }

    fn advance_generation(
        &self,
        state: &mut State,
        window: &Arc<tickets::Window>,
        policy: &Policy,
        clock: &dyn CostObservationClock,
        cutoff: u64,
    ) -> Result<Option<Publication>, FerrumError> {
        if state.session.is_none() {
            let opening = state.opening.unwrap_or(policy.opening);
            // The first active window predates the worker's first drain. Zero
            // is the lower FIFO boundary; every actual receipt remains numbered.
            state.open_attempted = true;
            state.session = Some(Session::open(self, policy, state.generation, opening, 0)?);
        }
        let now = clock.now_ns().ok_or_else(|| error("clock unavailable"))?;
        if !window.complete(now) {
            if window.audit().failed {
                return Err(error("offered window failed or expired"));
            }
            return Ok(None);
        }
        let session = state.session.as_mut().expect("opened above");
        if session.phase != window.phase || state.generation != window.generation {
            return Err(error("window generation/phase mismatch"));
        }
        let mut collected = self.collected.lock();
        if collected.failure.is_some()
            || collected.waves.len() != self.declaration.phase_offered_waves[session.phase]
        {
            return Err(error("closed population is incomplete"));
        }
        collected.waves.sort_by_key(|w| w.ticket);
        let offset = self.declaration.phase_offered_waves[..session.phase]
            .iter()
            .sum::<usize>() as u64;
        for wave in &collected.waves {
            session.record(&completed_record(wave, session.phase, offset)?)?;
            session.written_ticket = wave.ticket;
        }
        *collected = Collected::default();
        drop(collected);
        let freeze = session.collector.freeze(now).map_err(error)?;
        session.write_raw(&freeze)?;
        session.phase += 1;
        if session.phase < 3 {
            let at = clock
                .now_ns()
                .filter(|v| *v >= now)
                .ok_or_else(|| error("phase clock moved backwards"))?;
            session.record(&StructuredServiceRecordV6::PhaseOpen {
                phase: phase(session.phase),
                opened_at_ns: at,
                fifo_cutoff: cutoff,
            })?;
            session.written_ticket = 0;
            let opening = state.opening.unwrap_or(policy.opening);
            let deadline = opening
                .monotonic_ns
                .checked_add(self.declaration.maximum_window_ns)
                .ok_or_else(|| error("window clock overflow"))?;
            let next = tickets::Window::new(
                state.generation,
                session.phase,
                self.declaration.phase_offered_waves[session.phase],
                deadline,
            );
            if let Some(worker) = self.notification.get() {
                next.attach_worker(worker.clone());
            }
            *self.active.write() = next;
            return Ok(None);
        }
        let closing = ExportClockReading::closing(clock).map_err(error)?;
        session.record(&StructuredServiceRecordV6::Footer {
            offered: session.collector.offered(),
            accepted_fifo_cutoff: session.collector.last_fifo(),
            closing: paired(closing),
            failure: None,
        })?;
        session.footer_written = true;
        let staged = session.source.take().expect("open source");
        let unpublished = staged.unpublished_receipt();
        let source = match staged.publish() {
            Ok(source) => source,
            Err(reason) => {
                self.remember_failed_source(
                    state.generation,
                    Some(unpublished),
                    true,
                    true,
                    &reason.to_string(),
                );
                return Err(error(reason));
            }
        };
        state.published_source = Some(source.clone());
        state.finished = true;
        let session = state.session.take().expect("sealed session");
        if session.collector.qualified_children() == 0 {
            return Err(error("complete population qualified no owners"));
        }
        let limits = profile::load_limits(&policy.import);
        let installed_at = ExportClockReading::closing(clock).map_err(error)?;
        let imported = session
            .collector
            .publish_same_process(
                &source.path,
                source.bytes,
                source.digest,
                &session.profile_path,
                paired(installed_at),
                &limits,
            )
            .map_err(error)?;
        let (children, receipt) = profile::EngineCostSnapshot::live_service_catalog_with_domain(
            Some(&session.profile_path),
            imported,
            self.workload_domain(),
        )?;
        state.install_pending = true;
        Ok(Some(Publication { children, receipt }))
    }

    pub(in crate::continuous_engine::inner::cost_observation) fn note_publication(
        &self,
        result: Result<bool, FerrumError>,
    ) {
        if let Some(automatic) = &self.automatic {
            automatic.note_publication(self, result);
            return;
        }
        match result {
            Ok(true) => {
                self.worker.lock().install_pending = false;
                self.qualified_publications.fetch_add(1, Ordering::Release);
                *self.publication_error.lock() = None;
            }
            Ok(false) => {}
            Err(e) => {
                let mut state = self.worker.lock();
                if state.install_pending {
                    // The source/profile were complete before runtime install
                    // failed. Count this generation once without relabeling
                    // its successfully retired physical window as incomplete.
                    let reason = format!("complete source; runtime installation failed: {e}");
                    state.install_pending = false;
                    state.failure = Some(reason.clone());
                    self.remember_failed_source(
                        state.generation,
                        state.published_source.take(),
                        true,
                        false,
                        &reason,
                    );
                    let _ = self.failed_generations.fetch_update(
                        Ordering::AcqRel,
                        Ordering::Acquire,
                        |n| n.checked_add(1),
                    );
                }
                drop(state);
                *self.publication_error.lock() = Some(e.to_string());
            }
        }
    }
}
