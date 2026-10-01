//! Worker-owned automatic lifecycle. Discovery and each numerical phase use
//! disjoint private ticket populations; startup probes have no live tickets.
use super::super::{profile, profile_export::ExportClockReading};
use super::*;
use ferrum_scheduler::implementations::continuous::cost_profile::ImportedStructuredModelV2;
use ferrum_types::SloAutomaticCalibrationSettingsV1;
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    sync::atomic::AtomicBool,
};
mod owner_blocks;
mod session;
mod shadow;
mod source;
mod startup_activation;
use session::Session;
pub(in crate::continuous_engine::inner) use startup_activation::StartupOwnerSeries;
pub(in crate::continuous_engine::inner::cost_observation) use startup_activation::{
    StartupCatalogActivation, StartupSeriesActivation,
};

pub(super) const DISCOVERY_PHASE: usize = 3;

fn error(value: impl std::fmt::Display) -> FerrumError {
    FerrumError::config(format!("automatic live calibration: {value}"))
}

pub(super) struct Controller {
    settings: SloAutomaticCalibrationSettingsV1,
    import: ferrum_types::SloCostProfileImportConfig,
    diagnostics: Option<diagnostics::Store>,
    diagnostic_failures: Mutex<source::DiagnosticFailureAudit>,
    requested: AtomicBool,
    stopped: AtomicBool,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    generation: u64,
    cold_algorithm_seed: Option<ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1>,
    discovery: Option<discovery::DiscoveryWindow>,
    pending_discovery: Option<shadow::ShadowDiscoverySeed>,
    discovery_origin: Option<shadow::ShadowDiscoveryOrigin>,
    session: Option<Session>,
    blocks: Option<owner_blocks::BlockSession>,
    rolling: Option<owner_blocks::rolling::Rolling>,
    declaration_sha256: Option<[u8; 32]>,
    opening_ns: u64,
    failure: Option<String>,
    finished: bool,
    install_pending: bool,
    startup_install_pending: bool,
    startup_publication: Option<Result<u64, String>>,
    startup_series: Option<Arc<startup_activation::StartupSeriesAuthority>>,
    startup_series_attempts: usize,
    origins: VecDeque<[u8; 32]>,
    pending_origins: Option<VecDeque<[u8; 32]>>,
    retained_payload_bytes: usize,
    pending_payload_bytes: Option<usize>,
    last_failure: Option<RetainedFailure>,
    last_diagnostic_publication: Option<DiagnosticPublication>,
    history: VecDeque<GenerationAudit>,
    phase_populations: [Option<tickets::WindowAudit>; 4],
    last_published_population: Option<GenerationAudit>,
}

struct RetainedFailure {
    window: Arc<tickets::Window>,
    collected: Collected,
}

#[derive(Debug, Clone, Serialize)]
struct GenerationAudit {
    generation: u64,
    discovery_origin: Option<shadow::ShadowDiscoveryOrigin>,
    failed: bool,
    installed: bool,
    reason: Option<String>,
    first_settlement_failure: Option<SettlementFailureDiagnostic>,
    population: tickets::WindowAudit,
    phase_populations: [Option<tickets::WindowAudit>; 4],
}

/// Historical disk publication, subject to the configured diagnostic archive
/// retention. Predictor provenance always remains process-local memory.
#[derive(Debug, Clone, Serialize)]
pub(in crate::continuous_engine::inner::cost_observation) struct DiagnosticPublication {
    pub generation: u64,
    pub source: super::super::profile_export::PublishedFile,
}

#[derive(Debug, Clone, Serialize)]
pub(in crate::continuous_engine::inner::cost_observation) struct AutomaticAudit {
    pub start_requested: bool,
    pub stopped: bool,
    pub generation: u64,
    pub discovery_offered: usize,
    pub discovery_origin: Option<shadow::ShadowDiscoveryOrigin>,
    pub declaration_sha256: Option<[u8; 32]>,
    pub retained_origins: usize,
    /// Conservative retained model payload, excluding allocator metadata/RSS.
    pub retained_payload_bytes_upper_bound: usize,
    pub retained_failed_wave_count: usize,
    pub retained_failed_bytes_upper_bound: usize,
    pub retained_failed_ticket_population: Option<tickets::WindowAudit>,
    pub retained_failed_reason: Option<&'static str>,
    pub retained_failed_settlement_failure: Option<SettlementFailureDiagnostic>,
    pub last_diagnostic_publication: Option<DiagnosticPublication>,
    pub diagnostic_failures: source::DiagnosticFailureAudit,
    pub owner_blocks: Option<
        ferrum_scheduler::implementations::continuous::cost_profile::StructuredServiceAuditV7,
    >,
    pub rolling_owner_blocks: Option<owner_blocks::rolling::RollingAudit>,
    pub startup_publication: Option<Result<u64, String>>,
    history: Vec<GenerationAudit>,
    last_published_population: Option<GenerationAudit>,
}

impl LiveCalibration {
    #[cfg(test)]
    pub(in crate::continuous_engine::inner::cost_observation) fn hold_automatic_close_for_test(
        &self,
    ) -> Arc<owner_blocks::close_task::TestGate> {
        self.automatic
            .as_ref()
            .unwrap()
            .state
            .lock()
            .rolling
            .as_mut()
            .unwrap()
            .hold_close()
    }
    /// A unique owned close is still running or awaiting collection. It must
    /// finish before shutdown may seal feedback or release source reservations.
    pub(in crate::continuous_engine::inner::cost_observation) fn automatic_computation_pending(
        &self,
    ) -> bool {
        self.automatic.as_ref().is_some_and(|controller| {
            controller
                .state
                .lock()
                .rolling
                .as_ref()
                .is_some_and(|rolling| rolling.computation_pending())
        })
    }
    pub(in crate::continuous_engine::inner::cost_observation) fn automatic_finish_step_limit(
        &self,
    ) -> usize {
        self.automatic
            .as_ref()
            .filter(|c| c.uses_rolling_owner_blocks())
            .map_or(2, |c| c.settings.maximum_retained_generations.get() + 3)
    }

    pub(in crate::continuous_engine::inner::cost_observation) fn automatic_finish_pending(
        &self,
    ) -> bool {
        self.automatic
            .as_ref()
            .filter(|c| c.uses_rolling_owner_blocks())
            .is_some_and(|c| {
                c.state
                    .lock()
                    .rolling
                    .as_ref()
                    .is_some_and(|r| r.finish_pending())
            })
    }

    /// Call only after publishing this exact catalog. Time-only expiry pruning
    /// releases the same capture slots as a normal successful replacement.
    pub(in crate::continuous_engine::inner::cost_observation) fn note_automatic_catalog_origins(
        &self,
        children: &[ImportedStructuredModelV2],
    ) -> Result<(), FerrumError> {
        let Some(controller) = &self.automatic else {
            return Ok(());
        };
        let mut state = controller.state.lock();
        let mut captures = BTreeMap::<[u8; 32], usize>::new();
        for child in children {
            let charge = child
                .retained_payload_bytes()
                .ok_or_else(|| error("published catalog retained size overflow"))?;
            let origin = child.provenance().capture_identity;
            let next = captures
                .get(&origin)
                .copied()
                .unwrap_or(0)
                .checked_add(charge)
                .ok_or_else(|| error("published origin retained size overflow"))?;
            captures.insert(origin, next);
        }
        let bytes = captures
            .values()
            .try_fold(0usize, |sum, bytes| sum.checked_add(*bytes))
            .ok_or_else(|| error("published catalog retained sum overflow"))?;
        state.origins.retain(|origin| captures.contains_key(origin));
        for origin in captures.keys() {
            if !state.origins.contains(origin) {
                state.origins.push_back(*origin);
            }
        }
        state.retained_payload_bytes = bytes;
        Ok(())
    }

    pub(super) fn open_automatic(
        settings: &SloAutomaticCalibrationSettingsV1,
        identity: &ExecutorCostIdentityAvailability,
        import: &ferrum_types::SloCostProfileImportConfig,
    ) -> Result<Arc<Self>, FerrumError> {
        Self::open_automatic_with_domain(settings, identity, import, None)
    }

    pub(super) fn open_automatic_with_domain(
        settings: &SloAutomaticCalibrationSettingsV1,
        identity: &ExecutorCostIdentityAvailability,
        import: &ferrum_types::SloCostProfileImportConfig,
        workload_domain: Option<CostWorkloadDomainV1>,
    ) -> Result<Arc<Self>, FerrumError> {
        settings.validate().map_err(error)?;
        let ExecutorCostIdentityAvailability::Known(identity) = identity else {
            return Err(error("actual executor identity is required"));
        };
        if identity.schema_version != EXECUTOR_COST_IDENTITY_SCHEMA {
            return Err(error("executor identity schema differs"));
        }
        let retained = settings.maximum_retained_generations.get();
        let generation_bytes = settings.maximum_retained_numeric_bytes.get();
        // K prior activated origins, one candidate collector, one current raw
        // phase, and one retained failed raw phase all fit the configured sum.
        let share = generation_bytes
            .checked_mul(retained)
            .and_then(|bytes| bytes.checked_div(retained.checked_add(3)?))
            .map(|bytes| bytes.min(generation_bytes / 2))
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| error("retained generation budget cannot hold bounded state"))?;
        let mut numerical = automatic_numerical_settings(settings);
        numerical.max_phase_samples = settings
            .phase_offered_waves
            .iter()
            .map(|n| n.get())
            .max()
            .unwrap();
        if settings.population_schedule.uses_owner_blocks() {
            let schedule = owner_blocks::schedule(settings, &numerical)?;
            numerical.max_phase_samples = *schedule.maximum_phase_members.iter().max().unwrap();
        }
        numerical
            .validate()
            .map_err(|e| error(format!("native numerical settings: {e:?}")))?;
        if settings.maximum_window_ns.get() > numerical.max_sample_age_ns {
            return Err(error(
                "collection window exceeds native numerical sample age",
            ));
        }
        let declaration = Declaration {
            schema_version: 1,
            phase_offered_waves: settings.phase_offered_waves.map(|n| n.get()),
            maximum_window_ns: settings.maximum_window_ns.get(),
            settings: numerical,
            scopes: Vec::new(), // Filled only after a complete independent discovery.
        };
        let active = tickets::Window::new(
            0,
            DISCOVERY_PHASE,
            settings.discovery_offered_waves.get(),
            0,
        );
        active.close();
        if active.retained_bytes().is_none_or(|bytes| bytes > share) {
            return Err(error(
                "raw-state budget cannot hold the discovery ticket ledger",
            ));
        }
        Ok(Arc::new(Self {
            declaration,
            workload_domain,
            declaration_sha256: [0; 32],
            fingerprint: ExecutionFingerprint {
                model_weights: identity.model_weights,
                numerical_policy: identity.numerical_policy,
                device_runtime: identity.device_runtime,
                execution_config: identity.execution_config,
            },
            active: RwLock::new(active),
            collected: Mutex::new(Collected::default()),
            maximum_retained_bytes: share,
            producer: producer::ProducerIdentityCache::default(),
            publication: None,
            automatic: Some(Controller {
                settings: settings.clone(),
                import: import.clone(),
                diagnostics: diagnostics::Store::new(&settings.diagnostics),
                diagnostic_failures: Mutex::new(Default::default()),
                requested: AtomicBool::new(false),
                stopped: AtomicBool::new(false),
                state: Mutex::new(State::default()),
            }),
            worker: Mutex::new(publication::State::default()),
            notification: OnceLock::new(),
            reuse: OnceLock::new(),
            qualified_publications: AtomicU64::new(0),
            failed_generations: AtomicU64::new(0),
            last_failed_source: Mutex::new(None),
            publication_error: Mutex::new(None),
        }))
    }

    pub(in crate::continuous_engine::inner::cost_observation) fn begin_automatic_calibration(
        &self,
    ) -> Result<(), FerrumError> {
        if let Some(controller) = &self.automatic {
            let state = controller.state.lock();
            if controller.stopped.load(Ordering::Acquire) {
                return Err(error("capture is already stopping"));
            }
            if state
                .startup_series
                .as_ref()
                .is_some_and(|series| !series.closed())
            {
                return Err(error("startup source series is still active"));
            }
            if state.startup_install_pending {
                return Err(error("startup catalog installation is still pending"));
            }
            controller.requested.store(true, Ordering::Release);
            drop(state);
            if let Some(worker) = self.notification.get() {
                worker.unpark();
            }
        }
        Ok(())
    }

    /// A private startup source is independent of the ordinary live epoch.
    /// Only the original training worker may reserve this installation after
    /// draining the source's exact accepted FIFO boundary.
    pub(in crate::continuous_engine::inner::cost_observation) fn begin_startup_catalog(
        &self,
        series: Option<&StartupSeriesActivation>,
    ) -> Result<(), FerrumError> {
        let controller = self
            .automatic
            .as_ref()
            .ok_or_else(|| error("startup catalog requires automatic calibration"))?;
        let mut state = controller.state.lock();
        let active = self.active.read();
        let population = active.audit();
        let in_series = match series {
            Some(permit) => {
                permit.validate()?;
                if state
                    .startup_series
                    .as_ref()
                    .is_none_or(|authority| !permit.matches(authority))
                    || permit.ordinal() != state.startup_series_attempts
                {
                    return Err(error(
                        "startup source series authorization or order differs",
                    ));
                }
                true
            }
            None => false,
        };
        if controller.requested.load(Ordering::Acquire)
            || controller.stopped.load(Ordering::Acquire)
            || state.generation != 0
            || state.install_pending
            || (!in_series
                && (state.startup_publication.is_some() || state.startup_series.is_some()))
            || population.issued != 0
            || population.retired != 0
        {
            return Err(error(
                "startup catalog requires an unused automatic lifecycle",
            ));
        }
        if in_series {
            state.startup_series_attempts += 1;
        }
        state.install_pending = true;
        state.startup_install_pending = true;
        Ok(())
    }

    pub(in crate::continuous_engine::inner::cost_observation) fn finish_startup_catalog(
        &self,
        result: &Result<u64, FerrumError>,
    ) {
        let controller = self
            .automatic
            .as_ref()
            .expect("startup automatic controller");
        let mut state = controller.state.lock();
        debug_assert!(state.startup_install_pending && state.install_pending);
        state.startup_install_pending = false;
        state.install_pending = false;
        match result {
            Ok(epoch) => {
                state.origins = state
                    .pending_origins
                    .take()
                    .expect("installed startup retention");
                state.retained_payload_bytes = state
                    .pending_payload_bytes
                    .take()
                    .expect("installed startup payload charge");
                state.startup_publication = Some(Ok(*epoch));
                let _ = self.qualified_publications.fetch_update(
                    Ordering::AcqRel,
                    Ordering::Acquire,
                    |n| n.checked_add(1),
                );
                *self.publication_error.lock() = None;
            }
            Err(reason) => {
                state.pending_origins = None;
                state.pending_payload_bytes = None;
                // A later independent source cannot erase an installed prefix.
                if !matches!(state.startup_publication, Some(Ok(_))) {
                    state.startup_publication = Some(Err(reason.to_string()));
                }
                *self.publication_error.lock() = Some(reason.to_string());
            }
        }
    }

    pub(in crate::continuous_engine::inner::cost_observation) fn prepare_retention(
        &self,
        previous: &[ImportedStructuredModelV2],
        next: &[ImportedStructuredModelV2],
    ) -> Result<Option<Vec<[u8; 32]>>, FerrumError> {
        self.prepare_retention_preserving(previous, next, &[])
    }

    /// One activation may retain an original child instead of a narrower
    /// replacement. Protect only those existing origins during this pass.
    pub(in crate::continuous_engine::inner::cost_observation) fn prepare_retention_preserving(
        &self,
        previous: &[ImportedStructuredModelV2],
        next: &[ImportedStructuredModelV2],
        protected_origins: &[[u8; 32]],
    ) -> Result<Option<Vec<[u8; 32]>>, FerrumError> {
        self.automatic
            .as_ref()
            .map(|controller| {
                controller.prepare_retention(
                    previous,
                    next,
                    self.maximum_retained_bytes,
                    protected_origins,
                )
            })
            .transpose()
    }

    pub(in crate::continuous_engine::inner::cost_observation) fn persist_automatic_reference(
        &self,
        source: &[u8],
        reference: &[u8],
    ) -> Result<(), FerrumError> {
        if let Some(store) = self
            .automatic
            .as_ref()
            .and_then(|controller| controller.diagnostics.as_ref())
        {
            if let Err(reason) = store.publish_reference(source, reference) {
                self.record_automatic_diagnostic_failure(
                    0,
                    source::DiagnosticFailureStage::Reference,
                    &reason,
                );
            }
        }
        Ok(())
    }
}

impl Controller {
    pub(super) fn uses_owner_blocks(&self) -> bool {
        self.settings.population_schedule.uses_owner_blocks()
    }
    pub(super) fn uses_rolling_owner_blocks(&self) -> bool {
        self.settings.population_schedule
            == ferrum_types::SloAutomaticCalibrationPopulationScheduleV1::OwnerBlocksRollingV2
    }
    pub(super) fn stop(&self) {
        self.stopped.store(true, Ordering::Release);
    }

    pub(super) fn route_population(&self) -> ferrum_types::SloCalibrationRoutePopulationV1 {
        self.settings.route_population
    }

    pub(super) fn observe_outside(&self, ticket: &Ticket, fifo: u64) -> Result<(), &'static str> {
        let mut state = self.state.lock();
        if self.uses_rolling_owner_blocks() {
            return state
                .rolling
                .as_ref()
                .ok_or("automatic_rolling_unavailable")?
                .validate_ticket(ticket);
        }
        if ticket.window.generation != state.generation {
            return Err("automatic_generation_mismatch");
        }
        if self.uses_owner_blocks() {
            return state
                .blocks
                .as_ref()
                .ok_or("automatic_block_unavailable")?
                .validate_ticket(ticket);
        }
        if ticket.phase() == DISCOVERY_PHASE {
            state
                .discovery
                .as_mut()
                .ok_or("discovery_window_unavailable")?
                .observe_outside(ticket.ordinal(), fifo)
                .map_err(|_| "discovery_population_failed")?;
        } else {
            let session = state.session.as_mut().ok_or("automatic_phase_mismatch")?;
            if session.phase != ticket.phase() {
                return Err("automatic_phase_mismatch");
            }
            session.observe_shadow(ticket.ordinal(), fifo, None);
        }
        Ok(())
    }

    pub(super) fn observe(
        &self,
        ticket: &Ticket,
        fifo: u64,
        input: &StructuredInputV2,
    ) -> Result<(), &'static str> {
        let mut state = self.state.lock();
        if self.uses_rolling_owner_blocks() {
            return state
                .rolling
                .as_ref()
                .ok_or("automatic_rolling_unavailable")?
                .validate_ticket(ticket);
        }
        if ticket.window.generation != state.generation {
            return Err("automatic_generation_mismatch");
        }
        if self.uses_owner_blocks() {
            return state
                .blocks
                .as_ref()
                .ok_or("automatic_block_unavailable")?
                .validate_ticket(ticket);
        }
        if ticket.phase() == DISCOVERY_PHASE {
            let discovery = state
                .discovery
                .as_mut()
                .ok_or("discovery_window_unavailable")?;
            discovery
                .observe(ticket.ordinal(), fifo, input)
                .map_err(|_| "discovery_population_failed")?;
        } else {
            let session = state.session.as_mut().ok_or("automatic_phase_mismatch")?;
            if session.phase != ticket.phase() {
                return Err("automatic_phase_mismatch");
            }
            session.observe_shadow(ticket.ordinal(), fifo, Some(input));
        }
        Ok(())
    }

    pub(super) fn audit(&self) -> AutomaticAudit {
        let state = self.state.lock();
        AutomaticAudit {
            start_requested: self.requested.load(Ordering::Acquire),
            stopped: self.stopped.load(Ordering::Acquire),
            generation: state.generation,
            discovery_offered: state.discovery.as_ref().map_or(0, |value| value.offered()),
            discovery_origin: state.discovery_origin.clone(),
            declaration_sha256: state.declaration_sha256,
            retained_origins: state.origins.len(),
            retained_payload_bytes_upper_bound: state.retained_payload_bytes,
            retained_failed_wave_count: state
                .last_failure
                .as_ref()
                .map_or(0, |value| value.collected.waves.len()),
            retained_failed_bytes_upper_bound: state
                .last_failure
                .as_ref()
                .map_or(0, |value| value.collected.retained_bytes),
            retained_failed_ticket_population: state
                .last_failure
                .as_ref()
                .map(|value| value.window.audit()),
            retained_failed_reason: state
                .last_failure
                .as_ref()
                .and_then(|value| value.collected.failure),
            retained_failed_settlement_failure: state
                .last_failure
                .as_ref()
                .and_then(|value| value.collected.first_settlement_failure),
            last_diagnostic_publication: state.last_diagnostic_publication.clone(),
            diagnostic_failures: self.diagnostic_failures.lock().clone(),
            owner_blocks: state.blocks.as_ref().map(|blocks| blocks.audit()),
            rolling_owner_blocks: state.rolling.as_ref().map(|rolling| rolling.audit()),
            startup_publication: state.startup_publication.clone(),
            history: state.history.iter().cloned().collect(),
            last_published_population: state.last_published_population.clone(),
        }
    }

    pub(super) fn advance(
        &self,
        live: &LiveCalibration,
        clock: &dyn CostObservationClock,
        cutoff: u64,
        stopping: bool,
    ) -> Result<Option<publication::Publication>, FerrumError> {
        if self.uses_owner_blocks() {
            return self.advance_owner_blocks(live, clock, cutoff, stopping);
        }
        if stopping {
            self.stop();
        }
        let mut state = self.state.lock();
        if !self.requested.load(Ordering::Acquire)
            || (self.stopped.load(Ordering::Acquire) && state.generation == 0)
        {
            return Ok(None);
        }
        let mut window = live.active.read().clone();
        if state.finished && self.stopped.load(Ordering::Acquire) {
            return Ok(None);
        }
        if state.generation == 0 || state.finished {
            if window.audit().issued != window.audit().retired {
                return Ok(None);
            }
            let seed = state.pending_discovery.take().filter(|seed| {
                clock.now_ns().is_some_and(|now| {
                    seed.current(now, cutoff, self.settings.maximum_window_ns.get())
                })
            });
            if let Some(seed) = seed {
                self.open_shadow_generation(live, &mut state, seed, clock, cutoff)?;
            } else {
                self.open_discovery(live, &mut state, clock, cutoff)?;
            }
            window = live.active.read().clone();
        }
        if self.stopped.load(Ordering::Acquire) {
            window.fail();
            state
                .failure
                .get_or_insert_with(|| "shutdown before automatic generation publication".into());
        }
        let complete = match clock.now_ns() {
            Some(now) => window.complete(now),
            None => {
                window.fail();
                state
                    .failure
                    .get_or_insert_with(|| "clock unavailable".into());
                false
            }
        };
        if window.audit().failed {
            state.failure.get_or_insert_with(|| {
                live.collected.lock().failure.map_or_else(
                    || "automatic offered population failed or expired".into(),
                    |reason| format!("automatic offered population failed: {reason}"),
                )
            });
        }
        if state.failure.is_some() {
            return self.fail_generation(live, &mut state, &window, clock, cutoff);
        }
        if !complete {
            return Ok(None);
        }
        let result = if window.phase == DISCOVERY_PHASE {
            self.freeze_discovery(live, &mut state, clock, cutoff)
                .map(|_| None)
        } else {
            self.advance_phase(live, &mut state, &window, clock, cutoff)
        };
        match result {
            Ok(value) => Ok(value),
            Err(reason) => {
                window.fail();
                state.failure.get_or_insert_with(|| reason.to_string());
                self.fail_generation(live, &mut state, &window, clock, cutoff)
            }
        }
    }

    fn open_discovery(
        &self,
        live: &LiveCalibration,
        state: &mut State,
        clock: &dyn CostObservationClock,
        cutoff: u64,
    ) -> Result<(), FerrumError> {
        let generation = state.generation.checked_add(1).ok_or_else(|| {
            self.stop();
            error("generation counter exhausted")
        })?;
        let at = ExportClockReading::opening(clock).map_err(error)?;
        let discovery = discovery::DiscoveryWindow::new(
            discovery::DiscoveryPolicy {
                offered_waves: self.settings.discovery_offered_waves.get(),
                maximum_owners: self.settings.maximum_owners.get(),
                maximum_retained_bytes: self.settings.maximum_discovery_bytes.get(),
            },
            cutoff,
        )
        .map_err(|e| error(format!("discovery: {e:?}")))?;
        open_window(
            live,
            generation,
            DISCOVERY_PHASE,
            self.settings.discovery_offered_waves.get(),
            at.monotonic_ns,
        )?;
        state.phase_populations = std::array::from_fn(|_| None);
        state.generation = generation;
        state.opening_ns = at.monotonic_ns;
        state.discovery = Some(discovery);
        state.pending_discovery = None;
        state.discovery_origin = None;
        state.session = None;
        state.declaration_sha256 = None;
        state.failure = None;
        state.finished = false;
        state.install_pending = false;
        state.pending_origins = None;
        state.pending_payload_bytes = None;
        Ok(())
    }

    fn discovery_policy(&self) -> discovery::DiscoveryPolicy {
        discovery::DiscoveryPolicy {
            offered_waves: self.settings.discovery_offered_waves.get(),
            maximum_owners: self.settings.maximum_owners.get(),
            maximum_retained_bytes: self.settings.maximum_discovery_bytes.get(),
        }
    }

    fn open_shadow_generation(
        &self,
        live: &LiveCalibration,
        state: &mut State,
        seed: shadow::ShadowDiscoverySeed,
        clock: &dyn CostObservationClock,
        cutoff: u64,
    ) -> Result<(), FerrumError> {
        let generation = state.generation.checked_add(1).ok_or_else(|| {
            self.stop();
            error("generation counter exhausted")
        })?;
        // The original phase is already fully retired. Only its frozen input
        // facts cross this boundary; all numerical state remains in the failed
        // generation. There is no synthesized successful discovery ticket set.
        self.open_numerical(live, state, seed.frozen, generation, clock, cutoff)?;
        state.generation = generation;
        state.phase_populations = std::array::from_fn(|_| None);
        state.discovery_origin = Some(seed.origin);
        state.discovery = None;
        state.failure = None;
        state.finished = false;
        state.install_pending = false;
        state.pending_origins = None;
        state.pending_payload_bytes = None;
        Ok(())
    }

    fn freeze_discovery(
        &self,
        live: &LiveCalibration,
        state: &mut State,
        clock: &dyn CostObservationClock,
        cutoff: u64,
    ) -> Result<(), FerrumError> {
        state.phase_populations[DISCOVERY_PHASE] = Some(live.active.read().audit());
        let collected = live.collected.lock();
        if collected.failure.is_some()
            || collected.waves.len() != self.settings.discovery_offered_waves.get()
        {
            return Err(error("independent discovery population is incomplete"));
        }
        drop(collected);
        let frozen = state
            .discovery
            .take()
            .ok_or_else(|| error("discovery already frozen"))?
            .freeze()
            .map_err(|e| error(format!("discovery freeze: {e:?}")))?;
        let generation = state.generation;
        self.open_numerical(live, state, frozen, generation, clock, cutoff)
    }

    fn open_numerical(
        &self,
        live: &LiveCalibration,
        state: &mut State,
        frozen: discovery::FrozenDiscovery,
        generation: u64,
        clock: &dyn CostObservationClock,
        cutoff: u64,
    ) -> Result<(), FerrumError> {
        if cutoff < frozen.fifo_bounds().1 {
            return Err(error("discovery FIFO barrier moved backwards"));
        }
        let mut declaration = live.declaration.clone();
        declaration.scopes = frozen.into_scopes();
        declaration.validate()?;
        use sha2::Digest;
        state.declaration_sha256 =
            Some(sha2::Sha256::digest(serde_json::to_vec(&declaration).map_err(error)?).into());
        let opening = ExportClockReading::opening(clock).map_err(error)?;
        let session = Session::open(
            live,
            declaration,
            generation,
            opening,
            cutoff,
            &self.import,
            self.diagnostics.as_ref(),
            self.discovery_policy(),
        )?;
        state.opening_ns = opening.monotonic_ns;
        state.session = Some(session);
        open_window(
            live,
            generation,
            0,
            live.declaration.phase_offered_waves[0],
            opening.monotonic_ns,
        )
    }

    fn advance_phase(
        &self,
        live: &LiveCalibration,
        state: &mut State,
        window: &Arc<tickets::Window>,
        clock: &dyn CostObservationClock,
        cutoff: u64,
    ) -> Result<Option<publication::Publication>, FerrumError> {
        let session = state
            .session
            .as_mut()
            .ok_or_else(|| error("numerical session missing"))?;
        state.phase_populations[window.phase] = Some(window.audit());
        match session.complete_phase(live, window, clock, cutoff)? {
            session::PhaseCompletion::Continue => {}
            session::PhaseCompletion::AllChildrenFailed(seed) => {
                state.pending_discovery = seed;
                return Err(error("all frozen numerical children failed"));
            }
        }
        if session.phase < 3 {
            open_window(
                live,
                state.generation,
                session.phase,
                live.declaration.phase_offered_waves[session.phase],
                state.opening_ns,
            )?;
            return Ok(None);
        }
        let (publication, diagnostic) =
            state
                .session
                .take()
                .expect("completed session")
                .activate(live, clock, &self.import)?;
        if let Some(source) = diagnostic {
            state.last_diagnostic_publication = Some(DiagnosticPublication {
                generation: state.generation,
                source,
            });
        }
        state.install_pending = true;
        state.finished = true;
        self.record_history(
            state,
            GenerationAudit {
                generation: state.generation,
                discovery_origin: state.discovery_origin.clone(),
                failed: false,
                installed: false,
                reason: None,
                first_settlement_failure: None,
                population: live.active.read().audit(),
                phase_populations: state.phase_populations.clone(),
            },
        );
        Ok(Some(publication))
    }

    fn fail_generation(
        &self,
        live: &LiveCalibration,
        state: &mut State,
        window: &Arc<tickets::Window>,
        clock: &dyn CostObservationClock,
        cutoff: u64,
    ) -> Result<Option<publication::Publication>, FerrumError> {
        window.fail();
        let audit = window.audit();
        state.phase_populations[window.phase] = Some(audit.clone());
        if audit.issued != audit.retired {
            return Ok(None);
        }
        let reason = state
            .failure
            .clone()
            .unwrap_or_else(|| "automatic population failed".into());
        if let Some(session) = &mut state.session {
            session.persist_failure(live, window, clock, cutoff, &reason);
        } else if window.phase == DISCOVERY_PHASE {
            if let Some(store) = &self.diagnostics {
                if let Err(write) = source::failed_discovery(store, live, window, cutoff, &reason) {
                    live.record_automatic_diagnostic_failure(
                        state.generation,
                        source::DiagnosticFailureStage::FailedPopulation,
                        &write,
                    );
                    live.remember_failed_source(
                        state.generation,
                        None,
                        false,
                        false,
                        &format!("{reason}; failed discovery evidence: {write}"),
                    );
                }
            }
        }
        state.last_failure = Some(RetainedFailure {
            window: window.clone(),
            collected: std::mem::take(&mut *live.collected.lock()),
        });
        state.session = None;
        state.discovery = None;
        state.finished = true;
        state.install_pending = false;
        self.record_history(
            state,
            GenerationAudit {
                generation: state.generation,
                discovery_origin: state.discovery_origin.clone(),
                failed: true,
                installed: false,
                reason: Some(reason.chars().take(4096).collect()),
                first_settlement_failure: state
                    .last_failure
                    .as_ref()
                    .and_then(|value| value.collected.first_settlement_failure),
                population: audit,
                phase_populations: state.phase_populations.clone(),
            },
        );
        let _ = live
            .failed_generations
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1));
        live.remember_failed_source(state.generation, None, false, false, &reason);
        Err(error(reason))
    }

    fn record_history(&self, state: &mut State, audit: GenerationAudit) {
        state.history.push_back(audit);
        while state.history.len() > self.settings.maximum_retained_generations.get() {
            state.history.pop_front();
        }
    }

    fn prepare_retention(
        &self,
        previous: &[ImportedStructuredModelV2],
        next: &[ImportedStructuredModelV2],
        per_origin_bytes: usize,
        protected_origins: &[[u8; 32]],
    ) -> Result<Vec<[u8; 32]>, FerrumError> {
        let mut state = self.state.lock();
        if !state.install_pending {
            return Err(error("retention prepared outside pending publication"));
        }
        // Charge the complete candidate after replacing the same owners. An
        // encoded source length is never a proxy for retained model payload.
        let preserved: Vec<_> = previous
            .iter()
            .filter(|old| !next.iter().any(|new| old.same_population(new)))
            .collect();
        if protected_origins.len() > 128
            || protected_origins.iter().any(|origin| {
                !preserved
                    .iter()
                    .any(|child| child.provenance().capture_identity == *origin)
            })
        {
            return Err(error(
                "retention protection lacks an original preserved child",
            ));
        }
        let mut charges = BTreeMap::<[u8; 32], Option<usize>>::new();
        let mut owner_counts = BTreeMap::<[u8; 32], usize>::new();
        for child in preserved.iter().copied().chain(next.iter()) {
            let owners = owner_counts
                .entry(child.provenance().capture_identity)
                .or_default();
            *owners = owners
                .checked_add(1)
                .ok_or_else(|| error("retained owner count overflow"))?;
            let charge = charges
                .entry(child.provenance().capture_identity)
                .or_insert(Some(0));
            *charge =
                (*charge).and_then(|bytes| bytes.checked_add(child.retained_payload_bytes()?));
        }
        let fresh: BTreeSet<_> = next
            .iter()
            .map(|child| child.provenance().capture_identity)
            .collect();
        let mut origins = state.origins.clone();
        let mut initial: Vec<_> = preserved
            .iter()
            .map(|child| {
                (
                    child.provenance().loaded_unix_ns,
                    child.provenance().capture_identity,
                )
            })
            .collect();
        initial.sort_unstable();
        for (_, origin) in initial {
            if !origins.contains(&origin) {
                origins.push_back(origin);
            }
        }
        origins.retain(|origin| charges.contains_key(origin));
        for child in next {
            let origin = child.provenance().capture_identity;
            origins.retain(|old| *old != origin);
            origins.push_back(origin);
        }
        if let Some(rolling) = &state.rolling {
            rolling.check_fresh_catalog(&fresh, &charges)?;
        }
        let (origins, bytes) = retained_origins_within_budget(
            origins,
            &fresh,
            &charges,
            self.settings.maximum_retained_generations.get(),
            per_origin_bytes,
            &owner_counts,
            self.settings.maximum_owners.get(),
            protected_origins,
        )?;
        let allowed = origins.iter().copied().collect();
        if let Some(rolling) = &state.rolling {
            rolling.check_catalog_origins(&origins)?;
        }
        state.pending_origins = Some(origins);
        state.pending_payload_bytes = Some(bytes);
        Ok(allowed)
    }

    pub(super) fn note_publication(
        &self,
        live: &LiveCalibration,
        result: Result<bool, FerrumError>,
    ) {
        let mut state = self.state.lock();
        match result {
            Ok(true) => {
                if let Some(origins) = state.pending_origins.take() {
                    state.origins = origins;
                    state.retained_payload_bytes = state
                        .pending_payload_bytes
                        .take()
                        .expect("pending origins include payload charge");
                }
                state.install_pending = false;
                if let Some(rolling) = &mut state.rolling {
                    rolling.note_publication(None);
                }
                if let Some(last) = state.history.back_mut() {
                    last.installed = true;
                }
                state.last_published_population = state.history.back().cloned();
                let _ = live.qualified_publications.fetch_update(
                    Ordering::AcqRel,
                    Ordering::Acquire,
                    |n| n.checked_add(1),
                );
                *live.publication_error.lock() = None;
            }
            Ok(false) => {}
            Err(reason) => {
                if state.install_pending {
                    if self.uses_rolling_owner_blocks() {
                        state.install_pending = false;
                        state.pending_origins = None;
                        state.pending_payload_bytes = None;
                        if let Some(rolling) = &mut state.rolling {
                            rolling.note_publication(Some(reason.to_string()));
                        }
                        *live.publication_error.lock() = Some(reason.to_string());
                        return;
                    }
                    if self.uses_owner_blocks() {
                        // Keep the complete original block and journal until
                        // the worker can seal the failed epoch with its clock.
                        state.install_pending = false;
                        state.pending_origins = None;
                        state.pending_payload_bytes = None;
                        state.failure = Some(format!("runtime installation failed: {reason}"));
                        live.active.read().fail();
                        *live.publication_error.lock() = Some(reason.to_string());
                        return;
                    }
                    state.install_pending = false;
                    state.pending_origins = None;
                    state.pending_payload_bytes = None;
                    if let Some(last) = state.history.back_mut() {
                        last.failed = true;
                        last.reason = Some(reason.to_string().chars().take(4096).collect());
                    }
                    let _ = live.failed_generations.fetch_update(
                        Ordering::AcqRel,
                        Ordering::Acquire,
                        |n| n.checked_add(1),
                    );
                    state.last_failure = Some(RetainedFailure {
                        window: live.active.read().clone(),
                        collected: std::mem::take(&mut *live.collected.lock()),
                    });
                    live.remember_failed_source(
                        state.generation,
                        state
                            .last_diagnostic_publication
                            .as_ref()
                            .filter(|source| source.generation == state.generation)
                            .map(|source| source.source.clone()),
                        true,
                        false,
                        &reason.to_string(),
                    );
                }
                *live.publication_error.lock() = Some(reason.to_string());
            }
        }
    }
}

fn retained_origins_within_budget(
    mut origins: VecDeque<[u8; 32]>,
    fresh: &BTreeSet<[u8; 32]>,
    charges: &BTreeMap<[u8; 32], Option<usize>>,
    retained: usize,
    per_origin_bytes: usize,
    owner_counts: &BTreeMap<[u8; 32], usize>,
    maximum_owners: usize,
    protected_origins: &[[u8; 32]],
) -> Result<(VecDeque<[u8; 32]>, usize), FerrumError> {
    let maximum = retained
        .checked_mul(per_origin_bytes)
        .ok_or_else(|| error("retained model payload capacity overflow"))?;
    let fresh_bytes = fresh.iter().try_fold(0usize, |sum, origin| {
        sum.checked_add(charges.get(origin).copied().flatten()?)
    });
    let fresh_owners = fresh.iter().try_fold(0usize, |sum, origin| {
        sum.checked_add(*owner_counts.get(origin)?)
    });
    if fresh.is_empty()
        || fresh.len() > retained
        || fresh_bytes.is_none_or(|bytes| bytes > per_origin_bytes)
        || fresh_owners.is_none_or(|owners| owners > maximum_owners)
    {
        return Err(error(
            "fresh generation model payload exceeds its retained budget",
        ));
    }
    loop {
        let total = origins.iter().try_fold(0usize, |sum, origin| {
            let bytes = charges
                .get(origin)
                .copied()
                .flatten()
                .filter(|bytes| *bytes <= per_origin_bytes)?;
            sum.checked_add(bytes)
        });
        let owners = origins.iter().try_fold(0usize, |sum, origin| {
            sum.checked_add(*owner_counts.get(origin)?)
        });
        if origins.len() <= retained && owners.is_some_and(|owners| owners <= maximum_owners) {
            if let Some(bytes) = total.filter(|bytes| *bytes <= maximum) {
                return Ok((origins, bytes));
            }
        }
        let index = origins
            .iter()
            .position(|origin| !fresh.contains(origin) && !protected_origins.contains(origin))
            .ok_or_else(|| error("fresh generation cannot fit the retained model budget"))?;
        origins.remove(index);
    }
}

#[cfg(test)]
mod retention_tests;

fn open_window(
    live: &LiveCalibration,
    generation: u64,
    phase: usize,
    offers: usize,
    opening_ns: u64,
) -> Result<(), FerrumError> {
    let deadline = opening_ns
        .checked_add(live.declaration.maximum_window_ns)
        .ok_or_else(|| error("window deadline overflow"))?;
    let next = tickets::Window::new(generation, phase, offers, deadline);
    let ledger = next
        .retained_bytes()
        .filter(|bytes| *bytes <= live.maximum_retained_bytes)
        .ok_or_else(|| error("raw-state budget cannot hold the private ticket ledger"))?;
    if let Some(worker) = live.notification.get() {
        next.attach_worker(worker.clone());
    }
    *live.collected.lock() = Collected {
        retained_bytes: ledger,
        ..Default::default()
    };
    *live.active.write() = next;
    Ok(())
}
