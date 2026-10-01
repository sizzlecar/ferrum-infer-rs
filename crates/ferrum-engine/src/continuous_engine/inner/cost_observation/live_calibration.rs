//! Live ordinary-wave population. This is separate from complete-request
//! CalibrationSession sources: private tickets precede preparation/outcomes.
use super::trainer::host_content::statistical::SettlementFailureDiagnostic;
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::{
    structured_v2::{StructuredInputV2, StructuredScopeV2, StructuredSettingsV2},
    ExecutionFingerprint,
};
use ferrum_types::{FerrumError, SloLiveStructuredCalibration};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, io::Read, path::PathBuf};
mod automatic;
pub(in crate::continuous_engine::inner) use automatic::StartupOwnerSeries;
pub(super) use automatic::{StartupCatalogActivation, StartupSeriesActivation};
pub(super) mod diagnostics;
mod discovery;
mod no_submission;
mod producer;
mod publication;
pub(super) use publication::Publication;
mod tickets;
pub(super) use no_submission::NoSubmissionReceipt;
pub(in crate::continuous_engine::inner) use no_submission::OriginalNoSubmissionReceipt;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    OnceLock,
};
pub(super) use tickets::Ticket;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LiveConsumption {
    Eligible,
    OutsideDeclaredRoute { observed_at_ns: u64 },
    NoSubmission { observed_at_ns: u64 },
    Failed,
}

const MAX_DECLARATION_BYTES: u64 = 1024 * 1024;

/// The complete offered population and owner predicates are fixed before any
/// capture. Actual counts per owner are derived only after whole-window closure.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Declaration {
    schema_version: u32,
    phase_offered_waves: [usize; 3],
    maximum_window_ns: u64,
    settings: StructuredSettingsV2,
    scopes: Vec<StructuredScopeV2>,
}
impl Declaration {
    fn validate(&self) -> Result<(), FerrumError> {
        if self.schema_version != 1
            || self.scopes.is_empty()
            || self.scopes.len() > 128
            || self.maximum_window_ns == 0
            || self.maximum_window_ns > self.settings.max_sample_age_ns
            || self
                .phase_offered_waves
                .iter()
                .any(|n| *n == 0 || *n > 65_536)
        {
            return Err(FerrumError::config(
                "invalid live-window declaration/capacity",
            ));
        }
        self.settings
            .validate()
            .map_err(|e| FerrumError::config(format!("live settings: {e:?}")))?;
        for (i, scope) in self.scopes.iter().enumerate() {
            scope
                .validate()
                .map_err(|e| FerrumError::config(format!("live scope: {e:?}")))?;
            if self.scopes[..i]
                .iter()
                .any(|prior| prior.owner == scope.owner)
            {
                return Err(FerrumError::config(
                    "live scopes require unique declared owners",
                ));
            }
        }
        Ok(())
    }
}

struct Frontier {
    generation: u64,
    generated: u64,
    maximum_output: u64,
    kv: u32,
    terminal: bool,
}
enum WaveEvidence {
    Settled(Arc<HostStageEvidenceV1>),
    NoSubmission(Arc<NoSubmissionReceipt>),
}
struct Wave {
    ticket: u64,
    fifo: u64,
    evidence: WaveEvidence,
}
#[derive(Default)]
struct Collected {
    waves: Vec<Wave>,
    retained_bytes: usize,
    frontiers: HashMap<(RequestId, u64), Frontier>,
    failure: Option<&'static str>,
    first_settlement_failure: Option<SettlementFailureDiagnostic>,
}

/// No execution/witness authority. Only the original CPU worker consumes it.
pub(super) struct LiveCalibration {
    declaration: Declaration,
    workload_domain: Option<CostWorkloadDomainV1>,
    declaration_sha256: [u8; 32],
    fingerprint: ExecutionFingerprint,
    active: RwLock<Arc<tickets::Window>>,
    collected: Mutex<Collected>,
    maximum_retained_bytes: usize,
    producer: producer::ProducerIdentityCache,
    publication: Option<publication::Policy>,
    automatic: Option<automatic::Controller>,
    worker: Mutex<publication::State>,
    notification: OnceLock<std::thread::Thread>,
    reuse: OnceLock<Arc<super::automatic_reuse::AutomaticReuse>>,
    qualified_publications: AtomicU64,
    failed_generations: AtomicU64,
    last_failed_source: Mutex<Option<publication::FailedSourceReceipt>>,
    publication_error: Mutex<Option<String>>,
}
#[derive(Debug, Clone, Serialize)]
pub(super) struct LiveAudit {
    pub protocol: &'static str,
    pub declaration_sha256: [u8; 32],
    pub population: tickets::WindowAudit,
    pub retained_waves: usize,
    pub retained_numeric_and_evidence_bytes: usize,
    pub failure: Option<&'static str>,
    pub first_settlement_failure: Option<SettlementFailureDiagnostic>,
    pub qualified_publications: u64,
    pub failed_generations: u64,
    pub last_failed_source: Option<publication::FailedSourceReceipt>,
    pub publication_error: Option<String>,
    pub automatic: Option<automatic::AutomaticAudit>,
}

impl LiveCalibration {
    /// The executable digest is bounded cold setup, not a feedback FIFO task.
    /// Failure remains attached to the optional collector's original identity.
    pub(super) fn prepare_producer_identity(&self) {
        self.producer.prepare();
    }

    pub(super) fn install_automatic_reuse(
        &self,
        reuse: Arc<super::automatic_reuse::AutomaticReuse>,
    ) {
        let _ = self.reuse.set(reuse);
    }
    pub(super) fn workload_domain(&self) -> Option<&CostWorkloadDomainV1> {
        self.workload_domain.as_ref()
    }
    #[cfg(test)]
    pub(super) fn fixture(
        owner: ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredOwnerKeyV2,
        offers: usize,
    ) -> Arc<Self> {
        use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredCoverageV2;
        Arc::new(Self {
            workload_domain: None,
            declaration: Declaration {
                schema_version: 1,
                phase_offered_waves: [offers; 3],
                maximum_window_ns: 1_000,
                settings: StructuredSettingsV2::default(),
                scopes: vec![StructuredScopeV2 {
                    numerical_family: None,
                    owner,
                    coverage: StructuredCoverageV2 {
                        pending_eligible_positions: vec![],
                        authorized_pending_constraints: vec![],
                        pending_counts: vec![0],
                        length_counts: vec![0, 1],
                        pending_positions: vec![],
                        length_positions: vec![0],
                        joint_counts: vec![(0, 0), (0, 1)],
                    },
                }],
            },
            declaration_sha256: [7; 32],
            fingerprint: ExecutionFingerprint {
                model_weights: [1; 32],
                numerical_policy: [2; 32],
                device_runtime: [3; 32],
                execution_config: [4; 32],
            },
            active: RwLock::new(tickets::Window::new(1, 0, offers, 1_000)),
            collected: Mutex::new(Collected::default()),
            maximum_retained_bytes: 1 << 24,
            producer: producer::ProducerIdentityCache::default(),
            publication: None,
            automatic: None,
            worker: Mutex::new(publication::State::default()),
            notification: OnceLock::new(),
            reuse: OnceLock::new(),
            qualified_publications: AtomicU64::new(0),
            failed_generations: AtomicU64::new(0),
            last_failed_source: Mutex::new(None),
            publication_error: Mutex::new(None),
        })
    }

    pub fn open(
        policy: &SloLiveStructuredCalibration,
        identity: &ExecutorCostIdentityAvailability,
        clock: &dyn CostObservationClock,
        import: &ferrum_types::SloCostProfileImportConfig,
    ) -> Result<Option<Arc<Self>>, FerrumError> {
        Self::open_with_domain(policy, identity, clock, import, None)
    }

    pub fn open_with_domain(
        policy: &SloLiveStructuredCalibration,
        identity: &ExecutorCostIdentityAvailability,
        clock: &dyn CostObservationClock,
        import: &ferrum_types::SloCostProfileImportConfig,
        workload_domain: Option<CostWorkloadDomainV1>,
    ) -> Result<Option<Arc<Self>>, FerrumError> {
        if let SloLiveStructuredCalibration::AutomaticV1 { settings } = policy {
            return Self::open_automatic_with_domain(settings, identity, import, workload_domain)
                .map(Some);
        }
        let SloLiveStructuredCalibration::ServiceWindowsV1 {
            declaration,
            evidence_directory,
            maximum_retained_numeric_bytes,
            maximum_generations,
            maximum_source_bytes,
        } = policy
        else {
            return Ok(None);
        };
        let ExecutorCostIdentityAvailability::Known(identity) = identity else {
            return Err(FerrumError::config(
                "live calibration requires actual executor identity",
            ));
        };
        if identity.schema_version != EXECUTOR_COST_IDENTITY_SCHEMA {
            return Err(FerrumError::config(
                "live calibration identity schema differs",
            ));
        }
        let mut bytes = Vec::new();
        std::fs::File::open(declaration)
            .map_err(|e| FerrumError::config(format!("open live declaration: {e}")))?
            .take(MAX_DECLARATION_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| FerrumError::config(format!("read live declaration: {e}")))?;
        if bytes.len() as u64 > MAX_DECLARATION_BYTES {
            return Err(FerrumError::config(
                "live declaration exceeds bounded metadata capacity",
            ));
        }
        let declaration: Declaration = serde_json::from_slice(&bytes)
            .map_err(|e| FerrumError::config(format!("decode live declaration: {e}")))?;
        declaration.validate()?;
        use sha2::Digest;
        let declaration_sha256 = sha2::Sha256::digest(&bytes).into();
        let opening = super::profile_export::ExportClockReading::opening(clock)
            .map_err(|e| FerrumError::config(e.to_string()))?;
        let now = opening.monotonic_ns;
        let deadline = now
            .checked_add(declaration.maximum_window_ns)
            .ok_or_else(|| FerrumError::config("live calibration clock overflow"))?;
        let active = tickets::Window::new(1, 0, declaration.phase_offered_waves[0], deadline);
        Ok(Some(Arc::new(Self {
            declaration,
            workload_domain: None,
            declaration_sha256,
            fingerprint: ExecutionFingerprint {
                model_weights: identity.model_weights,
                numerical_policy: identity.numerical_policy,
                device_runtime: identity.device_runtime,
                execution_config: identity.execution_config,
            },
            active: RwLock::new(active),
            collected: Mutex::new(Collected::default()),
            maximum_retained_bytes: maximum_retained_numeric_bytes.get(),
            producer: producer::ProducerIdentityCache::default(),
            publication: Some(publication::Policy {
                directory: evidence_directory.clone(),
                maximum_generations: maximum_generations.get(),
                maximum_source_bytes: maximum_source_bytes.get(),
                import: import.clone(),
                opening,
            }),
            automatic: None,
            worker: Mutex::new(publication::State::default()),
            notification: OnceLock::new(),
            reuse: OnceLock::new(),
            qualified_publications: AtomicU64::new(0),
            failed_generations: AtomicU64::new(0),
            last_failed_source: Mutex::new(None),
            publication_error: Mutex::new(None),
        })))
    }

    pub fn attach_worker(&self, worker: std::thread::Thread) {
        let _ = self.notification.set(worker.clone());
        self.active.read().attach_worker(worker);
    }

    fn route_population(&self) -> ferrum_types::SloCalibrationRoutePopulationV1 {
        self.automatic.as_ref().map_or(
            ferrum_types::SloCalibrationRoutePopulationV1::AllAttempts,
            |a| a.route_population(),
        )
    }

    pub fn reserve(&self, now: Option<u64>) -> Option<Ticket> {
        // Never wait for background IO/numerical work. The publication lock is
        // held only while exchanging a scalar window owner, never during fit.
        let window = self.active.try_read()?;
        if window.generation == 0 {
            return None;
        }
        match now {
            Some(now) => window
                .reserve(now)
                .map(|ticket| ticket.with_population(self.route_population())),
            None => {
                window.fail();
                None
            }
        }
    }

    /// Original private HostSettled validation is mandatory even for owners
    /// outside the declared numerical scopes. Normal EOS/Stop/Length are
    /// accepted here only with the original complete no-extra-work receipt.
    pub fn consume(&self, ticket: Ticket, fifo: u64, entry: &CostEvidenceEntry) -> LiveConsumption {
        self.consume_projected(ticket, fifo, entry, None)
    }

    pub(super) fn consume_resolved(
        &self,
        ticket: Ticket,
        fifo: u64,
        resolved: &super::resolved::ResolvedCostEntry,
    ) -> LiveConsumption {
        self.consume_projected(ticket, fifo, resolved.entry(), Some(resolved))
    }

    fn consume_projected(
        &self,
        ticket: Ticket,
        fifo: u64,
        entry: &CostEvidenceEntry,
        resolved: Option<&super::resolved::ResolvedCostEntry>,
    ) -> LiveConsumption {
        let stages = match entry {
            CostEvidenceEntry::Training { stages, .. } => stages.as_ref(),
            CostEvidenceEntry::StagesOnly { stages, .. } => Some(stages),
        };
        if stages.is_some_and(|stages| {
            stages
                .route_evidence
                .as_ref()
                .is_some_and(|r| r.is_outside())
        }) {
            return self.consume_outside(ticket, fifo, entry, stages.unwrap());
        }
        // Only the compatibility/test entrypoint builds its own projection.
        // The production worker lends its one immutable result to all consumers.
        let fallback = resolved.is_none().then(|| {
            super::resolved::ResolvedCostEntry::project_with_domain(
                entry,
                self.workload_domain.as_ref(),
            )
        });
        let projection = match resolved {
            Some(resolved) => resolved.structured(),
            None => fallback
                .as_ref()
                .expect("projection created")
                .as_ref()
                .map_err(|e| *e),
        };
        let mut settlement_error = None;
        let validated = (|| {
            let stages = stages.ok_or("missing_host_settlement")?;
            if !ticket.matches(stages.call_id, fifo, stages.prepare_started_at_ns) {
                return Err("ticket_call_fifo_or_clock_mismatch");
            }
            if !ticket.route_population().is_all_attempts()
                && stages
                    .route_evidence
                    .as_ref()
                    .is_none_or(|r| r.is_outside())
            {
                return Err("missing_private_eligible_route");
            }
            let projected = projection.map_err(|error| match error {
                super::resolved::ProjectionError::Settlement(error) => {
                    settlement_error = Some(error);
                    "incomplete_private_settlement"
                }
                super::resolved::ProjectionError::MissingRecipe => "missing_actual_recipe",
                super::resolved::ProjectionError::Numeric(_) => "invalid_actual_numeric_projection",
            })?;
            let actual = &projected.actual;
            if actual.fingerprint != self.fingerprint {
                return Err("execution_fingerprint_changed");
            }
            if self.workload_domain.is_some() {
                projected
                    .physical_scope
                    .map_err(|_| "invalid_actual_physical_domain")?;
            }
            let input = projected.query.input();
            let numeric_bytes = input
                .retained_numeric_bytes()
                .ok_or("retained_capacity_overflow")?;
            let metadata_bytes =
                retained_wave_metadata_bytes(&input).ok_or("retained_capacity_overflow")?;
            let bytes = entry
                .retained_rows()
                .and_then(|rows| entry.retained_bytes(rows))
                .and_then(|n| n.checked_add(metadata_bytes))
                .and_then(|n| n.checked_add(numeric_bytes))
                .ok_or("retained_capacity_overflow")?;
            Ok((Arc::clone(stages), input, bytes))
        })();
        let validated = validated.and_then(|(stages, input, bytes)| {
            if let Some(automatic) = &self.automatic {
                automatic.observe(&ticket, fifo, &input)?;
            }
            Ok((stages, input, bytes))
        });
        let mut collected = self.collected.lock();
        let result = validated.and_then(|(stages, input, bytes)| {
            let total = collected
                .retained_bytes
                .checked_add(bytes)
                .filter(|n| *n <= self.maximum_retained_bytes)
                .ok_or("retained_capacity_exceeded")?;
            validate_frontiers(&mut collected.frontiers, &stages, &input)?;
            collected.waves.push(Wave {
                ticket: ticket.ordinal(),
                fifo,
                evidence: WaveEvidence::Settled(stages),
            });
            collected.retained_bytes = total;
            Ok(())
        });
        match result {
            Ok(()) => {
                ticket.complete();
                LiveConsumption::Eligible
            }
            Err(reason) => {
                if collected.failure.is_none() {
                    collected.first_settlement_failure = settlement_error
                        .map(|error| SettlementFailureDiagnostic::capture(entry, fifo, error));
                }
                collected.failure.get_or_insert(reason);
                drop(ticket);
                LiveConsumption::Failed
            }
        }
    }

    fn consume_outside(
        &self,
        ticket: Ticket,
        fifo: u64,
        entry: &CostEvidenceEntry,
        stages: &Arc<HostStageEvidenceV1>,
    ) -> LiveConsumption {
        let result = (|| {
            if ticket.route_population().is_all_attempts()
                || !ticket.matches(stages.call_id, fifo, stages.prepare_started_at_ns)
            {
                return Err("outside_ticket_policy_call_fifo_or_clock");
            }
            let proof = stages
                .route_evidence
                .as_ref()
                .ok_or("outside_private_selector_missing")?;
            let raw = proof
                .outside_record(
                    stages,
                    ticket.ordinal(),
                    publication::phase(ticket.phase().min(2)),
                    fifo,
                )
                .map_err(|_| "outside_raw_encoding")?;
            let (_, observed) = raw.validate_settlement(
                &ferrum_scheduler::implementations::continuous::cost_profile::ProfileFingerprint::from(&self.fingerprint),
                stages.prepare_started_at_ns.ok_or("outside_prepare_clock")?,
            ).map_err(|_| "outside_incomplete_original_settlement")?;
            let bytes = entry
                .retained_rows()
                .and_then(|rows| entry.retained_bytes(rows))
                .and_then(|n| n.checked_add(retained_wave_metadata_rows(stages.rows.len())?))
                .ok_or("retained_capacity_overflow")?;
            self.automatic
                .as_ref()
                .ok_or("outside_automatic_population_missing")?
                .observe_outside(&ticket, fifo)?;
            Ok((observed, bytes))
        })();
        let mut collected = self.collected.lock();
        let result = result.and_then(|(observed, bytes)| {
            let total = collected
                .retained_bytes
                .checked_add(bytes)
                .filter(|n| *n <= self.maximum_retained_bytes)
                .ok_or("retained_capacity_exceeded")?;
            let proof = stages
                .route_evidence
                .as_ref()
                .ok_or("outside_private_selector_missing")?;
            validate_frontier_states(
                &mut collected.frontiers,
                stages,
                proof.frontier_states().map(|host| {
                    (
                        host.state.generated_tokens_before,
                        host.state.maximum_output_tokens,
                    )
                }),
            )?;
            collected.waves.push(Wave {
                ticket: ticket.ordinal(),
                fifo,
                evidence: WaveEvidence::Settled(Arc::clone(stages)),
            });
            collected.retained_bytes = total;
            Ok(observed)
        });
        match result {
            Ok(observed_at_ns) => {
                ticket.complete_outside();
                LiveConsumption::OutsideDeclaredRoute { observed_at_ns }
            }
            Err(reason) => {
                collected.failure.get_or_insert(reason);
                drop(ticket);
                LiveConsumption::Failed
            }
        }
    }

    pub fn audit(&self) -> LiveAudit {
        // The worker holds controller state before taking collected. Read it
        // first and release that lock before obtaining the collection audit.
        let automatic = self.automatic.as_ref().map(|value| value.audit());
        let collected = self.collected.lock();
        LiveAudit {
            protocol: if self
                .automatic
                .as_ref()
                .is_some_and(|c| c.uses_owner_blocks())
            {
                "ferrum.structured-owner-block.source7"
            } else {
                "ferrum.structured-service-window.source6"
            },
            declaration_sha256: automatic
                .as_ref()
                .and_then(|value| value.declaration_sha256)
                .unwrap_or(self.declaration_sha256),
            population: self.active.read().audit(),
            retained_waves: collected.waves.len(),
            retained_numeric_and_evidence_bytes: collected.retained_bytes,
            failure: collected.failure,
            first_settlement_failure: collected.first_settlement_failure,
            qualified_publications: self.qualified_publications.load(Ordering::Acquire),
            failed_generations: self.failed_generations.load(Ordering::Acquire),
            last_failed_source: self.last_failed_source.lock().clone(),
            publication_error: self.publication_error.lock().clone(),
            automatic,
        }
    }
    pub fn poll(&self, now: Option<u64>) {
        let window = self.active.read();
        if window.generation == 0 {
            return;
        }
        match now {
            Some(now) => {
                let _ = window.complete(now);
            }
            None => window.fail(),
        }
    }

    /// Source7 bounds original-record conversion separately from the configured
    /// generic observation batch. Source6 keeps its original batch semantics.
    pub(super) fn incremental_owner_records(&self) -> bool {
        self.automatic
            .as_ref()
            .is_some_and(|controller| controller.incremental_owner_records())
    }

    pub(super) fn ingest_owner_record(&self, cutoff: u64) {
        if let Some(controller) = &self.automatic {
            controller.ingest_owner_record(self, cutoff);
        }
    }

    pub(super) fn has_pending_owner_records(&self) -> bool {
        self.automatic
            .as_ref()
            .is_some_and(|controller| controller.has_pending_owner_records(self))
    }
}

fn validate_frontiers(
    frontiers: &mut HashMap<(RequestId, u64), Frontier>,
    stages: &HostStageEvidenceV1,
    input: &StructuredInputV2,
) -> Result<(), &'static str> {
    let numeric = stages
        .actual_shape
        .as_ref()
        .and_then(|s| s.numeric_features.as_ref())
        .ok_or("missing_numeric_frontier")?;
    if numeric.rows.len() != stages.rows.len()
        || input.physical_host_rows().len() != stages.rows.len()
    {
        return Err("frontier_row_count");
    }
    validate_frontier_states(
        frontiers,
        stages,
        numeric
            .rows
            .iter()
            .map(|state| (state.generated_tokens_before, state.maximum_output_tokens)),
    )
}

fn validate_frontier_states(
    frontiers: &mut HashMap<(RequestId, u64), Frontier>,
    stages: &HostStageEvidenceV1,
    mut states: impl Iterator<Item = (u64, u64)>,
) -> Result<(), &'static str> {
    for actual in &stages.rows {
        let (generated, maximum_output) = states.next().ok_or("frontier_row_count")?;
        let key = (actual.request_id.clone(), actual.owner_incarnation);
        let kv = match actual.actual_work {
            HostStageWork::Decode { kv_tokens } => kv_tokens,
            HostStageWork::Prefill { offset, .. } => offset,
            _ => return Err("unsupported_frontier_work"),
        };
        if let Some(prior) = frontiers.get(&key) {
            if prior.terminal
                || prior.generation != actual.work_generation
                || prior.generated != generated
                || prior.maximum_output != maximum_output
                || prior.kv != kv
            {
                return Err("within_window_frontier_discontinuity");
            }
        }
        let emits = match actual.actual_work {
            HostStageWork::Decode { .. } => true,
            HostStageWork::Prefill {
                offset,
                count,
                total_prompt_tokens,
            } => offset.checked_add(count) == Some(total_prompt_tokens),
            _ => return Err("unsupported_frontier_work"),
        };
        let next_kv = match actual.actual_work {
            HostStageWork::Decode { kv_tokens } => kv_tokens.checked_add(1),
            HostStageWork::Prefill { offset, count, .. } => offset.checked_add(count),
            _ => None,
        }
        .ok_or("frontier_overflow")?;
        frontiers.insert(
            key,
            Frontier {
                generation: actual
                    .work_generation
                    .checked_add(1)
                    .ok_or("frontier_overflow")?,
                generated: generated
                    .checked_add(u64::from(emits))
                    .ok_or("frontier_overflow")?,
                maximum_output: maximum_output,
                kv: next_kv,
                terminal: actual.terminal.is_some(),
            },
        );
    }
    if states.next().is_some() {
        return Err("frontier_row_count");
    }
    Ok(())
}

// Only the original stages and these container/frontier allocations survive
// consume. The temporary input is still charged separately for the complete
// window as conservative validation/replay headroom, even after it is dropped.
// The factor covers geometric container growth; original queue evidence keeps
// its existing capacity-based charge.
fn retained_wave_metadata_bytes(input: &StructuredInputV2) -> Option<usize> {
    retained_wave_metadata_rows(input.physical_host_rows().len())
}

fn retained_wave_metadata_rows(rows: usize) -> Option<usize> {
    let metadata =
        rows.checked_mul(4 * (std::mem::size_of::<((RequestId, u64), Frontier)>() + 1))?;
    // The hash table also reserves a control group beyond its bucket array.
    metadata
        .checked_add(16)?
        .checked_add(4 * std::mem::size_of::<Wave>())
}
