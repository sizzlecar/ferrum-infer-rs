//! Bounded, same-attempt receipts using existing host measurements.

use std::sync::OnceLock;

use super::*;

const MAXIMUM_WAVES: usize = 4_096;
const MAXIMUM_ATTEMPTS: usize = MAX_DEFINITELY_NOT_SUBMITTED_RETRIES as usize + 1;
const GAP_REASONS: [DeviceReusableExecutionProgramGapReason; 11] = [
    DeviceReusableExecutionProgramGapReason::MissingComputeCommand,
    DeviceReusableExecutionProgramGapReason::ProviderReplayKeyMissing,
    DeviceReusableExecutionProgramGapReason::ReusableAddressScopeMissing,
    DeviceReusableExecutionProgramGapReason::ReusableAddressScopeConflict,
    DeviceReusableExecutionProgramGapReason::CaptureRejected,
    DeviceReusableExecutionProgramGapReason::CachedCaptureRejected,
    DeviceReusableExecutionProgramGapReason::WarmupRequired,
    DeviceReusableExecutionProgramGapReason::QuiescenceDeferred,
    DeviceReusableExecutionProgramGapReason::CapacityDeferred,
    DeviceReusableExecutionProgramGapReason::Evicted,
    DeviceReusableExecutionProgramGapReason::OutsidePreparation,
];

fn gap_index(reason: DeviceReusableExecutionProgramGapReason) -> usize {
    match reason {
        DeviceReusableExecutionProgramGapReason::MissingComputeCommand => 0,
        DeviceReusableExecutionProgramGapReason::ProviderReplayKeyMissing => 1,
        DeviceReusableExecutionProgramGapReason::ReusableAddressScopeMissing => 2,
        DeviceReusableExecutionProgramGapReason::ReusableAddressScopeConflict => 3,
        DeviceReusableExecutionProgramGapReason::CaptureRejected => 4,
        DeviceReusableExecutionProgramGapReason::CachedCaptureRejected => 5,
        DeviceReusableExecutionProgramGapReason::WarmupRequired => 6,
        DeviceReusableExecutionProgramGapReason::QuiescenceDeferred => 7,
        DeviceReusableExecutionProgramGapReason::CapacityDeferred => 8,
        DeviceReusableExecutionProgramGapReason::Evicted => 9,
        DeviceReusableExecutionProgramGapReason::OutsidePreparation => 10,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Disposition {
    Hit,
    NoResidentProgram,
    NoCachedRecipe,
    ImmutableCapabilityUnavailable,
    UnsupportedEncoder,
    BeforeDecision,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Outcome {
    Submitted,
    DefinitelyNotSubmitted,
    ContractError,
    ProviderError,
    InitializationError,
    InputUploadError,
    SubmissionIndeterminate,
    PostSubmitContract,
    QuiescentFailure,
}

impl Outcome {
    pub(super) fn from_dispatch<R: DeviceRuntime>(outcome: &DispatchOutcome<R>) -> Self {
        match outcome {
            DispatchOutcome::Submitted { .. } => Self::Submitted,
            DispatchOutcome::QuiescentFailure(_) => Self::QuiescentFailure,
            DispatchOutcome::SubmissionIndeterminate { .. } => Self::SubmissionIndeterminate,
            DispatchOutcome::PostSubmitContract { .. } => Self::PostSubmitContract,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum BackendIdentity {
    Exact,
    BothUnavailable,
    Mismatch,
}

fn identity_match(
    expected: Option<&DeviceReusableExecutionProgramId>,
    actual: Option<&DeviceReusableExecutionProgramId>,
) -> BackendIdentity {
    match (expected, actual) {
        (Some(expected), Some(actual)) if expected == actual => BackendIdentity::Exact,
        (None, None) => BackendIdentity::BothUnavailable,
        _ => BackendIdentity::Mismatch,
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
struct BackendPreparation {
    identity: BackendIdentity,
    capture_allowed_at_entry: bool,
    segments: DeviceReusableExecutionObservation,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct ProgramGaps {
    identity: BackendIdentity,
    node_counts_by_reason: [u64; GAP_REASONS.len()],
}

#[derive(Serialize)]
struct AttemptRecord {
    ordinal: usize,
    program_id: Option<DeviceReusableExecutionProgramId>,
    lane_epoch: u64,
    catalog_epoch: Option<u64>,
    catalog_miss: Option<VNextReusableExecutionCatalogMissReason>,
    disposition: Disposition,
    incomplete_declarations: bool,
    core_preparation: InvocationPreparationStats,
    outcome: Outcome,
    provider_encode_submit_ns: Option<u64>,
    provider_node_encode_ns: Option<u64>,
    enqueue_commands_ns: Option<u64>,
    backend_preparation: Option<BackendPreparation>,
    program_gaps: Option<ProgramGaps>,
    backend_submission_observation: Option<DeviceReusableExecutionObservation>,
    repeated_callback: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum PreparationPath {
    Ordinary,
    Bootstrap,
    Resident,
}

/// Basic metrics-only observation; owns no execution authority or request data.
#[derive(Default, Serialize)]
pub(super) struct DecodePreparationObservation {
    prepare_excluding_bootstrap_ns: Option<u64>,
    pub(super) bootstrap_ns: Option<u64>,
    pub(super) bootstrap_prepared: Option<bool>,
    prepared_path: Option<PreparationPath>,
    readback_mode: Option<&'static str>,
    greedy_participants: Option<usize>,
    full_participants: Option<usize>,
    pub(super) final_dispatch_path: Option<PreparationPath>,
    pub(super) resident_unavailable_fallbacks: u64,
    pub(super) fallback_prepare_ns: Option<u64>,
    cold_compile_ns: Option<u64>,
    cold_compile_attempts: u64,
    cold_compile_candidates: u64,
    cold_compile_errors: u64,
    resident_submitted: bool,
}

impl DecodePreparationObservation {
    pub(super) fn finish_prepare(&mut self, elapsed: Duration, path: PreparationPath) {
        self.prepare_excluding_bootstrap_ns =
            duration_ns(elapsed).checked_sub(self.bootstrap_ns.unwrap_or(0));
        self.prepared_path = Some(path);
    }

    pub(super) fn resolve_readback(&mut self, modes: &ProductReadbackModes) {
        self.readback_mode = Some(match modes.uniform() {
            Some(VNextProductOutputMode::GreedyToken) => "uniform_greedy",
            Some(VNextProductOutputMode::FullLogits) => "uniform_full",
            None => "participant",
        });
        self.full_participants = Some(modes.full_participants());
        self.greedy_participants = Some(modes.len() - modes.full_participants());
    }

    pub(super) fn record_compile(&mut self, elapsed: Duration, candidate: bool, error: bool) {
        self.cold_compile_ns = Some(
            self.cold_compile_ns
                .unwrap_or(0)
                .saturating_add(duration_ns(elapsed)),
        );
        self.cold_compile_attempts += 1;
        self.cold_compile_candidates += u64::from(candidate);
        self.cold_compile_errors += u64::from(error);
    }
}

#[derive(Serialize)]
struct WaveRecord {
    ordinal: u64,
    phase: &'static str,
    outcome: Outcome,
    host_encode_submit_ns: Option<u64>,
    attempts: Vec<AttemptRecord>,
    attempt_overflow: usize,
    decode_preparation: Option<DecodePreparationObservation>,
}

pub(super) struct Ledger {
    maximum_waves: usize,
    next_wave: AtomicU64,
    overflow: AtomicU64,
    unfinalized: AtomicU64,
    records: Mutex<Vec<WaveRecord>>,
}

impl Default for Ledger {
    fn default() -> Self {
        Self::with_capacity(MAXIMUM_WAVES)
    }
}

impl Ledger {
    pub(super) fn with_capacity(maximum_waves: usize) -> Self {
        Self {
            maximum_waves,
            next_wave: AtomicU64::new(0),
            overflow: AtomicU64::new(0),
            unfinalized: AtomicU64::new(0),
            records: Mutex::new(Vec::new()),
        }
    }

    pub(super) fn begin(
        &self,
        enabled: bool,
        kind: VNextExecutionWaveKind,
    ) -> Option<WaveObservation<'_>> {
        if !enabled {
            return None;
        }
        let ordinal = self
            .next_wave
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                Some(value.saturating_add(1))
            })
            .ok()?;
        if ordinal >= self.maximum_waves as u64 {
            self.overflow.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        Some(WaveObservation {
            ledger: self,
            ordinal,
            phase: match kind {
                VNextExecutionWaveKind::Prefill => "prefill",
                VNextExecutionWaveKind::Decode => "decode",
                VNextExecutionWaveKind::Mixed => "mixed",
            },
            attempts: Vec::with_capacity(MAXIMUM_ATTEMPTS),
            attempt_overflow: 0,
            decode_preparation: None,
            finished: false,
        })
    }

    pub(super) fn reset(&self) {
        // The existing startup reset runs only after preparation is quiescent.
        self.records.lock().clear();
        self.next_wave.store(0, Ordering::Relaxed);
        self.overflow.store(0, Ordering::Relaxed);
        self.unfinalized.store(0, Ordering::Relaxed);
    }

    pub(super) fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "collection": "host_timing_enabled_first_dispatch_waves_after_startup_reset",
            "maximum_waves": self.maximum_waves,
            "maximum_attempts_per_wave": MAXIMUM_ATTEMPTS,
            "waves_started": self.next_wave.load(Ordering::Relaxed),
            "overflow_waves": self.overflow.load(Ordering::Relaxed),
            "unfinalized_waves": self.unfinalized.load(Ordering::Relaxed),
            "gap_reason_order": GAP_REASONS.map(|reason| reason.as_str()),
            "records": &*self.records.lock(),
            "scope": [
                "submitted means dispatch returned a handle, not terminal device success",
                "host_encode_submit covers the final outer dispatch iteration including its internal retries; attempt intervals are nested and must not be added to it",
                "provider_encode_submit includes core dispatch; provider_node_encode and enqueue_commands reuse their original stage scopes",
                "missing callback or interval is null, not zero; prepare and final submission observations are separate boundaries",
                "program_gaps counts actual node gaps by reason before on-demand tombstone removal; segment counters have a different denominator",
                "program_id retains every field; exact backend identity comparison is diagnostic and never changes dispatch",
                "capture_allowed_at_entry is the backend quiescence decision, not proof of later completion",
                "first-wave prefix only; overflow and unfinalized waves cannot be used as complete cold-path history",
                "unwinding preserves original RAII metrics but does not finalize joint rows; snapshots during execution are not atomic",
                "attempt observation work is inside host_encode_submit; outer ledger reservation and final serialization are outside that interval",
                "decode_preparation is Basic metrics-only for successfully prepared Decode calls; earlier deferrals/errors have no joined row; null is missing or not entered, not zero",
                "prepare_excluding_bootstrap and bootstrap are disjoint before dispatch; fallback_prepare and cold_compile are later intervals nested in host dispatch and must not be added to host_encode_submit",
                "decode preparation times are host elapsed including instrumentation, locks and synchronization, not pure CPU or removable time; cold_compile precedes provider_encode_submit",
                "readback mode is the actual resolver result; participant counts are logical rows, not physical GPU width; compile candidates are not installations and submitted is not terminal completion",
                "after resident unavailable fallback the legacy host_encode_submit row covers the final dispatch iteration only; aggregate timing includes earlier iterations; no cohort-causal attribution is inferred"
            ]
        })
    }
}

pub(super) struct WaveObservation<'a> {
    ledger: &'a Ledger,
    ordinal: u64,
    phase: &'static str,
    attempts: Vec<AttemptRecord>,
    attempt_overflow: usize,
    pub(super) decode_preparation: Option<DecodePreparationObservation>,
    finished: bool,
}

impl WaveObservation<'_> {
    pub(super) fn record_attempt<T>(
        &mut self,
        observation: AttemptObservation<'_, T>,
        stats: InvocationPreparationStats,
        outcome: Outcome,
        provider_encode_submit: Option<Duration>,
    ) {
        if self.attempts.len() >= MAXIMUM_ATTEMPTS {
            self.attempt_overflow += 1;
            return;
        }
        self.attempts.push(observation.finish(
            self.attempts.len(),
            stats,
            outcome,
            provider_encode_submit,
        ));
    }

    pub(super) fn finish(mut self, host_elapsed: Option<Duration>, outcome: Outcome) {
        if let Some(preparation) = self.decode_preparation.as_mut() {
            preparation.resident_submitted = outcome == Outcome::Submitted
                && preparation.final_dispatch_path == Some(PreparationPath::Resident);
        }
        let record = WaveRecord {
            ordinal: self.ordinal,
            phase: self.phase,
            outcome,
            host_encode_submit_ns: host_elapsed.map(duration_ns),
            attempts: std::mem::take(&mut self.attempts),
            attempt_overflow: self.attempt_overflow,
            decode_preparation: self.decode_preparation.take(),
        };
        self.ledger.records.lock().push(record);
        self.finished = true;
    }
}

impl Drop for WaveObservation<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.ledger.unfinalized.fetch_add(1, Ordering::Relaxed);
        }
    }
}

pub(super) fn duration_ns(elapsed: Duration) -> u64 {
    elapsed.as_nanos().min(u128::from(u64::MAX)) as u64
}

fn disposition(stats: InvocationPreparationStats, outcome: Outcome) -> Disposition {
    let mut primary = [
        (stats.segment_hits, Disposition::Hit),
        (
            stats.segment_no_resident_program,
            Disposition::NoResidentProgram,
        ),
        (stats.segment_no_cached_recipe, Disposition::NoCachedRecipe),
        (
            stats.segment_immutable_capability_unavailable,
            Disposition::ImmutableCapabilityUnavailable,
        ),
        (
            stats.segment_unsupported_encoder,
            Disposition::UnsupportedEncoder,
        ),
    ]
    .into_iter()
    .filter(|(count, _)| *count != 0);
    match (primary.next(), primary.next()) {
        (Some((1, reason)), None) => reason,
        (None, None) if stats.segment_misses == 0 && outcome != Outcome::Submitted => {
            Disposition::BeforeDecision
        }
        _ => Disposition::Other,
    }
}

/// Claim before initializing so duplicate or concurrent callbacks never wait
/// for another OnceLock initializer. Only the winning callback can call set.
struct ObservationCell<T> {
    claimed: AtomicBool,
    value: OnceLock<T>,
}

impl<T> ObservationCell<T> {
    fn new() -> Self {
        Self {
            claimed: AtomicBool::new(false),
            value: OnceLock::new(),
        }
    }

    fn set(&self, value: T) -> bool {
        if self
            .claimed
            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return false;
        }
        self.value.set(value).is_ok()
    }

    fn into_inner(self) -> Option<T> {
        self.value.into_inner()
    }
}

/// Callbacks only borrow identity and copy fixed-size data. No locks, clocks,
/// allocation, or serialization occur on the backend submission thread.
pub(super) struct AttemptObservation<'a, T> {
    timing: &'a T,
    program_id: Option<DeviceReusableExecutionProgramId>,
    lane_epoch: u64,
    catalog_epoch: Option<u64>,
    catalog_miss: Option<VNextReusableExecutionCatalogMissReason>,
    provider_ns: ObservationCell<u64>,
    enqueue_ns: ObservationCell<u64>,
    preparation: ObservationCell<BackendPreparation>,
    gaps: ObservationCell<ProgramGaps>,
    submission: ObservationCell<DeviceReusableExecutionObservation>,
    repeated_callback: AtomicBool,
}

impl<'a, T> AttemptObservation<'a, T> {
    pub(super) fn new(
        timing: &'a T,
        program_id: Option<DeviceReusableExecutionProgramId>,
        lane_epoch: u64,
        catalog_epoch: Option<u64>,
        catalog_miss: Option<VNextReusableExecutionCatalogMissReason>,
    ) -> Self {
        Self {
            timing,
            program_id,
            lane_epoch,
            catalog_epoch,
            catalog_miss,
            provider_ns: ObservationCell::new(),
            enqueue_ns: ObservationCell::new(),
            preparation: ObservationCell::new(),
            gaps: ObservationCell::new(),
            submission: ObservationCell::new(),
            repeated_callback: AtomicBool::new(false),
        }
    }

    fn set<U>(&self, cell: &ObservationCell<U>, value: U) {
        if !cell.set(value) {
            self.repeated_callback.store(true, Ordering::Relaxed);
        }
    }

    fn finish(
        self,
        ordinal: usize,
        stats: InvocationPreparationStats,
        outcome: Outcome,
        provider_encode_submit: Option<Duration>,
    ) -> AttemptRecord {
        AttemptRecord {
            ordinal,
            program_id: self.program_id,
            lane_epoch: self.lane_epoch,
            catalog_epoch: self.catalog_epoch,
            catalog_miss: self.catalog_miss,
            disposition: disposition(stats, outcome),
            incomplete_declarations: stats.segment_incomplete_declarations != 0,
            core_preparation: stats,
            outcome,
            provider_encode_submit_ns: provider_encode_submit.map(duration_ns),
            provider_node_encode_ns: self.provider_ns.into_inner(),
            enqueue_commands_ns: self.enqueue_ns.into_inner(),
            backend_preparation: self.preparation.into_inner(),
            program_gaps: self.gaps.into_inner(),
            backend_submission_observation: self.submission.into_inner(),
            repeated_callback: self.repeated_callback.load(Ordering::Relaxed),
        }
    }
}

impl<T: DeviceSubmissionTimingSink> DeviceSubmissionTimingSink for AttemptObservation<'_, T> {
    const ENABLED: bool = T::ENABLED;

    fn record_device_enqueue(
        &self,
        stage: DeviceEnqueueStage,
        wall: Duration,
        thread_cpu: Option<Duration>,
    ) {
        self.timing.record_device_enqueue(stage, wall, thread_cpu);
    }

    fn record_device_submission(&self, stage: DeviceSubmissionStage, elapsed: Duration) {
        self.timing.record_device_submission(stage, elapsed);
        if T::ENABLED && stage == DeviceSubmissionStage::EnqueueCommands {
            self.set(&self.enqueue_ns, duration_ns(elapsed));
        }
    }

    fn record_reusable_execution(&self, observation: DeviceReusableExecutionObservation) {
        self.timing.record_reusable_execution(observation);
        if T::ENABLED {
            self.set(&self.submission, observation);
        }
    }

    fn record_reusable_preparation(
        &self,
        program_id: Option<&DeviceReusableExecutionProgramId>,
        capture_allowed: bool,
        observation: DeviceReusableExecutionObservation,
    ) {
        self.timing
            .record_reusable_preparation(program_id, capture_allowed, observation);
        if T::ENABLED {
            self.set(
                &self.preparation,
                BackendPreparation {
                    identity: identity_match(self.program_id.as_ref(), program_id),
                    capture_allowed_at_entry: capture_allowed,
                    segments: observation,
                },
            );
        }
    }

    fn record_reusable_program_gaps(
        &self,
        program_id: &DeviceReusableExecutionProgramId,
        gaps: &[DeviceReusableExecutionProgramGap],
    ) {
        self.timing.record_reusable_program_gaps(program_id, gaps);
        if T::ENABLED {
            let mut node_counts_by_reason = [0_u64; GAP_REASONS.len()];
            for gap in gaps {
                let index = gap_index(gap.reason());
                node_counts_by_reason[index] = node_counts_by_reason[index].saturating_add(1);
            }
            self.set(
                &self.gaps,
                ProgramGaps {
                    identity: identity_match(self.program_id.as_ref(), Some(program_id)),
                    node_counts_by_reason,
                },
            );
        }
    }
}

impl<T: SubmissionWaveDispatchTimingSink> SubmissionWaveDispatchTimingSink
    for AttemptObservation<'_, T>
{
    fn record(&self, stage: SubmissionWaveDispatchStage, elapsed: Duration) {
        self.timing.record(stage, elapsed);
        if T::ENABLED && stage == SubmissionWaveDispatchStage::ProviderNodeEncode {
            self.set(&self.provider_ns, duration_ns(elapsed));
        }
    }
}

#[cfg(test)]
mod tests;
