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
    PreparationDeferred,
    PreparationError,
    PreparationCompletedWithoutDispatch,
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
struct StageDurationRecord {
    samples: u64,
    total_ns: u64,
    overflowed: bool,
}

/// Existing stage callbacks may repeat within one attempt. Keep their totals
/// separate from the first-only provider and backend observations.
#[derive(Default)]
struct StageDurationAccumulator {
    samples: AtomicU64,
    total_ns: AtomicU64,
    overflowed: AtomicBool,
}

impl StageDurationAccumulator {
    fn add(&self, counter: &AtomicU64, value: u64) {
        if let Ok(previous) =
            counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_add(value))
            })
        {
            if previous.checked_add(value).is_none() {
                self.overflowed.store(true, Ordering::Relaxed);
            }
        }
    }

    fn record(&self, elapsed: Duration) {
        if elapsed.as_nanos() > u128::from(u64::MAX) {
            self.overflowed.store(true, Ordering::Relaxed);
        }
        self.add(&self.samples, 1);
        self.add(&self.total_ns, duration_ns(elapsed));
    }

    fn into_record(self) -> Option<StageDurationRecord> {
        let samples = self.samples.into_inner();
        (samples != 0).then(|| StageDurationRecord {
            samples,
            total_ns: self.total_ns.into_inner(),
            overflowed: self.overflowed.into_inner(),
        })
    }
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
    contract_validate_reserve: Option<StageDurationRecord>,
    segment_binding_prepare_encode: Option<StageDurationRecord>,
    segment_fresh_authority_and_windows: Option<StageDurationRecord>,
    segment_backing_permit_and_metadata: Option<StageDurationRecord>,
    segment_node_dependencies_and_regions: Option<StageDurationRecord>,
    segment_backend_encode_and_validate: Option<StageDurationRecord>,
    backend_preparation: Option<BackendPreparation>,
    program_gaps: Option<ProgramGaps>,
    backend_submission_observation: Option<DeviceReusableExecutionObservation>,
    repeated_callback: bool,
}

#[derive(Serialize)]
struct WaveRecord {
    ordinal: u64,
    phase: &'static str,
    outcome: Outcome,
    preparation: Option<DecodePreparationRecord>,
    host_encode_submit_ns: Option<u64>,
    attempts: Vec<AttemptRecord>,
    attempt_overflow: usize,
}

#[derive(Clone, Copy)]
pub(super) enum PreparationStage {
    CanonicalParticipants = 0,
    OperationLockWait = 1,
    TokensAndExtension = 2,
    ResourcePrepareAttempt = 3,
    StepRequestPrepare = 4,
    StepAdmission = 5,
    SubmissionWavePrepare = 6,
}

#[derive(Serialize)]
struct DecodePreparationRecord {
    input_width: u64,
    prepared_width: Option<u64>,
    reached_dispatch: bool,
    executor_to_dispatch_ns: Option<u64>,
    executor_returned_before_dispatch_ns: Option<u64>,
    executor_prefix_ns: Option<u64>,
    resource_prepare_ns: Option<u64>,
    dispatch_tail_ns: Option<u64>,
    canonical_participants: Option<StageDurationRecord>,
    operation_lock_wait: Option<StageDurationRecord>,
    tokens_and_extension: Option<StageDurationRecord>,
    resource_prepare_attempt: Option<StageDurationRecord>,
    step_request_prepare: Option<StageDurationRecord>,
    step_admission: Option<StageDurationRecord>,
    submission_wave_prepare: Option<StageDurationRecord>,
}

pub(super) struct DecodePreparation {
    started: Instant,
    input_width: u64,
    resource_started: OnceLock<Instant>,
    resource_ready: OnceLock<(Instant, u64)>,
    closed: OnceLock<(Instant, bool)>,
    stages: [StageDurationAccumulator; 7],
}

impl DecodePreparation {
    fn new(started: Instant, input_width: usize) -> Self {
        Self {
            started,
            input_width: u64::try_from(input_width).unwrap_or(u64::MAX),
            resource_started: OnceLock::new(),
            resource_ready: OnceLock::new(),
            closed: OnceLock::new(),
            stages: std::array::from_fn(|_| StageDurationAccumulator::default()),
        }
    }

    pub(super) fn begin_resource(&self) {
        let _ = self.resource_started.set(Instant::now());
    }

    pub(super) fn resource_ready(&self, width: u64) {
        let _ = self.resource_ready.set((Instant::now(), width));
    }

    fn close(&self, dispatched: bool) {
        let _ = self.closed.set((Instant::now(), dispatched));
    }

    fn into_record(self) -> DecodePreparationRecord {
        let closed = self.closed.into_inner();
        let resource_started = self.resource_started.into_inner();
        let resource_ready = self.resource_ready.into_inner();
        let elapsed =
            |end: Instant, start: Instant| end.checked_duration_since(start).map(duration_ns);
        let reached_dispatch = closed.is_some_and(|(_, dispatched)| dispatched);
        let total = closed.and_then(|(end, _)| elapsed(end, self.started));
        let [canonical, locks, tokens, resource, request, admission, wave] = self.stages;
        DecodePreparationRecord {
            input_width: self.input_width,
            prepared_width: resource_ready.map(|(_, width)| width),
            reached_dispatch,
            executor_to_dispatch_ns: reached_dispatch.then_some(total).flatten(),
            executor_returned_before_dispatch_ns: (!reached_dispatch).then_some(total).flatten(),
            executor_prefix_ns: resource_started.and_then(|end| elapsed(end, self.started)),
            resource_prepare_ns: resource_started
                .zip(resource_ready)
                .and_then(|(start, (end, _))| elapsed(end, start)),
            dispatch_tail_ns: resource_ready
                .zip(closed)
                .filter(|(_, (_, dispatched))| *dispatched)
                .and_then(|((start, _), (end, _))| elapsed(end, start)),
            canonical_participants: canonical.into_record(),
            operation_lock_wait: locks.into_record(),
            tokens_and_extension: tokens.into_record(),
            resource_prepare_attempt: resource.into_record(),
            step_request_prepare: request.into_record(),
            step_admission: admission.into_record(),
            submission_wave_prepare: wave.into_record(),
        }
    }
}

/// Reuses a live metrics timer when present. New prefix clocks are read only
/// for a bounded decode observer; no thread-local or aggregate subtraction.
pub(super) struct PreparationTimer<'a> {
    existing: Option<super::super::vnext_timing::AtomicDurationTimer<'a>>,
    started: Option<Instant>,
    observation: Option<&'a DecodePreparation>,
    stage: PreparationStage,
}

impl<'a> PreparationTimer<'a> {
    pub(super) fn start(
        observation: Option<&'a DecodePreparation>,
        stage: PreparationStage,
    ) -> Self {
        Self {
            existing: None,
            started: observation.map(|_| Instant::now()),
            observation,
            stage,
        }
    }

    pub(super) fn existing(
        existing: Option<super::super::vnext_timing::AtomicDurationTimer<'a>>,
        observation: Option<&'a DecodePreparation>,
        stage: PreparationStage,
    ) -> Self {
        Self {
            existing,
            started: None,
            observation,
            stage,
        }
    }
}

impl Drop for PreparationTimer<'_> {
    fn drop(&mut self) {
        let elapsed = self
            .existing
            .take()
            .and_then(|timer| timer.finish())
            .or_else(|| self.started.take().map(|started| started.elapsed()));
        if !std::thread::panicking() {
            if let (Some(observation), Some(elapsed)) = (self.observation, elapsed) {
                observation.stages[self.stage as usize].record(elapsed);
            }
        }
    }
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
            preparation: None,
            attempts: Vec::with_capacity(MAXIMUM_ATTEMPTS),
            attempt_overflow: 0,
            finished: false,
        })
    }

    pub(super) fn begin_decode(
        &self,
        enabled: bool,
        width: usize,
        started: Instant,
    ) -> Option<WaveObservation<'_>> {
        let mut wave = self.begin(enabled, VNextExecutionWaveKind::Decode)?;
        wave.preparation = Some(DecodePreparation::new(started, width));
        Some(wave)
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
            "collection": "host_timing_enabled_bounded_decode_entries_and_other_dispatch_waves_after_startup_reset",
            "maximum_waves": self.maximum_waves,
            "maximum_attempts_per_wave": MAXIMUM_ATTEMPTS,
            "waves_started": self.next_wave.load(Ordering::Relaxed),
            "overflow_waves": self.overflow.load(Ordering::Relaxed),
            "unfinalized_waves": self.unfinalized.load(Ordering::Relaxed),
            "gap_reason_order": GAP_REASONS.map(|reason| reason.as_str()),
            "records": &*self.records.lock(),
            "scope": [
                "submitted means dispatch returned a handle, not terminal device success",
                "preparation is present only for the PlanRuntime decode executor entry; its wall intervals include async operation-lock waits, not engine scheduling or preceding requests",
                "executor_prefix + resource_prepare + dispatch_tail close executor_to_dispatch only when all three boundaries were reached; prefix substages and resource sub-timers are nested, never additive to their parents",
                "preparation stage samples accumulate maintenance retries within this exact executor call; early errors or deferrals have no dispatch interval or invented prepared width; absent fields are null",
                "each wave owns one host_encode_submit interval including all retries; attempt intervals are nested and must not be added to it",
                "provider_encode_submit includes core dispatch; provider_node_encode and enqueue_commands reuse their original stage scopes",
                "missing callback or interval is null, not zero; prepare and final submission observations are separate boundaries",
                "selected stage records accumulate all callbacks within this attempt; null means no callback, and a sampled zero duration is valid; overflowed totals or counts saturate and are lower bounds",
                "selected stages reuse existing scopes after the attempt observer is attached; preceding program identity lookup and resource preparation are excluded; segment substages are nested inside segment_binding_prepare_encode and provider_node_encode and must not be added to their parents",
                "program_gaps counts actual node gaps by reason before on-demand tombstone removal; segment counters have a different denominator",
                "program_id retains every field; exact backend identity comparison is diagnostic and never changes dispatch",
                "capture_allowed_at_entry is the backend quiescence decision, not proof of later completion",
                "first-wave prefix only; overflow and unfinalized waves cannot be used as complete cold-path history",
                "unwinding preserves original RAII metrics but does not finalize joint rows; snapshots during execution are not atomic",
                "attempt observation work is inside host_encode_submit; outer ledger reservation and final serialization are outside that interval; decode entry boundary clocks are read only for a bounded observer"
            ]
        })
    }
}

pub(super) struct WaveObservation<'a> {
    ledger: &'a Ledger,
    ordinal: u64,
    phase: &'static str,
    preparation: Option<DecodePreparation>,
    attempts: Vec<AttemptRecord>,
    attempt_overflow: usize,
    finished: bool,
}

impl WaveObservation<'_> {
    pub(super) fn preparation(&self) -> Option<&DecodePreparation> {
        self.preparation.as_ref()
    }

    pub(super) fn close_preparation(&self, dispatched: bool) {
        if let Some(preparation) = &self.preparation {
            preparation.close(dispatched);
        }
    }

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
        let record = WaveRecord {
            ordinal: self.ordinal,
            phase: self.phase,
            outcome,
            preparation: self.preparation.take().map(DecodePreparation::into_record),
            host_encode_submit_ns: host_elapsed.map(duration_ns),
            attempts: std::mem::take(&mut self.attempts),
            attempt_overflow: self.attempt_overflow,
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

fn duration_ns(elapsed: Duration) -> u64 {
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

/// Callbacks only borrow identity and update fixed-size data. No locks, clocks,
/// allocation, or serialization occur on the backend submission thread.
pub(super) struct AttemptObservation<'a, T> {
    timing: &'a T,
    program_id: Option<DeviceReusableExecutionProgramId>,
    lane_epoch: u64,
    catalog_epoch: Option<u64>,
    catalog_miss: Option<VNextReusableExecutionCatalogMissReason>,
    provider_ns: ObservationCell<u64>,
    enqueue_ns: ObservationCell<u64>,
    contract_validate_reserve: StageDurationAccumulator,
    segment_binding_prepare_encode: StageDurationAccumulator,
    segment_fresh_authority_and_windows: StageDurationAccumulator,
    segment_backing_permit_and_metadata: StageDurationAccumulator,
    segment_node_dependencies_and_regions: StageDurationAccumulator,
    segment_backend_encode_and_validate: StageDurationAccumulator,
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
            contract_validate_reserve: StageDurationAccumulator::default(),
            segment_binding_prepare_encode: StageDurationAccumulator::default(),
            segment_fresh_authority_and_windows: StageDurationAccumulator::default(),
            segment_backing_permit_and_metadata: StageDurationAccumulator::default(),
            segment_node_dependencies_and_regions: StageDurationAccumulator::default(),
            segment_backend_encode_and_validate: StageDurationAccumulator::default(),
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
            contract_validate_reserve: self.contract_validate_reserve.into_record(),
            segment_binding_prepare_encode: self.segment_binding_prepare_encode.into_record(),
            segment_fresh_authority_and_windows: self
                .segment_fresh_authority_and_windows
                .into_record(),
            segment_backing_permit_and_metadata: self
                .segment_backing_permit_and_metadata
                .into_record(),
            segment_node_dependencies_and_regions: self
                .segment_node_dependencies_and_regions
                .into_record(),
            segment_backend_encode_and_validate: self
                .segment_backend_encode_and_validate
                .into_record(),
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
        if T::ENABLED {
            match stage {
                SubmissionWaveDispatchStage::ProviderNodeEncode => {
                    self.set(&self.provider_ns, duration_ns(elapsed));
                }
                SubmissionWaveDispatchStage::ContractValidateAndReserve => {
                    self.contract_validate_reserve.record(elapsed);
                }
                SubmissionWaveDispatchStage::SegmentBindingPrepareAndEncode => {
                    self.segment_binding_prepare_encode.record(elapsed);
                }
                SubmissionWaveDispatchStage::SegmentFreshAuthorityAndWindows => {
                    self.segment_fresh_authority_and_windows.record(elapsed);
                }
                SubmissionWaveDispatchStage::SegmentBackingPermitAndMetadata => {
                    self.segment_backing_permit_and_metadata.record(elapsed);
                }
                SubmissionWaveDispatchStage::SegmentNodeDependenciesAndRegions => {
                    self.segment_node_dependencies_and_regions.record(elapsed);
                }
                SubmissionWaveDispatchStage::SegmentBackendEncodeAndValidate => {
                    self.segment_backend_encode_and_validate.record(elapsed);
                }
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests;
