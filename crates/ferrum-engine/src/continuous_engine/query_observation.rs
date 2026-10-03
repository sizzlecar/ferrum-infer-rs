//! Bounded, passive planner diagnostics. No record can be imported as authority.
use ferrum_scheduler::implementations::continuous::{
    cost_model::structured_v2::{
        NumericalFamilyKeyV1, StructuredCostTemplatePolicyV1, StructuredOwnerKeyV2,
        StructuredQueryV2, StructuredUnknownV2,
    },
    slo_planner::*,
};
use ferrum_types::{
    FerrumError, SloRequiredQueryObservationConfig, SloRequiredQueryObservationLimits,
};
use parking_lot::Mutex;
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    fs::{File, OpenOptions},
    io::{self, Write},
    mem::{align_of, size_of},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, OnceLock,
    },
    thread::{self, JoinHandle, Thread},
};

#[cfg(test)]
mod tests;

const FOOTER_RESERVE: u64 = 8192;
const ERROR_CHARACTERS: usize = 256;
const SCHEMA: &str = "ferrum.planner-required-query-observation.v1";

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum Loss {
    Contention,
    Capacity,
    TransactionLimit,
    EventLimit,
    Closed,
    WriterFailed,
    Allocation,
    Encoding,
}
impl Loss {
    fn index(self) -> usize {
        self as usize
    }
}
const LOSSES: [&str; 8] = [
    "contention",
    "capacity",
    "transaction_limit",
    "event_limit",
    "closed",
    "writer_failed",
    "allocation",
    "encoding",
];

#[derive(Default)]
struct Counters {
    offered: AtomicU64,
    accepted: AtomicU64,
    written: AtomicU64,
    lost: [AtomicU64; 8],
    first_lost: AtomicU64,
    last_lost: AtomicU64,
    transactions: AtomicU64,
    active_transactions: AtomicU64,
    abandoned_transactions: AtomicU64,
    overflow: AtomicBool,
    failed: AtomicBool,
    closed: AtomicBool,
    footer_written: AtomicBool,
    flushed: AtomicBool,
    file_bytes: AtomicU64,
    first_error: OnceLock<String>,
}
impl Counters {
    fn next(&self, counter: &AtomicU64) -> Option<u64> {
        match counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1)) {
            Ok(n) => Some(n + 1),
            Err(_) => {
                self.overflow.store(true, Ordering::Release);
                None
            }
        }
    }
    fn loss(&self, ordinal: u64, reason: Loss) {
        self.next(&self.lost[reason.index()]);
        if ordinal != 0 {
            let _ = self
                .first_lost
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                    Some(if n == 0 { ordinal } else { n.min(ordinal) })
                });
            self.last_lost.fetch_max(ordinal, Ordering::Relaxed);
        }
    }
    fn fail(&self, error: impl std::fmt::Display) {
        self.failed.store(true, Ordering::Release);
        let _ = self
            .first_error
            .set(error.to_string().chars().take(ERROR_CHARACTERS).collect());
    }
    fn complete(&self) -> bool {
        !self.failed.load(Ordering::Acquire)
            && !self.overflow.load(Ordering::Acquire)
            && self.lost.iter().all(|n| n.load(Ordering::Relaxed) == 0)
            && self.active_transactions.load(Ordering::Acquire) == 0
            && self.abandoned_transactions.load(Ordering::Relaxed) == 0
            && self.offered.load(Ordering::Relaxed) == self.written.load(Ordering::Relaxed)
    }
    fn wire(&self) -> Value {
        let loss: serde_json::Map<String, Value> = LOSSES
            .iter()
            .zip(&self.lost)
            .map(|(k, v)| ((*k).into(), json!(v.load(Ordering::Relaxed))))
            .collect();
        json!({"offered":self.offered.load(Ordering::Relaxed), "accepted":self.accepted.load(Ordering::Relaxed),
            "written":self.written.load(Ordering::Relaxed), "lost":loss,
            "first_lost_ordinal":self.first_lost.load(Ordering::Relaxed), "last_lost_ordinal":self.last_lost.load(Ordering::Relaxed),
            "transactions":self.transactions.load(Ordering::Relaxed), "active_transactions":self.active_transactions.load(Ordering::Acquire),
            "abandoned_transactions":self.abandoned_transactions.load(Ordering::Relaxed), "counter_exhausted":self.overflow.load(Ordering::Acquire),
            "writer_failed":self.failed.load(Ordering::Acquire), "first_error":self.first_error.get(),
            "closed":self.closed.load(Ordering::Acquire), "footer_written":self.footer_written.load(Ordering::Acquire),
            "flushed_and_synced":self.flushed.load(Ordering::Acquire), "file_bytes":self.file_bytes.load(Ordering::Relaxed)})
    }
}

#[derive(Serialize)]
struct Frontier {
    request_id: ferrum_types::RequestId,
    incarnation: u64,
    work_generation: u64,
    ingress_at_ns: u64,
    first_commit_at_ns: Option<u64>,
    last_commit_at_ns: Option<u64>,
    committed_tokens: u32,
    maximum_output_tokens: u32,
    latency_budgets_ns: [u64; 3],
    slo_failed: bool,
    phase: Phase,
    readiness: DebugWire<RequestReadiness>,
    context_tokens: u32,
    recurrent_state_bytes: u64,
    output_credit: DebugWire<OutputCreditView>,
    output_policy_signature: [u8; 32],
    fairness_rank: u64,
    recovery_eligible_bypasses: usize,
    recovery_bound: usize,
    ranking_service_cost_ns: Option<u64>,
    optimistic_next_service: DebugWire<Option<OptimisticServiceLowerBound>>,
}
#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Phase {
    Decode,
    Prefill {
        admitted_at_ns: u64,
        reference_work_at_admission_ns: u64,
        offset: u32,
        total_prompt_tokens: u32,
        logical_high_water: u32,
        executable_until: u32,
        reference_version: u64,
        reference_curve_recorded: bool,
    },
}
impl Frontier {
    fn from_request(r: &RequestSchedulingView) -> Self {
        Self {
            request_id: r.key.request_id.clone(),
            incarnation: r.key.incarnation,
            work_generation: r.key.work_generation.get(),
            ingress_at_ns: r.timing.ingress_at_ns,
            first_commit_at_ns: r.timing.first_commit_at_ns,
            last_commit_at_ns: r.timing.last_commit_at_ns,
            committed_tokens: r.timing.committed_tokens,
            maximum_output_tokens: r.timing.maximum_output_tokens.get(),
            latency_budgets_ns: [
                r.timing.budgets.ttft_ns.get(),
                r.timing.budgets.tpot_ns.get(),
                r.timing.budgets.itl_ns.get(),
            ],
            slo_failed: r.timing.slo_failed,
            phase: match &r.phase {
                RequestPhaseView::Decode => Phase::Decode,
                RequestPhaseView::Prefill(p) => Phase::Prefill {
                    admitted_at_ns: p.admitted_at_ns,
                    reference_work_at_admission_ns: p.reference_work_at_admission_ns,
                    offset: p.offset,
                    total_prompt_tokens: p.total_prompt_tokens.get(),
                    logical_high_water: p.logical_high_water,
                    executable_until: p.executable_until,
                    reference_version: p.reference.version,
                    reference_curve_recorded: false,
                },
            },
            readiness: DebugWire(r.readiness),
            context_tokens: r.context_tokens,
            recurrent_state_bytes: r.recurrent_state_bytes,
            output_credit: DebugWire(r.output_credit),
            output_policy_signature: r.output_policy_signature,
            fairness_rank: r.fairness_rank,
            recovery_eligible_bypasses: r.recovery_service.eligible_bypasses(),
            recovery_bound: r.recovery_service.bound().get(),
            ranking_service_cost_ns: r.ranking_service_cost_ns,
            optimistic_next_service: DebugWire(r.optimistic_next_service),
        }
    }
}
struct DebugWire<T>(T);
impl<T: std::fmt::Debug> Serialize for DebugWire<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(&format_args!("{:?}", self.0))
    }
}
struct Fields<T>(T);
macro_rules! scalar_fields {
    ($ty:ty, $($field:ident),+ $(,)?) => {
        impl Serialize for Fields<$ty> {
            fn serialize<S: serde::Serializer>(&self, serializer:S)->Result<S::Ok,S::Error> {
                use serde::ser::SerializeMap;
                let mut map=serializer.serialize_map(None)?;
                $(map.serialize_entry(stringify!($field),&self.0.$field)?;)+
                map.end()
            }
        }
    };
}
scalar_fields!(
    PlanningCost,
    typical_ns,
    planning_ns,
    model_version,
    valid_for_ns
);
scalar_fields!(
    CapacityReadView,
    evidence_known,
    available_kv_tokens,
    maximum_context_tokens,
    available_workspace_bytes,
    available_output_bytes
);
scalar_fields!(
    PlanningScope,
    horizon_end_ns,
    reference_decode_token_ns,
    reference_work_version
);
impl Serialize for Fields<PlanningSearchStats> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let s = &self.0;
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("phase", &DebugWire(s.phase))?;
        macro_rules! field {($($name:ident),+)=>{$(map.serialize_entry(stringify!($name),&s.$name)?;)+};}
        field!(
            enumeration_attempts,
            expanded_candidates,
            generated_candidates,
            max_depth_reached,
            candidate_truncations,
            search_soft_stops,
            replay_reserve_stops,
            measured_replay_work_ns,
            replay_reserve_ns,
            beam_pruned_nodes,
            cost_unknown_candidates,
            shape_unknown_candidates,
            resource_unknown_candidates
        );
        map.end()
    }
}
#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Outcome {
    Known {
        cost: Fields<PlanningCost>,
    },
    StructuredUnknown {
        reason: DebugWire<StructuredUnknownV2>,
    },
    ModelUnavailable,
    ClockMappingFailed,
}
impl From<PlanningQueryOutcome> for Outcome {
    fn from(outcome: PlanningQueryOutcome) -> Self {
        match outcome {
            PlanningQueryOutcome::Known(cost) => Self::Known { cost: Fields(cost) },
            PlanningQueryOutcome::StructuredUnknown(reason) => Self::StructuredUnknown {
                reason: DebugWire(reason),
            },
            PlanningQueryOutcome::ModelUnavailable => Self::ModelUnavailable,
            PlanningQueryOutcome::ClockMappingFailed => Self::ClockMappingFailed,
        }
    }
}
#[derive(Serialize)]
struct Work {
    request_id: ferrum_types::RequestId,
    incarnation: u64,
    work_generation: u64,
    action: DebugWire<WaveAction>,
}
#[derive(Serialize)]
struct Snapshot {
    observed_at_ns: u64,
    generation: u64,
    cost_model_version: u64,
    model_weights: [u8; 32],
    numerical_policy: [u8; 32],
    device_runtime: [u8; 32],
    execution_config: [u8; 32],
    requests: Vec<Frontier>,
    capacity: Fields<CapacityReadView>,
    scope: Fields<PlanningScope>,
    has_unmodeled_maintenance: bool,
}
#[derive(Serialize)]
struct Attempt {
    attempt: u64,
    phase: DebugWire<PlanningQueryPhase>,
    depth: usize,
    requests: Vec<Frontier>,
    work: Vec<Work>,
}
#[derive(Serialize)]
struct Lookup {
    attempt: u64,
    alternative: usize,
    planning_now_ns: u64,
    cost_now_ns: Option<u64>,
    outcome: Outcome,
}

/// Constructed only by the bounded background writer from its already retained
/// checked query. No identity hash, recipe replay or new producer allocation.
#[derive(Serialize)]
struct QueryIdentity<'a> {
    domain_signature: &'a [u8; 32],
    owner: &'a StructuredOwnerKeyV2,
    numerical_family: Option<NumericalFamilyKeyV1>,
    numerical_family_error: Option<DebugWire<StructuredUnknownV2>>,
    physical_domain_signature: Option<&'a [u8; 32]>,
    /// Match the real snapshot lookup order, including an unavailable identity.
    installed_algorithm_set_identity: Option<TemplateIdentity<'a>>,
    ordered_identity: Option<TemplateIdentity<'a>>,
}

#[derive(Serialize)]
struct TemplateIdentity<'a> {
    owner: &'a StructuredOwnerKeyV2,
    domain_signature: &'a [u8; 32],
}

impl<'a> QueryIdentity<'a> {
    fn new(query: &'a StructuredQueryV2) -> Self {
        let input = query.input();
        let family = input.numerical_family_key();
        let identity = |policy| {
            input
                .cost_template_identity(policy)
                .map(|(owner, domain_signature)| TemplateIdentity {
                    owner,
                    domain_signature,
                })
        };
        Self {
            domain_signature: input.domain_signature(),
            owner: input.owner(),
            numerical_family: family.as_ref().ok().copied(),
            numerical_family_error: family.err().map(DebugWire),
            physical_domain_signature: input.physical_domain_signature(),
            installed_algorithm_set_identity: identity(
                StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1,
            ),
            ordered_identity: identity(StructuredCostTemplatePolicyV1::OrderedV1),
        }
    }
}
#[derive(Serialize)]
struct EndAttempt {
    attempt: u64,
    constructed: usize,
    queried: usize,
    not_queried_start: usize,
    not_queried_end: usize,
    reason: DebugWire<PlanningQueryAttemptEnd>,
}
#[derive(Serialize)]
struct EndTransaction {
    audit_available: bool,
    audit_times_exclude_end_recording: bool,
    outcome: &'static str,
    decision: &'static str,
    reason: &'static str,
    planning_wall_ns: u64,
    budget_ns: u64,
    planner_exhausted: bool,
    hard_exhausted: bool,
    search: Fields<PlanningSearchStats>,
}
/// Fixed-size, original transaction-clock diagnostics. The writer serializes
/// these asynchronously under the same byte/event limits as query evidence.
#[derive(Clone, Copy, Serialize)]
pub(super) struct ControllerCheckpoint {
    pub stage: &'static str,
    pub edge: &'static str,
    pub elapsed_ns: Option<u64>,
    pub hard_budget_ns: u64,
    pub optional_deadline_elapsed_ns: Option<u64>,
    pub completion_preparation_ns: Option<u64>,
    pub publication_reserve_ns: Option<u64>,
}

enum Event {
    ControllerCheckpoint(ControllerCheckpoint),
    Identity(Vec<u8>),
    Begin,
    ReplayBegin {
        replay: u64,
        waves: usize,
    },
    ReplayEnd {
        replay: u64,
        reason: PlanningQueryAttemptEnd,
    },
    SelectedReplay {
        replay: u64,
    },
    Snapshot(Snapshot),
    Attempt(Attempt),
    Query(
        PlanningQueryKey,
        Result<StructuredQueryV2, StructuredUnknownV2>,
    ),
    Lookup(Lookup),
    EndAttempt(EndAttempt),
    EndTransaction(EndTransaction),
}
impl Event {
    fn retained_bytes(&self) -> Option<usize> {
        let extra = match self {
            Self::Identity(bytes) => bytes.capacity(),
            Self::Snapshot(s) => s.requests.capacity().checked_mul(size_of::<Frontier>())?,
            Self::Attempt(a) => a
                .requests
                .capacity()
                .checked_mul(size_of::<Frontier>())?
                .checked_add(a.work.capacity().checked_mul(size_of::<Work>())?)?,
            Self::Query(_, Ok(q)) => q.observation_retained_bytes()?,
            _ => 0,
        };
        size_of::<Entry>().checked_add(extra)
    }
    fn encode(&self, entry: &Entry, out: &mut BoundedLine) -> serde_json::Result<()> {
        #[derive(Serialize)]
        struct Envelope<'a, T: Serialize> {
            schema: &'static str,
            ordinal: u64,
            retained_ordinal: u64,
            transaction: u64,
            event: &'static str,
            data: &'a T,
        }
        fn write<T: Serialize>(
            entry: &Entry,
            out: &mut BoundedLine,
            event: &'static str,
            data: &T,
        ) -> serde_json::Result<()> {
            serde_json::to_writer(
                out,
                &Envelope {
                    schema: SCHEMA,
                    ordinal: entry.ordinal,
                    retained_ordinal: entry.retained_ordinal,
                    transaction: entry.transaction,
                    event,
                    data,
                },
            )
        }
        match self {
            Self::ControllerCheckpoint(v) => write(entry, out, "controller_checkpoint", v),
            Self::Identity(bytes) => {
                write!(out, "{{\"schema\":\"{SCHEMA}\",\"ordinal\":{},\"retained_ordinal\":{},\"transaction\":0,\"event\":\"run_identity\",\"data\":",entry.ordinal,entry.retained_ordinal).map_err(serde_json::Error::io)?;
                out.write_all(bytes).map_err(serde_json::Error::io)?;
                out.write_all(b"}").map_err(serde_json::Error::io)
            }
            Self::Begin => write(entry, out, "transaction_begin", &()),
            Self::ReplayBegin { replay, waves } => write(
                entry,
                out,
                "replay_begin",
                &json!({"replay":replay,"waves":waves}),
            ),
            Self::ReplayEnd { replay, reason } => write(
                entry,
                out,
                "replay_end",
                &json!({"replay":replay,"reason":DebugWire(*reason)}),
            ),
            Self::SelectedReplay { replay } => {
                write(entry, out, "selected_replay", &json!({"replay":replay}))
            }
            Self::Snapshot(v) => write(entry, out, "snapshot", v),
            Self::Attempt(v) => write(entry, out, "attempt_begin", v),
            Self::Lookup(v) => write(entry, out, "query_lookup", v),
            Self::EndAttempt(v) => write(entry, out, "attempt_end", v),
            Self::EndTransaction(v) => write(entry, out, "transaction_end", v),
            Self::Query(key, result) => {
                #[derive(Serialize)]
                struct Query<'a, A: Serialize> {
                    attempt: u64,
                    alternative: usize,
                    demand: Option<&'a ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredQueryDemandV2>,
                    identity: Option<QueryIdentity<'a>>,
                    // Borrowed from the already retained original query, never
                    // reconstructed from a family hash or the installed model.
                    algorithm_axes: Option<A>,
                    prebound_algorithm_universe: Option<&'a [u8; 32]>,
                    input_unknown: Option<DebugWire<StructuredUnknownV2>>,
                    demand_error: Option<DebugWire<StructuredUnknownV2>>,
                }
                let demand = result
                    .as_ref()
                    .map_err(|e| *e)
                    .and_then(|q| q.required_coverage());
                write(
                    entry,
                    out,
                    "query_constructed",
                    &Query {
                        attempt: key.attempt,
                        alternative: key.alternative,
                        demand: demand.as_ref().ok(),
                        identity: result.as_ref().ok().map(QueryIdentity::new),
                        algorithm_axes: result
                            .as_ref()
                            .ok()
                            .map(|q| q.input().observation_algorithm_axes()),
                        prebound_algorithm_universe: result
                            .as_ref()
                            .ok()
                            .and_then(|q| q.input().algorithm_universe_signature()),
                        input_unknown: result.as_ref().err().copied().map(DebugWire),
                        demand_error: if result.is_ok() {
                            demand.as_ref().err().copied().map(DebugWire)
                        } else {
                            None
                        },
                    },
                )
            }
        }
    }
}
struct Entry {
    ordinal: u64,
    retained_ordinal: u64,
    transaction: u64,
    event: Event,
    bytes: usize,
}
// The high bit closes admission atomically with the active producer count.
// This includes producers admitted before payload construction/publication.
const CLOSED: usize = 1 << (usize::BITS - 1);
struct Shared {
    sender: crossbeam_channel::Sender<Entry>,
    receiver: crossbeam_channel::Receiver<Entry>,
    gate: AtomicUsize,
    retained_events: AtomicUsize,
    retained_bytes: AtomicUsize,
    event_capacity: usize,
    limits: SloRequiredQueryObservationLimits,
    counters: Counters,
    worker: OnceLock<Thread>,
    identity_bound: AtomicBool,
}
struct OfferPermit<'a>(&'a Shared);
impl Drop for OfferPermit<'_> {
    fn drop(&mut self) {
        self.0.gate.fetch_sub(1, Ordering::AcqRel);
        self.0.wake();
    }
}
struct Reservation<'a> {
    shared: &'a Shared,
    bytes: usize,
    published: bool,
}
impl Drop for Reservation<'_> {
    fn drop(&mut self) {
        if !self.published {
            self.shared.release(self.bytes);
        }
    }
}
fn reserve(counter: &AtomicUsize, amount: usize, limit: usize) -> bool {
    counter
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
            n.checked_add(amount).filter(|n| *n <= limit)
        })
        .is_ok()
}
impl Shared {
    fn wake(&self) {
        if let Some(worker) = self.worker.get() {
            worker.unpark();
        }
    }
    fn offer(
        &self,
        transaction: u64,
        transaction_event: Option<u64>,
        estimated: Option<usize>,
        make: impl FnOnce() -> Option<Event>,
        _startup: bool,
    ) -> bool {
        let entered = self
            .gate
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n & CLOSED == 0 && n < CLOSED - 1).then(|| n + 1)
            })
            .is_ok();
        let _permit = entered.then(|| OfferPermit(self));
        let ordinal = self.counters.next(&self.counters.offered).unwrap_or(0);
        let reject = |r| self.counters.loss(ordinal, r);
        if !entered {
            reject(Loss::Closed);
            return false;
        }
        if ordinal == 0 || ordinal > self.limits.max_events {
            reject(Loss::EventLimit);
            return false;
        }
        if transaction_event.is_none_or(|n| n > self.limits.max_transaction_events) {
            reject(Loss::TransactionLimit);
            return false;
        }
        if self.counters.failed.load(Ordering::Acquire) {
            reject(Loss::WriterFailed);
            return false;
        }
        let Some(bytes) = estimated.filter(|n| *n <= self.limits.max_event_bytes) else {
            reject(Loss::Capacity);
            return false;
        };
        let Some(additional) = bytes.checked_sub(size_of::<Entry>()) else {
            reject(Loss::Capacity);
            return false;
        };
        if !reserve(&self.retained_events, 1, self.event_capacity) {
            reject(Loss::Capacity);
            return false;
        }
        if !reserve(&self.retained_bytes, additional, self.limits.queue_bytes) {
            self.retained_events.fetch_sub(1, Ordering::AcqRel);
            reject(Loss::Capacity);
            return false;
        }
        // Queue + worker in-flight + admitted producers all retain their
        // reservation. Nothing allocates a DTO before this admission point.
        let mut reservation = Reservation {
            shared: self,
            bytes: additional,
            published: false,
        };
        let Some(event) = make() else {
            reject(Loss::Allocation);
            return false;
        };
        let Some(actual) = event.retained_bytes().filter(|n| *n <= bytes) else {
            reject(Loss::Capacity);
            return false;
        };
        self.retained_bytes
            .fetch_sub(bytes - actual, Ordering::AcqRel);
        reservation.bytes = actual - size_of::<Entry>();
        let entry = Entry {
            ordinal,
            retained_ordinal: 0,
            transaction,
            event,
            bytes: reservation.bytes,
        };
        match self.sender.try_send(entry) {
            Ok(()) => {
                reservation.published = true;
                self.counters.next(&self.counters.accepted);
                self.wake();
                true
            }
            Err(crossbeam_channel::TrySendError::Full(entry)) => {
                drop(entry);
                reject(Loss::Capacity);
                false
            }
            Err(crossbeam_channel::TrySendError::Disconnected(entry)) => {
                drop(entry);
                reject(Loss::WriterFailed);
                false
            }
        }
    }
    fn release(&self, bytes: usize) {
        self.retained_bytes.fetch_sub(bytes, Ordering::AcqRel);
        self.retained_events.fetch_sub(1, Ordering::AcqRel);
    }
}

/// One bounded scratch line. Serialization cannot grow beyond its declared cap.
struct BoundedLine {
    bytes: Vec<u8>,
    limit: usize,
}
impl Write for BoundedLine {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            return Err(io::Error::other("required-query record byte limit"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
trait Destination: Write + Send {
    fn sync(&mut self) -> io::Result<()>;
}
impl Destination for File {
    fn sync(&mut self) -> io::Result<()> {
        self.sync_all()
    }
}

pub(super) struct Writer {
    shared: Arc<Shared>,
    join: Mutex<Option<JoinHandle<()>>>,
}
impl Writer {
    pub fn open(
        config: &SloRequiredQueryObservationConfig,
    ) -> ferrum_types::Result<Option<Arc<Self>>> {
        config.validate().map_err(FerrumError::config)?;
        let (SloRequiredQueryObservationConfig::StructuredRequiredV1 { path, limits }
        | SloRequiredQueryObservationConfig::StructuredUncalibratedV1 { path, limits }) = config
        else {
            return Ok(None);
        };
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| FerrumError::internal(format!("required-query observation open: {e}")))?;
        Self::start(limits.clone(), Box::new(file)).map(Some)
    }
    fn start(
        limits: SloRequiredQueryObservationLimits,
        mut destination: Box<dyn Destination>,
    ) -> ferrum_types::Result<Arc<Self>> {
        let header = json!({"schema":SCHEMA,"event":"run_header","run_id":uuid::Uuid::new_v4(),"limits":limits,
            "authority":"diagnostic_only_not_profile_source_or_witness", "clock_scope":"original snapshot/planning/model monotonic clocks; no writer timestamp substitution",
            "identity_scope":"original snapshot fingerprints only; binary/source/config provenance must be bound externally",
            "scope":"actually visited construction and lookup events; not all required future branches", "frontier_scope":"original request numeric frontier; reference curve omitted and version retained",
            "memory_scope":"queue_bytes includes conservative bounded-channel slot storage plus all admitted producer, queued and worker-inflight numeric payload; plus two max_event_bytes worker buffers (serialization and demand), bounded header/footer, fixed channel metadata/counters/thread stack; no proof/source/device resources",
            "footer_reserve_bytes":FOOTER_RESERVE});
        let mut bytes =
            serde_json::to_vec(&header).map_err(|e| FerrumError::internal(e.to_string()))?;
        bytes.push(b'\n');
        if (bytes.len() as u64)
            .checked_add(FOOTER_RESERVE)
            .is_none_or(|n| n > limits.max_file_bytes)
        {
            return Err(FerrumError::config(
                "required-query file limit cannot hold header and reserved footer",
            ));
        }
        destination
            .write_all(&bytes)
            .and_then(|_| destination.flush())
            .map_err(|e| FerrumError::internal(format!("required-query header: {e}")))?;
        // crossbeam-channel's bounded array stores Entry plus an atomic stamp
        // per slot. Include conservative alignment padding as well. Channel
        // metadata is fixed overhead, like counters and the worker stack.
        let slot_bytes = size_of::<Entry>()
            .checked_add(size_of::<AtomicUsize>())
            .and_then(|n| n.checked_add(2 * align_of::<Entry>()))
            .ok_or_else(|| FerrumError::config("required-query slot size overflow"))?;
        let event_capacity = limits.queue_events.min(limits.queue_bytes / slot_bytes);
        if event_capacity == 0 {
            return Err(FerrumError::config(
                "required-query queue cannot hold one event",
            ));
        }
        let queue_storage_bytes = event_capacity
            .checked_mul(slot_bytes)
            .ok_or_else(|| FerrumError::config("required-query queue storage overflow"))?;
        let (sender, receiver) = crossbeam_channel::bounded(event_capacity);
        let shared = Arc::new(Shared {
            sender,
            receiver,
            gate: AtomicUsize::new(0),
            retained_events: AtomicUsize::new(0),
            retained_bytes: AtomicUsize::new(queue_storage_bytes),
            event_capacity,
            limits,
            counters: Counters::default(),
            worker: OnceLock::new(),
            identity_bound: AtomicBool::new(false),
        });
        shared
            .counters
            .file_bytes
            .store(bytes.len() as u64, Ordering::Relaxed);
        let worker_shared = shared.clone();
        let join = thread::Builder::new()
            .name("ferrum-required-query".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    consume(&worker_shared, &mut *destination)
                }));
                if result.is_err() {
                    worker_shared
                        .counters
                        .fail("required-query writer panicked");
                }
            })
            .map_err(|e| FerrumError::internal(format!("required-query worker: {e}")))?;
        let _ = shared.worker.set(join.thread().clone());
        Ok(Arc::new(Self {
            shared,
            join: Mutex::new(Some(join)),
        }))
    }
    /// Startup-only immutable configuration/import binding. The runtime identity
    /// remains the original per-snapshot identity, not this diagnostic record.
    pub fn bind_identity(
        &self,
        config: &ferrum_types::SloConfig,
        receipt: Option<&ferrum_types::SloCostProfileReceipt>,
    ) -> ferrum_types::Result<()> {
        if self.shared.identity_bound.swap(true, Ordering::AcqRel) {
            self.shared
                .counters
                .fail("required-query identity already bound");
            return Err(FerrumError::internal(
                "required-query identity already bound",
            ));
        }
        #[derive(Serialize)]
        struct Identity<'a> {
            config: &'a ferrum_types::SloConfig,
            cost_profile_receipt: Option<&'a ferrum_types::SloCostProfileReceipt>,
            #[serde(skip_serializing_if = "Option::is_none")]
            uncalibrated_scope: Option<&'static str>,
        }
        let mut body = BoundedLine {
            bytes: Vec::new(),
            limit: self
                .shared
                .limits
                .max_event_bytes
                .saturating_sub(size_of::<Entry>() + 256),
        };
        body.bytes.try_reserve_exact(body.limit).map_err(|e| {
            self.shared.counters.fail(&e);
            FerrumError::internal(e.to_string())
        })?;
        serde_json::to_writer(
            &mut body,
            &Identity {
                config,
                cost_profile_receipt: receipt,
                uncalibrated_scope: config.required_query_observation.is_uncalibrated().then_some(
                    "actual_root_candidates_only; model_version_zero_means_absent; unknown_stops_each_edge; not_full_required_future_coverage"
                ),
            },
        )
        .map_err(|e| {
            self.shared.counters.fail(&e);
            FerrumError::internal(format!("required-query identity: {e}"))
        })?;
        // Startup serialization uses a reserved buffer; shrink the owned record
        // before admission so unused scratch is not retained by the queue.
        body.bytes.shrink_to_fit();
        let bytes = body.bytes.capacity().checked_add(size_of::<Entry>());
        if !self.shared.offer(
            0,
            Some(1),
            bytes,
            || Some(Event::Identity(body.bytes)),
            true,
        ) {
            return Err(FerrumError::internal(
                "required-query identity was not retained",
            ));
        }
        Ok(())
    }
    pub fn begin(self: &Arc<Self>) -> Arc<Transaction> {
        let id = self
            .shared
            .counters
            .next(&self.shared.counters.transactions)
            .unwrap_or(0);
        self.shared
            .counters
            .next(&self.shared.counters.active_transactions);
        let transaction = Arc::new(Transaction {
            writer: self.clone(),
            id,
            events: AtomicU64::new(0),
            attempts: AtomicU64::new(0),
            replays: AtomicU64::new(0),
            finished: AtomicBool::new(false),
        });
        transaction.offer(Some(size_of::<Entry>()), || Some(Event::Begin));
        transaction
    }
    pub fn snapshot(&self) -> Value {
        let counters = &self.shared.counters;
        json!({"enabled":true,"schema":SCHEMA,"statistics":counters.wire(),"identity_bound":self.shared.identity_bound.load(Ordering::Acquire),"retained_events":self.shared.retained_events.load(Ordering::Acquire),"retained_bytes":self.shared.retained_bytes.load(Ordering::Acquire),
            "recording_complete":counters.closed.load(Ordering::Acquire)&&counters.footer_written.load(Ordering::Acquire)&&counters.flushed.load(Ordering::Acquire)&&counters.complete()})
    }
    pub fn close(&self) -> ferrum_types::Result<()> {
        // Close/join is a slow path owned by shared run/serve shutdown, never a callback.
        let mut join = self.join.lock();
        if !self.shared.identity_bound.load(Ordering::Acquire) {
            self.shared
                .counters
                .fail("required-query identity was never bound");
        }
        self.shared.gate.fetch_or(CLOSED, Ordering::AcqRel);
        if let Some(worker) = self.shared.worker.get() {
            worker.unpark();
        }
        if let Some(handle) = join.take() {
            if handle.join().is_err() {
                self.shared
                    .counters
                    .fail("required-query writer join failed");
            }
        }
        self.shared.counters.closed.store(true, Ordering::Release);
        let c = &self.shared.counters;
        if !c.complete()
            || !c.footer_written.load(Ordering::Acquire)
            || !c.flushed.load(Ordering::Acquire)
        {
            return Err(FerrumError::internal(format!(
                "required-query observation incomplete: {}",
                c.wire()
            )));
        }
        Ok(())
    }
}
impl Drop for Writer {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

pub(super) struct Transaction {
    writer: Arc<Writer>,
    id: u64,
    events: AtomicU64,
    attempts: AtomicU64,
    replays: AtomicU64,
    finished: AtomicBool,
}
impl Transaction {
    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn checkpoint(&self, checkpoint: ControllerCheckpoint) {
        self.offer(Some(size_of::<Entry>()), || {
            Some(Event::ControllerCheckpoint(checkpoint))
        });
    }
    fn offer(&self, bytes: Option<usize>, make: impl FnOnce() -> Option<Event>) {
        let event = self.writer.shared.counters.next(&self.events);
        if self.finished.load(Ordering::Acquire) {
            let ordinal = self
                .writer
                .shared
                .counters
                .next(&self.writer.shared.counters.offered)
                .unwrap_or(0);
            self.writer.shared.counters.loss(ordinal, Loss::Closed);
            return;
        }
        self.writer.shared.offer(self.id, event, bytes, make, false);
    }
    pub fn snapshot(&self, s: &SchedulerSnapshot) {
        self.offer(frontier_bytes(s.requests.len(), 0), || {
            Some(Event::Snapshot(Snapshot {
                observed_at_ns: s.observed_at_ns,
                generation: s.generation,
                cost_model_version: s.cost_model_version,
                model_weights: s.fingerprint.model_weights,
                numerical_policy: s.fingerprint.numerical_policy,
                device_runtime: s.fingerprint.device_runtime,
                execution_config: s.fingerprint.execution_config,
                requests: frontiers(&s.requests)?,
                capacity: Fields(s.capacity),
                scope: Fields(s.scope),
                has_unmodeled_maintenance: s.has_unmodeled_maintenance,
            }))
        });
    }
    #[allow(clippy::too_many_arguments)]
    pub fn finish(
        &self,
        outcome: &'static str,
        decision: &'static str,
        reason: &'static str,
        planning_wall_ns: u64,
        budget_ns: u64,
        planner_exhausted: bool,
        hard_exhausted: bool,
        search: &PlanningSearchStats,
    ) {
        self.finish_with_audit(
            outcome,
            decision,
            reason,
            planning_wall_ns,
            budget_ns,
            planner_exhausted,
            hard_exhausted,
            search,
            true,
        );
    }
    #[allow(clippy::too_many_arguments)]
    fn finish_with_audit(
        &self,
        outcome: &'static str,
        decision: &'static str,
        reason: &'static str,
        planning_wall_ns: u64,
        budget_ns: u64,
        planner_exhausted: bool,
        hard_exhausted: bool,
        search: &PlanningSearchStats,
        audit_available: bool,
    ) {
        if self.finished.swap(true, Ordering::AcqRel) {
            return;
        }
        let event = self.writer.shared.counters.next(&self.events);
        self.writer.shared.offer(
            self.id,
            event,
            Some(size_of::<Entry>()),
            || {
                Some(Event::EndTransaction(EndTransaction {
                    audit_available,
                    audit_times_exclude_end_recording: true,
                    outcome,
                    decision,
                    reason,
                    planning_wall_ns,
                    budget_ns,
                    planner_exhausted,
                    hard_exhausted,
                    search: Fields(*search),
                }))
            },
            false,
        );
        self.writer
            .shared
            .counters
            .active_transactions
            .fetch_sub(1, Ordering::AcqRel);
    }
}
impl Drop for Transaction {
    fn drop(&mut self) {
        if !self.finished.load(Ordering::Acquire) {
            self.writer
                .shared
                .counters
                .next(&self.writer.shared.counters.abandoned_transactions);
            self.finish_with_audit(
                "abandoned",
                "unknown",
                "transaction_dropped",
                0,
                0,
                false,
                false,
                &PlanningSearchStats::default(),
                false,
            );
        }
    }
}
fn frontier_bytes(requests: usize, work: usize) -> Option<usize> {
    size_of::<Entry>()
        .checked_add(requests.checked_mul(size_of::<Frontier>())?)?
        .checked_add(work.checked_mul(size_of::<Work>())?)
}
fn frontiers(requests: &[RequestSchedulingView]) -> Option<Vec<Frontier>> {
    let mut result = Vec::new();
    result.try_reserve_exact(requests.len()).ok()?;
    result.extend(requests.iter().map(Frontier::from_request));
    Some(result)
}
impl PlanningQueryObserver for Transaction {
    fn begin_replay(&self, waves: usize) -> u64 {
        let replay = self.writer.shared.counters.next(&self.replays).unwrap_or(0);
        self.offer(Some(size_of::<Entry>()), || {
            Some(Event::ReplayBegin { replay, waves })
        });
        replay
    }
    fn end_replay(&self, replay: u64, reason: PlanningQueryAttemptEnd) {
        self.offer(Some(size_of::<Entry>()), || {
            Some(Event::ReplayEnd { replay, reason })
        });
    }
    fn selected_replay(&self, replay: u64) {
        self.offer(Some(size_of::<Entry>()), || {
            Some(Event::SelectedReplay { replay })
        });
    }
    fn begin_attempt(
        &self,
        phase: PlanningQueryPhase,
        depth: usize,
        requests: &[RequestSchedulingView],
        work: &[CandidateWork],
    ) -> u64 {
        let attempt = self
            .writer
            .shared
            .counters
            .next(&self.attempts)
            .unwrap_or(0);
        self.offer(frontier_bytes(requests.len(), work.len()), || {
            let mut rows = Vec::new();
            rows.try_reserve_exact(work.len()).ok()?;
            rows.extend(work.iter().map(|w| Work {
                request_id: w.key.request_id.clone(),
                incarnation: w.key.incarnation,
                work_generation: w.key.work_generation.get(),
                action: DebugWire(w.action.clone()),
            }));
            Some(Event::Attempt(Attempt {
                attempt,
                phase: DebugWire(phase),
                depth,
                requests: frontiers(requests)?,
                work: rows,
            }))
        });
        attempt
    }
    fn constructed(
        &self,
        key: PlanningQueryKey,
        query: Result<&StructuredQueryV2, StructuredUnknownV2>,
    ) {
        let bytes = match query {
            Ok(q) => q
                .observation_demand_scratch_bytes()
                .filter(|n| *n <= self.writer.shared.limits.max_event_bytes)
                .and_then(|_| q.observation_retained_bytes())
                .and_then(|n| n.checked_add(size_of::<Entry>())),
            Err(_) => Some(size_of::<Entry>()),
        };
        self.offer(bytes, || Some(Event::Query(key, query.cloned())));
    }
    fn lookup(&self, key: PlanningQueryKey, planning_now_ns: u64, result: PlanningObservedCost) {
        self.offer(Some(size_of::<Entry>()), || {
            Some(Event::Lookup(Lookup {
                attempt: key.attempt,
                alternative: key.alternative,
                planning_now_ns,
                cost_now_ns: result.cost_now_ns,
                outcome: result.outcome.into(),
            }))
        });
    }
    fn end_attempt(
        &self,
        attempt: u64,
        constructed: usize,
        queried: usize,
        reason: PlanningQueryAttemptEnd,
    ) {
        self.offer(Some(size_of::<Entry>()), || {
            Some(Event::EndAttempt(EndAttempt {
                attempt,
                constructed,
                queried,
                not_queried_start: queried,
                not_queried_end: constructed,
                reason: DebugWire(reason),
            }))
        });
    }
}

fn release_entry(shared: &Shared, entry: Entry) {
    let bytes = entry.bytes;
    drop(entry);
    shared.release(bytes);
}

fn consume(shared: &Shared, destination: &mut dyn Destination) {
    let mut destination_failed = false;
    let mut line = BoundedLine {
        bytes: Vec::new(),
        limit: shared.limits.max_event_bytes,
    };
    if line.bytes.try_reserve_exact(line.limit).is_err() {
        shared
            .counters
            .fail("required-query serialization buffer allocation failed");
        return;
    }
    let mut retained_ordinal = 0u64;
    loop {
        // Never register a blocking channel receiver/select waiter: try_recv
        // plus park/unpark keeps sender notification off the receiver mutex.
        if let Ok(mut entry) = shared.receiver.try_recv() {
            let Some(next_ordinal) = retained_ordinal.checked_add(1) else {
                shared.counters.overflow.store(true, Ordering::Release);
                release_entry(shared, entry);
                continue;
            };
            retained_ordinal = next_ordinal;
            entry.retained_ordinal = retained_ordinal;
            if shared.counters.failed.load(Ordering::Acquire) {
                shared.counters.loss(entry.ordinal, Loss::WriterFailed);
                release_entry(shared, entry);
                continue;
            }
            line.bytes.clear();
            let encoded = entry
                .event
                .encode(&entry, &mut line)
                .map_err(|e| e.to_string())
                .and_then(|_| line.write_all(b"\n").map_err(|e| e.to_string()));
            if let Err(error) = encoded {
                shared.counters.loss(entry.ordinal, Loss::Encoding);
                shared.counters.fail(error);
                release_entry(shared, entry);
                continue;
            }
            let written = shared.counters.file_bytes.load(Ordering::Relaxed);
            let next = written.checked_add(line.bytes.len() as u64);
            if next
                .and_then(|n| n.checked_add(FOOTER_RESERVE))
                .is_none_or(|n| n > shared.limits.max_file_bytes)
            {
                shared.counters.loss(entry.ordinal, Loss::Capacity);
                release_entry(shared, entry);
                continue;
            }
            if let Err(error) = destination.write_all(&line.bytes) {
                destination_failed = true;
                shared.counters.loss(entry.ordinal, Loss::WriterFailed);
                shared.counters.fail(error);
            } else {
                shared
                    .counters
                    .file_bytes
                    .store(next.unwrap(), Ordering::Relaxed);
                shared.counters.next(&shared.counters.written);
            }
            release_entry(shared, entry);
        } else if shared.gate.load(Ordering::Acquire) == CLOSED {
            // The earlier Empty may precede an admitted producer's publish.
            // After the release/acquire producer gate reaches zero, check again.
            if shared.receiver.is_empty() {
                break;
            }
        } else {
            thread::park();
        }
    }
    // The footer's completion is conditional on the final flush+sync, exposed
    // separately in health and close(). Missing/truncated footer is incomplete.
    if !destination_failed {
        let footer = json!({"schema":SCHEMA,"event":"run_footer","recording_complete":shared.counters.complete(),"statistics":shared.counters.wire(),"requires_successful_close":true});
        match serde_json::to_vec(&footer) {
            Ok(mut bytes) if bytes.len() as u64 + 1 <= FOOTER_RESERVE => {
                bytes.push(b'\n');
                match destination
                    .write_all(&bytes)
                    .and_then(|_| destination.flush())
                    .and_then(|_| destination.sync())
                {
                    Ok(()) => {
                        shared
                            .counters
                            .file_bytes
                            .fetch_add(bytes.len() as u64, Ordering::Relaxed);
                        shared
                            .counters
                            .footer_written
                            .store(true, Ordering::Release);
                        shared.counters.flushed.store(true, Ordering::Release);
                    }
                    Err(e) => shared.counters.fail(e),
                }
            }
            _ => shared
                .counters
                .fail("required-query footer exceeded reserved bytes"),
        }
    }
}
