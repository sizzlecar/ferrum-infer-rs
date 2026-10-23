//! Owner-local evidence for an actual support rejection. No scheduler event ID
//! or second model evaluation is introduced. All snapshots share one bounded
//! slot, including snapshots retained by callers after a newer publication.

use super::snapshot::Domain;
use crate::model::{FeatureDomain, ModelFamily, WorkFeatures, FEATURE_COUNT, FEATURE_SCHEMA};
use ferrum_execution::{
    execution_attempt::{AttemptPrefixRestore, AttemptWorkKind},
    token_execution::ExecutionRange,
};
use ferrum_scheduler::cost_prediction::{CostExecution, CostQuery, CostWork};
use serde::Serialize;
use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex,
    },
    time::Instant,
};

const AXES: [&str; FEATURE_COUNT] = [
    "participants",
    "requested_tokens",
    "decode_context_sum",
    "prefill_causal_attention_pairs",
    "confirmed_restored_tokens",
    "confirmed_internal_reuse",
    "unknown_reuse_participants",
    "prefill_participants",
    "max_context",
];

/// Counts cover calls reaching the learned owner's OutsideObservedSupport
/// branch, including repeated/cached-round-independent calls from any caller.
/// Each axis can fire on the same call; these are not disjoint percentages.
/// Counters and the slot are read independently, so an in-flight health read is
/// not an atomic conservation statement. Neither counts nor last witness are
/// joined to scheduler admission/dispatch diagnostics. Cold returned copies are
/// caller-owned and excluded from the resident recorder's byte bound.
#[derive(Debug, Serialize)]
pub struct LearningQueryRejectionStats {
    pub feature_schema: u32,
    pub axes: [&'static str; FEATURE_COUNT],
    pub outside_observed_support: u64,
    pub below_axis_counts: [u64; FEATURE_COUNT],
    pub above_axis_counts: [u64; FEATURE_COUNT],
    pub stored: u64,
    pub overwritten: u64,
    pub contention_dropped: u64,
    pub oversize_dropped: u64,
    pub max_work_items: usize,
    /// Rust object/vector payload bound, not allocator or process RSS.
    pub retained_bytes_bound: usize,
    /// `last == None` with this flag does not mean no rejection was retained.
    pub last_witness_read_busy: bool,
    pub last: Option<OutsideSupportWitness>,
}

/// The last successfully written owner slot, not necessarily the latest query
/// clock under concurrent calls. Bounds, values and revisions belong to the
/// same immutable snapshot that rejected this call. Bounds describe a box of
/// measured coordinates, not proof of coverage of every combination inside it.
#[derive(Debug, Serialize)]
pub struct OutsideSupportWitness {
    /// Explicitly unjoined to scheduler latest-failure samples or frequencies.
    pub unjoined_to_scheduler: bool,
    pub snapshot_revision: u64,
    pub model_revision: u64,
    pub feature_schema: u32,
    pub family: ModelFamily,
    pub domain: FeatureDomain,
    pub values: [f64; FEATURE_COUNT],
    pub minimum: [f64; FEATURE_COUNT],
    pub maximum: [f64; FEATURE_COUNT],
    pub below_mask: u16,
    pub above_mask: u16,
    /// None if the caller's query clock precedes this source's creation or the
    /// nanosecond representation overflows. No wall-clock join is implied.
    pub query_since_source_start_ns: Option<u64>,
    pub model_valid_for_at_query_ns: Option<u64>,
    pub execution: &'static str,
    pub work_is_exact: bool,
    pub work: Vec<RejectionWork>,
}

/// Exact cost-query work without request IDs, prompts, responses or labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct RejectionWork {
    pub kind: &'static str,
    pub requested_start: usize,
    pub requested_end: usize,
    pub context_tokens: usize,
    /// None means NoneObserved; Some(0) preserves an acknowledged zero restore.
    pub prefix_restore_tokens: Option<usize>,
    pub internally_reused_tokens: Option<usize>,
}

impl From<CostWork> for RejectionWork {
    fn from(work: CostWork) -> Self {
        Self {
            kind: match work.kind {
                AttemptWorkKind::Prefill => "prefill",
                AttemptWorkKind::Decode => "decode",
            },
            requested_start: work.requested.start(),
            requested_end: work.requested.end(),
            context_tokens: work.context_tokens,
            prefix_restore_tokens: match work.prefix_restore {
                AttemptPrefixRestore::NoneObserved => None,
                AttemptPrefixRestore::Restored { tokens } => Some(tokens),
            },
            internally_reused_tokens: work.internally_reused_tokens,
        }
    }
}

#[derive(Clone, Copy)]
struct Evidence {
    snapshot_revision: u64,
    model_revision: u64,
    feature_schema: u32,
    family: ModelFamily,
    domain: FeatureDomain,
    values: [f64; FEATURE_COUNT],
    minimum: [f64; FEATURE_COUNT],
    maximum: [f64; FEATURE_COUNT],
    below_mask: u16,
    above_mask: u16,
    query_since_source_start_ns: Option<u64>,
    model_valid_for_at_query_ns: Option<u64>,
    execution: CostExecution,
    work_is_exact: bool,
}

struct Slot {
    evidence: Option<Evidence>,
    work: Box<[CostWork]>,
    work_len: usize,
}

pub(super) struct Recorder {
    started_at: Instant,
    max_work_items: usize,
    outside: AtomicU64,
    below: [AtomicU64; FEATURE_COUNT],
    above: [AtomicU64; FEATURE_COUNT],
    stored: AtomicU64,
    overwritten: AtomicU64,
    contention_dropped: AtomicU64,
    oversize_dropped: AtomicU64,
    slot: Mutex<Slot>,
}

impl Recorder {
    /// Includes the recorder (mutex, headers, fixed evidence and counters), Arc
    /// control counters, source/current-snapshot references and the fixed work
    /// payload. Allocator metadata and
    /// cold caller-owned serialized copies are not resident Rust payloads.
    pub(super) fn retained_bytes_bound(max_work_items: usize) -> Option<usize> {
        max_work_items
            .checked_mul(std::mem::size_of::<CostWork>())?
            .checked_add(std::mem::size_of::<Self>())?
            .checked_add(2 * std::mem::size_of::<usize>())
            .and_then(|bytes| bytes.checked_add(2 * std::mem::size_of::<std::sync::Arc<Self>>()))
    }

    pub(super) fn new(max_work_items: usize, started_at: Instant) -> Self {
        let empty = CostWork {
            kind: AttemptWorkKind::Decode,
            requested: ExecutionRange::new(0, 1).expect("constant nonempty range"),
            context_tokens: 1,
            prefix_restore: AttemptPrefixRestore::NoneObserved,
            internally_reused_tokens: None,
        };
        Self {
            started_at,
            max_work_items,
            outside: AtomicU64::new(0),
            below: std::array::from_fn(|_| AtomicU64::new(0)),
            above: std::array::from_fn(|_| AtomicU64::new(0)),
            stored: AtomicU64::new(0),
            overwritten: AtomicU64::new(0),
            contention_dropped: AtomicU64::new(0),
            oversize_dropped: AtomicU64::new(0),
            slot: Mutex::new(Slot {
                evidence: None,
                work: vec![empty; max_work_items].into_boxed_slice(),
                work_len: 0,
            }),
        }
    }

    /// Called only at the existing OutsideObservedSupport branch, using its
    /// already extracted finite features and immutable model. This method never
    /// allocates, waits for the training worker/health reader or calls a model.
    pub(super) fn record(
        &self,
        snapshot_revision: u64,
        domain: &Domain,
        features: &WorkFeatures,
        query: &CostQuery<'_>,
        now: Instant,
    ) {
        self.outside.fetch_add(1, Ordering::Relaxed);
        let (minimum, maximum) = domain.model.support_bounds();
        let mut below_mask = 0;
        let mut above_mask = 0;
        for (index, value) in features.values().iter().enumerate() {
            if *value < minimum[index] {
                below_mask |= 1 << index;
                self.below[index].fetch_add(1, Ordering::Relaxed);
            }
            if *value > maximum[index] {
                above_mask |= 1 << index;
                self.above[index].fetch_add(1, Ordering::Relaxed);
            }
        }
        if query.work.len() > self.max_work_items {
            self.oversize_dropped.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let Ok(mut slot) = self.slot.try_lock() else {
            self.contention_dropped.fetch_add(1, Ordering::Relaxed);
            return;
        };
        if slot.evidence.is_some() {
            self.overwritten.fetch_add(1, Ordering::Relaxed);
        }
        slot.work[..query.work.len()].copy_from_slice(query.work);
        slot.work_len = query.work.len();
        slot.evidence = Some(Evidence {
            snapshot_revision,
            model_revision: domain.model_revision,
            feature_schema: domain.model.feature_schema(),
            family: domain.model.family(),
            domain: features.domain(),
            values: *features.values(),
            minimum: *minimum,
            maximum: *maximum,
            below_mask,
            above_mask,
            query_since_source_start_ns: now
                .checked_duration_since(self.started_at)
                .and_then(|value| value.as_nanos().try_into().ok()),
            model_valid_for_at_query_ns: domain
                .model_expires_at
                .checked_duration_since(now)
                .and_then(|value| value.as_nanos().try_into().ok()),
            execution: query.execution,
            work_is_exact: query.work_is_exact,
        });
        self.stored.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn stats(&self) -> LearningQueryRejectionStats {
        let (last_witness_read_busy, last) = match self.slot.try_lock() {
            Ok(slot) => (
                false,
                slot.evidence.map(|e| OutsideSupportWitness {
                    unjoined_to_scheduler: true,
                    snapshot_revision: e.snapshot_revision,
                    model_revision: e.model_revision,
                    feature_schema: e.feature_schema,
                    family: e.family,
                    domain: e.domain,
                    values: e.values,
                    minimum: e.minimum,
                    maximum: e.maximum,
                    below_mask: e.below_mask,
                    above_mask: e.above_mask,
                    query_since_source_start_ns: e.query_since_source_start_ns,
                    model_valid_for_at_query_ns: e.model_valid_for_at_query_ns,
                    execution: match e.execution {
                        CostExecution::SingleInvocation => "single_invocation",
                        CostExecution::SequentialPrefillThenDecode => {
                            "sequential_prefill_then_decode"
                        }
                    },
                    work_is_exact: e.work_is_exact,
                    work: slot.work[..slot.work_len]
                        .iter()
                        .copied()
                        .map(RejectionWork::from)
                        .collect(),
                }),
            ),
            Err(_) => (true, None),
        };
        LearningQueryRejectionStats {
            feature_schema: FEATURE_SCHEMA,
            axes: AXES,
            outside_observed_support: self.outside.load(Ordering::Relaxed),
            below_axis_counts: std::array::from_fn(|i| self.below[i].load(Ordering::Relaxed)),
            above_axis_counts: std::array::from_fn(|i| self.above[i].load(Ordering::Relaxed)),
            stored: self.stored.load(Ordering::Relaxed),
            overwritten: self.overwritten.load(Ordering::Relaxed),
            contention_dropped: self.contention_dropped.load(Ordering::Relaxed),
            oversize_dropped: self.oversize_dropped.load(Ordering::Relaxed),
            max_work_items: self.max_work_items,
            retained_bytes_bound: Self::retained_bytes_bound(self.max_work_items)
                .expect("validated recorder memory bound"),
            last_witness_read_busy,
            last,
        }
    }

    #[cfg(test)]
    pub(super) fn with_slot_held(&self, f: impl FnOnce()) {
        let _guard = self.slot.lock().unwrap();
        f();
    }
}
