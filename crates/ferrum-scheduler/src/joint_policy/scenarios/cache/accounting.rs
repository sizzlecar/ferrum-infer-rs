//! Test-only, opt-in accounting for one existing synchronous diagnostic call.
//! Fixed thread-local report slots avoid shared test state and heap allocation.
//! This instrumentation is not a production counter or a timing baseline.

use super::*;
use serde::Serialize;
use std::cell::Cell;

#[derive(Clone, Copy, Default, Debug, Serialize)]
pub(crate) struct Bytes {
    pub slots: usize,
    pub hints: usize,
    pub generations: usize,
    pub trace: usize,
    pub work_payload: usize,
    pub service_time_payload: usize,
    pub retained: usize,
    pub sum_matches_ledger: bool,
}

#[derive(Clone, Copy, Default, Debug, Serialize)]
pub(crate) struct RejectedInsertion {
    pub before: Bytes,
    pub victim_payload_bytes: usize,
    pub proposed_retained_bytes: usize,
}

#[derive(Clone, Copy, Default, Debug, Serialize)]
pub(crate) struct TraceClears {
    pub endpoint_or_action_change: usize,
    pub preparation_failure: usize,
    pub descriptor_arithmetic: usize,
    pub index_arithmetic: usize,
    pub actual_descriptor_capacity_refused: usize,
    pub with_positive_trace_charge: usize,
    pub last_charge_before_clear: usize,
}

pub(in crate::joint_policy::scenarios) enum TraceClearReason {
    EndpointOrActionChange,
    PreparationFailure,
    DescriptorArithmetic,
    IndexArithmetic,
    ActualDescriptorCapacityRefused,
}

#[derive(Clone, Copy, Default, Debug, Serialize)]
pub(crate) struct Counts {
    pub known_insert_attempts: usize,
    pub empty_insertions: usize,
    pub exact_key_replacements: usize,
    pub collision_replacements: usize,
    pub byte_limit_rejections: usize,
    pub arithmetic_rejections: usize,
    pub occupied_slots: usize,
    pub first_byte_rejection: Option<RejectedInsertion>,
    pub last_byte_rejection: Option<RejectedInsertion>,
    pub trace_clears: TraceClears,
}

#[derive(Clone, Copy, Default, Debug, Serialize)]
pub(crate) struct Layout {
    pub entry: usize,
    pub option_entry: usize,
    pub cache_handle: usize,
    pub option_cache_handle: usize,
    pub wave_trace: usize,
    pub cost_work: usize,
    pub duration: usize,
}

#[derive(Clone, Copy, Default, Debug, Serialize)]
pub(crate) struct Report {
    /// Last retained-allocation ledger at cache teardown. Trace storage may
    /// already have been freed by normal evaluator local-drop order.
    pub at_cache_teardown_ledger: Bytes,
    pub counts: Counts,
    pub slots: usize,
    pub max_bytes: usize,
    pub hits: usize,
    pub hint_hits: usize,
    pub hash_lookups: usize,
    pub handle_hits: usize,
    pub layout: Layout,
}

#[derive(Clone, Copy, Default, Debug, Serialize)]
pub(crate) struct Capture {
    pub reports: [Option<Report>; 2],
    pub dropped_reports: usize,
}

#[derive(Clone, Copy)]
struct State {
    id: u64,
    capture: Capture,
}

thread_local! {
    static CURRENT: Cell<Option<State>> = const { Cell::new(None) };
    static NEXT: Cell<u64> = const { Cell::new(1) };
}

struct Restore(Option<State>);
impl Drop for Restore {
    fn drop(&mut self) {
        CURRENT.set(self.0);
    }
}

pub(crate) fn capture<T>(body: impl FnOnce() -> T) -> (T, Capture) {
    let id = NEXT.with(|next| {
        let id = next.get();
        next.set(
            id.checked_add(1)
                .expect("test-only capture identity exhausted"),
        );
        id
    });
    let restore = Restore(CURRENT.replace(Some(State {
        id,
        capture: Capture::default(),
    })));
    let result = body();
    let report = CURRENT.take().expect("live test-only capture").capture;
    drop(restore);
    (result, report)
}

pub(super) struct Counters {
    scope: u64,
    thread: std::thread::ThreadId,
    pub counts: Counts,
    pub work_payload: usize,
    pub service_time_payload: usize,
}

impl Counters {
    pub fn new() -> Option<Self> {
        CURRENT.get().map(|state| Self {
            scope: state.id,
            thread: std::thread::current().id(),
            counts: Counts::default(),
            work_payload: 0,
            service_time_payload: 0,
        })
    }

    pub fn publish(&self, report: Report) {
        if std::thread::current().id() != self.thread {
            return;
        }
        let Some(mut state) = CURRENT.get().filter(|state| state.id == self.scope) else {
            return;
        };
        if let Some(slot) = state.capture.reports.iter_mut().find(|slot| slot.is_none()) {
            *slot = Some(report);
        } else {
            state.capture.dropped_reports = state.capture.dropped_reports.saturating_add(1);
        }
        CURRENT.set(Some(state));
    }
}

impl ScenarioCache {
    pub(super) fn accounting_bytes(&self) -> Bytes {
        let Some(counters) = &self.accounting else {
            return Bytes::default();
        };
        let mut result = Bytes {
            slots: std::mem::size_of_val(self.slots.as_ref()),
            hints: std::mem::size_of_val(self.hints.as_ref()),
            generations: std::mem::size_of_val(self.generations.as_ref()),
            trace: self.trace_bytes,
            work_payload: counters.work_payload,
            service_time_payload: counters.service_time_payload,
            retained: self.retained_bytes,
            sum_matches_ledger: false,
        };
        result.sum_matches_ledger = [
            result.slots,
            result.hints,
            result.generations,
            result.trace,
            result.work_payload,
            result.service_time_payload,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
            == Some(result.retained);
        result
    }

    pub(in crate::joint_policy::scenarios) fn note_trace_clear(
        &mut self,
        reason: TraceClearReason,
    ) {
        let Some(counters) = &mut self.accounting else {
            return;
        };
        let clear = &mut counters.counts.trace_clears;
        let count = match reason {
            TraceClearReason::EndpointOrActionChange => &mut clear.endpoint_or_action_change,
            TraceClearReason::PreparationFailure => &mut clear.preparation_failure,
            TraceClearReason::DescriptorArithmetic => &mut clear.descriptor_arithmetic,
            TraceClearReason::IndexArithmetic => &mut clear.index_arithmetic,
            TraceClearReason::ActualDescriptorCapacityRefused => {
                &mut clear.actual_descriptor_capacity_refused
            }
        };
        *count = count.saturating_add(1);
        if self.trace_bytes > 0 {
            clear.with_positive_trace_charge = clear.with_positive_trace_charge.saturating_add(1);
        }
        clear.last_charge_before_clear = self.trace_bytes;
    }
}

impl Drop for ScenarioCache {
    fn drop(&mut self) {
        let Some(counters) = &self.accounting else {
            return;
        };
        // No slot scan, heap allocation, serialization or global shared lock.
        counters.publish(Report {
            at_cache_teardown_ledger: self.accounting_bytes(),
            counts: counters.counts,
            slots: self.slots.len(),
            max_bytes: self.max_bytes,
            hits: self.hits,
            hint_hits: self.hint_hits,
            hash_lookups: self.hash_lookups,
            handle_hits: self.handle_hits,
            layout: Layout {
                entry: std::mem::size_of::<Entry>(),
                option_entry: std::mem::size_of::<Option<Entry>>(),
                cache_handle: std::mem::size_of::<CacheHandle>(),
                option_cache_handle: std::mem::size_of::<Option<CacheHandle>>(),
                wave_trace: super::super::trace::wave_trace_size(),
                cost_work: std::mem::size_of::<CostWork>(),
                duration: std::mem::size_of::<Duration>(),
            },
        });
    }
}
