//! Bounded diagnostic history. Never consulted by scheduling or dispatch.
use std::sync::OnceLock;

use ferrum_interfaces::vnext::SubmissionWaveStructureObservation;
use parking_lot::Mutex;
use serde_json::json;

struct Counts {
    attempts: u64,
    completed: u64,
    failed_or_cancelled: u64,
    retries: u64,
    nondecode: u64,
    discontinuities: u64,
    missing_observation: u64,
    no_previous: u64,
    comparable_pairs: u64,
    same_observed_structure: u64,
    current_run: u64,
    longest_run: u64,
    resident_observations: u64,
    nonresident_observations: u64,
    preparation_gaps: u64,
    changes: [u64; 64],
}

impl Default for Counts {
    fn default() -> Self {
        Self {
            attempts: 0,
            completed: 0,
            failed_or_cancelled: 0,
            retries: 0,
            nondecode: 0,
            discontinuities: 0,
            missing_observation: 0,
            no_previous: 0,
            comparable_pairs: 0,
            same_observed_structure: 0,
            current_run: 0,
            longest_run: 0,
            resident_observations: 0,
            nonresident_observations: 0,
            preparation_gaps: 0,
            changes: [0; 64],
        }
    }
}

#[derive(Clone, Copy)]
struct Ticket {
    generation: u64,
    reset: u64,
}

struct History<T> {
    previous: Option<T>,
    counts: Counts,
    generation: u64,
    reset: u64,
    active: usize,
}

impl<T> Default for History<T> {
    fn default() -> Self {
        Self {
            previous: None,
            counts: Counts::default(),
            generation: 0,
            reset: 0,
            active: 0,
        }
    }
}

impl<T> History<T> {
    fn break_chain(&mut self) {
        self.previous = None;
        self.counts.current_run = 0;
        self.generation = self.generation.saturating_add(1);
    }

    fn begin(&mut self) -> Ticket {
        self.counts.attempts += 1;
        if self.active != 0 {
            self.counts.discontinuities += 1;
            self.break_chain();
        }
        self.active += 1;
        Ticket {
            generation: self.generation,
            reset: self.reset,
        }
    }

    fn finish(
        &mut self,
        ticket: Ticket,
        observation: Option<T>,
        compare: impl FnOnce(&T, &T) -> u8,
    ) {
        if ticket.reset != self.reset || self.reset == u64::MAX {
            return;
        }
        self.active = self.active.saturating_sub(1);
        self.counts.completed += 1;
        if ticket.generation != self.generation || self.active != 0 || self.generation == u64::MAX {
            self.counts.discontinuities += 1;
            self.break_chain();
            return;
        }
        let Some(observation) = observation else {
            self.counts.missing_observation += 1;
            self.break_chain();
            return;
        };
        if let Some(previous) = &self.previous {
            let mask = compare(&observation, previous);
            self.counts.comparable_pairs += 1;
            self.counts.changes[usize::from(mask)] += 1;
            if mask == 0 {
                self.counts.same_observed_structure += 1;
                self.counts.current_run += 1;
                self.counts.longest_run = self.counts.longest_run.max(self.counts.current_run);
            } else {
                self.counts.current_run = 0;
            }
        } else {
            self.counts.no_previous += 1;
        }
        self.previous = Some(observation);
    }

    fn abandon(&mut self, ticket: Ticket) {
        if ticket.reset != self.reset || self.reset == u64::MAX {
            return;
        }
        self.active = self.active.saturating_sub(1);
        self.counts.failed_or_cancelled += 1;
        self.break_chain();
    }

    fn reset(&mut self) {
        self.break_chain();
        self.reset = self.reset.saturating_add(1);
        self.active = 0;
        self.counts = Counts::default();
    }
}

#[derive(Default)]
pub(super) struct DecodeStructureMetrics {
    history: Mutex<History<SubmissionWaveStructureObservation>>,
}

impl DecodeStructureMetrics {
    pub(super) fn preparation(&self) -> PreparationBoundary<'_> {
        PreparationBoundary {
            metrics: self,
            reset: self.history.lock().reset,
            handed_off: false,
        }
    }
    pub(super) fn begin(&self) -> DecodeStructureAttempt<'_> {
        let ticket = self.history.lock().begin();
        DecodeStructureAttempt {
            metrics: self,
            ticket,
            observation: OnceLock::new(),
            finished: false,
        }
    }

    pub(super) fn nondecode(&self) {
        let mut state = self.history.lock();
        state.counts.nondecode += 1;
        state.break_chain();
    }

    pub(super) fn reset(&self) {
        self.history.lock().reset();
    }

    pub(super) fn snapshot(&self) -> serde_json::Value {
        let state = self.history.lock();
        let c = &state.counts;
        json!({
            "scope": "host_dispatch_timing_enabled_all_decode_execution_attempts_not_journal_sampling",
            "unit": "wave_not_node",
            "prepared_decode_executions": c.attempts, "successful_terminal_and_retired": c.completed,
            "failed_or_cancelled": c.failed_or_cancelled, "dns_retries": c.retries,
            "nondecode_boundaries": c.nondecode, "discontinuities": c.discontinuities,
            "preparation_failed_deferred_or_cancelled_boundaries": c.preparation_gaps,
            "missing_observation": c.missing_observation, "no_previous": c.no_previous,
            "comparable_successful_pairs": c.comparable_pairs,
            "same_observed_structure": c.same_observed_structure,
            "uninterrupted_successful_resident_program_observations": c.resident_observations,
            "uninterrupted_successful_no_resident_program_observations": c.nonresident_observations,
            "longest_unchanged_transition_run": c.longest_run,
            "joint_change_mask_counts": c.changes.as_slice(),
            "mask_bits": {"program_or_lane": 1, "dimensions": 2, "participants": 4,
                "physical_slots": 8, "captured_extents": 16, "plan": 32},
            "actual_chunk_owner": "unknown", "runtime_descriptor_immutability": "unknown",
            "provider_plan_immutability": "unknown", "graph_exec_identity": "unknown",
            "limitations": "Captured claim metadata only; no sealed authorization or hit rate. Same counts may include nonresident programs. Frame, attempt, claim issuance, logical used size and SequenceBackingGeneration are excluded; physical chunk generation remains. Explicit dimensions exclude pages but the exact program ID retains its pages field. Logical KV block crossings are not observed. Step preparation and sequence-extension errors/deferrals break adjacency outside the execution denominator; other outer prechecks are not observed. DNS retries stay within one prepared execution and break adjacency. One previous snapshot only; no IDs or token content are exported."
        })
    }
}

pub(super) struct PreparationBoundary<'a> {
    metrics: &'a DecodeStructureMetrics,
    reset: u64,
    handed_off: bool,
}

impl PreparationBoundary<'_> {
    pub(super) fn hand_off(mut self) {
        self.handed_off = true;
    }
}

impl Drop for PreparationBoundary<'_> {
    fn drop(&mut self) {
        if !self.handed_off {
            let mut state = self.metrics.history.lock();
            if self.reset == state.reset {
                state.counts.preparation_gaps += 1;
                state.break_chain();
            }
        }
    }
}

pub(super) struct DecodeStructureAttempt<'a> {
    metrics: &'a DecodeStructureMetrics,
    ticket: Ticket,
    observation: OnceLock<SubmissionWaveStructureObservation>,
    finished: bool,
}

impl DecodeStructureAttempt<'_> {
    pub(super) fn wants_observation(&self) -> bool {
        self.observation.get().is_none()
    }

    pub(super) fn observe(&self, observation: SubmissionWaveStructureObservation) {
        let _ = self.observation.set(observation);
    }

    pub(super) fn retry(&self) {
        let mut state = self.metrics.history.lock();
        if self.ticket.reset != state.reset {
            return;
        }
        state.counts.retries += 1;
        state.break_chain();
    }

    pub(super) fn complete(mut self) {
        let mut state = self.metrics.history.lock();
        if self.ticket.reset == state.reset
            && self.ticket.generation == state.generation
            && state.active == 1
            && state.generation != u64::MAX
        {
            if let Some(observation) = self.observation.get() {
                if observation.resident_program_selected() {
                    state.counts.resident_observations += 1;
                } else {
                    state.counts.nonresident_observations += 1;
                }
            }
        }
        state.finish(self.ticket, self.observation.take(), |next, previous| {
            next.change_mask(previous)
        });
        self.finished = true;
    }
}

impl Drop for DecodeStructureAttempt<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.metrics.history.lock().abandon(self.ticket);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn finish(h: &mut History<u8>, t: Ticket, value: u8) {
        h.finish(t, Some(value), |a, b| a ^ b);
    }

    #[test]
    fn decode_structure_publishes_only_success_and_joint_changes() {
        let mut h = History::default();
        let a = h.begin();
        assert!(h.previous.is_none());
        finish(&mut h, a, 0);
        let b = h.begin();
        finish(&mut h, b, 0);
        let c = h.begin();
        finish(&mut h, c, 5);
        assert_eq!(h.counts.same_observed_structure, 1);
        assert_eq!(h.counts.changes[5], 1);
        let failed = h.begin();
        h.abandon(failed);
        let d = h.begin();
        finish(&mut h, d, 5);
        assert_eq!(h.counts.comparable_pairs, 2);
        assert_eq!(h.counts.no_previous, 2);
    }

    #[test]
    fn decode_structure_overlap_and_retry_cannot_publish_old_candidates() {
        let mut h = History::default();
        let a = h.begin();
        let b = h.begin();
        finish(&mut h, b, 1);
        finish(&mut h, a, 1);
        assert!(h.previous.is_none());
        let c = h.begin();
        h.break_chain();
        finish(&mut h, c, 1);
        assert!(h.previous.is_none());
    }

    #[test]
    fn decode_structure_reset_clears_history_and_ignores_old_inflight_drop() {
        let mut h = History::default();
        let a = h.begin();
        finish(&mut h, a, 1);
        let old = h.begin();
        h.reset();
        let fresh = h.begin();
        finish(&mut h, fresh, 1);
        h.abandon(old);
        assert_eq!(h.previous, Some(1));
        assert_eq!(h.counts.attempts, 1);
        assert_eq!(h.counts.comparable_pairs, 0);
        assert_eq!(h.counts.failed_or_cancelled, 0);
    }

    #[test]
    fn decode_structure_raii_cancel_missing_observation_and_startup_reset() {
        let metrics = DecodeStructureMetrics::default();
        drop(metrics.begin());
        assert_eq!(metrics.snapshot()["failed_or_cancelled"], 1);
        metrics.begin().complete();
        assert_eq!(metrics.snapshot()["missing_observation"], 1);
        let old = metrics.begin();
        metrics.reset();
        drop(old);
        assert_eq!(metrics.snapshot()["prepared_decode_executions"], 0);
        assert_eq!(metrics.snapshot()["failed_or_cancelled"], 0);
    }

    #[test]
    fn decode_structure_preparation_handoff_and_gap_are_separate_from_executions() {
        let metrics = DecodeStructureMetrics::default();
        metrics.preparation().hand_off();
        assert_eq!(
            metrics.snapshot()["preparation_failed_deferred_or_cancelled_boundaries"],
            0
        );
        drop(metrics.preparation());
        assert_eq!(
            metrics.snapshot()["preparation_failed_deferred_or_cancelled_boundaries"],
            1
        );
        assert_eq!(metrics.snapshot()["prepared_decode_executions"], 0);
    }
}
