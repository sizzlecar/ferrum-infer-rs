use super::*;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

const SEED: u64 = 0x6a09_e667_f3bc_c909;
const SEEDED_STATE_SEQUENCE_COUNT: usize = 100_000;
const MIN_TRANSITIONS_PER_SEQUENCE: usize = 8;
const MAX_TRANSITIONS_PER_SEQUENCE: usize = 24;
const MAXIMUM_ACTIVE_SEQUENCES: u32 = 4;
const MAXIMUM_TOTALS: [u64; 2] = [16, 8];
const WAIT_WITNESS_LIMIT: usize = 64;

#[derive(Clone, Copy)]
struct DeterministicRng(u64);

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        assert_ne!(seed, 0);
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.0;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        self.0 = value;
        value.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn index(&mut self, upper: usize) -> usize {
        assert!(upper > 0);
        (self.next_u64() as usize) % upper
    }

    fn inclusive(&mut self, lower: u64, upper: u64) -> u64 {
        assert!(lower <= upper);
        lower + self.next_u64() % (upper - lower + 1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpectedDecision {
    Admitted,
    Deferred,
    PermanentRejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExpectedBlocker {
    domain: Option<u32>,
    kind: CapacityShortfallKind,
    requested: u64,
    available: u64,
    current_total: u64,
    maximum_total: u64,
}

impl ExpectedBlocker {
    fn from_shortfall(shortfall: &CapacityShortfall) -> Self {
        Self {
            domain: shortfall.domain().map(CapacityDomainId::get),
            kind: shortfall.kind(),
            requested: shortfall.requested().get(),
            available: shortfall.available().get(),
            current_total: shortfall.current_total().get(),
            maximum_total: shortfall.maximum_total().get(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExpectedEvaluation {
    decision: ExpectedDecision,
    blockers: Vec<ExpectedBlocker>,
    deferred_action: Option<DeferredAction>,
}

#[derive(Debug, Clone, Copy)]
struct DemandShape {
    immediate: [u64; 2],
    fit: [u64; 2],
    fit_policy: AdmissionFitPolicy,
    pressure_action: AdmissionPressureAction,
}

impl DemandShape {
    const fn exact(units: [u64; 2]) -> Self {
        Self {
            immediate: units,
            fit: units,
            fit_policy: AdmissionFitPolicy::ImmediateOnly,
            pressure_action: AdmissionPressureAction::WaitForRelease,
        }
    }

    fn combined(self, other: Self) -> Self {
        let immediate = [
            self.immediate[0] + other.immediate[0],
            self.immediate[1] + other.immediate[1],
        ];
        let fit = [self.fit[0] + other.fit[0], self.fit[1] + other.fit[1]];
        Self {
            immediate,
            fit,
            fit_policy: if self.fit_policy == AdmissionFitPolicy::FullInputMustFit
                || other.fit_policy == AdmissionFitPolicy::FullInputMustFit
            {
                AdmissionFitPolicy::FullInputMustFit
            } else {
                AdmissionFitPolicy::ImmediateOnly
            },
            pressure_action: if self.pressure_action == AdmissionPressureAction::PreemptAndRecompute
                || other.pressure_action == AdmissionPressureAction::PreemptAndRecompute
            {
                AdmissionPressureAction::PreemptAndRecompute
            } else {
                AdmissionPressureAction::WaitForRelease
            },
        }
    }
}

struct RequestEntry {
    lease: LogicalRequestLease,
    claims: [u64; 2],
}

struct SequenceEntry {
    request_key: u64,
    lease: LogicalAdmissionLease,
    claims: [u64; 2],
}

struct ChildEntry {
    sequence_key: u64,
    lease: LogicalCapacityLease,
    claims: [u64; 2],
}

struct WaitWitness {
    condition: CapacityWaitCondition,
    registration: CapacityWaitRegistration,
}

#[derive(Debug, Default, Serialize)]
struct ModelCounters {
    action_counts: BTreeMap<&'static str, u64>,
    outcome_counts: BTreeMap<&'static str, u64>,
    action_outcome_counts: BTreeMap<String, u64>,
    deferred_action_counts: BTreeMap<&'static str, u64>,
    blocker_kind_counts: BTreeMap<&'static str, u64>,
    invariant_checks: u64,
    exact_fit_admissions: u64,
    permanent_overflow_rejections: u64,
    zero_capacity_observations: u64,
    unit_capacity_observations: u64,
    request_authority_reuses: u64,
    sequence_authority_reuses: u64,
    deferred_wait_registrations: u64,
    deferred_wait_source_changes: u64,
    invalid_resize_rejections: u64,
}

impl ModelCounters {
    fn merge(&mut self, other: Self) {
        for (key, value) in other.action_counts {
            *self.action_counts.entry(key).or_default() += value;
        }
        for (key, value) in other.outcome_counts {
            *self.outcome_counts.entry(key).or_default() += value;
        }
        for (key, value) in other.action_outcome_counts {
            *self.action_outcome_counts.entry(key).or_default() += value;
        }
        for (key, value) in other.deferred_action_counts {
            *self.deferred_action_counts.entry(key).or_default() += value;
        }
        for (key, value) in other.blocker_kind_counts {
            *self.blocker_kind_counts.entry(key).or_default() += value;
        }
        self.invariant_checks += other.invariant_checks;
        self.exact_fit_admissions += other.exact_fit_admissions;
        self.permanent_overflow_rejections += other.permanent_overflow_rejections;
        self.zero_capacity_observations += other.zero_capacity_observations;
        self.unit_capacity_observations += other.unit_capacity_observations;
        self.request_authority_reuses += other.request_authority_reuses;
        self.sequence_authority_reuses += other.sequence_authority_reuses;
        self.deferred_wait_registrations += other.deferred_wait_registrations;
        self.deferred_wait_source_changes += other.deferred_wait_source_changes;
        self.invalid_resize_rejections += other.invalid_resize_rejections;
    }
}

#[derive(Debug, Serialize)]
struct StateModelReport {
    schema_version: u32,
    seed: u64,
    seed_derivation: &'static str,
    seeded_state_sequence_count: usize,
    unique_seed_count: usize,
    scripted_state_sequence_count: usize,
    transition_count: u64,
    minimum_transitions_per_seeded_sequence: usize,
    maximum_transitions_per_seeded_sequence: usize,
    maximum_active_sequences: u32,
    maximum_totals: [u64; 2],
    counters: ModelCounters,
    max_final_active_requests: u32,
    max_final_active_sequences: u32,
    max_final_active_child_claims: u64,
    max_final_used: [u64; 2],
    leaked_resources: u64,
    poisoned_state_sequences: u64,
}

struct StateModelRunReport {
    transitions: usize,
    counters: ModelCounters,
    final_active_requests: u32,
    final_active_sequences: u32,
    final_active_child_claims: u64,
    final_used: [u64; 2],
    leaked_resources: u64,
    poisoned: bool,
}

struct AdmissionStateModel {
    coordinator: LogicalAdmissionCoordinator,
    totals: [u64; 2],
    requests: BTreeMap<u64, RequestEntry>,
    sequences: BTreeMap<u64, SequenceEntry>,
    children: BTreeMap<u64, ChildEntry>,
    next_key: u64,
    request_generations: BTreeMap<u32, u64>,
    sequence_generations: BTreeMap<u32, u64>,
    wait_witnesses: VecDeque<WaitWitness>,
    last_release_epoch: u64,
    last_capacity_epoch: u64,
    transitions: usize,
    counters: ModelCounters,
}

impl AdmissionStateModel {
    fn new() -> Self {
        let coordinator = LogicalAdmissionCoordinator::new(
            vec![
                (
                    domain(1),
                    CapacityDomainSpec::new(
                        CapacityUnits::ZERO,
                        CapacityUnits::new(MAXIMUM_TOTALS[0]),
                    )
                    .unwrap(),
                ),
                (
                    domain(2),
                    CapacityDomainSpec::new(
                        CapacityUnits::ZERO,
                        CapacityUnits::new(MAXIMUM_TOTALS[1]),
                    )
                    .unwrap(),
                ),
            ],
            MAXIMUM_ACTIVE_SEQUENCES,
        )
        .unwrap();
        let snapshot = coordinator.snapshot().unwrap();
        Self {
            coordinator,
            totals: [0, 0],
            requests: BTreeMap::new(),
            sequences: BTreeMap::new(),
            children: BTreeMap::new(),
            next_key: 1,
            request_generations: BTreeMap::new(),
            sequence_generations: BTreeMap::new(),
            wait_witnesses: VecDeque::new(),
            last_release_epoch: snapshot.release_epoch(),
            last_capacity_epoch: snapshot.capacity_epoch(),
            transitions: 0,
            counters: ModelCounters::default(),
        }
    }

    fn next_key(&mut self) -> u64 {
        let key = self.next_key;
        self.next_key += 1;
        key
    }

    fn record_action(&mut self, action: &'static str) {
        *self.counters.action_counts.entry(action).or_default() += 1;
        self.transitions += 1;
    }

    fn record_outcome(&mut self, action: &'static str, outcome: ExpectedDecision) {
        let name = match outcome {
            ExpectedDecision::Admitted => "admitted",
            ExpectedDecision::Deferred => "deferred",
            ExpectedDecision::PermanentRejected => "permanent_rejected",
        };
        *self.counters.outcome_counts.entry(name).or_default() += 1;
        *self
            .counters
            .action_outcome_counts
            .entry(format!("{action}:{name}"))
            .or_default() += 1;
    }

    fn used(&self) -> [u64; 2] {
        self.requests
            .values()
            .map(|entry| entry.claims)
            .chain(self.sequences.values().map(|entry| entry.claims))
            .chain(self.children.values().map(|entry| entry.claims))
            .fold([0, 0], |mut used, claims| {
                used[0] += claims[0];
                used[1] += claims[1];
                used
            })
    }

    fn evaluate(&self, shape: DemandShape, consumes_sequence_slot: bool) -> ExpectedEvaluation {
        let used = self.used();
        let available = [
            self.totals[0].saturating_sub(used[0]),
            self.totals[1].saturating_sub(used[1]),
        ];
        let mut permanent = Vec::new();
        for requested in [shape.immediate, shape.fit] {
            for index in 0..2 {
                if requested[index] > MAXIMUM_TOTALS[index] {
                    permanent.push(ExpectedBlocker {
                        domain: Some(index as u32 + 1),
                        kind: CapacityShortfallKind::PermanentDomainMaximum,
                        requested: requested[index],
                        available: available[index],
                        current_total: self.totals[index],
                        maximum_total: MAXIMUM_TOTALS[index],
                    });
                }
            }
        }
        permanent.sort_by_key(|blocker| (blocker.domain, blocker.kind as u8, blocker.requested));
        permanent.dedup_by(|left, right| {
            left.domain == right.domain
                && left.kind == right.kind
                && left.requested == right.requested
        });
        if !permanent.is_empty() {
            return ExpectedEvaluation {
                decision: ExpectedDecision::PermanentRejected,
                blockers: permanent,
                deferred_action: None,
            };
        }

        let mut blockers = Vec::new();
        let mut growth_required = false;
        for index in 0..2 {
            let requested = shape.immediate[index];
            if requested > self.totals[index] {
                growth_required = true;
                blockers.push(ExpectedBlocker {
                    domain: Some(index as u32 + 1),
                    kind: CapacityShortfallKind::BackingGrowthRequired,
                    requested,
                    available: available[index],
                    current_total: self.totals[index],
                    maximum_total: MAXIMUM_TOTALS[index],
                });
            } else if requested > available[index] {
                blockers.push(ExpectedBlocker {
                    domain: Some(index as u32 + 1),
                    kind: CapacityShortfallKind::ImmediateAvailability,
                    requested,
                    available: available[index],
                    current_total: self.totals[index],
                    maximum_total: MAXIMUM_TOTALS[index],
                });
            }
        }
        if shape.fit_policy == AdmissionFitPolicy::FullInputMustFit {
            for index in 0..2 {
                let requested = shape.fit[index];
                if requested > self.totals[index] {
                    growth_required = true;
                    blockers.push(ExpectedBlocker {
                        domain: Some(index as u32 + 1),
                        kind: CapacityShortfallKind::BackingGrowthRequired,
                        requested,
                        available: available[index],
                        current_total: self.totals[index],
                        maximum_total: MAXIMUM_TOTALS[index],
                    });
                } else if requested > available[index] {
                    blockers.push(ExpectedBlocker {
                        domain: Some(index as u32 + 1),
                        kind: CapacityShortfallKind::FitAvailability,
                        requested,
                        available: available[index],
                        current_total: self.totals[index],
                        maximum_total: MAXIMUM_TOTALS[index],
                    });
                }
            }
        }
        if consumes_sequence_slot && self.sequences.len() >= MAXIMUM_ACTIVE_SEQUENCES as usize {
            blockers.push(ExpectedBlocker {
                domain: None,
                kind: CapacityShortfallKind::ActiveSequenceCeiling,
                requested: 1,
                available: 0,
                current_total: u64::from(MAXIMUM_ACTIVE_SEQUENCES),
                maximum_total: u64::from(MAXIMUM_ACTIVE_SEQUENCES),
            });
        }
        if blockers.is_empty() {
            ExpectedEvaluation {
                decision: ExpectedDecision::Admitted,
                blockers,
                deferred_action: None,
            }
        } else {
            ExpectedEvaluation {
                decision: ExpectedDecision::Deferred,
                blockers,
                deferred_action: Some(if growth_required {
                    DeferredAction::AwaitBackingGrowth
                } else {
                    match shape.pressure_action {
                        AdmissionPressureAction::WaitForRelease => DeferredAction::WaitForRelease,
                        AdmissionPressureAction::PreemptAndRecompute => {
                            DeferredAction::PreemptAndRecompute
                        }
                    }
                }),
            }
        }
    }

    fn note_exact_fit(&mut self, shape: DemandShape, expected: &ExpectedEvaluation) {
        if expected.decision != ExpectedDecision::Admitted {
            return;
        }
        let used = self.used();
        if (0..2).all(|index| shape.immediate[index] == self.totals[index] - used[index]) {
            self.counters.exact_fit_admissions += 1;
        }
    }

    fn note_request_authority(&mut self, authority: RequestAuthorityId) {
        if let Some(previous) = self
            .request_generations
            .insert(authority.sparse_id(), authority.generation())
        {
            assert!(authority.generation() > previous);
            self.counters.request_authority_reuses += 1;
        }
    }

    fn note_sequence_authority(&mut self, authority: SequenceAuthorityId) {
        if let Some(previous) = self
            .sequence_generations
            .insert(authority.sparse_id(), authority.generation())
        {
            assert!(authority.generation() > previous);
            self.counters.sequence_authority_reuses += 1;
        }
    }

    fn assert_blockers(&self, expected: &ExpectedEvaluation, actual: &[CapacityShortfall]) {
        assert_eq!(
            actual
                .iter()
                .map(ExpectedBlocker::from_shortfall)
                .collect::<Vec<_>>(),
            expected.blockers
        );
    }

    fn note_deferred(&mut self, deferred: &AdmissionDeferred, expected: &ExpectedEvaluation) {
        assert_eq!(expected.decision, ExpectedDecision::Deferred);
        assert_eq!(Some(deferred.action()), expected.deferred_action);
        self.assert_blockers(expected, deferred.blockers());
        *self
            .counters
            .deferred_action_counts
            .entry(deferred_action_name(deferred.action()))
            .or_default() += 1;
        for blocker in deferred.blockers() {
            *self
                .counters
                .blocker_kind_counts
                .entry(blocker_kind_name(blocker.kind()))
                .or_default() += 1;
        }
        let registration = self
            .coordinator
            .register_waiter(deferred.wait_condition().clone())
            .unwrap();
        assert!(!registration.recheck().unwrap().should_retry());
        if self.wait_witnesses.len() == WAIT_WITNESS_LIMIT {
            self.wait_witnesses.pop_front();
        }
        self.wait_witnesses.push_back(WaitWitness {
            condition: deferred.wait_condition().clone(),
            registration,
        });
        self.counters.deferred_wait_registrations += 1;
    }

    fn note_rejected(&mut self, rejected: &AdmissionRejected, expected: &ExpectedEvaluation) {
        assert_eq!(expected.decision, ExpectedDecision::PermanentRejected);
        self.assert_blockers(expected, rejected.blockers());
        for blocker in rejected.blockers() {
            *self
                .counters
                .blocker_kind_counts
                .entry(blocker_kind_name(blocker.kind()))
                .or_default() += 1;
        }
        self.counters.permanent_overflow_rejections += 1;
    }

    fn observe_wait_source_changes(&mut self) {
        if self.wait_witnesses.is_empty() {
            return;
        }
        let mut current = Vec::with_capacity(3);
        self.coordinator
            .write_availability_epochs(&mut current)
            .unwrap();
        self.wait_witnesses.retain(|witness| {
            let changed = witness.condition.changed_since(&current).unwrap();
            assert_eq!(
                witness.registration.recheck().unwrap().should_retry(),
                changed
            );
            if changed {
                self.counters.deferred_wait_source_changes += 1;
                false
            } else {
                true
            }
        });
    }

    fn attempt_request(&mut self, shape: DemandShape) -> Option<u64> {
        self.record_action("admit_request");
        let expected = self.evaluate(shape, false);
        self.note_exact_fit(shape, &expected);
        let decision = self.coordinator.try_admit_request(&demand(shape)).unwrap();
        let key = match decision {
            RequestAdmissionDecision::Admitted(lease) => {
                assert_eq!(expected.decision, ExpectedDecision::Admitted);
                self.note_request_authority(lease.request());
                let key = self.next_key();
                assert!(self
                    .requests
                    .insert(
                        key,
                        RequestEntry {
                            lease,
                            claims: shape.immediate,
                        },
                    )
                    .is_none());
                Some(key)
            }
            RequestAdmissionDecision::Deferred(deferred) => {
                self.note_deferred(&deferred, &expected);
                None
            }
            RequestAdmissionDecision::PermanentRejected(rejected) => {
                self.note_rejected(&rejected, &expected);
                None
            }
        };
        self.record_outcome("admit_request", expected.decision);
        self.verify();
        key
    }

    fn attempt_initial_bundle(
        &mut self,
        request_shape: DemandShape,
        mut sequence_shape: DemandShape,
    ) -> Option<(u64, u64)> {
        self.record_action("admit_initial_bundle");
        sequence_shape.pressure_action = request_shape.pressure_action;
        let combined = request_shape.combined(sequence_shape);
        let expected = self.evaluate(combined, true);
        self.note_exact_fit(combined, &expected);
        let observed = self
            .coordinator
            .observe_initial_sequence(&demand(request_shape), &demand(sequence_shape))
            .unwrap();
        match observed {
            AdmissionPreflightDecision::Eligible => {
                assert_eq!(expected.decision, ExpectedDecision::Admitted)
            }
            AdmissionPreflightDecision::Deferred(deferred) => {
                self.note_deferred(&deferred, &expected)
            }
            AdmissionPreflightDecision::PermanentRejected(rejected) => {
                assert_eq!(expected.decision, ExpectedDecision::PermanentRejected);
                self.assert_blockers(&expected, rejected.blockers());
            }
        }
        let decision = self
            .coordinator
            .try_admit_initial_sequence(&demand(request_shape), &demand(sequence_shape))
            .unwrap();
        let keys = match decision {
            InitialSequenceAdmissionDecision::Admitted(bundle) => {
                assert_eq!(expected.decision, ExpectedDecision::Admitted);
                let (request, sequence) = bundle.into_parts();
                assert_eq!(request.request(), sequence.request());
                self.note_request_authority(request.request());
                self.note_sequence_authority(sequence.sequence());
                let request_key = self.next_key();
                let sequence_key = self.next_key();
                self.requests.insert(
                    request_key,
                    RequestEntry {
                        lease: request,
                        claims: request_shape.immediate,
                    },
                );
                self.sequences.insert(
                    sequence_key,
                    SequenceEntry {
                        request_key,
                        lease: sequence,
                        claims: sequence_shape.immediate,
                    },
                );
                Some((request_key, sequence_key))
            }
            InitialSequenceAdmissionDecision::Deferred => {
                assert_eq!(expected.decision, ExpectedDecision::Deferred);
                None
            }
            InitialSequenceAdmissionDecision::PermanentRejected(rejected) => {
                self.note_rejected(&rejected, &expected);
                None
            }
        };
        self.record_outcome("admit_initial_bundle", expected.decision);
        self.verify();
        keys
    }

    fn attempt_sequence(&mut self, request_key: u64, shape: DemandShape) -> Option<u64> {
        self.record_action("admit_sequence");
        let expected = self.evaluate(shape, true);
        self.note_exact_fit(shape, &expected);
        let decision = {
            let request = &self.requests[&request_key].lease;
            self.coordinator
                .try_admit_sequence_for_request(request, &demand(shape))
                .unwrap()
        };
        let key = match decision {
            AdmissionDecision::Admitted(lease) => {
                assert_eq!(expected.decision, ExpectedDecision::Admitted);
                assert_eq!(lease.request(), self.requests[&request_key].lease.request());
                self.note_sequence_authority(lease.sequence());
                let key = self.next_key();
                self.sequences.insert(
                    key,
                    SequenceEntry {
                        request_key,
                        lease,
                        claims: shape.immediate,
                    },
                );
                Some(key)
            }
            AdmissionDecision::Deferred(deferred) => {
                self.note_deferred(&deferred, &expected);
                None
            }
            AdmissionDecision::PermanentRejected(rejected) => {
                self.note_rejected(&rejected, &expected);
                None
            }
        };
        self.record_outcome("admit_sequence", expected.decision);
        self.verify();
        key
    }

    fn attempt_child(&mut self, sequence_key: u64, shape: DemandShape) -> Option<u64> {
        self.record_action("claim_child");
        let expected = self.evaluate(shape, false);
        self.note_exact_fit(shape, &expected);
        let decision = {
            let sequence = &self.sequences[&sequence_key].lease;
            self.coordinator
                .try_claim_for_sequence(sequence, &demand(shape))
                .unwrap()
        };
        let key = match decision {
            CapacityClaimDecision::Claimed(lease) => {
                assert_eq!(expected.decision, ExpectedDecision::Admitted);
                assert_eq!(
                    lease.sequence(),
                    self.sequences[&sequence_key].lease.sequence()
                );
                let key = self.next_key();
                self.children.insert(
                    key,
                    ChildEntry {
                        sequence_key,
                        lease,
                        claims: shape.immediate,
                    },
                );
                Some(key)
            }
            CapacityClaimDecision::Deferred(deferred) => {
                self.note_deferred(&deferred, &expected);
                None
            }
            CapacityClaimDecision::PermanentRejected(rejected) => {
                self.note_rejected(&rejected, &expected);
                None
            }
        };
        self.record_outcome("claim_child", expected.decision);
        self.verify();
        key
    }

    fn release_child(&mut self, key: u64) {
        self.record_action("cancel_child");
        let child = self.children.remove(&key).unwrap();
        drop(child);
        self.observe_wait_source_changes();
        self.verify();
    }

    fn release_sequence(&mut self, key: u64) {
        assert!(!self
            .children
            .values()
            .any(|child| child.sequence_key == key));
        self.record_action("cancel_sequence");
        let sequence = self.sequences.remove(&key).unwrap();
        drop(sequence);
        self.observe_wait_source_changes();
        self.verify();
    }

    fn release_request(&mut self, key: u64) {
        assert!(!self
            .sequences
            .values()
            .any(|sequence| sequence.request_key == key));
        self.record_action("cancel_request");
        let request = self.requests.remove(&key).unwrap();
        drop(request);
        self.observe_wait_source_changes();
        self.verify();
    }

    fn resize(&mut self, totals: [u64; 2]) {
        self.record_action("resize_capacity");
        let used = self.used();
        let valid = (0..2)
            .all(|index| totals[index] >= used[index] && totals[index] <= MAXIMUM_TOTALS[index]);
        let before = self.coordinator.snapshot().unwrap();
        let result = self.coordinator.set_domain_totals(&[
            (domain(1), CapacityUnits::new(totals[0])),
            (domain(2), CapacityUnits::new(totals[1])),
        ]);
        if valid {
            result.unwrap();
            self.totals = totals;
            self.observe_wait_source_changes();
        } else {
            assert!(result.is_err());
            assert_eq!(self.coordinator.snapshot().unwrap(), before);
            self.counters.invalid_resize_rejections += 1;
        }
        self.verify();
    }

    fn verify(&mut self) {
        let snapshot = self.coordinator.snapshot().unwrap();
        let used = self.used();
        assert!(!snapshot.poisoned());
        assert_eq!(snapshot.active_requests() as usize, self.requests.len());
        assert_eq!(snapshot.live_request_records(), self.requests.len());
        assert_eq!(snapshot.active_sequences() as usize, self.sequences.len());
        assert_eq!(snapshot.live_sequence_records(), self.sequences.len());
        assert_eq!(snapshot.active_child_claims() as usize, self.children.len());
        assert_eq!(
            snapshot.maximum_active_sequences(),
            MAXIMUM_ACTIVE_SEQUENCES
        );
        assert_eq!(snapshot.domains().len(), 2);
        for index in 0..2 {
            assert_eq!(snapshot.domains()[index].domain(), domain(index as u32 + 1));
            assert_eq!(snapshot.domains()[index].total().get(), self.totals[index]);
            assert_eq!(
                snapshot.domains()[index].maximum_total().get(),
                MAXIMUM_TOTALS[index]
            );
            assert_eq!(snapshot.domains()[index].used().get(), used[index]);
            assert_eq!(
                snapshot.domains()[index].available().get(),
                self.totals[index] - used[index]
            );
        }
        assert!(snapshot.release_epoch() >= self.last_release_epoch);
        assert!(snapshot.capacity_epoch() >= self.last_capacity_epoch);
        self.last_release_epoch = snapshot.release_epoch();
        self.last_capacity_epoch = snapshot.capacity_epoch();
        if self.totals.contains(&0) {
            self.counters.zero_capacity_observations += 1;
        }
        if self.totals.contains(&1) {
            self.counters.unit_capacity_observations += 1;
        }
        let request_authorities = self
            .requests
            .values()
            .map(|entry| entry.lease.request())
            .collect::<BTreeSet<_>>();
        assert_eq!(request_authorities.len(), self.requests.len());
        let sequence_authorities = self
            .sequences
            .values()
            .map(|entry| entry.lease.sequence())
            .collect::<BTreeSet<_>>();
        assert_eq!(sequence_authorities.len(), self.sequences.len());
        for sequence in self.sequences.values() {
            assert_eq!(
                sequence.lease.request(),
                self.requests[&sequence.request_key].lease.request()
            );
        }
        for child in self.children.values() {
            assert!(self.sequences.contains_key(&child.sequence_key));
            assert_eq!(
                child.lease.sequence(),
                self.sequences[&child.sequence_key].lease.sequence()
            );
        }
        self.counters.invariant_checks += 1;
    }

    fn random_shape(&self, rng: &mut DeterministicRng) -> DemandShape {
        let mode = rng.index(10);
        let used = self.used();
        let available = [
            self.totals[0].saturating_sub(used[0]),
            self.totals[1].saturating_sub(used[1]),
        ];
        match mode {
            0..=4 => {
                let immediate = [rng.inclusive(1, 3), rng.inclusive(1, 2)];
                let full_fit = rng.index(2) == 0;
                let fit = if full_fit {
                    [
                        immediate[0] + rng.inclusive(0, 2),
                        immediate[1] + rng.inclusive(0, 1),
                    ]
                } else {
                    immediate
                };
                DemandShape {
                    immediate,
                    fit,
                    fit_policy: if full_fit {
                        AdmissionFitPolicy::FullInputMustFit
                    } else {
                        AdmissionFitPolicy::ImmediateOnly
                    },
                    pressure_action: if rng.index(2) == 0 {
                        AdmissionPressureAction::WaitForRelease
                    } else {
                        AdmissionPressureAction::PreemptAndRecompute
                    },
                }
            }
            5 => DemandShape::exact([available[0].max(1), available[1].max(1)]),
            6 => DemandShape::exact([
                self.totals[0].saturating_add(1).min(MAXIMUM_TOTALS[0]),
                self.totals[1].saturating_add(1).min(MAXIMUM_TOTALS[1]),
            ]),
            7 => DemandShape::exact([MAXIMUM_TOTALS[0] + 1, 1]),
            8 => DemandShape {
                immediate: [1, 1],
                fit: [MAXIMUM_TOTALS[0] + 1, 1],
                fit_policy: AdmissionFitPolicy::FullInputMustFit,
                pressure_action: AdmissionPressureAction::WaitForRelease,
            },
            _ => DemandShape {
                immediate: [1, 1],
                fit: [available[0].saturating_add(1).max(1), 1],
                fit_policy: AdmissionFitPolicy::FullInputMustFit,
                pressure_action: AdmissionPressureAction::PreemptAndRecompute,
            },
        }
    }

    fn random_transition(&mut self, rng: &mut DeterministicRng) {
        match rng.index(10) {
            0 => {
                let shape = self.random_shape(rng);
                self.attempt_request(shape);
            }
            1 => {
                let request = self.random_shape(rng);
                let sequence = self.random_shape(rng);
                self.attempt_initial_bundle(request, sequence);
            }
            2 => {
                if let Some(key) = choose_key(&self.requests, rng) {
                    let shape = self.random_shape(rng);
                    self.attempt_sequence(key, shape);
                } else {
                    let shape = self.random_shape(rng);
                    self.attempt_request(shape);
                }
            }
            3 => {
                if let Some(key) = choose_key(&self.sequences, rng) {
                    let shape = self.random_shape(rng);
                    self.attempt_child(key, shape);
                } else {
                    let shape = self.random_shape(rng);
                    self.attempt_request(shape);
                }
            }
            4 => {
                if let Some(key) = choose_key(&self.children, rng) {
                    self.release_child(key);
                } else {
                    let shape = self.random_shape(rng);
                    self.attempt_request(shape);
                }
            }
            5 => {
                let eligible = self
                    .sequences
                    .keys()
                    .copied()
                    .filter(|key| {
                        !self
                            .children
                            .values()
                            .any(|child| child.sequence_key == *key)
                    })
                    .collect::<Vec<_>>();
                if let Some(key) = choose_slice(&eligible, rng) {
                    self.release_sequence(key);
                } else {
                    let shape = self.random_shape(rng);
                    self.attempt_request(shape);
                }
            }
            6 => {
                let eligible = self
                    .requests
                    .keys()
                    .copied()
                    .filter(|key| {
                        !self
                            .sequences
                            .values()
                            .any(|sequence| sequence.request_key == *key)
                    })
                    .collect::<Vec<_>>();
                if let Some(key) = choose_slice(&eligible, rng) {
                    self.release_request(key);
                } else {
                    let shape = self.random_shape(rng);
                    self.attempt_request(shape);
                }
            }
            7 => {
                let used = self.used();
                let totals = [
                    rng.inclusive(used[0], MAXIMUM_TOTALS[0]),
                    rng.inclusive(used[1], MAXIMUM_TOTALS[1]),
                ];
                self.resize(totals);
            }
            8 => {
                let used = self.used();
                let index = rng.index(2);
                let mut totals = self.totals;
                totals[index] = used[index].saturating_sub(1);
                self.resize(totals);
            }
            _ => {
                self.attempt_request(DemandShape::exact([MAXIMUM_TOTALS[0] + 1, 1]));
            }
        }
    }

    fn finish(mut self) -> StateModelRunReport {
        for child in std::mem::take(&mut self.children).into_values() {
            drop(child);
        }
        self.observe_wait_source_changes();
        self.verify();
        for sequence in std::mem::take(&mut self.sequences).into_values() {
            drop(sequence);
        }
        self.observe_wait_source_changes();
        self.verify();
        for request in std::mem::take(&mut self.requests).into_values() {
            drop(request);
        }
        self.observe_wait_source_changes();
        self.verify();
        let snapshot = self.coordinator.snapshot().unwrap();
        let used = [
            snapshot.domains()[0].used().get(),
            snapshot.domains()[1].used().get(),
        ];
        let leaked_resources = u64::from(snapshot.active_requests())
            + u64::from(snapshot.active_sequences())
            + snapshot.active_child_claims()
            + used.iter().sum::<u64>();
        StateModelRunReport {
            transitions: self.transitions,
            counters: self.counters,
            final_active_requests: snapshot.active_requests(),
            final_active_sequences: snapshot.active_sequences(),
            final_active_child_claims: snapshot.active_child_claims(),
            final_used: used,
            leaked_resources,
            poisoned: snapshot.poisoned(),
        }
    }
}

fn deferred_action_name(action: DeferredAction) -> &'static str {
    match action {
        DeferredAction::WaitForRelease => "wait_for_release",
        DeferredAction::AwaitBackingGrowth => "await_backing_growth",
        DeferredAction::PreemptAndRecompute => "preempt_and_recompute",
    }
}

fn blocker_kind_name(kind: CapacityShortfallKind) -> &'static str {
    match kind {
        CapacityShortfallKind::ImmediateAvailability => "immediate_availability",
        CapacityShortfallKind::FitAvailability => "fit_availability",
        CapacityShortfallKind::BackingGrowthRequired => "backing_growth_required",
        CapacityShortfallKind::ActiveSequenceCeiling => "active_sequence_ceiling",
        CapacityShortfallKind::PermanentDomainMaximum => "permanent_domain_maximum",
    }
}

fn domain(value: u32) -> CapacityDomainId {
    CapacityDomainId::new(value).unwrap()
}

fn vector(units: [u64; 2]) -> CapacityVector {
    CapacityVector::new(vec![
        CapacityEntry::new(domain(1), CapacityUnits::new(units[0])).unwrap(),
        CapacityEntry::new(domain(2), CapacityUnits::new(units[1])).unwrap(),
    ])
    .unwrap()
}

fn demand(shape: DemandShape) -> AdmissionDemand {
    AdmissionDemand::from_plan(
        vector(shape.immediate),
        vector(shape.fit),
        shape.fit_policy,
        shape.pressure_action,
    )
    .unwrap()
}

fn choose_key<T>(values: &BTreeMap<u64, T>, rng: &mut DeterministicRng) -> Option<u64> {
    values.keys().nth(rng.index(values.len().max(1))).copied()
}

fn choose_slice(values: &[u64], rng: &mut DeterministicRng) -> Option<u64> {
    (!values.is_empty()).then(|| values[rng.index(values.len())])
}

fn state_sequence_seed(ordinal: usize) -> u64 {
    let mut value = SEED ^ (ordinal as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (value ^ (value >> 31)) | 1
}

fn scripted_boundary_sequence() -> StateModelRunReport {
    let mut model = AdmissionStateModel::new();
    model.verify();

    assert!(model.attempt_request(DemandShape::exact([1, 1])).is_none());
    model.resize([1, 1]);
    let request = model
        .attempt_request(DemandShape::exact([1, 1]))
        .expect("unit capacity must admit one exact request");
    assert!(model
        .attempt_initial_bundle(DemandShape::exact([1, 1]), DemandShape::exact([1, 1]))
        .is_none());
    assert!(model
        .attempt_initial_bundle(DemandShape::exact([16, 1]), DemandShape::exact([1, 1]))
        .is_none());
    assert!(model
        .attempt_sequence(request, DemandShape::exact([1, 1]))
        .is_none());
    assert!(model
        .attempt_request(DemandShape::exact([MAXIMUM_TOTALS[0] + 1, 1]))
        .is_none());
    model.release_request(request);

    model.resize([2, 2]);
    let (request, sequence) = model
        .attempt_initial_bundle(DemandShape::exact([1, 1]), DemandShape::exact([1, 1]))
        .expect("two-unit capacity must atomically admit request and first sequence");
    assert!(model
        .attempt_child(sequence, DemandShape::exact([1, 1]))
        .is_none());
    model.release_sequence(sequence);
    model.release_request(request);

    model.resize(MAXIMUM_TOTALS);
    let request = model
        .attempt_request(DemandShape::exact([1, 1]))
        .expect("maximum backing must admit a request");
    model.resize([0, MAXIMUM_TOTALS[1]]);
    let preempt_fit = DemandShape {
        immediate: [1, 1],
        fit: MAXIMUM_TOTALS,
        fit_policy: AdmissionFitPolicy::FullInputMustFit,
        pressure_action: AdmissionPressureAction::PreemptAndRecompute,
    };
    assert!(model.attempt_request(preempt_fit).is_none());
    let mut sequences = Vec::new();
    for _ in 0..MAXIMUM_ACTIVE_SEQUENCES {
        sequences.push(
            model
                .attempt_sequence(request, DemandShape::exact([1, 1]))
                .expect("scripted sequence ceiling fill must admit"),
        );
    }
    assert!(model
        .attempt_sequence(request, DemandShape::exact([1, 1]))
        .is_none());
    assert!(model
        .attempt_sequence(request, DemandShape::exact([MAXIMUM_TOTALS[0] + 1, 1]),)
        .is_none());
    let child = model
        .attempt_child(sequences[0], DemandShape::exact([1, 1]))
        .expect("scripted child claim must admit");
    assert!(model
        .attempt_child(sequences[0], DemandShape::exact([11, 1]))
        .is_none());
    assert!(model
        .attempt_child(sequences[0], DemandShape::exact([MAXIMUM_TOTALS[0] + 1, 1]),)
        .is_none());
    model.release_child(child);
    for sequence in sequences {
        model.release_sequence(sequence);
    }
    model.release_request(request);
    model.finish()
}

#[test]
fn seeded_admission_state_model_checks_one_hundred_thousand_state_sequences() {
    let mut counters = ModelCounters::default();
    let mut transition_count = 0_u64;
    let mut leaked_resources = 0_u64;
    let mut poisoned_state_sequences = 0_u64;
    let mut max_final_active_requests = 0_u32;
    let mut max_final_active_sequences = 0_u32;
    let mut max_final_active_child_claims = 0_u64;
    let mut max_final_used = [0_u64; 2];
    let mut unique_seeds = BTreeSet::new();

    let mut absorb = |run: StateModelRunReport| {
        assert_eq!(run.leaked_resources, 0);
        assert!(!run.poisoned);
        transition_count += run.transitions as u64;
        leaked_resources += run.leaked_resources;
        poisoned_state_sequences += u64::from(run.poisoned);
        max_final_active_requests = max_final_active_requests.max(run.final_active_requests);
        max_final_active_sequences = max_final_active_sequences.max(run.final_active_sequences);
        max_final_active_child_claims =
            max_final_active_child_claims.max(run.final_active_child_claims);
        for (maximum, observed) in max_final_used.iter_mut().zip(run.final_used) {
            *maximum = (*maximum).max(observed);
        }
        counters.merge(run.counters);
    };

    absorb(scripted_boundary_sequence());
    for ordinal in 0..SEEDED_STATE_SEQUENCE_COUNT {
        let seed = state_sequence_seed(ordinal);
        assert!(
            unique_seeds.insert(seed),
            "duplicate state-model seed {seed}"
        );
        let mut rng = DeterministicRng::new(seed);
        let transition_limit = MIN_TRANSITIONS_PER_SEQUENCE
            + rng.index(MAX_TRANSITIONS_PER_SEQUENCE - MIN_TRANSITIONS_PER_SEQUENCE + 1);
        let mut model = AdmissionStateModel::new();
        model.verify();
        for _ in 0..transition_limit {
            model.random_transition(&mut rng);
        }
        absorb(model.finish());
    }
    drop(absorb);

    for action in [
        "admit_request",
        "admit_initial_bundle",
        "admit_sequence",
        "claim_child",
        "cancel_child",
        "cancel_sequence",
        "cancel_request",
        "resize_capacity",
    ] {
        assert!(
            counters.action_counts[action] > 0,
            "missing action {action}"
        );
    }
    for action in [
        "admit_request",
        "admit_initial_bundle",
        "admit_sequence",
        "claim_child",
    ] {
        for outcome in ["admitted", "deferred", "permanent_rejected"] {
            let key = format!("{action}:{outcome}");
            assert!(counters.action_outcome_counts[&key] > 0, "missing {key}");
        }
    }
    for action in [
        "wait_for_release",
        "await_backing_growth",
        "preempt_and_recompute",
    ] {
        assert!(
            counters.deferred_action_counts[action] > 0,
            "missing deferred action {action}"
        );
    }
    for blocker in [
        "immediate_availability",
        "fit_availability",
        "backing_growth_required",
        "active_sequence_ceiling",
        "permanent_domain_maximum",
    ] {
        assert!(
            counters.blocker_kind_counts[blocker] > 0,
            "missing blocker {blocker}"
        );
    }
    assert_eq!(
        counters.action_counts.values().sum::<u64>(),
        transition_count
    );
    assert!(counters.exact_fit_admissions > 0);
    assert!(counters.permanent_overflow_rejections > 0);
    assert!(counters.zero_capacity_observations > 0);
    assert!(counters.unit_capacity_observations > 0);
    assert!(counters.request_authority_reuses > 0);
    assert!(counters.sequence_authority_reuses > 0);
    assert!(counters.deferred_wait_registrations > 0);
    assert!(counters.deferred_wait_source_changes > 0);
    assert!(counters.invalid_resize_rejections > 0);
    assert_eq!(leaked_resources, 0);
    assert_eq!(poisoned_state_sequences, 0);

    let report = StateModelReport {
        schema_version: 2,
        seed: SEED,
        seed_derivation: "splitmix64(base_seed xor ordinal*golden_ratio) | 1",
        seeded_state_sequence_count: SEEDED_STATE_SEQUENCE_COUNT,
        unique_seed_count: unique_seeds.len(),
        scripted_state_sequence_count: 1,
        transition_count,
        minimum_transitions_per_seeded_sequence: MIN_TRANSITIONS_PER_SEQUENCE,
        maximum_transitions_per_seeded_sequence: MAX_TRANSITIONS_PER_SEQUENCE,
        maximum_active_sequences: MAXIMUM_ACTIVE_SEQUENCES,
        maximum_totals: MAXIMUM_TOTALS,
        counters,
        max_final_active_requests,
        max_final_active_sequences,
        max_final_active_child_claims,
        max_final_used,
        leaked_resources,
        poisoned_state_sequences,
    };
    println!(
        "FERRUM G04 STATE MODEL KEEP: {}",
        serde_json::to_string(&report).unwrap()
    );
}
