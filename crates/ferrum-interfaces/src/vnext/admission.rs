use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use tokio::sync::watch;

use super::{DynamicAdmissionFaultKind, VNextError};

fn invalid_admission(reason: impl Into<String>) -> VNextError {
    admission_fault(DynamicAdmissionFaultKind::InvalidContract, reason)
}

fn admission_fault(kind: DynamicAdmissionFaultKind, reason: impl Into<String>) -> VNextError {
    VNextError::DynamicAdmissionContract {
        kind,
        reason: reason.into(),
    }
}

static NEXT_COORDINATOR_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct LogicalAdmissionCoordinatorId(u64);

impl LogicalAdmissionCoordinatorId {
    fn issue() -> Result<Self, VNextError> {
        NEXT_COORDINATOR_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map(Self)
            .map_err(|_| {
                admission_fault(
                    DynamicAdmissionFaultKind::AuthorityExhausted,
                    "logical admission coordinator id is exhausted",
                )
            })
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct CapacityDomainId(u32);

impl CapacityDomainId {
    pub fn new(value: u32) -> Result<Self, VNextError> {
        if value == 0 {
            return Err(invalid_admission("capacity domain id must be non-zero"));
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct CapacityUnits(u64);

impl CapacityUnits {
    pub const ZERO: Self = Self(0);

    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CapacityDomainSpec {
    total_units: CapacityUnits,
    maximum_total_units: CapacityUnits,
}

impl CapacityDomainSpec {
    pub fn new(
        total_units: CapacityUnits,
        maximum_total_units: CapacityUnits,
    ) -> Result<Self, VNextError> {
        if total_units.get() > maximum_total_units.get() {
            return Err(invalid_admission(
                "capacity domain total exceeds maximum total",
            ));
        }
        Ok(Self {
            total_units,
            maximum_total_units,
        })
    }

    pub const fn total_units(self) -> CapacityUnits {
        self.total_units
    }

    pub const fn maximum_total_units(self) -> CapacityUnits {
        self.maximum_total_units
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CapacityEntry {
    domain: CapacityDomainId,
    units: CapacityUnits,
}

impl CapacityEntry {
    pub fn new(domain: CapacityDomainId, units: CapacityUnits) -> Result<Self, VNextError> {
        if units.get() == 0 {
            return Err(invalid_admission("capacity demand units must be non-zero"));
        }
        Ok(Self { domain, units })
    }

    pub const fn domain(self) -> CapacityDomainId {
        self.domain
    }

    pub const fn units(self) -> CapacityUnits {
        self.units
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct CapacityVector(Vec<CapacityEntry>);

impl CapacityVector {
    pub fn new(mut entries: Vec<CapacityEntry>) -> Result<Self, VNextError> {
        entries.sort_by_key(|entry| entry.domain);
        if entries.is_empty() {
            return Err(invalid_admission("capacity vector must be non-empty"));
        }
        if entries
            .windows(2)
            .any(|pair| pair[0].domain == pair[1].domain)
        {
            return Err(invalid_admission(
                "capacity vector contains a duplicate domain",
            ));
        }
        Ok(Self(entries))
    }

    pub fn entries(&self) -> &[CapacityEntry] {
        &self.0
    }

    pub(crate) const fn empty() -> Self {
        Self(Vec::new())
    }

    pub const fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn units_for(&self, domain: CapacityDomainId) -> Option<CapacityUnits> {
        self.0
            .binary_search_by_key(&domain, |entry| entry.domain)
            .ok()
            .map(|index| self.0[index].units)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionFitPolicy {
    ImmediateOnly,
    /// Requires current capacity to fit the declared input/frontier, but does
    /// not reserve capacity beyond `immediate_claim`.
    FullInputMustFit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionPressureAction {
    WaitForRelease,
    PreemptAndRecompute,
}

/// Opaque demand derived by the execution plan. Product/backend callers can
/// inspect it but cannot construct or deserialize a lower demand.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AdmissionDemand {
    immediate_claim: CapacityVector,
    fit_requirement: CapacityVector,
    fit_policy: AdmissionFitPolicy,
    pressure_action: AdmissionPressureAction,
}

impl AdmissionDemand {
    pub(crate) fn from_plan(
        immediate_claim: CapacityVector,
        fit_requirement: CapacityVector,
        fit_policy: AdmissionFitPolicy,
        pressure_action: AdmissionPressureAction,
    ) -> Result<Self, VNextError> {
        for immediate in immediate_claim.entries() {
            let Some(fit_units) = fit_requirement.units_for(immediate.domain) else {
                return Err(invalid_admission(
                    "fit requirement omits an immediate-claim domain",
                ));
            };
            if fit_units.get() < immediate.units.get() {
                return Err(invalid_admission(
                    "fit requirement is smaller than the immediate claim",
                ));
            }
        }
        if fit_policy == AdmissionFitPolicy::ImmediateOnly && fit_requirement != immediate_claim {
            return Err(invalid_admission(
                "immediate-only admission requires identical immediate and fit vectors",
            ));
        }
        Ok(Self {
            immediate_claim,
            fit_requirement,
            fit_policy,
            pressure_action,
        })
    }

    pub fn immediate_claim(&self) -> &CapacityVector {
        &self.immediate_claim
    }

    pub fn fit_requirement(&self) -> &CapacityVector {
        &self.fit_requirement
    }

    pub const fn fit_policy(&self) -> AdmissionFitPolicy {
        self.fit_policy
    }

    pub const fn pressure_action(&self) -> AdmissionPressureAction {
        self.pressure_action
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CapacityShortfallKind {
    ImmediateAvailability,
    FitAvailability,
    BackingGrowthRequired,
    ActiveSequenceCeiling,
    PermanentDomainMaximum,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CapacityShortfall {
    domain: Option<CapacityDomainId>,
    kind: CapacityShortfallKind,
    requested: CapacityUnits,
    available: CapacityUnits,
    current_total: CapacityUnits,
    maximum_total: CapacityUnits,
}

impl CapacityShortfall {
    pub const fn domain(&self) -> Option<CapacityDomainId> {
        self.domain
    }

    pub const fn kind(&self) -> CapacityShortfallKind {
        self.kind
    }

    pub const fn requested(&self) -> CapacityUnits {
        self.requested
    }

    pub const fn available(&self) -> CapacityUnits {
        self.available
    }

    pub const fn current_total(&self) -> CapacityUnits {
        self.current_total
    }

    pub const fn maximum_total(&self) -> CapacityUnits {
        self.maximum_total
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DomainCapacitySnapshot {
    domain: CapacityDomainId,
    total: CapacityUnits,
    maximum_total: CapacityUnits,
    used: CapacityUnits,
    available: CapacityUnits,
}

impl DomainCapacitySnapshot {
    pub const fn domain(&self) -> CapacityDomainId {
        self.domain
    }

    pub const fn total(&self) -> CapacityUnits {
        self.total
    }

    pub const fn maximum_total(&self) -> CapacityUnits {
        self.maximum_total
    }

    pub const fn used(&self) -> CapacityUnits {
        self.used
    }

    pub const fn available(&self) -> CapacityUnits {
        self.available
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CapacitySnapshot {
    coordinator_id: LogicalAdmissionCoordinatorId,
    domains: Vec<DomainCapacitySnapshot>,
    active_requests: u32,
    active_sequences: u32,
    active_child_claims: u64,
    maximum_active_sequences: u32,
    release_epoch: u64,
    capacity_epoch: u64,
    live_sequence_records: usize,
    reusable_sequence_ids: usize,
    live_request_records: usize,
    reusable_request_ids: usize,
    poisoned: bool,
}

impl CapacitySnapshot {
    pub const fn coordinator_id(&self) -> LogicalAdmissionCoordinatorId {
        self.coordinator_id
    }

    pub fn domains(&self) -> &[DomainCapacitySnapshot] {
        &self.domains
    }

    pub const fn active_sequences(&self) -> u32 {
        self.active_sequences
    }

    pub const fn active_requests(&self) -> u32 {
        self.active_requests
    }

    pub const fn active_child_claims(&self) -> u64 {
        self.active_child_claims
    }

    pub const fn maximum_active_sequences(&self) -> u32 {
        self.maximum_active_sequences
    }

    pub const fn release_epoch(&self) -> u64 {
        self.release_epoch
    }

    pub const fn capacity_epoch(&self) -> u64 {
        self.capacity_epoch
    }

    pub const fn live_sequence_records(&self) -> usize {
        self.live_sequence_records
    }

    pub const fn reusable_sequence_ids(&self) -> usize {
        self.reusable_sequence_ids
    }

    pub const fn live_request_records(&self) -> usize {
        self.live_request_records
    }

    pub const fn reusable_request_ids(&self) -> usize {
        self.reusable_request_ids
    }

    pub const fn poisoned(&self) -> bool {
        self.poisoned
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CapacityEpochs {
    coordinator_id: LogicalAdmissionCoordinatorId,
    release_epoch: u64,
    capacity_epoch: u64,
}

impl CapacityEpochs {
    pub const fn coordinator_id(self) -> LogicalAdmissionCoordinatorId {
        self.coordinator_id
    }

    pub const fn release_epoch(self) -> u64 {
        self.release_epoch
    }

    pub const fn capacity_epoch(self) -> u64 {
        self.capacity_epoch
    }
}

/// One independently changing source that can make a deferred capacity
/// decision worth recomputing. Global release/capacity epochs remain audit
/// versions; scheduler retry eligibility is derived from these exact sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CapacityAvailabilitySource {
    Domain(CapacityDomainId),
    ActiveSequenceSlots,
    PlanDeviceBudget,
    ProcessDeviceCapacity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct CapacityAvailabilityEpoch {
    source: CapacityAvailabilitySource,
    epoch: u64,
}

impl CapacityAvailabilityEpoch {
    pub fn new(source: CapacityAvailabilitySource, epoch: u64) -> Result<Self, VNextError> {
        if epoch == 0 {
            return Err(invalid_admission(
                "capacity availability epoch must be non-zero",
            ));
        }
        Ok(Self { source, epoch })
    }

    pub const fn source(self) -> CapacityAvailabilitySource {
        self.source
    }

    pub const fn epoch(self) -> u64 {
        self.epoch
    }
}

/// Exact, non-authoritative retry predicate captured with one deferral.
/// Copying this value cannot allocate capacity; it can only suppress or permit
/// a later authoritative admission probe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CapacityWaitCondition {
    coordinator_id: LogicalAdmissionCoordinatorId,
    observed: Vec<CapacityAvailabilityEpoch>,
}

impl CapacityWaitCondition {
    pub fn from_observation(
        coordinator_id: u64,
        observed: Vec<CapacityAvailabilityEpoch>,
    ) -> Result<Self, VNextError> {
        if coordinator_id == 0 {
            return Err(invalid_admission(
                "capacity wait coordinator id must be non-zero",
            ));
        }
        Self::new(LogicalAdmissionCoordinatorId(coordinator_id), observed)
    }

    pub fn new(
        coordinator_id: LogicalAdmissionCoordinatorId,
        mut observed: Vec<CapacityAvailabilityEpoch>,
    ) -> Result<Self, VNextError> {
        if observed.is_empty() {
            return Err(invalid_admission(
                "capacity wait condition requires at least one availability source",
            ));
        }
        observed.sort_by_key(|entry| entry.source);
        if observed
            .windows(2)
            .any(|pair| pair[0].source == pair[1].source)
        {
            return Err(invalid_admission(
                "capacity wait condition contains a duplicate availability source",
            ));
        }
        Ok(Self {
            coordinator_id,
            observed,
        })
    }

    pub const fn coordinator_id(&self) -> LogicalAdmissionCoordinatorId {
        self.coordinator_id
    }

    pub fn observed(&self) -> &[CapacityAvailabilityEpoch] {
        &self.observed
    }

    pub fn validate_sources_present(
        &self,
        current: &[CapacityAvailabilityEpoch],
    ) -> Result<(), VNextError> {
        if current
            .windows(2)
            .any(|pair| pair[0].source >= pair[1].source)
        {
            return Err(invalid_admission(
                "capacity availability snapshot is not canonical",
            ));
        }
        for observed in &self.observed {
            current
                .binary_search_by_key(&observed.source, |entry| entry.source)
                .map_err(|_| {
                    invalid_admission("capacity availability snapshot omitted a waited source")
                })?;
        }
        Ok(())
    }

    pub fn changed_since(&self, current: &[CapacityAvailabilityEpoch]) -> Result<bool, VNextError> {
        self.validate_sources_present(current)?;
        let mut changed = false;
        for observed in &self.observed {
            let index = current
                .binary_search_by_key(&observed.source, |entry| entry.source)
                .expect("validated availability source remains present");
            let current_epoch = current[index].epoch;
            if current_epoch < observed.epoch {
                return Err(admission_fault(
                    DynamicAdmissionFaultKind::EpochRegression,
                    "capacity availability epoch regressed",
                ));
            }
            changed |= current_epoch > observed.epoch;
        }
        Ok(changed)
    }

    pub(crate) fn refreshed_from(
        &self,
        current: &[CapacityAvailabilityEpoch],
    ) -> Result<Self, VNextError> {
        self.validate_sources_present(current)?;
        Self::new(
            self.coordinator_id,
            self.observed
                .iter()
                .map(|observed| {
                    let index = current
                        .binary_search_by_key(&observed.source, |entry| entry.source)
                        .expect("validated availability source remains present");
                    current[index]
                })
                .collect(),
        )
    }
}

/// One coherent observation used to publish a deferred capacity decision.
///
/// The audit epochs and exact retry predicate are sampled while holding the
/// coordinator lock once. Callers that inspect a second resource owner must
/// capture this snapshot before that inspection, so a release racing with the
/// inspection is either visible to the inspection or advances this predicate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CapacityWaitSnapshot {
    epochs: CapacityEpochs,
    wait_condition: CapacityWaitCondition,
}

impl CapacityWaitSnapshot {
    fn new(epochs: CapacityEpochs, wait_condition: CapacityWaitCondition) -> Self {
        debug_assert_eq!(epochs.coordinator_id(), wait_condition.coordinator_id());
        Self {
            epochs,
            wait_condition,
        }
    }

    pub const fn epochs(&self) -> CapacityEpochs {
        self.epochs
    }

    pub fn wait_condition(&self) -> &CapacityWaitCondition {
        &self.wait_condition
    }

    pub(crate) fn narrow_to_domains(
        self,
        domains: impl IntoIterator<Item = CapacityDomainId>,
    ) -> Result<Self, VNextError> {
        let sources = domains
            .into_iter()
            .map(CapacityAvailabilitySource::Domain)
            .collect::<BTreeSet<_>>();
        if sources.is_empty() {
            return Err(invalid_admission(
                "capacity wait snapshot cannot be narrowed to no domains",
            ));
        }
        let observed = sources
            .into_iter()
            .map(|source| {
                self.wait_condition
                    .observed
                    .binary_search_by_key(&source, |entry| entry.source)
                    .map(|index| self.wait_condition.observed[index])
                    .map_err(|_| {
                        invalid_admission("capacity wait snapshot cannot add an unobserved domain")
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::new(
            self.epochs,
            CapacityWaitCondition::new(self.wait_condition.coordinator_id, observed)?,
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct SequenceAuthorityId {
    sparse_id: u32,
    generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct RequestAuthorityId {
    sparse_id: u32,
    generation: u64,
}

impl RequestAuthorityId {
    #[cfg(test)]
    pub(crate) const fn test_only(sparse_id: u32, generation: u64) -> Self {
        Self {
            sparse_id,
            generation,
        }
    }

    pub const fn sparse_id(self) -> u32 {
        self.sparse_id
    }

    pub const fn generation(self) -> u64 {
        self.generation
    }
}

impl SequenceAuthorityId {
    #[cfg(test)]
    pub(crate) const fn test_only(sparse_id: u32, generation: u64) -> Self {
        Self {
            sparse_id,
            generation,
        }
    }

    pub const fn sparse_id(self) -> u32 {
        self.sparse_id
    }

    pub const fn generation(self) -> u64 {
        self.generation
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeferredAction {
    WaitForRelease,
    AwaitBackingGrowth,
    PreemptAndRecompute,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AdmissionDeferred {
    immediate_requested: CapacityVector,
    fit_requested: CapacityVector,
    available: CapacitySnapshot,
    blockers: Vec<CapacityShortfall>,
    action: DeferredAction,
    release_epoch: u64,
    capacity_epoch: u64,
    wait_condition: CapacityWaitCondition,
}

impl AdmissionDeferred {
    pub fn immediate_requested(&self) -> &CapacityVector {
        &self.immediate_requested
    }

    pub fn fit_requested(&self) -> &CapacityVector {
        &self.fit_requested
    }

    pub fn available(&self) -> &CapacitySnapshot {
        &self.available
    }

    pub fn blockers(&self) -> &[CapacityShortfall] {
        &self.blockers
    }

    pub const fn action(&self) -> DeferredAction {
        self.action
    }

    pub const fn release_epoch(&self) -> u64 {
        self.release_epoch
    }

    pub const fn capacity_epoch(&self) -> u64 {
        self.capacity_epoch
    }

    pub const fn epochs(&self) -> CapacityEpochs {
        CapacityEpochs {
            coordinator_id: self.available.coordinator_id,
            release_epoch: self.release_epoch,
            capacity_epoch: self.capacity_epoch,
        }
    }

    pub fn wait_condition(&self) -> &CapacityWaitCondition {
        &self.wait_condition
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AdmissionRejected {
    immediate_requested: CapacityVector,
    fit_requested: CapacityVector,
    maximum: CapacitySnapshot,
    blockers: Vec<CapacityShortfall>,
}

impl AdmissionRejected {
    pub fn blockers(&self) -> &[CapacityShortfall] {
        &self.blockers
    }

    pub fn maximum(&self) -> &CapacitySnapshot {
        &self.maximum
    }
}

#[derive(Debug)]
pub enum AdmissionDecision {
    Admitted(LogicalAdmissionLease),
    Deferred(AdmissionDeferred),
    PermanentRejected(AdmissionRejected),
}

#[derive(Debug)]
pub enum RequestAdmissionDecision {
    Admitted(LogicalRequestLease),
    Deferred(AdmissionDeferred),
    PermanentRejected(AdmissionRejected),
}

pub(crate) enum AdmissionPreflightDecision {
    Eligible,
    Deferred(AdmissionDeferred),
    PermanentRejected(AdmissionRejected),
}

#[derive(Debug)]
pub enum CapacityClaimDecision {
    Claimed(LogicalCapacityLease),
    Deferred(AdmissionDeferred),
    PermanentRejected(AdmissionRejected),
}

#[derive(Debug)]
pub enum BatchCapacityClaimDecision {
    Claimed(LogicalBatchCapacityLease),
    Deferred(AdmissionDeferred),
    PermanentRejected(AdmissionRejected),
}

/// One exact sequence parent of a batch-scoped child capacity claim. The
/// coordinator derives this evidence from live leases and returns it in
/// canonical sequence-authority order; callers cannot construct authority by
/// copying these identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct SequenceCapacityParent {
    request: RequestAuthorityId,
    sequence: SequenceAuthorityId,
}

impl SequenceCapacityParent {
    pub const fn request(self) -> RequestAuthorityId {
        self.request
    }

    pub const fn sequence(self) -> SequenceAuthorityId {
        self.sequence
    }
}

#[derive(Debug)]
struct DomainState {
    spec: CapacityDomainSpec,
    used: u64,
    availability_epoch: u64,
}

#[derive(Debug, Clone, Copy)]
struct LiveSequenceRecord {
    generation: u64,
    request: RequestAuthorityId,
    active_child_claims: u64,
}

#[derive(Debug, Clone, Copy)]
struct LiveRequestRecord {
    generation: u64,
    active_sequences: u32,
}

#[derive(Debug)]
struct CoordinatorState {
    domains: BTreeMap<CapacityDomainId, DomainState>,
    maximum_active_sequences: u32,
    active_requests: u32,
    active_sequences: u32,
    active_child_claims: u64,
    live_requests: Vec<Option<LiveRequestRecord>>,
    reusable_request_ids: Vec<u32>,
    live_sequences: Vec<Option<LiveSequenceRecord>>,
    reusable_sequence_ids: Vec<u32>,
    next_request_generation: u64,
    next_sequence_generation: u64,
    release_epoch: u64,
    capacity_epoch: u64,
    active_sequence_availability_epoch: u64,
    poisoned: bool,
}

impl CoordinatorState {
    fn snapshot(&self, coordinator_id: LogicalAdmissionCoordinatorId) -> CapacitySnapshot {
        let domains = self
            .domains
            .iter()
            .map(|(domain, state)| {
                let total = state.spec.total_units.get();
                DomainCapacitySnapshot {
                    domain: *domain,
                    total: CapacityUnits::new(total),
                    maximum_total: state.spec.maximum_total_units,
                    used: CapacityUnits::new(state.used),
                    available: CapacityUnits::new(total.saturating_sub(state.used)),
                }
            })
            .collect();
        CapacitySnapshot {
            coordinator_id,
            domains,
            active_requests: self.active_requests,
            active_sequences: self.active_sequences,
            active_child_claims: self.active_child_claims,
            maximum_active_sequences: self.maximum_active_sequences,
            release_epoch: self.release_epoch,
            capacity_epoch: self.capacity_epoch,
            live_sequence_records: if self.poisoned {
                self.live_sequences
                    .iter()
                    .filter(|record| record.is_some())
                    .count()
            } else {
                self.active_sequences as usize
            },
            reusable_sequence_ids: self.reusable_sequence_ids.len(),
            live_request_records: if self.poisoned {
                self.live_requests
                    .iter()
                    .filter(|record| record.is_some())
                    .count()
            } else {
                self.active_requests as usize
            },
            reusable_request_ids: self.reusable_request_ids.len(),
            poisoned: self.poisoned,
        }
    }

    fn epochs(&self, coordinator_id: LogicalAdmissionCoordinatorId) -> CapacityEpochs {
        CapacityEpochs {
            coordinator_id,
            release_epoch: self.release_epoch,
            capacity_epoch: self.capacity_epoch,
        }
    }

    fn write_availability_epochs(&self, out: &mut Vec<CapacityAvailabilityEpoch>) {
        out.clear();
        out.extend(
            self.domains
                .iter()
                .map(|(domain, state)| CapacityAvailabilityEpoch {
                    source: CapacityAvailabilitySource::Domain(*domain),
                    epoch: state.availability_epoch,
                }),
        );
        out.push(CapacityAvailabilityEpoch {
            source: CapacityAvailabilitySource::ActiveSequenceSlots,
            epoch: self.active_sequence_availability_epoch,
        });
    }

    fn wait_condition_for_blockers(
        &self,
        coordinator_id: LogicalAdmissionCoordinatorId,
        blockers: &[CapacityShortfall],
    ) -> Result<CapacityWaitCondition, VNextError> {
        let mut sources = BTreeSet::new();
        for blocker in blockers {
            match (blocker.kind, blocker.domain) {
                (CapacityShortfallKind::ActiveSequenceCeiling, None) => {
                    sources.insert(CapacityAvailabilitySource::ActiveSequenceSlots);
                }
                (CapacityShortfallKind::PermanentDomainMaximum, _) => {
                    return Err(invalid_admission(
                        "permanent capacity blocker cannot produce a wait condition",
                    ));
                }
                (_, Some(domain)) => {
                    if !self.domains.contains_key(&domain) {
                        return Err(admission_fault(
                            DynamicAdmissionFaultKind::UnknownDomain,
                            "capacity blocker references an unknown wait domain",
                        ));
                    }
                    sources.insert(CapacityAvailabilitySource::Domain(domain));
                }
                (_, None) => {
                    return Err(invalid_admission(
                        "capacity blocker contains no availability source",
                    ));
                }
            }
        }
        self.wait_condition_for_sources(coordinator_id, sources)
    }

    fn wait_condition_for_domains(
        &self,
        coordinator_id: LogicalAdmissionCoordinatorId,
        domains: impl IntoIterator<Item = CapacityDomainId>,
    ) -> Result<CapacityWaitCondition, VNextError> {
        self.wait_condition_for_sources(
            coordinator_id,
            domains
                .into_iter()
                .map(CapacityAvailabilitySource::Domain)
                .collect(),
        )
    }

    fn wait_condition_for_sources(
        &self,
        coordinator_id: LogicalAdmissionCoordinatorId,
        sources: BTreeSet<CapacityAvailabilitySource>,
    ) -> Result<CapacityWaitCondition, VNextError> {
        let observed = sources
            .into_iter()
            .map(|source| {
                let epoch = match source {
                    CapacityAvailabilitySource::Domain(domain) => {
                        self.domains
                            .get(&domain)
                            .ok_or_else(|| {
                                admission_fault(
                                    DynamicAdmissionFaultKind::UnknownDomain,
                                    "capacity wait references an unknown domain",
                                )
                            })?
                            .availability_epoch
                    }
                    CapacityAvailabilitySource::ActiveSequenceSlots => {
                        self.active_sequence_availability_epoch
                    }
                    CapacityAvailabilitySource::PlanDeviceBudget
                    | CapacityAvailabilitySource::ProcessDeviceCapacity => {
                        return Err(invalid_admission(
                            "device capacity availability is owned by the device account",
                        ));
                    }
                };
                CapacityAvailabilityEpoch::new(source, epoch)
            })
            .collect::<Result<Vec<_>, _>>()?;
        CapacityWaitCondition::new(coordinator_id, observed)
    }

    fn availability_epoch_for(
        &self,
        source: CapacityAvailabilitySource,
    ) -> Result<u64, VNextError> {
        match source {
            CapacityAvailabilitySource::Domain(domain) => self
                .domains
                .get(&domain)
                .map(|state| state.availability_epoch)
                .ok_or_else(|| {
                    admission_fault(
                        DynamicAdmissionFaultKind::UnknownDomain,
                        "capacity wait references an unknown domain",
                    )
                }),
            CapacityAvailabilitySource::ActiveSequenceSlots => {
                Ok(self.active_sequence_availability_epoch)
            }
            CapacityAvailabilitySource::PlanDeviceBudget
            | CapacityAvailabilitySource::ProcessDeviceCapacity => Err(invalid_admission(
                "device capacity availability is owned by the device account",
            )),
        }
    }

    fn wait_condition_changed(
        &self,
        coordinator_id: LogicalAdmissionCoordinatorId,
        observed: &CapacityWaitCondition,
    ) -> Result<bool, VNextError> {
        if observed.coordinator_id != coordinator_id {
            return Err(admission_fault(
                DynamicAdmissionFaultKind::ForeignCoordinator,
                "capacity wait condition belongs to a different coordinator",
            ));
        }
        let mut changed = false;
        for entry in &observed.observed {
            let current = self.availability_epoch_for(entry.source)?;
            if current < entry.epoch {
                return Err(admission_fault(
                    DynamicAdmissionFaultKind::EpochRegression,
                    "capacity availability epoch regressed",
                ));
            }
            changed |= current > entry.epoch;
        }
        Ok(changed)
    }

    fn refresh_wait_condition(
        &self,
        coordinator_id: LogicalAdmissionCoordinatorId,
        observed: &CapacityWaitCondition,
    ) -> Result<CapacityWaitCondition, VNextError> {
        if observed.coordinator_id != coordinator_id {
            return Err(admission_fault(
                DynamicAdmissionFaultKind::ForeignCoordinator,
                "capacity wait condition belongs to a different coordinator",
            ));
        }
        CapacityWaitCondition::new(
            coordinator_id,
            observed
                .observed
                .iter()
                .map(|entry| {
                    CapacityAvailabilityEpoch::new(
                        entry.source,
                        self.availability_epoch_for(entry.source)?,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
    }

    fn preview_request_authority(&self) -> Result<RequestAuthorityReservation, VNextError> {
        let generation = self.next_request_generation;
        let next_generation = generation.checked_add(1).ok_or_else(|| {
            admission_fault(
                DynamicAdmissionFaultKind::AuthorityExhausted,
                "request authority generation is exhausted",
            )
        })?;
        let (sparse_id, source) = if let Some(id) = self.reusable_request_ids.last().copied() {
            (id, SparseIdSource::Reused)
        } else {
            if self.live_requests.len() >= u32::MAX as usize {
                return Err(admission_fault(
                    DynamicAdmissionFaultKind::AuthorityExhausted,
                    "request authority id space is exhausted",
                ));
            }
            (self.live_requests.len() as u32, SparseIdSource::Fresh)
        };
        if self
            .live_requests
            .get(sparse_id as usize)
            .is_some_and(Option::is_some)
        {
            return Err(invalid_admission(
                "request authority allocator selected a live sparse id",
            ));
        }
        Ok(RequestAuthorityReservation {
            authority: RequestAuthorityId {
                sparse_id,
                generation,
            },
            source,
            next_generation,
        })
    }

    fn prepare_request_storage(
        &mut self,
        reservation: RequestAuthorityReservation,
    ) -> Result<(), VNextError> {
        if matches!(reservation.source, SparseIdSource::Fresh) {
            self.live_requests.try_reserve(1).map_err(|_| {
                admission_fault(
                    DynamicAdmissionFaultKind::AllocationFailure,
                    "cannot reserve request authority slab",
                )
            })?;
            self.reusable_request_ids
                .try_reserve(self.live_requests.len() + 1)
                .map_err(|_| {
                    admission_fault(
                        DynamicAdmissionFaultKind::AllocationFailure,
                        "cannot reserve request release free-list",
                    )
                })?;
        }
        Ok(())
    }

    fn commit_request_authority(
        &mut self,
        reservation: RequestAuthorityReservation,
    ) -> RequestAuthorityId {
        let record = LiveRequestRecord {
            generation: reservation.authority.generation,
            active_sequences: 0,
        };
        match reservation.source {
            SparseIdSource::Reused => {
                self.reusable_request_ids.pop();
                self.live_requests[reservation.authority.sparse_id as usize] = Some(record);
            }
            SparseIdSource::Fresh => self.live_requests.push(Some(record)),
        }
        self.next_request_generation = reservation.next_generation;
        reservation.authority
    }

    fn preview_sequence_authority(&self) -> Result<SequenceAuthorityReservation, VNextError> {
        let generation = self.next_sequence_generation;
        let next_generation = generation.checked_add(1).ok_or_else(|| {
            admission_fault(
                DynamicAdmissionFaultKind::AuthorityExhausted,
                "sequence authority generation is exhausted",
            )
        })?;
        let (sparse_id, source) = if let Some(id) = self.reusable_sequence_ids.last().copied() {
            (id, SparseIdSource::Reused)
        } else {
            if self.live_sequences.len() >= u32::MAX as usize {
                return Err(admission_fault(
                    DynamicAdmissionFaultKind::AuthorityExhausted,
                    "sequence authority id space is exhausted",
                ));
            }
            (self.live_sequences.len() as u32, SparseIdSource::Fresh)
        };
        if self
            .live_sequences
            .get(sparse_id as usize)
            .is_some_and(Option::is_some)
        {
            return Err(invalid_admission(
                "sequence authority allocator selected a live sparse id",
            ));
        }
        Ok(SequenceAuthorityReservation {
            authority: SequenceAuthorityId {
                sparse_id,
                generation,
            },
            source,
            next_generation,
        })
    }

    fn prepare_sequence_storage(
        &mut self,
        reservation: SequenceAuthorityReservation,
    ) -> Result<(), VNextError> {
        if matches!(reservation.source, SparseIdSource::Fresh) {
            self.live_sequences.try_reserve(1).map_err(|_| {
                admission_fault(
                    DynamicAdmissionFaultKind::AllocationFailure,
                    "cannot reserve sequence authority slab",
                )
            })?;
            self.reusable_sequence_ids
                .try_reserve(self.live_sequences.len() + 1)
                .map_err(|_| {
                    admission_fault(
                        DynamicAdmissionFaultKind::AllocationFailure,
                        "cannot reserve sequence release free-list",
                    )
                })?;
        }
        Ok(())
    }

    fn commit_sequence_authority(
        &mut self,
        reservation: SequenceAuthorityReservation,
        request: RequestAuthorityId,
    ) -> SequenceAuthorityId {
        match reservation.source {
            SparseIdSource::Reused => {
                self.reusable_sequence_ids.pop();
                self.live_sequences[reservation.authority.sparse_id as usize] =
                    Some(LiveSequenceRecord {
                        generation: reservation.authority.generation,
                        request,
                        active_child_claims: 0,
                    });
            }
            SparseIdSource::Fresh => {
                self.live_sequences.push(Some(LiveSequenceRecord {
                    generation: reservation.authority.generation,
                    request,
                    active_child_claims: 0,
                }));
            }
        }
        self.next_sequence_generation = reservation.next_generation;
        reservation.authority
    }
}

#[derive(Debug, Clone, Copy)]
enum SparseIdSource {
    Reused,
    Fresh,
}

#[derive(Debug, Clone, Copy)]
struct RequestAuthorityReservation {
    authority: RequestAuthorityId,
    source: SparseIdSource,
    next_generation: u64,
}

#[derive(Debug, Clone, Copy)]
struct SequenceAuthorityReservation {
    authority: SequenceAuthorityId,
    source: SparseIdSource,
    next_generation: u64,
}

#[derive(Debug)]
struct CoordinatorInner {
    id: LogicalAdmissionCoordinatorId,
    state: Mutex<CoordinatorState>,
    epoch_tx: watch::Sender<CapacityEpochs>,
}

impl CoordinatorInner {
    fn lock_state(&self) -> Result<MutexGuard<'_, CoordinatorState>, VNextError> {
        match self.state.lock() {
            Ok(state) => Ok(state),
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                state.poisoned = true;
                let epochs = state.epochs(self.id);
                self.epoch_tx.send_replace(epochs);
                Err(admission_fault(
                    DynamicAdmissionFaultKind::Poisoned,
                    "coordinator state is poisoned",
                ))
            }
        }
    }

    fn lock_mutation(&self) -> Result<CoordinatorMutationGuard<'_>, VNextError> {
        Ok(CoordinatorMutationGuard {
            inner: self,
            state: self.lock_state()?,
            panicking_on_entry: std::thread::panicking(),
        })
    }
}

struct CoordinatorMutationGuard<'a> {
    inner: &'a CoordinatorInner,
    state: MutexGuard<'a, CoordinatorState>,
    panicking_on_entry: bool,
}

impl Deref for CoordinatorMutationGuard<'_> {
    type Target = CoordinatorState;

    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

impl DerefMut for CoordinatorMutationGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
}

impl Drop for CoordinatorMutationGuard<'_> {
    fn drop(&mut self) {
        if !self.panicking_on_entry && std::thread::panicking() {
            self.state.poisoned = true;
            self.inner
                .epoch_tx
                .send_replace(self.state.epochs(self.inner.id));
        }
    }
}

#[derive(Debug, Clone)]
pub struct LogicalAdmissionCoordinator {
    inner: Arc<CoordinatorInner>,
}

impl LogicalAdmissionCoordinator {
    pub(crate) fn new(
        domains: Vec<(CapacityDomainId, CapacityDomainSpec)>,
        maximum_active_sequences: u32,
    ) -> Result<Self, VNextError> {
        if maximum_active_sequences == 0 {
            return Err(invalid_admission(
                "coordinator requires a non-zero sequence ceiling",
            ));
        }
        let mut registered = BTreeMap::new();
        for (domain, spec) in domains {
            if registered
                .insert(
                    domain,
                    DomainState {
                        spec,
                        used: 0,
                        availability_epoch: 1,
                    },
                )
                .is_some()
            {
                return Err(invalid_admission("duplicate coordinator capacity domain"));
            }
        }
        let coordinator_id = LogicalAdmissionCoordinatorId::issue()?;
        let initial_epochs = CapacityEpochs {
            coordinator_id,
            release_epoch: 1,
            capacity_epoch: 1,
        };
        let (epoch_tx, _) = watch::channel(initial_epochs);
        Ok(Self {
            inner: Arc::new(CoordinatorInner {
                id: coordinator_id,
                state: Mutex::new(CoordinatorState {
                    domains: registered,
                    maximum_active_sequences,
                    active_requests: 0,
                    active_sequences: 0,
                    active_child_claims: 0,
                    live_requests: Vec::new(),
                    reusable_request_ids: Vec::new(),
                    live_sequences: Vec::new(),
                    reusable_sequence_ids: Vec::new(),
                    next_request_generation: 1,
                    next_sequence_generation: 1,
                    release_epoch: 1,
                    capacity_epoch: 1,
                    active_sequence_availability_epoch: 1,
                    poisoned: false,
                }),
                epoch_tx,
            }),
        })
    }

    pub fn id(&self) -> LogicalAdmissionCoordinatorId {
        self.inner.id
    }

    pub(crate) fn owns(&self, lease: &LogicalAdmissionLease) -> bool {
        self.id() == lease.coordinator_id() && Arc::ptr_eq(&self.inner, &lease.inner)
    }

    pub(crate) fn owns_request(&self, lease: &LogicalRequestLease) -> bool {
        self.id() == lease.coordinator_id() && Arc::ptr_eq(&self.inner, &lease.inner)
    }

    pub(crate) fn try_admit_request(
        &self,
        demand: &AdmissionDemand,
    ) -> Result<RequestAdmissionDecision, VNextError> {
        let mut state = self.inner.lock_mutation()?;
        if state.poisoned {
            return Err(admission_fault(
                DynamicAdmissionFaultKind::Poisoned,
                "coordinator is fail-closed",
            ));
        }

        let evaluation = evaluate_demand(&state, demand)?;
        if !evaluation.permanent.is_empty() {
            return Ok(RequestAdmissionDecision::PermanentRejected(
                AdmissionRejected {
                    immediate_requested: demand.immediate_claim.clone(),
                    fit_requested: demand.fit_requirement.clone(),
                    maximum: state.snapshot(self.id()),
                    blockers: evaluation.permanent,
                },
            ));
        }
        if !evaluation.blockers.is_empty() {
            let action = deferred_action(demand, evaluation.growth_required);
            let wait_condition =
                state.wait_condition_for_blockers(self.id(), &evaluation.blockers)?;
            let snapshot = state.snapshot(self.id());
            return Ok(RequestAdmissionDecision::Deferred(AdmissionDeferred {
                immediate_requested: demand.immediate_claim.clone(),
                fit_requested: demand.fit_requirement.clone(),
                release_epoch: snapshot.release_epoch,
                capacity_epoch: snapshot.capacity_epoch,
                available: snapshot,
                blockers: evaluation.blockers,
                action,
                wait_condition,
            }));
        }

        let request = state.preview_request_authority()?;
        let committed_claims = demand.immediate_claim.clone();
        let mut next_usage = Vec::with_capacity(demand.immediate_claim.entries().len());
        for entry in demand.immediate_claim.entries() {
            let domain = state
                .domains
                .get(&entry.domain)
                .expect("known request demand domain was preflight validated");
            let used = domain.used.checked_add(entry.units.get()).ok_or_else(|| {
                admission_fault(
                    DynamicAdmissionFaultKind::ArithmeticOverflow,
                    "request capacity usage overflows u64",
                )
            })?;
            next_usage.push((entry.domain, used));
        }
        let next_active_requests = state.active_requests.checked_add(1).ok_or_else(|| {
            admission_fault(
                DynamicAdmissionFaultKind::AuthorityExhausted,
                "active request count is exhausted",
            )
        })?;
        state
            .release_epoch
            .checked_add(u64::from(next_active_requests))
            .and_then(|epoch| epoch.checked_add(u64::from(state.active_sequences)))
            .and_then(|epoch| epoch.checked_add(state.active_child_claims))
            .ok_or_else(|| {
                admission_fault(
                    DynamicAdmissionFaultKind::EpochExhausted,
                    "release epoch cannot represent every outstanding lease release",
                )
            })?;
        state.prepare_request_storage(request)?;
        let request = state.commit_request_authority(request);
        for (domain, used) in next_usage {
            state
                .domains
                .get_mut(&domain)
                .expect("validated request capacity domain remains registered")
                .used = used;
        }
        state.active_requests = next_active_requests;
        Ok(RequestAdmissionDecision::Admitted(LogicalRequestLease {
            inner: Arc::clone(&self.inner),
            request,
            claims: committed_claims,
            released: false,
        }))
    }

    pub(crate) fn try_admit_sequence_for_request(
        &self,
        request: &LogicalRequestLease,
        demand: &AdmissionDemand,
    ) -> Result<AdmissionDecision, VNextError> {
        if !self.owns_request(request) {
            return Err(admission_fault(
                DynamicAdmissionFaultKind::ForeignCoordinator,
                "parent request belongs to another coordinator",
            ));
        }
        if request.released {
            return Err(invalid_admission(
                "sequence admission requires a live parent request",
            ));
        }
        let mut state = self.inner.lock_mutation()?;
        if state.poisoned {
            return Err(admission_fault(
                DynamicAdmissionFaultKind::Poisoned,
                "coordinator is fail-closed",
            ));
        }
        let request_record = state
            .live_requests
            .get(request.request.sparse_id as usize)
            .and_then(|record| *record)
            .filter(|record| record.generation == request.request.generation)
            .ok_or_else(|| invalid_admission("parent request authority is stale or not live"))?;

        let mut evaluation = evaluate_demand(&state, demand)?;
        if !evaluation.permanent.is_empty() {
            return Ok(AdmissionDecision::PermanentRejected(AdmissionRejected {
                immediate_requested: demand.immediate_claim.clone(),
                fit_requested: demand.fit_requirement.clone(),
                maximum: state.snapshot(self.id()),
                blockers: evaluation.permanent,
            }));
        }
        if state.active_sequences >= state.maximum_active_sequences {
            evaluation.blockers.push(CapacityShortfall {
                domain: None,
                kind: CapacityShortfallKind::ActiveSequenceCeiling,
                requested: CapacityUnits::new(1),
                available: CapacityUnits::ZERO,
                current_total: CapacityUnits::new(u64::from(state.maximum_active_sequences)),
                maximum_total: CapacityUnits::new(u64::from(state.maximum_active_sequences)),
            });
        }
        if !evaluation.blockers.is_empty() {
            let action = deferred_action(demand, evaluation.growth_required);
            let wait_condition =
                state.wait_condition_for_blockers(self.id(), &evaluation.blockers)?;
            let snapshot = state.snapshot(self.id());
            return Ok(AdmissionDecision::Deferred(AdmissionDeferred {
                immediate_requested: demand.immediate_claim.clone(),
                fit_requested: demand.fit_requirement.clone(),
                release_epoch: snapshot.release_epoch,
                capacity_epoch: snapshot.capacity_epoch,
                available: snapshot,
                blockers: evaluation.blockers,
                action,
                wait_condition,
            }));
        }

        let sequence = state.preview_sequence_authority()?;
        let committed_claims = demand.immediate_claim.clone();
        let mut next_usage = Vec::with_capacity(demand.immediate_claim.entries().len());
        for entry in demand.immediate_claim.entries() {
            let domain = state
                .domains
                .get(&entry.domain)
                .expect("known demand domain was preflight validated");
            let used = domain.used.checked_add(entry.units.get()).ok_or_else(|| {
                admission_fault(
                    DynamicAdmissionFaultKind::ArithmeticOverflow,
                    "capacity usage overflows u64",
                )
            })?;
            next_usage.push((entry.domain, used));
        }
        let next_active_sequences = state.active_sequences.checked_add(1).ok_or_else(|| {
            admission_fault(
                DynamicAdmissionFaultKind::ArithmeticOverflow,
                "active sequence count overflows u32",
            )
        })?;
        let next_request_sequences =
            request_record
                .active_sequences
                .checked_add(1)
                .ok_or_else(|| {
                    admission_fault(
                        DynamicAdmissionFaultKind::AuthorityExhausted,
                        "request child sequence count is exhausted",
                    )
                })?;
        state
            .release_epoch
            .checked_add(u64::from(state.active_requests))
            .and_then(|epoch| epoch.checked_add(u64::from(next_active_sequences)))
            .and_then(|epoch| epoch.checked_add(state.active_child_claims))
            .ok_or_else(|| {
                admission_fault(
                    DynamicAdmissionFaultKind::EpochExhausted,
                    "release epoch cannot represent every outstanding lease release",
                )
            })?;
        state.prepare_sequence_storage(sequence)?;
        let sequence = state.commit_sequence_authority(sequence, request.request);
        for (domain, used) in next_usage {
            state
                .domains
                .get_mut(&domain)
                .expect("validated capacity domain remains registered")
                .used = used;
        }
        state.active_sequences = next_active_sequences;
        state.live_requests[request.request.sparse_id as usize]
            .as_mut()
            .expect("validated parent request remains live")
            .active_sequences = next_request_sequences;
        Ok(AdmissionDecision::Admitted(LogicalAdmissionLease {
            inner: Arc::clone(&self.inner),
            request: request.request,
            sequence,
            claims: committed_claims,
            released: false,
        }))
    }

    /// Avoids physical backing work when the global logical sequence ceiling
    /// already makes admission impossible. Capacity-only blockers still flow
    /// through backing preparation so the caller receives exact pool growth
    /// evidence; the final admission remains the authoritative atomic check.
    pub(crate) fn preflight_sequence_ceiling_for_request(
        &self,
        request: &LogicalRequestLease,
        demand: &AdmissionDemand,
    ) -> Result<AdmissionPreflightDecision, VNextError> {
        if !self.owns_request(request) {
            return Err(admission_fault(
                DynamicAdmissionFaultKind::ForeignCoordinator,
                "parent request belongs to another coordinator",
            ));
        }
        if request.released {
            return Err(invalid_admission(
                "sequence admission requires a live parent request",
            ));
        }
        let state = self.inner.lock_state()?;
        if state.poisoned {
            return Err(admission_fault(
                DynamicAdmissionFaultKind::Poisoned,
                "coordinator is fail-closed",
            ));
        }
        state
            .live_requests
            .get(request.request.sparse_id as usize)
            .and_then(|record| *record)
            .filter(|record| record.generation == request.request.generation)
            .ok_or_else(|| invalid_admission("parent request authority is stale or not live"))?;

        let mut evaluation = evaluate_demand(&state, demand)?;
        if !evaluation.permanent.is_empty() {
            return Ok(AdmissionPreflightDecision::PermanentRejected(
                AdmissionRejected {
                    immediate_requested: demand.immediate_claim.clone(),
                    fit_requested: demand.fit_requirement.clone(),
                    maximum: state.snapshot(self.id()),
                    blockers: evaluation.permanent,
                },
            ));
        }
        if state.active_sequences < state.maximum_active_sequences {
            return Ok(AdmissionPreflightDecision::Eligible);
        }
        evaluation.blockers.push(CapacityShortfall {
            domain: None,
            kind: CapacityShortfallKind::ActiveSequenceCeiling,
            requested: CapacityUnits::new(1),
            available: CapacityUnits::ZERO,
            current_total: CapacityUnits::new(u64::from(state.maximum_active_sequences)),
            maximum_total: CapacityUnits::new(u64::from(state.maximum_active_sequences)),
        });
        let wait_condition = state.wait_condition_for_blockers(self.id(), &evaluation.blockers)?;
        let snapshot = state.snapshot(self.id());
        Ok(AdmissionPreflightDecision::Deferred(AdmissionDeferred {
            immediate_requested: demand.immediate_claim.clone(),
            fit_requested: demand.fit_requirement.clone(),
            release_epoch: snapshot.release_epoch,
            capacity_epoch: snapshot.capacity_epoch,
            available: snapshot,
            blockers: evaluation.blockers,
            action: match demand.pressure_action {
                AdmissionPressureAction::WaitForRelease => DeferredAction::WaitForRelease,
                AdmissionPressureAction::PreemptAndRecompute => DeferredAction::PreemptAndRecompute,
            },
            wait_condition,
        }))
    }

    pub(crate) fn try_claim_for_sequence(
        &self,
        sequence: &LogicalAdmissionLease,
        demand: &AdmissionDemand,
    ) -> Result<CapacityClaimDecision, VNextError> {
        match self.try_claim_for_sequences(&[sequence], demand)? {
            BatchCapacityClaimDecision::Claimed(batch) => {
                Ok(CapacityClaimDecision::Claimed(LogicalCapacityLease {
                    batch,
                }))
            }
            BatchCapacityClaimDecision::Deferred(deferred) => {
                Ok(CapacityClaimDecision::Deferred(deferred))
            }
            BatchCapacityClaimDecision::PermanentRejected(rejected) => {
                Ok(CapacityClaimDecision::PermanentRejected(rejected))
            }
        }
    }

    /// Claims one actual-shape capacity vector for a non-empty batch while
    /// binding its release authority to every participating sequence. Capacity
    /// is charged once for the batch, but every parent prevents early release
    /// until the shared child lease reaches its terminal state.
    pub(crate) fn try_claim_for_sequences(
        &self,
        sequences: &[&LogicalAdmissionLease],
        demand: &AdmissionDemand,
    ) -> Result<BatchCapacityClaimDecision, VNextError> {
        if sequences.is_empty() || demand.immediate_claim.is_empty() {
            return Err(invalid_admission(
                "batch child capacity claim requires live parents and non-empty demand",
            ));
        }
        let mut parents = Vec::with_capacity(sequences.len());
        for sequence in sequences {
            if !self.owns(sequence) {
                return Err(admission_fault(
                    DynamicAdmissionFaultKind::ForeignCoordinator,
                    "batch parent sequence belongs to another coordinator",
                ));
            }
            if sequence.released {
                return Err(invalid_admission(
                    "batch child capacity claim requires every parent to be live",
                ));
            }
            parents.push(SequenceCapacityParent {
                request: sequence.request,
                sequence: sequence.sequence,
            });
        }
        u32::try_from(parents.len())
            .map_err(|_| invalid_admission("batch parent count exceeds the protocol range"))?;
        parents.sort_by_key(|parent| (parent.sequence, parent.request));
        if parents.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(invalid_admission(
                "batch child capacity claim contains a duplicate parent sequence",
            ));
        }

        let mut state = self.inner.lock_mutation()?;
        if state.poisoned {
            return Err(admission_fault(
                DynamicAdmissionFaultKind::Poisoned,
                "coordinator is fail-closed",
            ));
        }
        let mut next_parent_child_claims = Vec::with_capacity(parents.len());
        for parent in &parents {
            let record = state
                .live_sequences
                .get(parent.sequence.sparse_id as usize)
                .and_then(|record| *record)
                .filter(|record| {
                    record.generation == parent.sequence.generation
                        && record.request == parent.request
                })
                .ok_or_else(|| {
                    invalid_admission("batch parent sequence authority is stale or not live")
                })?;
            let next = record.active_child_claims.checked_add(1).ok_or_else(|| {
                admission_fault(
                    DynamicAdmissionFaultKind::AuthorityExhausted,
                    "batch parent child capacity claim count is exhausted",
                )
            })?;
            next_parent_child_claims.push((parent.sequence.sparse_id, next));
        }

        let evaluation = evaluate_demand(&state, demand)?;
        if !evaluation.permanent.is_empty() {
            return Ok(BatchCapacityClaimDecision::PermanentRejected(
                AdmissionRejected {
                    immediate_requested: demand.immediate_claim.clone(),
                    fit_requested: demand.fit_requirement.clone(),
                    maximum: state.snapshot(self.id()),
                    blockers: evaluation.permanent,
                },
            ));
        }
        if !evaluation.blockers.is_empty() {
            let action = deferred_action(demand, evaluation.growth_required);
            let wait_condition =
                state.wait_condition_for_blockers(self.id(), &evaluation.blockers)?;
            let snapshot = state.snapshot(self.id());
            return Ok(BatchCapacityClaimDecision::Deferred(AdmissionDeferred {
                immediate_requested: demand.immediate_claim.clone(),
                fit_requested: demand.fit_requirement.clone(),
                release_epoch: snapshot.release_epoch,
                capacity_epoch: snapshot.capacity_epoch,
                available: snapshot,
                blockers: evaluation.blockers,
                action,
                wait_condition,
            }));
        }

        let committed_claims = demand.immediate_claim.clone();
        let mut next_usage = Vec::with_capacity(demand.immediate_claim.entries().len());
        for entry in demand.immediate_claim.entries() {
            let domain = state
                .domains
                .get(&entry.domain)
                .expect("known batch demand domain was preflight validated");
            let used = domain.used.checked_add(entry.units.get()).ok_or_else(|| {
                admission_fault(
                    DynamicAdmissionFaultKind::ArithmeticOverflow,
                    "batch child capacity usage overflows u64",
                )
            })?;
            next_usage.push((entry.domain, used));
        }
        let next_child_claims = state.active_child_claims.checked_add(1).ok_or_else(|| {
            admission_fault(
                DynamicAdmissionFaultKind::AuthorityExhausted,
                "active batch child capacity claim count is exhausted",
            )
        })?;
        state
            .release_epoch
            .checked_add(u64::from(state.active_requests))
            .and_then(|epoch| epoch.checked_add(u64::from(state.active_sequences)))
            .and_then(|epoch| epoch.checked_add(next_child_claims))
            .ok_or_else(|| {
                admission_fault(
                    DynamicAdmissionFaultKind::EpochExhausted,
                    "release epoch cannot represent every outstanding lease release",
                )
            })?;

        for (domain, used) in next_usage {
            state
                .domains
                .get_mut(&domain)
                .expect("validated batch capacity domain remains registered")
                .used = used;
        }
        state.active_child_claims = next_child_claims;
        for (sparse_id, next) in next_parent_child_claims {
            state.live_sequences[sparse_id as usize]
                .as_mut()
                .expect("validated batch parent sequence remains live")
                .active_child_claims = next;
        }
        Ok(BatchCapacityClaimDecision::Claimed(
            LogicalBatchCapacityLease {
                inner: Arc::clone(&self.inner),
                parents,
                claims: committed_claims,
                released: false,
            },
        ))
    }

    pub(crate) fn owns_capacity_claim(&self, lease: &LogicalCapacityLease) -> bool {
        self.owns_batch_capacity_claim(&lease.batch)
    }

    pub(crate) fn owns_batch_capacity_claim(&self, lease: &LogicalBatchCapacityLease) -> bool {
        self.id() == lease.coordinator_id() && Arc::ptr_eq(&self.inner, &lease.inner)
    }

    pub fn snapshot(&self) -> Result<CapacitySnapshot, VNextError> {
        match self.inner.state.lock() {
            Ok(state) => Ok(state.snapshot(self.id())),
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                state.poisoned = true;
                let snapshot = state.snapshot(self.id());
                self.inner.epoch_tx.send_replace(state.epochs(self.id()));
                Ok(snapshot)
            }
        }
    }

    pub fn epochs(&self) -> Result<CapacityEpochs, VNextError> {
        let state = self.inner.lock_state()?;
        if state.poisoned {
            return Err(admission_fault(
                DynamicAdmissionFaultKind::Poisoned,
                "coordinator is fail-closed",
            ));
        }
        Ok(state.epochs(self.id()))
    }

    pub(crate) fn subscribe_epochs(&self) -> watch::Receiver<CapacityEpochs> {
        self.inner.epoch_tx.subscribe()
    }

    /// Writes a canonical point-in-time availability vector into caller-owned
    /// storage. Reusing the buffer keeps steady scheduler ticks allocation-free.
    pub fn write_availability_epochs(
        &self,
        out: &mut Vec<CapacityAvailabilityEpoch>,
    ) -> Result<CapacityEpochs, VNextError> {
        let state = self.inner.lock_state()?;
        if state.poisoned {
            return Err(admission_fault(
                DynamicAdmissionFaultKind::Poisoned,
                "coordinator is fail-closed",
            ));
        }
        state.write_availability_epochs(out);
        Ok(state.epochs(self.id()))
    }

    pub(crate) fn wait_snapshot_for_domains(
        &self,
        domains: impl IntoIterator<Item = CapacityDomainId>,
    ) -> Result<CapacityWaitSnapshot, VNextError> {
        let state = self.inner.lock_state()?;
        if state.poisoned {
            return Err(admission_fault(
                DynamicAdmissionFaultKind::Poisoned,
                "coordinator is fail-closed",
            ));
        }
        Ok(CapacityWaitSnapshot::new(
            state.epochs(self.id()),
            state.wait_condition_for_domains(self.id(), domains)?,
        ))
    }

    pub(crate) fn refresh_wait_snapshot(
        &self,
        observed: &CapacityWaitCondition,
    ) -> Result<CapacityWaitSnapshot, VNextError> {
        let state = self.inner.lock_state()?;
        if state.poisoned {
            return Err(admission_fault(
                DynamicAdmissionFaultKind::Poisoned,
                "coordinator is fail-closed",
            ));
        }
        Ok(CapacityWaitSnapshot::new(
            state.epochs(self.id()),
            state.refresh_wait_condition(self.id(), observed)?,
        ))
    }

    pub(crate) fn set_domain_total(
        &self,
        domain: CapacityDomainId,
        new_total: CapacityUnits,
    ) -> Result<CapacityEpochs, VNextError> {
        self.set_domain_totals(&[(domain, new_total)])
    }

    pub(crate) fn set_domain_totals(
        &self,
        updates: &[(CapacityDomainId, CapacityUnits)],
    ) -> Result<CapacityEpochs, VNextError> {
        let mut state = self.inner.lock_mutation()?;
        if state.poisoned {
            return Err(admission_fault(
                DynamicAdmissionFaultKind::Poisoned,
                "coordinator is fail-closed",
            ));
        }
        let mut seen = BTreeSet::new();
        let mut changed = false;
        for (domain, new_total) in updates {
            if !seen.insert(*domain) {
                return Err(invalid_admission(
                    "capacity update contains a duplicate domain",
                ));
            }
            let domain_state = state.domains.get(domain).ok_or_else(|| {
                admission_fault(
                    DynamicAdmissionFaultKind::UnknownDomain,
                    "capacity update references unknown domain",
                )
            })?;
            if new_total.get() < domain_state.used
                || new_total.get() > domain_state.spec.maximum_total_units.get()
            {
                return Err(invalid_admission(
                    "capacity update is below live use or above maximum total",
                ));
            }
            if new_total.get() > domain_state.spec.total_units.get()
                && domain_state.availability_epoch == u64::MAX
            {
                return Err(admission_fault(
                    DynamicAdmissionFaultKind::EpochExhausted,
                    "domain availability epoch is exhausted",
                ));
            }
            changed |= *new_total != domain_state.spec.total_units;
        }
        if !changed {
            return Ok(state.epochs(self.id()));
        }
        let next_capacity_epoch = state.capacity_epoch.checked_add(1).ok_or_else(|| {
            admission_fault(
                DynamicAdmissionFaultKind::EpochExhausted,
                "capacity epoch is exhausted",
            )
        })?;
        for (domain, new_total) in updates {
            let domain_state = state
                .domains
                .get_mut(domain)
                .expect("validated capacity domain remains registered");
            if new_total.get() > domain_state.spec.total_units.get() {
                domain_state.availability_epoch += 1;
            }
            domain_state.spec.total_units = *new_total;
        }
        state.capacity_epoch = next_capacity_epoch;
        let epochs = state.epochs(self.id());
        self.inner.epoch_tx.send_replace(epochs);
        drop(state);
        Ok(epochs)
    }

    /// Publishes an allocator-visible availability change that does not alter
    /// the domain's total or used units, such as extent release or compaction
    /// increasing the largest contiguous range.
    pub(crate) fn notify_domain_availability_changed(
        &self,
        domain: CapacityDomainId,
    ) -> Result<CapacityEpochs, VNextError> {
        let mut state = self.inner.lock_mutation()?;
        if state.poisoned {
            return Err(admission_fault(
                DynamicAdmissionFaultKind::Poisoned,
                "coordinator is fail-closed",
            ));
        }
        if !state.domains.contains_key(&domain) {
            return Err(admission_fault(
                DynamicAdmissionFaultKind::UnknownDomain,
                "availability update references unknown domain",
            ));
        }
        let next_capacity_epoch = state.capacity_epoch.checked_add(1).ok_or_else(|| {
            admission_fault(
                DynamicAdmissionFaultKind::EpochExhausted,
                "capacity epoch is exhausted",
            )
        })?;
        let domain_state = state
            .domains
            .get_mut(&domain)
            .expect("validated availability domain remains registered");
        domain_state.availability_epoch = domain_state
            .availability_epoch
            .checked_add(1)
            .ok_or_else(|| {
                admission_fault(
                    DynamicAdmissionFaultKind::EpochExhausted,
                    "domain availability epoch is exhausted",
                )
            })?;
        state.capacity_epoch = next_capacity_epoch;
        let epochs = state.epochs(self.id());
        self.inner.epoch_tx.send_replace(epochs);
        drop(state);
        Ok(epochs)
    }

    pub fn register_waiter(
        &self,
        observed: CapacityWaitCondition,
    ) -> Result<CapacityWaitRegistration, VNextError> {
        if observed.coordinator_id != self.id() {
            return Err(admission_fault(
                DynamicAdmissionFaultKind::ForeignCoordinator,
                "wait observation belongs to a different coordinator",
            ));
        }
        let receiver = self.inner.epoch_tx.subscribe();
        let state = self.inner.lock_state()?;
        if state.poisoned {
            return Err(admission_fault(
                DynamicAdmissionFaultKind::Poisoned,
                "coordinator is fail-closed",
            ));
        }
        let registered = state.refresh_wait_condition(self.id(), &observed)?;
        Ok(CapacityWaitRegistration {
            inner: Arc::clone(&self.inner),
            observed,
            registered,
            receiver,
        })
    }
}

fn shortfall(
    requested: CapacityEntry,
    state: &DomainState,
    available: u64,
    kind: CapacityShortfallKind,
) -> CapacityShortfall {
    CapacityShortfall {
        domain: Some(requested.domain),
        kind,
        requested: requested.units,
        available: CapacityUnits::new(available),
        current_total: state.spec.total_units,
        maximum_total: state.spec.maximum_total_units,
    }
}

struct DemandEvaluation {
    permanent: Vec<CapacityShortfall>,
    blockers: Vec<CapacityShortfall>,
    growth_required: bool,
}

fn evaluate_demand(
    state: &CoordinatorState,
    demand: &AdmissionDemand,
) -> Result<DemandEvaluation, VNextError> {
    let mut permanent = Vec::new();
    for requested in demand
        .immediate_claim
        .entries()
        .iter()
        .chain(demand.fit_requirement.entries())
    {
        let Some(domain) = state.domains.get(&requested.domain) else {
            return Err(admission_fault(
                DynamicAdmissionFaultKind::UnknownDomain,
                format!(
                    "demand references unknown capacity domain {}",
                    requested.domain.get()
                ),
            ));
        };
        if requested.units.get() > domain.spec.maximum_total_units.get() {
            permanent.push(CapacityShortfall {
                domain: Some(requested.domain),
                kind: CapacityShortfallKind::PermanentDomainMaximum,
                requested: requested.units,
                available: CapacityUnits::new(
                    domain.spec.total_units.get().saturating_sub(domain.used),
                ),
                current_total: domain.spec.total_units,
                maximum_total: domain.spec.maximum_total_units,
            });
        }
    }
    permanent.sort_by_key(|shortfall| (shortfall.domain, shortfall.kind as u8));
    permanent.dedup_by(|left, right| {
        left.domain == right.domain && left.kind == right.kind && left.requested == right.requested
    });

    let mut blockers = Vec::new();
    let mut growth_required = false;
    if permanent.is_empty() {
        for requested in demand.immediate_claim.entries() {
            let domain = state
                .domains
                .get(&requested.domain)
                .expect("known demand domain was preflight validated");
            let available = domain.spec.total_units.get().saturating_sub(domain.used);
            if requested.units.get() > domain.spec.total_units.get() {
                growth_required = true;
                blockers.push(shortfall(
                    *requested,
                    domain,
                    available,
                    CapacityShortfallKind::BackingGrowthRequired,
                ));
            } else if requested.units.get() > available {
                blockers.push(shortfall(
                    *requested,
                    domain,
                    available,
                    CapacityShortfallKind::ImmediateAvailability,
                ));
            }
        }
        if demand.fit_policy == AdmissionFitPolicy::FullInputMustFit {
            for requested in demand.fit_requirement.entries() {
                let domain = state
                    .domains
                    .get(&requested.domain)
                    .expect("known fit domain was preflight validated");
                let available = domain.spec.total_units.get().saturating_sub(domain.used);
                if requested.units.get() > domain.spec.total_units.get() {
                    growth_required = true;
                    blockers.push(shortfall(
                        *requested,
                        domain,
                        available,
                        CapacityShortfallKind::BackingGrowthRequired,
                    ));
                } else if requested.units.get() > available {
                    blockers.push(shortfall(
                        *requested,
                        domain,
                        available,
                        CapacityShortfallKind::FitAvailability,
                    ));
                }
            }
        }
    }
    Ok(DemandEvaluation {
        permanent,
        blockers,
        growth_required,
    })
}

fn deferred_action(demand: &AdmissionDemand, growth_required: bool) -> DeferredAction {
    if growth_required {
        DeferredAction::AwaitBackingGrowth
    } else {
        match demand.pressure_action {
            AdmissionPressureAction::WaitForRelease => DeferredAction::WaitForRelease,
            AdmissionPressureAction::PreemptAndRecompute => DeferredAction::PreemptAndRecompute,
        }
    }
}

#[derive(Debug)]
#[must_use = "dropping the request lease releases request-scoped capacity"]
pub struct LogicalRequestLease {
    inner: Arc<CoordinatorInner>,
    request: RequestAuthorityId,
    claims: CapacityVector,
    released: bool,
}

impl LogicalRequestLease {
    pub fn coordinator_id(&self) -> LogicalAdmissionCoordinatorId {
        self.inner.id
    }

    pub const fn request(&self) -> RequestAuthorityId {
        self.request
    }

    pub fn claims(&self) -> &CapacityVector {
        &self.claims
    }

    fn release_inner(&mut self) -> bool {
        if self.released {
            return true;
        }
        let state = match self.inner.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                state.poisoned = true;
                let epochs = state.epochs(self.coordinator_id());
                self.inner.epoch_tx.send_replace(epochs);
                return false;
            }
        };
        let mut state = CoordinatorMutationGuard {
            inner: self.inner.as_ref(),
            state,
            panicking_on_entry: std::thread::panicking(),
        };
        let request_is_releasable = state
            .live_requests
            .get(self.request.sparse_id as usize)
            .and_then(|record| *record)
            .is_some_and(|record| {
                record.generation == self.request.generation && record.active_sequences == 0
            });
        if state.poisoned
            || !request_is_releasable
            || state.active_requests == 0
            || self.claims.entries().iter().any(|claim| {
                state
                    .domains
                    .get(&claim.domain)
                    .is_none_or(|domain| domain.used < claim.units.get())
            })
        {
            state.poisoned = true;
            let epochs = state.epochs(self.coordinator_id());
            self.inner.epoch_tx.send_replace(epochs);
            return false;
        }
        let Some(next_release_epoch) = state.release_epoch.checked_add(1) else {
            state.poisoned = true;
            let epochs = state.epochs(self.coordinator_id());
            self.inner.epoch_tx.send_replace(epochs);
            return false;
        };
        if self.claims.entries().iter().any(|claim| {
            state
                .domains
                .get(&claim.domain)
                .is_some_and(|domain| domain.availability_epoch == u64::MAX)
        }) {
            state.poisoned = true;
            let epochs = state.epochs(self.coordinator_id());
            self.inner.epoch_tx.send_replace(epochs);
            return false;
        }
        for claim in self.claims.entries() {
            let domain = state
                .domains
                .get_mut(&claim.domain)
                .expect("validated request capacity domain remains registered");
            domain.used -= claim.units.get();
            domain.availability_epoch += 1;
        }
        state.active_requests -= 1;
        state.live_requests[self.request.sparse_id as usize] = None;
        state.reusable_request_ids.push(self.request.sparse_id);
        state.release_epoch = next_release_epoch;
        let epochs = state.epochs(self.coordinator_id());
        self.inner.epoch_tx.send_replace(epochs);
        drop(state);
        self.released = true;
        true
    }
}

impl Drop for LogicalRequestLease {
    fn drop(&mut self) {
        let _ = self.release_inner();
    }
}

#[derive(Debug)]
#[must_use = "dropping the logical admission lease releases its capacity claim"]
pub struct LogicalAdmissionLease {
    inner: Arc<CoordinatorInner>,
    request: RequestAuthorityId,
    sequence: SequenceAuthorityId,
    claims: CapacityVector,
    released: bool,
}

impl LogicalAdmissionLease {
    pub fn coordinator_id(&self) -> LogicalAdmissionCoordinatorId {
        self.inner.id
    }

    pub const fn sequence(&self) -> SequenceAuthorityId {
        self.sequence
    }

    pub const fn request(&self) -> RequestAuthorityId {
        self.request
    }

    pub fn claims(&self) -> &CapacityVector {
        &self.claims
    }

    fn release_inner(&mut self) -> bool {
        if self.released {
            return true;
        }
        let state = match self.inner.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                state.poisoned = true;
                let epochs = state.epochs(self.coordinator_id());
                self.inner.epoch_tx.send_replace(epochs);
                return false;
            }
        };
        let mut state = CoordinatorMutationGuard {
            inner: self.inner.as_ref(),
            state,
            panicking_on_entry: std::thread::panicking(),
        };
        let sequence_record = state
            .live_sequences
            .get(self.sequence.sparse_id as usize)
            .and_then(|record| *record);
        let request_record = state
            .live_requests
            .get(self.request.sparse_id as usize)
            .and_then(|record| *record);
        if state.poisoned
            || sequence_record.is_none_or(|record| {
                record.generation != self.sequence.generation
                    || record.request != self.request
                    || record.active_child_claims != 0
            })
            || request_record.is_none_or(|record| {
                record.generation != self.request.generation || record.active_sequences == 0
            })
            || state.active_sequences == 0
            || self.claims.entries().iter().any(|claim| {
                state
                    .domains
                    .get(&claim.domain)
                    .is_none_or(|domain| domain.used < claim.units.get())
            })
        {
            state.poisoned = true;
            let epochs = state.epochs(self.coordinator_id());
            self.inner.epoch_tx.send_replace(epochs);
            return false;
        }
        let Some(next_release_epoch) = state.release_epoch.checked_add(1) else {
            state.poisoned = true;
            let epochs = state.epochs(self.coordinator_id());
            self.inner.epoch_tx.send_replace(epochs);
            return false;
        };
        if state.active_sequence_availability_epoch == u64::MAX
            || self.claims.entries().iter().any(|claim| {
                state
                    .domains
                    .get(&claim.domain)
                    .is_some_and(|domain| domain.availability_epoch == u64::MAX)
            })
        {
            state.poisoned = true;
            let epochs = state.epochs(self.coordinator_id());
            self.inner.epoch_tx.send_replace(epochs);
            return false;
        }
        for claim in self.claims.entries() {
            let domain = state
                .domains
                .get_mut(&claim.domain)
                .expect("validated logical claim domain remains registered");
            domain.used -= claim.units.get();
            domain.availability_epoch += 1;
        }
        state.active_sequences -= 1;
        state.active_sequence_availability_epoch += 1;
        state.live_requests[self.request.sparse_id as usize]
            .as_mut()
            .expect("validated parent request remains live")
            .active_sequences -= 1;
        state.live_sequences[self.sequence.sparse_id as usize] = None;
        state.reusable_sequence_ids.push(self.sequence.sparse_id);
        state.release_epoch = next_release_epoch;
        let epochs = state.epochs(self.coordinator_id());
        self.inner.epoch_tx.send_replace(epochs);
        drop(state);
        self.released = true;
        true
    }
}

impl Drop for LogicalAdmissionLease {
    fn drop(&mut self) {
        let _ = self.release_inner();
    }
}

#[derive(Debug)]
#[must_use = "dropping a child capacity lease releases its exact domain claims"]
pub struct LogicalCapacityLease {
    batch: LogicalBatchCapacityLease,
}

impl LogicalCapacityLease {
    pub fn coordinator_id(&self) -> LogicalAdmissionCoordinatorId {
        self.batch.coordinator_id()
    }

    pub fn sequence(&self) -> SequenceAuthorityId {
        self.batch.parents[0].sequence
    }

    pub fn request(&self) -> RequestAuthorityId {
        self.batch.parents[0].request
    }

    pub fn claims(&self) -> &CapacityVector {
        self.batch.claims()
    }
}

#[derive(Debug)]
#[must_use = "dropping a batch child lease releases its exact shared claim"]
pub struct LogicalBatchCapacityLease {
    inner: Arc<CoordinatorInner>,
    parents: Vec<SequenceCapacityParent>,
    claims: CapacityVector,
    released: bool,
}

impl LogicalBatchCapacityLease {
    pub fn coordinator_id(&self) -> LogicalAdmissionCoordinatorId {
        self.inner.id
    }

    pub fn parents(&self) -> &[SequenceCapacityParent] {
        &self.parents
    }

    pub fn claims(&self) -> &CapacityVector {
        &self.claims
    }

    fn release_inner(&mut self) -> bool {
        if self.released {
            return true;
        }
        let state = match self.inner.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                state.poisoned = true;
                let epochs = state.epochs(self.coordinator_id());
                self.inner.epoch_tx.send_replace(epochs);
                return false;
            }
        };
        let mut state = CoordinatorMutationGuard {
            inner: self.inner.as_ref(),
            state,
            panicking_on_entry: std::thread::panicking(),
        };
        let parents_are_live = !self.parents.is_empty()
            && self.parents.iter().all(|parent| {
                state
                    .live_sequences
                    .get(parent.sequence.sparse_id as usize)
                    .and_then(|record| *record)
                    .is_some_and(|record| {
                        record.generation == parent.sequence.generation
                            && record.request == parent.request
                            && record.active_child_claims > 0
                    })
            });
        if state.poisoned
            || !parents_are_live
            || state.active_child_claims == 0
            || self.claims.entries().iter().any(|claim| {
                state
                    .domains
                    .get(&claim.domain)
                    .is_none_or(|domain| domain.used < claim.units.get())
            })
        {
            state.poisoned = true;
            let epochs = state.epochs(self.coordinator_id());
            self.inner.epoch_tx.send_replace(epochs);
            return false;
        }
        let Some(next_release_epoch) = state.release_epoch.checked_add(1) else {
            state.poisoned = true;
            let epochs = state.epochs(self.coordinator_id());
            self.inner.epoch_tx.send_replace(epochs);
            return false;
        };
        if self.claims.entries().iter().any(|claim| {
            state
                .domains
                .get(&claim.domain)
                .is_some_and(|domain| domain.availability_epoch == u64::MAX)
        }) {
            state.poisoned = true;
            let epochs = state.epochs(self.coordinator_id());
            self.inner.epoch_tx.send_replace(epochs);
            return false;
        }
        for claim in self.claims.entries() {
            let domain = state
                .domains
                .get_mut(&claim.domain)
                .expect("validated child capacity domain remains registered");
            domain.used -= claim.units.get();
            domain.availability_epoch += 1;
        }
        state.active_child_claims -= 1;
        for parent in &self.parents {
            state.live_sequences[parent.sequence.sparse_id as usize]
                .as_mut()
                .expect("validated batch parent sequence remains live")
                .active_child_claims -= 1;
        }
        state.release_epoch = next_release_epoch;
        let epochs = state.epochs(self.coordinator_id());
        self.inner.epoch_tx.send_replace(epochs);
        drop(state);
        self.released = true;
        true
    }
}

impl Drop for LogicalBatchCapacityLease {
    fn drop(&mut self) {
        let _ = self.release_inner();
    }
}

#[derive(Debug)]
pub struct CapacityWaitRegistration {
    inner: Arc<CoordinatorInner>,
    observed: CapacityWaitCondition,
    registered: CapacityWaitCondition,
    receiver: watch::Receiver<CapacityEpochs>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapacityWaitRecheck {
    current: CapacityEpochs,
    changed_since_observation: bool,
    changed_since_registration: bool,
}

impl CapacityWaitRegistration {
    pub fn recheck(&self) -> Result<CapacityWaitRecheck, VNextError> {
        let state = self.inner.lock_state()?;
        if state.poisoned {
            return Err(admission_fault(
                DynamicAdmissionFaultKind::Poisoned,
                "coordinator is fail-closed",
            ));
        }
        let current = state.epochs(self.inner.id);
        Ok(CapacityWaitRecheck {
            current,
            changed_since_observation: state
                .wait_condition_changed(self.inner.id, &self.observed)?,
            changed_since_registration: state
                .wait_condition_changed(self.inner.id, &self.registered)?,
        })
    }

    pub async fn wait_for_change(mut self) -> Result<CapacityEpochs, VNextError> {
        loop {
            let recheck = self.recheck()?;
            if recheck.should_retry() {
                return Ok(recheck.current());
            }
            self.receiver.changed().await.map_err(|_| {
                admission_fault(
                    DynamicAdmissionFaultKind::Poisoned,
                    "capacity epoch listener closed",
                )
            })?;
            self.receiver.borrow_and_update();
        }
    }
}

impl CapacityWaitRecheck {
    pub(crate) const fn new(
        current: CapacityEpochs,
        changed_since_observation: bool,
        changed_since_registration: bool,
    ) -> Self {
        Self {
            current,
            changed_since_observation,
            changed_since_registration,
        }
    }

    pub const fn current(self) -> CapacityEpochs {
        self.current
    }

    pub const fn changed_since_observation(self) -> bool {
        self.changed_since_observation
    }

    pub const fn changed_since_registration(self) -> bool {
        self.changed_since_registration
    }

    pub const fn should_retry(self) -> bool {
        self.changed_since_observation || self.changed_since_registration
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Condvar, Mutex as StdMutex};
    use std::thread;

    const TEST_NATIVE_WORKER_LIMIT: usize = 8;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum TestGatePhase {
        Holding,
        Released,
        Cancelled,
    }

    #[derive(Debug)]
    struct TestGateState {
        ready: usize,
        phase: TestGatePhase,
    }

    #[derive(Debug)]
    struct CancellableTestGate {
        state: StdMutex<TestGateState>,
        changed: Condvar,
    }

    impl CancellableTestGate {
        fn new(worker_count: usize) -> Arc<Self> {
            assert!(worker_count > 0);
            assert!(worker_count <= TEST_NATIVE_WORKER_LIMIT);
            Arc::new(Self {
                state: StdMutex::new(TestGateState {
                    ready: 0,
                    phase: TestGatePhase::Holding,
                }),
                changed: Condvar::new(),
            })
        }

        fn arrive_and_wait(&self) -> bool {
            let mut state = self.state.lock().unwrap();
            if state.phase != TestGatePhase::Holding {
                return state.phase == TestGatePhase::Released;
            }
            state.ready += 1;
            self.changed.notify_all();
            while state.phase == TestGatePhase::Holding {
                state = self.changed.wait(state).unwrap();
            }
            state.phase == TestGatePhase::Released
        }

        fn wait_until_ready(&self, worker_count: usize) -> bool {
            assert!(worker_count <= TEST_NATIVE_WORKER_LIMIT);
            let mut state = self.state.lock().unwrap();
            while state.ready < worker_count && state.phase == TestGatePhase::Holding {
                state = self.changed.wait(state).unwrap();
            }
            state.ready == worker_count && state.phase == TestGatePhase::Holding
        }

        fn release(&self) {
            let mut state = self.state.lock().unwrap();
            assert_eq!(state.phase, TestGatePhase::Holding);
            state.phase = TestGatePhase::Released;
            self.changed.notify_all();
        }

        fn cancel(&self) {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if state.phase == TestGatePhase::Holding {
                state.phase = TestGatePhase::Cancelled;
                self.changed.notify_all();
            }
        }
    }

    struct CancelTestGateOnDrop<'a> {
        gate: &'a CancellableTestGate,
        armed: bool,
    }

    impl<'a> CancelTestGateOnDrop<'a> {
        fn new(gate: &'a CancellableTestGate) -> Self {
            Self { gate, armed: true }
        }

        fn disarm(&mut self) {
            self.armed = false;
        }
    }

    impl Drop for CancelTestGateOnDrop<'_> {
        fn drop(&mut self) {
            if self.armed {
                self.gate.cancel();
            }
        }
    }

    fn cancel_gate_on_unwind<T>(gate: &CancellableTestGate, work: impl FnOnce() -> T) -> T {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)) {
            Ok(value) => value,
            Err(payload) => {
                gate.cancel();
                std::panic::resume_unwind(payload);
            }
        }
    }

    fn domain(value: u32) -> CapacityDomainId {
        CapacityDomainId::new(value).unwrap()
    }

    fn vector(entries: &[(u32, u64)]) -> CapacityVector {
        CapacityVector::new(
            entries
                .iter()
                .map(|(domain_id, units)| {
                    CapacityEntry::new(domain(*domain_id), CapacityUnits::new(*units)).unwrap()
                })
                .collect(),
        )
        .unwrap()
    }

    fn demand(immediate: &[(u32, u64)], fit: &[(u32, u64)]) -> AdmissionDemand {
        AdmissionDemand::from_plan(
            vector(immediate),
            vector(fit),
            AdmissionFitPolicy::FullInputMustFit,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap()
    }

    fn coordinator(maximum_active_sequences: u32) -> LogicalAdmissionCoordinator {
        LogicalAdmissionCoordinator::new(
            vec![
                (
                    domain(1),
                    CapacityDomainSpec::new(CapacityUnits::new(10), CapacityUnits::new(20))
                        .unwrap(),
                ),
                (
                    domain(2),
                    CapacityDomainSpec::new(CapacityUnits::new(4), CapacityUnits::new(4)).unwrap(),
                ),
            ],
            maximum_active_sequences,
        )
        .unwrap()
    }

    fn wait_for_domains(
        coordinator: &LogicalAdmissionCoordinator,
        domains: &[u32],
    ) -> CapacityWaitCondition {
        coordinator
            .wait_snapshot_for_domains(domains.iter().copied().map(domain))
            .unwrap()
            .wait_condition()
            .clone()
    }

    fn admitted(decision: AdmissionDecision) -> LogicalAdmissionLease {
        match decision {
            AdmissionDecision::Admitted(lease) => lease,
            _ => panic!("expected admitted decision"),
        }
    }

    fn empty_demand() -> AdmissionDemand {
        AdmissionDemand::from_plan(
            CapacityVector::empty(),
            CapacityVector::empty(),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap()
    }

    fn admitted_request(decision: RequestAdmissionDecision) -> LogicalRequestLease {
        match decision {
            RequestAdmissionDecision::Admitted(lease) => lease,
            _ => panic!("expected admitted request"),
        }
    }

    fn request(coordinator: &LogicalAdmissionCoordinator) -> LogicalRequestLease {
        admitted_request(coordinator.try_admit_request(&empty_demand()).unwrap())
    }

    fn admit_sequence(
        coordinator: &LogicalAdmissionCoordinator,
        request: &LogicalRequestLease,
        demand: &AdmissionDemand,
    ) -> LogicalAdmissionLease {
        admitted(
            coordinator
                .try_admit_sequence_for_request(request, demand)
                .unwrap(),
        )
    }

    fn claimed_child(decision: CapacityClaimDecision) -> LogicalCapacityLease {
        match decision {
            CapacityClaimDecision::Claimed(lease) => lease,
            _ => panic!("expected child capacity claim"),
        }
    }

    fn claimed_batch(decision: BatchCapacityClaimDecision) -> LogicalBatchCapacityLease {
        match decision {
            BatchCapacityClaimDecision::Claimed(lease) => lease,
            _ => panic!("expected batch child capacity claim"),
        }
    }

    #[test]
    fn request_capacity_is_shared_once_across_multiple_sequences() {
        let coordinator = coordinator(2);
        let request = admitted_request(
            coordinator
                .try_admit_request(&demand(&[(1, 2), (2, 1)], &[(1, 2), (2, 1)]))
                .unwrap(),
        );
        let first = admit_sequence(
            &coordinator,
            &request,
            &demand(&[(1, 2), (2, 1)], &[(1, 2), (2, 1)]),
        );
        let second = admit_sequence(
            &coordinator,
            &request,
            &demand(&[(1, 2), (2, 1)], &[(1, 2), (2, 1)]),
        );
        let both = coordinator.snapshot().unwrap();
        assert_eq!(both.active_requests(), 1);
        assert_eq!(both.active_sequences(), 2);
        assert_eq!(both.domains()[0].used().get(), 6);
        assert_eq!(both.domains()[1].used().get(), 3);
        assert_eq!(first.request(), request.request());
        assert_eq!(second.request(), request.request());

        drop(first);
        let one = coordinator.snapshot().unwrap();
        assert_eq!(one.active_requests(), 1);
        assert_eq!(one.active_sequences(), 1);
        assert_eq!(one.domains()[0].used().get(), 4);
        assert_eq!(one.domains()[1].used().get(), 2);

        drop(second);
        let request_only = coordinator.snapshot().unwrap();
        assert_eq!(request_only.active_requests(), 1);
        assert_eq!(request_only.active_sequences(), 0);
        assert_eq!(request_only.domains()[0].used().get(), 2);
        assert_eq!(request_only.domains()[1].used().get(), 1);
        drop(request);
        let empty = coordinator.snapshot().unwrap();
        assert_eq!(empty.active_requests(), 0);
        assert!(empty
            .domains()
            .iter()
            .all(|domain| domain.used().get() == 0));
    }

    #[test]
    fn request_defer_and_reject_are_atomic_and_do_not_consume_sequence_slots() {
        let coordinator = coordinator(2);
        let held = admitted_request(
            coordinator
                .try_admit_request(&demand(&[(1, 1), (2, 3)], &[(1, 1), (2, 3)]))
                .unwrap(),
        );
        let before = coordinator.snapshot().unwrap();
        let deferred = coordinator
            .try_admit_request(&demand(&[(1, 1), (2, 2)], &[(1, 1), (2, 2)]))
            .unwrap();
        assert!(matches!(deferred, RequestAdmissionDecision::Deferred(_)));
        let rejected = coordinator
            .try_admit_request(&demand(&[(1, 21), (2, 1)], &[(1, 21), (2, 1)]))
            .unwrap();
        assert!(matches!(
            rejected,
            RequestAdmissionDecision::PermanentRejected(_)
        ));
        let after = coordinator.snapshot().unwrap();
        assert_eq!(before.domains, after.domains);
        assert_eq!(after.active_requests(), 1);
        assert_eq!(after.active_sequences(), 0);
        drop(held);
    }

    #[test]
    fn request_authority_reuses_sparse_storage_with_new_generation() {
        let coordinator = coordinator(u32::MAX);
        let mut previous_generation = 0;
        for _ in 0..100_000 {
            let request = request(&coordinator);
            assert_eq!(request.request().sparse_id(), 0);
            assert!(request.request().generation() > previous_generation);
            previous_generation = request.request().generation();
            drop(request);
        }
        let snapshot = coordinator.snapshot().unwrap();
        assert_eq!(snapshot.active_requests(), 0);
        assert_eq!(snapshot.live_request_records(), 0);
        assert_eq!(snapshot.reusable_request_ids(), 1);
        assert_eq!(snapshot.release_epoch(), 100_001);
    }

    #[test]
    fn early_request_release_with_live_sequence_fails_closed_and_retains_claims() {
        let coordinator = coordinator(1);
        let mut request = admitted_request(
            coordinator
                .try_admit_request(&demand(&[(1, 2)], &[(1, 2)]))
                .unwrap(),
        );
        let sequence = admit_sequence(&coordinator, &request, &demand(&[(1, 1)], &[(1, 1)]));
        assert!(!request.release_inner());
        let snapshot = coordinator.snapshot().unwrap();
        assert!(snapshot.poisoned());
        assert_eq!(snapshot.active_requests(), 1);
        assert_eq!(snapshot.active_sequences(), 1);
        assert_eq!(snapshot.domains()[0].used().get(), 3);
        drop(sequence);
        drop(request);
    }

    #[test]
    fn exact_fit_claims_only_immediate_and_release_retries() {
        let coordinator = coordinator(2);
        let request = request(&coordinator);
        let first = admit_sequence(
            &coordinator,
            &request,
            &demand(&[(1, 6), (2, 2)], &[(1, 10), (2, 2)]),
        );
        let snapshot = coordinator.snapshot().unwrap();
        assert_eq!(snapshot.domains()[0].used().get(), 6);
        assert_eq!(snapshot.domains()[0].available().get(), 4);

        let deferred = coordinator
            .try_admit_sequence_for_request(&request, &demand(&[(1, 5), (2, 1)], &[(1, 5), (2, 1)]))
            .unwrap();
        let observed = match deferred {
            AdmissionDecision::Deferred(value) => {
                assert_eq!(value.action(), DeferredAction::WaitForRelease);
                value.wait_condition().clone()
            }
            _ => panic!("expected capacity defer"),
        };
        drop(first);
        let registration = coordinator.register_waiter(observed).unwrap();
        assert!(registration.recheck().unwrap().should_retry());
        let retry = coordinator
            .try_admit_sequence_for_request(&request, &demand(&[(1, 5), (2, 1)], &[(1, 5), (2, 1)]))
            .unwrap();
        assert!(matches!(retry, AdmissionDecision::Admitted(_)));
    }

    #[test]
    fn child_claim_uses_parent_authority_without_consuming_sequence_slot() {
        let coordinator = coordinator(1);
        let request = request(&coordinator);
        let parent = admit_sequence(
            &coordinator,
            &request,
            &demand(&[(1, 2), (2, 1)], &[(1, 2), (2, 1)]),
        );
        let before = coordinator.snapshot().unwrap();
        let child = claimed_child(
            coordinator
                .try_claim_for_sequence(&parent, &demand(&[(1, 3), (2, 1)], &[(1, 3), (2, 1)]))
                .unwrap(),
        );
        assert_eq!(child.sequence(), parent.sequence());
        assert!(coordinator.owns_capacity_claim(&child));
        let claimed = coordinator.snapshot().unwrap();
        assert_eq!(claimed.active_sequences(), 1);
        assert_eq!(claimed.active_child_claims(), 1);
        assert_eq!(claimed.domains()[0].used().get(), 5);
        assert_eq!(claimed.domains()[1].used().get(), 2);

        drop(child);
        let released = coordinator.snapshot().unwrap();
        assert_eq!(released.active_sequences(), 1);
        assert_eq!(released.active_child_claims(), 0);
        assert_eq!(released.domains()[0].used().get(), 2);
        assert_eq!(released.domains()[1].used().get(), 1);
        assert_eq!(released.release_epoch(), before.release_epoch() + 1);
        drop(parent);
        assert_eq!(coordinator.snapshot().unwrap().active_sequences(), 0);
    }

    #[test]
    fn batch_child_claim_charges_once_and_binds_every_parent() {
        let coordinator = coordinator(2);
        let first_request = request(&coordinator);
        let second_request = request(&coordinator);
        let first = admit_sequence(&coordinator, &first_request, &empty_demand());
        let second = admit_sequence(&coordinator, &second_request, &empty_demand());
        let before = coordinator.snapshot().unwrap();

        let batch = claimed_batch(
            coordinator
                .try_claim_for_sequences(
                    &[&second, &first],
                    &demand(&[(1, 3), (2, 1)], &[(1, 3), (2, 1)]),
                )
                .unwrap(),
        );
        assert!(coordinator.owns_batch_capacity_claim(&batch));
        assert_eq!(batch.parents().len(), 2);
        assert!(batch
            .parents()
            .windows(2)
            .all(|pair| pair[0].sequence() < pair[1].sequence()));
        let claimed = coordinator.snapshot().unwrap();
        assert_eq!(claimed.active_requests(), 2);
        assert_eq!(claimed.active_sequences(), 2);
        assert_eq!(claimed.active_child_claims(), 1);
        assert_eq!(claimed.domains()[0].used().get(), 3);
        assert_eq!(claimed.domains()[1].used().get(), 1);
        let state = coordinator.inner.state.lock().unwrap();
        assert_eq!(
            state.live_sequences[first.sequence().sparse_id() as usize]
                .unwrap()
                .active_child_claims,
            1
        );
        assert_eq!(
            state.live_sequences[second.sequence().sparse_id() as usize]
                .unwrap()
                .active_child_claims,
            1
        );
        drop(state);

        drop(batch);
        let released = coordinator.snapshot().unwrap();
        assert_eq!(released.active_child_claims(), 0);
        assert_eq!(released.domains()[0].used().get(), 0);
        assert_eq!(released.domains()[1].used().get(), 0);
        assert_eq!(released.release_epoch(), before.release_epoch() + 1);
        drop(first);
        drop(second);
        drop(first_request);
        drop(second_request);
    }

    #[test]
    fn batch_child_rejects_duplicate_and_foreign_parents_atomically() {
        let local = coordinator(2);
        let local_request = request(&local);
        let first = admit_sequence(&local, &local_request, &empty_demand());
        let before = local.snapshot().unwrap();
        assert!(local
            .try_claim_for_sequences(&[&first, &first], &demand(&[(1, 1)], &[(1, 1)]))
            .is_err());
        let after_duplicate = local.snapshot().unwrap();
        assert_eq!(before.domains, after_duplicate.domains);
        assert_eq!(after_duplicate.active_child_claims(), 0);

        let foreign = coordinator(1);
        let foreign_request = request(&foreign);
        let foreign_sequence = admit_sequence(&foreign, &foreign_request, &empty_demand());
        assert!(matches!(
            local.try_claim_for_sequences(
                &[&first, &foreign_sequence],
                &demand(&[(1, 1)], &[(1, 1)])
            ),
            Err(VNextError::DynamicAdmissionContract {
                kind: DynamicAdmissionFaultKind::ForeignCoordinator,
                ..
            })
        ));
        let after_foreign = local.snapshot().unwrap();
        assert_eq!(before.domains, after_foreign.domains);
        assert_eq!(after_foreign.active_child_claims(), 0);
    }

    #[test]
    fn batch_child_defer_and_reject_have_zero_partial_parent_effect() {
        let coordinator = coordinator(2);
        let request = request(&coordinator);
        let first = admit_sequence(&coordinator, &request, &demand(&[(2, 3)], &[(2, 3)]));
        let second = admit_sequence(&coordinator, &request, &empty_demand());
        let before = coordinator.snapshot().unwrap();

        let deferred = coordinator
            .try_claim_for_sequences(
                &[&first, &second],
                &demand(&[(1, 1), (2, 2)], &[(1, 1), (2, 2)]),
            )
            .unwrap();
        assert!(matches!(deferred, BatchCapacityClaimDecision::Deferred(_)));
        let rejected = coordinator
            .try_claim_for_sequences(&[&first, &second], &demand(&[(1, 21)], &[(1, 21)]))
            .unwrap();
        assert!(matches!(
            rejected,
            BatchCapacityClaimDecision::PermanentRejected(_)
        ));
        let after = coordinator.snapshot().unwrap();
        assert_eq!(before.domains, after.domains);
        assert_eq!(after.active_child_claims(), 0);
        let state = coordinator.inner.state.lock().unwrap();
        assert_eq!(
            state.live_sequences[first.sequence().sparse_id() as usize]
                .unwrap()
                .active_child_claims,
            0
        );
        assert_eq!(
            state.live_sequences[second.sequence().sparse_id() as usize]
                .unwrap()
                .active_child_claims,
            0
        );
    }

    #[test]
    fn overlapping_batch_children_track_each_parent_without_double_charging() {
        let coordinator = coordinator(3);
        let request = request(&coordinator);
        let first = admit_sequence(&coordinator, &request, &empty_demand());
        let second = admit_sequence(&coordinator, &request, &empty_demand());
        let third = admit_sequence(&coordinator, &request, &empty_demand());
        let first_batch = claimed_batch(
            coordinator
                .try_claim_for_sequences(&[&first, &second], &demand(&[(1, 1)], &[(1, 1)]))
                .unwrap(),
        );
        let second_batch = claimed_batch(
            coordinator
                .try_claim_for_sequences(&[&second, &third], &demand(&[(1, 1)], &[(1, 1)]))
                .unwrap(),
        );
        let snapshot = coordinator.snapshot().unwrap();
        assert_eq!(snapshot.active_child_claims(), 2);
        assert_eq!(snapshot.domains()[0].used().get(), 2);
        let state = coordinator.inner.state.lock().unwrap();
        let child_counts = [&first, &second, &third].map(|parent| {
            state.live_sequences[parent.sequence().sparse_id() as usize]
                .unwrap()
                .active_child_claims
        });
        assert_eq!(child_counts, [1, 2, 1]);
        drop(state);
        drop(first_batch);
        assert_eq!(coordinator.snapshot().unwrap().active_child_claims(), 1);
        drop(second_batch);
        let released = coordinator.snapshot().unwrap();
        assert_eq!(released.active_child_claims(), 0);
        assert_eq!(released.domains()[0].used().get(), 0);
    }

    #[test]
    fn batch_child_unwind_releases_all_parent_edges() {
        let coordinator = coordinator(2);
        let request = request(&coordinator);
        let first = admit_sequence(&coordinator, &request, &empty_demand());
        let second = admit_sequence(&coordinator, &request, &empty_demand());
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _batch = claimed_batch(
                coordinator
                    .try_claim_for_sequences(&[&first, &second], &demand(&[(1, 2)], &[(1, 2)]))
                    .unwrap(),
            );
            panic!("inject batch invocation cancellation unwind");
        }));
        assert!(result.is_err());
        let snapshot = coordinator.snapshot().unwrap();
        assert!(!snapshot.poisoned());
        assert_eq!(snapshot.active_child_claims(), 0);
        assert_eq!(snapshot.domains()[0].used().get(), 0);
        let state = coordinator.inner.state.lock().unwrap();
        assert!([&first, &second].iter().all(|parent| state.live_sequences
            [parent.sequence().sparse_id() as usize]
            .unwrap()
            .active_child_claims
            == 0));
    }

    #[test]
    fn batch_child_rejects_stale_later_parent_without_partial_effect() {
        let coordinator = coordinator(2);
        let request = request(&coordinator);
        let first = admit_sequence(&coordinator, &request, &empty_demand());
        let second = admit_sequence(&coordinator, &request, &empty_demand());
        {
            let mut state = coordinator.inner.state.lock().unwrap();
            state.live_sequences[second.sequence().sparse_id() as usize]
                .as_mut()
                .unwrap()
                .generation += 1;
        }
        let before = coordinator.snapshot().unwrap();
        assert!(coordinator
            .try_claim_for_sequences(&[&first, &second], &demand(&[(1, 2)], &[(1, 2)]))
            .is_err());
        let after = coordinator.snapshot().unwrap();
        assert_eq!(before.domains, after.domains);
        assert_eq!(after.active_child_claims(), 0);
        {
            let mut state = coordinator.inner.state.lock().unwrap();
            state.live_sequences[second.sequence().sparse_id() as usize]
                .as_mut()
                .unwrap()
                .generation = second.sequence().generation();
        }
    }

    #[test]
    fn early_release_of_any_batch_parent_fails_closed_and_retains_shared_claim() {
        let coordinator = coordinator(2);
        let request = request(&coordinator);
        let first = admit_sequence(&coordinator, &request, &empty_demand());
        let mut second = admit_sequence(&coordinator, &request, &empty_demand());
        let batch = claimed_batch(
            coordinator
                .try_claim_for_sequences(&[&first, &second], &demand(&[(1, 2)], &[(1, 2)]))
                .unwrap(),
        );
        assert!(!second.release_inner());
        let snapshot = coordinator.snapshot().unwrap();
        assert!(snapshot.poisoned());
        assert_eq!(snapshot.active_sequences(), 2);
        assert_eq!(snapshot.active_child_claims(), 1);
        assert_eq!(snapshot.domains()[0].used().get(), 2);
        drop(batch);
        drop(first);
        drop(second);
        drop(request);
    }

    #[test]
    fn child_multi_domain_defer_and_reject_have_zero_partial_effect() {
        let coordinator = coordinator(2);
        let request = request(&coordinator);
        let parent = admit_sequence(
            &coordinator,
            &request,
            &demand(&[(1, 2), (2, 3)], &[(1, 2), (2, 3)]),
        );
        let before = coordinator.snapshot().unwrap();
        let deferred = coordinator
            .try_claim_for_sequence(&parent, &demand(&[(1, 2), (2, 2)], &[(1, 2), (2, 2)]))
            .unwrap();
        assert!(matches!(deferred, CapacityClaimDecision::Deferred(_)));
        let after_defer = coordinator.snapshot().unwrap();
        assert_eq!(before.domains, after_defer.domains);
        assert_eq!(after_defer.active_child_claims(), 0);

        let rejected = coordinator
            .try_claim_for_sequence(&parent, &demand(&[(1, 21), (2, 1)], &[(1, 21), (2, 1)]))
            .unwrap();
        assert!(matches!(
            rejected,
            CapacityClaimDecision::PermanentRejected(_)
        ));
        let after_reject = coordinator.snapshot().unwrap();
        assert_eq!(before.domains, after_reject.domains);
        assert_eq!(after_reject.active_child_claims(), 0);
    }

    #[test]
    fn concurrent_children_preserve_global_and_per_parent_counts() {
        const WORKER_COUNT: usize = 3;
        const _: () = assert!(WORKER_COUNT <= TEST_NATIVE_WORKER_LIMIT);

        let coordinator = coordinator(2);
        let request = Arc::new(request(&coordinator));
        let first = Arc::new(admit_sequence(
            &coordinator,
            &request,
            &demand(&[(1, 1)], &[(1, 1)]),
        ));
        let second = Arc::new(admit_sequence(
            &coordinator,
            &request,
            &demand(&[(1, 1)], &[(1, 1)]),
        ));
        let gate = CancellableTestGate::new(WORKER_COUNT);
        thread::scope(|scope| {
            let mut cancel_on_unwind = CancelTestGateOnDrop::new(&gate);
            let parents = [Arc::clone(&first), Arc::clone(&first), Arc::clone(&second)];
            let mut handles: Vec<thread::ScopedJoinHandle<'_, ()>> =
                Vec::with_capacity(WORKER_COUNT);
            for (worker_index, parent) in parents.into_iter().enumerate() {
                let worker_gate = Arc::clone(&gate);
                let coordinator = &coordinator;
                let handle = match thread::Builder::new()
                    .name(format!("admission-child-{worker_index}"))
                    .spawn_scoped(scope, move || {
                        cancel_gate_on_unwind(&worker_gate, || {
                            let child = claimed_child(
                                coordinator
                                    .try_claim_for_sequence(&parent, &demand(&[(1, 1)], &[(1, 1)]))
                                    .unwrap(),
                            );
                            let _ = worker_gate.arrive_and_wait();
                            drop(child);
                        });
                    }) {
                    Ok(handle) => handle,
                    Err(error) => {
                        gate.cancel();
                        for handle in handles.drain(..) {
                            let _ = handle.join();
                        }
                        panic!("failed to spawn bounded child worker: {error}");
                    }
                };
                handles.push(handle);
            }
            if !gate.wait_until_ready(WORKER_COUNT) {
                gate.cancel();
                for handle in handles.drain(..) {
                    let _ = handle.join();
                }
                panic!("bounded child worker cancelled before every worker became ready");
            }
            let snapshot = coordinator.snapshot().unwrap();
            assert_eq!(snapshot.active_requests(), 1);
            assert_eq!(snapshot.active_sequences(), 2);
            assert_eq!(snapshot.active_child_claims(), 3);
            let state = coordinator.inner.state.lock().unwrap();
            assert_eq!(
                state.live_sequences[first.sequence().sparse_id() as usize]
                    .unwrap()
                    .active_child_claims,
                2
            );
            assert_eq!(
                state.live_sequences[second.sequence().sparse_id() as usize]
                    .unwrap()
                    .active_child_claims,
                1
            );
            drop(state);
            gate.release();
            cancel_on_unwind.disarm();
            for handle in handles {
                handle.join().unwrap();
            }
        });
        assert_eq!(coordinator.snapshot().unwrap().active_child_claims(), 0);
    }

    #[test]
    fn child_unwind_releases_without_poisoning_parent() {
        let coordinator = coordinator(1);
        let request = request(&coordinator);
        let parent = admit_sequence(&coordinator, &request, &demand(&[(1, 1)], &[(1, 1)]));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _child = claimed_child(
                coordinator
                    .try_claim_for_sequence(&parent, &demand(&[(1, 2)], &[(1, 2)]))
                    .unwrap(),
            );
            panic!("inject invocation cancellation unwind");
        }));
        assert!(result.is_err());
        let snapshot = coordinator.snapshot().unwrap();
        assert!(!snapshot.poisoned());
        assert_eq!(snapshot.active_sequences(), 1);
        assert_eq!(snapshot.active_child_claims(), 0);
        assert_eq!(snapshot.domains()[0].used().get(), 1);
    }

    #[test]
    fn child_claim_rejects_stale_sequence_generation_without_side_effect() {
        let coordinator = coordinator(1);
        let request = request(&coordinator);
        let parent = admit_sequence(&coordinator, &request, &demand(&[(1, 1)], &[(1, 1)]));
        {
            let mut state = coordinator.inner.state.lock().unwrap();
            state.live_sequences[parent.sequence().sparse_id() as usize]
                .as_mut()
                .unwrap()
                .generation += 1;
        }
        let before = coordinator.snapshot().unwrap();
        assert!(coordinator
            .try_claim_for_sequence(&parent, &demand(&[(1, 1)], &[(1, 1)]))
            .is_err());
        let after = coordinator.snapshot().unwrap();
        assert_eq!(before.domains, after.domains);
        assert_eq!(after.active_child_claims(), 0);
        {
            let mut state = coordinator.inner.state.lock().unwrap();
            state.live_sequences[parent.sequence().sparse_id() as usize]
                .as_mut()
                .unwrap()
                .generation = parent.sequence().generation();
        }
    }

    #[test]
    fn child_claim_is_counted_in_future_request_epoch_headroom() {
        let coordinator = coordinator(1);
        let request = request(&coordinator);
        let parent = admit_sequence(&coordinator, &request, &demand(&[(1, 1)], &[(1, 1)]));
        let child = claimed_child(
            coordinator
                .try_claim_for_sequence(&parent, &demand(&[(1, 1)], &[(1, 1)]))
                .unwrap(),
        );
        {
            let mut state = coordinator.inner.state.lock().unwrap();
            state.release_epoch = u64::MAX - 3;
        }
        let before = coordinator.snapshot().unwrap();
        assert!(matches!(
            coordinator.try_admit_request(&empty_demand()),
            Err(VNextError::DynamicAdmissionContract {
                kind: DynamicAdmissionFaultKind::EpochExhausted,
                ..
            })
        ));
        let after = coordinator.snapshot().unwrap();
        assert_eq!(before.active_requests(), after.active_requests());
        assert_eq!(before.active_sequences(), after.active_sequences());
        assert_eq!(before.active_child_claims(), after.active_child_claims());
        assert_eq!(before.domains, after.domains);
        drop(child);
        drop(parent);
        drop(request);
        assert_eq!(coordinator.epochs().unwrap().release_epoch(), u64::MAX);
    }

    #[test]
    fn early_sequence_release_with_active_child_fails_closed() {
        let coordinator = coordinator(1);
        let request = request(&coordinator);
        let mut parent = admit_sequence(&coordinator, &request, &demand(&[(1, 1)], &[(1, 1)]));
        let child = claimed_child(
            coordinator
                .try_claim_for_sequence(&parent, &demand(&[(1, 1)], &[(1, 1)]))
                .unwrap(),
        );
        assert!(!parent.release_inner());
        let snapshot = coordinator.snapshot().unwrap();
        assert!(snapshot.poisoned());
        assert_eq!(snapshot.active_sequences(), 1);
        assert_eq!(snapshot.active_child_claims(), 1);
        drop(child);
        drop(parent);
        drop(request);
    }

    #[test]
    fn poisoned_child_drop_retains_capacity_and_child_counts() {
        let coordinator = coordinator(1);
        let request = request(&coordinator);
        let parent = admit_sequence(&coordinator, &request, &demand(&[(1, 1)], &[(1, 1)]));
        let child = claimed_child(
            coordinator
                .try_claim_for_sequence(&parent, &demand(&[(1, 2)], &[(1, 2)]))
                .unwrap(),
        );
        let inner = Arc::clone(&coordinator.inner);
        let _ = thread::spawn(move || {
            let _guard = inner.state.lock().unwrap();
            panic!("poison coordinator before child release");
        })
        .join();
        drop(child);
        let snapshot = coordinator.snapshot().unwrap();
        assert!(snapshot.poisoned());
        assert_eq!(snapshot.active_requests(), 1);
        assert_eq!(snapshot.active_sequences(), 1);
        assert_eq!(snapshot.active_child_claims(), 1);
        assert_eq!(snapshot.domains()[0].used().get(), 3);
        drop(parent);
        drop(request);
    }

    #[test]
    fn child_claim_rejects_foreign_parent_and_epoch_exhaustion_atomically() {
        let local = coordinator(2);
        let foreign = coordinator(2);
        let foreign_request = request(&foreign);
        let foreign_parent = admit_sequence(
            &foreign,
            &foreign_request,
            &demand(&[(1, 1), (2, 1)], &[(1, 1), (2, 1)]),
        );
        assert!(matches!(
            local.try_claim_for_sequence(
                &foreign_parent,
                &demand(&[(1, 1), (2, 1)], &[(1, 1), (2, 1)])
            ),
            Err(VNextError::DynamicAdmissionContract {
                kind: DynamicAdmissionFaultKind::ForeignCoordinator,
                ..
            })
        ));

        let local_request = request(&local);
        let parent = admit_sequence(
            &local,
            &local_request,
            &demand(&[(1, 1), (2, 1)], &[(1, 1), (2, 1)]),
        );
        {
            let mut state = local.inner.state.lock().unwrap();
            state.release_epoch = u64::MAX - 1;
        }
        let before = local.snapshot().unwrap();
        assert!(matches!(
            local.try_claim_for_sequence(&parent, &demand(&[(1, 1), (2, 1)], &[(1, 1), (2, 1)])),
            Err(VNextError::DynamicAdmissionContract {
                kind: DynamicAdmissionFaultKind::EpochExhausted,
                ..
            })
        ));
        let after = local.snapshot().unwrap();
        assert_eq!(before.domains, after.domains);
        assert_eq!(after.active_child_claims(), 0);
    }

    #[test]
    fn multi_domain_failure_has_zero_partial_effect() {
        let coordinator = coordinator(8);
        let request = request(&coordinator);
        let before = coordinator.snapshot().unwrap();
        let decision = coordinator
            .try_admit_sequence_for_request(&request, &demand(&[(1, 3), (2, 4)], &[(1, 3), (2, 5)]))
            .unwrap();
        assert!(matches!(decision, AdmissionDecision::PermanentRejected(_)));
        let after = coordinator.snapshot().unwrap();
        assert_eq!(before.domains, after.domains);
        assert_eq!(after.active_sequences(), 0);
        assert_eq!(after.live_sequence_records(), 0);
    }

    #[test]
    fn temporary_multi_domain_shortfall_has_zero_partial_effect() {
        let coordinator = coordinator(8);
        let request = request(&coordinator);
        let held = admit_sequence(
            &coordinator,
            &request,
            &demand(&[(1, 1), (2, 3)], &[(1, 1), (2, 3)]),
        );
        let before = coordinator.snapshot().unwrap();
        let decision = coordinator
            .try_admit_sequence_for_request(&request, &demand(&[(1, 2), (2, 2)], &[(1, 2), (2, 2)]))
            .unwrap();
        assert!(matches!(decision, AdmissionDecision::Deferred(_)));
        let after = coordinator.snapshot().unwrap();
        assert_eq!(before.domains, after.domains);
        assert_eq!(before.active_sequences(), after.active_sequences());
        drop(held);
    }

    #[test]
    fn growth_defer_and_permanent_reject_are_distinct() {
        let coordinator = coordinator(8);
        let request = request(&coordinator);
        let growth = coordinator
            .try_admit_sequence_for_request(
                &request,
                &demand(&[(1, 11), (2, 1)], &[(1, 11), (2, 1)]),
            )
            .unwrap();
        assert!(matches!(
            growth,
            AdmissionDecision::Deferred(AdmissionDeferred {
                action: DeferredAction::AwaitBackingGrowth,
                ..
            })
        ));
        let impossible = coordinator
            .try_admit_sequence_for_request(
                &request,
                &demand(&[(1, 21), (2, 1)], &[(1, 21), (2, 1)]),
            )
            .unwrap();
        assert!(matches!(
            impossible,
            AdmissionDecision::PermanentRejected(_)
        ));
        assert!(matches!(
            coordinator
                .try_admit_sequence_for_request(
                    &request,
                    &demand(&[(1, 2), (2, 1)], &[(1, 2), (2, 1)]),
                )
                .unwrap(),
            AdmissionDecision::Admitted(_)
        ));
    }

    #[test]
    fn lease_authority_is_bound_to_the_exact_coordinator() {
        let coordinator_a = coordinator(8);
        let coordinator_b = coordinator(8);
        assert_ne!(coordinator_a.id(), coordinator_b.id());
        let request = request(&coordinator_a);
        let lease = admit_sequence(
            &coordinator_a,
            &request,
            &demand(&[(1, 1), (2, 1)], &[(1, 1), (2, 1)]),
        );
        assert!(coordinator_a.owns(&lease));
        assert!(!coordinator_b.owns(&lease));
    }

    #[test]
    fn sparse_sequence_ids_reuse_storage_and_change_generation() {
        let coordinator = coordinator(u32::MAX);
        let parent_request = request(&coordinator);
        let sequence_demand = demand(&[(1, 1), (2, 1)], &[(1, 1), (2, 1)]);
        let mut previous_generation = 0;
        for _ in 0..1_000_000 {
            let lease = admit_sequence(&coordinator, &parent_request, &sequence_demand);
            assert_eq!(lease.sequence().sparse_id(), 0);
            assert!(lease.sequence().generation() > previous_generation);
            previous_generation = lease.sequence().generation();
            drop(lease);
        }
        let snapshot = coordinator.snapshot().unwrap();
        assert_eq!(snapshot.active_sequences(), 0);
        assert_eq!(snapshot.live_sequence_records(), 0);
        assert_eq!(snapshot.reusable_sequence_ids(), 1);
        assert_eq!(snapshot.release_epoch(), 1_000_001);
        assert_eq!(snapshot.capacity_epoch(), 1);
    }

    #[test]
    fn lease_release_uses_preallocated_reuse_storage() {
        let coordinator = coordinator(1);
        let request = request(&coordinator);
        let lease = admit_sequence(
            &coordinator,
            &request,
            &demand(&[(1, 1), (2, 1)], &[(1, 1), (2, 1)]),
        );
        let before = {
            let state = coordinator.inner.state.lock().unwrap();
            assert!(state.reusable_sequence_ids.capacity() >= state.live_sequences.len());
            (
                state.live_sequences.capacity(),
                state.reusable_sequence_ids.capacity(),
            )
        };
        drop(lease);
        let state = coordinator.inner.state.lock().unwrap();
        assert_eq!(state.live_sequences.capacity(), before.0);
        assert_eq!(state.reusable_sequence_ids.capacity(), before.1);
        assert_eq!(state.reusable_sequence_ids.as_slice(), &[0]);
    }

    #[test]
    fn snapshot_counts_live_authorities_independently_from_active_counter() {
        let coordinator = coordinator(1);
        let mut state = coordinator.inner.state.lock().unwrap();
        state.active_sequences = 1;
        state.poisoned = true;
        let snapshot = state.snapshot(coordinator.id());
        assert_eq!(snapshot.active_sequences(), 1);
        assert_eq!(snapshot.live_sequence_records(), 0);
    }

    #[test]
    fn concurrent_admission_never_exceeds_ceiling() {
        const SEQUENCE_CEILING: u32 = 4;
        const WORKER_COUNT: usize = 8;
        const _: () = assert!(WORKER_COUNT <= TEST_NATIVE_WORKER_LIMIT);

        let coordinator = Arc::new(coordinator(SEQUENCE_CEILING));
        let parent_request = Arc::new(request(&coordinator));
        let gate = CancellableTestGate::new(WORKER_COUNT);
        let mut handles: Vec<thread::JoinHandle<Option<AdmissionDecision>>> =
            Vec::with_capacity(WORKER_COUNT);
        for worker_index in 0..WORKER_COUNT {
            let coordinator = Arc::clone(&coordinator);
            let parent_request = Arc::clone(&parent_request);
            let worker_gate = Arc::clone(&gate);
            let handle = match thread::Builder::new()
                .name(format!("admission-sequence-{worker_index}"))
                .spawn(move || {
                    cancel_gate_on_unwind(&worker_gate, || {
                        if !worker_gate.arrive_and_wait() {
                            return None;
                        }
                        Some(
                            coordinator
                                .try_admit_sequence_for_request(&parent_request, &empty_demand())
                                .unwrap(),
                        )
                    })
                }) {
                Ok(handle) => handle,
                Err(error) => {
                    gate.cancel();
                    for handle in handles.drain(..) {
                        let _ = handle.join();
                    }
                    panic!("failed to spawn bounded admission worker: {error}");
                }
            };
            handles.push(handle);
        }
        if !gate.wait_until_ready(WORKER_COUNT) {
            gate.cancel();
            for handle in handles.drain(..) {
                let _ = handle.join();
            }
            panic!("bounded admission worker cancelled before every worker became ready");
        }
        gate.release();
        let mut leases = Vec::new();
        for handle in handles {
            if let Some(AdmissionDecision::Admitted(lease)) = handle.join().unwrap() {
                leases.push(lease);
            }
        }
        assert_eq!(leases.len(), SEQUENCE_CEILING as usize);
        assert_eq!(
            coordinator.snapshot().unwrap().active_sequences(),
            SEQUENCE_CEILING
        );

        let preempt_demand = AdmissionDemand::from_plan(
            CapacityVector::empty(),
            CapacityVector::empty(),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::PreemptAndRecompute,
        )
        .unwrap();
        let preflight = coordinator
            .preflight_sequence_ceiling_for_request(&parent_request, &preempt_demand)
            .unwrap();
        match preflight {
            AdmissionPreflightDecision::Deferred(deferred) => {
                assert_eq!(deferred.action(), DeferredAction::PreemptAndRecompute);
                assert!(deferred
                    .blockers()
                    .iter()
                    .any(|blocker| blocker.kind() == CapacityShortfallKind::ActiveSequenceCeiling));
            }
            _ => panic!("full sequence ceiling must defer preflight"),
        }

        for (pressure_action, expected) in [
            (
                AdmissionPressureAction::WaitForRelease,
                DeferredAction::WaitForRelease,
            ),
            (
                AdmissionPressureAction::PreemptAndRecompute,
                DeferredAction::PreemptAndRecompute,
            ),
        ] {
            let growth_demand = AdmissionDemand::from_plan(
                vector(&[(1, 11)]),
                vector(&[(1, 11)]),
                AdmissionFitPolicy::ImmediateOnly,
                pressure_action,
            )
            .unwrap();
            let decision = coordinator
                .preflight_sequence_ceiling_for_request(&parent_request, &growth_demand)
                .unwrap();
            match decision {
                AdmissionPreflightDecision::Deferred(deferred) => {
                    assert_eq!(deferred.action(), expected);
                    assert!(deferred.blockers().iter().any(|blocker| {
                        blocker.kind() == CapacityShortfallKind::ActiveSequenceCeiling
                    }));
                    assert!(deferred.blockers().iter().any(|blocker| {
                        blocker.kind() == CapacityShortfallKind::BackingGrowthRequired
                    }));
                }
                _ => panic!("mixed ceiling and growth pressure must remain a logical defer"),
            }
        }
        drop(leases);
        assert_eq!(coordinator.snapshot().unwrap().active_sequences(), 0);
    }

    #[test]
    fn zero_domain_plan_still_uses_dynamic_sequence_authority() {
        let coordinator = LogicalAdmissionCoordinator::new(Vec::new(), u32::MAX).unwrap();
        let parent_request = request(&coordinator);
        let sequence_demand = empty_demand();
        let lease = admit_sequence(&coordinator, &parent_request, &sequence_demand);
        assert!(lease.claims().is_empty());
        assert_eq!(coordinator.snapshot().unwrap().active_sequences(), 1);
        drop(lease);
        assert_eq!(coordinator.snapshot().unwrap().active_sequences(), 0);
    }

    #[test]
    fn sequence_issue_failure_has_zero_side_effect() {
        let coordinator = coordinator(8);
        let request = request(&coordinator);
        {
            let mut state = coordinator.inner.state.lock().unwrap();
            state.next_sequence_generation = u64::MAX;
        }
        let before = coordinator.snapshot().unwrap();
        assert!(
            coordinator
                .try_admit_sequence_for_request(
                    &request,
                    &demand(&[(1, 1), (2, 1)], &[(1, 1), (2, 1)]),
                )
                .is_err()
        );
        let after = coordinator.snapshot().unwrap();
        assert_eq!(before.domains, after.domains);
        assert_eq!(after.active_sequences(), 0);
        assert_eq!(after.live_sequence_records(), 0);
    }

    #[test]
    fn epoch_exhaustion_rejects_admission_before_claim() {
        let coordinator = coordinator(8);
        let request = request(&coordinator);
        {
            let mut state = coordinator.inner.state.lock().unwrap();
            state.release_epoch = u64::MAX;
        }
        let before = coordinator.snapshot().unwrap();
        assert!(
            coordinator
                .try_admit_sequence_for_request(
                    &request,
                    &demand(&[(1, 1), (2, 1)], &[(1, 1), (2, 1)]),
                )
                .is_err()
        );
        let after = coordinator.snapshot().unwrap();
        assert_eq!(before.domains, after.domains);
        assert_eq!(after.active_sequences(), 0);
    }

    #[test]
    fn waiter_recheck_closes_release_and_growth_races() {
        let coordinator = coordinator(1);
        let request = request(&coordinator);
        let observed = wait_for_domains(&coordinator, &[1]);
        let registration = coordinator.register_waiter(observed).unwrap();
        assert!(!registration.recheck().unwrap().should_retry());

        coordinator
            .set_domain_total(domain(1), CapacityUnits::new(11))
            .unwrap();
        let recheck = registration.recheck().unwrap();
        assert!(recheck.changed_since_registration());
        assert_eq!(recheck.current().capacity_epoch(), 2);

        let lease = admit_sequence(
            &coordinator,
            &request,
            &demand(&[(1, 1), (2, 1)], &[(1, 1), (2, 1)]),
        );
        let before_release = wait_for_domains(&coordinator, &[1]);
        drop(lease);
        let late_registration = coordinator.register_waiter(before_release).unwrap();
        assert!(late_registration
            .recheck()
            .unwrap()
            .changed_since_observation());
        assert_eq!(coordinator.epochs().unwrap().capacity_epoch(), 2);
    }

    #[test]
    fn waiter_retries_only_when_its_exact_domain_changes() {
        let coordinator = coordinator(8);
        let observed = wait_for_domains(&coordinator, &[1]);
        let registration = coordinator.register_waiter(observed).unwrap();

        let before = coordinator.epochs().unwrap();
        coordinator
            .notify_domain_availability_changed(domain(2))
            .unwrap();
        let unrelated = registration.recheck().unwrap();
        assert_eq!(
            unrelated.current().capacity_epoch(),
            before.capacity_epoch() + 1
        );
        assert!(!unrelated.changed_since_observation());
        assert!(!unrelated.changed_since_registration());
        assert!(!unrelated.should_retry());

        coordinator
            .notify_domain_availability_changed(domain(1))
            .unwrap();
        let relevant = registration.recheck().unwrap();
        assert!(relevant.changed_since_observation());
        assert!(relevant.changed_since_registration());
        assert!(relevant.should_retry());
    }

    #[test]
    fn wait_snapshot_keeps_pre_mutation_audit_and_exact_source_together() {
        let coordinator = coordinator(8);
        let snapshot = coordinator
            .wait_snapshot_for_domains([domain(1), domain(2)])
            .unwrap()
            .narrow_to_domains([domain(1)])
            .unwrap();
        let observed_epochs = snapshot.epochs();

        coordinator
            .notify_domain_availability_changed(domain(1))
            .unwrap();
        let mut current = Vec::new();
        let current_epochs = coordinator.write_availability_epochs(&mut current).unwrap();

        assert_eq!(
            snapshot.wait_condition().observed(),
            &[
                CapacityAvailabilityEpoch::new(CapacityAvailabilitySource::Domain(domain(1)), 1,)
                    .unwrap()
            ]
        );
        assert_eq!(observed_epochs.capacity_epoch(), 1);
        assert_eq!(current_epochs.capacity_epoch(), 2);
        assert!(snapshot.wait_condition().changed_since(&current).unwrap());
    }

    #[test]
    fn active_sequence_waiter_retries_when_a_slot_is_released() {
        let coordinator = coordinator(1);
        let first_request = request(&coordinator);
        let first = admit_sequence(&coordinator, &first_request, &empty_demand());
        let second_request = request(&coordinator);
        let deferred = match coordinator
            .try_admit_sequence_for_request(&second_request, &empty_demand())
            .unwrap()
        {
            AdmissionDecision::Deferred(deferred) => deferred,
            _ => panic!("active-sequence ceiling must defer the second sequence"),
        };
        assert_eq!(
            deferred.wait_condition().observed()[0].source(),
            CapacityAvailabilitySource::ActiveSequenceSlots
        );
        let registration = coordinator
            .register_waiter(deferred.wait_condition().clone())
            .unwrap();
        assert!(!registration.recheck().unwrap().should_retry());

        drop(first);
        assert!(registration.recheck().unwrap().should_retry());
    }

    #[test]
    fn changed_source_cannot_hide_another_source_regression() {
        let first = CapacityAvailabilitySource::Domain(domain(1));
        let second = CapacityAvailabilitySource::Domain(domain(2));
        let observed = CapacityWaitCondition::from_observation(
            41,
            vec![
                CapacityAvailabilityEpoch::new(first, 5).unwrap(),
                CapacityAvailabilityEpoch::new(second, 5).unwrap(),
            ],
        )
        .unwrap();
        let current = vec![
            CapacityAvailabilityEpoch::new(first, 6).unwrap(),
            CapacityAvailabilityEpoch::new(second, 4).unwrap(),
        ];

        assert!(matches!(
            observed.changed_since(&current),
            Err(VNextError::DynamicAdmissionContract {
                kind: DynamicAdmissionFaultKind::EpochRegression,
                ..
            })
        ));
    }

    #[test]
    fn allocator_availability_change_advances_capacity_epoch_without_recounting_units() {
        let coordinator = coordinator(8);
        let before = coordinator.snapshot().unwrap();
        let observed = wait_for_domains(&coordinator, &[1]);
        let registration = coordinator.register_waiter(observed).unwrap();

        let changed = coordinator
            .notify_domain_availability_changed(domain(1))
            .unwrap();
        let after = coordinator.snapshot().unwrap();
        assert_eq!(changed.capacity_epoch(), before.capacity_epoch() + 1);
        assert_eq!(changed.release_epoch(), before.release_epoch());
        assert_eq!(after.domains(), before.domains());
        assert!(registration.recheck().unwrap().should_retry());

        let before_reject = coordinator.epochs().unwrap();
        assert!(coordinator
            .notify_domain_availability_changed(domain(99))
            .is_err());
        assert_eq!(coordinator.epochs().unwrap(), before_reject);
    }

    #[test]
    fn multi_domain_growth_validates_all_before_one_epoch_commit() {
        let coordinator = coordinator(8);
        let before = coordinator.snapshot().unwrap();
        assert!(coordinator
            .set_domain_totals(&[
                (domain(1), CapacityUnits::new(11)),
                (domain(2), CapacityUnits::new(5)),
            ])
            .is_err());
        let rejected = coordinator.snapshot().unwrap();
        assert_eq!(before.domains, rejected.domains);
        assert_eq!(before.capacity_epoch(), rejected.capacity_epoch());

        let committed = coordinator
            .set_domain_totals(&[
                (domain(1), CapacityUnits::new(11)),
                (domain(2), CapacityUnits::new(4)),
            ])
            .unwrap();
        assert_eq!(committed.capacity_epoch(), before.capacity_epoch() + 1);
        assert_eq!(
            coordinator.snapshot().unwrap().domains()[0].total().get(),
            11
        );
    }

    #[tokio::test]
    async fn waiter_listener_closes_recheck_to_park_race() {
        let coordinator = coordinator(1);
        let request = request(&coordinator);
        let lease = admit_sequence(
            &coordinator,
            &request,
            &demand(&[(1, 1), (2, 1)], &[(1, 1), (2, 1)]),
        );
        let observed_epochs = coordinator.epochs().unwrap();
        let observed = wait_for_domains(&coordinator, &[1]);
        let registration = coordinator.register_waiter(observed).unwrap();
        assert!(!registration.recheck().unwrap().should_retry());
        drop(lease);
        let changed = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            registration.wait_for_change(),
        )
        .await
        .expect("listener must not miss a release between recheck and park")
        .unwrap();
        assert!(changed.release_epoch() > observed_epochs.release_epoch());
        assert_eq!(changed.coordinator_id(), coordinator.id());
    }

    #[tokio::test]
    async fn mutation_unwind_wakes_parked_waiter_with_terminal_error() {
        let coordinator = coordinator(1);
        let observed = wait_for_domains(&coordinator, &[1]);
        let registration = coordinator.register_waiter(observed).unwrap();
        let waiter = tokio::spawn(async move { registration.wait_for_change().await });
        tokio::task::yield_now().await;

        let inner = Arc::clone(&coordinator.inner);
        let _ = thread::spawn(move || {
            let _mutation = inner.lock_mutation().unwrap();
            panic!("inject mutation unwind");
        })
        .join();

        let result = tokio::time::timeout(std::time::Duration::from_secs(1), waiter)
            .await
            .expect("mutation unwind must wake an already parked waiter")
            .expect("waiter task must not panic");
        assert!(matches!(
            result,
            Err(VNextError::DynamicAdmissionContract {
                kind: DynamicAdmissionFaultKind::Poisoned,
                ..
            })
        ));
        assert!(coordinator.snapshot().unwrap().poisoned());
    }

    #[test]
    fn sequence_unwind_releases_lease_without_poisoning_coordinator() {
        let coordinator = coordinator(1);
        let request = request(&coordinator);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _lease = admit_sequence(
                &coordinator,
                &request,
                &demand(&[(1, 2), (2, 1)], &[(1, 2), (2, 1)]),
            );
            panic!("inject request panic outside coordinator mutation");
        }));
        assert!(result.is_err());

        let snapshot = coordinator.snapshot().unwrap();
        assert!(!snapshot.poisoned());
        assert_eq!(snapshot.active_sequences(), 0);
        assert_eq!(snapshot.live_sequence_records(), 0);
        assert!(snapshot
            .domains()
            .iter()
            .all(|domain| domain.used().get() == 0));

        let retry = coordinator
            .try_admit_sequence_for_request(&request, &demand(&[(1, 2), (2, 1)], &[(1, 2), (2, 1)]))
            .unwrap();
        assert!(matches!(retry, AdmissionDecision::Admitted(_)));
    }

    #[test]
    fn request_unwind_releases_request_capacity_without_poisoning_coordinator() {
        let coordinator = coordinator(1);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _request = admitted_request(
                coordinator
                    .try_admit_request(&demand(&[(1, 2), (2, 1)], &[(1, 2), (2, 1)]))
                    .unwrap(),
            );
            panic!("inject request scope unwind");
        }));
        assert!(result.is_err());
        let snapshot = coordinator.snapshot().unwrap();
        assert!(!snapshot.poisoned());
        assert_eq!(snapshot.active_requests(), 0);
        assert_eq!(snapshot.active_sequences(), 0);
        assert!(snapshot
            .domains()
            .iter()
            .all(|domain| domain.used().get() == 0));
    }

    #[test]
    fn poisoned_drop_retains_claim_and_is_observable() {
        let coordinator = coordinator(8);
        let request = request(&coordinator);
        let lease = admit_sequence(
            &coordinator,
            &request,
            &demand(&[(1, 2), (2, 1)], &[(1, 2), (2, 1)]),
        );
        let inner = Arc::clone(&coordinator.inner);
        let _ = thread::spawn(move || {
            let _guard = inner.state.lock().unwrap();
            panic!("poison admission state for conservative-drop test");
        })
        .join();
        drop(lease);
        let snapshot = coordinator.snapshot().unwrap();
        assert!(snapshot.poisoned());
        assert_eq!(snapshot.active_sequences(), 1);
        assert_eq!(snapshot.live_sequence_records(), 1);
        assert_eq!(snapshot.domains()[0].used().get(), 2);
    }
}
