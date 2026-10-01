//! An independent, fixed offered population discovers future numerical scopes.
//! It receives no latency, fit, residual, or qualification result. Discovery
//! cannot publish a model or authorize an execution; later phases still verify
//! their complete, disjoint original populations and empirical coverage.
use ferrum_interfaces::execution_cost::{
    HostContentDomainV1, HostPendingConstraintV2, HostRowRoleV2, HostTerminalExpectationV1,
    PlainTextSamplingRouteV2,
};
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    StructuredCoverageV2, StructuredInputV2, StructuredOwnerKeyV2, StructuredProductV2,
    StructuredScopeV2, StructuredWaveRoleV2,
};
use std::collections::HashSet;

#[cfg(test)]
mod tests;

const MAX_ROWS: usize = 128;

#[derive(Debug, Clone, Copy)]
pub(super) struct DiscoveryPolicy {
    pub offered_waves: usize,
    pub maximum_owners: usize,
    pub maximum_retained_bytes: usize,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DiscoveryError {
    InvalidPolicy,
    InvalidInput,
    TicketOrFifoDiscontinuity,
    WindowClosed,
    IncompletePopulation,
    OwnerCapacity,
    RetainedCapacity,
    Allocation,
}

/// Small physical facts only; numeric axes, request data, and timings are not
/// retained. The complete actual bitmaps let freeze evaluate challenges against
/// its final eligible set, including an earlier empty/full/intermediate wave.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct HostState {
    pending: u128,
    length: u128,
    eligible: u128,
    forces_full: u128,
    implicit_full: bool,
    plain_text: bool,
}
impl HostState {
    fn from_input(input: &StructuredInputV2) -> Result<Self, DiscoveryError> {
        let rows = input.physical_host_rows();
        if rows.is_empty() || rows.len() > MAX_ROWS || rows.len() != input.owner().rows as usize {
            return Err(DiscoveryError::InvalidInput);
        }
        let mut out = Self {
            pending: 0,
            length: 0,
            eligible: 0,
            forces_full: 0,
            implicit_full: input.owner().role == StructuredWaveRoleV2::Prefill,
            plain_text: true,
        };
        for (position, row) in rows.iter().enumerate() {
            if row.physical_position as usize != position {
                return Err(DiscoveryError::InvalidInput);
            }
            let bit = 1u128 << position;
            // Match the installed future producer: a sampling policy that
            // always reads FullLogits keeps that route even when every
            // unresolved pending bit is clear.
            let always_full = row.role == HostRowRoleV2::Decode
                && matches!(row.installed_policy.empirical_content_domain,
                    Some(HostContentDomainV1::PlainTextInstalledV2(policy))
                        if policy.sampling == PlainTextSamplingRouteV2::FullLogits);
            out.implicit_full |= always_full;
            out.pending |= if row.pending_decoded_utf8 { bit } else { 0 };
            out.length |= if row.terminal_expectation == HostTerminalExpectationV1::LengthBoundary {
                bit
            } else {
                0
            };
            if row.role == HostRowRoleV2::Decode
                && !row.no_generated_history
                && row.decode_requires_full_logits == Some(row.pending_decoded_utf8 || always_full)
            {
                out.eligible |= bit;
            }
            if row.decode_requires_full_logits == Some(true)
                || (row.role == HostRowRoleV2::Prefill && row.final_prefill)
            {
                out.forces_full |= bit;
            }
            out.plain_text &= matches!(
                row.installed_policy.empirical_content_domain,
                Some(
                    HostContentDomainV1::PlainTextGreedyV1
                        | HostContentDomainV1::PlainTextInstalledV2(_)
                )
            );
        }
        Ok(out)
    }
}

struct OwnerDiscovery {
    owner: StructuredOwnerKeyV2,
    states: HashSet<HostState>,
}

/// Only the original worker may feed this window, after private settlement and
/// discovery-ticket verification. Any lost/invalid ticket poisons the window;
/// callers must also propagate failures that never reached the sample FIFO.
pub(super) struct DiscoveryWindow {
    policy: DiscoveryPolicy,
    owners: Vec<OwnerDiscovery>,
    offered: usize,
    opening_fifo_cutoff: u64,
    last_fifo: u64,
    charged_bytes: usize,
    failure: Option<DiscoveryError>,
}

pub(super) struct FrozenDiscovery {
    scopes: Vec<StructuredScopeV2>,
    offered: usize,
    opening_fifo_cutoff: u64,
    closing_fifo_cutoff: u64,
}
impl FrozenDiscovery {
    pub fn scopes(&self) -> &[StructuredScopeV2] {
        &self.scopes
    }
    pub fn into_scopes(self) -> Vec<StructuredScopeV2> {
        self.scopes
    }
    pub fn offered(&self) -> usize {
        self.offered
    }
    pub fn fifo_bounds(&self) -> (u64, u64) {
        (self.opening_fifo_cutoff, self.closing_fifo_cutoff)
    }
    /// An unfamiliar owner belongs to a new independent discovery generation.
    /// It must never extend the scopes already frozen for active training.
    pub fn contains_owner(&self, input: &StructuredInputV2) -> bool {
        self.scopes
            .iter()
            .any(|scope| &scope.owner == input.owner())
    }
    pub fn is_after_discovery(&self, fifo: u64) -> bool {
        fifo > self.closing_fifo_cutoff
    }
}

impl DiscoveryWindow {
    pub fn new(policy: DiscoveryPolicy, opening_fifo_cutoff: u64) -> Result<Self, DiscoveryError> {
        if policy.offered_waves == 0
            || policy.offered_waves > 65_536
            || policy.maximum_owners == 0
            || policy.maximum_owners > 128
            || policy.maximum_retained_bytes < std::mem::size_of::<Self>()
            || policy.maximum_retained_bytes > 512 * 1024 * 1024
        {
            return Err(DiscoveryError::InvalidPolicy);
        }
        Ok(Self {
            policy,
            owners: Vec::new(),
            offered: 0,
            opening_fifo_cutoff,
            last_fifo: opening_fifo_cutoff,
            charged_bytes: std::mem::size_of::<Self>(),
            failure: None,
        })
    }

    pub fn fail(&mut self, reason: DiscoveryError) {
        self.failure.get_or_insert(reason);
    }
    pub fn failure(&self) -> Option<DiscoveryError> {
        self.failure
    }
    pub fn offered(&self) -> usize {
        self.offered
    }
    pub fn complete(&self) -> bool {
        self.failure.is_none() && self.offered == self.policy.offered_waves
    }
    pub fn retained_bytes_upper_bound(&self) -> usize {
        self.charged_bytes
    }

    /// `ticket` is the independent discovery ordinal, not a fit member or a
    /// count of successfully retained samples. FIFO ordinals may contain gaps
    /// for other consumers, but every discovery ticket must appear exactly once.
    pub fn observe(
        &mut self,
        ticket: u64,
        fifo: u64,
        input: &StructuredInputV2,
    ) -> Result<(), DiscoveryError> {
        if let Some(error) = self.failure {
            return Err(error);
        }
        if self.offered == self.policy.offered_waves {
            return Err(DiscoveryError::WindowClosed);
        }
        let result = self.observe_inner(ticket, fifo, input);
        if let Err(error) = result {
            self.fail(error);
        }
        result
    }

    /// A verified outside attempt consumes the original fixed quota without
    /// adding owner, host-policy, support or numerical facts.
    pub fn observe_outside(&mut self, ticket: u64, fifo: u64) -> Result<(), DiscoveryError> {
        if let Some(error) = self.failure {
            return Err(error);
        }
        if self.offered == self.policy.offered_waves {
            return Err(DiscoveryError::WindowClosed);
        }
        if ticket != self.offered as u64 + 1 || fifo <= self.last_fifo {
            self.fail(DiscoveryError::TicketOrFifoDiscontinuity);
            return Err(DiscoveryError::TicketOrFifoDiscontinuity);
        }
        self.offered += 1;
        self.last_fifo = fifo;
        Ok(())
    }

    fn observe_inner(
        &mut self,
        ticket: u64,
        fifo: u64,
        input: &StructuredInputV2,
    ) -> Result<(), DiscoveryError> {
        if ticket != self.offered as u64 + 1 || fifo <= self.last_fifo {
            return Err(DiscoveryError::TicketOrFifoDiscontinuity);
        }
        let state = HostState::from_input(input)?;
        let owner = self
            .owners
            .iter()
            .position(|entry| &entry.owner == input.owner());
        let new_owner = owner.is_none();
        if new_owner && self.owners.len() == self.policy.maximum_owners {
            return Err(DiscoveryError::OwnerCapacity);
        }
        let new_state = owner.is_none_or(|index| !self.owners[index].states.contains(&state));
        let extra =
            if new_owner { owner_charge() } else { 0 } + if new_state { state_charge() } else { 0 };
        let charged = self
            .charged_bytes
            .checked_add(extra)
            .filter(|bytes| *bytes <= self.policy.maximum_retained_bytes)
            .ok_or(DiscoveryError::RetainedCapacity)?;
        // Preserve the conservative bound even if a later allocation fails
        // after the owner container has already grown.
        self.charged_bytes = charged;
        let index = match owner {
            Some(index) => index,
            None => {
                self.owners
                    .try_reserve(1)
                    .map_err(|_| DiscoveryError::Allocation)?;
                self.owners.push(OwnerDiscovery {
                    owner: input.owner().clone(),
                    states: HashSet::new(),
                });
                self.owners.len() - 1
            }
        };
        if new_state {
            self.owners[index]
                .states
                .try_reserve(1)
                .map_err(|_| DiscoveryError::Allocation)?;
            self.owners[index].states.insert(state);
        }
        self.offered += 1;
        self.last_fifo = fifo;
        Ok(())
    }

    /// Consuming this object makes discovery/fitting population overlap a
    /// caller-visible boundary, rather than allowing scopes to mutate in place.
    pub fn freeze(self) -> Result<FrozenDiscovery, DiscoveryError> {
        if let Some(error) = self.failure {
            return Err(error);
        }
        if !self.complete() {
            return Err(DiscoveryError::IncompletePopulation);
        }
        let mut scopes = Vec::new();
        scopes
            .try_reserve_exact(self.owners.len())
            .map_err(|_| DiscoveryError::Allocation)?;
        for owner in self.owners {
            let scope = owner.freeze();
            scope.validate().map_err(|_| DiscoveryError::InvalidInput)?;
            scopes.push(scope);
        }
        Ok(FrozenDiscovery {
            scopes,
            offered: self.offered,
            opening_fifo_cutoff: self.opening_fifo_cutoff,
            closing_fifo_cutoff: self.last_fifo,
        })
    }
}

// Reserve for both retained discovery facts and simultaneously materialized
// scope vectors. Factors cover Vec growth, not inferred numerical coverage.
fn owner_charge() -> usize {
    4 * std::mem::size_of::<OwnerDiscovery>()
        + 4 * std::mem::size_of::<StructuredScopeV2>()
        + 4 * 3 * MAX_ROWS * std::mem::size_of::<u32>()
        + 8 * std::mem::size_of::<HostPendingConstraintV2>()
}
fn state_charge() -> usize {
    4 * (std::mem::size_of::<HostState>() + 1)
        + 16
        + 4 * (2 * std::mem::size_of::<u32>() + std::mem::size_of::<(u32, u32)>())
}

impl OwnerDiscovery {
    fn freeze(self) -> StructuredScopeV2 {
        let mut pending_positions = 0u128;
        let mut length_positions = 0u128;
        let mut eligible = u128::MAX;
        let mut pending_counts = [false; MAX_ROWS + 1];
        let mut length_counts = [false; MAX_ROWS + 1];
        let mut joint = [[false; MAX_ROWS + 1]; MAX_ROWS + 1];
        for state in &self.states {
            pending_positions |= state.pending;
            length_positions |= state.length;
            eligible &= state.eligible;
            let p = state.pending.count_ones() as usize;
            let l = state.length.count_ones() as usize;
            pending_counts[p] = true;
            length_counts[l] = true;
            joint[p][l] = true;
        }
        // A position never observed pending cannot acquire unresolved coverage
        // merely because its static role would permit a future pending state.
        eligible &= pending_positions;
        let mut any = Challenges::default();
        let mut nonempty = Challenges::default();
        for state in &self.states {
            if !state.plain_text {
                continue;
            }
            let fixed_full = state.implicit_full || state.forces_full & !eligible != 0;
            match self.owner.product {
                StructuredProductV2::GreedyToken
                    if eligible == 0 && state.pending == 0 && !fixed_full =>
                {
                    any.observe(state.pending & eligible, eligible);
                }
                StructuredProductV2::FullLogits if fixed_full => {
                    any.observe(state.pending & eligible, eligible);
                }
                StructuredProductV2::FullLogits if state.pending & eligible != 0 => {
                    nonempty.observe(state.pending & eligible, eligible);
                }
                _ => {}
            }
        }
        let mut authorized = Vec::new();
        if any.complete(eligible, true) {
            authorized.push(HostPendingConstraintV2::AnySubset);
        }
        if eligible != 0 && nonempty.complete(eligible, false) {
            authorized.push(HostPendingConstraintV2::NonEmptySubset);
        }
        StructuredScopeV2 {
            numerical_family: None,
            owner: self.owner,
            coverage: StructuredCoverageV2 {
                pending_eligible_positions: positions(eligible),
                authorized_pending_constraints: authorized,
                pending_counts: counts(&pending_counts),
                length_counts: counts(&length_counts),
                pending_positions: positions(pending_positions),
                length_positions: positions(length_positions),
                joint_counts: joint
                    .iter()
                    .enumerate()
                    .flat_map(|(p, lengths)| {
                        lengths
                            .iter()
                            .enumerate()
                            .filter_map(move |(l, seen)| seen.then_some((p as u32, l as u32)))
                    })
                    .collect(),
            },
        }
    }
}

#[derive(Default)]
struct Challenges {
    empty: bool,
    full: bool,
    intermediate: bool,
}
impl Challenges {
    fn observe(&mut self, pending: u128, eligible: u128) {
        self.empty |= pending == 0;
        self.full |= pending == eligible;
        self.intermediate |= pending != 0 && pending != eligible;
    }
    fn complete(&self, eligible: u128, require_empty: bool) -> bool {
        self.full
            && (!require_empty || self.empty)
            && (eligible.count_ones() <= 1 || self.intermediate)
    }
}
fn positions(bitmap: u128) -> Vec<u32> {
    (0..MAX_ROWS as u32)
        .filter(|position| bitmap & (1u128 << position) != 0)
        .collect()
}
fn counts(seen: &[bool; MAX_ROWS + 1]) -> Vec<u32> {
    seen.iter()
        .enumerate()
        .filter_map(|(count, present)| present.then_some(count as u32))
        .collect()
}
