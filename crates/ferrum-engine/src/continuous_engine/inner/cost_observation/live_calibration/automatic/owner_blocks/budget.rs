//! One work ledger per original offer stream. Source retries never mint work.
//! Memory, optional journal disk bytes, and CPU work are different quantities.
use super::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub(in crate::continuous_engine::inner::cost_observation) struct Work {
    pub canonical_source_bytes: u64,
    pub readiness_scalar_visits: u64,
    pub readiness_replay_upper_bound: u64,
    pub phase_transition_attempts: u64,
    pub imports: u64,
}
impl Work {
    pub(super) fn add(self, other: Self) -> Option<Self> {
        Some(Self {
            canonical_source_bytes: self
                .canonical_source_bytes
                .checked_add(other.canonical_source_bytes)?,
            readiness_scalar_visits: self
                .readiness_scalar_visits
                .checked_add(other.readiness_scalar_visits)?,
            readiness_replay_upper_bound: self
                .readiness_replay_upper_bound
                .checked_add(other.readiness_replay_upper_bound)?,
            phase_transition_attempts: self
                .phase_transition_attempts
                .checked_add(other.phase_transition_attempts)?,
            imports: self.imports.checked_add(other.imports)?,
        })
    }
    fn sub(self, other: Self) -> Option<Self> {
        Some(Self {
            canonical_source_bytes: self
                .canonical_source_bytes
                .checked_sub(other.canonical_source_bytes)?,
            readiness_scalar_visits: self
                .readiness_scalar_visits
                .checked_sub(other.readiness_scalar_visits)?,
            readiness_replay_upper_bound: self
                .readiness_replay_upper_bound
                .checked_sub(other.readiness_replay_upper_bound)?,
            phase_transition_attempts: self
                .phase_transition_attempts
                .checked_sub(other.phase_transition_attempts)?,
            imports: self.imports.checked_sub(other.imports)?,
        })
    }
    fn mul(self, n: u64) -> Option<Self> {
        Some(Self {
            canonical_source_bytes: self.canonical_source_bytes.checked_mul(n)?,
            readiness_scalar_visits: self.readiness_scalar_visits.checked_mul(n)?,
            readiness_replay_upper_bound: self.readiness_replay_upper_bound.checked_mul(n)?,
            phase_transition_attempts: self.phase_transition_attempts.checked_mul(n)?,
            imports: self.imports.checked_mul(n)?,
        })
    }
    fn min(self, cap: Self) -> Self {
        Self {
            canonical_source_bytes: self.canonical_source_bytes.min(cap.canonical_source_bytes),
            readiness_scalar_visits: self
                .readiness_scalar_visits
                .min(cap.readiness_scalar_visits),
            readiness_replay_upper_bound: self
                .readiness_replay_upper_bound
                .min(cap.readiness_replay_upper_bound),
            phase_transition_attempts: self
                .phase_transition_attempts
                .min(cap.phase_transition_attempts),
            imports: self.imports.min(cap.imports),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub(in crate::continuous_engine::inner::cost_observation) struct BudgetAudit {
    pub maximum_source_slots: usize,
    pub source_slot_bytes: usize,
    pub collector_bytes_per_source: usize,
    pub catalog_bytes_per_source: usize,
    pub source_metadata_bytes: usize,
    pub original_blocks_credited: u64,
    pub source_reservations: usize,
    pub available: Work,
    pub reserved_remaining: Work,
    pub spent: Work,
    /// A failed constructor did not return a collector audit. Its whole
    /// unobserved reservation is charged conservatively, never reported as
    /// an observed byte/visit count and never refunded.
    pub forfeited_upper_bound: Work,
    pub minted: Work,
    pub discarded_at_burst_limit: Work,
    pub shared_burst_capacity: Work,
    /// Work units above cover source encoding and readiness. Numerical
    /// transitions/imports are counted, not claimed as total CPU nanoseconds.
    pub numerical_cpu_time_measured: bool,
}

pub(super) struct Budget {
    pub maximum_sources: usize,
    pub slot_bytes: usize,
    pub collector_bytes: usize,
    pub catalog_bytes: usize,
    source_metadata_bytes: usize,
    quantum: Work,
    capacity: Work,
    available: Work,
    spent: Work,
    forfeited: Work,
    minted: Work,
    discarded: Work,
    reservations: Vec<(u64, Work)>,
    credited_block: u64,
    completed_blocks: u64,
}
impl Budget {
    pub fn new(
        settings: &SloAutomaticCalibrationSettingsV1,
        share: usize,
        schedule: &OwnerBlockScheduleV1,
        coordinator_metadata_per_source: usize,
    ) -> Result<Self, FerrumError> {
        let maximum_sources = settings
            .maximum_retained_generations
            .get()
            .checked_add(1)
            .ok_or_else(|| error("rolling source slots overflow"))?;
        let owners = u64::try_from(settings.maximum_owners.get()).map_err(error)?;
        let transitions = owners
            .checked_mul(3)
            .ok_or_else(|| error("rolling transition allowance overflow"))?;
        let visits = schedule
            .input_readiness
            .as_ref()
            .ok_or_else(|| error("rolling sources require input readiness"))?
            .maximum_geometry_visits;
        let quantum = Work {
            canonical_source_bytes: settings.maximum_encoded_source_bytes.get(),
            readiness_scalar_visits: transitions
                .checked_mul(visits / 2)
                .ok_or_else(|| error("rolling geometry allowance overflow"))?,
            readiness_replay_upper_bound: transitions
                .checked_mul(visits / 2)
                .ok_or_else(|| error("rolling replay geometry allowance overflow"))?,
            phase_transition_attempts: transitions,
            imports: owners,
        };
        let capacity = quantum
            .mul(maximum_sources as u64)
            .ok_or_else(|| error("rolling work capacity overflow"))?;
        let reservations = Vec::<(u64, Work)>::with_capacity(maximum_sources);
        let source_metadata_bytes = reservations
            .capacity()
            .checked_mul(std::mem::size_of::<(u64, Work)>())
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Self>()))
            .map(|bytes| bytes.div_ceil(maximum_sources))
            .and_then(|bytes| bytes.checked_add(coordinator_metadata_per_source))
            .ok_or_else(|| error("rolling coordinator retained metadata overflow"))?;
        let payload = share
            .checked_sub(source_metadata_bytes)
            .ok_or_else(|| error("rolling source slot cannot hold coordinator metadata"))?;
        let collector_bytes = payload / 2;
        if collector_bytes == 0 {
            return Err(error(
                "rolling source slot cannot hold collector/catalog partitions",
            ));
        }
        Ok(Self {
            maximum_sources,
            slot_bytes: share,
            collector_bytes,
            catalog_bytes: payload - collector_bytes,
            source_metadata_bytes,
            quantum,
            capacity,
            available: quantum,
            spent: Work::default(),
            forfeited: Work::default(),
            minted: quantum,
            discarded: Work::default(),
            reservations,
            credited_block: 0,
            completed_blocks: 0,
        })
    }
    /// Exactly one credit per genuinely completed original block. Repeated
    /// publication ACK/worker calls and failed/incomplete blocks cannot refill it.
    pub fn complete_original_block(&mut self, block: u64) -> Result<(), FerrumError> {
        if block == self.credited_block {
            return Ok(());
        }
        if block < self.credited_block {
            return Err(error("rolling original block credit sequence"));
        }
        let offered = self
            .available
            .add(self.quantum)
            .ok_or_else(|| error("rolling work refill overflow"))?;
        self.available = offered.min(
            self.capacity
                .sub(self.reserved_remaining())
                .ok_or_else(|| error("rolling shared work reservation overflow"))?,
        );
        self.minted = self
            .minted
            .add(self.quantum)
            .ok_or_else(|| error("rolling minted work overflow"))?;
        self.discarded = self
            .discarded
            .add(
                offered
                    .sub(self.available)
                    .ok_or_else(|| error("rolling discarded work differs"))?,
            )
            .ok_or_else(|| error("rolling discarded work overflow"))?;
        self.credited_block = block;
        self.completed_blocks = self
            .completed_blocks
            .checked_add(1)
            .ok_or_else(|| error("rolling original block count overflow"))?;
        Ok(())
    }
    pub fn reserve(&mut self, generation: u64) -> bool {
        if self.reservations.iter().any(|(id, _)| *id == generation)
            || self.reservations.len() >= self.maximum_sources
        {
            return false;
        }
        let Some(next) = self.available.sub(self.quantum) else {
            return false;
        };
        self.available = next;
        self.reservations.push((generation, Work::default()));
        true
    }
    pub fn charge(&mut self, generation: u64, observed: Work) -> Result<(), FerrumError> {
        let old = self
            .reservations
            .iter_mut()
            .find(|(id, _)| *id == generation)
            .map(|(_, used)| used)
            .ok_or_else(|| error("rolling source has no original work reservation"))?;
        self.quantum
            .sub(observed)
            .ok_or_else(|| error("rolling source exceeded original work reservation"))?;
        let delta = observed
            .sub(*old)
            .ok_or_else(|| error("rolling source work counter decreased"))?;
        self.spent = self
            .spent
            .add(delta)
            .ok_or_else(|| error("rolling cumulative work overflow"))?;
        *old = observed;
        Ok(())
    }
    pub fn release(&mut self, generation: u64) -> Result<(), FerrumError> {
        let index = self
            .reservations
            .iter()
            .position(|(id, _)| *id == generation)
            .ok_or_else(|| error("rolling source work released twice"))?;
        let (_, used) = self.reservations.remove(index);
        let remaining = self
            .quantum
            .sub(used)
            .ok_or_else(|| error("rolling remaining work overflow"))?;
        self.available = self
            .available
            .add(remaining)
            .ok_or_else(|| error("rolling work refund overflow"))?
            .min(
                self.capacity
                    .sub(self.reserved_remaining())
                    .ok_or_else(|| error("rolling shared work refund overflow"))?,
            );
        Ok(())
    }
    pub fn forfeit(&mut self, generation: u64) -> Result<(), FerrumError> {
        let index = self
            .reservations
            .iter()
            .position(|(id, _)| *id == generation)
            .ok_or_else(|| error("rolling source work forfeited twice"))?;
        let (_, used) = self.reservations.remove(index);
        self.forfeited = self
            .forfeited
            .add(
                self.quantum
                    .sub(used)
                    .ok_or_else(|| error("rolling forfeited work differs"))?,
            )
            .ok_or_else(|| error("rolling forfeited work overflow"))?;
        Ok(())
    }
    /// Consumes reservation ownership even when the final observation is
    /// outside its declared bounds. The invalid observation is not relabeled
    /// as measured work: its unobserved remainder is conservatively forfeited.
    pub fn close(&mut self, generation: u64, observed: Work) -> Result<(), FerrumError> {
        match self.charge(generation, observed) {
            Ok(()) => self.release(generation),
            Err(reason) => {
                self.forfeit(generation)?;
                Err(reason)
            }
        }
    }
    fn reserved_remaining(&self) -> Work {
        self.reservations
            .iter()
            .fold(Work::default(), |sum, (_, used)| {
                sum.add(
                    self.quantum
                        .sub(*used)
                        .expect("charged work fits reservation"),
                )
                .expect("reserved work fits capacity")
            })
    }
    pub fn audit(&self) -> BudgetAudit {
        let reserved_remaining = self.reserved_remaining();
        BudgetAudit {
            maximum_source_slots: self.maximum_sources,
            source_slot_bytes: self.slot_bytes,
            collector_bytes_per_source: self.collector_bytes,
            catalog_bytes_per_source: self.catalog_bytes,
            source_metadata_bytes: self.source_metadata_bytes,
            original_blocks_credited: self.completed_blocks,
            source_reservations: self.reservations.len(),
            available: self.available,
            reserved_remaining,
            spent: self.spent,
            forfeited_upper_bound: self.forfeited,
            minted: self.minted,
            discarded_at_burst_limit: self.discarded,
            shared_burst_capacity: self.capacity,
            numerical_cpu_time_measured: false,
        }
    }
}
