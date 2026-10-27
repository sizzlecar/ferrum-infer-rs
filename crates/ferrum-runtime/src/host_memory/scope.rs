//! Execution-scoped credit within an existing Host account. Credit owns no
//! allocation; ordinary consumers of the parent cannot spend reserved bytes.

use super::*;

const CLASSES: [HostMemoryClass; 4] = [
    HostMemoryClass::InputPreprocess,
    HostMemoryClass::TransferStaging,
    HostMemoryClass::Output,
    HostMemoryClass::WeightBacking,
];

#[derive(Debug)]
pub(super) struct ScopeAuthority {
    id: u64,
    classes: [bool; 4],
}

#[derive(Debug)]
pub(super) struct ScopeCredit {
    capacity: Counters,
    remaining: Counters,
}

/// Non-cloneable execution lifetime. Account views may be cloned into sessions
/// and outputs, but cannot keep unused credit reserved after this guard closes.
/// Existing allocations/reservations survive close and retain their real charge.
#[derive(Debug)]
pub struct HostMemoryScope {
    account: HostMemoryAccount,
}

impl HostMemoryScope {
    /// Same parent identity, alignment and limits; only admission routing differs.
    pub fn account(&self) -> HostMemoryAccount {
        self.account.clone()
    }

    /// Release unused credit immediately. Outstanding allocations/reservations
    /// remain valid; their eventual Drop returns capacity directly to the parent.
    pub fn close(self) -> Result<(), HostMemoryError> {
        self.release_unused()
    }

    fn release_unused(&self) -> Result<(), HostMemoryError> {
        let authority = self.account.1.as_ref().expect("scope has authority");
        let mut state = self
            .account
            .0
            .state
            .lock()
            .map_err(|_| HostMemoryError::Poisoned)?;
        if state.fail_closed {
            return Err(HostMemoryError::Poisoned);
        }
        if let Some(credit) = state.scopes.remove(&authority.id) {
            match state.committed.subtract(credit.remaining) {
                Some(committed) => state.committed = committed,
                None => {
                    state.fail_closed = true;
                    drop(state);
                    self.account.0.capacity_changed.notify();
                    return Err(HostMemoryError::Poisoned);
                }
            }
            drop(state);
            self.account.0.capacity_changed.notify();
        }
        Ok(())
    }
}

impl Drop for HostMemoryScope {
    fn drop(&mut self) {
        let _ = self.release_unused();
    }
}

impl HostMemoryAccount {
    /// Validate every physical class/total claim against hard account ceilings,
    /// independently of current occupancy. This does not reserve any capacity.
    pub fn check_scope_capacity(
        &self,
        budgets: &[(HostMemoryClass, usize)],
    ) -> Result<(), HostMemoryError> {
        let (demand, _) = self.scope_demand(budgets)?;
        self.check_hard_capacity(demand, Counters::default())
    }

    /// Check current scope admission without acquiring or refunding credit.
    /// A later `reserve_scope` still provides the actual atomic authorization.
    /// Schedulers can use this before multi-resource admission to avoid a
    /// rollback repeatedly waking the same unsuccessful admission attempt.
    pub fn check_scope_admission(
        &self,
        budgets: &[(HostMemoryClass, usize)],
    ) -> Result<(), HostMemoryError> {
        let (demand, _) = self.scope_demand(budgets)?;
        let state = self.0.state.lock().map_err(|_| HostMemoryError::Poisoned)?;
        if state.fail_closed {
            return Err(HostMemoryError::Poisoned);
        }
        self.check_admission(&state, demand).map(|_| ())
    }

    /// Prove that releasing externally owned Output could make this claim fit.
    /// Other committed classes form an unreleasable baseline for this check.
    /// `retained_output_bytes` is the physical Output credit already held by
    /// the selected execution on this same parent, which cannot be awaited.
    ///
    /// This is only a coherent metadata check, not an admission grant. On
    /// failure, `Capacity.requested` includes the unreleasable baseline and is
    /// therefore a hard requirement, rather than temporary Output occupancy.
    pub fn check_scope_after_output_release(
        &self,
        budgets: &[(HostMemoryClass, usize)],
        retained_output_bytes: usize,
    ) -> Result<(), HostMemoryError> {
        let (demand, _) = self.scope_demand(budgets)?;
        let state = self.0.state.lock().map_err(|_| HostMemoryError::Poisoned)?;
        if state.fail_closed {
            return Err(HostMemoryError::Poisoned);
        }
        let current_output = state.committed.classes[HostMemoryClass::Output.index()];
        if retained_output_bytes > current_output {
            return Err(HostMemoryError::InvalidReservation);
        }
        let external_output = current_output - retained_output_bytes;
        let baseline = state
            .committed
            .subtract(Counters::single(HostMemoryClass::Output, external_output))
            .ok_or(HostMemoryError::Poisoned)?;
        let required = baseline
            .add(demand)
            .ok_or(HostMemoryError::ArithmeticOverflow)?;
        self.check_hard_capacity(required, Counters::default())
    }

    fn scope_demand(
        &self,
        budgets: &[(HostMemoryClass, usize)],
    ) -> Result<(Counters, [bool; 4]), HostMemoryError> {
        if self.1.is_some() {
            return Err(HostMemoryError::NestedScope);
        }
        if budgets.is_empty() {
            return Err(HostMemoryError::InvalidReservation);
        }
        let mut demand = Counters::default();
        let mut classes = [false; 4];
        for &(class, bytes) in budgets {
            demand = demand
                .add(Counters::single(class, bytes))
                .ok_or(HostMemoryError::ArithmeticOverflow)?;
            classes[class.index()] = true;
        }
        Ok((demand, classes))
    }

    /// Reserve physical backing-byte credit for an execution before device work.
    /// Each supplied class receives a hard scope ceiling; classes absent here
    /// continue to use normal parent admission. Duplicate classes are summed.
    /// An explicit zero ceiling forbids new allocations in that class, including
    /// a fully warm execution that requires no additional backing.
    ///
    /// These are already-physical budgets: use `backing_bytes` on each planned
    /// logical allocation, then sum the required simultaneous backing. Padding
    /// is not reapplied to the aggregate. Released allocations return reusable
    /// credit while the scope lives. A live output retains only its own charge
    /// after the guard closes; unused credit is returned even if account views
    /// remain cached elsewhere. Closed views cannot reserve in scoped classes.
    pub fn reserve_scope(
        &self,
        budgets: &[(HostMemoryClass, usize)],
    ) -> Result<HostMemoryScope, HostMemoryError> {
        let (demand, classes) = self.scope_demand(budgets)?;
        let mut state = self.0.state.lock().map_err(|_| HostMemoryError::Poisoned)?;
        if state.fail_closed {
            return Err(HostMemoryError::Poisoned);
        }
        let next = self.check_admission(&state, demand)?;
        let id = state.next_scope;
        let next_id = id
            .checked_add(1)
            .ok_or(HostMemoryError::ArithmeticOverflow)?;
        let authority = Arc::new(ScopeAuthority { id, classes });
        state.scopes.insert(
            id,
            ScopeCredit {
                capacity: demand,
                remaining: demand,
            },
        );
        state.next_scope = next_id;
        state.committed = next;
        Ok(HostMemoryScope {
            account: HostMemoryAccount(self.0.clone(), Some(authority)),
        })
    }

    pub(super) fn check_admission(
        &self,
        state: &State,
        demand: Counters,
    ) -> Result<Counters, HostMemoryError> {
        self.check_hard_capacity(demand, state.committed)?;
        let next = state
            .committed
            .add(demand)
            .ok_or(HostMemoryError::ArithmeticOverflow)?;
        if next.total > self.0.limits.total_bytes {
            return Err(HostMemoryError::Capacity {
                scope: HostMemoryPressureScope::Total,
                requested: demand.total,
                committed: state.committed.total,
                limit: self.0.limits.total_bytes,
            });
        }
        for class in CLASSES {
            let index = class.index();
            if next.classes[index] > self.0.limits.classes[index] {
                return Err(HostMemoryError::Capacity {
                    scope: HostMemoryPressureScope::Class(class),
                    requested: demand.classes[index],
                    committed: state.committed.classes[index],
                    limit: self.0.limits.classes[index],
                });
            }
        }
        Ok(next)
    }

    fn check_hard_capacity(
        &self,
        demand: Counters,
        committed: Counters,
    ) -> Result<(), HostMemoryError> {
        if demand.total > self.0.limits.total_bytes {
            return Err(HostMemoryError::Capacity {
                scope: HostMemoryPressureScope::Total,
                requested: demand.total,
                committed: committed.total,
                limit: self.0.limits.total_bytes,
            });
        }
        for class in CLASSES {
            let index = class.index();
            if demand.classes[index] > self.0.limits.classes[index] {
                return Err(HostMemoryError::Capacity {
                    scope: HostMemoryPressureScope::Class(class),
                    requested: demand.classes[index],
                    committed: committed.classes[index],
                    limit: self.0.limits.classes[index],
                });
            }
        }
        Ok(())
    }

    pub(super) fn scoped_demand(
        &self,
        state: &State,
        demand: Counters,
    ) -> Result<Counters, HostMemoryError> {
        let Some(authority) = &self.1 else {
            return Ok(Counters::default());
        };
        let mut scoped = Counters::default();
        for class in CLASSES {
            let index = class.index();
            if !authority.classes[index] || demand.classes[index] == 0 {
                continue;
            }
            let credit = state
                .scopes
                .get(&authority.id)
                .ok_or(HostMemoryError::ScopeClosed)?;
            if demand.classes[index] > credit.remaining.classes[index] {
                return Err(HostMemoryError::ScopeCapacity {
                    class,
                    requested: demand.classes[index],
                    available: credit.remaining.classes[index],
                });
            }
            scoped = scoped
                .add(Counters::single(class, demand.classes[index]))
                .ok_or(HostMemoryError::ArithmeticOverflow)?;
        }
        Ok(scoped)
    }

    pub(super) fn consume_scope(&self, state: &mut State, demand: Counters) {
        if demand.total == 0 {
            return;
        }
        let authority = self.1.as_ref().expect("nonempty scoped demand");
        let credit = state
            .scopes
            .get_mut(&authority.id)
            .expect("scope checked under same lock");
        credit.remaining = credit
            .remaining
            .subtract(demand)
            .expect("scope demand admitted under same lock");
    }

    /// True means this class is still managed by a live scope, including a
    /// detected accounting inconsistency that must remain fail-closed.
    pub(super) fn refund_scope(&self, state: &mut State, amount: Counters) -> bool {
        let Some(authority) = &self.1 else {
            return false;
        };
        let Some(credit) = state.scopes.get_mut(&authority.id) else {
            return false;
        };
        if amount
            .classes
            .iter()
            .zip(authority.classes)
            .any(|(&bytes, scoped)| bytes != 0 && !scoped)
        {
            return false;
        }
        match credit.remaining.add(amount) {
            Some(remaining)
                if remaining.total <= credit.capacity.total
                    && remaining
                        .classes
                        .iter()
                        .zip(credit.capacity.classes)
                        .all(|(&bytes, cap)| bytes <= cap) =>
            {
                credit.remaining = remaining
            }
            _ => state.fail_closed = true,
        }
        true
    }
}

#[cfg(test)]
mod tests;
