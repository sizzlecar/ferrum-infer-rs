//! Explicit phase commitment on the original Host scope ledger.
use super::*;

#[derive(Clone, Copy)]
enum FreeCreditTarget {
    Exact,
    AtMost,
}

impl HostMemoryAccount {
    /// Return unused credit after a replacement/phase has actually committed.
    ///
    /// Unlike `exchange_scope`, this intentionally gives up the old rollback
    /// floor. Call it only after the new native work has been admitted and the
    /// previous execution no longer needs that credit. It neither admits work
    /// nor creates a scope, allocation, account, or replacement authority.
    ///
    /// `free_credit` is the exact remaining physical credit to retain in each
    /// scoped class, in addition to live allocations/reservations. Omitted
    /// scoped classes retain zero unused credit; they stay scoped, so a cached
    /// account view cannot escape its reduced ceiling through parent admission.
    /// Duplicate class entries are summed. A class that was not scoped is an
    /// invalid target, including an explicit zero. Increasing free credit fails
    /// with `ScopeCapacity`; use ordinary admission/exchange for growth.
    ///
    /// Live direct/inherited charges and their original refund routes remain
    /// unchanged. Their later refunds still become reusable scope credit while
    /// the guard lives, or return to the parent after close. No payload is copied.
    /// All checks precede mutation under the original account lock. Any error
    /// leaves the guard, remaining credit, charges and refund lineage unchanged.
    pub fn shrink_scope_unused(
        &self,
        scope: &HostMemoryScope,
        free_credit: &[(HostMemoryClass, usize)],
    ) -> Result<(), HostMemoryError> {
        self.shrink_scope_unused_impl(scope, free_credit, FreeCreditTarget::Exact)
    }

    /// Cap each scoped class's current unused credit after a committed phase.
    ///
    /// Each retained amount is `min(current_unused, maximum_free_credit)`.
    /// A maximum above current free credit does not grant additional bytes or
    /// fail because a still-live output consumed them. Omitted scoped classes
    /// retain zero free credit. Used/inherited bytes and later refund routing
    /// remain unchanged, as for `shrink_scope_unused`; old refunds may later
    /// make another explicit cap useful at the next completed phase boundary.
    ///
    /// This is not admission for the next phase. First obtain its true minimum
    /// through ordinary reservation/exchange. Parent, class, closed and checked
    /// arithmetic validation remain identical to the exact commitment method.
    pub fn shrink_scope_unused_to_at_most(
        &self,
        scope: &HostMemoryScope,
        maximum_free_credit: &[(HostMemoryClass, usize)],
    ) -> Result<(), HostMemoryError> {
        self.shrink_scope_unused_impl(scope, maximum_free_credit, FreeCreditTarget::AtMost)
    }

    fn shrink_scope_unused_impl(
        &self,
        scope: &HostMemoryScope,
        free_credit: &[(HostMemoryClass, usize)],
        mode: FreeCreditTarget,
    ) -> Result<(), HostMemoryError> {
        if self.1.is_some() {
            return Err(HostMemoryError::NestedScope);
        }
        if !self.same_parent(&scope.account) {
            return Err(HostMemoryError::InvalidReservation);
        }
        let (mut requested, requested_classes) = self.scope_demand(free_credit)?;
        let authority = scope
            .account
            .1
            .as_ref()
            .ok_or(HostMemoryError::InvalidReservation)?;
        if requested_classes
            .iter()
            .zip(authority.classes)
            .any(|(&requested, scoped)| requested && !scoped)
        {
            return Err(HostMemoryError::InvalidReservation);
        }
        let mut state = self.0.state.lock().map_err(|_| HostMemoryError::Poisoned)?;
        if state.fail_closed {
            return Err(HostMemoryError::Poisoned);
        }
        let old = state
            .scopes
            .get(&authority.id)
            .ok_or(HostMemoryError::ScopeClosed)?;
        for class in [
            HostMemoryClass::InputPreprocess,
            HostMemoryClass::TransferStaging,
            HostMemoryClass::Output,
            HostMemoryClass::WeightBacking,
        ] {
            let i = class.index();
            match mode {
                FreeCreditTarget::Exact if requested.classes[i] > old.remaining.classes[i] => {
                    return Err(HostMemoryError::ScopeCapacity {
                        class,
                        requested: requested.classes[i],
                        available: old.remaining.classes[i],
                    });
                }
                FreeCreditTarget::AtMost => {
                    requested.classes[i] = requested.classes[i].min(old.remaining.classes[i]);
                }
                FreeCreditTarget::Exact => (),
            }
        }
        requested.total = requested
            .classes
            .iter()
            .try_fold(0usize, |sum, n| sum.checked_add(*n))
            .ok_or(HostMemoryError::ArithmeticOverflow)?;
        let released = old
            .remaining
            .subtract(requested)
            .ok_or(HostMemoryError::Poisoned)?;
        let capacity = old
            .capacity
            .subtract(released)
            .ok_or(HostMemoryError::Poisoned)?;
        let committed = state
            .committed
            .subtract(released)
            .ok_or(HostMemoryError::Poisoned)?;
        // Everything fallible is complete. Used/inherited amounts, authority
        // identity and refund origins do not change during this commitment.
        let credit = state
            .scopes
            .get_mut(&authority.id)
            .expect("scope checked under the same ledger lock");
        credit.capacity = capacity;
        credit.remaining = requested;
        state.committed = committed;
        drop(state);
        if released.total != 0 {
            self.0.capacity_changed.notify();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
