//! Atomic replacement within the existing Host account and scope ledger.
use super::*;
use std::fmt;

/// Refund routing for actual allocations/reservations whose original scope
/// was exchanged. Only refunds follow this route: old account views cannot
/// spend the new authority. Entries disappear with their last original charge
/// or when the successor scope closes. Targets are flattened on every exchange.
#[derive(Debug)]
pub(super) struct ScopeRefund {
    pub(super) target: u64,
    pub(super) remaining: Counters,
}

/// Failure preserves the exact original guard and all its original claims.
#[derive(Debug)]
#[must_use = "recover the original scope; a failed exchange did not replace its authority"]
pub struct HostMemoryScopeExchangeFailure {
    error: HostMemoryError,
    scope: HostMemoryScope,
}
impl HostMemoryScopeExchangeFailure {
    pub fn error(&self) -> &HostMemoryError {
        &self.error
    }
    pub fn into_parts(self) -> (HostMemoryError, HostMemoryScope) {
        (self.error, self.scope)
    }
}
impl fmt::Display for HostMemoryScopeExchangeFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.error, f)
    }
}
impl std::error::Error for HostMemoryScopeExchangeFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

impl HostMemoryAccount {
    /// Replace an idle/fenced scope with a fresh authorization atomically.
    ///
    /// `minimum_free_credit` contains physical bytes still needed *in addition
    /// to already live allocations/reservations*. In particular an existing
    /// WeightBacking is not requested again. New free credit is the per-class
    /// maximum of the old unused credit and the requested minimum; this keeps
    /// the old execution's restart/rollback capacity until a new plan commits.
    /// Existing used bytes remain charged exactly once, with unchanged owners.
    /// Their later refunds recycle into the new scope while it remains live.
    ///
    /// Every capacity/identity/arithmetic check precedes mutation under the
    /// same account lock. Failure returns the unchanged old guard. Success
    /// closes the old scoped views and returns a different authority; there is
    /// no release/reacquire window in which queue admission can steal credit.
    /// Unscoped classes retain ordinary parent behavior, as with scope close.
    ///
    /// This is Host authorization only. The caller must first fence/pause old
    /// native work, freshly prepare the replacement and actually admit its
    /// device plan before publishing model/request membership changes. If that
    /// admission fails, return this new guard to the old execution owner.
    pub fn exchange_scope(
        &self,
        scope: HostMemoryScope,
        minimum_free_credit: &[(HostMemoryClass, usize)],
    ) -> Result<HostMemoryScope, HostMemoryScopeExchangeFailure> {
        let exchanged = (|| -> Result<HostMemoryScope, HostMemoryError> {
            if self.1.is_some() {
                return Err(HostMemoryError::NestedScope);
            }
            if !self.same_parent(&scope.account) {
                return Err(HostMemoryError::InvalidReservation);
            }
            let (requested, requested_classes) = self.scope_demand(minimum_free_credit)?;
            let old_authority = scope
                .account
                .1
                .as_ref()
                .ok_or(HostMemoryError::InvalidReservation)?;
            let mut state = self.0.state.lock().map_err(|_| HostMemoryError::Poisoned)?;
            if state.fail_closed {
                return Err(HostMemoryError::Poisoned);
            }
            let old = state
                .scopes
                .get(&old_authority.id)
                .ok_or(HostMemoryError::ScopeClosed)?;
            let used = old
                .capacity
                .subtract(old.remaining)
                .ok_or(HostMemoryError::Poisoned)?;
            let direct_used = used
                .subtract(old.inherited)
                .ok_or(HostMemoryError::Poisoned)?;
            let classes = std::array::from_fn(|i| old_authority.classes[i] || requested_classes[i]);
            let free_classes =
                std::array::from_fn(|i| old.remaining.classes[i].max(requested.classes[i]));
            let free = Counters {
                total: free_classes
                    .iter()
                    .try_fold(0usize, |sum, n| sum.checked_add(*n))
                    .ok_or(HostMemoryError::ArithmeticOverflow)?,
                classes: free_classes,
            };
            let capacity = used.add(free).ok_or(HostMemoryError::ArithmeticOverflow)?;
            let baseline = state
                .committed
                .subtract(old.remaining)
                .ok_or(HostMemoryError::Poisoned)?;
            let committed = self.check_admission_at(baseline, free)?;
            let next_id = state
                .next_scope
                .checked_add(1)
                .ok_or(HostMemoryError::ArithmeticOverflow)?;
            let new_id = state.next_scope;
            let authority = Arc::new(ScopeAuthority {
                id: new_id,
                classes,
            });
            // All fallible checks are complete. Existing refund origins stay
            // distinct, so chained exchanges cannot count inherited bytes twice.
            for refund in state.scope_refunds.values_mut() {
                if refund.target == old_authority.id {
                    refund.target = new_id;
                }
            }
            if direct_used.total != 0 {
                state.scope_refunds.insert(
                    old_authority.id,
                    ScopeRefund {
                        target: new_id,
                        remaining: direct_used,
                    },
                );
            }
            state.scopes.remove(&old_authority.id);
            state.scopes.insert(
                new_id,
                ScopeCredit {
                    capacity,
                    remaining: free,
                    inherited: used,
                },
            );
            state.next_scope = next_id;
            state.committed = committed;
            drop(state);
            self.0.capacity_changed.notify();
            Ok(HostMemoryScope {
                account: HostMemoryAccount(self.0.clone(), Some(authority)),
            })
        })();
        match exchanged {
            Ok(new) => {
                drop(scope);
                Ok(new)
            }
            Err(error) => Err(HostMemoryScopeExchangeFailure { error, scope }),
        }
    }
}

#[cfg(test)]
mod tests;
