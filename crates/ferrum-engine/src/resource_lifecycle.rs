use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ResourceKey {
    owner_kind: String,
    owner_id: String,
    resource_kind: String,
}

impl ResourceKey {
    fn new(owner_kind: &str, owner_id: &str, resource_kind: &str) -> Self {
        Self {
            owner_kind: owner_kind.to_string(),
            owner_id: owner_id.to_string(),
            resource_kind: resource_kind.to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ResourceLedgerState {
    reserved: i64,
    committed: i64,
    released: i64,
    rolled_back: i64,
    capacity: Option<i64>,
}

impl ResourceLedgerState {
    fn outstanding_reserved(self) -> i64 {
        self.reserved
            .saturating_sub(self.released)
            .saturating_sub(self.rolled_back)
    }

    fn outstanding_committed(self) -> i64 {
        self.committed
            .saturating_sub(self.released)
            .saturating_sub(self.rolled_back)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResourceLedgerTransition {
    pub(crate) before: i64,
    pub(crate) after: i64,
    pub(crate) capacity: Option<i64>,
    pub(crate) underflow_amount: Option<i64>,
}

#[derive(Debug, Default)]
pub(crate) struct ResourceLifecycleLedger {
    states: HashMap<ResourceKey, ResourceLedgerState>,
}

impl ResourceLifecycleLedger {
    pub(crate) fn close_owner(&mut self, owner_kind: &str, owner_id: &str) -> usize {
        let before = self.states.len();
        self.states
            .retain(|key, _state| key.owner_kind != owner_kind || key.owner_id != owner_id);
        before.saturating_sub(self.states.len())
    }

    #[cfg(test)]
    fn state_count(&self) -> usize {
        self.states.len()
    }

    pub(crate) fn reserve(
        &mut self,
        owner_kind: &str,
        owner_id: &str,
        resource_kind: &str,
        amount: i64,
        capacity: Option<i64>,
    ) -> ResourceLedgerTransition {
        let state = self.state_mut(owner_kind, owner_id, resource_kind, capacity);
        let amount = amount.max(0);
        let before = state.outstanding_reserved();
        state.reserved = state.reserved.saturating_add(amount);
        let after = state.outstanding_reserved();
        ResourceLedgerTransition {
            before,
            after,
            capacity: state.capacity,
            underflow_amount: None,
        }
    }

    pub(crate) fn commit(
        &mut self,
        owner_kind: &str,
        owner_id: &str,
        resource_kind: &str,
        amount: i64,
        capacity: Option<i64>,
    ) -> ResourceLedgerTransition {
        let state = self.state_mut(owner_kind, owner_id, resource_kind, capacity);
        let amount = amount.max(0);
        let before = state.outstanding_committed();
        state.committed = state.committed.saturating_add(amount);
        let after = state.outstanding_committed();
        ResourceLedgerTransition {
            before,
            after,
            capacity: state.capacity,
            underflow_amount: None,
        }
    }

    pub(crate) fn release(
        &mut self,
        owner_kind: &str,
        owner_id: &str,
        resource_kind: &str,
        amount: i64,
        capacity: Option<i64>,
    ) -> ResourceLedgerTransition {
        let state = self.state_mut(owner_kind, owner_id, resource_kind, capacity);
        let amount = amount.max(0);
        let before = state.outstanding_committed();
        state.released = state.released.saturating_add(amount);
        let after = before.saturating_sub(amount);
        ResourceLedgerTransition {
            before,
            after,
            capacity: state.capacity,
            underflow_amount: (amount > before).then_some(amount - before),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn rollback(
        &mut self,
        owner_kind: &str,
        owner_id: &str,
        resource_kind: &str,
        amount: i64,
        capacity: Option<i64>,
    ) -> ResourceLedgerTransition {
        let state = self.state_mut(owner_kind, owner_id, resource_kind, capacity);
        let amount = amount.max(0);
        let before = state.outstanding_reserved();
        state.rolled_back = state.rolled_back.saturating_add(amount);
        let after = before.saturating_sub(amount);
        ResourceLedgerTransition {
            before,
            after,
            capacity: state.capacity,
            underflow_amount: (amount > before).then_some(amount - before),
        }
    }

    fn state_mut(
        &mut self,
        owner_kind: &str,
        owner_id: &str,
        resource_kind: &str,
        capacity: Option<i64>,
    ) -> &mut ResourceLedgerState {
        let key = ResourceKey::new(owner_kind, owner_id, resource_kind);
        let state = self.states.entry(key).or_default();
        if capacity.is_some() {
            state.capacity = capacity;
        }
        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_commit_release_uses_ledger_state() {
        let mut ledger = ResourceLifecycleLedger::default();

        let reserve = ledger.reserve("request", "req-1", "kv_block", 4, Some(16));
        assert_eq!(
            reserve,
            ResourceLedgerTransition {
                before: 0,
                after: 4,
                capacity: Some(16),
                underflow_amount: None
            }
        );

        let commit = ledger.commit("request", "req-1", "kv_block", 4, None);
        assert_eq!(
            commit,
            ResourceLedgerTransition {
                before: 0,
                after: 4,
                capacity: Some(16),
                underflow_amount: None
            }
        );

        let release = ledger.release("request", "req-1", "kv_block", 2, None);
        assert_eq!(
            release,
            ResourceLedgerTransition {
                before: 4,
                after: 2,
                capacity: Some(16),
                underflow_amount: None
            }
        );

        let final_release = ledger.release("request", "req-1", "kv_block", 2, None);
        assert_eq!(
            final_release,
            ResourceLedgerTransition {
                before: 2,
                after: 0,
                capacity: Some(16),
                underflow_amount: None
            }
        );
    }

    #[test]
    fn rollback_clears_reserved_without_committing() {
        let mut ledger = ResourceLifecycleLedger::default();

        ledger.reserve("request", "req-1", "recurrent_state_slot", 3, Some(8));
        let rollback = ledger.rollback("request", "req-1", "recurrent_state_slot", 3, None);

        assert_eq!(
            rollback,
            ResourceLedgerTransition {
                before: 3,
                after: 0,
                capacity: Some(8),
                underflow_amount: None
            }
        );
    }

    #[test]
    fn release_and_rollback_report_underflow_in_transition() {
        let mut ledger = ResourceLifecycleLedger::default();

        let release = ledger.release("request", "req-1", "kv_block", 2, Some(4));
        assert_eq!(
            release,
            ResourceLedgerTransition {
                before: 0,
                after: -2,
                capacity: Some(4),
                underflow_amount: Some(2)
            }
        );

        ledger.reserve("request", "req-2", "kv_block", 1, Some(4));
        let rollback = ledger.rollback("request", "req-2", "kv_block", 2, None);
        assert_eq!(
            rollback,
            ResourceLedgerTransition {
                before: 1,
                after: -1,
                capacity: Some(4),
                underflow_amount: Some(1)
            }
        );
    }

    #[test]
    fn close_owner_removes_only_that_request_resources() {
        let mut ledger = ResourceLifecycleLedger::default();

        ledger.reserve("request", "req-1", "request_slot", 1, None);
        ledger.reserve("request", "req-1", "kv_block", 4, Some(16));
        ledger.reserve("request", "req-2", "kv_block", 2, Some(16));

        assert_eq!(ledger.state_count(), 3);
        assert_eq!(ledger.close_owner("request", "req-1"), 2);
        assert_eq!(ledger.state_count(), 1);
        assert_eq!(ledger.close_owner("request", "req-1"), 0);
        assert_eq!(ledger.close_owner("request", "req-2"), 1);
        assert_eq!(ledger.state_count(), 0);
    }
}
