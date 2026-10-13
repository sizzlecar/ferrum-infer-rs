//! Step-local retention in original first-seen order. The node directory only
//! narrows equality candidates; it neither issues nor substitutes authority.
use std::collections::BTreeMap;

use crate::vnext::{NodeId, RetainedPlanDependencyAuthority};

#[derive(Default)]
pub(in crate::vnext) struct StepPlanDependencyRetention {
    // Keep actual authorities first, with the original Vec drop order. The
    // directory holds no lease, allocation, invocation scope or device owner.
    authorities: Vec<RetainedPlanDependencyAuthority>,
    by_node: BTreeMap<NodeId, Vec<usize>>,
}

impl StepPlanDependencyRetention {
    pub(in crate::vnext) fn retain(&mut self, leases: Vec<RetainedPlanDependencyAuthority>) {
        for lease in leases {
            let already_retained =
                self.by_node
                    .get(lease.identity.node_id())
                    .is_some_and(|candidates| {
                        candidates
                            .iter()
                            .any(|&index| self.authorities[index].identity == lease.identity)
                    });
            if already_retained {
                // The caller has already validated this incoming invocation's
                // scope. Preserve the original complete identity equality and
                // retain the first authority, dropping this duplicate here.
                continue;
            }
            let node = lease.identity.node_id().clone();
            let index = self.authorities.len();
            self.authorities.push(lease);
            self.by_node.entry(node).or_default().push(index);
        }
    }

    #[cfg(test)]
    pub(in crate::vnext) fn authorities(&self) -> &[RetainedPlanDependencyAuthority] {
        &self.authorities
    }
}
