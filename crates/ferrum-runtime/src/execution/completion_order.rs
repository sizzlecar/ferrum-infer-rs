use super::{invalid_plan, BTreeMap, BTreeSet, NodeId, PlanNode, VNextError};

/// A bounded, single-successor cache for repeated completion-order queries.
///
/// Slot derivation visits intervals in first-user order. All candidate slots
/// for one interval ask about the same successor, so keeping its ancestors
/// avoids repeating the dependency traversal without retaining an N-by-N
/// reachability table. This cache changes no ordering or slot selection rule.
pub(super) struct CompletionOrderCache<'a> {
    nodes_by_id: &'a BTreeMap<NodeId, &'a PlanNode>,
    successor: Option<NodeId>,
    ancestors: BTreeSet<NodeId>,
    missing_dependency: bool,
    #[cfg(test)]
    traversals: usize,
}

impl<'a> CompletionOrderCache<'a> {
    pub(super) fn new(nodes_by_id: &'a BTreeMap<NodeId, &'a PlanNode>) -> Self {
        Self {
            nodes_by_id,
            successor: None,
            ancestors: BTreeSet::new(),
            missing_dependency: false,
            #[cfg(test)]
            traversals: 0,
        }
    }

    pub(super) fn precedes(
        &mut self,
        predecessor: &NodeId,
        successor: &NodeId,
    ) -> Result<bool, VNextError> {
        // Preserve the original early return, even for an absent node or a
        // cycle that makes a node reachable from itself.
        if predecessor == successor {
            return Ok(false);
        }
        if self.successor.as_ref() != Some(successor) {
            self.fill(successor)?;
        }
        if self.ancestors.contains(predecessor) {
            return Ok(true);
        }
        if self.missing_dependency {
            return Err(invalid_plan("completion-order dependency node is missing"));
        }
        Ok(false)
    }

    fn fill(&mut self, successor: &NodeId) -> Result<(), VNextError> {
        let node = self
            .nodes_by_id
            .get(successor)
            .ok_or_else(|| invalid_plan("completion-order successor node is missing"))?;
        let mut pending = node.dependencies.clone();
        self.ancestors.clear();
        self.missing_dependency = false;
        #[cfg(test)]
        {
            self.traversals += 1;
        }
        while let Some(node_id) = pending.pop() {
            if !self.ancestors.insert(node_id.clone()) {
                continue;
            }
            let Some(node) = self.nodes_by_id.get(&node_id) else {
                // The original query checks equality before resolving this
                // dependency. Retain this ID and the exact DFS prefix: a
                // predecessor already encountered still returns true, while
                // any other query observes the original missing-node error.
                self.missing_dependency = true;
                break;
            };
            pending.extend(node.dependencies.iter().cloned());
        }
        self.successor = Some(successor.clone());
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn traversal_count(&self) -> usize {
        self.traversals
    }

    #[cfg(test)]
    pub(super) fn retained_ancestor_count(&self) -> usize {
        self.ancestors.len()
    }
}
