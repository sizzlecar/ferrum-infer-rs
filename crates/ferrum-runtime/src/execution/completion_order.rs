use super::{invalid_plan, BTreeMap, NodeId, PlanNode, VNextError};

mod dag_index;
use dag_index::DagIndex;

/// Bounded reachability caches for repeated completion-order queries.
///
/// Slot derivation visits intervals in first-user order. All candidate slots
/// for one interval ask about the same successor, so keeping its ancestors
/// avoids repeating the dependency traversal. Offline placement also compares
/// one earlier interval with many later ones, so it can reuse the descendants
/// of one predecessor. Both caches retain O(nodes + edges) storage, not an
/// N-by-N reachability table, and change no ordering or slot selection rule.
pub(super) struct CompletionOrderCache<'a> {
    indices: BTreeMap<&'a NodeId, usize>,
    dependencies: Vec<Option<Vec<usize>>>,
    successors: Vec<Vec<usize>>,
    successor: Option<&'a NodeId>,
    ancestors: Vec<bool>,
    predecessor: Option<usize>,
    previous_query_predecessor: Option<usize>,
    descendants: Vec<bool>,
    complete_graph: bool,
    dag_index: Option<DagIndex>,
    pending: Vec<usize>,
    missing_dependency: bool,
    #[cfg(test)]
    traversals: usize,
}

impl<'a> CompletionOrderCache<'a> {
    pub(super) fn new(nodes_by_id: &'a BTreeMap<NodeId, &'a PlanNode>) -> Self {
        let mut indices = nodes_by_id
            .keys()
            .enumerate()
            .map(|(index, id)| (id, index))
            .collect::<BTreeMap<_, _>>();
        // Missing dependencies also need an identity: DFS encounters their ID
        // before reporting the missing node, and that prefix remains observable.
        for node in nodes_by_id.values() {
            for dependency in &node.dependencies {
                let next = indices.len();
                indices.entry(dependency).or_insert(next);
            }
        }
        let mut dependencies: Vec<Option<Vec<usize>>> = vec![None; indices.len()];
        for (id, node) in nodes_by_id {
            dependencies[indices[id]] = Some(
                node.dependencies
                    .iter()
                    .map(|dependency| indices[dependency])
                    .collect(),
            );
        }
        let complete_graph = dependencies.iter().all(Option::is_some);
        let mut successors = vec![Vec::new(); indices.len()];
        for (node, parents) in dependencies.iter().enumerate() {
            if let Some(parents) = parents {
                for &parent in parents {
                    successors[parent].push(node);
                }
            }
        }
        let dag_index = DagIndex::new(&dependencies, &successors);
        Self {
            ancestors: vec![false; indices.len()],
            descendants: vec![false; indices.len()],
            indices,
            dependencies,
            successors,
            successor: None,
            predecessor: None,
            previous_query_predecessor: None,
            complete_graph,
            dag_index,
            pending: Vec::new(),
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
        let predecessor_index = self.indices.get(predecessor).copied();
        if let (Some(index), Some(from), Some(&to)) = (
            &self.dag_index,
            predecessor_index,
            self.indices.get(successor),
        ) {
            if let Some(precedes) = index.precedes(from, to) {
                return Ok(precedes);
            }
        }
        let repeated_predecessor =
            predecessor_index.is_some() && predecessor_index == self.previous_query_predecessor;
        self.previous_query_predecessor = predecessor_index;
        if self.successor != Some(successor) {
            // Incomplete graphs must retain the original ancestor DFS's exact
            // missing-node error/prefix behavior. A reverse traversal cannot
            // establish that behavior, even if it finds a path.
            if self.complete_graph {
                if let (Some(from), Some(&to)) = (predecessor_index, self.indices.get(successor)) {
                    if self.predecessor == Some(from) {
                        return Ok(self.descendants[to]);
                    }
                    if repeated_predecessor {
                        self.fill_descendants(from);
                        return Ok(self.descendants[to]);
                    }
                }
            }
            self.fill(successor)?;
        }
        if self
            .indices
            .get(predecessor)
            .is_some_and(|&index| self.ancestors[index])
        {
            return Ok(true);
        }
        if self.missing_dependency {
            return Err(invalid_plan("completion-order dependency node is missing"));
        }
        Ok(false)
    }

    fn fill_descendants(&mut self, predecessor: usize) {
        self.pending.clear();
        self.pending
            .extend_from_slice(&self.successors[predecessor]);
        self.descendants.fill(false);
        while let Some(index) = self.pending.pop() {
            if std::mem::replace(&mut self.descendants[index], true) {
                continue;
            }
            self.pending.extend_from_slice(&self.successors[index]);
        }
        self.predecessor = Some(predecessor);
    }

    fn fill(&mut self, successor: &NodeId) -> Result<(), VNextError> {
        let (&id, &index) = self
            .indices
            .get_key_value(successor)
            .ok_or_else(|| invalid_plan("completion-order successor node is missing"))?;
        let dependencies = self.dependencies[index]
            .as_ref()
            .ok_or_else(|| invalid_plan("completion-order successor node is missing"))?;
        self.pending.clear();
        self.pending.extend_from_slice(dependencies);
        self.ancestors.fill(false);
        self.missing_dependency = false;
        #[cfg(test)]
        {
            self.traversals += 1;
        }
        while let Some(index) = self.pending.pop() {
            if std::mem::replace(&mut self.ancestors[index], true) {
                continue;
            }
            let Some(dependencies) = &self.dependencies[index] else {
                // The original query checks equality before resolving this
                // dependency. Retain this ID and the exact DFS prefix: a
                // predecessor already encountered still returns true, while
                // any other query observes the original missing-node error.
                self.missing_dependency = true;
                break;
            };
            self.pending.extend_from_slice(dependencies);
        }
        self.successor = Some(id);
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn traversal_count(&self) -> usize {
        self.traversals
    }

    #[cfg(test)]
    pub(super) fn retained_ancestor_count(&self) -> usize {
        self.ancestors.iter().filter(|&&visited| visited).count()
    }
}
