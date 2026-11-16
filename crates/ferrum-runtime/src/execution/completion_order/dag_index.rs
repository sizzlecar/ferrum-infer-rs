/// Linear-size completion proofs for a complete DAG.
///
/// Every dependency increases `depth`, so a non-increasing pair cannot be
/// ordered. One actual dependency per node also forms a forest: its ancestor
/// intervals prove reachability in constant time. Pairs not proved by either
/// rule still require the original graph traversal. This is not a transitive
/// closure, and it never treats a missing forest path as a missing graph path.
pub(super) struct DagIndex {
    depth: Vec<usize>,
    entered: Vec<usize>,
    finished: Vec<usize>,
}

impl DagIndex {
    pub(super) fn new(
        dependencies: &[Option<Vec<usize>>],
        successors: &[Vec<usize>],
    ) -> Option<Self> {
        let mut remaining = dependencies
            .iter()
            .map(|parents| parents.as_ref().map(Vec::len))
            .collect::<Option<Vec<_>>>()?;
        let count = remaining.len();
        let mut ready = remaining
            .iter()
            .enumerate()
            .filter_map(|(node, &parents)| (parents == 0).then_some(node))
            .collect::<Vec<_>>();
        let mut depth = vec![0; count];
        let mut parent = vec![None; count];
        let mut visited = 0;
        while let Some(node) = ready.pop() {
            visited += 1;
            for &successor in &successors[node] {
                // Choose an actual edge on a longest dependency path. This
                // makes sequential regions cheap without assuming node order.
                let next_depth = depth[node] + 1;
                if next_depth > depth[successor] {
                    depth[successor] = next_depth;
                    parent[successor] = Some(node);
                }
                remaining[successor] -= 1;
                if remaining[successor] == 0 {
                    ready.push(successor);
                }
            }
        }
        // Cycles and incomplete graphs preserve the original DFS behavior.
        if visited != count {
            return None;
        }
        let mut children = vec![Vec::new(); count];
        for (node, &parent) in parent.iter().enumerate() {
            if let Some(parent) = parent {
                children[parent].push(node);
            }
        }
        let mut entered = vec![0; count];
        let mut finished = vec![0; count];
        let mut pending = Vec::new();
        let mut clock = 0;
        for (root, parent) in parent.iter().enumerate() {
            if parent.is_some() {
                continue;
            }
            pending.push((root, false));
            while let Some((node, leaving)) = pending.pop() {
                if leaving {
                    finished[node] = clock;
                } else {
                    entered[node] = clock;
                    clock += 1;
                    pending.push((node, true));
                    pending.extend(children[node].iter().map(|&child| (child, false)));
                }
            }
        }
        Some(Self {
            depth,
            entered,
            finished,
        })
    }

    pub(super) fn precedes(&self, predecessor: usize, successor: usize) -> Option<bool> {
        if self.depth[predecessor] >= self.depth[successor] {
            return Some(false);
        }
        (self.entered[predecessor] <= self.entered[successor]
            && self.entered[successor] < self.finished[predecessor])
            .then_some(true)
    }
}
