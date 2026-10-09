//! Test-only mutations of real admitted pools. No authority is manufactured.
use super::*;

pub(crate) fn workspace_test_chunk_owners<R: DeviceRuntime>(
    root: &PlanRuntimeResources<R>,
) -> Vec<(DynamicBackingPoolId, u32, usize, usize)> {
    let mut owners = Vec::new();
    for (pool_id, pool) in &root.dynamic_pools.pools {
        let state = pool.state.lock().unwrap();
        for (ordinal, chunk) in &state.chunks {
            owners.push((
                pool_id.clone(),
                *ordinal,
                Arc::as_ptr(&chunk.backing) as usize,
                Arc::strong_count(&chunk.backing),
            ));
        }
    }
    owners
}

pub(crate) struct WorkspaceTestPoolPoison<R: DeviceRuntime> {
    pools: Vec<(Arc<DynamicBackingPool<R>>, bool)>,
}

pub(crate) fn workspace_test_poison_pools<R: DeviceRuntime>(
    root: &PlanRuntimeResources<R>,
) -> WorkspaceTestPoolPoison<R> {
    let pools = root
        .dynamic_pools
        .pools
        .values()
        .map(|pool| {
            let mut state = pool.state.lock().unwrap();
            let prior = state.poisoned;
            state.poisoned = true;
            (Arc::clone(pool), prior)
        })
        .collect();
    WorkspaceTestPoolPoison { pools }
}

impl<R: DeviceRuntime> Drop for WorkspaceTestPoolPoison<R> {
    fn drop(&mut self) {
        for (pool, prior) in &self.pools {
            pool.state.lock().unwrap().poisoned = *prior;
        }
    }
}

pub(crate) struct WorkspaceTestChunkReplacement<R: DeviceRuntime> {
    target: Arc<DynamicBackingPool<R>>,
    donor: Arc<DynamicBackingPool<R>>,
    ordinal: u32,
    original: Option<ResidentChunkState<R::Buffer>>,
    donor_identity: BackingChunkIdentity,
}

pub(crate) fn workspace_test_replace_chunk_from_donor<R: DeviceRuntime>(
    target: &PlanRuntimeResources<R>,
    donor: &PlanRuntimeResources<R>,
    pool_id: &DynamicBackingPoolId,
    ordinal: u32,
) -> WorkspaceTestChunkReplacement<R> {
    let target = Arc::clone(&target.dynamic_pools.pools[pool_id]);
    let donor = Arc::clone(&donor.dynamic_pools.pools[pool_id]);
    assert!(!Arc::ptr_eq(&target, &donor));
    // Keep both real grants and their pool ownership alive. The donor has not
    // materialized a view, so only its pool owns the physical backing Arc.
    {
        let target_state = target.state.lock().unwrap();
        let donor_state = donor.state.lock().unwrap();
        assert_eq!(
            target_state.chunks[&ordinal].backing.descriptor,
            donor_state.chunks[&ordinal].backing.descriptor,
        );
        assert_eq!(Arc::strong_count(&donor_state.chunks[&ordinal].backing), 1);
    }
    let original = target
        .state
        .lock()
        .unwrap()
        .chunks
        .remove(&ordinal)
        .unwrap();
    let mut replacement = donor.state.lock().unwrap().chunks.remove(&ordinal).unwrap();
    let donor_identity = replacement.backing.identity.clone();
    Arc::get_mut(&mut replacement.backing).unwrap().identity = original.backing.identity.clone();
    target
        .state
        .lock()
        .unwrap()
        .chunks
        .insert(ordinal, replacement);
    WorkspaceTestChunkReplacement {
        target,
        donor,
        ordinal,
        original: Some(original),
        donor_identity,
    }
}

impl<R: DeviceRuntime> Drop for WorkspaceTestChunkReplacement<R> {
    fn drop(&mut self) {
        let mut replacement = self
            .target
            .state
            .lock()
            .unwrap()
            .chunks
            .remove(&self.ordinal)
            .unwrap();
        Arc::get_mut(&mut replacement.backing).unwrap().identity = self.donor_identity.clone();
        self.donor
            .state
            .lock()
            .unwrap()
            .chunks
            .insert(self.ordinal, replacement);
        self.target
            .state
            .lock()
            .unwrap()
            .chunks
            .insert(self.ordinal, self.original.take().unwrap());
    }
}
