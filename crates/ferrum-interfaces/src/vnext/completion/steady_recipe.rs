//! Lane-local preparation hints. Fresh entry and wave authority still govern use.
use super::*;
use crate::vnext::{
    DeviceReusableExecutionEntryIdentity, DeviceReusableExecutionProgramId, SealedNodeRecipe,
};

#[cfg(test)]
mod tests;

pub(super) struct SteadyRecipeCache<R: DeviceRuntime> {
    // One current recipe per compiled node; this is not a history of requests,
    // programs, graph recaptures or dynamic batch shapes.
    nodes: Mutex<BTreeMap<usize, Arc<SealedNodeRecipe<R>>>>,
}

impl<R: DeviceRuntime> Default for SteadyRecipeCache<R> {
    fn default() -> Self {
        Self {
            nodes: Mutex::new(BTreeMap::new()),
        }
    }
}

impl<R: DeviceRuntime> ExecutionLane<R> {
    /// Inspect the exact backend entry at the same quiescent boundary as the
    /// catalog. A retained marker alone cannot authorize another dispatch.
    pub fn reusable_execution_entry_identity(
        &self,
        program: &DeviceReusableExecutionProgramId,
    ) -> Result<Option<DeviceReusableExecutionEntryIdentity>, VNextError> {
        self.with_quiescent_stream("inspect reusable execution entry", |runtime, stream| {
            runtime.reusable_execution_entry_identity(stream, program)
        })
    }

    /// Busy, unsupported and absent entries all lack a current usable identity.
    /// Runtime failures remain errors instead of becoming cache misses.
    pub(crate) fn try_reusable_execution_entry_identity(
        &self,
        program: &DeviceReusableExecutionProgramId,
    ) -> Result<Option<DeviceReusableExecutionEntryIdentity>, VNextError> {
        self.try_with_quiescent_stream(
            "inspect reusable execution entry",
            false,
            |runtime, stream| runtime.reusable_execution_entry_identity(stream, program),
        )
        .map(Option::flatten)
    }

    pub(crate) fn steady_recipe(
        &self,
        node_index: usize,
        provider_owner: &Arc<()>,
        program: &DeviceReusableExecutionProgramId,
        entry: &DeviceReusableExecutionEntryIdentity,
    ) -> Result<Option<Arc<SealedNodeRecipe<R>>>, VNextError> {
        if self.fail_closed.load(Ordering::Acquire) {
            return Err(invalid_completion(
                "fail-closed lane cannot reuse a steady recipe",
            ));
        }
        let Some(cache) = self.steady_recipes.get() else {
            return Ok(None);
        };
        let mut nodes = cache
            .nodes
            .lock()
            .map_err(|_| invalid_completion("steady recipe cache mutex is poisoned"))?;
        let current_provider = Arc::downgrade(provider_owner);
        let matching = nodes.get(&node_index).is_some_and(|recipe| {
            recipe.node_index == node_index
                && recipe.lane == self.id
                && recipe.epoch == self.reusable_execution_epoch()
                && recipe.program == *program
                && recipe.entry.same_entry(entry)
                && Weak::ptr_eq(&recipe.provider_owner, &current_provider)
        });
        let (selected, retired) = if matching {
            (nodes.get(&node_index).cloned(), None)
        } else {
            (None, nodes.remove(&node_index))
        };
        drop(nodes);
        drop(retired);
        Ok(selected)
    }

    /// Called only by successful full-wave completion. Publish only while the
    /// same backend entry is still resident at a quiescent lane boundary.
    pub(crate) fn record_successful_steady_recipe(
        &self,
        recipe: Arc<SealedNodeRecipe<R>>,
    ) -> Result<(), VNextError> {
        if recipe.lane != self.id {
            return Err(invalid_completion(
                "steady recipe belongs to another execution lane",
            ));
        }
        if self.fail_closed.load(Ordering::Acquire)
            || recipe.epoch != self.reusable_execution_epoch()
        {
            // Losing a performance hint must not turn a completed wave into a
            // failure merely because another operation invalidated its catalog.
            return Ok(());
        }
        let Some(_provider_owner) = recipe.provider_owner.upgrade() else {
            return Ok(());
        };
        let state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(()),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(invalid_completion("execution lane state mutex is poisoned"));
            }
        };
        if state.in_flight != 0 || recipe.epoch != self.reusable_execution_epoch() {
            return Ok(());
        }
        let current = self.checked_steady_entry(&state, &recipe.program)?;
        if !current.is_some_and(|entry| entry.same_entry(&recipe.entry)) {
            return Ok(());
        }
        // Lock order is lane, then cache. Runtime/provider work never executes
        // under the cache lock, and retired cold state is dropped after both.
        let cache = self.steady_recipes.get_or_init(SteadyRecipeCache::default);
        let mut nodes = cache
            .nodes
            .lock()
            .map_err(|_| invalid_completion("steady recipe cache mutex is poisoned"))?;
        if self.fail_closed.load(Ordering::Acquire)
            || recipe.epoch != self.reusable_execution_epoch()
        {
            drop(nodes);
            return Ok(());
        }
        let retired = nodes.insert(recipe.node_index, recipe);
        drop(nodes);
        drop(state);
        drop(retired);
        Ok(())
    }

    fn checked_steady_entry(
        &self,
        state: &ExecutionLaneState<R::Stream>,
        program: &DeviceReusableExecutionProgramId,
    ) -> Result<Option<DeviceReusableExecutionEntryIdentity>, VNextError> {
        if state.fail_closed || self.fail_closed.load(Ordering::Acquire) || state.in_flight != 0 {
            return Err(invalid_completion(
                "steady recipe requires a quiescent usable lane",
            ));
        }
        let inspected = catch_unwind(AssertUnwindSafe(|| {
            if !self.current_descriptor_matches_snapshot()
                || self.runtime.stream_state(&state.stream) != StreamState::Ready
            {
                return Err(invalid_completion(
                    "steady recipe entry inspection requires a stable lane",
                ));
            }
            let entry = self
                .runtime
                .reusable_execution_entry_identity(&state.stream, program)
                .map_err(|error| {
                    invalid_completion(format!("steady recipe entry inspection failed: {error}"))
                })?;
            if !self.current_descriptor_matches_snapshot()
                || self.runtime.stream_state(&state.stream) != StreamState::Ready
            {
                return Err(invalid_completion(
                    "steady recipe entry inspection changed the lane state",
                ));
            }
            Ok(entry)
        }));
        let result = inspected.unwrap_or_else(|_| {
            Err(invalid_completion(
                "device runtime panicked while inspecting a steady recipe entry",
            ))
        });
        if result.is_err() {
            self.fail_closed.store(true, Ordering::Release);
        }
        result
    }

    pub(super) fn clear_steady_recipes(&self) -> Result<(), VNextError> {
        let Some(cache) = self.steady_recipes.get() else {
            return Ok(());
        };
        let retired = {
            let mut nodes = cache
                .nodes
                .lock()
                .map_err(|_| invalid_completion("steady recipe cache mutex is poisoned"))?;
            std::mem::take(&mut *nodes)
        };
        drop(retired);
        Ok(())
    }
}

impl<R: DeviceRuntime> ExecutionLaneEnqueue<'_, R> {
    /// Revalidate under the stream guard retained through submission. An entry
    /// queried before provider encoding cannot authorize a replaced graph.
    pub(crate) fn validate_steady_recipe_entry(
        &self,
        program: &DeviceReusableExecutionProgramId,
        entry: &DeviceReusableExecutionEntryIdentity,
        epoch: u64,
    ) -> Result<(), VNextError> {
        if epoch != self.lane.reusable_execution_epoch() {
            return Err(invalid_completion(
                "steady recipe catalog epoch changed before enqueue",
            ));
        }
        let current = self.lane.checked_steady_entry(&self.state, program)?;
        if !current.is_some_and(|current| current.same_entry(entry)) {
            return Err(invalid_completion(
                "steady recipe executable entry changed before enqueue",
            ));
        }
        Ok(())
    }
}

impl<R: DeviceRuntime> CompletionReservation<R> {
    pub(crate) fn stage_steady_recipe(
        &mut self,
        lane: &Arc<ExecutionLane<R>>,
        recipe: Arc<SealedNodeRecipe<R>>,
    ) {
        match self
            .resources
            .as_mut()
            .expect("live completion reservation owns submission resources")
        {
            CompletionResourceLease::Wave(wave) => wave.stage_steady_recipe(lane, recipe),
            CompletionResourceLease::Invocation(_) => {
                unreachable!("steady recipe requires a submission wave reservation")
            }
        }
    }
}
