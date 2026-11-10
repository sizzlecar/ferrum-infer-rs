//! Prepare survivors during forward work and joins during initialization.
//! This owns only immutable geometry and metadata, never a native/Host grant.
use super::*;
use crate::image_generation::plan_cache::PendingCandidatePlanCache;
use expansion::downstream::{PreparationPermit, PreparationTask};

pub(super) struct PreparedContinuation<R: DeviceRuntime> {
    pub prepared: CompiledSteady<R>,
    pub candidate_cache: PendingCandidatePlanCache<R>,
}

pub(super) struct PendingMembership<T> {
    geometry: Arc<[RequestGeometry]>,
    task: PreparationTask<T>,
}

impl<T> PendingMembership<T> {
    pub(super) fn finish_for(self, geometry: &[&RequestGeometry]) -> Result<Option<T>> {
        if self.geometry.len() != geometry.len()
            || !self.geometry.iter().zip(geometry).all(|(a, b)| a == *b)
        {
            // Cancellation or changed membership invalidates the proposal.
            // Dropping joins its worker and rolls back uncommitted metadata.
            return Ok(None);
        }
        // Finish at the original membership admission boundary. In particular,
        // an unfinished optional join must not add extra old-cohort forwards.
        self.task.finish().map(Some)
    }
}

fn survivor_indices(steps: impl Iterator<Item = usize>, total: usize) -> Option<Vec<usize>> {
    let steps = steps.collect::<Vec<_>>();
    let first_finishing_step = *steps.iter().max()?;
    let remaining = total.checked_sub(first_finishing_step)?;
    if !(1..=2).contains(&remaining) {
        return None;
    }
    // One earlier opportunity avoids contending with downstream preparation
    // launched at the last-wave boundary. The recipe has no live timestep;
    // intervening joins/cancellations still require exact membership matching.
    let survivors = steps
        .into_iter()
        .enumerate()
        .filter_map(|(index, step)| (step < first_finishing_step).then_some(index))
        .collect::<Vec<_>>();
    (!survivors.is_empty()).then_some(survivors)
}

impl<R: DeviceRuntime> ContinuousBatch<R> {
    pub(super) fn prepare_survivors(
        &mut self,
        pipeline: &PH3ImagePipeline<R>,
        requests: &[&mut ContinuousRequest<R>],
    ) {
        if self.pending_membership.is_some() || pipeline.dit_input_diagnostics.is_some() {
            return;
        }
        let Some(indices) = survivor_indices(
            requests.iter().map(|request| request.step_index()),
            pipeline.execution.sources.assets.schedule.steps().len(),
        ) else {
            return;
        };
        let geometry: Arc<[RequestGeometry]> = indices
            .into_iter()
            .map(|index| requests[index].geometry.clone())
            .collect();
        self.prepare_membership(pipeline, geometry);
    }

    pub(super) fn prepare_join(
        &mut self,
        pipeline: &PH3ImagePipeline<R>,
        requests: &[&mut ContinuousRequest<R>],
        joining: &RequestGeometry,
    ) {
        // Never wait for or replace an existing proposal before initialization.
        // The original boundary will retire it if membership has changed.
        if self.pending_membership.is_some()
            || pipeline.dit_input_diagnostics.is_some()
            || self.active.is_some()
            || requests.is_empty()
            || requests.len() >= pipeline.execution.config.maximum_batch_size
            || self.members.len() != requests.len()
            || !Arc::ptr_eq(
                &self.pipeline,
                &pipeline.execution.service_preparation_identity,
            )
            || self.members.iter().zip(requests).any(|(id, request)| {
                id != &request.geometry.identity
                    || id == &joining.identity
                    || request.is_complete()
                    || !Arc::ptr_eq(&request.pipeline, &self.pipeline)
            })
        {
            return;
        }
        let geometry = requests
            .iter()
            .map(|r| r.geometry.clone())
            .chain(std::iter::once(joining.clone()))
            .collect();
        self.prepare_membership(pipeline, geometry);
    }

    fn prepare_membership(
        &mut self,
        pipeline: &PH3ImagePipeline<R>,
        geometry: Arc<[RequestGeometry]>,
    ) {
        // Share the existing single bounded preparation slot. A busy slot
        // leaves the original synchronous admission path unchanged.
        let Some(permit) = PreparationPermit::try_acquire(&pipeline.downstream_preparation, None)
        else {
            return;
        };
        let owned = geometry.clone();
        let fork = pipeline.metadata_preparation_fork();
        let task =
            match permit.spawn(move || fork.compile_continuation_recipe(owned.iter().collect())) {
                Ok(task) => task,
                Err(error) => {
                    tracing::debug!(%error, "H3 membership preparation uses synchronous fallback");
                    return;
                }
            };
        self.pending_membership = Some(PendingMembership { geometry, task });
    }
}

impl<R: DeviceRuntime> PH3ImagePipeline<R> {
    pub(super) fn compile_continuation_recipe(
        &self,
        members: Vec<&RequestGeometry>,
    ) -> Result<PreparedContinuation<R>> {
        ensure!(
            self.execution.execution_host.is_none(),
            "H3 continuation metadata must not retain a scoped execution Host account"
        );
        let candidate_cache = self.execution.plan_cache.begin_candidate()?;
        let assets = &self.execution.sources.assets;
        let geometry = BatchGeometry::new(
            assets.denoiser.transformer(),
            members,
            self.execution.config.maximum_batch_size,
            assets.limits.denoiser.max_metadata_bytes,
        )?;
        let prepared = self.compile_steady_batch(&geometry)?;
        Ok(PreparedContinuation {
            prepared,
            candidate_cache: candidate_cache.defer()?,
        })
    }
}

#[cfg(test)]
mod tests;
