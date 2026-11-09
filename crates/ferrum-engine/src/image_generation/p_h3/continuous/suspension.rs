//! An auxiliary phase may temporarily replace the native DiT installation.
//! Host samples, solver state and progress credit never leave their owners.
use super::super::stages::CompiledDenoiser;
use super::*;

/// Exact, separately charged input to RefineOnly. A refined cache is tied to
/// its original native transaction and cannot authenticate a cold reinstall.
pub(super) struct Restoration<R: DeviceRuntime> {
    pub conditioning: ManagedHostTensor,
    pub install: CompiledDenoiser<R>,
    pub refine: CompiledDenoiser<R>,
    pub work_budget: ExecutionHostBudget,
}

fn maximum(a: ExecutionHostBudget, b: ExecutionHostBudget) -> ExecutionHostBudget {
    ExecutionHostBudget {
        input: a.input.max(b.input),
        output: a.output.max(b.output),
        staging: a.staging.max(b.staging),
        weights: a.weights.max(b.weights),
    }
}

impl<R: DeviceRuntime> PH3ImagePipeline<R> {
    /// The auxiliary Host phase has already been admitted, without acquiring
    /// native resources. A false result leaves the original installation live.
    pub(super) fn suspend_continuous_batch(
        &mut self,
        batch: &mut ContinuousBatch<R>,
        requests: &mut [&mut ContinuousRequest<R>],
    ) -> std::result::Result<bool, ImageGenerationFailure<R>> {
        batch.validate(self, requests, false).map_err(process)?;
        ensure_ready(batch, requests).map_err(process)?;
        let mut budget = batch.work_budget;
        for request in requests.iter() {
            let restore = request
                .restoration
                .as_ref()
                .context("missing exact H3 connector")
                .map_err(process)?;
            require(
                request.cache.as_ref().is_some_and(|cache| {
                    cache
                        .key()
                        .matches_snapshot(request.weights().unwrap().weights().evidence())
                }),
                "H3 suspension requires its original authenticated cache",
            )
            .map_err(process)?;
            self.execution
                .validate_capacity_with_prefix(
                    &[
                        &restore.install.stage,
                        &restore.refine.stage,
                        &batch.prepared.stage,
                    ],
                    None,
                )
                .map_err(process)?;
            budget = maximum(budget, restore.work_budget);
        }
        // Preserve the old floor even if optional admission fails. This is
        // the same ledger exchange, with no detached grant or nested scope.
        let old = batch
            .work
            .take()
            .context("H3 suspended cohort lost work authority")
            .map_err(process)?;
        let scope = match self.execution.host.exchange_scope(old, &budget.claims()) {
            Ok(scope) => scope,
            Err(failure) => {
                let (source, old) = failure.into_parts();
                batch.work = Some(old);
                if source.is_temporary_capacity() {
                    return Ok(false);
                }
                return Err(process(source.into()));
            }
        };
        batch.work = Some(scope);
        batch.suspended_floor = Some(batch.work_budget);
        batch.work_budget = budget;
        batch.suspended = true;
        for request in requests.iter_mut() {
            drop(request.weights.take());
        }
        if !self
            .execution
            .release_weight_owner(ImageWeightOwnerKey::Denoiser, &mut batch.stages)?
        {
            // Another actual pin is still live. No close happened; restore
            // these exact attachments, without changing any cache identity.
            for request in requests.iter_mut() {
                let pin = self
                    .execution
                    .pin_weight_owner(ImageWeightOwnerKey::Denoiser)
                    .map_err(process)?;
                require(
                    request.cache.as_ref().is_some_and(|cache| {
                        cache.key().matches_snapshot(pin.weights().evidence())
                    }),
                    "unpinned H3 installation changed before suspension",
                )
                .map_err(process)?;
                request.weights = Some(pin);
            }
            batch.work_budget = batch
                .suspended_floor
                .take()
                .expect("saved original progress floor");
            batch.suspended = false;
            batch.shrink_idle_work(self).map_err(process)?;
            return Ok(false);
        }
        // Only the original acknowledged close makes these old-transaction
        // caches obsolete. Their refunds remain in each persistent scope.
        for request in requests.iter_mut() {
            drop(request.cache.take());
        }
        Ok(true)
    }

    /// Called only after the auxiliary's original Retired boundary. Every
    /// error is fatal to this suspended owner, including clean native refusal:
    /// the previous installation no longer exists and cannot be advertised as
    /// a runnable rollback. No sample, RNG or solver cursor is changed here.
    pub(super) fn restore_continuous_batch(
        &mut self,
        batch: &mut ContinuousBatch<R>,
        requests: &mut [&mut ContinuousRequest<R>],
    ) -> std::result::Result<(), ImageGenerationFailure<R>> {
        if !batch.suspended {
            return Ok(());
        }
        batch.validate(self, requests, false).map_err(process)?;
        let host = batch
            .work
            .as_ref()
            .context("H3 restoration lost its work grant")
            .map_err(process)?
            .account();
        require(
            self.execution.execution_host.is_none(),
            "H3 restoration overlaps another Host phase",
        )
        .map_err(process)?;
        self.execution.execution_host = Some(host.clone());
        let result = self.restore_continuous_inner(batch, requests, host);
        self.execution.execution_host = None;
        result
    }

    fn restore_continuous_inner(
        &mut self,
        batch: &mut ContinuousBatch<R>,
        requests: &mut [&mut ContinuousRequest<R>],
        host: HostMemoryAccount,
    ) -> std::result::Result<(), ImageGenerationFailure<R>> {
        let first = requests[0]
            .restoration
            .as_ref()
            .context("H3 restoration has no connector recipe")
            .map_err(process)?;
        let active = self.execution.start_stage(
            first.install.stage.clone(),
            batch.job,
            &mut batch.stages,
        )?;
        let _installation = match self
            .execution
            .pin_weight_owner(ImageWeightOwnerKey::Denoiser)
        {
            Ok(pin) => pin,
            Err(source) => return Err(active.processing(source)),
        };
        self.execution.finish_stage(active, &mut batch.stages)?;
        let mut restored = Vec::with_capacity(requests.len());
        for request in requests.iter() {
            let restore = request
                .restoration
                .as_ref()
                .context("H3 restoration has no original connector")
                .map_err(process)?;
            let pin = self
                .execution
                .pin_weight_owner(ImageWeightOwnerKey::Denoiser)
                .map_err(process)?;
            let mut cache = self.execute_refinement(
                &restore.refine,
                request.job,
                &request.persistent,
                host.clone(),
                &request.geometry.identity,
                &restore.conditioning,
                &pin,
                &mut batch.stages,
            )?;
            require(
                cache.len() == 1
                    && cache[0].key().request_identity() == &request.geometry.identity
                    && cache[0].key().matches_snapshot(pin.weights().evidence()),
                "restored H3 cache does not authenticate its new installation",
            )
            .map_err(process)?;
            restored.push((pin, cache.pop().unwrap()));
        }
        // Actual native admission precedes publication of any restored cache.
        let active = self.execution.start_stage(
            batch.prepared.stage.clone(),
            batch.job,
            &mut batch.stages,
        )?;
        if let Err(source) = batch
            .prepared
            .recipe
            .validate_attached_session(&active.session, restored[0].0.weights())
        {
            return Err(active.processing(source));
        }
        for (request, (pin, cache)) in requests.iter_mut().zip(restored) {
            request.weights = Some(pin);
            request.cache = Some(cache);
        }
        batch.active = Some(active);
        batch.work_budget = batch
            .suspended_floor
            .take()
            .context("H3 restoration lost its original progress floor")
            .map_err(process)?;
        batch.suspended = false;
        // Source conversion/upload/readback has fenced. Retain the old shared
        // steady/decoder floor, returning only the temporary restoration peak.
        batch.shrink_idle_work(self).map_err(process)?;
        Ok(())
    }
}

fn require(condition: bool, diagnostic: &'static str) -> Result<()> {
    ensure!(condition, "{diagnostic}");
    Ok(())
}

fn ensure_ready<R: DeviceRuntime>(
    batch: &ContinuousBatch<R>,
    requests: &[&mut ContinuousRequest<R>],
) -> Result<()> {
    ensure!(
        !batch.suspended && batch.suspended_floor.is_none() && batch.active.is_none(),
        "H3 suspension requires a fenced resident cohort"
    );
    ensure!(
        requests.iter().all(|request| !request.is_complete()
            && request.restoration.is_some()
            && request.weights.is_some()
            && request.cache.is_some()),
        "H3 suspension requires live requests and retained connector inputs"
    );
    Ok(())
}
