//! Process-local normalized weights, bound to the original verified source.
//! Only an admitted installation account may build the private file. Ready
//! mappings retain no Host scope and can be reused by later installations.
use super::*;
use ferrum_quantization::{
    ProcessMappedDenseLimits, ProcessMappedDenseUploadPolicy, ProcessMappedDenseWeights,
};
use std::{path::PathBuf, sync::Mutex};

pub(super) struct CacheBinding {
    directory: PathBuf,
    limits: ProcessMappedDenseLimits,
    pub(super) upload_policy: ProcessMappedDenseUploadPolicy,
    ready: Mutex<Option<Arc<ProcessMappedDenseWeights>>>,
}

impl CacheBinding {
    fn new(
        directory: PathBuf,
        limits: ProcessMappedDenseLimits,
        upload_policy: ProcessMappedDenseUploadPolicy,
    ) -> Result<Self, VNextError> {
        if directory.as_os_str().is_empty() || !directory.is_dir() {
            return Err(invalid(
                "process-mapped weight directory must be an existing directory",
            ));
        }
        Ok(Self {
            directory,
            limits,
            upload_policy,
            ready: Mutex::new(None),
        })
    }
}

impl ManagedProductionModel {
    /// Explicitly select a bounded, process-private execution-precision cache.
    /// This validates metadata only. The first actual installation builds it
    /// using its original Host account; no persistent artifact is trusted.
    pub fn configure_process_mapped_weights(
        &mut self,
        directory: PathBuf,
        limits: ProcessMappedDenseLimits,
    ) -> ferrum_types::Result<()> {
        self.configure_process_mapped_weights_with_upload_policy(
            directory,
            limits,
            ProcessMappedDenseUploadPolicy::ReadOnlyMapping,
        )
    }

    /// Select how runtimes may upload the immutable normalized bytes. Staged
    /// uses the runtime's existing accounted transfer staging; it does not
    /// change the representation, reserve capacity or retain a Host scope.
    pub fn configure_process_mapped_weights_with_upload_policy(
        &mut self,
        directory: PathBuf,
        limits: ProcessMappedDenseLimits,
        upload_policy: ProcessMappedDenseUploadPolicy,
    ) -> ferrum_types::Result<()> {
        if self.mapped_cache.is_some() {
            return Err(model_error("process-mapped weights are already configured"));
        }
        ProcessMappedDenseWeights::requirements(
            &self
                .prepared
                .family()
                .stored_weight_schema()
                .map_err(model_error)?
                .components,
            limits,
        )
        .map_err(model_error)?;
        self.mapped_cache = Some(Arc::new(
            CacheBinding::new(directory, limits, upload_policy).map_err(model_error)?,
        ));
        Ok(())
    }

    /// Share only the exact source mapping and physical representation recipe.
    /// This is used by separately prepared text and visual-causal profiles.
    pub fn share_process_mapped_weights_from(&mut self, other: &Self) -> ferrum_types::Result<()> {
        if self.mapped_cache.is_some()
            || !Arc::ptr_eq(&self.archive, &other.archive)
            || self
                .prepared
                .family()
                .stored_weight_schema()
                .map_err(model_error)?
                != other
                    .prepared
                    .family()
                    .stored_weight_schema()
                    .map_err(model_error)?
        {
            return Err(model_error(
                "process-mapped sharing requires an unconfigured exact source and physical schema",
            ));
        }
        self.mapped_cache = Some(Arc::clone(other.mapped_cache.as_ref().ok_or_else(
            || model_error("source model has no process-mapped weight configuration"),
        )?));
        Ok(())
    }

    pub fn process_mapped_weights_enabled(&self) -> bool {
        self.mapped_cache.is_some()
    }

    pub fn process_mapped_upload_policy(&self) -> Option<ProcessMappedDenseUploadPolicy> {
        self.mapped_cache.as_ref().map(|cache| cache.upload_policy)
    }

    pub(super) fn rebind_mapped_cache(
        &self,
        prepared: &PreparedProductionModel,
    ) -> ferrum_types::Result<Option<Arc<CacheBinding>>> {
        let Some(cache) = &self.mapped_cache else {
            return Ok(None);
        };
        if self
            .prepared
            .family()
            .stored_weight_schema()
            .map_err(model_error)?
            == prepared
                .family()
                .stored_weight_schema()
                .map_err(model_error)?
        {
            return Ok(Some(Arc::clone(cache)));
        }
        // A different valid execution precision/recipe gets its own bounded
        // cache. Source authority remains the original verified mapping.
        ProcessMappedDenseWeights::requirements(
            &prepared
                .family()
                .stored_weight_schema()
                .map_err(model_error)?
                .components,
            cache.limits,
        )
        .map_err(model_error)?;
        Ok(Some(Arc::new(
            CacheBinding::new(cache.directory.clone(), cache.limits, cache.upload_policy)
                .map_err(model_error)?,
        )))
    }
}

impl ManagedProductionWeightSource<'_> {
    /// Freeze the ready mapping into this source lease. Errors never install a
    /// partial cache or silently fall back to repeated source conversion.
    /// Cold capacity queries intentionally remain conservative after a hit.
    pub fn prepare_process_mapped_weights(&self) -> Result<(), VNextError> {
        let Some(binding) = self.mapped_cache else {
            return Ok(());
        };
        if self.mapped_ready.get().is_some() {
            return Ok(());
        }
        let components = &self.prepared.family().stored_weight_schema()?.components;
        let mut ready = binding
            .ready
            .lock()
            .map_err(|_| invalid("process-mapped cache lock poisoned"))?;
        let weights = match ready.as_ref() {
            Some(weights) => {
                if !weights.matches_components(components) {
                    return Err(invalid("process-mapped cache physical schema differs"));
                }
                Arc::clone(weights)
            }
            None => {
                let start = std::time::Instant::now();
                let weights = Arc::new(ProcessMappedDenseWeights::build(
                    self.archive,
                    components,
                    self.host,
                    &binding.directory,
                    binding.limits,
                )?);
                tracing::info!(
                    elapsed_seconds = start.elapsed().as_secs_f64(),
                    components = components.len(),
                    "Prepared process-mapped execution weights with the installation Host account"
                );
                *ready = Some(Arc::clone(&weights));
                weights
            }
        };
        // Concurrent callers can obtain only the same immutable binding.
        let _ = self.mapped_ready.set(weights);
        Ok(())
    }
}
