//! Explicit diagnostic capture on the original pipeline and Host parent.
//! The original source receipt still declares its external weight-lock limits.
use super::PH3ImagePipeline;
use crate::image_generation::continuous_trace::{
    ImageContinuousTraceHandle, ImageContinuousTraceLimits, ImageContinuousTraceRecorder,
};
use anyhow::{ensure, Result};
use ferrum_runtime::{DeviceRuntime, HostMemoryClass};
use serde_json::{json, Value};

/// Bounded host timing for recipe preparation. Nested intervals are diagnostic
/// subsets, not additional request time; no tensors or request text are logged.
pub(super) fn metadata_phase<T>(phase: &'static str, prepare: impl FnOnce() -> T) -> T {
    let started =
        tracing::enabled!(target: "ferrum.h3.metadata_preparation", tracing::Level::DEBUG)
            .then(std::time::Instant::now);
    let result = prepare();
    if let Some(started) = started {
        tracing::debug!(
            target: "ferrum.h3.metadata_preparation",
            phase,
            elapsed_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
            "H3 metadata preparation host interval"
        );
    }
    result
}

impl<R: DeviceRuntime> PH3ImagePipeline<R> {
    /// Opt in before requests start. The ordinary trace and HTTP response never
    /// acquire effective prompt content unless CaptureEffectivePrompt is set.
    pub fn set_prompt_expansion_diagnostics(
        &mut self,
        policy: super::PH3ImagePromptExpansionDiagnosticPolicy,
    ) -> Result<()> {
        use super::PH3ImagePromptExpansionDiagnosticPolicy as Policy;
        policy.validate()?;
        if policy == Policy::Disabled {
            self.prompt_expansion_diagnostics = policy;
            return Ok(());
        }
        ensure!(
            self.execution.next_job == 0
                && !self.execution.is_stopped()
                && self.execution.sources.assets.has_prompt_expansion(),
            "prompt expansion diagnostics require Enabled expansion before requests start"
        );
        let trace = self.execution.continuous_trace.as_ref().ok_or_else(|| {
            anyhow::anyhow!("prompt expansion diagnostics require a bounded continuous trace")
        })?;
        if let Policy::CaptureEffectivePrompt { maximum_utf8_bytes } = policy {
            ensure!(
                maximum_utf8_bytes <= trace.handle().status().maximum_bytes,
                "prompt capture ceiling exceeds bounded trace capacity"
            );
        }
        self.prompt_expansion_diagnostics = policy;
        Ok(())
    }

    /// Install a bounded diagnostic before admitting any request. `context`
    /// carries the composition root's actual backend and service configuration;
    /// it does not replace the runtime, source and capacity facts below.
    /// Export the handle after original shutdown. Its Output charge remains
    /// live until the pipeline and all handles are dropped.
    pub fn enable_continuous_trace(
        &mut self,
        limits: ImageContinuousTraceLimits,
        context: &Value,
    ) -> Result<ImageContinuousTraceHandle> {
        ensure!(
            self.execution.next_job == 0
                && !self.execution.preparation_only
                && !self.execution.is_stopped()
                && self.execution.continuous_trace.is_none(),
            "continuous trace must be installed once before the first image request"
        );
        limits.validate()?;
        let config = &self.execution.config;
        let host = self.execution.host.limits();
        let expansion = self
            .execution
            .sources
            .assets
            .expansion
            .as_ref()
            .map(|expansion| {
                Ok::<_, anyhow::Error>(json!({
                    "identity": expansion.assets.trace_identity()?,
                    "engine": expansion.options.engine,
                    "max_new_tokens": expansion.options.max_new_tokens,
                    "prefill_chunk_tokens": expansion.options.prefill_chunk_tokens,
                }))
            })
            .transpose()?;
        let header = json!({
            "kind":"header",
            "device":self.execution.runtime.descriptor(),
            "device_opaque_allowance_bytes":self.execution.device_opaque_allowance_bytes,
            "assets":self.execution.sources.assets.receipt(),
            "prompt_expansion":expansion,
            "noise_profile":self.options.noise_profile,
            "initial_noise_policy":self.options.initial_noise.identity(),
            "edit_noise_policy":self.options.edit_noise.identity(),
            "vae_decoder_tile_batch":self.options.vae_decoder_tile_batch,
            "pipeline":{
                "main_vision_reuse":self.execution.sources.assets.main_vision_reuse,
                "device_capacity_bytes":config.device_capacity_bytes,
                "maximum_user_tokens":config.maximum_user_tokens,
                "maximum_width":config.maximum_width,
                "maximum_height":config.maximum_height,
                "maximum_batch_size":config.maximum_batch_size,
                "decoder_batch_policy":config.decoder_batch_policy,
                "edit_prefix_policy":config.edit_prefix_policy,
                "plan_cache":config.plan_cache,
                "residency":config.residency,
                "dynamic_residency":config.dynamic_residency,
                "static_initialization":{
                    "maximum_staging_bytes":config.static_initialization.maximum_staging_bytes(),
                    "maximum_commands_per_batch":config.static_initialization.maximum_commands_per_batch(),
                },
                "executor":{
                    "maximum_backing_maintenance_passes":config.executor.maximum_backing_maintenance_passes,
                    "device_timing_mode":config.executor.device_timing_mode,
                },
            },
            "host":{
                "total_bytes":host.total_bytes(),
                "input_preprocess_bytes":host.class_bytes(HostMemoryClass::InputPreprocess),
                "output_bytes":host.class_bytes(HostMemoryClass::Output),
                "transfer_staging_bytes":host.class_bytes(HostMemoryClass::TransferStaging),
                "weight_backing_bytes":host.class_bytes(HostMemoryClass::WeightBacking),
                "allocation_alignment":self.execution.host.allocation_alignment(),
            },
            "context":context,
        });
        let (recorder, handle) = ImageContinuousTraceRecorder::new(&self.execution.host, limits)?;
        recorder.record(&header);
        self.execution.continuous_trace = Some(recorder);
        Ok(handle)
    }
}
