//! Installation-local conversion backing. This never caches prompts, KV,
//! execution weights across installations, or device state.
use super::*;
use ferrum_runtime::ExecutionPlan;
use serde::{Deserialize, Serialize};

/// Select conversion backing for one static prompt-upsampler installation.
/// The reusable slot keeps its full original Host charge until initialization
/// releases the local source; it is not a cross-request weight cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PH3ImagePromptExpansionSourceBacking {
    #[default]
    Fresh,
    ReusableSingleSlot,
    /// Convert and upload exact BF16 source ranges through one installation
    /// slot with at most 64 MiB of physical Host backing, including padding.
    BoundedStreaming,
}

const STREAMING_PHYSICAL_BYTES: usize = 64 << 20;

impl NativeTextExpansion {
    pub(crate) fn configure_source_backing(
        &mut self,
        policy: PH3ImagePromptExpansionSourceBacking,
    ) -> Result<()> {
        if self.source_conversion_policy.maximum_workers().get() > 1
            && policy != PH3ImagePromptExpansionSourceBacking::BoundedStreaming
        {
            return Err(FerrumError::config(
                "parallel source conversion requires bounded_streaming backing",
            ));
        }
        if policy != PH3ImagePromptExpansionSourceBacking::Fresh
            && self.model.process_mapped_weights_enabled()
        {
            return Err(FerrumError::config(
                "installation-local source backing cannot be combined with process-mapped weights",
            ));
        }
        self.source_backing = policy;
        Ok(())
    }

    pub(crate) fn configure_source_conversion_policy(
        &mut self,
        policy: ferrum_quantization::Bf16F16ConversionPolicy,
    ) -> Result<()> {
        if policy.maximum_workers().get() > 1
            && (self.source_backing != PH3ImagePromptExpansionSourceBacking::BoundedStreaming
                || self.model.process_mapped_weights_enabled())
        {
            return Err(FerrumError::config(
                "parallel source conversion requires bounded_streaming backing without process-mapped weights",
            ));
        }
        self.source_conversion_policy = policy;
        Ok(())
    }

    /// Metadata-only: validate this actual selected family/plan. Slot allocation
    /// belongs to core's hook after admission, not this preparation query.
    pub(super) fn validate_static_source_policy(
        &self,
        source: &ManagedProductionWeightSource<'_>,
        plan: &ExecutionPlan,
    ) -> Result<()> {
        match self.source_backing {
            PH3ImagePromptExpansionSourceBacking::Fresh => Ok(()),
            PH3ImagePromptExpansionSourceBacking::ReusableSingleSlot => source
                .prepare_reusable_static_backing(plan)
                .map_err(model_error),
            PH3ImagePromptExpansionSourceBacking::BoundedStreaming => source
                .prepare_bounded_streaming_static_backing_with_policy(
                    plan,
                    STREAMING_PHYSICAL_BYTES,
                    self.source_conversion_policy,
                )
                .map_err(model_error),
        }
    }

    pub(super) fn prepare_static_source(
        &self,
        source: &ManagedProductionWeightSource<'_>,
        plan: &ExecutionPlan,
    ) -> Result<()> {
        match self.source_backing {
            PH3ImagePromptExpansionSourceBacking::Fresh => {
                source.prepare_process_mapped_weights().map_err(model_error)
            }
            PH3ImagePromptExpansionSourceBacking::ReusableSingleSlot
            | PH3ImagePromptExpansionSourceBacking::BoundedStreaming => {
                self.validate_static_source_policy(source, plan)
            }
        }
    }
}
