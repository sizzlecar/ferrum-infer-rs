//! Capability-dependent execution defaults shared by run and serve.

use super::*;
use crate::{
    CausalDecodeAttentionGrouping, CausalDecodeKvReuse, CausalDecodePreparationMode,
    CompletionReadbackPolicy, InvocationPreparationStrategy, PrefillDecodeExecution,
    ProgramBindingUploadStrategy, Q4MmvqWeightDecode, Q5MmvqWeightDecode,
    Q8ProjectionWeightPreparation, SamplingReadbackMode, SegmentBindingOwnerViewMode,
    SequenceBackingGrowth,
};

impl FerrumConfigBuilder {
    pub(super) fn resolve_execution_defaults(
        &self,
        reusable_execution: bool,
        attention: Option<AttentionExecutionPolicy>,
    ) -> Result<Self, AutoConfigError> {
        // Only absent entries are automatic. An explicit conservative choice
        // remains an opt-out, including entries from config files and presets.
        macro_rules! select {
            ($key:expr, $ty:ty, $default:expr) => {
                self.entry($key)
                    .map(|entry| <$ty>::parse_runtime_value(&entry.effective_value))
                    .transpose()
                    .map_err(|reason| AutoConfigError::InvalidOverride {
                        key: $key.to_owned(),
                        reason,
                    })?
                    .unwrap_or($default)
            };
        }
        let cuda_plan = self.is_cuda_backend()
            && self.hardware.compiled_features.cuda
            && self.execution_resource_authority == ExecutionResourceAuthority::PlanRuntime;
        let upload = select!(
            "FERRUM_PROGRAM_BINDING_UPLOAD_STRATEGY",
            ProgramBindingUploadStrategy,
            if cuda_plan {
                ProgramBindingUploadStrategy::CompactScatter
            } else {
                ProgramBindingUploadStrategy::Sparse
            }
        );
        let requested_owner = self
            .entry(crate::SEGMENT_BINDING_OWNER_VIEW_MODE_CONFIG_KEY)
            .map(SegmentBindingOwnerViewMode::parse_config_entry)
            .transpose()
            .map_err(|reason| AutoConfigError::InvalidOverride {
                key: crate::SEGMENT_BINDING_OWNER_VIEW_MODE_CONFIG_KEY.to_owned(),
                reason,
            })?;
        let preparation = select!(
            "FERRUM_INVOCATION_PREPARATION_STRATEGY",
            InvocationPreparationStrategy,
            if cuda_plan
                && reusable_execution
                && self.hardware.graph_support
                && self.hardware.compiled_features.cuda_graph
                && upload == ProgramBindingUploadStrategy::CompactScatter
                && requested_owner != Some(SegmentBindingOwnerViewMode::Legacy)
            {
                InvocationPreparationStrategy::ResidentDecodeProgram
            } else {
                InvocationPreparationStrategy::Full
            }
        );
        let owner = requested_owner.unwrap_or(if cuda_plan && preparation.uses_decode_segment() {
            SegmentBindingOwnerViewMode::Indexed
        } else {
            SegmentBindingOwnerViewMode::Legacy
        });
        let native_attention = cuda_plan
            && self.hardware.compiled_features.vllm_paged_attn
            && attention == Some(AttentionExecutionPolicy::NativeAdaptive);
        let causal_preparation = select!(
            crate::CAUSAL_DECODE_PREPARATION_MODE_CONFIG_KEY,
            CausalDecodePreparationMode,
            if native_attention {
                CausalDecodePreparationMode::Packed
            } else {
                CausalDecodePreparationMode::PerParticipant
            }
        );
        let grouping = select!(
            crate::CAUSAL_DECODE_ATTENTION_GROUPING_CONFIG_KEY,
            CausalDecodeAttentionGrouping,
            if native_attention
                && self
                    .hardware
                    .compiled_features
                    .cuda_indexed_causal_attention
                && causal_preparation == CausalDecodePreparationMode::Packed
            {
                CausalDecodeAttentionGrouping::FixedExecutionCapacity
            } else {
                CausalDecodeAttentionGrouping::Contiguous
            }
        );
        let reuse = select!(
            crate::CAUSAL_DECODE_KV_REUSE_CONFIG_KEY,
            CausalDecodeKvReuse,
            if native_attention
                && self
                    .hardware
                    .compiled_features
                    .cuda_indexed_causal_attention
                && self.hardware.compiled_features.cuda_causal_gqa2
                && causal_preparation == CausalDecodePreparationMode::Packed
                && grouping == CausalDecodeAttentionGrouping::FixedExecutionCapacity
            {
                CausalDecodeKvReuse::PairedQueryHeadsUnifiedV2
            } else {
                CausalDecodeKvReuse::None
            }
        );
        let growth = select!(
            crate::SEQUENCE_BACKING_GROWTH_CONFIG_KEY,
            SequenceBackingGrowth,
            if cuda_plan && preparation == InvocationPreparationStrategy::ResidentDecodeProgram {
                SequenceBackingGrowth::PagedGroup4GrowOnly
            } else {
                SequenceBackingGrowth::Exact
            }
        );
        let readback = select!(
            "FERRUM_SAMPLING_READBACK_MODE",
            SamplingReadbackMode,
            if cuda_plan {
                SamplingReadbackMode::Participant
            } else {
                SamplingReadbackMode::Uniform
            }
        );
        let completion = select!(
            crate::COMPLETION_READBACK_POLICY_CONFIG_KEY,
            CompletionReadbackPolicy,
            if cuda_plan {
                CompletionReadbackPolicy::BatchedU32
            } else {
                CompletionReadbackPolicy::Synchronous
            }
        );
        // These are execution requests, not a claim that every model shape can
        // use a native kernel. Providers retain their geometry, dtype, current
        // lease and capacity checks, and their existing legal fallbacks.
        let mut resolved = self.clone();
        for (key, value) in [
            (
                "FERRUM_INVOCATION_PREPARATION_STRATEGY",
                preparation.as_runtime_value(),
            ),
            (
                "FERRUM_PROGRAM_BINDING_UPLOAD_STRATEGY",
                upload.as_runtime_value(),
            ),
            (
                crate::SEGMENT_BINDING_OWNER_VIEW_MODE_CONFIG_KEY,
                owner.as_runtime_value(),
            ),
            (
                crate::CAUSAL_DECODE_PREPARATION_MODE_CONFIG_KEY,
                causal_preparation.as_runtime_value(),
            ),
            (
                crate::CAUSAL_DECODE_ATTENTION_GROUPING_CONFIG_KEY,
                grouping.as_runtime_value(),
            ),
            (
                crate::CAUSAL_DECODE_KV_REUSE_CONFIG_KEY,
                reuse.as_runtime_value(),
            ),
            (
                crate::SEQUENCE_BACKING_GROWTH_CONFIG_KEY,
                growth.as_runtime_value(),
            ),
            ("FERRUM_SAMPLING_READBACK_MODE", readback.as_runtime_value()),
            (
                crate::COMPLETION_READBACK_POLICY_CONFIG_KEY,
                completion.as_runtime_value(),
            ),
            (
                "FERRUM_PREFILL_DECODE_EXECUTION",
                PrefillDecodeExecution::default().as_runtime_value(),
            ),
            // The builder does not know the selected numerical profile or
            // weight layout. Do not infer these numerical/storage choices
            // from model names or a generic quantization label.
            (
                crate::Q4_MMVQ_WEIGHT_DECODE_CONFIG_KEY,
                Q4MmvqWeightDecode::default().as_runtime_value(),
            ),
            (
                crate::Q5_MMVQ_WEIGHT_DECODE_CONFIG_KEY,
                Q5MmvqWeightDecode::default().as_runtime_value(),
            ),
            (
                crate::Q8_PROJECTION_WEIGHT_PREPARATION_CONFIG_KEY,
                Q8ProjectionWeightPreparation::default().as_runtime_value(),
            ),
        ] {
            if self.entry(key).is_none() {
                resolved
                    .runtime_config
                    .upsert(key, value, RuntimeConfigSource::Default);
            }
        }
        Ok(resolved)
    }
}

#[cfg(test)]
mod tests;
