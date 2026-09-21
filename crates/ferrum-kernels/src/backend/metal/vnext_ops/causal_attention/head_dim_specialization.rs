//! Optional fixed-dimension pipelines over the shared F16 attention kernels.
use super::*;
use metal::{FunctionConstantValues, LibraryRef, MTLDataType};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AttentionHeadDim {
    Head128,
    Head256,
}

impl AttentionHeadDim {
    pub(super) const ALL: [Self; 2] = [Self::Head128, Self::Head256];
    fn from_dimension(dimension: u32) -> Option<Self> {
        match dimension {
            128 => Some(Self::Head128),
            256 => Some(Self::Head256),
            _ => None,
        }
    }
    fn dimension(self) -> u32 {
        match self {
            Self::Head128 => 128,
            Self::Head256 => 256,
        }
    }
    fn index(self) -> usize {
        match self {
            Self::Head128 => 0,
            Self::Head256 => 1,
        }
    }
}

#[derive(Default)]
pub(super) struct SpecializedAttentionPipelines {
    pub(super) grouped: Option<(ComputePipelineState, ComputePipelineState)>,
    pub(super) batched_grouped: Option<(ComputePipelineState, ComputePipelineState)>,
    pub(super) tiled: Option<ComputePipelineState>,
    pub(super) gqa: Option<ComputePipelineState>,
    #[cfg(test)]
    pub(super) initialization_ns: u64,
}

fn supports_pipeline(
    execution_width: u64,
    maximum_threads: u64,
    static_bytes: u64,
    maximum_bytes: u64,
    threads: u64,
    dynamic_bytes: u64,
) -> bool {
    execution_width == SIMD_THREADS
        && maximum_threads >= threads
        && static_bytes
            .checked_add(dynamic_bytes)
            .is_some_and(|bytes| bytes <= maximum_bytes)
}

impl SpecializedAttentionPipelines {
    pub(super) fn new(
        device: &Device,
        library: &LibraryRef,
        dim: AttentionHeadDim,
        binding_length: u64,
        binding_alignment: u64,
    ) -> Self {
        #[cfg(test)]
        let started = std::time::Instant::now();
        let dimension = dim.dimension();
        let constants = FunctionConstantValues::new();
        constants.set_constant_value_at_index(
            (&dimension as *const u32).cast(),
            MTLDataType::UInt,
            0,
        );
        let pipeline = |name: &str,
                        threads: u64,
                        dynamic_bytes: u64,
                        page_table: bool|
         -> Option<ComputePipelineState> {
            let function = library.get_function(name, Some(constants.clone())).ok()?;
            if page_table {
                let encoder = function.new_argument_encoder(ATTENTION_PAGE_TABLE_INDEX);
                if encoder.encoded_length() != binding_length
                    || encoder.alignment() != binding_alignment
                {
                    return None;
                }
            }
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .ok()?;
            supports_pipeline(
                pipeline.thread_execution_width(),
                pipeline.max_total_threads_per_threadgroup(),
                pipeline.static_threadgroup_memory_length(),
                device.max_threadgroup_memory_length(),
                threads,
                dynamic_bytes,
            )
            .then_some(pipeline)
        };
        // Exactly the same dynamic memory formulas as the dispatch plans.
        let tile_bytes = |heads: u64, key_tile: u64| {
            heads
                * u64::from(TILED_PREFILL_QUERY_TILE)
                * (u64::from(dimension) + key_tile)
                * (std::mem::size_of::<half::f16>() + std::mem::size_of::<f32>()) as u64
        };
        let partial = pipeline(
            GROUPED_DECODE_PARTIAL_ATTENTION_KERNEL,
            SIMD_THREADS * TILED_PREFILL_SIMDGROUPS,
            tile_bytes(1, TILED_PREFILL_KEY_TILE),
            true,
        );
        let reduce = pipeline(
            GROUPED_DECODE_REDUCE_ATTENTION_KERNEL,
            SIMD_THREADS,
            grouped_decode_reduce_threadgroup_memory_bytes(),
            false,
        );
        let batched_partial = pipeline(
            "vnext_causal_attention_decode_batched_partial_f16",
            SIMD_THREADS * TILED_PREFILL_SIMDGROUPS,
            tile_bytes(1, TILED_PREFILL_KEY_TILE),
            true,
        );
        let batched_reduce = pipeline(
            "vnext_causal_attention_decode_batched_reduce_f16",
            SIMD_THREADS,
            grouped_decode_reduce_threadgroup_memory_bytes(),
            false,
        );
        let tiled = pipeline(
            TILED_PREFILL_ATTENTION_KERNEL,
            SIMD_THREADS * TILED_PREFILL_SIMDGROUPS,
            tile_bytes(1, TILED_PREFILL_KEY_TILE),
            true,
        );
        let gqa = (dim == AttentionHeadDim::Head256)
            .then(|| {
                pipeline(
                    GQA_TILED_PREFILL_ATTENTION_KERNEL,
                    SIMD_THREADS * GQA_TILED_PREFILL_SIMDGROUPS,
                    tile_bytes(
                        u64::from(GQA_TILED_PREFILL_QUERY_HEADS),
                        GQA_TILED_PREFILL_KEY_TILE,
                    ),
                    true,
                )
            })
            .flatten();
        Self {
            // Keep the two stages paired: unsupported partial or reduction
            // falls back to the original complete dynamic route.
            grouped: partial.zip(reduce),
            batched_grouped: batched_partial.zip(batched_reduce),
            tiled,
            gqa,
            #[cfg(test)]
            initialization_ns: started.elapsed().as_nanos() as u64,
        }
    }
}

impl MetalCausalAttentionPipelines {
    pub(super) fn specialization(
        &self,
        params: &CausalAttentionParams,
    ) -> Option<&SpecializedAttentionPipelines> {
        if self.kv_type != ElementType::F16 {
            return None;
        }
        AttentionHeadDim::from_dimension(params.head_dim).map(|dim| &self.specialized[dim.index()])
    }

    pub(super) fn attention_pipeline(
        &self,
        params: &CausalAttentionParams,
        kind: AttentionDispatchKind,
    ) -> &ComputePipelineState {
        let specialized = self.specialization(params);
        match kind {
            AttentionDispatchKind::General => &self.attention,
            AttentionDispatchKind::DirectDecode => &self.direct_decode_attention,
            AttentionDispatchKind::GroupedDecode => specialized
                .and_then(|p| p.grouped.as_ref())
                .map(|p| &p.0)
                .unwrap_or(&self.grouped_decode_partial_attention),
            AttentionDispatchKind::TiledPrefill => specialized
                .and_then(|p| p.tiled.as_ref())
                .unwrap_or(&self.tiled_prefill_attention),
            AttentionDispatchKind::GqaTiledPrefill => specialized
                .and_then(|p| p.gqa.as_ref())
                .unwrap_or(&self.gqa_tiled_prefill_attention),
        }
    }

    pub(super) fn grouped_reduce_pipeline(
        &self,
        params: &CausalAttentionParams,
    ) -> &ComputePipelineState {
        self.specialization(params)
            .and_then(|p| p.grouped.as_ref())
            .map(|p| &p.1)
            .unwrap_or(&self.grouped_decode_reduce_attention)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn head_dim_specialization_requires_matching_threads_and_memory_capacity() {
        assert!(supports_pipeline(32, 128, 0, 32768, 128, 13824));
        assert!(supports_pipeline(32, 256, 2048, 32768, 256, 30720));
        assert!(!supports_pipeline(16, 256, 0, 32768, 256, 30720));
        assert!(!supports_pipeline(32, 255, 0, 32768, 256, 30720));
        assert!(!supports_pipeline(32, 256, 2049, 32768, 256, 30720));
        assert!(!supports_pipeline(32, 256, u64::MAX, u64::MAX, 256, 1));
        assert_eq!(AttentionHeadDim::from_dimension(64), None);
    }
}
