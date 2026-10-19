//! Device-bound lazy quantized pipelines. No queue or mutable execution policy.

use super::MetalPipelines;
use metal::ComputePipelineState;
use std::sync::OnceLock;

#[derive(Default)]
pub(super) struct QuantizedPipelines {
    moe_post_ops_silu_pipeline: OnceLock<ComputePipelineState>,
    moe_post_ops_wsum_pipeline: OnceLock<ComputePipelineState>,
    moe_post_ops_wsum_res_pipeline: OnceLock<ComputePipelineState>,
    moe_post_ops_wsum_res_norm_pipeline: OnceLock<ComputePipelineState>,
    moe_post_ops_batched_silu_pipeline: OnceLock<ComputePipelineState>,
    moe_post_ops_batched_wsum_pipeline: OnceLock<ComputePipelineState>,
    moe_router_pipeline: OnceLock<ComputePipelineState>,
    moe_router_ids_tpe_pipeline: OnceLock<ComputePipelineState>,
    #[cfg(test)]
    q4_k_pipeline: OnceLock<ComputePipelineState>,
    q4_k_gemm_pipeline: OnceLock<ComputePipelineState>,
    q4_k_gemv_pipeline: OnceLock<ComputePipelineState>,
    q4_k_gemv_v2_pipeline: OnceLock<ComputePipelineState>,
    q4_k_moe_id_gate_up_silu_pipeline: OnceLock<ComputePipelineState>,
    q4_k_moe_id_gate_up_silu_batched_pipeline: OnceLock<ComputePipelineState>,
    q4_k_moe_id_gemm_pipeline: OnceLock<ComputePipelineState>,
    q4_k_moe_id_gemv_pipeline: OnceLock<ComputePipelineState>,
    q4_k_moe_id_gemv_batched_pipeline: OnceLock<ComputePipelineState>,
    q6_k_gemm_pipeline: OnceLock<ComputePipelineState>,
    q6_k_gemv_pipeline: OnceLock<ComputePipelineState>,
    q6_k_moe_id_gemm_pipeline: OnceLock<ComputePipelineState>,
    q6_k_moe_id_gemv_pipeline: OnceLock<ComputePipelineState>,
    q6_k_moe_id_gemv_batched_pipeline: OnceLock<ComputePipelineState>,
}

impl MetalPipelines {
    pub(crate) fn moe_post_ops_silu_pipeline(&self) -> &ComputePipelineState {
        self.quantized
            .moe_post_ops_silu_pipeline
            .get_or_init(|| crate::backend::metal::moe_post_ops::build_silu_pipeline(self.device()))
    }

    pub(crate) fn moe_post_ops_wsum_pipeline(&self) -> &ComputePipelineState {
        self.quantized
            .moe_post_ops_wsum_pipeline
            .get_or_init(|| crate::backend::metal::moe_post_ops::build_wsum_pipeline(self.device()))
    }

    pub(crate) fn moe_post_ops_wsum_res_pipeline(&self) -> &ComputePipelineState {
        self.quantized
            .moe_post_ops_wsum_res_pipeline
            .get_or_init(|| {
                crate::backend::metal::moe_post_ops::build_wsum_res_pipeline(self.device())
            })
    }

    pub(crate) fn moe_post_ops_wsum_res_norm_pipeline(&self) -> &ComputePipelineState {
        self.quantized
            .moe_post_ops_wsum_res_norm_pipeline
            .get_or_init(|| {
                crate::backend::metal::moe_post_ops::build_wsum_res_norm_pipeline(self.device())
            })
    }

    pub(crate) fn moe_post_ops_batched_silu_pipeline(&self) -> &ComputePipelineState {
        self.quantized
            .moe_post_ops_batched_silu_pipeline
            .get_or_init(|| {
                crate::backend::metal::moe_post_ops_batched::build_silu_pipeline(self.device())
            })
    }

    pub(crate) fn moe_post_ops_batched_wsum_pipeline(&self) -> &ComputePipelineState {
        self.quantized
            .moe_post_ops_batched_wsum_pipeline
            .get_or_init(|| {
                crate::backend::metal::moe_post_ops_batched::build_wsum_pipeline(self.device())
            })
    }

    pub(crate) fn moe_router_pipeline(&self) -> &ComputePipelineState {
        self.quantized
            .moe_router_pipeline
            .get_or_init(|| crate::backend::metal::moe_router::build_pipeline(self.device()))
    }

    pub(crate) fn moe_router_ids_tpe_pipeline(&self) -> &ComputePipelineState {
        self.quantized.moe_router_ids_tpe_pipeline.get_or_init(|| {
            crate::backend::metal::moe_router::build_ids_tpe_pipeline(self.device())
        })
    }

    #[cfg(test)]
    pub(crate) fn q4_k_pipeline(&self) -> &ComputePipelineState {
        self.quantized
            .q4_k_pipeline
            .get_or_init(|| crate::backend::metal::q4_k::build_pipeline(self.device()))
    }

    pub(crate) fn q4_k_gemm_pipeline(&self) -> &ComputePipelineState {
        self.quantized
            .q4_k_gemm_pipeline
            .get_or_init(|| crate::backend::metal::q4_k_gemm::build_pipeline(self.device()))
    }

    pub(crate) fn q4_k_gemv_pipeline(&self) -> &ComputePipelineState {
        self.quantized
            .q4_k_gemv_pipeline
            .get_or_init(|| crate::backend::metal::q4_k_gemv::build_pipeline(self.device()))
    }

    pub(crate) fn q4_k_gemv_v2_pipeline(&self) -> &ComputePipelineState {
        self.quantized
            .q4_k_gemv_v2_pipeline
            .get_or_init(|| crate::backend::metal::q4_k_gemv_v2::build_pipeline(self.device()))
    }

    pub(crate) fn q4_k_moe_id_gate_up_silu_pipeline(&self) -> &ComputePipelineState {
        self.quantized
            .q4_k_moe_id_gate_up_silu_pipeline
            .get_or_init(|| {
                crate::backend::metal::q4_k_moe_id_gate_up_silu::build_pipeline(self.device())
            })
    }

    pub(crate) fn q4_k_moe_id_gate_up_silu_batched_pipeline(&self) -> &ComputePipelineState {
        self.quantized
            .q4_k_moe_id_gate_up_silu_batched_pipeline
            .get_or_init(|| {
                crate::backend::metal::q4_k_moe_id_gate_up_silu_batched::build_pipeline(
                    self.device(),
                )
            })
    }

    pub(crate) fn q4_k_moe_id_gemm_pipeline(&self) -> &ComputePipelineState {
        self.quantized
            .q4_k_moe_id_gemm_pipeline
            .get_or_init(|| crate::backend::metal::q4_k_moe_id_gemm::build_pipeline(self.device()))
    }

    pub(crate) fn q4_k_moe_id_gemv_pipeline(&self) -> &ComputePipelineState {
        self.quantized
            .q4_k_moe_id_gemv_pipeline
            .get_or_init(|| crate::backend::metal::q4_k_moe_id_gemv::build_pipeline(self.device()))
    }

    pub(crate) fn q4_k_moe_id_gemv_batched_pipeline(&self) -> &ComputePipelineState {
        self.quantized
            .q4_k_moe_id_gemv_batched_pipeline
            .get_or_init(|| {
                crate::backend::metal::q4_k_moe_id_gemv_batched::build_pipeline(self.device())
            })
    }

    pub(crate) fn q6_k_gemm_pipeline(&self) -> &ComputePipelineState {
        self.quantized
            .q6_k_gemm_pipeline
            .get_or_init(|| crate::backend::metal::q6_k_gemm::build_pipeline(self.device()))
    }

    pub(crate) fn q6_k_gemv_pipeline(&self) -> &ComputePipelineState {
        self.quantized
            .q6_k_gemv_pipeline
            .get_or_init(|| crate::backend::metal::q6_k_gemv::build_pipeline(self.device()))
    }

    pub(crate) fn q6_k_moe_id_gemm_pipeline(&self) -> &ComputePipelineState {
        self.quantized
            .q6_k_moe_id_gemm_pipeline
            .get_or_init(|| crate::backend::metal::q6_k_moe_id_gemm::build_pipeline(self.device()))
    }

    pub(crate) fn q6_k_moe_id_gemv_pipeline(&self) -> &ComputePipelineState {
        self.quantized
            .q6_k_moe_id_gemv_pipeline
            .get_or_init(|| crate::backend::metal::q6_k_moe_id_gemv::build_pipeline(self.device()))
    }

    pub(crate) fn q6_k_moe_id_gemv_batched_pipeline(&self) -> &ComputePipelineState {
        self.quantized
            .q6_k_moe_id_gemv_batched_pipeline
            .get_or_init(|| {
                crate::backend::metal::q6_k_moe_id_gemv_batched::build_pipeline(self.device())
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn quantized_catalog_lazy_slots_belong_to_each_catalog_and_share_by_borrow() {
        let device = metal::Device::system_default().expect("Metal device for catalog cache test");
        let first = Arc::new(MetalPipelines::new(&device));
        let shared = Arc::clone(&first);
        let independent = MetalPipelines::new(&device);
        assert!(first.quantized.q4_k_gemv_pipeline.get().is_none());
        assert!(independent.quantized.q4_k_gemv_pipeline.get().is_none());

        let original = first.q4_k_gemv_pipeline();
        assert!(std::ptr::eq(original, shared.q4_k_gemv_pipeline()));
        assert!(std::ptr::eq(
            original,
            first.quantized.q4_k_gemv_pipeline.get().unwrap(),
        ));
        // Native Metal may deduplicate immutable objects. The authority is
        // this catalog's lazy slot, not distinct native pointer values.
        assert!(independent.quantized.q4_k_gemv_pipeline.get().is_none());
        let own = independent.q4_k_gemv_pipeline();
        assert!(std::ptr::eq(
            own,
            independent.quantized.q4_k_gemv_pipeline.get().unwrap(),
        ));
        assert!(first.quantized.q6_k_gemv_pipeline.get().is_none());
        independent.q6_k_gemv_pipeline();
        assert!(independent.quantized.q6_k_gemv_pipeline.get().is_some());
        assert!(first.quantized.q6_k_gemv_pipeline.get().is_none());
        drop(first);
        assert!(shared.quantized.q4_k_gemv_pipeline.get().is_some());
        assert!(std::ptr::eq(
            shared.q4_k_gemv_pipeline(),
            shared.quantized.q4_k_gemv_pipeline.get().unwrap(),
        ));
    }
}
