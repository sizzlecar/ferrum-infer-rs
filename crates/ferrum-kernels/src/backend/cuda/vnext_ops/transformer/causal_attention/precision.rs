//! Hidden-state ABI selection is independent of attention semantics and weights.
use super::CausalAttentionSemantics;
use ferrum_runtime::ElementType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CausalPrecision {
    F16,
    Bf16,
    F32Master,
}

impl CausalPrecision {
    pub(super) fn hidden(self) -> ElementType {
        match self {
            Self::F16 => ElementType::F16,
            Self::Bf16 => ElementType::Bf16,
            Self::F32Master => ElementType::F32,
        }
    }

    pub(super) fn storage(self) -> ElementType {
        match self {
            Self::Bf16 => ElementType::Bf16,
            _ => ElementType::F16,
        }
    }

    pub(super) fn provider_id(self, semantics: CausalAttentionSemantics) -> &'static str {
        match self {
            Self::F16 => semantics.provider_id(),
            Self::Bf16 if semantics == CausalAttentionSemantics::ExplicitRope => {
                "provider.cuda.causal_paged_attention.explicit-rope.bf16"
            }
            Self::Bf16 => "provider.cuda.causal_paged_attention.bf16",
            Self::F32Master if semantics.int8_kv() => {
                "provider.cuda.causal_paged_attention.f32-master.int8-kv"
            }
            Self::F32Master => "provider.cuda.causal_paged_attention.f32-master",
        }
    }

    pub(super) fn estimator_id(self, semantics: CausalAttentionSemantics) -> &'static str {
        match self {
            Self::F16 => semantics.estimator_id(),
            Self::Bf16 if semantics == CausalAttentionSemantics::ExplicitRope => {
                "resource-estimator.cuda.causal_paged_attention.explicit-rope.bf16"
            }
            Self::Bf16 => "resource-estimator.cuda.causal_paged_attention.bf16",
            Self::F32Master if semantics.int8_kv() => {
                "resource-estimator.cuda.causal_paged_attention.f32-master.int8-kv"
            }
            Self::F32Master => "resource-estimator.cuda.causal_paged_attention.f32-master",
        }
    }

    pub(super) fn rms_kernel(self) -> &'static str {
        match self {
            Self::F16 => "rms_norm_f16",
            Self::Bf16 => "rms_norm_bf16",
            Self::F32Master => "vnext_rms_norm_f32_to_f16",
        }
    }

    pub(super) fn residual_kernel(self) -> &'static str {
        match self {
            Self::F16 => "residual_add_f16",
            Self::Bf16 => "residual_add_bf16",
            Self::F32Master => "vnext_residual_add_f32_f16",
        }
    }

    pub(super) fn inplace_residual_kernel(self) -> Option<&'static str> {
        match self {
            Self::F16 => Some("residual_add_inplace_f16"),
            Self::Bf16 => Some("residual_add_inplace_bf16"),
            // The F32/F16 kernel admits input/output aliasing with its ordinary ABI.
            Self::F32Master => None,
        }
    }
}
