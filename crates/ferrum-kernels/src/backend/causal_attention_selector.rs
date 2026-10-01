//! Pure CUDA causal-attention selection shared by dispatch and cold metadata.
//! No device runtime, library handle, environment, or allocation is consulted.

use ferrum_interfaces::vnext::{
    DecodeContextBoundary, DecodeContextBoundaryKind, DecodeContextCoverage,
};
use ferrum_types::{AttentionExecutionPolicy, CUDA_NATIVE_ADAPTIVE_V1_MAX_SEQUENCE_TOKENS};
use std::num::NonZeroU64;

const MAXIMUM_VARLEN_HEAD_DIM: u64 = 256;
pub(super) const VARLEN_DYNAMIC_SHARED_BUDGET_BYTES: u64 = 48 * 1024 - 1024;
pub(super) const VARLEN_TILED_QUERY_TOKENS: u64 = 4;
// Bound cold metadata even for an unusually large compiled context. Truncation
// keeps the proven obligations and explicitly withdraws completeness.
const MAXIMUM_DECLARED_BOUNDARIES: usize = 4096;

#[derive(Debug, Clone, Copy)]
pub(crate) struct CausalAttentionSelectorShape {
    pub int8_kv: bool,
    pub uses_vllm_blocks: bool,
    pub head_dim: u64,
    pub sliding_window_tokens: u64,
    pub attention_scale: f32,
}

impl CausalAttentionSelectorShape {
    pub(crate) fn tiled_vllm_supported(self, native_compiled: bool) -> bool {
        native_compiled
            && self.uses_vllm_blocks
            && matches!(self.head_dim, 128 | 256)
            && self.sliding_window_tokens == 0
            && self.attention_scale.to_bits() == (1.0_f32 / (self.head_dim as f32).sqrt()).to_bits()
    }

    fn addressed_varlen_supported(self) -> bool {
        self.head_dim <= MAXIMUM_VARLEN_HEAD_DIM && self.sliding_window_tokens == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeDecodeKernel {
    V1,
    V2,
}

pub(crate) fn native_decode_kernel(sequence_tokens: u64) -> NativeDecodeKernel {
    if sequence_tokens <= CUDA_NATIVE_ADAPTIVE_V1_MAX_SEQUENCE_TOKENS {
        NativeDecodeKernel::V1
    } else {
        NativeDecodeKernel::V2
    }
}

pub(crate) fn native_decode_replay_capacity(
    kernel: NativeDecodeKernel,
    sequence_tokens: u64,
    maximum_context_tokens: u64,
) -> Result<u64, String> {
    match kernel {
        NativeDecodeKernel::V1 => {
            Ok(CUDA_NATIVE_ADAPTIVE_V1_MAX_SEQUENCE_TOKENS.min(maximum_context_tokens))
        }
        NativeDecodeKernel::V2 => sequence_tokens
            .div_ceil(CUDA_NATIVE_ADAPTIVE_V1_MAX_SEQUENCE_TOKENS)
            .checked_mul(CUDA_NATIVE_ADAPTIVE_V1_MAX_SEQUENCE_TOKENS)
            .map(|capacity| capacity.min(maximum_context_tokens))
            .ok_or_else(|| "causal attention replay sequence capacity overflows".to_owned()),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SelectedCausalAttentionPath {
    TokenMajorFallback,
    VllmAddressedFallback,
    VllmAddressedVarlen,
    VllmAddressedVarlenTiled,
    VllmAddressedDecodeV1,
    VllmAddressedDecodeV2,
}

pub(crate) fn select_causal_attention_path(
    policy: AttentionExecutionPolicy,
    shape: CausalAttentionSelectorShape,
    native_compiled: bool,
    active_tokens: u64,
    sequence_tokens: u64,
) -> Result<SelectedCausalAttentionPath, String> {
    use SelectedCausalAttentionPath::*;
    if shape.int8_kv && policy != AttentionExecutionPolicy::Portable {
        return Err("INT8 causal attention requires resolved portable execution".to_owned());
    }
    if !shape.uses_vllm_blocks {
        return Ok(TokenMajorFallback);
    }
    if active_tokens == 1 && shape.tiled_vllm_supported(native_compiled) {
        return match policy {
            AttentionExecutionPolicy::Portable => Ok(VllmAddressedFallback),
            AttentionExecutionPolicy::NativeAdaptive => {
                Ok(match native_decode_kernel(sequence_tokens) {
                    NativeDecodeKernel::V1 => VllmAddressedDecodeV1,
                    NativeDecodeKernel::V2 => VllmAddressedDecodeV2,
                })
            }
            AttentionExecutionPolicy::Auto => {
                Err("causal attention received an unresolved auto policy".to_owned())
            }
        };
    }
    if !shape.addressed_varlen_supported() {
        return Ok(VllmAddressedFallback);
    }
    let score_bytes = sequence_tokens
        .checked_mul(std::mem::size_of::<f32>() as u64)
        .ok_or_else(|| "causal attention varlen score bytes overflow".to_owned())?;
    if active_tokens >= VARLEN_TILED_QUERY_TOKENS
        && score_bytes
            .checked_mul(VARLEN_TILED_QUERY_TOKENS)
            .is_some_and(|bytes| bytes <= VARLEN_DYNAMIC_SHARED_BUDGET_BYTES)
    {
        Ok(VllmAddressedVarlenTiled)
    } else if score_bytes <= VARLEN_DYNAMIC_SHARED_BUDGET_BYTES {
        Ok(VllmAddressedVarlen)
    } else {
        Ok(VllmAddressedFallback)
    }
}

/// Describe only transitions established by the same pure dispatch selector.
/// Varlen's exact-shape eager topology cannot be represented as a replay
/// partition declaration, so retain its known family boundary as Unknown.
pub(crate) fn decode_context_coverage(
    policy: AttentionExecutionPolicy,
    shape: CausalAttentionSelectorShape,
    native_compiled: bool,
    maximum: NonZeroU64,
) -> DecodeContextCoverage {
    use SelectedCausalAttentionPath::*;
    let select =
        |sequence| select_causal_attention_path(policy, shape, native_compiled, 1, sequence);
    let mut known_boundaries = Vec::new();
    let complete = match select(1) {
        Ok(VllmAddressedDecodeV1) => {
            let mut first = CUDA_NATIVE_ADAPTIVE_V1_MAX_SEQUENCE_TOKENS.checked_add(1);
            let mut complete = true;
            while let Some(sequence) = first.filter(|sequence| *sequence <= maximum.get()) {
                if known_boundaries.len() + 2 > MAXIMUM_DECLARED_BOUNDARIES {
                    complete = false;
                    break;
                }
                let before = select(sequence - 1);
                let after = select(sequence);
                if before != after {
                    known_boundaries.push(DecodeContextBoundary {
                        first_sequence_tokens: NonZeroU64::new(sequence).unwrap(),
                        kind: DecodeContextBoundaryKind::KernelFamily,
                    });
                }
                let before_capacity = native_decode_replay_capacity(
                    native_decode_kernel(sequence - 1),
                    sequence - 1,
                    maximum.get(),
                );
                let after_capacity = native_decode_replay_capacity(
                    native_decode_kernel(sequence),
                    sequence,
                    maximum.get(),
                );
                if before_capacity != after_capacity {
                    known_boundaries.push(DecodeContextBoundary {
                        first_sequence_tokens: NonZeroU64::new(sequence).unwrap(),
                        kind: DecodeContextBoundaryKind::ReplayPartition,
                    });
                }
                first = sequence.checked_add(CUDA_NATIVE_ADAPTIVE_V1_MAX_SEQUENCE_TOKENS);
            }
            complete
        }
        Ok(TokenMajorFallback | VllmAddressedFallback) => true,
        Ok(VllmAddressedVarlen | VllmAddressedVarlenTiled) => {
            let first = VARLEN_DYNAMIC_SHARED_BUDGET_BYTES / std::mem::size_of::<f32>() as u64 + 1;
            if first <= maximum.get() && select(first - 1) != select(first) {
                known_boundaries.push(DecodeContextBoundary {
                    first_sequence_tokens: NonZeroU64::new(first).unwrap(),
                    kind: DecodeContextBoundaryKind::KernelFamily,
                });
            }
            false
        }
        Ok(VllmAddressedDecodeV2) | Err(_) => false,
    };
    if complete {
        DecodeContextCoverage::Declared {
            maximum_sequence_tokens: maximum,
            boundaries: known_boundaries,
        }
    } else {
        DecodeContextCoverage::Unknown {
            maximum_sequence_tokens: Some(maximum),
            known_boundaries,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> CausalAttentionSelectorShape {
        CausalAttentionSelectorShape {
            int8_kv: false,
            uses_vllm_blocks: true,
            head_dim: 128,
            sliding_window_tokens: 0,
            attention_scale: 1.0_f32 / 128.0_f32.sqrt(),
        }
    }

    #[test]
    fn decode_boundaries_use_current_token_frontier_and_real_partition_selector() {
        let partition = CUDA_NATIVE_ADAPTIVE_V1_MAX_SEQUENCE_TOKENS;
        let maximum = NonZeroU64::new(partition * 3).unwrap();
        let declared = decode_context_coverage(
            AttentionExecutionPolicy::NativeAdaptive,
            shape(),
            true,
            maximum,
        );
        assert!(declared.is_complete());
        assert_eq!(
            declared
                .known_boundaries()
                .iter()
                .map(|b| (b.first_sequence_tokens.get(), b.kind))
                .collect::<Vec<_>>(),
            [
                (partition + 1, DecodeContextBoundaryKind::KernelFamily),
                (partition + 1, DecodeContextBoundaryKind::ReplayPartition),
                (
                    partition * 2 + 1,
                    DecodeContextBoundaryKind::ReplayPartition
                ),
            ]
        );
        for boundary in declared.known_boundaries() {
            let first = boundary.first_sequence_tokens.get();
            let prior_kv = first - 1;
            assert_eq!(prior_kv + 1, first);
            match boundary.kind {
                DecodeContextBoundaryKind::KernelFamily => {
                    assert_ne!(native_decode_kernel(first - 1), native_decode_kernel(first))
                }
                DecodeContextBoundaryKind::ReplayPartition => assert_ne!(
                    native_decode_replay_capacity(
                        native_decode_kernel(first - 1),
                        first - 1,
                        maximum.get()
                    )
                    .unwrap(),
                    native_decode_replay_capacity(
                        native_decode_kernel(first),
                        first,
                        maximum.get()
                    )
                    .unwrap(),
                ),
            }
        }
        let clipped = decode_context_coverage(
            AttentionExecutionPolicy::NativeAdaptive,
            shape(),
            true,
            NonZeroU64::new(partition).unwrap(),
        );
        assert!(clipped.is_complete());
        assert!(clipped.known_boundaries().is_empty());
        assert_eq!(native_decode_kernel(partition), NativeDecodeKernel::V1);
        assert_eq!(native_decode_kernel(partition + 1), NativeDecodeKernel::V2);
    }

    #[test]
    fn static_shape_and_compiled_support_are_shared_with_dispatch() {
        let maximum = NonZeroU64::new(CUDA_NATIVE_ADAPTIVE_V1_MAX_SEQUENCE_TOKENS * 2).unwrap();
        for head_dim in [128, 256] {
            let shape = CausalAttentionSelectorShape {
                head_dim,
                attention_scale: 1.0_f32 / (head_dim as f32).sqrt(),
                ..shape()
            };
            assert!(shape.tiled_vllm_supported(true));
            assert!(decode_context_coverage(
                AttentionExecutionPolicy::NativeAdaptive,
                shape,
                true,
                maximum
            )
            .is_complete());
        }
        for (shape, compiled) in [
            (
                CausalAttentionSelectorShape {
                    uses_vllm_blocks: false,
                    ..shape()
                },
                true,
            ),
            (
                CausalAttentionSelectorShape {
                    head_dim: 64,
                    ..shape()
                },
                true,
            ),
            (
                CausalAttentionSelectorShape {
                    sliding_window_tokens: 128,
                    ..shape()
                },
                true,
            ),
            (
                CausalAttentionSelectorShape {
                    attention_scale: f32::from_bits(shape().attention_scale.to_bits() + 1),
                    ..shape()
                },
                true,
            ),
            (shape(), false),
        ] {
            assert!(!shape.tiled_vllm_supported(compiled));
            let path = select_causal_attention_path(
                AttentionExecutionPolicy::NativeAdaptive,
                shape,
                compiled,
                1,
                maximum.get(),
            )
            .unwrap();
            assert!(!matches!(
                path,
                SelectedCausalAttentionPath::VllmAddressedDecodeV1
                    | SelectedCausalAttentionPath::VllmAddressedDecodeV2
            ));
            assert!(decode_context_coverage(
                AttentionExecutionPolicy::NativeAdaptive,
                shape,
                compiled,
                maximum
            )
            .known_boundaries()
            .is_empty());
        }
        let int8 = CausalAttentionSelectorShape {
            int8_kv: true,
            uses_vllm_blocks: false,
            ..shape()
        };
        assert!(select_causal_attention_path(
            AttentionExecutionPolicy::NativeAdaptive,
            int8,
            true,
            1,
            1
        )
        .is_err());
        assert!(!decode_context_coverage(
            AttentionExecutionPolicy::NativeAdaptive,
            int8,
            true,
            maximum
        )
        .is_complete());
        assert!(
            decode_context_coverage(AttentionExecutionPolicy::Portable, int8, true, maximum)
                .is_complete()
        );
    }

    #[test]
    fn unresolved_and_exact_shape_routes_remain_unknown_without_losing_known_changes() {
        let maximum = NonZeroU64::new(32_768).unwrap();
        let unknown =
            decode_context_coverage(AttentionExecutionPolicy::Auto, shape(), true, maximum);
        assert!(!unknown.is_complete());
        let varlen =
            decode_context_coverage(AttentionExecutionPolicy::Portable, shape(), false, maximum);
        assert!(!varlen.is_complete());
        assert_eq!(varlen.known_boundaries().len(), 1);
        let boundary = varlen.known_boundaries()[0];
        assert_eq!(boundary.kind, DecodeContextBoundaryKind::KernelFamily);
        assert_ne!(
            select_causal_attention_path(
                AttentionExecutionPolicy::Portable,
                shape(),
                false,
                1,
                boundary.first_sequence_tokens.get() - 1
            )
            .unwrap(),
            select_causal_attention_path(
                AttentionExecutionPolicy::Portable,
                shape(),
                false,
                1,
                boundary.first_sequence_tokens.get()
            )
            .unwrap(),
        );
        let portable =
            decode_context_coverage(AttentionExecutionPolicy::Portable, shape(), true, maximum);
        assert!(portable.is_complete());
        assert!(portable.known_boundaries().is_empty());
    }

    #[test]
    fn metadata_limit_keeps_proven_boundaries_and_withdraws_completeness() {
        let maximum = NonZeroU64::new(
            CUDA_NATIVE_ADAPTIVE_V1_MAX_SEQUENCE_TOKENS * (MAXIMUM_DECLARED_BOUNDARIES as u64 + 1),
        )
        .unwrap();
        let coverage = decode_context_coverage(
            AttentionExecutionPolicy::NativeAdaptive,
            shape(),
            true,
            maximum,
        );
        assert!(!coverage.is_complete());
        assert!(!coverage.known_boundaries().is_empty());
        assert!(coverage.known_boundaries().len() <= MAXIMUM_DECLARED_BOUNDARIES);
    }

    #[test]
    fn shared_selector_keeps_prefill_shared_memory_boundaries() {
        use SelectedCausalAttentionPath::*;
        let scalar_limit = VARLEN_DYNAMIC_SHARED_BUDGET_BYTES / std::mem::size_of::<f32>() as u64;
        let tiled_limit = scalar_limit / VARLEN_TILED_QUERY_TOKENS;
        for (tokens, sequence, expected) in [
            (
                VARLEN_TILED_QUERY_TOKENS,
                tiled_limit,
                VllmAddressedVarlenTiled,
            ),
            (
                VARLEN_TILED_QUERY_TOKENS,
                tiled_limit + 1,
                VllmAddressedVarlen,
            ),
            (2, scalar_limit, VllmAddressedVarlen),
            (2, scalar_limit + 1, VllmAddressedFallback),
        ] {
            assert_eq!(
                select_causal_attention_path(
                    AttentionExecutionPolicy::NativeAdaptive,
                    shape(),
                    true,
                    tokens,
                    sequence
                )
                .unwrap(),
                expected,
            );
        }
    }

    #[test]
    fn native_replay_capacity_keeps_clamp_and_checked_rounding_order() {
        let partition = CUDA_NATIVE_ADAPTIVE_V1_MAX_SEQUENCE_TOKENS;
        for (kernel, sequence, maximum, expected) in [
            (NativeDecodeKernel::V1, 1, partition - 1, partition - 1),
            (NativeDecodeKernel::V1, partition, partition * 3, partition),
            (
                NativeDecodeKernel::V2,
                partition + 1,
                partition * 2 - 1,
                partition * 2 - 1,
            ),
            (
                NativeDecodeKernel::V2,
                partition * 2,
                partition * 3,
                partition * 2,
            ),
            (
                NativeDecodeKernel::V2,
                partition * 2 + 1,
                partition * 3,
                partition * 3,
            ),
        ] {
            assert_eq!(
                native_decode_replay_capacity(kernel, sequence, maximum).unwrap(),
                expected
            );
        }
        let last_full_partition = u64::MAX / partition * partition;
        assert_eq!(
            native_decode_replay_capacity(NativeDecodeKernel::V2, last_full_partition, u64::MAX)
                .unwrap(),
            last_full_partition,
        );
        // Rounding is checked before clamping, even when a small context could
        // otherwise hide the overflow. Live range validation remains with the
        // CUDA replay envelope, as it was before extracting this calculator.
        for maximum in [partition, u64::MAX] {
            for sequence in [last_full_partition + 1, u64::MAX] {
                assert!(
                    native_decode_replay_capacity(NativeDecodeKernel::V2, sequence, maximum)
                        .is_err()
                );
            }
        }
    }

    #[test]
    fn non_single_token_and_int8_portable_keep_dispatch_preconditions() {
        use SelectedCausalAttentionPath::*;
        let policies = [
            AttentionExecutionPolicy::Portable,
            AttentionExecutionPolicy::NativeAdaptive,
            AttentionExecutionPolicy::Auto,
        ];
        let scalar_limit = VARLEN_DYNAMIC_SHARED_BUDGET_BYTES / std::mem::size_of::<f32>() as u64;
        let tiled_limit = scalar_limit / VARLEN_TILED_QUERY_TOKENS;
        for compiled in [false, true] {
            for policy in policies {
                for (tokens, sequence, expected) in [
                    (2, tiled_limit, VllmAddressedVarlen),
                    (
                        VARLEN_TILED_QUERY_TOKENS,
                        tiled_limit,
                        VllmAddressedVarlenTiled,
                    ),
                    (
                        VARLEN_TILED_QUERY_TOKENS,
                        tiled_limit + 1,
                        VllmAddressedVarlen,
                    ),
                    (2, scalar_limit + 1, VllmAddressedFallback),
                ] {
                    assert_eq!(
                        select_causal_attention_path(policy, shape(), compiled, tokens, sequence)
                            .unwrap(),
                        expected
                    );
                }
            }
            // The production kv_layout() unconditionally returns token-major
            // pages for INT8; do not fabricate an INT8/vLLM-block combination.
            let int8 = CausalAttentionSelectorShape {
                int8_kv: true,
                uses_vllm_blocks: false,
                ..shape()
            };
            for tokens in [1, 2, VARLEN_TILED_QUERY_TOKENS] {
                for sequence in [
                    CUDA_NATIVE_ADAPTIVE_V1_MAX_SEQUENCE_TOKENS,
                    CUDA_NATIVE_ADAPTIVE_V1_MAX_SEQUENCE_TOKENS + 1,
                ] {
                    assert_eq!(
                        select_causal_attention_path(
                            AttentionExecutionPolicy::Portable,
                            int8,
                            compiled,
                            tokens,
                            sequence
                        )
                        .unwrap(),
                        TokenMajorFallback
                    );
                    for policy in [
                        AttentionExecutionPolicy::NativeAdaptive,
                        AttentionExecutionPolicy::Auto,
                    ] {
                        assert!(select_causal_attention_path(
                            policy, int8, compiled, tokens, sequence
                        )
                        .is_err());
                    }
                }
            }
        }
    }
}
