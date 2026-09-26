use super::*;
use crate::gguf_blocks::GgufBlockFormat;
use ferrum_interfaces::{execution_cost::SelectedReplayAlgorithmTemplateV1, vnext::WeightId};

fn make_library(
    s: CausalAttentionShape,
    counts: &[(u64, u64)],
    packed: bool,
    capture: SloStructuredCostCapture,
) -> Option<SelectedCommandCostEvidenceV1> {
    let rows = counts
        .iter()
        .map(|&(count, context)| {
            Row::new(s, AttentionExecutionPolicy::Portable, count, context, false)
        })
        .collect::<Option<Vec<_>>>()?;
    compute(
        s,
        CausalPrecision::F32Master,
        CausalProjection::F16,
        AttentionExecutionPolicy::Portable,
        ProjectionWork::DenseF16(CublasHandleApiIdentity::fixture_identity()),
        &rows,
        counts.iter().map(|r| r.0).sum(),
        packed,
        capture,
    )
}

#[test]
fn cublas_causal_complete_sequence_counts_packed_and_participant_library_calls() {
    use ferrum_interfaces::execution_cost::AlgorithmWorkKindV1;
    let s = shape();
    for counts in [&[(1, 7)][..], &[(3, 9)][..], &[(2, 9), (3, 17)][..]] {
        for packed in [false, true].into_iter().filter(|p| !p || counts.len() > 1) {
            let evidence =
                make_library(s, counts, packed, SloStructuredCostCapture::HostSettledV1).unwrap();
            let dispatches = physical_dispatch_count(
                counts.iter().map(|&(tokens, end)| {
                    CausalAttentionKernelPath::select(
                        AttentionExecutionPolicy::Portable,
                        s,
                        tokens,
                        end,
                    )
                    .unwrap()
                }),
                false,
                false,
                packed,
            );
            evidence
                .validate_command(counts.iter().map(|r| r.0).sum(), dispatches, 0)
                .unwrap();
            let table = evidence.algorithm_work().unwrap().unwrap();
            table.validate_command(&evidence).unwrap();
            let library_calls: u64 = table
                .entries()
                .iter()
                .filter(|entry| entry.kind() == AlgorithmWorkKindV1::LibraryCall)
                .map(|entry| entry.commands())
                .sum();
            assert_eq!(
                library_calls,
                if packed { 4 } else { 4 * counts.len() as u64 }
            );
            let kernels: u64 = table
                .entries()
                .iter()
                .filter(|entry| entry.kind() == AlgorithmWorkKindV1::Kernel)
                .map(|entry| entry.commands())
                .sum();
            assert!(
                kernels > 0,
                "projection evidence cannot omit attention/RMS/residual work"
            );
            assert_eq!(kernels + library_calls, dispatches);
        }
    }
}

#[test]
fn cublas_causal_resident_binding_keeps_gemm_abi_and_updates_context_work() {
    let mut s = shape();
    s.sliding_window_tokens = 128; // stable real addressed fallback
    let on = SloStructuredCostCapture::HostSettledV1;
    let first = make_library(s, &[(1, 37)], false, on).unwrap();
    let next = make_library(s, &[(1, 38)], false, on).unwrap();
    let dispatches = physical_dispatch_count(
        [CausalAttentionKernelPath::VllmAddressedFallback],
        false,
        false,
        false,
    );
    let template =
        SelectedReplayAlgorithmTemplateV1::from_selected(&first, 1, dispatches, 0).unwrap();
    template.validate_binding(&next).unwrap();
    assert_ne!(
        first.algorithm_work().unwrap(),
        next.algorithm_work().unwrap()
    );
    let changed_m = make_library(s, &[(2, 38)], false, on).unwrap();
    assert!(template.validate_binding(&changed_m).is_err());
    s.hidden_size = 512;
    let changed_k = make_library(s, &[(1, 38)], false, on).unwrap();
    assert!(template.validate_binding(&changed_k).is_err());
}

#[test]
fn cublas_causal_disabled_int8_and_wrong_projection_never_mint_table() {
    let on = SloStructuredCostCapture::HostSettledV1;
    assert!(make_library(
        shape(),
        &[(1, 1)],
        false,
        SloStructuredCostCapture::Disabled
    )
    .is_none());
    let mut int8 = shape();
    int8.int8_kv = true;
    assert!(make_library(int8, &[(1, 1)], false, on).is_none());
    // No identity can be manufactured by this producer: DenseF16's identity
    // is required by its type, and only actual/future entrypoints can obtain
    // it from the immutable runtime source after real handle observation.
    let s = shape();
    let rows = [Row::new(s, AttentionExecutionPolicy::Portable, 1, 1, false).unwrap()];
    assert!(compute(
        s,
        CausalPrecision::F32Master,
        CausalProjection::F16,
        AttentionExecutionPolicy::Portable,
        ProjectionWork::Native([&[], &[], &[], &[]]),
        &rows,
        1,
        false,
        on
    )
    .is_none());
    assert!(compute(
        s,
        CausalPrecision::F32Master,
        CausalProjection::Native {
            transform_bytes_per_token: 0,
        },
        AttentionExecutionPolicy::Portable,
        ProjectionWork::DenseF16(CublasHandleApiIdentity::fixture_identity()),
        &rows,
        1,
        false,
        on
    )
    .is_none());
}

fn shape() -> CausalAttentionShape {
    CausalAttentionShape {
        int8_kv: false,
        hidden_size: 256,
        query_heads: 2,
        key_value_heads: 2,
        head_dim: 128,
        query_features: 256,
        query_projection_features: 256,
        kv_features: 256,
        rope_dim: 128,
        rope_frequency_denominator: 128,
        rope_pair_offset: 64,
        maximum_context_tokens: 2048,
        epsilon: 1e-6,
        rope_theta: 10000.0,
        attention_scale: 1.0 / 128_f32.sqrt(),
        sliding_window_tokens: 0,
        rope_interleaved: false,
        output_gate: false,
        value_rms_norm: false,
        attention_k_eq_v: false,
        post_attention_norm: false,
    }
}
fn part(format: GgufBlockFormat) -> weights::MatrixPart {
    weights::MatrixPart {
        component_id: WeightId::new(format!("causal.weight.{format:?}")).unwrap(),
        format: weights::MatrixFormat::Block(format),
        rows: 256,
        columns: 256,
        output_offset: 0,
        transform: None,
        signs_region: None,
    }
}
fn make(
    s: CausalAttentionShape,
    policy: AttentionExecutionPolicy,
    counts: &[(u64, u64)],
    packed: bool,
    inplace: bool,
    capture: SloStructuredCostCapture,
) -> Option<SelectedCommandCostEvidenceV1> {
    let q = [part(GgufBlockFormat::Q5K)];
    let k = [part(GgufBlockFormat::Q4K)];
    let v = [part(GgufBlockFormat::Q8_0)];
    let o = [part(GgufBlockFormat::Q6K)];
    let rows = counts
        .iter()
        .map(|&(count, context)| Row::new(s, policy, count, context, inplace))
        .collect::<Option<Vec<_>>>()?;
    compute(
        s,
        CausalPrecision::F16,
        CausalProjection::Native {
            transform_bytes_per_token: 0,
        },
        policy,
        ProjectionWork::Native([&q, &k, &v, &o]),
        &rows,
        counts.iter().map(|r| r.0).sum(),
        packed,
        capture,
    )
}
#[test]
fn cuda_selected_causal_complete_native_work_and_alias_are_bound() {
    let on = SloStructuredCostCapture::HostSettledV1;
    let s = shape();
    for counts in [&[(4, 4)][..], &[(2, 9), (3, 17)][..]] {
        for packed in [false, true]
            .into_iter()
            .filter(|packed| !packed || counts.len() > 1)
        {
            let evidence = make(
                s,
                AttentionExecutionPolicy::Portable,
                counts,
                packed,
                false,
                on,
            )
            .unwrap();
            let dispatches = physical_dispatch_count(
                counts.iter().map(|&(tokens, end)| {
                    CausalAttentionKernelPath::select(
                        AttentionExecutionPolicy::Portable,
                        s,
                        tokens,
                        end,
                    )
                    .unwrap()
                }),
                false,
                false,
                packed,
            );
            let tokens = counts.iter().map(|r| r.0).sum();
            evidence.validate_command(tokens, dispatches, 0).unwrap();
            evidence
                .algorithm_work()
                .unwrap()
                .unwrap()
                .validate_command(&evidence)
                .unwrap();
            let template =
                SelectedReplayAlgorithmTemplateV1::from_selected(&evidence, tokens, dispatches, 0)
                    .unwrap();
            let alias = make(
                s,
                AttentionExecutionPolicy::Portable,
                counts,
                packed,
                true,
                on,
            )
            .unwrap();
            assert!(
                template.validate_binding(&alias).is_err(),
                "F16 in-place ABI cannot reuse out-of-place seal"
            );
        }
    }
}
#[test]
fn cuda_selected_causal_current_context_changes_work_without_reusing_capture_numbers() {
    let on = SloStructuredCostCapture::HostSettledV1;
    let mut s = shape();
    s.sliding_window_tokens = 128; // explicit stable fallback even without native vLLM feature
    let first = make(
        s,
        AttentionExecutionPolicy::Portable,
        &[(1, 37), (1, 81)],
        true,
        false,
        on,
    )
    .unwrap();
    let next = make(
        s,
        AttentionExecutionPolicy::Portable,
        &[(1, 38), (1, 82)],
        true,
        false,
        on,
    )
    .unwrap();
    let dispatches = physical_dispatch_count(
        [CausalAttentionKernelPath::VllmAddressedFallback; 2],
        false,
        false,
        true,
    );
    SelectedReplayAlgorithmTemplateV1::from_selected(&first, 2, dispatches, 0)
        .unwrap()
        .validate_binding(&next)
        .unwrap();
    assert_ne!(first.work().inner_work_units, next.work().inner_work_units);
    assert_ne!(
        first.algorithm_work().unwrap(),
        next.algorithm_work().unwrap()
    );
    let changed = make(
        s,
        AttentionExecutionPolicy::Portable,
        &[(2, 38), (1, 82)],
        true,
        false,
        on,
    )
    .unwrap();
    assert!(
        SelectedReplayAlgorithmTemplateV1::from_selected(&first, 2, dispatches, 0)
            .unwrap()
            .validate_binding(&changed)
            .is_err()
    );
}
#[test]
fn cuda_selected_causal_disabled_is_lazy_and_unsupported_paths_remain_unknown() {
    let lazy = std::iter::from_fn(|| -> Option<u64> {
        panic!("disabled capture must not enumerate binding writes")
    });
    assert!(binding_evidence(lazy, 3, SloStructuredCostCapture::Disabled).is_none());
    let mut s = shape();
    s.int8_kv = true;
    assert!(make(
        s,
        AttentionExecutionPolicy::Portable,
        &[(1, 1)],
        false,
        false,
        SloStructuredCostCapture::HostSettledV1
    )
    .is_none());
    assert!(Row::new(shape(), AttentionExecutionPolicy::Portable, 2, 1, false).is_none());
    assert!(Row::new(shape(), AttentionExecutionPolicy::Portable, 1, 2049, false).is_none());
}
#[test]
#[cfg(feature = "vllm-paged-attn-v2")]
fn cuda_selected_causal_native_v1_v2_and_batched_reduce_use_resident_geometry_and_current_work() {
    let on = SloStructuredCostCapture::HostSettledV1;
    let s = shape();
    for (first_counts, next_counts, packed) in [
        (&[(1, 32)][..], &[(1, 33)][..], false),
        (&[(1, 32), (1, 511)][..], &[(1, 33), (1, 512)][..], true),
        (
            &[(1, 32), (1, 510), (1, 513), (1, 1023), (1, 128)][..],
            &[(1, 33), (1, 511), (1, 514), (1, 1024), (1, 129)][..],
            true,
        ),
        (&[(1, 513), (1, 1023)][..], &[(1, 514), (1, 1024)][..], true),
    ] {
        let first = make(
            s,
            AttentionExecutionPolicy::NativeAdaptive,
            first_counts,
            packed,
            false,
            on,
        )
        .unwrap();
        let next = make(
            s,
            AttentionExecutionPolicy::NativeAdaptive,
            next_counts,
            packed,
            false,
            on,
        )
        .unwrap();
        let paths = first_counts.iter().map(|&(tokens, end)| {
            CausalAttentionKernelPath::select(
                AttentionExecutionPolicy::NativeAdaptive,
                s,
                tokens,
                end,
            )
            .unwrap()
        });
        let dispatches = physical_dispatch_count(paths, false, false, packed);
        let tokens = first_counts.iter().map(|r| r.0).sum();
        first.validate_command(tokens, dispatches, 0).unwrap();
        SelectedReplayAlgorithmTemplateV1::from_selected(&first, tokens, dispatches, 0)
            .unwrap()
            .validate_binding(&next)
            .unwrap();
        assert_ne!(first.work().inner_work_units, next.work().inner_work_units);
    }
    let resident = make(
        s,
        AttentionExecutionPolicy::NativeAdaptive,
        &[(1, 513), (1, 1023)],
        true,
        false,
        on,
    )
    .unwrap();
    let changed_grid = make(
        s,
        AttentionExecutionPolicy::NativeAdaptive,
        &[(1, 514), (1, 1025)],
        true,
        false,
        on,
    )
    .unwrap();
    assert!(
        SelectedReplayAlgorithmTemplateV1::from_selected(
            &resident,
            2,
            physical_dispatch_count(
                [CausalAttentionKernelPath::VllmAddressedDecodeV2; 2],
                false,
                false,
                true
            ),
            0
        )
        .unwrap()
        .validate_binding(&changed_grid)
        .is_err(),
        "same native family does not authorize a different resident partition grid"
    );
    let v1 = make(
        s,
        AttentionExecutionPolicy::NativeAdaptive,
        &[(1, 512)],
        false,
        false,
        on,
    )
    .unwrap();
    let v2 = make(
        s,
        AttentionExecutionPolicy::NativeAdaptive,
        &[(1, 513)],
        false,
        false,
        on,
    )
    .unwrap();
    assert!(SelectedReplayAlgorithmTemplateV1::from_selected(
        &v1,
        1,
        physical_dispatch_count(
            [CausalAttentionKernelPath::VllmAddressedDecodeV1],
            false,
            false,
            false
        ),
        0
    )
    .unwrap()
    .validate_binding(&v2)
    .is_err());
}
