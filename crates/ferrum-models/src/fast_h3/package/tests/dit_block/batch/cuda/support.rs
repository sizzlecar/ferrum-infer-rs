use super::*;
use ferrum_execution::{
    component::{ComponentInitializationConfig, TensorComponentPolicy},
    TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::vnext_ops::CudaTensorOperation as Op;

pub(super) fn ident<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
pub(super) fn initial(label: &str) -> ComponentInitializationConfig {
    ComponentInitializationConfig {
        run_id: ident(format!("run.batch.init.{label}")),
        request_id: ident(format!("request.batch.init.{label}")),
        transaction_id: ident(format!("transaction.batch.init.{label}")),
        static_policy: StaticInitializationPolicy::new(1 << 20, 128).unwrap(),
        executor: TensorExecutorConfig::default(),
    }
}
pub(super) fn operations() -> Vec<Op> {
    vec![
        Op::RowSelectedRmsModulation,
        Op::RowSelectedResidualRmsModulation,
        Op::RowSelectedAddcmulFmaBf16,
        Op::HeadRmsPartialRopeSingleRound,
        Op::MaskedTileMean,
        Op::PackedSwiGlu,
        Op::RowGather,
        Op::AffineLinearF32,
        Op::AffineLinearF32ToBf16,
        Op::CastF32ToBf16,
    ]
}
pub(super) fn contracts() -> Vec<Box<dyn OperationContract>> {
    vec![
        Box::new(row_selected_rms_modulation_bf16_contract().unwrap()),
        Box::new(row_selected_residual_rms_modulation_bf16_contract().unwrap()),
        Box::new(row_selected_addcmul_fma_bf16_contract().unwrap()),
        Box::new(head_rms_partial_rope_single_round_bf16_contract().unwrap()),
        Box::new(masked_tile_mean_bf16_f32_contract().unwrap()),
        Box::new(packed_swiglu_bf16_contract().unwrap()),
        Box::new(row_gather_bf16_contract().unwrap()),
        Box::new(affine_linear_f32_contract().unwrap()),
        Box::new(affine_linear_f32_to_bf16_contract().unwrap()),
        Box::new(cast_f32_to_bf16_contract().unwrap()),
    ]
}
pub(super) fn compile(
    runtime: &CudaDeviceRuntime,
    registry: &OperationRuntimeRegistry<CudaDeviceRuntime>,
    prepared: PreparedModelFamily,
    options: &ProgramPlanCompileOptions,
) -> CompiledTensorComponent<CudaDeviceRuntime> {
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        ident("provider.engine.batch"),
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )
    .unwrap();
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(d.clone(), vec![engine])
                .unwrap(),
        )
        .unwrap();
    CompiledTensorComponent::compile(
        prepared,
        registry,
        &catalog,
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect(),
        )
        .unwrap(),
        &materializers,
        options,
    )
    .unwrap()
}
pub(super) fn execute(
    session: &mut TensorComponentSession<CudaDeviceRuntime>,
    inputs: Vec<ManagedHostTensor>,
    label: &str,
) -> TensorExecutionOutput {
    let mut outputs = session
        .execute(
            ident(format!("run.batch.{label}")),
            ident(format!("request.batch.{label}")),
            inputs,
        )
        .unwrap_or_else(|error| {
            let message = format!("{error:?}");
            std::mem::forget(error);
            panic!("CUDA batch execution retains failed command owners: {message}")
        });
    assert_eq!(outputs.len(), 1);
    outputs.pop().unwrap()
}
pub(super) fn close(session: TensorComponentSession<CudaDeviceRuntime>) {
    match session.close() {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => {
            assert_eq!(
                receipt.released_static_resources(),
                0,
                "shared installation stays resident"
            );
        }
        Ok(_) => panic!("completed batch session still referenced"),
        Err(error) => {
            let message = format!("{:?}", error.failure());
            std::mem::forget(error);
            panic!("batch close retains failed owners: {message}")
        }
    }
}

/// Declared before GPU execution: finite BF16 values within two representable
/// steps of the independent B1 execution. This is batch equivalence, not a
/// general FastVideo numerical oracle. Signed zero compares arithmetically.
pub(super) fn compare(actual: &[u8], expected: &[u8]) -> u16 {
    assert_eq!(actual.len(), expected.len());
    let ordered = |x: u16| {
        if x & 0x8000 != 0 {
            0x8000 - (x & 0x7fff)
        } else {
            0x8000 + x
        }
    };
    let mut maximum = 0;
    for (index, (a, b)) in actual
        .chunks_exact(2)
        .zip(expected.chunks_exact(2))
        .enumerate()
    {
        let a = u16::from_le_bytes(a.try_into().unwrap());
        let b = u16::from_le_bytes(b.try_into().unwrap());
        assert!(
            bf16::from_bits(a).is_finite() && bf16::from_bits(b).is_finite(),
            "nonfinite at {index}"
        );
        let distance = ordered(a).abs_diff(ordered(b));
        maximum = maximum.max(distance);
        assert!(
            distance <= 2,
            "B2 vs B1 at {index}: {a:04x} vs {b:04x}, ULP {distance}"
        );
    }
    maximum
}

pub(super) fn fixture() -> (tempfile::TempDir, Arc<FastH3TransformerPackage>) {
    package_with(|fixture| {
        // Exact zero Q/K; V/output/FFN remain active sparse original NVFP4
        // matrices. No cross-request attention can be hidden by zero values.
        for (name, (_, dimensions, bytes)) in &mut fixture.packed {
            if name.ends_with("::_nvfp4_weight") {
                bytes.fill(0);
                if !name.contains("attn.to_q::")
                    && !name.contains("attn.to_k::")
                    && !name.contains("attn.to_gate_compress::")
                {
                    let packed_k = dimensions[1] as usize;
                    for row in 0..dimensions[0] as usize {
                        let column = row % (2 * packed_k);
                        bytes[row * packed_k + column / 2] = 2 << (4 * (column % 2));
                    }
                }
            } else if name.ends_with("::_nvfp4_weight_scale") {
                bytes.fill(0x38);
            } else if name.ends_with("::_nvfp4_alpha") {
                let scale: f32 = if name.contains("attn.to_v::") || name.contains("attn.to_out::") {
                    1.
                } else {
                    0.125
                };
                bytes.copy_from_slice(&scale.to_le_bytes());
            }
        }
        for (name, (_, _, bytes)) in &mut fixture.dense {
            if name.ends_with("adaln_proj.linear.bias") {
                let hidden = bytes.len() / 2 / 6;
                for dst in bytes[2 * hidden * 2..3 * hidden * 2].chunks_exact_mut(2) {
                    dst.copy_from_slice(&bf16::ONE.to_bits().to_le_bytes());
                }
            }
        }
    })
}
