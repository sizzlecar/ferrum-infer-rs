//! One two-layer tensor component versus independently prepared single-block
//! B1 executions. Intermediate Host handoffs exist only in this test oracle;
//! the production stack returns one final tensor and performs none of them.
use super::super::super::{
    PH3ImageCheckpointRotary, PH3ImageCurveBlockCheckpoint, PH3ImageCurveLerpArithmetic,
};
use super::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutionOutput, TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_tensor_runtime_config, cuda_vnext_affine_linear_f32_provider,
        cuda_vnext_tensor_providers, CudaTensorOperation,
    },
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use std::sync::Arc;
fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            256 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 64 << 20,
                transfer_staging_bytes: 64 << 20,
                output_bytes: 64 << 20,
                weight_backing_bytes: 64 << 20,
            },
        )
        .unwrap(),
    )
}
fn initial(name: &str) -> ComponentInitializationConfig {
    ComponentInitializationConfig {
        run_id: id(format!("run.curve_stack.{name}")),
        request_id: id(format!("request.curve_stack.{name}")),
        transaction_id: id(format!("transaction.curve_stack.{name}")),
        static_policy: StaticInitializationPolicy::new(1 << 20, 64).unwrap(),
        executor: TensorExecutorConfig::default(),
    }
}
fn compile<F: ModelFamilyProvider>(
    family: F,
    config: &F::Config,
    profile: &str,
    options: ProgramPlanCompileOptions,
    runtime: &CudaDeviceRuntime,
    registry: &OperationRuntimeRegistry<CudaDeviceRuntime>,
    catalog: &CapabilityCatalog,
    materializers: &WeightMaterializerRegistry,
) -> CompiledTensorComponent<CudaDeviceRuntime> {
    let prepared = TypedFamilyRegistration::new(family)
        .prepare_with_profile(&serde_json::to_value(config).unwrap(), &id(profile))
        .unwrap();
    let policy = TensorComponentPolicy::new(
        runtime.descriptor().total_memory_bytes,
        runtime
            .descriptor()
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    )
    .unwrap();
    CompiledTensorComponent::compile(
        prepared,
        registry,
        catalog,
        &policy,
        materializers,
        &options,
    )
    .unwrap()
}
fn input(
    account: &HostMemoryAccount,
    name: &str,
    rows: u64,
    h: u64,
    x: &[u16],
) -> ManagedHostTensor {
    let mut bytes = account
        .allocate_zeroed(HostMemoryClass::InputPreprocess, x.len() * 2)
        .unwrap();
    for (to, &value) in bytes.as_mut().chunks_exact_mut(2).zip(x) {
        to.copy_from_slice(&value.to_le_bytes());
    }
    ManagedHostTensor::new(
        id(name),
        ResolvedTensorSpec::new(
            vec![rows, h],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        bytes.freeze(),
    )
    .unwrap()
}
fn values(
    output: &[TensorExecutionOutput],
    account: &HostMemoryAccount,
    rows: u64,
    h: u64,
) -> Vec<u16> {
    assert_eq!(
        output.len(),
        1,
        "product stack retains only its final activation"
    );
    assert!(account.owns(output[0].bytes()));
    assert_eq!(output[0].tensor().dimensions(), &[rows, h]);
    output[0]
        .bytes()
        .bytes()
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn inputs(
    s: &Scene,
    source: &ComfyCanonicalWeightSource,
    account: &HostMemoryAccount,
    x: &[u16],
    positions: &[[f64; 3]],
    times: &[f32],
    name: &str,
) -> Vec<ManagedHostTensor> {
    let checkpoint = PH3ImageCurveStackCheckpoint::new(&s.config, source, limits()).unwrap();
    let basis = checkpoint
        .managed_basis(account, times, PH3ImageCurveLerpArithmetic::SeparateF32)
        .unwrap();
    let (cos, sin) = PH3ImageCheckpointRotary::new(&s.transformer, source)
        .unwrap()
        .managed_bf16_frequencies(
            account,
            positions,
            id(PH3ImageCurveStackFamily::COSINE_ID),
            id(PH3ImageCurveStackFamily::SINE_ID),
        )
        .unwrap();
    vec![
        input(
            account,
            name,
            positions.len() as u64,
            s.config.block.hidden_size(),
            x,
        ),
        basis,
        cos,
        sin,
    ]
}
#[test]
#[ignore = "requires configured CUDA host with admitted two-layer Curve stack fixture capacity"]
fn stored_mixed_curve_stack_stays_on_device_matches_sequential_b1_and_isolates_asynchronous_b2() {
    let mut s = scene(2, false, true);
    s.config = s
        .config
        .with_projection_storage(PH3ImageStackProjectionStorage::PerLayer(vec![
            PH3ImageBlockProjectionStorage::default(),
            PH3ImageBlockProjectionStorage::uniform(PH3ImageProjectionStorage::Int8W8A8),
        ]))
        .unwrap();
    assert_eq!(s.config.block.requests()[0].rows().unwrap(), 512);
    assert_eq!(s.config.block.requests()[1].rows().unwrap(), 516);
    let files = archive(&payloads(&s.config));
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 2 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 64 << 20,
            maximum_streams: 2,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 128,
            opaque_memory_profile: "curve-stack-fixture-unqualified-capacity".into(),
        })
        .unwrap(),
    );
    let mut providers = cuda_vnext_tensor_providers(
        &runtime,
        &[
            CudaTensorOperation::CastF32ToBf16,
            CudaTensorOperation::ReshapeCopy,
            CudaTensorOperation::RowGather,
            CudaTensorOperation::BatchedRowConcat,
            CudaTensorOperation::RowRmsNormSingleRound,
            CudaTensorOperation::ModulationScale,
            CudaTensorOperation::RowSelectedModulationBf16,
            CudaTensorOperation::RowSelectedAddcmulBf16,
            CudaTensorOperation::PlainResidualAdd,
            CudaTensorOperation::FusedHeadRmsPartialRope,
            CudaTensorOperation::Addcmul,
            CudaTensorOperation::SegmentedAttention,
            CudaTensorOperation::ComfyConvRotW4A8,
            CudaTensorOperation::ComfyConvRotSwiGluF32W4A8,
            CudaTensorOperation::ComfyConvRotW8A8,
            CudaTensorOperation::ComfyConvRotSwiGluF32W8A8,
        ],
    )
    .unwrap();
    providers.push(cuda_vnext_affine_linear_f32_provider(&runtime).unwrap());
    let contracts: Vec<Box<dyn OperationContract>> = vec![
        Box::new(affine_linear_f32_contract().unwrap()),
        Box::new(cast_f32_to_bf16_contract().unwrap()),
        Box::new(reshape_copy_bf16_contract().unwrap()),
        Box::new(row_gather_bf16_contract().unwrap()),
        Box::new(batched_row_concat_bf16_contract().unwrap()),
        Box::new(row_rms_norm_single_round_bf16_contract().unwrap()),
        Box::new(modulation_scale_bf16_contract().unwrap()),
        Box::new(row_selected_modulation_bf16_contract().unwrap()),
        Box::new(row_selected_addcmul_bf16_contract().unwrap()),
        Box::new(plain_residual_add_bf16_contract().unwrap()),
        Box::new(fused_head_rms_partial_rope_bf16_contract().unwrap()),
        Box::new(addcmul_bf16_contract().unwrap()),
        Box::new(segmented_attention_bf16_contract().unwrap()),
        Box::new(comfy_convrot_w4a8_bf16_contract().unwrap()),
        Box::new(comfy_convrot_swiglu_f32_w4a8_bf16_contract().unwrap()),
        Box::new(comfy_convrot_w8a8_bf16_contract().unwrap()),
        Box::new(comfy_convrot_swiglu_f32_w8a8_bf16_contract().unwrap()),
    ];
    let registry = OperationRuntimeRegistry::new(contracts, providers).unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.curve_stack.fixture"),
        ContractVersion::new(1, 0),
        runtime
            .descriptor()
            .runtime_implementation_fingerprint
            .clone(),
        runtime.descriptor().id.clone(),
        runtime.descriptor().capabilities.clone(),
    )
    .unwrap();
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(runtime.descriptor().clone(), vec![engine])
                .unwrap(),
        )
        .unwrap();
    let account = host();
    let checkpoint = PH3ImageCurveStackCheckpoint::new(&s.config, &files.source, limits()).unwrap();
    let family = PH3ImageCurveStackFamily::new(limits()).unwrap();
    let options = family.compile_options(&s.config).unwrap();
    assert!(options.completion_retention().is_empty());
    let mut stack = compile(
        family,
        &s.config,
        PH3ImageCurveStackFamily::cuda_profile_id(&s.config),
        options,
        runtime.as_ref(),
        &registry,
        &catalog,
        &materializers,
    )
    .initialize(
        runtime.clone(),
        account.clone(),
        ComponentWeightSource::Stored(checkpoint.source()),
        initial("init"),
    )
    .unwrap();
    let rows = s.config.block.rows().unwrap();
    let h = s.config.block.hidden_size();
    let output = stack
        .execute(
            id("run.curve_stack.execute"),
            id("request.curve_stack.execute"),
            inputs(
                &s,
                &files.source,
                &account,
                &s.x,
                &s.positions,
                &s.times,
                PH3ImageCurveStackFamily::INPUT_ID,
            ),
        )
        .unwrap();
    let packed = values(&output, &account, rows, h);
    assert_eq!(
        output[0].value_id().as_str(),
        PH3ImageCurveStackFamily::OUTPUT_ID
    );
    drop(output);
    let first_rows = s.config.block.requests()[0].rows().unwrap();
    let first = (first_rows * h) as usize;
    let mut sequential = s.x[..first].to_vec();
    for layer in 0..s.config.layers {
        let c = PH3ImageCurveBlockConfig::from_checkpoint(
            &s.transformer,
            PH3ImageCurveTimeEmbedding::new(4097, 16).unwrap(),
            layer,
            3,
            vec![s.config.block.requests()[0].clone()],
            &files.source,
        )
        .unwrap();
        let checkpoint =
            PH3ImageCurveBlockCheckpoint::new(&c, &files.source, limits().block()).unwrap();
        let mut session = compile(
            PH3ImageCurveBlockFamily::new(limits().block()).unwrap(),
            &c,
            PH3ImageCurveBlockFamily::cuda_profile_id(&c),
            c.compile_options().unwrap(),
            runtime.as_ref(),
            &registry,
            &catalog,
            &materializers,
        )
        .initialize(
            runtime.clone(),
            account.clone(),
            ComponentWeightSource::Stored(checkpoint.source()),
            initial(&format!("single{layer}")),
        )
        .unwrap();
        let output = session
            .execute(
                id(format!("run.curve_stack.single{layer}")),
                id(format!("request.curve_stack.single{layer}")),
                inputs(
                    &s,
                    &files.source,
                    &account,
                    &sequential,
                    &s.positions[..first_rows as usize],
                    &s.times,
                    PH3ImageCurveBlockFamily::INPUT_ID,
                ),
            )
            .unwrap();
        let next = values(&output, &account, first_rows, h);
        assert_ne!(
            next, sequential,
            "the real layer must change its residual stream"
        );
        sequential = next;
        drop(output);
        assert!(matches!(
            session
                .close()
                .unwrap_or_else(|e| panic!("single block close {:?}", e.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
    }
    assert_eq!(
        &packed[..first],
        sequential,
        "one two-layer device graph must equal independent sequential B1 execution"
    );
    s.x[first..]
        .iter_mut()
        .for_each(|v| *v = bf16::from_f32(bf16::from_bits(*v).to_f32() + 0.25).to_bits());
    s.times[1] = 0.75;
    let output = stack
        .execute(
            id("run.curve_stack.changed"),
            id("request.curve_stack.changed"),
            inputs(
                &s,
                &files.source,
                &account,
                &s.x,
                &s.positions,
                &s.times,
                PH3ImageCurveStackFamily::INPUT_ID,
            ),
        )
        .unwrap();
    let changed = values(&output, &account, rows, h);
    assert_eq!(&changed[..first], &packed[..first]);
    assert_ne!(&changed[first..], &packed[first..]);
    drop(output);
    assert!(matches!(
        stack
            .close()
            .unwrap_or_else(|e| panic!("stack close {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}
