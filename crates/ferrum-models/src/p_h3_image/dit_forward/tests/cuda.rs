//! Genuine full Stored graph and shared installation. No timing in this
//! fixture is a product SLO or a Comfy binary qualification.
use super::super::super::{
    PH3ImageCheckpointRotary, PH3ImageCurveLerpArithmetic, PH3ImagePackedBatchLayout,
};
use super::shared_fixture as fixture;
use super::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutionOutput, TensorExecutorConfig, TensorSerialActor,
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
fn initial(name: &str) -> ComponentInitializationConfig {
    ComponentInitializationConfig {
        run_id: id(format!("run.dit.{name}")),
        request_id: id(format!("request.dit.{name}")),
        transaction_id: id(format!("transaction.dit.{name}")),
        static_policy: StaticInitializationPolicy::new(1 << 20, 64).unwrap(),
        executor: TensorExecutorConfig::default(),
    }
}
fn compile(
    config: &PH3ImageDiTConfig,
    runtime: &CudaDeviceRuntime,
    registry: &OperationRuntimeRegistry<CudaDeviceRuntime>,
    catalog: &CapabilityCatalog,
    materializers: &WeightMaterializerRegistry,
    packing: ferrum_runtime::WeightArenaPacking,
) -> CompiledTensorComponent<CudaDeviceRuntime> {
    let family = PH3ImageDiTFamily::new(limits()).unwrap();
    let options = family
        .compile_options(config)
        .unwrap()
        .with_weight_arena_packing(packing);
    let prepared = TypedFamilyRegistration::new(family)
        .prepare_with_profile(
            &serde_json::to_value(config).unwrap(),
            &id(PH3ImageDiTFamily::cuda_profile_id(config)),
        )
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
fn registry(
    runtime: &CudaDeviceRuntime,
) -> (
    OperationRuntimeRegistry<CudaDeviceRuntime>,
    CapabilityCatalog,
    WeightMaterializerRegistry,
) {
    let mut providers = cuda_vnext_tensor_providers(
        runtime,
        &[
            CudaTensorOperation::AffineLinear,
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
            CudaTensorOperation::RowSelectedModulationBf16F32,
            CudaTensorOperation::ComfyConvRotW4A8,
            CudaTensorOperation::ComfyConvRotSwiGluF32W4A8,
            CudaTensorOperation::ComfyConvRotW8A8,
            CudaTensorOperation::ComfyConvRotSwiGluF32W8A8,
        ],
    )
    .unwrap();
    providers.push(cuda_vnext_affine_linear_f32_provider(runtime).unwrap());
    let contracts: Vec<Box<dyn OperationContract>> = vec![
        Box::new(affine_linear_bf16_contract().unwrap()),
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
        Box::new(row_selected_modulation_bf16_f32_contract().unwrap()),
        Box::new(comfy_convrot_w4a8_bf16_contract().unwrap()),
        Box::new(comfy_convrot_swiglu_f32_w4a8_bf16_contract().unwrap()),
        Box::new(comfy_convrot_w8a8_bf16_contract().unwrap()),
        Box::new(comfy_convrot_swiglu_f32_w8a8_bf16_contract().unwrap()),
    ];
    let registry = OperationRuntimeRegistry::new(contracts, providers).unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.dit.fixture"),
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
    (registry, catalog, materializers)
}
struct Samples {
    images: Vec<ManagedHostTensor>,
    audio: Vec<ManagedHostTensor>,
    references: Vec<ManagedHostTensor>,
    noise: Vec<ManagedHostTensor>,
    connector: ManagedHostTensor,
}
impl Samples {
    fn new(c: &PH3ImageDiTConfig, host: &HostMemoryAccount, second_delta: f32) -> Self {
        let images = (0..2)
            .map(|i| {
                f32_tensor(
                    host,
                    "value.sample.image",
                    vec![1, 2, 1, 4, 4],
                    (0..32).map(|n| {
                        ((n * 7 + i * 3) % 19) as f32 / 32. - 0.25
                            + if i == 1 { second_delta } else { 0. }
                    }),
                )
            })
            .collect();
        let audio = (0..2)
            .map(|i| {
                f32_tensor(
                    host,
                    "value.sample.audio",
                    vec![1, 2, 2, 2],
                    (0..8).map(|n| ((n * 3 + i * 7) % 13) as f32 / 32. - 0.125),
                )
            })
            .collect();
        let references = (0..2)
            .map(|i| {
                f32_tensor(
                    host,
                    "value.reference",
                    vec![1, 2, 1, 2, 4],
                    (0..16).map(|n| ((n * 5 + i * 11) % 17) as f32 / 32. - 0.25),
                )
            })
            .collect();
        let noise = (0..2)
            .map(|i| {
                f32_tensor(
                    host,
                    "value.reference.noise",
                    vec![2, 8],
                    (0..16).map(|n| ((n * 13 + i * 19) % 23) as f32 / 32. - 0.375),
                )
            })
            .collect();
        let connector = f32_tensor(
            host,
            "value.connector",
            vec![c.text_rows().unwrap(), 4],
            (0..4096).map(|n| ((n * 17) % 41) as f32 / 32. - 0.625),
        );
        Self {
            images,
            audio,
            references,
            noise,
            connector,
        }
    }
    fn prepare(
        &self,
        c: &PH3ImageDiTConfig,
        checkpoint: &PH3ImageDiTCheckpoint<'_>,
        host: &HostMemoryAccount,
        text: PH3ImageDiTTextInput<'_>,
        times: &[f32],
    ) -> PreparedPH3ImageDiTInputs {
        let t = transformer();
        let mut tags = vec![PH3ImageModality::Text; 512];
        tags[1] = PH3ImageModality::Image;
        let references = [PH3ImageReferenceImage::new(2, 4).unwrap()];
        let target = PH3ImageReferenceImage::new(4, 4).unwrap();
        let layout = PH3ImageRequestLayout::new(&t, &tags, target, &references).unwrap();
        let layouts = [layout, layout];
        let packed = PH3ImagePackedBatchLayout::new(&layouts).unwrap();
        let mut positions = vec![[0.; 3]; packed.rows()];
        packed.write_positions(&mut positions).unwrap();
        let basis = checkpoint
            .managed_basis(host, times, PH3ImageCurveLerpArithmetic::SeparateF32)
            .unwrap();
        let (cos, sin) = PH3ImageCheckpointRotary::new(&t, checkpoint.source())
            .unwrap()
            .managed_bf16_frequencies(
                host,
                &positions,
                id(PH3ImageDiTFamily::COSINE_ID),
                id(PH3ImageDiTFamily::SINE_ID),
            )
            .unwrap();
        let samples = (0..2)
            .map(|i| PH3ImageDiTSample {
                image: &self.images[i],
                audio: &self.audio[i],
            })
            .collect::<Vec<_>>();
        let refs = (0..2)
            .map(|i| {
                Some(PH3ImageDiTReference::default_augmented(
                    &self.references[i],
                    &self.noise[i],
                ))
            })
            .collect::<Vec<_>>();
        PH3ImageDiTInputAdapter::new(c, host)
            .unwrap()
            .prepare(&samples, &refs, text, basis, cos, sin)
            .unwrap()
    }
}
fn output<'a>(outputs: &'a [TensorExecutionOutput], id: &str) -> &'a TensorExecutionOutput {
    outputs
        .iter()
        .find(|v| v.value_id().as_str() == id)
        .unwrap()
}
#[test]
#[ignore = "requires configured CUDA host with admitted full DiT first/steady fixture capacity"]
fn full_dit_first_refine_then_steady_reuses_one_installation_and_isolates_requests() {
    shared_installation(ferrum_runtime::WeightArenaPacking::ByElementType);
}

#[test]
#[ignore = "requires configured CUDA host with admitted full DiT first/steady fixture capacity"]
fn bounded_dit_first_refine_then_steady_reuses_one_installation_and_isolates_requests() {
    shared_installation(ferrum_runtime::WeightArenaPacking::Bounded {
        target_bytes: std::num::NonZeroU64::new(512 << 10).unwrap(),
    });
}

fn shared_installation(packing: ferrum_runtime::WeightArenaPacking) {
    let c = fixture::config();
    let steady = c.clone().with_phase(PH3ImageDiTPhase::RefinedText);
    let files = fixture::files(&c);
    let checkpoint = PH3ImageDiTCheckpoint::new(&c, &files.source, limits()).unwrap();
    let steady_checkpoint = PH3ImageDiTCheckpoint::new(&steady, &files.source, limits()).unwrap();
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 2 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 64 << 20,
            maximum_streams: 2,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 256,
            opaque_memory_profile: "full-dit-fixture-unqualified-capacity".into(),
        })
        .unwrap(),
    );
    let (registry, catalog, materializers) = registry(&runtime);
    let first_plan = compile(&c, &runtime, &registry, &catalog, &materializers, packing);
    let steady_plan = compile(
        &steady,
        &runtime,
        &registry,
        &catalog,
        &materializers,
        packing,
    );
    let refine_config = c.clone().with_phase(PH3ImageDiTPhase::RefineOnly);
    let refine_plan = compile(
        &refine_config,
        &runtime,
        &registry,
        &catalog,
        &materializers,
        packing,
    );
    if matches!(packing, ferrum_runtime::WeightArenaPacking::Bounded { .. }) {
        let allocations = first_plan
            .executable()
            .execution_plan()
            .payload()
            .memory()
            .static_allocations();
        let types = allocations
            .iter()
            .map(|allocation| allocation.element_type())
            .collect::<BTreeSet<_>>();
        assert!(
            allocations.len() > types.len(),
            "bounded fixture must split at least one dtype arena"
        );
    }
    assert_eq!(
        first_plan
            .executable()
            .execution_plan()
            .payload()
            .execution_weights(),
        steady_plan
            .executable()
            .execution_plan()
            .payload()
            .execution_weights()
    );
    assert_eq!(
        first_plan
            .executable()
            .execution_plan()
            .payload()
            .execution_weights(),
        refine_plan
            .executable()
            .execution_plan()
            .payload()
            .execution_weights(),
    );
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            256 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 64 << 20,
                output_bytes: 64 << 20,
                transfer_staging_bytes: 64 << 20,
                weight_backing_bytes: 64 << 20,
            },
        )
        .unwrap(),
    );
    let (mut first, weights) = first_plan
        .initialize_shared_weights(
            runtime.clone(),
            host.clone(),
            ComponentWeightSource::Stored(&checkpoint),
            initial("first"),
        )
        .unwrap();
    let mut actor = TensorSerialActor::new(runtime.clone()).unwrap();
    let old_refine_scope = host
        .reserve_scope(&[
            (HostMemoryClass::InputPreprocess, 0),
            (HostMemoryClass::Output, 0),
        ])
        .unwrap();
    let mut refine = refine_plan
        .initialize_with_shared_weights(
            runtime.clone(),
            old_refine_scope.account(),
            &weights,
            initial("refine_only"),
        )
        .unwrap();
    drop(old_refine_scope);
    let refine_work = host
        .reserve_scope(&[
            (
                HostMemoryClass::InputPreprocess,
                host.backing_bytes(
                    (refine_config.text_rows().unwrap() * refine_config.text_dim * 2) as usize,
                )
                .unwrap(),
            ),
            (
                HostMemoryClass::Output,
                host.backing_bytes(
                    (refine_config.text_rows().unwrap() * refine_config.hidden * 2) as usize,
                )
                .unwrap(),
            ),
        ])
        .unwrap();
    refine
        .set_output_host_account(refine_work.account())
        .unwrap();
    let mut next = steady_plan
        .initialize_with_shared_weights(runtime.clone(), host.clone(), &weights, initial("steady"))
        .unwrap();
    let requests = [id("request.dit.doc0"), id("request.dit.doc1")];
    let samples = Samples::new(&c, &host, 0.);
    let keys = PH3ImageDiTConditioningKey::from_connector(
        &c,
        &samples.connector,
        &requests,
        weights.evidence(),
        1 << 20,
    )
    .unwrap();
    let scopes = c
        .requests
        .iter()
        .map(|r| {
            host.reserve_scope(&[(
                HostMemoryClass::InputPreprocess,
                host.backing_bytes((r.text_rows * c.hidden * 2) as usize)
                    .unwrap(),
            )])
            .unwrap()
        })
        .collect::<Vec<_>>();
    let destinations = requests
        .iter()
        .zip(&scopes)
        .map(|(request, scope)| PH3ImageRefinementDestination::new(request, scope))
        .collect::<Vec<_>>();
    let scoped_caches = PH3ImageTextRefinement::bind(
        &refine_config,
        limits(),
        &mut refine,
        &weights,
        refine_work.account(),
    )
    .unwrap()
    .execute_on_actor(
        &mut actor,
        id("run.dit.refine_only"),
        id("request.dit.refine_only"),
        &samples.connector,
        &destinations,
    )
    .unwrap();
    drop(destinations);
    let first_prepared = samples.prepare(
        &c,
        &checkpoint,
        &host,
        PH3ImageDiTTextInput::WithTextRefiner {
            connector: &samples.connector,
            keys: &keys,
        },
        &[0.125, 0.625, 0.999],
    );
    let first_output = first
        .execute_on_actor(
            &mut actor,
            id("run.dit.first"),
            id("request.dit.first"),
            first_prepared.inputs.clone(),
        )
        .unwrap();
    assert_eq!(first_output.len(), 3);
    let port = first_plan
        .executable()
        .execution_plan()
        .payload()
        .tensor_ports()
        .unwrap()
        .outputs()
        .iter()
        .find(|p| p.value_id().as_str() == PH3ImageDiTFamily::REFINED_TEXT_ID)
        .unwrap();
    let caches = PH3ImageRefinedTextCache::publish(
        &c,
        &host,
        &first_prepared,
        port,
        output(&first_output, PH3ImageDiTFamily::REFINED_TEXT_ID),
    )
    .unwrap();
    assert!(caches
        .iter()
        .zip(&keys)
        .all(|(cache, key)| cache.key() == key));
    let scoped_text = cache::concatenate(&steady, &host, &scoped_caches, &keys).unwrap();
    let first_text = cache::concatenate(&steady, &host, &caches, &keys).unwrap();
    assert_eq!(
        scoped_text.bytes().bytes(),
        first_text.bytes().bytes(),
        "fresh-only attached refinement equals the full first forward's refiner"
    );
    drop((scoped_text, first_text));
    let refined = samples.prepare(
        &steady,
        &steady_checkpoint,
        &host,
        PH3ImageDiTTextInput::RefinedText {
            caches: &scoped_caches,
            keys: &keys,
        },
        &[0.125, 0.625, 0.999],
    );
    let second_output = next
        .execute_on_actor(
            &mut actor,
            id("run.dit.steady"),
            id("request.dit.steady"),
            refined.inputs,
        )
        .unwrap();
    assert_eq!(second_output.len(), 2);
    for name in [
        PH3ImageDiTFamily::IMAGE_VELOCITY_ID,
        PH3ImageDiTFamily::AUDIO_VELOCITY_ID,
    ] {
        let a = output(&first_output, name);
        let b = output(&second_output, name);
        assert_eq!(a.tensor(), b.tensor());
        assert_eq!(
            a.bytes().bytes(),
            b.bytes().bytes(),
            "cached first/steady arithmetic {name}"
        );
        assert!(host.owns(a.bytes()) && host.owns(b.bytes()));
        assert!(a
            .bytes()
            .bytes()
            .chunks_exact(2)
            .all(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).is_finite()));
    }
    let changed = Samples::new(&steady, &host, 0.5);
    let prepared = changed.prepare(
        &steady,
        &steady_checkpoint,
        &host,
        PH3ImageDiTTextInput::RefinedText {
            caches: &scoped_caches,
            keys: &keys,
        },
        &[0.125, 0.75, 0.999],
    );
    let changed_output = next
        .execute_on_actor(
            &mut actor,
            id("run.dit.changed"),
            id("request.dit.changed"),
            prepared.inputs,
        )
        .unwrap();
    let target_bytes =
        (c.requests[0].target_rows(&c).unwrap() * c.image_width().unwrap() * 2) as usize;
    let a = output(&second_output, PH3ImageDiTFamily::IMAGE_VELOCITY_ID)
        .bytes()
        .bytes();
    let b = output(&changed_output, PH3ImageDiTFamily::IMAGE_VELOCITY_ID)
        .bytes()
        .bytes();
    assert_eq!(
        &a[..target_bytes],
        &b[..target_bytes],
        "asynchronous second document must not affect first image"
    );
    assert_ne!(
        &a[target_bytes..],
        &b[target_bytes..],
        "second request perturbation must be live"
    );
    let a = output(&second_output, PH3ImageDiTFamily::AUDIO_VELOCITY_ID)
        .bytes()
        .bytes();
    let b = output(&changed_output, PH3ImageDiTFamily::AUDIO_VELOCITY_ID)
        .bytes()
        .bytes();
    assert_eq!(
        &a[..16],
        &b[..16],
        "first stereo audio4 must remain isolated"
    );
    drop(first_output);
    drop(second_output);
    drop(changed_output);
    drop(first_prepared);
    drop(caches);
    drop(scoped_caches);
    drop(scopes);
    drop(keys);
    drop(samples);
    drop(changed);
    first
        .close()
        .unwrap_or_else(|failure| panic!("first close: {:?}", failure.failure()));
    next.close()
        .unwrap_or_else(|failure| panic!("steady close: {:?}", failure.failure()));
    refine
        .close()
        .unwrap_or_else(|failure| panic!("refine close: {:?}", failure.failure()));
    drop(weights);
    drop(first_plan);
    drop(steady_plan);
    drop(refine_plan);
    // The scope still owns its reserved credit after the sessions and their
    // readbacks retire. Close that original grant before checking total usage.
    refine_work.close().unwrap();
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
