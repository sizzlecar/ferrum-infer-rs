//! Actual two-layer Stored graph qualification. The declared architecture's
//! layer count is always executed; 36-layer coverage is separately checked by
//! the CPU graph/schema test, not claimed as a full-model GPU run here.
use super::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_tensor_runtime_config, cuda_vnext_affine_linear_f32_provider,
        cuda_vnext_tensor_providers, CudaTensorOperation,
    },
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use std::sync::{Arc, Mutex};
type Values = BTreeMap<ProgramValueId, Vec<f32>>;
pub(in crate::p_h3_image_vae) static GPU: Mutex<()> = Mutex::new(());
fn inputs(c: &PH3ImageVaeDecoderExecutionConfig, perturb: bool) -> Values {
    let host = host();
    let layout = crate::p_h3_image_vae::image_layout::PH3ImageVaeImageLayout::new(
        c.architecture().clone(),
        c.batch() as usize,
        c.latent_height() as usize,
        c.latent_width() as usize,
    )
    .unwrap();
    let (cosine, sine) = layout
        .managed_rotary(&host, id(Family::COSINE_ID), id(Family::SINE_ID))
        .unwrap();
    let read = |v: &ManagedHostTensor| {
        v.bytes()
            .bytes()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    };
    let per_batch = c.patches().unwrap() as usize * c.architecture().latent_channels() as usize;
    let values = (0..c.batch() as usize * per_batch)
        .map(|i| {
            let b = i / per_batch;
            ((i * 7 + b * 19) % 37) as f32 / 32. - 0.5
                + 0.000123
                + if perturb && b == 1 { 0.75 } else { 0. }
        })
        .collect();
    let result = BTreeMap::from([
        (id(Family::INPUT_ID), values),
        (id(Family::COSINE_ID), read(&cosine)),
        (id(Family::SINE_ID), read(&sine)),
    ]);
    drop(cosine);
    drop(sine);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    result
}
fn oracle(
    c: &PH3ImageVaeDecoderExecutionConfig,
    source: &ComfyCanonicalWeightSource,
    values: &Values,
) -> Vec<f32> {
    let post = affine(
        &values[&id::<ProgramValueId>(Family::INPUT_ID)],
        &floats(source, "post_quant_conv.weight"),
        &floats(source, "post_quant_conv.bias"),
    );
    let patches = affine(
        &post,
        &floats(source, "decoder.x_embedder.weight"),
        &floats(source, "decoder.x_embedder.bias"),
    );
    let h = c.architecture().hidden_size().unwrap() as usize;
    let p = c.patches().unwrap() as usize;
    let registers = floats(source, "decoder.register_tokens");
    let mut hidden = Vec::new();
    for b in 0..c.batch() as usize {
        hidden.extend_from_slice(&patches[b * p * h..(b + 1) * p * h]);
        hidden.extend_from_slice(&registers);
        hidden.extend(std::iter::repeat_n(0., h));
    }
    for layer in 0..c.architecture().layers() {
        let map = BTreeMap::from([
            (id(PH3ImageVaeDecoderBlockFamily::INPUT_ID), hidden),
            (
                id(PH3ImageVaeDecoderBlockFamily::COSINE_ID),
                values[&id::<ProgramValueId>(Family::COSINE_ID)].clone(),
            ),
            (
                id(PH3ImageVaeDecoderBlockFamily::SINE_ID),
                values[&id::<ProgramValueId>(Family::SINE_ID)].clone(),
            ),
        ]);
        hidden = crate::p_h3_image_vae::decoder_block::tests::oracle::execute(
            &c.block(layer).unwrap(),
            source,
            &map,
        );
    }
    let gamma = floats(source, "decoder.norm_out.weight");
    let beta = floats(source, "decoder.norm_out.bias");
    let eps = c.architecture().norm_epsilon();
    let eps = eps.numerator() as f64 / eps.denominator() as f64;
    let normalized: Vec<_> = hidden
        .chunks_exact(h)
        .flat_map(|row| {
            let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / h as f64;
            let variance = row
                .iter()
                .map(|&v| (f64::from(v) - mean).powi(2))
                .sum::<f64>()
                / h as f64;
            let inverse = 1. / (variance + eps).sqrt();
            row.iter().enumerate().map({
                let gamma = &gamma;
                let beta = &beta;
                move |(i, &x)| {
                    let n = ((f64::from(x) - mean) * inverse) as f32;
                    let scaled = n * gamma[i];
                    scaled + beta[i]
                }
            })
        })
        .collect();
    affine(
        &normalized,
        &floats(source, "decoder.proj_out.weight"),
        &floats(source, "decoder.proj_out.bias"),
    )
}
pub(in crate::p_h3_image_vae) struct Execution {
    pub(in crate::p_h3_image_vae) runtime: Arc<CudaDeviceRuntime>,
    pub(in crate::p_h3_image_vae) registry: OperationRuntimeRegistry<CudaDeviceRuntime>,
    pub(in crate::p_h3_image_vae) catalog: CapabilityCatalog,
    pub(in crate::p_h3_image_vae) materializers: WeightMaterializerRegistry,
}
impl Execution {
    pub(in crate::p_h3_image_vae) fn new() -> Self {
        let runtime = Arc::new(
            CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
                device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
                usable_memory_bytes: 1 << 30,
                bootstrap_memory_bytes: 256 << 20,
                stream_memory_bytes: 16 << 20,
                maximum_streams: 2,
                maximum_in_flight_per_stream: 1,
                maximum_commands_per_submission: 256,
                opaque_memory_profile: "unqualified-ph3-full-decoder-tiny-f32-v1".into(),
            })
            .unwrap(),
        );
        let mut providers = cuda_vnext_tensor_providers(
            &runtime,
            &[
                CudaTensorOperation::ComfyConvRotW8A8F32,
                CudaTensorOperation::HeadRmsNormF32,
                CudaTensorOperation::UnitHeadRmsPartialRopeF32,
                CudaTensorOperation::StructuredGqaAttentionF32,
                CudaTensorOperation::SiluMulF32,
                CudaTensorOperation::GroupedColumnSliceF32,
                CudaTensorOperation::ChannelBiasAddF32,
                CudaTensorOperation::ChannelScaledResidualF32,
                CudaTensorOperation::NanToNumF32,
                CudaTensorOperation::BatchedSuffixAppendF32,
                CudaTensorOperation::LayerNormF32,
            ],
        )
        .unwrap();
        providers.push(cuda_vnext_affine_linear_f32_provider(&runtime).unwrap());
        let registry = OperationRuntimeRegistry::new(
            vec![
                Box::new(comfy_convrot_w8a8_f32_contract().unwrap()),
                Box::new(head_rms_norm_f32_contract().unwrap()),
                Box::new(unit_head_rms_partial_rope_f32_contract().unwrap()),
                Box::new(structured_gqa_attention_f32_contract().unwrap()),
                Box::new(silu_mul_f32_contract().unwrap()),
                Box::new(grouped_column_slice_f32_contract().unwrap()),
                Box::new(channel_bias_add_f32_contract().unwrap()),
                Box::new(channel_scaled_residual_f32_contract().unwrap()),
                Box::new(nan_to_num_f32_contract().unwrap()),
                Box::new(batched_suffix_append_f32_contract().unwrap()),
                Box::new(layer_norm_f32_contract().unwrap()),
                Box::new(affine_linear_f32_contract().unwrap()),
            ],
            providers,
        )
        .unwrap();
        let materializers = WeightMaterializerRegistry::identity_only().unwrap();
        let d = runtime.descriptor();
        let engine = EngineProviderDescriptor::new(
            id("provider.engine.vae_decoder.full.fixture"),
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
        Self {
            runtime,
            registry,
            catalog,
            materializers,
        }
    }
    pub(in crate::p_h3_image_vae) fn run(
        &self,
        c: &PH3ImageVaeDecoderExecutionConfig,
        source: &ComfyCanonicalWeightSource,
        values: &Values,
    ) -> Vec<f32> {
        let family = Family::new(limits()).unwrap();
        let options = family.compile_options(c).unwrap();
        let prepared = TypedFamilyRegistration::new(family)
            .prepare_with_profile(
                &serde_json::to_value(c).unwrap(),
                &id(Family::CUDA_PROFILE_ID),
            )
            .unwrap();
        let d = self.runtime.descriptor();
        let compiled = CompiledTensorComponent::compile(
            prepared,
            &self.registry,
            &self.catalog,
            &TensorComponentPolicy::new(
                d.total_memory_bytes,
                d.dynamic_storage_profiles.iter().copied().collect(),
            )
            .unwrap(),
            &self.materializers,
            &options,
        )
        .unwrap();
        let host = host();
        let checkpoint =
            PH3ImageVaeDecoderCheckpoint::new(c, source, host.clone(), limits()).unwrap();
        let mut session = compiled
            .initialize(
                self.runtime.clone(),
                host.clone(),
                ComponentWeightSource::Stored(&checkpoint),
                ComponentInitializationConfig {
                    request_id: id("request.vae_full.init"),
                    run_id: id("run.vae_full.init"),
                    transaction_id: id("transaction.vae_full.init"),
                    static_policy: StaticInitializationPolicy::new(32 << 20, 1024).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap_or_else(|f| panic!("full decoder init: {f:?}"));
        let managed = c
            .input_tensor_specs()
            .unwrap()
            .into_iter()
            .map(|(name, spec)| {
                let bytes = f32_bytes(values[&name].iter().copied());
                let mut allocation = host
                    .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes.len())
                    .unwrap();
                allocation.as_mut().copy_from_slice(&bytes);
                ManagedHostTensor::new(
                    name,
                    ResolvedTensorSpec::new(spec.dimensions, spec.element_type, spec.layout)
                        .unwrap(),
                    allocation.freeze(),
                )
                .unwrap()
            })
            .collect();
        let outputs = session
            .execute(
                id("run.vae_full.execute"),
                id("request.vae_full.execute"),
                managed,
            )
            .unwrap_or_else(|f| panic!("full decoder execute: {f:?}"));
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].value_id().as_str(), Family::OUTPUT_ID);
        assert_eq!(
            outputs[0].tensor().dimensions(),
            [
                c.rows().unwrap(),
                c.architecture().patch_output_channels().unwrap()
            ]
        );
        let actual = outputs[0]
            .bytes()
            .bytes()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(matches!(
            session
                .close()
                .unwrap_or_else(|f| panic!("full decoder close: {:?}", f.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        assert!(host.owns(outputs[0].bytes()));
        drop(outputs);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        actual
    }
}
#[test]
#[ignore = "requires configured CUDA host and pinned native operator-set lock"]
fn actual_stored_full_decoder_two_layers_matches_oracle_b1_and_isolates_requests() {
    let _guard = GPU.lock().unwrap_or_else(|p| p.into_inner());
    let execution = Execution::new();
    let c = config(2, 2);
    let (_dir, source) = archive(&originals(&c));
    let values = inputs(&c, false);
    let actual = execution.run(&c, &source, &values);
    let wanted = oracle(&c, &source, &values);
    // Compound graph tolerance is declared before execution. The independent
    // oracle uses F64 statistics/attention and exact canonical A8 projections.
    for (i, (&got, &expected)) in actual.iter().zip(&wanted).enumerate() {
        let bound = 1e-4 * (1. + expected.abs());
        assert!(
            got.is_finite() && (got - expected).abs() <= bound,
            "decoder element {i}: actual={got} oracle={expected} bound={bound}"
        );
    }
    let per = c.sequence_length().unwrap() as usize
        * c.architecture().patch_output_channels().unwrap() as usize;
    for sample in 0..2 {
        let one = config(1, 2);
        let mut selected = Values::new();
        for (name, tensor) in c.input_tensor_specs().unwrap() {
            let elements = tensor.dimensions.iter().product::<u64>() as usize / 2;
            selected.insert(
                name.clone(),
                values[&name][sample * elements..(sample + 1) * elements].to_vec(),
            );
        }
        assert_eq!(
            execution.run(&one, &source, &selected),
            actual[sample * per..(sample + 1) * per]
        );
    }
    let changed = execution.run(&c, &source, &inputs(&c, true));
    assert_eq!(&changed[..per], &actual[..per]);
    assert!(changed[per..]
        .iter()
        .zip(&actual[per..])
        .any(|(a, b)| a != b));
}
