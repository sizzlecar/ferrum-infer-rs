//! Real Stored W8 + originally BF16 vectors, under Ferrum's explicit F32
//! full-attention profile. The test wrapper exposes existing graph boundaries;
//! production outputs remain unchanged. This does not qualify upstream's
//! optional INT8/BF16 attention or its default FP16 VAE route.
use super::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_tensor_runtime_config, cuda_vnext_tensor_providers, CudaTensorOperation},
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use std::sync::{Arc, Mutex};
static GPU: Mutex<()> = Mutex::new(());
type Values = BTreeMap<ProgramValueId, Vec<f32>>;

struct VerificationFamily {
    inner: Family,
    capture: bool,
}
impl ModelFamilyProvider for VerificationFamily {
    type Config = PH3ImageVaeDecoderBlockConfig;
    fn family_id(&self) -> &ModelFamilyId {
        self.inner.family_id()
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        self.inner.external_metadata_ids()
    }
    fn validate_config_identity(
        &self,
        r: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<(), VNextError> {
        self.inner.validate_config_identity(r, c)
    }
    fn validated_external_metadata_id(
        &self,
        r: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.inner.validated_external_metadata_id(r, c)
    }
    fn parse_config(&self, r: &serde_json::Value) -> Result<Self::Config, VNextError> {
        self.inner.parse_config(r)
    }
    fn weights(&self, c: &Self::Config) -> Result<ModelWeights, VNextError> {
        self.inner.weights(c)
    }
    fn numerical_profiles(&self, c: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        self.inner.numerical_profiles(c)
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let base = self.inner.semantic_program(c, p)?;
        if !self.capture {
            return Ok(base);
        }
        let outputs = base
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .flat_map(|n| n.outputs.clone())
            .collect();
        ModelProgram::new(
            base.family_id().clone(),
            base.inputs().to_vec(),
            base.blocks().to_vec(),
            base.states().to_vec(),
            base.weights().to_vec(),
            outputs,
        )?
        .with_input_tensor_specs(base.input_tensor_specs().clone())
    }
    fn semantic_metadata(&self, c: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        self.inner.semantic_metadata(c)
    }
}
struct Execution {
    runtime: Arc<CudaDeviceRuntime>,
    registry: OperationRuntimeRegistry<CudaDeviceRuntime>,
    catalog: CapabilityCatalog,
    materializers: WeightMaterializerRegistry,
}
impl Execution {
    fn new() -> Self {
        let runtime = Arc::new(
            CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
                device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
                usable_memory_bytes: 1 << 30,
                bootstrap_memory_bytes: 256 << 20,
                stream_memory_bytes: 16 << 20,
                maximum_streams: 2,
                maximum_in_flight_per_stream: 1,
                maximum_commands_per_submission: 256,
                opaque_memory_profile: "unqualified-ph3-vae-decoder-block-f32-v1".into(),
            })
            .unwrap(),
        );
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
            ],
            cuda_vnext_tensor_providers(
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
                ],
            )
            .unwrap(),
        )
        .unwrap();
        let materializers = WeightMaterializerRegistry::identity_only().unwrap();
        let d = runtime.descriptor();
        let engine = EngineProviderDescriptor::new(
            id("provider.engine.fixture.vae_decoder_f32"),
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
    fn run(
        &self,
        c: &PH3ImageVaeDecoderBlockConfig,
        source: &ComfyCanonicalWeightSource,
        values: &Values,
        label: &str,
        capture: bool,
    ) -> Values {
        let family = TypedFamilyRegistration::new(VerificationFamily {
            inner: Family::new(limits()).unwrap(),
            capture,
        })
        .prepare_with_profile(
            &serde_json::to_value(c).unwrap(),
            &id(Family::CUDA_F32_PROFILE_ID),
        )
        .unwrap();
        let d = self.runtime.descriptor();
        let options = c.compile_options().unwrap();
        let compiled = CompiledTensorComponent::compile(
            family,
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
        let host = HostMemoryAccount::new(
            HostMemoryLimits::new(
                64 << 20,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 8 << 20,
                    transfer_staging_bytes: 16 << 20,
                    output_bytes: 24 << 20,
                    weight_backing_bytes: 16 << 20,
                },
            )
            .unwrap(),
        );
        let checkpoint = PH3ImageVaeDecoderBlockCheckpoint::new(c, source, limits()).unwrap();
        let mut session = compiled
            .initialize(
                self.runtime.clone(),
                host.clone(),
                ComponentWeightSource::Stored(&checkpoint),
                ComponentInitializationConfig {
                    request_id: id(format!("request.vae_decoder.{label}.init")),
                    run_id: id(format!("run.vae_decoder.{label}.init")),
                    transaction_id: id(format!("transaction.vae_decoder.{label}.init")),
                    static_policy: StaticInitializationPolicy::new(8 << 20, 512).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap_or_else(|e| panic!("Stored decoder {label} init failed: {e:?}"));
        let inputs = c
            .input_tensor_specs()
            .unwrap()
            .into_iter()
            .map(|(name, spec)| {
                let data = f32_bytes(values[&name].iter().copied());
                let mut allocation = host
                    .allocate_zeroed(HostMemoryClass::InputPreprocess, data.len())
                    .unwrap();
                allocation.as_mut().copy_from_slice(&data);
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
                id(format!("run.vae_decoder.{label}")),
                id(format!("request.vae_decoder.{label}")),
                inputs,
            )
            .unwrap_or_else(|e| panic!("Stored decoder {label} execution failed: {e:?}"));
        let actual: Values = outputs
            .iter()
            .map(|output| {
                let spec = &options.tensor_specs()[output.value_id()];
                assert_eq!(output.tensor().dimensions(), spec.dimensions);
                assert_eq!(output.tensor().element_type(), ElementType::F32);
                assert!(host.owns(output.bytes()));
                (
                    output.value_id().clone(),
                    output
                        .bytes()
                        .bytes()
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect(),
                )
            })
            .collect();
        assert!(matches!(
            session
                .close()
                .unwrap_or_else(|f| panic!("decoder {label} close: {:?}", f.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        // Host readback owners survive closing every device/session owner.
        assert!(outputs.iter().all(|o| host.owns(o.bytes())));
        assert!(
            outputs
                .iter()
                .all(|o| o.bytes().bytes().len() as u64
                    == o.tensor().minimum_storage_bytes().unwrap())
        );
        drop(outputs);
        assert_eq!(
            host.usage().unwrap().committed_bytes(),
            0,
            "decoder {label} leaked Host ownership"
        );
        actual
    }
}
fn fixture_values(c: &PH3ImageVaeDecoderBlockConfig) -> Values {
    let mut values = input_values(c);
    let h = c.architecture().hidden_size().unwrap() as usize;
    let s = c.sequence_length() as usize;
    let rot = c.architecture().rotary_dim() as usize;
    assert_eq!(s, 2 + c.architecture().register_tokens() as usize + 1);
    let x = values
        .get_mut(&id::<ProgramValueId>(Family::INPUT_ID))
        .unwrap();
    for sample in 0..c.batch() as usize {
        for register in 0..4 {
            for column in 0..h {
                x[(sample * s + 2 + register) * h + column] =
                    bf16::from_f32(register as f32 / 8.0 + (column % 13) as f32 / 64.0 - 0.25)
                        .to_f32();
            }
        }
        x[(sample * s + s - 1) * h..(sample * s + s) * h].fill(0.0);
    }
    // Fixture-supplied F32 frequency tables: two patches at H=1,W=2,
    // plus the identity rotary payload of the four registers and zero CLS.
    // This tests the block's consumption, not a production position adapter.
    let mut cosine = vec![1.0; (c.rows().unwrap() as usize) * rot];
    let mut sine = vec![0.0; cosine.len()];
    let pairs = rot / 2;
    let per_axis = pairs / 3;
    let two_pi = 2.0_f32 * std::f32::consts::PI;
    for sample in 0..c.batch() as usize {
        for patch in 0..2 {
            let coordinates = [-6.0_f32 / 7.0, 0.0, if patch == 0 { -0.5 } else { 0.5 }];
            for axis in 0..3 {
                for frequency in 0..per_axis {
                    let inverse = 1.0_f32 / libm::powf(100.0, frequency as f32 / per_axis as f32);
                    let phase = (two_pi * coordinates[axis]) * inverse;
                    let pair = axis * per_axis + frequency;
                    let row = sample * s + patch;
                    for column in [pair, pair + pairs] {
                        cosine[row * rot + column] = libm::cosf(phase);
                        sine[row * rot + column] = libm::sinf(phase);
                    }
                }
            }
        }
    }
    values.insert(id(Family::COSINE_ID), cosine);
    values.insert(id(Family::SINE_ID), sine);
    values
}
fn assert_exact(a: &[f32], b: &[f32], name: &str) {
    assert_eq!(
        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "{name}"
    );
}
#[test]
#[ignore = "requires configured CUDA host and pinned native operator-set lock"]
fn actual_stored_decoder_block_f32_matches_oracle_b1_and_request_isolation() {
    let _guard = GPU.lock().unwrap_or_else(|p| p.into_inner());
    let execution = Execution::new();
    let c =
        PH3ImageVaeDecoderBlockConfig::new(architecture(4), 0, 2, 7, Preset::F32A8FullAttention)
            .unwrap();
    let (_directory, source) = archive(&original(&c));
    let values = fixture_values(&c);
    let expected = oracle::capture(&c, &source, &values);
    let actual = execution.run(&c, &source, &values, "b2.diagnostic", true);
    let product = execution.run(&c, &source, &values, "b2.production", false);
    assert_exact(
        &actual[&id::<ProgramValueId>(Family::OUTPUT_ID)],
        &product[&id::<ProgramValueId>(Family::OUTPUT_ID)],
        "diagnostic retention changed production output",
    );
    assert_eq!(
        actual.keys().collect::<Vec<_>>(),
        expected.keys().collect::<Vec<_>>()
    );
    for (port, reference) in &expected {
        let got = &actual[port];
        assert_eq!(got.len(), reference.len());
        let mut max_error = 0.0_f32;
        for (&got, &reference) in got.iter().zip(reference) {
            assert!(got.is_finite());
            let error = (got - reference).abs();
            max_error = max_error.max(error);
            // Predeclared independent F64 norm/attention bound. Quantized dot
            // products remain the canonical exact-I32 A8 scalar oracle.
            assert!(
                error <= 3.0e-5 * reference.abs().max(1.0),
                "{port}: native {got}, scalar {reference}, absolute error {error}"
            );
        }
        eprintln!("VAE F32 FullAttention boundary={port} rows=14 columns={} scalar_max_abs_error={max_error:e}",got.len()/14);
    }
    assert!(actual[&id::<ProgramValueId>(Family::OUTPUT_ID)]
        .iter()
        .any(|v| bf16::from_f32(*v).to_f32() != *v));
    for sample in 0..2 {
        let one = PH3ImageVaeDecoderBlockConfig::new(
            c.architecture().clone(),
            c.block_index(),
            1,
            c.sequence_length(),
            c.execution_preset(),
        )
        .unwrap();
        let single_inputs = values
            .iter()
            .map(|(id, data)| {
                let per_sample = data.len() / 2;
                (
                    id.clone(),
                    data[sample * per_sample..(sample + 1) * per_sample].to_vec(),
                )
            })
            .collect();
        let single = execution.run(
            &one,
            &source,
            &single_inputs,
            &format!("b1.sample{sample}"),
            true,
        );
        for (port, data) in single {
            let per_sample = actual[&port].len() / 2;
            assert_exact(
                &data,
                &actual[&port][sample * per_sample..(sample + 1) * per_sample],
                port.as_str(),
            );
        }
    }
    let mut changed = values.clone();
    let h = c.architecture().hidden_size().unwrap() as usize;
    let x = changed
        .get_mut(&id::<ProgramValueId>(Family::INPUT_ID))
        .unwrap();
    for value in &mut x[7 * h..9 * h] {
        *value = -0.75 * (*value) + 0.3125;
    }
    let perturbed = execution.run(&c, &source, &changed, "b2.second_patch_perturbed", true);
    for (port, data) in &actual {
        let per_sample = data.len() / 2;
        assert_exact(
            &data[..per_sample],
            &perturbed[port][..per_sample],
            port.as_str(),
        );
    }
    let output = id::<ProgramValueId>(Family::OUTPUT_ID);
    let original_cls = &actual[&output][13 * h..14 * h];
    let changed_cls = &perturbed[&output][13 * h..14 * h];
    assert!(
        original_cls
            .iter()
            .zip(changed_cls)
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "full bidirectional attention did not carry changed patch information into its own CLS row"
    );
}
