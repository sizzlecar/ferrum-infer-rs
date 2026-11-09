//! Actual Stored canonical weights + F32 decoder operators. These tiny tests
//! qualify component arithmetic and batching, not an image service, throughput
//! or bitwise parity with a particular Torch binary.
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
mod visual;
mod weight_source;
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
                opaque_memory_profile: "unqualified-qwen3vl-comfy-f32-component-v1".into(),
            })
            .unwrap(),
        );
        let registry = OperationRuntimeRegistry::new(
            vec![
                Box::new(comfy_convrot_w4a8_f32_contract().unwrap()),
                Box::new(comfy_convrot_w8a8_f32_contract().unwrap()),
                Box::new(head_rms_norm_f32_contract().unwrap()),
                Box::new(half_split_rope_f32_contract().unwrap()),
                Box::new(structured_gqa_attention_f32_contract().unwrap()),
                Box::new(plain_residual_add_f32_contract().unwrap()),
                Box::new(silu_mul_f32_contract().unwrap()),
                Box::new(row_scatter_replace_f32_contract().unwrap()),
                Box::new(row_scatter_add_f32_contract().unwrap()),
            ],
            cuda_vnext_tensor_providers(
                &runtime,
                &[
                    CudaTensorOperation::ComfyConvRotW4A8F32,
                    CudaTensorOperation::ComfyConvRotW8A8F32,
                    CudaTensorOperation::HeadRmsNormF32,
                    CudaTensorOperation::HalfSplitRopeF32,
                    CudaTensorOperation::StructuredGqaAttentionF32,
                    CudaTensorOperation::PlainResidualAddF32,
                    CudaTensorOperation::SiluMulF32,
                    CudaTensorOperation::RowScatterReplaceF32,
                    CudaTensorOperation::RowScatterAddF32,
                ],
            )
            .unwrap(),
        )
        .unwrap();
        let materializers = WeightMaterializerRegistry::identity_only().unwrap();
        let d = runtime.descriptor();
        let engine = EngineProviderDescriptor::new(
            id("provider.engine.fixture.qwen3vl_comfy"),
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
    fn execute(
        &self,
        c: &Qwen3VLTextComfyConfig,
        source: &ComfyCanonicalWeightSource,
        values: &BTreeMap<ProgramValueId, Vec<f32>>,
        label: &str,
    ) -> BTreeMap<ProgramValueId, Vec<f32>> {
        let family = TypedFamilyRegistration::new(Family::new(limits()).unwrap())
            .prepare_with_profile(
                &serde_json::to_value(c).unwrap(),
                &id(Family::cuda_profile_id(c)),
            )
            .unwrap();
        let identity = ExecutionWeightPlan::identity(&family).unwrap();
        let d = self.runtime.descriptor();
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
            &c.compile_options().unwrap(),
        )
        .unwrap();
        let host = HostMemoryAccount::new(
            HostMemoryLimits::new(
                32 << 20,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 4 << 20,
                    output_bytes: 4 << 20,
                    transfer_staging_bytes: 8 << 20,
                    weight_backing_bytes: 16 << 20,
                },
            )
            .unwrap(),
        );
        let checkpoint = Qwen3VLTextComfyCheckpoint::new(c, source, limits()).unwrap();
        let weight_source = weight_source::Source::new(&checkpoint, identity.schema(), &host);
        let mut session = compiled
            .initialize(
                self.runtime.clone(),
                host.clone(),
                ComponentWeightSource::Stored(&weight_source),
                ComponentInitializationConfig {
                    request_id: id(format!("request.comfy_te.{label}.init")),
                    run_id: id(format!("run.comfy_te.{label}.init")),
                    transaction_id: id(format!("transaction.comfy_te.{label}.init")),
                    static_policy: StaticInitializationPolicy::new(8 << 20, 512).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap_or_else(|e| panic!("Stored {label} initialization failed: {e:?}"));
        let inputs = c
            .input_tensor_specs()
            .unwrap()
            .into_iter()
            .map(|(id, spec)| {
                let bytes = floats(values[&id].iter().copied());
                let mut allocation = host
                    .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes.len())
                    .unwrap();
                allocation.as_mut().copy_from_slice(&bytes);
                ManagedHostTensor::new(
                    id,
                    ResolvedTensorSpec::new(spec.dimensions, spec.element_type, spec.layout)
                        .unwrap(),
                    allocation.freeze(),
                )
                .unwrap()
            })
            .collect();
        let outputs = session
            .execute(
                id(format!("run.comfy_te.{label}")),
                id(format!("request.comfy_te.{label}")),
                inputs,
            )
            .unwrap_or_else(|e| panic!("Stored {label} execution failed: {e:?}"));
        let actual = outputs
            .iter()
            .map(|output| {
                assert_eq!(output.tensor().element_type(), ElementType::F32);
                assert_eq!(
                    output.tensor().dimensions(),
                    [
                        c.text().layout().rows(),
                        c.text().architecture().hidden_size()
                    ]
                );
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
        drop(outputs);
        assert!(matches!(
            session
                .close()
                .unwrap_or_else(|f| panic!("close {label}: {:?}", f.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        assert_eq!(
            host.usage().unwrap().committed_bytes(),
            0,
            "Stored {label} leaked Host ownership"
        );
        actual
    }
}
fn qualify(layers: u64) {
    let execution = Execution::new();
    // Format choices are swapped at the same layer/projection identities. No
    // exclusion list can pass both cases while choosing from a layer number.
    for swapped in [false, true] {
        let c = config(text(layers, 2), swapped);
        let (_directory, source) = archive(&original(&c));
        let raw = inputs(&c);
        let expected = oracle::execute(&c, &source, &raw);
        let actual = execution.execute(&c, &source, &raw, &format!("l{layers}.b2.s{swapped}"));
        let mut max_error = 0.0_f32;
        for (port, values) in &expected {
            let got = &actual[port];
            assert_eq!(got.len(), values.len());
            for (&got, &expected) in got.iter().zip(values) {
                assert!(got.is_finite());
                let error = (got - expected).abs();
                max_error = max_error.max(error);
                // Predeclared F64-attention/scalar-normalization oracle bound.
                // A8 native output remains F32; no BF16-sized tolerance is used.
                assert!(
                    error <= 3.0e-5 * expected.abs().max(1.0),
                    "{port}: native {got} scalar {expected}, absolute error {error}"
                );
            }
        }
        let h = c.text().architecture().hidden_size() as usize;
        let s = c.text().layout().sequence_length() as usize;
        for sample in 0..2 {
            let layout = Qwen3VLTextLayout::new(
                s as u32,
                vec![c.text().layout().key_runs()[sample].clone()],
            )
            .unwrap();
            let one_text = Qwen3VLTextConfig::new(c.text().architecture().clone(), layout)
                .unwrap()
                .with_hidden_state_indices(c.text().hidden_state_indices().to_vec())
                .unwrap();
            let one = Qwen3VLTextComfyConfig::from_checkpoint(one_text, &source).unwrap();
            let single_inputs = raw
                .iter()
                .map(|(id, data)| {
                    let width = if id.as_str() == Family::EMBEDDINGS_ID {
                        h
                    } else {
                        c.text().architecture().head_dim() as usize
                    };
                    (
                        id.clone(),
                        data[sample * s * width..(sample + 1) * s * width].to_vec(),
                    )
                })
                .collect();
            let single = execution.execute(
                &one,
                &source,
                &single_inputs,
                &format!("l{layers}.b1.{sample}.s{swapped}"),
            );
            for (port, data) in single {
                let batch_row = &actual[&port][sample * s * h..(sample + 1) * s * h];
                assert_eq!(
                    data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    batch_row.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "{port}: batch membership changed sample {sample}"
                );
            }
        }
        eprintln!("Comfy F32 TE Stored layers={layers} batch=2 swapped={swapped} scalar_max_abs_error={max_error:e}; B1/B2 bit-exact");
    }
}
#[test]
#[ignore = "requires configured CUDA host and pinned native operator-set lock"]
fn actual_stored_single_decoder_matches_scalar_and_independent_b1_requests() {
    let _guard = GPU.lock().unwrap_or_else(|p| p.into_inner());
    qualify(1);
}
#[test]
#[ignore = "requires configured CUDA host and pinned native operator-set lock"]
fn actual_stored_decoder_stack_keeps_native_f32_taps_and_batch_independence() {
    let _guard = GPU.lock().unwrap_or_else(|p| p.into_inner());
    qualify(2);
}
