use super::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutionOutput, TensorExecutorConfig,
};
use ferrum_kernels::{
    backend::cuda::{
        vnext_ops::{
            cuda_tensor_runtime_config, cuda_tensor_runtime_enable_sage_vsa_fp4,
            cuda_tensor_runtime_enable_sage_vsa_fp4_quantize,
            cuda_vnext_sparse_fp4_attention_checked_host_provider,
            cuda_vnext_sparse_fp4_qkv_quantize_bf16_provider,
        },
        vnext_runtime::{
            CudaDeviceRuntime, CudaPinnedWeightUploadConfig, CudaTensorRuntimeConfig,
            CudaTensorRuntimeOptions,
        },
    },
    native_ops::sage_vsa_fp4::{CheckedHostVsaPattern, ListShape, SageVsaFp4Config},
};
use std::{num::NonZeroUsize, sync::Arc};

fn host() -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            32 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 16 << 20,
                transfer_staging_bytes: 8 << 20,
                output_bytes: 8 << 20,
                weight_backing_bytes: 8 << 20,
            },
        )
        .unwrap(),
        CudaDeviceRuntime::pinned_weight_upload_alignment().unwrap(),
    )
    .unwrap()
}
fn list(host: &HostMemoryAccount, values: &[i32]) -> ManagedHostBytes {
    let mut owner = host
        .allocate_zeroed(HostMemoryClass::TransferStaging, values.len() * 4)
        .unwrap();
    for (dst, v) in owner.as_mut().chunks_exact_mut(4).zip(values) {
        dst.copy_from_slice(&v.to_le_bytes());
    }
    owner.freeze()
}
#[test]
#[ignore = "requires SM120 and the locked nine-export Sage artifact; exact-domain six-layout/ownership qualification"]
fn sage_qkv_quantize_cuda_matches_independent_six_layout_oracle_and_shares_module() {
    let mut device = cuda_tensor_runtime_config(0, id("device.cuda.0").unwrap()).unwrap();
    cuda_tensor_runtime_enable_sage_vsa_fp4(&mut device).unwrap();
    cuda_tensor_runtime_enable_sage_vsa_fp4_quantize(&mut device).unwrap();
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager_with_options(
            CudaTensorRuntimeConfig {
                device,
                usable_memory_bytes: 1 << 30,
                bootstrap_memory_bytes: 256 << 20,
                stream_memory_bytes: 16 << 20,
                maximum_streams: 2,
                maximum_in_flight_per_stream: 1,
                maximum_commands_per_submission: 64,
                opaque_memory_profile: "sage-quantize-qualification".into(),
            },
            CudaTensorRuntimeOptions {
                pinned_weight_upload: Some(CudaPinnedWeightUploadConfig {
                    minimum_allocation_bytes: 0,
                    maximum_allocation_bytes: NonZeroUsize::new(8 << 20).unwrap(),
                    maximum_registered_bytes: NonZeroUsize::new(16 << 20).unwrap(),
                }),
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let account = host();
    let pattern = Arc::new(
        CheckedHostVsaPattern::new(
            ListShape::new(1, 1, 1, 1).unwrap(),
            list(&account, &[0]),
            list(&account, &[1]),
            list(&account, &[64, 64]),
        )
        .unwrap(),
    );
    let pattern_bytes = account.usage().unwrap().allocated_bytes();
    let native = SageVsaFp4Config {
        opaque_memory_bytes: 64 << 20,
        opaque_memory_profile: "sage-shared-quantize-attention-qualification".into(),
    };
    let quant = cuda_vnext_sparse_fp4_qkv_quantize_bf16_provider(&runtime, &native).unwrap();
    // Both actual factories acquire the same finite native-module slot while
    // the other owner is live. A per-operation admission identity would fail.
    let attention =
        cuda_vnext_sparse_fp4_attention_checked_host_provider(&runtime, &native, pattern.clone())
            .unwrap();
    let registry = OperationRuntimeRegistry::new(
        vec![
            Box::new(sparse_fp4_qkv_quantize_bf16_contract().unwrap()),
            Box::new(sparse_fp4_attention_checked_host_contract().unwrap()),
        ],
        vec![quant, attention],
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.sage_quantize").unwrap(),
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
    for c in [config(1, 2, 128, 256), config(2, 3, 256, 384)] {
        let prepared =
            TypedFamilyRegistration::new(FastH3VsaQkvQuantizeFamily::new(limits()).unwrap())
                .prepare_with_profile(
                    &serde_json::to_value(&c).unwrap(),
                    &NumericalProfileId::new(FastH3VsaQkvQuantizeFamily::PROFILE_ID).unwrap(),
                )
                .unwrap();
        let compiled = Arc::new(
            CompiledTensorComponent::compile(
                prepared,
                &registry,
                &catalog,
                &TensorComponentPolicy::new(
                    d.total_memory_bytes,
                    d.dynamic_storage_profiles.iter().copied().collect(),
                )
                .unwrap(),
                &materializers,
                &c.compile_options(limits()).unwrap(),
            )
            .unwrap(),
        );
        let owners = (compiled.clone(), runtime.clone(), pattern.clone());
        let mut s = compiled
            .initialize_retained(
                runtime.clone(),
                account.clone(),
                ComponentWeightSource::None,
                ComponentInitializationConfig {
                    run_id: id("run.sage_quantize.init").unwrap(),
                    request_id: id("request.sage_quantize.init").unwrap(),
                    transaction_id: id("transaction.sage_quantize.init").unwrap(),
                    static_policy: StaticInitializationPolicy::new(1024, 8).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
                owners,
            )
            .unwrap_or_else(|e| {
                std::mem::forget(e);
                panic!("quantize initialization owners retained")
            });
        let mut retained = None;
        let mut expected_last = None;
        for trial in 0..2 {
            let input = operands(&c, trial);
            let expected = oracle(&c, &input);
            let managed = FastH3VsaQkvQuantizeFamily::INPUT_IDS
                .into_iter()
                .zip(c.input_shapes().unwrap())
                .zip(&input)
                .map(|((name, shape), values)| {
                    let mut bytes = account
                        .allocate_zeroed(HostMemoryClass::InputPreprocess, values.len() * 2)
                        .unwrap();
                    for (dst, v) in bytes.as_mut().chunks_exact_mut(2).zip(values) {
                        dst.copy_from_slice(&v.to_le_bytes());
                    }
                    ManagedHostTensor::new(
                        id(name).unwrap(),
                        resolved(shape, ElementType::Bf16).unwrap(),
                        bytes.freeze(),
                    )
                    .unwrap()
                })
                .collect();
            let outputs = match s.session_mut().execute(
                id("run.sage_quantize.execute").unwrap(),
                id("request.sage_quantize.execute").unwrap(),
                managed,
            ) {
                Ok(v) => v,
                Err(e) => {
                    std::mem::forget((s, e));
                    panic!("quantize execution owners retained")
                }
            };
            verify(&c, &outputs, &expected);
            retained = Some(outputs);
            expected_last = Some(expected);
        }
        let (_, receipt) = s.close().unwrap_or_else(|e| {
            std::mem::forget(e);
            panic!("quantize close owners retained")
        });
        drop(receipt);
        verify(
            &c,
            retained.as_ref().unwrap(),
            expected_last.as_ref().unwrap(),
        );
        let output_bytes: usize = expected_last
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| account.backing_bytes(v.len()).unwrap())
            .sum();
        assert_eq!(
            account.usage().unwrap().allocated_bytes(),
            pattern_bytes + output_bytes
        );
        drop(retained);
        assert_eq!(account.usage().unwrap().allocated_bytes(), pattern_bytes);
    }
    drop(registry);
    drop(pattern);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
}
fn verify(
    c: &FastH3VsaQkvQuantizeConfig,
    outputs: &[TensorExecutionOutput],
    expected: &[Vec<u8>; 6],
) {
    assert_eq!(outputs.len(), 6);
    for ((name, shape), wanted) in FastH3VsaQkvQuantizeFamily::OUTPUT_IDS
        .into_iter()
        .zip(c.output_shapes())
        .zip(expected)
    {
        let actual = outputs
            .iter()
            .find(|o| o.value_id().as_str() == name)
            .unwrap();
        assert_eq!(actual.tensor().dimensions(), shape);
        assert_eq!(actual.tensor().element_type(), ElementType::U8);
        let bytes = actual.bytes().bytes();
        assert_eq!(bytes.len(), wanted.len());
        if let Some(i) = bytes.iter().zip(wanted).position(|(a, b)| a != b) {
            panic!("{name}: byte {i} is {}, expected {}", bytes[i], wanted[i]);
        }
    }
}
