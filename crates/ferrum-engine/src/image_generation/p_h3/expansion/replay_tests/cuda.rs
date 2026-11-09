//! Actual bounded CUDA causal admission, source installation and decode replay.
//! This tiny stateful model is a correctness fixture, not a performance claim.
use super::nonzero::{run, NonzeroFixture};
use super::*;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_tensor_and_dense_f16_visual_causal_runtime_config,
        cuda_vnext_dense_f16_visual_causal_contracts, cuda_vnext_dense_f16_visual_causal_providers,
    },
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
    CudaDenseF16CausalProfile, CudaTensorAndCausalRuntimeConfig,
};
use ferrum_runtime::{
    CapabilityId, ContractVersion, DeviceId, EngineProviderDescriptor, OperationRuntimeRegistry,
    ProviderId, WeightMaterializerRegistry, DEVICE_ON_DEMAND_REUSABLE_EXECUTION_CAPABILITY_ID,
    DEVICE_REUSABLE_EXECUTION_CAPABILITY_ID,
};

fn composition(
    replay: bool,
) -> (
    RuntimeComposition<CudaDeviceRuntime>,
    Arc<CudaDeviceRuntime>,
) {
    // Real products advertise visual capability even for a text-only request.
    let mut device = cuda_tensor_and_dense_f16_visual_causal_runtime_config(
        0,
        DeviceId::new("device.cuda.0").unwrap(),
    )
    .unwrap();
    let mut causal =
        CudaDenseF16CausalProfile::new(64 << 20, "unqualified-nonzero-replay-modules-v1")
            .unwrap()
            .with_visual_prompt_v1();
    if replay {
        causal = causal.with_decode_replay(1024, 64 << 20).unwrap();
        for capability in [
            DEVICE_REUSABLE_EXECUTION_CAPABILITY_ID,
            DEVICE_ON_DEMAND_REUSABLE_EXECUTION_CAPABILITY_ID,
        ] {
            device
                .capabilities
                .insert(CapabilityId::new(capability).unwrap());
        }
    }
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_and_causal_eager(CudaTensorAndCausalRuntimeConfig {
            tensor: CudaTensorRuntimeConfig {
                device,
                usable_memory_bytes: 1 << 30,
                bootstrap_memory_bytes: 128 << 20,
                stream_memory_bytes: 16 << 20,
                maximum_streams: 3,
                maximum_in_flight_per_stream: 1,
                maximum_commands_per_submission: 1024,
                opaque_memory_profile: "unqualified-nonzero-replay-runtime-v1".into(),
            },
            causal,
        })
        .expect("actual bounded CUDA runtime is required"),
    );
    let registry = OperationRuntimeRegistry::new(
        cuda_vnext_dense_f16_visual_causal_contracts().unwrap(),
        cuda_vnext_dense_f16_visual_causal_providers(&runtime).unwrap(),
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let descriptor = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        ProviderId::new("provider.engine.cuda.nonzero-replay-fixture").unwrap(),
        ContractVersion::new(1, 0),
        descriptor.runtime_implementation_fingerprint.clone(),
        descriptor.id.clone(),
        descriptor.capabilities.clone(),
    )
    .unwrap();
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(descriptor.clone(), vec![engine])
                .unwrap(),
        )
        .unwrap();
    ferrum_kernels::native_ops::validate_compiled_native_operator_provider_catalog_for_units(
        &catalog
            .native_operator_provider_catalog(ferrum_types::NativeOperatorBackend::Cuda)
            .unwrap(),
        ferrum_kernels::native_ops::compiled_native_operator_artifacts(),
        &[],
    )
    .unwrap();
    (
        RuntimeComposition::new(Arc::clone(&runtime), registry, materializers, catalog),
        runtime,
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires exclusive actual CUDA and pinned native operators; tiny stateful decode correctness, not official-model qualification"]
async fn cuda_bounded_decode_replay_matches_nonzero_eager_state_and_retires_original_owners() {
    let fixture = NonzeroFixture::new();
    let mut arms = Vec::new();
    for replay in [false, true] {
        let (composition, runtime) = composition(replay);
        let original = Arc::downgrade(&runtime);
        let mut results = Vec::new();
        for prompt in [[0, 0, 0], [1, 1, 1]] {
            let (tokens, metrics) = run(
                &fixture,
                &composition,
                ferrum_types::Device::CUDA(0),
                replay,
                &prompt,
            )
            .await;
            let direct = metrics["counters"]["reusable_execution"]["direct_waves"]
                .as_u64()
                .unwrap();
            if replay {
                assert!(direct > 0, "actual graph replay required: {metrics}");
                assert!(
                    metrics["counters"]["reusable_execution"]["catalog"]
                        ["programs_with_resident_segments"]
                        .as_u64()
                        .unwrap()
                        > 0
                );
            } else {
                assert_eq!(direct, 0);
            }
            eprintln!(
                "{}",
                serde_json::json!({"diagnostic":"bounded_causal_decode", "replay":replay, "prompt":prompt, "tokens":tokens, "metrics":metrics})
            );
            results.push(tokens);
        }
        drop(composition);
        drop(runtime);
        assert!(
            original.upgrade().is_none(),
            "original runtime retained after certified shutdown"
        );
        arms.push(results);
    }
    assert_eq!(
        arms[0], arms[1],
        "every stateful greedy result must match eager"
    );
    assert_ne!(
        &arms[0][0][1..],
        &arms[0][1][1..],
        "decode must preserve its own prefix state"
    );
    assert!(arms[0][0].windows(2).any(|v| v[0] != v[1]));
}
