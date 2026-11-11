//! Actual bounded CUDA causal admission, source installation and decode replay.
//! This tiny stateful model is a correctness fixture, not a performance claim.
use super::nonzero::{run_with_replay_limit, NonzeroFixture};
use super::*;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_tensor_and_dense_bf16_visual_causal_runtime_config,
        cuda_tensor_and_dense_f16_visual_causal_runtime_config,
        cuda_vnext_dense_bf16_visual_causal_contracts,
        cuda_vnext_dense_bf16_visual_causal_providers,
        cuda_vnext_dense_f16_visual_causal_contracts, cuda_vnext_dense_f16_visual_causal_providers,
    },
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
    CudaDenseF16CausalProfile, CudaTensorAndCausalRuntimeConfig,
};
use ferrum_models::vnext::qwen35::Qwen35DensePrecision;
use ferrum_runtime::{
    CapabilityId, ContractVersion, DeviceId, EngineProviderDescriptor, OperationRuntimeRegistry,
    ProviderId, WeightMaterializerRegistry, DEVICE_ON_DEMAND_REUSABLE_EXECUTION_CAPABILITY_ID,
    DEVICE_REUSABLE_EXECUTION_CAPABILITY_ID,
};

pub(in crate::image_generation::p_h3::expansion) fn composition(
    replay: bool,
) -> (
    RuntimeComposition<CudaDeviceRuntime>,
    Arc<CudaDeviceRuntime>,
) {
    composition_with_precision(replay, Qwen35DensePrecision::F16, 1024, 64 << 20)
}

fn composition_with_precision(
    replay: bool,
    precision: Qwen35DensePrecision,
    maximum_executables: usize,
    graph_memory_bytes: u64,
) -> (
    RuntimeComposition<CudaDeviceRuntime>,
    Arc<CudaDeviceRuntime>,
) {
    // Real products advertise visual capability even for a text-only request.
    let device_id = DeviceId::new("device.cuda.0").unwrap();
    let module_profile = "unqualified-nonzero-replay-modules-v1";
    let (device, causal) = match precision {
        Qwen35DensePrecision::F16 => (
            cuda_tensor_and_dense_f16_visual_causal_runtime_config(0, device_id),
            CudaDenseF16CausalProfile::new(64 << 20, module_profile),
        ),
        Qwen35DensePrecision::Bf16 => (
            cuda_tensor_and_dense_bf16_visual_causal_runtime_config(0, device_id),
            CudaDenseF16CausalProfile::new_bf16(64 << 20, module_profile),
        ),
    };
    let mut device = device.unwrap();
    let mut causal = causal.unwrap().with_visual_prompt_v1();
    if replay {
        causal = causal
            .with_decode_replay(maximum_executables, graph_memory_bytes)
            .unwrap();
        assert_eq!(
            causal.decode_replay_limits(),
            Some((maximum_executables, graph_memory_bytes))
        );
        assert_eq!(
            causal.decode_replay_memory_bytes_per_stream(),
            graph_memory_bytes + (32 << 20)
        );
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
    let (contracts, providers) = match precision {
        Qwen35DensePrecision::F16 => (
            cuda_vnext_dense_f16_visual_causal_contracts().unwrap(),
            cuda_vnext_dense_f16_visual_causal_providers(&runtime).unwrap(),
        ),
        Qwen35DensePrecision::Bf16 => (
            cuda_vnext_dense_bf16_visual_causal_contracts().unwrap(),
            cuda_vnext_dense_bf16_visual_causal_providers(&runtime).unwrap(),
        ),
    };
    let registry = OperationRuntimeRegistry::new(contracts, providers).unwrap();
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
    assert_replay_matches_eager(Qwen35DensePrecision::F16, 1024, 64 << 20).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires exclusive actual CUDA and pinned native operators; tiny BF16 N1 decode correctness, not official-model qualification"]
async fn cuda_bounded_bf16_decode_replay_matches_nonzero_eager_state_and_retires_original_owners() {
    // Same declared graph allowance as the product's ordinary F16 decode arm.
    // The original runtime additionally admits its 32 MiB BLAS workspace per lane.
    assert_replay_matches_eager(Qwen35DensePrecision::Bf16, 64, 128 << 20).await;
}

async fn assert_replay_matches_eager(
    precision: Qwen35DensePrecision,
    maximum_executables: usize,
    graph_memory_bytes: u64,
) {
    let fixture = NonzeroFixture::with_precision(precision);
    let mut arms = Vec::new();
    for replay in [false, true] {
        let (composition, runtime) =
            composition_with_precision(replay, precision, maximum_executables, graph_memory_bytes);
        let original = Arc::downgrade(&runtime);
        let mut results = Vec::new();
        for prompt in [[0, 0, 0], [1, 1, 1]] {
            let (tokens, metrics) = run_with_replay_limit(
                &fixture,
                &composition,
                ferrum_types::Device::CUDA(0),
                replay,
                &prompt,
                replay.then_some(maximum_executables as u64),
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
                serde_json::json!({"diagnostic":"bounded_causal_decode", "precision":precision, "replay":replay, "maximum_executables":maximum_executables, "graph_memory_bytes_per_lane":graph_memory_bytes, "prompt":prompt, "tokens":tokens, "metrics":metrics})
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
