//! Actual CUDA visual replacement, explicit RoPE and separate paged KV owners.
//! Same nonzero hidden/head-width-2 fixture as CPU: Portable attention accepts
//! this width and selects TokenMajorPages, without a vLLM head-width assumption.
//! Run the exact ignored test alone with the pinned native operator set. Missing
//! CUDA/providers or unsupported shapes are failures, never successful skips.
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
    let mut device = cuda_tensor_and_dense_f16_visual_causal_runtime_config(
        0,
        DeviceId::new("device.cuda.0").unwrap(),
    )
    .unwrap();
    let mut causal =
        CudaDenseF16CausalProfile::new(64 << 20, "unqualified-tiny-visual-cohort-modules-v1")
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
                opaque_memory_profile: "unqualified-tiny-visual-cohort-runtime-v1".into(),
            },
            causal,
        })
        .expect("visual cohort requires the actual CUDA tensor+visual causal runtime"),
    );
    let registry = OperationRuntimeRegistry::new(
        cuda_vnext_dense_f16_visual_causal_contracts().unwrap(),
        cuda_vnext_dense_f16_visual_causal_providers(&runtime)
            .expect("actual original dense F16 visual providers must initialize"),
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let descriptor = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        ProviderId::new("provider.engine.cuda.visual-cohort-fixture").unwrap(),
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
    .expect("selected CUDA providers require their original compiled operator set");
    (
        RuntimeComposition::new(Arc::clone(&runtime), registry, materializers, catalog),
        runtime,
    )
}

async fn run(
    fixture: &VisualFixture,
    composition: &RuntimeComposition<CudaDeviceRuntime>,
    cases: &[VisualCase],
    cancel: Option<(usize, bool)>,
) -> Observed {
    run_on(
        fixture,
        cases,
        cancel,
        host(),
        composition,
        ferrum_types::Device::CUDA(0),
        // CUDA retains its declared provider/context allowance across cases;
        // actual owner release is checked by Weak after the suite below.
        || {},
    )
    .await
}

async fn check(replay: bool) -> Vec<Vec<u32>> {
    let fixture = if replay {
        VisualFixture::new().with_decode_replay()
    } else {
        VisualFixture::new()
    };
    let (composition, runtime) = composition(replay);
    let original_runtime = Arc::downgrade(&runtime);
    let a = run(&fixture, &composition, &[MIXED], None).await;
    let b = run(&fixture, &composition, &[OTHER], None).await;
    // Hard numerical outcomes reject zero attention, missing visual replacement
    // and ignored explicit RoPE, even if both B1 and B2 share that defect.
    assert_eq!(a.tokens, [vec![SECOND_ANSWER_ID, EOS_ID]]);
    assert_eq!(b.tokens, [vec![ANSWER_ID, EOS_ID]]);
    assert_eq!(a.text, [Some(NEGATIVE_TEXT.to_owned())]);
    assert_eq!(b.text, [Some(POSITIVE_TEXT.to_owned())]);
    let pair = run(&fixture, &composition, &[MIXED, OTHER], None).await;
    assert_eq!(pair.tokens, [a.tokens[0].clone(), b.tokens[0].clone()]);
    assert_eq!(pair.text, [a.text[0].clone(), b.text[0].clone()]);
    let reverse = run(&fixture, &composition, &[OTHER, MIXED], None).await;
    assert_eq!(reverse.tokens, [b.tokens[0].clone(), a.tokens[0].clone()]);
    assert_eq!(reverse.text, [b.text[0].clone(), a.text[0].clone()]);
    for changed in [
        VisualCase {
            signs: &[1., 1.],
            ..MIXED
        },
        VisualCase {
            quarter_turn: -1.,
            ..MIXED
        },
    ] {
        let single = run(&fixture, &composition, &[changed], None).await;
        assert_eq!(single.tokens, [vec![ANSWER_ID, EOS_ID]]);
        assert_ne!(single.tokens[0], a.tokens[0]);
        let changed_pair = run(&fixture, &composition, &[changed, OTHER], None).await;
        assert_eq!(changed_pair.tokens[0], single.tokens[0]);
        assert_eq!(changed_pair.text[0], single.text[0]);
        assert_eq!(changed_pair.tokens[1], b.tokens[0]);
        assert_eq!(changed_pair.text[1], b.text[0]);
    }
    let cancel_cases = [MIXED, VisualCase { chunk: 2, ..OTHER }];
    let baseline = run(&fixture, &composition, &cancel_cases, None).await;
    assert_eq!(baseline, pair);
    for index in 0..2 {
        for after_visual_kv in [false, true] {
            let actual = run(
                &fixture,
                &composition,
                &cancel_cases,
                Some((index, after_visual_kv)),
            )
            .await;
            let survivor = 1 - index;
            assert!(actual.tokens[index].is_empty());
            assert_eq!(actual.text[index], None);
            assert_eq!(actual.tokens[survivor], baseline.tokens[survivor]);
            assert_eq!(actual.text[survivor], baseline.text[survivor]);
        }
    }
    let (repeated, metrics) = super::replay::run(
        &fixture,
        &composition,
        ferrum_types::Device::CUDA(0),
        &[MIXED, OTHER],
        None,
    )
    .await;
    let direct = metrics["counters"]["reusable_execution"]["direct_waves"]
        .as_u64()
        .unwrap();
    if replay {
        assert!(
            direct > 0,
            "visual decode must use actual replay: {metrics}"
        );
    } else {
        assert_eq!(direct, 0);
    }
    let reversed = super::replay::run(
        &fixture,
        &composition,
        ferrum_types::Device::CUDA(0),
        &[OTHER, MIXED],
        None,
    )
    .await
    .0;
    assert_eq!(repeated, [reversed[1].clone(), reversed[0].clone()]);
    for index in 0..2 {
        let cancelled = super::replay::run(
            &fixture,
            &composition,
            ferrum_types::Device::CUDA(0),
            &[MIXED, OTHER],
            Some(index),
        )
        .await
        .0;
        assert_eq!(cancelled[index], repeated[index][..5]);
        assert_eq!(cancelled[1 - index], repeated[1 - index]);
    }
    // Every run above certified its original executor Retired and refunded its
    // original Host account. Now release the one provider/context composition;
    // driver context caching is not misreported as a physical VRAM zero proof.
    drop(composition);
    drop(runtime);
    assert!(
        original_runtime.upgrade().is_none(),
        "original CUDA runtime still retained"
    );
    eprintln!(
        "{}",
        serde_json::json!({
            "diagnostic": "nonzero_cuda_visual_cohort",
            "decode_replay": replay,
            "repeated_decode_tokens": repeated,
            "replay_metrics": metrics,
            "independent_tokens": [a.tokens, b.tokens],
            "batch_tokens": pair.tokens,
            "reversed_tokens": reverse.tokens,
            "feature_and_rope_changes": "each changes only the selected member",
            "cancellation": "both members before admission and after actual visual KV",
            "original_runtime_released": true,
            "scope": "tiny actual CUDA causal fixture; no vision encoder or performance claim"
        })
    );
    repeated
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires exclusive actual CUDA and pinned native operators; nonzero synthetic visual causal correctness, not official image/performance qualification"]
async fn cuda_visual_cohort_nonzero_attention_isolates_features_rope_kv_and_cancellation() {
    let eager = check(false).await;
    let replay = check(true).await;
    assert_eq!(
        eager, replay,
        "every current visual/KV decode product must match eager"
    );
}
