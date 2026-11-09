#![cfg(feature = "cuda")]
//! Real managed affine/rank5 patch A/B. Test inputs are data, never CUDA results.
#[path = "support/tensor_numeric_component.rs"]
mod fixture;
use ferrum_execution::{
    component::{ComponentInitializationConfig, ComponentWeightSource},
    TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_tensor_runtime_config, cuda_vnext_affine_linear_f32_provider,
        cuda_vnext_dense_f32_provider, cuda_vnext_patch_projection_f32_provider,
        cuda_vnext_tensor_providers, CudaTensorOperation,
    },
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use ferrum_runtime::*;
use fixture::{id, Config, Kind, BIAS, INPUT, WEIGHT};
use std::sync::Arc;

fn runtime() -> Arc<CudaDeviceRuntime> {
    Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 2,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "unqualified-row-bias-correctness-fixture".into(),
        })
        .unwrap(),
    )
}

#[derive(Clone, Copy)]
enum Route {
    Saxpy,
    Broadcast,
    DotOnly,
}

fn execute(
    runtime: &Arc<CudaDeviceRuntime>,
    config: Config,
    route: Route,
    x: &[f32],
    w: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    let patch = matches!(config.kind, Kind::F32Patch);
    let config = if matches!(route, Route::DotOnly) {
        Config {
            kind: Kind::F32Dense,
            ..config
        }
    } else {
        config
    };
    let contract: Box<dyn OperationContract> = if matches!(route, Route::DotOnly) {
        Box::new(dense_linear_f32_contract().unwrap())
    } else if patch {
        Box::new(patch_projection_f32_contract().unwrap())
    } else {
        Box::new(affine_linear_f32_contract().unwrap())
    };
    let providers = match route {
        Route::Broadcast => cuda_vnext_tensor_providers(
            runtime,
            &[if patch {
                CudaTensorOperation::PatchProjectionF32
            } else {
                CudaTensorOperation::AffineLinearF32
            }],
        )
        .unwrap(),
        Route::Saxpy => vec![if patch {
            cuda_vnext_patch_projection_f32_provider(runtime).unwrap()
        } else {
            cuda_vnext_affine_linear_f32_provider(runtime).unwrap()
        }],
        Route::DotOnly => vec![cuda_vnext_dense_f32_provider(runtime).unwrap()],
    };
    let registry = OperationRuntimeRegistry::new(vec![contract], providers).unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.row_bias.test"),
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
    let component = config.compile(runtime.as_ref(), &registry, &catalog, &materializers);
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            256 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 128 << 20,
                transfer_staging_bytes: 64 << 20,
                output_bytes: 32 << 20,
                weight_backing_bytes: 32 << 20,
            },
        )
        .unwrap(),
    );
    let mut session = component
        .initialize(
            runtime.clone(),
            host.clone(),
            ComponentWeightSource::None,
            ComponentInitializationConfig {
                run_id: id("run.row_bias.init"),
                request_id: id("request.row_bias.init"),
                transaction_id: id("transaction.row_bias.init"),
                static_policy: StaticInitializationPolicy::new(1 << 20, 16).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    let managed = |name, values: &[f32]| {
        config.managed(
            &host,
            name,
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    };
    let mut inputs = vec![managed(INPUT, x), managed(WEIGHT, w)];
    if !matches!(route, Route::DotOnly) {
        inputs.push(managed(BIAS, bias));
    }
    let output = session
        .execute(
            id("run.row_bias.execute"),
            id("request.row_bias.execute"),
            inputs,
        )
        .unwrap();
    let values = output[0]
        .bytes()
        .bytes()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|e| panic!("close: {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    drop((output, component, catalog, registry));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    values
}

fn equal_float(actual: f32, expected: f32) {
    if expected.is_nan() {
        assert!(actual.is_nan());
    } else {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
#[ignore = "requires exclusive CUDA; actual same-runtime SGEMM/SAXPY and retained broadcast"]
fn affine_and_original_rank5_patch_match_saxpy_for_tails_official_widths_and_rounding() {
    let runtime = runtime();
    // R=1, non-tile tails, official hidden/QKV/MLP/merger widths and exact
    // original [O,3,2,16,16] patch storage. This is correctness, not throughput.
    for (rows, inputs, outputs, kind) in [
        (1, 3, 5, Kind::F32Affine),
        (7, 19, 257, Kind::F32Affine),
        (3, 1024, 3072, Kind::F32Affine),
        (3, 1024, 4096, Kind::F32Affine),
        (3, 4096, 1024, Kind::F32Affine),
        (5, 4096, 2560, Kind::F32Affine),
        (7, 13, 31, Kind::F32Patch),
        (3, 1536, 1024, Kind::F32Patch),
    ] {
        let config = Config {
            rows,
            inputs,
            outputs,
            kind,
        };
        let x: Vec<_> = (0..rows * inputs)
            .map(|i| ((i * 7 % 23) as f32 - 11.0) / 16.0)
            .collect();
        let w: Vec<_> = (0..outputs * inputs)
            .map(|i| ((i * 11 % 31) as f32 - 15.0) / 32.0)
            .collect();
        let bias: Vec<_> = (0..outputs)
            .map(|i| ((i % 17) as f32 - 8.0) / 64.0)
            .collect();
        let old = execute(&runtime, config, Route::Saxpy, &x, &w, &bias);
        let new = execute(&runtime, config, Route::Broadcast, &x, &w, &bias);
        let expected_len = usize::try_from(rows.checked_mul(outputs).unwrap()).unwrap();
        assert_eq!(old.len(), expected_len);
        assert_eq!(new.len(), expected_len);
        for (a, b) in new.into_iter().zip(old) {
            equal_float(a, b);
        }
    }
    // A single-product dot makes the stored F32 boundary observable. Bias
    // cancels 1.0 after rounding of (1+2^-23)*(1-2^-23), unlike fused FMA.
    for kind in [Kind::F32Affine, Kind::F32Patch] {
        let config = Config {
            rows: 1,
            inputs: 1,
            outputs: 8,
            kind,
        };
        let x = [f32::from_bits(0x3f800001)];
        let w = [
            f32::from_bits(0x3f7ffffe),
            1.0,
            -1.0,
            0.0,
            -0.0,
            1.0,
            1.0,
            1.0,
        ];
        let bias = [
            -1.0,
            f32::from_bits(0x33800000),
            f32::from_bits(0xb3800000),
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        let dot = execute(&runtime, config, Route::DotOnly, &x, &w, &[]);
        let new = execute(&runtime, config, Route::Broadcast, &x, &w, &bias);
        let old = execute(&runtime, config, Route::Saxpy, &x, &w, &bias);
        let expected_len =
            usize::try_from(config.rows.checked_mul(config.outputs).unwrap()).unwrap();
        assert_eq!(dot.len(), expected_len);
        assert_eq!(old.len(), expected_len);
        assert_eq!(new.len(), expected_len);
        assert_eq!(bias.len(), expected_len);
        for ((a, b), (d, bias)) in new.into_iter().zip(old).zip(dot.into_iter().zip(bias)) {
            equal_float(a, b);
            equal_float(a, d + bias);
        }
        // Signed-zero inputs do not guarantee SGEMM stores a negative zero.
        // This compares the observed dot-store boundary; it does not qualify
        // every combination of signed-zero dot and signed-zero bias.
        for x in [
            0.0,
            -0.0,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ] {
            let w = [1.0, -1.0, 0.0, -0.0, 1.0, 1.0, 1.0, 1.0];
            let bias = [
                0.0,
                -0.0,
                f32::from_bits(1),
                -f32::from_bits(1),
                f32::MAX,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
            ];
            let dot = execute(&runtime, config, Route::DotOnly, &[x], &w, &[]);
            let old = execute(&runtime, config, Route::Saxpy, &[x], &w, &bias);
            let new = execute(&runtime, config, Route::Broadcast, &[x], &w, &bias);
            assert_eq!(dot.len(), expected_len);
            assert_eq!(old.len(), expected_len);
            assert_eq!(new.len(), expected_len);
            assert_eq!(bias.len(), expected_len);
            for ((a, b), (d, bias)) in new.into_iter().zip(old).zip(dot.into_iter().zip(bias)) {
                equal_float(a, b);
                equal_float(a, d + bias);
            }
        }
    }
}
