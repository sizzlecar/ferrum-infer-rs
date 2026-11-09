#![cfg(feature = "cuda")]
//! Explicit GPU qualification of managed full-F32 and W4/W8 A8 integer paths.
#[path = "support/tensor_numeric_component.rs"]
mod fixture;
use ferrum_execution::{
    component::{ComponentInitializationConfig, ComponentWeightSource},
    TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_tensor_runtime_config, cuda_vnext_affine_linear_f32_provider,
        cuda_vnext_dense_f32_provider, cuda_vnext_tensor_providers, CudaTensorOperation,
    },
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use ferrum_runtime::*;
use fixture::*;
use half::bf16;
use std::sync::{Arc, Mutex};

static GPU: Mutex<()> = Mutex::new(());
fn runtime() -> Arc<CudaDeviceRuntime> {
    // Declared fixture allowances; not measured native-memory qualification.
    Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 1024 * 1024 * 1024,
            bootstrap_memory_bytes: 256 * 1024 * 1024,
            stream_memory_bytes: 16 * 1024 * 1024,
            maximum_streams: 2,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "unqualified-numeric-component-test-v1".into(),
        })
        .unwrap(),
    )
}
fn catalog(
    runtime: &CudaDeviceRuntime,
    registry: &OperationRuntimeRegistry<CudaDeviceRuntime>,
    materializers: &WeightMaterializerRegistry,
) -> CapabilityCatalog {
    let descriptor = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.numeric.component.test"),
        ContractVersion::new(1, 0),
        descriptor.runtime_implementation_fingerprint.clone(),
        descriptor.id.clone(),
        descriptor.capabilities.clone(),
    )
    .unwrap();
    materializers
        .augment_catalog(
            registry
                .capability_catalog(descriptor.clone(), vec![engine])
                .unwrap(),
        )
        .unwrap()
}
fn initialization() -> ComponentInitializationConfig {
    ComponentInitializationConfig {
        run_id: id("run.cuda.numeric.init"),
        request_id: id("request.cuda.numeric.init"),
        transaction_id: id("transaction.cuda.numeric.init"),
        static_policy: StaticInitializationPolicy::new(1 << 20, 16).unwrap(),
        executor: TensorExecutorConfig::default(),
    }
}

#[test]
#[ignore = "requires an explicitly configured CUDA host; exercises real pedantic F32 SGEMM"]
fn pedantic_f32_dense_executes_all_rows_without_tf32_or_half_operand_rounding() {
    let _gpu = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let runtime = runtime();
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(dense_linear_f32_contract().unwrap())],
        vec![cuda_vnext_dense_f32_provider(&runtime).unwrap()],
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let catalog = catalog(&runtime, &registry, &materializers);
    let component = f32_config().compile(runtime.as_ref(), &registry, &catalog, &materializers);
    let host = host();
    let mut session = component
        .initialize(
            runtime.clone(),
            host.clone(),
            ComponentWeightSource::None,
            initialization(),
        )
        .unwrap();
    let output = execute_f32(&mut session, &host);
    assert!(matches!(
        session.close().unwrap_or_else(|failure| panic!(
            "CUDA component close failed: {:?}",
            failure.failure()
        )),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    drop(output);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
#[ignore = "requires configured CUDA host; actual pedantic SGEMM then F32 SAXPY bias"]
fn pedantic_f32_affine_executes_real_bias_without_half_precision_rounding() {
    let _gpu = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let runtime = runtime();
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(affine_linear_f32_contract().unwrap())],
        vec![cuda_vnext_affine_linear_f32_provider(&runtime).unwrap()],
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let catalog = catalog(&runtime, &registry, &materializers);
    let component =
        affine_f32_config().compile(runtime.as_ref(), &registry, &catalog, &materializers);
    let host = host();
    let mut session = component
        .initialize(
            runtime.clone(),
            host.clone(),
            ComponentWeightSource::None,
            initialization(),
        )
        .unwrap();
    for trial in 0..2 {
        let output = execute_affine_f32(&mut session, &host, trial);
        drop(output);
    }
    assert!(matches!(
        session.close().unwrap_or_else(|failure| panic!(
            "CUDA F32 affine close failed: {:?}",
            failure.failure()
        )),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
#[ignore = "requires configured CUDA host; actual managed IEEE F32 to BF16 conversion"]
fn f32_to_bf16_cast_executes_ties_subnormals_overflow_and_nan_classes() {
    let _gpu = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let runtime = runtime();
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(cast_f32_to_bf16_contract().unwrap())],
        cuda_vnext_tensor_providers(&runtime, &[CudaTensorOperation::CastF32ToBf16]).unwrap(),
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let catalog = catalog(&runtime, &registry, &materializers);
    let component =
        cast_f32_to_bf16_config().compile(runtime.as_ref(), &registry, &catalog, &materializers);
    let host = host();
    let mut session = component
        .initialize(
            runtime.clone(),
            host.clone(),
            ComponentWeightSource::None,
            initialization(),
        )
        .unwrap();
    for trial in 0..2 {
        drop(execute_cast_f32_to_bf16(&mut session, &host, trial));
    }
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|failure| panic!("CUDA cast close failed: {:?}", failure.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
#[ignore = "requires an explicitly configured CUDA host; exercises real managed INT8 GEMM"]
fn canonical_w4_a8_integer_gemm_executes_through_managed_component() {
    let _gpu = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let runtime = runtime();
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(comfy_convrot_w4a8_bf16_contract().unwrap())],
        cuda_vnext_tensor_providers(&runtime, &[CudaTensorOperation::ComfyConvRotW4A8]).unwrap(),
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let catalog = catalog(&runtime, &registry, &materializers);
    let config = Config {
        rows: 3,
        inputs: 256,
        outputs: 4,
        kind: Kind::ComfyW4A8,
    };
    let component = config.compile(runtime.as_ref(), &registry, &catalog, &materializers);
    let host = host();
    let mut session = component
        .initialize(
            runtime.clone(),
            host.clone(),
            ComponentWeightSource::None,
            initialization(),
        )
        .unwrap();
    let packed: Vec<u8> = [0x00_u8, 0x11, 0x22, 0xff]
        .into_iter()
        .flat_map(|value| std::iter::repeat_n(value, 128))
        .collect();
    let relative = vec![0x38_u8; 4 * 16];
    let channel: Vec<u8> = [0.5_f32, 0.25, 0.125, 1.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let mut table = [0.0_f32; 16];
    table[0] = -1.0;
    table[1] = 1.0;
    table[2] = 2.0;
    table[15] = 0.5;
    let codebook: Vec<u8> = table.into_iter().flat_map(f32::to_le_bytes).collect();
    for (trial, values) in [[1.0_f32, -0.5, 0.0], [0.0, 0.0, 0.0]]
        .into_iter()
        .enumerate()
    {
        let x: Vec<u8> = values
            .into_iter()
            .flat_map(|value| std::iter::repeat_n(bf16::from_f32(value).to_bits(), 256))
            .flat_map(u16::to_le_bytes)
            .collect();
        let names = [INPUT, WEIGHT, RELATIVE, CHANNEL, CODEBOOK];
        let data = [
            x.as_slice(),
            packed.as_slice(),
            relative.as_slice(),
            channel.as_slice(),
            codebook.as_slice(),
        ];
        let batch = names
            .into_iter()
            .zip(data)
            .map(|(name, bytes)| config.managed(&host, name, bytes))
            .collect();
        let output = session
            .execute(
                id(format!("run.cuda.comfy_convrot.{trial}")),
                id(format!("request.cuda.comfy_convrot.{trial}")),
                batch,
            )
            .unwrap();
        assert_eq!(output[0].tensor().dimensions(), &[3, 4]);
        let actual: Vec<_> = output[0]
            .bytes()
            .bytes()
            .chunks_exact(2)
            .map(|bytes| bf16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32())
            .collect();
        let expected = if trial == 0 {
            [
                -128.0, 64.0, 64.0, 0.0, 64.0, -32.0, -32.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        } else {
            [0.0; 12]
        };
        assert_eq!(actual, expected);
    }
    assert!(matches!(
        session.close().unwrap_or_else(|failure| panic!(
            "CUDA component close failed: {:?}",
            failure.failure()
        )),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
#[ignore = "requires an explicitly configured CUDA host; exercises direct signed INT8 weights"]
fn canonical_w8_a8_integer_gemm_preserves_minimum_weight_code_and_scale_scope() {
    let _gpu = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let runtime = runtime();
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(comfy_convrot_w8a8_bf16_contract().unwrap())],
        cuda_vnext_tensor_providers(&runtime, &[CudaTensorOperation::ComfyConvRotW8A8]).unwrap(),
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let catalog = catalog(&runtime, &registry, &materializers);
    let config = Config {
        rows: 3,
        inputs: 256,
        outputs: 4,
        kind: Kind::ComfyW8A8,
    };
    let component = config.compile(runtime.as_ref(), &registry, &catalog, &materializers);
    let host = host();
    let mut session = component
        .initialize(
            runtime.clone(),
            host.clone(),
            ComponentWeightSource::None,
            initialization(),
        )
        .unwrap();
    let weight: Vec<u8> = [-128_i8, 127, -2, 0]
        .into_iter()
        .flat_map(|value| std::iter::repeat_n(value as u8, 256))
        .collect();
    let scales: Vec<u8> = [0.5_f32, 0.25, 0.125, 1.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    for (trial, values) in [[1.0_f32, -0.5, 0.0], [0.0, 0.0, 0.0]]
        .into_iter()
        .enumerate()
    {
        let activation: Vec<u8> = values
            .into_iter()
            .flat_map(|value| std::iter::repeat_n(bf16::from_f32(value).to_bits(), 256))
            .flat_map(u16::to_le_bytes)
            .collect();
        let output = session
            .execute(
                id(format!("run.cuda.comfy_convrot.w8.{trial}")),
                id(format!("request.cuda.comfy_convrot.w8.{trial}")),
                [INPUT, WEIGHT, CHANNEL]
                    .into_iter()
                    .zip([activation.as_slice(), weight.as_slice(), scales.as_slice()])
                    .map(|(name, bytes)| config.managed(&host, name, bytes))
                    .collect(),
            )
            .unwrap();
        assert_eq!(output[0].tensor().dimensions(), &[3, 4]);
        let actual: Vec<_> = output[0]
            .bytes()
            .bytes()
            .chunks_exact(2)
            .map(|bytes| bf16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32())
            .collect();
        let expected = if trial == 0 {
            [
                -16384.0, 8128.0, -64.0, 0.0, 8192.0, -4064.0, 32.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        } else {
            [0.0; 12]
        };
        assert_eq!(actual, expected);
    }
    assert!(matches!(
        session.close().unwrap_or_else(|failure| panic!(
            "CUDA component close failed: {:?}",
            failure.failure()
        )),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
#[ignore = "requires configured CUDA host; F32 SwiGLU product reaches managed W4 A8 GEMM"]
fn fused_swiglu_f32_w4_a8_avoids_intermediate_bf16_rounding() {
    let _gpu = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    fused_swiglu_projection(true);
}

#[test]
#[ignore = "requires configured CUDA host; F32 SwiGLU product reaches managed W8 A8 GEMM"]
fn fused_swiglu_f32_w8_a8_avoids_intermediate_bf16_rounding() {
    let _gpu = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    fused_swiglu_projection(false);
}

fn fused_swiglu_projection(w4: bool) {
    let runtime = runtime();
    let (contract, operation, kind) = if w4 {
        (
            comfy_convrot_swiglu_f32_w4a8_bf16_contract().unwrap(),
            CudaTensorOperation::ComfyConvRotSwiGluF32W4A8,
            Kind::ComfySwiGluF32W4A8,
        )
    } else {
        (
            comfy_convrot_swiglu_f32_w8a8_bf16_contract().unwrap(),
            CudaTensorOperation::ComfyConvRotSwiGluF32W8A8,
            Kind::ComfySwiGluF32W8A8,
        )
    };
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(contract)],
        cuda_vnext_tensor_providers(&runtime, &[operation]).unwrap(),
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let catalog = catalog(&runtime, &registry, &materializers);
    let config = Config {
        rows: 3,
        inputs: 256,
        outputs: 4,
        kind,
    };
    let component = config.compile(runtime.as_ref(), &registry, &catalog, &materializers);
    let host = host();
    let mut session = component
        .initialize(
            runtime.clone(),
            host.clone(),
            ComponentWeightSource::None,
            initialization(),
        )
        .unwrap();
    let weights: Vec<u8> = if w4 {
        [0x00_u8, 0x11, 0x22, 0xff]
            .into_iter()
            .flat_map(|v| std::iter::repeat_n(v, 128))
            .collect()
    } else {
        [-1_i8, 1, 2, 0]
            .into_iter()
            .flat_map(|v| std::iter::repeat_n(v as u8, 256))
            .collect()
    };
    let relative = vec![0x38_u8; 4 * 16];
    let scales: Vec<u8> = [0.5_f32, 0.25, 0.125, 1.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let mut table = [0.0_f32; 16];
    table[0] = -1.0;
    table[1] = 1.0;
    table[2] = 2.0;
    let table: Vec<u8> = table.into_iter().flat_map(f32::to_le_bytes).collect();
    for (trial, ups) in [[3.0_f32, -1.5, 0.0], [0.0, 0.0, 0.0]]
        .into_iter()
        .enumerate()
    {
        let gate_up: Vec<u8> = ups
            .into_iter()
            .flat_map(|up| {
                [0.5_f32, up]
                    .into_iter()
                    .flat_map(|v| std::iter::repeat_n(bf16::from_f32(v).to_bits(), 256))
            })
            .flat_map(u16::to_le_bytes)
            .collect();
        let mut data = vec![(INPUT, gate_up.as_slice()), (WEIGHT, weights.as_slice())];
        if w4 {
            data.push((RELATIVE, relative.as_slice()));
        }
        data.push((CHANNEL, scales.as_slice()));
        if w4 {
            data.push((CODEBOOK, table.as_slice()));
        }
        let output = session
            .execute(
                id(format!("run.cuda.comfy_swiglu_f32.{trial}")),
                id(format!("request.cuda.comfy_swiglu_f32.{trial}")),
                data.into_iter()
                    .map(|(name, bytes)| config.managed(&host, name, bytes))
                    .collect(),
            )
            .unwrap();
        assert_eq!(output[0].tensor().dimensions(), &[3, 4]);
        let actual: Vec<f32> = output[0]
            .bytes()
            .bytes()
            .chunks_exact(2)
            .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect();
        let expected = if trial == 0 {
            // B(B(silu(.5))*3) would give -119 instead of -119.5.
            [
                -119.5, 59.75, 59.75, 0.0, 59.75, -29.875, -29.875, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        } else {
            [0.0; 12]
        };
        assert_eq!(actual, expected);
    }
    assert!(matches!(
        session.close().unwrap_or_else(|failure| panic!(
            "CUDA fused SwiGLU close failed: {:?}",
            failure.failure()
        )),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
