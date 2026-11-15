#![cfg(feature = "cuda-cutlass-nvfp4")]
//! Real managed packed GEMM. Dyadic operands make the selected scalar oracle
//! exact before BF16 rounding; this is not end-to-end FastVideo qualification.
#[path = "support/tensor_numeric_component.rs"]
mod fixture;
#[path = "support/nvfp4_stored_component.rs"]
mod stored;
use ferrum_execution::{
    component::{ComponentInitializationConfig, ComponentWeightSource},
    TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_tensor_runtime_config, cuda_tensor_runtime_enable_cutlass_nvfp4,
        cuda_vnext_nvfp4_matmul_bf16_provider, CutlassNvfp4Config,
    },
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use ferrum_runtime::*;
use fixture::{id, Config, Kind, BIAS, CHANNEL, INPUT, RELATIVE, WEIGHT};
use std::sync::Arc;

fn operand(rows: usize, k: usize, branch: usize) -> (Vec<u8>, Vec<u8>, Vec<f64>) {
    let layout = Nvfp4MatrixLayout::new(rows as u64, k as u64).unwrap();
    let mut packed = vec![0; layout.packed_bytes() as usize];
    // Inactive row padding is deliberately nonfinite and must not contaminate
    // active rows. This provider explicitly excludes K tails.
    let mut scales = vec![0x7f; layout.scale_bytes() as usize];
    let mut values = vec![0.; rows * k];
    let magnitudes = [0., 0.5, 1., 1.5, 2., 3., 4., 6.];
    for row in 0..rows {
        for group in 0..k / 16 {
            let power = (row + group + branch) % 3;
            let (code, scale) = [(0x30, 0.5), (0x38, 1.), (0x40, 2.)][power];
            scales[layout.scale_index(row as u64, group as u64).unwrap() as usize] = code;
            for offset in 0..16 {
                let column = group * 16 + offset;
                let nibble = if row == 0 && branch == 0 {
                    0
                } else {
                    ((row * 7 + column * 3 + branch * 5) % 16) as u8
                };
                let index = row * k + column;
                packed[index / 2] |= nibble << ((index % 2) * 4);
                let sign = if nibble & 8 == 0 { 1. } else { -1. };
                values[index] = magnitudes[(nibble & 7) as usize] * sign * scale;
            }
        }
    }
    (packed, scales, values)
}

#[test]
#[ignore = "requires configured SM120 host and locked cuda-cutlass-nvfp4 artifact"]
fn nvfp4_uses_managed_submission_and_matches_exact_dyadic_oracle() {
    let mut device = cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap();
    cuda_tensor_runtime_enable_cutlass_nvfp4(&mut device).unwrap();
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device,
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 2,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "nvfp4-correctness-unqualified-capacity".into(),
        })
        .unwrap(),
    );
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(nvfp4_matmul_bf16_contract().unwrap())],
        vec![cuda_vnext_nvfp4_matmul_bf16_provider(
            &runtime,
            &CutlassNvfp4Config {
                opaque_memory_bytes: 64 << 20,
                opaque_memory_profile: "nvfp4-module-correctness-unqualified-capacity".into(),
            },
        )
        .unwrap()],
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let descriptor = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.nvfp4.test"),
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
    for (m, n, k, stored_weights) in [
        (3, 8, 128, false),
        (129, 136, 256, false),
        (129, 136, 256, true),
        // Cross M and multiple N tiles, with inactive scale
        // padding still nonfinite. Exercise dynamic and Stored operands.
        (129, 264, 256, false),
        (129, 264, 256, true),
    ] {
        let config = Config {
            rows: m,
            inputs: k,
            outputs: n,
            kind: Kind::Nvfp4Matmul,
        };
        let (a, sa, av) = operand(m as usize, k as usize, 0);
        let (b, sb, bv) = operand(n as usize, k as usize, 1);
        let source = stored::Source::new(config, b.clone(), sb.clone());
        let component = if stored_weights {
            stored::compile(
                config,
                runtime.as_ref(),
                &registry,
                &catalog,
                &materializers,
            )
        } else {
            config.compile(runtime.as_ref(), &registry, &catalog, &materializers)
        };
        let host = fixture::host();
        let mut session = component
            .initialize(
                runtime.clone(),
                host.clone(),
                if stored_weights {
                    ComponentWeightSource::Stored(&source)
                } else {
                    ComponentWeightSource::None
                },
                ComponentInitializationConfig {
                    run_id: id("run.nvfp4.init"),
                    request_id: id("request.nvfp4.init"),
                    transaction_id: id("transaction.nvfp4.init"),
                    static_policy: StaticInitializationPolicy::new(1024, 8).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap();
        let expected_reads = if stored_weights { 2 } else { 0 };
        assert_eq!(source.component_reads(), expected_reads);
        let alpha = 0.75_f32;
        let expected = (0..m as usize)
            .flat_map(|i| {
                let av = &av;
                let bv = &bv;
                (0..n as usize).map(move |j| {
                    let dot = (0..k as usize)
                        .map(|l| av[i * k as usize + l] * bv[j * k as usize + l])
                        .sum::<f64>();
                    half::bf16::from_f32((dot * alpha as f64) as f32).to_bits()
                })
            })
            .collect::<Vec<_>>();
        let mut retained = None;
        for trial in 0..2 {
            let inputs = [
                (INPUT, a.as_slice()),
                (WEIGHT, b.as_slice()),
                (RELATIVE, sa.as_slice()),
                (CHANNEL, sb.as_slice()),
                (BIAS, &alpha.to_le_bytes()),
            ]
            .into_iter()
            .filter(|(name, _)| !stored_weights || !matches!(*name, WEIGHT | CHANNEL))
            .map(|(name, bytes)| config.managed(&host, name, bytes))
            .collect();
            let output = session
                .execute(
                    id(format!("run.nvfp4.{trial}")),
                    id(format!("request.nvfp4.{trial}")),
                    inputs,
                )
                .unwrap();
            let actual = output[0]
                .bytes()
                .bytes()
                .chunks_exact(2)
                .map(|v| u16::from_le_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(
                actual, expected,
                "shape {m}x{n}x{k}, stored={stored_weights}"
            );
            assert_eq!(source.component_reads(), expected_reads);
            retained = Some(output);
        }
        assert!(matches!(
            session
                .close()
                .unwrap_or_else(|e| panic!("NVFP4 close: {:?}", e.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        let retained = retained.unwrap();
        assert!(host.owns(retained[0].bytes()));
        drop(retained);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}
