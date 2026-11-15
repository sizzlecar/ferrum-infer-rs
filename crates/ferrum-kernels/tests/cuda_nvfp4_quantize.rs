#![cfg(all(feature = "cuda-cutlass-nvfp4", feature = "cuda-nvfp4-diagnostics"))]
//! Byte-exact independent quantizer qualification, through normal managed
//! plans. The explicit test artifact config is not a product environment mode.
#[path = "support/nvfp4_quantize_component.rs"]
mod fixture;
use ferrum_execution::{
    component::{ComponentInitializationConfig, ComponentWeightSource},
    TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_tensor_runtime_config, cuda_tensor_runtime_enable_cutlass_nvfp4,
        cuda_vnext_nvfp4_bf16_providers, cuda_vnext_nvfp4_reference_provider,
        CudaNvfp4ReferenceConfig, CutlassNvfp4Config,
    },
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use ferrum_runtime::*;
use fixture::{id, Config, GLOBAL, INPUT};
use std::sync::Arc;

fn compare(actual: &[Vec<u8>], expected: &[Vec<u8>], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: output count");
    for (port, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{context}: output {port} length"
        );
        if let Some((offset, (a, e))) = actual
            .iter()
            .zip(expected)
            .enumerate()
            .find(|(_, (a, e))| a != e)
        {
            panic!("{context}: output {port}, byte {offset}: actual={a:#04x}, expected={e:#04x}");
        }
    }
}

fn execute(
    runtime: &Arc<CudaDeviceRuntime>,
    registry: &OperationRuntimeRegistry<CudaDeviceRuntime>,
    config: Config,
    input: &[u8],
    global: f32,
) -> Vec<Vec<u8>> {
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.quantizer.qualification"),
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
    let component = config.compile(runtime.as_ref(), registry, &catalog, &materializers);
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            16 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 4 << 20,
                transfer_staging_bytes: 4 << 20,
                output_bytes: 4 << 20,
                weight_backing_bytes: 4 << 20,
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
                run_id: id("run.quantizer.init"),
                request_id: id("request.quantizer.init"),
                transaction_id: id("transaction.quantizer.init"),
                static_policy: StaticInitializationPolicy::new(1024, 8).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    let mut prior: Option<Vec<Vec<u8>>> = None;
    let mut retained = None;
    for trial in 0..2 {
        let out = session
            .execute(
                id(format!("run.quantizer.{trial}")),
                id(format!("request.quantizer.{trial}")),
                vec![
                    config.managed(&host, INPUT, input),
                    config.managed(&host, GLOBAL, &global.to_le_bytes()),
                ],
            )
            .unwrap();
        let bytes = out
            .iter()
            .map(|v| v.bytes().bytes().to_vec())
            .collect::<Vec<_>>();
        if let Some(prior) = &prior {
            compare(&bytes, prior, "repeat changed quantization output");
        }
        prior = Some(bytes);
        retained = Some(out);
    }
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|e| panic!("quantizer close: {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    let retained = retained.unwrap();
    assert!(retained.iter().all(|v| host.owns(v.bytes())));
    drop(retained);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    prior.unwrap()
}

#[test]
#[ignore = "SM120; set FERRUM_NVFP4_REFERENCE_CONFIG to explicit CudaNvfp4ReferenceConfig JSON"]
fn managed_quantizer_matches_independent_flashinfer_at_boundaries() {
    let path = std::env::var_os("FERRUM_NVFP4_REFERENCE_CONFIG")
        .expect("explicit reference artifact config required");
    let config_bytes = std::fs::read(path).unwrap();
    assert!(config_bytes.len() <= 16 * 1024);
    let reference: CudaNvfp4ReferenceConfig = serde_json::from_slice(&config_bytes).unwrap();
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
            opaque_memory_profile: "nvfp4-quantizer-correctness-unqualified-capacity".into(),
        })
        .unwrap(),
    );
    let providers = cuda_vnext_nvfp4_bf16_providers(
        &runtime,
        &CutlassNvfp4Config {
            opaque_memory_bytes: 64 << 20,
            opaque_memory_profile: "nvfp4-quantizer-module-unqualified-capacity".into(),
        },
    )
    .unwrap();
    let candidate = OperationRuntimeRegistry::new(
        vec![
            Box::new(nvfp4_quantize_bf16_contract().unwrap()),
            Box::new(nvfp4_matmul_bf16_contract().unwrap()),
        ],
        providers,
    )
    .unwrap();
    let oracle = OperationRuntimeRegistry::new(
        vec![Box::new(nvfp4_quantize_bf16_contract().unwrap())],
        vec![cuda_vnext_nvfp4_reference_provider(&runtime, &reference).unwrap()],
    )
    .unwrap();

    let dyadic = [
        -6., -4., -3., -2., -1.5, -1., -0.5, -0., 0., 0.5, 1., 1.5, 2., 3., 4., 6.,
    ];
    let input = dyadic
        .into_iter()
        .flat_map(|v| half::bf16::from_f32(v).to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    let c = Config { rows: 1, k: 16 };
    let expected = execute(&runtime, &oracle, c, &input, 1.);
    assert_eq!(
        expected[0],
        [0xef, 0xcd, 0xab, 0x89, 0x10, 0x32, 0x54, 0x76]
    );
    assert_eq!(expected[1][0], 0x38);
    assert!(expected[1][1..].iter().all(|b| *b == 0));
    compare(
        &execute(&runtime, &candidate, c, &input, 1.),
        &expected,
        "dyadic codes",
    );

    // Exact E2M1 ties and the adjacent BF16 values, with amax=6 and SF=1.
    // The middle row has an independent RNE-even byte oracle.
    let c = Config { rows: 3, k: 16 };
    let input = (-1i32..=1)
        .flat_map(|delta| {
            let mut row = Vec::new();
            for midpoint in [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.] {
                let bits = (i32::from(half::bf16::from_f32(midpoint).to_bits()) + delta) as u16;
                row.extend(bits.to_le_bytes());
                row.extend((bits | 0x8000).to_le_bytes());
            }
            row.extend(0u16.to_le_bytes());
            row.extend(half::bf16::from_f32(6.).to_bits().to_le_bytes());
            row
        })
        .collect::<Vec<_>>();
    let expected = execute(&runtime, &oracle, c, &input, 1.);
    assert_eq!(
        &expected[0][8..16],
        &[0x80, 0xa2, 0xa2, 0xc4, 0xc4, 0xe6, 0xe6, 0x70]
    );
    compare(
        &execute(&runtime, &candidate, c, &input, 1.),
        &expected,
        "E2M1 ties and neighbors",
    );

    // Every finite BF16 bit pattern appears in the large case, including
    // both signed zeros and all BF16 subnormals. Small cases exercise K tails
    // and both scale-tile dimensions without padded packed-value rows.
    for c in [
        Config { rows: 3, k: 80 },
        Config { rows: 129, k: 144 },
        Config { rows: 129, k: 512 },
        Config { rows: 2048, k: 256 },
    ] {
        let input = (0..c.rows * c.k)
            .flat_map(|i| {
                // Sweep every positive finite BF16 amax in separate groups;
                // random mixtures alone would hide tiny-scale underflow behind
                // a larger group maximum. Include both signs and exact zeros.
                if c.rows == 2048 {
                    let raw = (i / 16) as u16;
                    let maximum =
                        half::bf16::from_bits(if raw < 0x7f80 { raw } else { 0 }).to_f32();
                    let factor = [-1., -0.5, -0.25, -0., 0., 0.25, 0.5, 1.][(i % 8) as usize];
                    return half::bf16::from_f32(maximum * factor)
                        .to_bits()
                        .to_le_bytes();
                }
                let raw = (i.wrapping_mul(40503).wrapping_add(17)) as u16;
                let finite = if raw & 0x7f80 == 0x7f80 {
                    raw & 0x8000
                } else {
                    raw
                };
                finite.to_le_bytes()
            })
            .collect::<Vec<_>>();
        for global in [
            1.,
            0.125,
            1.125,
            448.,
            2.0f32.powi(-120),
            2.0f32.powi(120),
            f32::from_bits(1),
            f32::MIN_POSITIVE,
            f32::MAX,
        ] {
            let expected = execute(&runtime, &oracle, c, &input, global);
            let actual = execute(&runtime, &candidate, c, &input, global);
            compare(
                &actual,
                &expected,
                &format!("shape {}x{}, global {global:e}", c.rows, c.k),
            );
            let layout = Nvfp4MatrixLayout::new(c.rows, c.k).unwrap();
            let mut active = vec![false; layout.scale_bytes() as usize];
            for row in 0..c.rows {
                for group in 0..c.k / 16 {
                    active[layout.scale_index(row, group).unwrap() as usize] = true;
                }
            }
            assert!(
                actual[1]
                    .iter()
                    .zip(active)
                    .all(|(byte, active)| active || *byte == 0),
                "scale padding must be zero"
            );
        }
    }
}
