//! D128 GQA uses the same public managed path as the text encoder. Expected
//! values below are analytic fixtures, not a replay of the candidate kernel.
use super::*;
use half::bf16;

#[path = "gqa_d128/family.rs"]
mod family;
use family::{GqaConfig, GqaFamily};

const PORTS: [&str; 3] = [QUERY, KEY, VALUE];
const GQA_ONLY: [CudaTensorOperation; 1] = [CudaTensorOperation::StructuredGqaAttention];

struct Case {
    name: &'static str,
    inputs: [Vec<u16>; 3],
    expected: Vec<u16>,
}

fn scaled(bits: u16, exponent: usize) -> u16 {
    bf16::from_f32(bf16::from_bits(bits).to_f32() * (1_u32 << exponent) as f32).to_bits()
}

// Each KV head has its own power-of-two scale. Dimensions exercise opposite
// signs, gradual underflow, -0, visible V NaN/Inf, and both BF16 tie parities.
fn value_bits(key: usize, head: usize, dimension: usize) -> u16 {
    let a = scaled(0x3f80, head);
    match (dimension % 9, key) {
        (0, 0) | (2, 0) => 0,
        (1, 0) | (3, 0) | (4, _) => 0x8000,
        (0, 2) | (5, 2) => a,
        (1, 2) => a | 0x8000,
        (2, 2) => 0x0001,
        (3, 2) => 0x8001,
        (0, 4) | (5, 4) => a | 0x8000,
        (1, 4) => a,
        (2, 4) => 0x8001,
        (3, 4) => 0x0001,
        (5, 0) => 0x7fc1,
        (6, 0 | 4) => 0xff80,
        (6, 2) => 0x7f80,
        (7, 0 | 2) => 0x3f80,
        (7, 4) | (8, 0 | 2) => 0x3f81,
        (8, 4) => 0x3f82,
        _ => panic!("fixture has no visible value for key {key}"),
    }
}

fn expected_wide(batch: usize, row: usize, head: usize, d: usize, poison_future: bool) -> u16 {
    let kv = head / 4;
    if batch == 1 && row < 2 {
        return 0; // These queries contain NaNs but have no valid causal keys.
    }
    if row == 3 || (row == 4 && poison_future) {
        return 0x7fc0; // Visible nonfinite score poisons the entire head.
    }
    if row < 2 {
        return value_bits(0, kv, d);
    }
    if batch == 1 && row == 2 {
        return value_bits(2, kv, d); // First visible key bypasses weighted addition.
    }
    if row == 4 {
        return match d % 9 {
            0..=3 => 0, // Exact cancellation, including BF16 subnormals.
            4 => 0x8000,
            5 if batch == 0 => 0x7fc0,
            5 => 0,
            6 => 0x7fc0,               // +Inf and -Inf are both visible.
            7 => 0x3f80,               // B1: mean(1, 1+1/128) ties to the even lower bit.
            8 if batch == 1 => 0x3f82, // Opposite tie parity rounds upward.
            8 => 0x3f81,
            _ => unreachable!(),
        };
    }
    // B0/query2 has exactly two visible scores, [0,+1] or [0,-1].
    // sigmoid(+1) -> 0x3f3b; sigmoid(-1) -> 0x3e8a. These are far from
    // BF16 midpoints; multiplying by an exact power of two preserves RNE.
    let positive = head % 2 == 0;
    match d % 9 {
        0 => scaled(if positive { 0x3f3b } else { 0x3e8a }, kv),
        1 => scaled(if positive { 0x3f3b } else { 0x3e8a }, kv) | 0x8000,
        2 => u16::from(positive),
        3 => 0x8000 | u16::from(positive),
        4 => 0x8000,
        5 | 6 => 0x7fc0,
        7 => 0x3f80,
        8 => 0x3f81,
        _ => unreachable!(),
    }
}

fn wide_case(poison_future: bool) -> Case {
    let mut query = vec![0; 2 * 5 * 32 * 128];
    let mut key = vec![0; 2 * 5 * 8 * 128];
    let mut value = key.clone();
    let mut expected = query.clone();
    for batch in 0..2 {
        for row in 0..5 {
            for head in 0..32 {
                for d in 0..128 {
                    let i = ((batch * 5 + row) * 32 + head) * 128 + d;
                    query[i] = if row == 3 || (batch == 1 && row < 2) {
                        0x7fc1
                    } else if row == 2 {
                        if head % 2 == 0 {
                            0x3f80
                        } else {
                            0xbf80
                        }
                    } else {
                        0
                    };
                    expected[i] = expected_wide(batch, row, head, d, poison_future);
                }
            }
            for head in 0..8 {
                let base = ((batch * 5 + row) * 8 + head) * 128;
                let masked = row == 1 || row == 3 || (batch == 1 && row == 0);
                if masked {
                    key[base..base + 128].fill(0x7f80);
                    value[base..base + 128].fill(0xffc1);
                    continue;
                }
                for d in 0..128 {
                    value[base + d] = value_bits(row, head, d);
                }
                if row == 2 {
                    // Both placements have adjacent-tree sum 1. For even
                    // heads, a conventional descending shuffle tree gives 2.
                    // For odd heads the cancellation is inside one lane.
                    let positions = if head % 2 == 0 {
                        [0, 4, 8, 12]
                    } else {
                        [0, 1, 2, 3]
                    };
                    for (d, bits) in positions.into_iter().zip([0x4b80, 0x3f80, 0xcb80, 0x3f80]) {
                        key[base + d] = bits;
                    }
                } else if row == 4 && poison_future {
                    key[base..base + 128].fill(0x7f80);
                }
            }
        }
    }
    Case {
        name: if poison_future {
            "future_poison"
        } else {
            "finite_future"
        },
        inputs: [query, key, value],
        expected,
    }
}

fn run_cases(config: GqaConfig, cases: Vec<Case>) {
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: USABLE_BYTES,
            bootstrap_memory_bytes: 1 << 30,
            stream_memory_bytes: 64 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "experimental-unqualified-gqa-d128-fixture-v1".into(),
        })
        .unwrap(),
    );
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(structured_gqa_attention_bf16_contract().unwrap())],
        cuda_vnext_tensor_providers(&runtime, &GQA_ONLY).unwrap(),
    )
    .unwrap();
    assert!(cuda_vnext_tensor_providers(&runtime, &GQA_ONLY).is_err());
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.gqa_d128.fixture"),
        ContractVersion::new(1, 0),
        runtime
            .descriptor()
            .runtime_implementation_fingerprint
            .clone(),
        runtime.descriptor().id.clone(),
        runtime.descriptor().capabilities.clone(),
    )
    .unwrap();
    let catalog = registry
        .capability_catalog(runtime.descriptor().clone(), vec![engine])
        .unwrap();
    let family = TypedFamilyRegistration::new(GqaFamily)
        .prepare_with_profile(
            &serde_json::to_value(&config).unwrap(),
            &NumericalProfileId::new(TEXT_PROFILE).unwrap(),
        )
        .unwrap();
    let mut options = ProgramPlanCompileOptions::new(config.inputs()).unwrap();
    options
        .insert_tensor_spec(id(ATTENDED), config.spec(true))
        .unwrap();
    options.retain_completion_value(id(ATTENDED));
    let compiled = ProgramPlanCompiler::compile(
        &family,
        &catalog,
        &policy(runtime.as_ref()),
        &registry.planning(),
        &options,
    )
    .unwrap();
    let executable = compiled.executable();
    let resources = no_static(runtime.clone(), executable.execution_plan());
    let query_elements = (config.rows * config.query_columns) as usize;
    let kv_elements = (config.rows * config.kv_columns) as usize;
    let input_bytes = 2 * (query_elements + 2 * kv_elements);
    let output_bytes = 2 * query_elements;
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            input_bytes + output_bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: input_bytes,
                output_bytes,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let mut executor = PreparedTensorExecutor::new(
        executable.clone(),
        registry.bind_plan(executable).unwrap(),
        resources.clone(),
        host.clone(),
        TensorExecutorConfig::default(),
    )
    .unwrap();
    let mut final_output = None;
    for case in cases {
        assert_eq!(case.expected.len(), query_elements);
        assert_eq!(case.inputs[0].len(), query_elements);
        assert_eq!(case.inputs[1].len(), kv_elements);
        assert_eq!(case.inputs[2].len(), kv_elements);
        // Drop the previous completed output before admitting this execution.
        drop(final_output.take());
        let inputs = PORTS
            .into_iter()
            .zip(&case.inputs)
            .enumerate()
            .map(|(i, (name, bits))| {
                managed_input(
                    &host,
                    name,
                    vec![
                        config.rows,
                        if i == 0 {
                            config.query_columns
                        } else {
                            config.kv_columns
                        },
                    ],
                    bits,
                )
            })
            .collect();
        let outputs = executor
            .execute(
                id(format!("run.gqa.{}", case.name)),
                id(format!("request.gqa.{}", case.name)),
                inputs,
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(host.owns(outputs[0].bytes()));
        assert_eq!(outputs[0].bytes().len(), output_bytes);
        for (i, (bytes, expected)) in outputs[0]
            .bytes()
            .bytes()
            .chunks_exact(2)
            .zip(&case.expected)
            .enumerate()
        {
            let actual = u16::from_le_bytes([bytes[0], bytes[1]]);
            if bf16::from_bits(*expected).is_nan() {
                assert!(
                    bf16::from_bits(actual).is_nan(),
                    "{} index {i}: {actual:04x}",
                    case.name
                );
            } else {
                assert_eq!(actual, *expected, "{} index {i}", case.name);
            }
        }
        final_output = Some(outputs);
    }
    drop((executor, compiled, catalog, registry));
    close_resources(resources);
    // Completion Host backing survives device-root retirement. The provider
    // permit can be acquired again without retaining the old plan.
    assert_eq!(host.usage().unwrap().allocated_bytes(), output_bytes);
    drop(cuda_vnext_tensor_providers(&runtime, &GQA_ONLY).unwrap());
    drop(final_output);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
#[ignore = "requires exclusive CUDA; D128 causal GQA analytic arithmetic, masks and ownership"]
fn managed_cuda_gqa_d128_causal_holes_group_mapping_and_ieee() {
    run_cases(GqaConfig::wide(), vec![wide_case(true), wide_case(false)]);
    // Three groups leave the fourth warp of the last CTA inactive. Every live
    // warp still has 32 active lanes. Uniform causal means are exactly 1,2,3.
    run_cases(
        GqaConfig::tail(),
        vec![Case {
            name: "partial_cta",
            inputs: [
                vec![0; 3 * 128],
                vec![0; 3 * 128],
                [0x3f80, 0x4040, 0x40a0]
                    .into_iter()
                    .flat_map(|v| [v; 128])
                    .collect(),
            ],
            expected: [0x3f80, 0x4000, 0x4040]
                .into_iter()
                .flat_map(|v| [v; 128])
                .collect(),
        }],
    );
}

#[test]
fn d128_fixture_uses_analytic_probabilities_and_both_rne_ties() {
    let p = 1.0_f64 / (1.0 + (-1.0_f64).exp());
    assert_eq!(bf16::from_f64(p).to_bits(), 0x3f3b);
    assert_eq!(bf16::from_f64(1.0 - p).to_bits(), 0x3e8a);
    assert_eq!(
        bf16::from_f64((1.0 + 129.0 / 128.0) / 2.0).to_bits(),
        0x3f80
    );
    assert_eq!(
        bf16::from_f64((129.0 / 128.0 + 130.0 / 128.0) / 2.0).to_bits(),
        0x3f82
    );
    assert_ne!(
        expected_wide(0, 2, 0, 0, false),
        expected_wide(0, 2, 4, 0, false)
    );
    assert_ne!(
        expected_wide(0, 2, 0, 0, false),
        expected_wide(0, 2, 1, 0, false)
    );
    for config in [GqaConfig::wide(), GqaConfig::tail()] {
        TypedFamilyRegistration::new(GqaFamily)
            .prepare_with_profile(
                &serde_json::to_value(&config).unwrap(),
                &NumericalProfileId::new(TEXT_PROFILE).unwrap(),
            )
            .unwrap();
    }
}
