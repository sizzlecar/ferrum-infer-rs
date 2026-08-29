//! CUDA parity fixtures for Marlin execution formats used by locked Qwen3.8
//! checkpoints.
//!
//! Run on an sm89 CUDA host with the versioned native-operator lock:
//! `cargo test -p ferrum-kernels --features vllm-marlin --release \
//!   --test compressed_tensors_marlin_eq -- --ignored --nocapture --test-threads=8`

#![cfg(all(feature = "cuda", feature = "vllm-marlin"))]

use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
use ferrum_interfaces::vnext::{
    numeric_weight_quality_authority_implementation_fingerprint, CanonicalRational,
    ContractVersion, WeightMaterializationFidelity, WeightMaterializerSelection,
    NUMERIC_WEIGHT_QUALITY_ARTIFACT_SCHEMA_ID, NUMERIC_WEIGHT_QUALITY_AUTHORITY_ID,
};
use ferrum_kernels::marlin_fp8_materializer::{
    block_fp8_to_marlin_fp8_weight_materializer, BLOCK_FP8_TO_MARLIN_FP8_WEIGHT_MATERIALIZER_ID,
    MARLIN_FP8_QUANTIZATION_FORMAT_ID, MARLIN_FP8_WEIGHT_FORMAT_ID, MARLIN_FP8_WEIGHT_LAYOUT_ID,
};
use ferrum_kernels::marlin_repack::{
    prepare_block_fp8_weight_for_fp8_marlin, repack_compressed_tensors_zero_points_to_marlin,
    repack_gptq_to_marlin, repack_scales_to_marlin,
};
use ferrum_kernels::vllm_marlin::{
    launch_marlin_mm_f16_weight, MarlinF16WeightType, MarlinMmBuffers, MarlinMmExecution,
    MarlinMmF16WeightRequest, MarlinMmProblem,
};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::os::raw::c_void;

const BLOCK_FP8_SOURCE_WEIGHT_FORMAT_ID: &str =
    "weight-format.safetensors.fp8-e4m3-block-grid-inverse-scale";
const RELATIVE_L2_REPORT_DENOMINATOR: u64 = 100_000_000;
const LOCKED_QUALITY_VECTOR_JSON: &str = include_str!(
    "../../../scripts/release/configs/vnext_model_adoption/qwen38_27b_fp8_m3_quality_vector.json"
);
const QUALITY_VECTOR_PAYLOAD_KEYS: [&str; 10] = [
    "schema_version",
    "fixture_id",
    "checkpoint",
    "generator",
    "source_contract",
    "activation_contract",
    "reference_contract",
    "weight_shapes",
    "activation_batches",
    "cases",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumericArtifactAuthority {
    id: String,
    implementation_fingerprint: String,
    version: ContractVersion,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumericArtifactMaterializer {
    fidelity: WeightMaterializationFidelity,
    id: String,
    implementation_fingerprint: String,
    version: ContractVersion,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumericArtifactCheckpoint {
    id: String,
    repository: String,
    revision: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumericArtifactSource {
    weight_format_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumericArtifactExecution {
    quantization_format_ids: Vec<String>,
    weight_format_id: String,
    weight_layout_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumericArtifactContract {
    execution_contract_fingerprint: String,
    quality_vector_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumericArtifactCase {
    actual_f16_bits: Vec<u16>,
    actual_f16le_sha256: String,
    case_id: String,
    inf_count: u64,
    nan_count: u64,
    reference_f32_bits: Vec<u32>,
    reference_f32le_sha256: String,
    relative_l2_upper_bound: CanonicalRational,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumericQualityArtifactV1 {
    authority: NumericArtifactAuthority,
    cases: Vec<NumericArtifactCase>,
    checkpoint: NumericArtifactCheckpoint,
    contract: NumericArtifactContract,
    execution: NumericArtifactExecution,
    materializer: NumericArtifactMaterializer,
    quality_vector_payload: Value,
    schema_id: String,
    source: NumericArtifactSource,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn conservative_relative_l2_upper_bound(relative_l2: f64) -> CanonicalRational {
    assert!(
        relative_l2.is_finite() && relative_l2 >= 0.0,
        "relative L2 must be finite and non-negative"
    );
    let mut numerator = (relative_l2 * RELATIVE_L2_REPORT_DENOMINATOR as f64).ceil() as i64;
    while numerator as f64 / (RELATIVE_L2_REPORT_DENOMINATOR as f64) < relative_l2 {
        numerator += 1;
    }
    CanonicalRational::new(numerator, RELATIVE_L2_REPORT_DENOMINATOR)
        .expect("canonical relative L2 upper bound")
}

fn recursively_sort_json(value: Value) -> Value {
    match value {
        Value::Array(values) => {
            Value::Array(values.into_iter().map(recursively_sort_json).collect())
        }
        Value::Object(values) => {
            let mut entries = values.into_iter().collect::<Vec<_>>();
            entries.sort_by(|(left, _), (right, _)| left.cmp(right));
            Value::Object(
                entries
                    .into_iter()
                    .map(|(key, value)| (key, recursively_sort_json(value)))
                    .collect(),
            )
        }
        scalar => scalar,
    }
}

fn canonical_json_bytes<T: Serialize>(value: &T) -> Vec<u8> {
    let value = serde_json::to_value(value).expect("serialize numeric artifact value");
    serde_json::to_vec(&recursively_sort_json(value)).expect("serialize canonical numeric artifact")
}

fn locked_quality_vector_payload() -> Value {
    let document: Value =
        serde_json::from_str(LOCKED_QUALITY_VECTOR_JSON).expect("parse locked quality vector");
    let root = document
        .as_object()
        .expect("locked quality vector root is an object");
    let payload = QUALITY_VECTOR_PAYLOAD_KEYS
        .into_iter()
        .map(|key| {
            (
                key.to_owned(),
                root.get(key)
                    .unwrap_or_else(|| panic!("locked quality vector is missing `{key}`"))
                    .clone(),
            )
        })
        .collect();
    recursively_sort_json(Value::Object(payload))
}

#[derive(Clone, Copy)]
struct Fixture {
    name: &'static str,
    rows: usize,
    input_features: usize,
    output_features: usize,
}

const GROUP_SIZE: usize = 32;

const FIXTURES: [Fixture; 4] = [
    Fixture {
        name: "asymmetric-packing",
        rows: 3,
        input_features: 128,
        output_features: 64,
    },
    Fixture {
        name: "single-projection",
        rows: 1,
        input_features: 256,
        output_features: 128,
    },
    Fixture {
        name: "fused-qkv-gate-up",
        rows: 2,
        input_features: 256,
        output_features: 256,
    },
    Fixture {
        name: "mixed-dense-linear-attention-segment",
        rows: 4,
        input_features: 128,
        output_features: 192,
    },
];

struct HostFixture {
    input: Vec<f16>,
    qweight_gptq: Vec<i32>,
    scales_grouped: Vec<f16>,
    zero_points_compressed_tensors: Vec<i32>,
    logical_weights: Vec<u8>,
    logical_scales: Vec<f16>,
    logical_zero_points: Vec<u8>,
}

fn build_host_fixture(fixture: Fixture) -> HostFixture {
    let m = fixture.rows;
    let k = fixture.input_features;
    let n = fixture.output_features;
    let groups = k / GROUP_SIZE;
    assert!(k.is_multiple_of(128));
    assert!(n.is_multiple_of(64));

    let input = (0..m * k)
        .map(|index| {
            let centered = ((index * 29 + m * 17) % 257) as f32 - 128.0;
            f16::from_f32(centered / 113.0)
        })
        .collect::<Vec<_>>();
    let logical_weights = (0..k * n)
        .map(|index| ((index * 7 + index / 11 + n / 64) & 0x0f) as u8)
        .collect::<Vec<_>>();
    let logical_scales = (0..groups * n)
        .map(|index| f16::from_f32(0.0125 + ((index * 13 % 17) as f32) * 0.00075))
        .collect::<Vec<_>>();
    let logical_zero_points = (0..groups * n)
        .map(|index| ((index * 5 + index / 9 + 3) & 0x0f) as u8)
        .collect::<Vec<_>>();

    // Marlin weight repacking consumes GPTQ's `[K / 8, N]` word layout.
    // The production source adapter reaches this layout by transposing the
    // checkpoint's compressed-tensors `[N, K / 8]` storage.
    let mut qweight_gptq = vec![0_i32; (k / 8) * n];
    for packed_input in 0..k / 8 {
        for output in 0..n {
            qweight_gptq[packed_input * n + output] = (0..8).fold(0_u32, |word, lane| {
                word | (u32::from(logical_weights[(packed_input * 8 + lane) * n + output])
                    << (lane * 4))
            }) as i32;
        }
    }

    // compressed-tensors packs zero points as `[N / 8, K / G]` with
    // little-endian output-channel nibbles.
    let mut zero_points_compressed_tensors = vec![0_i32; (n / 8) * groups];
    for packed_output in 0..n / 8 {
        for group in 0..groups {
            zero_points_compressed_tensors[packed_output * groups + group] =
                (0..8).fold(0_u32, |word, lane| {
                    word | (u32::from(logical_zero_points[group * n + packed_output * 8 + lane])
                        << (lane * 4))
                }) as i32;
        }
    }

    HostFixture {
        input,
        qweight_gptq,
        scales_grouped: logical_scales.clone(),
        zero_points_compressed_tensors,
        logical_weights,
        logical_scales,
        logical_zero_points,
    }
}

fn cpu_reference(fixture: Fixture, host: &HostFixture) -> Vec<f16> {
    let m = fixture.rows;
    let k = fixture.input_features;
    let n = fixture.output_features;
    let mut output = vec![f16::ZERO; m * n];
    for row in 0..m {
        for output_feature in 0..n {
            let mut sum = 0.0_f32;
            for input_feature in 0..k {
                let group = input_feature / GROUP_SIZE;
                let quantized = host.logical_weights[input_feature * n + output_feature] as i32;
                let zero_point = host.logical_zero_points[group * n + output_feature] as i32;
                let scale = host.logical_scales[group * n + output_feature].to_f32();
                sum += host.input[row * k + input_feature].to_f32()
                    * (quantized - zero_point) as f32
                    * scale;
            }
            output[row * n + output_feature] = f16::from_f32(sum);
        }
    }
    output
}

fn run_fixture(context: &std::sync::Arc<CudaContext>, fixture: Fixture) -> (f64, usize) {
    let stream = context.default_stream();
    let host = build_host_fixture(fixture);
    let reference = cpu_reference(fixture, &host);
    let k = fixture.input_features;
    let n = fixture.output_features;
    let groups = k / GROUP_SIZE;

    let packed_weight = repack_gptq_to_marlin(&host.qweight_gptq, k, n);
    let packed_scales = repack_scales_to_marlin(&host.scales_grouped, k, n, GROUP_SIZE);
    let packed_zero_points = repack_compressed_tensors_zero_points_to_marlin(
        &host.zero_points_compressed_tensors,
        groups,
        n,
    );

    let input_device: CudaSlice<f16> = stream.clone_htod(&host.input).expect("upload input");
    let weight_device: CudaSlice<i32> = stream
        .clone_htod(&packed_weight)
        .expect("upload packed weight");
    let scales_device: CudaSlice<f16> = stream
        .clone_htod(&packed_scales)
        .expect("upload packed scales");
    let zero_points_device: CudaSlice<i32> = stream
        .clone_htod(&packed_zero_points)
        .expect("upload packed zero points");
    let mut output_device: CudaSlice<f16> = stream
        .alloc_zeros(fixture.rows * n)
        .expect("allocate output");
    let sms = context
        .attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .expect("query SM count");
    let workspace: CudaSlice<i32> = stream
        .alloc_zeros(usize::try_from(sms).expect("positive SM count"))
        .expect("allocate workspace");

    {
        let (input_pointer, _input_guard) = input_device.device_ptr(&stream);
        let (weight_pointer, _weight_guard) = weight_device.device_ptr(&stream);
        let (scales_pointer, _scales_guard) = scales_device.device_ptr(&stream);
        let (zero_points_pointer, _zero_points_guard) = zero_points_device.device_ptr(&stream);
        let (output_pointer, _output_guard) = output_device.device_ptr_mut(&stream);
        let (workspace_pointer, _workspace_guard) = workspace.device_ptr(&stream);
        unsafe {
            launch_marlin_mm_f16_weight(MarlinMmF16WeightRequest {
                weight_type: MarlinF16WeightType::U4,
                buffers: MarlinMmBuffers {
                    a: input_pointer as *const c_void,
                    b: weight_pointer as *const c_void,
                    c: output_pointer as *mut c_void,
                    c_tmp: std::ptr::null_mut(),
                    a_scales: std::ptr::null_mut(),
                    b_scales: scales_pointer as *mut c_void,
                    zero_points: zero_points_pointer as *mut c_void,
                    group_index: std::ptr::null_mut(),
                    permutation: std::ptr::null_mut(),
                    a_tmp: std::ptr::null_mut(),
                    workspace: workspace_pointer as *mut c_void,
                },
                problem: MarlinMmProblem {
                    m: fixture.rows as i32,
                    n: n as i32,
                    k: k as i32,
                    lda: k as i32,
                    num_groups: groups as i32,
                    group_size: GROUP_SIZE as i32,
                },
                execution: MarlinMmExecution {
                    device: 0,
                    stream: stream.cu_stream(),
                    sms,
                    has_act_order: false,
                    is_k_full: true,
                    use_atomic_add: false,
                    use_fp32_reduce: false,
                },
            });
        }
        stream.synchronize().expect("Marlin synchronize");
    }
    let actual = stream
        .clone_dtoh(&output_device)
        .expect("download Marlin output");

    let mut reference_squared = 0.0_f64;
    let mut error_squared = 0.0_f64;
    let mut non_finite = 0;
    for (actual, expected) in actual.iter().zip(reference.iter()) {
        let actual = actual.to_f32();
        let expected = expected.to_f32();
        if !actual.is_finite() {
            non_finite += 1;
            continue;
        }
        let error = f64::from(actual - expected);
        reference_squared += f64::from(expected) * f64::from(expected);
        error_squared += error * error;
    }
    let relative_l2 = (error_squared / reference_squared.max(f64::MIN_POSITIVE)).sqrt();
    (relative_l2, non_finite)
}

#[test]
#[ignore = "requires an sm89 CUDA host and the versioned vLLM Marlin native artifact"]
fn qwen38_compressed_tensors_w4a16_matches_cpu_reference_for_four_fixed_fixtures() {
    let context = CudaContext::new(0).expect("CUDA context");
    for fixture in FIXTURES {
        let (relative_l2, non_finite) = run_fixture(&context, fixture);
        eprintln!(
            "QWEN38_CT_MARLIN_FIXTURE name={} rel_err={relative_l2:.8} non_finite={non_finite}",
            fixture.name
        );
        assert_eq!(non_finite, 0, "{} emitted NaN/Inf", fixture.name);
        assert!(
            relative_l2 < 0.05,
            "{} rel_err={relative_l2:.8} exceeds 0.05",
            fixture.name
        );
    }
}

#[test]
#[ignore = "requires an sm89 CUDA host and the versioned vLLM Marlin native artifact"]
fn qwen38_block_fp8_marlin_matches_four_locked_quality_vector_cases() {
    const BLOCK_SHAPE: [usize; 2] = [128, 128];
    const LCG_MULTIPLIER: u64 = 0x5851_f42d_4c95_7f2d;
    const LCG_INCREMENT: u64 = 0x1405_7b7e_f767_814f;
    const ROOT_SEED: u64 = 0x5147_454e_3338_4650;
    const SHAPE_SEED_XOR: u64 = 0x9e37_79b9_7f4a_7c15;
    const WEIGHT_SEED_XOR: u64 = 0x5745_4947_4854_5f31;
    const SCALE_SEED_XOR: u64 = 0x5343_414c_455f_5f31;
    const ACTIVATION_SEED_XOR: u64 = 0x4143_5449_5641_5445;
    const CHANNEL_SCALE_PERMUTATION: [usize; 32] = [
        0, 1, 8, 9, 16, 17, 24, 25, 2, 3, 10, 11, 18, 19, 26, 27, 4, 5, 12, 13, 20, 21, 28, 29, 6,
        7, 14, 15, 22, 23, 30, 31,
    ];
    const CASES: [(&str, usize, usize, usize, usize, &str, &str, &str, &str); 4] = [
        (
            "weight-256x128-batch-1",
            0,
            256,
            128,
            1,
            "bad109739a3d2cc2f6ef463fb2d28adace77215b57c1c7ea121e7e897301154e",
            "1bf6bf27cbd8c14186b0cda8f5a8ed5093e65dafaf9362936f68283332992f85",
            "72d95dc1e2393a03a975d51e3f04cb6059fc8fa7fafa5b7f5285bee77f9dac1d",
            "7c9ce69dd6365a3f04122ba9d718586ddfd1a5d8a1bc88f50d116741041c748c",
        ),
        (
            "weight-256x128-batch-4",
            0,
            256,
            128,
            4,
            "bad109739a3d2cc2f6ef463fb2d28adace77215b57c1c7ea121e7e897301154e",
            "1bf6bf27cbd8c14186b0cda8f5a8ed5093e65dafaf9362936f68283332992f85",
            "330473b51568e047fb583cdc94db872ac7ad308ccb807fb1f8deb099bc006056",
            "a81ae3c8aec953fe4254623a8adb48e2fb06ad754b87eb926299eab91f1fea47",
        ),
        (
            "weight-256x256-batch-1",
            1,
            256,
            256,
            1,
            "2d134465e4591910d21b7b3a8674fbe2b56c3b0c98b184e0d289b98e071d8f92",
            "8d9dcc1756740e90089dd2a7191653d7b4bd97013fcc999a71402f28897cb6b7",
            "3034df7557de9898996f9a480964b86b0b788db52a1d19ff1ca838b29232baa9",
            "d6b7eac6e8c85e42bcf198eee9aed09b3435a0ca94d39484c89406c739db8492",
        ),
        (
            "weight-256x256-batch-4",
            1,
            256,
            256,
            4,
            "2d134465e4591910d21b7b3a8674fbe2b56c3b0c98b184e0d289b98e071d8f92",
            "8d9dcc1756740e90089dd2a7191653d7b4bd97013fcc999a71402f28897cb6b7",
            "e34838c39c331fcd8137ecf1d49c36f9ec7773aafef032cb6992f17885ff6926",
            "1adb773fe9ac46786102e92c40bb84abaace5e4dd1f314c02ddebcdd22d0b24b",
        ),
    ];

    let next = |state: &mut u64| {
        *state = state
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add(LCG_INCREMENT);
        *state
    };
    let stream_seed = |shape_index: usize, stream_xor: u64| {
        let mut state = ROOT_SEED
            ^ (u64::try_from(shape_index + 1)
                .unwrap()
                .wrapping_mul(SHAPE_SEED_XOR))
            ^ stream_xor;
        next(&mut state)
    };
    let materializer = block_fp8_to_marlin_fp8_weight_materializer()
        .expect("construct block-FP8 Marlin materializer");
    let descriptor = materializer.descriptor();
    assert_eq!(
        descriptor.id().as_str(),
        BLOCK_FP8_TO_MARLIN_FP8_WEIGHT_MATERIALIZER_ID
    );
    let quality_contract = descriptor
        .approximate_quality_contract()
        .expect("block-FP8 materializer quality contract");
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let mut artifact_cases = Vec::with_capacity(CASES.len());
    for (
        case_id,
        shape_index,
        n,
        k,
        batch,
        expected_values_sha,
        expected_scales_sha,
        expected_activations_sha,
        expected_reference_sha,
    ) in CASES
    {
        let mut activation_state = stream_seed(shape_index, ACTIVATION_SEED_XOR);
        let input = (0..batch * k)
            .map(|_| {
                let signed = i32::try_from(next(&mut activation_state) % 129).unwrap() - 64;
                f16::from_f32(signed as f32 / 64.0)
            })
            .collect::<Vec<_>>();
        let input_bytes = input
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let scale_rows = n.div_ceil(BLOCK_SHAPE[0]);
        let scale_columns = k.div_ceil(BLOCK_SHAPE[1]);
        let mut scale_state = stream_seed(shape_index, SCALE_SEED_XOR);
        let inverse_scales = (0..scale_rows * scale_columns)
            .map(|_| bf16::from_bits(0x3b80 + 0x20 * (next(&mut scale_state) % 5) as u16))
            .collect::<Vec<_>>();
        let inverse_scale_bytes = inverse_scales
            .iter()
            .flat_map(|scale| scale.to_le_bytes())
            .collect::<Vec<_>>();
        let mut weight_state = stream_seed(shape_index, WEIGHT_SEED_XOR);
        let source_values = (0..n * k)
            .map(|index| {
                let word = next(&mut weight_state);
                if word & 0x0f == 0 {
                    0
                } else {
                    let output_channel = index / k;
                    let exponent_tier = ((output_channel % 8) / 2) as u8;
                    let magnitude = 0x20 + exponent_tier * 8 + ((word >> 9) & 0x07) as u8;
                    let sign = ((word >> 31) & 0x80) as u8;
                    magnitude | sign
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(sha256_hex(&source_values), expected_values_sha, "{case_id}");
        assert_eq!(
            sha256_hex(&inverse_scale_bytes),
            expected_scales_sha,
            "{case_id}"
        );
        assert_eq!(
            sha256_hex(&input_bytes),
            expected_activations_sha,
            "{case_id}"
        );

        let mut reference = vec![0.0_f32; batch * n];
        for row in 0..batch {
            for output in 0..n {
                let mut sum = 0.0_f32;
                for input_feature in 0..k {
                    let source =
                        float8::F8E4M3::from_bits(source_values[output * k + input_feature])
                            .to_f32();
                    let scale_index =
                        (output / BLOCK_SHAPE[0]) * scale_columns + input_feature / BLOCK_SHAPE[1];
                    let decoded_weight = source * inverse_scales[scale_index].to_f32();
                    let product = input[row * k + input_feature].to_f32() * decoded_weight;
                    sum += product;
                }
                reference[row * n + output] = sum;
            }
        }
        let reference_bytes = reference
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(
            sha256_hex(&reference_bytes),
            expected_reference_sha,
            "{case_id}"
        );

        let prepared = prepare_block_fp8_weight_for_fp8_marlin(
            &source_values,
            &inverse_scale_bytes,
            n,
            k,
            BLOCK_SHAPE,
        )
        .expect("prepare block-FP8 Marlin weight");
        let (packed_weight, packed_scales) = prepared.into_parts();

        // Canary for the exact class of bug this fixture is meant to expose:
        // channel-wise weights are quantized correctly, but logical scales are
        // uploaded in identity order instead of Marlin's 32-channel order.
        // The fixture's four power-of-two channel tiers are deliberately not
        // invariant under CHANNEL_SCALE_PERMUTATION.
        let mut quantization_scales = vec![0.0_f32; n];
        let mut logical_runtime_scales = vec![0.0_f32; n];
        for output in 0..n {
            let mut maximum = 0.0_f32;
            for input_feature in 0..k {
                let source =
                    float8::F8E4M3::from_bits(source_values[output * k + input_feature]).to_f32();
                let scale_index =
                    (output / BLOCK_SHAPE[0]) * scale_columns + input_feature / BLOCK_SHAPE[1];
                maximum = maximum.max((source * inverse_scales[scale_index].to_f32()).abs());
            }
            let scale = maximum / 448.0;
            quantization_scales[output] = scale;
            logical_runtime_scales[output] = f16::from_f32(scale * 256.0).to_f32() / 256.0;
        }

        let mut identity_scale_for_output = vec![0.0_f32; n];
        for chunk_start in (0..n).step_by(CHANNEL_SCALE_PERMUTATION.len()) {
            for (destination, source) in CHANNEL_SCALE_PERMUTATION.iter().copied().enumerate() {
                let expected = f16::from_f32(quantization_scales[chunk_start + source] * 256.0);
                assert_eq!(
                    packed_scales[chunk_start + destination].to_bits(),
                    expected.to_bits(),
                    "{case_id} prepared scale permutation drifted"
                );
                identity_scale_for_output[chunk_start + source] =
                    logical_runtime_scales[chunk_start + destination];
            }
        }
        assert_ne!(
            identity_scale_for_output, logical_runtime_scales,
            "{case_id} fixture does not distinguish identity scale layout"
        );

        let mut identity_scale_output = vec![0.0_f32; batch * n];
        for row in 0..batch {
            for output in 0..n {
                let mut sum = 0.0_f32;
                for input_feature in 0..k {
                    let source =
                        float8::F8E4M3::from_bits(source_values[output * k + input_feature])
                            .to_f32();
                    let scale_index =
                        (output / BLOCK_SHAPE[0]) * scale_columns + input_feature / BLOCK_SHAPE[1];
                    let decoded = source * inverse_scales[scale_index].to_f32();
                    let quantized = if quantization_scales[output] == 0.0 {
                        0.0
                    } else {
                        float8::F8E4M3::from_f32(decoded / quantization_scales[output]).to_f32()
                    };
                    sum += input[row * k + input_feature].to_f32()
                        * quantized
                        * identity_scale_for_output[output];
                }
                identity_scale_output[row * n + output] = sum;
            }
        }
        let identity_error_squared = identity_scale_output
            .iter()
            .zip(reference.iter())
            .map(|(actual, expected)| {
                let error = f64::from(actual - expected);
                error * error
            })
            .sum::<f64>();
        let reference_squared = reference
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>();
        let identity_relative_l2 =
            identity_error_squared.sqrt() / reference_squared.sqrt().max(1.0e-6);
        eprintln!(
            "QWEN38_BLOCK_FP8_IDENTITY_SCALE_CANARY name={case_id} rel_err={identity_relative_l2:.8}"
        );
        assert!(
            identity_relative_l2 > 0.05,
            "{case_id} identity-scale canary rel_err={identity_relative_l2:.8} did not exceed 0.05"
        );

        let input_device: CudaSlice<f16> = stream.clone_htod(&input).expect("upload input");
        let weight_device: CudaSlice<u8> = stream
            .clone_htod(&packed_weight)
            .expect("upload packed weight");
        let scales_device: CudaSlice<f16> = stream
            .clone_htod(&packed_scales)
            .expect("upload packed scales");
        let mut output_device: CudaSlice<f16> =
            stream.alloc_zeros(batch * n).expect("allocate output");
        let sms = context
            .attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .expect("query SM count");
        let workspace: CudaSlice<i32> = stream
            .alloc_zeros(usize::try_from(sms).expect("positive SM count"))
            .expect("allocate workspace");

        {
            let (input_pointer, _input_guard) = input_device.device_ptr(&stream);
            let (weight_pointer, _weight_guard) = weight_device.device_ptr(&stream);
            let (scales_pointer, _scales_guard) = scales_device.device_ptr(&stream);
            let (output_pointer, _output_guard) = output_device.device_ptr_mut(&stream);
            let (workspace_pointer, _workspace_guard) = workspace.device_ptr(&stream);
            unsafe {
                launch_marlin_mm_f16_weight(MarlinMmF16WeightRequest {
                    weight_type: MarlinF16WeightType::E4M3Fn,
                    buffers: MarlinMmBuffers {
                        a: input_pointer as *const c_void,
                        b: weight_pointer as *const c_void,
                        c: output_pointer as *mut c_void,
                        c_tmp: std::ptr::null_mut(),
                        a_scales: std::ptr::null_mut(),
                        b_scales: scales_pointer as *mut c_void,
                        zero_points: std::ptr::null_mut(),
                        group_index: std::ptr::null_mut(),
                        permutation: std::ptr::null_mut(),
                        a_tmp: std::ptr::null_mut(),
                        workspace: workspace_pointer as *mut c_void,
                    },
                    problem: MarlinMmProblem {
                        m: batch as i32,
                        n: n as i32,
                        k: k as i32,
                        lda: k as i32,
                        num_groups: 1,
                        group_size: -1,
                    },
                    execution: MarlinMmExecution {
                        device: 0,
                        stream: stream.cu_stream(),
                        sms,
                        has_act_order: false,
                        is_k_full: true,
                        use_atomic_add: false,
                        use_fp32_reduce: false,
                    },
                });
            }
            stream.synchronize().expect("Marlin synchronize");
        }

        let actual = stream
            .clone_dtoh(&output_device)
            .expect("download Marlin output");
        let mut reference_squared = 0.0_f64;
        let mut error_squared = 0.0_f64;
        let mut nan_count = 0_usize;
        let mut infinity_count = 0_usize;
        for (actual, expected) in actual.iter().zip(reference.iter().copied()) {
            let actual = actual.to_f32();
            reference_squared += f64::from(expected) * f64::from(expected);
            if actual.is_nan() {
                nan_count += 1;
            } else if actual.is_infinite() {
                infinity_count += 1;
            } else {
                let error = f64::from(actual - expected);
                error_squared += error * error;
            }
        }
        let relative_l2 = if nan_count == 0 && infinity_count == 0 {
            error_squared.sqrt() / reference_squared.sqrt().max(1.0e-6)
        } else {
            f64::INFINITY
        };
        eprintln!(
            "QWEN38_BLOCK_FP8_MARLIN_FIXTURE name={case_id} rel_err={relative_l2:.8} nan_count={nan_count} infinity_count={infinity_count}"
        );

        assert_eq!(nan_count, 0, "{case_id} emitted NaN");
        assert_eq!(infinity_count, 0, "{case_id} emitted Inf");
        assert!(
            relative_l2 <= 0.05,
            "{case_id} rel_err={relative_l2:.8} exceeds 0.05"
        );

        let actual_bytes = actual
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        artifact_cases.push(NumericArtifactCase {
            actual_f16_bits: actual.iter().map(|value| value.to_bits()).collect(),
            actual_f16le_sha256: sha256_hex(&actual_bytes),
            case_id: case_id.to_owned(),
            inf_count: u64::try_from(infinity_count).expect("Inf count fits u64"),
            nan_count: u64::try_from(nan_count).expect("NaN count fits u64"),
            reference_f32_bits: reference.iter().map(|value| value.to_bits()).collect(),
            reference_f32le_sha256: sha256_hex(&reference_bytes),
            relative_l2_upper_bound: conservative_relative_l2_upper_bound(relative_l2),
        });
    }

    assert_eq!(
        artifact_cases.len(),
        usize::try_from(quality_contract.required_case_count()).expect("case count fits usize")
    );
    let quality_vector_payload = locked_quality_vector_payload();
    assert_eq!(
        sha256_hex(&canonical_json_bytes(&quality_vector_payload)),
        quality_contract.quality_vector_digest(),
        "locked quality vector payload digest drifted"
    );
    let artifact = NumericQualityArtifactV1 {
        authority: NumericArtifactAuthority {
            id: NUMERIC_WEIGHT_QUALITY_AUTHORITY_ID.to_owned(),
            implementation_fingerprint:
                numeric_weight_quality_authority_implementation_fingerprint()
                    .expect("numeric quality authority implementation fingerprint"),
            version: ContractVersion::new(1, 0),
        },
        cases: artifact_cases,
        checkpoint: NumericArtifactCheckpoint {
            id: "qwen38-27b-fp8".to_owned(),
            repository: "Qwen/Qwen3.8-27B-FP8".to_owned(),
            revision: "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a".to_owned(),
        },
        contract: NumericArtifactContract {
            execution_contract_fingerprint: quality_contract
                .execution_contract_fingerprint()
                .to_owned(),
            quality_vector_digest: quality_contract.quality_vector_digest().to_owned(),
        },
        execution: NumericArtifactExecution {
            quantization_format_ids: vec![MARLIN_FP8_QUANTIZATION_FORMAT_ID.to_owned()],
            weight_format_id: MARLIN_FP8_WEIGHT_FORMAT_ID.to_owned(),
            weight_layout_id: MARLIN_FP8_WEIGHT_LAYOUT_ID.to_owned(),
        },
        materializer: NumericArtifactMaterializer {
            fidelity: descriptor.fidelity(),
            id: descriptor.id().as_str().to_owned(),
            implementation_fingerprint: descriptor.implementation_fingerprint().to_owned(),
            version: descriptor.version(),
        },
        quality_vector_payload,
        schema_id: NUMERIC_WEIGHT_QUALITY_ARTIFACT_SCHEMA_ID.to_owned(),
        source: NumericArtifactSource {
            weight_format_id: BLOCK_FP8_SOURCE_WEIGHT_FORMAT_ID.to_owned(),
        },
    };
    let canonical_json = canonical_json_bytes(&artifact);
    let decoded: NumericQualityArtifactV1 =
        serde_json::from_slice(&canonical_json).expect("parse canonical numeric artifact");
    assert_eq!(
        decoded, artifact,
        "numeric artifact typed roundtrip drifted"
    );
    assert_eq!(
        canonical_json_bytes(&decoded),
        canonical_json,
        "numeric artifact canonical JSON drifted after roundtrip"
    );
    let selection = WeightMaterializerSelection::numeric_quality_artifact(
        descriptor.id().clone(),
        canonical_json.clone(),
    )
    .expect("typed numeric quality artifact selection");
    assert_eq!(selection.materializer_id(), descriptor.id());
    assert!(selection.has_numeric_quality_artifact());
    let artifact_sha256 = sha256_hex(&canonical_json);
    let artifact_bytes = canonical_json.len();
    let artifact_json =
        std::str::from_utf8(&canonical_json).expect("canonical artifact is valid UTF-8");
    eprintln!(
        "QWEN38_BLOCK_FP8_NUMERIC_ARTIFACT_V1 sha256={artifact_sha256} bytes={artifact_bytes} json={artifact_json}"
    );
}
