//! CUDA parity fixtures for Marlin execution formats used by locked Qwen3.8
//! checkpoints.
//!
//! Run on an sm89 CUDA host with the versioned native-operator lock:
//! `cargo test -p ferrum-kernels --features vllm-marlin --release \
//!   --test compressed_tensors_marlin_eq -- --ignored --nocapture --test-threads=8`

#![cfg(all(feature = "cuda", feature = "vllm-marlin"))]

use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
use ferrum_interfaces::vnext::{CanonicalRational, ContractVersion, WeightMaterializationFidelity};
use ferrum_kernels::marlin_fp8_materializer::{
    block_fp8_to_marlin_fp8_weight_materializer, BLOCK_FP8_TO_MARLIN_FP8_WEIGHT_MATERIALIZER_ID,
    MARLIN_FP8_GROUP128_QUANTIZATION_FORMAT_ID, MARLIN_FP8_GROUP128_WEIGHT_FORMAT_ID,
    MARLIN_FP8_GROUP128_WEIGHT_LAYOUT_ID,
};
use ferrum_kernels::marlin_repack::{
    block_fp8_group128_scales_to_marlin_f16_reference,
    repack_compressed_tensors_zero_points_to_marlin, repack_gptq_to_marlin,
    repack_scales_to_marlin,
};
use ferrum_kernels::vllm_marlin::{
    launch_block_fp8_group128_repack, launch_block_fp8_group128_scales,
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
const BLOCK_FP8_EXACT_PARITY_ARTIFACT_SCHEMA_ID: &str =
    "validation.weight-materializer.exact-parity.v1";
const BLOCK_FP8_QUALITY_VECTOR_DIGEST: &str =
    "4c8b44a6a6e2ca803f6a3916b033a50a8a007cb2452a0e9246ed6c7f3cacbb51";
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
struct ExactParityArtifactV1 {
    cases: Vec<NumericArtifactCase>,
    checkpoint: NumericArtifactCheckpoint,
    execution: NumericArtifactExecution,
    materializer: NumericArtifactMaterializer,
    quality_vector_payload: Value,
    quality_vector_digest: String,
    schema_id: String,
    source: NumericArtifactSource,
}

const GEMMA4_NUMERIC_ARTIFACT_SCHEMA_ID: &str =
    "validation.gemma4-compressed-tensors-symmetric-marlin.exact-parity.v1";
const GEMMA4_NUMERIC_MATERIALIZER_ID: &str =
    "weight-materializer.cuda.compressed-tensors-int4-symmetric-to-marlin";
const GEMMA4_NUMERIC_QUANTIZATION_FORMAT_ID: &str =
    "quantization.marlin.compressed-tensors-int4-symmetric";
const GEMMA4_NUMERIC_WEIGHT_FORMAT_ID: &str =
    "weight-format.safetensors.compressed-tensors-marlin-int4-symmetric";
const GEMMA4_NUMERIC_WEIGHT_LAYOUT_ID: &str =
    "weight-layout.gemma4_unified.text.compressed_tensors_marlin_symmetric";
const GEMMA4_REFERENCE_SEMANTICS: &str =
    "source signed INT4 values decoded with locked BF16 group-32 scales and multiplied by locked F16 activations in F32 accumulation order";
// Locked from the first bounded CUDA diagnostic after the fixture generator,
// source hashes, and reference semantics were frozen.
const GEMMA4_QUALITY_VECTOR_DIGEST: &str =
    "4da1220e3be4163e88e28d9c428829572c6ea508ff513801842bf7f7ba2fa91a";
const GEMMA4_LOCKED_CASE_HASHES: [(&str, &str, &str, &str, &str); 4] = [
    (
        "attention-projection-batch-1",
        "94a5b2d4f873a3fc51f78e0de31216cd15453186f04533aa019a98beca3e567c",
        "c0d1be12357545afe4ae031f1368bcf6344cd41e54cadef02ccca5aad88fec3a",
        "a1acae30a2ea2cb6bc314e7a2bd5ac38792009828b436809ec7a2c72a6a01f5e",
        "0ce762c1a380bce24a3c4f6d630cd2792b2470ad476a9223cbeb29649510605e",
    ),
    (
        "attention-projection-batch-4",
        "94a5b2d4f873a3fc51f78e0de31216cd15453186f04533aa019a98beca3e567c",
        "c0d1be12357545afe4ae031f1368bcf6344cd41e54cadef02ccca5aad88fec3a",
        "04fdaf18dbab3e6102855eb53f827709007aec569d2eb43230bc821cf8e2d7e7",
        "54e8101a95943bb7be573bf2833c8c6d4de2ca0470bcda6cd51df1f0b0d17d78",
    ),
    (
        "mlp-down-projection-batch-1",
        "ae71bdb9c1ac20134883688ce9f2e6abd0197744c276d42e5da5b30ef5953124",
        "c26d234b528374d04d22681a01fbe4ecc8b5aa413bcdaf7a3cc86d98e2f73887",
        "ae3dac36c9aecb6ec2ab6cd87616ea447c24da708a7ee7ee1424a809bef731bb",
        "b2a43aa504886c803a97422375ec497bea2d5f9d483f571188c5df50ef048972",
    ),
    (
        "mlp-down-projection-batch-4",
        "ae71bdb9c1ac20134883688ce9f2e6abd0197744c276d42e5da5b30ef5953124",
        "c26d234b528374d04d22681a01fbe4ecc8b5aa413bcdaf7a3cc86d98e2f73887",
        "805ec81fbdedc96fec0c16f7910495a8b557f6ce8ec9e4575cbf7716182f1dc5",
        "67a7fbb6bdd06c402e92033be709262bcc2de66c0cace1c6464850a653f03e95",
    ),
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gemma4NumericArtifactCase {
    activation_batch: u64,
    activation_f16le_sha256: String,
    actual_f16le_sha256: String,
    case_id: String,
    inf_count: u64,
    nan_count: u64,
    reference_f32le_sha256: String,
    relative_l2_upper_bound: CanonicalRational,
    source_packed_i32le_sha256: String,
    source_scales_bf16le_sha256: String,
    weight_shape: [u64; 2],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gemma4ExactParityArtifactV1 {
    cases: Vec<Gemma4NumericArtifactCase>,
    checkpoint: NumericArtifactCheckpoint,
    execution: NumericArtifactExecution,
    materializer: NumericArtifactMaterializer,
    quality_vector_digest: String,
    quality_vector_payload: Value,
    schema_id: String,
    source: NumericArtifactSource,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Pack exact row-major E4M3 bytes from `[N, K]` into the final 16-by-64
/// Marlin W8A16 tile ABI. This test-only oracle is deliberately independent
/// of both the product CUDA transform and its staging-layout reference.
fn block_fp8_group128_raw_bits_to_final_marlin_u32_reference(
    source_fp8_e4m3: &[u8],
    n: usize,
    k: usize,
) -> Vec<u32> {
    assert_eq!(n % 128, 0, "group-128 output dimension");
    assert_eq!(k % 128, 0, "group-128 input dimension");
    assert_eq!(source_fp8_e4m3.len(), n * k, "group-128 value count");

    let mut packed = Vec::with_capacity(n * k / 4);
    for k_tile in 0..k / 16 {
        for n_tile in 0..n / 64 {
            for marlin_thread in 0..32 {
                let tensor_core_column = marlin_thread / 4;
                let tensor_core_row = (marlin_thread % 4) * 2;
                for warp in 0..4 {
                    for column_half in 0..2 {
                        let output = n_tile * 64 + warp * 16 + tensor_core_column + column_half * 8;
                        let source_base = output * k + k_tile * 16;
                        packed.push(u32::from_le_bytes([
                            source_fp8_e4m3[source_base + tensor_core_row],
                            source_fp8_e4m3[source_base + tensor_core_row + 8],
                            source_fp8_e4m3[source_base + tensor_core_row + 1],
                            source_fp8_e4m3[source_base + tensor_core_row + 9],
                        ]));
                    }
                }
            }
        }
    }
    assert_eq!(packed.len(), n * k / 4, "final Marlin word count");
    packed
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

fn gemma4_quality_vector_payload() -> Value {
    recursively_sort_json(serde_json::json!({
        "schema_version": 1,
        "fixture_id": "gemma4-12b-w4a16-ct-symmetric-marlin-v1",
        "checkpoint": {
            "repository": "google/gemma-4-12B-it-qat-w4a16-ct",
            "revision": "1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee",
        },
        "generator": {
            "algorithm": "coordinate-splitmix64-v1",
            "weight_seed_hex": format!("{GEMMA4_WEIGHT_SEED:016x}"),
            "scale_seed_hex": format!("{GEMMA4_SCALE_SEED:016x}"),
            "input_seed_hex": format!("{GEMMA4_INPUT_SEED:016x}"),
        },
        "source_contract": {
            "format": "compressed-tensors",
            "quantization": "signed-int4-symmetric",
            "group_size": GROUP_SIZE,
            "packing": "I32[N,K/8], q+8, low-input-lane-first",
            "scale_dtype": "BF16",
            "zero_point": null,
        },
        "activation_contract": {
            "dtype": "F16",
            "batches": [1, 4],
        },
        "reference_semantics": GEMMA4_REFERENCE_SEMANTICS,
        "weight_shapes": GEMMA4_SYMMETRIC_SHAPES
            .iter()
            .map(|shape| [shape.output_features, shape.input_features])
            .collect::<Vec<_>>(),
    }))
}

fn gemma4_materializer_implementation_fingerprint() -> String {
    let mut digest = Sha256::new();
    digest.update(GEMMA4_NUMERIC_MATERIALIZER_ID.as_bytes());
    digest.update(include_bytes!("../src/marlin_repack.rs"));
    digest.update(include_bytes!("../src/backend/cuda/vllm_marlin.rs"));
    digest.update(include_bytes!(
        "../src/backend/cuda/vnext_ops/transformer/moe_weights.rs"
    ));
    digest.update(include_bytes!(
        "../../ferrum-quantization/src/compressed_tensors_marlin_source.rs"
    ));
    format!("{:x}", digest.finalize())
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

#[derive(Clone, Copy)]
struct Gemma4SymmetricShape {
    name: &'static str,
    output_features: usize,
    input_features: usize,
}

struct Gemma4SymmetricMarlinWeight {
    logical_scales: Vec<f16>,
    packed_scales: Vec<f16>,
    packed_weight: Vec<i32>,
    source_packed_i32le_sha256: String,
    source_scales_bf16le_sha256: String,
}

struct Gemma4SymmetricRun {
    activation_f16le_sha256: String,
    actual_f16le_sha256: String,
    inf_count: usize,
    nan_count: usize,
    reference_f32le_sha256: String,
    relative_l2: f64,
}

const GEMMA4_SYMMETRIC_SHAPES: [Gemma4SymmetricShape; 2] = [
    Gemma4SymmetricShape {
        name: "attention-projection",
        output_features: 4_096,
        input_features: 3_840,
    },
    Gemma4SymmetricShape {
        name: "mlp-down-projection",
        output_features: 3_840,
        input_features: 15_360,
    },
];

const GEMMA4_WEIGHT_SEED: u64 = 0x4745_4d4d_4134_5734;
const GEMMA4_SCALE_SEED: u64 = 0x4745_4d4d_4134_5334;
const GEMMA4_INPUT_SEED: u64 = 0x4745_4d4d_4134_4934;

fn gemma4_mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn gemma4_coordinate_hash(seed: u64, shape_index: usize, first: usize, second: usize) -> u64 {
    gemma4_mix64(
        seed ^ u64::try_from(shape_index + 1)
            .expect("shape index fits u64")
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ u64::try_from(first + 1)
                .expect("first coordinate fits u64")
                .wrapping_mul(0xd1b5_4a32_d192_ed03)
            ^ u64::try_from(second + 1)
                .expect("second coordinate fits u64")
                .wrapping_mul(0x8cb9_2baa_5f3d_8dd7),
    )
}

fn gemma4_symmetric_quantized_value(
    shape_index: usize,
    output_feature: usize,
    input_feature: usize,
) -> i32 {
    let code = gemma4_coordinate_hash(
        GEMMA4_WEIGHT_SEED,
        shape_index,
        output_feature,
        input_feature,
    ) & 0x0f;
    i32::try_from(code).expect("INT4 code fits i32") - 8
}

fn gemma4_symmetric_source_scale(shape_index: usize, output_feature: usize, group: usize) -> bf16 {
    let numerator = 1
        + (gemma4_coordinate_hash(GEMMA4_SCALE_SEED, shape_index, output_feature, group) & 0x1f)
            as usize;
    // These are exact powers-of-two fractions. Passing through BF16 first
    // mirrors Gemma 4's checkpoint dtype and the source adapter's BF16->F16
    // transcode without introducing a second reference convention.
    bf16::from_f32(numerator as f32 / 512.0)
}

fn gemma4_symmetric_input(batch: usize, input_features: usize) -> Vec<f16> {
    (0..batch)
        .flat_map(|row| {
            (0..input_features).map(move |input_feature| {
                let bits = gemma4_coordinate_hash(GEMMA4_INPUT_SEED, 0, row, input_feature);
                let centered =
                    i32::try_from(bits % 127).expect("input fixture value fits i32") - 63;
                f16::from_f32(centered as f32 / 64.0)
            })
        })
        .collect()
}

fn gemma4_symmetric_marlin_weight(
    shape_index: usize,
    shape: Gemma4SymmetricShape,
) -> Gemma4SymmetricMarlinWeight {
    let n = shape.output_features;
    let k = shape.input_features;
    assert!(k.is_multiple_of(128), "Gemma 4 K must be Marlin aligned");
    assert!(n.is_multiple_of(64), "Gemma 4 N must be Marlin aligned");
    assert!(k.is_multiple_of(GROUP_SIZE), "Gemma 4 group-32 K");

    // Build the checkpoint's exact compressed-tensors source layout:
    // I32[N, K/8], low input lane in the low nibble, signed q stored as q+8.
    let mut checkpoint_words = vec![0_i32; n * (k / 8)];
    for output in 0..n {
        for packed_input in 0..k / 8 {
            checkpoint_words[output * (k / 8) + packed_input] = (0..8).fold(0_u32, |word, lane| {
                let quantized =
                    gemma4_symmetric_quantized_value(shape_index, output, packed_input * 8 + lane);
                let biased_code =
                    u32::try_from(quantized + 8).expect("symmetric signed INT4 maps to code 0..15");
                word | (biased_code << (lane * 4))
            }) as i32;
        }
    }
    let mut source_packed_digest = Sha256::new();
    for word in &checkpoint_words {
        source_packed_digest.update(word.to_le_bytes());
    }
    let source_packed_i32le_sha256 = format!("{:x}", source_packed_digest.finalize());

    // The product source adapter performs this transpose before the common
    // Marlin tile permutation. No zero-point tensor or second bias exists.
    let mut gptq_words = vec![0_i32; checkpoint_words.len()];
    for output in 0..n {
        for packed_input in 0..k / 8 {
            gptq_words[packed_input * n + output] =
                checkpoint_words[output * (k / 8) + packed_input];
        }
    }
    let packed_weight = repack_gptq_to_marlin(&gptq_words, k, n);

    let groups = k / GROUP_SIZE;
    let mut logical_scales = vec![f16::ZERO; groups * n];
    let mut source_scales = vec![bf16::ZERO; n * groups];
    for output in 0..n {
        for group in 0..groups {
            let source_scale = gemma4_symmetric_source_scale(shape_index, output, group);
            source_scales[output * groups + group] = source_scale;
            logical_scales[group * n + output] = f16::from_f32(source_scale.to_f32());
        }
    }
    let mut source_scales_digest = Sha256::new();
    for scale in &source_scales {
        source_scales_digest.update(scale.to_le_bytes());
    }
    let source_scales_bf16le_sha256 = format!("{:x}", source_scales_digest.finalize());
    let packed_scales = repack_scales_to_marlin(&logical_scales, k, n, GROUP_SIZE);
    Gemma4SymmetricMarlinWeight {
        logical_scales,
        packed_scales,
        packed_weight,
        source_packed_i32le_sha256,
        source_scales_bf16le_sha256,
    }
}

fn gemma4_symmetric_cpu_reference(
    shape_index: usize,
    shape: Gemma4SymmetricShape,
    batch: usize,
    input: &[f16],
    logical_scales: &[f16],
) -> Vec<f32> {
    let n = shape.output_features;
    let k = shape.input_features;
    assert_eq!(logical_scales.len(), (k / GROUP_SIZE) * n);

    let mut reference = vec![0.0_f32; batch * n];
    for row in 0..batch {
        for output in 0..n {
            let mut sum = 0.0_f32;
            for input_feature in 0..k {
                let quantized =
                    gemma4_symmetric_quantized_value(shape_index, output, input_feature);
                let scale = logical_scales[(input_feature / GROUP_SIZE) * n + output].to_f32();
                sum += input[row * k + input_feature].to_f32() * quantized as f32 * scale;
            }
            reference[row * n + output] = sum;
        }
    }
    reference
}

fn run_gemma4_symmetric_fixture(
    context: &std::sync::Arc<CudaContext>,
    shape_index: usize,
    shape: Gemma4SymmetricShape,
    batch: usize,
    weight: &Gemma4SymmetricMarlinWeight,
) -> Gemma4SymmetricRun {
    let stream = context.default_stream();
    let n = shape.output_features;
    let k = shape.input_features;
    let groups = k / GROUP_SIZE;
    let input = gemma4_symmetric_input(batch, k);
    let reference =
        gemma4_symmetric_cpu_reference(shape_index, shape, batch, &input, &weight.logical_scales);

    let input_device: CudaSlice<f16> = stream.clone_htod(&input).expect("upload Gemma 4 input");
    let weight_device: CudaSlice<i32> = stream
        .clone_htod(&weight.packed_weight)
        .expect("upload Gemma 4 U4B8 weight");
    let scales_device: CudaSlice<f16> = stream
        .clone_htod(&weight.packed_scales)
        .expect("upload Gemma 4 group-32 scales");
    let mut output_device: CudaSlice<f16> = stream
        .alloc_zeros(batch * n)
        .expect("allocate Gemma 4 output");
    let sms = context
        .attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .expect("query SM count");
    let workspace: CudaSlice<i32> = stream
        .alloc_zeros(usize::try_from(sms).expect("positive SM count"))
        .expect("allocate Gemma 4 Marlin workspace");

    {
        let (input_pointer, _input_guard) = input_device.device_ptr(&stream);
        let (weight_pointer, _weight_guard) = weight_device.device_ptr(&stream);
        let (scales_pointer, _scales_guard) = scales_device.device_ptr(&stream);
        let (output_pointer, _output_guard) = output_device.device_ptr_mut(&stream);
        let (workspace_pointer, _workspace_guard) = workspace.device_ptr(&stream);
        unsafe {
            launch_marlin_mm_f16_weight(MarlinMmF16WeightRequest {
                weight_type: MarlinF16WeightType::U4B8,
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
                    m: i32::try_from(batch).expect("batch fits i32"),
                    n: i32::try_from(n).expect("N fits i32"),
                    k: i32::try_from(k).expect("K fits i32"),
                    lda: i32::try_from(k).expect("LDA fits i32"),
                    num_groups: i32::try_from(groups).expect("group count fits i32"),
                    group_size: i32::try_from(GROUP_SIZE).expect("group size fits i32"),
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
        stream.synchronize().expect("Gemma 4 Marlin synchronize");
    }

    let actual = stream
        .clone_dtoh(&output_device)
        .expect("download Gemma 4 output");
    let mut reference_squared = 0.0_f64;
    let mut error_squared = 0.0_f64;
    let mut nan_count = 0_usize;
    let mut inf_count = 0_usize;
    for (actual, expected) in actual.iter().zip(reference.iter().copied()) {
        let actual = actual.to_f32();
        reference_squared += f64::from(expected) * f64::from(expected);
        if actual.is_nan() {
            nan_count += 1;
        } else if actual.is_infinite() {
            inf_count += 1;
        } else {
            let error = f64::from(actual - expected);
            error_squared += error * error;
        }
    }
    let relative_l2 = if nan_count == 0 && inf_count == 0 {
        error_squared.sqrt() / reference_squared.sqrt().max(1.0e-6)
    } else {
        f64::INFINITY
    };
    let activation_f16le_sha256 = sha256_hex(
        &input
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let reference_f32le_sha256 = sha256_hex(
        &reference
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let actual_f16le_sha256 = sha256_hex(
        &actual
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    Gemma4SymmetricRun {
        activation_f16le_sha256,
        actual_f16le_sha256,
        inf_count,
        nan_count,
        reference_f32le_sha256,
        relative_l2,
    }
}

#[test]
#[ignore = "requires an sm89 CUDA host and the versioned vLLM Marlin native artifact"]
fn gemma4_symmetric_compressed_tensors_w4a16_matches_two_shapes_and_batches_1_4() {
    let context = CudaContext::new(0).expect("CUDA context");
    let quality_vector_payload = gemma4_quality_vector_payload();
    let quality_vector_digest = sha256_hex(&canonical_json_bytes(&quality_vector_payload));
    let implementation_fingerprint = gemma4_materializer_implementation_fingerprint();
    let mut artifact_cases = Vec::with_capacity(4);
    for (shape_index, shape) in GEMMA4_SYMMETRIC_SHAPES.into_iter().enumerate() {
        let weight = gemma4_symmetric_marlin_weight(shape_index, shape);
        for batch in [1_usize, 4] {
            let case_id = format!("{}-batch-{batch}", shape.name);
            let result = run_gemma4_symmetric_fixture(&context, shape_index, shape, batch, &weight);
            eprintln!(
                "GEMMA4_CT_MARLIN_CASE case_id={case_id} weight_shape=[{},{}] activation_batch={} rel_err={:.8} nan_count={} inf_count={} source_packed_sha256={} source_scales_sha256={} activation_sha256={} reference_sha256={} actual_sha256={} quality_vector_digest={quality_vector_digest} implementation_fingerprint={implementation_fingerprint}",
                shape.output_features,
                shape.input_features,
                batch,
                result.relative_l2,
                result.nan_count,
                result.inf_count,
                weight.source_packed_i32le_sha256,
                weight.source_scales_bf16le_sha256,
                result.activation_f16le_sha256,
                result.reference_f32le_sha256,
                result.actual_f16le_sha256,
            );
            assert_eq!(
                result.nan_count, 0,
                "{} batch={batch} emitted NaN",
                shape.name
            );
            assert_eq!(
                result.inf_count, 0,
                "{} batch={batch} emitted Inf",
                shape.name
            );
            assert!(
                result.relative_l2 <= 0.05,
                "{} batch={batch} rel_err={:.8} exceeds 0.05",
                shape.name,
                result.relative_l2,
            );
            artifact_cases.push(Gemma4NumericArtifactCase {
                activation_batch: u64::try_from(batch).expect("batch fits u64"),
                activation_f16le_sha256: result.activation_f16le_sha256,
                actual_f16le_sha256: result.actual_f16le_sha256,
                case_id,
                inf_count: u64::try_from(result.inf_count).expect("Inf count fits u64"),
                nan_count: u64::try_from(result.nan_count).expect("NaN count fits u64"),
                reference_f32le_sha256: result.reference_f32le_sha256,
                relative_l2_upper_bound: conservative_relative_l2_upper_bound(result.relative_l2),
                source_packed_i32le_sha256: weight.source_packed_i32le_sha256.clone(),
                source_scales_bf16le_sha256: weight.source_scales_bf16le_sha256.clone(),
                weight_shape: [
                    u64::try_from(shape.output_features).expect("N fits u64"),
                    u64::try_from(shape.input_features).expect("K fits u64"),
                ],
            });
        }
    }
    assert_eq!(
        artifact_cases.len(),
        4,
        "Gemma 4 numeric matrix must be 4/4"
    );

    let artifact = Gemma4ExactParityArtifactV1 {
        cases: artifact_cases,
        checkpoint: NumericArtifactCheckpoint {
            id: "gemma4-12b-w4a16-ct".to_owned(),
            repository: "google/gemma-4-12B-it-qat-w4a16-ct".to_owned(),
            revision: "1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee".to_owned(),
        },
        execution: NumericArtifactExecution {
            quantization_format_ids: vec![GEMMA4_NUMERIC_QUANTIZATION_FORMAT_ID.to_owned()],
            weight_format_id: GEMMA4_NUMERIC_WEIGHT_FORMAT_ID.to_owned(),
            weight_layout_id: GEMMA4_NUMERIC_WEIGHT_LAYOUT_ID.to_owned(),
        },
        materializer: NumericArtifactMaterializer {
            fidelity: WeightMaterializationFidelity::Exact,
            id: GEMMA4_NUMERIC_MATERIALIZER_ID.to_owned(),
            implementation_fingerprint,
            version: ContractVersion::new(1, 0),
        },
        quality_vector_digest: quality_vector_digest.clone(),
        quality_vector_payload,
        schema_id: GEMMA4_NUMERIC_ARTIFACT_SCHEMA_ID.to_owned(),
        source: NumericArtifactSource {
            weight_format_id: GEMMA4_NUMERIC_WEIGHT_FORMAT_ID.to_owned(),
        },
    };
    let canonical_json = canonical_json_bytes(&artifact);
    let decoded: Gemma4ExactParityArtifactV1 =
        serde_json::from_slice(&canonical_json).expect("parse Gemma 4 exact-parity artifact");
    assert_eq!(
        decoded, artifact,
        "Gemma 4 typed artifact roundtrip drifted"
    );
    assert_eq!(
        canonical_json_bytes(&decoded),
        canonical_json,
        "Gemma 4 canonical artifact changed after typed roundtrip"
    );
    eprintln!(
        "GEMMA4_CT_MARLIN_EXACT_PARITY_ARTIFACT_V1 sha256={} bytes={} json={}",
        sha256_hex(&canonical_json),
        canonical_json.len(),
        std::str::from_utf8(&canonical_json).expect("Gemma 4 artifact is UTF-8"),
    );

    // These locks intentionally run after all four diagnostics so a fixture
    // update yields one complete REJECT artifact rather than four blind reruns.
    assert_eq!(
        quality_vector_digest, GEMMA4_QUALITY_VECTOR_DIGEST,
        "Gemma 4 quality vector digest drifted"
    );
    for (case, expected) in artifact.cases.iter().zip(GEMMA4_LOCKED_CASE_HASHES) {
        let (case_id, packed_sha, scales_sha, activation_sha, reference_sha) = expected;
        assert_eq!(case.case_id, case_id, "Gemma 4 case ordering drifted");
        assert_eq!(
            case.source_packed_i32le_sha256, packed_sha,
            "{case_id} source packed values drifted"
        );
        assert_eq!(
            case.source_scales_bf16le_sha256, scales_sha,
            "{case_id} source scales drifted"
        );
        assert_eq!(
            case.activation_f16le_sha256, activation_sha,
            "{case_id} activations drifted"
        );
        assert_eq!(
            case.reference_f32le_sha256, reference_sha,
            "{case_id} reference drifted"
        );
    }
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
    assert_eq!(descriptor.version(), ContractVersion::new(2, 0));
    assert_eq!(descriptor.fidelity(), WeightMaterializationFidelity::Exact);
    assert!(descriptor.approximate_quality_contract().is_none());
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

        let expected_packed =
            block_fp8_group128_raw_bits_to_final_marlin_u32_reference(&source_values, n, k);
        let expected_scales =
            block_fp8_group128_scales_to_marlin_f16_reference(&inverse_scale_bytes, n, k)
                .expect("build exact group-128 scale oracle")
                .into_iter()
                .map(f16::to_bits)
                .collect::<Vec<_>>();
        let source_device: CudaSlice<u8> = stream
            .clone_htod(&source_values)
            .expect("upload row-major block-FP8 values");
        let inverse_scale_words = inverse_scales
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        let inverse_scale_device: CudaSlice<u16> = stream
            .clone_htod(&inverse_scale_words)
            .expect("upload BF16 inverse-scale grid");
        let mut weight_device: CudaSlice<u32> = stream
            .alloc_zeros(expected_packed.len())
            .expect("allocate exact group-128 packed weight");
        let mut scales_device: CudaSlice<u16> = stream
            .alloc_zeros(expected_scales.len())
            .expect("allocate exact group-128 Marlin scales");
        {
            let (source_pointer, _source_guard) = source_device.device_ptr(&stream);
            let (inverse_scale_pointer, _inverse_scale_guard) =
                inverse_scale_device.device_ptr(&stream);
            let (weight_pointer, _weight_guard) = weight_device.device_ptr_mut(&stream);
            let (scales_pointer, _scales_guard) = scales_device.device_ptr_mut(&stream);
            unsafe {
                launch_block_fp8_group128_repack(
                    &stream,
                    source_pointer,
                    weight_pointer,
                    u64::try_from(k).expect("K fits u64"),
                    u64::try_from(n).expect("N fits u64"),
                )
            }
            .expect("launch exact group-128 value transform");
            unsafe {
                launch_block_fp8_group128_scales(
                    &stream,
                    inverse_scale_pointer,
                    scales_pointer,
                    u64::try_from(k).expect("K fits u64"),
                    u64::try_from(n).expect("N fits u64"),
                )
            }
            .expect("launch exact group-128 scale transform");
        }
        stream
            .synchronize()
            .expect("synchronize exact group-128 transforms");
        assert_eq!(
            stream
                .clone_dtoh(&weight_device)
                .expect("download transformed weight"),
            expected_packed,
            "{case_id} exact group-128 weight transform drifted"
        );
        assert_eq!(
            stream
                .clone_dtoh(&scales_device)
                .expect("download transformed scales"),
            expected_scales,
            "{case_id} exact group-128 scale transform drifted"
        );
        let input_device: CudaSlice<f16> = stream.clone_htod(&input).expect("upload input");
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
                        num_groups: i32::try_from(k / BLOCK_SHAPE[1])
                            .expect("group count fits i32"),
                        group_size: i32::try_from(BLOCK_SHAPE[1]).expect("group size fits i32"),
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

    assert_eq!(artifact_cases.len(), CASES.len());
    let quality_vector_payload = locked_quality_vector_payload();
    let quality_vector_digest = sha256_hex(&canonical_json_bytes(&quality_vector_payload));
    assert_eq!(
        quality_vector_digest, BLOCK_FP8_QUALITY_VECTOR_DIGEST,
        "locked quality vector payload digest drifted"
    );
    let artifact = ExactParityArtifactV1 {
        cases: artifact_cases,
        checkpoint: NumericArtifactCheckpoint {
            id: "qwen38-27b-fp8".to_owned(),
            repository: "Qwen/Qwen3.8-27B-FP8".to_owned(),
            revision: "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a".to_owned(),
        },
        execution: NumericArtifactExecution {
            quantization_format_ids: vec![MARLIN_FP8_GROUP128_QUANTIZATION_FORMAT_ID.to_owned()],
            weight_format_id: MARLIN_FP8_GROUP128_WEIGHT_FORMAT_ID.to_owned(),
            weight_layout_id: MARLIN_FP8_GROUP128_WEIGHT_LAYOUT_ID.to_owned(),
        },
        materializer: NumericArtifactMaterializer {
            fidelity: descriptor.fidelity(),
            id: descriptor.id().as_str().to_owned(),
            implementation_fingerprint: descriptor.implementation_fingerprint().to_owned(),
            version: descriptor.version(),
        },
        quality_vector_payload,
        quality_vector_digest,
        schema_id: BLOCK_FP8_EXACT_PARITY_ARTIFACT_SCHEMA_ID.to_owned(),
        source: NumericArtifactSource {
            weight_format_id: BLOCK_FP8_SOURCE_WEIGHT_FORMAT_ID.to_owned(),
        },
    };
    let canonical_json = canonical_json_bytes(&artifact);
    let decoded: ExactParityArtifactV1 =
        serde_json::from_slice(&canonical_json).expect("parse canonical exact-parity artifact");
    assert_eq!(
        decoded, artifact,
        "exact-parity artifact typed roundtrip drifted"
    );
    assert_eq!(
        canonical_json_bytes(&decoded),
        canonical_json,
        "exact-parity artifact canonical JSON drifted after roundtrip"
    );
    let artifact_sha256 = sha256_hex(&canonical_json);
    let artifact_bytes = canonical_json.len();
    let artifact_json =
        std::str::from_utf8(&canonical_json).expect("canonical artifact is valid UTF-8");
    eprintln!(
        "QWEN38_BLOCK_FP8_EXACT_PARITY_ARTIFACT_V1 sha256={artifact_sha256} bytes={artifact_bytes} json={artifact_json}"
    );
}
