//! Isolated Q4 prefill experiments. Production selection is intact.
//! Both arms execute the same fused M32/N64/K32 MMA and complete SwiGLU.
use super::*;
use crate::backend::metal::k_quant_gemm::SHADER_SOURCE as GEMM_SOURCE;
use crate::backend::metal::vnext_ops::numerical_tolerance;
use q4_b4_ffn_cooperative::oracle::{matrix, metrics};

mod coefficients;
mod direct_store;
mod full_ffn;
mod reference;

const WEIGHT_PREFIX: usize = 2;
const HALF_PREFIX: usize = 3;
const GUARD: usize = 17;
const BYTE_GUARD: u8 = 0xa7;
const HALF_GUARD: f16 = f16::from_bits(0x57b0);
const FULL_TOLERANCE: &str =
    "runtime-vnext.metal.dense-swiglu.v1.operation.fp16.gguf-q4-k-q6-k.full-pipeline";

struct ExperimentalPipelines {
    experiment: Experiment,
    fused: ComputePipelineState,
    coefficients: ComputePipelineState,
    direct_store_probe: Option<ComputePipelineState>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Experiment {
    U16Bytes,
    DirectF16,
}
impl Experiment {
    fn emit(self, mut record: serde_json::Value) {
        if self == Self::DirectF16 {
            if let Some(kind) = record["kind"].as_str() {
                record["kind"] = kind
                    .replace("q4_prefill_u16", "q4_prefill_direct_f16")
                    .into();
            }
        }
        record["experiment"] = format!("{self:?}").into();
        println!("{record}");
    }
    fn coefficient_probe(self) -> &'static str {
        match self {
            Self::U16Bytes => "stage_q4k_f16_u16_bytes_probe",
            Self::DirectF16 => "stage_q4k_f16",
        }
    }
}

impl ExperimentalPipelines {
    fn new(device: &Device) -> Self {
        Self::for_experiment(device, Experiment::U16Bytes)
    }
    fn for_experiment(device: &Device, experiment: Experiment) -> Self {
        let (define, entry) = match experiment {
            Experiment::U16Bytes => (
                "FERRUM_TEST_Q4_U16_BYTES",
                "gemm_f16a_q4kw_tiled_u16_bytes_experiment",
            ),
            Experiment::DirectF16 => (
                "FERRUM_TEST_Q4_DIRECT_F16_STORE",
                "gemm_f16a_q4kw_tiled_direct_f16_experiment",
            ),
        };
        let source = format!("#define {define} 1\n{GEMM_SOURCE}");
        let library = device
            .new_library_with_source(&source, &CompileOptions::new())
            .expect("compile isolated Q4 experiment with original options");
        let pipeline = |name| {
            let function = library.get_function(name, None).unwrap();
            device
                .new_compute_pipeline_state_with_function(&function)
                .unwrap()
        };
        let result = Self {
            experiment,
            fused: pipeline(entry),
            coefficients: pipeline(experiment.coefficient_probe()),
            direct_store_probe: (experiment == Experiment::DirectF16)
                .then(|| pipeline("full_tile_f16_direct_store_probe")),
        };
        assert_eq!(result.fused.thread_execution_width(), 32);
        assert!(result.fused.max_total_threads_per_threadgroup() >= 128);
        assert!(
            result.fused.static_threadgroup_memory_length() + 8192
                <= device.max_threadgroup_memory_length()
        );
        result
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    Production,
    // Kept for the existing u16 experiment's identity; the explicit pipeline
    // parameter supplies this slot for either isolated candidate, never both.
    U16Bytes,
}

impl Arm {
    fn label(self, experiment: Experiment) -> &'static str {
        match (self, experiment) {
            (Self::Production, _) => "Production",
            (Self::U16Bytes, Experiment::U16Bytes) => "U16Bytes",
            (Self::U16Bytes, Experiment::DirectF16) => "DirectF16",
        }
    }
}

fn selects_candidate(arm: Arm, format: GgufBlockFormat, kind: LinearDispatchKind) -> bool {
    arm == Arm::U16Bytes && format == GgufBlockFormat::Q4K && kind == LinearDispatchKind::TiledGemm
}

fn bits_report(actual: &[f16], expected: &[f16], width: usize) -> serde_json::Value {
    let mut report = metrics(actual, expected, width);
    report["bitwise_differences"] = actual
        .iter()
        .zip(expected)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
        .into();
    report["first_bitwise_difference"] = actual
        .iter()
        .zip(expected)
        .position(|(a, b)| a.to_bits() != b.to_bits())
        .map(|index| {
            serde_json::json!({"index":index,"row":index/width,"column":index%width,
            "actual_bits":actual[index].to_bits(),"expected_bits":expected[index].to_bits()})
        })
        .into();
    report
}

fn bitwise_equal(actual: &[f16], expected: &[f16]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

fn read_halves(buffer: &Buffer) -> &[f16] {
    // SAFETY: these buffers contain shared halves; all callers synchronize
    // command completion before reading and hold the allocation for this view.
    unsafe {
        std::slice::from_raw_parts(
            buffer.contents().cast::<f16>(),
            buffer.length() as usize / 2,
        )
    }
}

fn byte_buffer(device: &Device, bytes: &[u8]) -> Buffer {
    let mut guarded = vec![BYTE_GUARD; WEIGHT_PREFIX];
    guarded.extend_from_slice(bytes);
    guarded.extend([BYTE_GUARD; GUARD]);
    buffer(device, &guarded)
}

fn immutable_bytes(buffer: &Buffer, expected: &[u8]) -> bool {
    // SAFETY: shared byte allocation, read only after command completion.
    let actual = unsafe {
        std::slice::from_raw_parts(buffer.contents().cast::<u8>(), buffer.length() as usize)
    };
    actual.len() == WEIGHT_PREFIX + expected.len() + GUARD
        && actual[..WEIGHT_PREFIX].iter().all(|&v| v == BYTE_GUARD)
        && &actual[WEIGHT_PREFIX..WEIGHT_PREFIX + expected.len()] == expected
        && actual[WEIGHT_PREFIX + expected.len()..]
            .iter()
            .all(|&v| v == BYTE_GUARD)
}

#[test]
fn q4_prefill_u16_equivalence_preserves_signed_zero_subnormal_and_length() {
    assert!(bitwise_equal(&[f16::ZERO], &[f16::ZERO]));
    assert!(!bitwise_equal(&[f16::ZERO], &[]));
    assert!(!bitwise_equal(&[f16::ZERO], &[f16::from_bits(0x8000)]));
    assert!(!bitwise_equal(&[f16::from_bits(1)], &[f16::ZERO]));
    let report = bits_report(&[f16::ZERO], &[f16::from_bits(0x8000)], 1);
    assert_eq!(report["bitwise_differences"], 1);
    assert_eq!(report["max_abs_error"], 0.0);
}
