//! CPU-only, node-local replay of saved Block0 SiLU-mul boundaries.
//! This never executes CUDA or changes the original Block0 acceptance criterion.
#[path = "verify_qwen_image21_projection/artifacts.rs"]
mod artifacts;
#[path = "diagnose_qwen_image21_block0_dense/source.rs"]
mod source;

use anyhow::{ensure, Context, Result};
use ferrum_models::qwen_image21::QwenImage21Block0MlpFamily as Family;
use half::bf16;
use serde::Serialize;
use serde_json::json;
use std::{collections::BTreeMap, ffi::OsString, fs::OpenOptions, io::Write, path::PathBuf};

const ABSOLUTE_TOLERANCE: f64 = 0.02;
const RELATIVE_TOLERANCE: f64 = 0.02;
const USAGE: &str = "diagnose_qwen_image21_block0_silu --reference-artifact-dir REF --candidate-artifact-dir CUDA --artifact-dir NEW_EXTERNAL_DIRECTORY\nCPU only. Verifies saved input/output identities and scans every activation element. Writes all original-threshold failures to activation-failures.jsonl; no model weights or CUDA execution are needed. The original Block0 verdict remains unchanged.";

fn main() -> Result<()> {
    let Some(args) = arguments(std::env::args_os().skip(1))? else {
        println!("{USAGE}");
        return Ok(());
    };
    let reference = source::Artifact::open(&args.reference, "reference")?;
    let candidate = source::Artifact::open(&args.candidate, "cuda")?;
    ensure!(
        reference.manifest.identity == candidate.manifest.identity,
        "canonical config, original input and declared weight identities must match exactly"
    );
    let reference_values = Boundaries::load(&reference)?;
    let candidate_values = Boundaries::load(&candidate)?;
    let output = artifacts::Artifacts::create(&args.output)?;
    output.json("declaration.json", &json!({
        "schema":"ferrum.qwen_image21.block0-silu-replay.v1",
        "reference_manifest_sha256":reference.manifest_sha256,
        "candidate_manifest_sha256":candidate.manifest_sha256,
        "identity":reference.manifest.identity,
        "reference_boundaries":reference_values.metadata,
        "candidate_boundaries":candidate_values.metadata,
        "criterion":"finite BF16 activation values, decoded to F64: abs(candidate-reference)<=0.02+0.02*abs(reference); nonfinite observed activations fail",
        "absolute_tolerance":ABSOLUTE_TOLERANCE,"relative_tolerance":RELATIVE_TOLERANCE,
        "replay":"CPU libm=0.2.15 expf(-gate); FP32 add/div; half=2.7.1 BF16 RNE; decode; FP32 multiply by up; BF16 RNE",
        "reference_scalar_source_sha256":artifacts::sha256(include_bytes!("../../ferrum-kernels/src/backend/reference/runtime/transcendental.rs")),
        "reference_profile_source_sha256":artifacts::sha256(include_bytes!("../../ferrum-kernels/src/backend/reference/transcendental.rs")),
        "diagnostic_source_sha256":artifacts::sha256(include_bytes!("diagnose_qwen_image21_block0_silu.rs")),
        "original_criterion_source_sha256":artifacts::sha256(include_bytes!("verify_qwen_image21_block0/evidence.rs")),
        "identity_scope":"original inputs and consumed complete gate/up/activation files are size/SHA checked; weight identities are compared between manifests, without opening or authenticating checkpoint payloads",
        "scope":"all saved activation elements scanned; only original-threshold failures emitted as points; complete saved-tensor replay mismatch counts also reported",
        "interpretation":"saved upstream tensors are explicit node-local inputs, not an independent full-block oracle; candidate-vs-CPU residual does not by itself prove a CUDA defect",
        "original_block_verdict":"unchanged; CPU replay cannot close R36, qualify libdevice, or establish Torch/vLLM parity",
        "memory_scope":"existing bounded artifact reader (16MiB per tensor); diagnostic CPU vectors/control are not a product Host ledger; point records stream with fixed writer buffering"
    }))?;
    let outcome: Result<()> = (|| {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output.directory().join("activation-failures.jsonl"))?;
        let mut writer = std::io::BufWriter::new(file);
        let summary = scan(reference_values.view(), candidate_values.view(), |point| {
            serde_json::to_writer(&mut writer, point)?;
            writer.write_all(b"\n")?;
            Ok(())
        })?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        output.json("result.json", &json!({
            "diagnostic_completed":true,"summary":summary,
            "points_file":"activation-failures.jsonl",
            "gpu_reexecuted":false,"original_block_verdict":"unchanged",
            "decomposition":"candidate_observed-reference_observed = candidate_minus_cpu_replay + cpu_replay_upstream_effect + reference_replay_minus_observed; differences reported in F64, not an exact-dyadic proof",
            "scope":"completed diagnosis is not a numerical acceptance pass; bit mismatch counts are observations, including signed-zero and NaN payload differences"
        }))?;
        Ok(())
    })();
    if let Err(error) = &outcome {
        output.json("failure.json", &json!({"diagnostic_completed":false,"error":format!("{error:#}"),"partial_points_are_not_a_complete_report":true}))?;
    }
    outcome
}

struct Arguments {
    reference: PathBuf,
    candidate: PathBuf,
    output: PathBuf,
}
fn arguments(args: impl Iterator<Item = OsString>) -> Result<Option<Arguments>> {
    let mut args = args.peekable();
    if args.peek().is_some_and(|x| x == "--help" || x == "-h") {
        args.next();
        ensure!(args.next().is_none(), "--help takes no other arguments");
        return Ok(None);
    }
    let mut values = BTreeMap::new();
    while let Some(name) = args.next() {
        let name = name
            .into_string()
            .map_err(|_| anyhow::anyhow!("option names require UTF-8"))?;
        let value = args.next().context("every option requires one value")?;
        ensure!(values.insert(name, value).is_none(), "duplicate option");
    }
    let mut path = |name: &str| -> Result<PathBuf> {
        Ok(values
            .remove(name)
            .with_context(|| format!("missing {name}; {USAGE}"))?
            .into())
    };
    let result = Arguments {
        reference: path("--reference-artifact-dir")?,
        candidate: path("--candidate-artifact-dir")?,
        output: path("--artifact-dir")?,
    };
    ensure!(values.is_empty(), "unknown options: {:?}", values.keys());
    Ok(Some(result))
}

struct Boundaries {
    metadata: [source::Tensor; 3],
    bytes: [Vec<u8>; 3],
}
impl Boundaries {
    fn load(artifact: &source::Artifact) -> Result<Self> {
        let (gate, gate_bytes) = artifact.output(Family::GATE_ID)?;
        let (up, up_bytes) = artifact.output(Family::UP_ID)?;
        let (activation, activation_bytes) = artifact.output(Family::ACTIVATION_ID)?;
        ensure!(
            gate.shape == up.shape && gate.shape == activation.shape,
            "SiLU-mul boundary shapes differ"
        );
        Ok(Self {
            metadata: [gate, up, activation],
            bytes: [gate_bytes, up_bytes, activation_bytes],
        })
    }
    fn view(&self) -> BoundaryView<'_> {
        BoundaryView {
            shape: &self.metadata[0].shape,
            gate: &self.bytes[0],
            up: &self.bytes[1],
            activation: &self.bytes[2],
        }
    }
}

#[derive(Clone, Copy)]
struct BoundaryView<'a> {
    shape: &'a [u64],
    gate: &'a [u8],
    up: &'a [u8],
    activation: &'a [u8],
}
impl BoundaryView<'_> {
    fn validate(self) -> Result<(usize, usize)> {
        ensure!(
            self.shape.len() == 2 && !self.shape.contains(&0),
            "SiLU-mul needs nonempty rank-two boundaries"
        );
        let elements = self.shape[0]
            .checked_mul(self.shape[1])
            .context("element extent overflow")?;
        let bytes = usize::try_from(elements.checked_mul(2).context("byte extent overflow")?)?;
        ensure!(
            [self.gate, self.up, self.activation]
                .iter()
                .all(|x| x.len() == bytes),
            "boundary byte lengths differ from shape"
        );
        Ok((usize::try_from(elements)?, usize::try_from(self.shape[1])?))
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
struct Scalar {
    bits: u16,
    value: Option<f64>,
    class: &'static str,
}
impl Scalar {
    fn new(bits: u16) -> Self {
        let value = f64::from(bf16::from_bits(bits).to_f32());
        let class = if value.is_nan() {
            "nan"
        } else if value == f64::INFINITY {
            "positive_infinity"
        } else if value == f64::NEG_INFINITY {
            "negative_infinity"
        } else if value == 0.0 && value.is_sign_negative() {
            "negative_zero"
        } else if value == 0.0 {
            "positive_zero"
        } else {
            "finite"
        };
        Self {
            bits,
            value: value.is_finite().then_some(value),
            class,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
struct Replay {
    gate: Scalar,
    up: Scalar,
    exponential_f32_bits: u32,
    denominator_f32_bits: u32,
    silu: Scalar,
    activation: Scalar,
}
fn replay(gate: u16, up: u16) -> Replay {
    // Match Reference scalar order; do not substitute stable sigmoid or fuse the
    // product across the first BF16 boundary. This is CPU replay, not CUDA.
    let x = bf16::from_bits(gate).to_f32();
    let exponential = libm::expf(-x);
    let denominator = 1.0_f32 + exponential;
    let silu = bf16::from_f32(x / denominator);
    let activation = bf16::from_f32(silu.to_f32() * bf16::from_bits(up).to_f32());
    Replay {
        gate: Scalar::new(gate),
        up: Scalar::new(up),
        exponential_f32_bits: exponential.to_bits(),
        denominator_f32_bits: denominator.to_bits(),
        silu: Scalar::new(silu.to_bits()),
        activation: Scalar::new(activation.to_bits()),
    }
}

#[derive(Debug, Serialize)]
struct Point {
    row: usize,
    channel: usize,
    reference_observed_activation: Scalar,
    candidate_observed_activation: Scalar,
    reference_inputs_replay: Replay,
    candidate_inputs_replay: Replay,
    original_absolute_error: Option<f64>,
    original_allowed_error: Option<f64>,
    candidate_minus_cpu_replay: Option<f64>,
    cpu_replay_upstream_effect: Option<f64>,
    reference_replay_minus_observed: Option<f64>,
}
fn difference(lhs: Scalar, rhs: Scalar) -> Option<f64> {
    Some(lhs.value? - rhs.value?)
}
#[derive(Default, Debug, Serialize)]
struct Summary {
    elements: usize,
    original_activation_failure_count: usize,
    original_activation_nonfinite_count: usize,
    reference_replay_bit_mismatch_count: usize,
    candidate_replay_bit_mismatch_count: usize,
    changed_gate_or_up_bits_count: usize,
    replay_nonfinite_activation_count: usize,
}
fn scan(
    reference: BoundaryView<'_>,
    candidate: BoundaryView<'_>,
    mut emit: impl FnMut(&Point) -> Result<()>,
) -> Result<Summary> {
    let (elements, width) = reference.validate()?;
    candidate.validate()?;
    ensure!(
        reference.shape == candidate.shape,
        "reference and candidate boundary shapes differ"
    );
    let bits = |data: &[u8], i: usize| u16::from_le_bytes([data[2 * i], data[2 * i + 1]]);
    let mut summary = Summary {
        elements,
        ..Summary::default()
    };
    for i in 0..elements {
        let r = replay(bits(reference.gate, i), bits(reference.up, i));
        let c = replay(bits(candidate.gate, i), bits(candidate.up, i));
        let observed_r = Scalar::new(bits(reference.activation, i));
        let observed_c = Scalar::new(bits(candidate.activation, i));
        summary.reference_replay_bit_mismatch_count +=
            usize::from(r.activation.bits != observed_r.bits);
        summary.candidate_replay_bit_mismatch_count +=
            usize::from(c.activation.bits != observed_c.bits);
        summary.changed_gate_or_up_bits_count +=
            usize::from(r.gate.bits != c.gate.bits || r.up.bits != c.up.bits);
        summary.replay_nonfinite_activation_count +=
            usize::from(r.activation.value.is_none() || c.activation.value.is_none());
        let nonfinite = observed_r.value.is_none() || observed_c.value.is_none();
        summary.original_activation_nonfinite_count += usize::from(nonfinite);
        let error = difference(observed_c, observed_r).map(f64::abs);
        let allowed = observed_r
            .value
            .map(|x| ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * x.abs());
        if nonfinite
            || error
                .zip(allowed)
                .is_some_and(|(error, allowed)| error > allowed)
        {
            summary.original_activation_failure_count += 1;
            emit(&Point {
                row: i / width,
                channel: i % width,
                reference_observed_activation: observed_r,
                candidate_observed_activation: observed_c,
                reference_inputs_replay: r,
                candidate_inputs_replay: c,
                original_absolute_error: error,
                original_allowed_error: allowed,
                candidate_minus_cpu_replay: difference(observed_c, c.activation),
                cpu_replay_upstream_effect: difference(c.activation, r.activation),
                reference_replay_minus_observed: difference(r.activation, observed_r),
            })?;
        }
    }
    Ok(summary)
}

#[cfg(test)]
#[path = "diagnose_qwen_image21_block0_silu/tests.rs"]
mod tests;
