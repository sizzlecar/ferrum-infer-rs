//! Optional full-OI deployment-library diagnosis of existing block-0 Q/V.
//!
//! Set FERRUM_TEST_TORCH_LINEAR_CONFIG to the explicit JSON ReplayConfig below.
//! Build (no GPU): cargo test -p ferrum-models --test torch_block0_linear_oracle
//! build_deployed_torch_linear_oracle -- --ignored --nocapture --test-threads=1
//! Exclusive GPU: use executable.mode=prebuilt, then select
//! deployed_torch_linear_replays_real_block0_qv with the same Cargo arguments.
//! A successful diagnostic is NOT block parity: the original R36 verdict and
//! abs <= .02 + .02*abs(reference) criterion remain unchanged. This calls BF16
//! ATen in the deployed libraries, not Omni's FP8 quantized Q/V path.

#[path = "../examples/verify_qwen_image21_projection/artifacts.rs"]
mod artifacts;
// Full-matrix replay uses exact dots without the selected-point report helpers.
#[allow(dead_code)]
#[path = "../examples/diagnose_qwen_image21_block0_dense/dyadic.rs"]
mod dyadic;
#[path = "support/torch_linear_oracle.rs"]
mod linear;
// Share native process/library ownership, not the Philox generation entrypoint.
#[allow(dead_code)]
#[path = "support/torch_native_oracle.rs"]
mod native;
#[path = "../examples/diagnose_qwen_image21_block0_dense/source.rs"]
mod source;

use anyhow::{ensure, Context, Result};
use ferrum_models::qwen_image21::{
    QwenImage21Block0AttentionFamily as Attention, QwenImage21Block0Family as Block,
};
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::{HostMemoryAccount, HostMemoryClassLimits, HostMemoryLimits};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

/// Paths select user-owned existing artifacts and checkpoint. Channel lists
/// are diagnostic selections, never a success ratio or fixed-fixture gate.
/// For the R36 first divergence use query_channels=[953] and
/// value_channels=[126,668,1602], while replaying both complete weight matrices.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ReplayConfig {
    torch: native::Config,
    reference_artifact_dir: PathBuf,
    candidate_artifact_dir: PathBuf,
    model_dir: PathBuf,
    index_file: String,
    artifact_dir: PathBuf,
    policy: linear::MathPolicy,
    query_channels: Vec<usize>,
    value_channels: Vec<usize>,
}
impl ReplayConfig {
    fn from_environment() -> Result<Self> {
        let path = std::env::var_os("FERRUM_TEST_TORCH_LINEAR_CONFIG")
            .context("explicit FERRUM_TEST_TORCH_LINEAR_CONFIG is required")?;
        let config: Self = serde_json::from_slice(&native::read_bounded(
            Path::new(&path),
            native::METADATA_LIMIT,
        )?)?;
        ensure!(
            [
                &config.reference_artifact_dir,
                &config.candidate_artifact_dir,
                &config.model_dir,
                &config.artifact_dir
            ]
            .iter()
            .all(|p| p.is_absolute()),
            "diagnostic source/output paths must be explicit absolute paths"
        );
        for channels in [&config.query_channels, &config.value_channels] {
            ensure!(
                !channels.is_empty()
                    && channels.len() <= 64
                    && channels.iter().collect::<BTreeSet<_>>().len() == channels.len(),
                "diagnostic channel lists require 1..64 distinct indices"
            );
        }
        Ok(config)
    }
}

#[test]
#[ignore = "explicit Torch Build configuration and Linux C++ toolchain; does not execute a GPU operation"]
fn build_deployed_torch_linear_oracle() {
    let config = ReplayConfig::from_environment().unwrap();
    assert!(matches!(
        &config.torch.executable,
        native::Executable::Build { .. }
    ));
    let oracle = linear::Oracle::prepare(config.torch).unwrap();
    eprintln!(
        "{}",
        json!({"executable":oracle.native.executable,
        "executable_sha256":oracle.native.executable_sha256,"build_receipt":oracle.native.build_receipt,
        "gpu_executed":false,"program":"aten.linear.no_bias"})
    );
}

#[derive(Debug, Serialize)]
struct Comparison {
    elements: usize,
    exact_bit_mismatches: usize,
    nonfinite_pairs: usize,
    failures_under_original_threshold: usize,
    maximum_absolute_error: f64,
    mean_absolute_error_over_finite_pairs: f64,
}
fn comparison(torch: &[u8], baseline: &[u8]) -> Result<Comparison> {
    ensure!(
        !torch.is_empty() && torch.len() % 2 == 0 && torch.len() == baseline.len(),
        "comparison extents differ"
    );
    let mut stats = Comparison {
        elements: torch.len() / 2,
        exact_bit_mismatches: 0,
        nonfinite_pairs: 0,
        failures_under_original_threshold: 0,
        maximum_absolute_error: 0.0,
        mean_absolute_error_over_finite_pairs: 0.0,
    };
    let mut sum = 0.0_f64;
    for (actual, expected) in torch.chunks_exact(2).zip(baseline.chunks_exact(2)) {
        let a = u16::from_le_bytes([actual[0], actual[1]]);
        let b = u16::from_le_bytes([expected[0], expected[1]]);
        stats.exact_bit_mismatches += usize::from(a != b);
        let (a, b) = (dyadic::value(a), dyadic::value(b));
        if !a.is_finite() || !b.is_finite() {
            stats.nonfinite_pairs += 1;
            stats.failures_under_original_threshold += 1;
            continue;
        }
        let distance = (a - b).abs();
        sum += distance;
        stats.maximum_absolute_error = stats.maximum_absolute_error.max(distance);
        stats.failures_under_original_threshold += usize::from(distance > 0.02 + 0.02 * b.abs());
    }
    let finite = stats.elements - stats.nonfinite_pairs;
    stats.mean_absolute_error_over_finite_pairs = if finite == 0 {
        0.0
    } else {
        sum / finite as f64
    };
    Ok(stats)
}
fn bits(bytes: &[u8], index: usize) -> u16 {
    u16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]])
}

fn diagnose(config: ReplayConfig) -> Result<()> {
    ensure!(matches!(&config.torch.executable, native::Executable::Prebuilt { .. }),
        "GPU diagnostic requires separately built/prebuilt oracle; do not compile while holding the GPU window");
    let reference = source::Artifact::open(&config.reference_artifact_dir, "reference")?;
    let candidate = source::Artifact::open(&config.candidate_artifact_dir, "cuda")?;
    ensure!(
        reference.manifest.identity == candidate.manifest.identity,
        "canonical model config, input and complete weight identities differ"
    );
    let (input_meta, input) = reference.output(Block::MODULATED1_ID)?;
    let (candidate_input_meta, candidate_input) = candidate.output(Block::MODULATED1_ID)?;
    ensure!(
        input_meta.shape == candidate_input_meta.shape && input == candidate_input,
        "first Q/V replay requires identical actual upstream bytes to isolate linear arithmetic"
    );
    let rows = usize::try_from(input_meta.shape[0])?;
    let k = usize::try_from(input_meta.shape[1])?;
    let archive = SafetensorsArchive::open_with_index(&config.model_dir, &config.index_file)?;
    let files = artifacts::Artifacts::create(&config.artifact_dir)?;
    files.json("declaration.json", &json!({"schema":"ferrum.block0.torch-linear-diagnosis.v1",
        "configuration":config,"reference_manifest_sha256":reference.manifest_sha256,
        "candidate_manifest_sha256":candidate.manifest_sha256,"identity":reference.manifest.identity,
        "input":input_meta,"scope":"complete OI BF16 ATen linear in explicitly selected deployed libraries; not Omni FP8 Q/V or full-block parity",
        "original_block_criterion":"abs(candidate-reference)<=0.02+0.02*abs(reference), unchanged",
        "original_block_verdict":"unchanged; the R36 failure is not closed by this diagnostic",
        "weight_transport":"original safetensors shard segment, checked against mmap bytes and manifest SHA; no copied weight payload file",
        "file_contract":"immutable local sources throughout child execution; no adversarial concurrent mutation guarantee",
        "math_environment":"known TF32/BLAS preference/workspace overrides cleared; policy and actual ATen Context values recorded"}))?;
    let mut oracle = linear::Oracle::prepare(config.torch.clone())?;
    let outcome: Result<()> = (|| {
        let mut records = Vec::new();
        let mut nonfinite = 0;
        for (stage, output_id, weight_name, channels) in [
            (
                "query",
                Attention::QUERY_ID,
                Block::WEIGHT_NAMES[0],
                &config.query_channels,
            ),
            (
                "value",
                Attention::VALUE_ID,
                Block::WEIGHT_NAMES[4],
                &config.value_channels,
            ),
        ] {
            let output_meta = reference.output_metadata(output_id)?;
            let candidate_meta = candidate.output_metadata(output_id)?;
            ensure!(
                output_meta.shape == candidate_meta.shape
                    && output_meta.shape.len() == 2
                    && output_meta.shape[0] == rows as u64
                    && output_meta.dtype == "BF16"
                    && candidate_meta.dtype == "BF16",
                "Q/V output geometry differs"
            );
            let n = usize::try_from(output_meta.shape[1])?;
            ensure!(
                channels.iter().all(|c| *c < n),
                "selected diagnostic channel exceeds complete output width"
            );
            let weight = reference.weight(&archive, weight_name, k, n)?;
            // Independent selected exact-dot records are prepared before any
            // Torch/candidate outputs are inspected. Capacity failure is an
            // unresolved diagnostic, never substituted with a floating sum.
            let mut exact = Vec::new();
            for row in 0..rows {
                for &channel in channels {
                    exact.push((
                        row,
                        channel,
                        dyadic::Dot::prepare(
                            &input[row * k * 2..(row + 1) * k * 2],
                            &weight[channel * k * 2..(channel + 1) * k * 2],
                        )?,
                    ));
                }
            }
            let tensor = archive.tensor(weight_name)?;
            let case = linear::Case {
                rows: rows as u64,
                in_features: k as u64,
                out_features: n as u64,
                policy: config.policy,
                input: linear::FileSegment {
                    path: config
                        .reference_artifact_dir
                        .join(&input_meta.file)
                        .canonicalize()?,
                    offset: 0,
                    bytes: input_meta.bytes,
                    sha256: input_meta.sha256.clone(),
                },
                weight: linear::FileSegment {
                    path: config.model_dir.join(tensor.source_file()).canonicalize()?,
                    offset: u64::try_from(tensor.retained_host_memory().offset_bytes())?,
                    bytes: weight.len() as u64,
                    sha256: native::bytes_sha(weight),
                },
            };
            let output_bytes = usize::try_from(case.output_bytes()?)?;
            let host = HostMemoryAccount::new(HostMemoryLimits::new(
                output_bytes,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 0,
                    output_bytes,
                    transfer_staging_bytes: 0,
                    weight_backing_bytes: 0,
                },
            )?);
            let torch = oracle.linear(&case, &host)?;
            // Existing outputs are observations, never arguments to ATen.
            let (_, reference_bytes) = reference.output(output_id)?;
            let (_, candidate_bytes) = candidate.output(output_id)?;
            let vs_reference = comparison(torch.bytes.bytes(), &reference_bytes)?;
            let vs_candidate = comparison(torch.bytes.bytes(), &candidate_bytes)?;
            nonfinite += vs_reference.nonfinite_pairs + vs_candidate.nonfinite_pairs;
            let selected = exact.into_iter().map(|(row, channel, dot)| {
                let index = row*n + channel;
                let (t, r, c) = (bits(torch.bytes.bytes(), index), bits(&reference_bytes, index), bits(&candidate_bytes, index));
                json!({"row":row,"channel":channel,"torch_bits":t,"reference_bits":r,"candidate_bits":c,
                    "torch_value":dyadic::value(t).is_finite().then(||dyadic::value(t)),
                    "reference_value":dyadic::value(r).is_finite().then(||dyadic::value(r)),
                    "candidate_value":dyadic::value(c).is_finite().then(||dyadic::value(c)),
                    "exact_dot":dot})
            }).collect::<Vec<_>>();
            files.bytes(&format!("{stage}-torch.bf16"), torch.bytes.bytes())?;
            let record = json!({"stage":stage,"case":case,"output_port":output_id,
                "torch_sha256":native::bytes_sha(torch.bytes.bytes()),"reference_sha256":output_meta.sha256,
                "candidate_sha256":candidate_meta.sha256,"torch_vs_reference":vs_reference,
                "torch_vs_candidate":vs_candidate,"selected":selected,"metadata":torch.metadata,
                "actual_library_sha256":torch.library_sha256});
            files.json(&format!("{stage}.json"), &record)?;
            drop(torch);
            ensure!(
                host.usage()?.committed_bytes() == 0 && host.usage()?.allocated_bytes() == 0,
                "managed Torch output did not retire"
            );
            records.push(record);
        }
        let result = json!({"diagnostic_completed":true,"records":records,
            "oracle_executable_sha256":oracle.native.executable_sha256,
            "nonfinite_pairs":nonfinite,"original_block_verdict":"unchanged",
            "block_or_deployment_parity_established":false,
            "interpretation":"same-shape local observations discriminate reduction/profile effects; matching selected points does not close the original block failure"});
        files.json("result.json", &result)?;
        eprintln!("{result}");
        ensure!(
            nonfinite == 0,
            "Torch/baseline nonfinite data; diagnostic output is not numerically usable"
        );
        Ok(())
    })();
    if let Err(error) = &outcome {
        files.json(
            "failure.json",
            &json!({"diagnostic_completed":false,"error":format!("{error:#}"),
            "unresolved_is_not_pass":true,"original_block_verdict":"unchanged"}),
        )?;
    }
    outcome
}

#[test]
#[ignore = "exclusive GPU window, explicit immutable R36 artifacts/checkpoint and separately built deployed LibTorch oracle; diagnostic only"]
fn deployed_torch_linear_replays_real_block0_qv() {
    diagnose(ReplayConfig::from_environment().unwrap()).unwrap();
}

#[test]
fn comparisons_preserve_bits_nonfinite_classification_and_original_absolute_relative_threshold() {
    let bytes = |values: &[u16]| {
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let baseline = bytes(&[0, 0x3f80, 0x4000]);
    let same = comparison(&baseline, &baseline).unwrap();
    assert_eq!(same.exact_bit_mismatches, 0);
    let changed = comparison(&bytes(&[0x8000, 0x3f81, 0x4080]), &baseline).unwrap();
    assert_eq!(changed.exact_bit_mismatches, 3);
    assert_eq!(changed.failures_under_original_threshold, 1);
    let nonfinite = comparison(&bytes(&[0, 0x7fc1, 0x7f80]), &baseline).unwrap();
    assert_eq!(nonfinite.nonfinite_pairs, 2);
    assert_eq!(nonfinite.failures_under_original_threshold, 2);
    assert!(comparison(&[0], &[0]).is_err());
}
