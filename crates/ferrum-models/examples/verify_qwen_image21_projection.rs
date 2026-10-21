//! Reproducible, bounded img_in verification through the managed Reference executor.
//!
//! Usage: verify_qwen_image21_projection --config CONFIG --weight-fragment BF16
//! --weight-sha256 HEX --artifact-dir NEW_DIRECTORY_OUTSIDE_A_GIT_WORKTREE
//!
//! The supplied digest binds fragment integrity, not checkpoint authenticity.
//! Candle F64 is an independent mathematical reference, not deployed Torch/vLLM
//! parity. Preparation and Candle allocations are outside the runtime Host ledger.

#[path = "verify_qwen_image21_projection/artifacts.rs"]
mod artifacts;
#[path = "verify_qwen_image21_projection/execution.rs"]
mod execution;
#[path = "verify_qwen_image21_projection/oracle.rs"]
mod oracle;

use std::ffi::OsString;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use ferrum_models::qwen_image21::{QwenImage21Config, StructureLimits};
use serde::Serialize;
use serde_json::json;

const ROWS: usize = 4;
const MAX_WEIGHT_BYTES: usize = 512 * 1024;
const USAGE: &str = "verify_qwen_image21_projection --config CONFIG --weight-fragment BF16 --weight-sha256 HEX --artifact-dir NEW_DIRECTORY_OUTSIDE_A_GIT_WORKTREE";

struct Arguments {
    config: PathBuf,
    weight_fragment: PathBuf,
    weight_sha256: String,
    artifact_dir: PathBuf,
}

impl Arguments {
    fn parse(arguments: impl IntoIterator<Item = OsString>) -> Result<Option<Self>> {
        let mut arguments = arguments.into_iter().peekable();
        if arguments
            .peek()
            .is_some_and(|value| value == "--help" || value == "-h")
        {
            arguments.next();
            if arguments.next().is_some() {
                bail!("--help accepts no other arguments");
            }
            return Ok(None);
        }
        let (mut config, mut fragment, mut digest, mut directory) = (None, None, None, None);
        while let Some(flag) = arguments.next() {
            let value = arguments.next().context("every option requires a value")?;
            let slot = match flag.to_str() {
                Some("--config") => &mut config,
                Some("--weight-fragment") => &mut fragment,
                Some("--weight-sha256") => &mut digest,
                Some("--artifact-dir") => &mut directory,
                _ => bail!("unknown option {flag:?}; {USAGE}"),
            };
            if slot.replace(value).is_some() {
                bail!("duplicate option {flag:?}");
            }
        }
        let weight_sha256 = digest
            .context(USAGE)?
            .into_string()
            .map_err(|_| anyhow::anyhow!("SHA256 must be ASCII hexadecimal"))?;
        if weight_sha256.len() != 64 || !weight_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            bail!("--weight-sha256 requires exactly 64 hexadecimal digits");
        }
        Ok(Some(Self {
            config: config.context(USAGE)?.into(),
            weight_fragment: fragment.context(USAGE)?.into(),
            weight_sha256: weight_sha256.to_ascii_lowercase(),
            artifact_dir: directory.context(USAGE)?.into(),
        }))
    }
}

#[derive(Serialize)]
struct CaseResult {
    name: &'static str,
    input_sha256: String,
    output_sha256: String,
    comparison: oracle::Comparison,
}

fn run(arguments: Arguments) -> Result<()> {
    let limits = StructureLimits::default();
    let config_bytes =
        artifacts::read_bounded(&arguments.config, limits.metadata_file_bytes() as usize)?;
    let config = QwenImage21Config::from_hf_json(&config_bytes, limits)?;
    let inputs = usize::try_from(config.in_channels())?;
    let outputs = usize::try_from(config.hidden_size())?;
    let weight_length = inputs
        .checked_mul(outputs)
        .and_then(|n| n.checked_mul(2))
        .context("weight extent overflows")?;
    if weight_length == 0 || weight_length > MAX_WEIGHT_BYTES {
        bail!("projection fragment exceeds the bounded {MAX_WEIGHT_BYTES}-byte tool scope");
    }
    let weight = artifacts::read_bounded(&arguments.weight_fragment, MAX_WEIGHT_BYTES)?;
    if weight.len() != weight_length {
        bail!(
            "fragment has {} bytes; configuration requires {weight_length}",
            weight.len()
        );
    }
    let digest = artifacts::sha256(&weight);
    if digest != arguments.weight_sha256 {
        bail!("fragment SHA256 differs from the caller's integrity pin");
    }
    let shape = oracle::ProjectionShape::new(ROWS, inputs, outputs)?;
    let probes = oracle::analytic_inputs(shape)?;
    let dense = oracle::dense_inputs(shape)?;
    // Establish the declared domain before any runtime execution or artifacts.
    oracle::validate_domain(shape, &probes, &weight)?;
    oracle::validate_domain(shape, &dense, &weight)?;

    let artifacts = artifacts::Artifacts::create(&arguments.artifact_dir)?;
    artifacts.json("declaration.json", &json!({
        "schema": "ferrum.qwen_image21.projection-verification.v1",
        "scope": "partial img_in only; independent CPU mathematics; no Torch/vLLM parity",
        "source": {
            "config_path": arguments.config.canonicalize()?.display().to_string(),
            "config_sha256": artifacts::sha256(&config_bytes),
            "weight_fragment_path": arguments.weight_fragment.canonicalize()?.display().to_string(),
            "caller_sha256": arguments.weight_sha256,
            "observed_sha256": digest,
            "integrity_scope": "caller-supplied digest; original checkpoint/revision authenticity not established",
            "tensor": "img_in.weight", "dtype": "BF16", "shape": [outputs, inputs],
            "payload_bytes": weight.len()
        },
        "input_shape": [ROWS, inputs], "output_shape": [ROWS, outputs],
        "bits_encoding": "contiguous row-major BF16, little-endian u16; complete tensors",
        "oracle": oracle::declaration(shape),
        "memory_scope": "executor inputs/outputs use its injected Host account; fragment preparation, archive mmap, Candle and artifact/control allocations are separate; not an RSS bound"
    }))?;
    artifacts.bytes("transformer-config.json", &config_bytes)?;
    artifacts.single_weight_archive(&weight, outputs, inputs, &digest)?;
    artifacts.bytes("analytic-input.bf16", &probes)?;
    artifacts.bytes("dense-input.bf16", &dense)?;

    let result = (|| -> Result<()> {
        let mut executor =
            execution::ProjectionExecutor::new(&config, artifacts.directory(), shape)?;
        artifacts.json("execution.json", &executor.provenance())?;
        let mut reports = Vec::with_capacity(2);
        for (name, input) in [("analytic", probes.as_slice()), ("dense", dense.as_slice())] {
            let output = executor.execute(name, input)?;
            artifacts.bytes(&format!("{name}-output.bf16"), output.bytes())?;
            let comparison = if name == "analytic" {
                oracle::compare_analytic(shape, &weight, output.bytes())?
            } else {
                oracle::compare_dense(shape, input, &weight, output.bytes())?
            };
            reports.push(CaseResult {
                name,
                input_sha256: artifacts::sha256(input),
                output_sha256: artifacts::sha256(output.bytes()),
                comparison,
            });
        }
        let cleanup = executor.close()?;
        let passed = reports.iter().all(|report| report.comparison.passed());
        artifacts.json("result.json", &json!({
            "passed": passed,
            "cases": reports,
            "cleanup": cleanup,
            "scope": "sampled partial projection numerical verification, not full model or upstream parity"
        }))?;
        if !passed {
            bail!("projection comparison failed; inspect result.json and the complete output bits");
        }
        println!(
            "Projection comparison passed: {}",
            artifacts.directory().display()
        );
        Ok(())
    })();
    if let Err(error) = &result {
        artifacts.json(
            "failure.json",
            &json!({"error": format!("{error:#}"), "passed": false}),
        )?;
    }
    result
}

fn main() -> Result<()> {
    match Arguments::parse(std::env::args_os().skip(1))? {
        Some(arguments) => run(arguments),
        None => {
            println!("{USAGE}\nCreates a new external artifact directory; never overwrites files.");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arguments_reject_duplicate_unknown_and_malformed_digest() {
        for args in [
            vec!["--config", "a", "--config", "b"],
            vec!["--unknown", "a"],
            vec!["--weight-sha256", "bad"],
            vec!["--config"],
        ] {
            assert!(Arguments::parse(args.into_iter().map(OsString::from)).is_err());
        }
    }

    #[test]
    fn tool_executes_real_projection_family_and_writes_complete_analytic_bits() {
        let parent = tempfile::tempdir().unwrap();
        let config_path = parent.path().join("config.json");
        // A small valid architecture configuration, not a claimed model
        // checkpoint. The full adapter validates its otherwise unused fields.
        let config = json!({
            "_class_name": "QwenImage21Transformer2DModel",
            "num_layers": 1, "num_attention_heads": 1,
            "attention_head_dim": 6, "context_in_dim": 8,
            "in_channels": 3, "out_channels": 4,
            "patch_size": 1, "mlp_ratio": 3,
            "axes_dims_rope": [2, 2, 2], "causal_condition": true,
            "eps": 0.000001
        });
        std::fs::write(&config_path, serde_json::to_vec(&config).unwrap()).unwrap();
        let encode = |values: &[f32]| {
            values
                .iter()
                .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_le_bytes())
                .collect::<Vec<_>>()
        };
        let weight = encode(&[
            2., -1., 1., 1., 3., -2., 1., 2., 3., -2., 0., 1., 0., 1., 2., 3., 2., -1.,
        ]);
        let fragment = parent.path().join("weight.bf16");
        std::fs::write(&fragment, &weight).unwrap();
        let directory = parent.path().join("evidence");
        run(Arguments {
            config: config_path,
            weight_fragment: fragment,
            weight_sha256: artifacts::sha256(&weight),
            artifact_dir: directory.clone(),
        })
        .unwrap();
        // Independent column picks/sign changes: the expected output does
        // not call either matrix multiplication implementation.
        let expected = encode(&[
            0., 0., 0., 0., 0., 0., 2., 1., 1., -2., 0., 3., -1., 2., -3., -1., -2., 1., -1., 3.,
            2., 0., 1., 2.,
        ]);
        assert_eq!(
            std::fs::read(directory.join("analytic-output.bf16")).unwrap(),
            expected
        );
        let result: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("result.json")).unwrap()).unwrap();
        assert_eq!(result["passed"], true);
        assert_eq!(result["cleanup"]["runtime_live_allocations"], 0);
        assert_eq!(result["cleanup"]["managed_host_committed_bytes"], 0);
        assert_eq!(
            std::fs::read(directory.join("dense-output.bf16"))
                .unwrap()
                .len(),
            expected.len()
        );
    }
}
