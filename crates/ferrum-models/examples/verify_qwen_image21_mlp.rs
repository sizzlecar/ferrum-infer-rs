//! Verifies block 0 MLP from its already-modulated input z, not a full block.
//! The single basis row and sampled output coordinates are declared before execution.

#[path = "verify_qwen_image21_projection/artifacts.rs"]
mod artifacts;
#[path = "verify_qwen_image21_projection/arguments.rs"]
mod backend;
#[path = "verify_qwen_image21_mlp/execution.rs"]
mod execution;
#[path = "verify_qwen_image21_mlp/oracle.rs"]
mod oracle;
#[path = "support/tensor_component.rs"]
mod shared;

use anyhow::{bail, Context, Result};
use ferrum_models::qwen_image21::{
    QwenImage21Block0MlpFamily as Family, QwenImage21Config, StructureLimits,
};
use ferrum_quantization::SafetensorsArchive;
use serde_json::json;
use std::{collections::BTreeMap, ffi::OsString, path::PathBuf};

const USAGE: &str = "verify_qwen_image21_mlp --config CONFIG --model-dir TRANSFORMER_DIRECTORY --artifact-dir NEW_EXTERNAL_DIRECTORY [--index-file diffusion_pytorch_model.safetensors.index.json] [--backend reference|cuda] [CUDA budget options from verify_qwen_image21_projection]";

struct Arguments {
    config: PathBuf,
    model_dir: PathBuf,
    index_file: String,
    artifact_dir: PathBuf,
    backend: backend::Backend,
}

impl Arguments {
    fn parse(arguments: impl IntoIterator<Item = OsString>) -> Result<Option<Self>> {
        let mut args = arguments.into_iter().peekable();
        if args
            .peek()
            .is_some_and(|value| value == "--help" || value == "-h")
        {
            args.next();
            if args.next().is_some() {
                bail!("--help accepts no other arguments");
            }
            return Ok(None);
        }
        let mut values = BTreeMap::new();
        while let Some(flag) = args.next() {
            let flag = flag
                .into_string()
                .map_err(|_| anyhow::anyhow!("option names require UTF-8"))?;
            let value = args.next().context("every option requires a value")?;
            if values.insert(flag.clone(), value).is_some() {
                bail!("duplicate option {flag}");
            }
        }
        let config = values.remove("--config").context(USAGE)?.into();
        let model_dir = values.remove("--model-dir").context(USAGE)?.into();
        let artifact_dir = values.remove("--artifact-dir").context(USAGE)?.into();
        let index_file = values
            .remove("--index-file")
            .unwrap_or_else(|| "diffusion_pytorch_model.safetensors.index.json".into())
            .into_string()
            .map_err(|_| anyhow::anyhow!("index name requires UTF-8"))?;
        let backend = backend::Backend::parse_options(&mut values)?;
        backend.ensure_available()?;
        Ok(Some(Self {
            config,
            model_dir,
            index_file,
            artifact_dir,
            backend,
        }))
    }
}

fn run(args: Arguments) -> Result<()> {
    let limits = StructureLimits::default();
    let config_bytes =
        artifacts::read_bounded(&args.config, limits.metadata_file_bytes() as usize)?;
    let config = QwenImage21Config::from_hf_json(&config_bytes, limits)?;
    let source = SafetensorsArchive::open_with_index(&args.model_dir, &args.index_file)?;
    let h = usize::try_from(config.hidden_size())?;
    let m = usize::try_from(config.mlp_hidden_size())?;
    let tensors = Family::WEIGHT_NAMES
        .iter()
        .map(|name| source.tensor(name))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut evidence = Vec::new();
    for (index, tensor) in tensors.iter().enumerate() {
        let expected = if index == 2 {
            vec![h as u64, m as u64]
        } else {
            vec![m as u64, h as u64]
        };
        if tensor.dtype() != safetensors::Dtype::BF16 || tensor.shape() != expected {
            bail!(
                "MLP weight {} has wrong dtype/shape",
                Family::WEIGHT_NAMES[index]
            );
        }
        evidence.push(json!({"tensor": Family::WEIGHT_NAMES[index], "source_file": tensor.source_file(), "dtype": "BF16", "shape": expected, "bytes": tensor.bytes().len(), "payload_sha256": artifacts::sha256(tensor.bytes())}));
    }
    // All expectations, allowed activation sets and output envelopes precede
    // runtime creation. No candidate intermediate is fed into the oracle.
    let prepared = oracle::Prepared::new(
        h,
        m,
        tensors[0].bytes(),
        tensors[1].bytes(),
        tensors[2].bytes(),
    )?;
    let output = artifacts::Artifacts::create(&args.artifact_dir)?;
    output.json("declaration.json", &json!({
        "schema": "ferrum.qwen_image21.block0-mlp-verification.v1",
        "scope": "partial z -> gate/up -> SiLU-mul -> out; z is the already-modulated MLP entrance; no attention, residual gate, full block or image claim",
        "config": {"path": args.config.canonicalize()?, "sha256": artifacts::sha256(&config_bytes)},
        "source": {"directory": args.model_dir.canonicalize()?, "index": args.index_file, "weights": evidence, "integrity": "observed selected payload digests; original checkpoint authenticity and complete shard hashes not established"},
        "backend": args.backend,
        "oracle": prepared.declaration(),
        "memory_scope": "managed runtime input/output/static staging; source mmap, SHA scan, oracle vectors and control allocations excluded from Host/RSS bound"
    }))?;
    output.bytes("input-z.bf16", prepared.input())?;
    let result = execution::execute(
        &config,
        &source,
        &args.backend,
        prepared.input(),
        |values, provenance| {
            output.json("execution.json", provenance)?;
            for value in values {
                let name = match value.value_id().as_str() {
                    Family::GATE_ID => "gate.bf16",
                    Family::UP_ID => "up.bf16",
                    Family::ACTIVATION_ID => "activation.bf16",
                    Family::OUTPUT_ID => "output.bf16",
                    _ => bail!("unexpected MLP output port"),
                };
                output.bytes(name, value.bytes().bytes())?;
            }
            prepared.compare(values)
        },
    );
    match result {
        Ok((comparison, cleanup)) => {
            output.json(
                "result.json",
                &json!({"comparison": comparison, "cleanup": cleanup}),
            )?;
            if comparison["passed"] != true {
                bail!("MLP comparison failed; inspect complete tensors and result.json");
            }
            println!(
                "MLP partial verification passed: {}",
                output.directory().display()
            );
            Ok(())
        }
        Err(error) => {
            output.json(
                "failure.json",
                &json!({"passed": false, "error": format!("{error:#}")}),
            )?;
            Err(error)
        }
    }
}

fn main() -> Result<()> {
    match Arguments::parse(std::env::args_os().skip(1))? {
        Some(args) => run(args),
        None => {
            println!("{USAGE}\n{}", backend::USAGE);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_config_and_model_directory_are_required() {
        for args in [
            vec!["--model-dir", "model"],
            vec![
                "--config",
                "config",
                "--model-dir",
                "model",
                "--artifact-dir",
                "a",
                "--unknown",
                "1",
            ],
            vec!["--help", "extra"],
        ] {
            assert!(Arguments::parse(args.into_iter().map(OsString::from)).is_err());
        }
    }

    #[test]
    fn tiny_real_archive_runs_all_four_managed_nodes_and_retires() {
        let root = tempfile::tempdir().unwrap();
        let config = root.path().join("config.json");
        std::fs::write(&config, serde_json::to_vec(&json!({"_class_name":"QwenImage21Transformer2DModel", "num_layers":1, "num_attention_heads":1, "attention_head_dim":6, "context_in_dim":8, "in_channels":3, "out_channels":4, "patch_size":1, "mlp_ratio":3, "axes_dims_rope":[2,2,2], "causal_condition":true, "eps":0.000001})).unwrap()).unwrap();
        let gate = (0..108)
            .flat_map(|index| {
                half::bf16::from_f32(if index % 6 == 0 { 1.0 } else { 0.0 })
                    .to_bits()
                    .to_le_bytes()
            })
            .collect::<Vec<_>>();
        let up = gate.clone();
        let out = (0..108)
            .flat_map(|_| half::bf16::from_f32(0.25).to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let views = [
            safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![18, 6], &gate)
                .unwrap(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![18, 6], &up)
                .unwrap(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![6, 18], &out)
                .unwrap(),
        ];
        safetensors::tensor::serialize_to_file(
            Family::WEIGHT_NAMES.into_iter().zip(views),
            &None,
            &root.path().join("weights.safetensors"),
        )
        .unwrap();
        let weights = Family::WEIGHT_NAMES
            .into_iter()
            .map(|name| (name, "weights.safetensors"))
            .collect::<BTreeMap<_, _>>();
        std::fs::write(
            root.path().join("test.index.json"),
            serde_json::to_vec(&json!({"weight_map":weights})).unwrap(),
        )
        .unwrap();
        let artifact_dir = root.path().join("evidence");
        run(Arguments {
            config,
            model_dir: root.path().into(),
            index_file: "test.index.json".into(),
            artifact_dir: artifact_dir.clone(),
            backend: backend::Backend::Reference,
        })
        .unwrap();
        let result: serde_json::Value =
            serde_json::from_slice(&std::fs::read(artifact_dir.join("result.json")).unwrap())
                .unwrap();
        assert_eq!(result["comparison"]["passed"], true);
        assert_eq!(result["cleanup"]["managed_host_committed_bytes"], 0);
        assert_eq!(
            std::fs::read(artifact_dir.join("activation.bf16"))
                .unwrap()
                .len(),
            36
        );
    }
}
