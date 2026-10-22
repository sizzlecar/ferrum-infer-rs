//! Real block-0 Q/K projection and head RMS normalization, optionally followed
//! by adjacent RoPE with explicit identity/quadrant frequency probes.
//! One basis input is explicitly already normalized/modulated at the entrance.

#[path = "verify_qwen_image21_projection/artifacts.rs"]
mod artifacts;
#[path = "verify_qwen_image21_projection/arguments.rs"]
mod backend;
#[path = "verify_qwen_image21_qk_norm/execution.rs"]
mod execution;
#[path = "verify_qwen_image21_qk_norm/oracle.rs"]
mod oracle;
#[path = "support/tensor_component.rs"]
mod shared;

use anyhow::{bail, Context, Result};
use ferrum_models::qwen_image21::{
    QwenImage21Block0QkNormFamily as Family, QwenImage21Config, QwenImage21QkFrontend as Frontend,
    StructureLimits,
};
use ferrum_quantization::SafetensorsArchive;
use serde_json::json;
use std::{collections::BTreeMap, ffi::OsString, path::PathBuf};

const USAGE: &str = "verify_qwen_image21_qk_norm --config CONFIG --model-dir TRANSFORMER_DIRECTORY --artifact-dir NEW_EXTERNAL_DIRECTORY [--index-file diffusion_pytorch_model.safetensors.index.json] [--backend reference|cuda] [--frontend norm|rotary] [--rotary-probe identity|quadrants (required only for rotary)] [CUDA budget options from verify_qwen_image21_projection]";

struct Arguments {
    config: PathBuf,
    model_dir: PathBuf,
    index_file: String,
    artifact_dir: PathBuf,
    backend: backend::Backend,
    frontend: Frontend,
    rotary_probe: Option<oracle::RotaryProbe>,
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
        let frontend = match values.remove("--frontend").as_deref() {
            None => Frontend::Norm,
            Some(value) if value == "norm" => Frontend::Norm,
            Some(value) if value == "rotary" => Frontend::Rotary,
            _ => bail!("--frontend must be norm or rotary"),
        };
        let rotary_probe = match values.remove("--rotary-probe").as_deref() {
            None => None,
            Some(value) if value == "identity" => Some(oracle::RotaryProbe::Identity),
            Some(value) if value == "quadrants" => Some(oracle::RotaryProbe::Quadrants),
            _ => bail!("--rotary-probe must be identity or quadrants"),
        };
        if (frontend == Frontend::Rotary) != rotary_probe.is_some() {
            bail!("--frontend rotary requires --rotary-probe; norm accepts no rotary probe");
        }
        let backend = backend::Backend::parse_options(&mut values)?;
        backend.ensure_available()?;
        Ok(Some(Self {
            config,
            model_dir,
            index_file,
            artifact_dir,
            backend,
            frontend,
            rotary_probe,
        }))
    }
}

fn run(args: Arguments) -> Result<()> {
    let limits = StructureLimits::default();
    let config_bytes =
        artifacts::read_bounded(&args.config, limits.metadata_file_bytes() as usize)?;
    let config = QwenImage21Config::from_hf_json(&config_bytes, limits)?;
    let source = SafetensorsArchive::open_with_index(&args.model_dir, &args.index_file)?;
    let hidden = usize::try_from(config.hidden_size())?;
    let head_dim = usize::try_from(config.head_dim())?;
    let tensors = Family::WEIGHT_NAMES
        .iter()
        .map(|name| source.tensor(name))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut evidence = Vec::new();
    for (index, tensor) in tensors.iter().enumerate() {
        let expected = if index < 2 {
            vec![config.hidden_size(), config.hidden_size()]
        } else {
            vec![config.head_dim()]
        };
        if tensor.dtype() != safetensors::Dtype::BF16 || tensor.shape() != expected {
            bail!(
                "Q/K norm weight {} has wrong dtype/shape",
                Family::WEIGHT_NAMES[index]
            );
        }
        evidence.push(json!({"tensor":Family::WEIGHT_NAMES[index],"source_file":tensor.source_file(),"dtype":"BF16","shape":expected,"bytes":tensor.bytes().len(),"payload_sha256":artifacts::sha256(tensor.bytes())}));
    }
    let epsilon =
        (config.epsilon().numerator() as f64 / config.epsilon().denominator() as f64) as f32;
    // All expected sets precede both runtime construction and execution.
    let prepared = oracle::Prepared::new(
        hidden,
        head_dim,
        epsilon,
        std::array::from_fn(|i| tensors[i].bytes()),
    )?;
    let prepared = match (args.frontend, args.rotary_probe) {
        (Frontend::Norm, None) => prepared,
        (Frontend::Rotary, Some(probe)) => prepared.with_rotary(probe)?,
        _ => bail!("frontend and rotary probe disagree"),
    };
    let output = artifacts::Artifacts::create(&args.artifact_dir)?;
    output.json("declaration.json", &json!({
        "schema":"ferrum.qwen_image21.block0-qk-frontend-verification.v2", "frontend":args.frontend,"rotary_probe":args.rotary_probe,
        "scope":"partial already-normalized/modulated x -> Q/K Dense -> HeadRmsNorm, optionally adjacent RoPE with declared probes; no generated model frequencies, V, attention, residual, full block or image claim",
        "config":{"path":args.config.canonicalize()?,"sha256":artifacts::sha256(&config_bytes)},
        "source":{"directory":args.model_dir.canonicalize()?,"index":args.index_file,"weights":evidence,"integrity":"observed selected payload digests, not original checkpoint authentication or complete shard hashes"},
        "backend":args.backend,"oracle":prepared.declaration(),
        "memory_scope":"managed runtime input/output/static staging; source mmap, hashing, oracle vectors and control allocations excluded from Host/RSS bound"
    }))?;
    output.bytes("input-x.bf16", prepared.input())?;
    if let Some(frequencies) = prepared.frequencies() {
        output.bytes("input-frequencies.f32", frequencies)?;
    }
    let result = execution::execute(
        &config,
        &source,
        &args.backend,
        args.frontend,
        prepared.input(),
        prepared.frequencies(),
        |values, provenance| {
            output.json("execution.json", provenance)?;
            for value in values {
                let name = match value.value_id().as_str() {
                    Family::QUERY_ID => "query.bf16",
                    Family::KEY_ID => "key.bf16",
                    Family::QUERY_NORMALIZED_ID => "query-normalized.bf16",
                    Family::KEY_NORMALIZED_ID => "key-normalized.bf16",
                    Family::QUERY_ROTATED_ID => "query-rotated.bf16",
                    Family::KEY_ROTATED_ID => "key-rotated.bf16",
                    _ => bail!("unexpected Q/K norm output port"),
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
                &json!({"comparison":comparison,"cleanup":cleanup}),
            )?;
            if comparison["passed"] != true {
                bail!("Q/K norm comparison failed; inspect complete outputs and result.json");
            }
            println!(
                "Q/K frontend partial verification passed: {}",
                output.directory().display()
            );
            Ok(())
        }
        Err(error) => {
            output.json(
                "failure.json",
                &json!({"passed":false,"error":format!("{error:#}")}),
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
    fn explicit_inputs_and_known_options_are_required() {
        for args in [
            vec!["--model-dir", "model"],
            vec![
                "--config",
                "c",
                "--model-dir",
                "m",
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
    fn tiny_real_archive_executes_two_projections_and_two_head_norms() {
        let directory = tempfile::tempdir().unwrap();
        let config = directory.path().join("config.json");
        std::fs::write(&config,serde_json::to_vec(&json!({"_class_name":"QwenImage21Transformer2DModel","num_layers":1,"num_attention_heads":2,"attention_head_dim":6,"context_in_dim":8,"in_channels":3,"out_channels":4,"patch_size":1,"mlp_ratio":3,"axes_dims_rope":[2,2,2],"causal_condition":true,"eps":0.75})).unwrap()).unwrap();
        let bytes = |values: Vec<f32>| {
            values
                .into_iter()
                .flat_map(|x| half::bf16::from_f32(x).to_bits().to_le_bytes())
                .collect::<Vec<_>>()
        };
        let query = bytes(
            (0..144)
                .map(|i| {
                    if i % 12 == 0 {
                        if i / 12 % 2 == 0 {
                            0.5
                        } else {
                            -0.5
                        }
                    } else {
                        0.0
                    }
                })
                .collect(),
        );
        let key = bytes(
            (0..144)
                .map(|i| {
                    if i % 12 == 0 {
                        if i / 12 % 2 == 0 {
                            -0.5
                        } else {
                            0.5
                        }
                    } else {
                        0.0
                    }
                })
                .collect(),
        );
        let gamma_q = bytes(vec![1., 2., 3., 4., 5., 6.]);
        let gamma_k = bytes(vec![-1., 2., -3., 4., -5., 6.]);
        let views = [
            safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![12, 12], &query)
                .unwrap(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![12, 12], &key)
                .unwrap(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![6], &gamma_q)
                .unwrap(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![6], &gamma_k)
                .unwrap(),
        ];
        safetensors::tensor::serialize_to_file(
            Family::WEIGHT_NAMES.into_iter().zip(views),
            &None,
            &directory.path().join("weights.safetensors"),
        )
        .unwrap();
        let weight_map = Family::WEIGHT_NAMES
            .into_iter()
            .map(|name| (name, "weights.safetensors"))
            .collect::<BTreeMap<_, _>>();
        std::fs::write(
            directory.path().join("test.index.json"),
            serde_json::to_vec(&json!({"weight_map":weight_map})).unwrap(),
        )
        .unwrap();
        let expected = bytes(vec![
            0.5, -1., 1.5, -2., 2.5, -3., 0.5, -1., 1.5, -2., 2.5, -3.,
        ]);
        for (name, frontend, rotary_probe) in [
            ("norm", Frontend::Norm, None),
            (
                "identity",
                Frontend::Rotary,
                Some(oracle::RotaryProbe::Identity),
            ),
            (
                "quadrants",
                Frontend::Rotary,
                Some(oracle::RotaryProbe::Quadrants),
            ),
        ] {
            let artifacts = directory.path().join(name);
            run(Arguments {
                config: config.clone(),
                model_dir: directory.path().into(),
                index_file: "test.index.json".into(),
                artifact_dir: artifacts.clone(),
                backend: backend::Backend::Reference,
                frontend,
                rotary_probe,
            })
            .unwrap();
            let result: serde_json::Value =
                serde_json::from_slice(&std::fs::read(artifacts.join("result.json")).unwrap())
                    .unwrap();
            assert_eq!(result["comparison"]["passed"], true);
            assert_eq!(result["cleanup"]["managed_host_committed_bytes"], 0);
            assert_eq!(
                std::fs::read(artifacts.join("query-normalized.bf16")).unwrap(),
                expected
            );
            let execution: serde_json::Value =
                serde_json::from_slice(&std::fs::read(artifacts.join("execution.json")).unwrap())
                    .unwrap();
            assert_eq!(execution["family"], frontend.family_id());
            assert_eq!(
                execution["nodes"].as_array().unwrap().len(),
                frontend.output_ids().len()
            );
            assert_eq!(
                execution["managed_input_bytes"],
                if frontend == Frontend::Rotary { 48 } else { 24 }
            );
            assert_eq!(
                execution["managed_output_bytes"],
                frontend.output_ids().len() * 24
            );
            match rotary_probe {
                Some(oracle::RotaryProbe::Identity) => assert_eq!(
                    std::fs::read(artifacts.join("query-rotated.bf16")).unwrap(),
                    expected
                ),
                Some(oracle::RotaryProbe::Quadrants) => assert_eq!(
                    std::fs::read(artifacts.join("query-rotated.bf16")).unwrap(),
                    bytes(vec![
                        0.5, -1., 2., 1.5, -2.5, 3., 0.5, -1., 2., 1.5, -2.5, 3.
                    ])
                ),
                None => assert!(!artifacts.join("query-rotated.bf16").exists()),
            }
        }
    }

    #[test]
    fn frontend_defaults_to_norm_and_rotary_requires_explicit_probe() {
        let base = ["--config", "c", "--model-dir", "m", "--artifact-dir", "a"];
        let parse = |tail: &[&str]| {
            Arguments::parse(
                base.into_iter()
                    .chain(tail.iter().copied())
                    .map(OsString::from),
            )
        };
        assert_eq!(parse(&[]).unwrap().unwrap().frontend, Frontend::Norm);
        let args = parse(&["--frontend", "rotary", "--rotary-probe", "quadrants"])
            .unwrap()
            .unwrap();
        assert_eq!(args.frontend, Frontend::Rotary);
        assert_eq!(args.rotary_probe, Some(oracle::RotaryProbe::Quadrants));
        for tail in [
            vec!["--frontend", "rotary"],
            vec!["--rotary-probe", "identity"],
            vec!["--frontend", "unknown"],
            vec!["--frontend", "rotary", "--rotary-probe", "unknown"],
        ] {
            assert!(parse(&tail).is_err());
        }
    }
}
