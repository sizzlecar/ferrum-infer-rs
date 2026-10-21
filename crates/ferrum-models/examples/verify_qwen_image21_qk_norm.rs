//! Real block-0 Q/K projection and head RMS normalization, before RoPE.
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
    QwenImage21Block0QkNormFamily as Family, QwenImage21Config, StructureLimits,
};
use ferrum_quantization::SafetensorsArchive;
use serde_json::json;
use std::{collections::BTreeMap, ffi::OsString, path::PathBuf};

const USAGE: &str = "verify_qwen_image21_qk_norm --config CONFIG --model-dir TRANSFORMER_DIRECTORY --artifact-dir NEW_EXTERNAL_DIRECTORY [--index-file diffusion_pytorch_model.safetensors.index.json] [--backend reference|cuda] [CUDA budget options from verify_qwen_image21_projection]";

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
    let output = artifacts::Artifacts::create(&args.artifact_dir)?;
    output.json("declaration.json", &json!({
        "schema":"ferrum.qwen_image21.block0-qk-norm-verification.v1",
        "scope":"partial already-normalized/modulated x -> Q/K Dense -> HeadRmsNorm; no RoPE, V, attention, residual, full block or image claim",
        "config":{"path":args.config.canonicalize()?,"sha256":artifacts::sha256(&config_bytes)},
        "source":{"directory":args.model_dir.canonicalize()?,"index":args.index_file,"weights":evidence,"integrity":"observed selected payload digests, not original checkpoint authentication or complete shard hashes"},
        "backend":args.backend,"oracle":prepared.declaration(),
        "memory_scope":"managed runtime input/output/static staging; source mmap, hashing, oracle vectors and control allocations excluded from Host/RSS bound"
    }))?;
    output.bytes("input-x.bf16", prepared.input())?;
    let result = execution::execute(
        &config,
        &source,
        &args.backend,
        prepared.input(),
        |values, provenance| {
            output.json("execution.json", provenance)?;
            for value in values {
                let name = match value.value_id().as_str() {
                    Family::QUERY_ID => "query.bf16",
                    Family::KEY_ID => "key.bf16",
                    Family::QUERY_NORMALIZED_ID => "query-normalized.bf16",
                    Family::KEY_NORMALIZED_ID => "key-normalized.bf16",
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
                "Q/K norm partial verification passed: {}",
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
        let artifacts = directory.path().join("evidence");
        run(Arguments {
            config,
            model_dir: directory.path().into(),
            index_file: "test.index.json".into(),
            artifact_dir: artifacts.clone(),
            backend: backend::Backend::Reference,
        })
        .unwrap();
        let result: serde_json::Value =
            serde_json::from_slice(&std::fs::read(artifacts.join("result.json")).unwrap()).unwrap();
        assert_eq!(result["comparison"]["passed"], true);
        assert_eq!(result["cleanup"]["managed_host_committed_bytes"], 0);
        let actual = std::fs::read(artifacts.join("query-normalized.bf16")).unwrap();
        let expected = bytes(vec![
            0.5, -1., 1.5, -2., 2.5, -3., 0.5, -1., 1.5, -2., 2.5, -3.,
        ]);
        assert_eq!(actual, expected);
    }
}
