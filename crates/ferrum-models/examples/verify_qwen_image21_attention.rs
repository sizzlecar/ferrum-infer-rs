//! Real block-0 attention branch after LN/modulation. Single-key probes verify
//! wiring only; causal/full pairs exercise nontrivial softmax and visibility.
#[path = "verify_qwen_image21_projection/artifacts.rs"]
mod artifacts;
#[path = "verify_qwen_image21_projection/arguments.rs"]
mod backend;
#[path = "verify_qwen_image21_attention/execution.rs"]
mod execution;
#[path = "verify_qwen_image21_attention/oracle.rs"]
mod oracle;
#[path = "support/tensor_component.rs"]
mod shared;

use anyhow::{bail, Context, Result};
use ferrum_models::qwen_image21::{
    QwenImage21AttentionConfig, QwenImage21AttentionLayout,
    QwenImage21Block0AttentionFamily as Family, QwenImage21Config, StructureLimits,
};
use ferrum_quantization::SafetensorsArchive;
use serde::Serialize;
use serde_json::json;
use std::{collections::BTreeMap, ffi::OsString, path::PathBuf};

const USAGE:&str="verify_qwen_image21_attention --config CONFIG --model-dir TRANSFORMER_DIRECTORY --artifact-dir NEW_EXTERNAL_DIRECTORY --probe single|causal-pair|full-pair [--index-file diffusion_pytorch_model.safetensors.index.json] [--backend reference|cuda] [CUDA budget options from verify_qwen_image21_projection]";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Probe {
    Single,
    CausalPair,
    FullPair,
}
impl Probe {
    pub(crate) const fn rows(self) -> usize {
        match self {
            Self::Single => 1,
            Self::CausalPair | Self::FullPair => 2,
        }
    }
    fn layout(self) -> Result<QwenImage21AttentionLayout> {
        Ok(QwenImage21AttentionLayout::uniform(
            self.rows() as u32,
            self == Self::FullPair,
        )?)
    }
    fn scope(self) -> &'static str {
        match self { Self::Single=>"single visible key: nonzero real-weight wiring only; not a multi-key softmax or mask verification",Self::CausalPair|Self::FullPair=>"two nonzero opposite basis rows; declared visibility and nontrivial online-softmax sample; not full-domain qualification" }
    }
}

struct Arguments {
    config: PathBuf,
    model_dir: PathBuf,
    index_file: String,
    artifact_dir: PathBuf,
    backend: backend::Backend,
    probe: Probe,
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
        let probe = match values.remove("--probe").as_deref() {
            Some(value) if value == "single" => Probe::Single,
            Some(value) if value == "causal-pair" => Probe::CausalPair,
            Some(value) if value == "full-pair" => Probe::FullPair,
            _ => bail!(
                "--probe single|causal-pair|full-pair is required; single does not verify softmax"
            ),
        };
        let backend = backend::Backend::parse_options(&mut values)?;
        backend.ensure_available()?;
        Ok(Some(Self {
            config,
            model_dir,
            index_file,
            artifact_dir,
            backend,
            probe,
        }))
    }
}

fn run(args: Arguments) -> Result<()> {
    let limits = StructureLimits::default();
    let config_bytes =
        artifacts::read_bounded(&args.config, usize::try_from(limits.metadata_file_bytes())?)?;
    let transformer = QwenImage21Config::from_hf_json(&config_bytes, limits)?;
    let config = QwenImage21AttentionConfig::new(transformer, args.probe.layout()?)?;
    let source = SafetensorsArchive::open_with_index(&args.model_dir, &args.index_file)?;
    let tensors = Family::WEIGHT_NAMES
        .iter()
        .map(|name| source.tensor(name))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut evidence = Vec::new();
    for (index, tensor) in tensors.iter().enumerate() {
        let expected = if index == 2 || index == 3 {
            vec![config.transformer().head_dim()]
        } else {
            vec![config.transformer().hidden_size(); 2]
        };
        if tensor.dtype() != safetensors::Dtype::BF16 || tensor.shape() != expected {
            bail!(
                "attention weight {} has wrong dtype/shape",
                Family::WEIGHT_NAMES[index]
            );
        }
        evidence.push(json!({"tensor":Family::WEIGHT_NAMES[index],"source_file":tensor.source_file(),"dtype":"BF16","shape":expected,"bytes":tensor.bytes().len(),"payload_sha256":artifacts::sha256(tensor.bytes())}));
    }
    // Every expected output and numerical-domain declaration precedes runtime
    // construction. The comparison never uses candidate Q/K or V as input.
    let prepared = oracle::Prepared::new(
        config.transformer(),
        args.probe,
        std::array::from_fn(|i| tensors[i].bytes()),
    )?;
    let output = artifacts::Artifacts::create(&args.artifact_dir)?;
    output.json("declaration.json",&json!({"schema":"ferrum.qwen_image21.block0-attention-verification.v1","probe":args.probe,"probe_scope":args.probe.scope(),"scope":"already-normalized/modulated x -> Q/K/V, QK normalization and identity-frequency RoPE -> structured attention -> BF16 -> output projection; no residual, MLP, prefix cache, full block or image claim","config":{"path":args.config.canonicalize()?,"sha256":artifacts::sha256(&config_bytes),"canonical_attention":config},"source":{"directory":args.model_dir.canonicalize()?,"index":args.index_file,"weights":evidence,"integrity":"observed selected payload digests, not checkpoint authentication or complete shard hashes"},"backend":args.backend,"oracle":prepared.declaration(),"memory_scope":"managed runtime input/output/static staging; source mmap, hashing, oracle vectors and control allocations excluded from Host/RSS bound"}))?;
    output.bytes("input-x.bf16", prepared.input())?;
    output.bytes("input-frequencies.f32", prepared.frequencies())?;
    let result = execution::execute(
        &config,
        &source,
        &args.backend,
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
                    Family::VALUE_ID => "value.bf16",
                    Family::ATTENTION_ID => "attention.bf16",
                    Family::OUTPUT_ID => "projected.bf16",
                    _ => bail!("unexpected attention output port"),
                };
                output.bytes(name, value.bytes().bytes())?;
            }
            prepared.compare(values)
        },
    );
    match result {
        Ok((comparison, cleanup)) => {
            output.json("result.json",&json!({"probe":args.probe,"probe_scope":args.probe.scope(),"comparison":comparison,"cleanup":cleanup}))?;
            if comparison["passed"] != true {
                bail!("attention comparison failed; inspect complete outputs and result.json");
            }
            println!(
                "Attention partial verification passed ({:?}): {}",
                args.probe,
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
#[path = "verify_qwen_image21_attention/tests.rs"]
mod tests;
