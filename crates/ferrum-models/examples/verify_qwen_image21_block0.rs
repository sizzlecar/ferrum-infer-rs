//! Complete block-0 execution and optional Reference-artifact differential.
//! This is not an independent whole-block numerical oracle or Torch parity.
#[path = "verify_qwen_image21_projection/artifacts.rs"]
mod artifacts;
#[path = "verify_qwen_image21_projection/arguments.rs"]
mod backend;
#[path = "verify_qwen_image21_block0/evidence.rs"]
mod evidence;
#[path = "verify_qwen_image21_block0/execution.rs"]
mod execution;
#[path = "verify_qwen_image21_block0/probe.rs"]
mod probe;
#[path = "support/tensor_component.rs"]
mod shared;

use anyhow::{bail, Context, Result};
use ferrum_models::qwen_image21::{
    QwenImage21Block0Family as Family, QwenImage21Config, StructureLimits,
};
use ferrum_quantization::SafetensorsArchive;
use serde_json::json;
use std::{collections::BTreeMap, ffi::OsString, path::PathBuf};

const USAGE: &str = "verify_qwen_image21_block0 --config CONFIG --model-dir TRANSFORMER_DIRECTORY --artifact-dir NEW_EXTERNAL_DIRECTORY [--reference-artifact-dir REFERENCE_RUN_DIRECTORY] [--index-file diffusion_pytorch_model.safetensors.index.json] [--backend reference|cuda] [--reference-memory-bytes BYTES (Reference only; default 1073741824)] [CUDA budget options from verify_qwen_image21_projection]";

const DEFAULT_REFERENCE_MEMORY_BYTES: u64 = 1 << 30;

struct Arguments {
    config: PathBuf,
    model_dir: PathBuf,
    index_file: String,
    artifact_dir: PathBuf,
    reference_artifact_dir: Option<PathBuf>,
    reference_memory_bytes: u64,
    backend: backend::Backend,
}
impl Arguments {
    fn parse(arguments: impl IntoIterator<Item = OsString>) -> Result<Option<Self>> {
        let mut args = arguments.into_iter().peekable();
        if args.peek().is_some_and(|v| v == "--help" || v == "-h") {
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
        let reference_artifact_dir = values.remove("--reference-artifact-dir").map(PathBuf::from);
        let index_file = values
            .remove("--index-file")
            .unwrap_or_else(|| "diffusion_pytorch_model.safetensors.index.json".into())
            .into_string()
            .map_err(|_| anyhow::anyhow!("index name requires UTF-8"))?;
        let explicit_reference_memory = values.remove("--reference-memory-bytes");
        let reference_memory_bytes = match explicit_reference_memory.as_ref() {
            Some(value) => value
                .to_str()
                .context("Reference memory budget requires UTF-8")?
                .parse::<u64>()?,
            None => DEFAULT_REFERENCE_MEMORY_BYTES,
        };
        if reference_memory_bytes == 0 || reference_memory_bytes >= isize::MAX as u64 {
            bail!("Reference memory budget must be positive and below isize::MAX");
        }
        let backend = backend::Backend::parse_options(&mut values)?;
        if explicit_reference_memory.is_some() && !matches!(backend, backend::Backend::Reference) {
            bail!("--reference-memory-bytes applies only to the Reference backend");
        }
        backend.ensure_available()?;
        Ok(Some(Self {
            config,
            model_dir,
            index_file,
            artifact_dir,
            reference_artifact_dir,
            reference_memory_bytes,
            backend,
        }))
    }
}

fn run(args: Arguments) -> Result<()> {
    let limits = StructureLimits::default();
    let config_bytes =
        artifacts::read_bounded(&args.config, usize::try_from(limits.metadata_file_bytes())?)?;
    let transformer = QwenImage21Config::from_hf_json(&config_bytes, limits)?;
    let config = probe::config(transformer)?;
    let inputs = probe::Inputs::new(&config)?;
    let source = SafetensorsArchive::open_with_index(&args.model_dir, &args.index_file)?;
    let identity = evidence::Identity::new(&config, &config_bytes, &inputs, &source)?;
    let specs = probe::output_specs(&config)?;
    // Bind the complete declared reference before creating a candidate runtime.
    let reference = args
        .reference_artifact_dir
        .as_ref()
        .map(|path| evidence::Reference::load(path, &identity, &specs))
        .transpose()?;
    let output = artifacts::Artifacts::create(&args.artifact_dir)?;
    output.json("declaration.json", &json!({
        "schema":"ferrum.qwen_image21.block0-execution.v1", "identity":identity,
        "probe":probe::declaration(), "backend":args.backend,
        "validation":evidence::declaration(reference.is_some()),
        "reference_directory":reference.as_ref().map(|r|r.directory()),
        "source":{"config_path":args.config.canonicalize()?,"model_directory":args.model_dir.canonicalize()?,"index":args.index_file,
            "integrity":"observed selected payload SHA-256, not checkpoint authentication or complete shard hashes"},
        "scope":"one complete block with supplied conditioning and identity-frequency probe; not conditioning encoder, full transformer, diffusion step or image generation",
        "memory_scope":"managed input/output/static staging and plan retirement; source mmap, hashing, probe/reference vectors and control allocations are outside the Host/RSS bound"
    }))?;
    for (input, bytes) in identity.inputs.iter().zip(&inputs.bytes) {
        output.bytes(&input.file, bytes)?;
    }
    let result = execution::execute(
        &config,
        &source,
        &args.backend,
        args.reference_memory_bytes,
        &inputs,
        |values, provenance| {
            output.json("execution.json", provenance)?;
            if values.len() != specs.len() {
                bail!("execution did not return all 21 boundaries");
            }
            let mut outputs = Vec::new();
            let mut comparisons = Vec::new();
            for (index, spec) in specs.iter().enumerate() {
                let matches = values
                    .iter()
                    .filter(|value| value.value_id().as_str() == spec.value_id)
                    .collect::<Vec<_>>();
                if matches.len() != 1 {
                    bail!("missing or repeated output {}", spec.value_id);
                }
                let value = matches[0];
                if value.tensor().dimensions() != spec.shape
                    || value.tensor().element_type() != ferrum_runtime::ElementType::Bf16
                    || value.tensor().layout() != &ferrum_runtime::ResolvedTensorLayout::Contiguous
                {
                    bail!("output shape/dtype/layout differs from declared boundary");
                }
                let bytes = value.bytes().bytes();
                output.bytes(&spec.file, bytes)?;
                outputs.push(evidence::TensorDigest::from_spec(spec, bytes)?);
                comparisons.push(evidence::compare(
                    spec,
                    bytes,
                    reference.as_ref().map(|r| r.output(index)),
                )?);
            }
            output.json(
                "manifest.json",
                &evidence::Manifest {
                    schema: evidence::MANIFEST_SCHEMA.into(),
                    backend: evidence::backend_name(&args.backend).into(),
                    identity: identity.clone(),
                    outputs,
                },
            )?;
            Ok(evidence::report(comparisons, reference.is_some()))
        },
    );
    match result {
        Ok((validation, cleanup)) => {
            output.json(
                "result.json",
                &json!({"validation":validation,"cleanup":cleanup}),
            )?;
            if validation["passed"] != true {
                bail!("block-0 finite/differential validation failed; inspect result.json and complete outputs");
            }
            println!(
                "Block-0 {} completed: {}",
                if reference.is_some() {
                    "Reference differential"
                } else {
                    "execution_smoke_only"
                },
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
#[path = "verify_qwen_image21_block0/tests.rs"]
mod tests;
