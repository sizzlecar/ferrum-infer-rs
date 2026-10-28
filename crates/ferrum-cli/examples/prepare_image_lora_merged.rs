//! Export an equivalent BF16 checkpoint for fixed-LoRA performance comparisons.
//! Reuse the production composition contract, with one output matrix at a time.
//! Other model components are immutable hard links, never duplicate model copies.
use anyhow::{ensure, Context};
use clap::Parser;
use ferrum_models::qwen_image21::{ExpectedWeights, QwenImage21Config, StructureLimits};
use ferrum_quantization::{
    linear_adapter::{ImportedLinearAdapter, LinearAdapterScaling, LinearAdapterSource},
    safetensors_archive::SafetensorsNaming,
    SafetensorsArchive,
};
use ferrum_runtime::{HostMemoryAccount, HostMemoryClass, HostMemoryClassLimits, HostMemoryLimits};
use safetensors::{serialize_to_file, tensor::TensorView, Dtype};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

#[derive(Parser)]
struct Arguments {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    adapter: PathBuf,
    #[arg(long)]
    base_model: String,
    #[arg(long, default_value_t = 1.0)]
    strength: f32,
    #[arg(long)]
    alpha: Option<f32>,
    #[arg(long)]
    output_directory: PathBuf,
}

fn host(bytes: usize) -> anyhow::Result<HostMemoryAccount> {
    Ok(HostMemoryAccount::new(HostMemoryLimits::new(
        bytes.max(1),
        HostMemoryClassLimits {
            input_preprocess_bytes: 0,
            transfer_staging_bytes: 0,
            output_bytes: 0,
            weight_backing_bytes: bytes,
        },
    )?))
}

fn write_new(path: &Path, value: &serde_json::Value) -> anyhow::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

fn hard_link_tree(source: &Path, destination: &Path) -> anyhow::Result<()> {
    let kind = fs::symlink_metadata(source)?.file_type();
    if kind.is_dir() {
        fs::create_dir(destination)?;
        for entry in fs::read_dir(source)? {
            let entry = entry?;
            hard_link_tree(&entry.path(), &destination.join(entry.file_name()))?;
        }
    } else {
        ensure!(
            kind.is_file(),
            "model assets must be regular files: {source:?}"
        );
        fs::hard_link(source, destination).context("output must share the source filesystem")?;
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Arguments::parse();
    let started = Instant::now();
    let model = fs::canonicalize(&args.model)?;
    let output = if args.output_directory.is_absolute() {
        args.output_directory.clone()
    } else {
        std::env::current_dir()?.join(&args.output_directory)
    };
    let parent = fs::canonicalize(output.parent().context("output has no parent")?)?;
    ensure!(
        !parent.starts_with(&model),
        "output must be outside the immutable source model"
    );
    let mut config = Vec::new();
    File::open(model.join("transformer/config.json"))?
        .take(4 * 1024 * 1024 + 1)
        .read_to_end(&mut config)?;
    ensure!(
        config.len() <= 4 * 1024 * 1024,
        "transformer metadata exceeds 4 MiB"
    );
    let expected = ExpectedWeights::from_config(
        &QwenImage21Config::from_hf_json(&config, StructureLimits::default())?,
        StructureLimits::default(),
    )?;
    let archive = SafetensorsArchive::open_standard_component(
        &model,
        "transformer",
        SafetensorsNaming::Diffusers,
        4 * 1024 * 1024,
    )?;
    ensure!(
        archive.tensor_count() == expected.tensors().len(),
        "base tensor inventory differs"
    );
    let mut owners = BTreeMap::new();
    for tensor in expected.tensors().values() {
        let view = archive.tensor(tensor.external_name())?;
        ensure!(
            view.dtype() == Dtype::BF16 && view.shape() == tensor.dimensions(),
            "base tensor shape/type differs"
        );
        let bytes = view.retained_host_memory().owner_bytes();
        owners.insert(bytes.as_ptr() as usize, bytes.len());
    }
    let source = if args.adapter.is_dir() {
        ensure!(args.alpha.is_none(), "PEFT uses its own alpha");
        LinearAdapterSource::Peft {
            directory: args.adapter.clone(),
            base_model: Some(args.base_model.clone()),
        }
    } else {
        LinearAdapterSource::Diffusers {
            file: args.adapter.clone(),
            base_model: args.base_model.clone(),
            scaling: args.alpha.map_or(LinearAdapterScaling::Unit, |alpha| {
                LinearAdapterScaling::Alpha { alpha }
            }),
        }
    };
    let prepared = Arc::new(expected.bind_lora(
        Arc::new(ImportedLinearAdapter::open(&source)?),
        args.strength,
    )?);
    let schema = expected.weight_schema()?;
    let sizing = host(1)?;
    let base_bytes = owners.values().try_fold(0usize, |sum, bytes| {
        sum.checked_add(sizing.backing_bytes(*bytes)?)
            .context("base mapping bytes overflow")
    })?;
    let adapter_bytes = prepared.mapped_backing_bytes(&sizing)?;
    let mut maximum_active_bytes = 0usize;
    for component in &schema.components {
        maximum_active_bytes = maximum_active_bytes.max(
            prepared
                .workspace_bytes(&sizing, component)?
                .checked_add(prepared.output_backing_bytes(&sizing, component)?)
                .context("composition bytes overflow")?,
        );
    }
    let limit = base_bytes
        .checked_add(adapter_bytes)
        .and_then(|n| n.checked_add(maximum_active_bytes))
        .context("Host limit overflows")?;
    let account = host(limit)?;
    let _base_claims = owners
        .values()
        .map(|bytes| account.reserve(HostMemoryClass::WeightBacking, *bytes))
        .collect::<Result<Vec<_>, _>>()?;
    let adapter = prepared.manage(&account)?;
    fs::create_dir(&output)?;
    for component in [
        "model_index.json",
        "scheduler",
        "processor",
        "text_encoder",
        "vae",
    ] {
        hard_link_tree(&model.join(component), &output.join(component))?;
    }
    let transformer = output.join("transformer");
    fs::create_dir(&transformer)?;
    fs::hard_link(
        model.join("transformer/config.json"),
        transformer.join("config.json"),
    )?;
    let mut index = BTreeMap::new();
    let mut source_hash = Sha256::new();
    let mut output_hash = Sha256::new();
    let mut compose_time = Duration::ZERO;
    let mut composed = 0usize;
    for (ordinal, component) in schema.components.iter().enumerate() {
        let name = &component.external_names[0];
        let original = archive.tensor(name)?;
        let begin = Instant::now();
        let merged = if adapter.affects(component) {
            composed += 1;
            Some(adapter.compose(&account, component, original.bytes())?)
        } else {
            None
        };
        compose_time += begin.elapsed();
        let bytes = merged
            .as_ref()
            .map_or(original.bytes(), |buffer| buffer.bytes());
        for (hash, data) in [
            (&mut source_hash, original.bytes()),
            (&mut output_hash, bytes),
        ] {
            hash.update((name.len() as u64).to_le_bytes());
            hash.update(name.as_bytes());
            hash.update((data.len() as u64).to_le_bytes());
            hash.update(data);
        }
        let shape = original
            .shape()
            .iter()
            .copied()
            .map(usize::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        let tensor = TensorView::new(Dtype::BF16, shape, bytes)?;
        let filename = format!("diffusion_pytorch_model-{ordinal:05}.safetensors");
        let path = transformer.join(&filename);
        serialize_to_file([(name.as_str(), &tensor)], &None, &path)?;
        File::open(path)?.sync_all()?;
        index.insert(name.clone(), filename);
    }
    write_new(
        &transformer.join("diffusion_pytorch_model.safetensors.index.json"),
        &serde_json::json!({"metadata":{"total_size":expected.payload_bytes()},"weight_map":index}),
    )?;
    let report = serde_json::json!({
        "source_model":model,"adapter":source,"strength":args.strength,"adapter_identity":adapter.identity(),
        "output_directory":output,"tensors":schema.components.len(),"composed_tensors":composed,
        "payload_bytes":expected.payload_bytes(),"source_canonical_tensor_sha256":format!("{:x}",source_hash.finalize()),
        "output_canonical_tensor_sha256":format!("{:x}",output_hash.finalize()),
        "host_limit_bytes":limit,"mapped_base_bytes":base_bytes,"mapped_adapter_bytes":adapter_bytes,
        "maximum_active_composition_bytes":maximum_active_bytes,"compose_seconds":compose_time.as_secs_f64(),
        "export_seconds":started.elapsed().as_secs_f64(),
        "scope":"equivalent premerged BF16 performance baseline; production composition reused, not an independent numerical oracle; remaining components are immutable hard links",
    });
    write_new(&output.join("ferrum-lora-export.json"), &report)?;
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}
