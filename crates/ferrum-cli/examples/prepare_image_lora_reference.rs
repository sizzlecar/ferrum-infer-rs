//! Export a complete model-bound adapter to ordinary PEFT for independent
//! reference runners. No inference or floating-point weight arithmetic occurs:
//! only declared names and model-declared B-row slices change. Outputs are new
//! files outside the repository; source tensors remain immutable mmap views.
use anyhow::{ensure, Context};
use clap::Parser;
use ferrum_models::qwen_image21::{ExpectedWeights, QwenImage21Config, StructureLimits};
use ferrum_quantization::linear_adapter::{
    ImportedLinearAdapter, LinearAdapterScaling, LinearAdapterSource,
};
use ferrum_runtime::{ElementType, HostMemoryAccount, HostMemoryClassLimits, HostMemoryLimits};
use safetensors::{serialize_to_file, tensor::TensorView, Dtype};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::PathBuf,
    sync::Arc,
};

#[derive(Parser)]
struct Arguments {
    #[arg(long)]
    transformer_config: PathBuf,
    #[arg(long)]
    adapter: PathBuf,
    #[arg(long)]
    base_model: String,
    #[arg(long)]
    alpha: Option<f32>,
    #[arg(long)]
    output_directory: PathBuf,
    /// Explicit naming differences in the reference implementation: FROM=TO.
    /// Each suffix must match at least one complete model-declared target.
    #[arg(long)]
    rename_module_suffix: Vec<String>,
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
fn dtype(kind: ElementType) -> anyhow::Result<Dtype> {
    match kind {
        ElementType::Bf16 => Ok(Dtype::BF16),
        ElementType::F16 => Ok(Dtype::F16),
        ElementType::F32 => Ok(Dtype::F32),
        _ => anyhow::bail!("adapter storage is not a supported float type"),
    }
}
fn write_new(path: PathBuf, value: &serde_json::Value) -> anyhow::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}
fn main() -> anyhow::Result<()> {
    let args = Arguments::parse();
    let mut config = Vec::new();
    File::open(&args.transformer_config)?
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
    let imported = Arc::new(ImportedLinearAdapter::open(&source)?);
    let prepared = Arc::new(expected.bind_lora(imported.clone(), 1.0)?);
    let account = host(prepared.mapped_backing_bytes(&host(1)?)?)?;
    let admitted = prepared.manage(&account)?;
    let first = imported
        .matrices()
        .values()
        .next()
        .context("empty adapter")?;
    let rank = first.rank();
    let multiplier = first.multiplier();
    let mut rewrites = args
        .rename_module_suffix
        .iter()
        .map(|value| {
            let (from, to) = value.split_once('=').context("rewrite must be FROM=TO")?;
            ensure!(!from.is_empty() && !to.is_empty(), "empty module suffix");
            Ok((from, to, 0usize))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut tensors = BTreeMap::new();
    let mut targets = BTreeSet::new();
    let mut routes = Vec::new();
    for (module, matrix) in imported.matrices() {
        ensure!(
            matrix.rank() == rank && matrix.multiplier().to_bits() == multiplier.to_bits(),
            "reference PEFT export requires a uniform declared rank and alpha/r"
        );
        let a = matrix.a();
        let b = matrix.b();
        for slice in expected.lora_target(module)?.slices() {
            let original = slice
                .tensor()
                .external_name()
                .strip_suffix(".weight")
                .context("target is not a weight")?;
            let mut target = original.to_owned();
            for (from, to, matched) in &mut rewrites {
                if let Some(prefix) = target.strip_suffix(*from) {
                    target = format!("{prefix}{to}");
                    *matched += 1;
                }
            }
            ensure!(
                targets.insert(target.clone()),
                "target rewrite collides with another matrix"
            );
            let rows = slice.adapter_output_rows();
            let row_bytes = usize::try_from(rank)?
                .checked_mul(b.element_type.size_bytes() as usize)
                .context("row bytes overflow")?;
            let start = usize::try_from(rows.start)?
                .checked_mul(row_bytes)
                .context("row offset overflow")?;
            let end = usize::try_from(rows.end)?
                .checked_mul(row_bytes)
                .context("row offset overflow")?;
            let prefix = format!("base_model.model.transformer.{target}");
            tensors.insert(
                format!("{prefix}.lora_A.weight"),
                TensorView::new(
                    dtype(a.element_type)?,
                    a.dimensions
                        .into_iter()
                        .map(usize::try_from)
                        .collect::<Result<_, _>>()?,
                    a.bytes,
                )?,
            );
            tensors.insert(
                format!("{prefix}.lora_B.weight"),
                TensorView::new(
                    dtype(b.element_type)?,
                    vec![
                        usize::try_from(rows.end - rows.start)?,
                        usize::try_from(rank)?,
                    ],
                    b.bytes
                        .get(start..end)
                        .context("B-row slice exceeds storage")?,
                )?,
            );
            routes.push(serde_json::json!({"source_module":module,"target_module":target,"b_rows":[rows.start,rows.end]}));
        }
    }
    ensure!(
        rewrites.iter().all(|(_, _, matched)| *matched != 0),
        "a declared reference rewrite matched no target"
    );
    // Directory creation is exclusive; serialize_to_file cannot overwrite a
    // pre-existing user file. Publish the weights only after the full write.
    std::fs::create_dir(&args.output_directory)?;
    let partial = args
        .output_directory
        .join("adapter_model.safetensors.partial");
    serialize_to_file(
        tensors.iter().map(|(name, tensor)| (name.as_str(), tensor)),
        &None,
        &partial,
    )?;
    File::open(&partial)?.sync_all()?;
    std::fs::rename(
        &partial,
        args.output_directory.join("adapter_model.safetensors"),
    )?;
    write_new(
        args.output_directory.join("adapter_config.json"),
        &serde_json::json!({
            "peft_type":"LORA","r":rank,"lora_alpha":multiplier * rank as f32,
            "base_model_name_or_path":args.base_model,"target_modules":targets,
            "bias":"none","fan_in_fan_out":false,"use_dora":false,"use_rslora":false,
        }),
    )?;
    let report = serde_json::json!({
        "source":source,"admitted_identity_at_unit_strength":admitted.identity(),
        "mapped_backing_bytes":admitted.mapped_backing_bytes(),"source_modules":imported.matrices().len(),
        "exported_matrices":tensors.len(),"routes":routes,"rewrites":args.rename_module_suffix,
        "arithmetic":"none; exact encoded A and model-declared B-row slices; scaling retained in PEFT config",
    });
    write_new(args.output_directory.join("export-receipt.json"), &report)?;
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}
