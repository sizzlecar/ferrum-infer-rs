//! Complete managed single-frame VAE decoder from an explicitly typed BF16
//! latent file. Uses source-backed packing/denormalization; does not run DiT.
#[path = "verify_qwen_image21_vae/arguments.rs"]
mod arguments;
#[path = "verify_qwen_image21_projection/artifacts.rs"]
mod artifacts;
#[path = "verify_qwen_image21_projection/arguments.rs"]
mod backend;
#[path = "verify_qwen_image21_vae/execution.rs"]
mod execution;
#[path = "verify_qwen_image21_vae/latent.rs"]
mod latent;
#[path = "verify_qwen_image21_vae/output.rs"]
mod output;
#[path = "verify_qwen_image21_vae/png_output.rs"]
mod png_output;
#[path = "support/tensor_component.rs"]
mod shared;

use anyhow::{ensure, Context, Result};
use arguments::{Arguments, LatentDomain};
use ferrum_models::qwen_image21::QwenImage21VaeDecoderConfig;
use ferrum_quantization::SafetensorsArchive;
use serde_json::{json, Value};

fn run(args: Arguments) -> Result<()> {
    args.backend.ensure_available()?;
    let config_bytes = artifacts::read_bounded(&args.config, 4 * 1024 * 1024)?;
    let config = QwenImage21VaeDecoderConfig::from_hf_json(
        &config_bytes,
        args.batch,
        args.height,
        args.width,
    )?;
    let raw_config: Value = serde_json::from_slice(&config_bytes)?;
    let output_tensor = config.output_tensor()?;
    png_output::dimensions(output_tensor.dimensions()[3], output_tensor.dimensions()[2])?;
    let statistics = if args.domain == LatentDomain::Normalized {
        Some(latent::Statistics::from_config(
            &raw_config,
            usize::try_from(config.latent_channels())?,
        )?)
    } else {
        None
    };
    let budget = latent::Budget::new(&config, args.staging_bytes(), args.maximum_host_bytes)?;
    let host = budget.account()?;
    let loaded = latent::load(
        &args.latent_file,
        &config,
        args.layout,
        args.domain,
        statistics.as_ref(),
        &host,
    )?;
    let vae_dir = args.model_root.join("vae");
    let source = match &args.index_file {
        Some(index) => SafetensorsArchive::open_with_index(&vae_dir, index)?,
        None => SafetensorsArchive::open(&vae_dir)?,
    };
    let mut weights = Vec::new();
    let mut weight_bytes = 0_u64;
    for expected in config.expected_weights()? {
        let actual = source.tensor(expected.external_name())?;
        ensure!(
            actual.dtype() == safetensors::Dtype::BF16
                && actual.shape() == expected.dimensions()
                && u64::try_from(actual.bytes().len())? == expected.bytes(),
            "weight header mismatch for {}",
            expected.external_name()
        );
        weight_bytes = weight_bytes
            .checked_add(expected.bytes())
            .context("weight extent overflow")?;
        weights.push(json!({"name":expected.external_name(),"shape":expected.dimensions(),"bytes":expected.bytes(),"shard":actual.source_file(),"sha256":artifacts::sha256(actual.bytes())}));
    }
    let artifacts = artifacts::Artifacts::create(&args.artifact_dir)?;
    artifacts.json("declaration.json",&json!({
        "schema":"ferrum.qwen_image21.vae-decoder-execution.v1","model_root":args.model_root.canonicalize()?,"config_path":args.config.canonicalize()?,"config_sha256":artifacts::sha256(&config_bytes),"config":config,"backend":args.backend,"cudnn":args.cudnn,
        "latent":{"path":args.latent_file.canonicalize()?,"layout":args.layout,"domain":args.domain,"input_shape":loaded.tensor.tensor().dimensions(),"source_sha256":loaded.source_sha256,"prepared_nchw_sha256":loaded.prepared_sha256,"arithmetic":"statistics: JSON->F32->BF16; normalized: B(B(x*B(std))+B(mean)); denormalized: unchanged bits"},
        "weights":weights,"selected_weight_bytes":weight_bytes,"integrity":"observed selected payload SHA-256; not checkpoint authentication or full shard hashes",
        "managed_host_budget":budget,"memory_scope":"input/output/static upload backing uses one Host account; source mmap residency, small metadata/weight digests/control allocations, fixed 8/16 KiB I/O stack buffers and bounded PNG codec state are outside this backing budget; CUDA opaque allowances remain experimental",
        "png_memory":{"strategy":"streaming, no complete image/PNG buffer","maximum_rgba_row_bytes":png_output::MAX_ROW_BYTES,"encoder_row_buffers":3,"idat_buffer_bytes":16384,"decoder_allocation_limit_bytes":8388608,"scope":"png crate row/deflate/decoder internals; not a managed-Host or total-RSS claim"},
        "scope":"full single-frame decoder without temporal cache/time_conv; raw RGBA and clamped pixel artifact; no DiT/scheduler, no image-generation end-to-end or Torch parity claim",
        "source":"https://github.com/vllm-project/vllm-omni/blob/c0551d22674d0db81305c800b66ad83035fdc980/vllm_omni/diffusion/models/qwen_image_21/pipeline_qwen_image_21.py#L1025"
    }))?;
    artifacts.bytes(
        "latent-denormalized-nchw.bf16",
        loaded.tensor.bytes().bytes(),
    )?;
    let result = execution::run(
        &config,
        loaded.tensor,
        &host,
        &budget,
        &source,
        &args,
        weight_bytes,
        |rgba| output::save(rgba, &artifacts),
    );
    match result {
        Ok((validation, provenance, cleanup)) => {
            artifacts.json("execution.json", &provenance)?;
            artifacts.json(
                "result.json",
                &json!({"validation":validation,"cleanup":cleanup}),
            )?;
            ensure!(
                validation["passed"] == true,
                "VAE produced nonfinite output; raw diagnostic retained"
            );
            println!(
                "Full single-frame VAE execution/finite smoke completed: {}",
                artifacts.directory().display()
            );
            Ok(())
        }
        Err(error) => {
            artifacts.json(
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
            println!("{}\n{}", arguments::USAGE, backend::USAGE);
            Ok(())
        }
    }
}
#[cfg(test)]
#[path = "verify_qwen_image21_vae/tests.rs"]
mod tests;
