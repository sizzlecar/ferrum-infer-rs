//! One managed full-transformer velocity evaluation over externally supplied
//! latent/text tensors. No synthetic replacement for text encoding, solver or VAE.

#[path = "verify_qwen_image21_transformer/arguments.rs"]
mod arguments;
#[path = "verify_qwen_image21_projection/artifacts.rs"]
mod artifacts;
#[path = "verify_qwen_image21_projection/arguments.rs"]
mod backend;
#[path = "verify_qwen_image21_transformer/execution.rs"]
mod execution;
#[path = "verify_qwen_image21_transformer/inputs.rs"]
mod inputs;
#[path = "support/tensor_component.rs"]
mod shared;

use anyhow::{bail, Context, Result};
use ferrum_models::qwen_image21::{
    ExpectedWeights, QwenImage21Config, QwenImage21RotaryFrequencies, QwenImage21T2iLayout,
    QwenImage21Timestep256, QwenImage21TransformerConfig, QwenImage21TransformerFamily as Family,
    StructureLimits,
};
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::{ElementType, ProgramValueId};
use serde_json::json;

fn run(args: arguments::Arguments) -> Result<()> {
    args.backend.ensure_available()?;
    let limits = StructureLimits::default();
    let config_bytes =
        artifacts::read_bounded(&args.config, usize::try_from(limits.metadata_file_bytes())?)?;
    let transformer = QwenImage21Config::from_hf_json(&config_bytes, limits)?;
    let config = QwenImage21TransformerConfig::new(
        transformer,
        QwenImage21T2iLayout::new(
            args.text_tokens,
            args.latent_height,
            args.latent_width,
            args.valid_text_lengths.clone(),
        )?,
    )?;
    let inventory = ExpectedWeights::from_config(config.transformer(), limits)?;
    let capacity = match &args.backend {
        backend::Backend::Reference => args.reference_memory_bytes,
        backend::Backend::Cuda(options) => options.usable_memory_bytes,
    };
    if inventory.payload_bytes() > capacity {
        bail!("the declared device capacity cannot hold {} weight bytes, before dynamic/opaque overhead",inventory.payload_bytes());
    }
    let inputs = inputs::Inputs::prepare(&config, &args)?;
    let source = SafetensorsArchive::open_with_index(&args.model_dir, &args.index_file)?;
    // Metadata checks cover the complete expected schema before hashing any
    // payload. Hashing observes source integrity, not checkpoint authenticity.
    for expected in inventory.tensors().values() {
        let tensor = source.tensor(expected.external_name())?;
        if tensor.dtype() != safetensors::Dtype::BF16
            || tensor.shape() != expected.dimensions()
            || tensor.bytes().len() as u64 != expected.bytes()
        {
            bail!(
                "weight {} differs from the complete typed transformer schema",
                expected.external_name()
            );
        }
    }
    let output = artifacts::Artifacts::create(&args.artifact_dir)?;
    let weights=inventory.tensors().values().map(|expected| {
        let tensor=source.tensor(expected.external_name())?;
        Ok(json!({"tensor":expected.external_name(),"shape":expected.dimensions(),"dtype":"BF16",
            "bytes":expected.bytes(),"sha256":artifacts::sha256(tensor.bytes())}))
    }).collect::<Result<Vec<_>>>()?;
    let specs = config.input_tensor_specs()?;
    let names = [
        "latents.bf16",
        "text.bf16",
        "time_features.bf16",
        "frequencies.f32",
    ];
    let mut input_records = Vec::new();
    for ((name, data), value) in names.into_iter().zip(&inputs.bytes).zip(Family::INPUT_IDS) {
        let spec = &specs[&ProgramValueId::new(value)?];
        output.bytes(name, data)?;
        input_records.push(json!({"value_id":value,"file":name,"shape":spec.dimensions,
            "dtype":spec.element_type,"bytes":data.len(),"sha256":artifacts::sha256(data)}));
    }
    output.json("declaration.json",&json!({
        "schema":"ferrum.qwen_image21.transformer-step.v1",
        "scope":"one full configured transformer velocity evaluation; supplied text/latent tensors; no TE, scheduler update, VAE, image or throughput claim",
        "validation":{"mode":"execution_smoke_only","finite_velocity_required":true,"numerical_verification":"not performed"},
        "canonical_config":config,"config_file_sha256":artifacts::sha256(&config_bytes),
        "source":{"config":args.config.canonicalize()?,"model_directory":args.model_dir.canonicalize()?,
            "index":args.index_file,"integrity":"observed payload hashes, not checkpoint authentication"},
        "inputs":input_records,"weights":weights,"weight_bytes":inventory.payload_bytes(),"backend":args.backend,
        "preprocessing":{"scheduler_timesteps":args.scheduler_timesteps,
            "normalized_model_timestep_bf16_bits":inputs.model_timesteps.iter().map(|value|value.bf16_bits()).collect::<Vec<_>>(),
            "timestep_profile":QwenImage21Timestep256::NUMERICAL_PROFILE_ID,
            "rotary_profile":QwenImage21RotaryFrequencies::NUMERICAL_PROFILE_ID,
            "timestep_boundary":"B(RN32(B(raw scheduler timestep)/1000)); early model B retained; causal shared t0 appended",
            "frequency_layout":"F32[S,D/2,2], frame/height/width axes, padded text T and centered target grid",
            "qualification":"libm0.2.15 CPU arithmetic, no Torch/libdevice bitwise claim"},
        "memory_scope":"managed execution input/output/static staging and retirement; source mmap/hash scans, source/preprocessing vectors and control allocations outside Host/RSS accounting",
        "maximum_input_payload_bytes":args.maximum_input_bytes,
    }))?;
    let started = std::time::Instant::now();
    let result = execution::execute(&config, &source, &args, &inputs, |values, provenance| {
        output.json("execution.json", provenance)?;
        let [velocity] = values else {
            bail!("full transformer must retain exactly one velocity output");
        };
        if velocity.value_id().as_str() != Family::VELOCITY_ID
            || velocity.tensor().element_type() != ElementType::Bf16
        {
            bail!("transformer returned the wrong completion ABI");
        }
        let bytes = velocity.bytes().bytes();
        output.bytes("velocity.bf16", bytes)?;
        let mut nonfinite = 0_usize;
        let mut maximum_absolute = 0.0_f64;
        let mut absolute_sum = 0.0_f64;
        for chunk in bytes.chunks_exact(2) {
            let number = f32::from_bits(u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16);
            if !number.is_finite() {
                nonfinite += 1;
            } else {
                let magnitude = f64::from(number).abs();
                maximum_absolute = maximum_absolute.max(magnitude);
                absolute_sum += magnitude;
            }
        }
        let elements = bytes.len() / 2;
        Ok(
            json!({"mode":"execution_smoke_only","numerical_verification":"not performed","passed":nonfinite==0,
            "nonfinite_count":nonfinite,"elements":elements,"shape":velocity.tensor().dimensions(),
            "output_file":"velocity.bf16","output_sha256":artifacts::sha256(bytes),
            "maximum_absolute_value":maximum_absolute,"mean_absolute_value":absolute_sum/elements as f64}),
        )
    });
    match result {
        Ok((validation, cleanup)) => {
            output.json("result.json",&json!({"validation":validation,"cleanup":cleanup,
                "cold_execution_and_inspection_seconds":started.elapsed().as_secs_f64(),
                "timing_scope":"runtime creation, compile, weight initialization, execution, readback and artifact inspection; not a throughput benchmark"}))?;
            if validation["passed"] != true {
                bail!("transformer produced nonfinite velocity; inspect saved result");
            }
            println!(
                "Transformer execution smoke completed: {}",
                output.directory().display()
            );
            Ok(())
        }
        Err(error) => {
            output.json(
                "failure.json",
                &json!({"passed":false,"error":format!("{error:#}")}),
            )?;
            Err(error).context("managed transformer step failed")
        }
    }
}

fn main() -> Result<()> {
    match arguments::Arguments::parse(std::env::args_os().skip(1))? {
        Some(args) => run(args),
        None => {
            println!("{}\n{}", arguments::USAGE, backend::USAGE);
            Ok(())
        }
    }
}
