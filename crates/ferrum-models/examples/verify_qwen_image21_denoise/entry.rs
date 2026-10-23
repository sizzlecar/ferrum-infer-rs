use super::{
    arguments::Arguments,
    artifacts, backend, composition,
    inputs::{HostBudget, Inputs, TextMetadata},
    schedule::Schedule,
};
use anyhow::{ensure, Context, Result};
use ferrum_models::qwen_image21::{
    ExpectedWeights, QwenImage21Config, QwenImage21RotaryFrequencies, QwenImage21T2iLayout,
    QwenImage21Timestep256, QwenImage21TransformerConfig, StructureLimits,
};
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::ElementType;
use serde_json::json;

pub fn run(args: Arguments) -> Result<()> {
    args.backend.ensure_available()?;
    let config_bytes = artifacts::read_bounded(&args.config, 4 * 1024 * 1024)?;
    let text_metadata_bytes = artifacts::read_bounded(&args.text_metadata, 1024 * 1024)?;
    let text_metadata: TextMetadata = serde_json::from_slice(&text_metadata_bytes)?;
    let config = QwenImage21TransformerConfig::new(
        QwenImage21Config::from_hf_json(&config_bytes, StructureLimits::default())?,
        QwenImage21T2iLayout::new(
            text_metadata.text_tokens,
            args.latent_height,
            args.latent_width,
            text_metadata.valid_text_lengths.clone(),
        )?,
    )?;
    let model_index = artifacts::read_bounded(&args.model_index, 4 * 1024 * 1024)?;
    let scheduler_config = artifacts::read_bounded(&args.scheduler_config, 1024 * 1024)?;
    let schedule = Schedule::parse(&model_index, &scheduler_config)?;
    let inventory = ExpectedWeights::from_config(config.transformer(), StructureLimits::default())?;
    let weight_bytes = inventory
        .tensors()
        .values()
        .try_fold(0_u64, |sum, t| sum.checked_add(t.bytes()))
        .context("weight bytes overflow")?;
    ensure!(weight_bytes<args.device_bytes(),"declared device capacity cannot hold {weight_bytes} DiT weights before dynamic/opaque overhead");
    let budget = HostBudget::new(&config, args.staging_bytes(), args.maximum_host_bytes)?;
    text_metadata.validate(&config, &budget)?;
    let source = SafetensorsArchive::open_with_index(&args.model_dir, &args.index_file)?;
    for w in inventory.tensors().values() {
        let t = source.tensor(w.external_name())?;
        ensure!(
            t.element_type() == Some(ElementType::Bf16)
                && t.shape() == w.dimensions()
                && t.bytes().len() as u64 == w.bytes(),
            "DiT weight metadata mismatch: {}",
            w.external_name()
        );
    }
    let host = budget.account()?;
    let inputs = Inputs::read(
        &config,
        &host,
        &budget,
        &args.latent_file,
        &args.text_file,
        &text_metadata,
    )?;
    let output = artifacts::Artifacts::create(&args.artifact_dir)?;
    let weights=inventory.tensors().values().map(|w|{let t=source.tensor(w.external_name())?;
        Ok(json!({"name":w.external_name(),"shape":w.dimensions(),"bytes":w.bytes(),"sha256":artifacts::sha256(t.bytes())}))}).collect::<Result<Vec<_>>>()?;
    let initial_hash = artifacts::sha256(inputs.sample.bytes().bytes());
    output.bytes("initial_latent.bf16", inputs.sample.bytes().bytes())?;
    output.bytes("frequencies.f32", inputs.frequencies.bytes().bytes())?;
    output.json("schedule.json", &schedule.declaration)?;
    output.json("declaration.json",&json!({"schema":"ferrum.qwen_image21.denoise.v1","mode":"execution_smoke_only",
        "scope":"configured DiT repeated over explicit sigmas and managed BF16 Euler; no decoded image, full-trajectory numerical parity or performance claim",
        "config":config,"config_sha256":artifacts::sha256(&config_bytes),"weights":weights,"initialized_weight_bytes":weight_bytes,
        "model_index_sha256":artifacts::sha256(&model_index),"scheduler_config_sha256":artifacts::sha256(&scheduler_config),
        "schedule":schedule.declaration,"text_metadata":text_metadata,"text_metadata_sha256":artifacts::sha256(&text_metadata_bytes),
        "initial_latent":{"file":"initial_latent.bf16","sha256":initial_hash,"shape":inputs.sample.tensor().dimensions(),"domain":"normalized","layout":"packed spatial","source":"explicit caller file; RNG/seed provenance not established"},
        "frequencies":{"file":"frequencies.f32","sha256":artifacts::sha256(inputs.frequencies.bytes().bytes()),"shape":inputs.frequencies.tensor().dimensions(),"profile":QwenImage21RotaryFrequencies::NUMERICAL_PROFILE_ID},
        "timestep_profile":QwenImage21Timestep256::NUMERICAL_PROFILE_ID,"backend":args.backend,"host_budget":budget,
        "source_authenticity":"content SHA binding, not checkpoint certification","memory_scope":"managed latent/text/time/frequency/velocity/staging; mmap/hash scans and metadata are outside Host/RSS accounting"}))?;
    let started = std::time::Instant::now();
    let result: Result<()> = (|| {
        let result = composition::run(
            &config,
            &source,
            &args,
            &host,
            &budget,
            inputs,
            schedule.solver,
            |record, _sample| {
                let step = record["step"]
                    .as_u64()
                    .context("step receipt lacks index")?;
                output.json(&format!("step-{step:03}.json"), record)
            },
        )?;
        output.json("execution.json", &result.provenance)?;
        output.bytes("latents.bf16", result.sample.bytes().bytes())?;
        let latent_record = json!({"schema":"ferrum.qwen_image21.denoised-latent.v1","file":"latents.bf16",
            "sha256":artifacts::sha256(result.sample.bytes().bytes()),"bytes":result.sample.bytes().len(),"dtype":"BF16",
            "domain":"normalized","layout":"packed","shape":result.sample.tensor().dimensions(),
            "logical_shape":[config.layout().batch(),config.layout().target_tokens(),config.transformer().in_channels()],
            "batch":config.layout().batch(),"latent_height":config.layout().latent_height(),"latent_width":config.layout().latent_width(),
            "channels":config.transformer().in_channels(),"frames":1,"completed_steps":result.steps.len(),"terminal_sigma":0.0,"nonfinite_count":0,
            "vae_handoff":"unpack to NCHW then explicitly denormalize using VAE channel mean/std; these bytes are not yet denormalized"});
        output.json("latent.json", &latent_record)?;
        drop(result.sample);
        let usage = host.usage()?;
        ensure!(
            usage.committed_bytes() == 0 && usage.allocated_bytes() == 0,
            "Host owners remain after final latent artifact"
        );
        output.json("result.json",&json!({"passed":true,"mode":"execution_smoke_only","numerical_parity":"not established",
            "latent":latent_record,"steps":result.steps,"retirement":result.retirement,
            "host_after_artifact":{"committed":usage.committed_bytes(),"allocated":usage.allocated_bytes()},
            "cold_seconds":started.elapsed().as_secs_f64(),"timing_scope":"compile+one static init+all DiT/Euler steps+readbacks+artifacts+close; not throughput"}))?;
        Ok(())
    })();
    if let Err(error) = &result {
        output.json("failure.json",&json!({"passed":false,"error":format!("{error:#}"),
        "scope":"no complete trajectory or safe-retirement claim; prior step receipts remain evidence of completed intervals only"}))?;
    }
    result
}
pub fn main() -> Result<()> {
    match Arguments::parse(std::env::args_os().skip(1))? {
        Some(args) => run(args),
        None => {
            println!("{}\n{}", super::arguments::USAGE, backend::USAGE);
            Ok(())
        }
    }
}
