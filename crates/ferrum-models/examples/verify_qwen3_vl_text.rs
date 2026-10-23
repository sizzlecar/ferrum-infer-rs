//! Real pure-T2I prompt -> full Qwen3-VL text tower -> DiT conditioning artifact.
//! Only execution/finite-output evidence; no Torch parity or image claim.
#[path = "verify_qwen3_vl_text/arguments.rs"]
mod arguments;
#[path = "verify_qwen_image21_projection/artifacts.rs"]
mod artifacts;
#[path = "verify_qwen_image21_projection/arguments.rs"]
mod backend;
#[path = "verify_qwen3_vl_text/execution.rs"]
mod execution;
#[path = "support/tensor_component.rs"]
mod shared;

use anyhow::{ensure, Context, Result};
use ferrum_models::{
    qwen3_vl_text::{
        Qwen3VLTextArchitecture, Qwen3VLTextConfig, Qwen3VLTextEncoderFamily as Family,
    },
    qwen_image21::t2i_prompt::{QwenImage21T2iTokenizer, T2I_TEXT_ROTARY_PROFILE},
};
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::*;
use serde_json::json;

fn run(args: arguments::Arguments) -> Result<()> {
    args.backend.ensure_available()?;
    let config_bytes = artifacts::read_bounded(
        &args.model_dir.join("text_encoder/config.json"),
        4 * 1024 * 1024,
    )?;
    let architecture =
        Qwen3VLTextArchitecture::from_hf_value(&serde_json::from_slice(&config_bytes)?)?;
    let tokenizer_bytes = artifacts::read_bounded(
        &args.model_dir.join("processor/tokenizer.json"),
        32 * 1024 * 1024,
    )?;
    let tokenizer_config = artifacts::read_bounded(
        &args.model_dir.join("processor/tokenizer_config.json"),
        1024 * 1024,
    )?;
    let template = artifacts::read_bounded(
        &args.model_dir.join("processor/chat_template.jinja"),
        64 * 1024,
    )?;
    let tokenizer =
        QwenImage21T2iTokenizer::from_source_bytes(&tokenizer_bytes, &tokenizer_config, &template)?;
    let prompts = tokenizer.prepare(&[&args.prompt], args.maximum_user_tokens)?;
    let config = Qwen3VLTextConfig::new(architecture.clone(), prompts.layout().clone())?;
    let weights = config.expected_weights()?;
    let weight_bytes = weights
        .iter()
        .try_fold(0_u64, |n, w| n.checked_add(w.bytes()))
        .context("weight byte count overflows")?;
    ensure!(weight_bytes<args.device_bytes(),"declared device capacity cannot hold {weight_bytes} text layer weight bytes before dynamic/opaque overhead");
    if let backend::Backend::Cuda(options) = &args.backend {
        let minimum = usize::try_from(architecture.layers())?
            .checked_mul(17)
            .and_then(|n| n.checked_add(3))
            .context("command count overflows")?;
        ensure!(options.maximum_commands_per_submission>=minimum,"CUDA command limit must cover {minimum} commands; set --cuda-maximum-commands-per-submission");
    }
    let budget = execution::HostBudget::new(
        &config,
        &prompts,
        args.staging_bytes(),
        args.maximum_host_bytes,
    )?;
    let host = budget.account()?;
    let source = SafetensorsArchive::open(args.model_dir.join("text_encoder"))?;
    // Check all executed weight metadata before payload hashing or GPU startup.
    for w in &weights {
        let t = source.tensor(w.external_name())?;
        ensure!(
            t.element_type() == Some(ElementType::Bf16)
                && t.shape() == w.dimensions()
                && t.bytes().len() as u64 == w.bytes(),
            "text weight metadata mismatch: {}",
            w.external_name()
        );
    }
    let embedding = source.tensor(Family::EMBEDDING_EXTERNAL_NAME)?;
    let inputs = prompts.managed_inputs(&host, &architecture, &embedding)?;
    let output = artifacts::Artifacts::create(&args.artifact_dir)?;
    let weights=weights.iter().map(|w| {
        let t=source.tensor(w.external_name())?;
        Ok(json!({"name":w.external_name(),"shape":w.dimensions(),"bytes":w.bytes(),"sha256":artifacts::sha256(t.bytes())}))
    }).collect::<Result<Vec<_>>>()?;
    let input_records=inputs.tensors().iter().zip(["embeddings.bf16","cosine.bf16","sine.bf16"]).map(|(tensor,file)| {
        output.bytes(file,tensor.bytes().bytes())?;
        Ok(json!({"value":tensor.value_id(),"shape":tensor.tensor().dimensions(),"file":file,"bytes":tensor.bytes().len(),"sha256":artifacts::sha256(tensor.bytes().bytes())}))
    }).collect::<Result<Vec<_>>>()?;
    output.json("declaration.json",&json!({"schema":"ferrum.qwen3vl.t2i-text.v1","mode":"execution_smoke_only",
        "scope":"actual tokenizer and full configured text tower pre-final-norm output; no vision, Torch parity, first image or performance claim",
        "config":config,"config_source_sha256":artifacts::sha256(&config_bytes),"prompt":args.prompt,"tokenizer_source_sha256":tokenizer.source_sha256(),
        "input_ids":prompts.input_ids(),"input_valid_lengths":prompts.valid_input_lengths(),"drop_prefix_tokens":prompts.drop_prefix_tokens(),
        "position_semantics":"fresh cache-free T2I, all three axes equal full padded column; not per-valid-token cumsum",
        "rotary_profile":T2I_TEXT_ROTARY_PROFILE,"tokenizer_qualification":"local tokenizers0.21; no deployed0.23.3 exact-token oracle",
        "embedding":{"name":embedding.external_name(),"shape":embedding.shape(),"bytes":embedding.bytes().len(),"sha256":artifacts::sha256(embedding.bytes()),"device_upload":"selected token rows only"},
        "weights":weights,"initialized_weight_bytes":weight_bytes,"inputs":input_records,"backend":args.backend,"host_budget":budget,
        "memory_scope":"one Host account for gathered embeddings, rotary, uncropped output, cropped features and static staging; source mmap/hash scans, tokenizer and metadata allocations outside Host/RSS accounting"}))?;
    let start = std::time::Instant::now();
    let result = (|| {
        let result = execution::run(
            &config,
            &prompts,
            inputs,
            &host,
            &budget,
            &source,
            &args,
            weight_bytes,
        )?;
        output.json("execution.json", &result.provenance)?;
        let features = result.features;
        let bytes = features.tensor().bytes().bytes();
        output.bytes("text.bf16", bytes)?;
        let feature_record = json!({"file":"text.bf16","sha256":artifacts::sha256(bytes),"bytes":bytes.len(),"shape":features.tensor().tensor().dimensions(),
            "dtype":"BF16","text_tokens":features.sequence_length(),"valid_text_lengths":features.valid_lengths(),"nonfinite_count":0});
        output.json("text.json", &feature_record)?;
        drop(features);
        let usage = host.usage()?;
        ensure!(
            usage.committed_bytes() == 0 && usage.allocated_bytes() == 0,
            "Host owners remain after text artifact completion"
        );
        output.json("result.json",&json!({"passed":true,"mode":"execution_smoke_only","numerical_parity":"not established",
            "features":feature_record,"retirement":result.retirement,"host_after_artifact":{"committed":usage.committed_bytes(),"allocated":usage.allocated_bytes()},
            "cold_execution_seconds":start.elapsed().as_secs_f64(),"timing_scope":"compile+static init+execution+readback+crop+retirement+artifact; no throughput comparison"}))?;
        Ok(())
    })();
    if let Err(error) = &result {
        output.json("failure.json",&json!({"passed":false,"error":format!("{error:#}"),"scope":"no successful execution or retirement claim"}))?;
    }
    result
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
