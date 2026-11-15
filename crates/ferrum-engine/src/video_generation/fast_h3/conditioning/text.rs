use super::*;
use anyhow::{ensure, Context};
use ferrum_models::{
    fast_h3::{FastH3ImagePresentation, FastH3PresentationLimits, FastH3VisionFeatureView},
    qwen3_vl_text::{
        Qwen3VLTextEncoderFamily, Qwen3VLTextNvfp4Attention, Qwen3VLTextNvfp4Checkpoint,
        Qwen3VLTextNvfp4Limits,
    },
    qwen3_vl_vision::Qwen3VLVisionConfig,
};
use ferrum_quantization::SafetensorsArchive;

pub struct PreparedTextConditioning {
    pub checkpoint: Arc<Qwen3VLTextNvfp4Checkpoint>,
    pub presentation: FastH3ImagePresentation,
}
#[allow(clippy::too_many_arguments)]
pub fn prepare_text(
    metadata: &serde_json::Value,
    archive: Arc<SafetensorsArchive>,
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    vision: &Qwen3VLVisionConfig,
    presentation_limits: FastH3PresentationLimits,
    source_limits: Qwen3VLTextNvfp4Limits,
) -> anyhow::Result<PreparedTextConditioning> {
    for (token, field) in [
        ("<|image_pad|>", "image_token_id"),
        ("<|video_pad|>", "video_token_id"),
        ("<|vision_start|>", "vision_start_token_id"),
        ("<|vision_end|>", "vision_end_token_id"),
    ] {
        ensure!(
            tokenizer.token_to_id(token).map(u64::from)
                == metadata.get(field).and_then(|v| v.as_u64())
                && metadata.get(field).is_some(),
            "tokenizer/checkpoint {field} mismatch"
        );
    }
    let presentation =
        FastH3ImagePresentation::single_image(tokenizer, prompt, vision, presentation_limits)?;
    let layers = metadata
        .get("num_hidden_layers_override")
        .and_then(|v| v.as_u64())
        .context("checkpoint lacks declared exported text layer count")?;
    let checkpoint = Qwen3VLTextNvfp4Checkpoint::from_shared_archive(
        metadata,
        archive,
        presentation.layout()?,
        layers,
        source_limits,
    )?
    .with_deepstack(presentation.deepstack()?)?
    .with_attention(Qwen3VLTextNvfp4Attention::VendorFusedCausal)?;
    Ok(PreparedTextConditioning {
        checkpoint: Arc::new(checkpoint),
        presentation,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn encode_text<R: DeviceRuntime>(
    prepared: &PreparedTextConditioning,
    vision: &[[TensorExecutionOutput; 4]],
    compiled: Arc<CompiledTensorComponent<R>>,
    runtime: Arc<R>,
    host: &HostMemoryAccount,
    actor: &mut TensorSerialActor<R>,
    initialization: ComponentInitializationConfig,
    run: RunId,
    request: RequestIdentity,
) -> Result<TensorExecutionOutput, ConditioningFailure> {
    let declared = serde_json::to_value(prepared.checkpoint.config())
        .map_err(|e| ConditioningFailure::preparation("text family", e))?;
    if compiled.family().canonical_config() != &declared {
        return Err(ConditioningFailure::preparation(
            "text family",
            "compiled component differs from original text layout/profile",
        ));
    }
    let views = vision
        .iter()
        .map(|image| {
            std::array::from_fn(|i| {
                FastH3VisionFeatureView::new(
                    image[i].value_id(),
                    image[i].tensor(),
                    image[i].bytes(),
                )
            })
        })
        .collect::<Vec<_>>();
    let inputs = prepared
        .presentation
        .managed_text_inputs(
            prepared.checkpoint.config().text(),
            prepared.checkpoint.archive(),
            &views,
            host,
        )
        .map_err(|e| ConditioningFailure::preparation("text inputs", e))?;
    let source = prepared
        .checkpoint
        .managed_source(host)
        .map_err(|e| ConditioningFailure::preparation("text source", e))?;
    let mut outputs = forward(
        compiled,
        runtime,
        host,
        actor,
        initialization,
        run,
        request,
        &source,
        prepared.checkpoint.clone(),
        inputs,
    )?;
    if outputs.len() != 1
        || outputs[0].value_id().as_str() != Qwen3VLTextEncoderFamily::OUTPUT_ID
        || outputs[0].tensor().element_type() != ElementType::Bf16
        || !outputs[0]
            .bytes()
            .bytes()
            .chunks_exact(2)
            .all(|b| u16::from_le_bytes(b.try_into().unwrap()) & 0x7f80 != 0x7f80)
    {
        return Err(ConditioningFailure::preparation(
            "text outputs",
            "invalid original text features",
        ));
    }
    Ok(outputs.remove(0))
}
