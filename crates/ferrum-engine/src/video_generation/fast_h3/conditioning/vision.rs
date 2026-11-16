use super::*;
use ferrum_models::{fast_h3::PreparedFastH3ReferenceImage, qwen3_vl_vision::*};
use ferrum_quantization::SafetensorsArchive;

pub struct PreparedVisionConditioning {
    pub checkpoint: Arc<PreparedQwen3VLVisionCheckpoint>,
    pub inputs: Vec<ManagedHostTensor>,
}
pub fn prepare_vision(
    architecture: Qwen3VLVisionArchitecture,
    archive: Arc<SafetensorsArchive>,
    processor: &Qwen3VLVisionImageProcessor,
    reference: &PreparedFastH3ReferenceImage,
    host: &HostMemoryAccount,
    limits: Qwen3VLVisionCheckpointLimits,
) -> anyhow::Result<PreparedVisionConditioning> {
    let image = reference.image();
    let plan = processor.prepare_image(architecture, image.width(), image.height())?;
    let checkpoint = Arc::new(PreparedQwen3VLVisionCheckpoint::prepare(
        plan.config().clone(),
        archive,
        &NumericalProfileId::new(Qwen3VLVisionFamily::CUDA_PROFILE_ID)
            .map_err(anyhow::Error::msg)?,
        limits,
    )?);
    let mut inputs = checkpoint.managed_position_inputs(host)?.to_vec();
    inputs.push(plan.managed_patches(host, image)?);
    Ok(PreparedVisionConditioning { checkpoint, inputs })
}

#[allow(clippy::too_many_arguments)]
pub fn encode_vision<R: DeviceRuntime>(
    prepared: PreparedVisionConditioning,
    compiled: Arc<CompiledTensorComponent<R>>,
    runtime: Arc<R>,
    host: &HostMemoryAccount,
    actor: &mut TensorSerialActor<R>,
    initialization: ComponentInitializationConfig,
    run: RunId,
    request: RequestIdentity,
) -> Result<[TensorExecutionOutput; 4], ConditioningFailure> {
    if compiled
        .family()
        .fingerprint()
        .map_err(|e| ConditioningFailure::preparation("vision family", e))?
        != prepared
            .checkpoint
            .family()
            .fingerprint()
            .map_err(|e| ConditioningFailure::preparation("vision family", e))?
    {
        return Err(ConditioningFailure::preparation(
            "vision family",
            "compiled component differs from original vision graph",
        ));
    }
    let source = prepared.checkpoint.managed_source(host);
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
        prepared.inputs,
    )?;
    let mut ordered = Vec::with_capacity(4);
    for id in [
        Qwen3VLVisionFamily::OUTPUT_ID,
        Qwen3VLVisionFamily::DEEPSTACK_IDS[0],
        Qwen3VLVisionFamily::DEEPSTACK_IDS[1],
        Qwen3VLVisionFamily::DEEPSTACK_IDS[2],
    ] {
        let index = outputs
            .iter()
            .position(|o| o.value_id().as_str() == id)
            .ok_or_else(|| {
                ConditioningFailure::preparation(
                    "vision outputs",
                    "required original output missing",
                )
            })?;
        let output = outputs.swap_remove(index);
        if output.tensor().element_type() != ElementType::Bf16
            || output.tensor().dimensions()
                != [
                    prepared
                        .checkpoint
                        .config()
                        .merged_rows()
                        .map_err(|e| ConditioningFailure::preparation("vision rows", e))?,
                    prepared.checkpoint.config().architecture().output_size(),
                ]
            || !output
                .bytes()
                .bytes()
                .chunks_exact(2)
                .all(|b| u16::from_le_bytes(b.try_into().unwrap()) & 0x7f80 != 0x7f80)
        {
            return Err(ConditioningFailure::preparation(
                "vision outputs",
                "invalid native BF16 features",
            ));
        }
        ordered.push(output);
    }
    if !outputs.is_empty() {
        return Err(ConditioningFailure::preparation(
            "vision outputs",
            "unexpected output",
        ));
    }
    ordered
        .try_into()
        .map_err(|_| ConditioningFailure::preparation("vision outputs", "output count differs"))
}
