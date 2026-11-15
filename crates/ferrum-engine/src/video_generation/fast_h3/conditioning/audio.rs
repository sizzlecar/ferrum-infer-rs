use super::*;
use ferrum_models::fast_h3::audio_vae::{
    FastH3AudioComponentLimits, FastH3AudioVaeConfig, PreparedFastH3AudioEncoderProjection,
    PreparedFastH3AudioEncoderTrunk,
};
use ferrum_quantization::SafetensorsArchive;

/// Original waveform -> normalized posterior mode on the same runtime, Host
/// account and serial actor as visual conditioning. Each source channel is a
/// separate mono batch item. CNN resources retire before causal projection.
#[allow(clippy::too_many_arguments)]
pub fn encode_audio_reference<R: DeviceRuntime>(
    architecture: &FastH3AudioVaeConfig,
    archive: Arc<SafetensorsArchive>,
    waveform: ManagedHostTensor,
    sampling_rate: u32,
    compilation: &ConditioningCompilation<'_, R>,
    runtime: Arc<R>,
    host: &HostMemoryAccount,
    actor: &mut TensorSerialActor<R>,
    limits: FastH3AudioComponentLimits,
    mut initialization: impl FnMut(&str) -> Result<ComponentInitializationConfig, VNextError>,
) -> Result<ManagedHostTensor, ConditioningFailure> {
    let prep = |e| ConditioningFailure::preparation("audio reference encoder", e);
    let dimensions = waveform.tensor().dimensions();
    if dimensions.len() != 3
        || dimensions[0] == 0
        || dimensions[1] != 1
        || dimensions[2] == 0
        || waveform.tensor().element_type() != ElementType::F32
        || waveform.tensor().layout() != &ResolvedTensorLayout::Contiguous
        || !host.owns(waveform.bytes())
        || sampling_rate != architecture.sampling_rate()
    {
        return Err(ConditioningFailure::preparation(
            "audio reference input",
            "requires owned independent mono NCL channels at the original VAE sampling rate",
        ));
    }
    let (batch, samples) = (dimensions[0], dimensions[2]);
    let trunk = Arc::new(
        PreparedFastH3AudioEncoderTrunk::prepare(
            architecture,
            archive.clone(),
            batch,
            samples,
            host,
            limits,
        )
        .map_err(prep)?,
    );
    let latent_samples = trunk.output_spec().dimensions()[3];
    let input = trunk.managed_input(&waveform, host).map_err(prep)?;
    drop(waveform);
    let compiled = compilation.compile_prepared(trunk.family().clone(), trunk.compile_options())?;
    let init = initialization("audio.encoder.trunk")
        .map_err(|e| ConditioningFailure::preparation("audio encoder identity", e))?;
    let run = RunId::new(format!("{}.forward", init.run_id))
        .map_err(|e| ConditioningFailure::preparation("audio encoder identity", e))?;
    let request = init.request_id.clone();
    let output = forward(
        compiled,
        runtime.clone(),
        host,
        actor,
        init,
        run,
        request,
        trunk.as_ref(),
        trunk.clone(),
        vec![input],
    )?;
    drop(trunk);
    let [trunk_output]: [TensorExecutionOutput; 1] = output.try_into().map_err(|_| {
        ConditioningFailure::preparation(
            "audio encoder output",
            "expected one completed CNN output",
        )
    })?;
    let projection = Arc::new(
        PreparedFastH3AudioEncoderProjection::prepare(
            architecture,
            archive,
            batch,
            latent_samples,
            host,
            limits,
        )
        .map_err(prep)?,
    );
    let input = projection
        .managed_input(&trunk_output, host)
        .map_err(prep)?;
    drop(trunk_output);
    let compiled =
        compilation.compile_prepared(projection.family().clone(), projection.compile_options())?;
    let init = initialization("audio.encoder.projection")
        .map_err(|e| ConditioningFailure::preparation("audio encoder identity", e))?;
    let run = RunId::new(format!("{}.forward", init.run_id))
        .map_err(|e| ConditioningFailure::preparation("audio encoder identity", e))?;
    let request = init.request_id.clone();
    let output = forward(
        compiled,
        runtime,
        host,
        actor,
        init,
        run,
        request,
        projection.as_ref(),
        projection.clone(),
        vec![input],
    )?;
    let [output]: [TensorExecutionOutput; 1] = output.try_into().map_err(|_| {
        ConditioningFailure::preparation(
            "audio encoder output",
            "expected one completed posterior mean",
        )
    })?;
    projection.normalized_mode(&output, host).map_err(prep)
}
