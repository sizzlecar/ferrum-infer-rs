use super::*;
use ferrum_models::fast_h3::video_vae::{
    sample_fast_h3_video_posterior, FastH3VideoVaeConfig, PreparedFastH3TiledVideoEncoder,
};
use std::sync::atomic::{AtomicBool, Ordering};

#[allow(clippy::too_many_arguments)]
pub fn encode_video_reference<R: DeviceRuntime>(
    architecture: &FastH3VideoVaeConfig,
    prepared: Arc<PreparedFastH3TiledVideoEncoder>,
    compiled: Arc<CompiledTensorComponent<R>>,
    runtime: Arc<R>,
    host: &HostMemoryAccount,
    actor: &mut TensorSerialActor<R>,
    initialization: ComponentInitializationConfig,
    pixels: ManagedHostTensor,
    cancelled: &AtomicBool,
) -> Result<Option<ManagedHostTensor>, ConditioningFailure> {
    let request_id = initialization.request_id.clone();
    let encoder = prepared.tile_encoder();
    if compiled
        .family()
        .fingerprint()
        .map_err(|e| ConditioningFailure::preparation("video encoder family", e))?
        != encoder
            .family()
            .fingerprint()
            .map_err(|e| ConditioningFailure::preparation("video encoder family", e))?
    {
        return Err(ConditioningFailure::preparation(
            "video encoder family",
            "compiled component differs from original encoder",
        ));
    }
    let mut request = prepared
        .begin(&pixels, request_id.clone(), host)
        .map_err(|e| ConditioningFailure::preparation("video encoder preparation", e))?;
    let phase = (prepared.clone(), compiled.clone(), runtime.clone(), pixels);
    let mut session = compiled
        .initialize_retained_with_pool_policy(
            runtime,
            host.clone(),
            ComponentWeightSource::Stored(encoder.weight_source()),
            initialization,
            phase,
            ferrum_execution::component::ColdPoolInitializationPolicy::OnExecutionAdmission,
        )
        .map_err(|e| ConditioningFailure::retain("video encoder initialization", e, ()))?;
    loop {
        if cancelled.load(Ordering::Acquire) {
            let (returned, _) = session.close().map_err(|e| {
                ConditioningFailure::retain("cancelled video encoder close", e, request)
            })?;
            drop(returned);
            return Ok(None);
        }
        let run = match RunId::new(format!(
            "{}.reference_video.{}",
            request_id,
            request.executed_tiles()
        )) {
            Ok(run) => run,
            Err(e) => {
                return Err(ConditioningFailure::retain(
                    "video encoder identity",
                    e,
                    (session, request),
                ))
            }
        };
        match request.advance_on_actor(session.session_mut(), run, actor) {
            Ok(true) => break,
            Ok(false) => {}
            Err(e) => {
                return Err(ConditioningFailure::retain(
                    "video encoder tile",
                    e,
                    (session, request),
                ))
            }
        }
    }
    let moments = match request.finish() {
        Ok(moments) => moments,
        Err(e) => {
            return Err(ConditioningFailure::retain(
                "video encoder stitching",
                e,
                session,
            ))
        }
    };
    let (returned, _) = session
        .close()
        .map_err(|e| ConditioningFailure::retain("video encoder close", e, ()))?;
    drop(returned);
    if cancelled.load(Ordering::Acquire) {
        return Ok(None);
    }
    let id = ProgramValueId::new("value.fast_h3.normalized_reference_video")
        .map_err(|e| ConditioningFailure::preparation("video posterior identity", e))?;
    sample_fast_h3_video_posterior(architecture, moments.tensor(), moments.bytes(), id, host)
        .map(Some)
        .map_err(|e| ConditioningFailure::preparation("video posterior", e))
}
