use super::*;
use ferrum_models::{
    fast_h3::video_vae::{
        sample_fast_h3_keyframe_posterior, FastH3VideoVaeConfig, PreparedFastH3TiledImageEncoder,
    },
    fast_h3::{
        FastH3ReferenceImagePlan, PreparedFastH3ReferenceImage, FAST_H3_REFERENCE_IMAGE_SHORT_EDGE,
    },
    qwen_image21::image_input::{ImageInputLimits, ManagedRgba8Image},
};

pub fn prepare_reference_image(
    encoded_png: &ManagedHostBytes,
    host: &HostMemoryAccount,
    limits: ImageInputLimits,
) -> anyhow::Result<PreparedFastH3ReferenceImage> {
    let image = ManagedRgba8Image::decode_png(host, encoded_png, limits)?;
    FastH3ReferenceImagePlan::new(
        image.width(),
        image.height(),
        FAST_H3_REFERENCE_IMAGE_SHORT_EDGE,
        limits,
    )?
    .prepare(&image, host)
}

#[allow(clippy::too_many_arguments)]
pub fn encode_keyframe<R: DeviceRuntime>(
    architecture: &FastH3VideoVaeConfig,
    prepared: Arc<PreparedFastH3TiledImageEncoder>,
    compiled: Arc<CompiledTensorComponent<R>>,
    runtime: Arc<R>,
    host: &HostMemoryAccount,
    actor: &mut TensorSerialActor<R>,
    initialization: ComponentInitializationConfig,
    request_id: RequestIdentity,
    pixels: ManagedHostTensor,
) -> Result<ManagedHostTensor, ConditioningFailure> {
    if compiled.family().canonical_config() != prepared.family().canonical_config()
        || compiled
            .family()
            .fingerprint()
            .map_err(|e| ConditioningFailure::preparation("keyframe family", e))?
            != prepared
                .family()
                .fingerprint()
                .map_err(|e| ConditioningFailure::preparation("keyframe family", e))?
    {
        return Err(ConditioningFailure::preparation(
            "keyframe family",
            "compiled component differs from original encoder",
        ));
    }
    let mut request = prepared
        .begin(&pixels, request_id.clone(), host)
        .map_err(|e| ConditioningFailure::preparation("keyframe preparation", e))?;
    let source = prepared.weight_source(host.clone());
    let phase = (prepared.clone(), compiled.clone(), runtime.clone(), pixels);
    let mut session = compiled
        .initialize_retained_with_pool_policy(
            runtime,
            host.clone(),
            ComponentWeightSource::Stored(&source),
            initialization,
            phase,
            ferrum_execution::component::ColdPoolInitializationPolicy::OnExecutionAdmission,
        )
        .map_err(|e| ConditioningFailure::retain("keyframe initialization", e, ()))?;
    loop {
        let run = match RunId::new(format!(
            "{}.keyframe.{}",
            request_id,
            request.completed_tiles()
        )) {
            Ok(run) => run,
            Err(e) => {
                return Err(ConditioningFailure::retain(
                    "keyframe run identity",
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
                    "keyframe tile",
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
                "keyframe stitching",
                e,
                session,
            ))
        }
    };
    let (returned, _) = session
        .close()
        .map_err(|e| ConditioningFailure::retain("keyframe close", e, ()))?;
    drop(returned);
    let id = ProgramValueId::new("value.fast_h3.normalized_keyframe")
        .map_err(|e| ConditioningFailure::preparation("keyframe posterior identity", e))?;
    sample_fast_h3_keyframe_posterior(architecture, moments.tensor(), moments.bytes(), id, host)
        .map_err(|e| ConditioningFailure::preparation("keyframe posterior", e))
}
