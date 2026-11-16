//! Ordered media conditioning with a bounded last-domain compilation.
//! Immutable graph metadata and an explicitly bounded completed Host result may
//! survive requests. Auxiliary sessions always retire before publication.
use crate::video_generation::{
    fast_h3::{assets::FastH3Assets, conditioning::*, cuda::FastH3CudaComposition},
    VideoGenerationRequest, VideoReference,
};
use ferrum_execution::{
    component::{CompiledTensorComponent, ComponentInitializationConfig},
    TensorExecutionOutput, TensorSerialActor,
};
use ferrum_kernels::backend::cuda::vnext_runtime::CudaDeviceRuntime;
use ferrum_models::{
    audio_input::{AudioInputLimits, WaveAudioPlan},
    fast_h3::{
        audio_vae::{
            FastH3AudioComponentLimits, PreparedFastH3AudioEncoderProjection,
            PreparedFastH3AudioEncoderTrunk,
        },
        video_vae::{
            FastH3ImageEncoderLimits, FastH3ImageEncoderProfile, FastH3ImageSpatialPolicy,
            PreparedFastH3TiledImageEncoder,
        },
        FastH3InputLimits, FastH3MediaPresentation, FastH3PresentationLimits,
        FastH3PresentationReference, FastH3Reference, FastH3ReferenceLatents,
        FastH3TextRefinerLimits, FastH3TextTagRun,
    },
    qwen3_vl_text::{
        Qwen3VLTextNvfp4Attention, Qwen3VLTextNvfp4Checkpoint, Qwen3VLTextNvfp4Family,
        Qwen3VLTextNvfp4Limits,
    },
    qwen3_vl_vision::{
        PreparedQwen3VLVisionCheckpoint, Qwen3VLVisionCheckpointLimits, Qwen3VLVisionFamily,
    },
    qwen_image21::image_input::ImageInputLimits,
};
use ferrum_runtime::*;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

type Device = CudaDeviceRuntime;
type Compiled = Arc<CompiledTensorComponent<Device>>;
type Result<T> = std::result::Result<T, ConditioningFailure>;

mod plan;
mod reference;
mod video;
mod vision;
pub(super) use plan::ConditioningPlan;
use plan::ReferenceConditioningPlan;
use reference::ConditionedReference;

#[derive(Clone, Copy, Debug)]
pub struct ConditioningLimits {
    pub reuse: FastH3ConditioningReuse,
    pub image: ImageInputLimits,
    pub audio: AudioInputLimits,
    pub audio_encoder: FastH3AudioComponentLimits,
    pub keyframe: FastH3ImageEncoderLimits,
    pub maximum_image_tiles: usize,
    pub video_encoder: ferrum_models::fast_h3::video_vae::FastH3VideoEncoderLimits,
    pub reference_video: ferrum_models::fast_h3::FastH3ReferenceVideoLimits,
    pub vision: Qwen3VLVisionCheckpointLimits,
    pub text: Qwen3VLTextNvfp4Limits,
    pub presentation: FastH3PresentationLimits,
    pub input: FastH3InputLimits,
    pub refiner: FastH3TextRefinerLimits,
}
impl ConditioningLimits {
    /// Bounds for the qualified single-reference deployment; these do not
    /// allocate memory or replace the service's Host/device admission.
    pub fn rtx5090() -> Self {
        Self {
            reuse: FastH3ConditioningReuse::default(),
            image: ImageInputLimits::default(),
            audio: AudioInputLimits::default(),
            audio_encoder: FastH3AudioComponentLimits {
                maximum_metadata_bytes: 16 << 20,
                maximum_source_bytes: 2 << 30,
                maximum_stored_bytes: 2 << 30,
                maximum_plan_nodes: 2048,
                maximum_activation_bytes: 1 << 30,
            },
            keyframe: FastH3ImageEncoderLimits {
                maximum_metadata_bytes: 64 << 20,
                maximum_stored_bytes: 4 << 30,
                maximum_plan_nodes: 2048,
                maximum_input_bytes: 4 << 30,
                maximum_output_bytes: 4 << 30,
            },
            maximum_image_tiles: 512,
            video_encoder: ferrum_models::fast_h3::video_vae::FastH3VideoEncoderLimits {
                maximum_metadata_bytes: 64 << 20,
                maximum_stored_bytes: 4 << 30,
                maximum_plan_nodes: 4096,
                maximum_activation_bytes: 4 << 30,
            },
            reference_video: ferrum_models::fast_h3::FastH3ReferenceVideoLimits {
                image: ImageInputLimits::default(),
                maximum_source_frames: 8192,
                maximum_frames: 362,
                maximum_output_bytes: 2 << 30,
            },
            vision: Qwen3VLVisionCheckpointLimits {
                maximum_metadata_bytes: 8 << 20,
                maximum_plan_nodes: 4096,
                maximum_source_bytes: 32 << 30,
            },
            text: Qwen3VLTextNvfp4Limits {
                maximum_projections: 4096,
                maximum_source_bytes: 32usize << 30,
            },
            presentation: FastH3PresentationLimits {
                maximum_prompt_bytes: 64 << 10,
                maximum_tokens: 32768,
            },
            input: FastH3InputLimits {
                maximum_rows: 32768,
                maximum_segments: 32,
                maximum_metadata_bytes: 8 << 20,
                maximum_stored_bytes: 32 << 30,
            },
            refiner: FastH3TextRefinerLimits {
                maximum_rows: 32768,
                maximum_requests: 1,
                maximum_layers: 128,
                maximum_metadata_bytes: 8 << 20,
                maximum_stored_bytes: 32 << 30,
            },
        }
    }
}

#[derive(Clone)]
pub(super) struct ConditionedRequest {
    pub references: Vec<ConditionedReference>,
    pub text: TensorExecutionOutput,
    pub tags: Vec<FastH3TextTagRun>,
}
struct KeyframeDomain {
    checkpoint: Arc<PreparedFastH3TiledImageEncoder>,
    compiled: Compiled,
}
struct VisionDomain {
    checkpoint: Arc<PreparedQwen3VLVisionCheckpoint>,
    compiled: Compiled,
}
struct TextDomain {
    checkpoint: Arc<Qwen3VLTextNvfp4Checkpoint>,
    compiled: Compiled,
}
pub(super) struct PreparedConditioner {
    assets: Arc<FastH3Assets>,
    cuda: Arc<FastH3CudaComposition>,
    limits: ConditioningLimits,
    registry: Arc<OperationRuntimeRegistry<Device>>,
    catalog: CapabilityCatalog,
    last_keyframe: Option<KeyframeDomain>,
    last_video: Option<Compiled>,
    video_input: ferrum_models::video_input::VideoInputCodec,
    last_vision: Option<VisionDomain>,
    last_text: Option<TextDomain>,
    last_result: Option<(ConditioningKey, ConditionedRequest)>,
}
impl PreparedConditioner {
    pub(super) fn plan(
        &self,
        request: &VideoGenerationRequest,
        host: &HostMemoryAccount,
        input_floor: usize,
        output_floor: usize,
        weight_floor: usize,
    ) -> anyhow::Result<ConditioningPlan> {
        plan::plan(
            &self.assets,
            &self.video_input,
            self.limits,
            request,
            host,
            input_floor,
            output_floor,
            weight_floor,
        )
    }
    pub(super) fn new(
        assets: Arc<FastH3Assets>,
        cuda: Arc<FastH3CudaComposition>,
        limits: ConditioningLimits,
        video_input: ferrum_models::video_input::VideoInputCodec,
    ) -> Result<Self> {
        limits
            .reuse
            .validate()
            .map_err(|e| prep("conditioning reuse", e))?;
        let registry = cuda
            .registry(None)
            .map_err(|e| prep("conditioning registry", e))?;
        let catalog = cuda
            .catalog(&registry)
            .map_err(|e| prep("conditioning catalog", e))?;
        Ok(Self {
            assets,
            cuda,
            limits,
            registry,
            catalog,
            last_keyframe: None,
            last_video: None,
            video_input,
            last_vision: None,
            last_text: None,
            last_result: None,
        })
    }

    pub(super) fn reuse(&self, key: ConditioningKey) -> Option<ConditionedRequest> {
        self.last_result
            .as_ref()
            .filter(|(old, _)| *old == key)
            .map(|(_, result)| result.clone())
    }

    /// Eviction drops only our owners. Active borrowers remain charged in the
    /// original Host account; admission must be retried rather than assumed.
    pub(super) fn evict(&mut self) -> bool {
        self.last_result.take().is_some()
    }

    pub(super) fn retain(
        &mut self,
        key: ConditioningKey,
        result: &ConditionedRequest,
        admitted: FastH3ConditioningReuse,
    ) {
        let bytes = result
            .references
            .iter()
            .try_fold(result.text.bytes().backing_bytes(), |sum, value| {
                sum.checked_add(value.backing_bytes())
            });
        self.last_result = bytes
            .filter(|&n| self.limits.reuse.retains(n) && admitted.retains(n))
            .map(|_| (key, result.clone()));
        tracing::debug!(
            retained_bytes = bytes,
            retained = self.last_result.is_some(),
            "completed FastH3 conditioning reuse"
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute(
        &mut self,
        request: &VideoGenerationRequest,
        plan: ConditioningPlan,
        host: &HostMemoryAccount,
        actor: &mut TensorSerialActor<Device>,
        cancellation: &AtomicBool,
        mut initialization: impl FnMut(
            &str,
        )
            -> std::result::Result<ComponentInitializationConfig, VNextError>,
    ) -> Result<Option<ConditionedRequest>> {
        let cancelled = || cancellation.load(Ordering::Acquire);
        if !host.same_parent(&self.cuda.host) {
            return Err(prep("conditioning Host", "foreign parent account"));
        }
        self.assets
            .validate_request(request)
            .map_err(|e| prep("conditioning request", e))?;
        if cancelled() {
            return Ok(None);
        }
        if request.references.len() != plan.references.len() {
            return Err(prep(
                "conditioning plan",
                "reference count differs from admitted plan",
            ));
        }
        let mut references = Vec::with_capacity(plan.references.len());
        let mut all_visual_outputs = Vec::new();
        let mut original_request = None;
        for (reference, reference_plan) in request.references.iter().zip(&plan.references) {
            let (encoded, image_plan) = match (reference, reference_plan) {
                (VideoReference::Image(encoded), ReferenceConditioningPlan::Image(image)) => {
                    (encoded, image)
                }
                (VideoReference::Video(_), ReferenceConditioningPlan::Video(video)) => {
                    let Some((reference, visual)) = self.execute_reference_video(
                        video,
                        host,
                        actor,
                        cancellation,
                        &mut initialization,
                        &mut original_request,
                    )?
                    else {
                        return Ok(None);
                    };
                    references.push(reference);
                    all_visual_outputs.push(visual);
                    continue;
                }
                (VideoReference::Audio(_), ReferenceConditioningPlan::Audio(audio)) => {
                    if cancelled() {
                        return Ok(None);
                    }
                    let waveform = audio
                        .wave
                        .prepare_stereo(audio.maximum_samples, host)
                        .map_err(|e| prep("reference waveform", e))?;
                    let compilation = ConditioningCompilation {
                        registry: &self.registry,
                        catalog: &self.catalog,
                        policy: &self.cuda.policy,
                        materializers: &self.cuda.materializers,
                    };
                    let posterior = encode_audio_reference(
                        &self.assets.audio_vae,
                        self.assets.audio_archive.clone(),
                        waveform,
                        audio.wave.sampling_rate(),
                        &compilation,
                        self.cuda.runtime.clone(),
                        host,
                        actor,
                        self.limits.audio_encoder,
                        |name| {
                            let init = initialization(name)?;
                            if original_request
                                .as_ref()
                                .is_some_and(|id| id != &init.request_id)
                            {
                                return Err(VNextError::InvalidExecutionPlan {
                                    reason: "conditioning changed request identity".into(),
                                });
                            }
                            original_request = Some(init.request_id.clone());
                            Ok(init)
                        },
                    )?;
                    references.push(ConditionedReference::Audio(posterior));
                    if cancelled() {
                        return Ok(None);
                    }
                    continue;
                }
                _ => {
                    return Err(prep(
                        "conditioning plan",
                        "reference kind differs from admitted plan",
                    ))
                }
            };
            let decoded = ferrum_models::qwen_image21::image_input::ManagedRgba8Image::decode_png(
                host,
                encoded,
                self.limits.image,
            )
            .map_err(|e| prep("reference decode", e))?;
            let reference = image_plan
                .reference
                .prepare(&decoded, host)
                .map_err(|e| prep("reference image", e))?;
            drop(decoded);
            let image = reference.image();
            let shape = [1, 3, u64::from(image.height()), u64::from(image.width())];
            if !self
                .last_keyframe
                .as_ref()
                .is_some_and(|old| old.checkpoint.input_spec().dimensions() == shape)
            {
                let checkpoint = Arc::new(
                    PreparedFastH3TiledImageEncoder::prepare(
                        &self.assets.video_vae,
                        self.assets.video_encoder.clone(),
                        1,
                        shape[2],
                        shape[3],
                        FastH3ImageSpatialPolicy::FastVideoTiled {
                            maximum_tiles: self.limits.maximum_image_tiles,
                        },
                        self.limits.keyframe,
                        FastH3ImageEncoderProfile::Cuda,
                        host,
                    )
                    .map_err(|e| prep("keyframe source", e))?,
                );
                let compiled =
                    self.compile(checkpoint.family().clone(), checkpoint.compile_options())?;
                self.last_keyframe = Some(KeyframeDomain {
                    checkpoint,
                    compiled,
                });
            }
            let pixels = reference
                .managed_vae_pixels(
                    host,
                    usize::try_from(self.limits.keyframe.maximum_input_bytes)
                        .map_err(|e| prep("keyframe input capacity", e))?,
                )
                .map_err(|e| prep("keyframe pixels", e))?;
            if cancelled() {
                return Ok(None);
            }
            let init =
                initialization("keyframe").map_err(|e| prep("keyframe initialization", e))?;
            let request_id = init.request_id.clone();
            if original_request
                .as_ref()
                .is_some_and(|id| id != &request_id)
            {
                return Err(prep(
                    "reference identity",
                    "conditioning changed request identity",
                ));
            }
            original_request = Some(request_id.clone());
            let keyframe = self
                .last_keyframe
                .as_ref()
                .expect("prepared keyframe domain");
            let posterior = encode_keyframe(
                &self.assets.video_vae,
                keyframe.checkpoint.clone(),
                keyframe.compiled.clone(),
                self.cuda.runtime.clone(),
                host,
                actor,
                init,
                request_id.clone(),
                pixels,
            )?;
            references.push(ConditionedReference::Image(posterior));
            if cancelled() {
                return Ok(None);
            }
            let vision_plan = &image_plan.vision;
            let patches = vision_plan
                .managed_patches(host, image)
                .map_err(|e| prep("vision patches", e))?;
            drop(reference);
            if cancelled() {
                return Ok(None);
            }
            let init = checked_init(&mut initialization, "vision", &request_id)?;
            let visual_outputs =
                self.vision_forward(vision_plan.config(), patches, host, actor, init)?;
            all_visual_outputs.push(visual_outputs);
            if cancelled() {
                return Ok(None);
            }
        }
        let request_id =
            original_request.ok_or_else(|| prep("conditioning", "no visual reference"))?;
        let presentation = plan.presentation;
        let layout = presentation.layout().map_err(|e| prep("text layout", e))?;
        let deepstack = presentation
            .deepstack()
            .map_err(|e| prep("text DeepStack routing", e))?;
        if !self.last_text.as_ref().is_some_and(|old| {
            let c = old.checkpoint.config();
            c.text().layout() == &layout
                && c.text().deepstack() == Some(&deepstack)
                && c.attention() == Qwen3VLTextNvfp4Attention::VendorFusedCausal
                && c.executed_layers() == self.assets.exported_text_layers
        }) {
            let checkpoint = Arc::new(
                Qwen3VLTextNvfp4Checkpoint::from_shared_archive(
                    &self.assets.encoder_metadata,
                    self.assets.encoder.clone(),
                    layout,
                    self.assets.exported_text_layers,
                    self.limits.text,
                )
                .map_err(|e| prep("text source", e))?
                .with_deepstack(deepstack)
                .map_err(|e| prep("text DeepStack configuration", e))?
                .with_attention(Qwen3VLTextNvfp4Attention::VendorFusedCausal)
                .map_err(|e| prep("text attention profile", e))?,
            );
            let config = checkpoint.config();
            let profile = NumericalProfileId::new(config.numerical_profile_id())
                .map_err(|e| prep("text profile", e))?;
            let family = TypedFamilyRegistration::new(
                Qwen3VLTextNvfp4Family::new().map_err(|e| prep("text family", e))?,
            )
            .prepare_with_profile(
                &serde_json::to_value(config).map_err(|e| prep("text configuration", e))?,
                &profile,
            )
            .map_err(|e| prep("text family", e))?;
            let options = config
                .compile_options()
                .map_err(|e| prep("text plan options", e))?;
            let compiled = self.compile(family, &options)?;
            self.last_text = Some(TextDomain {
                checkpoint,
                compiled,
            });
        }
        let tags = presentation.text_tags().to_vec();
        if cancelled() {
            return Ok(None);
        }
        let cached_text = self.last_text.as_ref().expect("prepared text domain");
        let init = checked_init(&mut initialization, "text", &request_id)?;
        let run = forward_run(&init)?;
        let text = encode_text(
            &PreparedTextConditioning {
                checkpoint: cached_text.checkpoint.clone(),
                presentation,
            },
            &all_visual_outputs,
            cached_text.compiled.clone(),
            self.cuda.runtime.clone(),
            host,
            actor,
            init,
            run,
            request_id.clone(),
        )?;
        drop(all_visual_outputs);
        if cancelled() {
            return Ok(None);
        }
        let compilation = ConditioningCompilation {
            registry: &self.registry,
            catalog: &self.catalog,
            policy: &self.cuda.policy,
            materializers: &self.cuda.materializers,
        };
        let refined = refine_text(
            self.assets.transformer.clone(),
            &text,
            &compilation,
            self.cuda.runtime.clone(),
            host,
            actor,
            self.limits.input,
            self.limits.refiner,
            |name| {
                let init = initialization(name)?;
                if init.request_id != request_id {
                    return Err(VNextError::InvalidExecutionPlan {
                        reason: "conditioning initialization changed request identity".into(),
                    });
                }
                Ok(init)
            },
        )?;
        if cancelled() {
            return Ok(None);
        }
        Ok(Some(ConditionedRequest {
            references,
            text: refined,
            tags,
        }))
    }

    fn compile(
        &self,
        family: PreparedModelFamily,
        options: &ProgramPlanCompileOptions,
    ) -> Result<Compiled> {
        Ok(Arc::new(
            CompiledTensorComponent::compile(
                family,
                &self.registry,
                &self.catalog,
                &self.cuda.policy,
                &self.cuda.materializers,
                options,
            )
            .map_err(|e| prep("conditioning component compile", e))?,
        ))
    }
}
fn prep(stage: &'static str, error: impl std::fmt::Display) -> ConditioningFailure {
    ConditioningFailure::preparation(stage, error)
}
fn checked_init(
    initialization: &mut impl FnMut(
        &str,
    )
        -> std::result::Result<ComponentInitializationConfig, VNextError>,
    stage: &'static str,
    request: &RequestIdentity,
) -> Result<ComponentInitializationConfig> {
    let init = initialization(stage).map_err(|e| prep("conditioning initialization", e))?;
    if &init.request_id != request {
        return Err(prep(
            "conditioning initialization",
            "request identity changed between stages",
        ));
    }
    Ok(init)
}
fn forward_run(init: &ComponentInitializationConfig) -> Result<RunId> {
    RunId::new(format!("{}.forward", init.run_id.as_str()))
        .map_err(|e| prep("conditioning run identity", e))
}
