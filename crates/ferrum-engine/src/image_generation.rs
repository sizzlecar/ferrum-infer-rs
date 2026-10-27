//! Qwen-Image 2.1 request batches on injected runtime owners.
//!
//! Cold is the default. Optional DenoiserAndDecoder residency retains two
//! exact-plan sessions and immutable weight snapshots across compatible plans,
//! releasing idle dynamic backing between stages. AllModelComponents also
//! retains TE; both keep noise cold. AllStages explicitly retains weightless
//! sessions too, while each invocation still consumes fresh inputs. A bounded optional cache
//! reuses compiled plans. Compatible
//! requests share real batched component forwards. These policies are not
//! throughput qualification or proof that every configured shape fits.
//!
//! Metadata preparation is separate from backend construction. The pipeline
//! never constructs a runtime or Host account, launches a helper process, or
//! exchanges tensor payloads through files. All component plans compile before
//! the first upload. Each edit request has its own reference image; compatible
//! edits share batched text/denoising while B1 vision/encoding sessions process
//! distinct references in caller order. Edits may reuse the timestep-zero prefix
//! through admitted fixed Request state. Its D2D handoff accounts for both cache
//! copies; the request retires before decoding. Failures retain recovery owners
//! and stop this pipeline instance. CFG and video are not implemented.
//! Host claims cover managed tensor payloads and static transfer staging, not
//! tokenizer/metadata containers, mapped checkpoints, or total process RSS.

mod assets;
mod attention_policy;
mod composition;
pub use composition::ImagePipelineLimits;
#[cfg(feature = "cuda")]
pub mod cuda;
mod decoder;
mod editing;
mod host_cache;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;
mod metal_attention_policy;
pub use metal_attention_policy::ImageMetalAttentionPolicy;
mod prefix_denoising;
pub use editing::{ImageEditPrefixPolicy, QwenImage21EditProfiles};
mod plan_cache;
mod precision;
mod prepare;
mod residency;
pub mod service;
mod stage;
mod stream_requirements;
pub use assets::{ImageAssetReceipt, QwenImage21Assets, QwenImage21Metadata, QwenImage21Sources};
pub use attention_policy::ImageCudaAttentionPolicy;
pub use host_cache::{ImageFp8HostCacheBudgets, ImageFp8HostCachePolicy};
pub use plan_cache::{ImagePlanCachePolicy, ImagePlanCacheReceipt};
/// Shared component precision contract; keep the original CUDA spelling compatible.
pub use precision::ImageCudaPrecisionPolicy as ImagePrecisionPolicy;
pub use precision::{
    ImageComponentPrecision, ImageCudaPrecisionPolicy, ImageDenoiserPrecision, ImageTextPrecision,
};
pub use residency::{
    ImageDynamicResidencyPolicy, ImageResidencyPolicy, ImageResidentStageSet,
    ImageStageRetentionBudgets,
};
pub use stage::{
    ActiveImageStage, ImageGenerationError, ImageGenerationFailure, ImageStage,
    ImageStageExecutionReceipt, ImageStagePlanReceipt, ImageStagePlanRole, ImageStageReceipt,
    ImageStageRetirement,
};
pub use stream_requirements::{
    ImageStreamLimit, ImageStreamLimitSource, ImageStreamPlan, ImageStreamRequirements,
};

use anyhow::{ensure, Context};
use ferrum_diffusion::{Bf16EulerProfile, ManagedBf16Euler};
use ferrum_execution::component::{
    CompiledTensorComponent, ComponentInitializationConfig, ComponentInitializationError,
    ComponentInitializationReceipt, ComponentWeightSource, TensorComponentPolicy,
    TensorComponentSession,
};
use ferrum_execution::{
    TensorExecutionError, TensorExecutionOutput, TensorExecutorConfig, TensorSerialActor,
};
use ferrum_models::normal_noise::{NormalNoiseConfig, NormalNoiseFamily};
use ferrum_models::qwen3_vl_text::{
    Qwen3VLTextArchitecture, Qwen3VLTextConfig, Qwen3VLTextEncoderFamily,
};
use ferrum_models::qwen_image21::pipeline_adapters::{
    QwenImage21LatentDomain, QwenImage21LatentLayout, QwenImage21LatentShape,
    QwenImage21LatentStatistics, QwenImage21ManagedLatents, QwenImage21RgbaOutput,
};
use ferrum_models::qwen_image21::sampling_schedule::QwenImage21SamplingSchedule;
use ferrum_models::qwen_image21::t2i_prompt::{QwenImage21T2iPromptBatch, QwenImage21T2iTokenizer};
use ferrum_models::qwen_image21::{
    QwenImage21Config, QwenImage21T2iLayout, QwenImage21TransformerConfig,
    QwenImage21TransformerFamily, QwenImage21VaeDecoderConfig, QwenImage21VaeDecoderFamily,
    StructureLimits,
};
use ferrum_quantization::{SafetensorsArchive, TensorArchive};
use ferrum_runtime::*;
use prepare::PreparedRequest;
use serde::{Deserialize, Serialize};
use stage::PreparedStage;
use std::{collections::BTreeMap, sync::Arc};

/// All arithmetic identities are explicit selections, not backend-name guesses.
#[derive(Debug, Clone)]
pub struct QwenImage21Profiles {
    pub text_encoder: NumericalProfileId,
    pub noise: NumericalProfileId,
    pub denoiser: NumericalProfileId,
    pub decoder: NumericalProfileId,
    pub euler: Bf16EulerProfile,
}

/// Controls only the decoder's compiled batch geometry. TE and DiT retain
/// the admitted request batch. Neither choice implies cross-batch residency
/// or bitwise agreement between backend algorithms selected for different B.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageDecoderBatchPolicy {
    /// Initialize one B1 decoder, then execute it once per request.
    #[default]
    SingleImage,
    /// Diagnostic path: initialize one decoder at the complete request B.
    FullBatch,
}
impl ImageDecoderBatchPolicy {
    fn batch_size(self, request_batch: usize) -> usize {
        match self {
            Self::SingleImage => 1,
            Self::FullBatch => request_batch,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ImagePipelineConfig {
    pub device_capacity_bytes: u64,
    pub maximum_user_tokens: usize,
    pub maximum_width: u32,
    pub maximum_height: u32,
    /// Admission ceiling for one real request batch. Product configuration
    /// defaults to one; larger cold batches require explicit opt-in.
    pub maximum_batch_size: usize,
    pub decoder_batch_policy: ImageDecoderBatchPolicy,
    pub edit_prefix_policy: ImageEditPrefixPolicy,
    pub plan_cache: ImagePlanCachePolicy,
    pub residency: ImageResidencyPolicy,
    pub dynamic_residency: ImageDynamicResidencyPolicy,
    pub static_initialization: StaticInitializationPolicy,
    pub executor: TensorExecutorConfig,
}
impl ImagePipelineConfig {
    pub const DEFAULT_MAXIMUM_BATCH_SIZE: usize = 1;

    fn validate(&self) -> anyhow::Result<()> {
        self.plan_cache.validate()?;
        self.dynamic_residency.validate(self.residency)?;
        ensure!(
            self.device_capacity_bytes > 0 && self.maximum_width > 0 && self.maximum_height > 0,
            "image capacity and dimension limits must be positive"
        );
        ensure!(
            (1..=8192).contains(&self.maximum_user_tokens),
            "user-token limit must be 1..=8192"
        );
        ensure!(
            (1..=STRUCTURED_ATTENTION_BF16_MAX_BATCH).contains(&self.maximum_batch_size),
            "image maximum_batch_size exceeds the supported attention layout capacity"
        );
        Ok(())
    }
}

/// One prompt and one returned image. Extra serialized controls (negative
/// prompt, CFG, step override, image input, video, etc.) are rejected, not ignored.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImageGenerationRequest {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub seed: u64,
    pub n: u32,
}

/// One encoded PNG reference; its admitted backing belongs to this pipeline's
/// Host account throughout queueing, preprocessing and execution.
#[derive(Debug, Clone)]
pub struct ImageEditRequest {
    pub generation: ImageGenerationRequest,
    pub image: ManagedHostBytes,
}
impl ImageGenerationRequest {
    fn latent_dimensions(
        &self,
        config: &ImagePipelineConfig,
        spatial_scale: u32,
    ) -> anyhow::Result<(u32, u32)> {
        ensure!(self.n == 1, "the current image pipeline supports n=1 only");
        ensure!(
            self.prompt.len() <= 1024 * 1024,
            "prompt exceeds the 1 MiB tokenizer bound"
        );
        ensure!(
            spatial_scale > 0
                && self.width > 0
                && self.height > 0
                && self.width <= config.maximum_width
                && self.height <= config.maximum_height,
            "image dimensions exceed the declared positive bounds"
        );
        ensure!(
            self.width % spatial_scale == 0 && self.height % spatial_scale == 0,
            "image dimensions must divide exactly by the decoder spatial scale"
        );
        let (height, width) = (self.height / spatial_scale, self.width / spatial_scale);
        ensure!(
            height % 2 == 0 && width % 2 == 0 && height <= 2048 && width <= 2048,
            "rotary layout requires even latent axes <=2048; dimensions are never rounded"
        );
        Ok((height, width))
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageDenoiseStepReceipt {
    pub index: usize,
    pub sigma: f32,
    pub next_sigma: f32,
    pub scheduler_timestep: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageGenerationReceipt {
    pub job: u64,
    pub batch_size: usize,
    pub batch_index: usize,
    pub decoder_batch_policy: ImageDecoderBatchPolicy,
    pub residency: ImageResidencyPolicy,
    pub width: u32,
    pub height: u32,
    pub seed: u64,
    pub stages: Vec<ImageStageReceipt>,
    pub steps: Vec<ImageDenoiseStepReceipt>,
    pub runtime_fingerprint: String,
    /// Runtime-lifetime native samples, independent of this request's plan
    /// budgets. Shared by CLI receipts and the serving worker's diagnostic log.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_memory: Option<DeviceNativeMemorySnapshot>,
    pub assets: ImageAssetReceipt,
    pub euler_profile: &'static str,
}

/// Immutable BF16 `[1,4,H,W]` with its actual Output backing/Host claim. The
/// caller may close/drop the pipeline while keeping this artifact alive.
pub struct ImageGenerationOutput {
    pub rgba: TensorExecutionOutput,
    pub receipt: ImageGenerationReceipt,
}

pub struct QwenImage21Pipeline<R: DeviceRuntime> {
    runtime: Arc<R>,
    host: HostMemoryAccount,
    registry: OperationRuntimeRegistry<R>,
    catalog: CapabilityCatalog,
    materializers: WeightMaterializerRegistry,
    assets: QwenImage21Assets,
    profiles: QwenImage21Profiles,
    editing_profiles: Option<QwenImage21EditProfiles>,
    config: ImagePipelineConfig,
    policy: TensorComponentPolicy,
    next_job: u64,
    stopped: bool,
    plan_cache: plan_cache::StagePlanCache<R>,
    resident_stages: residency::ResidentStages<R>,
    // Composition-supplied planning allowance. Actual native owners still
    // reserve through the existing device account; this never claims bytes.
    device_opaque_allowance_bytes: u64,
    // One serial lane spans every component and prefix request. Keep this
    // field after resident owners so implicit drop also releases them first.
    serial_actor: Option<TensorSerialActor<R>>,
}

impl<R: DeviceRuntime> QwenImage21Pipeline<R> {
    /// The composition root supplies one combined provider registry. In
    /// particular, CUDA noise/DiT/decoder providers share its native module;
    /// creating separate backend runtimes for stages is not supported here.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        runtime: Arc<R>,
        host: HostMemoryAccount,
        registry: OperationRuntimeRegistry<R>,
        catalog: CapabilityCatalog,
        materializers: WeightMaterializerRegistry,
        assets: QwenImage21Assets,
        profiles: QwenImage21Profiles,
        config: ImagePipelineConfig,
    ) -> anyhow::Result<Self> {
        config.validate()?;
        ensure!(
            config.device_capacity_bytes <= runtime.descriptor().total_memory_bytes,
            "component capacity exceeds physical device descriptor"
        );
        ensure!(
            catalog.device() == runtime.descriptor(),
            "catalog and actual runtime device descriptors differ"
        );
        let policy = TensorComponentPolicy::new(
            config.device_capacity_bytes,
            runtime
                .descriptor()
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect(),
        )?;
        let mut serial_actor = TensorSerialActor::new(runtime.clone())?;
        serial_actor.set_admission_control(ferrum_execution::TensorAdmissionControl::new());
        Ok(Self {
            runtime,
            host,
            registry,
            catalog,
            materializers,
            assets,
            profiles,
            editing_profiles: None,
            config,
            policy,
            next_job: 0,
            stopped: false,
            plan_cache: plan_cache::StagePlanCache::new(),
            resident_stages: residency::ResidentStages::new(),
            device_opaque_allowance_bytes: 0,
            serial_actor: Some(serial_actor),
        })
    }

    /// Include the composition's one aggregate native allowance in resident
    /// batch planning. Reference has no native overhead and keeps zero. This
    /// does not reduce the runtime ceiling or duplicate a capacity reservation.
    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos"), test))]
    pub(super) fn with_device_opaque_allowance(mut self, bytes: u64) -> anyhow::Result<Self> {
        ensure!(
            bytes < self.config.device_capacity_bytes,
            "device opaque allowance leaves no component storage capacity"
        );
        self.device_opaque_allowance_bytes = bytes;
        Ok(self)
    }

    pub fn host_account(&self) -> &HostMemoryAccount {
        &self.host
    }

    /// Both synchronous CLI calls and the service worker use this admission
    /// contract. Replace a cancelled control before a new independent batch.
    pub fn set_admission_control(&mut self, control: ferrum_execution::TensorAdmissionControl) {
        if let Some(actor) = self.serial_actor.as_mut() {
            actor.set_admission_control(control);
        }
    }

    pub fn admission_control(&self) -> Option<&ferrum_execution::TensorAdmissionControl> {
        self.serial_actor
            .as_ref()
            .and_then(TensorSerialActor::admission_control)
    }
    pub fn is_stopped(&self) -> bool {
        self.stopped
            || self
                .serial_actor
                .as_ref()
                .is_none_or(TensorSerialActor::is_failed)
    }

    pub fn generate(
        &mut self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationOutput, ImageGenerationError<R>> {
        let mut outputs = self.generate_batch(vec![request])?;
        Ok(outputs.remove(0))
    }

    /// One real shared TE/DiT forward per stage (and timestep), with each
    /// request's independent fresh Philox stream. All requests must have the
    /// same geometry and this pipeline's fixed schedule/profile. Components
    /// follow the explicit residency policy; exact-plan misses close the old
    /// session before initialization and do not reuse incompatible masks.
    /// VAE follows the explicit decoder policy, reusing one initialized session
    /// within this batch. Output views retain their actual execution backing:
    /// one image under SingleImage, the whole batch under FullBatch.
    /// Execution uses canonical token-length order to reuse exact mask plans;
    /// outputs and receipt batch indices retain the caller's original order.
    /// Per-execution stage timings follow that internal execution order.
    pub fn generate_batch(
        &mut self,
        requests: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageGenerationError<R>> {
        let empty_error = |failure| ImageGenerationError {
            failure,
            completed_stages: Vec::new(),
            completed_steps: Vec::new(),
        };
        if self.is_stopped() {
            return Err(empty_error(ImageGenerationFailure::Unavailable));
        }
        let prepared = self
            .prepare_batch(&requests)
            .map_err(|e| empty_error(ImageGenerationFailure::Preflight(e)))?;
        self.execute_prepared(prepared, &requests)
    }

    fn execute_prepared(
        &mut self,
        prepared: PreparedRequest<R>,
        requests: &[ImageGenerationRequest],
    ) -> Result<Vec<ImageGenerationOutput>, ImageGenerationError<R>> {
        let empty_error = |failure| ImageGenerationError {
            failure,
            completed_stages: Vec::new(),
            completed_steps: Vec::new(),
        };
        let job = self.next_job;
        self.next_job = self.next_job.checked_add(1).ok_or_else(|| {
            empty_error(ImageGenerationFailure::Preflight(anyhow::anyhow!(
                "image job identity exhausted"
            )))
        })?;
        let mut stages = Vec::with_capacity(4);
        let mut steps = Vec::with_capacity(self.assets.schedule.evaluation_count());
        let executed = self.run(prepared, &requests, job, &mut stages, &mut steps);
        let native_memory = self.runtime.native_memory_snapshot();
        if let Some(memory) = native_memory {
            tracing::info!(
                target: "ferrum_engine::image_generation::native_memory",
                job,
                source = memory.source,
                scope = ?memory.scope,
                allocated_bytes = memory.allocated_bytes,
                observed_peak_allocated_bytes = memory.observed_peak_allocated_bytes,
                samples = memory.samples,
                execution_succeeded = executed.is_ok(),
                "image native resource allocation observation"
            );
        }
        let result = executed.and_then(|outputs| {
            if outputs.len() != requests.len() {
                return Err(ImageGenerationFailure::Processing {
                    source: anyhow::anyhow!("decoder returned an unexpected image count"),
                    active: None,
                });
            }
            requests
                .iter()
                .zip(outputs)
                .enumerate()
                .map(|(batch_index, (request, rgba))| {
                    Ok(ImageGenerationOutput {
                        rgba,
                        receipt: ImageGenerationReceipt {
                            job,
                            batch_size: requests.len(),
                            batch_index,
                            decoder_batch_policy: self.config.decoder_batch_policy,
                            residency: self.config.residency,
                            width: request.width,
                            height: request.height,
                            seed: request.seed,
                            stages: stages.clone(),
                            steps: steps.clone(),
                            runtime_fingerprint: self
                                .runtime
                                .descriptor()
                                .runtime_implementation_fingerprint
                                .clone(),
                            native_memory,
                            assets: self.assets.receipt().clone(),
                            euler_profile: self.profiles.euler.id(),
                        },
                    })
                })
                .collect()
        });
        match result {
            Ok(outputs) => Ok(outputs),
            Err(failure) => {
                let failure = self.retire_failed_execution(failure, &mut stages);
                Err(ImageGenerationError {
                    failure,
                    completed_stages: stages,
                    completed_steps: steps,
                })
            }
        }
    }

    fn retire_failed_execution(
        &mut self,
        failure: ImageGenerationFailure<R>,
        stages: &mut Vec<ImageStageReceipt>,
    ) -> ImageGenerationFailure<R> {
        let failure = match failure {
            ImageGenerationFailure::Execution {
                source: TensorExecutionError::CancelledBeforeSubmission(cancellation),
                active,
            } => match self.finish_stage(active, stages) {
                Ok(()) => ImageGenerationFailure::Cancelled(cancellation),
                Err(failure) => failure,
            },
            failure => failure,
        };
        self.stopped = !matches!(&failure, ImageGenerationFailure::Cancelled(_));
        failure
    }

    fn run(
        &mut self,
        prepared: PreparedRequest<R>,
        requests: &[ImageGenerationRequest],
        job: u64,
        stages: &mut Vec<ImageStageReceipt>,
        steps: &mut Vec<ImageDenoiseStepReceipt>,
    ) -> Result<Vec<TensorExecutionOutput>, ImageGenerationFailure<R>> {
        let PreparedRequest {
            prompts,
            request_order,
            transformer,
            decoder,
            latent_shape,
            text_stage,
            noise_stage,
            denoise_stage,
            prefix_decode,
            decode_stage,
            edit,
        } = prepared;
        let process = |source| ImageGenerationFailure::Processing {
            source,
            active: None,
        };
        let ordered_requests = request_order.requests(requests).map_err(process)?;

        let (text_input, reference) = if let Some(edit) = edit {
            let (text, reference) = self.run_edit_conditioning(edit, text_stage, job, stages)?;
            (text, Some(reference))
        } else {
            let inputs = prompts
                .managed_inputs(
                    &self.host,
                    &self.assets.text,
                    &self
                        .assets
                        .sources
                        .text_encoder
                        .tensor(Qwen3VLTextEncoderFamily::EMBEDDING_EXTERNAL_NAME)
                        .map_err(|e| process(e.into()))?,
                )
                .map_err(process)?
                .into_parts()
                .1;
            let text = self.start_stage(text_stage, job, stages)?;
            let (text, outputs) = self.execute_stage(text, job, Vec::from(inputs))?;
            self.finish_stage(text, stages)?;
            let hidden =
                only_output(outputs, Qwen3VLTextEncoderFamily::OUTPUT_ID).map_err(process)?;
            let features = prompts
                .crop_hidden(&self.host, &self.assets.text, &hidden)
                .map_err(process)?;
            drop(hidden);
            (features.into_tensor(), None)
        };

        // A batch has independent seed streams, not one longer draw. The
        // noise session is reused while per-sample outputs are copied directly
        // into one admitted NCHW backing. Heavy model stages execute batch B.
        let [_, _, height, width] = latent_shape.nchw_dimensions();
        let item_shape = QwenImage21LatentShape::new(1, height, width).map_err(process)?;
        let mut batch_noise = self
            .host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, latent_shape.byte_len())
            .map_err(|e| process(e.into()))?;
        let mut noise = self.start_stage(noise_stage, job, stages)?;
        for (index, request) in ordered_requests.enumerate() {
            let state = match noise_state(&self.host, request.seed) {
                Ok(state) => state,
                Err(error) => return Err(noise.processing(error)),
            };
            let (active, outputs) = self.execute_stage(noise, job, vec![state])?;
            noise = active;
            let copy = (|| -> anyhow::Result<()> {
                let output = only_output(outputs, NormalNoiseFamily::OUTPUT_ID)?;
                let checked = QwenImage21ManagedLatents::from_output(
                    &self.host,
                    &output,
                    item_shape,
                    QwenImage21LatentLayout::Nchw,
                    QwenImage21LatentDomain::Normalized,
                )?;
                let stride = item_shape.byte_len();
                batch_noise.as_mut()[index * stride..(index + 1) * stride]
                    .copy_from_slice(checked.bytes().bytes());
                Ok(())
            })();
            if let Err(error) = copy {
                return Err(noise.processing(error));
            }
        }
        self.finish_stage(noise, stages)?;
        let batch_noise = ManagedHostTensor::new(
            id(NormalNoiseFamily::OUTPUT_ID).map_err(process)?,
            latent_shape
                .tensor(QwenImage21LatentLayout::Nchw)
                .map_err(process)?,
            batch_noise.freeze(),
        )
        .map_err(|e| process(e.into()))?;
        let latents = QwenImage21ManagedLatents::new(
            &self.host,
            batch_noise,
            latent_shape,
            QwenImage21LatentLayout::Nchw,
            QwenImage21LatentDomain::Normalized,
        )
        .map_err(process)?;
        let mut sample = latents
            .repack(
                &self.host,
                QwenImage21LatentLayout::Packed,
                id(QwenImage21TransformerFamily::LATENTS_ID).map_err(process)?,
            )
            .map_err(process)?
            .into_input_tensor()
            .map_err(process)?;
        drop(latents);

        let rotary_started = std::time::Instant::now();
        let frequencies = transformer
            .managed_rotary_frequencies(&self.host)
            .map_err(process)?;
        tracing::debug!(
            target: "ferrum_engine::image_generation::timing",
            job,
            phase = ?transformer.execution_phase(),
            rotary_us = trace_elapsed_us(rotary_started),
            "image denoising rotary inputs prepared"
        );
        let mut solver = ManagedBf16Euler::new(
            self.assets.schedule.solver().clone(),
            self.host.clone(),
            sample.tensor().clone(),
            self.profiles.euler,
        )
        .map_err(|e| process(e.into()))?;
        if let Some(decode) = prefix_decode {
            sample = self.run_prefix_denoising(
                prefix_denoising::PrefixDenoise {
                    prefill: denoise_stage,
                    decode,
                    config: transformer,
                    sample,
                    text: text_input.clone(),
                    reference: reference
                        .clone()
                        .context("prefix cache requires a reference image")
                        .map_err(process)?,
                    frequencies: frequencies.clone(),
                },
                &mut solver,
                job,
                stages,
                steps,
            )?;
        } else {
            let mut denoiser = self.start_stage(denoise_stage, job, stages)?;
            let mut euler_us = 0_u64;
            while let Some(step) = solver.current_step() {
                let time =
                    match transformer.managed_time_features(&self.host, step.timestep().as_f32()) {
                        Ok(value) => value,
                        Err(error) => return Err(denoiser.processing(error)),
                    };
                let mut inputs = vec![
                    sample.clone(),
                    text_input.clone(),
                    time,
                    frequencies.clone(),
                ];
                if let Some(reference) = &reference {
                    inputs.push(reference.clone());
                }
                let (active, outputs) = self.execute_stage(denoiser, job, inputs)?;
                denoiser = active;
                let velocity = match only_output(outputs, QwenImage21TransformerFamily::VELOCITY_ID)
                {
                    Ok(value) => value,
                    Err(error) => return Err(denoiser.processing(error)),
                };
                let euler_started = std::time::Instant::now();
                sample = match solver.advance(&sample, velocity.tensor(), velocity.bytes()) {
                    Ok((_, next)) => next,
                    Err(error) => return Err(denoiser.processing(error.into())),
                };
                euler_us = euler_us.saturating_add(trace_elapsed_us(euler_started));
                steps.push(ImageDenoiseStepReceipt {
                    index: step.index(),
                    sigma: step.sigma(),
                    next_sigma: step.next_sigma(),
                    scheduler_timestep: step.timestep().as_f32(),
                });
            }
            tracing::debug!(
                target: "ferrum_engine::image_generation::timing",
                job,
                euler_us,
                "image denoising host solver completed"
            );
            self.finish_stage(denoiser, stages)?;
        }
        drop((text_input, frequencies, solver, reference));

        let normalized = QwenImage21ManagedLatents::new(
            &self.host,
            sample,
            latent_shape,
            QwenImage21LatentLayout::Packed,
            QwenImage21LatentDomain::Normalized,
        )
        .map_err(process)?;
        let input = normalized
            .denormalize_for_vae(&self.host, &self.assets.statistics)
            .map_err(process)?
            .into_input_tensor()
            .map_err(process)?;
        drop(normalized);
        let outputs =
            self.decode_batch(decode_stage, &decoder, input, requests.len(), job, stages)?;
        request_order.restore_outputs(outputs).map_err(process)
    }

    fn execute_stage(
        &mut self,
        active: ActiveImageStage<R>,
        job: u64,
        inputs: Vec<ManagedHostTensor>,
    ) -> Result<(ActiveImageStage<R>, Vec<TensorExecutionOutput>), ImageGenerationFailure<R>> {
        let Some(actor) = self.serial_actor.as_mut() else {
            return Err(active.processing(anyhow::anyhow!("image serial actor is closed")));
        };
        active.execute_on_actor(actor, job, inputs)
    }
}

fn trace_elapsed_us(started: std::time::Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}

fn id<T: TryFrom<String, Error = VNextError>>(value: impl Into<String>) -> anyhow::Result<T> {
    Ok(T::try_from(value.into())?)
}

fn noise_state(host: &HostMemoryAccount, seed: u64) -> anyhow::Result<ManagedHostTensor> {
    let mut state = host.allocate_zeroed(HostMemoryClass::InputPreprocess, 16)?;
    state
        .as_mut()
        .copy_from_slice(&PhiloxSeedOffset::fresh(seed).little_endian_bytes());
    Ok(ManagedHostTensor::new(
        id(NormalNoiseFamily::STATE_ID)?,
        NormalNoiseConfig::state_tensor()?,
        state.freeze(),
    )?)
}

fn only_output(
    mut outputs: Vec<TensorExecutionOutput>,
    expected: &str,
) -> anyhow::Result<TensorExecutionOutput> {
    ensure!(
        outputs.len() == 1 && outputs[0].value_id().as_str() == expected,
        "component returned an unexpected public output set"
    );
    Ok(outputs.remove(0))
}

#[cfg(test)]
mod tests;
