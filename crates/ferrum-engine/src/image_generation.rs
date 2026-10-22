//! Serial, single-image Qwen-Image 2.1 generation on injected runtime owners.
//!
//! This cold BF16 path qualifies G2 functionality. It is not the G3 serving
//! architecture: it recompiles/reloads stages for every request. Throughput
//! qualification additionally requires compatible request batching and retained
//! component sessions with explicit residency/offload policies. Queuing several
//! calls to this single-image path does not provide parallel image generation.
//!
//! Metadata preparation is separate from backend construction. The pipeline
//! never constructs a runtime or Host account, launches a helper process, or
//! exchanges tensor payloads through files. All four component plans compile
//! before the first upload. Each stage must close successfully before the next
//! begins; a stage failure returns the original recovery owner and stops this
//! pipeline instance. This first path has no CFG, image editing, video, batching,
//! device-resident denoise cache, or claim of Torch parity/performance parity.
//! Host claims cover managed tensor payloads and static transfer staging, not
//! tokenizer/metadata containers, mapped checkpoints, or total process RSS.

mod assets;
#[cfg(feature = "cuda")]
pub mod cuda;
mod prepare;
mod stage;
pub use assets::{ImageAssetReceipt, QwenImage21Assets, QwenImage21Metadata, QwenImage21Sources};
pub use stage::{
    ActiveImageStage, ImageGenerationError, ImageGenerationFailure, ImageStage,
    ImageStageExecutionReceipt, ImageStagePlanReceipt, ImageStageReceipt,
};

use anyhow::{ensure, Context};
use ferrum_diffusion::{Bf16EulerProfile, ManagedBf16Euler};
use ferrum_execution::component::{
    CompiledTensorComponent, ComponentInitializationConfig, ComponentInitializationError,
    ComponentInitializationReceipt, ComponentWeightSource, TensorComponentPolicy,
    TensorComponentSession,
};
use ferrum_execution::{TensorExecutionError, TensorExecutionOutput, TensorExecutorConfig};
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
use ferrum_quantization::SafetensorsArchive;
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

#[derive(Debug, Clone)]
pub struct ImagePipelineConfig {
    pub device_capacity_bytes: u64,
    pub maximum_user_tokens: usize,
    pub maximum_width: u32,
    pub maximum_height: u32,
    pub static_initialization: StaticInitializationPolicy,
    pub executor: TensorExecutorConfig,
}
impl ImagePipelineConfig {
    fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            self.device_capacity_bytes > 0 && self.maximum_width > 0 && self.maximum_height > 0,
            "image capacity and dimension limits must be positive"
        );
        ensure!(
            (1..=8192).contains(&self.maximum_user_tokens),
            "user-token limit must be 1..=8192"
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
    pub width: u32,
    pub height: u32,
    pub seed: u64,
    pub stages: Vec<ImageStageReceipt>,
    pub steps: Vec<ImageDenoiseStepReceipt>,
    pub runtime_fingerprint: String,
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
    config: ImagePipelineConfig,
    policy: TensorComponentPolicy,
    next_job: u64,
    stopped: bool,
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
        Ok(Self {
            runtime,
            host,
            registry,
            catalog,
            materializers,
            assets,
            profiles,
            config,
            policy,
            next_job: 0,
            stopped: false,
        })
    }

    pub fn host_account(&self) -> &HostMemoryAccount {
        &self.host
    }
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }

    pub fn generate(
        &mut self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationOutput, ImageGenerationError<R>> {
        let empty_error = |failure| ImageGenerationError {
            failure,
            completed_stages: Vec::new(),
            completed_steps: Vec::new(),
        };
        if self.stopped {
            return Err(empty_error(ImageGenerationFailure::Unavailable));
        }
        let prepared = self
            .prepare(&request)
            .map_err(|e| empty_error(ImageGenerationFailure::Preflight(e)))?;
        let job = self.next_job;
        self.next_job = self.next_job.checked_add(1).ok_or_else(|| {
            empty_error(ImageGenerationFailure::Preflight(anyhow::anyhow!(
                "image job identity exhausted"
            )))
        })?;
        let mut stages = Vec::with_capacity(4);
        let mut steps = Vec::with_capacity(self.assets.schedule.evaluation_count());
        match self.run(prepared, &request, job, &mut stages, &mut steps) {
            Ok(rgba) => Ok(ImageGenerationOutput {
                rgba,
                receipt: ImageGenerationReceipt {
                    job,
                    width: request.width,
                    height: request.height,
                    seed: request.seed,
                    stages,
                    steps,
                    runtime_fingerprint: self
                        .runtime
                        .descriptor()
                        .runtime_implementation_fingerprint
                        .clone(),
                    assets: self.assets.receipt().clone(),
                    euler_profile: self.profiles.euler.id(),
                },
            }),
            Err(failure) => {
                self.stopped = true;
                Err(ImageGenerationError {
                    failure,
                    completed_stages: stages,
                    completed_steps: steps,
                })
            }
        }
    }

    fn run(
        &self,
        prepared: PreparedRequest<R>,
        request: &ImageGenerationRequest,
        job: u64,
        stages: &mut Vec<ImageStageReceipt>,
        steps: &mut Vec<ImageDenoiseStepReceipt>,
    ) -> Result<TensorExecutionOutput, ImageGenerationFailure<R>> {
        let PreparedRequest {
            prompts,
            transformer,
            decoder,
            latent_shape,
            text_stage,
            noise_stage,
            denoise_stage,
            decode_stage,
        } = prepared;
        let process = |source| ImageGenerationFailure::Processing {
            source,
            active: None,
        };

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
        let text = text_stage.start(
            self,
            job,
            ComponentWeightSource::Stored(self.assets.sources.text_encoder.as_ref()),
        )?;
        let (text, outputs) = text.execute(job, Vec::from(inputs))?;
        stages.push(text.close()?);
        let hidden = only_output(outputs, Qwen3VLTextEncoderFamily::OUTPUT_ID).map_err(process)?;
        let features = prompts
            .crop_hidden(&self.host, &self.assets.text, &hidden)
            .map_err(process)?;
        drop(hidden);
        let text_input = features.into_tensor();

        let mut state = self
            .host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 16)
            .map_err(|e| process(e.into()))?;
        state
            .as_mut()
            .copy_from_slice(&PhiloxSeedOffset::fresh(request.seed).little_endian_bytes());
        let state = ManagedHostTensor::new(
            id(NormalNoiseFamily::STATE_ID).map_err(process)?,
            NormalNoiseConfig::state_tensor().map_err(|e| process(e.into()))?,
            state.freeze(),
        )
        .map_err(|e| process(e.into()))?;
        let noise = noise_stage.start(self, job, ComponentWeightSource::None)?;
        let (noise, outputs) = noise.execute(job, vec![state])?;
        stages.push(noise.close()?);
        let noise_output = only_output(outputs, NormalNoiseFamily::OUTPUT_ID).map_err(process)?;
        let latents = QwenImage21ManagedLatents::from_output(
            &self.host,
            &noise_output,
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
        drop((latents, noise_output));

        let frequencies = transformer
            .managed_rotary_frequencies(&self.host)
            .map_err(process)?;
        let mut solver = ManagedBf16Euler::new(
            self.assets.schedule.solver().clone(),
            self.host.clone(),
            sample.tensor().clone(),
            self.profiles.euler,
        )
        .map_err(|e| process(e.into()))?;
        let mut denoiser = denoise_stage.start(
            self,
            job,
            ComponentWeightSource::Stored(self.assets.sources.transformer.as_ref()),
        )?;
        while let Some(step) = solver.current_step() {
            let time = match transformer.managed_time_features(&self.host, step.timestep().as_f32())
            {
                Ok(value) => value,
                Err(error) => return Err(denoiser.processing(error)),
            };
            let (active, outputs) = denoiser.execute(
                job,
                vec![
                    sample.clone(),
                    text_input.clone(),
                    time,
                    frequencies.clone(),
                ],
            )?;
            denoiser = active;
            let velocity = match only_output(outputs, QwenImage21TransformerFamily::VELOCITY_ID) {
                Ok(value) => value,
                Err(error) => return Err(denoiser.processing(error)),
            };
            sample = match solver.advance(&sample, velocity.tensor(), velocity.bytes()) {
                Ok((_, next)) => next,
                Err(error) => return Err(denoiser.processing(error.into())),
            };
            steps.push(ImageDenoiseStepReceipt {
                index: step.index(),
                sigma: step.sigma(),
                next_sigma: step.next_sigma(),
                scheduler_timestep: step.timestep().as_f32(),
            });
        }
        stages.push(denoiser.close()?);
        drop((text_input, frequencies, solver));

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
        let vae = decode_stage.start(
            self,
            job,
            ComponentWeightSource::Stored(self.assets.sources.vae.as_ref()),
        )?;
        let (vae, outputs) = vae.execute(job, vec![input])?;
        stages.push(vae.close()?);
        let rgba = only_output(outputs, QwenImage21VaeDecoderFamily::OUTPUT_ID).map_err(process)?;
        let validation = (|| -> anyhow::Result<()> {
            ensure!(
                rgba.tensor() == &decoder.output_tensor()?,
                "actual RGBA differs from requested decoder shape"
            );
            QwenImage21RgbaOutput::new(&self.host, rgba.tensor(), rgba.bytes())?;
            Ok(())
        })();
        validation.map_err(process)?;
        Ok(rgba)
    }
}

fn id<T: TryFrom<String, Error = VNextError>>(value: impl Into<String>) -> anyhow::Result<T> {
    Ok(T::try_from(value.into())?)
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
