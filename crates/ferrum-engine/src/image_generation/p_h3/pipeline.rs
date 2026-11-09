//! H3 composition over the shared execution owner. Numerical selections are
//! explicit; constructing this object neither generates an image nor grants
//! per-request tensor capacity.
use super::{PH3ImageAssets, PH3ImageInitialNoisePolicy, PH3ImageSourceAdapter};
use crate::image_generation::{
    ImageDecoderBatchPolicy, ImageEditPrefixPolicy, ImageGenerationError, ImagePipelineConfig,
    ImagePipelineExecution, ImageStageReceipt,
};
use anyhow::{ensure, Result};
use ferrum_models::p_h3_image_vae::{
    PH3ImageVaeDecoderPreset, PH3ImageVaeEncoderPreset, PH3ImageVaeSpatialPolicy,
    PH3ImageVaeTileBatch,
};
use ferrum_runtime::*;
use std::sync::Arc;

/// The current native compressed route uses F32 sampling and VAE activations.
/// Its explicit profiles do not assert Torch RNG or FP16 decoder equivalence.
#[derive(Clone, Debug)]
pub struct PH3ImagePipelineOptions {
    pub noise_profile: NumericalProfileId,
    pub initial_noise: PH3ImageInitialNoisePolicy,
    pub edit_noise: super::PH3ImageEditNoisePolicy,
    pub edit_pixels: ferrum_models::p_h3_image::PH3ImageEditPixelPolicy,
    pub vae_spatial_policy: PH3ImageVaeSpatialPolicy,
    pub vae_encoder: PH3ImageVaeEncoderPreset,
    pub vae_decoder: PH3ImageVaeDecoderPreset,
    /// Adjacent spatial tiles of one image per decoder execution. This does
    /// not change the service's admitted image-request batch capacity.
    pub vae_decoder_tile_batch: PH3ImageVaeTileBatch,
}
impl PH3ImagePipelineOptions {
    pub fn comfy_f32(noise_profile: NumericalProfileId, maximum_tiles: usize) -> Result<Self> {
        ensure!(maximum_tiles > 0, "H3 tile capacity must be positive");
        Ok(Self {
            noise_profile,
            initial_noise: PH3ImageInitialNoisePolicy::default(),
            edit_noise: super::PH3ImageEditNoisePolicy::default(),
            edit_pixels: ferrum_models::p_h3_image::PH3ImageEditPixelPolicy::default(),
            vae_spatial_policy: PH3ImageVaeSpatialPolicy::comfy_default(maximum_tiles),
            vae_encoder: PH3ImageVaeEncoderPreset::CudaF32Fma,
            vae_decoder: PH3ImageVaeDecoderPreset::F32A8FullAttention,
            vae_decoder_tile_batch: PH3ImageVaeTileBatch::default(),
        })
    }
}

pub struct PH3ImagePipeline<R: DeviceRuntime> {
    pub(super) execution: ImagePipelineExecution<R, PH3ImageSourceAdapter>,
    pub(super) options: PH3ImagePipelineOptions,
    pub(super) prompt_expansion_diagnostics: super::PH3ImagePromptExpansionDiagnosticPolicy,
    pub(super) main_text_diagnostics: Option<super::conditioning_diagnostics::MainTextDiagnostics>,
    pub(super) main_vision_diagnostics:
        Option<super::conditioning_diagnostics::MainVisionDiagnostics>,
    pub(super) final_latent_diagnostics:
        Option<super::final_latent_diagnostics::FinalLatentDiagnostics>,
    pub(super) dit_input_diagnostics: Option<super::dit_input_diagnostics::DitInputDiagnostics>,
}
impl<R: DeviceRuntime> PH3ImagePipeline<R> {
    /// The composition root provides one backend and provider registry for all
    /// model stages. The shared owner creates its one original serial actor;
    /// this facade has no separate runtime, allocator or residency ledger.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        runtime: Arc<R>,
        host: HostMemoryAccount,
        registry: OperationRuntimeRegistry<R>,
        catalog: CapabilityCatalog,
        materializers: WeightMaterializerRegistry,
        assets: PH3ImageAssets,
        config: ImagePipelineConfig,
        options: PH3ImagePipelineOptions,
    ) -> Result<Self> {
        ensure!(
            config.decoder_batch_policy == ImageDecoderBatchPolicy::SingleImage,
            "H3 request-local tiled decoding requires the single-image decoder policy"
        );
        ensure!(
            config.edit_prefix_policy == ImageEditPrefixPolicy::Recompute,
            "H3 bidirectional denoising does not declare a causal edit-prefix cache"
        );
        let execution = ImagePipelineExecution::new(
            runtime,
            host,
            registry,
            catalog,
            materializers,
            config,
            PH3ImageSourceAdapter::new(Arc::new(assets)),
        )?;
        Ok(Self {
            execution,
            options,
            prompt_expansion_diagnostics: super::PH3ImagePromptExpansionDiagnosticPolicy::Disabled,
            main_vision_diagnostics: None,
            main_text_diagnostics: None,
            final_latent_diagnostics: None,
            dit_input_diagnostics: None,
        })
    }
    /// Enable one synchronous borrowed capture after each validated main vision
    /// execution. The limit is RGBA plus four F32 logical payloads per callback;
    /// it is not an additional Host reservation or a total RSS limit.
    pub fn with_main_vision_observer(
        mut self,
        maximum_payload_bytes: std::num::NonZeroUsize,
        observer: Box<dyn super::PH3ImageMainVisionObserver>,
    ) -> Result<Self> {
        ensure!(
            self.execution.next_job == 0,
            "main vision diagnostics must be selected before request preparation"
        );
        ensure!(
            self.main_text_diagnostics.is_none()
                && self.final_latent_diagnostics.is_none()
                && self.dit_input_diagnostics.is_none(),
            "select only one image diagnostic capture"
        );
        self.main_vision_diagnostics =
            Some(super::conditioning_diagnostics::MainVisionDiagnostics::new(
                maximum_payload_bytes,
                observer,
            ));
        Ok(self)
    }

    /// Borrow the five validated main-text readbacks synchronously, before
    /// their original owners retire. The limit covers raw F32 bytes per callback.
    pub fn with_main_text_observer(
        mut self,
        maximum_payload_bytes: std::num::NonZeroUsize,
        observer: Box<dyn super::PH3ImageMainTextObserver>,
    ) -> Result<Self> {
        ensure!(
            self.execution.next_job == 0,
            "main text diagnostics must be selected before request preparation"
        );
        ensure!(
            self.main_vision_diagnostics.is_none()
                && self.final_latent_diagnostics.is_none()
                && self.dit_input_diagnostics.is_none(),
            "select only one image diagnostic capture"
        );
        self.main_text_diagnostics =
            Some(super::conditioning_diagnostics::MainTextDiagnostics::new(
                maximum_payload_bytes,
                observer,
            ));
        Ok(self)
    }

    /// Borrow each B1 step's original inputs and solver state before execution.
    /// Disabled by default; synchronous capture is not a performance run.
    pub fn with_dit_input_observer(
        mut self,
        maximum_payload_bytes: std::num::NonZeroUsize,
        observer: Box<dyn super::PH3ImageDitInputObserver>,
    ) -> Result<Self> {
        ensure!(
            self.execution.next_job == 0,
            "DiT input diagnostics must precede request preparation"
        );
        ensure!(
            self.main_text_diagnostics.is_none()
                && self.main_vision_diagnostics.is_none()
                && self.final_latent_diagnostics.is_none(),
            "select only one image diagnostic capture"
        );
        self.dit_input_diagnostics = Some(super::dit_input_diagnostics::DitInputDiagnostics::new(
            maximum_payload_bytes,
            observer,
        ));
        Ok(self)
    }

    /// Borrow the original completed target after successful decoding. The
    /// per-callback limit covers raw latent bytes, not PNG or process RSS.
    pub fn with_final_latent_observer(
        mut self,
        maximum_payload_bytes: std::num::NonZeroUsize,
        observer: Box<dyn super::PH3ImageFinalLatentObserver>,
    ) -> Result<Self> {
        ensure!(
            self.execution.next_job == 0,
            "final latent diagnostics must be selected before request preparation"
        );
        ensure!(
            self.main_text_diagnostics.is_none()
                && self.main_vision_diagnostics.is_none()
                && self.dit_input_diagnostics.is_none(),
            "select only one image diagnostic capture"
        );
        self.final_latent_diagnostics = Some(
            super::final_latent_diagnostics::FinalLatentDiagnostics::new(
                maximum_payload_bytes,
                observer,
            ),
        );
        Ok(self)
    }

    pub fn is_stopped(&self) -> bool {
        self.execution.is_stopped()
    }
    pub fn host_account(&self) -> &HostMemoryAccount {
        self.execution.host_account()
    }
    pub fn residency_policy(&self) -> crate::image_generation::ImageResidencyPolicy {
        self.execution.config.residency
    }
    pub fn runtime(&self) -> &Arc<R> {
        self.execution.runtime()
    }
    pub fn shutdown(&mut self) -> Result<Vec<ImageStageReceipt>, ImageGenerationError<R>> {
        self.execution.shutdown()
    }
    pub fn with_device_timing_mode(mut self, mode: DeviceTimingMode) -> Result<Self> {
        ensure!(
            self.execution.next_job == 0,
            "device timing must be selected before request preparation"
        );
        self.execution.config.executor.device_timing_mode = mode;
        Ok(self)
    }
}
