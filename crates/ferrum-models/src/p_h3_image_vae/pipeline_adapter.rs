//! Opaque model adapter for a shared pipeline executor. Runtime construction,
//! static installation/residency, streams and Host scopes remain with that
//! executor. Source views preserve the released precision and last-time-tap
//! mapping; a generic BF16 loader is not an equivalent source.
use super::{
    decode_tiles::{PH3ImageVaeDecodeError, PH3ImageVaeTileDecoder},
    decoder::{
        self, PH3ImageVaeDecoderCheckpoint, PH3ImageVaeDecoderExecutionConfig,
        PH3ImageVaeDecoderFamily, PH3ImageVaeDecoderLimits,
    },
    decoder_block::PH3ImageVaeDecoderBlockExecutionPreset,
    encode_tiles::{PH3ImageVaeEncodeError, PH3ImageVaeTileEncoder},
    encoder::{
        self, PH3ImageVaeEncoderCheckpoint, PH3ImageVaeEncoderConfig, PH3ImageVaeEncoderFamily,
        PH3ImageVaeEncoderLimits,
    },
    encoder_vertical::PH3ImageVaeEncoderArchitecture,
    image_layout::PH3ImageVaeImageLayout,
    latent_statistics::PH3ImageVaeLatentStatistics,
    tiling::{PH3ImageVaeCanvas, PH3ImageVaeMomentCanvas, PH3ImageVaeTilePlan},
    PH3ImageVaeDecoderConfig,
};
pub use super::{encode_layout::PH3ImageVaePixelDomain, tiling::PH3ImageVaeSpatialPolicy};
use crate::image_pixels::{ImagePixelEncoding, OwnedImagePixels};
use anyhow::{ensure, Context, Result};
use ferrum_execution::{
    component::TensorComponentSession, TensorExecutionError, TensorSerialActor,
};
use ferrum_quantization::comfy_convrot::ComfyCanonicalWeightSource;
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
mod encoder_host_requirements;
mod host_backing;
mod host_requirements;
pub use encoder_host_requirements::{
    PH3ImageVaeEncoderHostPhase, PH3ImageVaeEncoderHostPhaseBudget,
    PH3ImageVaeEncoderHostRequirements,
};
pub use host_requirements::{
    PH3ImageVaeDecoderHostPhase, PH3ImageVaeDecoderHostPhaseBudget,
    PH3ImageVaeDecoderHostRequirements,
};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PH3ImageVaeAdapterLimits {
    pub max_metadata_bytes: usize,
    pub max_stored_bytes: u64,
    pub max_plan_nodes: usize,
}
impl PH3ImageVaeAdapterLimits {
    fn encoder(self) -> PH3ImageVaeEncoderLimits {
        PH3ImageVaeEncoderLimits {
            max_metadata_bytes: self.max_metadata_bytes,
            max_stored_bytes: self.max_stored_bytes,
            max_plan_nodes: self.max_plan_nodes,
        }
    }
    fn decoder(self) -> PH3ImageVaeDecoderLimits {
        PH3ImageVaeDecoderLimits {
            max_metadata_bytes: self.max_metadata_bytes,
            max_stored_bytes: self.max_stored_bytes,
            max_plan_nodes: self.max_plan_nodes,
        }
    }
}
/// A declared route, not hardware qualification or a Torch binary-parity claim.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PH3ImageVaeEncoderPreset {
    ReferenceF32,
    CudaF32Fma,
}
impl PH3ImageVaeEncoderPreset {
    fn profile(self) -> &'static str {
        match self {
            Self::ReferenceF32 => PH3ImageVaeEncoderFamily::REFERENCE_PROFILE_ID,
            Self::CudaF32Fma => PH3ImageVaeEncoderFamily::CUDA_PROFILE_ID,
        }
    }
}
/// Explicit F32 activations/A8 projections/full F32 attention. This is not
/// upstream's default FP16 or opportunistic INT8-attention preset.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PH3ImageVaeDecoderPreset {
    F32A8FullAttention,
}

struct ValidatedSource {
    source: Arc<ComfyCanonicalWeightSource>,
    encoder: PH3ImageVaeEncoderArchitecture,
    decoder: PH3ImageVaeDecoderConfig,
    limits: PH3ImageVaeAdapterLimits,
    encoder_schema: String,
    decoder_schema: String,
    encoder_stored_bytes: u64,
    decoder_stored_bytes: u64,
    // Weight geometry is independent of activation batch/tile dimensions.
    // Keep these already-validated recipes once, without any Host account.
    encoder_source: PH3ImageVaeEncoderConfig,
    decoder_source: PH3ImageVaeDecoderExecutionConfig,
}
#[derive(Clone)]
pub struct PH3ImageVaeAdapter {
    validated: Arc<ValidatedSource>,
}
impl PH3ImageVaeAdapter {
    /// Validate the immutable complete source once. Activation geometry is not
    /// inferred from a filename/model name. Later prepares compare the actual
    /// physical Stored schema and do not rescan every large F32 CNN weight.
    pub fn new(
        source: Arc<ComfyCanonicalWeightSource>,
        hf_metadata: &[u8],
        limits: PH3ImageVaeAdapterLimits,
    ) -> Result<Self> {
        ensure!(
            hf_metadata.len() <= limits.max_metadata_bytes,
            "VAE metadata exceeds adapter capacity"
        );
        let encoder = PH3ImageVaeEncoderArchitecture::from_hf_json(hf_metadata)?;
        let decoder = PH3ImageVaeDecoderConfig::from_hf_json(hf_metadata)?;
        encoder
            .latent_channels()
            .checked_mul(2)
            .context("VAE mean/log-variance channel extent overflows")?;
        ensure!(
            encoder.latent_channels() == decoder.latent_channels()
                && encoder.in_channels() == 3
                && decoder.out_channels() == 3,
            "VAE adapter requires consistent RGB/latent architecture"
        );
        let minimum = minimum_encoder_extent(&encoder)?;
        let ec = PH3ImageVaeEncoderConfig::new(encoder.clone(), 1, minimum, minimum)?;
        let dc = PH3ImageVaeDecoderExecutionConfig::new(
            decoder.clone(),
            1,
            1,
            1,
            PH3ImageVaeDecoderBlockExecutionPreset::F32A8FullAttention,
        )?;
        encoder::checkpoint_source::validate(&ec, source.archive(), limits.encoder())?;
        decoder::checkpoint_source::validate(&dc, &source, limits.decoder())?;
        PH3ImageVaeLatentStatistics::new(&decoder, &source)?;
        // Use the core's canonical prepared schema ordering. Raw family
        // inventory order is not its normalized Stored-schema fingerprint.
        let encoder_family =
            TypedFamilyRegistration::new(PH3ImageVaeEncoderFamily::new(limits.encoder())?)
                .prepare_with_profile(
                    &serde_json::to_value(&ec)?,
                    &profile(PH3ImageVaeEncoderFamily::REFERENCE_PROFILE_ID)?,
                )?;
        let decoder_family =
            TypedFamilyRegistration::new(PH3ImageVaeDecoderFamily::new(limits.decoder())?)
                .prepare_with_profile(
                    &serde_json::to_value(&dc)?,
                    &profile(PH3ImageVaeDecoderFamily::CUDA_PROFILE_ID)?,
                )?;
        let encoder_schema = encoder_family.weights().stored_schema()?.fingerprint()?;
        let decoder_schema = decoder_family.weights().stored_schema()?.fingerprint()?;
        let encoder_stored_bytes = stored_bytes(encoder_family.weights().stored_schema()?)?;
        let decoder_stored_bytes = stored_bytes(decoder_family.weights().stored_schema()?)?;
        Ok(Self {
            validated: Arc::new(ValidatedSource {
                source,
                encoder,
                decoder,
                limits,
                encoder_schema,
                decoder_schema,
                encoder_stored_bytes,
                decoder_stored_bytes,
                encoder_source: ec,
                decoder_source: dc,
            }),
        })
    }

    pub fn latent_channels(&self) -> u64 {
        self.validated.decoder.latent_channels()
    }
    pub fn spatial_scale(&self) -> u64 {
        self.validated.decoder.patch_size()
    }
    pub fn encoder_stored_bytes(&self) -> u64 {
        self.validated.encoder_stored_bytes
    }
    pub fn decoder_stored_bytes(&self) -> u64 {
        self.validated.decoder_stored_bytes
    }

    pub fn encoder_weight_source(&self, host: HostMemoryAccount) -> PH3ImageVaeWeightSource<'_> {
        PH3ImageVaeWeightSource {
            kind: WeightSourceKind::Encoder(&self.validated),
            host,
        }
    }
    pub fn decoder_weight_source(&self, host: HostMemoryAccount) -> PH3ImageVaeWeightSource<'_> {
        PH3ImageVaeWeightSource {
            kind: WeightSourceKind::Decoder(&self.validated),
            host,
        }
    }
    /// The shared executor supplies the actual prepared family/compiled plan.
    /// Their full identity is checked by the core grouping helper; the source
    /// domain additionally requires the already validated physical schema.
    pub fn maximum_encoder_group_backing_bytes(
        &self,
        family: &PreparedModelFamily,
        plan: &ExecutionPlan,
        host: &HostMemoryAccount,
    ) -> Result<usize> {
        ensure!(
            family.family_id()
                == PH3ImageVaeEncoderFamily::new(self.validated.limits.encoder())?.family_id()
                && family.stored_weight_schema()?.fingerprint()? == self.validated.encoder_schema,
            "compiled encoder does not belong to this validated VAE source domain"
        );
        host_backing::maximum(family, plan, host, |component| {
            Ok(encoder::checkpoint_source::temporary_backing_bytes(
                &self.validated.encoder_source,
                component,
            )?)
        })
    }
    pub fn maximum_decoder_group_backing_bytes(
        &self,
        family: &PreparedModelFamily,
        plan: &ExecutionPlan,
        host: &HostMemoryAccount,
    ) -> Result<usize> {
        ensure!(
            family.family_id()
                == PH3ImageVaeDecoderFamily::new(self.validated.limits.decoder())?.family_id()
                && family.stored_weight_schema()?.fingerprint()? == self.validated.decoder_schema,
            "compiled decoder does not belong to this validated VAE source domain"
        );
        host_backing::maximum(family, plan, host, |component| {
            decoder::checkpoint_source::temporary_backing_bytes(&self.validated.source, component)
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_encoder(
        &self,
        batch: u64,
        width: usize,
        height: usize,
        policy: PH3ImageVaeSpatialPolicy,
        preset: PH3ImageVaeEncoderPreset,
    ) -> Result<PreparedPH3ImageVaeEncoder> {
        let v = &self.validated;
        let plan = PH3ImageVaeTilePlan::new(
            width,
            height,
            usize::try_from(v.decoder.patch_size())?,
            policy,
        )?;
        let tile = plan.tile(0)?;
        let config = PH3ImageVaeEncoderConfig::new(
            v.encoder.clone(),
            batch,
            u64::try_from(tile.height)?,
            u64::try_from(tile.width)?,
        )?;
        let family =
            TypedFamilyRegistration::new(PH3ImageVaeEncoderFamily::new(v.limits.encoder())?)
                .prepare_with_profile(
                    &serde_json::to_value(&config)?,
                    &profile(preset.profile())?,
                )?;
        ensure!(
            family.weights().stored_schema()?.fingerprint()? == v.encoder_schema,
            "prepared encoder physical weights differ from validated source"
        );
        let options = config.compile_options()?;
        // The literal encoder's temporal-only downsamples need not be a pure
        // spatial scale. Reject such geometry rather than silently reshaping it.
        let expected = spec(&[
            batch,
            v.encoder.latent_channels() * 2,
            u64::try_from(tile.height / plan.spatial_scale())?,
            u64::try_from(tile.width / plan.spatial_scale())?,
        ])?;
        ensure!(
            resolved(config.output_tensor_spec()?)? == expected,
            "encoder tile output differs from the declared spatial compression"
        );
        Ok(PreparedPH3ImageVaeEncoder {
            validated: Arc::clone(v),
            config,
            family,
            options,
            plan,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_decoder(
        &self,
        batch: u64,
        width: usize,
        height: usize,
        policy: PH3ImageVaeSpatialPolicy,
        preset: PH3ImageVaeDecoderPreset,
    ) -> Result<PreparedPH3ImageVaeDecoder> {
        let v = &self.validated;
        let scale = usize::try_from(v.decoder.patch_size())?;
        let plan = PH3ImageVaeTilePlan::new(width, height, scale, policy)?;
        let tile = plan.tile(0)?;
        let execution_preset = match preset {
            PH3ImageVaeDecoderPreset::F32A8FullAttention => {
                PH3ImageVaeDecoderBlockExecutionPreset::F32A8FullAttention
            }
        };
        let config = PH3ImageVaeDecoderExecutionConfig::new(
            v.decoder.clone(),
            batch,
            u64::try_from(tile.height / scale)?,
            u64::try_from(tile.width / scale)?,
            execution_preset,
        )?;
        let provider = PH3ImageVaeDecoderFamily::new(v.limits.decoder())?;
        let options = provider.compile_options(&config)?;
        let family = TypedFamilyRegistration::new(provider).prepare_with_profile(
            &serde_json::to_value(&config)?,
            &profile(PH3ImageVaeDecoderFamily::CUDA_PROFILE_ID)?,
        )?;
        ensure!(
            family.weights().stored_schema()?.fingerprint()? == v.decoder_schema,
            "prepared decoder physical weights differ from validated source"
        );
        Ok(PreparedPH3ImageVaeDecoder {
            validated: Arc::clone(v),
            config,
            family,
            options,
            plan,
        })
    }
}

/// Owns only small geometry/program metadata and Arc source owners. Safe to
/// retain in an executor cache requiring Send+Sync+'static; it borrows neither
/// its own config nor a temporary checkpoint.
pub struct PreparedPH3ImageVaeEncoder {
    validated: Arc<ValidatedSource>,
    config: PH3ImageVaeEncoderConfig,
    family: PreparedModelFamily,
    options: ProgramPlanCompileOptions,
    plan: PH3ImageVaeTilePlan,
}
impl PreparedPH3ImageVaeEncoder {
    pub fn family(&self) -> &PreparedModelFamily {
        &self.family
    }
    pub fn compile_options(&self) -> &ProgramPlanCompileOptions {
        &self.options
    }
    /// Bind only the current static-installation Host view. Cached prepared
    /// metadata never retains a per-request scope or falls back to a root.
    pub fn weight_source(&self, host: HostMemoryAccount) -> PH3ImageVaeWeightSource<'_> {
        PH3ImageVaeWeightSource {
            kind: WeightSourceKind::Encoder(&self.validated),
            host,
        }
    }
    pub fn pixel_input_spec(&self) -> Result<ResolvedTensorSpec> {
        let [h, w] = self.plan.dimensions();
        spec(&[self.config.batch(), 3, h as u64, w as u64])
    }
    pub fn posterior_noise_spec(&self) -> Result<ResolvedTensorSpec> {
        self.normalized_latent_spec()
    }
    pub fn normalized_latent_spec(&self) -> Result<ResolvedTensorSpec> {
        let [h, w] = self.plan.dimensions();
        let s = self.plan.spatial_scale();
        spec(&[
            self.config.batch(),
            self.validated.encoder.latent_channels(),
            (h / s) as u64,
            (w / s) as u64,
        ])
    }
    pub fn tile_input_spec(&self) -> Result<ResolvedTensorSpec> {
        resolved(
            self.config.input_tensor_specs()?
                [&ProgramValueId::new(PH3ImageVaeEncoderFamily::INPUT_ID)?]
                .clone(),
        )
    }
    /// Extra source-owned WeightBacking for the actual compiled static groups.
    /// Materializer preparation/retention and TransferStaging are separate.
    pub fn maximum_group_backing_bytes(
        &self,
        plan: &ExecutionPlan,
        host: &HostMemoryAccount,
    ) -> Result<usize> {
        host_backing::maximum(&self.family, plan, host, |component| {
            Ok(encoder::checkpoint_source::temporary_backing_bytes(
                &self.config,
                component,
            )?)
        })
    }
    /// Whole immutable RGB/noise plus the larger of one normalized tile and
    /// the normalized posterior successor. Every backing rounds separately.
    pub fn input_peak(&self, host: &HostMemoryAccount) -> Result<usize> {
        let image = backing(host, &self.pixel_input_spec()?)?;
        let noise = backing(host, &self.posterior_noise_spec()?)?;
        let tile = backing(host, &self.tile_input_spec()?)?;
        checked_sum(&[image, noise, tile.max(noise)])
    }
    /// Includes full moments, raw neighbour row/current readback and working
    /// tile. Input tile, immutable RGB/noise and posterior have separate costs.
    pub fn output_peak(&self, host: &HostMemoryAccount) -> Result<usize> {
        PH3ImageVaeMomentCanvas::output_peak(
            host,
            &self.plan,
            usize::try_from(self.config.batch())?,
            usize::try_from(self.validated.encoder.latent_channels() * 2)?,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn bind<'a, R: DeviceRuntime>(
        &'a self,
        session: &'a mut TensorComponentSession<R>,
        pixels: &'a ManagedHostTensor,
        pixel_domain: PH3ImageVaePixelDomain,
        noise: &'a ManagedHostTensor,
        request: RequestIdentity,
        input_host: HostMemoryAccount,
        output_host: HostMemoryAccount,
    ) -> Result<PH3ImageVaeEncoderConsumer<'a, R>> {
        let v = &self.validated;
        let checkpoint = PH3ImageVaeEncoderCheckpoint::view_validated(
            &self.config,
            v.source.archive(),
            input_host.clone(),
        );
        let statistics = PH3ImageVaeLatentStatistics::new(&v.decoder, &v.source)?;
        Ok(PH3ImageVaeEncoderConsumer {
            inner: PH3ImageVaeTileEncoder::bind(
                session,
                &checkpoint,
                v.limits.encoder(),
                self.plan.clone(),
                pixels,
                pixel_domain,
                noise,
                statistics,
                request,
                input_host,
                output_host,
            )?,
        })
    }
}

pub struct PreparedPH3ImageVaeDecoder {
    validated: Arc<ValidatedSource>,
    config: PH3ImageVaeDecoderExecutionConfig,
    family: PreparedModelFamily,
    options: ProgramPlanCompileOptions,
    plan: PH3ImageVaeTilePlan,
}
impl PreparedPH3ImageVaeDecoder {
    pub fn family(&self) -> &PreparedModelFamily {
        &self.family
    }
    pub fn compile_options(&self) -> &ProgramPlanCompileOptions {
        &self.options
    }
    pub fn weight_source(&self, host: HostMemoryAccount) -> PH3ImageVaeWeightSource<'_> {
        PH3ImageVaeWeightSource {
            kind: WeightSourceKind::Decoder(&self.validated),
            host,
        }
    }
    pub fn maximum_group_backing_bytes(
        &self,
        plan: &ExecutionPlan,
        host: &HostMemoryAccount,
    ) -> Result<usize> {
        host_backing::maximum(&self.family, plan, host, |component| {
            decoder::checkpoint_source::temporary_backing_bytes(&self.validated.source, component)
        })
    }
    pub fn normalized_latent_spec(&self) -> Result<ResolvedTensorSpec> {
        let [h, w] = self.plan.dimensions();
        let s = self.plan.spatial_scale();
        spec(&[
            self.config.batch(),
            self.validated.decoder.latent_channels(),
            (h / s) as u64,
            (w / s) as u64,
        ])
    }
    pub fn pixel_output_spec(&self) -> Result<ResolvedTensorSpec> {
        let [h, w] = self.plan.dimensions();
        spec(&[self.config.batch(), 3, h as u64, w as u64])
    }
    pub const fn pixel_encoding(&self) -> ImagePixelEncoding {
        ImagePixelEncoding::F32UnitRgbRne
    }
    /// Reservation bound for actual postprocessing and readback owners:
    /// canvas/work/old+new seam storage, one pending unpatchified tile, and one
    /// complete decoder readback. No uncharged temporary tile is assumed.
    pub fn output_peak(&self, host: &HostMemoryAccount) -> Result<usize> {
        let batch = usize::try_from(self.config.batch())?;
        let tile = self.plan.tile(0)?;
        let pixels = spec(&[
            self.config.batch(),
            3,
            u64::try_from(tile.height)?,
            u64::try_from(tile.width)?,
        ])?;
        let (_, _, patches) = self.tile_layout_specs()?;
        checked_sum(&[
            PH3ImageVaeCanvas::output_peak(host, &self.plan, batch, 3)?,
            backing(host, &pixels)?,
            backing(host, &patches)?,
        ])
    }
    /// Immutable normalized full sample plus one gathered tile and the two
    /// cached frequency backings; all use the same parent Input account.
    pub fn input_peak(&self, host: &HostMemoryAccount) -> Result<usize> {
        let (rows, frequencies, _) = self.tile_layout_specs()?;
        checked_sum(&[
            backing(host, &self.normalized_latent_spec()?)?,
            backing(host, &rows)?,
            backing(host, &frequencies)?,
            backing(host, &frequencies)?,
        ])
    }
    pub fn tile_layout_specs(
        &self,
    ) -> Result<(ResolvedTensorSpec, ResolvedTensorSpec, ResolvedTensorSpec)> {
        let layout = PH3ImageVaeImageLayout::new(
            self.validated.decoder.clone(),
            usize::try_from(self.config.batch())?,
            usize::try_from(self.config.latent_height())?,
            usize::try_from(self.config.latent_width())?,
        )?;
        Ok((
            layout.row_spec()?,
            layout.frequency_spec()?,
            layout.patch_output_spec()?,
        ))
    }
    pub fn bind<'a, R: DeviceRuntime>(
        &'a self,
        session: &'a mut TensorComponentSession<R>,
        normalized: &'a ManagedHostTensor,
        request: RequestIdentity,
        input_host: HostMemoryAccount,
        output_host: HostMemoryAccount,
    ) -> Result<PH3ImageVaeDecoderConsumer<'a, R>> {
        let v = &self.validated;
        let checkpoint = PH3ImageVaeDecoderCheckpoint::view_validated(
            &self.config,
            &v.source,
            input_host.clone(),
        );
        Ok(PH3ImageVaeDecoderConsumer {
            inner: PH3ImageVaeTileDecoder::bind(
                session,
                &checkpoint,
                v.limits.decoder(),
                self.plan.clone(),
                normalized,
                request,
                input_host,
                output_host.clone(),
            )?,
            output_host,
        })
    }
}
/// Per-installation source binding; all managed promotions/repacking consume
/// the injected caller's actual WeightBacking grants. Payload owners survive
/// independently through the ordinary initialization transaction/fence.
pub struct PH3ImageVaeWeightSource<'a> {
    kind: WeightSourceKind<'a>,
    host: HostMemoryAccount,
}
enum WeightSourceKind<'a> {
    Encoder(&'a ValidatedSource),
    Decoder(&'a ValidatedSource),
}
impl WeightComponentSource for PH3ImageVaeWeightSource<'_> {
    fn component<'s>(
        &'s self,
        c: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'s>, VNextError> {
        match self.kind {
            WeightSourceKind::Encoder(p) => encoder::checkpoint_source::component(
                &p.encoder_source,
                p.source.archive(),
                &self.host,
                c,
            ),
            WeightSourceKind::Decoder(p) => {
                decoder::checkpoint_source::component(&p.decoder_source, &p.source, &self.host, c)
            }
        }
    }
}

#[must_use = "execution errors may own submitted work and resource retirement"]
pub enum PH3ImageVaePipelineError<R: DeviceRuntime> {
    Preparation(anyhow::Error),
    Execution(TensorExecutionError<R>),
}
impl<R: DeviceRuntime> std::fmt::Debug for PH3ImageVaePipelineError<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Preparation(e) => f.debug_tuple("Preparation").field(e).finish(),
            Self::Execution(e) => f.debug_tuple("Execution").field(e).finish(),
        }
    }
}
pub struct PH3ImageVaeEncoderConsumer<'a, R: DeviceRuntime> {
    inner: PH3ImageVaeTileEncoder<'a, R>,
}
impl<R: DeviceRuntime> PH3ImageVaeEncoderConsumer<'_, R> {
    pub fn advance(
        &mut self,
        actor: &mut TensorSerialActor<R>,
        run: RunId,
    ) -> std::result::Result<bool, PH3ImageVaePipelineError<R>> {
        self.inner.advance(actor, run).map_err(|e| match e {
            PH3ImageVaeEncodeError::Preparation(e) => PH3ImageVaePipelineError::Preparation(e),
            PH3ImageVaeEncodeError::Execution(e) => PH3ImageVaePipelineError::Execution(e),
        })
    }
    pub fn completed_tiles(&self) -> usize {
        self.inner.completed_tiles()
    }
    pub fn executed_tiles(&self) -> usize {
        self.inner.executed_tiles()
    }
    pub fn take_execution_profile(
        &mut self,
    ) -> Option<ferrum_execution::tensor::TensorExecutionProfile> {
        self.inner.take_execution_profile()
    }
    pub fn finish(&mut self, output_id: ProgramValueId) -> Result<ManagedHostTensor> {
        self.inner.finish(output_id)
    }
}
pub struct PH3ImageVaeDecoderConsumer<'a, R: DeviceRuntime> {
    inner: PH3ImageVaeTileDecoder<'a, R>,
    output_host: HostMemoryAccount,
}
impl<R: DeviceRuntime> PH3ImageVaeDecoderConsumer<'_, R> {
    /// Successful tensor readbacks, including a tile awaiting Host stitching.
    /// Retrying that tile cannot count or submit the same device work twice.
    pub fn executed_tiles(&self) -> usize {
        self.inner.executed_tiles()
    }
    pub fn take_execution_profile(
        &mut self,
    ) -> Option<ferrum_execution::tensor::TensorExecutionProfile> {
        self.inner.take_execution_profile()
    }
    pub fn advance(
        &mut self,
        actor: &mut TensorSerialActor<R>,
        run: RunId,
    ) -> std::result::Result<bool, PH3ImageVaePipelineError<R>> {
        self.inner.advance(actor, run).map_err(|e| match e {
            PH3ImageVaeDecodeError::Preparation(e) => PH3ImageVaePipelineError::Preparation(e),
            PH3ImageVaeDecodeError::Execution(e) => PH3ImageVaePipelineError::Execution(e),
        })
    }
    pub fn completed_tiles(&self) -> usize {
        self.inner.completed_tiles()
    }
    pub fn finish(self) -> Result<OwnedImagePixels> {
        let (spec, bytes) = self.inner.finish()?;
        OwnedImagePixels::new(
            &self.output_host,
            spec,
            bytes,
            ImagePixelEncoding::F32UnitRgbRne,
        )
    }
}
fn minimum_encoder_extent(a: &PH3ImageVaeEncoderArchitecture) -> Result<u64> {
    let mut n = 2u64;
    for (&spatial, &temporal) in a
        .spatial_downsample_factors()
        .iter()
        .zip(a.temporal_downsample_factors())
        .rev()
    {
        if spatial > 1 || temporal > 1 {
            let kernel_minus_pad = if spatial == 2 { 2 } else { 3 };
            n = n
                .checked_sub(1)
                .and_then(|v| v.checked_mul(spatial))
                .and_then(|v| v.checked_add(kernel_minus_pad))
                .context("minimal encoder validation extent overflows")?;
        }
    }
    Ok(n)
}
fn stored_bytes(schema: &WeightSchema) -> Result<u64> {
    schema.components.iter().try_fold(0u64, |a, c| {
        a.checked_add(c.physical_bytes()?)
            .context("VAE physical Stored bytes overflow")
    })
}
fn backing(host: &HostMemoryAccount, tensor: &ResolvedTensorSpec) -> Result<usize> {
    Ok(host.backing_bytes(usize::try_from(tensor.minimum_storage_bytes()?)?)?)
}
fn checked_sum(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(0usize, |a, &b| {
        a.checked_add(b).context("VAE Host backing sum overflows")
    })
}
fn profile(name: &str) -> Result<NumericalProfileId> {
    NumericalProfileId::new(name).map_err(anyhow::Error::msg)
}
fn resolved(t: ProgramTensorSpec) -> Result<ResolvedTensorSpec> {
    Ok(ResolvedTensorSpec::new(
        t.dimensions,
        t.element_type,
        t.layout,
    )?)
}
fn spec(d: &[u64]) -> Result<ResolvedTensorSpec> {
    Ok(ResolvedTensorSpec::new(
        d.to_vec(),
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )?)
}

#[cfg(test)]
mod tests;
