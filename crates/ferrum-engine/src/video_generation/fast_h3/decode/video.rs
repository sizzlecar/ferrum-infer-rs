use super::*;
use ferrum_models::fast_h3::video_vae::{
    FastH3VideoClipConfig, FastH3VideoClipFamily, FastH3VideoClipLimits, FastH3VideoClipSource,
    FastH3VideoDecoderPackage, FastH3VideoLatentLayout, FastH3VideoLatentUnits,
    FastH3VideoRgbChunk, FastH3VideoSpatialPlan, FastH3VideoSpatialPolicy,
    FastH3VideoTemporalAssembler, FastH3VideoTemporalLimits,
};
use ferrum_models::spatial_tiles::SpatialTileBatch;

mod overlap;
mod postprocess;
mod readback;
use postprocess::{DecodedTiles, VideoPostprocessor};
pub use readback::FastH3VideoReadbackAdmission;

/// The optional readback grant shares native custody through initialization,
/// execution and close failures; a failed fence cannot refund it early.
pub type FastH3VideoDecodePhase<R, P> = (
    Arc<FastH3VideoDecodePlan<R>>,
    P,
    Option<FastH3VideoReadbackAdmission>,
);

#[derive(Clone, Copy, Debug)]
pub struct FastH3VideoDecodeGeometry {
    /// [requests,latent_channels,time,height,width]. Batch never denotes time.
    pub latent_shape: [usize; 5],
    pub spatial: FastH3VideoSpatialPolicy,
    /// Upper bound for uniform real spatial groups. The largest admitted
    /// divisor of the tile count reuses one plan and weight installation.
    pub tile_batch: SpatialTileBatch,
    pub temporal: FastH3VideoTemporalLimits,
    pub postprocessing: FastH3VideoPostprocessing,
}

pub struct FastH3VideoDecodePlan<R: DeviceRuntime> {
    source: FastH3VideoClipSource,
    compiled: Arc<CompiledTensorComponent<R>>,
    config: FastH3VideoClipConfig,
    limits: FastH3VideoClipLimits,
    geometry: FastH3VideoDecodeGeometry,
    spatial: FastH3VideoSpatialPlan,
    output_shape: [usize; 5],
    input_spec: ResolvedTensorSpec,
    output_bytes: usize,
    tile_batch: SpatialTileBatch,
    maximum_readback_bytes: usize,
}
impl<R: DeviceRuntime> FastH3VideoDecodePlan<R> {
    #[allow(clippy::too_many_arguments)]
    pub fn compile(
        package: Arc<FastH3VideoDecoderPackage>,
        geometry: FastH3VideoDecodeGeometry,
        limits: FastH3VideoClipLimits,
        host: &HostMemoryAccount,
        registry: &OperationRuntimeRegistry<R>,
        catalog: &CapabilityCatalog,
        materializers: &WeightMaterializerRegistry,
        policy: &TensorComponentPolicy,
    ) -> anyhow::Result<Self> {
        let a = package.config();
        let [batch, channels, time, height, width] = geometry.latent_shape;
        let scale = a.spatial_compression_ratio()?;
        let height = height
            .checked_mul(scale)
            .context("decoded height overflow")?;
        let width = width.checked_mul(scale).context("decoded width overflow")?;
        let temporal =
            FastH3VideoTemporalAssembler::new(a, time, batch, height, width, geometry.temporal)?;
        let frames = temporal.plan().output_frames();
        let output_shape = [batch, frames, height, width, 3];
        let output_bytes = output_shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .filter(|n| *n <= isize::MAX as usize)
            .context("RGB output extent overflow")?;
        ensure!(
            output_bytes <= geometry.temporal.maximum_buffer_bytes,
            "RGB output exceeds declared Host buffer capacity"
        );
        let spatial = FastH3VideoSpatialPlan::new(
            a,
            geometry.latent_shape,
            geometry.spatial,
            geometry.temporal.maximum_buffer_bytes,
        )?;
        // Validate the actual final DiT target shape, not a raw byte alias.
        let patch =
            ferrum_diffusion::SpatiotemporalLatentShape::new(geometry.latent_shape, [1, 2, 2])?;
        let input_spec = ResolvedTensorSpec::new(
            vec![
                (patch.element_count() / (channels * 4)) as u64,
                (channels * 4) as u64,
            ],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )?;
        let mut selected = None;
        for tile_batch in spatial.uniform_tile_group_candidates(geometry.tile_batch) {
            let grid = spatial.grouped_clip_grid(
                temporal.plan().chunk(0)?.latent_count,
                tile_batch.maximum_tiles(),
            )?;
            let (config, source) =
                FastH3VideoClipSource::prepare(package.clone(), grid, host, limits)?;
            let family = TypedFamilyRegistration::new(FastH3VideoClipFamily::new(limits)?)
                .prepare_with_profile(
                    &serde_json::to_value(&config)?,
                    &NumericalProfileId::new(FastH3VideoClipFamily::PROFILE_ID)
                        .map_err(anyhow::Error::msg)?,
                )?;
            let compiled = CompiledTensorComponent::compile(
                family,
                registry,
                catalog,
                policy,
                materializers,
                &config.compile_options(limits)?,
            );
            match compiled {
                Ok(compiled) => {
                    selected = Some((config, source, compiled, tile_batch));
                    break;
                }
                Err(VNextError::PlanCapacityExceeded { .. }) if tile_batch.maximum_tiles() > 1 => {
                    tracing::debug!(
                        tiles = tile_batch.maximum_tiles(),
                        "decoder group exceeds compiled capacity; trying smaller real group"
                    );
                }
                Err(error) => return Err(error.into()),
            }
        }
        let (config, source, compiled, tile_batch) =
            selected.context("decoder has no admitted spatial group")?;
        let mut maximum_readback_bytes = 0;
        for ordinal in 0..temporal.plan().chunk_count() {
            maximum_readback_bytes =
                maximum_readback_bytes.max(spatial.maximum_grouped_readback_backing_bytes(
                    temporal.plan().chunk(ordinal)?.raw_frames,
                    tile_batch,
                    host,
                )?);
        }
        tracing::debug!(
            tiles = tile_batch.maximum_tiles(),
            total_tiles = spatial.tile_count(),
            "compiled uniform video decoder groups"
        );
        Ok(Self {
            source,
            compiled: Arc::new(compiled),
            config,
            limits,
            geometry,
            spatial,
            output_shape,
            input_spec,
            output_bytes,
            tile_batch,
            maximum_readback_bytes,
        })
    }
    pub fn compiled(&self) -> &Arc<CompiledTensorComponent<R>> {
        &self.compiled
    }
    pub fn input_spec(&self) -> &ResolvedTensorSpec {
        &self.input_spec
    }
    /// Full target repacking coexists with one tile group and its two RoPE
    /// tables. The caller separately owns the original packed target copy.
    pub(crate) fn additional_input_host_bytes(
        &self,
        host: &HostMemoryAccount,
    ) -> anyhow::Result<usize> {
        host.backing_bytes(usize::try_from(self.input_spec.minimum_storage_bytes()?)?)?
            .checked_add(super::input_backing_bytes(
                self.compiled.family().program(),
                host,
            )?)
            .context("video decoder input Host bound overflow")
    }
    pub fn output_shape(&self) -> [usize; 5] {
        self.output_shape
    }
    pub fn spatial_tiles(&self) -> usize {
        self.spatial.tile_count()
    }

    /// Optional CPU overlap is priced only at the completed denoising fence.
    /// Refusal keeps the already admitted inline path runnable.
    pub fn try_admit_postprocessing(
        &self,
        parent: &HostMemoryAccount,
    ) -> anyhow::Result<Option<FastH3VideoReadbackAdmission>> {
        Ok(readback::try_admit(
            self.geometry.postprocessing,
            parent,
            self.maximum_readback_bytes,
        )?)
    }

    pub fn initialize<P: Send + 'static>(
        self: Arc<Self>,
        runtime: Arc<R>,
        host: HostMemoryAccount,
        initialization: ComponentInitializationConfig,
        phase: P,
        readbacks: Option<FastH3VideoReadbackAdmission>,
    ) -> Result<
        FastH3VideoDecodeSession<R, P>,
        RetainedComponentInitializationError<R, FastH3VideoDecodePhase<R, P>>,
    > {
        let readback_host = readbacks
            .as_ref()
            .map(|r| (r.scope.account(), r.maximum_readback_bytes));
        let session = self.compiled.initialize_retained_with_pool_policy(
            runtime,
            host.clone(),
            ComponentWeightSource::Stored(&self.source),
            initialization,
            (self.clone(), phase, readbacks),
            ferrum_execution::component::ColdPoolInitializationPolicy::OnExecutionAdmission,
        )?;
        Ok(FastH3VideoDecodeSession {
            plan: self,
            session,
            host,
            readback_host,
        })
    }
}

/// Contiguous RGB24 [requests,frames,height,width,3], retaining its original
/// Output grant. A request slice is zero-copy and retains the shared backing.
pub struct FastH3DecodedVideo {
    bytes: ManagedHostBytes,
    shape: [usize; 5],
}
impl FastH3DecodedVideo {
    pub fn bytes(&self) -> &ManagedHostBytes {
        &self.bytes
    }
    pub fn into_bytes(self) -> ManagedHostBytes {
        self.bytes
    }
    pub fn nthwc_shape(&self) -> [usize; 5] {
        self.shape
    }
    pub fn request_bytes(&self, request: usize) -> anyhow::Result<ManagedHostBytes> {
        ensure!(
            request < self.shape[0],
            "video request outside decoded batch"
        );
        let stride = self.bytes.len() / self.shape[0];
        Ok(self.bytes.slice(request * stride..(request + 1) * stride)?)
    }
}

pub struct FastH3VideoDecodeSession<R: DeviceRuntime, P: Send + 'static> {
    plan: Arc<FastH3VideoDecodePlan<R>>,
    session: RetainedTensorComponentSession<R, FastH3VideoDecodePhase<R, P>>,
    host: HostMemoryAccount,
    readback_host: Option<(HostMemoryAccount, usize)>,
}
impl<R: DeviceRuntime, P: Send + 'static> FastH3VideoDecodeSession<R, P> {
    /// Final normalized DiT target rows only. The original full-depth decoder
    /// executes for every spatial tile and temporal chunk on the same actor.
    pub fn decode(
        &mut self,
        actor: &mut TensorSerialActor<R>,
        run: RunId,
        request: RequestIdentity,
        target_latents: &ManagedHostTensor,
        cancelled: impl Fn() -> bool,
    ) -> Result<FastH3DecodedVideo, FastH3DecodeError<R>> {
        if cancelled() {
            return Err(FastH3DecodeError::Cancelled);
        }
        if target_latents.tensor() != &self.plan.input_spec
            || !self.host.owns(target_latents.bytes())
        {
            return Err(
                anyhow::anyhow!("video target shape/dtype/layout or Host parent differs").into(),
            );
        }
        if let Some((host, capacity)) = &self.readback_host {
            if !host.same_parent(&self.host) || *capacity < self.plan.maximum_readback_bytes {
                return Err(anyhow::anyhow!(
                    "decoder readback admission has another parent or insufficient extent"
                )
                .into());
            }
            self.session
                .session_mut()
                .set_output_host_account(host.clone())
                .map_err(FastH3DecodeError::Execution)?;
        }
        let plan = self.plan.clone();
        let p = plan.as_ref();
        let latent = FastH3VideoLatentLayout::PackedSpatial2x2.managed_ncthw(
            p.geometry.latent_shape,
            target_latents.bytes(),
            &self.host,
            p.geometry.temporal.maximum_buffer_bytes,
        )?;
        let (cosine, sine) = p.config.managed_bf16_frequencies(&self.host, p.limits)?;
        let mut processor = VideoPostprocessor::new(p, &self.host)?;
        let schedule = processor.schedule().clone();
        let mut completed = false;
        if self.readback_host.is_some() {
            match overlap::run(
                |sender| {
                    self.execute_tiles(
                        actor,
                        &run,
                        &request,
                        &latent,
                        &cosine,
                        &sine,
                        &schedule,
                        &cancelled,
                        |tiles| {
                            sender
                                .send(tiles)
                                .map_err(|_| overlap::ProductionError::ConsumerStopped)
                        },
                    )
                },
                |tiles| processor.push(tiles),
            ) {
                Ok(result) => {
                    result?;
                    completed = true;
                }
                Err(error) => tracing::debug!(%error,
                    "video Host worker unavailable; using admitted inline processing"),
            }
        }
        if !completed {
            self.execute_tiles(
                actor,
                &run,
                &request,
                &latent,
                &cosine,
                &sine,
                &schedule,
                &cancelled,
                |tiles| {
                    processor.push(tiles).map_err(|error| {
                        overlap::ProductionError::Original(FastH3DecodeError::Validation(error))
                    })
                },
            )
            .map_err(|error| match error {
                overlap::ProductionError::Original(error) => error,
                overlap::ProductionError::ConsumerStopped => {
                    FastH3DecodeError::Validation(anyhow::anyhow!("inline video consumer stopped"))
                }
            })?;
        }
        if cancelled() {
            return Err(FastH3DecodeError::Cancelled);
        }
        let output = processor.finish()?;
        if cancelled() {
            return Err(FastH3DecodeError::Cancelled);
        }
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_tiles(
        &mut self,
        actor: &mut TensorSerialActor<R>,
        run: &RunId,
        request: &RequestIdentity,
        latent: &ManagedHostBytes,
        cosine: &ManagedHostTensor,
        sine: &ManagedHostTensor,
        schedule: &ferrum_models::fast_h3::video_vae::FastH3VideoDecodePlan,
        cancelled: &impl Fn() -> bool,
        mut consume: impl FnMut(
            DecodedTiles,
        ) -> Result<(), overlap::ProductionError<FastH3DecodeError<R>>>,
    ) -> Result<(), overlap::ProductionError<FastH3DecodeError<R>>> {
        use overlap::ProductionError::Original;
        let p = &self.plan;
        for ordinal in 0..schedule.chunk_count() {
            for first in (0..p.spatial.tile_count()).step_by(p.tile_batch.maximum_tiles()) {
                if cancelled() {
                    return Err(Original(FastH3DecodeError::Cancelled));
                }
                let (input, run) = (|| -> anyhow::Result<_> {
                    let group = p.spatial.tile_group(first, p.tile_batch)?;
                    let input = p.spatial.managed_tile_group_input(
                        schedule,
                        ordinal,
                        group,
                        FastH3VideoLatentUnits::Normalized,
                        latent,
                        &self.host,
                    )?;
                    Ok((input, derived_run(run, "video", ordinal, first)?))
                })()
                .map_err(|error| Original(FastH3DecodeError::Validation(error)))?;
                let mut output = self
                    .session
                    .session_mut()
                    .execute_on_actor(
                        actor,
                        run,
                        request.clone(),
                        vec![input, cosine.clone(), sine.clone()],
                    )
                    .map_err(|error| Original(FastH3DecodeError::Execution(error)))?;
                if output.len() != 1 {
                    return Err(Original(FastH3DecodeError::Validation(anyhow::anyhow!(
                        "video decoder must return one patch tensor"
                    ))));
                }
                // execute_on_actor has fenced and retired the device wave.
                // Only the managed Host output crosses the rendezvous.
                consume(DecodedTiles {
                    ordinal,
                    first,
                    output: output.pop().unwrap(),
                })?;
            }
        }
        Ok(())
    }
    pub fn initialization_receipt(
        &self,
    ) -> &ferrum_execution::component::ComponentInitializationReceipt {
        self.session.session().initialization_receipt()
    }
    pub fn close(
        self,
    ) -> Result<
        (FastH3VideoDecodePhase<R, P>, PlanRuntimeCloseReceipt),
        RetainedComponentCloseError<R, FastH3VideoDecodePhase<R, P>>,
    > {
        self.session.close()
    }
}

fn append_rgb(
    chunk: &FastH3VideoRgbChunk,
    destination: &mut ManagedHostBuffer,
    shape: [usize; 5],
    published: &mut usize,
) -> anyhow::Result<()> {
    let [batch, frames, height, width, channels] = chunk.nthwc_shape();
    let end = published
        .checked_add(frames)
        .context("RGB cursor overflow")?;
    ensure!(
        chunk.output_frame_start() == *published
            && [batch, height, width, channels] == [shape[0], shape[2], shape[3], shape[4]]
            && end <= shape[1],
        "RGB chunk routing differs"
    );
    let frame_bytes = height * width * channels;
    let source_stride = frames * frame_bytes;
    let destination_stride = shape[1] * frame_bytes;
    ensure!(
        destination.len() == batch * destination_stride
            && chunk.bytes().len() == batch * source_stride,
        "RGB exact physical extent differs"
    );
    for b in 0..batch {
        let from = b * source_stride;
        let to = b * destination_stride + *published * frame_bytes;
        destination.as_mut()[to..to + source_stride]
            .copy_from_slice(&chunk.bytes().bytes()[from..from + source_stride]);
    }
    *published = end;
    Ok(())
}
