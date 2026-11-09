//! Metadata-only shape recipes and a session-independent request consumer.
//! The executor still owns actual compilation, original static sharing,
//! dynamic capacity, stream/fence retirement and the one active shape session.
use super::*;
use crate::p_h3_image_vae::decode_tiles::{DecodeShape, PH3ImageVaeDecodeState};

pub struct PreparedPH3ImageVaeDecoderTiles {
    shapes: Vec<PreparedPH3ImageVaeDecoder>,
    tile_batch: PH3ImageVaeTileBatch,
}
impl PH3ImageVaeAdapter {
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_decoder_with_tile_batch(
        &self,
        request_batch: u64,
        width: usize,
        height: usize,
        policy: PH3ImageVaeSpatialPolicy,
        preset: PH3ImageVaeDecoderPreset,
        tile_batch: PH3ImageVaeTileBatch,
    ) -> Result<PreparedPH3ImageVaeDecoderTiles> {
        let plan = PH3ImageVaeTilePlan::new(
            width,
            height,
            usize::try_from(self.spatial_scale())?,
            policy,
        )?;
        let shapes = plan
            .group_shapes(tile_batch)
            .into_iter()
            .map(|count| {
                self.prepare_decoder_shape(request_batch, count, width, height, policy, preset)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(PreparedPH3ImageVaeDecoderTiles { shapes, tile_batch })
    }
}
impl PreparedPH3ImageVaeDecoderTiles {
    /// Largest real group first, then a real tail if present. No padded shape.
    pub fn shapes(&self) -> &[PreparedPH3ImageVaeDecoder] {
        &self.shapes
    }
    pub fn shape(&self, tile_count: usize) -> Result<&PreparedPH3ImageVaeDecoder> {
        self.shapes
            .iter()
            .find(|s| s.tile_count == tile_count)
            .context("undeclared VAE tile group shape")
    }
    pub fn tile_batch(&self) -> PH3ImageVaeTileBatch {
        self.tile_batch
    }
    pub fn tile_count(&self) -> usize {
        self.shapes[0].plan.tile_count()
    }
    pub fn normalized_latent_spec(&self) -> Result<ResolvedTensorSpec> {
        self.shapes[0].normalized_latent_spec()
    }
    pub fn pixel_output_spec(&self) -> Result<ResolvedTensorSpec> {
        self.shapes[0].pixel_output_spec()
    }
    pub fn host_requirements(
        &self,
        host: &HostMemoryAccount,
    ) -> Result<PH3ImageVaeDecoderHostRequirements> {
        host_requirements::decoder_requirements(&self.shapes[0], self.tile_batch, host)
    }
    /// Bind the request once, before selecting the first physical session.
    /// All source parameters are borrowed from this exact prepared source.
    pub fn bind<'a>(
        &'a self,
        normalized: &'a ManagedHostTensor,
        request: RequestIdentity,
        input_host: HostMemoryAccount,
        output_host: HostMemoryAccount,
    ) -> Result<PH3ImageVaeDecoderTilesConsumer<'a>> {
        let p = &self.shapes[0];
        let shapes = self
            .shapes
            .iter()
            .map(|p| {
                DecodeShape::new(
                    &p.config,
                    p.tile_count,
                    p.family.family_id().clone(),
                    p.family.program().fingerprint()?,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(PH3ImageVaeDecoderTilesConsumer {
            inner: PH3ImageVaeDecodeState::new(
                p.plan.clone(),
                p.request_batch,
                self.tile_batch,
                shapes,
                PH3ImageVaeLatentStatistics::new(&p.validated.decoder, &p.validated.source)?,
                normalized,
                request,
                input_host,
                output_host.clone(),
            )?,
            output_host,
        })
    }
}

pub struct PH3ImageVaeDecoderTilesConsumer<'a> {
    inner: PH3ImageVaeDecodeState<'a>,
    output_host: HostMemoryAccount,
}
impl PH3ImageVaeDecoderTilesConsumer<'_> {
    /// A pending Host retry returns its original group, including when earlier
    /// members have committed. Do not switch sessions until that group drains.
    pub fn next_group(&self) -> Result<Option<PH3ImageVaeTileGroup>> {
        self.inner.next_group()
    }
    pub fn completed_tiles(&self) -> usize {
        self.inner.completed_tiles()
    }
    pub fn executed_tiles(&self) -> usize {
        self.inner.executed_tiles()
    }
    pub fn executed_groups(&self) -> usize {
        self.inner.executed_groups()
    }
    pub fn has_pending_output(&self) -> bool {
        self.inner.has_pending_output()
    }
    pub fn execution_failed(&self) -> bool {
        self.inner.execution_failed()
    }
    /// One actual group per call. Core execution errors seal the consumer; the
    /// original error, session and static owner must follow normal retirement.
    /// No new session, weight installation or fallback occurs here.
    pub fn advance_with_session<R: DeviceRuntime>(
        &mut self,
        session: &mut TensorComponentSession<R>,
        actor: &mut TensorSerialActor<R>,
        run: RunId,
    ) -> std::result::Result<bool, PH3ImageVaePipelineError<R>> {
        self.inner
            .advance_with_session(session, actor, run)
            .map_err(|e| match e {
                PH3ImageVaeDecodeError::Preparation(e) => PH3ImageVaePipelineError::Preparation(e),
                PH3ImageVaeDecodeError::Execution(e) => PH3ImageVaePipelineError::Execution(e),
            })
    }
    pub fn finish(self) -> Result<OwnedImagePixels> {
        let (tensor, bytes) = self.inner.finish()?;
        OwnedImagePixels::new(
            &self.output_host,
            tensor,
            bytes,
            ImagePixelEncoding::F32UnitRgbRne,
        )
    }
}
