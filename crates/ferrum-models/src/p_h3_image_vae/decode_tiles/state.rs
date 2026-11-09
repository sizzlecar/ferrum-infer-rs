use super::*;
use crate::p_h3_image_vae::decoder::PH3ImageVaeDecoderExecutionConfig;

pub(super) enum Pending {
    Patches(Vec<TensorExecutionOutput>),
    Pixels(ResolvedTensorSpec, ManagedHostBytes),
}

pub(crate) struct DecodeShape {
    pub tile_count: usize,
    pub layout: PH3ImageVaeImageLayout,
    family: ModelFamilyId,
    fingerprint: String,
    config: PH3ImageVaeDecoderExecutionConfig,
}
impl DecodeShape {
    pub fn new(
        config: &PH3ImageVaeDecoderExecutionConfig,
        tile_count: usize,
        family: ModelFamilyId,
        fingerprint: String,
    ) -> Result<Self> {
        ensure!(tile_count > 0, "decoder shape requires real tiles");
        Ok(Self {
            tile_count,
            layout: PH3ImageVaeImageLayout::new(
                config.architecture().clone(),
                usize::try_from(config.batch())?,
                usize::try_from(config.latent_height())?,
                usize::try_from(config.latent_width())?,
            )?,
            family,
            fingerprint,
            config: config.clone(),
        })
    }
    pub fn validate_session<R: DeviceRuntime>(
        &self,
        session: &TensorComponentSession<R>,
        input_host: &HostMemoryAccount,
    ) -> Result<()> {
        let compiled = session.executable().execution_plan().payload();
        ensure!(
            session.host_account().same_parent(input_host)
                && compiled.family_id() == &self.family
                && compiled.program_fingerprint() == self.fingerprint,
            "session differs from the declared complete Stored VAE decoder shape or Host parent"
        );
        Ok(())
    }
}

/// All pending outputs and the ordered canvas are independent of a physical
/// shape session. An execution error seals this state: the caller must retain
/// the original failure/session until its ordinary retirement proves cleanup.
pub(crate) struct PH3ImageVaeDecodeState<'a> {
    plan: PH3ImageVaeTilePlan,
    request_batch: u64,
    tile_batch: PH3ImageVaeTileBatch,
    shapes: Vec<DecodeShape>,
    statistics: PH3ImageVaeLatentStatistics<'a>,
    normalized: &'a ManagedHostTensor,
    request: RequestIdentity,
    input_host: HostMemoryAccount,
    output_host: HostMemoryAccount,
    pub(super) cosine: ManagedHostTensor,
    pub(super) sine: ManagedHostTensor,
    canvas: PH3ImageVaeCanvas,
    pub(super) pending: Option<Pending>,
    pending_group: Option<PH3ImageVaeTileGroup>,
    next_commit: usize,
    executed_tiles: usize,
    executed_groups: usize,
    execution_failed: bool,
}
impl<'a> PH3ImageVaeDecodeState<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        plan: PH3ImageVaeTilePlan,
        request_batch: u64,
        tile_batch: PH3ImageVaeTileBatch,
        shapes: Vec<DecodeShape>,
        statistics: PH3ImageVaeLatentStatistics<'a>,
        normalized: &'a ManagedHostTensor,
        request: RequestIdentity,
        input_host: HostMemoryAccount,
        output_host: HostMemoryAccount,
    ) -> Result<Self> {
        ensure!(
            input_host.same_parent(&output_host)
                && input_host.owns(normalized.bytes())
                && normalized.bytes().class() == HostMemoryClass::InputPreprocess,
            "VAE decoder inputs and outputs must use the existing Host parent"
        );
        let expected = plan.group_shapes(tile_batch);
        ensure!(
            shapes.iter().map(|s| s.tile_count).collect::<Vec<_>>() == expected,
            "decoder recipes must describe exactly the real row groups and tail"
        );
        let first = &shapes[0];
        let architecture = first.config.architecture();
        let scale = usize::try_from(architecture.patch_size())?;
        ensure!(
            architecture.out_channels() == 3 && plan.spatial_scale() == scale,
            "decoder must preserve RGB architecture and spatial compression"
        );
        let tile = plan.tile(0)?;
        for shape in &shapes {
            ensure!(
                shape.config.architecture() == architecture
                    && shape.config.batch()
                        == request_batch
                            .checked_mul(u64::try_from(shape.tile_count)?)
                            .context("decoder physical batch overflows")?
                    && usize::try_from(shape.config.latent_height())?.checked_mul(scale)
                        == Some(tile.height)
                    && usize::try_from(shape.config.latent_width())?.checked_mul(scale)
                        == Some(tile.width),
                "decoder physical batch or tile shape differs from its semantic request batch"
            );
        }
        let [height, width] = plan.dimensions();
        let whole = PH3ImageVaeImageLayout::new(
            architecture.clone(),
            usize::try_from(request_batch)?,
            height / scale,
            width / scale,
        )?;
        ensure!(
            normalized.tensor() == &whole.latent_spec()?,
            "normalized sampler shape differs from the semantic request canvas"
        );
        // Every batch lane has identical local coordinates. Retain the largest
        // two actual backings once; a smaller real tail uses a checked prefix
        // view, still charged at the full original allocation size.
        let (cosine, sine) = first.layout.managed_rotary(
            &input_host,
            ProgramValueId::new(PH3ImageVaeDecoderFamily::COSINE_ID)?,
            ProgramValueId::new(PH3ImageVaeDecoderFamily::SINE_ID)?,
        )?;
        let canvas = PH3ImageVaeCanvas::new(
            plan.clone(),
            output_host.clone(),
            usize::try_from(request_batch)?,
            3,
        )?;
        Ok(Self {
            plan,
            request_batch,
            tile_batch,
            shapes,
            statistics,
            normalized,
            request,
            input_host,
            output_host,
            cosine,
            sine,
            canvas,
            pending: None,
            pending_group: None,
            next_commit: 0,
            executed_tiles: 0,
            executed_groups: 0,
            execution_failed: false,
        })
    }
    pub fn next_group(&self) -> Result<Option<PH3ImageVaeTileGroup>> {
        ensure!(
            !self.execution_failed,
            "decoder execution failed; original session/failure retirement is required"
        );
        if let Some(group) = self.pending_group {
            return Ok(Some(group));
        }
        if self.canvas.completed_tiles() == self.plan.tile_count() {
            return Ok(None);
        }
        Ok(Some(
            self.plan
                .group(self.canvas.completed_tiles(), self.tile_batch)?,
        ))
    }
    fn shape(&self, count: usize) -> Result<&DecodeShape> {
        self.shapes
            .iter()
            .find(|s| s.tile_count == count)
            .context("undeclared decoder group shape")
    }
    pub fn advance_with_session<R: DeviceRuntime>(
        &mut self,
        session: &mut TensorComponentSession<R>,
        actor: &mut TensorSerialActor<R>,
        run: RunId,
    ) -> std::result::Result<bool, PH3ImageVaeDecodeError<R>> {
        let group = self
            .next_group()
            .map_err(PH3ImageVaeDecodeError::Preparation)?;
        let Some(group) = group else {
            return Ok(true);
        };
        self.shape(group.tile_count())
            .and_then(|s| s.validate_session(session, &self.input_host))
            .map_err(PH3ImageVaeDecodeError::Preparation)?;
        if self.pending.is_none() {
            let inputs = self
                .inputs(group)
                .map_err(PH3ImageVaeDecodeError::Preparation)?;
            // Set before entering execution: even a panic cannot reopen this
            // consumer while native submission/retirement authority is unknown.
            self.execution_failed = true;
            let outputs = session
                .execute_on_actor(actor, run, self.request.clone(), inputs)
                .map_err(PH3ImageVaeDecodeError::Execution)?;
            self.execution_failed = false;
            self.pending = Some(Pending::Patches(outputs));
            self.pending_group = Some(group);
            self.next_commit = 0;
            self.executed_tiles += group.tile_count();
            self.executed_groups += 1;
        }
        self.commit_pending()
            .map_err(PH3ImageVaeDecodeError::Preparation)
    }
    fn inputs(&self, group: PH3ImageVaeTileGroup) -> Result<Vec<ManagedHostTensor>> {
        let rows = self.plan.managed_tile_rows(
            &self.input_host,
            self.normalized,
            &self.statistics,
            group.first_tile(),
            group.tile_count(),
            ProgramValueId::new(PH3ImageVaeDecoderFamily::INPUT_ID)?,
        )?;
        let frequencies = self.shape(group.tile_count())?.layout.frequency_spec()?;
        let bytes = usize::try_from(frequencies.minimum_storage_bytes()?)?;
        let prefix = |owner: &ManagedHostTensor| -> Result<ManagedHostTensor> {
            Ok(ManagedHostTensor::new(
                owner.value_id().clone(),
                frequencies.clone(),
                owner.bytes().slice(0..bytes)?,
            )?)
        };
        Ok(vec![rows, prefix(&self.cosine)?, prefix(&self.sine)?])
    }
    /// Host-only continuation of a completed native group. This is shared by
    /// capacity-one and grouped execution; it never reruns the decoder.
    fn commit_pending(&mut self) -> Result<bool> {
        let group = self
            .pending_group
            .context("decoder has no completed group")?;
        if let Some(Pending::Patches(outputs)) = &self.pending {
            let layout = &self.shape(group.tile_count())?.layout;
            ensure!(
                outputs.len() == 1,
                "decoder must return exactly one patch tensor"
            );
            ensure!(
                outputs[0].value_id().as_str() == PH3ImageVaeDecoderFamily::OUTPUT_ID
                    && outputs[0].tensor() == &layout.patch_output_spec()?
                    && self.output_host.owns(outputs[0].bytes()),
                "decoder patch output or Host owner differs"
            );
            let (tensor, bytes) =
                layout.unpatchify(&self.output_host, outputs[0].tensor(), outputs[0].bytes())?;
            self.pending = Some(Pending::Pixels(tensor, bytes));
        }
        while self.next_commit < group.tile_count() {
            let index = group.first_tile() + self.next_commit;
            let tile = self.plan.tile(index)?;
            let tensor = ResolvedTensorSpec::new(
                vec![self.request_batch, 3, tile.height as u64, tile.width as u64],
                ElementType::F32,
                ResolvedTensorLayout::Contiguous,
            )?;
            let bytes_per_tile = usize::try_from(tensor.minimum_storage_bytes()?)?;
            let Some(Pending::Pixels(group_tensor, owner)) = &self.pending else {
                anyhow::bail!("completed decoder group lacks pixels");
            };
            ensure!(
                group_tensor == &self.shape(group.tile_count())?.layout.pixel_spec()?,
                "pending group pixels differ from physical batch"
            );
            let start = self
                .next_commit
                .checked_mul(bytes_per_tile)
                .context("tile view offset overflows")?;
            let slice = owner.slice(
                start
                    ..start
                        .checked_add(bytes_per_tile)
                        .context("tile view extent overflows")?,
            )?;
            self.canvas.push(index, &tensor, &slice)?;
            // Only advance after the unchanged transactional Canvas commits.
            // A later failure retains the full group owner and this cursor.
            self.next_commit += 1;
        }
        self.pending = None;
        self.pending_group = None;
        Ok(self.canvas.completed_tiles() == self.plan.tile_count())
    }
    pub fn completed_tiles(&self) -> usize {
        self.canvas.completed_tiles()
    }
    pub fn executed_tiles(&self) -> usize {
        self.executed_tiles
    }
    pub fn executed_groups(&self) -> usize {
        self.executed_groups
    }
    pub fn has_pending_output(&self) -> bool {
        self.pending.is_some()
    }
    pub fn execution_failed(&self) -> bool {
        self.execution_failed
    }
    pub fn finish(self) -> Result<(ResolvedTensorSpec, ManagedHostBytes)> {
        ensure!(
            !self.execution_failed && self.pending.is_none(),
            "cannot publish unresolved decoder work"
        );
        self.canvas.finish(PH3ImageVaeCanvasDomain::UnitRgb)
    }
}

#[cfg(test)]
mod tests;
