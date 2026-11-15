//! Request-local CPU state. It owns only already fenced Host readbacks and
//! their original Output credit; no runtime, actor or native session moves here.
use super::*;
use ferrum_execution::TensorExecutionOutput;
use ferrum_models::fast_h3::video_vae::FastH3VideoSpatialCanvas;

pub(super) struct DecodedTiles {
    pub ordinal: usize,
    pub first: usize,
    pub output: TensorExecutionOutput,
}

pub(super) struct VideoPostprocessor<'a> {
    config: &'a FastH3VideoClipConfig,
    spatial: &'a FastH3VideoSpatialPlan,
    geometry: FastH3VideoDecodeGeometry,
    limits: FastH3VideoClipLimits,
    tile_batch: SpatialTileBatch,
    host: HostMemoryAccount,
    shape: [usize; 5],
    temporal: FastH3VideoTemporalAssembler,
    canvas: Option<FastH3VideoSpatialCanvas>,
    ordinal: usize,
    published: usize,
    rgb: ManagedHostBuffer,
}

impl<'a> VideoPostprocessor<'a> {
    pub fn new<R: DeviceRuntime>(
        plan: &'a FastH3VideoDecodePlan<R>,
        host: &HostMemoryAccount,
    ) -> anyhow::Result<Self> {
        let [batch, _, time, _, _] = plan.geometry.latent_shape;
        let [_, _, height, width, _] = plan.output_shape;
        let temporal = FastH3VideoTemporalAssembler::new(
            plan.config.architecture(),
            time,
            batch,
            height,
            width,
            plan.geometry.temporal,
        )?;
        // Both allocations precede the first native submission, in either mode.
        let rgb = host.allocate_zeroed(HostMemoryClass::Output, plan.output_bytes)?;
        let canvas = plan
            .spatial
            .begin_canvas(temporal.plan().chunk(0)?.raw_frames, host)?;
        Ok(Self {
            config: &plan.config,
            spatial: &plan.spatial,
            geometry: plan.geometry,
            limits: plan.limits,
            tile_batch: plan.tile_batch,
            host: host.clone(),
            shape: plan.output_shape,
            temporal,
            canvas: Some(canvas),
            ordinal: 0,
            published: 0,
            rgb,
        })
    }

    pub fn schedule(&self) -> &ferrum_models::fast_h3::video_vae::FastH3VideoDecodePlan {
        self.temporal.plan()
    }

    pub fn push(&mut self, tiles: DecodedTiles) -> anyhow::Result<()> {
        ensure!(
            tiles.ordinal == self.ordinal,
            "out-of-order temporal readback"
        );
        let canvas = self
            .canvas
            .as_mut()
            .context("readback after final spatial chunk")?;
        ensure!(
            tiles.first == canvas.completed_tiles(),
            "out-of-order grouped readback"
        );
        let group = self.spatial.tile_group(tiles.first, self.tile_batch)?;
        let batch = self.geometry.latent_shape[0];
        for offset in 0..group.tile_count() {
            let raw = self.config.unpack_execution_batch_range(
                &tiles.output,
                offset * batch..(offset + 1) * batch,
                &self.host,
                self.limits,
            )?;
            canvas.push(tiles.first + offset, &raw)?;
        }
        // Temporal/RGB staging never retains this grouped readback. The next
        // producer readback has separate, explicitly reserved Output credit.
        drop(tiles.output);
        if canvas.completed_tiles() == self.spatial.tile_count() {
            let raw = self.canvas.take().unwrap().finish()?;
            if let Some(chunk) = self
                .temporal
                .prepare_chunk(self.ordinal, &raw, &self.host)?
                .commit()
            {
                let chunk =
                    chunk.managed_rgb8(&self.host, self.geometry.temporal.maximum_buffer_bytes)?;
                append_rgb(&chunk, &mut self.rgb, self.shape, &mut self.published)?;
            }
            drop(raw);
            self.ordinal += 1;
            if self.ordinal < self.temporal.plan().chunk_count() {
                self.canvas = Some(self.spatial.begin_canvas(
                    self.temporal.plan().chunk(self.ordinal)?.raw_frames,
                    &self.host,
                )?);
            }
        }
        Ok(())
    }

    pub fn finish(mut self) -> anyhow::Result<FastH3DecodedVideo> {
        ensure!(
            self.ordinal == self.temporal.plan().chunk_count() && self.canvas.is_none(),
            "decoder readbacks ended before the declared temporal schedule"
        );
        if let Some(chunk) = self.temporal.prepare_finish(&self.host)?.commit() {
            let chunk =
                chunk.managed_rgb8(&self.host, self.geometry.temporal.maximum_buffer_bytes)?;
            append_rgb(&chunk, &mut self.rgb, self.shape, &mut self.published)?;
        }
        ensure!(
            self.temporal.is_finished() && self.published == self.shape[1],
            "decoder did not finalize its exact declared frame count"
        );
        Ok(FastH3DecodedVideo {
            bytes: self.rgb.freeze(),
            shape: self.shape,
        })
    }
}
