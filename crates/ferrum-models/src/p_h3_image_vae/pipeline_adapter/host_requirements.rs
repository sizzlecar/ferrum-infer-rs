//! Physical payload peaks of the existing tiled consumers, without another
//! Host account, capacity ledger, reservation or model execution.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PH3ImageVaeDecoderHostPhase {
    ExecuteReadback,
    Unpatchify,
    Stitch,
}
/// The largest live total of this phase over the declared tile traversal.
/// `tile_index` identifies an actual tile attaining that peak.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PH3ImageVaeDecoderHostPhaseBudget {
    pub phase: PH3ImageVaeDecoderHostPhase,
    pub tile_index: usize,
    pub input_bytes: usize,
    pub output_bytes: usize,
    pub total_bytes: usize,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PH3ImageVaeDecoderHostRequirements {
    /// Caller-owned normalized sample. Its actual owner may exceed this view's
    /// logical shape; no existing charge is repeated in the new grant below.
    pub borrowed_normalized_latents: ResolvedTensorSpec,
    /// Two separately allocated frequency backings, cached until finish/drop.
    pub cached_rotary_input_bytes: usize,
    /// Largest actual gathered/denormalized group-row allocation during device execution.
    pub work_input_bytes: usize,
    /// Full canvas is allocated at bind and returned as the final pixel owner.
    pub canvas_output_bytes: usize,
    /// Peak additional Output over the canvas: exact old/new seam coexistence,
    /// patch readback or pixel/working tiles at their real phase boundaries.
    pub work_output_peak_bytes: usize,
    pub phases: [PH3ImageVaeDecoderHostPhaseBudget; 3],
    pub input_peak_bytes: usize,
    pub output_peak_bytes: usize,
    pub live_peak_bytes: usize,
    /// One fixed consumer scope reserves the independent class peaks, whose
    /// sum may exceed the actual maximum live phase total.
    pub single_scope_bytes: usize,
    pub retained_pixel_output_bytes: usize,
}
impl PH3ImageVaeDecoderHostRequirements {
    /// The current consumer uses one Input view and one Output view for its
    /// full lifetime; canvas/rotary are included in these combined claims.
    /// Static backing/staging and borrowed caller payloads are separate.
    pub fn scope_claims(&self) -> [(HostMemoryClass, usize); 2] {
        [
            (HostMemoryClass::InputPreprocess, self.input_peak_bytes),
            (HostMemoryClass::Output, self.output_peak_bytes),
        ]
    }
}
impl PreparedPH3ImageVaeDecoder {
    pub fn host_requirements(
        &self,
        host: &HostMemoryAccount,
    ) -> Result<PH3ImageVaeDecoderHostRequirements> {
        ensure!(
            self.tile_count == 1,
            "use the grouped decoder requirements for a spatial batch recipe"
        );
        decoder_requirements(self, PH3ImageVaeTileBatch::default(), host)
    }
}

pub(super) fn decoder_requirements(
    prepared: &PreparedPH3ImageVaeDecoder,
    tile_batch: PH3ImageVaeTileBatch,
    host: &HostMemoryAccount,
) -> Result<PH3ImageVaeDecoderHostRequirements> {
    let layout = |count: usize| {
        PH3ImageVaeImageLayout::new(
            prepared.validated.decoder.clone(),
            usize::try_from(prepared.request_batch)?
                .checked_mul(count)
                .context("physical decoder batch overflows")?,
            usize::try_from(prepared.config.latent_height())?,
            usize::try_from(prepared.config.latent_width())?,
        )
    };
    let largest = prepared.plan.group_shapes(tile_batch)[0];
    let maximum = layout(largest)?;
    let cached_rotary_input_bytes = backing(host, &maximum.frequency_spec()?)?
        .checked_mul(2)
        .context("VAE decoder rotary backing overflows")?;
    let work_input_bytes = backing(host, &maximum.row_spec()?)?;
    let canvas_output_bytes = backing(host, &prepared.pixel_output_spec()?)?;
    let [_, canvas_width] = prepared.plan.dimensions();
    let batch = prepared.request_batch;
    let pixels = |height: usize, width: usize| -> Result<usize> {
        if height == 0 || width == 0 {
            return Ok(0);
        }
        backing(host, &spec(&[batch, 3, height as u64, width as u64])?)
    };
    let mut peaks: [Option<PH3ImageVaeDecoderHostPhaseBudget>; 3] = [None; 3];
    let mut input_peak_bytes = 0;
    let mut output_peak_bytes = 0;
    let mut observe = |slot: usize, phase: PH3ImageVaeDecoderHostPhaseBudget| {
        input_peak_bytes = input_peak_bytes.max(phase.input_bytes);
        output_peak_bytes = output_peak_bytes.max(phase.output_bytes);
        if peaks[slot].is_none_or(|old| phase.total_bytes > old.total_bytes) {
            peaks[slot] = Some(phase);
        }
    };
    let mut first = 0;
    while first < prepared.plan.tile_count() {
        let group = prepared.plan.group(first, tile_batch)?;
        let actual = layout(group.tile_count())?;
        let readback = backing(host, &actual.patch_output_spec()?)?;
        let group_pixels = backing(host, &actual.pixel_spec()?)?;
        for index in first..first + group.tile_count() {
            let tile = prepared.plan.tile(index)?;
            let tile_pixels = pixels(tile.height, tile.width)?;
            let upper = pixels(tile.overlap_top, canvas_width)?;
            let left = pixels(tile.height, tile.overlap_left)?;
            let building = if tile.x != 0 {
                pixels(tile.trim_bottom, canvas_width)?
            } else {
                0
            };
            let base = checked_sum(&[canvas_output_bytes, upper, left, building])?;
            if index == first {
                observe(
                    0,
                    decoder_phase(
                        PH3ImageVaeDecoderHostPhase::ExecuteReadback,
                        index,
                        checked_sum(&[
                            cached_rotary_input_bytes,
                            backing(host, &actual.row_spec()?)?,
                        ])?,
                        checked_sum(&[base, readback])?,
                    )?,
                );
                observe(
                    1,
                    decoder_phase(
                        PH3ImageVaeDecoderHostPhase::Unpatchify,
                        index,
                        cached_rotary_input_bytes,
                        checked_sum(&[base, readback, group_pixels])?,
                    )?,
                );
            }
            // Every tile view retains the full group's original pixel backing.
            // Account old/new seams at each cursor, including a retry after
            // earlier group members have committed. The patch owner is gone.
            let new_left = pixels(tile.height, tile.trim_right)?;
            let new_strip = if tile.x == 0 {
                pixels(tile.trim_bottom, canvas_width)?
            } else {
                0
            };
            observe(
                2,
                decoder_phase(
                    PH3ImageVaeDecoderHostPhase::Stitch,
                    index,
                    cached_rotary_input_bytes,
                    checked_sum(&[base, group_pixels, tile_pixels, new_left, new_strip])?,
                )?,
            );
        }
        first += group.tile_count();
    }
    let phases = peaks.map(|p| p.expect("validated VAE plan contains a tile"));
    Ok(PH3ImageVaeDecoderHostRequirements {
        borrowed_normalized_latents: prepared.normalized_latent_spec()?,
        cached_rotary_input_bytes,
        work_input_bytes,
        canvas_output_bytes,
        work_output_peak_bytes: output_peak_bytes
            .checked_sub(canvas_output_bytes)
            .context("VAE decoder canvas backing exceeds its peak")?,
        input_peak_bytes,
        output_peak_bytes,
        live_peak_bytes: phases.iter().map(|p| p.total_bytes).max().unwrap(),
        single_scope_bytes: checked_sum(&[input_peak_bytes, output_peak_bytes])?,
        retained_pixel_output_bytes: canvas_output_bytes,
        phases,
    })
}

fn decoder_phase(
    phase: PH3ImageVaeDecoderHostPhase,
    tile_index: usize,
    input_bytes: usize,
    output_bytes: usize,
) -> Result<PH3ImageVaeDecoderHostPhaseBudget> {
    Ok(PH3ImageVaeDecoderHostPhaseBudget {
        phase,
        tile_index,
        input_bytes,
        output_bytes,
        total_bytes: checked_sum(&[input_bytes, output_bytes])?,
    })
}
