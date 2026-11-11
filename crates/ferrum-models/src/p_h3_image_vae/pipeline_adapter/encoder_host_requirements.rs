//! Physical Host payload requirements of the existing raw-neighbour encoder
//! consumer. Original pixels/noise and static installation stay caller-owned.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PH3ImageVaeEncoderHostPhase {
    ExecuteReadback,
    Stitch,
    Posterior,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PH3ImageVaeEncoderHostPhaseBudget {
    pub phase: PH3ImageVaeEncoderHostPhase,
    /// Posterior runs after all tiles have committed and has no tile index.
    pub tile_index: Option<usize>,
    pub input_bytes: usize,
    pub output_bytes: usize,
    pub total_bytes: usize,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PH3ImageVaeEncoderHostRequirements {
    /// Only expected views: the caller retains each actual original backing,
    /// including any larger parent owner retained by a slice.
    pub borrowed_pixels: ResolvedTensorSpec,
    pub borrowed_posterior_noise: ResolvedTensorSpec,
    pub tile_input_bytes: usize,
    pub posterior_input_bytes: usize,
    /// Full moments persist across tiles and a failed posterior admission.
    pub moment_canvas_output_bytes: usize,
    pub tile_readback_output_bytes: usize,
    /// At most one raw tile per column remains cached without payload copies.
    pub raw_neighbour_output_peak_bytes: usize,
    /// Additional Output above the full canvas, including exactly the raw
    /// neighbours/current readback/working tile which actually coexist.
    pub work_output_peak_bytes: usize,
    pub phases: [PH3ImageVaeEncoderHostPhaseBudget; 3],
    pub input_peak_bytes: usize,
    pub output_peak_bytes: usize,
    pub live_peak_bytes: usize,
    pub single_scope_bytes: usize,
    pub retained_latent_input_bytes: usize,
}
impl PH3ImageVaeEncoderHostRequirements {
    /// Already-physical new consumer credit. Do not add borrowed RGB/noise or
    /// charge the same moment canvas a second time in another work grant.
    /// WeightBacking/TransferStaging remain common-stage responsibilities.
    pub fn scope_claims(&self) -> [(HostMemoryClass, usize); 2] {
        [
            (HostMemoryClass::InputPreprocess, self.input_peak_bytes),
            (HostMemoryClass::Output, self.output_peak_bytes),
        ]
    }
}
impl PreparedPH3ImageVaeEncoder {
    pub fn host_requirements(
        &self,
        host: &HostMemoryAccount,
    ) -> Result<PH3ImageVaeEncoderHostRequirements> {
        let tile_input_bytes = backing(host, &self.tile_input_spec()?)?;
        let borrowed_posterior_noise = self.posterior_noise_spec()?;
        let posterior_input_bytes = backing(host, &self.normalized_latent_spec()?)?;
        let [height, width] = self.plan.dimensions();
        let moment_channels = self
            .validated
            .encoder
            .latent_channels()
            .checked_mul(2)
            .context("VAE encoder moment width overflows")?;
        let full = spec(&[
            self.config.batch(),
            moment_channels,
            (height / self.plan.spatial_scale()) as u64,
            (width / self.plan.spatial_scale()) as u64,
        ])?;
        let moment_canvas_output_bytes = backing(host, &full)?;
        let output = self.config.output_tensor_spec()?;
        let output_id = ProgramValueId::new(PH3ImageVaeEncoderFamily::OUTPUT_ID)?;
        ensure!(
            self.options.tensor_specs().get(&output_id) == Some(&output),
            "prepared encoder output differs from its actual tile moments"
        );
        let tile_readback_output_bytes = backing(host, &resolved(output)?)?;
        let mut columns = 0usize;
        for index in 0..self.plan.tile_count() {
            if self.plan.tile(index)?.y != 0 {
                break;
            }
            columns = columns
                .checked_add(1)
                .context("VAE encoder tile columns overflow")?;
        }
        let raw_neighbour_output_peak_bytes = tile_readback_output_bytes
            .checked_mul(columns)
            .context("VAE encoder raw-neighbour backing overflows")?;
        let mut peaks: [Option<PH3ImageVaeEncoderHostPhaseBudget>; 2] = [None; 2];
        for index in 0..self.plan.tile_count() {
            let tile = self.plan.tile(index)?;
            // First-row columns fill raw_row one by one. After that row, all
            // columns retain their raw source until replaced or finish drops
            // the row. The current readback does not alias its upper source.
            let before = if tile.y == 0 { index } else { columns };
            let readbacks = before
                .checked_add(1)
                .and_then(|n| n.checked_mul(tile_readback_output_bytes))
                .context("VAE encoder readback backing overflows")?;
            let working = before
                .checked_add(2)
                .and_then(|n| n.checked_mul(tile_readback_output_bytes))
                .context("VAE encoder stitch backing overflows")?;
            let phases = [
                encoder_phase(
                    PH3ImageVaeEncoderHostPhase::ExecuteReadback,
                    Some(index),
                    tile_input_bytes,
                    checked_sum(&[moment_canvas_output_bytes, readbacks])?,
                )?,
                // The actual Input tile retires before Canvas::push creates
                // its working allocation and retains the original readback.
                encoder_phase(
                    PH3ImageVaeEncoderHostPhase::Stitch,
                    Some(index),
                    0,
                    checked_sum(&[moment_canvas_output_bytes, working])?,
                )?,
            ];
            for (peak, phase) in peaks.iter_mut().zip(phases) {
                if peak
                    .as_ref()
                    .is_none_or(|old| phase.total_bytes > old.total_bytes)
                {
                    *peak = Some(phase);
                }
            }
        }
        let [execute, stitch] = peaks.map(|p| p.expect("validated encoder plan contains a tile"));
        let phases = [
            execute,
            stitch,
            // canvas.finish consumes raw_row before posterior_from_noise
            // allocates the new Input. Failure retains only full moments.
            encoder_phase(
                PH3ImageVaeEncoderHostPhase::Posterior,
                None,
                posterior_input_bytes,
                moment_canvas_output_bytes,
            )?,
        ];
        let input_peak_bytes = tile_input_bytes.max(posterior_input_bytes);
        let output_peak_bytes = phases.iter().map(|p| p.output_bytes).max().unwrap();
        Ok(PH3ImageVaeEncoderHostRequirements {
            borrowed_pixels: self.pixel_input_spec()?,
            borrowed_posterior_noise,
            tile_input_bytes,
            posterior_input_bytes,
            moment_canvas_output_bytes,
            tile_readback_output_bytes,
            raw_neighbour_output_peak_bytes,
            work_output_peak_bytes: output_peak_bytes
                .checked_sub(moment_canvas_output_bytes)
                .context("VAE encoder canvas backing exceeds its peak")?,
            live_peak_bytes: phases.iter().map(|p| p.total_bytes).max().unwrap(),
            single_scope_bytes: checked_sum(&[input_peak_bytes, output_peak_bytes])?,
            retained_latent_input_bytes: posterior_input_bytes,
            input_peak_bytes,
            output_peak_bytes,
            phases,
        })
    }
}
fn encoder_phase(
    phase: PH3ImageVaeEncoderHostPhase,
    tile_index: Option<usize>,
    input_bytes: usize,
    output_bytes: usize,
) -> Result<PH3ImageVaeEncoderHostPhaseBudget> {
    Ok(PH3ImageVaeEncoderHostPhaseBudget {
        phase,
        tile_index,
        input_bytes,
        output_bytes,
        total_bytes: checked_sum(&[input_bytes, output_bytes])?,
    })
}
