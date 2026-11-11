//! Physical Host payload requirements of the existing conditioning controller.
//! WeightBacking, static-upload TransferStaging, allocator metadata and parser
//! scratch remain separate responsibilities of the common execution owner.
use super::*;

/// The actual controller's four successive payload ownership boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PH3ImageTextConditioningHostPhase {
    EncoderReadback,
    StandardizeTaps,
    Project,
    Finish,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PH3ImageTextConditioningHostPhaseBudget {
    pub phase: PH3ImageTextConditioningHostPhase,
    pub input_bytes: usize,
    pub output_bytes: usize,
    pub total_bytes: usize,
}

/// Geometry-specific metadata only: this value neither retains an account nor
/// grants capacity. Every distinct backing includes the supplied allocator's
/// alignment separately. Existing caller-owned payloads must be added by the
/// caller; these are the controller's newly allocated Input/Output payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PH3ImageTextConditioningHostRequirements {
    pub phases: [PH3ImageTextConditioningHostPhaseBudget; 4],
    pub input_peak_bytes: usize,
    pub output_peak_bytes: usize,
    /// Maximum simultaneously live Input + Output payloads in one phase.
    pub live_peak_bytes: usize,
    /// A single fixed scope reserves both independent class peaks. Its actual
    /// total credit is their sum, which may exceed `live_peak_bytes`.
    pub single_scope_bytes: usize,
    /// The final conditioning owns this Input charge after the controller
    /// returns. Closing a scope returns unused credit, not this live charge.
    pub retained_conditioning_bytes: usize,
}
impl PH3ImageTextConditioningHostRequirements {
    /// Already-physical class credit, without reapplying aggregate padding.
    /// Weight/transfer classes are deliberately managed by the common stage.
    pub fn scope_claims(self) -> [(HostMemoryClass, usize); 2] {
        [
            (HostMemoryClass::InputPreprocess, self.input_peak_bytes),
            (HostMemoryClass::Output, self.output_peak_bytes),
        ]
    }
}

impl PH3ImageTextConditioningConfig {
    pub fn host_requirements(
        &self,
        host: &HostMemoryAccount,
        precision: PH3ImageConditioningPrecision,
    ) -> Result<PH3ImageTextConditioningHostRequirements> {
        let text = self.encoder().text();
        let rows = text.layout().rows();
        // managed_inputs creates three independent allocations: IDs, cosine
        // and sine. Use the same typed specs rather than a total-byte estimate.
        let encoder_input =
            self.input_tensor_specs()?
                .values()
                .try_fold(0usize, |sum, spec| -> Result<_> {
                    let backing = host.backing_bytes(usize::try_from(spec.byte_len()?)?)?;
                    sum.checked_add(backing)
                        .context("conditioning encoder Input backing overflows")
                })?;
        let tap_logical = bytes(&[rows, text.architecture().hidden_size(), 4])?;
        // execute_bound reserves all five readbacks before device work. These
        // are separate allocations even when their device values share owners.
        let taps = host
            .backing_bytes(tap_logical)?
            .checked_mul(text.hidden_state_indices().len())
            .context("conditioning tap Output backing overflows")?;
        let connector = self.connector_config()?;
        let normalized = host.backing_bytes(bytes(&[rows, connector.input_dim(), 4])?)?;
        let projected = host.backing_bytes(bytes(&[rows, connector.out_channels(), 4])?)?;
        let element_bytes = match precision {
            PH3ImageConditioningPrecision::F32 => 4,
            PH3ImageConditioningPrecision::Bf16 => 2,
        };
        let conditioning =
            host.backing_bytes(bytes(&[rows, connector.out_channels(), element_bytes])?)?;
        let phases = [
            phase(
                PH3ImageTextConditioningHostPhase::EncoderReadback,
                encoder_input,
                taps,
            )?,
            // Encoder input owners retire before execute returns. The five
            // borrowed readbacks stay live while normalized Input is created.
            phase(
                PH3ImageTextConditioningHostPhase::StandardizeTaps,
                normalized,
                taps,
            )?,
            // Controller explicitly drops all taps before projection starts;
            // projection readback is admitted while normalized Input is live.
            phase(
                PH3ImageTextConditioningHostPhase::Project,
                normalized,
                projected,
            )?,
            // Normalized Input retires on execute return. Final sink/affine
            // output is a new Input owner while projection Output stays live.
            phase(
                PH3ImageTextConditioningHostPhase::Finish,
                conditioning,
                projected,
            )?,
        ];
        let input_peak_bytes = phases.iter().map(|p| p.input_bytes).max().unwrap();
        let output_peak_bytes = phases.iter().map(|p| p.output_bytes).max().unwrap();
        Ok(PH3ImageTextConditioningHostRequirements {
            live_peak_bytes: phases.iter().map(|p| p.total_bytes).max().unwrap(),
            single_scope_bytes: input_peak_bytes
                .checked_add(output_peak_bytes)
                .context("conditioning fixed-scope backing overflows")?,
            phases,
            input_peak_bytes,
            output_peak_bytes,
            retained_conditioning_bytes: conditioning,
        })
    }
}

fn bytes(factors: &[u64]) -> Result<usize> {
    let logical = factors.iter().try_fold(1u64, |n, &factor| {
        n.checked_mul(factor)
            .context("conditioning logical payload extent overflows")
    })?;
    Ok(usize::try_from(logical)?)
}
fn phase(
    phase: PH3ImageTextConditioningHostPhase,
    input_bytes: usize,
    output_bytes: usize,
) -> Result<PH3ImageTextConditioningHostPhaseBudget> {
    Ok(PH3ImageTextConditioningHostPhaseBudget {
        phase,
        input_bytes,
        output_bytes,
        total_bytes: input_bytes
            .checked_add(output_bytes)
            .context("conditioning phase backing overflows")?,
    })
}

#[cfg(test)]
mod tests;
