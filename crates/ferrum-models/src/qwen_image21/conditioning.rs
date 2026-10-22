//! Managed storage for the already-defined transformer conditioning adapters.
//!
//! Arithmetic, positions, early BF16 timestep casts, and the causal shared t0
//! remain in `transformer::conditioning`. This module only resolves typed
//! ports, admits InputPreprocess backing in an injected Host account, invokes
//! those adapters, and publishes little-endian immutable bytes. It creates no
//! backend, source file, hash receipt, or input-sized temporary payload.

use super::{
    QwenImage21ModelTimestep, QwenImage21Timestep256, QwenImage21TransformerConfig,
    QwenImage21TransformerFamily as Family,
};
use anyhow::{ensure, Context, Result};
use ferrum_runtime::{
    ElementType, HostMemoryAccount, HostMemoryClass, ManagedHostTensor, ProgramValueId,
    ResolvedTensorLayout, ResolvedTensorSpec, STRUCTURED_ATTENTION_BF16_MAX_BATCH,
};

impl QwenImage21TransformerConfig {
    /// One F32 `[S,D/2,2]` frequency table shared across batches and heads.
    /// Padded text positions and centered image coordinates are taken directly
    /// from this config's checked layout. The numerical profile is unchanged:
    /// `QwenImage21RotaryFrequencies::NUMERICAL_PROFILE_ID`.
    pub fn managed_rotary_frequencies(
        &self,
        host: &HostMemoryAccount,
    ) -> Result<ManagedHostTensor> {
        let elements = self.rotary_frequencies()?.output_elements(self.layout())?;
        let (id, spec, bytes) =
            input_spec(self, Family::FREQUENCIES_ID, ElementType::F32, elements)?;
        let mut output = host.allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)?;
        write_rotary_bytes(self, output.as_mut())?;
        Ok(ManagedHostTensor::new(id, spec, output.freeze())?)
    }

    /// BF16 `[B + causal_condition,256]` features for a shared scheduler time.
    /// Every batch item receives the supplied time; causal conditioning appends
    /// exactly one positive t0 row. Input is the scheduler's 0..1000 coordinate,
    /// not an already-normalized model timestep. The existing adapter applies
    /// `B(RN32(B(t)/1000))` before its checked F32 projection/final BF16 cast.
    ///
    /// Invalid time/geometry fails before reservation. Admission/alignment or
    /// adapter errors drop the private backing without publishing a tensor.
    pub fn managed_time_features(
        &self,
        host: &HostMemoryAccount,
        scheduler_timestep: f32,
    ) -> Result<ManagedHostTensor> {
        let time = QwenImage21ModelTimestep::from_scheduler_timestep(scheduler_timestep)?;
        let batch = usize::try_from(self.layout().batch())?;
        ensure!(
            (1..=STRUCTURED_ATTENTION_BF16_MAX_BATCH).contains(&batch),
            "timestep batch exceeds checked conditioning capacity"
        );
        // Bounded control values, not a [B,256] feature buffer.
        let timesteps = [time; STRUCTURED_ATTENTION_BF16_MAX_BATCH];
        let adapter = QwenImage21Timestep256::new()?;
        let elements = QwenImage21Timestep256::output_elements(self)?;
        let (id, spec, bytes) =
            input_spec(self, Family::TIME_FEATURES_ID, ElementType::Bf16, elements)?;
        let mut output = host.allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)?;
        write_time_bytes(self, &adapter, &timesteps[..batch], output.as_mut())?;
        Ok(ManagedHostTensor::new(id, spec, output.freeze())?)
    }
}

fn input_spec(
    config: &QwenImage21TransformerConfig,
    name: &str,
    dtype: ElementType,
    elements: usize,
) -> Result<(ProgramValueId, ResolvedTensorSpec, usize)> {
    let id = ProgramValueId::new(name)?;
    let declared = config
        .input_tensor_specs()?
        .remove(&id)
        .context("conditioning port missing from typed transformer config")?;
    ensure!(
        declared.element_type == dtype && declared.layout == ResolvedTensorLayout::Contiguous,
        "conditioning port differs from its mathematical adapter dtype/layout"
    );
    let spec =
        ResolvedTensorSpec::new(declared.dimensions, declared.element_type, declared.layout)?;
    let bytes = usize::try_from(spec.minimum_storage_bytes()?)?;
    let expected = elements
        .checked_mul(usize::try_from(dtype.size_bytes())?)
        .filter(|&length| length <= isize::MAX as usize)
        .context("conditioning payload exceeds addressable storage")?;
    ensure!(
        bytes == expected,
        "conditioning port extent differs from mathematical adapter"
    );
    Ok((id, spec, bytes))
}

fn write_rotary_bytes(config: &QwenImage21TransformerConfig, output: &mut [u8]) -> Result<()> {
    // Host promises byte alignment only. Preserve the verified example's safe
    // checked cast; alignment failure is explicit and never uses unsafe access.
    let values = bytemuck::try_cast_slice_mut::<u8, f32>(output)
        .map_err(|e| anyhow::anyhow!("managed rotary buffer alignment/extent: {e}"))?;
    config
        .rotary_frequencies()?
        .write_into(config.layout(), values)?;
    for value in values {
        *value = f32::from_bits(value.to_bits().to_le());
    }
    Ok(())
}

fn write_time_bytes(
    config: &QwenImage21TransformerConfig,
    adapter: &QwenImage21Timestep256,
    times: &[QwenImage21ModelTimestep],
    output: &mut [u8],
) -> Result<()> {
    let values = bytemuck::try_cast_slice_mut::<u8, u16>(output)
        .map_err(|e| anyhow::anyhow!("managed timestep buffer alignment/extent: {e}"))?;
    adapter.write_into(config, times, values)?;
    for value in values {
        *value = value.to_le();
    }
    Ok(())
}

#[cfg(test)]
mod tests;
