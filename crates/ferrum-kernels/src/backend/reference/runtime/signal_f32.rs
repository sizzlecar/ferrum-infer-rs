use super::*;
use ferrum_runtime::{SignalF32Geometry, SignalF32Op};

pub(crate) struct ReferenceSignalF32Launch {
    pub(crate) inputs: Vec<ReferenceBufferRegion>,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) geometry: SignalF32Geometry,
}
impl ReferenceSignalF32Launch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        let g = self.geometry;
        if self.inputs.len() != g.kind.input_count() {
            return Err(ReferenceDeviceRuntimeError::contract(
                "signal input count differs",
            ));
        }
        for (r, bytes) in self
            .inputs
            .iter()
            .enumerate()
            .map(|(n, r)| (r, g.input_elements(n).unwrap() * 4))
            .chain(std::iter::once((&self.output, g.output_elements() * 4)))
        {
            r.validate_range()?;
            if r.element_type != ElementType::F32
                || r.length_bytes as u64 != bytes
                || (r.offset_bytes - r.allocation.logical_offset_bytes) % 4 != 0
            {
                return Err(ReferenceDeviceRuntimeError::contract(
                    "signal requires exact aligned F32 physical extents",
                ));
            }
        }
        if self.inputs.iter().any(|r| self.output.overlaps(r)) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "signal output aliases input",
            ));
        }
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        for r in self.inputs.iter().chain(std::iter::once(&self.output)) {
            r.validate_runtime(runtime)?;
        }
        self.validate()
    }
}
pub(super) fn execute(l: &ReferenceSignalF32Launch) {
    let g = l.geometry;
    // Unused internal lock slots reuse the already-retained input. They are
    // not extra graph operands or allocations.
    let mut s = AllocationLocks::new([
        &l.inputs[0],
        l.inputs.get(1).unwrap_or(&l.inputs[0]),
        l.inputs.get(2).unwrap_or(&l.inputs[0]),
        &l.output,
    ]);
    for index in 0..g.output_elements() {
        let value = match g.kind {
            SignalF32Op::FirResample => {
                let y = index % g.output_width;
                let channel = index / g.output_width;
                let position = y + g.crop[0];
                let padded = g.input_width + g.padding[0] + g.padding[1];
                let mut sum = 0.0f32;
                for tap in 0..g.taps {
                    let input = if g.transpose {
                        if position < tap || (position - tap) % g.stride != 0 {
                            continue;
                        }
                        let j = (position - tap) / g.stride;
                        if j >= padded {
                            continue;
                        }
                        j
                    } else {
                        position * g.stride + tap
                    };
                    let input = input.saturating_sub(g.padding[0]).min(g.input_width - 1);
                    sum = s
                        .read_f32(0, (channel * g.input_width + input) as usize)
                        .mul_add(s.read_f32(1, tap as usize), sum);
                }
                sum
            }
            SignalF32Op::SnakeBetaLog => {
                let c = index / (g.height * g.input_width) % g.channels;
                let x = s.read_f32(0, index as usize);
                let alpha = s.read_f32(1, c as usize).exp();
                let beta = s.read_f32(2, c as usize).exp();
                let sine = (alpha * x).sin();
                let square = sine * sine;
                let reciprocal = 1.0f32 / (beta + 1.0e-9f32);
                x + reciprocal * square
            }
            SignalF32Op::SnakeAlpha => {
                let c = index / (g.height * g.input_width) % g.channels;
                let x = s.read_f32(0, index as usize);
                let alpha = s.read_f32(1, c as usize);
                let sine = (alpha * x).sin();
                let reciprocal = 1.0f32 / (alpha + 1.0e-9f32);
                x + reciprocal * (sine * sine)
            }
            SignalF32Op::Clamp => {
                let x = s.read_f32(0, index as usize);
                if x < g.minimum {
                    g.minimum
                } else if x > g.maximum {
                    g.maximum
                } else {
                    x
                }
            }
        };
        s.write_f32(3, index as usize, value);
    }
}

#[cfg(test)]
#[path = "signal_f32/tests.rs"]
mod tests;
