use ferrum_runtime::{BatchedOperationInvocation, ElementType, ResolvedValueRole};

use super::super::vnext_runtime::{
    CpuBufferRegion, CpuDeviceBuffer, CpuRegionSet, CpuRuntimeError,
};
use super::bindings::{binding, scratch_region, token_region, value_region};
use super::matrix::{CpuMatrix, CpuMatrixFormat};
use super::provider::unsigned;
use super::scalar::CpuFloat;
use super::weights::matrix_parts;

pub(super) struct CpuTailTwoGreedyLaunch {
    rows: usize,
    hidden: usize,
    vocabulary: usize,
    input: usize,
    weight: usize,
    decisions: usize,
    carrier: usize,
    scratch: usize,
}

pub(super) fn prepare(
    invocation: &BatchedOperationInvocation<'_, CpuDeviceBuffer>,
    index: usize,
    rows: usize,
    workspace_offset: u64,
    regions: &mut Vec<CpuBufferRegion>,
) -> Result<CpuTailTwoGreedyLaunch, CpuRuntimeError> {
    use ResolvedValueRole::{Input, Output};
    let participant = &invocation.participants()[index];
    let hidden = unsigned(participant.attributes(), "hidden_size")?;
    let vocabulary = unsigned(participant.attributes(), "vocab_size")?;
    if rows == 0 || hidden == 0 || vocabulary == 0 || vocabulary > u32::MAX as u64 {
        return Err(CpuRuntimeError::new("CPU tail head dimensions are invalid"));
    }
    let parts = matrix_parts(
        participant,
        binding(participant, Input, 1)?,
        &[vocabulary, hidden],
        regions,
    )?;
    let [weight] = parts.as_slice() else {
        return Err(CpuRuntimeError::new(
            "CPU tail head requires one dense F16 weight matrix",
        ));
    };
    if !matches!(weight.format, CpuMatrixFormat::Dense(CpuFloat::F16)) || weight.output_offset != 0
    {
        return Err(CpuRuntimeError::new(
            "CPU tail head does not support quantized weights",
        ));
    }
    let weight = weight.region;
    let mut retain = |region| {
        let i = regions.len();
        regions.push(region);
        i
    };
    let input = retain(token_region(
        invocation,
        index,
        Input,
        0,
        ElementType::F16,
        false,
    )?);
    let decisions = retain(value_region(
        participant,
        binding(participant, Output, 0)?,
        ElementType::U32,
        None,
    )?);
    let carrier = retain(value_region(
        participant,
        binding(participant, Output, 1)?,
        ElementType::F16,
        None,
    )?);
    let scratch = retain(scratch_region(
        &invocation.participants()[0],
        workspace_offset,
        vocabulary
            .checked_mul(2)
            .ok_or_else(|| CpuRuntimeError::new("CPU tail head scratch overflows"))?,
    )?);
    let hidden = usize::try_from(hidden)
        .map_err(|_| CpuRuntimeError::new("CPU tail hidden exceeds usize"))?;
    let vocabulary = usize::try_from(vocabulary)
        .map_err(|_| CpuRuntimeError::new("CPU tail vocabulary exceeds usize"))?;
    if regions[input].length_bytes() != CpuFloat::F16.byte_len(rows, hidden)?
        || regions[decisions].length_bytes() != 8
        || regions[carrier].length_bytes() != CpuFloat::F16.byte_len(2, hidden)?
    {
        return Err(CpuRuntimeError::new(
            "CPU tail head buffers differ from typed extents",
        ));
    }
    Ok(CpuTailTwoGreedyLaunch {
        rows,
        hidden,
        vocabulary,
        input,
        weight,
        decisions,
        carrier,
        scratch,
    })
}

impl CpuTailTwoGreedyLaunch {
    pub(super) fn execute(&self, views: &mut CpuRegionSet<'_>) -> Result<(), CpuRuntimeError> {
        // One borrow validates all reads/writes and aliases before any mutation.
        views.with_io(
            [self.input, self.weight],
            [self.decisions, self.carrier, self.scratch],
            |[input, weight], [decisions, carrier, scratch]| {
                evaluate(
                    input,
                    weight,
                    self.rows,
                    self.hidden,
                    self.vocabulary,
                    decisions,
                    carrier,
                    scratch,
                )
            },
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate(
    input: &[u8],
    weight: &[u8],
    rows: usize,
    hidden: usize,
    vocabulary: usize,
    decisions: &mut [u8],
    carrier: &mut [u8],
    scratch: &mut [u8],
) -> Result<(), CpuRuntimeError> {
    let dtype = CpuFloat::F16;
    let stride = dtype.byte_len(1, hidden)?;
    if rows == 0
        || vocabulary == 0
        || vocabulary > u32::MAX as usize
        || input.len() != dtype.byte_len(rows, hidden)?
        || decisions.len() != 8
        || carrier.len() != dtype.byte_len(2, hidden)?
        || scratch.len() != dtype.byte_len(1, vocabulary)?
    {
        return Err(CpuRuntimeError::new("CPU tail head extents are invalid"));
    }
    let matrix = CpuMatrix::new(weight, CpuMatrixFormat::Dense(dtype), vocabulary, hidden)?;
    let tail_rows = rows.min(2);
    for slot in 0..tail_rows {
        let row = rows - tail_rows + slot;
        let source = &input[row * stride..(row + 1) * stride];
        // Reuse the existing dense-head accumulation and F16 output rounding.
        matrix.linear(source, dtype, scratch, dtype, 1, vocabulary, 0)?;
        let mut best = u32::MAX;
        let mut maximum = f32::NEG_INFINITY;
        for token in 0..vocabulary {
            super::elementwise::select_finite_max(
                &mut best,
                &mut maximum,
                token as u32,
                dtype.read(scratch, token),
            );
        }
        decisions[slot * 4..(slot + 1) * 4].copy_from_slice(&best.to_le_bytes());
        carrier[slot * stride..(slot + 1) * stride].copy_from_slice(source);
    }
    if rows == 1 {
        // Both slots name the same row; preserve its rounded decision without
        // projecting that row a second time.
        decisions.copy_within(..4, 4);
        carrier.copy_within(..stride, stride);
    }
    Ok(())
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod arithmetic_tests {
    use super::*;
    fn f16(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
            .collect()
    }
    #[test]
    fn cpu_tail_head_compares_rounded_logits_and_skips_nonfinite_values() {
        // Row1 has a larger F32 dot but rounds to the same F16 as row0.
        // NaN and overflowing logits must not displace a finite winner.
        let input = f16(&[1., 1. / 4096.]);
        let weight = f16(&[1., 0., 1., 1., f32::NAN, 0., 65504., 65504.]);
        let mut decisions = [0u8; 8];
        let mut carrier = [0u8; 8];
        let mut scratch = [0u8; 8];
        // Make the final row nonfinite explicitly without relying on an
        // overflow threshold accidentally different from F16 rounding.
        let mut weight = weight;
        weight[12..14].copy_from_slice(&half::f16::INFINITY.to_le_bytes());
        evaluate(
            &input,
            &weight,
            1,
            2,
            4,
            &mut decisions,
            &mut carrier,
            &mut scratch,
        )
        .unwrap();
        assert_eq!(decisions, [0; 8]);
        assert_eq!(&carrier[..4], input);
        assert_eq!(&carrier[4..], input);

        let none = f16(&[f32::NAN, 0., f32::INFINITY, 0.]);
        evaluate(
            &input,
            &none,
            1,
            2,
            2,
            &mut decisions,
            &mut carrier,
            &mut scratch[..4],
        )
        .unwrap();
        assert_eq!(decisions, [255; 8]);
    }
    #[test]
    fn cpu_tail_head_extent_failure_does_not_modify_destinations() {
        let input = f16(&[1., 2.]);
        let weight = f16(&[1., 0.]);
        let mut decisions = [0x55u8; 8];
        let mut carrier = [0x55u8; 8];
        let mut scratch = [0x55u8; 2];
        assert!(evaluate(
            &input,
            &weight,
            2,
            2,
            1,
            &mut decisions,
            &mut carrier,
            &mut scratch
        )
        .is_err());
        assert_eq!(decisions, [0x55; 8]);
        assert_eq!(carrier, [0x55; 8]);
        assert_eq!(scratch, [0x55; 2]);
    }
}
