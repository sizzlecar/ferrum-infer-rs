use ferrum_runtime::{BatchedOperationInvocation, ElementType, ResolvedValueRole};

use super::super::vnext_runtime::{
    CpuBufferRegion, CpuDeviceBuffer, CpuRegionSet, CpuRuntimeError,
};
use super::bindings::{binding, scratch_region, token_region, value_region};
use super::elementwise;
use super::gated_delta::{
    self, GatedDeltaScratch, GatedDeltaShape, GatedDeltaState, GatedDeltaWeights,
};
use super::matrix::CpuMatrix;
use super::scalar::CpuFloat;
use super::weights::{matrix_parts, CpuMatrixPart};

pub(super) struct CpuGatedDeltaLaunch {
    shape: GatedDeltaShape,
    dtype: CpuFloat,
    rows: usize,
    input: usize,
    output: usize,
    input_norm: usize,
    convolution_weight: usize,
    decay: usize,
    bias: usize,
    gated_norm: usize,
    convolution_state: usize,
    recurrent_state: usize,
    in_projection: Vec<CpuMatrixPart>,
    out_projection: Vec<CpuMatrixPart>,
    workspace: [usize; 6],
    residual_in_place: bool,
    prefix_candidates: Option<PrefixCandidates>,
}

struct PrefixCandidates {
    convolution: usize,
    recurrent: usize,
    convolution_bytes: usize,
    recurrent_bytes: usize,
    reads: Vec<usize>,
    writes: Vec<usize>,
}

fn workspace_layout(shape: GatedDeltaShape) -> Result<[(u64, u64); 6], CpuRuntimeError> {
    let mut offset = 0_u64;
    let mut layout = [(0, 0); 6];
    for (slot, (width, dtype)) in layout.iter_mut().zip([
        (shape.hidden, CpuFloat::F16),
        (shape.mixed(), CpuFloat::F16),
        (shape.qkv(), CpuFloat::F32),
        (shape.values(), CpuFloat::F32),
        (shape.values(), CpuFloat::F16),
        (shape.hidden, CpuFloat::F16),
    ]) {
        let bytes = dtype.byte_len(1, width)? as u64;
        *slot = (offset, bytes);
        offset = offset
            .checked_add(bytes)
            .and_then(|end| end.checked_add(15))
            .map(|end| end & !15)
            .ok_or_else(|| CpuRuntimeError::new("CPU GDN workspace overflows"))?;
    }
    Ok(layout)
}

pub(super) fn workspace_bytes(shape: GatedDeltaShape) -> Result<u64, CpuRuntimeError> {
    let (offset, bytes) = workspace_layout(shape)?[5];
    offset
        .checked_add(bytes)
        .and_then(|end| end.checked_add(15))
        .map(|end| end & !15)
        .ok_or_else(|| CpuRuntimeError::new("CPU GDN workspace overflows"))
}

pub(super) fn prepare(
    invocation: &BatchedOperationInvocation<'_, CpuDeviceBuffer>,
    index: usize,
    rows: usize,
    dtype: CpuFloat,
    workspace_offset: u64,
    regions: &mut Vec<CpuBufferRegion>,
) -> Result<CpuGatedDeltaLaunch, CpuRuntimeError> {
    use ResolvedValueRole::{Input, Output};
    let participant = &invocation.participants()[index];
    let shape = GatedDeltaShape::from_attributes(participant.attributes())?;
    let in_projection = matrix_parts(
        participant,
        binding(participant, Input, 2)?,
        &[shape.mixed() as u64, shape.hidden as u64],
        regions,
    )?;
    let out_projection = matrix_parts(
        participant,
        binding(participant, Input, 7)?,
        &[shape.hidden as u64, shape.values() as u64],
        regions,
    )?;
    let mut retain = |region| {
        let index = regions.len();
        regions.push(region);
        index
    };
    let input_region = token_region(invocation, index, Input, 0, dtype.element_type(), false)?;
    let output_region = token_region(invocation, index, Output, 0, dtype.element_type(), false)?;
    if input_region.length_bytes() != dtype.byte_len(rows, shape.hidden)?
        || output_region.length_bytes() != input_region.length_bytes()
    {
        return Err(CpuRuntimeError::new(
            "CPU GDN hidden buffers differ from their token shape",
        ));
    }
    let residual_in_place = input_region.same_physical_region(&output_region);
    let input = retain(input_region);
    let output = retain(output_region);
    let mut value = |ordinal, dtype| {
        value_region(
            participant,
            binding(participant, Input, ordinal)?,
            dtype,
            None,
        )
        .map(&mut retain)
    };
    let input_norm = value(1, ElementType::F16)?;
    let convolution_weight = value(3, ElementType::F16)?;
    let decay = value(4, ElementType::F32)?;
    let bias = value(5, ElementType::F32)?;
    let gated_norm = value(6, ElementType::F32)?;
    let convolution_state = value(8, ElementType::F16)?;
    let recurrent_state = value(9, ElementType::F32)?;
    let mut workspace = [0; 6];
    for (slot, (offset, bytes)) in workspace.iter_mut().zip(workspace_layout(shape)?) {
        let offset = workspace_offset
            .checked_add(offset)
            .ok_or_else(|| CpuRuntimeError::new("CPU GDN participant scratch offset overflows"))?;
        *slot = retain(scratch_region(
            &invocation.participants()[0],
            offset,
            bytes,
        )?);
    }
    Ok(CpuGatedDeltaLaunch {
        shape,
        dtype,
        rows,
        input,
        output,
        input_norm,
        convolution_weight,
        decay,
        bias,
        gated_norm,
        convolution_state,
        recurrent_state,
        in_projection,
        out_projection,
        workspace,
        residual_in_place,
        prefix_candidates: None,
    })
}

/// Token-row scan retaining the last two candidate states in admitted output
/// storage; the original state and every other input remain read-only.
pub(super) fn prepare_candidate_scan(
    invocation: &BatchedOperationInvocation<'_, CpuDeviceBuffer>,
    index: usize,
    rows: usize,
    dtype: CpuFloat,
    workspace_offset: u64,
    regions: &mut Vec<CpuBufferRegion>,
) -> Result<CpuGatedDeltaLaunch, CpuRuntimeError> {
    let participant = &invocation.participants()[index];
    if rows == 0 {
        return Err(CpuRuntimeError::new(
            "CPU GDN candidate scan requires at least one row",
        ));
    }
    let region_start = regions.len();
    let mut launch = prepare(invocation, index, rows, dtype, workspace_offset, regions)?;
    let convolution_bytes = launch
        .shape
        .qkv()
        .checked_mul(launch.shape.convolution - 1)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| CpuRuntimeError::new("CPU GDN convolution state overflows"))?;
    let recurrent_bytes = launch
        .shape
        .values()
        .checked_mul(launch.shape.key_dim)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| CpuRuntimeError::new("CPU GDN recurrent state overflows"))?;
    if regions[launch.convolution_state].length_bytes() != convolution_bytes
        || regions[launch.recurrent_state].length_bytes() != recurrent_bytes
    {
        return Err(CpuRuntimeError::new(
            "CPU GDN candidate scan original state extent differs",
        ));
    }
    let mut writes = vec![launch.output];
    writes.extend(launch.workspace);
    let reads = (region_start..regions.len())
        .filter(|i| !writes.contains(i))
        .collect();
    let mut candidate = |ordinal, dtype, bytes: usize| {
        let region = value_region(
            participant,
            binding(participant, ResolvedValueRole::Output, ordinal)?,
            dtype,
            None,
        )?;
        if bytes.checked_mul(2) != Some(region.length_bytes()) {
            return Err(CpuRuntimeError::new(
                "CPU GDN candidate scan candidate extent differs",
            ));
        }
        let index = regions.len();
        regions.push(region);
        writes.push(index);
        Ok(index)
    };
    let convolution = candidate(1, ElementType::F16, convolution_bytes)?;
    let recurrent = candidate(2, ElementType::F32, recurrent_bytes)?;
    launch.prefix_candidates = Some(PrefixCandidates {
        convolution,
        recurrent,
        convolution_bytes,
        recurrent_bytes,
        reads,
        writes,
    });
    Ok(launch)
}

impl CpuGatedDeltaLaunch {
    pub(super) fn execute(&self, views: &mut CpuRegionSet<'_>) -> Result<(), CpuRuntimeError> {
        let [normalized, mixed, qkv, core, context, projected] = self.workspace;
        let hidden_bytes = self.dtype.byte_len(1, self.shape.hidden)?;
        if let Some(prefix) = &self.prefix_candidates {
            // Validate all candidate writes (including hidden and workspace)
            // against every read and each other before the first mutation.
            views.with_io_slices(&prefix.reads, &prefix.writes, |_, _| Ok(()))?;
            views.with_io(
                [self.convolution_state, self.recurrent_state],
                [prefix.convolution, prefix.recurrent],
                |[original_conv, original_delta], [conv, delta]| {
                    conv[prefix.convolution_bytes..].copy_from_slice(original_conv);
                    delta[prefix.recurrent_bytes..].copy_from_slice(original_delta);
                    Ok(())
                },
            )?;
        }
        let (convolution_state, recurrent_state) = self
            .prefix_candidates
            .as_ref()
            .map(|prefix| (prefix.convolution, prefix.recurrent))
            .unwrap_or((self.convolution_state, self.recurrent_state));
        for row in 0..self.rows {
            let span = row * hidden_bytes..(row + 1) * hidden_bytes;
            views.with_io(
                [self.input, self.input_norm],
                [normalized],
                |[input, weight], [output]| {
                    elementwise::rms_norm(
                        &input[span.clone()],
                        self.dtype,
                        weight,
                        output,
                        CpuFloat::F16,
                        self.shape.hidden,
                        self.shape.epsilon,
                    )
                },
            )?;
            project(
                views,
                &self.in_projection,
                normalized,
                mixed,
                self.shape.mixed(),
            )?;
            views.with_io(
                [
                    mixed,
                    self.convolution_weight,
                    self.decay,
                    self.bias,
                    self.gated_norm,
                ],
                [convolution_state, recurrent_state, qkv, core, context],
                |[mixed, convolution, decay, dt_bias, norm],
                 [convolution_state, recurrent, qkv, core, output]| {
                    let (convolution_state, recurrent) =
                        if let Some(prefix) = &self.prefix_candidates {
                            let c = prefix.convolution_bytes;
                            let d = prefix.recurrent_bytes;
                            if row + 1 == self.rows {
                                // Preserve the N-1 boundary before computing row N.
                                // For N=1 this is the untouched original state.
                                convolution_state.copy_within(c.., 0);
                                recurrent.copy_within(d.., 0);
                            }
                            (&mut convolution_state[c..], &mut recurrent[d..])
                        } else {
                            (convolution_state, recurrent)
                        };
                    gated_delta::step(
                        self.shape,
                        mixed,
                        GatedDeltaWeights {
                            convolution,
                            decay,
                            dt_bias,
                            norm,
                        },
                        GatedDeltaState {
                            convolution: convolution_state,
                            recurrent,
                        },
                        GatedDeltaScratch { qkv, core, output },
                    )
                },
            )?;
            project(
                views,
                &self.out_projection,
                context,
                projected,
                self.shape.hidden,
            )?;
            if self.residual_in_place {
                views.with_io([projected], [self.output], |[right], [output]| {
                    elementwise::residual_add_in_place(
                        &mut output[span.clone()],
                        self.dtype,
                        Some(right),
                    )
                })?;
            } else {
                views.with_io(
                    [self.input, projected],
                    [self.output],
                    |[left, right], [output]| {
                        elementwise::residual_add(
                            &left[span.clone()],
                            self.dtype,
                            right,
                            &mut output[span.clone()],
                        )
                    },
                )?;
            }
        }
        Ok(())
    }
}

fn project(
    views: &mut CpuRegionSet<'_>,
    parts: &[CpuMatrixPart],
    input: usize,
    output: usize,
    width: usize,
) -> Result<(), CpuRuntimeError> {
    for part in parts {
        views.with_io(
            [input, part.region],
            [output],
            |[input, weight], [output]| {
                CpuMatrix::new(weight, part.format, part.rows, part.columns)?.linear(
                    input,
                    CpuFloat::F16,
                    output,
                    CpuFloat::F16,
                    1,
                    width,
                    part.output_offset,
                )
            },
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
