use ferrum_interfaces::vnext::{BatchedOperationInvocation, ElementType, ResolvedValueRole};

use super::super::vnext_runtime::{
    CpuBufferRegion, CpuDeviceBuffer, CpuRegionSet, CpuRuntimeError,
};
use super::bindings::{
    binding, binding_region, paged_regions, scratch_region, token_region, value_region,
};
use super::causal_attention::{
    self, CausalInputs, CausalScratch, CausalShape, CpuKvPages, KV_PAGE_BYTES,
};
use super::elementwise;
use super::launch::{CpuAction, CpuOperatorLaunch};
use super::matrix::CpuMatrix;
use super::scalar::CpuFloat;
use super::weights::{matrix_parts, CpuMatrixPart};

pub(super) struct CpuCausalLaunch {
    shape: CausalShape,
    dtype: CpuFloat,
    rows: usize,
    input: usize,
    output: usize,
    input_norm: usize,
    query_norm: usize,
    key_norm: usize,
    state: Vec<usize>,
    binding: usize,
    projections: [Vec<CpuMatrixPart>; 4],
    workspace: [usize; 8],
    residual_in_place: bool,
}

fn workspace_layout(shape: CausalShape) -> Result<[(u64, u64); 8], CpuRuntimeError> {
    let mut offset = 0_u64;
    let mut layout = [(0, 0); 8];
    for (slot, (width, dtype)) in layout.iter_mut().zip([
        (shape.hidden, CpuFloat::F16),
        (shape.query_projection(), CpuFloat::F16),
        (shape.kv(), CpuFloat::F16),
        (shape.kv(), CpuFloat::F16),
        (shape.queries(), CpuFloat::F16),
        (shape.queries(), CpuFloat::F32),
        (shape.queries(), CpuFloat::F16),
        (shape.hidden, CpuFloat::F16),
    ]) {
        let bytes = dtype.byte_len(1, width)? as u64;
        *slot = (offset, bytes);
        offset = offset
            .checked_add(bytes)
            .and_then(|end| end.checked_add(15))
            .map(|end| end & !15)
            .ok_or_else(|| CpuRuntimeError::new("CPU attention workspace overflows"))?;
    }
    Ok(layout)
}

pub(super) fn workspace_bytes(shape: CausalShape) -> Result<u64, CpuRuntimeError> {
    let (offset, bytes) = workspace_layout(shape)?[7];
    offset
        .checked_add(bytes)
        .and_then(|end| end.checked_add(15))
        .map(|end| end & !15)
        .ok_or_else(|| CpuRuntimeError::new("CPU attention workspace overflows"))
}

pub(super) fn prepare(
    invocation: &BatchedOperationInvocation<'_, CpuDeviceBuffer>,
    index: usize,
    rows: usize,
    dtype: CpuFloat,
    workspace_offset: u64,
    regions: &mut Vec<CpuBufferRegion>,
) -> Result<(CpuCausalLaunch, CpuOperatorLaunch), CpuRuntimeError> {
    use ResolvedValueRole::{Input, Output};
    let participant = &invocation.participants()[index];
    let shape = CausalShape::from_attributes(participant.attributes())?;
    let source = invocation
        .participant_token_ranges()
        .get(index)
        .ok_or_else(|| CpuRuntimeError::new("CPU causal participant lacks source positions"))?
        .source_token_range();
    if source.start >= source.end
        || source.end > shape.maximum_context as u64
        || source.end - source.start != rows as u64
    {
        return Err(CpuRuntimeError::new(
            "CPU causal source positions differ from the admitted token work",
        ));
    }
    let projections = [
        matrix_parts(
            participant,
            binding(participant, Input, 2)?,
            &[shape.query_projection() as u64, shape.hidden as u64],
            regions,
        )?,
        matrix_parts(
            participant,
            binding(participant, Input, 3)?,
            &[shape.kv() as u64, shape.hidden as u64],
            regions,
        )?,
        matrix_parts(
            participant,
            binding(participant, Input, 4)?,
            &[shape.kv() as u64, shape.hidden as u64],
            regions,
        )?,
        matrix_parts(
            participant,
            binding(participant, Input, 5)?,
            &[shape.hidden as u64, shape.queries() as u64],
            regions,
        )?,
    ];
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
            "CPU attention hidden buffers differ from their token shape",
        ));
    }
    let residual_in_place = input_region.same_physical_region(&output_region);
    let input = retain(input_region);
    let output = retain(output_region);
    let mut value = |ordinal| {
        value_region(
            participant,
            binding(participant, Input, ordinal)?,
            ElementType::F16,
            None,
        )
        .map(&mut retain)
    };
    let input_norm = value(1)?;
    let query_norm = value(6)?;
    let key_norm = value(7)?;
    let state_binding = binding(participant, Input, 8)?;
    let [component] = state_binding.storage().components() else {
        return Err(CpuRuntimeError::new(
            "CPU KV state requires one logical storage component",
        ));
    };
    if component.offset_bytes() != 0
        || component.element_type() != ElementType::F16
        || state_binding.tensor().element_type() != ElementType::F16
        || state_binding.tensor().dimensions() != [2, shape.kv_heads as u64, shape.head_dim as u64]
    {
        return Err(CpuRuntimeError::new(
            "CPU KV state differs from its token-scaled ABI",
        ));
    }
    let state_bytes = source
        .end
        .checked_mul(shape.state_bytes_per_token() as u64)
        .and_then(|bytes| bytes.checked_add(KV_PAGE_BYTES - 1))
        .map(|bytes| bytes / KV_PAGE_BYTES * KV_PAGE_BYTES)
        .ok_or_else(|| CpuRuntimeError::new("CPU KV frontier overflows"))?;
    let state = paged_regions(
        participant,
        component.resource_id(),
        state_bytes,
        KV_PAGE_BYTES,
        ElementType::F16,
    )?
    .into_iter()
    .map(&mut retain)
    .collect();
    let binding_offset = (index as u64)
        .checked_mul(16)
        .ok_or_else(|| CpuRuntimeError::new("CPU attention binding offset overflows"))?;
    let bound = binding_region(&invocation.participants()[0], binding_offset, 16)?;
    let binding = retain(bound.clone());
    let preparation = CpuOperatorLaunch {
        regions: vec![bound],
        actions: vec![CpuAction::CausalBinding {
            binding: 0,
            source_start: source.start,
            tokens: rows as u64,
        }],
    };
    let mut workspace = [0; 8];
    for (slot, (offset, bytes)) in workspace.iter_mut().zip(workspace_layout(shape)?) {
        let offset = workspace_offset.checked_add(offset).ok_or_else(|| {
            CpuRuntimeError::new("CPU attention participant scratch offset overflows")
        })?;
        *slot = retain(scratch_region(
            &invocation.participants()[0],
            offset,
            bytes,
        )?);
    }
    Ok((
        CpuCausalLaunch {
            shape,
            dtype,
            rows,
            input,
            output,
            input_norm,
            query_norm,
            key_norm,
            state,
            binding,
            projections,
            workspace,
            residual_in_place,
        },
        preparation,
    ))
}

impl CpuCausalLaunch {
    pub(super) fn execute(&self, views: &mut CpuRegionSet<'_>) -> Result<(), CpuRuntimeError> {
        let bound = views.read(self.binding);
        if bound.len() != 16 {
            return Err(CpuRuntimeError::new(
                "CPU attention binding is not a position/count pair",
            ));
        }
        let start = usize::try_from(u64::from_le_bytes(bound[..8].try_into().unwrap()))
            .map_err(|_| CpuRuntimeError::new("CPU attention start exceeds usize"))?;
        let rows = u64::from_le_bytes(bound[8..].try_into().unwrap());
        if rows != self.rows as u64
            || start
                .checked_add(self.rows)
                .is_none_or(|end| end > self.shape.maximum_context)
        {
            return Err(CpuRuntimeError::new(
                "CPU attention binding differs from its admitted work",
            ));
        }
        let [normalized, raw_query, raw_key, raw_value, query, accumulated, context, projected] =
            self.workspace;
        let hidden_bytes = self.dtype.byte_len(1, self.shape.hidden)?;
        let mut attention_outputs = self.state.clone();
        attention_outputs.extend([query, accumulated, context]);
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
            for (parts, destination, width) in [
                (
                    &self.projections[0],
                    raw_query,
                    self.shape.query_projection(),
                ),
                (&self.projections[1], raw_key, self.shape.kv()),
                (&self.projections[2], raw_value, self.shape.kv()),
            ] {
                project(views, parts, normalized, destination, width)?;
            }
            views.with_io_slices(
                &[
                    raw_query,
                    raw_key,
                    raw_value,
                    self.query_norm,
                    self.key_norm,
                ],
                &attention_outputs,
                |inputs, outputs| {
                    let (pages, scratch) = outputs.split_at_mut(self.state.len());
                    let [query, accumulated, context] = scratch else {
                        return Err(CpuRuntimeError::new("CPU causal workspace arity differs"));
                    };
                    let mut state = CpuKvPages::new(pages)?;
                    causal_attention::step(
                        self.shape,
                        start + row,
                        CausalInputs {
                            query: inputs[0],
                            key: inputs[1],
                            value: inputs[2],
                            query_norm: inputs[3],
                            key_norm: inputs[4],
                        },
                        &mut state,
                        CausalScratch {
                            query,
                            accumulated,
                            context,
                        },
                    )
                },
            )?;
            project(
                views,
                &self.projections[3],
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
