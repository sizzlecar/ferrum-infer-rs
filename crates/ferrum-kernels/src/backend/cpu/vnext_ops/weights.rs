use super::super::vnext_runtime::{CpuBufferRegion, CpuDeviceBuffer, CpuRuntimeError};
use super::bindings::physical_region;
use super::matrix::CpuMatrixFormat;
use ferrum_interfaces::vnext::{
    ElementType, OperationInvocation, PhysicalWeightLayout, PhysicalWeightPadding,
    ResolvedValueBinding, ResolvedWeightComponentLayout, WeightEncoding, WeightId,
};
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
pub(super) struct CpuMatrixPart {
    pub(super) region: usize,
    pub(super) format: CpuMatrixFormat,
    pub(super) rows: usize,
    pub(super) columns: usize,
    pub(super) output_offset: usize,
}

fn product(dimensions: &[u64]) -> Result<u64, CpuRuntimeError> {
    dimensions
        .iter()
        .try_fold(1_u64, |total, &extent| {
            total.checked_mul(extent).filter(|&n| n != 0)
        })
        .ok_or_else(|| CpuRuntimeError::new("CPU weight dimensions are empty or overflow"))
}

pub(super) fn matrix_parts(
    participant: &OperationInvocation<'_, CpuDeviceBuffer>,
    binding: &ResolvedValueBinding,
    shape: &[u64],
    regions: &mut Vec<CpuBufferRegion>,
) -> Result<Vec<CpuMatrixPart>, CpuRuntimeError> {
    if shape.len() < 2
        || binding.tensor().dimensions() != shape
        || binding.tensor().element_type() != ElementType::F16
    {
        return Err(CpuRuntimeError::new(
            "CPU matrix differs from its logical weight signature",
        ));
    }
    let weight = binding
        .weight()
        .ok_or_else(|| CpuRuntimeError::new("CPU matrix lacks typed physical metadata"))?;
    let storage = binding
        .storage()
        .components()
        .iter()
        .map(|component| {
            component
                .component_id()
                .map(|id| (id, component))
                .ok_or_else(|| {
                    CpuRuntimeError::new("CPU weight storage lacks a component identity")
                })
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    if storage.len() != weight.components().len()
        || storage.len() != binding.storage().components().len()
    {
        return Err(CpuRuntimeError::new(
            "CPU weight components are duplicated or incomplete",
        ));
    }
    let mut metadata = BTreeMap::new();
    for component in weight.components() {
        let stored = storage
            .get(component.component_id())
            .ok_or_else(|| CpuRuntimeError::new("CPU weight component is absent"))?;
        if stored.element_type() != component.physical_element_type()
            || stored.length_bytes() != component.physical_bytes()?
        {
            return Err(CpuRuntimeError::new(
                "CPU weight storage differs from its physical encoding",
            ));
        }
        let index = regions.len();
        regions.push(physical_region(
            participant,
            stored.resource_id(),
            stored.offset_bytes(),
            stored.length_bytes(),
            stored.element_type(),
        )?);
        if metadata
            .insert(component.component_id(), (index, component))
            .is_some()
        {
            return Err(CpuRuntimeError::new("CPU weight identity is duplicated"));
        }
    }
    let mut parts = Vec::new();
    flatten(weight.physical_layout(), shape, 0, &metadata, &mut parts)?;
    parts.sort_unstable_by_key(|part| part.output_offset);
    let expected = usize::try_from(product(&shape[..shape.len() - 1])?)
        .map_err(|_| CpuRuntimeError::new("CPU matrix rows exceed usize"))?;
    let mut cursor = 0;
    for part in &parts {
        if part.output_offset != cursor {
            return Err(CpuRuntimeError::new(
                "CPU matrix partitions overlap or leave a gap",
            ));
        }
        cursor = cursor
            .checked_add(part.rows)
            .ok_or_else(|| CpuRuntimeError::new("CPU matrix output range overflows"))?;
    }
    if cursor != expected {
        return Err(CpuRuntimeError::new(
            "CPU matrix partitions do not cover its output",
        ));
    }
    Ok(parts)
}

fn flatten(
    layout: &PhysicalWeightLayout,
    shape: &[u64],
    output_offset: u64,
    metadata: &BTreeMap<&WeightId, (usize, &ResolvedWeightComponentLayout)>,
    output: &mut Vec<CpuMatrixPart>,
) -> Result<(), CpuRuntimeError> {
    if let PhysicalWeightLayout::Composite { parts } = layout {
        for part in parts {
            if part.extents.len() != shape.len()
                || part.logical_offsets.len() != shape.len()
                || part.extents.last() != shape.last()
                || part.logical_offsets.last() != Some(&0)
            {
                return Err(CpuRuntimeError::new(
                    "CPU matrix partitions must preserve complete input rows",
                ));
            }
            let mut start = 0_u64;
            let mut end = 0_u64;
            for ((&offset, &extent), &dimension) in part
                .logical_offsets
                .iter()
                .zip(&part.extents)
                .zip(shape)
                .take(shape.len() - 1)
            {
                let limit = offset
                    .checked_add(extent)
                    .filter(|&end| extent != 0 && end <= dimension)
                    .ok_or_else(|| {
                        CpuRuntimeError::new("CPU matrix partition exceeds its logical shape")
                    })?;
                start = start
                    .checked_mul(dimension)
                    .and_then(|n| n.checked_add(offset))
                    .ok_or_else(|| CpuRuntimeError::new("CPU matrix partition offset overflows"))?;
                end = end
                    .checked_mul(dimension)
                    .and_then(|n| n.checked_add(limit - 1))
                    .ok_or_else(|| CpuRuntimeError::new("CPU matrix partition end overflows"))?;
            }
            if end.checked_sub(start).and_then(|n| n.checked_add(1))
                != Some(product(&part.extents[..part.extents.len() - 1])?)
            {
                return Err(CpuRuntimeError::new(
                    "CPU matrix partition rows are not contiguous",
                ));
            }
            flatten(
                &part.layout,
                &part.extents,
                output_offset
                    .checked_add(start)
                    .ok_or_else(|| CpuRuntimeError::new("CPU matrix output offset overflows"))?,
                metadata,
                output,
            )?;
        }
        return Ok(());
    }
    let (id, native) = match layout {
        PhysicalWeightLayout::Dense { component_id } => (component_id, false),
        PhysicalWeightLayout::Stored { component } => (&component.component_id, false),
        PhysicalWeightLayout::BlockQuantized {
            blocks,
            block_axis,
            block_padding,
        } if *block_axis as usize == shape.len() - 1
            && *block_padding == PhysicalWeightPadding::Exact =>
        {
            (&blocks.component_id, true)
        }
        _ => {
            return Err(CpuRuntimeError::new(
                "CPU matrix physical layout has no installed kernel",
            ))
        }
    };
    let &(region, component) = metadata
        .get(id)
        .ok_or_else(|| CpuRuntimeError::new("CPU matrix leaf references an absent weight"))?;
    if native != matches!(component.encoding(), WeightEncoding::BlockQuantized(_)) {
        return Err(CpuRuntimeError::new(
            "CPU matrix leaf disagrees with its declared encoding",
        ));
    }
    let format = CpuMatrixFormat::from_encoding(component.encoding())?;
    let columns = *shape.last().unwrap();
    let rows = product(&shape[..shape.len() - 1])?;
    let physical_columns = match component.encoding() {
        WeightEncoding::BlockQuantized(spec)
            if columns.is_multiple_of(u64::from(spec.logical_values_per_block)) =>
        {
            columns / u64::from(spec.logical_values_per_block)
        }
        WeightEncoding::Dense {
            element_type: ElementType::F16,
        } => columns,
        _ => {
            return Err(CpuRuntimeError::new(
                "CPU matrix leaf has an incompatible row ABI",
            ))
        }
    };
    if component.physical_dimensions().last() != Some(&physical_columns)
        || product(component.physical_dimensions())?
            != rows
                .checked_mul(physical_columns)
                .ok_or_else(|| CpuRuntimeError::new("CPU matrix physical size overflows"))?
    {
        return Err(CpuRuntimeError::new(
            "CPU matrix leaf physical dimensions differ",
        ));
    }
    let to_usize = |value| {
        usize::try_from(value)
            .map_err(|_| CpuRuntimeError::new("CPU matrix dimensions exceed addressable memory"))
    };
    output.push(CpuMatrixPart {
        region,
        format,
        rows: to_usize(rows)?,
        columns: to_usize(columns)?,
        output_offset: to_usize(output_offset)?,
    });
    Ok(())
}
