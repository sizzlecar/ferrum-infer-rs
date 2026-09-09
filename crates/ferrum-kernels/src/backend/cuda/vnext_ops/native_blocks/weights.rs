//! Retained native matrix parts, addressed by the declared physical component IDs.

use std::collections::BTreeMap;

use ferrum_interfaces::vnext::{
    ElementType, OperationInvocation, PhysicalStorageLayout, PhysicalWeightLayout,
    PhysicalWeightPadding, ResolvedValueBinding, ResolvedWeightBinding,
    ResolvedWeightComponentLayout, WeightEncoding, WeightId,
};

use super::super::super::vnext_runtime::{CudaBufferRegion, CudaDeviceBuffer};
use crate::gguf_blocks::GgufBlockFormat;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::backend::cuda::vnext_ops) enum MatrixFormat {
    DenseF16,
    Block(GgufBlockFormat),
}

impl MatrixFormat {
    pub(in crate::backend::cuda::vnext_ops) fn parameters(self) -> [u32; 3] {
        match self {
            Self::DenseF16 => [1, 1, 2],
            Self::Block(format) => [
                format.ggml_type_id(),
                format.block_values() as u32,
                format.block_bytes() as u32,
            ],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::backend::cuda::vnext_ops) struct MatrixPart {
    pub(in crate::backend::cuda::vnext_ops) component_id: WeightId,
    pub(in crate::backend::cuda::vnext_ops) format: MatrixFormat,
    pub(in crate::backend::cuda::vnext_ops) rows: u32,
    pub(in crate::backend::cuda::vnext_ops) columns: u32,
    pub(in crate::backend::cuda::vnext_ops) output_offset: u32,
}

pub(in crate::backend::cuda::vnext_ops) struct MatrixWeight {
    pub(in crate::backend::cuda::vnext_ops) parts: Vec<MatrixPart>,
    // Exactly one region per part, in logical output order.
    pub(in crate::backend::cuda::vnext_ops) regions: Vec<CudaBufferRegion>,
}

pub(in crate::backend::cuda::vnext_ops) fn resolve(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    binding: &ResolvedValueBinding,
    shape: &[u64],
) -> Result<MatrixWeight, String> {
    if binding.tensor().dimensions() != shape || binding.tensor().element_type() != ElementType::F16
    {
        return Err("CUDA native matrix differs from its logical signature".into());
    }
    let weight = binding
        .weight()
        .ok_or("CUDA native matrix has no physical weight metadata")?;
    let parts = matrix_parts(weight, shape)?;
    let components = binding.storage().components();
    let mut storage = BTreeMap::new();
    for component in components {
        let id = component
            .component_id()
            .ok_or("CUDA native matrix storage lacks a component ID")?;
        if storage.insert(id, component).is_some() {
            return Err("CUDA native matrix storage has duplicate component IDs".into());
        }
    }
    if storage.len() != weight.components().len() {
        return Err("CUDA native matrix storage component inventory differs".into());
    }
    let metadata = weight
        .components()
        .iter()
        .map(|component| (component.component_id(), component))
        .collect::<BTreeMap<_, _>>();
    let mut regions = Vec::with_capacity(parts.len());
    for part in &parts {
        let component = metadata
            .get(&part.component_id)
            .ok_or("CUDA native matrix metadata component is absent")?;
        let stored = storage
            .get(&part.component_id)
            .ok_or("CUDA native matrix storage component is absent")?;
        let expected_bytes = component
            .physical_bytes()
            .map_err(|error| error.to_string())?;
        let expected_type = component.physical_element_type();
        if stored.length_bytes() != expected_bytes || stored.element_type() != expected_type {
            return Err("CUDA native matrix storage disagrees with the physical ABI".into());
        }
        let mut views = participant
            .views()
            .iter()
            .filter(|view| view.resource_id() == stored.resource_id());
        let view = views
            .next()
            .ok_or("CUDA native matrix resource view is absent")?;
        if views.next().is_some() {
            return Err("CUDA native matrix resource view is ambiguous".into());
        }
        let translated = view
            .translate(stored.offset_bytes(), stored.length_bytes())
            .map_err(|error| error.to_string())?;
        let mut physical = translated.iter();
        let range = physical
            .next()
            .ok_or("CUDA native matrix physical range is absent")?;
        if physical.next().is_some() {
            return Err("CUDA native matrix requires contiguous physical rows".into());
        }
        let (buffer, range, retention) = range.buffer_and_physical_range();
        let region = buffer
            .retained_region(range, retention)
            .map_err(|error| error.to_string())?;
        if region.element_type() != expected_type || region.length_bytes() != expected_bytes {
            return Err("CUDA native matrix retained a different physical range".into());
        }
        regions.push(region);
    }
    Ok(MatrixWeight { parts, regions })
}

fn product(shape: &[u64]) -> Result<u64, String> {
    if shape.is_empty() {
        return Err("CUDA native matrix shape is empty".into());
    }
    shape.iter().try_fold(1_u64, |total, &extent| {
        total
            .checked_mul(extent)
            .filter(|&n| n != 0)
            .ok_or_else(|| "CUDA native matrix shape is zero or overflows".into())
    })
}

fn dimension(value: u64) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| "CUDA native matrix extent exceeds u32".into())
}

pub(in crate::backend::cuda::vnext_ops) fn matrix_parts(
    weight: &ResolvedWeightBinding,
    shape: &[u64],
) -> Result<Vec<MatrixPart>, String> {
    if shape.len() < 2 {
        return Err("CUDA native matrix needs at least two logical dimensions".into());
    }
    let rows = dimension(product(&shape[..shape.len() - 1])?)?;
    let columns = dimension(product(&shape[shape.len() - 1..])?)?;
    let metadata = weight
        .components()
        .iter()
        .map(|component| (component.component_id(), component))
        .collect::<BTreeMap<_, _>>();
    let mut result = Vec::new();
    flatten(weight.physical_layout(), shape, 0, &metadata, &mut result)?;
    result.sort_unstable_by_key(|part| part.output_offset);
    let mut cursor = 0_u32;
    for part in &result {
        if part.output_offset != cursor || part.columns != columns {
            return Err("CUDA native matrix partitions overlap or leave a gap".into());
        }
        cursor = cursor
            .checked_add(part.rows)
            .ok_or("CUDA native matrix partition end overflows")?;
    }
    if cursor != rows {
        return Err("CUDA native matrix partitions do not cover its output".into());
    }
    Ok(result)
}

fn flatten(
    layout: &PhysicalWeightLayout,
    shape: &[u64],
    output_offset: u64,
    metadata: &BTreeMap<&WeightId, &ResolvedWeightComponentLayout>,
    parts: &mut Vec<MatrixPart>,
) -> Result<(), String> {
    if let PhysicalWeightLayout::Composite { parts: children } = layout {
        for child in children {
            if child.extents.len() != shape.len()
                || child.logical_offsets.len() != shape.len()
                || child.extents.last() != shape.last()
                || child.logical_offsets.last() != Some(&0)
            {
                return Err(
                    "CUDA native matrix partitions must preserve complete input rows".into(),
                );
            }
            let mut start = 0_u64;
            let mut end = 0_u64;
            for ((&offset, &extent), &dimension) in child
                .logical_offsets
                .iter()
                .zip(&child.extents)
                .zip(shape)
                .take(shape.len() - 1)
            {
                let limit = offset
                    .checked_add(extent)
                    .filter(|&end| extent != 0 && end <= dimension)
                    .ok_or("CUDA native matrix partition exceeds its logical extent")?;
                start = start
                    .checked_mul(dimension)
                    .and_then(|n| n.checked_add(offset))
                    .ok_or("CUDA native matrix partition offset overflows")?;
                end = end
                    .checked_mul(dimension)
                    .and_then(|n| n.checked_add(limit - 1))
                    .ok_or("CUDA native matrix partition end overflows")?;
            }
            if end.checked_sub(start).and_then(|n| n.checked_add(1))
                != Some(product(&child.extents[..shape.len() - 1])?)
            {
                return Err("CUDA native matrix partition rows are not contiguous".into());
            }
            flatten(
                &child.layout,
                &child.extents,
                output_offset
                    .checked_add(start)
                    .ok_or("CUDA native matrix output offset overflows")?,
                metadata,
                parts,
            )?;
        }
        return Ok(());
    }
    let (id, is_block) = match layout {
        PhysicalWeightLayout::Dense { component_id } => (component_id, false),
        PhysicalWeightLayout::Stored { component } => {
            exact_storage(&component.storage)?;
            (&component.component_id, false)
        }
        PhysicalWeightLayout::BlockQuantized {
            blocks,
            block_axis,
            block_padding,
        } if *block_axis as usize == shape.len() - 1
            && *block_padding == PhysicalWeightPadding::Exact =>
        {
            exact_storage(&blocks.storage)?;
            (&blocks.component_id, true)
        }
        _ => return Err("CUDA native matrix physical layout has no installed kernel".into()),
    };
    let component = metadata
        .get(id)
        .ok_or("CUDA native matrix references an absent component")?;
    let columns = *shape.last().unwrap();
    let rows = product(&shape[..shape.len() - 1])?;
    let (format, physical_columns) = match component.encoding() {
        WeightEncoding::Dense {
            element_type: ElementType::F16,
        } if !is_block => (MatrixFormat::DenseF16, columns),
        WeightEncoding::BlockQuantized(spec) if is_block => {
            let format = GgufBlockFormat::from_spec(spec)?;
            if !columns.is_multiple_of(format.block_values() as u64) {
                return Err("CUDA native matrix row ends inside a quantization block".into());
            }
            (
                MatrixFormat::Block(format),
                columns / format.block_values() as u64,
            )
        }
        _ => return Err("CUDA native matrix component encoding differs from its layout".into()),
    };
    if component.physical_dimensions().last() != Some(&physical_columns)
        || product(component.physical_dimensions())?
            != rows
                .checked_mul(physical_columns)
                .ok_or("CUDA native matrix physical element count overflows")?
    {
        return Err("CUDA native matrix physical dimensions differ from its logical rows".into());
    }
    parts.push(MatrixPart {
        component_id: id.clone(),
        format,
        rows: dimension(rows)?,
        columns: dimension(columns)?,
        output_offset: dimension(output_offset)?,
    });
    Ok(())
}

fn exact_storage(storage: &PhysicalStorageLayout) -> Result<(), String> {
    if matches!(
        storage,
        PhysicalStorageLayout::Contiguous {
            padding: PhysicalWeightPadding::Exact
        }
    ) {
        Ok(())
    } else {
        Err("CUDA native matrix requires exact contiguous component storage".into())
    }
}

#[cfg(test)]
mod tests;
