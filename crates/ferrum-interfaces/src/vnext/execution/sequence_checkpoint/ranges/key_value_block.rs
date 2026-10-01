//! Initialized paged key/value bytes: full blocks plus two packed tail rectangles.
use super::*;
use crate::vnext::{ElementType, ResolvedTensorLayout};

pub(super) fn source_copy_plan(
    tensor: &ResolvedTensorSpec,
    tokens_per_block: u64,
    key_pack_elements: u64,
    offset: u64,
    boundary: u64,
    logical_capacity: u64,
    physical_capacity: u64,
) -> Result<(Vec<Range<u64>>, Vec<StridedCopyRegion>), VNextError> {
    let [2, heads, dimension] = tensor.dimensions() else {
        return Err(invalid_plan(
            "blocked key/value prefix has invalid tensor geometry",
        ));
    };
    if tensor.element_type() != ElementType::F16
        || !matches!(tensor.layout(), ResolvedTensorLayout::Contiguous)
        || tokens_per_block == 0
        || key_pack_elements == 0
        || !dimension.is_multiple_of(key_pack_elements)
    {
        return Err(invalid_plan(
            "blocked key/value prefix has invalid element or block geometry",
        ));
    }
    let error = || invalid_plan("blocked key/value checkpoint geometry overflows u64");
    let mul = |a: u64, b: u64| a.checked_mul(b).ok_or_else(error);
    let add = |a: u64, b: u64| a.checked_add(b).ok_or_else(error);
    let element_bytes = tensor.element_type().size_bytes();
    let bytes_per_token = tensor.minimum_storage_bytes()?;
    if boundary == 0 || boundary > logical_capacity / bytes_per_token {
        return Err(invalid_plan(
            "blocked key/value prefix exceeds its proven token capacity",
        ));
    }
    let block_bytes = mul(bytes_per_token, tokens_per_block)?;
    let full_bytes = mul(boundary / tokens_per_block, block_bytes)?;
    let tail_tokens = boundary % tokens_per_block;
    let mut ranges = Vec::new();
    let mut strided = Vec::new();
    if full_bytes != 0 {
        let end = add(offset, full_bytes)?;
        if end > physical_capacity {
            return Err(invalid_plan(
                "full key/value blocks exceed physical capacity",
            ));
        }
        ranges.push(offset..end);
    }
    if tail_tokens != 0 {
        let key_width = mul(mul(tail_tokens, key_pack_elements)?, element_bytes)?;
        let key_rows = mul(*heads, dimension / key_pack_elements)?;
        let key_pitch = mul(mul(tokens_per_block, key_pack_elements)?, element_bytes)?;
        let value_width = mul(tail_tokens, element_bytes)?;
        let value_rows = mul(*heads, *dimension)?;
        let value_pitch = mul(tokens_per_block, element_bytes)?;
        let tail_base = add(offset, full_bytes)?;
        let value_base = add(tail_base, block_bytes / 2)?;
        let key = StridedCopyRegion::new(
            tail_base, full_bytes, key_width, key_rows, key_pitch, key_width,
        )?;
        let value = StridedCopyRegion::new(
            value_base,
            add(full_bytes, key.length_bytes()?)?,
            value_width,
            value_rows,
            value_pitch,
            value_width,
        )?;
        if key.source_end_bytes()? > physical_capacity
            || value.source_end_bytes()? > physical_capacity
        {
            return Err(invalid_plan(
                "tail key/value slots exceed physical capacity",
            ));
        }
        strided.extend([key, value]);
    }
    Ok((ranges, strided))
}

#[cfg(test)]
mod tests;
