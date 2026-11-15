//! Checked model layout -> original charged Host region owners. This payload
//! remains separate from Stored weights and must be bound to the native factory.
use super::*;
use ferrum_kernels::native_ops::sage_vsa_fp4::{CheckedVsaRegions, ListShape};
use std::sync::Arc;

pub(super) fn prepare(
    layout: &FastH3VsaLayout,
    batches: u64,
    heads: u64,
    selection: FastH3VsaSelection,
    host: &HostMemoryAccount,
    maximum_bytes: u64,
) -> Result<Arc<CheckedVsaRegions>, VNextError> {
    let prefix = layout.prefix_tiles();
    let tiles = layout.tile_count();
    if prefix == 0 || prefix >= tiles {
        return Err(invalid(
            "dynamic regions require nonempty prefix and video suffix",
        ));
    }
    let shape = ListShape::new(
        batches.try_into().map_err(invalid)?,
        heads.try_into().map_err(invalid)?,
        (tiles - prefix).try_into().map_err(invalid)?,
        tiles.try_into().map_err(invalid)?,
    )
    .map_err(invalid)?;
    let region_bytes = layout
        .regions()
        .len()
        .checked_mul(12)
        .ok_or_else(|| invalid("region bytes overflow"))?;
    let half_bytes = tiles
        .checked_mul(8)
        .ok_or_else(|| invalid("half-valid bytes overflow"))?;
    if region_bytes
        .checked_add(half_bytes)
        .is_none_or(|n| n as u64 > maximum_bytes)
    {
        return Err(invalid("region metadata exceeds admitted payload capacity"));
    }
    let mut regions = host
        .allocate_zeroed(HostMemoryClass::TransferStaging, region_bytes)
        .map_err(invalid)?;
    for (record, region) in regions.as_mut().chunks_exact_mut(12).zip(layout.regions()) {
        let range = region.tiles();
        let keep = match selection {
            FastH3VsaSelection::Trained => region.trained_key_tiles(),
            FastH3VsaSelection::Dense => range.len(),
        };
        for (chunk, value) in record
            .chunks_exact_mut(4)
            .zip([range.start, range.end, keep])
        {
            chunk.copy_from_slice(&i32::try_from(value).map_err(invalid)?.to_le_bytes());
        }
    }
    let mut halves = host
        .allocate_zeroed(HostMemoryClass::TransferStaging, half_bytes)
        .map_err(invalid)?;
    for (record, &count) in halves
        .as_mut()
        .chunks_exact_mut(8)
        .zip(layout.valid_rows_per_tile())
    {
        record[..4].copy_from_slice(&i32::from(count.min(64)).to_le_bytes());
        record[4..].copy_from_slice(&i32::from(count.saturating_sub(64)).to_le_bytes());
    }
    Ok(Arc::new(
        CheckedVsaRegions::new(shape, regions.freeze(), halves.freeze()).map_err(invalid)?,
    ))
}
