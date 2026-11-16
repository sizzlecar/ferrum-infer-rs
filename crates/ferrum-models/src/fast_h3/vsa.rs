//! Checked geometry and mask lowering for FastH3's trained tile-128 VSA.
//!
//! These CPU metadata contracts are not an attention implementation or GPU
//! qualification. The numerical/provider gaps are recorded in `vsa/README.md`.
//! A descriptor describes independent batches with the same document geometry;
//! it never concatenates requests into a joint attention sequence.

use std::{mem::size_of, ops::Range};

use anyhow::{ensure, Context, Result};

use super::{
    FastH3Checkpoint, FastH3PackedLayout, FastH3Reference, FastH3Segment, FastH3SegmentKind,
    FastH3VideoGeometry,
};

mod block_lists;
mod key_center;
mod layout;
mod qkv_quantize;
pub use key_center::{FastH3KeyCenterConfig, FastH3KeyCenterFamily, FastH3KeyCenterLimits};
pub use qkv_quantize::{
    FastH3VsaQkvQuantizeConfig, FastH3VsaQkvQuantizeFamily, FastH3VsaQkvQuantizeLimits,
};
mod head_axis_pack;
pub use head_axis_pack::{
    FastH3HeadAxisPackConfig, FastH3HeadAxisPackFamily, FastH3HeadAxisPackLimits,
};
mod pooled_attention;
mod pooling;
pub use pooled_attention::{
    FastH3PooledAttentionConfig, FastH3PooledAttentionFamily, FastH3PooledAttentionLimits,
};
mod prefix_attention;
mod row_transport;
pub use pooling::{
    FastH3VsaPoolConfig, FastH3VsaPoolFamily, FastH3VsaPoolLimits, FastH3VsaPoolSource,
};
pub use prefix_attention::{
    FastH3PrefixAttentionArithmetic, FastH3PrefixAttentionConfig, FastH3PrefixAttentionFamily,
    FastH3PrefixAttentionLimits, FastH3PrefixAttentionSource,
};
pub use row_transport::{
    FastH3VsaRowTransport, FastH3VsaRowTransportConfig, FastH3VsaRowTransportFamily,
    FastH3VsaRowTransportLimits, FastH3VsaRowTransportSource,
};

pub use block_lists::{FastH3VsaBlockLists, FastH3VsaMaskLimits};

/// The fixed checkpoint's `[t,h,w]` token tile, after `[1,2,2]` patching.
pub const FAST_H3_VSA_TILE_SHAPE: [usize; 3] = [4, 4, 8];
pub const FAST_H3_VSA_TILE_ROWS: usize = 128;
/// A tile-buffer row that must be initialized to positive zero, never gathered.
pub const FAST_H3_VSA_PADDING_ROW: u32 = u32::MAX;

/// Caller-declared logical metadata limits, checked before materializing maps.
/// These bound Rust metadata payloads, not allocator overhead or device memory.
#[derive(Clone, Copy, Debug)]
pub struct FastH3VsaLayoutLimits {
    pub maximum_packed_rows: usize,
    pub maximum_tiles: usize,
    pub maximum_metadata_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FastH3VsaRegionKind {
    ReferenceVideo,
    TargetVideo,
}

/// A sparse video region; image/audio references belong to the dense prefix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FastH3VsaRegion {
    kind: FastH3VsaRegionKind,
    tiles: Range<usize>,
    trained_key_tiles: usize,
}
impl FastH3VsaRegion {
    pub fn kind(&self) -> FastH3VsaRegionKind {
        self.kind
    }
    pub fn tiles(&self) -> Range<usize> {
        self.tiles.clone()
    }
    pub fn trained_key_tiles(&self) -> usize {
        self.trained_key_tiles
    }
}

/// Dense steps/layers override every region's sparsity, including references.
/// The mode does not change FP4 probability quantization or prefix precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FastH3VsaSelection {
    Trained,
    Dense,
}

/// Private fields prevent forging a map into padding or another document.
#[derive(Clone, Debug)]
pub struct FastH3VsaLayout {
    packed: FastH3PackedLayout,
    prefix_tiles: usize,
    regions: Vec<FastH3VsaRegion>,
    valid_rows: Vec<u16>,
    packed_to_tile: Vec<u32>,
    tile_to_packed: Vec<u32>,
    metadata_bytes: usize,
}
impl FastH3VsaLayout {
    pub fn packed(&self) -> &FastH3PackedLayout {
        &self.packed
    }
    pub fn prefix_tiles(&self) -> usize {
        self.prefix_tiles
    }
    pub fn regions(&self) -> &[FastH3VsaRegion] {
        &self.regions
    }
    pub fn tile_count(&self) -> usize {
        self.valid_rows.len()
    }
    pub fn padded_rows(&self) -> usize {
        self.tile_to_packed.len()
    }
    /// Each tile's valid rows are contiguous at its beginning, even when the
    /// corresponding spatial border has holes in the original raster order.
    pub fn valid_rows_per_tile(&self) -> &[u16] {
        &self.valid_rows
    }
    /// The upstream `untile_combined_index`: packed row -> non-padding slot.
    pub fn packed_to_tile_rows(&self) -> &[u32] {
        &self.packed_to_tile
    }
    /// Tile slot -> packed row, or [`FAST_H3_VSA_PADDING_ROW`].
    pub fn tile_to_packed_rows(&self) -> &[u32] {
        &self.tile_to_packed
    }
    /// Peak logical metadata payload, including temporary segment geometry.
    pub fn metadata_bytes(&self) -> usize {
        self.metadata_bytes
    }
}

fn product(values: &[usize], context: &'static str) -> Result<usize> {
    values
        .iter()
        .try_fold(1usize, |n, &v| n.checked_mul(v))
        .context(context)
}

fn filled<T: Clone>(len: usize, value: T) -> Result<Vec<T>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(len)
        .context("VSA metadata allocation failed")?;
    result.resize(len, value);
    Ok(result)
}

#[cfg(test)]
mod tests;
