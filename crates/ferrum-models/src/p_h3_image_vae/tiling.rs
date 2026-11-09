//! Single-frame spatial tiling from pinned ComfyUI 56f179 VAE split_tiles.
//! Pixel geometry is explicit; splitting never silently resizes the canvas.
use super::latent_statistics::PH3ImageVaeLatentStatistics;
use anyhow::{ensure, Context, Result};
use ferrum_runtime::{
    ElementType, HostMemoryAccount, HostMemoryClass, ManagedHostBuffer, ManagedHostBytes,
    ManagedHostTensor, ProgramValueId, ResolvedTensorLayout, ResolvedTensorSpec,
};

mod canvas;
mod groups;
mod moments;
pub(crate) use canvas::{PH3ImageVaeCanvas, PH3ImageVaeCanvasDomain};
pub use groups::{PH3ImageVaeTileBatch, PH3ImageVaeTileGroup};
pub(crate) use moments::PH3ImageVaeMomentCanvas;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PH3ImageVaeSpatialPolicy {
    WholeImage,
    Tiled {
        tile_pixels: usize,
        minimum_overlap_pixels: usize,
        max_tiles: usize,
    },
}
impl PH3ImageVaeSpatialPolicy {
    pub fn comfy_default(max_tiles: usize) -> Self {
        Self::Tiled {
            tile_pixels: 256,
            minimum_overlap_pixels: 64,
            max_tiles,
        }
    }
}

#[derive(Clone, Debug)]
struct Axis {
    length: usize,
    starts: Vec<usize>,
    extent: usize,
    overlaps: Vec<usize>,
}
impl Axis {
    fn count(length: usize, tile: usize, overlap: usize) -> Result<usize> {
        if length <= tile {
            return Ok(1);
        }
        let step = tile
            .checked_sub(overlap)
            .filter(|&n| n > 0)
            .context("tile overlap must leave a positive step")?;
        Ok((length - overlap).div_ceil(step))
    }
    fn new(length: usize, tile: usize, overlap: usize, scale: usize, count: usize) -> Result<Self> {
        if count == 1 {
            return Ok(Self {
                length,
                starts: vec![0],
                extent: length,
                overlaps: vec![],
            });
        }
        let edges = count - 1;
        let capacity = tile
            .checked_mul(count)
            .and_then(|v| v.checked_sub(overlap.checked_mul(edges)?))
            .context("tile coverage overflows")?;
        let extra = capacity
            .checked_sub(length)
            .context("tile coverage is short")?
            / scale;
        let overlaps = (0..edges)
            .map(|i| overlap + (extra / edges + usize::from(i < extra % edges)) * scale)
            .collect::<Vec<_>>();
        ensure!(
            overlaps
                .iter()
                .all(|&v| v > 0 && v < tile && v % scale == 0),
            "tile overlap must preserve a nonempty aligned advance"
        );
        let mut starts = Vec::with_capacity(count);
        starts.push(0);
        for &overlap in &overlaps {
            starts.push(starts.last().unwrap() + tile - overlap);
        }
        ensure!(
            starts.last().unwrap().checked_add(tile) == Some(length),
            "tile coverage must exactly equal the declared canvas"
        );
        Ok(Self {
            length,
            starts,
            extent: tile,
            overlaps,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PH3ImageVaeTile {
    pub index: usize,
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    pub overlap_left: usize,
    pub overlap_top: usize,
    pub trim_right: usize,
    pub trim_bottom: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct PH3ImageVaeTilePlan {
    x: Axis,
    y: Axis,
    scale: usize,
    count: usize,
}
impl PH3ImageVaeTilePlan {
    pub fn new(
        width: usize,
        height: usize,
        spatial_scale: usize,
        policy: PH3ImageVaeSpatialPolicy,
    ) -> Result<Self> {
        ensure!(
            width > 0
                && height > 0
                && spatial_scale > 0
                && width % spatial_scale == 0
                && height % spatial_scale == 0,
            "VAE canvas must have positive exactly aligned spatial dimensions"
        );
        let (tile, overlap, maximum) = match policy {
            PH3ImageVaeSpatialPolicy::WholeImage => (width.max(height), 0, 1),
            PH3ImageVaeSpatialPolicy::Tiled {
                tile_pixels,
                minimum_overlap_pixels,
                max_tiles,
            } => {
                ensure!(
                    tile_pixels > 0
                        && tile_pixels % spatial_scale == 0
                        && minimum_overlap_pixels > 0
                        && minimum_overlap_pixels < tile_pixels
                        && minimum_overlap_pixels % spatial_scale == 0
                        && max_tiles > 0,
                    "invalid tiled VAE policy or capacity"
                );
                (tile_pixels, minimum_overlap_pixels, max_tiles)
            }
        };
        let nx = Axis::count(width, tile, overlap)?;
        let ny = Axis::count(height, tile, overlap)?;
        let count = nx
            .checked_mul(ny)
            .filter(|&n| n <= maximum)
            .context("VAE tile count exceeds declared capacity")?;
        nx.checked_add(ny)
            .and_then(|v| v.checked_mul(2 * std::mem::size_of::<usize>()))
            .filter(|&n| n <= isize::MAX as usize)
            .context("tile metadata exceeds address space")?;
        // No coordinate arrays are allocated until the complete declared tile
        // count and address extents have passed admission.
        Ok(Self {
            x: Axis::new(width, tile, overlap, spatial_scale, nx)?,
            y: Axis::new(height, tile, overlap, spatial_scale, ny)?,
            scale: spatial_scale,
            count,
        })
    }
    pub fn dimensions(&self) -> [usize; 2] {
        [self.y.length, self.x.length]
    }
    pub fn spatial_scale(&self) -> usize {
        self.scale
    }
    pub fn tile_count(&self) -> usize {
        self.count
    }
    pub fn tile(&self, index: usize) -> Result<PH3ImageVaeTile> {
        ensure!(index < self.count, "tile index exceeds canvas plan");
        let row = index / self.x.starts.len();
        let column = index % self.x.starts.len();
        Ok(PH3ImageVaeTile {
            index,
            x: self.x.starts[column],
            y: self.y.starts[row],
            width: self.x.extent,
            height: self.y.extent,
            overlap_left: column.checked_sub(1).map_or(0, |i| self.x.overlaps[i]),
            overlap_top: row.checked_sub(1).map_or(0, |i| self.y.overlaps[i]),
            trim_right: self.x.overlaps.get(column).copied().unwrap_or(0),
            trim_bottom: self.y.overlaps.get(row).copied().unwrap_or(0),
        })
    }

    /// Extract adjacent tiles of one row directly from immutable normalized
    /// NCHW sampler storage into denormalized [tiles*B*TH*TW,C]. Tiles precede
    /// requests in batch order, matching the pinned decoder microbatch layout.
    /// No temporary full-resolution or tile NCHW copy is created.
    pub fn managed_tile_rows(
        &self,
        host: &HostMemoryAccount,
        normalized: &ManagedHostTensor,
        statistics: &PH3ImageVaeLatentStatistics<'_>,
        first_tile: usize,
        tile_count: usize,
        output_id: ProgramValueId,
    ) -> Result<ManagedHostTensor> {
        ensure!(tile_count > 0, "decoder microbatch must contain a tile");
        let tile = self.tile(first_tile)?;
        ensure!(
            tile_count <= self.x.starts.len() - first_tile % self.x.starts.len(),
            "decoder microbatch must stay within one tile row"
        );
        let [batch, channels, height, width] = normalized.tensor().dimensions() else {
            anyhow::bail!("VAE sampler input must be NCHW");
        };
        ensure!(
            *batch > 0
                && *channels == statistics.channels() as u64
                && *height == (self.y.length / self.scale) as u64
                && *width == (self.x.length / self.scale) as u64
                && normalized.tensor().element_type() == ElementType::F32
                && host.owns(normalized.bytes())
                && normalized.bytes().class() == HostMemoryClass::InputPreprocess,
            "VAE sampler input differs from planned geometry, statistics or Host owner"
        );
        let (batch, channels, height, width) = (
            usize::try_from(*batch)?,
            usize::try_from(*channels)?,
            usize::try_from(*height)?,
            usize::try_from(*width)?,
        );
        let (th, tw) = (tile.height / self.scale, tile.width / self.scale);
        let rows = tile_count
            .checked_mul(batch)
            .and_then(|v| v.checked_mul(th))
            .and_then(|v| v.checked_mul(tw))
            .context("tile batch extent overflows")?;
        let tensor = f32_spec(&[rows, channels])?;
        let mut output =
            host.allocate_zeroed(HostMemoryClass::InputPreprocess, byte_len(&tensor)?)?;
        for n in 0..tile_count {
            let tile = self.tile(first_tile + n)?;
            for b in 0..batch {
                for y in 0..th {
                    for x in 0..tw {
                        for c in 0..channels {
                            let src = (((b * channels + c) * height + tile.y / self.scale + y)
                                * width
                                + tile.x / self.scale
                                + x)
                                * 4;
                            let value =
                                statistics.denormalize(c, read(normalized.bytes().bytes(), src))?;
                            let dst = ((((n * batch + b) * th + y) * tw + x) * channels + c) * 4;
                            output.as_mut()[dst..dst + 4].copy_from_slice(&value.to_le_bytes());
                        }
                    }
                }
            }
        }
        Ok(ManagedHostTensor::new(output_id, tensor, output.freeze())?)
    }
}
fn f32_spec(dimensions: &[usize]) -> Result<ResolvedTensorSpec> {
    let tensor = ResolvedTensorSpec::new(
        dimensions.iter().map(|&n| n as u64).collect(),
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )?;
    byte_len(&tensor)?;
    Ok(tensor)
}
fn byte_len(tensor: &ResolvedTensorSpec) -> Result<usize> {
    let bytes = usize::try_from(tensor.minimum_storage_bytes()?)?;
    ensure!(
        bytes > 0 && bytes <= isize::MAX as usize,
        "VAE tile tensor exceeds Host address space"
    );
    Ok(bytes)
}
fn read(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

#[cfg(test)]
mod tests;
