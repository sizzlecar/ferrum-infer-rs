//! Original video-decoder spatial attention domains and raw-neighbour stitching.
//! The decoder session/source is reused; each tile has its own local rotary grid.
use super::{
    FastH3VideoClipFamily, FastH3VideoDecodePlan, FastH3VideoRawClip, FastH3VideoVaeConfig,
};
use crate::p_h3_image_vae::tiling::{PH3ImageVaeSpatialPolicy, PH3ImageVaeTilePlan};
use crate::spatial_tiles::{SpatialTileBatch, SpatialTileGroup, SpatialTileGroups};
use anyhow::{ensure, Context, Result};
use ferrum_diffusion::SpatiotemporalLatentShape;
use ferrum_runtime::*;
use half::bf16;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FastH3VideoLatentUnits {
    #[default]
    Normalized,
    Decoder,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FastH3VideoSpatialPolicy {
    FastVideoTiled {
        maximum_tiles: usize,
    },
    /// Whole-image attention and global coordinates are a distinct diagnostic.
    WholeSpatialDiagnostic,
}
impl Default for FastH3VideoSpatialPolicy {
    fn default() -> Self {
        Self::FastVideoTiled {
            maximum_tiles: 1024,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FastH3VideoSpatialPlan {
    architecture: FastH3VideoVaeConfig,
    shape: [usize; 5],
    tiles: PH3ImageVaeTilePlan,
    maximum_bytes: usize,
}
impl FastH3VideoSpatialPlan {
    pub const BLEND_PROFILE_ID:&'static str="fasth3.video.tiles256_overlap64.raw_neighbours.vertical_then_horizontal.bf16_staged.scalar_reciprocal.v1";
    pub fn new(
        architecture: &FastH3VideoVaeConfig,
        shape: [usize; 5],
        policy: FastH3VideoSpatialPolicy,
        maximum_bytes: usize,
    ) -> Result<Self> {
        architecture.validate()?;
        ensure!(
            !shape.contains(&0) && shape[1] == architecture.latent_channels(),
            "spatial decoder latent shape differs"
        );
        let bytes = extent(shape, 4)?;
        ensure!(
            bytes <= maximum_bytes,
            "latent input exceeds spatial capacity"
        );
        let ratio = architecture.spatial_compression_ratio()?;
        let h = shape[3]
            .checked_mul(ratio)
            .context("pixel height overflow")?;
        let w = shape[4]
            .checked_mul(ratio)
            .context("pixel width overflow")?;
        let policy = match policy {
            FastH3VideoSpatialPolicy::FastVideoTiled { maximum_tiles } => {
                PH3ImageVaeSpatialPolicy::Tiled {
                    tile_pixels: 256,
                    minimum_overlap_pixels: 64,
                    max_tiles: maximum_tiles,
                }
            }
            FastH3VideoSpatialPolicy::WholeSpatialDiagnostic => {
                PH3ImageVaeSpatialPolicy::WholeImage
            }
        };
        Ok(Self {
            architecture: architecture.clone(),
            shape,
            tiles: PH3ImageVaeTilePlan::new(w, h, ratio, policy)?,
            maximum_bytes,
        })
    }
    pub fn tile_count(&self) -> usize {
        self.tiles.tile_count()
    }
    /// Output backing simultaneously retained by a spatial decode, including
    /// the canvas, one row of original neighbours, the current Core readback,
    /// its unpacked copy and the trimmed blend staging. The caller separately
    /// accounts for the final RGB artifact and prior temporal overlap.
    /// This is metadata only; every allocation is aligned independently.
    pub fn output_backing_bytes(
        &self,
        raw_frames: usize,
        host: &HostMemoryAccount,
    ) -> Result<usize> {
        self.grouped_output_backing_bytes(raw_frames, SpatialTileBatch::default(), host)
    }
    /// One grouped Core readback remains live while each tile is unpacked into
    /// independent backing. Raw neighbours therefore retain individual tiles,
    /// never an implicitly larger group allocation.
    pub fn grouped_output_backing_bytes(
        &self,
        raw_frames: usize,
        batch: SpatialTileBatch,
        host: &HostMemoryAccount,
    ) -> Result<usize> {
        ensure!(raw_frames > 0, "empty spatial clip");
        let [height, width] = self.tiles.dimensions();
        let canvas =
            host.backing_bytes(extent([self.shape[0], 3, raw_frames, height, width], 2)?)?;
        let columns = (0..self.tile_count())
            .take_while(|&i| self.tiles.tile(i).is_ok_and(|t| t.y == 0))
            .count();
        let mut row = vec![0usize; columns];
        let mut retained = 0usize;
        let mut maximum = canvas;
        let mut first = 0;
        while first < self.tile_count() {
            let group = self.tile_group(first, batch)?;
            let tile = self.tiles.tile(first)?;
            let readback = host.backing_bytes(extent(
                [
                    self.shape[0],
                    group.tile_count(),
                    3,
                    raw_frames,
                    tile.height,
                    tile.width,
                ],
                2,
            )?)?;
            for index in first..first + group.tile_count() {
                let tile = self.tiles.tile(index)?;
                let raw = host.backing_bytes(extent(
                    [self.shape[0], 3, raw_frames, tile.height, tile.width],
                    2,
                )?)?;
                let staged = host.backing_bytes(extent(
                    [
                        self.shape[0],
                        3,
                        raw_frames,
                        tile.height - tile.trim_bottom,
                        tile.width - tile.trim_right,
                    ],
                    2,
                )?)?;
                let live = [canvas, retained, readback, raw, staged]
                    .into_iter()
                    .try_fold(0usize, |n, bytes| n.checked_add(bytes))
                    .context("spatial Output admission overflow")?;
                maximum = maximum.max(live);
                let column = index % columns;
                retained = retained
                    .checked_sub(row[column])
                    .and_then(|n| n.checked_add(raw))
                    .context("spatial retained row overflow")?;
                row[column] = raw;
            }
            first += group.tile_count();
        }
        Ok(maximum)
    }
    pub fn tile_groups(&self, batch: SpatialTileBatch) -> SpatialTileGroups {
        batch
            .groups(self.tile_count())
            .expect("validated nonempty video tile plan")
    }
    /// Largest single grouped readback. An overlapped CPU consumer can retain
    /// the preceding readback while the serial device lane produces this one.
    /// This is a physical Host bound; it does not allocate or grant credit.
    pub fn maximum_grouped_readback_backing_bytes(
        &self,
        raw_frames: usize,
        batch: SpatialTileBatch,
        host: &HostMemoryAccount,
    ) -> Result<usize> {
        ensure!(raw_frames > 0, "empty spatial clip");
        let mut maximum = 0;
        for first in (0..self.tile_count()).step_by(batch.maximum_tiles()) {
            let group = self.tile_group(first, batch)?;
            let tile = self.tiles.tile(first)?;
            maximum = maximum.max(host.backing_bytes(extent(
                [
                    self.shape[0],
                    group.tile_count(),
                    3,
                    raw_frames,
                    tile.height,
                    tile.width,
                ],
                2,
            )?)?);
        }
        Ok(maximum)
    }
    pub fn tile_group(&self, first: usize, batch: SpatialTileBatch) -> Result<SpatialTileGroup> {
        self.tile_groups(batch).group(first)
    }
    pub fn tile_group_shapes(&self, batch: SpatialTileBatch) -> Vec<usize> {
        self.tile_groups(batch).shapes()
    }
    /// Uniform real groups retain one exact compiled shape and scalar fallback.
    pub fn uniform_tile_group_candidates(
        &self,
        maximum: SpatialTileBatch,
    ) -> Vec<SpatialTileBatch> {
        self.tile_groups(maximum).uniform_batches().collect()
    }
    /// Leading decoder items are [tile, request]; time stays inside each item.
    pub fn grouped_clip_grid(&self, latent_count: usize, tiles: usize) -> Result<[u64; 4]> {
        ensure!(
            tiles > 0 && tiles <= self.tile_count(),
            "invalid decoder tile group"
        );
        let mut grid = self.clip_grid(latent_count)?;
        grid[0] = grid[0]
            .checked_mul(tiles as u64)
            .context("grouped decoder batch overflow")?;
        Ok(grid)
    }
    pub fn clip_grid(&self, latent_count: usize) -> Result<[u64; 4]> {
        let tile = self.tiles.tile(0)?;
        let ratio = self.tiles.spatial_scale();
        Ok([
            self.shape[0] as u64,
            latent_count as u64,
            (tile.height / ratio) as u64,
            (tile.width / ratio) as u64,
        ])
    }
    /// Directly crop a temporal window and spatial tile from original F32
    /// NCTHW storage, then pack N,(T H W),C. Denormalization is per channel.
    pub fn managed_tile_input(
        &self,
        temporal: &FastH3VideoDecodePlan,
        chunk: usize,
        tile: usize,
        units: FastH3VideoLatentUnits,
        source: &ManagedHostBytes,
        host: &HostMemoryAccount,
    ) -> Result<ManagedHostTensor> {
        self.managed_tile_group_input(
            temporal,
            chunk,
            self.tile_group(tile, SpatialTileBatch::default())?,
            units,
            source,
            host,
        )
    }
    /// Pack real consecutive tiles without padding. The caller executes the
    /// exact group shape and later separates its leading [tile,request] axis.
    pub fn managed_tile_group_input(
        &self,
        temporal: &FastH3VideoDecodePlan,
        chunk: usize,
        group: SpatialTileGroup,
        units: FastH3VideoLatentUnits,
        source: &ManagedHostBytes,
        host: &HostMemoryAccount,
    ) -> Result<ManagedHostTensor> {
        ensure!(
            temporal.latent_frames() == self.shape[2]
                && host.owns(source)
                && source.class() == HostMemoryClass::InputPreprocess
                && source.len() == extent(self.shape, 4)?,
            "spatial latent source/temporal geometry/Host parent differs"
        );
        let temporal = temporal.chunk(chunk)?;
        let tiles = (group.first_tile()..group.first_tile() + group.tile_count())
            .map(|index| self.tiles.tile(index))
            .collect::<Result<Vec<_>>>()?;
        let tile = tiles.first().context("empty decoder tile group")?;
        ensure!(
            tiles
                .iter()
                .all(|t| t.width == tile.width && t.height == tile.height),
            "decoder tile group has mixed spatial shapes"
        );
        let ratio = self.tiles.spatial_scale();
        let [batch, channels, time, height, width] = self.shape;
        let th = tile.height / ratio;
        let tw = tile.width / ratio;
        let shape = SpatiotemporalLatentShape::new(
            [
                batch
                    .checked_mul(tiles.len())
                    .context("grouped decoder batch overflow")?,
                channels,
                temporal.latent_count,
                th,
                tw,
            ],
            [1, 1, 1],
        )?;
        ensure!(
            shape.byte_len::<f32>()? <= self.maximum_bytes,
            "tile input exceeds capacity"
        );
        let mut out =
            host.allocate_zeroed(HostMemoryClass::InputPreprocess, shape.byte_len::<f32>()?)?;
        let mut finite = true;
        let source = source.bytes();
        let destination = out.as_mut();
        shape.for_each_packed_index(|p, d| {
            let x = d % tw;
            let y = d / tw % th;
            let t = d / (tw * th) % temporal.latent_count;
            let c = d / (tw * th * temporal.latent_count) % channels;
            let item = d / (tw * th * temporal.latent_count * channels);
            let tile = &tiles[item / batch];
            let b = item % batch;
            let source_t = (temporal.latent_start + t).min(time - 1);
            let index = ((((b * channels + c) * time + source_t) * height + tile.y / ratio + y)
                * width
                + tile.x / ratio
                + x)
                * 4;
            let value = f32::from_le_bytes(source[index..index + 4].try_into().unwrap());
            let value = match units {
                FastH3VideoLatentUnits::Decoder => value,
                FastH3VideoLatentUnits::Normalized => {
                    let scaled = value * self.architecture.latent_std()[c] as f32;
                    scaled + self.architecture.latent_mean()[c] as f32
                }
            };
            finite &= value.is_finite();
            destination[p * 4..p * 4 + 4].copy_from_slice(&value.to_le_bytes());
        });
        ensure!(finite, "nonfinite spatial latent or denormalization");
        Ok(ManagedHostTensor::new(
            ProgramValueId::new(FastH3VideoClipFamily::INPUT_ID)?,
            ResolvedTensorSpec::new(
                vec![
                    shape.element_count() as u64 / channels as u64,
                    channels as u64,
                ],
                ElementType::F32,
                ResolvedTensorLayout::Contiguous,
            )?,
            out.freeze(),
        )?)
    }
    pub fn begin_canvas(
        &self,
        raw_frames: usize,
        host: &HostMemoryAccount,
    ) -> Result<FastH3VideoSpatialCanvas> {
        ensure!(raw_frames > 0, "empty spatial clip");
        let [h, w] = self.tiles.dimensions();
        let shape = [self.shape[0], 3, raw_frames, h, w];
        let bytes = extent(shape, 2)?;
        ensure!(
            bytes <= self.maximum_bytes,
            "raw spatial canvas exceeds capacity"
        );
        let columns = (0..self.tile_count())
            .take_while(|&i| self.tiles.tile(i).is_ok_and(|t| t.y == 0))
            .count();
        Ok(FastH3VideoSpatialCanvas {
            plan: self.tiles.clone(),
            host: host.clone(),
            shape,
            canvas: host.allocate_zeroed(HostMemoryClass::Output, bytes)?,
            raw_row: (0..columns).map(|_| None).collect(),
            next: 0,
            maximum_bytes: self.maximum_bytes,
        })
    }
}

/// Keeps only one row of original readbacks. Blended pixels never become a
/// neighbour. All staging is admitted before publication; allocation failure
/// or an invalid tile leaves the prior canvas/progress untouched.
pub struct FastH3VideoSpatialCanvas {
    plan: PH3ImageVaeTilePlan,
    host: HostMemoryAccount,
    shape: [usize; 5],
    canvas: ManagedHostBuffer,
    raw_row: Vec<Option<ManagedHostBytes>>,
    next: usize,
    maximum_bytes: usize,
}
impl FastH3VideoSpatialCanvas {
    pub fn completed_tiles(&self) -> usize {
        self.next
    }
    pub fn push(&mut self, index: usize, raw: &FastH3VideoRawClip) -> Result<()> {
        ensure!(index == self.next, "out-of-order spatial tile");
        let tile = self.plan.tile(index)?;
        let [batch, _, time, height, width] = self.shape;
        let shape = [batch, 3, time, tile.height, tile.width];
        ensure!(
            raw.ncthw_shape() == shape
                && self.host.owns(raw.bytes())
                && raw.bytes().class() == HostMemoryClass::Output
                && raw.bytes().len() == extent(shape, 2)?,
            "raw spatial output shape/owner differs"
        );
        ensure!(
            raw.bytes()
                .bytes()
                .chunks_exact(2)
                .all(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).is_finite()),
            "nonfinite spatial tile"
        );
        let column = index % self.raw_row.len();
        let top = if tile.overlap_top > 0 {
            Some(
                self.raw_row[column]
                    .as_ref()
                    .context("missing raw upper tile")?,
            )
        } else {
            None
        };
        let left = if tile.overlap_left > 0 {
            Some(
                self.raw_row[column - 1]
                    .as_ref()
                    .context("missing raw left tile")?,
            )
        } else {
            None
        };
        let keep_h = tile.height - tile.trim_bottom;
        let keep_w = tile.width - tile.trim_right;
        let size = extent([batch, 3, time, keep_h, keep_w], 2)?;
        ensure!(
            size <= self.maximum_bytes,
            "spatial staging exceeds capacity"
        );
        let mut staged = self.host.allocate_zeroed(HostMemoryClass::Output, size)?;
        // Coefficients depend only on the overlap coordinate, not on the
        // request, frame, channel or pixel in the other axis. Compute the
        // original separately rounded BF16 coefficients once per coordinate.
        let vertical = (0..keep_h.min(tile.overlap_top))
            .map(|y| SpatialBlendWeights::new(y, tile.overlap_top))
            .collect::<Vec<_>>();
        let horizontal = (0..keep_w.min(tile.overlap_left))
            .map(|x| SpatialBlendWeights::new(x, tile.overlap_left))
            .collect::<Vec<_>>();
        // Accumulate failures without an early-return branch in the pixel
        // loops. Include the vertical intermediate even if horizontal blending
        // would overwrite it; the canvas remains untouched until validation.
        let mut nonfinite = false;
        let raw_bytes = raw.bytes().bytes();
        let top = top.map(ManagedHostBytes::bytes);
        let left = left.map(ManagedHostBytes::bytes);
        let staged_bytes = staged.as_mut();
        for plane in 0..batch * 3 * time {
            for y in 0..keep_h {
                let source_start = 2 * (plane * tile.height + y) * tile.width;
                let source = &raw_bytes[source_start..source_start + keep_w * 2];
                let destination_start = 2 * (plane * keep_h + y) * keep_w;
                let destination =
                    &mut staged_bytes[destination_start..destination_start + keep_w * 2];
                if let Some(weights) = vertical.get(y) {
                    let upper_start =
                        2 * (plane * tile.height + tile.height - tile.overlap_top + y) * tile.width;
                    let upper = &top.unwrap()[upper_start..upper_start + keep_w * 2];
                    for ((out, a), b) in destination
                        .chunks_exact_mut(2)
                        .zip(upper.chunks_exact(2))
                        .zip(source.chunks_exact(2))
                    {
                        let value = weights.apply(word(a), word(b));
                        nonfinite |= value & 0x7f80 == 0x7f80;
                        out.copy_from_slice(&value.to_le_bytes());
                    }
                } else {
                    destination.copy_from_slice(source);
                }
                // The horizontal pass consumes the rounded vertical result
                // and the original left neighbour, preserving eager order.
                if let Some(left) = left {
                    let left_start = source_start + 2 * (tile.width - tile.overlap_left);
                    let left = &left[left_start..left_start + horizontal.len() * 2];
                    for ((out, a), weights) in destination
                        .chunks_exact_mut(2)
                        .zip(left.chunks_exact(2))
                        .zip(&horizontal)
                    {
                        let value = weights.apply(word(a), word(out));
                        nonfinite |= value & 0x7f80 == 0x7f80;
                        out.copy_from_slice(&value.to_le_bytes());
                    }
                }
            }
        }
        ensure!(!nonfinite, "BF16 spatial blend overflow");
        // No fallible work remains once the canvas is modified.
        let source = staged.as_ref();
        let canvas = self.canvas.as_mut();
        for plane in 0..batch * 3 * time {
            for y in 0..keep_h {
                let src = 2 * (plane * keep_h + y) * keep_w;
                let dst = 2 * ((plane * height + tile.y + y) * width + tile.x);
                canvas[dst..dst + keep_w * 2].copy_from_slice(&source[src..src + keep_w * 2]);
            }
        }
        self.raw_row[column] = Some(raw.bytes().clone());
        self.next += 1;
        Ok(())
    }
    pub fn finish(self) -> Result<FastH3VideoRawClip> {
        ensure!(
            self.next == self.plan.tile_count(),
            "spatial canvas incomplete"
        );
        Ok(FastH3VideoRawClip {
            shape: self.shape,
            bytes: self.canvas.freeze(),
        })
    }
}
fn extent<const N: usize>(shape: [usize; N], bytes: usize) -> Result<usize> {
    shape
        .into_iter()
        .try_fold(bytes, |n, d| n.checked_mul(d))
        .filter(|n| *n <= isize::MAX as usize)
        .context("spatial extent overflow")
}
fn word(bytes: &[u8]) -> u16 {
    u16::from_le_bytes(bytes.try_into().unwrap())
}

struct SpatialBlendWeights {
    previous: f32,
    current: f32,
}
impl SpatialBlendWeights {
    fn new(position: usize, extent: usize) -> Self {
        // TensorIterator scalar division: F64 reciprocal -> opmath F32
        // multiply, then each eager BF16 tensor stage rounds independently.
        let pos = bf16::from_f32(position as f32).to_f32();
        let inverse = (1f64 / extent as f64) as f32;
        let current = bf16::from_f32(pos * inverse).to_f32();
        let previous = bf16::from_f32(1f32 - current).to_f32();
        Self { previous, current }
    }
    fn apply(&self, a: u16, b: u16) -> u16 {
        let a = bf16::from_f32(bf16::from_bits(a).to_f32() * self.previous).to_f32();
        let b = bf16::from_f32(bf16::from_bits(b).to_f32() * self.current).to_f32();
        bf16::from_f32(a + b).to_bits()
    }
}

#[cfg(test)]
mod tests;
