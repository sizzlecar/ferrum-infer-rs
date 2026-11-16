//! Ref2VA positions from FastVideo 866a3cdb, packing.py. Coordinates remain
//! F64 until RoPE's explicit F32 conversion. No expanded position tensor is
//! retained: regions are bounded metadata, and output buffers use the caller's
//! existing Host account. Continuation conditioning is not part of this plan.

use anyhow::{ensure, Context, Result};

use super::{FastH3PackedLayout, FastH3Reference, FastH3VideoGeometry};

mod rotary;

/// Bounds checked before constructing the region metadata or packed layout.
#[derive(Clone, Copy, Debug)]
pub struct FastH3PositionLimits {
    pub max_sequence_rows: usize,
    pub max_regions: usize,
}

#[derive(Clone, Copy, Debug)]
struct SpatialAxis {
    left: f64,
    step: f64,
    count: usize,
}
impl SpatialAxis {
    fn new(dimension: usize, sqrt_area: f64) -> Self {
        let ratio = dimension as f64 / sqrt_area;
        let left = (1.0 - ratio) / 2.0;
        let count = dimension / 2;
        // Preserve linspace's subtraction of its independently rounded stop.
        let stop = left + ratio;
        Self {
            left,
            step: (stop - left) / count as f64,
            count,
        }
    }
    fn at(self, index: usize) -> f64 {
        (index as f64 * self.step + self.left) * 32.0
    }
}

#[derive(Clone, Copy, Debug)]
struct SpatialGrid {
    height: SpatialAxis,
    width: SpatialAxis,
    rows: usize,
}
impl SpatialGrid {
    fn new(height: usize, width: usize) -> Result<Self> {
        let area = height
            .checked_mul(width)
            .context("position area overflow")?;
        ensure!(
            height > 0 && width > 0 && height % 2 == 0 && width % 2 == 0,
            "FastH3 positions need nonempty [1,2,2] patch geometry"
        );
        let sqrt_area = libm::sqrt(area as f64);
        Ok(Self {
            height: SpatialAxis::new(height, sqrt_area),
            width: SpatialAxis::new(width, sqrt_area),
            rows: area / 4,
        })
    }
}

#[derive(Clone, Copy, Debug)]
enum Region {
    Text,
    Visual {
        origin: f64,
        grid: SpatialGrid,
    },
    Audio {
        origin: f64,
        per_channel: usize,
        width: SpatialAxis,
    },
}

/// The ordered packed layout and its coordinates are built together from the
/// same prepared references. Audio is channel-major; video and its audio share
/// an origin, while successive references advance one shared Ref2VA clock.
#[derive(Clone, Debug)]
pub struct FastH3PositionPlan {
    layout: FastH3PackedLayout,
    regions: Vec<Region>,
}
impl FastH3PositionPlan {
    pub fn ref2va(
        text_rows: usize,
        references: &[FastH3Reference],
        target: &FastH3VideoGeometry,
        limits: FastH3PositionLimits,
    ) -> Result<Self> {
        let regions = references.iter().try_fold(3_usize, |n, reference| {
            n.checked_add(match reference {
                FastH3Reference::Video {
                    audio_latents_per_channel: Some(_),
                    ..
                } => 2,
                _ => 1,
            })
            .context("position region count overflow")
        })?;
        ensure!(
            regions <= limits.max_regions,
            "position regions exceed capacity"
        );
        let layout =
            FastH3PackedLayout::ref2va(text_rows, references, target, limits.max_sequence_rows)?;
        let target_grid = SpatialGrid::new(target.latent_height(), target.latent_width())?;
        let mut plan = Self {
            layout,
            regions: Vec::new(),
        };
        plan.regions.try_reserve_exact(regions)?;
        plan.regions.push(Region::Text);
        let mut origin = text_rows as f64;
        for reference in references {
            match *reference {
                FastH3Reference::Image {
                    latent_height,
                    latent_width,
                } => {
                    let grid = SpatialGrid::new(latent_height, latent_width)?;
                    plan.regions.push(Region::Visual { origin, grid });
                    origin += 1.0;
                }
                FastH3Reference::Audio {
                    latents_per_channel,
                } => {
                    plan.regions.push(Region::Audio {
                        origin,
                        per_channel: latents_per_channel,
                        width: target_grid.width,
                    });
                    origin += latents_per_channel as f64;
                }
                FastH3Reference::Video {
                    latent_frames,
                    latent_height,
                    latent_width,
                    audio_latents_per_channel,
                } => {
                    let grid = SpatialGrid::new(latent_height, latent_width)?;
                    if let Some(per_channel) = audio_latents_per_channel {
                        plan.regions.push(Region::Audio {
                            origin,
                            per_channel,
                            width: grid.width,
                        });
                    }
                    plan.regions.push(Region::Visual { origin, grid });
                    // Ref2VA uses Python's sequential sum, unlike the separate
                    // keyframe path's NumPy pairwise temporal span helper.
                    let span = (0..latent_frames).fold(0.0, |sum, frame| sum + frame_span(frame));
                    origin += span.max(audio_latents_per_channel.unwrap_or(0) as f64);
                }
            }
        }
        plan.regions.push(Region::Audio {
            origin,
            per_channel: target.audio_latents_per_channel(),
            width: target_grid.width,
        });
        plan.regions.push(Region::Visual {
            origin,
            grid: target_grid,
        });
        ensure!(
            plan.regions.len() == plan.layout.segments().len(),
            "position regions disagree with packed layout"
        );
        Ok(plan)
    }

    pub fn layout(&self) -> &FastH3PackedLayout {
        &self.layout
    }

    /// Streams positions in packed row order with O(1) iterator state.
    pub fn positions(&self) -> impl ExactSizeIterator<Item = [f64; 3]> + '_ {
        Positions {
            plan: self,
            region: 0,
            row: 0,
            temporal_prefix: 0.0,
            remaining: self.layout.sequence_rows(),
        }
    }
}

fn frame_span(frame: usize) -> f64 {
    (5.0 / 3.0) * if frame % 5 == 0 { 1.0 } else { 4.0 }
}

struct Positions<'a> {
    plan: &'a FastH3PositionPlan,
    region: usize,
    row: usize,
    temporal_prefix: f64,
    remaining: usize,
}
impl Iterator for Positions<'_> {
    type Item = [f64; 3];
    fn next(&mut self) -> Option<Self::Item> {
        let region = *self.plan.regions.get(self.region)?;
        let position = match region {
            Region::Text => [self.row as f64, 0.0, 0.0],
            Region::Audio {
                origin,
                per_channel,
                width,
            } => [
                origin + (self.row % per_channel) as f64,
                0.0,
                width.at(if self.row < per_channel {
                    0
                } else {
                    width.count - 1
                }),
            ],
            Region::Visual { origin, grid } => {
                let spatial = self.row % grid.rows;
                [
                    origin + self.temporal_prefix,
                    grid.height.at(spatial / grid.width.count),
                    grid.width.at(spatial % grid.width.count),
                ]
            }
        };
        self.row += 1;
        if let Region::Visual { grid, .. } = region {
            if self.row % grid.rows == 0 {
                self.temporal_prefix += frame_span(self.row / grid.rows - 1);
            }
        }
        if self.row == self.plan.layout.segments()[self.region].packed_rows.len() {
            self.region += 1;
            self.row = 0;
            self.temporal_prefix = 0.0;
        }
        self.remaining -= 1;
        Some(position)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}
impl ExactSizeIterator for Positions<'_> {}
impl std::iter::FusedIterator for Positions<'_> {}

#[cfg(test)]
mod tests;
