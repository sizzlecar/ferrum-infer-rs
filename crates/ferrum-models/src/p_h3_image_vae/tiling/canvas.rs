//! Ordered, transactional stitching of completed F32 tiles. Vertical overlap
//! consumes the previous blended strip before horizontal overlap consumes the
//! blended left tail. Final pixel normalization occurs after the whole canvas.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PH3ImageVaeCanvasDomain {
    RawDecoder,
    UnitRgb,
}

pub(crate) struct PH3ImageVaeCanvas {
    plan: PH3ImageVaeTilePlan,
    host: HostMemoryAccount,
    batch: usize,
    channels: usize,
    next: usize,
    canvas: ManagedHostBuffer,
    strip: Option<ManagedHostBytes>,
    building_strip: Option<ManagedHostBuffer>,
    left: Option<ManagedHostBytes>,
}
impl PH3ImageVaeCanvas {
    pub const BLEND_PROFILE_ID: &'static str =
        "p_h3_image.vae.tile_blend.f32.vertical_then_horizontal.separate_mul_add.v1";

    pub fn new(
        plan: PH3ImageVaeTilePlan,
        host: HostMemoryAccount,
        batch: usize,
        channels: usize,
    ) -> Result<Self> {
        ensure!(
            batch > 0 && channels > 0,
            "canvas batch and channel dimensions must be positive"
        );
        let tensor = f32_spec(&[batch, channels, plan.y.length, plan.x.length])?;
        let canvas = host.allocate_zeroed(HostMemoryClass::Output, byte_len(&tensor)?)?;
        Ok(Self {
            plan,
            host,
            batch,
            channels,
            next: 0,
            canvas,
            strip: None,
            building_strip: None,
            left: None,
        })
    }
    /// Bound simultaneously live canvas/work/old+new strip/old+new tail owners.
    /// The caller separately budgets completed decoder readbacks and input
    /// tiles. Each allocation's real alignment is included independently.
    pub fn output_peak(
        host: &HostMemoryAccount,
        plan: &PH3ImageVaeTilePlan,
        batch: usize,
        channels: usize,
    ) -> Result<usize> {
        let backing = |height, width| -> Result<usize> {
            if height == 0 || width == 0 {
                return Ok(0);
            }
            Ok(host.backing_bytes(byte_len(&f32_spec(&[batch, channels, height, width])?)?)?)
        };
        let strip = backing(
            plan.y.overlaps.iter().copied().max().unwrap_or(0),
            plan.x.length,
        )?;
        let tail = backing(
            plan.y.extent,
            plan.x.overlaps.iter().copied().max().unwrap_or(0),
        )?;
        [
            backing(plan.y.length, plan.x.length)?,
            backing(plan.y.extent, plan.x.extent)?,
            strip,
            strip,
            tail,
            tail,
        ]
        .into_iter()
        .try_fold(0usize, |a, b| {
            a.checked_add(b).context("canvas Host peak overflows")
        })
    }
    pub fn next_tile(&self) -> Result<PH3ImageVaeTile> {
        self.plan.tile(self.next)
    }
    pub fn completed_tiles(&self) -> usize {
        self.next
    }

    pub fn push(
        &mut self,
        index: usize,
        tensor: &ResolvedTensorSpec,
        source: &ManagedHostBytes,
    ) -> Result<()> {
        ensure!(
            index == self.next,
            "VAE tiles must commit in declared row-major order"
        );
        let tile = self.plan.tile(index)?;
        let expected = f32_spec(&[self.batch, self.channels, tile.height, tile.width])?;
        ensure!(
            tensor == &expected
                && source.len() == byte_len(&expected)?
                && self.host.owns(source)
                && source.class() == HostMemoryClass::Output,
            "completed tile differs from its shape or original Output owner"
        );
        ensure!(
            source
                .bytes()
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
            "completed VAE tile contains nonfinite pixels"
        );
        ensure!(
            (tile.overlap_top > 0) == self.strip.is_some()
                && (tile.overlap_left > 0) == self.left.is_some(),
            "tile neighbours differ from the declared traversal"
        );
        let mut working = self
            .host
            .allocate_zeroed(HostMemoryClass::Output, source.len())?;
        working.as_mut().copy_from_slice(source.bytes());
        if let Some(strip) = &self.strip {
            ensure!(
                strip.len()
                    == byte_len(&f32_spec(&[
                        self.batch,
                        self.channels,
                        tile.overlap_top,
                        self.plan.x.length
                    ])?)?,
                "vertical strip extent differs"
            );
            for bc in 0..self.batch * self.channels {
                for y in 0..tile.overlap_top {
                    for x in 0..tile.width {
                        let a = ((bc * tile.overlap_top + y) * self.plan.x.length + tile.x + x) * 4;
                        let b = ((bc * tile.height + y) * tile.width + x) * 4;
                        let result = blend(
                            read(strip.bytes(), a),
                            read(working.as_ref(), b),
                            y,
                            tile.overlap_top,
                        )?;
                        working.as_mut()[b..b + 4].copy_from_slice(&result.to_le_bytes());
                    }
                }
            }
        }
        if let Some(left) = &self.left {
            ensure!(
                left.len()
                    == byte_len(&f32_spec(&[
                        self.batch,
                        self.channels,
                        tile.height,
                        tile.overlap_left
                    ])?)?,
                "left tail extent differs"
            );
            for bc in 0..self.batch * self.channels {
                for y in 0..tile.height {
                    for x in 0..tile.overlap_left {
                        let a = ((bc * tile.height + y) * tile.overlap_left + x) * 4;
                        let b = ((bc * tile.height + y) * tile.width + x) * 4;
                        let result = blend(
                            read(left.bytes(), a),
                            read(working.as_ref(), b),
                            x,
                            tile.overlap_left,
                        )?;
                        working.as_mut()[b..b + 4].copy_from_slice(&result.to_le_bytes());
                    }
                }
            }
        }
        // Allocate every new neighbour before modifying any persistent owner.
        // A late capacity failure leaves the cursor, canvas and prior seams
        // unchanged, so the same completed tile may be retried safely.
        let mut new_left = if tile.trim_right > 0 {
            Some(self.host.allocate_zeroed(
                HostMemoryClass::Output,
                byte_len(&f32_spec(&[
                    self.batch,
                    self.channels,
                    tile.height,
                    tile.trim_right,
                ])?)?,
            )?)
        } else {
            None
        };
        let mut new_strip = if tile.x == 0 && tile.trim_bottom > 0 {
            Some(self.host.allocate_zeroed(
                HostMemoryClass::Output,
                byte_len(&f32_spec(&[
                    self.batch,
                    self.channels,
                    tile.trim_bottom,
                    self.plan.x.length,
                ])?)?,
            )?)
        } else {
            None
        };
        let kept_width = tile.width - tile.trim_right;
        let kept_height = tile.height - tile.trim_bottom;
        if let Some(left) = new_left.as_mut() {
            for bc in 0..self.batch * self.channels {
                for y in 0..tile.height {
                    let a = ((bc * tile.height + y) * tile.width + kept_width) * 4;
                    let b = (bc * tile.height + y) * tile.trim_right * 4;
                    left.as_mut()[b..b + tile.trim_right * 4]
                        .copy_from_slice(&working.as_ref()[a..a + tile.trim_right * 4]);
                }
            }
        }
        let strip = match new_strip.as_mut() {
            Some(strip) => Some(strip),
            None => self.building_strip.as_mut(),
        };
        if let Some(strip) = strip {
            for bc in 0..self.batch * self.channels {
                for y in 0..tile.trim_bottom {
                    let a = ((bc * tile.height + kept_height + y) * tile.width) * 4;
                    let b = ((bc * tile.trim_bottom + y) * self.plan.x.length + tile.x) * 4;
                    strip.as_mut()[b..b + kept_width * 4]
                        .copy_from_slice(&working.as_ref()[a..a + kept_width * 4]);
                }
            }
        }
        for bc in 0..self.batch * self.channels {
            for y in 0..kept_height {
                let a = (bc * tile.height + y) * tile.width * 4;
                let b = ((bc * self.plan.y.length + tile.y + y) * self.plan.x.length + tile.x) * 4;
                self.canvas.as_mut()[b..b + kept_width * 4]
                    .copy_from_slice(&working.as_ref()[a..a + kept_width * 4]);
            }
        }
        self.left = new_left.map(ManagedHostBuffer::freeze);
        if new_strip.is_some() {
            self.building_strip = new_strip;
        }
        self.next += 1;
        if self.next % self.plan.x.starts.len() == 0 {
            self.strip = self.building_strip.take().map(ManagedHostBuffer::freeze);
            self.left = None;
        }
        Ok(())
    }
    pub fn finish(
        mut self,
        domain: PH3ImageVaeCanvasDomain,
    ) -> Result<(ResolvedTensorSpec, ManagedHostBytes)> {
        ensure!(
            self.next == self.plan.count,
            "cannot publish an incomplete VAE canvas"
        );
        if domain == PH3ImageVaeCanvasDomain::UnitRgb {
            ensure!(
                self.channels == 3,
                "unit RGB finalization requires exactly three channels"
            );
            let positions = self.plan.x.length * self.plan.y.length;
            for b in 0..self.batch {
                for c in 0..3 {
                    for p in 0..positions {
                        let offset = ((b * 3 + c) * positions + p) * 4;
                        let raw = read(self.canvas.as_ref(), offset);
                        let scaled = raw * [0.229_f32, 0.224, 0.225][c];
                        let shifted = scaled + [0.485_f32, 0.456, 0.406][c];
                        self.canvas.as_mut()[offset..offset + 4]
                            .copy_from_slice(&shifted.clamp(0.0, 1.0).to_le_bytes());
                    }
                }
            }
        }
        let tensor = f32_spec(&[
            self.batch,
            self.channels,
            self.plan.y.length,
            self.plan.x.length,
        ])?;
        Ok((tensor, self.canvas.freeze()))
    }
}
fn blend(a: f32, b: f32, position: usize, extent: usize) -> Result<f32> {
    let ratio = position as f32 / extent as f32;
    let left = a * (1.0 - ratio);
    let right = b * ratio;
    let result = left + right;
    ensure!(
        result.is_finite(),
        "VAE tile blend overflowed its F32 profile"
    );
    Ok(result)
}
