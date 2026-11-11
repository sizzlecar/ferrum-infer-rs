//! Encoder seams consume raw neighbours, unlike the decoder's recursively
//! blended seams. This preserves pinned ComfyUI tiled_encode semantics while
//! retaining at most one raw tile row and one working tile on the same Host.
use super::*;

pub(crate) struct PH3ImageVaeMomentCanvas {
    plan: PH3ImageVaeTilePlan,
    host: HostMemoryAccount,
    batch: usize,
    channels: usize,
    next: usize,
    canvas: ManagedHostBuffer,
    raw_row: Vec<Option<ManagedHostBytes>>,
}
impl PH3ImageVaeMomentCanvas {
    pub const PROFILE_ID: &'static str =
        "p_h3_image.vae.encoder_tile_blend.f32.raw_neighbours.vertical_then_horizontal.v1";
    pub fn new(
        plan: PH3ImageVaeTilePlan,
        host: HostMemoryAccount,
        batch: usize,
        channels: usize,
    ) -> Result<Self> {
        ensure!(
            batch > 0 && channels > 0,
            "moment canvas requires positive batch and channels"
        );
        plan.x
            .starts
            .len()
            .checked_mul(std::mem::size_of::<Option<ManagedHostBytes>>())
            .filter(|&n| n <= isize::MAX as usize)
            .context("raw tile row metadata exceeds address space")?;
        let tensor = f32_spec(&[
            batch,
            channels,
            plan.y.length / plan.scale,
            plan.x.length / plan.scale,
        ])?;
        let canvas = host.allocate_zeroed(HostMemoryClass::Output, byte_len(&tensor)?)?;
        let raw_row = vec![None; plan.x.starts.len()];
        Ok(Self {
            plan,
            host,
            batch,
            channels,
            next: 0,
            canvas,
            raw_row,
        })
    }
    /// Includes the current encoder readback, all retained raw neighbours and
    /// a separate working tile. Each real allocation's padding is charged once.
    /// Encoder device scratch and normalized input tiles are budgeted separately.
    pub fn output_peak(
        host: &HostMemoryAccount,
        plan: &PH3ImageVaeTilePlan,
        batch: usize,
        channels: usize,
    ) -> Result<usize> {
        let full = f32_spec(&[
            batch,
            channels,
            plan.y.length / plan.scale,
            plan.x.length / plan.scale,
        ])?;
        let tile = f32_spec(&[
            batch,
            channels,
            plan.y.extent / plan.scale,
            plan.x.extent / plan.scale,
        ])?;
        let row_and_work = plan
            .x
            .starts
            .len()
            .checked_add(2)
            .and_then(|n| n.checked_mul(host.backing_bytes(byte_len(&tile).ok()?).ok()?))
            .context("moment canvas output peak overflows")?;
        host.backing_bytes(byte_len(&full)?)?
            .checked_add(row_and_work)
            .context("moment canvas output peak overflows")
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
            "encoder tiles must commit in row-major order"
        );
        let tile = self.plan.tile(index)?;
        let scale = self.plan.scale;
        let (height, width) = (tile.height / scale, tile.width / scale);
        let (top, left) = (tile.overlap_top / scale, tile.overlap_left / scale);
        let expected = f32_spec(&[self.batch, self.channels, height, width])?;
        ensure!(
            tensor == &expected
                && source.len() == byte_len(&expected)?
                && source.class() == HostMemoryClass::Output
                && self.host.owns(source),
            "encoder tile differs from expected F32 moments or Output owner"
        );
        ensure!(
            source
                .bytes()
                .chunks_exact(4)
                .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite()),
            "encoder moments contain a nonfinite value"
        );
        let column = index % self.raw_row.len();
        ensure!(
            (top > 0) == self.raw_row[column].is_some(),
            "encoder upper neighbour differs from traversal"
        );
        if let Some(previous) = &self.raw_row[column] {
            ensure!(
                previous.len() == source.len(),
                "encoder upper moment shape differs"
            );
        }
        if column > 0 {
            ensure!(
                self.raw_row[column - 1]
                    .as_ref()
                    .is_some_and(|v| v.len() == source.len()),
                "encoder left neighbour differs from traversal"
            );
        }
        let mut working = self
            .host
            .allocate_zeroed(HostMemoryClass::Output, source.len())?;
        working.as_mut().copy_from_slice(source.bytes());
        if top > 0 {
            let previous = self.raw_row[column].as_ref().unwrap();
            for bc in 0..self.batch * self.channels {
                for y in 0..top {
                    for x in 0..width {
                        let a = ((bc * height + height - top + y) * width + x) * 4;
                        let b = ((bc * height + y) * width + x) * 4;
                        let value =
                            blend(read(previous.bytes(), a), read(working.as_ref(), b), y, top)?;
                        working.as_mut()[b..b + 4].copy_from_slice(&value.to_le_bytes());
                    }
                }
            }
        }
        if left > 0 {
            // The previously committed raw tile is retained without copying.
            // Its horizontally/vertically blended working copy was discarded.
            let previous = self.raw_row[column - 1].as_ref().unwrap();
            for bc in 0..self.batch * self.channels {
                for y in 0..height {
                    for x in 0..left {
                        let a = ((bc * height + y) * width + width - left + x) * 4;
                        let b = ((bc * height + y) * width + x) * 4;
                        let value = blend(
                            read(previous.bytes(), a),
                            read(working.as_ref(), b),
                            x,
                            left,
                        )?;
                        working.as_mut()[b..b + 4].copy_from_slice(&value.to_le_bytes());
                    }
                }
            }
        }
        // No persistent state changes before all allocation and numerical
        // checks succeed. A rejected tile may be retried with the same owners.
        let kept_width = width - tile.trim_right / scale;
        let kept_height = height - tile.trim_bottom / scale;
        let full_height = self.plan.y.length / scale;
        let full_width = self.plan.x.length / scale;
        for bc in 0..self.batch * self.channels {
            for y in 0..kept_height {
                let a = (bc * height + y) * width * 4;
                let b = ((bc * full_height + tile.y / scale + y) * full_width + tile.x / scale) * 4;
                self.canvas.as_mut()[b..b + kept_width * 4]
                    .copy_from_slice(&working.as_ref()[a..a + kept_width * 4]);
            }
        }
        self.raw_row[column] = Some(source.clone());
        self.next += 1;
        Ok(())
    }
    pub fn finish(self) -> Result<(ResolvedTensorSpec, ManagedHostBytes)> {
        ensure!(
            self.next == self.plan.count,
            "cannot publish incomplete encoder moments"
        );
        let tensor = f32_spec(&[
            self.batch,
            self.channels,
            self.plan.y.length / self.plan.scale,
            self.plan.x.length / self.plan.scale,
        ])?;
        Ok((tensor, self.canvas.freeze()))
    }
}
fn blend(a: f32, b: f32, position: usize, extent: usize) -> Result<f32> {
    let ratio = position as f32 / extent as f32;
    let first = a * (1.0 - ratio);
    let second = b * ratio;
    let result = first + second;
    ensure!(
        result.is_finite(),
        "encoder tile blend overflowed its F32 profile"
    );
    Ok(result)
}

#[cfg(test)]
mod tests;
