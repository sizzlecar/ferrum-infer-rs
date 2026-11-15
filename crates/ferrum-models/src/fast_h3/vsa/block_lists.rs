use super::*;

#[derive(Clone, Copy, Debug)]
pub struct FastH3VsaMaskLimits {
    pub maximum_mask_entries: usize,
    pub maximum_descriptor_bytes: usize,
}

/// The tile128 specialization of `vsa_tile_mask_to_fp4_blocks`.
///
/// `indices` is `[batch,heads,query_blocks,key_blocks]`, `counts` drops the last
/// axis, and `half_valid` has `2*key_blocks` elements. Each batch is a separate
/// document with the same checked layout, never another span of one document.
/// Tile128 needs no quadrant mask. The native kernel visits each list in
/// reverse, so descending indices make the nonempty prefix block zero first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FastH3VsaBlockLists {
    batch: usize,
    heads: usize,
    tiles: usize,
    prefix_tiles: usize,
    indices: Vec<i32>,
    counts: Vec<i32>,
    half_valid: Vec<i32>,
    descriptor_bytes: usize,
}

impl FastH3VsaLayout {
    /// Lower an already selected dynamic mask, after validating the trained
    /// region budgets and dense-prefix policy. This verifies visibility and
    /// memory safety; it does not prove that the caller chose the highest
    /// pooled QK scores. The future selection operator owns that calculation.
    ///
    /// `mask` uses `[batch,heads,query_tile,key_tile]` row-major indexing. The
    /// shape is fixed before any descriptor allocation; heads/batches may have
    /// different selections, but must share this layout and selection mode.
    pub fn lower_mask(
        &self,
        batch: usize,
        heads: usize,
        mask: &[bool],
        selection: FastH3VsaSelection,
        limits: FastH3VsaMaskLimits,
    ) -> Result<FastH3VsaBlockLists> {
        ensure!(
            batch > 0 && heads > 0 && batch <= i32::MAX as usize && heads <= i32::MAX as usize,
            "VSA mask batch and heads must fit positive native int dimensions"
        );
        let tiles = self.tile_count();
        let row_count = product(&[batch, heads, tiles], "VSA mask row count overflow")?;
        // The upstream scheduler's total_blocks and counts row addressing
        // multiply these dimensions in signed int, unlike its index payload.
        ensure!(
            row_count <= i32::MAX as usize,
            "VSA native work-grid extent exceeds i32"
        );
        let entries = product(&[row_count, tiles], "VSA mask element count overflow")?;
        ensure!(
            entries <= limits.maximum_mask_entries && mask.len() == entries,
            "VSA mask shape or capacity mismatch"
        );
        let half_count = product(&[tiles, 2], "VSA half count overflow")?;
        let descriptor_bytes = entries
            .checked_add(row_count)
            .and_then(|v| v.checked_add(half_count))
            .and_then(|v| v.checked_mul(size_of::<i32>()))
            .context("VSA descriptor byte count overflow")?;
        ensure!(
            descriptor_bytes <= limits.maximum_descriptor_bytes,
            "VSA block lists exceed descriptor capacity"
        );
        // Reject malformed dynamic masks before allocating any output. The
        // private layout guarantees contiguous, exhaustive region tile spans.
        for (row, visible) in mask.chunks_exact(tiles).enumerate() {
            ensure!(
                visible[..self.prefix_tiles].iter().all(|&v| v),
                "VSA exempt mode requires every prefix key for every query"
            );
            if row % tiles < self.prefix_tiles {
                ensure!(
                    visible.iter().all(|&v| v),
                    "VSA prefix queries must be dense"
                );
            } else {
                for region in &self.regions {
                    let expected = match selection {
                        FastH3VsaSelection::Trained => region.trained_key_tiles,
                        FastH3VsaSelection::Dense => region.tiles.len(),
                    };
                    ensure!(
                        visible[region.tiles.clone()].iter().filter(|&&v| v).count() == expected,
                        "VSA query does not meet its region key-tile budget"
                    );
                }
            }
        }
        let mut indices = filled(entries, 0i32)?;
        let mut counts = filled(row_count, 0i32)?;
        for (row, visible) in mask.chunks_exact(tiles).enumerate() {
            let output = &mut indices[row * tiles..(row + 1) * tiles];
            let mut count = 0usize;
            for key in (0..tiles).rev().filter(|&key| visible[key]) {
                output[count] = key as i32;
                count += 1;
            }
            // Every tile has a nonempty first half. Prefix block zero is
            // selected above, including for padded query rows in either half.
            ensure!(
                count > 0 && output[count - 1] == 0,
                "VSA first visited block must be the dense prefix"
            );
            counts[row] = count as i32;
        }
        let mut half_valid = filled(half_count, 0i32)?;
        for (tile, &valid) in self.valid_rows.iter().enumerate() {
            half_valid[2 * tile] = i32::from(valid.min(64));
            half_valid[2 * tile + 1] = i32::from(valid.saturating_sub(64));
        }
        Ok(FastH3VsaBlockLists {
            batch,
            heads,
            tiles,
            prefix_tiles: self.prefix_tiles,
            indices,
            counts,
            half_valid,
            descriptor_bytes,
        })
    }
}

impl FastH3VsaBlockLists {
    pub fn batch(&self) -> usize {
        self.batch
    }
    pub fn heads(&self) -> usize {
        self.heads
    }
    pub fn query_blocks(&self) -> usize {
        self.tiles
    }
    pub fn key_blocks(&self) -> usize {
        self.tiles
    }
    pub fn indices(&self) -> &[i32] {
        &self.indices
    }
    pub fn counts(&self) -> &[i32] {
        &self.counts
    }
    pub fn half_valid(&self) -> &[i32] {
        &self.half_valid
    }
    pub fn descriptor_bytes(&self) -> usize {
        self.descriptor_bytes
    }
    /// Prefix outputs use their own dense FP8-QK branch; the FP4 query view
    /// starts here while retaining all key blocks and the original row lists.
    pub fn video_query_blocks(&self) -> Range<usize> {
        self.prefix_tiles..self.tiles
    }
    pub fn selected_blocks(&self, batch: usize, head: usize, query: usize) -> Option<&[i32]> {
        if batch >= self.batch || head >= self.heads || query >= self.tiles {
            return None;
        }
        let row = (batch * self.heads + head) * self.tiles + query;
        let count = self.counts[row] as usize;
        Some(&self.indices[row * self.tiles..row * self.tiles + count])
    }
}
