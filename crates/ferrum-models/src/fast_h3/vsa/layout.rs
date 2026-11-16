use super::*;

#[derive(Clone, Copy)]
struct Segment {
    start: usize,
    rows: usize,
    grid: Option<[usize; 3]>,
    kind: FastH3SegmentKind,
}

impl FastH3VsaLayout {
    /// Build the trained document geometry. Checkpoint tile64/256 metadata is
    /// parsed elsewhere but is deliberately not executable by this descriptor.
    pub fn ref2va(
        checkpoint: &FastH3Checkpoint,
        text_rows: usize,
        references: &[FastH3Reference],
        target: &FastH3VideoGeometry,
        limits: FastH3VsaLayoutLimits,
    ) -> Result<Self> {
        ensure!(
            checkpoint.vsa_tile_size() == FAST_H3_VSA_TILE_ROWS,
            "FastH3 VSA descriptor currently requires trained tile128"
        );
        ensure!(
            limits.maximum_packed_rows > 0
                && limits.maximum_tiles > 0
                && limits.maximum_metadata_bytes > 0,
            "VSA metadata limits must be positive"
        );
        let segment_count = references
            .iter()
            .try_fold(3usize, |n, reference| {
                n.checked_add(match reference {
                    FastH3Reference::Video {
                        audio_latents_per_channel: Some(_),
                        ..
                    } => 2,
                    _ => 1,
                })
            })
            .context("VSA segment count overflow")?;
        let segment_bytes = product(
            &[
                segment_count,
                size_of::<FastH3Segment>() + size_of::<Segment>(),
            ],
            "VSA segment byte count overflow",
        )?;
        ensure!(
            segment_bytes <= limits.maximum_metadata_bytes,
            "VSA segment geometry exceeds metadata capacity"
        );
        let packed =
            FastH3PackedLayout::ref2va(text_rows, references, target, limits.maximum_packed_rows)?;
        ensure!(
            packed.sequence_rows() < FAST_H3_VSA_PADDING_ROW as usize,
            "VSA packed row index exceeds descriptor representation"
        );
        let mut video_references = references.iter().filter_map(|r| match r {
            FastH3Reference::Video {
                latent_frames,
                latent_height,
                latent_width,
                ..
            } => Some([*latent_frames, latent_height / 2, latent_width / 2]),
            _ => None,
        });
        let mut segments = Vec::new();
        segments.try_reserve_exact(segment_count)?;
        let mut tile_count = 0usize;
        let mut region_count = 0usize;
        for s in packed.segments() {
            let grid = match s.kind {
                FastH3SegmentKind::ReferenceVideo => Some(
                    video_references
                        .next()
                        .context("missing reference video geometry")?,
                ),
                FastH3SegmentKind::TargetVideo => Some([
                    target.latent_frames(),
                    target.latent_height() / 2,
                    target.latent_width() / 2,
                ]),
                _ => None,
            };
            let rows = s.packed_rows.end - s.packed_rows.start;
            let tiles = if let Some(grid) = grid {
                ensure!(
                    grid.iter().all(|&v| v > 0)
                        && product(&grid, "VSA token grid overflow")? == rows,
                    "VSA region grid disagrees with packed rows"
                );
                region_count += 1;
                product(
                    &std::array::from_fn::<_, 3, _>(|axis| {
                        grid[axis].div_ceil(FAST_H3_VSA_TILE_SHAPE[axis])
                    }),
                    "VSA tile count overflow",
                )?
            } else {
                rows.div_ceil(FAST_H3_VSA_TILE_ROWS)
            };
            tile_count = tile_count
                .checked_add(tiles)
                .context("VSA tile count overflow")?;
            ensure!(
                tile_count <= limits.maximum_tiles,
                "VSA tile capacity exceeded"
            );
            segments.push(Segment {
                start: s.packed_rows.start,
                rows,
                grid,
                kind: s.kind,
            });
        }
        ensure!(
            video_references.next().is_none(),
            "unconsumed VSA reference region"
        );
        let padded_rows = product(
            &[tile_count, FAST_H3_VSA_TILE_ROWS],
            "VSA padded rows overflow",
        )?;
        ensure!(
            padded_rows <= i32::MAX as usize && tile_count <= i32::MAX as usize,
            "VSA tile index exceeds native descriptor representation"
        );
        let metadata_bytes = [
            segment_bytes,
            product(
                &[region_count, size_of::<FastH3VsaRegion>()],
                "VSA region bytes overflow",
            )?,
            product(
                &[tile_count, size_of::<u16>()],
                "VSA validity bytes overflow",
            )?,
            product(
                &[packed.sequence_rows(), size_of::<u32>()],
                "VSA inverse map bytes overflow",
            )?,
            product(
                &[padded_rows, size_of::<u32>()],
                "VSA tile map bytes overflow",
            )?,
        ]
        .into_iter()
        .try_fold(0usize, |sum, n| sum.checked_add(n))
        .context("VSA metadata bytes overflow")?;
        ensure!(
            metadata_bytes <= limits.maximum_metadata_bytes,
            "VSA maps exceed metadata capacity"
        );
        let mut layout = Self {
            packed_to_tile: filled(packed.sequence_rows(), FAST_H3_VSA_PADDING_ROW)?,
            tile_to_packed: filled(padded_rows, FAST_H3_VSA_PADDING_ROW)?,
            packed,
            prefix_tiles: 0,
            regions: Vec::new(),
            valid_rows: Vec::new(),
            metadata_bytes,
        };
        layout.regions.try_reserve_exact(region_count)?;
        layout.valid_rows.try_reserve_exact(tile_count)?;
        // Prefix chunks never cross a source segment, even if two adjacent
        // short segments would fit in the same 128-row block.
        for s in segments.iter().filter(|s| s.grid.is_none()) {
            for offset in (0..s.rows).step_by(FAST_H3_VSA_TILE_ROWS) {
                let valid = (s.rows - offset).min(FAST_H3_VSA_TILE_ROWS);
                let slot = layout.valid_rows.len() * FAST_H3_VSA_TILE_ROWS;
                for row in 0..valid {
                    layout.set_row(slot + row, s.start + offset + row)?;
                }
                layout.valid_rows.push(valid as u16);
            }
        }
        layout.prefix_tiles = layout.valid_rows.len();
        ensure!(
            layout.prefix_tiles > 0,
            "FP4 exempt attention needs a nonempty dense prefix"
        );
        for s in segments.iter().filter(|s| s.grid.is_some()) {
            let [t, h, w] = s.grid.context("missing VSA grid")?;
            let first = layout.valid_rows.len();
            for t0 in (0..t).step_by(FAST_H3_VSA_TILE_SHAPE[0]) {
                for h0 in (0..h).step_by(FAST_H3_VSA_TILE_SHAPE[1]) {
                    for w0 in (0..w).step_by(FAST_H3_VSA_TILE_SHAPE[2]) {
                        let slot = layout.valid_rows.len() * FAST_H3_VSA_TILE_ROWS;
                        let mut valid = 0usize;
                        for ti in t0..t.min(t0 + FAST_H3_VSA_TILE_SHAPE[0]) {
                            for hi in h0..h.min(h0 + FAST_H3_VSA_TILE_SHAPE[1]) {
                                for wi in w0..w.min(w0 + FAST_H3_VSA_TILE_SHAPE[2]) {
                                    layout
                                        .set_row(slot + valid, s.start + (ti * h + hi) * w + wi)?;
                                    valid += 1;
                                }
                            }
                        }
                        layout.valid_rows.push(valid as u16);
                    }
                }
            }
            let end = layout.valid_rows.len();
            let (kind, mut sparsity) = if s.kind == FastH3SegmentKind::TargetVideo {
                (FastH3VsaRegionKind::TargetVideo, checkpoint.vsa_sparsity())
            } else {
                (
                    FastH3VsaRegionKind::ReferenceVideo,
                    1.0 - checkpoint.vsa_reference_keep_rate(),
                )
            };
            if checkpoint.vsa_sparsity() <= 0.0 {
                sparsity = 0.0;
            }
            // Preserve the source's f64 subtraction and ceil order, including
            // reference keep -> sparsity -> keep; rationalization can change k.
            let trained_key_tiles =
                (((1.0 - sparsity) * (end - first) as f64).ceil() as usize).clamp(1, end - first);
            layout.regions.push(FastH3VsaRegion {
                kind,
                tiles: first..end,
                trained_key_tiles,
            });
        }
        ensure!(
            layout.valid_rows.len() == tile_count
                && !layout.packed_to_tile.contains(&FAST_H3_VSA_PADDING_ROW)
                && layout
                    .regions
                    .last()
                    .is_some_and(|r| r.kind == FastH3VsaRegionKind::TargetVideo),
            "VSA geometry failed to cover the packed document"
        );
        Ok(layout)
    }

    fn set_row(&mut self, tile_row: usize, packed_row: usize) -> Result<()> {
        let inverse = self
            .packed_to_tile
            .get_mut(packed_row)
            .context("VSA packed row out of bounds")?;
        let forward = self
            .tile_to_packed
            .get_mut(tile_row)
            .context("VSA tile row out of bounds")?;
        ensure!(
            *inverse == FAST_H3_VSA_PADDING_ROW && *forward == FAST_H3_VSA_PADDING_ROW,
            "VSA map must be injective"
        );
        *inverse = tile_row as u32;
        *forward = packed_row as u32;
        Ok(())
    }
}
