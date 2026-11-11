//! Explicit spatial grouping; the image request batch remains independent.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "usize", into = "usize")]
pub struct PH3ImageVaeTileBatch(usize);
impl PH3ImageVaeTileBatch {
    pub fn new(maximum_tiles: usize) -> Result<Self> {
        ensure!(
            (1..=4).contains(&maximum_tiles),
            "VAE tile batch must be in 1..=4"
        );
        Ok(Self(maximum_tiles))
    }
    pub const fn maximum_tiles(self) -> usize {
        self.0
    }
}
impl Default for PH3ImageVaeTileBatch {
    fn default() -> Self {
        Self(1)
    }
}
impl TryFrom<usize> for PH3ImageVaeTileBatch {
    type Error = anyhow::Error;
    fn try_from(value: usize) -> Result<Self> {
        Self::new(value)
    }
}
impl From<PH3ImageVaeTileBatch> for usize {
    fn from(value: PH3ImageVaeTileBatch) -> Self {
        value.0
    }
}

/// Consecutive tiles in row-major order, without padding. A group may cross a
/// row boundary; callers select the admitted recipe with this exact tile count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PH3ImageVaeTileGroup {
    first_tile: usize,
    tile_count: usize,
}
impl PH3ImageVaeTileGroup {
    pub const fn first_tile(self) -> usize {
        self.first_tile
    }
    pub const fn tile_count(self) -> usize {
        self.tile_count
    }
}
impl PH3ImageVaeTilePlan {
    pub fn group(
        &self,
        first_tile: usize,
        batch: PH3ImageVaeTileBatch,
    ) -> Result<PH3ImageVaeTileGroup> {
        self.tile(first_tile)?;
        let largest = batch.0.min(self.x.starts.len());
        ensure!(
            first_tile % largest == 0,
            "VAE group must start at its declared traversal boundary"
        );
        Ok(PH3ImageVaeTileGroup {
            first_tile,
            tile_count: largest.min(self.tile_count() - first_tile),
        })
    }
    pub fn group_shapes(&self, batch: PH3ImageVaeTileBatch) -> Vec<usize> {
        let largest = batch.0.min(self.x.starts.len());
        let tail = self.tile_count() % largest;
        if tail == 0 {
            vec![largest]
        } else {
            vec![largest, tail]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn declared_groups_cross_rows_in_order_with_only_a_final_real_tail() {
        let mut crossed_row = false;
        let mut saw_tail = false;
        for columns in 1..=7 {
            for rows in 1..=4 {
                let plan = PH3ImageVaeTilePlan::new(
                    (columns + 1) * 16,
                    (rows + 1) * 16,
                    16,
                    PH3ImageVaeSpatialPolicy::Tiled {
                        tile_pixels: 32,
                        minimum_overlap_pixels: 16,
                        max_tiles: 32,
                    },
                )
                .unwrap();
                for maximum in 1..=4 {
                    let batch = PH3ImageVaeTileBatch::new(maximum).unwrap();
                    let largest = maximum.min(columns);
                    let mut visited = Vec::new();
                    let mut shapes = Vec::new();
                    while visited.len() < plan.tile_count() {
                        let first = visited.len();
                        let g = plan.group(first, batch).unwrap();
                        assert_eq!(g.first_tile(), first);
                        assert!((1..=largest).contains(&g.tile_count()));
                        if g.tile_count() < largest {
                            saw_tail = true;
                            assert_eq!(first + g.tile_count(), plan.tile_count());
                        }
                        if !shapes.contains(&g.tile_count()) {
                            shapes.push(g.tile_count());
                        }
                        let initial = plan.tile(first).unwrap();
                        for index in first..first + g.tile_count() {
                            let tile = plan.tile(index).unwrap();
                            assert_eq!((tile.width, tile.height), (initial.width, initial.height));
                            crossed_row |= tile.y != initial.y;
                            visited.push(tile.index);
                        }
                    }
                    assert_eq!(visited, (0..columns * rows).collect::<Vec<_>>());
                    assert_eq!(plan.group_shapes(batch), shapes);
                    assert!(plan.group(plan.tile_count(), batch).is_err());
                    assert!(plan.group(usize::MAX, batch).is_err());
                    if largest > 1 {
                        assert!(plan.group(1, batch).is_err());
                    }
                }
            }
        }
        assert!(crossed_row && saw_tail);
        for invalid in [0, 5, usize::MAX] {
            assert!(PH3ImageVaeTileBatch::new(invalid).is_err());
            assert!(
                serde_json::from_value::<PH3ImageVaeTileBatch>(serde_json::json!(invalid)).is_err()
            );
        }
        assert_eq!(
            serde_json::to_value(PH3ImageVaeTileBatch::default()).unwrap(),
            1
        );
    }
}
