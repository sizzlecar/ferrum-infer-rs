//! Real spatial groups, independent of request batch and temporal extent.
use anyhow::{ensure, Result};
use std::num::NonZeroUsize;

/// A grouping preference. The validated canvas and physical memory admission
/// bound the actual group; this value never authorizes padding or allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "usize", into = "usize")]
pub struct SpatialTileBatch(NonZeroUsize);

impl SpatialTileBatch {
    pub fn new(maximum_tiles: usize) -> Result<Self> {
        Ok(Self(NonZeroUsize::new(maximum_tiles).ok_or_else(|| {
            anyhow::anyhow!("spatial tile batch must be nonzero")
        })?))
    }
    pub const fn maximum_tiles(self) -> usize {
        self.0.get()
    }
    pub fn groups(self, total_tiles: usize) -> Result<SpatialTileGroups> {
        ensure!(total_tiles > 0, "spatial groups require a nonempty canvas");
        Ok(SpatialTileGroups {
            total_tiles,
            maximum_tiles: self.maximum_tiles().min(total_tiles),
        })
    }
}
impl Default for SpatialTileBatch {
    fn default() -> Self {
        Self(NonZeroUsize::MIN)
    }
}
impl TryFrom<usize> for SpatialTileBatch {
    type Error = anyhow::Error;
    fn try_from(value: usize) -> Result<Self> {
        Self::new(value)
    }
}
impl From<SpatialTileBatch> for usize {
    fn from(value: SpatialTileBatch) -> Self {
        value.maximum_tiles()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpatialTileGroup {
    first_tile: usize,
    tile_count: usize,
}
impl SpatialTileGroup {
    pub const fn first_tile(self) -> usize {
        self.first_tile
    }
    pub const fn tile_count(self) -> usize {
        self.tile_count
    }
}

/// Consecutive groups over a validated, finite row-major tile inventory.
#[derive(Clone, Copy, Debug)]
pub struct SpatialTileGroups {
    total_tiles: usize,
    maximum_tiles: usize,
}
impl SpatialTileGroups {
    pub fn group(self, first_tile: usize) -> Result<SpatialTileGroup> {
        ensure!(
            first_tile < self.total_tiles,
            "spatial group starts outside canvas"
        );
        ensure!(
            first_tile % self.maximum_tiles == 0,
            "spatial group starts outside traversal boundary"
        );
        Ok(SpatialTileGroup {
            first_tile,
            tile_count: self.maximum_tiles.min(self.total_tiles - first_tile),
        })
    }
    pub fn shapes(self) -> Vec<usize> {
        let tail = self.total_tiles % self.maximum_tiles;
        if tail == 0 {
            vec![self.maximum_tiles]
        } else {
            vec![self.maximum_tiles, tail]
        }
    }
    /// Descending real divisors, always ending in scalar fallback. Every
    /// candidate uses one exact shape and needs no padded or alternate tail.
    pub fn uniform_batches(self) -> impl Iterator<Item = SpatialTileBatch> {
        (1..=self.maximum_tiles)
            .rev()
            .filter(move |count| self.total_tiles % count == 0)
            .map(|count| SpatialTileBatch(NonZeroUsize::new(count).expect("positive divisor")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_groups_cover_the_canvas_once_with_only_a_final_tail() {
        for total in 1..=35 {
            for maximum in [1, 2, 3, 4, 7, 12, 16, usize::MAX] {
                let groups = SpatialTileBatch::new(maximum)
                    .unwrap()
                    .groups(total)
                    .unwrap();
                let expected: Vec<_> = (0..total).collect();
                let mut visited = Vec::new();
                let mut shapes = Vec::new();
                while visited.len() < total {
                    let group = groups.group(visited.len()).unwrap();
                    let end = group.first_tile() + group.tile_count();
                    if group.tile_count() < maximum.min(total) {
                        assert_eq!(end, total);
                    }
                    if !shapes.contains(&group.tile_count()) {
                        shapes.push(group.tile_count());
                    }
                    visited.extend(group.first_tile()..end);
                }
                assert_eq!(visited, expected);
                assert_eq!(groups.shapes(), shapes);
                assert!(groups.group(total).is_err());
                assert!(groups.group(usize::MAX).is_err());
                if maximum.min(total) > 1 {
                    assert!(groups.group(1).is_err());
                }
                let candidates: Vec<_> = groups
                    .uniform_batches()
                    .map(SpatialTileBatch::maximum_tiles)
                    .collect();
                assert_eq!(candidates.last(), Some(&1));
                assert!(candidates.windows(2).all(|w| w[0] > w[1]));
                assert!(candidates.iter().all(|n| *n <= maximum && total % n == 0));
            }
        }
    }

    #[test]
    fn limits_are_positive_and_clamped_before_traversal() {
        assert!(SpatialTileBatch::new(0).is_err());
        assert!(serde_json::from_str::<SpatialTileBatch>("0").is_err());
        assert!(SpatialTileBatch::default().groups(0).is_err());
        let huge = SpatialTileBatch::new(usize::MAX).unwrap();
        assert_eq!(
            huge.groups(12)
                .unwrap()
                .uniform_batches()
                .map(SpatialTileBatch::maximum_tiles)
                .collect::<Vec<_>>(),
            [12, 6, 4, 3, 2, 1]
        );
        assert_eq!(
            huge.groups(usize::MAX)
                .unwrap()
                .group(0)
                .unwrap()
                .tile_count(),
            usize::MAX
        );
        assert_eq!(
            serde_json::to_string(&SpatialTileBatch::default()).unwrap(),
            "1"
        );
    }
}
