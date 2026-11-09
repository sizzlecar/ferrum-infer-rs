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

/// Actual adjacent tiles in one row, without padding. Returned by the model's
/// traversal; callers select the admitted recipe with this exact tile count.
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
        let column = first_tile % self.x.starts.len();
        ensure!(
            column % batch.0 == 0,
            "VAE group must start at its declared row boundary"
        );
        Ok(PH3ImageVaeTileGroup {
            first_tile,
            tile_count: batch.0.min(self.x.starts.len() - column),
        })
    }
    pub fn group_shapes(&self, batch: PH3ImageVaeTileBatch) -> Vec<usize> {
        let largest = batch.0.min(self.x.starts.len());
        let tail = self.x.starts.len() % largest;
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
    fn declared_groups_have_real_row_tails_and_no_padding() {
        for (size, expected, total) in [(2912, vec![4, 3], 60), (1024, vec![4, 1], 10)] {
            let plan = PH3ImageVaeTilePlan::new(
                size,
                size,
                16,
                PH3ImageVaeSpatialPolicy::comfy_default(256),
            )
            .unwrap();
            let batch = PH3ImageVaeTileBatch::new(4).unwrap();
            assert_eq!(plan.group_shapes(batch), expected);
            let mut first = 0;
            let mut groups = 0;
            while first < plan.tile_count() {
                let g = plan.group(first, batch).unwrap();
                assert_eq!(g.first_tile(), first);
                assert_eq!(
                    plan.tile(first).unwrap().y,
                    plan.tile(first + g.tile_count() - 1).unwrap().y
                );
                first += g.tile_count();
                groups += 1;
            }
            assert_eq!(first, plan.tile_count());
            assert_eq!(groups, total);
            assert!(plan.group(1, batch).is_err());
        }
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
