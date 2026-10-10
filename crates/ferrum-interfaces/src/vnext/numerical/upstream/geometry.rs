//! Explicit geometry-dependent arithmetic. Bounds are declaration data, not a
//! provider heuristic or a statement that a shape has passed a performance gate.
use serde::{Deserialize, Serialize};

use super::{UpstreamProjectionArithmetic, UpstreamProjectionLayout, UpstreamProjectionPolicy};
use crate::vnext::ProjectionBlockFormat;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum UpstreamProjectionGeometrySelection {
    /// Replace the declared Columns/M8 Q4_K or Q5_K MMVQ route by MMQ DS4
    /// only when both logical dimensions of this physical leaf meet the bounds.
    /// Other rows, layouts and dimensions retain their existing declared route.
    M8Q4Q5Mmq {
        minimum_input_features: u64,
        minimum_output_features: u64,
    },
}

impl UpstreamProjectionGeometrySelection {
    pub(super) fn validate(&self, policy: &UpstreamProjectionPolicy) -> Result<(), String> {
        let Self::M8Q4Q5Mmq {
            minimum_input_features,
            minimum_output_features,
        } = self;
        if *minimum_input_features == 0 || *minimum_output_features == 0 {
            return Err("geometry selection requires positive explicit K/N bounds".into());
        }
        if !matches!(
            policy.format,
            ProjectionBlockFormat::Q4K | ProjectionBlockFormat::Q5K
        ) || !policy.routes.iter().any(|route| {
            route.layout == UpstreamProjectionLayout::Columns
                && route.contains_rows(8)
                && route.arithmetic == UpstreamProjectionArithmetic::MmvqQ8_1MarkerV2
        }) {
            return Err(
                "M8 geometry selection requires the Q4K/Q5K Columns MarkerV2 MMVQ base route"
                    .into(),
            );
        }
        Ok(())
    }

    fn selects_mmq(
        self,
        format: ProjectionBlockFormat,
        layout: UpstreamProjectionLayout,
        local_rows: u32,
        input_features: u64,
        leaf_output_features: u64,
    ) -> bool {
        let Self::M8Q4Q5Mmq {
            minimum_input_features,
            minimum_output_features,
        } = self;
        matches!(
            format,
            ProjectionBlockFormat::Q4K | ProjectionBlockFormat::Q5K
        ) && layout == UpstreamProjectionLayout::Columns
            && local_rows == 8
            && input_features >= minimum_input_features
            && leaf_output_features >= minimum_output_features
    }
}

impl UpstreamProjectionPolicy {
    /// Select from a validated declaration using the actual local width and
    /// logical K/N of one physical weight leaf. N is not a composite total,
    /// padded row count or allocation capacity. This grants no resource authority
    /// and does not replace alignment, native geometry or bounds validation.
    pub fn select_arithmetic(
        &self,
        layout: UpstreamProjectionLayout,
        local_rows: u32,
        input_features: u64,
        leaf_output_features: u64,
    ) -> Option<UpstreamProjectionArithmetic> {
        let route = self
            .routes
            .iter()
            .find(|route| route.layout == layout && route.contains_rows(local_rows))?;
        if self.geometry_selection.is_some_and(|selection| {
            selection.selects_mmq(
                self.format,
                layout,
                local_rows,
                input_features,
                leaf_output_features,
            )
        }) {
            Some(UpstreamProjectionArithmetic::MmqDs4MarkerV2)
        } else {
            Some(route.arithmetic)
        }
    }
}
