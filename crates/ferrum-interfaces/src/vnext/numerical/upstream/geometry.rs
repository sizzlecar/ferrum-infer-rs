//! Physical-leaf selection for the independent Q6 F16 boundary. These are
//! declared numerical domains, not runtime model-name or performance heuristics.
use serde::{Deserialize, Serialize};

use super::{UpstreamProjectionArithmetic, UpstreamProjectionLayout, UpstreamProjectionPolicy};
use crate::vnext::ProjectionBlockFormat;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum UpstreamProjectionGeometrySelection {
    Q6F16Mmq {
        minimum_input_features: u64,
        minimum_output_features: u64,
        minimum_rows_for_small_outputs: u32,
        minimum_output_features_for_small_rows: u64,
    },
}

impl UpstreamProjectionGeometrySelection {
    pub(super) fn validate(self, policy: &UpstreamProjectionPolicy) -> Result<(), String> {
        let Self::Q6F16Mmq {
            minimum_input_features,
            minimum_output_features,
            minimum_rows_for_small_outputs,
            minimum_output_features_for_small_rows,
        } = self;
        if minimum_input_features == 0
            || minimum_output_features == 0
            || !(1..=32).contains(&minimum_rows_for_small_outputs)
            || minimum_output_features_for_small_rows < minimum_output_features
            || policy.format != ProjectionBlockFormat::Q6K
            || policy.routes.iter().any(|r| {
                r.layout != UpstreamProjectionLayout::Columns
                    || r.arithmetic != UpstreamProjectionArithmetic::MmqD4Q6F16MarkerV1
            })
        {
            return Err("Q6 F16 selection requires positive bounds and its Columns ABI".into());
        }
        Ok(())
    }
}

impl UpstreamProjectionPolicy {
    /// Uses logical dimensions of one physical leaf, never a joined/padded N.
    /// Selection grants no resource authority and cannot bypass native bounds.
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
            .find(|r| r.layout == layout && r.contains_rows(local_rows))?;
        if let Some(UpstreamProjectionGeometrySelection::Q6F16Mmq {
            minimum_input_features,
            minimum_output_features,
            minimum_rows_for_small_outputs,
            minimum_output_features_for_small_rows,
        }) = self.geometry_selection
        {
            return (self.format == ProjectionBlockFormat::Q6K
                && layout == UpstreamProjectionLayout::Columns
                && input_features >= minimum_input_features
                && input_features % 256 == 0
                && leaf_output_features >= minimum_output_features
                && (local_rows >= minimum_rows_for_small_outputs
                    || leaf_output_features >= minimum_output_features_for_small_rows))
                .then_some(route.arithmetic);
        }
        Some(route.arithmetic)
    }
}
