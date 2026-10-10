//! Q6 F16 is a separate arithmetic and boundary ABI. Existing physical leaves
//! keep their exact declarations; selection bounds do not describe ABI limits.
use super::*;
use crate::vnext::{
    UpstreamProjectionGeometrySelection,
    COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_Q6_F16,
};

pub(super) fn arithmetic(profile: UpstreamMarkerV2Profile) -> CompositeNumericalArithmetic {
    let inherited = match profile {
        UpstreamMarkerV2Profile::SwiGluQ6F16 => UpstreamMarkerV2Profile::SwiGluExtraAllRows,
        UpstreamMarkerV2Profile::GatedDeltaQ6F16 => UpstreamMarkerV2Profile::GatedDeltaM8Geometry,
        UpstreamMarkerV2Profile::CausalQ6F16 => UpstreamMarkerV2Profile::CausalM8Geometry,
        _ => unreachable!("closed Q6 F16 profile"),
    };
    let mut contract = inherited.arithmetic();
    contract.schema_version = COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_Q6_F16;
    for projection in &mut contract.projections {
        projection.leaves.push(QuantizedProjectionLeafContract {
            format: ProjectionBlockFormat::Q6K,
            shape: ProjectionShapeEligibility {
                minimum_input_features: 5120,
                minimum_output_features: 1024,
                input_features_multiple: 256,
            },
            arithmetic: UpstreamProjectionPolicy {
                format: ProjectionBlockFormat::Q6K,
                routes: vec![UpstreamProjectionRouteDeclaration {
                    arithmetic: UpstreamProjectionArithmetic::MmqD4Q6F16MarkerV1,
                    layout: UpstreamProjectionLayout::Columns,
                    local_rows: (4..=32).collect(),
                    prefill_rows: None,
                }],
                geometry_selection: Some(UpstreamProjectionGeometrySelection::Q6F16Mmq {
                    minimum_input_features: 5120,
                    minimum_output_features: 1024,
                    minimum_rows_for_small_outputs: 8,
                    minimum_output_features_for_small_rows: 5120,
                }),
                fallback: StrictProjectionFallback::RetainBaseArithmetic {},
            }
            .staged()
            .expect("closed Q6 F16 policy"),
        });
    }
    contract
}
