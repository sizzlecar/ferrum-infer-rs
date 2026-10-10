//! Independently versioned attention arithmetic. These lower bounds are an
//! explicit numerical policy, not an adaptive backend tuning decision.
use super::*;
use crate::vnext::{
    UpstreamProjectionGeometrySelection,
    COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_GEOMETRY,
};

pub(super) fn arithmetic(profile: UpstreamMarkerV2Profile) -> CompositeNumericalArithmetic {
    let old = match profile {
        UpstreamMarkerV2Profile::GatedDeltaM8Geometry => {
            UpstreamMarkerV2Profile::GatedDeltaExtraAllRows
        }
        UpstreamMarkerV2Profile::CausalM8Geometry => UpstreamMarkerV2Profile::CausalExtraAllRows,
        _ => unreachable!("closed attention geometry profile"),
    };
    let mut contract = old.arithmetic();
    contract.schema_version = COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_GEOMETRY;
    for projection in &mut contract.projections {
        for leaf in &mut projection.leaves {
            if matches!(
                leaf.format,
                ProjectionBlockFormat::Q4K | ProjectionBlockFormat::Q5K
            ) {
                let mut policy = leaf
                    .arithmetic
                    .upstream_policy()
                    .expect("closed upstream leaf")
                    .clone();
                policy.geometry_selection = Some(UpstreamProjectionGeometrySelection::M8Q4Q5Mmq {
                    minimum_input_features: 5120,
                    minimum_output_features: 5120,
                });
                leaf.arithmetic = policy.staged().expect("closed geometry policy");
            }
        }
    }
    contract
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_geometry_profile_changes_only_m8_large_q4_q5_leaf_arithmetic() {
        for (new, old) in [
            (
                UpstreamMarkerV2Profile::GatedDeltaM8Geometry,
                UpstreamMarkerV2Profile::GatedDeltaExtraAllRows,
            ),
            (
                UpstreamMarkerV2Profile::CausalM8Geometry,
                UpstreamMarkerV2Profile::CausalExtraAllRows,
            ),
        ] {
            let current = new.arithmetic();
            let prior = old.arithmetic();
            current.validate().unwrap();
            assert_ne!(new.operation_id(), old.operation_id());
            assert_ne!(new.capability_id(), old.capability_id());
            assert_eq!(new.strict_operation_id(), old.strict_operation_id());
            assert!(new.prefill() && new.extra_all_rows() && new.extra_prefill());
            let mut restored = current.clone();
            restored.schema_version = prior.schema_version;
            for projection in &mut restored.projections {
                for leaf in &mut projection.leaves {
                    let policy = leaf.arithmetic.upstream_policy().unwrap();
                    for (k, n) in [(4096, 5120), (5120, 4096), (5120, 5120), (6144, 12288)] {
                        for m in [1, 2, 4, 8, 16, 32, 33, 2048] {
                            let old_policy = prior
                                .projections
                                .iter()
                                .find(|p| p.role == projection.role)
                                .unwrap()
                                .leaves
                                .iter()
                                .find(|l| l.format == leaf.format)
                                .unwrap()
                                .arithmetic
                                .upstream_policy()
                                .unwrap();
                            let expected = if m == 8
                                && k >= 5120
                                && n >= 5120
                                && matches!(
                                    leaf.format,
                                    ProjectionBlockFormat::Q4K | ProjectionBlockFormat::Q5K
                                ) {
                                Some(UpstreamProjectionArithmetic::MmqDs4MarkerV2)
                            } else {
                                old_policy.select_arithmetic(
                                    UpstreamProjectionLayout::Columns,
                                    m,
                                    k,
                                    n,
                                )
                            };
                            assert_eq!(
                                policy.select_arithmetic(
                                    UpstreamProjectionLayout::Columns,
                                    m,
                                    k,
                                    n
                                ),
                                expected
                            );
                        }
                    }
                    if policy.geometry_selection.is_some() {
                        let mut restored_policy = policy.clone();
                        restored_policy.geometry_selection = None;
                        leaf.arithmetic = restored_policy.staged().unwrap();
                    }
                }
            }
            assert_eq!(
                serde_json::to_vec(&restored).unwrap(),
                serde_json::to_vec(&prior).unwrap()
            );
        }
    }
}
