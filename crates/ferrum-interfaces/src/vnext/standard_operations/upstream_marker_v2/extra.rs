//! The original three-format declarations are inherited verbatim. Extra
//! formats use a separate native arithmetic family and never the G32 branch.
use super::*;

pub(super) fn arithmetic(profile: UpstreamMarkerV2Profile) -> CompositeNumericalArithmetic {
    let base = match profile {
        UpstreamMarkerV2Profile::SwiGluExtraPrefill
        | UpstreamMarkerV2Profile::SwiGluExtraAllRows
        | UpstreamMarkerV2Profile::SwiGluExtraLargePrefill => {
            UpstreamMarkerV2Profile::SwiGluPrefill
        }
        UpstreamMarkerV2Profile::GatedDeltaExtraPrefill
        | UpstreamMarkerV2Profile::GatedDeltaExtraAllRows
        | UpstreamMarkerV2Profile::GatedDeltaExtraLargePrefill => {
            UpstreamMarkerV2Profile::GatedDeltaPrefill
        }
        UpstreamMarkerV2Profile::CausalExtraPrefill
        | UpstreamMarkerV2Profile::CausalExtraAllRows
        | UpstreamMarkerV2Profile::CausalExtraLargePrefill => {
            UpstreamMarkerV2Profile::CausalPrefill
        }
        _ => unreachable!("closed extra profile"),
    };
    let mut contract = base.arithmetic();
    contract.schema_version = if profile.extra_prefill() {
        crate::vnext::COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_EXTRA_PREFILL
    } else {
        crate::vnext::COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_EXTRA
    };
    for projection in &mut contract.projections {
        projection.leaves.extend(
            [
                ProjectionBlockFormat::Q3K,
                ProjectionBlockFormat::Iq3S,
                ProjectionBlockFormat::Iq4Nl,
            ]
            .into_iter()
            .map(|format| QuantizedProjectionLeafContract {
                format,
                // Weight QK=32 for IQ4_NL does not relax the activation/native K256 ABI.
                shape: ProjectionShapeEligibility {
                    minimum_input_features: 256,
                    minimum_output_features: 1,
                    input_features_multiple: 256,
                },
                arithmetic: {
                    let mut policy = policy(format);
                    if profile.extra_all_rows() {
                        // Keep the pre-existing MMVQ rows and route ordering.
                        // Only previously undeclared local widths acquire MMQ.
                        let missing = (1..=32)
                            .filter(|rows| !policy.routes.iter().any(|r| r.contains_rows(*rows)))
                            .collect::<Vec<_>>();
                        policy
                            .routes
                            .iter_mut()
                            .find(|route| {
                                route.arithmetic == UpstreamProjectionArithmetic::MmqD4ExtraMarkerV2
                            })
                            .expect("closed MMQ route")
                            .local_rows
                            .extend(missing);
                    }
                    if profile.extra_prefill() {
                        policy.routes.push(UpstreamProjectionRouteDeclaration {
                            arithmetic: UpstreamProjectionArithmetic::MmqD4ExtraMarkerV2,
                            layout: UpstreamProjectionLayout::Columns,
                            local_rows: Default::default(),
                            prefill_rows: Some(crate::vnext::UpstreamPrefillRows {
                                first: 33,
                                last: 2048,
                            }),
                        });
                    }
                    policy.staged().expect("closed extra policy")
                },
            }),
        );
    }
    contract
}

// Explicit local-row policy. Partial widths and extra-format prefill remain
// strict; no shape, role, model name or offered concurrency selects arithmetic.
// IQ4_NL M8 keeps MMVQ across roles rather than adding a near-tie shape split.
fn policy(format: ProjectionBlockFormat) -> UpstreamProjectionPolicy {
    let (mmvq, mmq): (&[u32], &[u32]) = match format {
        ProjectionBlockFormat::Q3K => (&[1], &[4, 8, 16, 32]),
        ProjectionBlockFormat::Iq3S => (&[1, 4], &[8, 16, 32]),
        ProjectionBlockFormat::Iq4Nl => (&[1, 4, 8], &[16, 32]),
        _ => unreachable!("closed extra policy builder"),
    };
    UpstreamProjectionPolicy {
        format,
        fallback: StrictProjectionFallback::RetainBaseArithmetic {},
        routes: [
            (UpstreamProjectionArithmetic::MmvqQ8_1ExtraMarkerV2, mmvq),
            (UpstreamProjectionArithmetic::MmqD4ExtraMarkerV2, mmq),
        ]
        .into_iter()
        .map(|(arithmetic, rows)| UpstreamProjectionRouteDeclaration {
            arithmetic,
            layout: UpstreamProjectionLayout::Columns,
            local_rows: rows.iter().copied().collect(),
            prefill_rows: None,
        })
        .collect(),
    }
}
