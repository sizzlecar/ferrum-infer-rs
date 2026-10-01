//! Boundary proof only: no collector, qualified profile, or measured GPU walls.
//! Located here to reuse the production numerical/coverage gates without
//! exporting private phase authority merely for a test.
use super::*;
use ferrum_interfaces::{execution_cost::*, vnext::DeviceCommandPhase};
mod fixture;
use fixture::{checked, domain, fit, population, Wave};

#[test]
fn numerical_family_cross_width_preserves_exact_owners_and_numeric_coordinates() {
    let waves: Vec<_> = [2, 4, 8]
        .into_iter()
        .map(|rows| {
            Wave {
                rows,
                ..Default::default()
            }
            .build()
        })
        .collect();
    let inputs: Vec<_> = waves.iter().map(|w| checked(w).unwrap()).collect();
    for (wave, input) in waves.iter().zip(&inputs) {
        input.same_family(&inputs[0]).unwrap();
        assert_eq!(
            input.original_input().owner().rows as usize,
            wave.exact.rows.len()
        );
        assert_eq!(
            wave.exact.recurrent_state_bytes,
            wave.exact.rows.len() as u64 * 32
        );
        let selected = wave.statistical.as_ref().unwrap();
        let original = StructuredInputV2::from_actual_with_domain(
            &wave.exact,
            selected,
            selected.structured_capture().unwrap().unwrap(),
            &domain(),
        )
        .unwrap();
        assert_eq!(input.original_input(), &original);
        assert_eq!(input.regression_axes(), original.regression_axes());
    }
    for pair in inputs.windows(2) {
        assert_ne!(
            pair[0].original_input().owner(),
            pair[1].original_input().owner()
        );
        assert_ne!(
            pair[0].original_input().domain_signature(),
            pair[1].original_input().domain_signature()
        );
        assert_ne!(pair[0].regression_axes(), pair[1].regression_axes());
        assert_eq!(
            pair[0]
                .original_input()
                .same_domain(pair[1].original_input()),
            Err(StructuredUnknown::WrongDomain)
        );
    }
    let a = waves[0].statistical.as_ref().unwrap();
    let b = waves[1].statistical.as_ref().unwrap();
    assert!(NumericalFamilyInputV1::from_actual(
        &waves[0].exact,
        a,
        b.structured_capture().unwrap().unwrap(),
        &domain()
    )
    .is_err());
}

#[test]
fn numerical_family_separates_algorithm_policy_route_and_domain() {
    let base = checked(&Wave::default().build()).unwrap();
    let changed_policy = HostCostPolicyV2 {
        categorical_signature: [5; 32],
        ..Wave::default().policy
    };
    for wave in [
        Wave {
            algorithm: "fixture.other",
            ..Default::default()
        },
        Wave {
            graph: ActualWaveGraphState::ConfiguredEager,
            ..Default::default()
        },
        Wave {
            retries: 1,
            ..Default::default()
        },
        Wave {
            order: ActualWaveRowOrder::IndependentRows,
            ..Default::default()
        },
        Wave {
            policy: changed_policy,
            ..Default::default()
        },
        Wave {
            policy: HostCostPolicyV2 {
                decoder_scratch_bytes_per_token: 16,
                ..Wave::default().policy
            },
            ..Default::default()
        },
        Wave {
            product: CostProductOutput::FullLogits,
            ..Default::default()
        },
        Wave {
            readback: CoreReadbackRoute::HostSynchronized,
            ..Default::default()
        },
    ] {
        let other = checked(&wave.build()).unwrap();
        assert_eq!(
            base.same_family(&other),
            Err(StructuredUnknown::WrongDomain)
        );
        assert_ne!(base.family().signature(), other.family().signature());
    }
    let d = domain();
    let changed = CostWorkloadDomainV1::new_vnext(
        &ExecutorCostIdentity {
            schema_version: EXECUTOR_COST_IDENTITY_SCHEMA,
            model_weights: [9; 32],
            numerical_policy: [2; 32],
            device_runtime: [3; 32],
            execution_config: [4; 32],
        },
        *d.limits(),
    )
    .unwrap();
    let wave = Wave::default().build();
    let selected = wave.statistical.as_ref().unwrap();
    let other = NumericalFamilyInputV1::from_actual(
        &wave.exact,
        selected,
        selected.structured_capture().unwrap().unwrap(),
        &changed,
    )
    .unwrap();
    assert_eq!(
        base.same_family(&other),
        Err(StructuredUnknown::WrongDomain)
    );
}

#[test]
fn numerical_family_rejects_heterogeneous_policy_first_decode_and_invalid_physical_limits() {
    let heterogeneous = Wave {
        last_row_policy: Some(HostCostPolicyV2 {
            categorical_signature: [5; 32],
            ..Wave::default().policy
        }),
        ..Default::default()
    }
    .build();
    assert!(matches!(
        checked(&heterogeneous),
        Err(StructuredUnknown::UnsupportedScope)
    ));
    assert!(matches!(
        checked(
            &Wave {
                generated: 0,
                ..Default::default()
            }
            .build()
        ),
        Err(StructuredUnknown::UnsupportedScope)
    ));
    // Build genuinely bound inputs with these totals, rather than only corrupting
    // a digest: the declared physical domain must reject the original recipe.
    assert!(matches!(
        checked(
            &Wave {
                state_bytes_per_row: 33,
                ..Default::default()
            }
            .build()
        ),
        Err(StructuredUnknown::WrongDomain)
    ));
    assert!(matches!(
        checked(
            &Wave {
                rows: 9,
                ..Default::default()
            }
            .build()
        ),
        Err(StructuredUnknown::WrongDomain)
    ));
}

#[test]
fn numerical_family_preserves_position_moments_and_actual_row_boundaries() {
    let a = checked(
        &Wave {
            rows: 2,
            pending: vec![1],
            length: vec![1],
            ..Default::default()
        }
        .build(),
    )
    .unwrap();
    let b = checked(
        &Wave {
            rows: 8,
            pending: vec![7],
            length: vec![7],
            ..Default::default()
        }
        .build(),
    )
    .unwrap();
    a.same_family(&b).unwrap();
    for (input, expected) in [(&a, [1., 2., 4.]), (&b, [1., 8., 64.])] {
        let original = input.original_input();
        let offset = original.pending_basis_offset;
        assert_eq!(&original.regression_axes()[offset..], &expected);
        assert_eq!(&original.regression_axes()[offset - 3..offset], &expected);
        assert!(original
            .clone()
            .with_settled_completion(&[original.owner().rows])
            .is_err());
    }
}

#[test]
fn numerical_family_pooled_fit_replays_certificate_and_keeps_both_mask_failures() {
    let (family, clean_fit) = population(StructuredPhaseV2::Fit, false);
    let clean = fit(&clean_fit);
    let (residual_family, with_mask) = population(StructuredPhaseV2::Residual, true);
    assert_eq!(family, residual_family);
    let unseen = envelope::membership_axes(&with_mask[0].input).unwrap();
    assert_eq!(
        clean.check_observed_axes(&unseen),
        Err(StructuredUnknown::UnidentifiedDirection)
    );

    let (family, fit_samples) = population(StructuredPhaseV2::Fit, true);
    let fitted = fit(&fit_samples);
    let (residual_family, clean_residual) = population(StructuredPhaseV2::Residual, false);
    assert_eq!(family, residual_family);
    for sample in &clean_residual {
        fitted
            .check_observed_axes(&envelope::membership_axes(&sample.input).unwrap())
            .unwrap();
    }
    let coverage = ChallengeCoverage::observe(&clean_residual).unwrap();
    coverage.validate_branches(&clean_residual).unwrap();
    assert_eq!(
        coverage.validate_positive_axes(&fitted.certificate().column_maxima),
        Err(StructuredUnknown::QualificationCoverage)
    );

    // Replay the original pooled Fit certificate with every original axis/wall.
    let axes: Vec<_> = fit_samples
        .iter()
        .map(|s| envelope::axes(&s.input).unwrap())
        .collect();
    let numeric: Vec<_> = fit_samples
        .iter()
        .zip(&axes)
        .map(|(s, axes)| FitSample {
            axes,
            wall_ns: s.wall_ns,
        })
        .collect();
    let replayed = NonNegativeFit::from_certificate(
        &numeric,
        &StructuredSettingsV2::default(),
        EnvelopeSettings::default(),
        fitted.certificate().clone(),
    )
    .unwrap();
    assert_eq!(replayed.certificate(), fitted.certificate());
    for phase in [
        StructuredPhaseV2::Residual,
        StructuredPhaseV2::Qualification,
    ] {
        let (phase_family, samples) = population(phase, true);
        assert_eq!(family, phase_family);
        let coverage = ChallengeCoverage::observe(&samples).unwrap();
        coverage.validate_branches(&samples).unwrap();
        coverage
            .validate_positive_axes(&replayed.certificate().column_maxima)
            .unwrap();
        for sample in &samples {
            let input = envelope::membership_axes(&sample.input).unwrap();
            replayed.check_observed_axes(&input).unwrap();
            assert_eq!(
                replayed.predict_fitted(&input).unwrap(),
                fitted.predict_fitted(&input).unwrap()
            );
        }
    }
}
