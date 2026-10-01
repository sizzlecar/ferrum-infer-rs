//! Predeclared synthetic laws, canonical producers and the unchanged 64-sweep
//! solver. This is an identification/solver experiment, not hardware evidence.
use super::*;
use crate::implementations::continuous::cost_model::structured_v2::fit::{FitRow, RowSpaceFit};

const MEMBERS: usize = 48;
const HELD_OUT: [(u32, u32); 3] = [(3, 47), (6, 191), (8, 767)];

// These matrices and this nonnegative work law are fixed before observing any
// prediction. No samples, solver settings or qualification limits are tuned.
fn work_cost(rows: u32, kv: u32) -> u64 {
    1000 + 100 * u64::from(rows) + 10 * u64::from(rows) * (u64::from(kv) + 1)
}

fn grid(rows: &[u32], contexts: &[u32]) -> Vec<(StructuredInputV2, u64)> {
    rows.iter()
        .flat_map(|&b| {
            contexts
                .iter()
                .map(move |&kv| (input(b, kv), work_cost(b, kv)))
        })
        .collect()
}

fn declared_populations(
    u: &DeclaredAlgorithmUniverseV1,
    points: &[Vec<(StructuredInputV2, u64)>; 3],
) -> [Vec<StructuredNumericObservationV2>; 3] {
    let mut phases = repeated_populations(u, MEMBERS);
    for (phase, points) in phases.iter_mut().zip(points) {
        assert_eq!(MEMBERS % points.len(), 0);
        for (i, sample) in phase.iter_mut().enumerate() {
            let (input, wall) = &points[i % points.len()];
            sample.input = input.clone().with_algorithm_universe(u).unwrap();
            sample.wall_ns = *wall;
        }
    }
    for pair in phases.windows(2) {
        assert!(pair[0].last().unwrap().call_id < pair[1][0].call_id);
        assert!(pair[0].last().unwrap().ordinal < pair[1][0].ordinal);
    }
    phases
}

fn finish_reported(
    label: &str,
    fitted: FittedStructuredModelV2,
    phases: &[Vec<StructuredNumericObservationV2>; 3],
) -> Result<QualifiedStructuredModelV2> {
    let calibrated = fitted
        .calibrate_service_window(
            close(StructuredPhaseV2::Residual, &phases[1]),
            &phases[1],
            phases[1].last().unwrap().observed_at_ns,
        )
        .map_err(|reason| {
            eprintln!(
                "{}",
                serde_json::json!({"event":"independent_work_stage_v1",
            "case":label,"phase":"residual","error":format!("{reason:?}")})
            );
            reason
        })?;
    eprintln!(
        "{}",
        serde_json::json!({"event":"independent_work_stage_v1",
        "case":label,"phase":"residual","residual_ns":calibrated.residual_ns,
        "fit_floor_ns":calibrated.fitted.numerical.fit_error_floor_ns(),
        "learned_span_margin_ns":calibrated.learned_span_margin_ns})
    );
    calibrated
        .qualify_service_window(
            close(StructuredPhaseV2::Qualification, &phases[2]),
            &phases[2],
            phases[2].last().unwrap().observed_at_ns,
        )
        .map_err(|reason| {
            eprintln!(
                "{}",
                serde_json::json!({"event":"independent_work_stage_v1",
            "case":label,"phase":"qualification","error":format!("{reason:?}")})
            );
            reason
        })
}

fn qualify_and_replay(
    label: &str,
    u: &DeclaredAlgorithmUniverseV1,
    phases: &[Vec<StructuredNumericObservationV2>; 3],
    probes: &[(StructuredInputV2, u64)],
) -> QualifiedStructuredModelV2 {
    qualify_and_replay_using(
        label,
        u,
        phases,
        probes,
        NonNegativePlanningEstimatorV1::FittedResidualV1,
    )
}

fn qualify_and_replay_using(
    label: &str,
    u: &DeclaredAlgorithmUniverseV1,
    phases: &[Vec<StructuredNumericObservationV2>; 3],
    probes: &[(StructuredInputV2, u64)],
    estimator: NonNegativePlanningEstimatorV1,
) -> QualifiedStructuredModelV2 {
    let fitting = &phases[0];
    let fit = FittedStructuredModelV2::fit_service_window(
        fitting[0].fingerprint.clone(),
        settings(),
        scope(&fitting[0].input),
        source_contract_for(u, estimator, MEMBERS),
        close(StructuredPhaseV2::Fit, fitting),
        fitting,
        fitting.last().unwrap().observed_at_ns,
    )
    .unwrap_or_else(|reason| panic!("{label}: original Fit failed: {reason:?}"));
    let certificate = fit.nonnegative_fit_certificate().unwrap().clone();
    assert_eq!(
        certificate.signed_basis.is_some(),
        estimator == NonNegativePlanningEstimatorV1::IdentifiedEnvelopeV2
    );
    eprintln!(
        "{}",
        serde_json::json!({"event":"independent_work_stage_v1",
        "case":label,"phase":"fit","members":MEMBERS,"estimator":estimator,
        "rank":certificate.geometry_rank,"epsilon_ns":certificate.epsilon_ns,
        "fit_floor_ns":fit.numerical.fit_error_floor_ns()})
    );
    // These are diagnostic evaluations only. They cannot authorize a query
    // before the following independent Residual and Qualification transitions.
    for (i, (raw, actual)) in probes.iter().enumerate() {
        let mapped = raw.clone().with_algorithm_universe(u).unwrap();
        let point = fit.numerical.actual_upper(&mapped).unwrap();
        eprintln!(
            "{}",
            serde_json::json!({"event":"independent_work_unqualified_point_v1",
            "case":label,"probe":i,"actual_ns":actual,"point_ns":point,
            "signed_error_ns":i128::from(point)-i128::from(*actual)})
        );
    }
    let replay_fit = FittedStructuredModelV2::fit_service_window_from_certificate(
        fitting[0].fingerprint.clone(),
        settings(),
        scope(&fitting[0].input),
        source_contract_for(u, estimator, MEMBERS),
        close(StructuredPhaseV2::Fit, fitting),
        fitting,
        fitting.last().unwrap().observed_at_ns,
        certificate,
    )
    .unwrap();
    let qualified = finish_reported(label, fit, phases)
        .unwrap_or_else(|reason| panic!("{label}: independent phase rejected: {reason:?}"));
    let replay = finish_reported(&format!("{label}.replay"), replay_fit, phases).unwrap();
    assert_eq!(
        qualified.parameters_signature(),
        replay.parameters_signature()
    );
    let now = phases[2].last().unwrap().observed_at_ns + 10;
    for (raw, _) in probes {
        let query = StructuredQueryV2::exact(raw.clone());
        let original = qualified
            .predict_query(&fitting[0].fingerprint, &query, now)
            .unwrap();
        let independent = replay
            .predict_query(&fitting[0].fingerprint, &query, now)
            .unwrap();
        assert_eq!(
            (original.fitted_upper_ns, original.planning_ns),
            (independent.fitted_upper_ns, independent.planning_ns)
        );
        assert_eq!(
            query.input(),
            raw,
            "numerical projection must preserve raw execution input"
        );
    }
    qualified
}

fn identified_bounds(
    label: &str,
    model: &QualifiedStructuredModelV2,
    phases: &[Vec<StructuredNumericObservationV2>; 3],
    probes: &[(StructuredInputV2, u64)],
) -> Vec<u64> {
    probes
        .iter()
        .enumerate()
        .map(|(index, (input, actual))| {
            let query = StructuredQueryV2::exact(input.clone());
            let prediction = model
                .predict_query(
                    &phases[0][0].fingerprint,
                    &query,
                    phases[2].last().unwrap().observed_at_ns + 10,
                )
                .unwrap();
            // The public field is this strategy's certified base upper. The
            // unqualified point was logged separately by qualify_and_replay_using.
            assert_eq!(
                prediction.planning_ns,
                prediction.fitted_upper_ns
                    + prediction.fit_error_floor_ns.max(prediction.residual_ns)
                    + settings().static_margin_ns
                    + prediction.learned_span_margin_ns
            );
            assert_eq!(query.input(), input);
            // Reuse this same verified Fit population's original positive
            // certificates. No second fit or changed residual policy is used.
            let physical = model.calibrated.fitted.numerical.physical().unwrap();
            let mapped = physical
                .contract
                .project_input(
                    input
                        .clone()
                        .with_cost_template_policy(physical.contract.template_policy)
                        .unwrap(),
                )
                .unwrap();
            let numeric_query = StructuredQueryV2::exact(mapped);
            let upper_axes =
                envelope::query_upper(&numeric_query, &physical.contract.workload_domain).unwrap();
            let positive_upper = physical.fit.predict_detailed(&upper_axes).unwrap().upper_ns;
            assert!(
                prediction.fitted_upper_ns <= positive_upper,
                "an optional signed witness cannot widen the original positive bound"
            );
            eprintln!(
                "{}",
                serde_json::json!({"event":"identified_geometry_prediction_v2",
            "case":label,"probe":index,"rows":input.owner().rows,"actual_ns":actual,
            "base_upper_ns":prediction.fitted_upper_ns,"positive_upper_ns":positive_upper,
            "planning_ns":prediction.planning_ns,
            "rank":prediction.identified_rank,"fit_floor_ns":prediction.fit_error_floor_ns,
            "residual_ns":prediction.residual_ns,"span_ns":prediction.learned_span_margin_ns})
            );
            prediction.fitted_upper_ns
        })
        .collect()
}

#[test]
fn identified_envelope_geometry_fixed_shape_retains_parameter_ambiguity() {
    let u = DeclaredAlgorithmUniverseV1::from_inputs([&input(1, 64)], 4096).unwrap();
    let phases = declared_populations(&u, &std::array::from_fn(|_| grid(&[1], &[64])));
    let probes: Vec<_> = HELD_OUT
        .into_iter()
        .map(|(b, kv)| (input(b, kv), work_cost(b, kv)))
        .collect();
    let model = qualify_and_replay_using(
        "identified_fixed_shape",
        &u,
        &phases,
        &probes,
        NonNegativePlanningEstimatorV1::IdentifiedEnvelopeV2,
    );
    let bounds = identified_bounds("identified_fixed_shape", &model, &phases, &probes);
    for ((&(rows, kv), &bound), (_, actual)) in HELD_OUT.iter().zip(&bounds).zip(&probes) {
        let anchor = work_cost(1, 64);
        let compatible = [
            anchor,
            anchor * u64::from(rows),
            (anchor * u64::from(rows) * (u64::from(kv) + 1)).div_ceil(65),
        ];
        assert!(
            compatible.iter().all(|&cost| cost <= bound),
            "a signed proposal cannot discard any compatible nonnegative law"
        );
        assert_eq!(
            bound,
            *compatible.iter().max().unwrap(),
            "the original positive certificate still supplies the ambiguity upper"
        );
        assert!(bound >= *actual);
        assert!(
            bound > anchor + settings().static_margin_ns,
            "one observed direction cannot become a fitted-point authority"
        );
    }
}

#[test]
fn identified_envelope_geometry_independent_b_kv_tightens_with_strict_replay() {
    let u = DeclaredAlgorithmUniverseV1::from_inputs([&input(1, 64)], 4096).unwrap();
    let phases = declared_populations(
        &u,
        &[
            grid(&[1, 2, 4, 8], &[7, 63, 255, 511]),
            grid(&[1, 3, 6, 8], &[15, 95, 319]),
            grid(&[1, 2, 5, 7], &[31, 127, 383]),
        ],
    );
    let probes: Vec<_> = HELD_OUT
        .into_iter()
        .map(|(b, kv)| (input(b, kv), work_cost(b, kv)))
        .collect();
    let point_model = qualify_and_replay("point_independent_b_kv", &u, &phases, &probes);
    let model = qualify_and_replay_using(
        "identified_independent_b_kv",
        &u,
        &phases,
        &probes,
        NonNegativePlanningEstimatorV1::IdentifiedEnvelopeV2,
    );
    let bounds = identified_bounds("identified_independent_b_kv", &model, &phases, &probes);
    for ((input, actual), bound) in probes.iter().zip(bounds) {
        let query = StructuredQueryV2::exact(input.clone());
        let now = phases[2].last().unwrap().observed_at_ns + 10;
        let point = point_model
            .predict_query(&phases[0][0].fingerprint, &query, now)
            .unwrap();
        let identified = model
            .predict_query(&phases[0][0].fingerprint, &query, now)
            .unwrap();
        assert_eq!(
            (
                identified.fit_error_floor_ns,
                identified.residual_ns,
                identified.learned_span_margin_ns
            ),
            (
                point.fit_error_floor_ns,
                point.residual_ns,
                point.learned_span_margin_ns
            ),
            "calibration must still use the same fitted point and original populations"
        );
        assert!(
            bound >= *actual,
            "the declared law remains in the feasible set"
        );
        // Use the unchanged fixture's existing 100 ns static margin as a
        // predeclared tightness yardstick, not a new production tolerance.
        assert!(bound - actual <= settings().static_margin_ns,
            "independent work directions should give a tight integer witness, actual={actual}, upper={bound}");
    }
}

#[test]
fn identified_envelope_geometry_fixed_overhead_anchor_tightens_composition() {
    let raw = [
        original(&["H"], 1, false),
        original(&["H", "A"], 1, false),
        original(&["H", "B"], 1, false),
        original(&["H", "A", "B"], 1, false),
    ];
    let costs = [1200u64, 1700, 1900, 2400];
    let u = DeclaredAlgorithmUniverseV1::from_inputs(raw.iter(), 4096).unwrap();
    let mut bounds = Vec::new();
    for anchored in [false, true] {
        let indices: &[usize] = if anchored { &[0, 1, 2] } else { &[1, 2] };
        let points = std::array::from_fn(|_| {
            indices
                .iter()
                .map(|&i| {
                    (
                        raw[i]
                            .clone()
                            .with_cost_template_policy(
                                StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1,
                            )
                            .unwrap(),
                        costs[i],
                    )
                })
                .collect()
        });
        let phases = declared_populations(&u, &points);
        let probes = [(raw[3].clone(), costs[3])];
        let label = if anchored {
            "identified_composition_with_h_anchor"
        } else {
            "identified_composition_without_h_anchor"
        };
        let model = qualify_and_replay_using(
            label,
            &u,
            &phases,
            &probes,
            NonNegativePlanningEstimatorV1::IdentifiedEnvelopeV2,
        );
        let bound = identified_bounds(label, &model, &phases, &probes)[0];
        assert!(bound >= costs[3]);
        if anchored {
            assert!(
                bound - costs[3] <= settings().static_margin_ns,
                "the real H fixed-overhead anchor should identify HAB tightly"
            );
        } else {
            // A third compatible law has intercept=H=0, A=1700, B=1900:
            // HA/HB still match, but HAB=3600. It must remain covered.
            assert!(
                bound >= costs[1] + costs[2],
                "two rows cannot determine the shared overhead subtraction"
            );
        }
        bounds.push(bound);
    }
    assert!(
        bounds[1] < bounds[0],
        "the added input direction must tighten the bound"
    );
}

fn predictions(
    label: &str,
    model: &QualifiedStructuredModelV2,
    phases: &[Vec<StructuredNumericObservationV2>; 3],
    probes: &[(StructuredInputV2, u64)],
) -> Vec<(u64, u64)> {
    probes
        .iter()
        .enumerate()
        .map(|(i, (raw, actual))| {
            let query = StructuredQueryV2::exact(raw.clone());
            let prediction = model
                .predict_query(
                    &phases[0][0].fingerprint,
                    &query,
                    phases[2].last().unwrap().observed_at_ns + 10,
                )
                .unwrap();
            eprintln!(
                "{}",
                serde_json::json!({"event":"independent_work_prediction_v1",
            "case":label,"probe":i,"rows":raw.owner().rows,
            "actual_ns":actual,"point_ns":prediction.fitted_upper_ns,
            "planning_ns":prediction.planning_ns,
            "signed_error_ns":i128::from(prediction.fitted_upper_ns)-i128::from(*actual),
            "planning_underestimate_ns":actual.saturating_sub(prediction.planning_ns),
            "rank":prediction.identified_rank,"fit_floor_ns":prediction.fit_error_floor_ns,
            "residual_ns":prediction.residual_ns,"span_ns":prediction.learned_span_margin_ns})
            );
            (prediction.fitted_upper_ns, prediction.planning_ns)
        })
        .collect()
}

#[test]
fn fitted_residual_independent_geometry_fixed_shape_negative_control() {
    let u = DeclaredAlgorithmUniverseV1::from_inputs([&input(1, 64)], 4096).unwrap();
    let phases = declared_populations(&u, &std::array::from_fn(|_| grid(&[1], &[64])));
    let probes: Vec<_> = HELD_OUT
        .into_iter()
        .map(|(b, kv)| (input(b, kv), work_cost(b, kv)))
        .collect();
    let model = qualify_and_replay("fixed_shape", &u, &phases, &probes);
    let values = predictions("fixed_shape", &model, &phases, &probes);
    for ((point, planning), (_, actual)) in values.iter().zip(&probes) {
        assert_eq!(*point, work_cost(1, 64));
        assert_eq!(*planning, work_cost(1, 64) + settings().static_margin_ns);
        assert!(
            *planning < *actual,
            "the unseen scaling remains unidentified"
        );
    }
}

#[test]
fn fitted_residual_independent_geometry_learns_predeclared_work_and_qualifies() {
    let u = DeclaredAlgorithmUniverseV1::from_inputs([&input(1, 64)], 4096).unwrap();
    let points = [
        grid(&[1, 2, 4, 8], &[7, 63, 255, 511]),
        grid(&[1, 3, 6, 8], &[15, 95, 319]),
        grid(&[1, 2, 5, 7], &[31, 127, 383]),
    ];
    let phases = declared_populations(&u, &points);
    let probes: Vec<_> = HELD_OUT
        .into_iter()
        .map(|(b, kv)| (input(b, kv), work_cost(b, kv)))
        .collect();
    let model = qualify_and_replay("independent_b_kv", &u, &phases, &probes);
    let values = predictions("independent_b_kv", &model, &phases, &probes);
    assert!(values.windows(2).all(|v| v[0].0 < v[1].0));
    let mut fitted_error = 0u64;
    let mut fixed_error = 0u64;
    for ((point, planning), (_, actual)) in values.iter().zip(&probes) {
        fitted_error += point.abs_diff(*actual);
        fixed_error += work_cost(1, 64).abs_diff(*actual);
        assert!(*planning >= *actual,
            "predeclared held-out law exceeds unchanged planning authority: actual={actual}, point={point}, planning={planning}");
    }
    assert!(
        fitted_error < fixed_error,
        "independent geometry must improve over the predeclared fixed-shape control"
    );
}

#[test]
fn fitted_residual_independent_geometry_composition_needs_fixed_overhead_anchor() {
    // H is real common work, not a fabricated zero-command/zero-row owner.
    // Law: intercept=1000, H=200, A=500, B=700 ns per selected command.
    let raw = [
        original(&["H"], 1, false),
        original(&["H", "A"], 1, false),
        original(&["H", "B"], 1, false),
        original(&["H", "A", "B"], 1, false),
    ];
    let u = DeclaredAlgorithmUniverseV1::from_inputs(raw.iter(), 4096).unwrap();
    let inputs: Vec<_> = raw
        .iter()
        .cloned()
        .map(|input| {
            input
                .with_cost_template_policy(StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1)
                .unwrap()
                .with_algorithm_universe(&u)
                .unwrap()
        })
        .collect();
    let axes: Vec<_> = inputs.iter().map(|i| i.regression_axes()).collect();
    for j in 0..axes[0].len() {
        assert_eq!(
            axes[0][j] + axes[3][j],
            axes[1][j] + axes[2][j],
            "full canonical affine composition failed at axis {j}"
        );
    }
    let costs = [1200u64, 1700, 1900, 2400];
    // Another entirely nonnegative law, intercept=1200,H=200,A=300,B=500,
    // agrees at HA/HB but gives 2200 instead of 2400 at HAB.
    let alternative = [1400u64, 1700, 1900, 2200];
    assert_eq!(&costs[1..3], &alternative[1..3]);
    assert_ne!(costs[3], alternative[3]);
    let mut outputs = Vec::new();
    for anchored in [false, true] {
        let indices: &[usize] = if anchored { &[0, 1, 2] } else { &[1, 2] };
        let rows: Vec<_> = (0..MEMBERS)
            .map(|i| {
                let j = indices[i % indices.len()];
                FitRow {
                    basis: axes[j],
                    wall_ns: costs[j],
                }
            })
            .collect();
        let geometry = RowSpaceFit::fit(&rows, &settings()).unwrap();
        if anchored {
            assert_eq!(geometry.rank(), 3);
            geometry.identify(axes[3]).unwrap();
        } else {
            assert_eq!(geometry.rank(), 2);
            assert!(matches!(
                geometry.identify(axes[3]),
                Err(StructuredUnknown::UnidentifiedDirection)
            ));
        }
        let declarations = std::array::from_fn(|_| {
            indices
                .iter()
                .map(|&i| {
                    (
                        raw[i]
                            .clone()
                            .with_cost_template_policy(
                                StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1,
                            )
                            .unwrap(),
                        costs[i],
                    )
                })
                .collect()
        });
        let phases = declared_populations(&u, &declarations);
        let probes = [(raw[3].clone(), costs[3])];
        let label = if anchored {
            "composition_with_h_anchor"
        } else {
            "composition_without_h_anchor"
        };
        let model = qualify_and_replay(label, &u, &phases, &probes);
        let values = predictions(label, &model, &phases, &probes);
        if anchored {
            assert!(
                values[0].1 >= costs[3],
                "identified composition must pass the unchanged planning bound"
            );
        }
        outputs.push(values[0]);
    }
    eprintln!(
        "{}",
        serde_json::json!({"event":"independent_composition_comparison_v1",
        "compatible_unanchored_costs_ns":[costs[3],alternative[3]],
        "without_anchor":outputs[0],"with_anchor":outputs[1]})
    );
}
