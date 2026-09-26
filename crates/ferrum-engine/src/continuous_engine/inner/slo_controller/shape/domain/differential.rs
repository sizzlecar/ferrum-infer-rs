//! Differential oracle over real provider projections. The reference retains
//! every history and invokes the same product projector without deduplication.
use super::*;
use ferrum_interfaces::vnext::{ExecutionCostRouteProjection, ResourcePlanningUnknown};

impl ExecutorShape<'_> {
    pub(in crate::continuous_engine::inner::slo_controller) fn assert_decode_domain_equivalence(
        &self,
        waves: usize,
    ) {
        let mut frontiers = self.initial_frontiers(&mut || Ok(())).unwrap();
        let mut compact = RouteDomain::initial(self.captured.route.initial_state());
        let mut reference = vec![self.captured.route.initial_state()];
        let mut merged_different_upload_receipts = false;
        let mut checked_transition_after_merge = false;
        for wave in 0..waves {
            let work = [CandidateWork {
                key: frontiers.request(0).key.clone(),
                action: WaveAction::Decode,
            }];
            let prepared = frontiers.prepare(&work, &mut || Ok(())).unwrap();
            let modes = self
                .future_modes(frontiers.requests(), &prepared.rows, &mut || Ok(()))
                .unwrap()
                .unwrap();
            let mut full = Vec::new();
            let mut unknown = None;
            for previous in &reference {
                let representative = compact
                    .states
                    .iter()
                    .find(|state| state.same_future_state(previous, &mut || true).unwrap())
                    .unwrap();
                for &mode in &modes {
                    let mut original_unknown = None;
                    let original = self
                        .project_with_diagnostic(
                            previous,
                            frontiers.requests(),
                            &prepared.rows,
                            mode,
                            &mut || Ok(()),
                            &mut original_unknown,
                        )
                        .unwrap();
                    let mut representative_unknown = None;
                    let repeated = self
                        .project_with_diagnostic(
                            representative,
                            frontiers.requests(),
                            &prepared.rows,
                            mode,
                            &mut || Ok(()),
                            &mut representative_unknown,
                        )
                        .unwrap();
                    assert_eq!(original_unknown, representative_unknown);
                    assert_eq!(original.is_some(), repeated.is_some());
                    checked_transition_after_merge |= reference.len() > compact.states.len();
                    if let Some((projection, forecast)) = original {
                        let (other, other_forecast) = repeated.unwrap();
                        assert!(same_alternative(
                            &projection,
                            forecast.as_ref().unwrap(),
                            &other.shape,
                            other.statistical_evidence.as_ref().unwrap(),
                            other_forecast.as_ref().unwrap(),
                        ));
                        assert!(projection
                            .state
                            .same_future_state(&other.state, &mut || true)
                            .unwrap());
                        full.push((projection, forecast));
                    } else {
                        unknown = Some(original_unknown.expect("typed rejection is required"));
                    }
                }
            }
            if let Some(reason) = unknown {
                // This fixture captures exactly the configured horizon. A
                // fourth projection must retain the real resource boundary,
                // including after histories have already been merged.
                assert_eq!(
                    wave,
                    self.captured
                        .route
                        .resource_view()
                        .limits()
                        .maximum_projected_waves
                );
                assert_eq!(
                    reason,
                    ExecutionCostRouteUnknown::Resource(ResourcePlanningUnknown::LimitExceeded)
                );
                assert!(full.is_empty(), "the limit applies to every history");
                let mut compact_unknown = None;
                assert!(
                    self.project_domain_with_diagnostic(
                        &compact,
                        frontiers.requests(),
                        &prepared.rows,
                        &mut || Ok(()),
                        &mut compact_unknown,
                    )
                    .unwrap()
                    .is_none(),
                    "a domain may not publish a partial known subset"
                );
                assert_eq!(compact_unknown, Some(reason));
                eprintln!("future domain wave {wave}: both populations rejected {reason:?}");
                break;
            }
            let (shapes, statistics, forecasts, next) = self
                .project_domain(&compact, frontiers.requests(), &prepared.rows, &mut || {
                    Ok(())
                })
                .unwrap()
                .unwrap();
            let statistics = statistics.as_ref().unwrap().shapes();
            let forecasts = forecasts.as_ref().unwrap().shapes();
            assert_eq!(shapes.shapes().len(), statistics.len());
            assert_eq!(shapes.shapes().len(), forecasts.len());
            // Set equality preserves every cost/host alternative, including
            // unknown FullLogits owners. Only duplicate multiplicity can fall.
            for (projection, forecast) in &full {
                assert!(shapes.shapes().iter().enumerate().any(|(index, shape)| {
                    same_alternative(
                        projection,
                        forecast.as_ref().unwrap(),
                        shape,
                        &statistics[index],
                        &forecasts[index],
                    )
                }));
                assert!(next.states.iter().any(|state| {
                    state
                        .same_future_state(&projection.state, &mut || true)
                        .unwrap()
                }));
            }
            for (index, shape) in shapes.shapes().iter().enumerate() {
                assert!(full.iter().any(|(projection, forecast)| same_alternative(
                    projection,
                    forecast.as_ref().unwrap(),
                    shape,
                    &statistics[index],
                    &forecasts[index],
                )));
            }
            for state in &next.states {
                assert!(full.iter().any(|(projection, _)| {
                    state
                        .same_future_state(&projection.state, &mut || true)
                        .unwrap()
                }));
            }
            for (index, (left, _)) in full.iter().enumerate() {
                for (right, _) in &full[index + 1..] {
                    if left.state.last_token_mask_uploads() != right.state.last_token_mask_uploads()
                        && left
                            .state
                            .same_future_state(&right.state, &mut || true)
                            .unwrap()
                    {
                        // An upload miss followed by a hit can reach the same
                        // resident mask. Their different current costs must
                        // both survive the alternative-set checks above.
                        assert_ne!(left.shape, right.shape);
                        merged_different_upload_receipts = true;
                    }
                }
            }
            eprintln!(
                "future domain wave {wave}: unmerged={}, alternatives={}, distinct={}",
                full.len(),
                shapes.shapes().len(),
                next.states.len(),
            );
            // Distinct-state capacity admits identical successors at its exact
            // boundary; another frontier is still rejected as a new state.
            let mut at_limit = vec![next.states[0].clone()];
            retain_distinct_state(&mut at_limit, next.states[0].clone(), 1, &mut || Ok(()))
                .unwrap();
            assert_eq!(
                retain_distinct_state(
                    &mut at_limit,
                    self.captured.route.initial_state(),
                    1,
                    &mut || Ok(())
                ),
                Err(PlanningUnknownReason::ShapeCapacity)
            );
            let mut polls = 0;
            assert_eq!(
                retain_distinct_state(&mut at_limit, next.states[0].clone(), 1, &mut || {
                    polls += 1;
                    if polls < 3 {
                        Ok(())
                    } else {
                        Err(PlanningUnknownReason::ComputeBudgetExhausted)
                    }
                }),
                Err(PlanningUnknownReason::ComputeBudgetExhausted)
            );
            reference = full
                .into_iter()
                .map(|(projection, _)| projection.state)
                .collect();
            compact = next;
            frontiers.advance(prepared).unwrap();
        }
        assert!(
            reference.len() > compact.states.len(),
            "reference must retain duplicate histories"
        );
        assert!(
            merged_different_upload_receipts,
            "real mask hits and misses must converge without losing their current costs"
        );
        assert!(
            checked_transition_after_merge,
            "compare a subsequent transition from merged histories"
        );
    }
}

fn same_alternative(
    expected: &ExecutionCostRouteProjection,
    expected_forecast: &HostContentForecastV2,
    shape: &CanonicalWaveCostShape,
    statistics: &StatisticalWaveEvidenceV1,
    forecast: &HostContentForecastV2,
) -> bool {
    let expected_statistics = expected.statistical_evidence.as_ref().unwrap();
    if &expected.shape != shape
        || expected_statistics != statistics
        || expected_statistics.independent_attention_v2() != statistics.independent_attention_v2()
        || expected_statistics.structured_capture() != statistics.structured_capture()
    {
        return false;
    }
    let expected_recipe = expected_statistics.structured_capture().unwrap().unwrap();
    let recipe = statistics.structured_capture().unwrap().unwrap();
    assert_eq!(expected_recipe.algorithm_work(), recipe.algorithm_work());
    expected_forecast
        .validate(&expected.shape, expected_recipe)
        .unwrap();
    forecast.validate(shape, recipe).unwrap();
    match (expected_forecast, forecast) {
        (HostContentForecastV2::Exact, HostContentForecastV2::Exact) => true,
        (HostContentForecastV2::Unresolved(a), HostContentForecastV2::Unresolved(b)) => {
            a.eligible_positions() == b.eligible_positions() && a.constraint() == b.constraint()
        }
        _ => false,
    }
}
