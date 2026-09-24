//! Real native route evidence, not a latency-model qualification. One actual
//! partial wave publishes an ExactV1 snapshot at its unchanged min_samples=8;
//! that snapshot can still return insufficient samples. The assertions below
//! compare complete projected shapes to the actual native/host recorder.
use super::*;
use crate::continuous_engine::inner::calibration::{
    CalibrationAction, CalibrationBlockReason, CalibrationFrontier, CalibrationLimits,
    CalibrationObservation, CalibrationSession, CalibrationSubmissionState, CalibrationTurn,
    CalibrationWaveReport, CalibrationWork,
};
use ferrum_interfaces::vnext::{ResourcePlanningAvailability, ResourcePlanningUnknown};
use ferrum_scheduler::implementations::continuous::slo_planner::{
    canonical_cost_shape, PlanningExecutionState, ProjectedExecution,
};

// Reuse the already exercised real model generator; no production export and
// no second safetensors/config implementation.
mod fixture;
#[path = "../../../../../../ferrum-models/src/executor/vnext_executor/cost_observation/product_tests/weights.rs"]
mod weights;
use fixture::{add, admit, fixture, frontier, ready, wave};

/// This artificial cost is used only to exercise certificate delivery. The
/// actual min-eight-samples model above remains unqualified; native canonical
/// shapes and physical resource projections still come from the real backend.
fn selected_partial_proof(inner: &EngineInner, captured: &mut ControllerSnapshot) -> SelectedWave {
    use ferrum_scheduler::implementations::continuous::cost_model::{
        ExecutionFingerprint, WaveExecutionShape,
    };
    struct WiringCost(u64);
    impl PlanningCostModel for WiringCost {
        fn model_version(&self) -> u64 {
            self.0
        }
        fn predict(
            &self,
            _: &ExecutionFingerprint,
            _: &WaveExecutionShape,
            _: u64,
        ) -> Option<PlanningCost> {
            Some(PlanningCost {
                typical_ns: 1,
                planning_ns: 1,
                model_version: self.0,
                valid_for_ns: u64::MAX,
            })
        }
    }
    struct FixedClock(u64);
    impl PlanningClock for FixedClock {
        fn now_ns(&mut self) -> u64 {
            self.0
        }
    }
    // This tiny fixture has a 30-second TTFT and a four-token prompt. Limit
    // this wiring check to its legal one-token partial edge and finite horizon,
    // without pretending unknown post-first-token content/cost is supported.
    captured.snapshot.capabilities.prefill_chunk_sizes = vec![n32(1)];
    captured.snapshot.scope.horizon_end_ns = captured.snapshot.observed_at_ns + 1_000_000_000;
    let mut config = inner.config.scheduler.slo.planner.clone();
    config.lookahead_waves = NonZeroUsize::new(1).unwrap();
    config.candidate_limit = NonZeroUsize::new(1).unwrap();
    config.beam_width = NonZeroUsize::new(1).unwrap();
    let planner = BoundedSloPlanner {
        settings: BoundedPlannerSettings { search: config },
    };
    let context = shape::ExecutorShape {
        engine: inner,
        captured,
    };
    let decision = planner.propose_with_execution(
        &captured.snapshot,
        &WiringCost(captured.snapshot.cost_model_version),
        &context,
        &mut FixedClock(captured.snapshot.observed_at_ns),
    );
    match decision {
        PlanningDecision::FeasibleWithinHorizon { first_wave, .. } => first_wave,
        other => panic!("real native first-canonical delivery with test-only cost: {other:?}"),
    }
}

fn n32(value: u32) -> NonZeroU32 {
    NonZeroU32::new(value).unwrap()
}

fn project<'a>(
    parent: &dyn PlanningExecutionState<'a>,
    requests: &[RequestSchedulingView],
    work: &[CandidateWork],
) -> Option<ProjectedExecution<'a>> {
    let rows: Vec<_> = work
        .iter()
        .map(|work| {
            let request = requests.iter().find(|r| r.key == work.key).unwrap();
            let actual = match work.action {
                WaveAction::Decode => ActualRowWork::Decode {
                    kv_tokens: request.context_tokens,
                },
                WaveAction::Prefill { offset, count } => {
                    let RequestPhaseView::Prefill(progress) = &request.phase else {
                        panic!("actual prefill parent")
                    };
                    ActualRowWork::Prefill {
                        offset,
                        count: count.get(),
                        total_prompt_tokens: progress.total_prompt_tokens.get(),
                    }
                }
            };
            PlanningShapeRow {
                request,
                work: actual,
            }
        })
        .collect();
    let prefill = rows
        .iter()
        .any(|r| matches!(r.work, ActualRowWork::Prefill { .. }));
    let decode = rows
        .iter()
        .any(|r| matches!(r.work, ActualRowWork::Decode { .. }));
    parent
        .project(
            &PlanningExecutionInput {
                work,
                requests,
                kind: match (prefill, decode) {
                    (true, true) => ActualWaveKind::Mixed,
                    (true, false) => ActualWaveKind::Prefill,
                    (false, true) => ActualWaveKind::Decode,
                    _ => panic!("nonempty actual work"),
                },
                recurrent_state_bytes: rows.iter().map(|r| r.request.recurrent_state_bytes).sum(),
                rows: &rows,
            },
            &mut || Ok(()),
        )
        .unwrap()
}

fn assert_actual(expected: &CanonicalWaveCostShape, report: &CalibrationWaveReport) {
    assert!(
        report.error.is_none(),
        "actual execution error: {:?}",
        report.error
    );
    assert_eq!(
        report.submission,
        CalibrationSubmissionState::HostReconciled
    );
    let CalibrationObservation::Observed {
        sample,
        actual_rows,
        commits,
        ..
    } = &report.observation
    else {
        panic!(
            "missing actual joined native/host observation: {:?}",
            report.observation
        )
    };
    // The representation conversion preserves every canonical field. Verify
    // physical row order separately because the cost DTO groups phase arrays.
    assert_eq!(
        actual_rows.iter().map(|r| r.work).collect::<Vec<_>>(),
        expected.rows
    );
    assert_eq!(actual_rows.len(), commits.len());
    assert_eq!(canonical_cost_shape(expected).unwrap(), sample.actual_shape);
    assert_eq!(sample.actual_shape.restore_bytes, 0);
    assert_eq!(sample.actual_shape.maintenance_bytes, 0);
    let stages = report
        .host_stages
        .as_ref()
        .expect("actual host-stage evidence");
    assert_eq!(stages.actual_shape.as_ref(), Some(&sample.actual_shape));
}

fn capture(inner: &EngineInner) -> ControllerSnapshot {
    inner
        .capture_slo_controller_snapshot(
            &ferrum_interfaces::BatchHint::simple(8),
            ControllerBudget::new(slo_clock_now(), Duration::from_secs(30)).unwrap(),
        )
        .unwrap_or_else(|error| panic!("actual controller capture: {error:?}"))
}

fn assert_no_live_effects(
    inner: &EngineInner,
    captured: &ControllerSnapshot,
    counters: &serde_json::Value,
) {
    assert_eq!(
        &inner.model_executor.cache_metrics_snapshot().unwrap()["counters"],
        counters
    );
    let requests: Vec<_> = captured
        .fences
        .iter()
        .map(|f| ExecutorResourcePlanningRequest {
            request_id: &f.key.request_id,
            cache_id: f.resource_cache_id(),
        })
        .collect();
    let ResourcePlanningAvailability::Known(after) = inner
        .model_executor
        .execution_resource_planning_view(&requests, captured.resources.limits(), &mut || true)
    else {
        panic!("same live native resource evidence unavailable")
    };
    assert!(
        captured.resources.same_live_evidence(&after),
        "pure projection changed physical/logical allocation evidence"
    );
    let sequences = inner.sequences.read();
    for fence in &captured.fences {
        let actual = &sequences[&fence.key.request_id];
        assert_eq!(
            actual.prefill_tokens_processed,
            fence.prefill_tokens_processed
        );
        assert_eq!(actual.generated_tokens.len(), fence.generated);
        assert_eq!(
            actual.cost_frontier.unwrap().work_generation.get(),
            fence.generation
        );
    }
}

#[tokio::test]
async fn unified_execution_metal_partial_final_successor_and_recaptured_decode_mixed_match_native()
{
    let (mut session, directory) = fixture().await;
    let inner = session.test_engine_inner();
    let (id, output) = add(&mut session).await;
    admit(&mut session, &id).await;

    // Bootstrap only the real observation runtime. No synthetic costs or
    // thresholds: the original default min_samples remains eight.
    let warm = frontier(&session, &id).prefill_work(n32(1)).unwrap();
    let first = wave(&mut session, vec![warm]).await;
    assert!(matches!(
        &first.observation,
        CalibrationObservation::Observed { .. }
    ));
    let frozen = session.freeze_cost_model().await.unwrap();
    assert!(frozen.model_version().is_some());
    let CalibrationObservation::Observed { sample, .. } = &first.observation else {
        unreachable!()
    };
    assert!(matches!(frozen.predict(&sample.actual_shape).unwrap(), Some(
        ferrum_scheduler::implementations::continuous::cost_model::CostPrediction::Unknown(
            ferrum_scheduler::implementations::continuous::cost_model::CostUnknownReason::InsufficientSamples
        )
    )), "real route evidence must not be mislabeled as trained cost coverage");
    assert_eq!(
        session
            .configuration()
            .scheduler
            .slo
            .cost_observation
            .model
            .min_samples
            .get(),
        8
    );

    let mut captured = capture(&inner);
    let replayed = selected_partial_proof(&inner, &mut captured);
    let valid_until = captured
        .origin
        .instant_at_ns(replayed.planning_observed_at_ns + replayed.witness_valid_for_ns)
        .unwrap();
    let delivered = inner
        .controller_first_wave_shape(&captured, &replayed, valid_until)
        .expect("consume the real final-replay first canonical without a third projection");
    let mut absent = replayed.clone();
    absent.final_replay_first_wave = None;
    assert!(inner
        .controller_first_wave_shape(&captured, &absent, valid_until)
        .is_none());
    let mut changed = replayed;
    changed.candidate.work[0].key.incarnation += 1;
    assert!(inner
        .controller_first_wave_shape(&captured, &changed, valid_until)
        .is_none());
    let original = captured.snapshot.requests.clone();
    let counters = inner.model_executor.cache_metrics_snapshot().unwrap()["counters"].clone();
    let context = shape::ExecutorShape {
        engine: &inner,
        captured: &captured,
    };
    let parent = context.begin(&captured.snapshot, &mut || Ok(())).unwrap();
    let work = [CandidateWork {
        key: original[0].key.clone(),
        action: WaveAction::Prefill {
            offset: 1,
            count: n32(1),
        },
    }];
    let partial = project(parent.as_ref(), &original, &work).expect("real partial route");
    let partial_shape = partial.canonical_domain.exact().unwrap().clone();
    assert_eq!(
        delivered, partial_shape,
        "selected replay edge equals the real native route"
    );
    // A sibling reads the unchanged parent, not the preceding successor.
    assert_eq!(
        project(parent.as_ref(), &original, &work)
            .unwrap()
            .canonical_domain,
        partial.canonical_domain
    );
    let mut after_partial = original.clone();
    after_partial[0].context_tokens = 2;
    let RequestPhaseView::Prefill(progress) = &mut after_partial[0].phase else {
        panic!()
    };
    progress.offset = 2;
    progress.logical_high_water = 2;
    let final_work = [CandidateWork {
        key: original[0].key.clone(),
        action: WaveAction::Prefill {
            offset: 2,
            count: n32(2),
        },
    }];
    let final_edge = project(partial.successor.as_ref(), &after_partial, &final_work)
        .expect("same-capture final successor");
    let final_shape = final_edge.canonical_domain.exact().unwrap().clone();
    assert_eq!(captured.snapshot.requests, original);
    assert_no_live_effects(&inner, &captured, &counters);

    // Content after this hypothetical final token is unknown to the exact
    // model. This test does not manufacture an empirical future domain.
    let mut after_final = after_partial;
    after_final[0].phase = RequestPhaseView::Decode;
    after_final[0].context_tokens = 4;
    after_final[0].timing.committed_tokens = 1;
    after_final[0].timing.first_commit_at_ns = Some(captured.snapshot.observed_at_ns);
    after_final[0].timing.last_commit_at_ns = Some(captured.snapshot.observed_at_ns);
    assert!(project(
        final_edge.successor.as_ref(),
        &after_final,
        &[CandidateWork {
            key: original[0].key.clone(),
            action: WaveAction::Decode,
        }]
    )
    .is_none());
    assert_no_live_effects(&inner, &captured, &counters);
    drop(final_edge);
    drop(partial);
    drop(parent);
    drop(context);
    drop(captured);

    let selected = frontier(&session, &id).prefill_work(n32(1)).unwrap();
    assert_actual(&delivered, &wave(&mut session, vec![selected]).await);
    let selected = frontier(&session, &id).prefill_work(n32(2)).unwrap();
    assert_actual(&final_shape, &wave(&mut session, vec![selected]).await);
    ready(&inner, &id).await;

    // A fresh capture now knows the actual committed token and its policy.
    let captured = capture(&inner);
    let counters = inner.model_executor.cache_metrics_snapshot().unwrap()["counters"].clone();
    let context = shape::ExecutorShape {
        engine: &inner,
        captured: &captured,
    };
    let parent = context.begin(&captured.snapshot, &mut || Ok(())).unwrap();
    let edge = project(
        parent.as_ref(),
        &captured.snapshot.requests,
        &[CandidateWork {
            key: captured.snapshot.requests[0].key.clone(),
            action: WaveAction::Decode,
        }],
    )
    .expect("recaptured actual decode route");
    let expected = edge.canonical_domain.exact().unwrap().clone();
    assert_no_live_effects(&inner, &captured, &counters);
    drop(edge);
    drop(parent);
    drop(context);
    drop(captured);
    let selected = frontier(&session, &id).decode_work().unwrap();
    assert_actual(&expected, &wave(&mut session, vec![selected]).await);
    ready(&inner, &id).await;

    let (other, other_output) = add(&mut session).await;
    admit(&mut session, &other).await;
    let captured = capture(&inner);
    let counters = inner.model_executor.cache_metrics_snapshot().unwrap()["counters"].clone();
    let context = shape::ExecutorShape {
        engine: &inner,
        captured: &captured,
    };
    let parent = context.begin(&captured.snapshot, &mut || Ok(())).unwrap();
    let work: Vec<_> = captured
        .snapshot
        .requests
        .iter()
        .rev()
        .map(|r| CandidateWork {
            key: r.key.clone(),
            action: if r.key.request_id == id {
                WaveAction::Decode
            } else {
                WaveAction::Prefill {
                    offset: 0,
                    count: n32(1),
                }
            },
        })
        .collect();
    // Startup prepares declared single-owner packed buckets, not this
    // two-owner transient mixed workspace. Preserve the cold Unknown result.
    assert!(project(parent.as_ref(), &captured.snapshot.requests, &work).is_none());
    assert_no_live_effects(&inner, &captured, &counters);
    drop(parent);
    drop(context);
    drop(captured);
    let selected = vec![
        frontier(&session, &other).prefill_work(n32(1)).unwrap(),
        frontier(&session, &id).decode_work().unwrap(),
    ];
    // A genuine completion-only guarded wave establishes the physical pool.
    // No private fixture warm, invented maintenance or synthetic cost sample.
    let warm_mixed = wave(&mut session, selected).await;
    assert_eq!(
        warm_mixed.submission,
        CalibrationSubmissionState::HostReconciled
    );
    let CalibrationObservation::Observed {
        sample,
        actual_rows,
        ..
    } = &warm_mixed.observation
    else {
        panic!(
            "actual mixed preparation evidence: {:?}",
            warm_mixed.observation
        )
    };
    assert_eq!(
        sample.actual_shape.kind,
        ferrum_scheduler::implementations::continuous::cost_model::WaveKind::Mixed
    );
    assert_eq!(actual_rows.len(), 2);
    assert!(actual_rows
        .iter()
        .any(|r| matches!(r.work, ActualRowWork::Decode { .. })));
    assert!(actual_rows.iter().any(|r| r.work
        == ActualRowWork::Prefill {
            offset: 0,
            count: 1,
            total_prompt_tokens: 4
        }));
    ready(&inner, &id).await;
    ready(&inner, &other).await;

    let captured = capture(&inner);
    let counters = inner.model_executor.cache_metrics_snapshot().unwrap()["counters"].clone();
    let context = shape::ExecutorShape {
        engine: &inner,
        captured: &captured,
    };
    let parent = context.begin(&captured.snapshot, &mut || Ok(())).unwrap();
    let work: Vec<_> = captured
        .snapshot
        .requests
        .iter()
        .rev()
        .map(|r| CandidateWork {
            key: r.key.clone(),
            action: if r.key.request_id == id {
                WaveAction::Decode
            } else {
                WaveAction::Prefill {
                    offset: 1,
                    count: n32(1),
                }
            },
        })
        .collect();
    let edge = project(parent.as_ref(), &captured.snapshot.requests, &work)
        .expect("recaptured resident mixed native route");
    let expected = edge.canonical_domain.exact().unwrap().clone();
    assert_eq!(expected.kind, ActualWaveKind::Mixed);
    assert_no_live_effects(&inner, &captured, &counters);
    drop(edge);
    drop(parent);
    drop(context);
    drop(captured);
    let selected = vec![
        frontier(&session, &other).prefill_work(n32(1)).unwrap(),
        frontier(&session, &id).decode_work().unwrap(),
    ];
    assert_actual(&expected, &wave(&mut session, selected).await);
    session.shutdown().await.unwrap();
    output.await.unwrap();
    other_output.await.unwrap();
    drop(inner);
    drop(directory);
}
