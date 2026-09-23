use super::*;
use ferrum_interfaces::{execution_cost::*, vnext::DeviceCommandPhase};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

/// Deliberately synthetic complete route. It is never a production fallback.
pub(super) struct TestResolver;
impl PlanningShapeResolver for TestResolver {
    fn resolve(
        &self,
        query: &PlanningShapeQuery<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<CanonicalWaveCostShape>, PlanningUnknownReason> {
        let mut builder = CanonicalWaveCostBuilder::new(0, CostProductOutput::FullLogits);
        builder
            .physical_command(CostPhysicalCommand {
                native_op_id: "fixture.wave",
                command_index: 0,
                node_index: None,
                command_phase: DeviceCommandPhase::Compute,
                provider: None,
                path: CostCommandPath::Eager,
                participant_start: 0,
                participant_count: query.rows.len() as u32,
                token_count: query
                    .rows
                    .iter()
                    .map(|row| match row.work {
                        ActualRowWork::Decode { .. } => 1,
                        ActualRowWork::Prefill { count, .. } => u64::from(count),
                        _ => 0,
                    })
                    .sum(),
                batching_form: "fixture",
                compute_dispatch_count: 1,
                transfer_command_count: 0,
                reusable_graph_node_count: None,
            })
            .map_err(|_| PlanningUnknownReason::InvalidShapeEvidence)?;
        for row in query.rows {
            poll()?;
            let output = match row.work {
                ActualRowWork::Decode { .. } => CostRowOutput::Decode {
                    requires_full_logits: true,
                    repetition_tokens: 0,
                    repetition_penalty_bits: 1.0f32.to_bits(),
                },
                ActualRowWork::Prefill {
                    offset,
                    count,
                    total_prompt_tokens,
                } => CostRowOutput::Prefill {
                    final_logits: offset.checked_add(count) == Some(total_prompt_tokens),
                },
                _ => return Err(PlanningUnknownReason::InvalidShapeEvidence),
            };
            builder
                .row(CanonicalCostRow {
                    host_features: None,
                    work: row.work,
                    host_policy_signature: host_history_cost_signature(
                        row.request.output_policy_signature,
                        u64::from(row.request.timing.committed_tokens),
                    ),
                    mask_upload_required: false,
                    output,
                })
                .map_err(|_| PlanningUnknownReason::InvalidShapeEvidence)?;
        }
        let caps = &query.snapshot.capabilities;
        let path = match caps.path {
            WaveExecutionPath::PlanRuntime => ActualWavePath::PlanRuntime,
            WaveExecutionPath::NativeUnified => ActualWavePath::NativeUnified,
            WaveExecutionPath::LegacySplit => ActualWavePath::LegacySplit,
            WaveExecutionPath::UnsupportedFallback => ActualWavePath::UnsupportedFallback,
            WaveExecutionPath::CapacityFallback => ActualWavePath::CapacityFallback,
        };
        let graph = match caps.graph_state {
            WaveGraphState::Disabled => ActualWaveGraphState::Disabled,
            WaveGraphState::Cold => ActualWaveGraphState::Cold,
            WaveGraphState::Warm => ActualWaveGraphState::Warm,
        };
        let order = match caps.order {
            BatchOrderSemantics::Ordered => ActualWaveRowOrder::Ordered,
            BatchOrderSemantics::IndependentRows => ActualWaveRowOrder::IndependentRows,
        };
        builder
            .finish(query.kind, path, graph, order, query.recurrent_state_bytes)
            .map(Some)
            .map_err(|_| PlanningUnknownReason::InvalidShapeEvidence)
    }
}

struct UnknownResolver;
impl PlanningShapeResolver for UnknownResolver {
    fn resolve(
        &self,
        _: &PlanningShapeQuery<'_>,
        _: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<CanonicalWaveCostShape>, PlanningUnknownReason> {
        Ok(None)
    }
}

#[test]
fn absent_route_evidence_cannot_use_capacity_constants_or_call_the_cost_model() {
    let snapshot = snapshot(vec![decode(1)]);
    let model = Model(|_: &WaveExecutionShape| -> Option<u64> {
        panic!("unknown route must not query cost")
    });
    let result = planner(1).propose(&snapshot, &model, &UnknownResolver, &mut Clock(100));
    assert!(
        matches!(result,PlanningDecision::Unknown {reason:PlanningUnknownReason::ShapeUnavailable,search} if search.shape_unknown_candidates>0)
    );
}

#[test]
fn heterogeneous_row_policies_are_resolved_independently() {
    let mut second = decode(2);
    second.output_policy_signature = [4; 32];
    let mut snapshot = snapshot(vec![decode(1), second]);
    snapshot.capabilities.decode_batch_sizes = vec![nz(2)];
    let first = feasible(planner(1).propose(
        &snapshot,
        &Model(|_: &WaveExecutionShape| Some(5)),
        &TestResolver,
        &mut Clock(100),
    ))
    .0;
    assert_eq!(first.candidate.work.len(), 2);
    snapshot.requests[1].output_policy_signature = [6; 32];
    let second = feasible(planner(1).propose(
        &snapshot,
        &Model(|_: &WaveExecutionShape| Some(5)),
        &TestResolver,
        &mut Clock(100),
    ))
    .0;
    assert_ne!(
        first
            .candidate
            .execution_shape
            .exact()
            .unwrap()
            .output_policy_signature,
        second
            .candidate
            .execution_shape
            .exact()
            .unwrap()
            .output_policy_signature
    );
}

struct RecordingResolver(RefCell<Vec<(u32, u32, [u8; 32])>>);
impl PlanningShapeResolver for RecordingResolver {
    fn resolve(
        &self,
        query: &PlanningShapeQuery<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<CanonicalWaveCostShape>, PlanningUnknownReason> {
        let result = TestResolver.resolve(query, poll)?;
        for row in query.rows {
            self.0.borrow_mut().push((
                row.request.context_tokens,
                row.request.timing.committed_tokens,
                result.as_ref().unwrap().output_policy_signature,
            ));
        }
        Ok(result)
    }
}
#[test]
fn rollout_reresolves_numeric_context_and_history_instead_of_reusing_the_first_key() {
    let mut request = decode(1);
    request.timing.maximum_output_tokens = n32(3);
    request.timing.budgets.itl_ns = n64(10);
    request.timing.budgets.tpot_ns = n64(10);
    let mut snapshot = snapshot(vec![request]);
    snapshot.observed_at_ns = 95;
    snapshot.scope.horizon_end_ns = 115;
    let resolver = RecordingResolver(RefCell::new(Vec::new()));
    let (_, witness, _) = feasible(planner(2).propose(
        &snapshot,
        &Model(|_: &WaveExecutionShape| Some(5)),
        &resolver,
        &mut Clock(95),
    ));
    assert_eq!(witness.predicted_output_tokens, 2);
    let seen = resolver.0.borrow();
    let first = seen
        .iter()
        .find(|&&(context, n, _)| context == 10 && n == 1)
        .unwrap();
    let second = seen
        .iter()
        .find(|&&(context, n, _)| context == 11 && n == 2)
        .unwrap();
    assert_ne!(first.2, second.2);
}

struct BadResolver;
impl PlanningShapeResolver for BadResolver {
    fn resolve(
        &self,
        query: &PlanningShapeQuery<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<CanonicalWaveCostShape>, PlanningUnknownReason> {
        let mut result = TestResolver.resolve(query, poll)?.unwrap();
        result.rows[0] = ActualRowWork::Decode { kv_tokens: 999 };
        Ok(Some(result))
    }
}
#[test]
fn resolver_cannot_rewrite_validated_logical_progress() {
    assert!(matches!(
        planner(1).propose(
            &snapshot(vec![decode(1)]),
            &Model(|_: &WaveExecutionShape| Some(1)),
            &BadResolver,
            &mut Clock(100)
        ),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::InvalidShapeEvidence,
            ..
        }
    ));
}

struct SlowResolver(Rc<Cell<u64>>);
impl PlanningShapeResolver for SlowResolver {
    fn resolve(
        &self,
        query: &PlanningShapeQuery<'_>,
        _: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<CanonicalWaveCostShape>, PlanningUnknownReason> {
        let result = TestResolver.resolve(query, &mut || Ok(()));
        self.0.set(10_000_000);
        result
    }
}
#[test]
fn caller_checks_wall_budget_even_when_the_resolver_never_polls() {
    struct SharedClock(Rc<Cell<u64>>);
    impl PlanningClock for SharedClock {
        fn now_ns(&mut self) -> u64 {
            self.0.get()
        }
    }
    let now = Rc::new(Cell::new(100));
    assert!(matches!(
        planner(1).propose(
            &snapshot(vec![decode(1)]),
            &Model(|_: &WaveExecutionShape| Some(1)),
            &SlowResolver(now.clone()),
            &mut SharedClock(now)
        ),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::ComputeBudgetExhausted,
            ..
        }
    ));
}

fn recomputing() -> RequestSchedulingView {
    let mut request = prefill(1);
    request.timing.first_commit_at_ns = Some(90);
    request.timing.last_commit_at_ns = Some(90);
    request.timing.committed_tokens = 1;
    request.timing.maximum_output_tokens = n32(3);
    request.timing.budgets.itl_ns = n64(20);
    request.timing.budgets.tpot_ns = n64(20);
    request.context_tokens = 4;
    if let RequestPhaseView::Prefill(progress) = &mut request.phase {
        progress.offset = 4;
        progress.logical_high_water = 8;
    }
    request
}
#[test]
fn final_recompute_commits_a_token_without_resetting_first_commit_or_useful_work() {
    let mut snapshot = snapshot(vec![recomputing()]);
    snapshot.scope.horizon_end_ns = 120;
    let model = Model(|_: &WaveExecutionShape| Some(5));
    let (selected, witness, _) =
        feasible(planner(1).propose(&snapshot, &model, &TestResolver, &mut Clock(100)));
    assert_eq!(witness.predicted_output_tokens, 1);
    assert_eq!(witness.net_prefill_reference_work_ns, 0);
    let state = simulation::simulate(
        &snapshot,
        &[selected.candidate],
        &model,
        &TestResolver,
        None,
        &mut || Ok(()),
        100,
        true,
        None,
    )
    .unwrap();
    let request = &state.requests[0];
    assert_eq!(request.timing.first_commit_at_ns, Some(90));
    assert_eq!(request.timing.last_commit_at_ns, Some(105));
    assert_eq!(request.timing.committed_tokens, 2);
    assert_eq!(request.timing.next_deadline_ns(), Some(125));
    assert_eq!(request.output_credit.available_token_commands, 15);
    assert_eq!(
        request.output_credit.byte_backing,
        OutputByteBacking::Incremental {
            available_bytes: 992,
            bytes_per_token_upper_bound: Some(n64(32)),
        }
    );
}
#[test]
fn recompute_output_credit_and_following_decode_obligation_share_one_budget() {
    let mut snapshot = snapshot(vec![recomputing()]);
    snapshot.scope.horizon_end_ns = 130;
    let model = Model(|_: &WaveExecutionShape| Some(5));
    let (_, witness, _) =
        feasible(planner(2).propose(&snapshot, &model, &TestResolver, &mut Clock(100)));
    assert_eq!(witness.predicted_output_tokens, 2);
    assert_eq!(witness.waves, 2);
    snapshot.requests[0].output_credit.available_token_commands = 1;
    assert!(matches!(
        planner(2).propose(&snapshot, &model, &TestResolver, &mut Clock(100)),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::OutputOrResourceBlocked,
            ..
        }
    ));
    snapshot.requests[0].output_credit.available_token_commands = 0;
    assert!(matches!(
        planner(1).propose(&snapshot, &model, &TestResolver, &mut Clock(100)),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::OutputOrResourceBlocked,
            ..
        }
    ));
}
#[test]
fn canonical_and_actual_conversion_preserve_the_identical_cost_key() {
    let snapshot = snapshot(vec![decode(1), prefill(2)]);
    let rows = [
        PlanningShapeRow {
            request: &snapshot.requests[0],
            work: ActualRowWork::Decode { kv_tokens: 10 },
        },
        PlanningShapeRow {
            request: &snapshot.requests[1],
            work: ActualRowWork::Prefill {
                offset: 0,
                count: 4,
                total_prompt_tokens: 8,
            },
        },
    ];
    let canonical = TestResolver
        .resolve(
            &PlanningShapeQuery {
                snapshot: &snapshot,
                prior_waves: &[],
                kind: ActualWaveKind::Mixed,
                rows: &rows,
                recurrent_state_bytes: 0,
            },
            &mut || Ok(()),
        )
        .unwrap()
        .unwrap();
    let actual = ActualWaveShape {
        numeric_features: canonical.numeric_features.clone(),
        row_multiset_features: canonical.row_multiset_features.clone(),
        host_content_features: canonical.host_content_features,
        kind: canonical.kind,
        path: canonical.path,
        graph: canonical.graph,
        row_order: canonical.row_order,
        provider_signature: canonical.provider_signature,
        output_policy_signature: canonical.output_policy_signature,
        rows: rows
            .iter()
            .enumerate()
            .map(|(index, row)| ActualWaveRow {
                request_id: row.request.key.request_id.clone(),
                owner_incarnation: row.request.key.incarnation,
                work_generation: 0,
                input_index: index as u32,
                work: row.work,
            })
            .collect(),
        recurrent_state_bytes: 0,
        restore_bytes: 0,
        maintenance_bytes: 0,
        maintenance_units: 0,
    };
    let projected = canonical_cost_shape(&canonical).unwrap();
    assert_eq!(projected, actual_cost_shape(&actual).unwrap());
    assert_eq!(projected.provider_signature, canonical.provider_signature);
    assert_eq!(
        projected.output_policy_signature,
        canonical.output_policy_signature
    );
}

#[test]
fn enforced_resolution_session_limit_stops_invoking_the_callback() {
    struct Counted(Cell<usize>);
    impl PlanningShapeResolver for Counted {
        fn resolve(
            &self,
            _: &PlanningShapeQuery<'_>,
            _: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
        ) -> Result<Option<CanonicalWaveCostShape>, PlanningUnknownReason> {
            self.0.set(self.0.get() + 1);
            Ok(None)
        }
    }
    let mut settings = planner(1).settings;
    settings.search.candidate_limit = nz(1);
    settings.search.beam_width = nz(1);
    let resolver = Counted(Cell::new(0));
    let session = shape::ResolutionSession::new(&resolver, &settings);
    let snapshot = snapshot(vec![decode(1)]);
    let rows = [PlanningShapeRow {
        request: &snapshot.requests[0],
        work: ActualRowWork::Decode { kv_tokens: 10 },
    }];
    let query = PlanningShapeQuery {
        snapshot: &snapshot,
        prior_waves: &[],
        kind: ActualWaveKind::Decode,
        rows: &rows,
        recurrent_state_bytes: 0,
    };
    for _ in 0..18 {
        assert!(session.resolve(&query, &mut || Ok(())).unwrap().is_none());
    }
    assert_eq!(
        session.resolve(&query, &mut || Ok(())),
        Err(PlanningUnknownReason::SearchIncomplete)
    );
    assert_eq!(resolver.0.get(), 18);
}

#[test]
fn missing_future_route_keeps_the_new_decode_obligation_unknown() {
    struct FirstOnly;
    impl PlanningShapeResolver for FirstOnly {
        fn resolve(
            &self,
            query: &PlanningShapeQuery<'_>,
            poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
        ) -> Result<Option<CanonicalWaveCostShape>, PlanningUnknownReason> {
            if query
                .rows
                .iter()
                .any(|row| row.request.timing.committed_tokens > 1)
            {
                Ok(None)
            } else {
                TestResolver.resolve(query, poll)
            }
        }
    }
    let mut request = decode(1);
    request.timing.maximum_output_tokens = n32(3);
    request.timing.budgets.itl_ns = n64(10);
    request.timing.budgets.tpot_ns = n64(10);
    let mut snapshot = snapshot(vec![request]);
    snapshot.observed_at_ns = 95;
    snapshot.scope.horizon_end_ns = 115;
    assert!(matches!(
        planner(2).propose(
            &snapshot,
            &Model(|_: &WaveExecutionShape| Some(5)),
            &FirstOnly,
            &mut Clock(95)
        ),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::ShapeUnavailable,
            ..
        }
    ));
}

#[test]
fn oversized_resolver_output_is_rejected_before_cost_lookup() {
    struct Oversized;
    impl PlanningShapeResolver for Oversized {
        fn resolve(
            &self,
            query: &PlanningShapeQuery<'_>,
            poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
        ) -> Result<Option<CanonicalWaveCostShape>, PlanningUnknownReason> {
            let mut canonical = TestResolver.resolve(query, poll)?.unwrap();
            canonical.rows.resize(MAX_COST_ROWS + 1, canonical.rows[0]);
            Ok(Some(canonical))
        }
    }
    assert!(matches!(
        planner(1).propose(
            &snapshot(vec![decode(1)]),
            &Model(|_: &WaveExecutionShape| -> Option<u64> {
                panic!("oversized shape must not query cost")
            }),
            &Oversized,
            &mut Clock(100)
        ),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::InvalidShapeEvidence,
            ..
        }
    ));
}

#[test]
fn resolver_cannot_swallow_a_failed_budget_poll() {
    struct IgnorePoll;
    impl PlanningShapeResolver for IgnorePoll {
        fn resolve(
            &self,
            query: &PlanningShapeQuery<'_>,
            poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
        ) -> Result<Option<CanonicalWaveCostShape>, PlanningUnknownReason> {
            let _ = poll();
            TestResolver.resolve(query, &mut || Ok(()))
        }
    }
    let snapshot = snapshot(vec![decode(1)]);
    let work = [CandidateWork {
        key: snapshot.requests[0].key.clone(),
        action: WaveAction::Decode,
    }];
    let mut calls = 0;
    let result = shape::resolve(
        &snapshot,
        &snapshot.requests,
        &work,
        &IgnorePoll,
        &mut || {
            calls += 1;
            // Legal-row check and caller precheck succeed; the callback's own
            // poll fails, then the outer postcheck succeeds again.
            if calls == 3 {
                Err(PlanningUnknownReason::ClockMovedBackwards)
            } else {
                Ok(())
            }
        },
    );
    assert_eq!(result, Err(PlanningUnknownReason::ClockMovedBackwards));
}
