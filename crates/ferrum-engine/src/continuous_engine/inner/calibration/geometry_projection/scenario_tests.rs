use super::tests::{assert_unsubmitted_clean, fixture, limits, requests};
use super::*;
use ferrum_interfaces::Tokenizer;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::prefixes::StructuredPrefixSlotV5;
use ferrum_tokenizer::implementations::HuggingFaceTokenizer;
use ferrum_types::TokenId;

async fn fragment_tokenizer() -> HuggingFaceTokenizer {
    let vocab = (0..64)
        .map(|i| {
            let surface = match i {
                0 => "<unk>".to_owned(),
                10 => "a".to_owned(),
                11 => "Ã".to_owned(),
                12 => "©".to_owned(),
                _ => format!("v{i}"),
            };
            (surface, i)
        })
        .collect();
    let mut tokenizer = tokenizers::Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".into())
            .build()
            .unwrap(),
    );
    tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    HuggingFaceTokenizer::from_source_bytes(
        tokenizer.to_string(false).unwrap().as_bytes(),
        None,
        None,
    )
    .await
    .unwrap()
}

fn prefix(
    tokenizer: &dyn Tokenizer,
    request_id: RequestId,
    tokens: &[u32],
) -> GeometryPrefixConstraint {
    let token_ids: Vec<_> = tokens.iter().copied().map(TokenId::new).collect();
    let mut scratch = vec![0; tokenizer.bounded_token_bytes_bound().unwrap().get()];
    let token_bytes = token_ids
        .iter()
        .map(|&token| {
            let count = tokenizer
                .token_bytes_bounded_into(token, &mut scratch)
                .unwrap()
                .unwrap();
            scratch[..count].to_vec()
        })
        .collect();
    GeometryPrefixConstraint {
        request_id,
        release_generated: tokens.len() as u32,
        slot: StructuredPrefixSlotV5 {
            tokenizer_policy_sha256: tokenizer.host_output_policy_identity().unwrap(),
            token_ids,
            token_bytes,
        },
    }
}

fn layouts(
    session: &CalibrationSession,
    probes: &[ProbeRequest],
) -> [[GeometryPrefixConstraint; 2]; 2] {
    let tokenizer = session.engine.inner.tokenizer.as_ref();
    [
        [
            prefix(tokenizer, probes[0].request.id.clone(), &[10, 10]),
            prefix(tokenizer, probes[1].request.id.clone(), &[10, 11]),
        ],
        [
            prefix(tokenizer, probes[0].request.id.clone(), &[10, 11]),
            prefix(tokenizer, probes[1].request.id.clone(), &[10, 10]),
        ],
    ]
}

fn target() -> [GeometryInputTarget; 1] {
    [GeometryInputTarget::Decode(GeometryProjectionPoint {
        rows: 2,
        sequence_tokens: 3,
    })]
}

async fn shared_prefill_projection(
    session: &mut CalibrationSession,
    reuse: PrefillReuse,
    budget: GeometryProjectionLimits,
) -> GeometryInputReport {
    let probes = requests(session, 2);
    let layouts = layouts(session, &probes);
    let targets = [
        GeometryInputTarget::Decode(GeometryProjectionPoint {
            rows: 2,
            sequence_tokens: 3,
        }),
        GeometryInputTarget::Decode(GeometryProjectionPoint {
            rows: 1,
            sequence_tokens: 3,
        }),
    ];
    let scenarios = layouts.each_ref().map(|prefixes| GeometryInputScenario {
        targets: &targets,
        prefixes,
    });
    session
        .project_geometry_inputs_inner(probes, &scenarios, budget, 0, reuse)
        .await
        .unwrap()
}

#[tokio::test]
async fn geometry_prefill_shares_successor_and_keeps_each_prefix_decode_independent() {
    let (mut session, executor) = fixture(2).await;
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    executor
        .single_row_prefill_only
        .store(true, Ordering::Release);
    Arc::get_mut(&mut session.engine.inner).unwrap().tokenizer =
        Arc::new(fragment_tokenizer().await);
    let replay = shared_prefill_projection(&mut session, PrefillReuse::Replay, limits()).await;
    let shared = shared_prefill_projection(&mut session, PrefillReuse::Share, limits()).await;
    assert_eq!(replay.outcomes.len(), 4);
    assert_eq!(shared.admitted_requests, replay.admitted_requests);
    for (expected, actual) in replay.outcomes.iter().zip(&shared.outcomes) {
        assert_eq!(expected.unknown, None);
        super::trajectory_tests::assert_equivalent(expected, actual);
    }
    // The two widths require respectively two and one real prefill projections.
    // The second prefix layout reuses only those three projections, preserving
    // both layouts' independently projected decode histories and pending rows.
    assert_eq!(replay.projection_attempts - shared.projection_attempts, 3);
    let pending = |outcome: &GeometryInputOutcome| {
        outcome.branches[0]
            .query
            .input()
            .physical_host_rows()
            .iter()
            .map(|row| row.pending_decoded_utf8)
            .collect::<Vec<_>>()
    };
    assert_ne!(pending(&shared.outcomes[0]), pending(&shared.outcomes[2]));
    // A new owner group/captured view starts empty even when inputs are equal.
    let fresh = shared_prefill_projection(&mut session, PrefillReuse::Share, limits()).await;
    assert_eq!(fresh.projection_attempts, shared.projection_attempts);
    for (expected, actual) in shared.outcomes.iter().zip(&fresh.outcomes) {
        super::trajectory_tests::assert_equivalent(expected, actual);
    }
    assert_unsubmitted_clean(&session, &executor);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn geometry_prefill_evicts_within_original_state_slots_without_losing_branches() {
    let (mut session, executor) = fixture(2).await;
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    executor
        .single_row_prefill_only
        .store(true, Ordering::Release);
    Arc::get_mut(&mut session.engine.inner).unwrap().tokenizer =
        Arc::new(fragment_tokenizer().await);
    let replay = shared_prefill_projection(&mut session, PrefillReuse::Replay, limits()).await;
    for maximum_route_states in [1, 2] {
        let mut budget = limits();
        budget.maximum_route_states = maximum_route_states;
        let bounded = shared_prefill_projection(&mut session, PrefillReuse::Share, budget).await;
        // Widths 2,1,2,1 need respectively 2,1,2,1 serial prefill queries,
        // followed by two decode queries apiece: replay costs 14. One active
        // slot leaves no cache. With two slots, the second target's width-one
        // prefix can seed the third target's width-two prefix: its one-token
        // prompt has identical segmentation, so only the new owner is prepared.
        // Width two cannot seed width one; the fourth target still replays.
        assert_eq!(replay.projection_attempts, 4 + 3 + 4 + 3);
        assert_eq!(
            bounded.projection_attempts,
            if maximum_route_states == 1 {
                4 + 3 + 4 + 3
            } else {
                4 + 3 + 3 + 3
            }
        );
        for (expected, actual) in replay.outcomes.iter().zip(&bounded.outcomes) {
            super::trajectory_tests::assert_equivalent(expected, actual);
        }
    }
    assert_unsubmitted_clean(&session, &executor);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn geometry_prefill_savings_are_real_work_under_the_same_projection_cap() {
    let (mut session, executor) = fixture(2).await;
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    Arc::get_mut(&mut session.engine.inner).unwrap().tokenizer =
        Arc::new(fragment_tokenizer().await);
    let complete = shared_prefill_projection(&mut session, PrefillReuse::Share, limits()).await;
    assert!(complete
        .outcomes
        .iter()
        .all(|outcome| outcome.unknown.is_none()));
    let mut exact = limits();
    exact.maximum_projections = complete.projection_attempts;
    let shared = shared_prefill_projection(&mut session, PrefillReuse::Share, exact).await;
    assert_eq!(shared.projection_attempts, exact.maximum_projections);
    assert!(shared
        .outcomes
        .iter()
        .all(|outcome| outcome.unknown.is_none()));
    let replay = shared_prefill_projection(&mut session, PrefillReuse::Replay, exact).await;
    assert_eq!(replay.projection_attempts, exact.maximum_projections);
    assert!(replay
        .outcomes
        .iter()
        .any(|outcome| outcome.unknown == Some(GeometryProjectionUnknown::BudgetExhausted)));
    exact.maximum_projections -= 1;
    let short = shared_prefill_projection(&mut session, PrefillReuse::Share, exact).await;
    assert_eq!(short.projection_attempts, exact.maximum_projections);
    assert!(short
        .outcomes
        .iter()
        .any(|outcome| outcome.unknown == Some(GeometryProjectionUnknown::BudgetExhausted)));
    assert_unsubmitted_clean(&session, &executor);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn geometry_prefill_reserves_cache_headers_before_any_owner_admission() {
    let (mut session, executor) = fixture(2).await;
    Arc::get_mut(&mut session.engine.inner).unwrap().tokenizer =
        Arc::new(fragment_tokenizer().await);
    let probes = requests(&session, 2);
    let layouts = layouts(&session, &probes);
    let targets = target();
    let scenarios = layouts.each_ref().map(|prefixes| GeometryInputScenario {
        targets: &targets,
        prefixes,
    });
    let mut budget = limits();
    budget.maximum_retained_bytes = std::mem::size_of::<GeometryInputReport>()
        + 2 * std::mem::size_of::<GeometryInputOutcome>()
        + PrefillStates::overhead_bytes(2, budget.maximum_route_states, PrefillReuse::Share)
            .unwrap()
        - 1;
    let result = session
        .project_geometry_input_scenarios(probes, &scenarios, budget)
        .await;
    assert!(result.is_err());
    assert_unsubmitted_clean(&session, &executor);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn geometry_trajectory_keeps_pending_prefix_and_later_unresolved_host_branches() {
    let (mut session, executor) = fixture(2).await;
    Arc::get_mut(&mut session.engine.inner).unwrap().tokenizer =
        Arc::new(fragment_tokenizer().await);
    let probes = requests(&session, 2);
    let layouts = layouts(&session, &probes);
    let targets = [
        GeometryInputTarget::Decode(GeometryProjectionPoint {
            rows: 2,
            sequence_tokens: 3,
        }),
        GeometryInputTarget::Decode(GeometryProjectionPoint {
            rows: 2,
            sequence_tokens: 4,
        }),
    ];
    let mut scenarios = vec![GeometryInputScenario {
        targets: &targets,
        prefixes: &layouts[0],
    }];
    scenarios.extend(targets.iter().map(|target| GeometryInputScenario {
        targets: std::slice::from_ref(target),
        prefixes: &layouts[0],
    }));
    let report = session
        .project_geometry_input_scenarios(probes, &scenarios, limits())
        .await
        .unwrap();
    assert_eq!(report.outcomes.len(), targets.len() * 2);
    for index in 0..targets.len() {
        let independent = &report.outcomes[targets.len() + index];
        assert_eq!(independent.unknown, None);
        super::trajectory_tests::assert_equivalent(&report.outcomes[index], independent);
    }
    assert!(report.outcomes[0].branches[0]
        .query
        .input()
        .physical_host_rows()
        .iter()
        .any(|row| row.pending_decoded_utf8));
    assert_unsubmitted_clean(&session, &executor);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn geometry_scenarios_share_real_owners_without_sharing_prefix_results() {
    let (mut session, executor) = fixture(2).await;
    Arc::get_mut(&mut session.engine.inner).unwrap().tokenizer =
        Arc::new(fragment_tokenizer().await);
    let probes = requests(&session, 2);
    let layouts = layouts(&session, &probes);
    let targets = target();
    let scenarios = layouts.each_ref().map(|prefixes| GeometryInputScenario {
        targets: &targets,
        prefixes,
    });
    let report = session
        .project_geometry_input_scenarios(probes, &scenarios, limits())
        .await
        .unwrap();
    assert_eq!(report.admitted_requests, 2);
    assert_eq!(report.outcomes.len(), 2);
    let mut pending = Vec::new();
    for (index, outcome) in report.outcomes.iter().enumerate() {
        assert_eq!(outcome.scenario_index, index);
        assert_eq!(outcome.unknown, None);
        assert_eq!(outcome.branches.len(), 1);
        assert_eq!(
            outcome.branches[0].host_branch,
            Some(GeometryHostBranch::FullLogits)
        );
        assert_eq!(
            outcome.prefix_condition,
            Some(GeometryPrefixCondition {
                release_generated: 2,
                first_ordinary_wave: true,
            })
        );
        let rows: Vec<_> = outcome.branches[0]
            .query
            .input()
            .physical_host_rows()
            .iter()
            .map(|row| row.pending_decoded_utf8)
            .collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows.iter().filter(|pending| **pending).count(), 1);
        pending.push(rows);
    }
    // Physical row order may differ from original admission order, but is the
    // same for both scenarios sharing this route view. Swapping the real slot
    // trajectories must swap every row's pending fact, not reuse a query.
    assert!(pending[0]
        .iter()
        .zip(&pending[1])
        .all(|(first, second)| first != second));
    assert_unsubmitted_clean(&session, &executor);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn geometry_scenarios_consume_one_projection_allowance_and_clean_all_roots() {
    let (mut session, executor) = fixture(2).await;
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    Arc::get_mut(&mut session.engine.inner).unwrap().tokenizer =
        Arc::new(fragment_tokenizer().await);
    let targets = target();
    let probes = requests(&session, 2);
    let first_layouts = layouts(&session, &probes);
    let first = session
        .project_geometry_inputs(probes, &targets, limits(), &first_layouts[0])
        .await
        .unwrap();
    assert_eq!(first.outcomes[0].unknown, None);
    let first_attempts = first.projection_attempts;
    assert!(first_attempts > 0);
    drop(first);
    assert_unsubmitted_clean(&session, &executor);

    let probes = requests(&session, 2);
    let layouts = layouts(&session, &probes);
    let scenarios = layouts.each_ref().map(|prefixes| GeometryInputScenario {
        targets: &targets,
        prefixes,
    });
    let mut budget = limits();
    budget.maximum_projections = first_attempts;
    let report = session
        .project_geometry_input_scenarios(probes, &scenarios, budget)
        .await
        .unwrap();
    assert_eq!(report.admitted_requests, 2);
    assert_eq!(report.projection_attempts, first_attempts);
    assert_eq!(report.outcomes.len(), 2);
    assert_eq!(report.outcomes[0].scenario_index, 0);
    assert_eq!(report.outcomes[0].unknown, None);
    assert_eq!(report.outcomes[1].scenario_index, 1);
    assert_eq!(
        report.outcomes[1].unknown,
        Some(GeometryProjectionUnknown::BudgetExhausted)
    );
    assert!(report.outcomes[1].branches.is_empty());
    assert_unsubmitted_clean(&session, &executor);
    session.shutdown().await.unwrap();
}

// Preserve the inventory order: original first-prefill/decode scenarios,
// continuation-prefill scenarios, then ordinary trajectory scenarios. The
// captured owner group remains the same throughout this complete traversal.
async fn complete_prefill_geometry(
    session: &mut CalibrationSession,
    reuse: PrefillReuse,
    budget: GeometryProjectionLimits,
) -> GeometryInputReport {
    let mut probes = requests(session, 4);
    for probe in &mut probes {
        probe.request.prompt = "test ok v7".into();
        probe.request.sampling_params.max_tokens = 2;
    }
    let decode = |rows| {
        GeometryInputTarget::Decode(GeometryProjectionPoint {
            rows,
            sequence_tokens: 4,
        })
    };
    let mut targets = Vec::new();
    for rows in 1..=4 {
        targets.push(GeometryInputTarget::InitialPrefill { rows });
        targets.push(decode(rows));
    }
    // With a whole-wave allowance of eight, widths three and four use
    // two-token row chunks. Their three-token prompts have one continuation.
    for rows in [3, 4] {
        targets.push(GeometryInputTarget::PrefillSpan { rows, offset: 2 });
    }
    targets.extend((1..=4).map(decode));
    let scenarios: Vec<_> = targets
        .iter()
        .map(|target| GeometryInputScenario {
            targets: std::slice::from_ref(target),
            prefixes: &[],
        })
        .collect();
    session
        .project_geometry_inputs_inner(probes, &scenarios, budget, 0, reuse)
        .await
        .unwrap()
}

#[tokio::test]
async fn geometry_complete_prefill_inventory_reuses_only_identical_checked_paths() {
    let (mut session, executor) = super::tests::fixture_with_domain(4, 32, 8).await;
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    let mut budget = limits();
    budget.prefill_chunk = NonZeroU32::new(8).unwrap();
    budget.maximum_route_states = 16;
    let replay = complete_prefill_geometry(&mut session, PrefillReuse::Replay, budget).await;
    let shared = complete_prefill_geometry(&mut session, PrefillReuse::Share, budget).await;
    assert_eq!(shared.admitted_requests, replay.admitted_requests);
    assert_eq!(shared.outcomes.len(), replay.outcomes.len());
    for (expected, actual) in replay.outcomes.iter().zip(&shared.outcomes) {
        assert_eq!(expected.unknown, None);
        super::trajectory_tests::assert_equivalent(expected, actual);
    }
    // Four independent initial joint queries. The first width-one successor
    // supplies its sequential prefix; width two adds one original owner.
    // Width three changes segmentation and must replay all six row chunks.
    // Width four has the same two-token segments and adds two chunks. The
    // two joint continuation targets each advance their own joint predecessor.
    // All eight original decode queries still run with unchanged host evidence.
    let required = 4 + (0 + 1 + 6 + 2) + 2 + 8;
    assert_eq!(shared.projection_attempts, required);
    assert!(replay.projection_attempts > shared.projection_attempts);

    budget.maximum_projections = required;
    let exact = complete_prefill_geometry(&mut session, PrefillReuse::Share, budget).await;
    assert_eq!(exact.projection_attempts, required);
    for (expected, actual) in shared.outcomes.iter().zip(&exact.outcomes) {
        super::trajectory_tests::assert_equivalent(expected, actual);
    }
    budget.maximum_projections -= 1;
    let short = complete_prefill_geometry(&mut session, PrefillReuse::Share, budget).await;
    assert_eq!(short.projection_attempts, budget.maximum_projections);
    assert_eq!(
        short.outcomes.last().unwrap().unknown,
        Some(GeometryProjectionUnknown::BudgetExhausted)
    );
    budget = limits();
    budget.prefill_chunk = NonZeroU32::new(8).unwrap();
    budget.maximum_route_states = 1;
    let uncached = complete_prefill_geometry(&mut session, PrefillReuse::Share, budget).await;
    assert_eq!(uncached.projection_attempts, replay.projection_attempts);
    for (expected, actual) in replay.outcomes.iter().zip(&uncached.outcomes) {
        super::trajectory_tests::assert_equivalent(expected, actual);
    }
    assert_unsubmitted_clean(&session, &executor);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn geometry_prefill_failed_target_never_installs_its_partial_joint_successor() {
    let (mut session, executor) = super::tests::fixture_with_domain(1, 32, 4).await;
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    let mut reports = Vec::new();
    for reuse in [PrefillReuse::Replay, PrefillReuse::Share] {
        let calls = std::sync::atomic::AtomicUsize::new(0);
        *executor.projection_readiness_fault.lock() = Some(Box::new(move |_, _| {
            // The first ancestor succeeds, but the complete target does not.
            (calls.fetch_add(1, Ordering::AcqRel) == 1)
                .then_some(ExecutionCostRouteUnknown::Unsupported)
        }));
        let mut probes = requests(&session, 1);
        probes[0].request.prompt = "test ok v7".into();
        probes[0].request.sampling_params.max_tokens = 2;
        let targets = [
            GeometryInputTarget::PrefillSpan { rows: 1, offset: 2 },
            GeometryInputTarget::PrefillSpan { rows: 1, offset: 2 },
            GeometryInputTarget::Decode(GeometryProjectionPoint {
                rows: 1,
                sequence_tokens: 4,
            }),
        ];
        let scenarios: Vec<_> = targets
            .iter()
            .map(|target| GeometryInputScenario {
                targets: std::slice::from_ref(target),
                prefixes: &[],
            })
            .collect();
        let mut budget = limits();
        budget.prefill_chunk = NonZeroU32::new(1).unwrap();
        reports.push(
            session
                .project_geometry_inputs_inner(probes, &scenarios, budget, 0, reuse)
                .await
                .unwrap(),
        );
        *executor.projection_readiness_fault.lock() = None;
        assert_unsubmitted_clean(&session, &executor);
    }
    let [replay, shared]: [_; 2] = reports.try_into().ok().unwrap();
    for (expected, actual) in replay.outcomes.iter().zip(&shared.outcomes) {
        super::trajectory_tests::assert_equivalent(expected, actual);
    }
    assert_eq!(
        shared.outcomes[0].unknown,
        Some(GeometryProjectionUnknown::Route(
            ExecutionCostRouteUnknown::Unsupported
        ))
    );
    assert_eq!(shared.outcomes[1].unknown, None);
    assert_eq!(shared.outcomes[2].unknown, None);
    // A failed target leaves neither of its ancestors installed: the second
    // target still needs all three prefill waves. Only its complete, checked
    // width-one successor may eliminate the final decode's serial preparation.
    assert_eq!(shared.projection_attempts, 2 + 3 + 1);
    assert_eq!(replay.projection_attempts, 2 + 3 + 3 + 1);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn geometry_prefill_partial_width_one_joint_state_is_not_a_completed_serial_prefix() {
    let (mut session, executor) = super::tests::fixture_with_domain(1, 32, 4).await;
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    let mut reports = Vec::new();
    for reuse in [PrefillReuse::Replay, PrefillReuse::Share] {
        let mut probes = requests(&session, 1);
        probes[0].request.prompt = "test ok v7".into();
        probes[0].request.sampling_params.max_tokens = 2;
        let targets = [
            GeometryInputTarget::InitialPrefill { rows: 1 },
            GeometryInputTarget::Decode(GeometryProjectionPoint {
                rows: 1,
                sequence_tokens: 4,
            }),
        ];
        let scenarios: Vec<_> = targets
            .iter()
            .map(|target| GeometryInputScenario {
                targets: std::slice::from_ref(target),
                prefixes: &[],
            })
            .collect();
        let mut budget = limits();
        budget.prefill_chunk = NonZeroU32::new(1).unwrap();
        reports.push(
            session
                .project_geometry_inputs_inner(probes, &scenarios, budget, 0, reuse)
                .await
                .unwrap(),
        );
        assert_unsubmitted_clean(&session, &executor);
    }
    let [replay, shared]: [_; 2] = reports.try_into().ok().unwrap();
    assert_eq!(shared.projection_attempts, 1 + 3 + 1);
    assert_eq!(shared.projection_attempts, replay.projection_attempts);
    for (expected, actual) in replay.outcomes.iter().zip(&shared.outcomes) {
        assert_eq!(expected.unknown, None);
        super::trajectory_tests::assert_equivalent(expected, actual);
    }
    session.shutdown().await.unwrap();
}
