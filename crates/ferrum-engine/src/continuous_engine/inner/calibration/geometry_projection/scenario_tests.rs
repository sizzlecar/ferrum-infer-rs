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
        // One active slot leaves no cache; two leave one cache entry. Alternating
        // widths evicts that entry before its next use, so original replay wins.
        assert_eq!(bounded.projection_attempts, replay.projection_attempts);
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
