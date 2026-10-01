//! The real Metal model is constructed through the product startup hook. No
//! fixture warm() helper is called: every declared Step/Invocation slot must
//! already be resident before the first product request.
use super::*;
use ferrum_interfaces::model_executor::ExecutorResourcePlanningRequest;
use ferrum_interfaces::model_executor::{
    ExecutorResourcePreparationOutcome, ExecutorResourcePreparationReceipt,
    ExecutorResourcePreparationRequest,
};
use std::num::NonZeroUsize;

fn resource_request(
    rows: usize,
    frontier: usize,
    chunk: usize,
    kind: ActualWaveKind,
) -> ExecutorResourcePreparationRequest {
    ExecutorResourcePreparationRequest::new(
        NonZeroUsize::new(rows).unwrap(),
        NonZeroUsize::new(frontier).unwrap(),
        NonZeroUsize::new(chunk).unwrap(),
        kind,
    )
    .unwrap()
}

fn capture_resource_frontier(
    fixture: &Fixture,
    rows: usize,
    prompt: usize,
    frontier: usize,
    chunk: usize,
) -> ResourcePlanningView {
    let inputs = (0..rows)
        .map(|_| {
            PlanRuntimePrefillInput::new(
                RequestId::new(),
                vec![TokenId::new(1); prompt],
                frontier,
                PrefillChunk::new(0, chunk, prompt).unwrap(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    for input in &inputs {
        fixture.admit(input);
    }
    let requests = inputs
        .iter()
        .map(|input| ExecutorResourcePlanningRequest {
            request_id: &input.request_id,
            cache_id: None,
        })
        .collect::<Vec<_>>();
    let view = match fixture.executor.execution_resource_planning_view(
        &requests,
        ResourcePlanningLimits {
            maximum_projected_waves: prompt.div_ceil(chunk) + frontier - prompt,
            ..Default::default()
        },
        &mut || true,
    ) {
        ResourcePlanningAvailability::Known(view) => view,
        ResourcePlanningAvailability::Unknown(reason) => {
            panic!("resource-only capture: {reason:?}")
        }
    };
    for input in &inputs {
        assert!(fixture.executor.cancel_prefill_admission(&input.request_id));
    }
    view
}

fn project_resource_frontier(
    fixture: &Fixture,
    view: &ResourcePlanningView,
    width: usize,
    prompt: usize,
    frontier: usize,
    chunk: usize,
) -> std::result::Result<(), (ResourcePlanningUnknown, usize, usize, ActualWaveKind)> {
    let mut state = view.initial_state();
    let mut start = 0;
    while start < frontier {
        let (count, kind) = if start < prompt {
            (chunk.min(prompt - start), ActualWaveKind::Prefill)
        } else {
            (1, ActualWaveKind::Decode)
        };
        let rows = (0..width)
            .map(|participant_index| ResourcePlanningRow {
                participant_index,
                start_token: start as u64,
                token_count: count as u64,
            })
            .collect::<Vec<_>>();
        match fixture.executor.project_execution_resource_wave_for_kind(
            view,
            &state,
            &rows,
            kind,
            &mut || true,
        ) {
            ResourcePlanningAvailability::Known(projected) => state = projected.state,
            ResourcePlanningAvailability::Unknown(reason) => {
                return Err((reason, start, count, kind))
            }
        }
        start += count;
    }
    Ok(())
}

#[tokio::test]
async fn workspace_resource_readiness_materializes_declared_frontiers_without_model_work() {
    let fixture = Fixture::grouped().await;
    for (rows, prompt, output, chunk) in [(1, 257, 2, 16), (3, 129, 3, 21), (8, 509, 3, 8)] {
        let frontier = prompt + output - 1;
        let cold = capture_resource_frontier(&fixture, rows, prompt, frontier, chunk);
        let prior = project_resource_frontier(&fixture, &cold, rows, prompt, frontier, chunk);
        if rows == 1 {
            assert!(
                prior.is_err(),
                "cold long-context backing must not be fabricated"
            );
        }
        let mut shapes = vec![(ActualWaveKind::Prefill, chunk)];
        if prompt % chunk != 0 {
            shapes.push((ActualWaveKind::Prefill, prompt % chunk));
        }
        shapes.push((ActualWaveKind::Decode, 1));
        // Preserve the product declaration order: eager multirow prefill
        // precedes the retained decode slot. The implementation must use its
        // real bucket selector to establish retention dependencies.
        let requests = shapes
            .into_iter()
            .map(|(kind, work)| resource_request(rows, frontier, work, kind))
            .collect::<Vec<_>>();
        assert_eq!(
            fixture
                .executor
                .prepare_execution_resources(&requests, &mut || true)
                .unwrap(),
            ExecutorResourcePreparationReceipt {
                outcome: ExecutorResourcePreparationOutcome::Prepared,
                prepared_participants: rows * requests.len(),
            }
        );
        assert_eq!(fixture.executor.sequences.lock().total_len(), 0);
        assert_eq!(fixture.submissions(), 0);
        // The old numerical capture cannot inherit newly resident backing.
        assert_eq!(
            project_resource_frontier(&fixture, &cold, rows, prompt, frontier, chunk),
            prior
        );
        let fresh = capture_resource_frontier(&fixture, rows, prompt, frontier, chunk);
        assert_eq!(
            project_resource_frontier(&fixture, &fresh, rows, prompt, frontier, chunk),
            Ok(()),
            "width={rows} prompt={prompt} output={output} chunk={chunk}"
        );
        let status = fixture
            .executor
            .plan_resources
            .dynamic_pool_status()
            .unwrap();
        assert!(status.process_claimed_bytes() <= status.effective_device_usable_ceiling_bytes());
        assert_eq!(fixture.submissions(), 0);
    }
}

#[tokio::test]
async fn workspace_resource_readiness_capacity_and_mid_admission_deadline_preserve_cleanup() {
    let fixture = Fixture::grouped().await;
    let request = resource_request(3, 129, 16, ActualWaveKind::Prefill);
    assert_eq!(
        fixture
            .executor
            .prepare_execution_resources(&[request], &mut || false)
            .unwrap(),
        ExecutorResourcePreparationReceipt {
            outcome: ExecutorResourcePreparationOutcome::Unavailable,
            prepared_participants: 0
        }
    );
    // Expire after a real disposable owner exists, without tying the test to
    // any particular number of budget polls.
    assert_eq!(
        fixture
            .executor
            .prepare_execution_resources(&[request], &mut || fixture
                .executor
                .sequences
                .lock()
                .total_len()
                == 0)
            .unwrap(),
        ExecutorResourcePreparationReceipt {
            outcome: ExecutorResourcePreparationOutcome::Unavailable,
            prepared_participants: 0
        }
    );
    assert_eq!(fixture.executor.sequences.lock().total_len(), 0);
    assert_eq!(fixture.submissions(), 0);
    for request in [
        resource_request(9, 129, 1, ActualWaveKind::Decode),
        resource_request(1, 513, 1, ActualWaveKind::Decode),
        resource_request(8, 129, 9, ActualWaveKind::Prefill),
    ] {
        assert_eq!(
            fixture
                .executor
                .prepare_execution_resources(&[request], &mut || true)
                .unwrap(),
            ExecutorResourcePreparationReceipt {
                outcome: ExecutorResourcePreparationOutcome::Unavailable,
                prepared_participants: 0
            }
        );
    }
    assert_eq!(fixture.executor.sequences.lock().total_len(), 0);
    assert_eq!(fixture.submissions(), 0);
    assert_eq!(
        fixture
            .executor
            .prepare_execution_resources(
                &[resource_request(1, 129, 16, ActualWaveKind::Prefill)],
                &mut || true
            )
            .unwrap(),
        ExecutorResourcePreparationReceipt {
            outcome: ExecutorResourcePreparationOutcome::Prepared,
            prepared_participants: 1
        }
    );
}

#[tokio::test]
async fn workspace_resource_readiness_partial_batch_receipt_preserves_completed_owners() {
    let fixture = Fixture::grouped().await;
    let requests = [
        resource_request(1, 129, 16, ActualWaveKind::Prefill),
        resource_request(3, 129, 1, ActualWaveKind::Decode),
    ];
    let mut saw_owner = false;
    let receipt = fixture
        .executor
        .prepare_execution_resources(&requests, &mut || {
            let active = fixture.executor.sequences.lock().total_len() != 0;
            saw_owner |= active;
            !saw_owner || active
        })
        .unwrap();
    assert_eq!(
        receipt,
        ExecutorResourcePreparationReceipt {
            outcome: ExecutorResourcePreparationOutcome::Unavailable,
            prepared_participants: 1,
        }
    );
    assert_eq!(fixture.executor.sequences.lock().total_len(), 0);
    assert_eq!(fixture.submissions(), 0);
}

fn startup_report(fixture: &Fixture) -> serde_json::Value {
    let state = fixture.executor.startup_preparation.lock();
    let VNextStartupPreparationState::Ready { report } = &*state else {
        panic!("product startup did not reach Ready")
    };
    serde_json::to_value(report).unwrap()
}

#[tokio::test]
async fn workspace_startup_metal_default_does_no_preparation() {
    let fixture = Fixture::new(8, false).await;
    let report = startup_report(&fixture);
    assert!(report.get("workspace_preparation").is_none());
    assert_eq!(report["eager_warmup_waves"], 0);
    assert_eq!(report["capture_waves"], 0);
    assert_eq!(fixture.submissions(), 0);
    assert_eq!(fixture.executor.sequences.lock().total_len(), 0);
}

#[tokio::test]
async fn workspace_startup_metal_resident_buckets_project_before_first_decode() {
    let fixture = Fixture::workspace_startup().await;
    let startup = startup_report(&fixture);
    let report = &startup["workspace_preparation"];
    assert_eq!(report["mode"], "startup");
    assert_eq!(report["maximum_simultaneous_startup_owners"], 8);
    assert_eq!(report["encoded_model_waves"], 0);
    assert_eq!(report["submitted_model_waves"], 0);
    assert_eq!(fixture.submissions(), 0);
    assert_eq!(fixture.executor.sequences.lock().total_len(), 0);
    let expected = fixture
        .executor
        .resolved_plan
        .execution_plan()
        .payload()
        .memory()
        .reusable_execution()
        .unwrap()
        .buckets()
        .len();
    assert_eq!(
        report["prepared_buckets"].as_array().unwrap().len(),
        expected
    );
    assert!(
        report["after"]["process_claimed_bytes"].as_u64().unwrap()
            <= report["after"]["effective_device_usable_ceiling_bytes"]
                .as_u64()
                .unwrap()
    );
    let before = fixture
        .executor
        .plan_resources
        .dynamic_pool_status()
        .unwrap();
    fixture.executor.prepare_startup().await.unwrap();
    let after = fixture
        .executor
        .plan_resources
        .dynamic_pool_status()
        .unwrap();
    assert_eq!(before.epochs(), after.epochs());
    assert_eq!(
        before.process_claimed_bytes(),
        after.process_claimed_bytes()
    );
    assert_eq!(startup_report(&fixture), startup);

    let mut decodes = Vec::new();
    for _ in 0..8 {
        let input = prompt(&[1], 1);
        fixture.admit(&input);
        let PlanRuntimePrefillOutcome::Completed(output) = fixture
            .executor
            .plan_runtime_prefill_with_capacity(&input)
            .await
            .unwrap()
        else {
            panic!("actual fresh prefill did not complete")
        };
        decodes.push(PlanRuntimeDecodeInput::new(
            input.request_id,
            TokenId::new(1),
            Arc::clone(output.output().kv_cache()),
        ));
    }
    // These are real active product owners, not temporary startup IDs. Their
    // future decode projection must reuse each declared resident bucket.
    let cache_ids = decodes
        .iter()
        .map(|input| input.kv_cache.cache_id())
        .collect::<Vec<_>>();
    let requests: Vec<_> = decodes
        .iter()
        .zip(&cache_ids)
        .map(|(input, cache_id)| ExecutorResourcePlanningRequest {
            request_id: &input.request_id,
            cache_id: Some(cache_id.as_str()),
        })
        .collect();
    let deadline = Instant::now() + Duration::from_secs(5);
    let view = loop {
        match fixture.executor.execution_resource_planning_view(
            &requests,
            ResourcePlanningLimits::default(),
            &mut || true,
        ) {
            ResourcePlanningAvailability::Known(view) => break view,
            ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::ReadUnavailable(_))
                if Instant::now() < deadline =>
            {
                std::thread::yield_now()
            }
            ResourcePlanningAvailability::Unknown(reason) => panic!("resource capture: {reason:?}"),
        }
    };
    let submissions = fixture.submissions();
    for width in [1, 2, 4, 8] {
        let rows = (0..width)
            .map(|participant_index| ResourcePlanningRow {
                participant_index,
                start_token: 1,
                token_count: 1,
            })
            .collect::<Vec<_>>();
        match fixture.executor.project_execution_resource_wave_for_kind(
            &view,
            &view.initial_state(),
            &rows,
            ActualWaveKind::Decode,
            &mut || true,
        ) {
            ResourcePlanningAvailability::Known(_) => {}
            ResourcePlanningAvailability::Unknown(reason) => {
                panic!("resident width {width}: {reason:?}")
            }
        }
    }
    assert_eq!(fixture.submissions(), submissions);
    let outputs = decode_outputs(
        fixture
            .executor
            .plan_runtime_batch_decode_with_capacity(&decodes)
            .await
            .unwrap(),
    );
    assert_eq!(outputs.len(), decodes.len());
    assert_eq!(fixture.submissions(), submissions + 1);
    for input in decodes {
        fixture.executor.release_cache(&input.kv_cache.cache_id());
    }
    assert_eq!(fixture.executor.sequences.lock().total_len(), 0);
}
