//! The real Metal model is constructed through the product startup hook. No
//! fixture warm() helper is called: every declared Step/Invocation slot must
//! already be resident before the first product request.
use super::*;
use ferrum_interfaces::model_executor::ExecutorResourcePlanningRequest;

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
