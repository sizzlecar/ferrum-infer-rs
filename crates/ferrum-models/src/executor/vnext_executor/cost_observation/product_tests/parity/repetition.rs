//! Genuine greedy repetition dispatch compared with the pure first-wave route.
//! Histories contain real vocabulary IDs; the executor owns dedup and padding.
use super::*;
use ferrum_interfaces::model_executor::GreedyRepetitionPenalty;

#[tokio::test]
async fn future_metal_real_repetition_history_matches_actual_padded_uploads() {
    let fixture = Fixture::with_structured_capture(8).await;
    let prefills = [prompt(&[0, 1, 2, 1], 4)];
    fixture.admit(&prefills[0]);
    let mut probe = Probe::new(&[&prefills[0].request_id]);
    declare_numeric_context(&mut probe, &[0]);
    let output = match executed(
        fixture
            .executor
            .plan_runtime_batch_prefill_with_capacity_observed(&prefills, &mut probe.context())
            .await,
    ) {
        PlanRuntimeBatchPrefillOutcome::Completed(mut outputs) => outputs.remove(0),
        _ => panic!("initial native prefill did not complete"),
    };
    let mut decode = PlanRuntimeDecodeInput::new(
        prefills[0].request_id.clone(),
        TokenId::new(1),
        Arc::clone(output.output().kv_cache()),
    );
    // Empty, repeated, and growing unique histories exercise both actual
    // upload branches, including whole-capacity padding for every nonempty set.
    for (step, ids) in [vec![], vec![1], vec![1, 1, 2], vec![1, 2, 0, 1]]
        .into_iter()
        .enumerate()
    {
        let repetition = GreedyRepetitionPenalty::new(1.1, ids);
        let expected_unique = repetition.token_ids().len();
        decode.logits_policy = LogitsReturnPolicy::GreedyArgmax {
            token_mask: None,
            repetition_penalty: Some(repetition),
        };
        let mut probe = Probe::new(&[&decode.request_id]);
        declare_numeric_context(&mut probe, &[step as u64 + 1]);
        let predicted = predict(&fixture, &[], std::slice::from_ref(&decode), &probe);
        assert_eq!(
            predicted.numeric_features.as_ref().unwrap().rows[0].repetition_tokens,
            expected_unique as u64
        );
        let mut outputs = decode_outputs(executed(
            fixture
                .executor
                .plan_runtime_batch_decode_with_capacity_observed(
                    std::slice::from_ref(&decode),
                    &mut probe.context(),
                )
                .await,
        ));
        assert_canonical(&mut probe, &predicted);
        decode.kv_cache = Arc::clone(&outputs.remove(0).kv_cache);
    }
}

#[tokio::test]
async fn future_metal_numeric_repetition_interval_has_real_anchor_and_same_device_route() {
    let fixture = Fixture::with_structured_capture(8).await;
    let prefills = [prompt(&[0, 1, 2, 1], 4)];
    fixture.admit(&prefills[0]);
    let mut prefill = Probe::new(&[&prefills[0].request_id]);
    declare_numeric_context(&mut prefill, &[0]);
    let output = match executed(
        fixture
            .executor
            .plan_runtime_batch_prefill_with_capacity_observed(&prefills, &mut prefill.context())
            .await,
    ) {
        PlanRuntimeBatchPrefillOutcome::Completed(mut outputs) => outputs.remove(0),
        _ => panic!("native prefill did not complete"),
    };
    let mut decode = PlanRuntimeDecodeInput::new(
        prefills[0].request_id.clone(),
        TokenId::new(1),
        Arc::clone(output.output().kv_cache()),
    );
    for ids in [vec![1], vec![1, 2], vec![1, 2, 0]] {
        let count = ids.len() as u64;
        decode.logits_policy = LogitsReturnPolicy::GreedyArgmax {
            token_mask: None,
            repetition_penalty: Some(GreedyRepetitionPenalty::new(1.1, ids)),
        };
        let mut probe = Probe::new(&[&decode.request_id]);
        probe.recorder = BoundedWaveRecorder::new(
            NonZeroU64::new(1).unwrap(),
            CostRecorderLimits {
                max_waves: 1,
                max_rows_per_wave: 8,
                max_retained_rows: ferrum_types::SloCostObservationConfig::default()
                    .max_retained_rows_per_call
                    .get(),
            },
        )
        .unwrap();
        declare_numeric_context(&mut probe, &[3]);
        probe.participants[0]
            .host_features
            .as_mut()
            .unwrap()
            .policy
            .empirical_content_domain = Some(HostContentDomainV1::PlainTextInstalledV2(
            PlainTextPolicyCapabilityV2 {
                sampling: PlainTextSamplingRouteV2::Greedy {
                    repetition_penalty: true,
                },
                model_eos: true,
                user_stop: false,
            },
        ));
        let host = FutureHostPendingQueryV2 {
            eligible_rows: &[],
            constraint: HostPendingConstraintV2::AnySubset,
        };
        let forecast = predict_with_host_and_numeric_repetition(
            &fixture,
            &[],
            std::slice::from_ref(&decode),
            &probe,
            Some(&host),
            true,
        );
        let HostContentForecastV2::Unresolved(bound) = &forecast.host_content else {
            panic!("numeric history must retain its uncertainty");
        };
        assert_eq!(bound.repetition_upper_sum(), Some(3));
        assert_eq!(
            forecast
                .projection
                .shape
                .numeric_features
                .as_ref()
                .unwrap()
                .rows[0]
                .repetition_tokens,
            count
        );
        let recipe = forecast
            .projection
            .statistical_evidence
            .as_ref()
            .unwrap()
            .structured_capture()
            .unwrap()
            .unwrap();
        // Reproject another genuine valid history at the same live frontier,
        // before executing either. This isolates repetition from KV growth.
        let mut alternate = decode.clone();
        alternate.logits_policy = LogitsReturnPolicy::GreedyArgmax {
            token_mask: None,
            repetition_penalty: Some(GreedyRepetitionPenalty::new(1.1, vec![0, 1, 2])),
        };
        let alternate = predict_with_host_and_numeric_repetition(
            &fixture,
            &[],
            std::slice::from_ref(&alternate),
            &probe,
            Some(&host),
            true,
        );
        let alternate_recipe = alternate
            .projection
            .statistical_evidence
            .as_ref()
            .unwrap()
            .structured_capture()
            .unwrap()
            .unwrap();
        // Both histories use the same padded device program. Their exact
        // evidence still binds the distinct numerical repetition counts.
        recipe.validate_exact(&forecast.projection.shape).unwrap();
        alternate_recipe
            .validate_exact(&alternate.projection.shape)
            .unwrap();
        if count != 3 {
            assert!(recipe.validate_exact(&alternate.projection.shape).is_err());
            assert!(alternate_recipe
                .validate_exact(&forecast.projection.shape)
                .is_err());
        }
        let device = recipe.device();
        let other_device = alternate_recipe.device();
        assert_eq!(
            (
                device.ordered_template(),
                device.provider_grouped_template(),
                device.physical_commands(),
                device.product(),
                device.readback(),
                device.retries(),
                device.aggregate_work(),
                device.replay_work(),
            ),
            (
                other_device.ordered_template(),
                other_device.provider_grouped_template(),
                other_device.physical_commands(),
                other_device.product(),
                other_device.readback(),
                other_device.retries(),
                other_device.aggregate_work(),
                other_device.replay_work(),
            )
        );
        let work = device.algorithm_work().unwrap();
        let other_work = other_device.algorithm_work().unwrap();
        assert_eq!(work.entries(), other_work.entries());
        assert_eq!(
            work.ordered_command_binding(),
            other_work.ordered_command_binding()
        );
        assert_eq!(
            work.physical_command_count(),
            other_work.physical_command_count()
        );
        assert_eq!(
            work.selected_command_count(),
            other_work.selected_command_count()
        );
        assert_eq!(work.aggregate_work(), other_work.aggregate_work());
        let mut outputs = decode_outputs(executed(
            fixture
                .executor
                .plan_runtime_batch_decode_with_capacity_observed(
                    std::slice::from_ref(&decode),
                    &mut probe.context().with_structured_capture(true),
                )
                .await,
        ));
        assert_canonical(&mut probe, &forecast.projection.shape);
        let observations = probe.recorder.observations();
        let actual = observations[0].shape.as_ref().unwrap();
        let actual_recipe = actual
            .statistical_evidence
            .as_ref()
            .unwrap()
            .structured_capture()
            .unwrap()
            .unwrap();
        recipe.validate_actual(actual).unwrap();
        assert_eq!(actual_recipe.device(), recipe.device());
        assert_eq!(
            actual_recipe.algorithm_work().unwrap(),
            recipe.algorithm_work().unwrap()
        );
        decode.kv_cache = Arc::clone(&outputs.remove(0).kv_cache);
    }
}
