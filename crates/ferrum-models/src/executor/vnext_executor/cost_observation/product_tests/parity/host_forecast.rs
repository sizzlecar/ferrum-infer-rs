//! Real Metal commands with declared host-policy fixtures. This tests device
//! route equivalence, not real tokenizer UTF-8 outcomes or model qualification.
use super::*;

#[tokio::test]
async fn future_metal_pending_host_subsets_match_actual_full_and_greedy_routes() {
    let fixture = Fixture::with_structured_capture(8).await;
    let prefills = [prompt(&[0, 1, 2, 1], 4), prompt(&[1, 2, 0, 1], 4)];
    for input in &prefills {
        fixture.admit(input);
    }
    let mut prefill_probe = Probe::new(&[&prefills[0].request_id, &prefills[1].request_id]);
    declare_numeric_context(&mut prefill_probe, &[0, 0]);
    predict(&fixture, &prefills, &[], &prefill_probe);
    let outputs = match executed(
        fixture
            .executor
            .plan_runtime_batch_prefill_with_capacity_observed(
                &prefills,
                &mut prefill_probe.context(),
            )
            .await,
    ) {
        PlanRuntimeBatchPrefillOutcome::Completed(outputs) => outputs,
        _ => panic!("initial native prefill did not complete"),
    };
    let mut decodes: Vec<_> = prefills
        .iter()
        .zip(outputs)
        .map(|(input, output)| {
            PlanRuntimeDecodeInput::new(
                input.request_id.clone(),
                TokenId::new(1),
                Arc::clone(output.output().kv_cache()),
            )
        })
        .collect();
    let clean = LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(TokenSelectionMask::new(vec![1, 0, 1])),
        repetition_penalty: None,
    };
    let eligible = [
        FutureHostPendingRowV2 {
            physical_position: 0,
            clean_policy: &clean,
        },
        FutureHostPendingRowV2 {
            physical_position: 1,
            clean_policy: &clean,
        },
    ];
    // Both singleton pending choices, their union, then conditional clean. All
    // are actual native invocations; every prediction also checks no live effects.
    for (step, subset) in [1_u32, 2, 3, 0].into_iter().enumerate() {
        let mut probe = Probe::new(&[&decodes[0].request_id, &decodes[1].request_id]);
        // Structured capture retains algorithm tables in addition to physical
        // rows. Use the product's declared capture budget before execution,
        // instead of the ordinary fixture's intentionally tiny row-only budget.
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
        declare_numeric_context(&mut probe, &[step as u64 + 1; 2]);
        for (position, (decode, participant)) in
            decodes.iter_mut().zip(&mut probe.participants).enumerate()
        {
            let pending = subset & (1 << position) != 0;
            decode.logits_policy = if pending {
                LogitsReturnPolicy::FullLogits
            } else {
                clean.clone()
            };
            let host = participant.host_features.as_mut().unwrap();
            host.policy.empirical_content_domain = Some(HostContentDomainV1::PlainTextGreedyV1);
            host.state.pending_decoded_utf8 = pending;
        }
        let host = FutureHostPendingQueryV2 {
            eligible_rows: if subset == 0 { &[] } else { &eligible },
            constraint: if subset == 0 {
                HostPendingConstraintV2::AnySubset
            } else {
                HostPendingConstraintV2::NonEmptySubset
            },
        };
        let forecast = predict_with_host(&fixture, &[], &decodes, &probe, Some(&host));
        let recipe = forecast
            .projection
            .statistical_evidence
            .as_ref()
            .unwrap()
            .structured_capture()
            .unwrap()
            .unwrap();
        forecast
            .host_content
            .validate(&forecast.projection.shape, recipe)
            .unwrap();
        let HostContentForecastV2::Unresolved(pending) = &forecast.host_content else {
            panic!("conditional future content must remain unresolved");
        };
        assert_eq!(
            pending.eligible_positions(),
            if subset == 0 { &[][..] } else { &[0, 1][..] }
        );
        let before = fixture.submissions();
        let outputs = decode_outputs(executed(
            fixture
                .executor
                .plan_runtime_batch_decode_with_capacity_observed(
                    &decodes,
                    &mut probe.context().with_structured_capture(true),
                )
                .await,
        ));
        assert_eq!(fixture.submissions(), before + 1);
        assert_canonical(&probe, &forecast.projection.shape);
        let observations = probe.recorder.observations();
        let actual = observations[0].shape.as_ref().unwrap();
        let actual_recipe = actual
            .statistical_evidence
            .as_ref()
            .unwrap()
            .structured_capture()
            .unwrap()
            .unwrap();
        assert_eq!(actual_recipe.device(), recipe.device());
        assert_eq!(
            actual_recipe.algorithm_work().unwrap(),
            recipe.algorithm_work().unwrap()
        );
        for (decode, output) in decodes.iter_mut().zip(outputs) {
            decode.kv_cache = Arc::clone(&output.kv_cache);
        }
    }
}
