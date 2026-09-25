//! Whole-model future/actual parity through the ordinary observed product API.
//! The host feature context below is a declared contract fixture, not evidence
//! of an engine tokenizer/output actor. Device commands and work are real.
use super::*;
use ferrum_interfaces::model_executor::{ExecutorResourcePlanningRequest, TokenSelectionMask};

fn mask_counts(fixture: &Fixture) -> (u64, u64) {
    (
        fixture
            .executor
            .metrics
            .token_mask_upload_participants
            .load(Ordering::Relaxed),
        fixture
            .executor
            .metrics
            .token_mask_cache_hit_participants
            .load(Ordering::Relaxed),
    )
}

fn declare_numeric_context(probe: &mut Probe, generated: &[u64]) {
    assert_eq!(probe.participants.len(), generated.len());
    for (participant, &generated_tokens_before) in probe.participants.iter_mut().zip(generated) {
        participant.host_features = Some(HostCostFeaturesV1 {
            policy: HostCostPolicyV2 {
                empirical_content_domain: None,
                categorical_signature: [11; 32],
                decoder_text_bytes_per_token: 8,
                decoder_scratch_bytes_per_token: 16,
                raw_token_bytes_bound: 8,
            },
            state: HostCostStateV1 {
                generated_tokens_before,
                maximum_output_tokens: 16,
                sampling_history_tokens: generated_tokens_before,
                sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
                pending_decoded_utf8: false,
                completion_state_signature: satisfied_completion_cost_signature(),
            },
        });
    }
}

/// Snapshot after explicit resource-only fixture preparation, never after
/// encoding the wave being predicted. The ordinary API below still owns its
/// real Step, upload, selected providers, native submission and completion.
fn predict(
    fixture: &Fixture,
    prefills: &[PlanRuntimePrefillInput],
    decodes: &[PlanRuntimeDecodeInput],
    probe: &Probe,
) -> CanonicalWaveCostShape {
    predict_with_host(fixture, prefills, decodes, probe, None)
        .projection
        .shape
}

fn predict_with_host(
    fixture: &Fixture,
    prefills: &[PlanRuntimePrefillInput],
    decodes: &[PlanRuntimeDecodeInput],
    probe: &Probe,
    host: Option<&FutureHostPendingQueryV2<'_>>,
) -> ExecutionCostRouteForecastV2 {
    fixture.warm(prefills, decodes);
    let before = fixture.target_resource_evidence(prefills, decodes);
    let submissions = fixture.submissions();
    let masks = mask_counts(fixture);
    let rows = fixture.rows(prefills, decodes);
    let requests: Vec<_> = rows
        .iter()
        .map(|(sequence, _, _)| ExecutorResourcePlanningRequest {
            request_id: sequence.request_id(),
            cache_id: decodes
                .iter()
                .any(|input| &input.request_id == sequence.request_id())
                .then_some(sequence.cache_id.as_str()),
        })
        .collect();
    let deadline = Instant::now() + Duration::from_secs(5);
    let view = loop {
        match fixture.executor.execution_cost_route_view(
            &requests,
            ResourcePlanningLimits::default(),
            &mut || true,
        ) {
            ExecutionCostRouteAvailability::Known(view) => break view,
            ExecutionCostRouteAvailability::Unknown(ExecutionCostRouteUnknown::Resource(
                ResourcePlanningUnknown::ReadUnavailable(_),
            )) if Instant::now() < deadline => std::thread::yield_now(),
            ExecutionCostRouteAvailability::Unknown(reason) => panic!("future capture: {reason:?}"),
        }
    }
    .with_structured_capture(host.is_some());
    let query_rows: Vec<_> = rows
        .iter()
        .enumerate()
        .map(|(participant_index, (sequence, _, range))| {
            let host = probe
                .participants
                .iter()
                .find(|p| &p.request_id == sequence.request_id())
                .unwrap();
            let (work, output) = if let Some(input) = prefills
                .iter()
                .find(|p| &p.request_id == sequence.request_id())
            {
                (
                    ActualRowWork::Prefill {
                        offset: range.start.try_into().unwrap(),
                        count: range.len().try_into().unwrap(),
                        total_prompt_tokens: input.input_tokens.len().try_into().unwrap(),
                    },
                    FutureCostOutput::Prefill {
                        final_logits: input.chunk.is_final(),
                    },
                )
            } else {
                let input = decodes
                    .iter()
                    .find(|p| &p.request_id == sequence.request_id())
                    .unwrap();
                (
                    ActualRowWork::Decode {
                        kv_tokens: range.start.try_into().unwrap(),
                    },
                    FutureCostOutput::Decode {
                        policy: &input.logits_policy,
                    },
                )
            };
            FutureWaveCostRow {
                participant_index,
                work,
                output,
                host_policy_signature: host.output_policy_signature.unwrap(),
                host_features: host.host_features,
            }
        })
        .collect();
    let query = FutureWaveCostQuery {
        kind: if decodes.is_empty() {
            ActualWaveKind::Prefill
        } else if prefills.is_empty() {
            ActualWaveKind::Decode
        } else {
            ActualWaveKind::Mixed
        },
        rows: &query_rows,
    };
    let query_once = || {
        let result = if let Some(host) = host {
            fixture
                .executor
                .project_execution_cost_wave_with_host_content(
                    &view,
                    &view.initial_state(),
                    &query,
                    host,
                    &mut || true,
                )
        } else {
            match fixture.executor.project_execution_cost_wave(
                &view,
                &view.initial_state(),
                &query,
                &mut || true,
            ) {
                ExecutionCostRouteAvailability::Known(projection) => {
                    ExecutionCostRouteAvailability::Known(ExecutionCostRouteForecastV2 {
                        projection,
                        host_content: HostContentForecastV2::Exact,
                    })
                }
                ExecutionCostRouteAvailability::Unknown(reason) => {
                    ExecutionCostRouteAvailability::Unknown(reason)
                }
            }
        };
        match result {
            ExecutionCostRouteAvailability::Known(projection) => projection,
            ExecutionCostRouteAvailability::Unknown(reason) => {
                panic!("future projection: {reason:?}")
            }
        }
    };
    let projection = query_once();
    let shape = &projection.projection.shape;
    assert_eq!(
        query_once().projection.shape,
        *shape,
        "pure projections from one view must agree"
    );
    assert_eq!(view.initial_state().projected_waves(), 0);
    assert_eq!(projection.projection.state.projected_waves(), 1);
    assert!(
        shape.numeric_features.is_some(),
        "also compare the core readback/host numeric identity"
    );
    assert_eq!(
        fixture.submissions(),
        submissions,
        "query must not encode or submit"
    );
    assert_eq!(
        mask_counts(fixture),
        masks,
        "query must not publish mask residency"
    );
    assert_eq!(
        fixture.target_resource_evidence(prefills, decodes),
        before,
        "query must preserve target backing/frame/initialization evidence"
    );
    projection
}

mod host_forecast;

fn assert_canonical(probe: &Probe, expected: &CanonicalWaveCostShape) {
    let observed = &probe.recorder.observations()[0];
    let actual = observed.shape.as_ref().unwrap();
    probe.assert_wave(
        actual.kind,
        &probe
            .participants
            .iter()
            .map(|participant| {
                actual
                    .rows
                    .iter()
                    .find(|row| row.request_id == participant.request_id)
                    .unwrap()
                    .work
            })
            .collect::<Vec<_>>(),
    );
    assert_eq!(actual.restore_bytes, 0);
    assert_eq!(actual.maintenance_bytes, 0);
    assert_eq!(actual.maintenance_units, 0);
    // Equality covers ordered row work, every command's provider identity and
    // dispatch/transfer counts, output mode/masks, readback and numeric context.
    assert_eq!(
        &CanonicalWaveCostShape {
            kind: actual.kind,
            path: actual.path,
            graph: actual.graph,
            row_order: actual.row_order,
            provider_signature: actual.provider_signature,
            output_policy_signature: actual.output_policy_signature,
            numeric_features: actual.numeric_features.clone(),
            host_content_features: actual.host_content_features,
            row_multiset_features: actual.row_multiset_features.clone(),
            rows: actual.rows.iter().map(|row| row.work).collect(),
            recurrent_state_bytes: actual.recurrent_state_bytes,
        },
        expected
    );
}

#[tokio::test]
async fn future_metal_product_full_canonical_partial_final_decode_and_mixed() {
    let fixture = Fixture::new(8, false).await;
    let mut input = prompt(&[0, 1, 2, 1], 2);
    fixture.admit(&input);
    let mut probe = Probe::new(&[&input.request_id]);
    declare_numeric_context(&mut probe, &[0]);
    let expected = predict(&fixture, std::slice::from_ref(&input), &[], &probe);
    let before = fixture.submissions();
    let partial = executed(
        fixture
            .executor
            .plan_runtime_prefill_with_capacity_observed(&input, &mut probe.context())
            .await,
    );
    assert!(matches!(partial, PlanRuntimePrefillOutcome::Completed(_)));
    assert_eq!(fixture.submissions(), before + 1);
    assert_canonical(&probe, &expected);

    input.chunk = PrefillChunk::new(2, 2, 4).unwrap();
    let mut probe = Probe::new(&[&input.request_id]);
    declare_numeric_context(&mut probe, &[0]);
    let expected = predict(&fixture, std::slice::from_ref(&input), &[], &probe);
    let final_output = match executed(
        fixture
            .executor
            .plan_runtime_batch_prefill_with_capacity_observed(
                std::slice::from_ref(&input),
                &mut probe.context(),
            )
            .await,
    ) {
        PlanRuntimeBatchPrefillOutcome::Completed(mut outputs) => outputs.remove(0),
        _ => panic!("real final prefill did not complete"),
    };
    assert_canonical(&probe, &expected);
    let mut decode = PlanRuntimeDecodeInput::new(
        input.request_id.clone(),
        TokenId::new(1),
        Arc::clone(final_output.output().kv_cache()),
    );
    let selection = || LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(TokenSelectionMask::new(vec![0, 1, 0])),
        repetition_penalty: None,
    };
    // Residency deliberately retains only Weak mask storage. Keep the first
    // source alive while checking a distinct allocation with equal contents.
    // Replacing the sole strong owner would correctly make both future and
    // actual routes upload again, rather than exercise the equal-content hit.
    let resident_selection = selection();
    let equal_selection = selection();
    let (
        LogitsReturnPolicy::GreedyArgmax {
            token_mask: Some(resident),
            ..
        },
        LogitsReturnPolicy::GreedyArgmax {
            token_mask: Some(equal),
            ..
        },
    ) = (&resident_selection, &equal_selection)
    else {
        unreachable!("selection fixture")
    };
    assert!(!Arc::ptr_eq(
        &resident.valid_token_mask,
        &equal.valid_token_mask
    ));
    assert_eq!(
        resident.valid_token_mask.as_ref(),
        equal.valid_token_mask.as_ref()
    );
    // New Selection content is uploaded, distinct equal storage hits, then
    // AllValid and Selection transitions each upload their actual contents.
    for (generated, policy, uploads) in [
        (1, resident_selection.clone(), 1),
        (2, equal_selection, 0),
        (3, LogitsReturnPolicy::FullLogits, 1),
        (4, selection(), 1),
    ] {
        decode.logits_policy = policy;
        let mut probe = Probe::new(&[&decode.request_id]);
        declare_numeric_context(&mut probe, &[generated]);
        let expected = predict(&fixture, &[], std::slice::from_ref(&decode), &probe);
        let masks = mask_counts(&fixture);
        let outputs = decode_outputs(executed(
            fixture
                .executor
                .plan_runtime_batch_decode_with_capacity_observed(
                    std::slice::from_ref(&decode),
                    &mut probe.context(),
                )
                .await,
        ));
        assert_canonical(&probe, &expected);
        assert_eq!(mask_counts(&fixture).0 - masks.0, uploads);
        assert_eq!(mask_counts(&fixture).1 - masks.1, 1 - uploads);
        decode.kv_cache = Arc::clone(&outputs[0].kv_cache);
    }
    // The ledger must not keep mask storage alive. After dropping every old
    // strong source, equal newly allocated content requires a real upload.
    let expired_mask = match &decode.logits_policy {
        LogitsReturnPolicy::GreedyArgmax {
            token_mask: Some(mask),
            ..
        } => Arc::downgrade(&mask.valid_token_mask),
        _ => unreachable!("last wave selected a mask"),
    };
    drop(resident_selection);
    decode.logits_policy = selection();
    assert!(expired_mask.upgrade().is_none());
    let mut probe = Probe::new(&[&decode.request_id]);
    declare_numeric_context(&mut probe, &[5]);
    let expected = predict(&fixture, &[], std::slice::from_ref(&decode), &probe);
    let masks = mask_counts(&fixture);
    let outputs = decode_outputs(executed(
        fixture
            .executor
            .plan_runtime_batch_decode_with_capacity_observed(
                std::slice::from_ref(&decode),
                &mut probe.context(),
            )
            .await,
    ));
    assert_canonical(&probe, &expected);
    assert_eq!(mask_counts(&fixture).0 - masks.0, 1);
    assert_eq!(mask_counts(&fixture).1 - masks.1, 0);
    decode.kv_cache = Arc::clone(&outputs[0].kv_cache);
    let fresh = prompt(&[1, 0], 2);
    fixture.admit(&fresh);
    // Reverse the real authority order in the correlated host context. Do not
    // assume allocator slot IDs are monotone in request admission order.
    let rows = fixture.rows(std::slice::from_ref(&fresh), std::slice::from_ref(&decode));
    let ids: Vec<_> = rows
        .iter()
        .rev()
        .map(|(sequence, _, _)| sequence.request_id())
        .collect();
    let mut probe = Probe::new(&ids);
    let generated: Vec<_> = ids
        .iter()
        .map(|id| if **id == fresh.request_id { 0 } else { 6 })
        .collect();
    declare_numeric_context(&mut probe, &generated);
    drop(ids);
    drop(rows);
    let expected = predict(
        &fixture,
        std::slice::from_ref(&fresh),
        std::slice::from_ref(&decode),
        &probe,
    );
    let output = executed(
        fixture
            .executor
            .plan_runtime_mixed_batch_with_capacity_observed(
                std::slice::from_ref(&fresh),
                std::slice::from_ref(&decode),
                &mut probe.context(),
            )
            .await,
    );
    let PlanRuntimeMixedBatchOutcome::Completed { prefills, decodes } = output else {
        panic!("real mixed wave did not complete");
    };
    assert_canonical(&probe, &expected);
    assert_eq!(
        probe.recorder.observations()[0]
            .shape
            .as_ref()
            .unwrap()
            .rows
            .iter()
            .map(|row| row.input_index)
            .collect::<Vec<_>>(),
        vec![1, 0]
    );
    assert_eq!(decodes.len(), 1);
    fixture.executor.release_cache(&decode.kv_cache.cache_id());
    fixture
        .executor
        .discard_plan_runtime_prefill(
            prefills
                .into_iter()
                .next()
                .unwrap()
                .into_parts()
                .0
                .into_parts()
                .0,
        )
        .unwrap();
    assert_eq!(fixture.executor.device_timing_mode(), DeviceTimingMode::Off);
}

mod grouped;

mod independent_rows;
