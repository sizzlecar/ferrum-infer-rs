//! Real product path, Qwen GQA geometry with tiny hidden/FFN weights. No mock
//! provider or fabricated physical ranges. Context crosses a KV page boundary.
use super::*;

#[tokio::test]
async fn future_metal_grouped_decode_long_context_and_rollout_match_actual() {
    let fixture = Fixture::grouped().await;
    let mut decodes = Vec::new();
    for length in [255, 256] {
        let mut input = PlanRuntimePrefillInput::new(
            RequestId::new(),
            (0..length)
                .map(|index| TokenId::new((index % 3) as u32))
                .collect::<Vec<_>>(),
            512,
            PrefillChunk::new(0, 64, length).unwrap(),
        )
        .unwrap();
        fixture.admit(&input);
        for start in (0..length).step_by(64) {
            input.chunk = PrefillChunk::new(start, (length - start).min(64), length).unwrap();
            let output = executed(
                fixture
                    .executor
                    .plan_runtime_prefill_with_capacity_observed(
                        &input,
                        &mut Probe::new(&[&input.request_id]).context(),
                    )
                    .await,
            );
            let PlanRuntimePrefillOutcome::Completed(output) = output else {
                panic!("prefill deferred");
            };
            assert_eq!(
                output.completed_chunk(),
                input.chunk,
                "prefill must keep the declared frontier"
            );
            if input.chunk.is_final() {
                decodes.push(PlanRuntimeDecodeInput::new(
                    input.request_id.clone(),
                    TokenId::new(1),
                    Arc::clone(output.output().kv_cache()),
                ));
            }
        }
    }
    assert_eq!(decodes.len(), 2);
    // Capture before any of the three decode submissions. Reserve reusable
    // workspace through the existing resource-only auxiliary fixture, preserving
    // each target's exact KV frontier and all native submission counters.
    fixture.warm(&[], &decodes);
    let rows = fixture.rows(&[], &decodes);
    let requests: Vec<_> = rows
        .iter()
        .map(|(sequence, _, _)| ExecutorResourcePlanningRequest {
            request_id: sequence.request_id(),
            cache_id: Some(sequence.cache_id.as_str()),
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
            ExecutionCostRouteAvailability::Unknown(reason) => panic!("capture: {reason:?}"),
        }
    };
    let mut state = view.initial_state();
    let mut predicted = Vec::new();
    let before = fixture.submissions();
    let resource_before = fixture.target_resource_evidence(&[], &decodes);
    for wave in 0..3_u64 {
        let ids: Vec<_> = rows
            .iter()
            .map(|(sequence, _, _)| sequence.request_id())
            .collect();
        let mut probe = Probe::new(&ids);
        declare_numeric_context(&mut probe, &[1 + wave; 2]);
        let query_rows: Vec<_> = rows
            .iter()
            .enumerate()
            .map(|(index, (_, _, range))| FutureWaveCostRow {
                participant_index: index,
                work: ActualRowWork::Decode {
                    kv_tokens: u32::try_from(range.start).unwrap() + wave as u32,
                },
                output: FutureCostOutput::Decode {
                    policy: &decodes[0].logits_policy,
                },
                host_policy_signature: probe.participants[index].output_policy_signature.unwrap(),
                host_features: probe.participants[index].host_features,
            })
            .collect();
        match fixture.executor.project_execution_cost_wave(
            &view,
            &state,
            &FutureWaveCostQuery {
                kind: ActualWaveKind::Decode,
                rows: &query_rows,
            },
            &mut || true,
        ) {
            ExecutionCostRouteAvailability::Known(projection) => {
                state = projection.state;
                predicted.push(projection.shape);
            }
            ExecutionCostRouteAvailability::Unknown(reason) => {
                panic!("rollout wave {wave}: {reason:?}")
            }
        }
    }
    assert_eq!(fixture.submissions(), before);
    assert_eq!(
        fixture.target_resource_evidence(&[], &decodes),
        resource_before
    );
    drop(requests);
    drop(rows);
    for (wave, expected) in predicted.iter().enumerate() {
        let ids: Vec<_> = decodes.iter().map(|row| &row.request_id).collect();
        let mut probe = Probe::new(&ids);
        declare_numeric_context(&mut probe, &[1 + wave as u64; 2]);
        let outputs = decode_outputs(executed(
            fixture
                .executor
                .plan_runtime_batch_decode_with_capacity_observed(&decodes, &mut probe.context())
                .await,
        ));
        assert_canonical(&probe, expected);
        assert_eq!(outputs.len(), decodes.len());
        for (input, output) in decodes.iter_mut().zip(&outputs) {
            assert_eq!(input.kv_cache.cache_id(), output.kv_cache.cache_id());
            input.kv_cache = Arc::clone(&output.kv_cache);
        }
    }
    assert_eq!(fixture.submissions(), before + 3);
    for input in decodes {
        fixture.executor.release_cache(&input.kv_cache.cache_id());
    }
}
