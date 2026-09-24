//! Real ordinary product execution and future projection, with independent
//! request KV pages. Host features are a contract fixture, not tokenizer SLO
//! or empirical timing qualification. No allocator order is fabricated.
use super::*;

async fn run(lengths: [usize; 3]) -> ([u8; 32], [u8; 32], DeviceNumericWorkV1, Vec<u32>) {
    let fixture = Fixture::grouped().await;
    let mut decodes = Vec::new();
    for length in lengths {
        let mut input = PlanRuntimePrefillInput::new(
            RequestId::new(),
            (0..length)
                .map(|index| TokenId::new((index % 3) as u32))
                .collect::<Vec<_>>(),
            512,
            PrefillChunk::new(0, 64.min(length), length).unwrap(),
        )
        .unwrap();
        fixture.admit(&input);
        for start in (0..length).step_by(64) {
            input.chunk = PrefillChunk::new(start, (length - start).min(64), length).unwrap();
            let PlanRuntimePrefillOutcome::Completed(output) = executed(
                fixture
                    .executor
                    .plan_runtime_prefill_with_capacity_observed(
                        &input,
                        &mut Probe::new(&[&input.request_id]).context(),
                    )
                    .await,
            ) else {
                panic!("real prefill must complete");
            };
            assert_eq!(output.completed_chunk(), input.chunk);
            if input.chunk.is_final() {
                decodes.push(PlanRuntimeDecodeInput::new(
                    input.request_id.clone(),
                    TokenId::new(1),
                    Arc::clone(output.output().kv_cache()),
                ));
            }
        }
    }
    fixture.warm(&[], &decodes);
    let ids = decodes
        .iter()
        .map(|row| &row.request_id)
        .collect::<Vec<_>>();
    let mut probe = Probe::new(&ids);
    declare_numeric_context(&mut probe, &[1; 3]);
    for participant in &mut probe.participants {
        participant
            .host_features
            .as_mut()
            .unwrap()
            .policy
            .empirical_content_domain = Some(HostContentDomainV1::PlainTextGreedyV1);
    }
    let rows = fixture.rows(&[], &decodes);
    let requests = rows
        .iter()
        .map(|(sequence, _, _)| ExecutorResourcePlanningRequest {
            request_id: sequence.request_id(),
            cache_id: Some(sequence.cache_id.as_str()),
        })
        .collect::<Vec<_>>();
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
            other => panic!("real capture: {other:?}"),
        }
    };
    let query_rows = rows
        .iter()
        .enumerate()
        .map(|(participant_index, (sequence, _, range))| {
            let host = probe
                .participants
                .iter()
                .find(|p| &p.request_id == sequence.request_id())
                .unwrap();
            let input = decodes
                .iter()
                .find(|d| &d.request_id == sequence.request_id())
                .unwrap();
            FutureWaveCostRow {
                participant_index,
                work: ActualRowWork::Decode {
                    kv_tokens: u32::try_from(range.start).unwrap(),
                },
                output: FutureCostOutput::Decode {
                    policy: &input.logits_policy,
                },
                host_policy_signature: host.output_policy_signature.unwrap(),
                host_features: host.host_features,
            }
        })
        .collect::<Vec<_>>();
    let before = fixture.submissions();
    let resource_before = fixture.target_resource_evidence(&[], &decodes);
    let predicted = match fixture.executor.project_execution_cost_wave(
        &view,
        &view.initial_state(),
        &FutureWaveCostQuery {
            kind: ActualWaveKind::Decode,
            rows: &query_rows,
        },
        &mut || true,
    ) {
        ExecutionCostRouteAvailability::Known(v) => v,
        other => panic!("real future route: {other:?}"),
    };
    assert_eq!(fixture.submissions(), before);
    assert_eq!(
        fixture.target_resource_evidence(&[], &decodes),
        resource_before
    );
    let future = predicted
        .statistical_evidence
        .as_ref()
        .expect("complete actual-selected producer chain");
    let future_v2 = future
        .independent_attention_v2()
        .expect("real independent-row producer");
    future_v2.validate_exact(&predicted.shape).unwrap();
    drop(query_rows);
    drop(requests);
    drop(rows);
    let output = decode_outputs(executed(
        fixture
            .executor
            .plan_runtime_batch_decode_with_capacity_observed(&decodes, &mut probe.context())
            .await,
    ));
    assert_eq!(output.len(), 3);
    assert_eq!(fixture.submissions(), before + 1);
    assert_canonical(&probe, &predicted.shape);
    let observations = probe.recorder.observations();
    let actual = observations[0].shape.as_ref().unwrap();
    let observed = actual
        .statistical_evidence
        .as_ref()
        .expect("real native whole-wave statistics");
    let observed_v2 = observed
        .independent_attention_v2()
        .expect("real native independent-row evidence");
    observed_v2.validate_actual(actual).unwrap();
    assert_eq!(
        observed, future,
        "ordered V1 stays actual/future equivalent"
    );
    assert_eq!(
        observed_v2, future_v2,
        "compare V2 explicitly: legacy equality excludes the sidecar"
    );
    let order = actual
        .rows
        .iter()
        .map(|row| match row.work {
            ActualRowWork::Decode { kv_tokens } => kv_tokens,
            _ => panic!("decode row"),
        })
        .collect();
    let result = (
        *observed.family_signature(),
        *observed_v2.family_signature(),
        observed.work(),
        order,
    );
    for input in decodes {
        fixture.executor.release_cache(&input.kv_cache.cache_id());
    }
    result
}

#[tokio::test]
async fn independent_rows_v2_real_metal_future_native_permutations_preserve_exact_order() {
    // Separate ordinary coordinators; input/admission order is explicit. The
    // assertion below inspects the actual authority order instead of changing it.
    let mut results = Vec::new();
    for lengths in [[100, 300, 120], [300, 100, 120], [100, 120, 300]] {
        results.push(run(lengths).await);
    }
    let group_positions = results
        .iter()
        .map(|(_, _, _, rows)| rows.iter().position(|&kv| kv >= 255).unwrap())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        group_positions.len(),
        3,
        "fixture must really execute DGD/GDD/DDG, not infer from input order"
    );
    assert_eq!(
        results.iter().map(|r| r.0).collect::<BTreeSet<_>>().len(),
        3
    );
    for result in &results[1..] {
        assert_eq!(
            results[0].1, result.1,
            "only empirical row-group family is order invariant"
        );
        assert_eq!(results[0].2, result.2);
    }
}
