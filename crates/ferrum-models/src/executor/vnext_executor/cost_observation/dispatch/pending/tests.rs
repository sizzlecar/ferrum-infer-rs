use super::*;
use ferrum_interfaces::vnext::{
    DeviceBatchingForm, DeviceCommandPhase, DeviceCostGraphConfiguration,
    DeviceCostGraphStreamState, DeviceNativeOperationId, DeviceNativeWorkAttribution,
    DeviceSubmissionGraphEvidence,
};

#[path = "../../../../../../../ferrum-interfaces/tests/vnext_device_operation_contract/mod.rs"]
mod contract;

fn projection(replayed_without_segment: bool) -> FrozenActualProjection {
    // The observed CUDA disposition: the original OnDemand stream is unchanged,
    // but capture was requested for one candidate and produced no graph work.
    let state =
        DeviceCostGraphStreamState::new(DeviceCostGraphConfiguration::OnDemand, 35, 35, 0).unwrap();
    let graph = DeviceSubmissionGraphEvidence::new(
        state,
        state,
        true,
        1,
        0,
        0,
        0,
        u64::from(replayed_without_segment),
    )
    .unwrap();
    let attribution = DeviceSubmissionAttribution::new(vec![DeviceNativeWorkAttribution::new(
        0,
        Some(0),
        DeviceCommandPhase::Compute,
        DeviceNativeOperationId::new("fixture.native").unwrap(),
        if replayed_without_segment {
            DeviceExecutionPath::Replayed
        } else {
            DeviceExecutionPath::Eager
        },
        DeviceBatchingForm::Packed,
        1,
        1,
        1,
        0,
        replayed_without_segment.then_some(1),
    )
    .unwrap()])
    .unwrap()
    .with_graph_evidence(graph)
    .unwrap();
    FrozenActualProjection {
        providers: Arc::new(ProviderIdentityTable {
            rows: vec![OwnedProviderIdentity {
                provider_id: "fixture.provider".into(),
                implementation_fingerprint: "fixture-implementation".into(),
                operation_fingerprint: "fixture-operation".into(),
            }]
            .into_boxed_slice(),
            retained_bytes: 1024,
        }),
        attribution,
        rows: vec![ActualWaveRow {
            request_id: ferrum_types::RequestId::new(),
            owner_incarnation: 4,
            work_generation: 9,
            input_index: 0,
            work: ActualRowWork::Decode { kv_tokens: 12 },
        }],
        canonical_rows: vec![CanonicalCostRow {
            work: ActualRowWork::Decode { kv_tokens: 12 },
            host_policy_signature: host_history_cost_signature([3; 32], 1),
            host_features: None,
            mask_upload_required: true,
            output: CostRowOutput::Decode {
                requires_full_logits: false,
                repetition_tokens: 0,
                repetition_penalty_bits: 1.0f32.to_bits(),
            },
        }],
        kind: ActualWaveKind::Decode,
        product: CostProductOutput::GreedyToken,
        graph_capability: DeviceCostGraphCaptureCapability::Supported,
        readback: CoreReadbackRoute::HostSynchronized,
        retries: 0,
        recurrent_state_bytes: 128,
        structured_capture: false,
        numeric_observation: DeviceCostObservationDemand::NotRequired,
        bounds: PendingWaveBounds {
            retained_bytes: 4096,
            maximum_resolved_bytes: 4096,
            retained_rows: 1,
        },
    }
}

#[test]
fn pending_graph_rejection_preserves_only_validated_actual_physical_work() {
    let original = projection(false);
    let projected = original.project_with_physical_evidence();
    assert_eq!(projected.shape, Err(ActualWaveEvidenceUnknown::GraphPath));
    assert_eq!(
        original.project(),
        Err(ActualWaveEvidenceUnknown::GraphPath)
    );
    let physical = projected.physical_evidence.unwrap();
    assert_eq!(physical.kind, ActualWaveKind::Decode);
    assert_eq!(physical.path, ActualWavePath::PlanRuntime);
    assert_eq!(physical.row_order, ActualWaveRowOrder::Ordered);
    assert_eq!(
        (
            physical.restore_bytes,
            physical.maintenance_bytes,
            physical.maintenance_units
        ),
        (0, 0, 0)
    );

    let mut retried = projection(false);
    retried.retries = 1;
    let projected = retried.project_with_physical_evidence();
    assert_eq!(projected.shape, Err(ActualWaveEvidenceUnknown::GraphPath));
    assert_eq!(
        projected.physical_evidence.unwrap().path,
        ActualWavePath::UnsupportedFallback
    );

    // A graph rejection must not mask a missing logical replay ledger, a wrong
    // wave role or invalid canonical row. Keep the original numerical reason.
    let replay_missing = projection(true);
    let mut wrong_kind = projection(false);
    wrong_kind.kind = ActualWaveKind::Prefill;
    let mut invalid_row = projection(false);
    invalid_row.canonical_rows[0].output = CostRowOutput::Prefill { final_logits: true };
    for invalid in [replay_missing, wrong_kind, invalid_row] {
        let projected = invalid.project_with_physical_evidence();
        assert_eq!(projected.shape, Err(ActualWaveEvidenceUnknown::GraphPath));
        assert!(projected.physical_evidence.is_none());
    }
    let mut missing_provider = projection(false);
    missing_provider.providers = Arc::new(ProviderIdentityTable {
        rows: Box::new([]),
        retained_bytes: 0,
    });
    let projected = missing_provider.project_with_physical_evidence();
    assert_eq!(
        projected.shape,
        Err(ActualWaveEvidenceUnknown::ProviderPath)
    );
    assert!(projected.physical_evidence.is_none());
}

#[test]
fn pending_eligible_graph_keeps_original_numeric_projection() {
    let mut original = projection(false);
    let state =
        DeviceCostGraphStreamState::new(DeviceCostGraphConfiguration::OnDemand, 35, 35, 0).unwrap();
    original.attribution =
        DeviceSubmissionAttribution::new(original.attribution.commands().to_vec())
            .unwrap()
            .with_graph_evidence(
                DeviceSubmissionGraphEvidence::new(state, state, false, 0, 0, 0, 0, 0).unwrap(),
            )
            .unwrap();
    let projected = original.project_with_physical_evidence();
    let shape = projected.shape.unwrap();
    assert_eq!(shape.graph, ActualWaveGraphState::ConfiguredEager);
    assert_eq!(shape.rows, original.rows);
    assert_eq!(shape.path, ActualWavePath::PlanRuntime);
    assert!(projected.physical_evidence.is_none());
    assert_eq!(original.project().unwrap(), shape);
}

#[test]
fn pending_replay_structure_graph_error_never_gets_physical_evidence() {
    use ferrum_interfaces::vnext::{
        DeviceReplayedLogicalCommandAttribution, DeviceReplayedSegmentAttribution,
        DeviceReusableExecutionProgramId, DeviceReusableExecutionSegment, PlanHash,
        PlanRuntimeCloseOutcome, PlanRuntimeResources, ReusableExecutionBucketSpec,
        ReusableExecutionCapacity, ReusableExecutionClassId,
    };
    // This typed ledger is valid device attribution, but the physical launch
    // plus its logical commands exceeds the existing canonical capacity. The
    // replay_segment error also maps to GraphPath, before graph classification.
    let count = u32::try_from(MAX_COST_COMMANDS).unwrap();
    let bucket = ReusableExecutionBucketSpec::new(
        ReusableExecutionClassId::new("pending-replay-test").unwrap(),
        ReusableExecutionCapacity::new(1, 1, 1).unwrap(),
    )
    .unwrap();
    let plan: PlanHash = serde_json::from_value(serde_json::json!("a".repeat(64))).unwrap();
    let fixture = contract::fixture();
    let lane = fixture.plan_resources.create_execution_lane().unwrap();
    let program = DeviceReusableExecutionProgramId::new(
        plan,
        "b".repeat(64),
        lane.id(),
        bucket.bucket_id().clone(),
        "c".repeat(64),
        "d".repeat(64),
        0,
        1,
        1,
        1,
    )
    .unwrap();
    let logical = (0..count)
        .map(|index| {
            DeviceReplayedLogicalCommandAttribution::new(
                index,
                index,
                DeviceNativeOperationId::new("fixture.native").unwrap(),
                DeviceBatchingForm::Packed,
                1,
                1,
                1,
                0,
                1,
            )
            .unwrap()
        })
        .collect();
    let segment = DeviceReplayedSegmentAttribution::new(
        0,
        program,
        DeviceReusableExecutionSegment::new(0, 0, count, count).unwrap(),
        "e".repeat(64),
        logical,
    )
    .unwrap();
    let physical = DeviceNativeWorkAttribution::new(
        0,
        Some(0),
        DeviceCommandPhase::Compute,
        DeviceNativeOperationId::new("fixture.native").unwrap(),
        DeviceExecutionPath::Replayed,
        DeviceBatchingForm::Packed,
        1,
        1,
        1,
        0,
        Some(u64::from(count)),
    )
    .unwrap();
    let mut original = projection(false);
    original.attribution =
        DeviceSubmissionAttribution::with_replayed_segments(vec![physical], vec![segment]).unwrap();
    // The failing stage is the canonical replay ledger, not an absent graph.
    assert!(matches!(
        route::actual_route_components(
            Some(&original.attribution),
            |index| original.providers.get(index),
            original.graph_capability,
            original.product,
            0,
            false,
            false,
        ),
        Err(ActualWaveEvidenceUnknown::GraphPath)
    ));
    let projected = original.project_with_physical_evidence();
    assert_eq!(projected.shape, Err(ActualWaveEvidenceUnknown::GraphPath));
    assert!(projected.physical_evidence.is_none());
    drop(lane);
    drop(fixture.registry);
    drop(fixture.impostor_registry);
    drop(fixture.runtime);
    assert!(matches!(
        PlanRuntimeResources::close(fixture.plan_resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
}
