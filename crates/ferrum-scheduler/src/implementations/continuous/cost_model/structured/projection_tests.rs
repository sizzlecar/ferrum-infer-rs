//! Typed producer-to-model contract checks. These do not use GPU timings and
//! must not be represented as heldout quality/performance qualification.
use super::*;
use ferrum_interfaces::{execution_cost::*, vnext::DeviceCommandPhase};

fn wave(terminal: usize, capture: bool, work_a: u64, algorithm: &str) -> CanonicalStructuredWave {
    let mut command = if capture {
        SelectedCommandCostBuilderV1::new_with_algorithm_work(2)
    } else {
        SelectedCommandCostBuilderV1::new(2)
    };
    for (entry, work) in [(algorithm, work_a), ("fixture.second", 32 - work_a)] {
        command
            .kernel(
                SelectedAlgorithmClassV1::new(entry, 1, [1; 32], [2; 32]).unwrap(),
                KernelNumericWorkV1 {
                    logical_units: work,
                    padded_units: work,
                    inner_units_per_logical_unit: 2,
                    grid: [1, 1, 1],
                    scratch_bytes: 64,
                    staged_weight_bytes: 0,
                },
            )
            .unwrap();
    }
    let selected = command.finish().unwrap();
    let mut b =
        CanonicalWaveCostBuilder::new_with_structured_statistics(0, CostProductOutput::FullLogits);
    b.physical_command(CostPhysicalCommand {
        native_op_id: "fixture.structured",
        command_index: 0,
        node_index: Some(0),
        command_phase: DeviceCommandPhase::Compute,
        provider: Some(CostProviderIdentity {
            provider_id: "fixture.provider",
            implementation_fingerprint: "impl-v1",
            operation_fingerprint: "op-v1",
        }),
        path: CostCommandPath::Eager,
        participant_start: 0,
        participant_count: 2,
        token_count: 2,
        batching_form: "packed",
        compute_dispatch_count: 2,
        transfer_command_count: 0,
        reusable_graph_node_count: None,
        statistical_evidence: Some(&selected),
    })
    .unwrap();
    b.core_readback_route(CoreReadbackRoute::HostSynchronized)
        .unwrap();
    for position in 0..2 {
        b.row(CanonicalCostRow {
            work: ActualRowWork::Decode { kv_tokens: 64 },
            output: CostRowOutput::Decode {
                requires_full_logits: true,
                repetition_tokens: 2,
                repetition_penalty_bits: 1f32.to_bits(),
            },
            host_policy_signature: [3; 32],
            mask_upload_required: false,
            host_features: Some(HostCostFeaturesV1 {
                policy: HostCostPolicyV2 {
                    empirical_content_domain: Some(HostContentDomainV1::PlainTextGreedyV1),
                    categorical_signature: [4; 32],
                    decoder_text_bytes_per_token: 4,
                    decoder_scratch_bytes_per_token: 8,
                    raw_token_bytes_bound: 4,
                },
                state: HostCostStateV1 {
                    generated_tokens_before: 2,
                    maximum_output_tokens: if position == terminal { 3 } else { 20 },
                    sampling_history_tokens: 2,
                    sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
                    pending_decoded_utf8: false,
                    completion_state_signature: satisfied_completion_cost_signature(),
                },
            }),
        })
        .unwrap();
    }
    let wave = b
        .finish_with_captured_structure(
            ActualWaveKind::Decode,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Disabled,
            ActualWaveRowOrder::Ordered,
            64,
        )
        .unwrap();
    let structured = wave
        .statistical
        .as_ref()
        .unwrap()
        .structured_capture()
        .unwrap()
        .map(|recipe| recipe.as_ref().clone());
    CanonicalStructuredWave {
        exact: wave.exact,
        statistical: wave.statistical,
        structured,
    }
}
fn project(w: &CanonicalStructuredWave) -> Result<StructuredInputV1> {
    StructuredInputV1::from_future(
        &w.exact,
        w.statistical.as_ref().unwrap(),
        w.structured.as_ref().unwrap(),
    )
}
#[test]
fn checked_projection_shares_domain_but_preserves_terminal_position_and_exact_binding() {
    let a = wave(0, true, 8, "fixture.first");
    let b = wave(1, true, 8, "fixture.first");
    assert_ne!(a.exact, b.exact);
    let ia = project(&a).unwrap();
    let ib = project(&b).unwrap();
    assert_eq!(ia.domain, ib.domain);
    assert_ne!(ia.basis, ib.basis);
    assert_ne!(ia.physical_host_rows, ib.physical_host_rows);
    assert!(ia.validate(&StructuredSettingsV1::default()).is_ok());
    assert!(ib.validate(&StructuredSettingsV1::default()).is_ok());
    assert!(StructuredInputV1::from_future(
        &b.exact,
        b.statistical.as_ref().unwrap(),
        a.structured.as_ref().unwrap()
    )
    .is_err());
}
#[test]
fn checked_projection_retains_algorithm_allocation_not_just_aggregate_work() {
    let a = wave(0, true, 8, "fixture.first");
    let b = wave(0, true, 24, "fixture.first");
    assert_eq!(
        a.structured.as_ref().unwrap().device().aggregate_work(),
        b.structured.as_ref().unwrap().device().aggregate_work()
    );
    let ia = project(&a).unwrap();
    let ib = project(&b).unwrap();
    assert_eq!(ia.domain, ib.domain);
    assert_ne!(ia.basis, ib.basis);
    assert_ne!(ia.support, ib.support);
}
#[test]
fn missing_per_algorithm_producer_or_changed_route_is_never_upgraded() {
    let missing = wave(0, false, 8, "fixture.first");
    assert!(matches!(
        project(&missing),
        Err(StructuredUnknown::MissingEvidence)
    ));
    let original = project(&wave(0, true, 8, "fixture.first")).unwrap();
    let changed = project(&wave(0, true, 8, "fixture.different")).unwrap();
    assert!(matches!(
        original.same_domain(&changed),
        Err(StructuredUnknown::WrongDomain)
    ));
}

#[test]
fn equal_legacy_identity_cannot_exchange_different_algorithm_assignments() {
    let a = wave(0, true, 8, "fixture.first");
    let b = wave(0, true, 24, "fixture.first");
    assert_eq!(a.exact, b.exact);
    assert_eq!(a.statistical, b.statistical);
    let a_recipe = a.structured.as_ref().unwrap();
    let b_recipe = b.structured.as_ref().unwrap();
    assert_eq!(
        a_recipe.device().aggregate_work(),
        b_recipe.device().aggregate_work()
    );
    assert_ne!(a_recipe, b_recipe);
    assert!(matches!(
        StructuredInputV1::from_future(&a.exact, a.statistical.as_ref().unwrap(), b_recipe),
        Err(StructuredUnknown::MissingEvidence)
    ));
    assert!(matches!(
        StructuredInputV1::from_future(&b.exact, b.statistical.as_ref().unwrap(), a_recipe),
        Err(StructuredUnknown::MissingEvidence)
    ));
    assert!(project(&a).is_ok());
    assert!(project(&b).is_ok());
}
