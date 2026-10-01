use super::*;
use crate::execution_cost::{
    satisfied_completion_cost_signature, CostSamplingHistoryScope, HostContentDomainV1,
    HostCostFeaturesV1, HostCostPolicyV2, HostCostStateV1,
};
fn identity() -> ExecutorCostIdentity {
    ExecutorCostIdentity {
        schema_version: 1,
        model_weights: [1; 32],
        numerical_policy: [2; 32],
        device_runtime: [3; 32],
        execution_config: [4; 32],
    }
}
fn limits() -> CostWorkloadLimitsV1 {
    CostWorkloadLimitsV1 {
        maximum_rows: NonZeroU32::new(2).unwrap(),
        maximum_context_tokens: NonZeroU32::new(16).unwrap(),
        maximum_scheduled_tokens_per_wave: NonZeroU64::new(4).unwrap(),
        output_vocabulary_elements: NonZeroU64::new(32).unwrap(),
        repetition_slot_capacity: 8,
        fixed_state_bytes_per_row: 64,
    }
}
fn domain() -> CostWorkloadDomainV1 {
    CostWorkloadDomainV1::new_vnext(&identity(), limits()).unwrap()
}
fn host(generated: u64) -> HostCostFeaturesV1 {
    HostCostFeaturesV1 {
        policy: HostCostPolicyV2 {
            empirical_content_domain: Some(HostContentDomainV1::PlainTextGreedyV1),
            categorical_signature: [9; 32],
            decoder_text_bytes_per_token: 4,
            decoder_scratch_bytes_per_token: 8,
            raw_token_bytes_bound: 16,
        },
        state: HostCostStateV1 {
            generated_tokens_before: generated,
            maximum_output_tokens: 1_000_000,
            sampling_history_tokens: generated,
            sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
            pending_decoded_utf8: false,
            completion_state_signature: satisfied_completion_cost_signature(),
        },
    }
}
fn decode(kv_tokens: u32) -> CanonicalCostRow {
    CanonicalCostRow {
        work: ActualRowWork::Decode { kv_tokens },
        host_policy_signature: [8; 32],
        host_features: Some(host(2)),
        mask_upload_required: false,
        output: CostRowOutput::Decode {
            requires_full_logits: false,
            repetition_tokens: 0,
            repetition_penalty_bits: 1f32.to_bits(),
        },
    }
}
fn prefill(offset: u32, count: u32, total: u32) -> CanonicalCostRow {
    CanonicalCostRow {
        work: ActualRowWork::Prefill {
            offset,
            count,
            total_prompt_tokens: total,
        },
        host_policy_signature: [8; 32],
        host_features: Some(host(0)),
        mask_upload_required: false,
        output: CostRowOutput::Prefill {
            final_logits: offset + count == total,
        },
    }
}

#[test]
fn workload_domain_strict_wire_uses_checked_conversion_and_recomputes_digest() {
    let d = domain();
    let wire = serde_json::to_value(&d).unwrap();
    let restored: CostWorkloadDomainV1 = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(restored, d);
    for parent in ["", "limits", "identity"] {
        let mut bad = wire.clone();
        let object = if parent.is_empty() {
            &mut bad
        } else {
            &mut bad[parent]
        };
        object["unexpected"] = true.into();
        assert!(serde_json::from_value::<CostWorkloadDomainV1>(bad).is_err());
    }
    for name in [
        "maximum_rows",
        "maximum_context_tokens",
        "maximum_scheduled_tokens_per_wave",
        "output_vocabulary_elements",
    ] {
        let mut bad = wire.clone();
        bad["limits"][name] = 0.into();
        assert!(serde_json::from_value::<CostWorkloadDomainV1>(bad).is_err());
    }
    for key in ["schema_version", "identity"] {
        let mut bad = wire.clone();
        if key == "identity" {
            bad[key]["schema_version"] = 2.into();
        } else {
            bad[key] = 2.into();
        }
        assert!(serde_json::from_value::<CostWorkloadDomainV1>(bad).is_err());
    }
    let mut bad = wire;
    bad["sha256"] = "00".into();
    assert!(serde_json::from_value::<CostWorkloadDomainV1>(bad).is_err());
}
#[test]
fn workload_domain_digest_binds_every_capacity_and_existing_identity() {
    let original = domain();
    let wire = serde_json::to_value(&original).unwrap();
    for key in [
        "maximum_rows",
        "maximum_context_tokens",
        "maximum_scheduled_tokens_per_wave",
        "output_vocabulary_elements",
        "repetition_slot_capacity",
        "fixed_state_bytes_per_row",
    ] {
        let mut next = wire.clone();
        let value = next["limits"][key].as_u64().unwrap();
        next["limits"][key] = (value + 1).into();
        let next: CostWorkloadDomainV1 = serde_json::from_value(next).unwrap();
        assert_ne!(next.sha256(), original.sha256());
        assert!(next.matches_execution_identity(&identity()));
        assert!(!next.matches_runtime_domain(&original));
    }
    for key in [
        "model_weights",
        "numerical_policy",
        "device_runtime",
        "execution_config",
    ] {
        let mut next = wire.clone();
        next["identity"][key][0] = 19.into();
        let next: CostWorkloadDomainV1 = serde_json::from_value(next).unwrap();
        assert_ne!(next.sha256(), original.sha256());
        assert!(!next.matches_execution_identity(&identity()));
    }
}
#[test]
fn workload_domain_counts_real_mixed_units_without_restricting_user_output_limit() {
    let d = domain();
    assert!(d.validate_workload_rows(&[decode(15)], 64).is_ok());
    assert!(d
        .validate_workload_rows(&[prefill(0, 3, 10), decode(8)], 128)
        .is_ok());
    assert_eq!(
        d.validate_workload_rows(&[prefill(0, 4, 10), decode(8)], 128),
        Err(CostWorkloadDomainError::ScheduledTokenCapacity)
    );
    assert_eq!(
        d.validate_workload_rows(&[decode(16)], 64),
        Err(CostWorkloadDomainError::ContextCapacity)
    );
    assert_eq!(
        d.validate_workload_rows(&[prefill(14, 2, 17)], 64),
        Err(CostWorkloadDomainError::ContextCapacity)
    );
    assert_eq!(
        d.validate_workload_rows(&[decode(1); 3], 192),
        Err(CostWorkloadDomainError::RowCapacity)
    );
    assert_eq!(
        d.validate_workload_rows(&[decode(8)], 63),
        Err(CostWorkloadDomainError::RecurrentStateMismatch)
    );
}
#[test]
fn workload_domain_preserves_host_history_repetition_and_decoder_bounds() {
    let d = domain();
    let mut r = decode(8);
    r.output = CostRowOutput::Decode {
        requires_full_logits: false,
        repetition_tokens: 9,
        repetition_penalty_bits: 1.25f32.to_bits(),
    };
    assert_eq!(
        d.validate_workload_rows(&[r], 64),
        Err(CostWorkloadDomainError::RepetitionCapacity)
    );
    r = decode(8);
    r.host_features
        .as_mut()
        .unwrap()
        .state
        .sampling_history_tokens = 1;
    assert_eq!(
        d.validate_workload_rows(&[r], 64),
        Err(CostWorkloadDomainError::UnsupportedHostPolicy)
    );
    r = decode(8);
    r.host_features
        .as_mut()
        .unwrap()
        .policy
        .decoder_text_bytes_per_token = u64::MAX;
    assert_eq!(
        d.validate_workload_rows(&[r], 64),
        Err(CostWorkloadDomainError::InvalidHostWork)
    );
    r = decode(8);
    r.host_features = None;
    assert_eq!(
        d.validate_workload_rows(&[r], 64),
        Err(CostWorkloadDomainError::UnsupportedHostPolicy)
    );
    let mut r = prefill(0, 2, 3);
    r.output = CostRowOutput::Prefill { final_logits: true };
    assert_eq!(
        d.validate_workload_rows(&[r], 64),
        Err(CostWorkloadDomainError::InvalidRow)
    );
}

// Use the original producer builders for both projections. No fabricated host
// state is reconstructed from numeric fields by the new domain validator.
fn projected(rows: &[CanonicalCostRow]) -> (CanonicalWaveCostShape, Vec<StructuredHostRowV1>) {
    use crate::execution_cost::{
        ActualWaveGraphState, ActualWaveKind, ActualWavePath, ActualWaveRowOrder,
        CanonicalWaveCostBuilder, CoreReadbackRoute, CostCommandPath, CostPhysicalCommand,
        CostProductOutput, CostProviderIdentity, KernelNumericWorkV1, SelectedAlgorithmClassV1,
        SelectedCommandCostBuilderV1,
    };
    use crate::vnext::DeviceCommandPhase;
    let tokens = rows
        .iter()
        .map(|r| match r.work {
            ActualRowWork::Decode { .. } => 1u64,
            ActualRowWork::Prefill { count, .. } => u64::from(count),
            _ => panic!("fixture only supplies ordinary typed rows"),
        })
        .sum::<u64>();
    let mut evidence = SelectedCommandCostBuilderV1::new(tokens);
    evidence
        .kernel(
            SelectedAlgorithmClassV1::new("domain.fixture.kernel", 1, [1; 32], [2; 32]).unwrap(),
            KernelNumericWorkV1 {
                logical_units: tokens,
                padded_units: tokens,
                inner_units_per_logical_unit: 64,
                grid: [u32::try_from(tokens).unwrap(), 1, 1],
                scratch_bytes: 0,
                staged_weight_bytes: 0,
            },
        )
        .unwrap();
    let evidence = evidence.finish().unwrap();
    let mut builder =
        CanonicalWaveCostBuilder::new_with_structured_statistics(0, CostProductOutput::GreedyToken);
    builder
        .physical_command(CostPhysicalCommand {
            native_op_id: "domain.fixture.native",
            command_index: 0,
            node_index: Some(0),
            command_phase: DeviceCommandPhase::Compute,
            provider: Some(CostProviderIdentity {
                provider_id: "domain.fixture.provider",
                implementation_fingerprint: "fixture.impl",
                operation_fingerprint: "fixture.op",
            }),
            path: CostCommandPath::Eager,
            participant_start: 0,
            participant_count: rows.len() as u32,
            token_count: tokens,
            batching_form: "packed",
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            reusable_graph_node_count: None,
            statistical_evidence: Some(&evidence),
        })
        .unwrap();
    builder
        .core_readback_route(CoreReadbackRoute::SubmissionStaged)
        .unwrap();
    for row in rows {
        builder.row(*row).unwrap();
    }
    let decode = rows
        .iter()
        .any(|r| matches!(r.work, ActualRowWork::Decode { .. }));
    let prefill = rows
        .iter()
        .any(|r| matches!(r.work, ActualRowWork::Prefill { .. }));
    let kind = match (decode, prefill) {
        (true, true) => ActualWaveKind::Mixed,
        (true, false) => ActualWaveKind::Decode,
        (false, true) => ActualWaveKind::Prefill,
        _ => panic!("nonempty fixture"),
    };
    let built = builder
        .finish_with_structure(
            kind,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::ConfiguredEager,
            ActualWaveRowOrder::Ordered,
            rows.len() as u64 * 64,
        )
        .unwrap();
    let host = built.structured.unwrap().physical_host_rows().to_vec();
    (built.exact, host)
}

#[test]
fn workload_domain_projected_matches_original_limits_and_preserves_large_output_limit() {
    let d = domain();
    for rows in [
        vec![decode(15)],
        vec![prefill(0, 3, 10), decode(8)],
        vec![prefill(0, 4, 10), decode(8)],
        vec![decode(16)],
        vec![prefill(14, 2, 17)],
        vec![decode(1); 3],
        vec![prefill(0, 3, 3)],
    ] {
        let (shape, host) = projected(&rows);
        assert_eq!(
            d.validate_projected_workload(&shape, &host),
            d.validate_workload_rows(&rows, shape.recurrent_state_bytes)
        );
    }
    let (mut shape, host) = projected(&[decode(15)]);
    assert!(
        shape.numeric_features.as_ref().unwrap().rows[0].maximum_output_tokens
            > u64::from(d.limits().maximum_context_tokens.get())
    );
    assert!(d.validate_projected_workload(&shape, &host).is_ok());
    shape.recurrent_state_bytes -= 1;
    assert_eq!(
        d.validate_projected_workload(&shape, &host),
        Err(CostWorkloadDomainError::RecurrentStateMismatch)
    );
}

#[test]
fn workload_domain_projected_checks_installed_policy_repetition_and_host_prefix() {
    let mut row = decode(15);
    row.host_features = Some(host(12));
    row.output = CostRowOutput::Decode {
        requires_full_logits: false,
        repetition_tokens: 9,
        repetition_penalty_bits: 1.1f32.to_bits(),
    };
    let (shape, physical) = projected(&[row]);
    assert_eq!(
        domain().validate_projected_workload(&shape, &physical),
        Err(CostWorkloadDomainError::RepetitionCapacity)
    );
    let mut l = limits();
    l.repetition_slot_capacity = 32;
    l.output_vocabulary_elements = NonZeroU64::new(8).unwrap();
    assert_eq!(
        CostWorkloadDomainV1::new_vnext(&identity(), l)
            .unwrap()
            .validate_projected_workload(&shape, &physical),
        Err(CostWorkloadDomainError::RepetitionCapacity)
    );
    let mut row = decode(15);
    row.host_features = Some(host(16));
    let (shape, physical) = projected(&[row]);
    assert_eq!(
        domain().validate_projected_workload(&shape, &physical),
        Err(CostWorkloadDomainError::ContextCapacity)
    );
    let (shape, mut physical) = projected(&[decode(15)]);
    physical[0].installed_policy.empirical_content_domain = None;
    assert_eq!(
        domain().validate_projected_workload(&shape, &physical),
        Err(CostWorkloadDomainError::UnsupportedHostPolicy)
    );
    let mut installed = decode(15);
    installed
        .host_features
        .as_mut()
        .unwrap()
        .policy
        .empirical_content_domain = Some(HostContentDomainV1::PlainTextInstalledV2(
        crate::execution_cost::PlainTextPolicyCapabilityV2 {
            sampling: crate::execution_cost::PlainTextSamplingRouteV2::Greedy {
                repetition_penalty: false,
            },
            model_eos: true,
            user_stop: true,
        },
    ));
    let (shape, mut physical) = projected(&[installed]);
    assert!(domain()
        .validate_projected_workload(&shape, &physical)
        .is_ok());
    physical[0].repetition_penalty_bits = Some(f32::NAN.to_bits());
    assert_eq!(
        domain().validate_projected_workload(&shape, &physical),
        Err(CostWorkloadDomainError::InvalidRow)
    );
}

#[test]
fn workload_domain_projected_requires_exact_physical_alignment_and_emission() {
    let d = domain();
    let (shape, physical) = projected(&[prefill(0, 3, 10), decode(8)]);
    let check = |s: &CanonicalWaveCostShape, h: &[StructuredHostRowV1]| {
        assert!(d.validate_projected_workload(s, h).is_err());
    };
    check(&shape, &physical[..1]);
    let mut s = shape.clone();
    s.numeric_features = None;
    check(&s, &physical);
    let mut s = shape.clone();
    s.numeric_features.as_mut().unwrap().rows.pop();
    check(&s, &physical);
    let mut s = shape.clone();
    s.numeric_features.as_mut().unwrap().schema_version += 1;
    check(&s, &physical);
    let mut h = physical.clone();
    h.swap(0, 1);
    check(&shape, &h);
    let mut h = physical.clone();
    h[0].physical_position = 1;
    check(&shape, &h);
    let mut h = physical.clone();
    h[0].role = HostRowRoleV2::Decode;
    check(&shape, &h);
    let mut h = physical.clone();
    h[0].initial_prefill = false;
    check(&shape, &h);
    let mut h = physical.clone();
    h[0].final_prefill = true;
    check(&shape, &h);
    let mut h = physical.clone();
    h[0].terminal_expectation = HostTerminalExpectationV1::TokenMayTerminate;
    check(&shape, &h);
    let mut h = physical.clone();
    h[1].no_generated_history = true;
    check(&shape, &h);
    let mut h = physical.clone();
    h[1].decode_requires_full_logits = None;
    check(&shape, &h);
    let mut h = physical.clone();
    h[1].installed_policy.decoder_scratch_bytes_per_token += 1;
    check(&shape, &h);
    let mut s = shape.clone();
    s.numeric_features.as_mut().unwrap().rows[1].sampling_history_tokens = 1;
    check(&s, &physical);
    // This altered prefix remains a valid stand-alone numeric row but cannot
    // represent the decode emission bound to the original physical host row.
    let mut s = shape.clone();
    let n = &mut s.numeric_features.as_mut().unwrap().rows[1];
    n.decoded_prefix_tokens = 2;
    n.decoded_text_bytes_bound = 8;
    n.decode_scratch_bytes_bound = 16;
    n.validate().unwrap();
    check(&s, &physical);
}

#[test]
fn workload_domain_projected_final_prefill_length_boundary_and_checked_capacity() {
    let mut row = prefill(0, 3, 3);
    row.host_features
        .as_mut()
        .unwrap()
        .state
        .maximum_output_tokens = 1;
    let (shape, mut physical) = projected(&[row]);
    assert_eq!(
        physical[0].terminal_expectation,
        HostTerminalExpectationV1::LengthBoundary
    );
    assert!(domain()
        .validate_projected_workload(&shape, &physical)
        .is_ok());
    physical[0].terminal_expectation = HostTerminalExpectationV1::TokenMayTerminate;
    assert_eq!(
        domain().validate_projected_workload(&shape, &physical),
        Err(CostWorkloadDomainError::InvalidHostWork)
    );
    let (mut shape, physical) = projected(&[decode(8), decode(8)]);
    let mut l = limits();
    l.fixed_state_bytes_per_row = u64::MAX;
    let d = CostWorkloadDomainV1::new_vnext(&identity(), l).unwrap();
    shape.recurrent_state_bytes = u64::MAX;
    assert_eq!(
        d.validate_projected_workload(&shape, &physical),
        Err(CostWorkloadDomainError::ArithmeticOverflow)
    );
}
