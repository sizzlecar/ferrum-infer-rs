use super::*;
use crate::execution_cost::*;
use crate::vnext::DeviceCommandPhase;

mod row_multiset;

fn host(generated: u64, maximum: u64) -> HostCostFeaturesV1 {
    HostCostFeaturesV1 {
        policy: HostCostPolicyV2 {
            empirical_content_domain: None,
            categorical_signature: [8; 32],
            decoder_text_bytes_per_token: 12,
            decoder_scratch_bytes_per_token: 4,
            raw_token_bytes_bound: 4,
        },
        state: HostCostStateV1 {
            generated_tokens_before: generated,
            maximum_output_tokens: maximum,
            sampling_history_tokens: generated,
            sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
            pending_decoded_utf8: false,
            completion_state_signature: satisfied_completion_cost_signature(),
        },
    }
}
fn row(host: Option<HostCostFeaturesV1>) -> CanonicalCostRow {
    CanonicalCostRow {
        work: ActualRowWork::Decode { kv_tokens: 80 },
        host_policy_signature: host_history_cost_signature(
            [2; 32],
            host.map_or(3, |h| h.state.generated_tokens_before),
        ),
        host_features: host,
        mask_upload_required: false,
        output: CostRowOutput::Decode {
            requires_full_logits: false,
            repetition_tokens: 0,
            repetition_penalty_bits: 1f32.to_bits(),
        },
    }
}
fn shape(rows: &[CanonicalCostRow], route: Option<CoreReadbackRoute>) -> CanonicalWaveCostShape {
    shape_product(rows, route, CostProductOutput::GreedyToken)
}
fn shape_product(
    rows: &[CanonicalCostRow],
    route: Option<CoreReadbackRoute>,
    product: CostProductOutput,
) -> CanonicalWaveCostShape {
    let mut builder = CanonicalWaveCostBuilder::new(0, product);
    builder
        .physical_command(CostPhysicalCommand {
            statistical_evidence: None,
            native_op_id: "fixture.operation",
            command_index: 0,
            node_index: Some(0),
            command_phase: DeviceCommandPhase::Compute,
            provider: Some(CostProviderIdentity {
                provider_id: "fixture",
                implementation_fingerprint: "impl.v1",
                operation_fingerprint: "op.v1",
            }),
            path: CostCommandPath::Eager,
            participant_start: 0,
            participant_count: rows.len() as u32,
            token_count: rows.len() as u64,
            batching_form: "packed",
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            reusable_graph_node_count: None,
        })
        .unwrap();
    if let Some(route) = route {
        builder.core_readback_route(route).unwrap();
    }
    for row in rows {
        builder.row(*row).unwrap();
    }
    let kind = if rows
        .iter()
        .any(|r| matches!(r.work, ActualRowWork::Prefill { .. }))
    {
        ActualWaveKind::Mixed
    } else {
        ActualWaveKind::Decode
    };
    builder
        .finish(
            kind,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Disabled,
            ActualWaveRowOrder::Ordered,
            0,
        )
        .unwrap()
}

#[test]
fn numeric_projection_scales_full_history_and_preserves_exact_v1() {
    let baseline = shape(
        &[row(Some(host(3, 20)))],
        Some(CoreReadbackRoute::SubmissionStaged),
    );
    let exact = shape(&[row(None)], Some(CoreReadbackRoute::SubmissionStaged));
    assert_eq!(baseline.provider_signature, exact.provider_signature);
    assert_eq!(
        baseline.output_policy_signature,
        exact.output_policy_signature
    );
    assert!(exact.numeric_features.is_none());
    let longer = shape(
        &[row(Some(host(7, 40)))],
        Some(CoreReadbackRoute::SubmissionStaged),
    );
    assert_ne!(
        baseline.output_policy_signature,
        longer.output_policy_signature
    );
    let before = baseline.numeric_features.unwrap();
    let after = longer.numeric_features.unwrap();
    assert_eq!(
        before.output_policy_signature,
        after.output_policy_signature
    );
    assert_eq!(before.rows[0].decoded_prefix_tokens, 4);
    assert_eq!(after.rows[0].decoded_text_bytes_bound, 96);
    assert_eq!(after.rows[0].decode_scratch_bytes_bound, 32);
    assert_ne!(
        before.rows[0].maximum_output_tokens,
        after.rows[0].maximum_output_tokens
    );
}

#[test]
fn empirical_host_content_marginalizes_bytes_without_rewriting_exact_keys() {
    let mut clean = host(3, 20);
    let old = shape(
        &[row(Some(clean))],
        Some(CoreReadbackRoute::SubmissionStaged),
    );
    clean.policy.empirical_content_domain = Some(HostContentDomainV1::PlainTextGreedyV1);
    let with_domain = shape(
        &[row(Some(clean))],
        Some(CoreReadbackRoute::SubmissionStaged),
    );
    assert_eq!(
        old.output_policy_signature,
        with_domain.output_policy_signature
    );
    assert_eq!(old.numeric_features, with_domain.numeric_features);
    assert!(old.host_content_features.is_none());
    assert!(with_domain.host_content_features.is_some());

    let mut pending = clean;
    pending.state.pending_decoded_utf8 = true;
    let clean_row = row(Some(clean));
    let mut pending_row = row(Some(pending));
    pending_row.output = CostRowOutput::Decode {
        requires_full_logits: true,
        repetition_tokens: 0,
        repetition_penalty_bits: 1f32.to_bits(),
    };
    let a = shape_product(
        &[clean_row],
        Some(CoreReadbackRoute::SubmissionStaged),
        CostProductOutput::FullLogits,
    );
    let b = shape_product(
        &[pending_row],
        Some(CoreReadbackRoute::SubmissionStaged),
        CostProductOutput::FullLogits,
    );
    assert_ne!(a.numeric_features, b.numeric_features);
    assert_ne!(a.output_policy_signature, b.output_policy_signature);
    assert_eq!(a.host_content_features, b.host_content_features);
    // Full-logits CPU work and greedy-token host work never share a bucket.
    assert_ne!(a.host_content_features, with_domain.host_content_features);
}

#[test]
fn empirical_host_content_retains_readback_terminal_and_upload_branches() {
    let mut admitted = host(3, 20);
    admitted.policy.empirical_content_domain = Some(HostContentDomainV1::PlainTextGreedyV1);
    let baseline = shape(
        &[row(Some(admitted))],
        Some(CoreReadbackRoute::SubmissionStaged),
    );
    let synchronized = shape(
        &[row(Some(admitted))],
        Some(CoreReadbackRoute::HostSynchronized),
    );
    assert_ne!(
        baseline.host_content_features,
        synchronized.host_content_features
    );
    let mut terminal = admitted;
    terminal.state.maximum_output_tokens = 4;
    assert_ne!(
        baseline.host_content_features,
        shape(
            &[row(Some(terminal))],
            Some(CoreReadbackRoute::SubmissionStaged)
        )
        .host_content_features
    );
    let mut upload = row(Some(admitted));
    upload.mask_upload_required = true;
    assert_ne!(
        baseline.host_content_features,
        shape(&[upload], Some(CoreReadbackRoute::SubmissionStaged)).host_content_features
    );
    let mut other_policy = admitted;
    other_policy.policy.categorical_signature = [9; 32];
    assert_ne!(
        baseline.host_content_features,
        shape(
            &[row(Some(other_policy))],
            Some(CoreReadbackRoute::SubmissionStaged)
        )
        .host_content_features
    );
}

#[test]
fn empirical_host_content_requires_every_rows_installed_domain_and_known_readback() {
    let mut admitted = host(3, 20);
    admitted.policy.empirical_content_domain = Some(HostContentDomainV1::PlainTextGreedyV1);
    assert!(shape(
        &[row(Some(admitted)), row(Some(host(3, 20)))],
        Some(CoreReadbackRoute::SubmissionStaged)
    )
    .host_content_features
    .is_none());
    assert!(shape(&[row(Some(admitted))], None)
        .host_content_features
        .is_none());
    admitted.state.completion_state_signature = [17; 32];
    assert!(shape(
        &[row(Some(admitted))],
        Some(CoreReadbackRoute::SubmissionStaged)
    )
    .host_content_features
    .is_none());
    let invalid = HostContentCostFeaturesV1 {
        schema_version: 2,
        output_policy_signature: [0; 32],
    };
    assert_eq!(invalid.validate(), Err(CostFeatureError::UnsupportedSchema));
}

#[test]
fn true_branch_changes_remain_categorical() {
    let original = row(Some(host(3, 20)));
    let baseline = shape(&[original], Some(CoreReadbackRoute::SubmissionStaged))
        .numeric_features
        .unwrap()
        .output_policy_signature;
    let mut pending = original;
    pending
        .host_features
        .as_mut()
        .unwrap()
        .state
        .pending_decoded_utf8 = true;
    let mut completion = original;
    completion
        .host_features
        .as_mut()
        .unwrap()
        .state
        .completion_state_signature = [4; 32];
    let mut codec = original;
    codec
        .host_features
        .as_mut()
        .unwrap()
        .policy
        .categorical_signature = [9; 32];
    let mut mask = original;
    mask.mask_upload_required = true;
    let mut full = original;
    full.output = CostRowOutput::Decode {
        requires_full_logits: true,
        repetition_tokens: 0,
        repetition_penalty_bits: 1f32.to_bits(),
    };
    for changed in [
        pending,
        completion,
        codec,
        mask,
        full,
        row(Some(host(3, 4))),
    ] {
        assert_ne!(
            baseline,
            shape(&[changed], Some(CoreReadbackRoute::SubmissionStaged))
                .numeric_features
                .unwrap()
                .output_policy_signature
        );
    }
}

#[test]
fn readback_fallback_is_distinct_even_without_different_gpu_commands() {
    let rows = [row(Some(host(3, 20)))];
    let staged = shape(&rows, Some(CoreReadbackRoute::SubmissionStaged));
    let synchronized = shape(&rows, Some(CoreReadbackRoute::HostSynchronized));
    assert_eq!(staged.provider_signature, synchronized.provider_signature);
    assert_eq!(
        staged.output_policy_signature,
        synchronized.output_policy_signature
    );
    let staged_hash = staged.numeric_features.unwrap().output_policy_signature;
    let synchronized_hash = synchronized
        .numeric_features
        .unwrap()
        .output_policy_signature;
    let fallback = shape(
        &rows,
        Some(CoreReadbackRoute::SubmissionFallbackSynchronized),
    );
    assert_eq!(fallback.provider_signature, synchronized.provider_signature);
    assert_eq!(
        fallback.output_policy_signature,
        synchronized.output_policy_signature
    );
    let fallback_hash = fallback.numeric_features.unwrap().output_policy_signature;
    assert_ne!(staged_hash, synchronized_hash);
    assert_ne!(staged_hash, fallback_hash);
    assert_ne!(synchronized_hash, fallback_hash);
    assert!(shape(&rows, None).numeric_features.is_none());
    assert!(shape(&rows, Some(CoreReadbackRoute::Unknown))
        .numeric_features
        .is_none());
}

#[test]
fn ordered_mixed_roles_do_not_merge_and_partial_features_stay_exact_only() {
    let decode = row(Some(host(3, 20)));
    let mut prefill = row(Some(host(0, 20)));
    prefill.work = ActualRowWork::Prefill {
        offset: 0,
        count: 4,
        total_prompt_tokens: 4,
    };
    prefill.output = CostRowOutput::Prefill { final_logits: true };
    let route = Some(CoreReadbackRoute::SubmissionStaged);
    let first = shape(&[decode, prefill], route).numeric_features.unwrap();
    let second = shape(&[prefill, decode], route).numeric_features.unwrap();
    assert_ne!(
        first.output_policy_signature,
        second.output_policy_signature
    );
    assert_eq!(first.rows[1].decoded_prefix_tokens, 1);
    assert!(shape(&[decode, row(None)], route)
        .numeric_features
        .is_none());
}

#[test]
fn invalid_history_limits_repetition_and_overflow_are_not_zero_work() {
    let work = ActualRowWork::Decode { kv_tokens: 8 };
    let output = row(None).output;
    let mut invalid = host(3, 3);
    assert!(project_host_cost_features(invalid, work, output).is_err());
    invalid = host(3, 20);
    invalid.state.sampling_history_tokens = 2;
    assert!(project_host_cost_features(invalid, work, output).is_err());
    invalid = host(3, 20);
    invalid.policy.decoder_text_bytes_per_token = u64::MAX;
    assert_eq!(
        project_host_cost_features(invalid, work, output),
        Err(CostFeatureError::Overflow)
    );
    let repetition = CostRowOutput::Decode {
        requires_full_logits: false,
        repetition_tokens: 4,
        repetition_penalty_bits: 1.2f32.to_bits(),
    };
    assert!(project_host_cost_features(host(3, 20), work, repetition).is_err());
    assert_eq!(
        project_host_cost_features(host(u64::MAX, u64::MAX), work, output),
        Err(CostFeatureError::Overflow)
    );
}

#[test]
fn wire_rows_are_strict_bounded_and_version_checked() {
    let features = shape(
        &[row(Some(host(3, 20)))],
        Some(CoreReadbackRoute::NoReadback),
    )
    .numeric_features
    .unwrap();
    let mut json = serde_json::to_value(&features).unwrap();
    json["rows"][0]["invented"] = true.into();
    assert!(serde_json::from_value::<CanonicalWaveCostFeatures>(json).is_err());
    let mut json = serde_json::to_value(&features).unwrap();
    json["rows"] = serde_json::to_value(vec![features.rows[0]; MAX_COST_ROWS + 1]).unwrap();
    assert!(serde_json::from_value::<CanonicalWaveCostFeatures>(json).is_err());
    let mut unsupported = features.clone();
    unsupported.schema_version += 1;
    assert_eq!(
        unsupported.validate(1),
        Err(CostFeatureError::UnsupportedSchema)
    );
    assert_eq!(features.validate(2), Err(CostFeatureError::RowCapacity));
    let mut oversized = features;
    oversized.rows.reserve(MAX_COST_ROWS * 2);
    assert_eq!(oversized.validate(1), Err(CostFeatureError::RowCapacity));
}
