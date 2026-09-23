use super::*;

fn admitted(generated: u64, maximum: u64) -> CanonicalCostRow {
    let mut features = host(generated, maximum);
    features.policy.empirical_content_domain = Some(HostContentDomainV1::PlainTextGreedyV1);
    row(Some(features))
}

#[test]
fn row_multiset_evidence_follows_physical_rows_without_changing_legacy_hashes() {
    let first = admitted(3, 20);
    let mut terminal = admitted(7, 8);
    terminal.work = ActualRowWork::Decode { kv_tokens: 160 };
    terminal.mask_upload_required = true;
    let a = shape(
        &[first, terminal],
        Some(CoreReadbackRoute::SubmissionStaged),
    );
    let b = shape(
        &[terminal, first],
        Some(CoreReadbackRoute::SubmissionStaged),
    );
    assert_eq!(a.provider_signature, b.provider_signature);
    assert_ne!(a.output_policy_signature, b.output_policy_signature);
    assert_ne!(a.host_content_features, b.host_content_features);
    let a_rows = a.row_multiset_features.as_ref().unwrap();
    let b_rows = b.row_multiset_features.as_ref().unwrap();
    assert_eq!(a_rows.wave_policy_signature, b_rows.wave_policy_signature);
    assert_ne!(a_rows.rows[0], a_rows.rows[1]);
    assert_eq!(a_rows.rows[0], b_rows.rows[1]);
    assert_eq!(a_rows.rows[1], b_rows.rows[0]);
    assert_eq!(a.rows, vec![first.work, terminal.work]);
    assert_eq!(b.rows, vec![terminal.work, first.work]);

    let mut legacy = first;
    legacy
        .host_features
        .as_mut()
        .unwrap()
        .policy
        .empirical_content_domain = None;
    let legacy_shape = shape(&[legacy], Some(CoreReadbackRoute::SubmissionStaged));
    let new_shape = shape(&[first], Some(CoreReadbackRoute::SubmissionStaged));
    assert_eq!(
        legacy_shape.provider_signature,
        new_shape.provider_signature
    );
    assert_eq!(
        legacy_shape.output_policy_signature,
        new_shape.output_policy_signature
    );
    assert_eq!(legacy_shape.numeric_features, new_shape.numeric_features);
    assert!(legacy_shape.row_multiset_features.is_none());
}

#[test]
fn row_multiset_categories_preserve_execution_branches_and_actual_product() {
    let base = admitted(3, 20);
    let baseline = shape(&[base], Some(CoreReadbackRoute::SubmissionStaged));
    let reference = baseline.row_multiset_features.unwrap();
    let mut upload = base;
    upload.mask_upload_required = true;
    let terminal = admitted(3, 4);
    let mut policy = base;
    policy
        .host_features
        .as_mut()
        .unwrap()
        .policy
        .categorical_signature = [12; 32];
    let first = admitted(0, 20);
    for changed in [upload, terminal, policy, first] {
        let altered = shape(&[changed], Some(CoreReadbackRoute::SubmissionStaged))
            .row_multiset_features
            .unwrap();
        assert_eq!(
            altered.wave_policy_signature,
            reference.wave_policy_signature
        );
        assert_ne!(altered.rows, reference.rows);
    }
    for (route, product) in [
        (
            CoreReadbackRoute::HostSynchronized,
            CostProductOutput::GreedyToken,
        ),
        (
            CoreReadbackRoute::SubmissionFallbackSynchronized,
            CostProductOutput::GreedyToken,
        ),
        (
            CoreReadbackRoute::SubmissionStaged,
            CostProductOutput::FullLogits,
        ),
    ] {
        let altered = shape_product(&[base], Some(route), product)
            .row_multiset_features
            .unwrap();
        assert_eq!(altered.rows, reference.rows);
        assert_ne!(
            altered.wave_policy_signature,
            reference.wave_policy_signature
        );
    }
}

#[test]
fn row_multiset_requires_complete_domain_and_tracks_interleaved_roles() {
    let decode = admitted(3, 20);
    let mut prefill = admitted(0, 20);
    prefill.work = ActualRowWork::Prefill {
        offset: 0,
        count: 8,
        total_prompt_tokens: 16,
    };
    prefill.output = CostRowOutput::Prefill {
        final_logits: false,
    };
    let mixed = shape(
        &[prefill, decode, prefill],
        Some(CoreReadbackRoute::SubmissionStaged),
    );
    assert_eq!(
        mixed
            .row_multiset_features
            .unwrap()
            .rows
            .iter()
            .map(|r| r.role)
            .collect::<Vec<_>>(),
        vec![
            HostRowRoleV2::Prefill,
            HostRowRoleV2::Decode,
            HostRowRoleV2::Prefill
        ]
    );
    assert!(shape(&[decode], None).row_multiset_features.is_none());
    assert!(shape(
        &[decode, row(None)],
        Some(CoreReadbackRoute::SubmissionStaged)
    )
    .row_multiset_features
    .is_none());
    let mut unsupported = decode;
    unsupported
        .host_features
        .as_mut()
        .unwrap()
        .state
        .completion_state_signature = [7; 32];
    assert!(shape(
        &[decode, unsupported],
        Some(CoreReadbackRoute::SubmissionStaged)
    )
    .row_multiset_features
    .is_none());
}

#[test]
fn row_multiset_wire_rejects_oversize_rows_and_invalid_schema_or_count() {
    let mut evidence = shape(
        &[admitted(3, 20)],
        Some(CoreReadbackRoute::SubmissionStaged),
    )
    .row_multiset_features
    .unwrap();
    assert_eq!(evidence.validate(1), Ok(()));
    assert_eq!(evidence.validate(2), Err(CostFeatureError::InvalidState));
    evidence.schema_version += 1;
    assert_eq!(
        evidence.validate(1),
        Err(CostFeatureError::UnsupportedSchema)
    );
    evidence.schema_version = HOST_ROW_MULTISET_FEATURE_SCHEMA_V2;
    evidence.rows = vec![evidence.rows[0]; MAX_COST_ROWS];
    let bytes = serde_json::to_vec(&evidence).unwrap();
    assert!(serde_json::from_slice::<HostRowMultisetCostFeaturesV2>(&bytes).is_ok());
    evidence.rows.push(evidence.rows[0]);
    let bytes = serde_json::to_vec(&evidence).unwrap();
    assert!(serde_json::from_slice::<HostRowMultisetCostFeaturesV2>(&bytes).is_err());
}
