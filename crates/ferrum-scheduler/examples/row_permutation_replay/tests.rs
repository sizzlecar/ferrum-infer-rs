use super::*;
use ferrum_interfaces::execution_cost::CostRowNumericFeatures;
fn shape() -> WaveExecutionShape {
    WaveExecutionShape {
        row_multiset_features: None,
        host_content_features: Some(HostContentCostFeaturesV1 {
            schema_version: 1,
            output_policy_signature: [8; 32],
        }),
        numeric_features: Some(CanonicalWaveCostFeatures {
            schema_version: 1,
            output_policy_signature: [7; 32],
            rows: [9, 3]
                .map(|n| CostRowNumericFeatures {
                    generated_tokens_before: n,
                    maximum_output_tokens: 20,
                    sampling_history_tokens: n,
                    repetition_tokens: 0,
                    decoded_prefix_tokens: n + 1,
                    decoded_text_bytes_bound: (n + 1) * 4,
                    decode_scratch_bytes_bound: (n + 1) * 2,
                })
                .to_vec(),
        }),
        kind: WaveKind::Decode,
        path: WaveExecutionPath::PlanRuntime,
        provider_signature: [6; 32],
        output_policy_signature: [5; 32],
        graph_state: WaveGraphState::Disabled,
        order: BatchOrderSemantics::Ordered,
        decode_kv_tokens: vec![90, 30],
        prefill_chunks: vec![],
        recurrent_state_bytes: 0,
        restore_bytes: 0,
        maintenance_bytes: 0,
        maintenance_units: 0,
    }
}
#[test]
fn legacy_diagnostic_keeps_correlated_rows_and_does_not_invent_v2_metadata() {
    let mut value = shape();
    let old = value.host_content_features;
    assert!(diagnostic_permute(&mut value).unwrap());
    assert_eq!(value.decode_kv_tokens, vec![30, 90]);
    assert_eq!(
        value
            .numeric_features
            .unwrap()
            .rows
            .iter()
            .map(|r| r.generated_tokens_before)
            .collect::<Vec<_>>(),
        vec![3, 9]
    );
    assert_eq!(value.host_content_features, old);
    assert!(value.row_multiset_features.is_none());
}
#[test]
fn legacy_mixed_or_missing_numeric_evidence_stays_excluded() {
    let mut value = shape();
    value.kind = WaveKind::Mixed;
    let original = value.clone();
    assert!(!diagnostic_permute(&mut value).unwrap());
    assert_eq!(value, original);
    value = shape();
    value.numeric_features = None;
    assert!(!diagnostic_permute(&mut value).unwrap());
}
