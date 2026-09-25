use super::*;
use crate::commands::calibrate_slo::report::Phase;
use ferrum_interfaces::execution_cost::{CoreReadbackRoute, HostCostPolicyV2, HostRowRoleV2};
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    StructuredProductV2, StructuredTemplateV2, StructuredWaveRoleV2,
};

// A display-data fixture tests aggregation only. It is not a StructuredInputV2,
// qualified receipt or source; none of those types is constructed from this DTO.
fn report(domain: u8, pending: u32, length: u32, coordinate: u64) -> DiscoveryReportV2 {
    let rows = (0..2)
        .map(|position| StructuredHostRowV1 {
            physical_position: position,
            role: HostRowRoleV2::Decode,
            installed_policy: HostCostPolicyV2 {
                empirical_content_domain: None,
                categorical_signature: [5; 32],
                decoder_text_bytes_per_token: 16,
                decoder_scratch_bytes_per_token: 32,
                raw_token_bytes_bound: 16,
            },
            no_generated_history: false,
            pending_decoded_utf8: position == pending,
            initial_prefill: false,
            final_prefill: false,
            mask_upload_required: false,
            decode_requires_full_logits: Some(position == pending),
            repetition_penalty_bits: Some(1f32.to_bits()),
            terminal_expectation: if position == length {
                HostTerminalExpectationV1::LengthBoundary
            } else {
                HostTerminalExpectationV1::TokenMayTerminate
            },
        })
        .collect();
    DiscoveryReportV2 {
        model_revision: MODEL_REVISION_V2,
        input: DiscoveryInputV2::Known {
            owner: StructuredOwnerKeyV2 {
                rows: 2,
                role: StructuredWaveRoleV2::OrdinaryDecode,
                product: StructuredProductV2::FullLogits,
                readback: CoreReadbackRoute::HostSynchronized,
                provider_template: StructuredTemplateV2::Ordered([domain; 32]),
                algorithm_domain: [4; 32],
                installed_policy: [5; 32],
            },
            domain_signature: [domain; 32],
            basis: vec![1., coordinate as f64],
            support: vec![coordinate],
            physical_host_rows: rows,
        },
    }
}
#[test]
fn discovery_summary_keeps_phase_patterns_and_numeric_extrema_separate() {
    let mut summary = DiscoverySummaryV2::new();
    let first = report(1, 0, 1, 10);
    summary.observe(Phase::Warmup, &first);
    summary.observe(Phase::Qualification, &first);
    assert_eq!(summary.wave_reports_seen, 0);
    summary.observe(Phase::Discovery, &first);
    summary.observe(Phase::Discovery, &report(1, 1, 0, 20));
    let d = &summary.domains[0];
    assert_eq!(d.waves, 2);
    assert_eq!(d.pending_positions, BTreeSet::from([0, 1]));
    assert_eq!(d.length_positions, BTreeSet::from([0, 1]));
    assert_eq!(d.pending_counts, BTreeSet::from([1]));
    assert_eq!(d.length_counts, BTreeSet::from([1]));
    assert_eq!(d.joint_counts, BTreeSet::from([(1, 1)]));
    assert_eq!(
        (d.support_ranges[0].minimum, d.support_ranges[0].maximum),
        (10, 20)
    );
    assert!(!summary.collection_completed);
    assert!(!summary.inventory_truncated);
    let wire = serde_json::to_value(&summary).unwrap();
    assert!(wire.get("qualified").is_none());
    assert!(wire.get("source").is_none());
}
#[test]
fn discovery_unknown_is_counted_without_creating_a_domain_or_model() {
    let mut summary = DiscoverySummaryV2::new();
    let input = inspect(SloStructuredCostCapture::HostSettledV1, || {
        Err(StructuredUnknownV2::MissingEvidence)
    })
    .unwrap();
    summary.observe(Phase::Discovery, &input);
    assert_eq!(summary.wave_reports_seen, 1);
    assert_eq!(summary.unknown_input_reports, 1);
    assert_eq!(summary.unknown_reasons["MissingEvidence"], 1);
    assert!(summary.domains.is_empty());
    assert_eq!(summary.known_input_waves, 0);
    assert!(!summary.collection_completed);
}
#[test]
fn discovery_domain_and_total_coordinate_caps_mark_incomplete_inventory() {
    let mut summary = DiscoverySummaryV2::new();
    for domain in 0..=MAX_DOMAINS {
        summary.observe(Phase::Discovery, &report(domain as u8, 0, 1, 1));
    }
    assert_eq!(summary.domains.len(), MAX_DOMAINS);
    assert_eq!(summary.omitted_known_waves, 1);
    assert!(summary.inventory_truncated);
    // A later observation of a retained owner still contributes; truncation is sticky.
    summary.observe(Phase::Discovery, &report(0, 1, 0, 2));
    assert_eq!(summary.domains[0].waves, 2);
    let mut summary = DiscoverySummaryV2::new();
    for domain in 0..=(MAX_COORDINATES / 8192) {
        let mut input = report(domain as u8, 0, 1, 1);
        if let DiscoveryInputV2::Known { basis, support, .. } = &mut input.input {
            *basis = vec![1.; 4096];
            *support = vec![1; 4096];
        }
        summary.observe(Phase::Discovery, &input);
    }
    assert_eq!(summary.coordinates, MAX_COORDINATES);
    assert!(summary.inventory_truncated);
    assert_eq!(summary.omitted_known_waves, 1);
}
