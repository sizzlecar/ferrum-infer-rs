//! Original CPU submit -> host settlement -> FIFO -> source7 replay/renewal.
//! The issued declaration borrows a genuinely qualified runtime prediction;
//! this tests the source boundary, not controller search or hardware accuracy.
use super::*;
use crate::continuous_engine::inner::cost_observation::prospective_capture::{
    ProspectiveCapture, ProspectiveCaptureOutcomeV1,
};
use ferrum_scheduler::implementations::continuous::cost_profile::StructuredServiceWaveV7;
use std::time::{Duration, Instant};

fn record_issued(f: &mut Families) {
    let w = Families::wave(A);
    let query = StructuredQueryV2::from_future_with_domain(
        &w.prepared.exact,
        &w.prepared.selected,
        &w.prepared.recipe,
        &ferrum_interfaces::execution_cost::HostContentForecastV2::Exact,
        &f.domain,
    )
    .unwrap();
    let prediction = f
        .runtime
        .snapshot()
        .unwrap()
        .audit_structured_query_v2(&query, f.clock.now_ns().unwrap())
        .unwrap();
    let participants = w
        .actual
        .rows
        .iter()
        .map(|row| CostObservationParticipant {
            request_id: row.request_id.clone(),
            owner_incarnation: row.owner_incarnation,
            work_generation: row.work_generation,
            input_index: row.input_index,
            output_policy_signature: Some([6; 32]),
            host_features: Some(w.host),
        })
        .collect();
    let issued = ProspectiveCapture::from_fixture(
        &f.runtime,
        w.prepared.exact.clone(),
        w.prepared.selected.clone(),
        &query,
        participants,
        Instant::now() + Duration::from_secs(30),
    )
    .with_issued_bound_for_test(&f.runtime, prediction.planning_ns);
    let stages = fixture::record_issued_cohort_route(&f.runtime, &f.clock, w, issued.clone())
        .expect("real CPU submission and complete original host settlement");
    let receipt = stages.prospective_capture.as_ref().unwrap();
    assert_eq!(receipt.outcome(), ProspectiveCaptureOutcomeV1::Matched);
    assert!(receipt.validates_host_stages(&stages));

    let diagnostic = serde_json::to_value(stages.structured_diagnostic_view()).unwrap();
    assert!(diagnostic.get("prospective_capture").is_some());
    let source = serde_json::to_value(stages.structured_source_view()).unwrap();
    assert!(source.get("prospective_capture").is_none());
    assert_eq!(
        source["structured_evidence"],
        diagnostic["structured_evidence"]
    );
    let mut legacy = stages.as_ref().clone();
    legacy.prospective_capture = None;
    assert_eq!(
        serde_json::to_vec(&stages.structured_source_view()).unwrap(),
        serde_json::to_vec(&legacy.structured_diagnostic_view()).unwrap(),
        "all original source bytes, including the original qualifier, are preserved"
    );
    let independent = stages
        .statistical_evidence
        .as_ref()
        .and_then(|v| v.independent_attention_v2())
        .map(|v| v.to_wire_v2());
    assert!(
        StructuredServiceWaveV7::from_diagnostic(
            1,
            stages.prepare_started_at_ns.unwrap(),
            1,
            diagnostic,
            independent.clone(),
        )
        .is_err(),
        "diagnostic sidecars do not expand the strict source schema"
    );
    let mut unknown = source;
    unknown["untrusted_extra"] = serde_json::json!(true);
    assert!(StructuredServiceWaveV7::from_diagnostic(
        1,
        stages.prepare_started_at_ns.unwrap(),
        1,
        unknown,
        independent,
    )
    .is_err());

    f.runtime.consume_samples();
    f.recorded += 1;
    let audit = f.runtime.audit_snapshot();
    assert_eq!(audit.sink.raw_accepted, f.recorded);
    assert_eq!(audit.sink.raw_resolved, f.recorded);
    assert_eq!(audit.sink.raw_lost, 0);
    assert_eq!(audit.sink.raw_resolution_failed, 0);
    assert!(
        audit.prospective_capture.is_none(),
        "passive capture stays disabled"
    );
    let issued_audit = audit.issued_structured_prediction.unwrap();
    assert_eq!(issued_audit.paired_count, f.recorded - (4 * OFFERS) as u64);
    assert_eq!(issued_audit.invalid_measurement, 0);
    assert_eq!(issued_audit.matched_but_not_retained, 0);
    assert_eq!(f.live().audit().failed_generations, 0);
    assert!(f.live().audit().publication_error.is_none());
    assert!(receipt.validates_host_stages(&stages));
}

#[tokio::test]
async fn automatic_issued_source7_renews_from_original_fifo_without_serializing_diagnostic_sidecar()
{
    let mut f = Families::new();
    complete_epoch(&mut f, 1, 0);
    let first = only_child(&f);
    let original_snapshot = f.runtime.snapshot().unwrap();
    let original_epoch = original_snapshot.model_version();
    f.runtime.consume_samples();
    assert_eq!(f.live().audit().population.generation, 2);
    for block in 0..4 {
        f.runtime.consume_samples();
        for _ in 0..OFFERS {
            record_issued(&mut f);
        }
        let audit = f.live().audit();
        assert_eq!(audit.qualified_publications, 1 + u64::from(block == 3));
        let blocks = audit
            .automatic
            .as_ref()
            .unwrap()
            .owner_blocks
            .as_ref()
            .unwrap();
        assert_eq!(blocks.block, block as u64 + 1);
        let phases = [
            Some(ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredPhaseV2::Fit),
            Some(ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredPhaseV2::Residual),
            Some(ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredPhaseV2::Qualification),
            None,
        ];
        assert_eq!(blocks.owners[0].phase, phases[block]);
    }
    let second = only_child(&f);
    assert_eq!(
        second.provenance().phases.each_ref().map(|p| p.members),
        [OFFERS; 3]
    );
    assert_ne!(
        first.provenance().source_sha256,
        second.provenance().source_sha256
    );
    assert!(
        second.provenance().phases[0].accepted_fifo_cutoff
            > first.provenance().phases[2].accepted_fifo_cutoff
    );
    assert!(!original_snapshot.current());
    let renewed = f.runtime.snapshot().unwrap();
    assert!(renewed.model_version() > original_epoch);
    let prediction = renewed
        .audit_structured_query_v2(&f.query(A), f.clock.now_ns().unwrap())
        .unwrap();
    assert_eq!(prediction.model_version, renewed.model_version());
    f.runtime.consume_samples();
    record_issued(&mut f);
    let audit = f.runtime.audit_snapshot();
    assert_eq!(
        audit.issued_structured_prediction.unwrap().paired_count,
        (4 * OFFERS + 1) as u64
    );
    assert!(audit.structured_feedback.unwrap().revoked.is_none());
    f.runtime.shutdown().await.unwrap();
}
