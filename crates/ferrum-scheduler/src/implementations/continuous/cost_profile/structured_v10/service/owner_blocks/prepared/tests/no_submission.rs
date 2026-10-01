//! Source DTO tests validate original replay rules; engine tests separately
//! obtain NoSubmission from the private producer rather than these wire DTOs.
use super::*;
use ferrum_interfaces::execution_cost::{
    no_submission_participant_signature, CallNoSubmissionParticipantV1,
};
fn id(n: u8) -> ferrum_types::RequestId {
    serde_json::from_value(serde_json::json!(format!(
        "00000000-0000-0000-0000-{n:012}"
    )))
    .unwrap()
}
fn setup() -> StructuredPreparedOwnerBlockCollectorV8 {
    let mut h = header();
    h.declaration.population.route_population =
        ferrum_types::SloCalibrationRoutePopulationV1::WarmOrGraphDisabledWithNoSubmissionV2;
    let h = StructuredPreparedOwnerBlockHeaderV8::new(
        h.capture_identity,
        h.generation,
        h.fingerprint,
        h.producer,
        h.opening,
        h.declaration,
        h.maximum_file_bytes,
    )
    .unwrap();
    let mut c =
        StructuredPreparedOwnerBlockCollectorV8::new(h, CostProfileLoadLimits::default()).unwrap();
    c.open_block(2, 0).unwrap();
    for value in [
        serde_json::json!({"kind":"cohort_begin","phase":"fit","cohort":0,"manifest_case":0,"repetition":0}),
        serde_json::json!({"kind":"request_admitted","phase":"fit","cohort":0,"slot":0,"request_id":id(1).to_string(),"maximum_output":3}),
    ] {
        c.push(&StructuredPreparedOwnerBlockRecordV8::Cohort(
            StructuredCohortEventV8::from_diagnostic(value).unwrap(),
        ))
        .unwrap();
    }
    c
}
fn offer(ticket: u64) -> StructuredPreparedOwnerBlockRecordV8 {
    StructuredPreparedOwnerBlockRecordV8::Preparation(StructuredPreparationEventV8::from_diagnostic(serde_json::json!({
        "kind":"preparation_offered","offered":ticket,"phase":"fit","cohort":0,
        "rows":[{"before":{"request_id":id(1).to_string(),"owner_incarnation":1,"work_generation":1,"generated_tokens":0,"kv_tokens":0,"model_cache_id":null,"pending_utf8":[],"output_accepted_ordinal":0},
            "work":{"kind":"prefill","offset":0,"count":64,"total_prompt_tokens":64}}]
    })).unwrap())
}
fn attempt(request: u8) -> StructuredServiceNoSubmissionV7 {
    let participants = vec![CallNoSubmissionParticipantV1 {
        request_id: id(request),
        owner_incarnation: 1,
        work_generation: 1,
        input_index: 0,
    }];
    serde_json::from_value(serde_json::json!({"ticket":1,"issued_at_ns":3,"fifo":1,"protocol":"ferrum.inference-not-submitted.v1",
        "fingerprint":old::header().fingerprint,"call_id":1,"prepare_started_at_ns":3,"returned_at_ns":4,"call_returned_at_ns":5,"finalized_at_ns":6,
        "participant_signature":no_submission_participant_signature(&participants).unwrap(),"participants":participants,
        "reason":{"kind":"capacity","stage":"submission_wave"}})).unwrap()
}
fn outcome(request: u8, cohort: usize) -> StructuredPreparedOwnerBlockRecordV8 {
    StructuredPreparedOwnerBlockRecordV8::PreparationDisposition(
        StructuredPreparationDispositionV8::PreparationNotSubmitted {
            phase: StructuredProfilePhaseV10::Fit,
            cohort,
            attempt: attempt(request),
        },
    )
}
#[test]
fn source8_preparation_no_submission_counts_original_offer_then_retries_unchanged_frontier() {
    let mut c = setup();
    c.push(&offer(1)).unwrap();
    c.push(&outcome(1, 0)).unwrap();
    assert_eq!(
        (
            c.offered(),
            c.last_fifo(),
            c.preparation_attempts(),
            c.qualified_children()
        ),
        (1, 1, 1, 0)
    );
    c.push(&offer(2)).unwrap();
    assert!(c
        .close_block(StructuredServiceClockV7 {
            monotonic_ns: 7,
            wall_unix_ns: 1_000_006
        })
        .is_err());
}
#[test]
fn source8_preparation_no_submission_rejects_foreign_cohort_or_duplicate_outcome() {
    for (request, cohort) in [(2, 0), (1, 1)] {
        let mut c = setup();
        c.push(&offer(1)).unwrap();
        assert!(c.push(&outcome(request, cohort)).is_err());
        assert!(c.audit().poisoned);
    }
    let mut c = setup();
    c.push(&offer(1)).unwrap();
    c.push(&outcome(1, 0)).unwrap();
    assert!(c.push(&outcome(1, 0)).is_err());
    let mut c = setup();
    c.push(&offer(1)).unwrap();
    assert!(c
        .push(&StructuredPreparedOwnerBlockRecordV8::Population(
            StructuredServiceRecordV7::NotSubmitted {
                attempt: attempt(1)
            }
        ))
        .is_err());
}
#[test]
fn source8_ordinary_no_submission_cannot_count_unadmitted_request() {
    let mut c = setup();
    assert!(c
        .push(&StructuredPreparedOwnerBlockRecordV8::Population(
            StructuredServiceRecordV7::NotSubmitted {
                attempt: attempt(2)
            }
        ))
        .is_err());
    assert!(c.audit().poisoned);
    assert_eq!(c.qualified_children(), 0);
}
