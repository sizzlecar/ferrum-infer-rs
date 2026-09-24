use super::*;
use ferrum_interfaces::model_executor::PrefillChunk;
use ferrum_scheduler::implementations::continuous::slo_planner::PlanningTimeOrigin;
use ferrum_testkit::MockKvCacheHandle;
use ferrum_types::{InferenceRequest, SloLatencyBudgets, TokenId};
use std::time::Duration;

mod fixture;
mod product;
use fixture::*;

pub(super) fn calibration_runtime() -> Arc<EnginePrefillReferenceRuntime> {
    fixture::shared_runtime()
}

pub(super) fn calibration_artifact(
) -> ferrum_scheduler::implementations::continuous::prefill_reference::ReferenceCalibrationV1 {
    fixture::artifact()
}

fn budgets() -> SloLatencyBudgets {
    SloLatencyBudgets {
        ttft_ms: n64(10_000),
        tpot_ms: n64(100),
        itl_ms: n64(100),
    }
}
fn sequence(
    runtime: &EnginePrefillReferenceRuntime,
    id: RequestId,
    ingress: Instant,
    admitted: Instant,
    prefix: usize,
    length: usize,
) -> SequenceState {
    let mut request = InferenceRequest::new("one two three", "reference-test");
    request.id = id;
    request.sampling_params.max_tokens = 8;
    let mut sequence = SequenceState::new(request, vec![TokenId::new(1); length]);
    sequence.slo = Some(
        ferrum_interfaces::RequestSloState::new(ingress, "reference".into(), budgets()).unwrap(),
    );
    sequence.prefill_tokens_processed = prefix;
    sequence.prefill_reference = Some(match runtime.bind(&sequence, admitted) {
        Ok(binding) => SequencePrefillReference::Known(binding),
        Err(reason) => SequencePrefillReference::Unknown(reason),
    });
    sequence
}
fn known(sequence: &SequenceState) -> &SequenceReferenceBinding {
    sequence
        .prefill_reference
        .as_ref()
        .unwrap()
        .known()
        .unwrap()
}
fn commit(sequence: &mut SequenceState, chunk: PrefillChunk, output: bool) {
    let receipt = sequence.prepare_prefill_reference_commit(chunk);
    if output {
        sequence.generated_tokens.push(TokenId::new(4));
    }
    sequence.commit_plan_runtime_prefill_chunk_resources(
        Arc::new(MockKvCacheHandle::new(
            sequence.request_id.clone(),
            1,
            chunk.end(),
        )),
        chunk.end(),
        chunk.is_final(),
    );
    sequence.publish_prefill_reference_commit(receipt);
}

#[test]
fn absent_config_never_queries_identity_and_explicit_errors_stay_typed() {
    assert!(
        EnginePrefillReferenceRuntime::load(None, || panic!("Off queried identity"))
            .unwrap()
            .is_none()
    );
    let file = ArtifactFile::new();
    assert!(matches!(
        EnginePrefillReferenceRuntime::load(Some(&file.0), Default::default),
        Err(PrefillReferenceLoadError::IdentityUnavailable)
    ));
    let mut incompatible = file.0.clone();
    incompatible.expected_protocol_sha256 = [44; 32];
    assert!(matches!(
        EnginePrefillReferenceRuntime::load(Some(&incompatible), identity),
        Err(PrefillReferenceLoadError::Artifact(
            ReferenceError::Incompatible
        ))
    ));
    let wrong_identity = || {
        let ExecutorCostIdentityAvailability::Known(value) = identity() else {
            unreachable!()
        };
        let mut value = (*value).clone();
        value.model_weights = [45; 32];
        ExecutorCostIdentityAvailability::Known(Arc::new(value))
    };
    assert!(matches!(
        EnginePrefillReferenceRuntime::load(Some(&file.0), wrong_identity),
        Err(PrefillReferenceLoadError::Artifact(
            ReferenceError::Incompatible
        ))
    ));
    std::fs::write(&file.0.artifact_path, b"{malformed}").unwrap();
    assert!(matches!(
        EnginePrefillReferenceRuntime::load(Some(&file.0), identity),
        Err(PrefillReferenceLoadError::Artifact(ReferenceError::Json(_)))
    ));
}

#[test]
fn original_ingress_and_admission_survive_snapshot_origin_changes_and_recompute() {
    let file = ArtifactFile::new();
    let runtime = file.runtime();
    let ingress = Instant::now().checked_sub(Duration::from_secs(2)).unwrap();
    let admission = ingress + Duration::from_millis(20);
    let mut sequence = sequence(&runtime, RequestId::new(), ingress, admission, 0, 4);
    let original = known(&sequence).identity();
    commit(&mut sequence, PrefillChunk::new(0, 2, 4).unwrap(), false);
    assert_eq!(known(&sequence).binding().logical_high_water(), 2);
    let old_resources = sequence.take_physical_resources_for_recompute();
    assert_eq!(sequence.prefill_tokens_processed, 0);
    assert_eq!(known(&sequence).binding().logical_high_water(), 2);
    commit(&mut sequence, PrefillChunk::new(0, 2, 4).unwrap(), false);
    assert_eq!(known(&sequence).binding().logical_high_water(), 2);
    assert_eq!(known(&sequence).binding().admitted_at_ns(), 20_000_000);
    assert_eq!(known(&sequence).identity(), original);
    assert_eq!(known(&sequence).tau_ref_ns().get(), 7);
    assert!(Arc::ptr_eq(
        known(&sequence).binding().reference(),
        &runtime
            .calibration()
            .curve(NonZeroU32::new(4).unwrap())
            .unwrap()
    ));
    let observed = ingress + Duration::from_secs(1);
    let earlier =
        PlanningTimeOrigin::from_origin(ingress - Duration::from_secs(1), observed).unwrap();
    let later = PlanningTimeOrigin::from_origin(ingress, observed).unwrap();
    let point = ingress + Duration::from_secs(5);
    let a = known(&sequence)
        .project_progress(&earlier, 2, 4, &[earlier.at_ns(point).unwrap()])
        .unwrap();
    let b = known(&sequence)
        .project_progress(&later, 2, 4, &[later.at_ns(point).unwrap()])
        .unwrap();
    assert_eq!(a.admitted_at_ns - b.admitted_at_ns, 1_000_000_000);
    assert_eq!(a.milestones[0].at_ns - b.milestones[0].at_ns, 1_000_000_000);
    assert_eq!(
        a.milestones[0].required_reference_work_ns,
        b.milestones[0].required_reference_work_ns
    );
    drop(old_resources);
}

#[test]
fn verified_admission_prefix_is_baseline_and_uncovered_length_is_unknown() {
    let file = ArtifactFile::new();
    let runtime = file.runtime();
    let ingress = Instant::now();
    let mut bound = sequence(&runtime, RequestId::new(), ingress, ingress, 2, 4);
    assert_eq!(known(&bound).binding().reference_work_at_admission_ns(), 5);
    commit(&mut bound, PrefillChunk::new(2, 2, 4).unwrap(), true);
    assert!(known(&bound).binding().first_token_committed());
    let unknown = sequence(&runtime, RequestId::new(), ingress, ingress, 0, 3);
    assert!(matches!(
        unknown.prefill_reference.as_ref().unwrap().known(),
        Err(ReferenceBindingUnknown::Reference(
            ReferenceUnknown::LengthNotCalibrated
        ))
    ));
    assert!(unknown.generated_tokens.is_empty());
}

#[test]
fn stale_and_duplicate_receipts_do_not_advance_or_poison_replacement_owner() {
    let file = ArtifactFile::new();
    let runtime = file.runtime();
    let ingress = Instant::now();
    let id = RequestId::new();
    let mut first = sequence(&runtime, id.clone(), ingress, ingress, 0, 4);
    let chunk = PrefillChunk::new(0, 2, 4).unwrap();
    let stale = first.prepare_prefill_reference_commit(chunk);
    let duplicate = first.prepare_prefill_reference_commit(chunk);
    commit(&mut first, chunk, false);
    first.publish_prefill_reference_commit(duplicate);
    assert_eq!(known(&first).binding().logical_high_water(), 2);
    let mut replacement = sequence(&runtime, id, ingress, ingress, 0, 4);
    replacement.publish_prefill_reference_commit(stale);
    assert_eq!(known(&replacement).binding().logical_high_water(), 0);
    let abandoned = replacement.prepare_prefill_reference_commit(chunk);
    drop(abandoned);
    assert_eq!(known(&replacement).binding().logical_high_water(), 0);
}

#[test]
fn final_kv_without_output_and_unmeasured_endpoint_never_earn_credit() {
    let file = ArtifactFile::new();
    let runtime = file.runtime();
    let ingress = Instant::now();
    let mut no_output = sequence(&runtime, RequestId::new(), ingress, ingress, 0, 4);
    commit(&mut no_output, PrefillChunk::new(0, 4, 4).unwrap(), false);
    assert!(matches!(
        no_output.prefill_reference.as_ref().unwrap().known(),
        Err(ReferenceBindingUnknown::InvalidReceipt)
    ));
    let mut unmeasured = sequence(&runtime, RequestId::new(), ingress, ingress, 0, 4);
    commit(&mut unmeasured, PrefillChunk::new(0, 1, 4).unwrap(), false);
    assert!(matches!(
        unmeasured.prefill_reference.as_ref().unwrap().known(),
        Err(ReferenceBindingUnknown::Reference(
            ReferenceUnknown::MissingEndpoint
        ))
    ));
}

pub(super) fn piecewise_runtime() -> Arc<EnginePrefillReferenceRuntime> {
    fixture::piecewise_runtime()
}
