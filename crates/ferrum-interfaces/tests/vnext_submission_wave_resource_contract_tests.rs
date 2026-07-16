#[path = "vnext_resource_contract/support.rs"]
mod resource_support;
mod vnext_core_contract;

use ferrum_interfaces::vnext::*;
use std::collections::BTreeSet;
use std::sync::Arc;
use vnext_core_contract as core;

fn sequential_scratch_plan() -> (
    ExecutionPlan,
    OperationRuntimeRegistry<core::PlanningTestRuntime>,
) {
    let registration = TypedFamilyRegistration::new(core::SequentialScratchFamily);
    let family = registration
        .prepare(&serde_json::json!({"width": 4}))
        .unwrap();
    let catalog = core::catalog();
    let policy = core::policy(4096);
    let descriptor = catalog.providers_for(&core::id("operation.main")).unwrap()[0].clone();
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(core::TestOperationContract {
            descriptor: core::operation(),
            calls: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            reject_signature: false,
        }) as Box<dyn OperationContract>],
        vec![Box::new(core::SequentialScratchEstimator { descriptor })
            as Box<dyn OperationProvider<core::PlanningTestRuntime>>],
    )
    .unwrap();
    let planning = registry.planning();
    let first = PlanNodeResolution::resolve(
        &family,
        &catalog,
        &policy,
        &planning,
        core::id("node.first"),
        core::sequential_resolved_values(
            "value.input",
            "resource.sequential.input",
            "value.intermediate",
            "resource.sequential.intermediate",
        ),
        BTreeSet::new(),
        None,
    )
    .unwrap();
    let second = PlanNodeResolution::resolve(
        &family,
        &catalog,
        &policy,
        &planning,
        core::id("node.second"),
        core::sequential_resolved_values(
            "value.intermediate",
            "resource.sequential.intermediate",
            "value.output",
            "resource.sequential.output",
        ),
        BTreeSet::new(),
        None,
    )
    .unwrap();
    let plan = ExecutionPlan::build(
        PlanBuildRequest::new(&family, &catalog, &policy, vec![first, second]).unwrap(),
    )
    .unwrap();
    (plan, registry)
}

fn begin_step(
    batch: &ExecutionBatchParticipants<resource_support::TestRuntime>,
) -> Arc<StepResourceLease<resource_support::TestRuntime>> {
    let request = StepResourceAdmissionRequest::new(
        batch
            .bind_work_shape(vec![resource_support::one_token_span()])
            .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    for attempt in 0..=3 {
        match batch.try_begin_step(request.clone()).unwrap() {
            StepResourceAdmissionDecision::Admitted(step) => return step,
            StepResourceAdmissionDecision::BackingDeferred(deferred) if attempt < 3 => {
                deferred.maintain().unwrap();
            }
            _ => panic!("sequential scratch step admission did not converge"),
        }
    }
    unreachable!("bounded step admission returns or panics")
}

fn submission_requests(
    plan: &ExecutionPlan,
    step: &Arc<StepResourceLease<resource_support::TestRuntime>>,
) -> Vec<InvocationResourceAdmissionRequest> {
    plan.payload()
        .nodes()
        .iter()
        .map(|node| {
            InvocationResourceAdmissionRequest::for_all_step_participants(
                node.id().clone(),
                step.bind_all_invocation_work_shape(vec![resource_support::one_token_span()])
                    .unwrap(),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap()
        })
        .collect()
}

fn prepare_wave(
    plan: &ExecutionPlan,
    step: &Arc<StepResourceLease<resource_support::TestRuntime>>,
) -> PreparedStepSubmissionWave<resource_support::TestRuntime> {
    let requests = submission_requests(plan, step);
    for attempt in 0..=3 {
        match step.try_prepare_submission_wave(requests.clone()).unwrap() {
            StepSubmissionWaveAdmissionDecision::Prepared(wave) => return wave,
            StepSubmissionWaveAdmissionDecision::BackingDeferred(deferred) if attempt < 3 => {
                deferred.maintain().unwrap();
            }
            _ => panic!("sequential scratch wave admission did not converge"),
        }
    }
    unreachable!("bounded wave admission returns or panics")
}

#[test]
fn total_order_invocation_scratch_claims_peak_once_for_the_whole_wave() {
    let (plan, registry) = sequential_scratch_plan();
    assert_eq!(plan.payload().memory().minimum_invocation_peak_bytes(), 96);
    let (driver, _trace) = resource_support::configured_driver(&plan, &[], &[]);
    let root = resource_support::plan_runtime(&plan, driver, "submission-wave-peak");
    let sequence = resource_support::admit_logical_sequence(
        &root,
        "run.submission-wave-peak",
        "request.submission-wave-peak",
    );
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let step = begin_step(&batch);

    let wave = prepare_wave(&plan, &step);
    let claimed = wave.claimed_backing();
    let immediate_bytes = claimed
        .demand()
        .immediate_claim()
        .entries()
        .iter()
        .map(|entry| entry.units().get())
        .sum::<u64>();
    assert_eq!(wave.node_count(), 2);
    assert_eq!(claimed.node_work_shapes().len(), 2);
    assert_eq!(immediate_bytes, 96);
    assert_eq!(claimed.physical_claim_count(), 1);
    assert_eq!(claimed.backing_slices().len(), 2);
    assert!(claimed.backing_slices().iter().all(|slice| {
        slice.evidence().physical_size_bytes() == 96
            && slice.evidence().physical_offset_bytes() == 0
    }));
    let capacity = claimed
        .logical_capacity()
        .expect("non-empty invocation peak owns one logical claim");
    assert_eq!(capacity.claims(), claimed.demand().immediate_claim());
    assert_eq!(capacity.parents().len(), 1);

    drop(wave);
    step.try_retire_normal().unwrap();
    drop(batch);
    session.try_complete().unwrap();
    drop(session);
    drop(sequence);
    drop(registry);
    resource_support::close_plan_runtime(root);
}

#[test]
fn submission_wave_backing_deferral_retains_step_until_exact_retry() {
    let (plan, registry) = sequential_scratch_plan();
    let (driver, _trace) = resource_support::configured_driver(&plan, &[], &[]);
    let root = resource_support::plan_runtime(&plan, driver, "submission-wave-deferral-owner");
    let first_sequence = resource_support::admit_logical_sequence(
        &root,
        "run.submission-wave-deferral-owner-first",
        "request.submission-wave-deferral-owner-first",
    );
    let first_session = first_sequence.open_session().unwrap();
    let first_batch = ExecutionBatchParticipants::new(vec![Arc::clone(&first_session)]).unwrap();
    let first_step = begin_step(&first_batch);
    let first_wave = prepare_wave(&plan, &first_step);

    let second_sequence = resource_support::admit_logical_sequence(
        &root,
        "run.submission-wave-deferral-owner-second",
        "request.submission-wave-deferral-owner-second",
    );
    let second_session = second_sequence.open_session().unwrap();
    let second_batch = ExecutionBatchParticipants::new(vec![Arc::clone(&second_session)]).unwrap();
    let step = begin_step(&second_batch);
    let requests = submission_requests(&plan, &step);

    let deferred = match step.try_prepare_submission_wave(requests.clone()).unwrap() {
        StepSubmissionWaveAdmissionDecision::BackingDeferred(deferred) => deferred,
        _ => panic!("zero-resident invocation backing must defer the submission wave"),
    };
    assert_eq!(
        deferred.node_work_fingerprints().len(),
        plan.payload().nodes().len()
    );
    assert!(deferred
        .node_work_fingerprints()
        .iter()
        .zip(plan.payload().nodes())
        .all(|((node_id, _), node)| node_id == node.id()));

    let failure = step
        .try_retire_normal()
        .expect_err("the deferred wave must retain the exact step parent");
    assert!(failure
        .error()
        .to_string()
        .contains("step cannot finalize while an invocation or scheduler clone retains it"));
    let step = failure.into_step();
    assert!(matches!(
        deferred.maintain().unwrap(),
        DynamicDeferredMaintenanceOutcome::Maintained(_)
    ));
    drop(deferred);

    let wave = match step.try_prepare_submission_wave(requests).unwrap() {
        StepSubmissionWaveAdmissionDecision::Prepared(wave) => wave,
        _ => panic!("maintained invocation backing must prepare the exact wave"),
    };
    drop(wave);
    step.try_retire_normal().unwrap();
    drop(second_batch);
    second_session.try_complete().unwrap();
    drop(second_session);
    drop(second_sequence);

    drop(first_wave);
    first_step.try_retire_normal().unwrap();
    drop(first_batch);
    first_session.try_complete().unwrap();
    drop(first_session);
    drop(first_sequence);
    drop(registry);
    resource_support::close_plan_runtime(root);
}
