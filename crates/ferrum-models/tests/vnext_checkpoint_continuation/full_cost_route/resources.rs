//! Actual resource-only preparation for the selected-route fixture.
//! It never encodes, submits, initializes model state, or mints a cost receipt.
use super::*;

pub(super) fn make_resident(
    fixture: &Fixture,
    tokens: &[Arc<[u32]>],
    bucket: Option<&ReusableExecutionBucketId>,
) {
    let owners = tokens
        .iter()
        .enumerate()
        .map(|(index, tokens)| {
            fixture.admit(
                &format!("route-resource-preparation.{index}"),
                Arc::clone(tokens),
            )
        })
        .collect::<Vec<_>>();
    let batch = ExecutionBatchParticipants::new(owners.clone()).unwrap();
    let ordered = batch
        .sessions()
        .iter()
        .map(|session| {
            let index = owners
                .iter()
                .position(|owner| Arc::ptr_eq(owner, session))
                .unwrap();
            Arc::clone(&tokens[index])
        })
        .collect::<Vec<_>>();
    let (step, wave) = prepare(fixture, &batch, &ordered, bucket);
    // A prepared-but-unsubmitted Invocation prevents rollback. Drop its real
    // leases, then explicitly abort only these disposable owners, as the
    // product resource-only startup path does. The actual test owners are
    // admitted afterwards and have no copied completed/model frontier.
    drop(wave);
    step.try_abort().unwrap();
    drop(batch);
    for owner in owners {
        owner.try_abort().unwrap();
    }
}

pub(super) fn prepare(
    fixture: &Fixture,
    batch: &ExecutionBatchParticipants<Runtime>,
    tokens: &[Arc<[u32]>],
    bucket: Option<&ReusableExecutionBucketId>,
) -> (
    Arc<StepResourceLease<Runtime>>,
    PreparedStepSubmissionWave<Runtime>,
) {
    let work = batch
        .bind_work_shape(
            tokens
                .iter()
                .map(|tokens| token_span(Arc::clone(tokens), 0..tokens.len()))
                .collect(),
        )
        .unwrap();
    let mut request = StepResourceAdmissionRequest::new(
        work,
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    if let Some(bucket) = bucket {
        request = request.with_reusable_execution_bucket(bucket.clone());
    }
    let step = loop {
        match batch
            .try_begin_step(request.clone(), &fixture.lane)
            .unwrap()
        {
            StepResourceAdmissionDecision::Admitted(step) => break step,
            StepResourceAdmissionDecision::BackingDeferred(deferred) => {
                require_progress(deferred.maintain().unwrap())
            }
            StepResourceAdmissionDecision::Deferred(reason) => require_progress(
                fixture
                    .resources
                    .maintain_for_admission_deferred(&reason)
                    .unwrap(),
            ),
            other => panic!("step unavailable: {:?}", std::mem::discriminant(&other)),
        }
    };
    let wave = loop {
        match step
            .try_prepare_full_plan_submission_wave(
                Arc::new(step.work_shape().clone()),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap()
        {
            StepSubmissionWaveAdmissionDecision::Prepared(wave) => break wave,
            StepSubmissionWaveAdmissionDecision::BackingDeferred(deferred) => {
                require_progress(deferred.maintain().unwrap())
            }
            StepSubmissionWaveAdmissionDecision::Deferred(reason) => require_progress(
                fixture
                    .resources
                    .maintain_for_admission_deferred(&reason)
                    .unwrap(),
            ),
            other => panic!("wave unavailable: {:?}", std::mem::discriminant(&other)),
        }
    };
    (step, wave)
}
