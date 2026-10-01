//! Real resource admissions for the opt-in CPU projection fixture.
use super::*;

impl ControlledExecutor {
    /// The CPU logits fill does not allocate the core fixture's tensor arenas.
    /// Prepare and release its actual Step and whole-plan Invocation resources
    /// once, before any query snapshot. Future projection remains read-only and
    /// must fit the resulting real resident pool; this creates no cost sample.
    pub(in crate::continuous_engine::inner) fn prepare_structured_query_resources(&self) {
        let physical_before = self.physical.load(Ordering::Acquire);
        let batch =
            vnext::ExecutionBatchParticipants::new(self.evidence.sessions.snapshot()).unwrap();
        let work = || vec![contract::one_token_span(); batch.sessions().len()];
        let request = vnext::StepResourceAdmissionRequest::new(
            batch.bind_work_shape(work()).unwrap(),
            vnext::AdmissionFitPolicy::ImmediateOnly,
            vnext::AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
        let lane = self.evidence.lane.as_ref().unwrap();
        let mut prepared_step = None;
        for attempt in 0..=3 {
            match batch.try_begin_step(request.clone(), lane).unwrap() {
                vnext::StepResourceAdmissionDecision::Admitted(step) => {
                    prepared_step = Some(step);
                    break;
                }
                vnext::StepResourceAdmissionDecision::BackingDeferred(deferred) if attempt < 3 => {
                    deferred.maintain().unwrap();
                }
                _ => panic!("CPU query fixture could not admit its real multirow Step resources"),
            }
        }
        let step = prepared_step.expect("bounded Step admission");
        let fixture = self.evidence.fixture.as_ref().unwrap();
        let requests = fixture
            .resolved
            .execution_plan()
            .payload()
            .nodes()
            .iter()
            .map(|node| {
                vnext::InvocationResourceAdmissionRequest::for_all_step_participants(
                    node.id().clone(),
                    step.bind_all_invocation_work_shape(work()).unwrap(),
                    vnext::AdmissionFitPolicy::ImmediateOnly,
                    vnext::AdmissionPressureAction::WaitForRelease,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let mut prepared_wave = None;
        for attempt in 0..=3 {
            match step.try_prepare_submission_wave(requests.clone()).unwrap() {
                vnext::StepSubmissionWaveAdmissionDecision::Prepared(wave) => {
                    prepared_wave = Some(wave);
                    break;
                }
                vnext::StepSubmissionWaveAdmissionDecision::BackingDeferred(deferred)
                    if attempt < 3 =>
                {
                    deferred.maintain().unwrap();
                }
                _ => panic!(
                    "CPU query fixture could not admit its real whole-plan Invocation resources"
                ),
            }
        }
        drop(prepared_wave.expect("bounded whole-plan admission"));
        step.try_retire_normal().unwrap();
        assert_eq!(self.physical.load(Ordering::Acquire), physical_before);
    }
}
