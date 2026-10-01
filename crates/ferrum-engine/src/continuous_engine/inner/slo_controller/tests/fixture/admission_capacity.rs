//! Opt-in real allocator admission for controller tests that submit no model
//! work. Capacity, wait evidence and release epochs come from owned leases.
use super::*;

pub(super) struct RealAdmission {
    fixture: Option<contract::Fixture>,
    leases: std::collections::HashMap<
        RequestId,
        Arc<vnext::AdmittedSequenceResources<contract::TestRuntime>>,
    >,
}

impl RealAdmission {
    fn new(capacity: u32) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let device = vnext::DeviceId::new(format!(
            "device.controller-admission.{}",
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
        .unwrap();
        let fixture = contract::fixture_with_device_and_sequence_capacity(device, Some(capacity));
        assert_eq!(
            fixture
                .plan_resources
                .dynamic_pool_status()
                .unwrap()
                .maximum_active_sequences(),
            capacity
        );
        Self {
            fixture: Some(fixture),
            leases: Default::default(),
        }
    }

    pub(super) fn epochs(
        &self,
        sources: &mut Vec<vnext::CapacityAvailabilityEpoch>,
    ) -> ExecutorAdmissionEpochs {
        let epoch = self
            .fixture
            .as_ref()
            .unwrap()
            .plan_resources
            .write_dynamic_capacity_availability(sources)
            .unwrap();
        ExecutorAdmissionEpochs::new(
            NonZeroU64::new(epoch.coordinator_id().get()).unwrap(),
            epoch.release_epoch(),
            epoch.capacity_epoch(),
        )
    }

    pub(super) fn admit(
        &mut self,
        input: ExecutorPrefillAdmission<'_>,
    ) -> ExecutorPrefillAdmissionDecision {
        assert!(
            !self.leases.contains_key(input.request_id),
            "duplicate allocator admission"
        );
        let tokens = input
            .input_tokens
            .iter()
            .map(|token| token.get())
            .collect::<Vec<_>>();
        let span = vnext::TokenSpanWork::from_token_ids_with_fit(
            &tokens,
            0..tokens.len(),
            input.maximum_sequence_tokens,
        )
        .unwrap();
        let work = vnext::ResourceWorkShape::single(span).unwrap();
        let request = vnext::RequestResourceAdmissionRequest::new(
            work.clone(),
            vnext::AdmissionFitPolicy::ImmediateOnly,
            vnext::AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
        let sequence = vnext::SequenceResourceAdmissionRequest::new(
            work,
            vnext::AdmissionFitPolicy::ImmediateOnly,
            vnext::AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
        let binding = self
            .fixture
            .as_ref()
            .unwrap()
            .plan_resources
            .trusted_runtime_binding()
            .unwrap();
        // Cold fixture backing may need real maintenance; each retry follows
        // a bounded allocator-issued maintenance result, never a fabricated
        // admission or an unbounded retry after logical capacity deferral.
        let mut maintenance_attempts = 0;
        loop {
            match binding
                .try_admit_initial_sequence(
                    request.clone(),
                    sequence.clone(),
                    vnext::RunId::new("run.controller-admission").unwrap(),
                    vnext::RequestIdentity::new(input.request_id.to_string()).unwrap(),
                )
                .unwrap()
            {
                vnext::InitialSequenceResourceAdmissionDecision::Admitted(lease) => {
                    self.leases.insert(input.request_id.clone(), lease);
                    return ExecutorPrefillAdmissionDecision::Admitted(
                        ExecutorPrefillAdmissionReceipt {
                            request_id: input.request_id.clone(),
                        },
                    );
                }
                vnext::InitialSequenceResourceAdmissionDecision::Deferred(value) => {
                    return ExecutorPrefillAdmissionDecision::Deferred(value);
                }
                vnext::InitialSequenceResourceAdmissionDecision::PermanentRejected(value) => {
                    return ExecutorPrefillAdmissionDecision::PermanentRejected(value);
                }
                vnext::InitialSequenceResourceAdmissionDecision::BackingDeferred(value) => {
                    assert!(
                        maintenance_attempts < 3,
                        "bounded allocator fixture maintenance"
                    );
                    maintenance_attempts += 1;
                    value.maintain().unwrap();
                }
            }
        }
    }

    pub(super) fn cancel(&mut self, id: &RequestId) -> bool {
        self.leases.remove(id).is_some()
    }
}

impl Drop for RealAdmission {
    fn drop(&mut self) {
        let leaked = !self.leases.is_empty();
        self.leases.clear();
        if !std::thread::panicking() {
            assert!(!leaked, "test retained actual admission leases");
        }
        if let Some(fixture) = self.fixture.take() {
            drop(fixture.registry);
            drop(fixture.impostor_registry);
            drop(fixture.runtime);
            assert!(matches!(
                vnext::PlanRuntimeResources::close(fixture.plan_resources),
                Ok(vnext::PlanRuntimeCloseOutcome::Closed(_))
            ));
        }
    }
}

impl ControlledExecutor {
    pub(in crate::continuous_engine::inner::slo_controller) fn enable_real_admission_capacity(
        &self,
        capacity: u32,
    ) {
        assert!(self
            .admission_capacity
            .lock()
            .replace(RealAdmission::new(capacity))
            .is_none());
    }

    pub(in crate::continuous_engine::inner::slo_controller) fn retained_admission_owners(
        &self,
    ) -> usize {
        self.admission_capacity
            .lock()
            .as_ref()
            .unwrap()
            .leases
            .len()
    }
}
