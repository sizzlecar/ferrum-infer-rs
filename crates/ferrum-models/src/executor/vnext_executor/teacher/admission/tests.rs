//! Teacher branch tests using the real core allocator, never fabricated
//! growth receipts or a second capacity ledger.
use super::*;
use ferrum_interfaces::vnext::{
    CapacityShortfallKind, InitialSequenceResourceAdmissionDecision,
    RequestResourceAdmissionRequest, SequenceResourceAdmissionRequest,
};

#[path = "../../../../../../ferrum-interfaces/tests/vnext_device_operation_contract/mod.rs"]
mod contract;
use contract::{fixture_with_token_scaled_paged_state, logical_resources, Fixture, TestRuntime};

struct PressureFixture {
    fixture: Fixture,
    owners: Vec<Arc<AdmittedSequenceResources<TestRuntime>>>,
    work: ResourceWorkShape,
}

impl PressureFixture {
    fn new() -> Self {
        let fixture = fixture_with_token_scaled_paged_state();
        // The fixture declares 4 bytes/token and a 16-token ceiling. Its
        // State pool is first grown through its authentic fit demand. Live
        // one-token owners then leave fewer than the 64 bytes needed for fit.
        let tokens = vec![1_u32; 16];
        let work = ResourceWorkShape::single(TokenSpanWork::from_token_ids(&tokens, 0..1).unwrap())
            .unwrap();
        let mut harness = Self {
            fixture,
            owners: Vec::new(),
            work,
        };
        match harness.admit("prime-fit") {
            InitialSequenceResourceAdmissionDecision::Deferred(deferred) => {
                assert!(harness
                    .fixture
                    .plan_resources
                    .try_maintain_for_capacity_pressure(&deferred)
                    .unwrap()
                    .is_some());
            }
            InitialSequenceResourceAdmissionDecision::Admitted(owner) => drop(owner),
            _ => panic!("fixture fit priming must be a real logical admission or growth demand"),
        }
        let fixture = &harness.fixture;
        let status = fixture.plan_resources.dynamic_pool_status().unwrap();
        let domain = status
            .pools()
            .iter()
            .find(|pool| pool.contract().minimum_sequence_bytes() > 0)
            .unwrap()
            .domain_id();
        loop {
            let snapshot = fixture.plan_resources.dynamic_pool_status().unwrap();
            let capacity = snapshot
                .pools()
                .iter()
                .find(|d| d.domain_id() == domain)
                .unwrap();
            assert!(capacity.resident_bytes() >= 64);
            if capacity.free_bytes() < 64 {
                break;
            }
            assert!(harness.owners.len() + 1 < status.maximum_active_sequences() as usize);
            harness.owners.push(logical_resources(
                &fixture.plan_resources,
                "run.teacher-pressure",
                &format!("request.teacher-pressure-held-{}", harness.owners.len()),
            ));
        }
        harness
    }

    fn admit(&self, suffix: &str) -> InitialSequenceResourceAdmissionDecision<TestRuntime> {
        self.fixture
            .plan_resources
            .trusted_runtime_binding()
            .unwrap()
            .try_admit_initial_sequence(
                RequestResourceAdmissionRequest::new(
                    self.work.clone(),
                    AdmissionFitPolicy::FullInputMustFit,
                    AdmissionPressureAction::WaitForRelease,
                )
                .unwrap(),
                SequenceResourceAdmissionRequest::new(
                    self.work.clone(),
                    AdmissionFitPolicy::FullInputMustFit,
                    AdmissionPressureAction::WaitForRelease,
                )
                .unwrap(),
                contract::id("run.teacher-pressure"),
                contract::id(format!("request.teacher-pressure-{suffix}")),
            )
            .unwrap()
    }

    fn deferred(&self, suffix: &str) -> AdmissionDeferred {
        let InitialSequenceResourceAdmissionDecision::Deferred(deferred) = self.admit(suffix)
        else {
            panic!("the real occupied pool must defer the unchanged full-input fit");
        };
        assert_eq!(deferred.action(), DeferredAction::WaitForRelease);
        assert!(deferred.blockers().iter().any(|blocker| blocker.kind()
            == CapacityShortfallKind::FitAvailability
            && blocker.requested().get() <= blocker.current_total().get()
            && blocker.requested().get() > blocker.available().get()));
        deferred
    }

    fn close(self) {
        drop(self.owners);
        drop(self.fixture.registry);
        drop(self.fixture.impostor_registry);
        drop(self.fixture.runtime);
        assert!(matches!(
            PlanRuntimeResources::close(self.fixture.plan_resources),
            Ok(PlanRuntimeCloseOutcome::Closed(_))
        ));
    }
}

#[test]
fn teacher_pressure_grows_real_available_capacity_then_reprobes_without_claiming_fit() {
    let fixture = PressureFixture::new();
    let deferred = fixture.deferred("new-owner");
    let before = fixture
        .fixture
        .plan_resources
        .dynamic_pool_status()
        .unwrap();
    let mut maintenance = TeacherAdmissionPressureMaintenance::default();
    assert!(maintenance
        .try_maintain(&fixture.fixture.plan_resources, &deferred)
        .unwrap());
    assert_eq!(deferred.action(), DeferredAction::WaitForRelease);
    let after = fixture
        .fixture
        .plan_resources
        .dynamic_pool_status()
        .unwrap();
    assert!(after.epochs().capacity_epoch() > before.epochs().capacity_epoch());
    assert!(after.budget_claimed_bytes() > before.budget_claimed_bytes());
    assert_eq!(
        after.maximum_active_sequences(),
        before.maximum_active_sequences()
    );
    assert_eq!(
        deferred.available().active_sequences() as usize,
        fixture.owners.len()
    );
    for prior in before.pools() {
        let current = after
            .pools()
            .iter()
            .find(|pool| pool.pool_id() == prior.pool_id())
            .unwrap();
        assert_eq!(
            current.resident_bytes() - current.free_bytes(),
            prior.resident_bytes() - prior.free_bytes(),
            "maintenance must not acquire any of the owner's fit claims"
        );
    }
    let InitialSequenceResourceAdmissionDecision::Admitted(owner) = fixture.admit("new-owner")
    else {
        panic!("only the fresh authoritative probe can admit after real maintenance");
    };
    // Once this owner consumes its immediate claim, another full fit defers.
    // The same teacher attempt cannot keep allocating for repeated pressure.
    let next = fixture.deferred("another-owner");
    assert_eq!(
        next.available().active_sequences() as usize,
        fixture.owners.len() + 1
    );
    let allocations = fixture
        .fixture
        .runtime_trace
        .lock()
        .unwrap()
        .allocation_calls;
    assert!(!maintenance
        .try_maintain(&fixture.fixture.plan_resources, &next)
        .unwrap());
    assert_eq!(
        fixture
            .fixture
            .runtime_trace
            .lock()
            .unwrap()
            .allocation_calls,
        allocations
    );
    drop(owner);
    fixture.close();
}

#[test]
fn teacher_pressure_rechecks_real_release_without_allocating_stale_shortfall() {
    let mut fixture = PressureFixture::new();
    let deferred = fixture.deferred("released-owner");
    drop(fixture.owners.pop().unwrap());
    let allocations = fixture
        .fixture
        .runtime_trace
        .lock()
        .unwrap()
        .allocation_calls;
    let mut maintenance = TeacherAdmissionPressureMaintenance::default();
    assert!(maintenance
        .try_maintain(&fixture.fixture.plan_resources, &deferred)
        .unwrap());
    assert_eq!(
        fixture
            .fixture
            .runtime_trace
            .lock()
            .unwrap()
            .allocation_calls,
        allocations
    );
    let InitialSequenceResourceAdmissionDecision::Admitted(owner) = fixture.admit("released-owner")
    else {
        panic!("real release must allow a fresh probe without fabricated growth");
    };
    drop(owner);
    fixture.close();
}

#[test]
fn teacher_pressure_rejects_foreign_authority_without_allocating_or_retrying() {
    let source = PressureFixture::new();
    let deferred = source.deferred("foreign-owner");
    let target = fixture_with_token_scaled_paged_state();
    let before = target.plan_resources.dynamic_pool_status().unwrap();
    let allocations = target.runtime_trace.lock().unwrap().allocation_calls;
    let mut maintenance = TeacherAdmissionPressureMaintenance::default();
    assert!(maintenance
        .try_maintain(&target.plan_resources, &deferred)
        .is_err());
    assert!(!maintenance
        .try_maintain(&target.plan_resources, &deferred)
        .unwrap());
    assert_eq!(target.plan_resources.dynamic_pool_status().unwrap(), before);
    assert_eq!(
        target.runtime_trace.lock().unwrap().allocation_calls,
        allocations
    );
    drop(target.registry);
    drop(target.impostor_registry);
    drop(target.runtime);
    assert!(matches!(
        PlanRuntimeResources::close(target.plan_resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
    source.close();
}
