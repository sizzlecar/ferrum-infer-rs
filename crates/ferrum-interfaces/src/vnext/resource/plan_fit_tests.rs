use super::*;
use crate::vnext::CapacityShortfallKind;

fn initial_admission(
    binding: &TrustedPlanRuntimeBinding<TestRuntime>,
) -> InitialSequenceResourceAdmissionDecision<TestRuntime> {
    binding
        .try_admit_initial_sequence(
            RequestResourceAdmissionRequest::new(
                work_with_ceiling(1, 3),
                AdmissionFitPolicy::FullInputMustFit,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap(),
            SequenceResourceAdmissionRequest::new(
                chunked_work(3, 0..1),
                AdmissionFitPolicy::FullInputMustFit,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap(),
            RunId::new("run/plan-fit").unwrap(),
            RequestIdentity::new("request/plan-fit").unwrap(),
        )
        .unwrap()
}

fn two_state_pools_with_workspace() -> PoolCatalog {
    combine_catalogs(&[
        pool_catalog(
            paged_profile(),
            AllocationLifetime::Sequence,
            'a',
            1,
            256,
            TestDemand::Tokens,
        ),
        pool_catalog(
            paged_profile(),
            AllocationLifetime::Sequence,
            'b',
            1,
            256,
            TestDemand::Tokens,
        ),
        pool_catalog(
            linear_profile(),
            AllocationLifetime::Step,
            'c',
            1,
            64,
            TestDemand::Fixed,
        ),
    ])
}

fn assert_plan_budget(rejected: &AdmissionRejected, required: u64, maximum: u64) {
    assert_eq!(rejected.blockers().len(), 1);
    let blocker = &rejected.blockers()[0];
    assert_eq!(blocker.kind(), CapacityShortfallKind::PermanentPlanBudget);
    assert_eq!(blocker.domain(), None);
    assert_eq!(blocker.requested().get(), required);
    assert_eq!(blocker.maximum_total().get(), maximum);
}

#[test]
fn initial_bundle_rejects_joint_fit_before_allocating_or_creating_an_owner() {
    let catalog = two_state_pools_with_workspace();
    let runtime = new_runtime(&catalog, 384);
    let h = harness(Arc::clone(&runtime), catalog, 384, false);
    let binding = h.root.trusted_runtime_binding().unwrap();
    let before = h.root.maintenance_controller.status().unwrap();
    let InitialSequenceResourceAdmissionDecision::PermanentRejected(rejected) =
        initial_admission(&binding)
    else {
        panic!("two 192-byte states plus an unavoidable 64-byte workspace cannot fit")
    };
    assert_plan_budget(&rejected, 448, 384);
    assert_eq!(runtime.allocate_calls(), 0);
    assert_eq!(h.root.maintenance_controller.status().unwrap(), before);
    let logical = h.root.dynamic_pools.logical_admission.snapshot().unwrap();
    assert_eq!(logical.active_requests(), 0);
    assert_eq!(logical.active_sequences(), 0);
    drop(binding);
    close_dynamic_test_root(h.root);
}

#[test]
fn exact_joint_fit_remains_admissible_without_reserving_full_input() {
    let catalog = two_state_pools_with_workspace();
    let runtime = new_runtime(&catalog, 448);
    let h = harness(runtime, catalog, 448, false);
    h.root
        .maintenance_controller
        .initialize_pools(&h.pool_ids)
        .unwrap();
    let binding = h.root.trusted_runtime_binding().unwrap();
    let InitialSequenceResourceAdmissionDecision::Deferred(deferred) = initial_admission(&binding)
    else {
        panic!("an exact fit must request ordinary backing growth")
    };
    assert!(matches!(
        h.root.maintain_for_admission_deferred(&deferred).unwrap(),
        DynamicDeferredMaintenanceOutcome::Maintained(_)
    ));
    let InitialSequenceResourceAdmissionDecision::Admitted(sequence) = initial_admission(&binding)
    else {
        panic!("materializing the exact fit must make admission succeed")
    };
    let status = h.root.maintenance_controller.status().unwrap();
    assert_eq!(status.budget_claimed_bytes(), 448);
    assert_eq!(
        status
            .pools()
            .iter()
            .map(|pool| pool.live_occupancy().total().physical_bytes())
            .sum::<u64>(),
        128
    );
    drop(sequence);
    drop(binding);
    close_dynamic_test_root(h.root);
}

#[test]
fn child_fit_includes_its_retained_request_once() {
    let request = pool_catalog(
        paged_profile(),
        AllocationLifetime::Request,
        'a',
        1,
        256,
        TestDemand::Tokens,
    );
    let sequence = pool_catalog(
        paged_profile(),
        AllocationLifetime::Sequence,
        'b',
        1,
        256,
        TestDemand::Tokens,
    );
    let request_id = request.pool_id.clone();
    let catalog = combine_catalogs(&[request, sequence]);
    let runtime = new_runtime(&catalog, 320);
    let h = harness(Arc::clone(&runtime), catalog, 320, false);
    h.root
        .maintenance_controller
        .initialize_pools(&h.pool_ids)
        .unwrap();
    h.root
        .maintenance_controller
        .grow_pool(&request_id, 128)
        .unwrap();
    let binding = h.root.trusted_runtime_binding().unwrap();
    let RequestResourceAdmissionDecision::Admitted(request) = binding
        .try_admit_request(
            RequestResourceAdmissionRequest::new(
                work_with_ceiling(1, 3),
                AdmissionFitPolicy::FullInputMustFit,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap(),
            RunId::new("run/retained-parent-fit").unwrap(),
            RequestIdentity::new("request/retained-parent-fit").unwrap(),
        )
        .unwrap()
    else {
        panic!("the 192-byte parent plus 64-byte sequence minimum must fit")
    };
    let before = h.root.maintenance_controller.status().unwrap();
    let allocations_before = runtime.allocate_calls();
    let SequenceResourceAdmissionDecision::PermanentRejected(rejected) = request
        .try_admit_sequence(
            SequenceResourceAdmissionRequest::new(
                chunked_work(3, 0..1),
                AdmissionFitPolicy::FullInputMustFit,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap(),
        )
        .unwrap()
    else {
        panic!("the child's 192-byte fit must include its retained 192-byte parent")
    };
    assert_plan_budget(&rejected, 384, 320);
    assert_eq!(runtime.allocate_calls(), allocations_before);
    assert_eq!(h.root.maintenance_controller.status().unwrap(), before);
    // A smaller child is possible. Its parent's existing claim must not be
    // charged a second time or confused with current free device bytes.
    let SequenceResourceAdmissionDecision::Admitted(child) = request
        .try_admit_sequence(
            SequenceResourceAdmissionRequest::new(
                work(1),
                AdmissionFitPolicy::FullInputMustFit,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap(),
        )
        .unwrap()
    else {
        panic!("a 64-byte child plus its once-counted parent must still fit")
    };
    drop(child);
    drop(request);
    drop(binding);
    close_dynamic_test_root(h.root);
}

#[test]
fn standalone_request_accounts_for_other_runnable_pool_minima() {
    let catalog = combine_catalogs(&[
        pool_catalog(
            paged_profile(),
            AllocationLifetime::Request,
            'a',
            1,
            256,
            TestDemand::Tokens,
        ),
        pool_catalog(
            linear_profile(),
            AllocationLifetime::Step,
            'b',
            1,
            128,
            TestDemand::Fixed,
        ),
    ]);
    let runtime = new_runtime(&catalog, 224);
    let h = harness(Arc::clone(&runtime), catalog, 224, false);
    let binding = h.root.trusted_runtime_binding().unwrap();
    let RequestResourceAdmissionDecision::PermanentRejected(rejected) = binding
        .try_admit_request(
            RequestResourceAdmissionRequest::new(
                work_with_ceiling(1, 3),
                AdmissionFitPolicy::FullInputMustFit,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap(),
            RunId::new("run/request-plan-fit").unwrap(),
            RequestIdentity::new("request/request-plan-fit").unwrap(),
        )
        .unwrap()
    else {
        panic!("the request alone fits, but its plan's unavoidable workspace does not")
    };
    assert_plan_budget(&rejected, 256, 224);
    assert_eq!(runtime.allocate_calls(), 0);
    drop(binding);
    close_dynamic_test_root(h.root);
}

#[test]
fn another_plans_temporary_process_ceiling_is_not_a_permanent_fit_limit() {
    let catalog = combine_catalogs(&[
        pool_catalog(
            paged_profile(),
            AllocationLifetime::Sequence,
            'a',
            1,
            256,
            TestDemand::Tokens,
        ),
        pool_catalog(
            paged_profile(),
            AllocationLifetime::Sequence,
            'b',
            1,
            256,
            TestDemand::Tokens,
        ),
    ]);
    let runtime = new_runtime(&catalog, 384);
    let h = harness(runtime, catalog, 384, false);
    h.root
        .maintenance_controller
        .initialize_pools(&h.pool_ids)
        .unwrap();
    let other_budget = h
        .root
        .dynamic_pools
        .budget
        .account
        .register_budget(256)
        .unwrap();
    let binding = h.root.trusted_runtime_binding().unwrap();
    let InitialSequenceResourceAdmissionDecision::Deferred(deferred) = initial_admission(&binding)
    else {
        panic!("the complete fit fits this plan; another plan's ceiling is temporary")
    };
    let DynamicDeferredMaintenanceOutcome::WaitForRelease { wait_condition, .. } =
        h.root.maintain_for_admission_deferred(&deferred).unwrap()
    else {
        panic!("the competing plan's process ceiling must still be enforced physically")
    };
    let waiter = h.root.register_capacity_waiter(&wait_condition).unwrap();
    assert!(!waiter.recheck().unwrap().should_retry());
    drop(other_budget);
    assert!(waiter.recheck().unwrap().should_retry());
    assert!(matches!(
        h.root.maintain_for_admission_deferred(&deferred).unwrap(),
        DynamicDeferredMaintenanceOutcome::Maintained(_)
    ));
    let InitialSequenceResourceAdmissionDecision::Admitted(sequence) = initial_admission(&binding)
    else {
        panic!("removing the other plan must allow this exact fit to run")
    };
    drop(sequence);
    drop(waiter);
    drop(binding);
    close_dynamic_test_root(h.root);
}

#[test]
fn per_domain_maximum_remains_the_specific_rejection() {
    let catalog = pool_catalog(
        paged_profile(),
        AllocationLifetime::Sequence,
        'a',
        1,
        128,
        TestDemand::Tokens,
    );
    let runtime = new_runtime(&catalog, 128);
    let h = harness(Arc::clone(&runtime), catalog, 128, false);
    let binding = h.root.trusted_runtime_binding().unwrap();
    let InitialSequenceResourceAdmissionDecision::PermanentRejected(rejected) =
        initial_admission(&binding)
    else {
        panic!("a per-domain maximum violation must remain permanently rejected")
    };
    assert_eq!(
        rejected.blockers()[0].kind(),
        CapacityShortfallKind::PermanentDomainMaximum
    );
    assert_eq!(runtime.allocate_calls(), 0);
    drop(binding);
    close_dynamic_test_root(h.root);
}
