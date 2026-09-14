use super::*;
use checkpoint_fixture::{Fixture, Spec};
use ferrum_interfaces::vnext::*;

#[path = "../../../../../ferrum-interfaces/tests/vnext_resource_contract/support.rs"]
mod resource_fixture;

#[test]
fn prefix_restore_extension_retains_request_fit_without_reserving_future_tokens() {
    for (suffix, dependency, maximum_tokens) in [
        ("prefix", CheckpointInputDependency::ExactTokenPrefix, 12),
        ("whole", CheckpointInputDependency::EntireTokenInput, 60),
    ] {
        // The shared fixture's selected provider declares 64 KiB pages.
        // Use two prefix positions per supported page so restoring five
        // positions crosses a real physical-allocation boundary.
        let page_bytes = 65_536;
        let bytes_per_token = page_bytes / 2;
        let resident_target = 4 * page_bytes;
        let profile = vnext_core_contract::paged_storage_profile(page_bytes);
        let mut spec = Spec {
            device_id: Some(resource_fixture::id(format!(
                "device.restore-frontier.{suffix}"
            ))),
            profile,
            port_profile: profile,
            dependency,
            checkpoint_capacity: Some(CheckpointCapacityPolicy::new(1 << 20).unwrap()),
            ..Spec::default()
        };
        spec.states[0].tensor.dimensions = vec![bytes_per_token];
        spec.states[0].capacity_demand = StateCapacityDemand::TokenScaled {
            bytes_per_token,
            maximum_tokens: 64,
        };
        let fixture = Fixture::build(spec).unwrap();
        assert_eq!(fixture.layout().input_dependency(), dependency);
        let (mut driver, _) = resource_fixture::configured_driver(&fixture.plan, &[], &[]);
        let runtime = Arc::get_mut(&mut driver.runtime).unwrap();
        runtime.descriptor.dynamic_storage_profiles.insert(profile);
        runtime
            .descriptor
            .capabilities
            .insert(resource_fixture::id("capability.compute"));
        runtime.alternate_descriptor = runtime.descriptor.clone();
        let committed = resource_fixture::transaction(&fixture.plan, driver, suffix)
            .reserve()
            .unwrap()
            .commit()
            .unwrap();
        let controller = committed.maintenance_controller();
        for pool_id in controller.pool_ids() {
            controller.initialize_pool(pool_id).unwrap();
        }
        let state_pool = fixture
            .plan
            .payload()
            .memory()
            .dynamic_descriptors()
            .iter()
            .find(|descriptor| descriptor.base_resource_id().as_str() == "resource.state.0")
            .unwrap()
            .pool_id();
        let resident = controller
            .status()
            .unwrap()
            .pools()
            .iter()
            .find(|pool| pool.pool_id() == state_pool)
            .unwrap()
            .resident_bytes();
        // Five half-page prefix positions require three pages; the independent
        // 4-byte boundary value needs one more page. No future suffix or output
        // capacity is resident in this pool.
        assert!(resident <= resident_target);
        if resident < resident_target {
            controller
                .grow_pool(state_pool, resident_target - resident)
                .unwrap();
        }
        let root = match committed.into_plan_runtime() {
            Ok(root) => root,
            Err(failure) => panic!("fixture runtime handoff failed: {}", failure.error()),
        };
        let binding = root.trusted_runtime_binding().unwrap();
        let tokens = [11, 13, 17, 19, 23, 29, 31, 37, 41];
        let admitted_work = ResourceWorkShape::single(
            TokenSpanWork::from_token_ids_with_fit(&tokens, 0..1, maximum_tokens).unwrap(),
        )
        .unwrap();
        let RequestResourceAdmissionDecision::Admitted(request) = binding
            .try_admit_request(
                RequestResourceAdmissionRequest::new(
                    admitted_work.clone(),
                    AdmissionFitPolicy::FullInputMustFit,
                    AdmissionPressureAction::WaitForRelease,
                )
                .unwrap(),
                resource_fixture::id(format!("run.restore-frontier.{suffix}")),
                resource_fixture::id(format!("request.restore-frontier.{suffix}")),
            )
            .unwrap()
        else {
            panic!("the full request fit must remain admissible")
        };
        let SequenceResourceAdmissionDecision::Admitted(sequence) = request
            .try_admit_sequence(
                SequenceResourceAdmissionRequest::new(
                    admitted_work.clone(),
                    AdmissionFitPolicy::ImmediateOnly,
                    AdmissionPressureAction::WaitForRelease,
                )
                .unwrap(),
            )
            .unwrap()
        else {
            panic!("initial sequence backing must admit")
        };
        let session = sequence.open_session().unwrap();
        let identity = session.fingerprint().clone();
        let initial = root.dynamic_pool_status().unwrap();

        // The existing minimum pool fits this actual prefix but not the
        // complete prompt/output ceiling. No maintenance grows the pool here.
        let extension = SequenceResourceExtensionRequest::new(
            prefix_restore_extension_work(&tokens[..5]).unwrap(),
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
        let SequenceResourceExtensionDecision::Extended(backing) =
            session.try_ensure_backing_covers(extension).unwrap()
        else {
            panic!("restoring a resident prefix must not wait for future-token backing")
        };
        assert_eq!(backing.committed_tokens(), 5);
        assert_eq!(session.fingerprint(), &identity);
        assert_eq!(request.work_shape(), &admitted_work);
        assert_eq!(admitted_work.fit_tokens(), maximum_tokens as u64);
        let after = root.dynamic_pool_status().unwrap();
        for (before, after) in initial.pools().iter().zip(after.pools()) {
            assert_eq!(before.pool_id(), after.pool_id());
            assert_eq!(before.resident_bytes(), after.resident_bytes());
        }

        // A future extension still has to acquire its own capacity; the short
        // restore neither reserves it nor weakens the admitted request ceiling.
        let future = SequenceResourceExtensionRequest::new(
            ResourceWorkShape::single(
                TokenSpanWork::from_token_ids_with_fit(&tokens, 0..tokens.len(), maximum_tokens)
                    .unwrap(),
            )
            .unwrap(),
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
        assert!(matches!(
            session.try_ensure_backing_covers(future).unwrap(),
            SequenceResourceExtensionDecision::BackingDeferred(_)
        ));
        let beyond = SequenceResourceExtensionRequest::new(
            ResourceWorkShape::single(
                TokenSpanWork::from_token_ids_with_fit(&tokens, 0..1, maximum_tokens + 1).unwrap(),
            )
            .unwrap(),
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
        assert!(session.try_ensure_backing_covers(beyond).is_err());
        drop(backing);
        session.try_abort_if_quiescent().unwrap();
        drop(session);
        drop(sequence);
        drop(request);
        drop(binding);
        resource_fixture::close_plan_runtime(root);
    }
}
