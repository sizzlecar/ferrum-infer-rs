//! Poll work, rather than machine time, bounds a real coalesced-slot capture.
use super::*;

fn capture_poll_count(projections: usize) -> usize {
    capture_measurements(projections, 1)[0].1
}

fn capture_measurements(projections: usize, captures: usize) -> Vec<(u128, usize)> {
    resident_measurements(projections, captures, false)
}

fn resident_measurements(
    projections: usize,
    captures: usize,
    measure_projection: bool,
) -> Vec<(u128, usize)> {
    let catalog = pool_catalog_with_options(
        linear_profile(),
        AllocationLifetime::Step,
        'b',
        projections,
        256,
        TestDemand::Tokens,
        "activations",
        true,
        StateInitialization::None,
    );
    let (memory, bucket) = reusable_step_memory_plan(catalog.pool_id.clone());
    let harness = harness_with_reusable(new_runtime(&catalog, 256), catalog, 256, memory);
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 256)
        .unwrap();
    let lane = harness.root.create_execution_lane().unwrap();
    let sequence = admitted_sequence_with_ceiling(&harness.root, "linear-capture", 4);
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let actual = step(&batch, &lane, 1, Some(&bucket));
    assert_eq!(actual.backing_slices().len(), projections);
    for slice in actual.backing_slices() {
        assert!(slice
            .evidence()
            .physical_claim_identity()
            .shares_resource_id_storage(&slice.segment_lease.claim_identity));
    }
    actual.try_retire_normal().unwrap();
    let allocations = harness.runtime.allocate_calls();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let mut measurements = Vec::new();
    let mut last_view = None;
    for _ in 0..captures {
        let (view, polls, wall_ns) = loop {
            let mut polls = 0;
            let began = std::time::Instant::now();
            let captured = harness.root.resource_planning_view_on_lane(
                &[&session],
                &lane,
                ResourcePlanningLimits::default(),
                &mut || {
                    polls += 1;
                    true
                },
            );
            match captured {
                ResourcePlanningAvailability::Known(view) => {
                    break (view, polls, began.elapsed().as_nanos())
                }
                ResourcePlanningAvailability::Unknown(
                    ResourcePlanningUnknown::ReadUnavailable(_),
                ) if std::time::Instant::now() < deadline => std::thread::yield_now(),
                other => panic!("actual shared-claim capture failed: {other:?}"),
            }
        };
        if measure_projection {
            let initial = view.initial_state();
            let mut projection_polls = 0;
            let began = std::time::Instant::now();
            let projected = known(harness.root.project_resource_wave_with_bucket(
                &view,
                &initial,
                &[row(0, 0, 1)],
                Some(bucket.bucket_id()),
                &mut || {
                    projection_polls += 1;
                    true
                },
            ));
            measurements.push((began.elapsed().as_nanos(), projection_polls));
            assert!(projected.selected_step_slot().is_some());
        } else {
            measurements.push((wall_ns, polls));
        }
        last_view = Some(view);
    }
    assert_eq!(harness.runtime.allocate_calls(), allocations);
    session.try_abort_if_quiescent().unwrap();
    drop(batch);
    drop(session);
    drop(sequence);
    drop(lane);
    close_dynamic_test_root(harness.root);
    assert_eq!(
        last_view.unwrap().participants().len(),
        1,
        "numeric evidence pins no lease"
    );
    measurements
}

/// Explicit local CPU diagnostic, not a CUDA/Metal or serving benchmark.
/// All snapshots use an actually allocated and retired resident slot.
#[test]
#[ignore = "local capture timing diagnostic; no timing assertion"]
fn planning_workspace_resident_capture_timing() {
    for projections in [16, 128, 512] {
        let mut measurements = capture_measurements(projections, 101);
        let (first_ns, first_polls) = measurements.remove(0);
        measurements.sort_unstable_by_key(|m| m.0);
        eprintln!("resident_capture projections={projections} first_ns={first_ns} first_polls={first_polls} warm_samples={} warm_p50_ns={} warm_p99_ns={} warm_polls={}", measurements.len(), measurements[49].0, measurements[98].0, measurements[49].1);
    }
}

#[test]
#[ignore = "local full resource projection timing diagnostic; no timing assertion"]
fn planning_workspace_resident_projection_timing() {
    for projections in [16, 128, 512] {
        let mut measurements = resident_measurements(projections, 101, true);
        let (first_ns, first_polls) = measurements.remove(0);
        measurements.sort_unstable_by_key(|m| m.0);
        eprintln!("resident_projection projections={projections} first_ns={first_ns} first_polls={first_polls} warm_samples={} warm_p50_ns={} warm_p99_ns={} warm_polls={}", measurements.len(), measurements[49].0, measurements[98].0, measurements[49].1);
    }
}

#[test]
fn planning_workspace_shared_claim_capture_poll_work_scales_linearly() {
    let small = capture_poll_count(16);
    let large = capture_poll_count(64);
    assert!(large > small, "each real projection still needs validation");
    assert!(
        large <= small * 4,
        "four times as many shared projections must not cause quadratic identity scans: {small} -> {large}"
    );
}

#[test]
fn compiled_step_layout_preserves_dynamic_fit_and_physical_capacity() {
    let catalog = pool_catalog_with_options(
        linear_profile(),
        AllocationLifetime::Step,
        'b',
        16,
        256,
        TestDemand::Tokens,
        "activations",
        true,
        StateInitialization::None,
    );
    let (memory, bucket) = reusable_step_memory_plan(catalog.pool_id.clone());
    let harness = harness_with_reusable(new_runtime(&catalog, 256), catalog, 256, memory);
    let binding = harness.root.trusted_runtime_binding().unwrap();
    for (immediate, fit) in [(1, 3), (2, 4), (4, 4), (1, 1)] {
        for reusable in [None, Some(&bucket)] {
            let (demand, requests) = binding
                .scoped_demand(
                    AllocationLifetime::Step,
                    None,
                    DynamicResourceShape::from_validated(1, immediate, 0),
                    DynamicResourceShape::from_validated(1, fit, 0),
                    reusable,
                    AdmissionFitPolicy::FullInputMustFit,
                    AdmissionPressureAction::WaitForRelease,
                )
                .unwrap();
            assert_eq!(
                demand.immediate_claim().entries()[0].units().get(),
                immediate * 64
            );
            assert_eq!(
                demand.fit_requirement().entries()[0].units().get(),
                fit * 64
            );
            assert_eq!(
                requests.len(),
                1,
                "shared projections consume one physical slot"
            );
            let request = &requests[0];
            let physical = if reusable.is_some() {
                256
            } else {
                immediate * 64
            };
            assert_eq!(request.capacity_size_bytes, physical);
            assert_eq!(request.claim_identity.resource_ids().len(), 16);
            assert_eq!(request.projections.len(), 16);
            for projection in &request.projections {
                assert_eq!(projection.logical_size_bytes, immediate * 64);
                assert_eq!(projection.capacity_size_bytes, physical);
                assert_eq!(projection.physical_offset_bytes, 0);
            }
            if reusable.is_some() {
                assert_compiled_layout_identity(
                    &requests,
                    AllocationLifetime::Step,
                    harness
                        .root
                        .dynamic_pools
                        .lane_stable_layouts
                        .get(Some(bucket.bucket_id()), AllocationLifetime::Step)
                        .unwrap(),
                );
            }
        }
    }
    assert!(
        binding
            .scoped_demand(
                AllocationLifetime::Step,
                None,
                DynamicResourceShape::from_validated(1, 1, 0),
                DynamicResourceShape::from_validated(1, 5, 0),
                Some(&bucket),
                AdmissionFitPolicy::FullInputMustFit,
                AdmissionPressureAction::WaitForRelease,
            )
            .is_err(),
        "fit outside the real bucket remains rejected"
    );
    drop(binding);
    close_dynamic_test_root(harness.root);
}

// Compare complete identity with the original encoder for both Step and
// Invocation layouts. Mutations exercise fields on later projections too;
// an old fingerprint must never hide a changed physical resource contract.
pub(super) fn assert_compiled_layout_identity(
    requests: &[EvaluatedBackingRequest<'_>],
    lifetime: AllocationLifetime,
    compiled: &crate::vnext::resource::lane_stable_arena::CompiledLaneStableLayout,
) {
    let lane = ExecutionLaneId::mint().unwrap();
    let key = |requests: &[EvaluatedBackingRequest<'_>], use_compiled: bool| {
        let mut canonical: Vec<_> = requests.iter().collect();
        canonical.sort_unstable_by(|a, b| a.claim_identity.cmp(&b.claim_identity));
        lane_stable_layout_key(lane, lifetime, &canonical, use_compiled.then_some(compiled))
            .unwrap()
    };
    let original = key(requests, false);
    assert_eq!(key(requests, true), original);
    let mut dynamic = requests.to_vec();
    for request in &mut dynamic {
        for projection in &mut request.projections {
            projection.logical_size_bytes = 1;
        }
    }
    assert_eq!(
        key(&dynamic, true),
        original,
        "logical demand is checked separately"
    );
    for field in 0..6 {
        let mut changed = requests.to_vec();
        let request = changed.last_mut().unwrap();
        match field {
            0 => request.capacity_size_bytes += 16,
            1 => {
                request
                    .projections
                    .last_mut()
                    .unwrap()
                    .physical_offset_bytes += 16
            }
            2 => request.projections.last_mut().unwrap().capacity_size_bytes += 16,
            3 => {
                let last = request.projections.len() - 1;
                assert!(last > 0);
                request.projections.swap(0, last);
            }
            4 => {
                let mut ids = request.claim_identity.resource_ids().to_vec();
                *ids.last_mut().unwrap() =
                    ResourceId::new("resource/changed-physical-layout").unwrap();
                request.claim_identity = PhysicalBackingClaimIdentity::new(
                    request.claim_identity.pool_id().clone(),
                    ids,
                )
                .unwrap();
            }
            _ => {
                request.projections.pop();
            }
        }
        let independent = key(&changed, false);
        assert_ne!(
            independent, original,
            "changed physical identity field {field}"
        );
        assert_eq!(
            key(&changed, true),
            independent,
            "compiled layout cannot mask field {field}"
        );
    }
    let refs: Vec<_> = requests.iter().collect();
    let different_lane = ExecutionLaneId::mint().unwrap();
    let other = lane_stable_layout_key(different_lane, lifetime, &refs, Some(compiled)).unwrap();
    assert_ne!(other, original);
    assert_eq!(other.layout_fingerprint, original.layout_fingerprint);
}
