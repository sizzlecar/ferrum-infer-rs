mod vnext_device_operation_contract;
mod vnext_device_operation_wave_contract;

use vnext_device_operation_contract::*;
use vnext_device_operation_wave_contract::prepare_wave;

const CHILD_SEQUENCE_COUNT: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
// These are resource-shape proxies; provider math remains the same so the
// lifecycle implementation is the only independent variable.
enum SyntheticLifecycleProfile {
    Dense,
    Moe,
    Hybrid,
}

impl SyntheticLifecycleProfile {
    const ALL: [Self; 3] = [Self::Dense, Self::Moe, Self::Hybrid];

    fn fixture(self) -> Fixture {
        match self {
            Self::Dense => fixture_with_token_scaled_paged_state_and_provider_behavior(
                ProviderBehavior::Success,
            ),
            Self::Moe => fixture_with_token_scaled_paged_state_and_provider_behavior(
                ProviderBehavior::ScratchOverwrite,
            ),
            Self::Hybrid => {
                fixture_with_hybrid_state_and_provider_behavior(ProviderBehavior::ScratchOverwrite)
            }
        }
    }

    const fn has_invocation_scratch(self) -> bool {
        matches!(self, Self::Moe | Self::Hybrid)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum LifecycleStage {
    Plan,
    Request,
    ChildSequence(usize),
    Step,
    InvocationPrepared,
    InFlight,
    FenceTerminal,
    StepRetired,
    RequestRetained,
    Empty,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
struct ScopeOccupancy {
    total: u64,
    total_bytes: u64,
    plan: u64,
    plan_bytes: u64,
    request: u64,
    request_bytes: u64,
    sequence: u64,
    sequence_bytes: u64,
    step: u64,
    step_bytes: u64,
    invocation: u64,
    invocation_bytes: u64,
    initial_sequence_bundle: u64,
    initial_sequence_bundle_bytes: u64,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct LifecycleSnapshot {
    stage: LifecycleStage,
    occupancy: ScopeOccupancy,
}

#[derive(Debug, Serialize)]
struct LifecycleAnalysis {
    profile: SyntheticLifecycleProfile,
    contract_shape: ProfileContractShape,
    snapshots: Vec<LifecycleSnapshot>,
    per_child_sequence_claims: u64,
    per_child_sequence_bytes: u64,
    invocation_peak_claims: u64,
    invocation_peak_bytes: u64,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct ProfileContractShape {
    sequence_values: BTreeSet<ResourceId>,
    invocation_scratch_resources: usize,
    minimum_invocation_peak_bytes: u64,
}

fn assert_profile_contract(
    root: &PlanRuntimeResources<TestRuntime>,
    profile: SyntheticLifecycleProfile,
) -> ProfileContractShape {
    let status = root.dynamic_pool_status().unwrap();
    let mut sequence_values = BTreeSet::new();
    let mut invocation_scratch_resources = 0;
    let mut minimum_invocation_peak_bytes = 0;
    for pool in status.pools() {
        minimum_invocation_peak_bytes += pool.contract().minimum_invocation_peak_bytes();
        for resource in pool.contract().resources() {
            if resource.lifetime() == AllocationLifetime::Sequence
                && resource.kind() == &AllocationKind::Value
            {
                sequence_values.insert(resource.resource_id().clone());
            }
            if matches!(resource.kind(), AllocationKind::Scratch { .. }) {
                assert_eq!(resource.lifetime(), AllocationLifetime::Invocation);
                invocation_scratch_resources += 1;
            }
            match resource.resource_id().as_str() {
                "resource.state" => {
                    assert_eq!(resource.lifetime(), AllocationLifetime::Sequence);
                    assert_eq!(resource.kind(), &AllocationKind::Value);
                    assert!(matches!(
                        resource.demand(),
                        DynamicResourceDemand::Tokens {
                            bytes_per_token: 4,
                            maximum_tokens: 16,
                        }
                    ));
                    assert_eq!(pool.storage_profile(), paged_storage_profile());
                }
                "resource.recurrent-state" => {
                    assert_eq!(resource.lifetime(), AllocationLifetime::Sequence);
                    assert_eq!(resource.kind(), &AllocationKind::Value);
                    assert!(matches!(
                        resource.demand(),
                        DynamicResourceDemand::Fixed { bytes: 4 }
                    ));
                    assert_eq!(pool.storage_profile(), contiguous_storage_profile());
                }
                _ => {}
            }
        }
    }
    let mut expected_sequence_values = BTreeSet::from([id("resource.state")]);
    if profile == SyntheticLifecycleProfile::Hybrid {
        expected_sequence_values.insert(id("resource.recurrent-state"));
    }
    assert_eq!(sequence_values, expected_sequence_values);
    assert_eq!(
        invocation_scratch_resources,
        if profile.has_invocation_scratch() {
            2
        } else {
            0
        },
        "the two-node MoE resource proxy declares one scratch formula per node"
    );
    assert_eq!(
        minimum_invocation_peak_bytes,
        if profile.has_invocation_scratch() {
            16
        } else {
            0
        },
        "sequential node scratch must plan max(node), not sum(nodes)"
    );
    ProfileContractShape {
        sequence_values,
        invocation_scratch_resources,
        minimum_invocation_peak_bytes,
    }
}

fn occupancy(root: &PlanRuntimeResources<TestRuntime>) -> ScopeOccupancy {
    let status = root.dynamic_pool_status().unwrap();
    let mut summary = ScopeOccupancy::default();
    for pool in status.pools() {
        let live = pool.live_occupancy();
        summary.total += live.total().claim_count();
        summary.total_bytes += live.total().physical_bytes();
        for residency in [live.transient(), live.lane_stable()] {
            summary.plan += residency.plan().claim_count();
            summary.plan_bytes += residency.plan().physical_bytes();
            summary.request += residency.request().claim_count();
            summary.request_bytes += residency.request().physical_bytes();
            summary.sequence += residency.sequence().claim_count();
            summary.sequence_bytes += residency.sequence().physical_bytes();
            summary.step += residency.step().claim_count();
            summary.step_bytes += residency.step().physical_bytes();
            summary.invocation += residency.invocation().claim_count();
            summary.invocation_bytes += residency.invocation().physical_bytes();
            summary.initial_sequence_bundle += residency.initial_sequence_bundle().claim_count();
            summary.initial_sequence_bundle_bytes +=
                residency.initial_sequence_bundle().physical_bytes();
        }
    }
    assert_eq!(
        summary.total,
        summary.plan
            + summary.request
            + summary.sequence
            + summary.step
            + summary.invocation
            + summary.initial_sequence_bundle,
        "typed live occupancy must equal the sum of its exact scopes"
    );
    assert_eq!(
        summary.total_bytes,
        summary.plan_bytes
            + summary.request_bytes
            + summary.sequence_bytes
            + summary.step_bytes
            + summary.invocation_bytes
            + summary.initial_sequence_bundle_bytes,
        "typed live bytes must equal the sum of their exact scopes"
    );
    summary
}

fn snapshot(
    snapshots: &mut Vec<LifecycleSnapshot>,
    root: &PlanRuntimeResources<TestRuntime>,
    stage: LifecycleStage,
) -> ScopeOccupancy {
    let occupancy = occupancy(root);
    snapshots.push(LifecycleSnapshot { stage, occupancy });
    occupancy
}

fn begin_step(
    batch: &ExecutionBatchParticipants<TestRuntime>,
    lane: &Arc<ExecutionLane<TestRuntime>>,
) -> Arc<StepResourceLease<TestRuntime>> {
    let work = vec![one_token_span(); CHILD_SEQUENCE_COUNT];
    let request = StepResourceAdmissionRequest::new(
        batch.bind_work_shape(work).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    for attempt in 0..=3 {
        match batch.try_begin_step(request.clone(), lane).unwrap() {
            StepResourceAdmissionDecision::Admitted(step) => return step,
            StepResourceAdmissionDecision::BackingDeferred(deferred) if attempt < 3 => {
                deferred.maintain().unwrap();
            }
            _ => panic!("synthetic lifecycle step admission did not converge"),
        }
    }
    unreachable!("bounded step admission returns or panics")
}

fn analyze(profile: SyntheticLifecycleProfile) -> LifecycleAnalysis {
    let fixture = profile.fixture();
    let root = Arc::clone(&fixture.plan_resources);
    let mut snapshots = Vec::new();
    let plan = snapshot(&mut snapshots, &root, LifecycleStage::Plan);
    assert_eq!(plan, ScopeOccupancy::default());
    let contract_shape = assert_profile_contract(&root, profile);

    let work = one_token_work();
    let request = admit_request_resources_with_work(
        &root,
        "run.g04.synthetic-lifecycle",
        &format!("request.g04.synthetic-lifecycle.{profile:?}"),
        work.clone(),
    );
    let request_only = snapshot(&mut snapshots, &root, LifecycleStage::Request);
    assert_eq!(
        request_only.request, 0,
        "product I/O is Step-scoped; Request authority need not retain physical backing"
    );
    assert_eq!(request_only.request_bytes, 0);
    assert_eq!(request_only.sequence, 0);

    let mut sequences = Vec::with_capacity(CHILD_SEQUENCE_COUNT);
    let mut per_child_sequence_claims = None;
    let mut per_child_sequence_bytes = None;
    for child_index in 0..CHILD_SEQUENCE_COUNT {
        let sequence = admit_sequence_resources_with_work(&request, work.clone());
        assert!(Arc::ptr_eq(sequence.request_resources(), &request));
        assert_eq!(sequence.request_authority(), request.request_authority());
        assert_eq!(sequence.coordinator_id(), request.coordinator_id());
        assert_eq!(sequence.run_id(), request.run_id());
        assert_eq!(sequence.request_id(), request.request_id());
        sequences.push(sequence);
        let admitted = snapshot(
            &mut snapshots,
            &root,
            LifecycleStage::ChildSequence(child_index + 1),
        );
        assert_eq!(
            admitted.request, request_only.request,
            "child admission must not duplicate Request-lifetime extents"
        );
        let current_per_child = admitted.sequence / (child_index as u64 + 1);
        assert_eq!(
            admitted.sequence,
            current_per_child * (child_index as u64 + 1),
            "Sequence-lifetime extents must scale exactly with child count"
        );
        match per_child_sequence_claims {
            Some(expected) => assert_eq!(current_per_child, expected),
            None => per_child_sequence_claims = Some(current_per_child),
        }
        let current_per_child_bytes = admitted.sequence_bytes / (child_index as u64 + 1);
        assert_eq!(
            admitted.sequence_bytes,
            current_per_child_bytes * (child_index as u64 + 1),
            "Sequence-lifetime physical bytes must scale exactly with child count"
        );
        match per_child_sequence_bytes {
            Some(expected) => assert_eq!(current_per_child_bytes, expected),
            None => per_child_sequence_bytes = Some(current_per_child_bytes),
        }
    }
    let per_child_sequence_claims = per_child_sequence_claims.unwrap();
    let per_child_sequence_bytes = per_child_sequence_bytes.unwrap();
    assert!(per_child_sequence_claims > 0);
    assert!(per_child_sequence_bytes > 0);
    assert_eq!(
        sequences
            .iter()
            .map(|sequence| sequence.sequence_authority())
            .collect::<BTreeSet<_>>()
            .len(),
        CHILD_SEQUENCE_COUNT,
        "one Request must own N distinct child Sequence authorities"
    );

    let sessions = sequences
        .iter()
        .map(|sequence| sequence.open_session().unwrap())
        .collect::<Vec<_>>();
    let batch = ExecutionBatchParticipants::new(sessions.iter().map(Arc::clone).collect()).unwrap();
    let lane = root.create_execution_lane().unwrap();
    let step = begin_step(&batch, &lane);
    assert_eq!(step.work_shape().participants().len(), CHILD_SEQUENCE_COUNT);
    let step_live = snapshot(&mut snapshots, &root, LifecycleStage::Step);
    assert!(step_live.step > 0);
    assert_eq!(step_live.request, request_only.request);
    assert_eq!(
        step_live.sequence,
        per_child_sequence_claims * CHILD_SEQUENCE_COUNT as u64
    );

    let wave = prepare_wave(&root, &fixture.plan, &step);
    let active_bindings = batch
        .sessions()
        .iter()
        .map(|session| TrustedActiveSequenceBinding::from_session(session).unwrap())
        .collect::<Vec<_>>();
    assert!(wave
        .nodes()
        .iter()
        .all(|node| { node.participant_frames().len() == CHILD_SEQUENCE_COUNT }));
    let prepared = snapshot(&mut snapshots, &root, LifecycleStage::InvocationPrepared);
    assert_eq!(prepared.request, step_live.request);
    assert_eq!(prepared.sequence, step_live.sequence);
    assert_eq!(prepared.step, step_live.step);
    assert_eq!(
        prepared.invocation > 0,
        profile.has_invocation_scratch(),
        "only MoE and hybrid profiles declare Invocation scratch"
    );
    assert_eq!(
        prepared.invocation,
        u64::from(profile.has_invocation_scratch()),
        "the two sequential nodes must share one Invocation peak claim"
    );
    assert_eq!(
        prepared.invocation_bytes,
        if profile.has_invocation_scratch() {
            16 * CHILD_SEQUENCE_COUNT as u64
        } else {
            0
        },
        "Invocation scratch must equal max(node bytes), not the two-node sum"
    );

    let providers = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| fixture.registry.bind(&fixture.resolved, node.id()).unwrap())
        .collect::<Vec<_>>();
    let batch_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let reaper = CompletionReaper::new();
    let handle = OperationDispatch::encode_and_submit_wave(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    assert_eq!(lane.in_flight_count(), 1);
    assert_eq!(reaper.retained_count(), 1);
    let in_flight = snapshot(&mut snapshots, &root, LifecycleStage::InFlight);
    assert_eq!(in_flight, prepared);

    let completion = match handle.wait().unwrap() {
        CompletionObservation::Terminal(completion) => completion,
        other => panic!("synthetic lifecycle did not reach a terminal fence: {other:?}"),
    };
    assert_eq!(
        completion.disposition(),
        &OperationCompletionDisposition::Succeeded
    );
    assert_eq!(
        completion.participants().len(),
        CHILD_SEQUENCE_COUNT * fixture.plan.payload().nodes().len(),
        "completion must project every child of every submitted node"
    );
    assert!(completion.participants().iter().all(|participant| {
        participant.disposition() == &OperationParticipantCompletionDisposition::Succeeded
    }));
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);
    let terminal = snapshot(&mut snapshots, &root, LifecycleStage::FenceTerminal);
    assert_eq!(terminal.invocation, 0);
    assert_eq!(terminal.step, step_live.step);
    assert_eq!(terminal.sequence, step_live.sequence);
    assert_eq!(terminal.request, step_live.request);

    {
        let runtime_trace = fixture.runtime_trace.lock().unwrap();
        assert_eq!(runtime_trace.submit_calls, 1);
        assert_eq!(runtime_trace.next_fence, 1);
    }
    {
        let provider_trace = fixture.provider_trace.lock().unwrap();
        assert_eq!(
            provider_trace.encode_calls,
            fixture.plan.payload().nodes().len() as u64
        );
        assert_eq!(provider_trace.last_participant_count, CHILD_SEQUENCE_COUNT);
    }

    drop(handle);
    drop(providers);
    drop(active_bindings);
    drop(reaper);
    step.try_retire_normal().unwrap();
    let retired = snapshot(&mut snapshots, &root, LifecycleStage::StepRetired);
    assert_eq!(retired.step, 0);
    assert_eq!(retired.invocation, 0);
    assert_eq!(retired.sequence, terminal.sequence);
    assert_eq!(retired.request, terminal.request);
    drop(batch);
    for session in &sessions {
        session.try_complete().unwrap();
    }
    drop(sessions);
    drop(sequences);
    let request_retained = snapshot(&mut snapshots, &root, LifecycleStage::RequestRetained);
    assert_eq!(request_retained, request_only);
    drop(request);
    let empty = snapshot(&mut snapshots, &root, LifecycleStage::Empty);
    assert_eq!(empty, ScopeOccupancy::default());

    drop(lane);
    drop(root);
    drop(fixture.registry);
    drop(fixture.impostor_registry);
    drop(fixture.runtime);
    match PlanRuntimeResources::close(fixture.plan_resources) {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => {
            assert_eq!(receipt.released_static_resources(), 2)
        }
        Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => {
            panic!("synthetic lifecycle retained {strong_count} root references")
        }
        Err(failure) => panic!("synthetic lifecycle close failed: {:?}", failure.failure()),
    }

    LifecycleAnalysis {
        profile,
        contract_shape,
        snapshots,
        per_child_sequence_claims,
        per_child_sequence_bytes,
        invocation_peak_claims: prepared.invocation,
        invocation_peak_bytes: prepared.invocation_bytes,
    }
}

#[test]
fn dense_moe_and_hybrid_share_one_exact_resource_lifecycle() {
    let analyses = SyntheticLifecycleProfile::ALL.map(analyze);
    let expected_stages = vec![
        LifecycleStage::Plan,
        LifecycleStage::Request,
        LifecycleStage::ChildSequence(1),
        LifecycleStage::ChildSequence(2),
        LifecycleStage::ChildSequence(3),
        LifecycleStage::Step,
        LifecycleStage::InvocationPrepared,
        LifecycleStage::InFlight,
        LifecycleStage::FenceTerminal,
        LifecycleStage::StepRetired,
        LifecycleStage::RequestRetained,
        LifecycleStage::Empty,
    ];
    for analysis in &analyses {
        assert_eq!(
            analysis
                .snapshots
                .iter()
                .map(|snapshot| snapshot.stage)
                .collect::<Vec<_>>(),
            expected_stages,
            "{:?} diverged from the shared lifecycle",
            analysis.profile
        );
    }
    let [dense, moe, hybrid] = &analyses;
    assert_eq!(dense.profile, SyntheticLifecycleProfile::Dense);
    assert_eq!(moe.profile, SyntheticLifecycleProfile::Moe);
    assert_eq!(hybrid.profile, SyntheticLifecycleProfile::Hybrid);
    assert_eq!(
        dense.per_child_sequence_claims,
        moe.per_child_sequence_claims
    );
    assert_eq!(dense.per_child_sequence_bytes, moe.per_child_sequence_bytes);
    assert!(hybrid.per_child_sequence_claims > dense.per_child_sequence_claims);
    assert!(hybrid.per_child_sequence_bytes > dense.per_child_sequence_bytes);
    assert_eq!(
        (dense.invocation_peak_claims, dense.invocation_peak_bytes),
        (0, 0)
    );
    assert_eq!(
        (moe.invocation_peak_claims, moe.invocation_peak_bytes),
        (1, 48)
    );
    assert_eq!(
        (hybrid.invocation_peak_claims, hybrid.invocation_peak_bytes),
        (1, 48)
    );
    println!(
        "FERRUM G04 SYNTHETIC LIFECYCLE KEEP: {}",
        serde_json::to_string(&analyses).unwrap()
    );
}
