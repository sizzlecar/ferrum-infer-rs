#[path = "vnext_resource_contract/support.rs"]
mod resource_support;
mod vnext_core_contract;

use ferrum_interfaces::vnext::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use vnext_core_contract as core;

fn sequential_scratch_plan_with_policy(
    policy: ResolvedRuntimePolicy,
) -> (
    ExecutionPlan,
    OperationRuntimeRegistry<core::PlanningTestRuntime>,
) {
    let registration = TypedFamilyRegistration::new(core::SequentialScratchFamily);
    let family = registration
        .prepare(&serde_json::json!({"width": 4}))
        .unwrap();
    let catalog = core::catalog();
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

fn sequential_scratch_plan() -> (
    ExecutionPlan,
    OperationRuntimeRegistry<core::PlanningTestRuntime>,
) {
    sequential_scratch_plan_with_policy(core::policy(4096))
}

fn reusable_sequential_scratch_plan() -> (
    ExecutionPlan,
    OperationRuntimeRegistry<core::PlanningTestRuntime>,
    ReusableExecutionBucketSpec,
) {
    let bucket = ReusableExecutionBucketSpec::new(
        ReusableExecutionClassId::new("execution.test-decode").unwrap(),
        ReusableExecutionCapacity::new(1, 1, 1).unwrap(),
    )
    .unwrap();
    let reusable_execution = ReusableExecutionPolicy::new(1, vec![bucket.clone()]).unwrap();
    let policy = ResolvedRuntimePolicy::new(
        "runtime-policy.test-reusable",
        ContractVersion::new(1, 0),
        SchedulingDiscipline::FirstReady,
        RuntimeMemoryPolicy {
            capacity_bytes: 4096,
            reserve_bytes: 128,
            maximum_active_sequences: 3,
            dynamic_storage_profile_order: vec![core::contiguous_storage_profile()],
        },
        serde_json::from_value(serde_json::json!({
            "maximum_queue_depth": 8,
            "maximum_scheduled_tokens": 4096,
            "sequence_fit_policy": "immediate_only",
            "allow_defer": true,
            "cancellation_check_interval_steps": 1
        }))
        .unwrap(),
        Some(reusable_execution),
    )
    .unwrap();
    let (plan, registry) = sequential_scratch_plan_with_policy(policy);
    (plan, registry, bucket)
}

fn reusable_token_scaled_plan() -> (
    ExecutionPlan,
    core::TestRegistry,
    core::TestPlanningRegistry,
    ReusableExecutionBucketSpec,
) {
    let bucket = ReusableExecutionBucketSpec::new(
        ReusableExecutionClassId::new("execution.test-token-scaled").unwrap(),
        ReusableExecutionCapacity::new(1, 4, 1).unwrap(),
    )
    .unwrap();
    let policy = ResolvedRuntimePolicy::new(
        "runtime-policy.test-token-scaled",
        ContractVersion::new(1, 0),
        SchedulingDiscipline::FirstReady,
        RuntimeMemoryPolicy {
            capacity_bytes: 4096,
            reserve_bytes: 128,
            maximum_active_sequences: 3,
            dynamic_storage_profile_order: vec![core::contiguous_storage_profile()],
        },
        serde_json::from_value(serde_json::json!({
            "maximum_queue_depth": 8,
            "maximum_scheduled_tokens": 4,
            "sequence_fit_policy": "immediate_only",
            "allow_defer": true,
            "cancellation_check_interval_steps": 1
        }))
        .unwrap(),
        Some(ReusableExecutionPolicy::new(1, vec![bucket.clone()]).unwrap()),
    )
    .unwrap();
    let model_registry = core::TestRegistry::new();
    let family = model_registry.prepare();
    let catalog = core::catalog();
    let planning = core::TestPlanningRegistry::new(
        &catalog,
        32,
        32,
        core::EstimateBehavior::TokenScaledScratch,
    );
    let resolution = core::node_resolution(&family, &catalog, &policy, 0, &planning);
    let plan = ExecutionPlan::build(
        PlanBuildRequest::new(&family, &catalog, &policy, vec![resolution]).unwrap(),
    )
    .unwrap();
    (plan, model_registry, planning, bucket)
}

fn begin_step(
    batch: &ExecutionBatchParticipants<resource_support::TestRuntime>,
    lane: &Arc<ExecutionLane<resource_support::TestRuntime>>,
) -> Arc<StepResourceLease<resource_support::TestRuntime>> {
    begin_step_with_bucket(batch, lane, None)
}

fn begin_step_with_bucket(
    batch: &ExecutionBatchParticipants<resource_support::TestRuntime>,
    lane: &Arc<ExecutionLane<resource_support::TestRuntime>>,
    bucket: Option<&ReusableExecutionBucketSpec>,
) -> Arc<StepResourceLease<resource_support::TestRuntime>> {
    begin_step_for_span(batch, lane, bucket, resource_support::one_token_span())
}

fn begin_step_for_span(
    batch: &ExecutionBatchParticipants<resource_support::TestRuntime>,
    lane: &Arc<ExecutionLane<resource_support::TestRuntime>>,
    bucket: Option<&ReusableExecutionBucketSpec>,
    token_span: TokenSpanWork,
) -> Arc<StepResourceLease<resource_support::TestRuntime>> {
    let request = StepResourceAdmissionRequest::new(
        batch.bind_work_shape(vec![token_span]).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let request = match bucket {
        Some(bucket) => request.with_reusable_execution_bucket(bucket.bucket_id().clone()),
        None => request,
    };
    for attempt in 0..=3 {
        match batch.try_begin_step(request.clone(), lane).unwrap() {
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
    let work_shape = step
        .shared_all_invocation_work_shape(&[resource_support::one_token_span()])
        .unwrap();
    plan.payload()
        .nodes()
        .iter()
        .map(|node| {
            InvocationResourceAdmissionRequest::for_all_step_participants(
                node.id().clone(),
                Arc::clone(&work_shape),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap()
        })
        .collect()
}

fn prepare_wave(
    _plan: &ExecutionPlan,
    step: &Arc<StepResourceLease<resource_support::TestRuntime>>,
) -> PreparedStepSubmissionWave<resource_support::TestRuntime> {
    prepare_wave_for_span(step, resource_support::one_token_span())
}

fn prepare_wave_for_span(
    step: &Arc<StepResourceLease<resource_support::TestRuntime>>,
    token_span: TokenSpanWork,
) -> PreparedStepSubmissionWave<resource_support::TestRuntime> {
    let work_shape = step
        .shared_all_invocation_work_shape(&[token_span])
        .unwrap();
    for attempt in 0..=3 {
        match step
            .try_prepare_full_plan_submission_wave(
                Arc::clone(&work_shape),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap()
        {
            StepSubmissionWaveAdmissionDecision::Prepared(wave) => return wave,
            StepSubmissionWaveAdmissionDecision::BackingDeferred(deferred) if attempt < 3 => {
                deferred.maintain().unwrap();
            }
            _ => panic!("sequential scratch wave admission did not converge"),
        }
    }
    unreachable!("bounded wave admission returns or panics")
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PhysicalSliceIdentity {
    resource_id: ResourceId,
    pool_instance_id: u64,
    segment_generation: u64,
    segments: Vec<BackingSegment>,
    physical_offset_bytes: u64,
    capacity_size_bytes: u64,
    physical_size_bytes: u64,
}

fn physical_slice_identities(
    slices: &[LogicalBackingSliceAuthority],
) -> Vec<PhysicalSliceIdentity> {
    slices
        .iter()
        .map(|slice| {
            let evidence = slice.evidence();
            PhysicalSliceIdentity {
                resource_id: evidence.resource_id().clone(),
                pool_instance_id: evidence.pool_instance_id(),
                segment_generation: evidence.segment_generation(),
                segments: evidence.segments().to_vec(),
                physical_offset_bytes: evidence.physical_offset_bytes(),
                capacity_size_bytes: evidence.capacity_size_bytes(),
                physical_size_bytes: evidence.physical_size_bytes(),
            }
        })
        .collect()
}

fn logical_slice_sizes(slices: &[LogicalBackingSliceAuthority]) -> Vec<(ResourceId, u64)> {
    slices
        .iter()
        .map(|slice| (slice.resource_id().clone(), slice.size_bytes()))
        .collect()
}

fn live_segments_by_pool(
    root: &PlanRuntimeResources<resource_support::TestRuntime>,
) -> BTreeMap<DynamicBackingPoolId, u64> {
    root.dynamic_pool_status()
        .unwrap()
        .pools()
        .iter()
        .map(|pool| (pool.pool_id().clone(), pool.live_segments()))
        .collect()
}

fn assert_physical_identities_do_not_overlap(
    left: &[PhysicalSliceIdentity],
    right: &[PhysicalSliceIdentity],
) {
    for left in left {
        for right in right {
            for left_segment in &left.segments {
                for right_segment in &right.segments {
                    if left_segment.chunk() != right_segment.chunk() {
                        continue;
                    }
                    let left_end = left_segment
                        .offset_bytes()
                        .checked_add(left_segment.length_bytes())
                        .unwrap();
                    let right_end = right_segment
                        .offset_bytes()
                        .checked_add(right_segment.length_bytes())
                        .unwrap();
                    assert!(
                        left_end <= right_segment.offset_bytes()
                            || right_end <= left_segment.offset_bytes(),
                        "concurrent arena slots overlap one physical extent"
                    );
                }
            }
        }
    }
}

#[test]
fn eager_submission_wave_releases_invocation_backing_while_step_and_lane_remain_live() {
    let (plan, registry) = sequential_scratch_plan();
    let (driver, _trace) = resource_support::configured_driver(&plan, &[], &[]);
    let lane = ExecutionLane::create(Arc::clone(&driver.runtime)).unwrap();
    let root = resource_support::plan_runtime(&plan, driver, "eager-wave-release");
    let sequence = resource_support::admit_logical_sequence(
        &root,
        "run.eager-wave-release",
        "request.eager-wave-release",
    );
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let step = begin_step(&batch, &lane);
    let before_wave = live_segments_by_pool(&root);

    let wave = prepare_wave(&plan, &step);
    assert!(std::ptr::eq(
        step.work_shape(),
        wave.nodes()[0].work_shape()
    ));
    assert!(step
        .shared_all_invocation_work_shape(&[TokenSpanWork::from_token_ids(&[1, 2], 0..2).unwrap()])
        .is_err());
    let during_wave = live_segments_by_pool(&root);
    assert!(during_wave.iter().any(|(pool_id, live_segments)| {
        *live_segments > before_wave.get(pool_id).copied().unwrap_or_default()
    }));

    drop(wave);
    assert_eq!(
        live_segments_by_pool(&root),
        before_wave,
        "ordinary eager Invocation backing must release while its Step and lane remain live"
    );

    step.try_retire_normal().unwrap();
    drop(batch);
    session.try_complete().unwrap();
    drop(session);
    drop(sequence);
    drop(lane);
    drop(registry);
    resource_support::close_plan_runtime(root);
}

#[test]
fn reusable_lane_same_shape_reuses_step_and_invocation_physical_generations() {
    let (plan, registry, bucket) = reusable_sequential_scratch_plan();
    let (driver, _trace) = resource_support::configured_driver(&plan, &[], &[]);
    let lane = ExecutionLane::create(Arc::clone(&driver.runtime)).unwrap();
    let root = resource_support::plan_runtime(&plan, driver, "lane-arena-reuse");
    let sequence = resource_support::admit_logical_sequence(
        &root,
        "run.lane-arena-reuse",
        "request.lane-arena-reuse",
    );
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();

    let first_step = begin_step_with_bucket(&batch, &lane, Some(&bucket));
    let first_wave = prepare_wave(&plan, &first_step);
    let first_step_identity = physical_slice_identities(first_step.backing_slices());
    let first_wave_identity =
        physical_slice_identities(first_wave.claimed_backing().backing_slices());
    drop(first_wave);
    first_step.try_retire_normal().unwrap();

    let second_step = begin_step_with_bucket(&batch, &lane, Some(&bucket));
    let second_wave = prepare_wave(&plan, &second_step);
    assert_eq!(
        physical_slice_identities(second_step.backing_slices()),
        first_step_identity
    );
    assert_eq!(
        physical_slice_identities(second_wave.claimed_backing().backing_slices()),
        first_wave_identity
    );

    drop(second_wave);
    second_step.try_retire_normal().unwrap();
    drop(batch);
    session.try_complete().unwrap();
    drop(session);
    drop(sequence);
    drop(lane);
    drop(registry);
    resource_support::close_plan_runtime(root);
}

#[test]
fn reusable_wave_keeps_compiled_physical_layout_while_logical_demand_changes() {
    let (plan, model_registry, planning, bucket) = reusable_token_scaled_plan();
    let (driver, _trace) = resource_support::configured_driver(&plan, &[], &[]);
    let lane = ExecutionLane::create(Arc::clone(&driver.runtime)).unwrap();
    let root = resource_support::plan_runtime(&plan, driver, "lane-arena-token-scaled");
    let sequence = resource_support::admit_logical_sequence(
        &root,
        "run.lane-arena-token-scaled",
        "request.lane-arena-token-scaled",
    );
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();

    let first_span = TokenSpanWork::from_token_ids(&[1], 0..1).unwrap();
    let first_step = begin_step_for_span(&batch, &lane, Some(&bucket), first_span.clone());
    let first_wave = prepare_wave_for_span(&first_step, first_span);
    let first_slices = first_wave.claimed_backing().backing_slices();
    let first_physical = physical_slice_identities(first_slices);
    let first_logical = logical_slice_sizes(first_slices);
    let wire = serde_json::to_value(first_slices[0].evidence()).unwrap();
    assert_eq!(
        wire.as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            "alignment_bytes",
            "capacity_size_bytes",
            "domain_id",
            "element_type",
            "initialization",
            "physical_claim_identity",
            "physical_offset_bytes",
            "physical_size_bytes",
            "pool_id",
            "pool_instance_id",
            "resource_id",
            "reusable_execution_bucket_id",
            "segment_generation",
            "segments",
            "size_bytes",
            "storage_profile",
            "usage",
        ]),
        "allocation caching must preserve the public evidence wire shape"
    );
    drop(first_wave);
    first_step.try_retire_normal().unwrap();

    let second_span = TokenSpanWork::from_token_ids(&[1, 2], 0..2).unwrap();
    let second_step = begin_step_for_span(&batch, &lane, Some(&bucket), second_span.clone());
    let second_wave = prepare_wave_for_span(&second_step, second_span);
    let second_slices = second_wave.claimed_backing().backing_slices();
    assert_eq!(
        physical_slice_identities(second_slices),
        first_physical,
        "one reusable bucket must retain its precompiled physical capacity layout"
    );
    assert_ne!(
        logical_slice_sizes(second_slices),
        first_logical,
        "logical per-wave demand must remain dynamic inside a reusable capacity bucket"
    );

    drop(second_wave);
    second_step.try_retire_normal().unwrap();
    drop(batch);
    session.try_complete().unwrap();
    drop(session);
    drop(sequence);
    drop(lane);
    drop(planning);
    drop(model_registry);
    resource_support::close_plan_runtime(root);
}

#[test]
fn reusable_lane_overlapping_steps_use_disjoint_slots_then_reuse_released_slots() {
    let (plan, registry, bucket) = reusable_sequential_scratch_plan();
    let (driver, _trace) = resource_support::configured_driver(&plan, &[], &[]);
    let lane = ExecutionLane::create(Arc::clone(&driver.runtime)).unwrap();
    let root = resource_support::plan_runtime(&plan, driver, "lane-arena-overlap");
    let first_sequence = resource_support::admit_logical_sequence(
        &root,
        "run.lane-arena-overlap.first",
        "request.lane-arena-overlap.first",
    );
    let second_sequence = resource_support::admit_logical_sequence(
        &root,
        "run.lane-arena-overlap.second",
        "request.lane-arena-overlap.second",
    );
    let first_session = first_sequence.open_session().unwrap();
    let second_session = second_sequence.open_session().unwrap();
    let first_batch = ExecutionBatchParticipants::new(vec![Arc::clone(&first_session)]).unwrap();
    let second_batch = ExecutionBatchParticipants::new(vec![Arc::clone(&second_session)]).unwrap();

    let first_step = begin_step_with_bucket(&first_batch, &lane, Some(&bucket));
    let first_wave = prepare_wave(&plan, &first_step);
    let first_step_identity = physical_slice_identities(first_step.backing_slices());
    let first_wave_identity =
        physical_slice_identities(first_wave.claimed_backing().backing_slices());

    let second_step = begin_step_with_bucket(&second_batch, &lane, Some(&bucket));
    let second_wave = prepare_wave(&plan, &second_step);
    let second_step_identity = physical_slice_identities(second_step.backing_slices());
    let second_wave_identity =
        physical_slice_identities(second_wave.claimed_backing().backing_slices());
    assert_physical_identities_do_not_overlap(&first_step_identity, &second_step_identity);
    assert_physical_identities_do_not_overlap(&first_wave_identity, &second_wave_identity);

    drop(first_wave);
    first_step.try_retire_normal().unwrap();
    let third_step = begin_step_with_bucket(&first_batch, &lane, Some(&bucket));
    let third_wave = prepare_wave(&plan, &third_step);
    assert_eq!(
        physical_slice_identities(third_step.backing_slices()),
        first_step_identity,
        "the first released Step slot must be reusable while another slot remains in flight"
    );
    assert_eq!(
        physical_slice_identities(third_wave.claimed_backing().backing_slices()),
        first_wave_identity,
        "the first released Invocation slot must be reusable while another slot remains in flight"
    );

    drop(third_wave);
    third_step.try_retire_normal().unwrap();
    drop(second_wave);
    second_step.try_retire_normal().unwrap();
    drop(first_batch);
    drop(second_batch);
    first_session.try_complete().unwrap();
    second_session.try_complete().unwrap();
    drop(first_session);
    drop(second_session);
    drop(first_sequence);
    drop(second_sequence);
    drop(lane);
    drop(registry);
    resource_support::close_plan_runtime(root);
}

#[test]
fn released_reusable_lane_slot_wakes_capacity_waiter_and_retries_without_growth() {
    let (plan, registry, bucket) = reusable_sequential_scratch_plan();
    let (driver, _trace) = resource_support::configured_driver(&plan, &[], &[]);
    let lane = ExecutionLane::create(Arc::clone(&driver.runtime)).unwrap();
    let root = resource_support::plan_runtime(&plan, driver, "lane-arena-waiter");
    let first_sequence = resource_support::admit_logical_sequence(
        &root,
        "run.lane-arena-waiter.first",
        "request.lane-arena-waiter.first",
    );
    let second_sequence = resource_support::admit_logical_sequence(
        &root,
        "run.lane-arena-waiter.second",
        "request.lane-arena-waiter.second",
    );
    let first_session = first_sequence.open_session().unwrap();
    let second_session = second_sequence.open_session().unwrap();
    let first_batch = ExecutionBatchParticipants::new(vec![Arc::clone(&first_session)]).unwrap();
    let second_batch = ExecutionBatchParticipants::new(vec![Arc::clone(&second_session)]).unwrap();
    let first_step = begin_step_with_bucket(&first_batch, &lane, Some(&bucket));
    let first_wave = prepare_wave(&plan, &first_step);
    let first_wave_identity =
        physical_slice_identities(first_wave.claimed_backing().backing_slices());
    let second_step = begin_step_with_bucket(&second_batch, &lane, Some(&bucket));
    let requests = submission_requests(&plan, &second_step);

    let deferred = match second_step
        .try_prepare_submission_wave(requests.clone())
        .unwrap()
    {
        StepSubmissionWaveAdmissionDecision::BackingDeferred(deferred) => deferred,
        _ => panic!("a busy exact-shape slot with no resident spare must defer"),
    };
    let waiter = deferred.register_waiter().unwrap();
    assert!(!waiter.recheck().unwrap().changed_since_registration());

    drop(first_wave);
    assert!(
        waiter.recheck().unwrap().changed_since_registration(),
        "slot release must wake an already-registered capacity waiter"
    );
    drop(waiter);
    drop(deferred);
    let second_wave = match second_step.try_prepare_submission_wave(requests).unwrap() {
        StepSubmissionWaveAdmissionDecision::Prepared(wave) => wave,
        _ => panic!("released exact-shape slot must be immediately reusable without growth"),
    };
    assert_eq!(
        physical_slice_identities(second_wave.claimed_backing().backing_slices()),
        first_wave_identity
    );

    drop(second_wave);
    first_step.try_retire_normal().unwrap();
    second_step.try_retire_normal().unwrap();
    drop(first_batch);
    drop(second_batch);
    first_session.try_complete().unwrap();
    second_session.try_complete().unwrap();
    drop(first_session);
    drop(second_session);
    drop(first_sequence);
    drop(second_sequence);
    drop(lane);
    drop(registry);
    resource_support::close_plan_runtime(root);
}

#[test]
fn concurrent_lanes_receive_disjoint_step_and_invocation_arenas() {
    let (plan, registry) = sequential_scratch_plan();
    let (driver, _trace) = resource_support::configured_driver(&plan, &[], &[]);
    let first_lane = ExecutionLane::create(Arc::clone(&driver.runtime)).unwrap();
    let second_lane = ExecutionLane::create(Arc::clone(&driver.runtime)).unwrap();
    let root = resource_support::plan_runtime(&plan, driver, "lane-arena-isolation");
    let first_sequence = resource_support::admit_logical_sequence(
        &root,
        "run.lane-arena-isolation.first",
        "request.lane-arena-isolation.first",
    );
    let second_sequence = resource_support::admit_logical_sequence(
        &root,
        "run.lane-arena-isolation.second",
        "request.lane-arena-isolation.second",
    );
    let first_session = first_sequence.open_session().unwrap();
    let second_session = second_sequence.open_session().unwrap();
    let first_batch = ExecutionBatchParticipants::new(vec![Arc::clone(&first_session)]).unwrap();
    let second_batch = ExecutionBatchParticipants::new(vec![Arc::clone(&second_session)]).unwrap();

    let first_step = begin_step(&first_batch, &first_lane);
    let first_wave = prepare_wave(&plan, &first_step);
    let second_step = begin_step(&second_batch, &second_lane);
    let second_wave = prepare_wave(&plan, &second_step);
    let first_step_identity = physical_slice_identities(first_step.backing_slices());
    let second_step_identity = physical_slice_identities(second_step.backing_slices());
    let first_wave_identity =
        physical_slice_identities(first_wave.claimed_backing().backing_slices());
    let second_wave_identity =
        physical_slice_identities(second_wave.claimed_backing().backing_slices());
    assert!(first_step_identity
        .iter()
        .zip(&second_step_identity)
        .all(|(left, right)| left.segment_generation != right.segment_generation));
    assert!(first_wave_identity
        .iter()
        .zip(&second_wave_identity)
        .all(|(left, right)| left.segment_generation != right.segment_generation));
    assert_physical_identities_do_not_overlap(&first_step_identity, &second_step_identity);
    assert_physical_identities_do_not_overlap(&first_wave_identity, &second_wave_identity);

    drop(first_wave);
    drop(second_wave);
    first_step.try_retire_normal().unwrap();
    second_step.try_retire_normal().unwrap();
    drop(first_batch);
    drop(second_batch);
    first_session.try_complete().unwrap();
    second_session.try_complete().unwrap();
    drop(first_session);
    drop(second_session);
    drop(first_sequence);
    drop(second_sequence);
    drop(first_lane);
    drop(second_lane);
    drop(registry);
    resource_support::close_plan_runtime(root);
}

#[test]
fn total_order_invocation_scratch_claims_peak_once_for_the_whole_wave() {
    let (plan, registry) = sequential_scratch_plan();
    assert_eq!(plan.payload().memory().minimum_invocation_peak_bytes(), 96);
    let (driver, _trace) = resource_support::configured_driver(&plan, &[], &[]);
    let lane = ExecutionLane::create(Arc::clone(&driver.runtime)).unwrap();
    let root = resource_support::plan_runtime(&plan, driver, "submission-wave-peak");
    let sequence = resource_support::admit_logical_sequence(
        &root,
        "run.submission-wave-peak",
        "request.submission-wave-peak",
    );
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let step = begin_step(&batch, &lane);

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
    assert_eq!(claimed.node_count(), 2);
    assert_eq!(claimed.work_shape(), wave.nodes()[0].work_shape());
    assert!(std::ptr::eq(
        wave.nodes()[0].work_shape(),
        wave.nodes()[1].work_shape()
    ));
    assert_eq!(claimed.plan_hash(), plan.plan_hash());
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
    let first_lane = ExecutionLane::create(Arc::clone(&driver.runtime)).unwrap();
    let second_lane = ExecutionLane::create(Arc::clone(&driver.runtime)).unwrap();
    let root = resource_support::plan_runtime(&plan, driver, "submission-wave-deferral-owner");
    let first_sequence = resource_support::admit_logical_sequence(
        &root,
        "run.submission-wave-deferral-owner-first",
        "request.submission-wave-deferral-owner-first",
    );
    let first_session = first_sequence.open_session().unwrap();
    let first_batch = ExecutionBatchParticipants::new(vec![Arc::clone(&first_session)]).unwrap();
    let first_step = begin_step(&first_batch, &first_lane);
    let first_wave = prepare_wave(&plan, &first_step);

    let second_sequence = resource_support::admit_logical_sequence(
        &root,
        "run.submission-wave-deferral-owner-second",
        "request.submission-wave-deferral-owner-second",
    );
    let second_session = second_sequence.open_session().unwrap();
    let second_batch = ExecutionBatchParticipants::new(vec![Arc::clone(&second_session)]).unwrap();
    let step = begin_step(&second_batch, &second_lane);
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

#[test]
fn capacity_deferred_unsubmitted_step_rolls_back_without_poisoning_its_session() {
    let (plan, registry) = sequential_scratch_plan();
    let (driver, _trace) = resource_support::configured_driver(&plan, &[], &[]);
    let lane = ExecutionLane::create(Arc::clone(&driver.runtime)).unwrap();
    let root = resource_support::plan_runtime(&plan, driver, "submission-wave-step-rollback");
    let sequence = resource_support::admit_logical_sequence(
        &root,
        "run.submission-wave-step-rollback",
        "request.submission-wave-step-rollback",
    );
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();

    let step = begin_step(&batch, &lane);
    let first_step_id = step.batch_step_id();
    let first_frame = step
        .participant_frames()
        .next()
        .expect("single-participant step owns one frame")
        .frame_id();
    let receipt = step.try_rollback_unsubmitted().unwrap();
    assert_eq!(receipt.batch_step_id(), first_step_id);
    assert_eq!(receipt.participants().len(), 1);
    assert_eq!(
        receipt.participants()[0].assignment().frame_id(),
        first_frame
    );
    assert_eq!(
        receipt.participants()[0].disposition(),
        StepParticipantRetirementDisposition::RolledBackUnsubmitted
    );

    let retry = begin_step(&batch, &lane);
    assert_ne!(retry.batch_step_id(), first_step_id);
    assert_eq!(
        retry
            .participant_frames()
            .next()
            .expect("single-participant retry owns one frame")
            .frame_id(),
        first_frame,
        "a fresh physical step attempt must reuse the request frame that never executed"
    );
    retry.try_retire_normal().unwrap();

    drop(batch);
    session.try_complete().unwrap();
    drop(session);
    drop(sequence);
    drop(registry);
    resource_support::close_plan_runtime(root);
}
