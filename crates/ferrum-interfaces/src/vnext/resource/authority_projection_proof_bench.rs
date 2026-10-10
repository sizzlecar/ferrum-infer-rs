//! Synthetic complete backing preparation, shared unchanged by two source trees.
//! External old/new executable ABBA pairs are the comparison; no timing is a gate.
use super::*;
use std::hint::black_box;
use std::time::{Duration, Instant};

const WARMUP_ITERATIONS: usize = 8;
const MEASURED_ITERATIONS: usize = 64;
const ROUNDS: usize = 4;

fn prepare_and_drop(
    wave: &PreparedStepSubmissionWave<TestRuntime>,
    requests: &[SegmentBackingRequest],
    windows: &[SegmentBackingWindow],
) {
    let prepared = black_box(wave)
        .prepare_segment_backings(black_box(requests), black_box(windows))
        .expect("the real admitted backing preparation must succeed");
    black_box(&prepared);
    drop(prepared);
}

fn elapsed_calls(
    wave: &PreparedStepSubmissionWave<TestRuntime>,
    requests: &[SegmentBackingRequest],
    windows: &[SegmentBackingWindow],
    iterations: usize,
) -> Duration {
    let started = Instant::now();
    for _ in 0..iterations {
        prepare_and_drop(wave, requests, windows);
    }
    started.elapsed()
}

fn round(resources_per_lifetime: usize, request_count: usize, round_index: usize) {
    let per_pool = u64::try_from(resources_per_lifetime).unwrap() * 64;
    let catalog = combine_catalogs(&[
        pool_catalog(
            linear_profile(),
            AllocationLifetime::Request,
            'a',
            resources_per_lifetime,
            per_pool,
            TestDemand::Fixed,
        ),
        pool_catalog(
            paged_profile(),
            AllocationLifetime::Sequence,
            'b',
            resources_per_lifetime,
            per_pool,
            TestDemand::Tokens,
        ),
        pool_catalog(
            linear_profile(),
            AllocationLifetime::Step,
            'c',
            resources_per_lifetime,
            per_pool,
            TestDemand::Fixed,
        ),
    ]);
    let mut unique_requests = Vec::new();
    for (digit, lifetime, profile) in [
        ('a', AllocationLifetime::Request, linear_profile()),
        ('b', AllocationLifetime::Sequence, paged_profile()),
        ('c', AllocationLifetime::Step, linear_profile()),
    ] {
        for index in 0..resources_per_lifetime {
            unique_requests.push(SegmentBackingRequest {
                participant_index: 0,
                resource_id: ResourceId::new(format!("resource/dynamic-{digit}-{index:02}"))
                    .unwrap(),
                lifetime,
                expected: SegmentBackingExpectation {
                    logical_bytes: 64,
                    logical_size_rule: SegmentLogicalSizeRule::Exact,
                    usage: BufferUsage::State,
                    storage_profile: profile,
                    element_type: ElementType::U8,
                    alignment_bytes: 16,
                },
            });
        }
    }
    let requests = (0..request_count)
        .map(|index| {
            let mut request = unique_requests[index % unique_requests.len()].clone();
            if request.lifetime == AllocationLifetime::Sequence && index % 2 == 0 {
                request.expected.logical_bytes = 32;
                request.expected.logical_size_rule = SegmentLogicalSizeRule::AtLeast;
            }
            request
        })
        .collect::<Vec<_>>();
    let windows = requests
        .iter()
        .enumerate()
        .flat_map(|(resource_index, request)| {
            (0..1 + resource_index % 3).map(move |part| SegmentBackingWindow {
                resource_index,
                offset_bytes: if part == 0 { 0 } else { 16 },
                length_bytes: if part == 0 {
                    request.expected.logical_bytes
                } else {
                    16
                },
                element_type: ElementType::U8,
                alignment_bytes: 16,
            })
        })
        .collect::<Vec<_>>();
    let runtime = new_runtime(&catalog, per_pool * 3);
    let harness = harness_with_nodes(
        runtime,
        catalog,
        per_pool * 3,
        false,
        Arc::from(vec![PlanNode::resource_test_node(
            NodeId::new("node/complete-backing-benchmark").unwrap(),
        )]),
    );
    for pool in &harness.pool_ids {
        harness
            .root
            .maintenance_controller
            .initialize_pool(pool)
            .unwrap();
    }
    let admission_started = Instant::now();
    let sequence = admitted_sequence(&harness.root, "complete-backing-benchmark");
    let session = sequence.open_session().unwrap();
    let lane = harness.root.create_execution_lane().unwrap();
    let (batch, step) = begin_request_state_test_step(vec![Arc::clone(&session)], &lane);
    let wave = match step
        .try_prepare_full_plan_submission_wave(
            step.bind_all_invocation_work_shape(vec![token_span(1)])
                .unwrap()
                .into(),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap()
    {
        StepSubmissionWaveAdmissionDecision::Prepared(wave) => wave,
        _ => panic!("the resident complete-plan wave must prepare"),
    };
    let admission_elapsed = admission_started.elapsed();

    // Admission and all request/window construction are outside both timings.
    // Each round starts with fresh Request/Sequence/Step authorities. The first
    // call therefore exposes first-use cost without artificially cloning claims.
    let first_use = elapsed_calls(&wave, &requests, &windows, 1);
    let prepared = wave.prepare_segment_backings(&requests, &windows).unwrap();
    assert_eq!(prepared.resources().len(), request_count);
    assert_eq!(prepared.bindings().len(), unique_requests.len());
    let mut physical_buffers = BTreeSet::new();
    for binding in prepared.bindings() {
        physical_buffers.insert(binding.buffer() as *const TestBuffer as usize);
    }
    for (index, window) in windows.iter().enumerate() {
        let view = prepared.window(index).unwrap();
        assert_eq!(
            view.physical_regions()
                .map(|region| region.length_bytes())
                .sum::<u64>(),
            window.length_bytes,
        );
    }
    drop(prepared);
    let snapshot = sequence.backing_snapshot().unwrap();
    let mut claim_resource_id_counts = BTreeMap::<usize, usize>::new();
    for authority in sequence
        .request_resources()
        .backing_slices()
        .iter()
        .chain(snapshot.backing_slices())
        .chain(step.backing_slices())
    {
        *claim_resource_id_counts
            .entry(
                authority
                    .evidence()
                    .physical_claim_identity()
                    .resource_ids()
                    .len(),
            )
            .or_default() += 1;
    }
    assert_eq!(
        claim_resource_id_counts.values().sum::<usize>(),
        unique_requests.len()
    );
    drop(snapshot);

    for _ in 0..WARMUP_ITERATIONS {
        prepare_and_drop(&wave, &requests, &windows);
    }
    let elapsed = elapsed_calls(&wave, &requests, &windows, MEASURED_ITERATIONS);
    println!(
        "AUTHORITY_PROJECTION_PROOF_BENCH {}",
        json!({
            "round": round_index,
            "participants": 1,
            "plan_nodes": 1,
            "pools": 3,
            "distinct_authorities": unique_requests.len(),
            "actual_physical_buffers": physical_buffers.len(),
            "physical_claim_resource_ids_histogram": claim_resource_id_counts,
            "logical_authority_size_bytes": std::mem::size_of::<LogicalBackingSliceAuthority>(),
            "logical_evidence_size_bytes": std::mem::size_of::<LogicalBackingSliceEvidence>(),
            "requests": requests.len(),
            "windows": windows.len(),
            "warmup_iterations": WARMUP_ITERATIONS,
            "iterations": MEASURED_ITERATIONS,
            "first_use_prepare_and_drop_ns": first_use.as_nanos(),
            "new_request_sequence_step_wave_admission_ns": admission_elapsed.as_nanos(),
            "reused_wave_prepare_and_drop_total_ns": elapsed.as_nanos(),
            "reused_wave_prepare_and_drop_mean_ns": elapsed.as_nanos() as f64 / MEASURED_ITERATIONS as f64,
            "errors": 0,
            "scope": "complete wave.prepare_segment_backings plus result drop; snapshot lookup, dedup/reserve, canonical locking, current owner/chunk/extent validation, windows and retained owners included; admission, metadata capability getters, backend encoding, submission and GPU excluded",
            "reuse_limit": "same wave/Step reused within a round, so warmed result is an upper-bound reuse scenario; each round also reports first preparation on entirely fresh Request/Sequence/Step claims and separate admission time; admission may itself validate proofs, so first preparation does not promise an empty proof cache; no whole-model extrapolation",
        })
    );
    drop(wave);
    step.try_retire_normal().unwrap();
    session.try_complete().unwrap();
    drop(batch);
    drop(session);
    drop(sequence);
    drop(lane);
    close_dynamic_test_root(harness.root);
}

#[test]
#[ignore = "diagnostic: compare identical fixture in separately built old/new executables"]
fn authority_projection_proof_complete_backing_microbench() {
    for (resources_per_lifetime, requests) in [(16, 264), (138, 2285)] {
        for round_index in 0..ROUNDS {
            round(resources_per_lifetime, requests, round_index);
        }
    }
}
