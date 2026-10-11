//! Two real CUDA lanes share one provisioned Plan, including its weights and
//! validation flags. Request/sequence state and admitted step scratch remain
//! independent. This positive gate does not qualify alias/generation rejection
//! or prove that the device executions overlap in time.

use super::*;
use std::sync::Barrier;

struct RequestOnLane {
    lane: Arc<ExecutionLane<Runtime>>,
    reaper: Arc<CompletionReaper<Runtime>>,
    session: Arc<SequenceSession<Runtime>>,
    tokens: Arc<[u32]>,
}

impl RequestOnLane {
    fn execute(&self, fixture: &Fixture, range: Range<usize>, path: Path) -> BatchObservation {
        fixture.execute_participants_on_lane(
            &self.lane,
            &self.reaper,
            std::slice::from_ref(&self.session),
            std::slice::from_ref(&self.tokens),
            range,
            path,
        )
    }
}

fn execute_pair(
    fixture: &Fixture,
    requests: &[RequestOnLane; 2],
    range: Range<usize>,
    path: Path,
) -> [BatchObservation; 2] {
    // Host release is simultaneous, including the first encode using these
    // Plan flags. No sleep, device synchronization, or fake pending event is
    // inserted to force a particular registry publication order.
    let start = Barrier::new(3);
    std::thread::scope(|scope| {
        let first_range = range.clone();
        let first = scope.spawn(|| {
            start.wait();
            requests[0].execute(fixture, first_range, path)
        });
        let second = scope.spawn(|| {
            start.wait();
            requests[1].execute(fixture, range, path)
        });
        start.wait();
        [first.join().unwrap(), second.join().unwrap()]
    })
}

fn verify(kind: AttentionKind) {
    verify_policy(kind, false, false)
}
fn verify_policy(kind: AttentionKind, hybrid: bool, extra: bool) {
    verify_with_family(
        kind,
        || {
            if extra {
                Family::extra(kind)
            } else if hybrid {
                Family::hybrid(kind, 2052)
            } else {
                Family::new(kind)
            }
        },
        hybrid,
    )
}

fn verify_with_family(kind: AttentionKind, definition: impl Fn() -> Family, large_prefix: bool) {
    let make = |reusable| Fixture::for_family(kind, reusable, 2, definition());
    let eager = make(false);
    let shared = make(true);
    // ExecutionLane::create calls CudaDeviceRuntime::create_stream; both lanes
    // come from the SAME actual PlanRuntimeResources, not cloned fixtures.
    let second_lane = shared.resources.create_execution_lane().unwrap();
    second_lane
        .configure_reusable_executables(DeviceReusableExecutionPlan::on_demand(8).unwrap())
        .unwrap();
    assert_ne!(shared.lane.id(), second_lane.id());
    let lanes = [Arc::clone(&shared.lane), second_lane];
    let tokens: [Arc<[u32]>; 2] = std::array::from_fn(|sequence| {
        (0..if large_prefix { 37 } else { 36 })
            .map(|position| ((position * 7 + sequence * 3 + 1) % 32) as u32)
            .collect()
    });
    let eager_sessions: [_; 2] = std::array::from_fn(|sequence| {
        eager.admit(
            &format!("two-stream-eager-{sequence}"),
            Arc::clone(&tokens[sequence]),
        )
    });
    let requests: [RequestOnLane; 2] = std::array::from_fn(|sequence| RequestOnLane {
        lane: Arc::clone(&lanes[sequence]),
        reaper: if sequence == 0 {
            Arc::clone(&shared.reaper)
        } else {
            CompletionReaper::new()
        },
        session: shared.admit(
            &format!("two-stream-shared-{sequence}"),
            Arc::clone(&tokens[sequence]),
        ),
        tokens: Arc::clone(&tokens[sequence]),
    });
    println!(
        "{}",
        serde_json::json!({"kind":"marker_two_stream_plan", "attention":format!("{kind:?}"),
            "lane_ids":lanes.iter().map(|lane| format!("{:?}",lane.id())).collect::<Vec<_>>(),
            "shared_plan_resources":true,"independent_requests":true,
            "host_start_barrier":true,"device_overlap_measured":false})
    );

    // Keep the existing cold admission/memory policy. Two independent prefix
    // waves contain four tokens each, then single-token waves cross KV block32.
    let prefix_path = if kind == AttentionKind::Causal {
        Path::EagerBoundary
    } else {
        Path::Warm
    };
    let waves = (0..32)
        .step_by(4)
        .map(|start| {
            (
                start..start + 4,
                if start == 0 { Path::Warm } else { prefix_path },
            )
        })
        .chain((32..36).map(|position| {
            (
                position..position + 1,
                if position < 34 {
                    Path::Warm
                } else {
                    Path::Replay
                },
            )
        }));
    let waves: Vec<_> = if large_prefix {
        std::iter::once((0..33, Path::Warm))
            .chain((33..37).map(|i| (i..i + 1, if i < 35 { Path::Warm } else { Path::Replay })))
            .collect()
    } else {
        waves.collect()
    };
    for (range, path) in waves {
        let expected: [_; 2] = std::array::from_fn(|sequence| {
            eager.execute_participants(
                std::slice::from_ref(&eager_sessions[sequence]),
                std::slice::from_ref(&tokens[sequence]),
                range.clone(),
                Path::Eager,
            )
        });
        if range.start == 0 {
            assert_ne!(
                expected[0].values, expected[1].values,
                "independent requests must exercise distinguishable output/state"
            );
        }
        let actual = execute_pair(&shared, &requests, range.clone(), path);
        for sequence in 0..2 {
            // Full residual F32 output and every valid recurrent/KV element.
            // Path::Replay additionally requires the actual lane catalog and
            // determinism_replayed; it never falls back to eager execution.
            expected[sequence].assert_same(&actual[sequence]);
            println!(
                "{}",
                serde_json::json!({"kind":"marker_two_stream_comparison", "attention":format!("{kind:?}"),
                    "lane_id":format!("{:?}",requests[sequence].lane.id()),"sequence":sequence,
                    "source_start":range.start,"source_end":range.end,"path":format!("{path:?}"),
                    "all_output_state_bytes_equal":true})
            );
            actual[sequence].dump(kind, 1, range.clone(), path);
        }
    }
    for sequence in 0..2 {
        assert_eq!(requests[sequence].lane.in_flight_count(), 0);
        requests[sequence].session.try_complete().unwrap();
        eager_sessions[sequence].try_complete().unwrap();
    }
}

#[test]
#[ignore = "requires exclusive CUDA and actual MarkerV2 native/provider artifacts"]
fn upstream_marker_shared_plan_two_cuda_streams_preserve_eager_and_replay_state() {
    for kind in [AttentionKind::GatedDelta, AttentionKind::Causal] {
        verify(kind);
    }
}

#[test]
#[ignore = "requires qualified G32/MMQ providers, real Plan flags and two CUDA streams"]
fn hybrid_shared_plan_two_streams_mmq_prefix_then_g32_replay() {
    verify_policy(AttentionKind::GatedDelta, true, false);
    verify_policy(AttentionKind::Causal, true, false);
}

pub(super) fn verify_extra(kind: AttentionKind) {
    verify_policy(kind, false, true);
}

pub(super) fn verify_extra_prefill(kind: AttentionKind) {
    verify_with_family(kind, || Family::extra_prefill(kind, 2052), true);
}

pub(super) fn verify_extra_all_rows(kind: AttentionKind) {
    verify_with_family(kind, || Family::extra_all_rows(kind, 2052), true);
}
