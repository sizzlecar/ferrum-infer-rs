use super::*;
use ferrum_types::RequestId;
use std::time::Duration;

fn request(prompt: usize, output: usize) -> SimulatedRequest {
    SimulatedRequest {
        id: RequestId::new(),
        prompt_tokens: prompt,
        prefill_tokens: if output > 0 { prompt } else { 0 },
        output_tokens: output,
        output_cap: 4,
        scenario_end_tokens: 4,
        age: Duration::ZERO,
        central_clock: RequestClock::default(),
        risk_clock: RequestClock::default(),
    }
}

fn shape(cap: Option<usize>) -> RequestShape {
    RequestShape {
        restored_tokens: 0,
        execution_ceiling: None,
        phase_cap: cap.and_then(NonZeroUsize::new),
        decode_only_allowed: true,
    }
}

fn trajectory(targets: Vec<PrefillTarget>, shapes: Vec<RequestShape>) -> CohortTrajectory {
    CohortTrajectory {
        first_wave: Vec::new(),
        cap_commit_targets: targets,
        shapes,
        admission: AdmissionSlotPlan::new(
            &[
                InitialAdmission::Resident,
                InitialAdmission::Waiting { order: 0 },
                InitialAdmission::Waiting { order: 1 },
            ],
            3,
            3,
        )
        .unwrap(),
        execution: CostExecution::SequentialPrefillThenDecode,
        shape_policy: PrefillShapePolicy::Identity,
        max_batch: 3,
        max_tokens: 5,
        prefill_step_chunk: Some(4),
        active_decode_prefill_chunk: Some(4),
        active_decode_prefill_budget: None,
        decode_pressure_threshold: 3,
        default_active_chunk: 4,
        chunking: true,
    }
}

fn members(wave: &[GroupMember]) -> Vec<(usize, CostWork)> {
    wave.iter()
        .map(|member| (member.request, member.work))
        .collect()
}

fn advance(requests: &mut [SimulatedRequest], wave: &[GroupMember]) {
    for member in wave {
        let request = &mut requests[member.request];
        if member.work.kind == AttemptWorkKind::Prefill {
            assert_eq!(
                member.work.requested.as_range().start,
                request.prefill_tokens
            );
            request.prefill_tokens = member.work.requested.as_range().end;
            if request.prefill_tokens == request.prompt_tokens {
                request.output_tokens = 1;
            }
        } else {
            request.output_tokens += 1;
        }
    }
}

#[test]
fn admission_cap_preview_refills_budget_without_extending_its_target_set() {
    let mut requests = vec![request(1, 1), request(7, 0), request(7, 0)];
    let target = PrefillTarget {
        request: 1,
        prompt_tokens: 7,
    };
    // The third member's existing cap remains three; it is not a new target.
    let preview = trajectory(vec![target], vec![shape(None), shape(None), shape(Some(3))]);
    let installed = trajectory(
        Vec::new(),
        vec![shape(None), shape(Some(1)), shape(Some(3))],
    );
    let baseline = preview.natural_members(&requests).unwrap();
    assert_eq!(
        baseline
            .iter()
            .map(|m| (m.request, m.work.requested.as_range().len()))
            .collect::<Vec<_>>(),
        vec![(0, 1), (1, 4)]
    );
    let limit = NonZeroUsize::new(1).unwrap();
    assert!(preview.allows_cap_commit(&requests, limit));
    let first = preview
        .natural_members_with_phase(&requests, Some((limit, &preview.cap_commit_targets)))
        .unwrap();
    assert_eq!(
        first
            .iter()
            .map(|m| (m.request, m.work.requested.as_range().len()))
            .collect::<Vec<_>>(),
        vec![(0, 1), (1, 1), (2, 3)],
        "freed nominal tokens belong to the normal first-wave packing policy"
    );
    assert_eq!(
        members(&first),
        members(&installed.natural_members(&requests).unwrap())
    );

    let mut reached_common_continuation = false;
    // This finite three-request fixture advances positive prompt/output work
    // on every wave. Compare complete work descriptors through all endpoints.
    let bound = requests
        .iter()
        .map(|r| r.prompt_tokens + r.output_cap)
        .sum::<usize>();
    for _ in 0..bound {
        if requests.iter().all(SimulatedRequest::finished) {
            break;
        }
        let active = requests[target.request].prefill_tokens < target.prompt_tokens;
        reached_common_continuation |= !active;
        let wave = preview
            .natural_members_with_phase(
                &requests,
                active.then_some((limit, preview.cap_commit_targets.as_slice())),
            )
            .unwrap();
        assert_eq!(
            members(&wave),
            members(&installed.natural_members(&requests).unwrap())
        );
        advance(&mut requests, &wave);
    }
    assert!(reached_common_continuation);
    assert!(requests.iter().all(SimulatedRequest::finished));
    assert_eq!(preview.cap_commit_targets, vec![target]);
    assert_eq!(installed.shapes[2].phase_cap.unwrap().get(), 3);
}
