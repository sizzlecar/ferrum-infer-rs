use super::*;

fn prefill_work(snapshot: &SchedulerSnapshot, index: usize, offset: u32) -> CandidateWork {
    CandidateWork {
        key: snapshot.requests[index].key.clone(),
        action: WaveAction::Prefill {
            offset,
            count: n32(4),
        },
    }
}

#[test]
fn committed_milestone_credit_does_not_expire_during_a_peer_wave() {
    let mut source = prefill(1);
    source.context_tokens = 4;
    let RequestPhaseView::Prefill(progress) = &mut source.phase else {
        unreachable!()
    };
    progress.offset = 4;
    progress.logical_high_water = 4;
    progress.milestones = Arc::from([PrefillMilestone {
        at_ns: 110,
        required_reference_work_ns: 40,
    }]);
    let snapshot = snapshot(vec![source, prefill(2)]);
    let original = snapshot.clone();
    let context = execution::ReplayContext {
        resolver: &TestResolver,
        resources: None,
    };
    let model = Model(|_: &WaveExecutionShape| Some(5));
    let initial = simulation::begin(&snapshot, &context, &mut || Ok(()), 100).unwrap();
    let next = simulation::advance(
        &snapshot,
        &initial,
        &[prefill_work(&snapshot, 1, 0)],
        &model,
        false,
        &mut || Ok(()),
        true,
        None,
    )
    .unwrap_or_else(|_| panic!("the peer wave preserves the committed prefix"));
    assert!(next.state.minimum_start_slack_ns >= 6);
    // Replay crosses the old milestone's wall clock, but the source had
    // already satisfied it before either planning root was captured.
    let replay = simulation::simulate(
        &snapshot,
        &[next.wave],
        &model,
        &TestResolver,
        None,
        &mut || Ok(()),
        106,
        true,
        None,
    )
    .unwrap();
    assert_eq!(replay.now_ns, 111);
    assert_eq!(replay.requests[0], original.requests[0]);
    assert_eq!(snapshot, original);
}

#[test]
fn newly_completed_milestone_keeps_its_first_completion_slack() {
    let mut request = prefill(1);
    let RequestPhaseView::Prefill(progress) = &mut request.phase else {
        unreachable!()
    };
    progress.milestones = Arc::from([PrefillMilestone {
        at_ns: 110,
        required_reference_work_ns: 40,
    }]);
    let snapshot = snapshot(vec![request]);
    let context = execution::ReplayContext {
        resolver: &TestResolver,
        resources: None,
    };
    let model = Model(|_: &WaveExecutionShape| Some(5));
    let initial = simulation::begin(&snapshot, &context, &mut || Ok(()), 100).unwrap();
    let first = simulation::advance(
        &snapshot,
        &initial,
        &[prefill_work(&snapshot, 0, 0)],
        &model,
        false,
        &mut || Ok(()),
        true,
        None,
    )
    .unwrap_or_else(|_| panic!("first fragment completes the milestone"));
    assert_eq!(first.state.minimum_start_slack_ns, 5);
    let second = simulation::advance(
        &snapshot,
        &first.state,
        &[prefill_work(&snapshot, 0, 4)],
        &model,
        false,
        &mut || Ok(()),
        true,
        None,
    )
    .unwrap_or_else(|_| panic!("later fragment must not redate completion"));
    assert_eq!(second.state.minimum_start_slack_ns, 5);
    let waves = [first.wave, second.wave];
    let replay = simulation::simulate(
        &snapshot,
        &waves,
        &model,
        &TestResolver,
        None,
        &mut || Ok(()),
        104,
        true,
        None,
    )
    .unwrap();
    assert_eq!(replay.now_ns, 114);
    assert_eq!(replay.minimum_start_slack_ns, 1);
    assert_eq!(replay.requests[0].timing.first_commit_at_ns, Some(114));
    // A delay beyond the first completion's actual allowance still fails.
    assert!(matches!(
        simulation::simulate(
            &snapshot,
            &waves,
            &model,
            &TestResolver,
            None,
            &mut || Ok(()),
            106,
            true,
            None,
        ),
        Err(simulation::SimulationFailure::SequenceViolation)
    ));
}
