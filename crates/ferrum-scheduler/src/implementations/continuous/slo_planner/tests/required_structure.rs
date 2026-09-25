use super::*;

fn work(request: &RequestSchedulingView, action: WaveAction) -> CandidateWork {
    CandidateWork {
        key: request.key.clone(),
        action,
    }
}
fn p(offset: u32, count: u32) -> WaveAction {
    WaveAction::Prefill {
        offset,
        count: n32(count),
    }
}
fn rejected(snapshot: &SchedulerSnapshot, work: &[CandidateWork]) {
    let initial = snapshot.requests.clone();
    assert!(
        super::super::shape::validate_work(snapshot, &initial, work, &mut || Ok(()))
            .unwrap()
            .is_none()
    );
    assert!(matches!(
        PlanningStructure::new(snapshot).advance(work, &mut || Ok(())),
        Err(PlanningUnknownReason::InvalidShapeEvidence)
    ));
    assert_eq!(snapshot.requests, initial);
}

#[test]
fn required_future_structure_uses_production_prefill_and_readiness_rules() {
    let mut snap = snapshot(vec![prefill(1)]);
    let legal = vec![work(&snap.requests[0], p(0, 4))];
    let first = PlanningStructure::new(&snap)
        .advance(&legal, &mut || Ok(()))
        .unwrap();
    assert_eq!(first.requests()[0].context_tokens, 4);
    assert_eq!(
        first.requests()[0].output_credit,
        snap.requests[0].output_credit
    );
    rejected(&snap, &[work(&snap.requests[0], p(0, 2))]); // undeclared, unaligned endpoint
    rejected(&snap, &[work(&snap.requests[0], p(4, 4))]); // wrong current offset
    snap.capabilities.max_prefill_tokens_per_wave = n64(3);
    rejected(&snap, &legal);
    snap.capabilities.max_prefill_tokens_per_wave = n64(32);
    snap.requests[0].readiness = RequestReadiness::OutputBlocked;
    rejected(&snap, &legal);
    snap.requests[0].readiness = RequestReadiness::Ready;
    if let RequestPhaseView::Prefill(progress) = &mut snap.requests[0].phase {
        progress.executable_until = 3;
    }
    rejected(&snap, &legal);

    let mut missing = snapshot(vec![prefill(1)]);
    if let RequestPhaseView::Prefill(progress) = &mut missing.requests[0].phase {
        Arc::make_mut(&mut progress.reference)
            .points
            .retain(|point| point.prompt_tokens != 4);
    }
    rejected(&missing, &[work(&missing.requests[0], p(0, 4))]); // no reference endpoint
}

#[test]
fn required_future_structure_recomputes_active_decode_and_mixed_policy() {
    let mut snap = snapshot(vec![decode(1), prefill(2)]);
    let mixed = vec![
        work(&snap.requests[0], WaveAction::Decode),
        work(&snap.requests[1], p(0, 4)),
    ];
    assert!(PlanningStructure::new(&snap)
        .advance(&mixed, &mut || Ok(()))
        .is_ok());
    snap.capabilities.native_mixed = false;
    rejected(&snap, &mixed);
    snap.capabilities.native_mixed = true;
    snap.capabilities.work_policy.allow_mixed = false;
    rejected(&snap, &mixed);
    snap.capabilities.work_policy.allow_mixed = true;
    snap.capabilities.work_policy.maximum_wave_tokens = 4;
    rejected(&snap, &mixed);
    snap.capabilities.work_policy.maximum_wave_tokens = u64::MAX;
    snap.capabilities.work_policy.active_decode_prefill_chunk = Some(2);
    rejected(&snap, &[work(&snap.requests[1], p(0, 4))]); // unselected live decoder still matters

    let mut snap = snapshot(vec![prefill(1), prefill(2)]);
    snap.requests[0].timing.maximum_output_tokens = n32(3);
    snap.capabilities.work_policy.active_decode_prefill_chunk = Some(2);
    let first = PlanningStructure::new(&snap)
        .advance(&[work(&snap.requests[0], p(0, 8))], &mut || Ok(()))
        .unwrap();
    assert!(matches!(
        first.requests()[0].phase,
        RequestPhaseView::Decode
    ));
    assert!(matches!(
        first.advance(&[work(&snap.requests[1], p(0, 4))], &mut || Ok(())),
        Err(PlanningUnknownReason::InvalidShapeEvidence)
    )); // final prefill created active decoder
}

#[test]
fn required_future_structure_consumes_credit_and_shared_bytes_without_time_or_refill() {
    let mut request = decode(1);
    request.timing.maximum_output_tokens = n32(5);
    request.output_credit.available_token_commands = 1;
    let snap = snapshot(vec![request]);
    let action = vec![work(&snap.requests[0], WaveAction::Decode)];
    let start = PlanningStructure::new(&snap);
    let next = start.advance(&action, &mut || Ok(())).unwrap();
    assert_eq!(
        next.requests()[0].timing.first_commit_at_ns,
        snap.requests[0].timing.first_commit_at_ns
    );
    assert_eq!(
        next.requests()[0].timing.last_commit_at_ns,
        snap.requests[0].timing.last_commit_at_ns
    );
    assert_eq!(next.requests()[0].timing.committed_tokens, 2);
    assert_eq!(next.requests()[0].output_credit.available_token_commands, 0);
    assert!(matches!(
        next.advance(&action, &mut || Ok(())),
        Err(PlanningUnknownReason::OutputOrResourceBlocked)
    ));
    assert_eq!(start.requests(), snap.requests);

    let mut snap = snapshot(vec![decode(1), decode(2)]);
    snap.capacity.available_output_bytes = 63; // each row needs the original bound of 32
    let action = snap
        .requests
        .iter()
        .map(|r| work(r, WaveAction::Decode))
        .collect::<Vec<_>>();
    let start = PlanningStructure::new(&snap);
    assert!(matches!(
        start.advance(&action, &mut || Ok(())),
        Err(PlanningUnknownReason::OutputOrResourceBlocked)
    ));
    assert_eq!(start.requests(), snap.requests); // failure on second row did not spend first
    assert!(start.advance(&action[..1], &mut || Ok(())).is_ok());

    let mut snap = snapshot(vec![prefill(1)]);
    snap.requests[0].output_credit.available_token_commands = 0;
    let first = PlanningStructure::new(&snap)
        .advance(&[work(&snap.requests[0], p(0, 4))], &mut || Ok(()))
        .unwrap();
    assert!(matches!(
        first.advance(&[work(&snap.requests[0], p(4, 4))], &mut || Ok(())),
        Err(PlanningUnknownReason::OutputOrResourceBlocked)
    )); // final prefill emits exactly once
}

#[test]
fn required_future_structure_preserves_prepaid_backing_and_budget_failure() {
    let mut request = decode(1);
    request.timing.maximum_output_tokens = n32(5);
    request.output_credit = OutputCreditView {
        available_token_commands: 2,
        byte_backing: OutputByteBacking::PrepaidLifetime {
            remaining_token_commands: 2,
            remaining_wire_bytes: 500,
        },
    };
    let mut snap = snapshot(vec![request]);
    snap.capacity.available_output_bytes = 0;
    let action = [work(&snap.requests[0], WaveAction::Decode)];
    let start = PlanningStructure::new(&snap);
    let one = start.advance(&action, &mut || Ok(())).unwrap();
    let two = one.advance(&action, &mut || Ok(())).unwrap();
    assert_eq!(
        two.requests()[0].output_credit.byte_backing,
        OutputByteBacking::PrepaidLifetime {
            remaining_token_commands: 0,
            remaining_wire_bytes: 500
        }
    );
    assert!(matches!(
        two.advance(&action, &mut || Ok(())),
        Err(PlanningUnknownReason::OutputOrResourceBlocked)
    ));
    let mut remaining = 4usize;
    let result = start.advance(&action, &mut || {
        remaining = remaining.saturating_sub(1);
        if remaining == 0 {
            Err(PlanningUnknownReason::ComputeBudgetExhausted)
        } else {
            Ok(())
        }
    });
    assert!(matches!(
        result,
        Err(PlanningUnknownReason::ComputeBudgetExhausted)
    ));
    assert_eq!(start.requests(), snap.requests);
}
