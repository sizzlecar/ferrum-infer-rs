use super::super::super::work_policy::PlanningWorkPolicy;
use super::*;

#[test]
fn frontier_keeps_all_obligations_but_uses_split_and_total_work_limits() {
    let mut s = snapshot(vec![decode(1), prefill(2), prefill(3)]);
    s.capabilities.prefill_batch_sizes = vec![nz(1), nz(2)];
    s.capabilities.work_policy = PlanningWorkPolicy {
        maximum_wave_tokens: 10,
        active_decode_prefill_chunk: Some(4),
        active_decode_prefill_token_budget: Some(4),
        allow_mixed: false,
        ..Default::default()
    };
    let before = s.clone();
    let set = candidates::enumerate(
        &s,
        &s.requests,
        100,
        16,
        None,
        &TestResolver,
        &mut || Ok(()),
    )
    .unwrap();
    assert!(set.waves.iter().any(|wave| wave
        .work
        .iter()
        .any(|row| matches!(row.action, WaveAction::Prefill { .. }))));
    assert!(set.waves.iter().any(|wave| wave
        .work
        .iter()
        .any(|row| matches!(row.action, WaveAction::Decode))));
    for wave in set.waves {
        assert!(candidates::within_work_envelope(
            &s.capabilities,
            &s.requests,
            &wave.work,
            &mut || Ok(())
        )
        .unwrap());
        assert_eq!(wave.work.len(), 1);
    }
    assert_eq!(s, before);
    assert_eq!(s.requests.len(), 3);
}

#[test]
fn final_prefill_activates_successor_policy_without_locking_prompt_frontier() {
    let mut first = prefill(1);
    first.timing.maximum_output_tokens = n32(2);
    let mut s = snapshot(vec![first, prefill(2)]);
    s.capabilities.work_policy.active_decode_prefill_chunk = Some(4);
    let context = execution::ReplayContext {
        resolver: &TestResolver,
        resources: None,
    };
    let start =
        simulation::begin(&s, &context, &mut || Ok(()), 100).unwrap_or_else(|_| panic!("begin"));
    let model = Model(|_: &WaveExecutionShape| Some(1));
    let row = |index: usize, offset: u32, count: u32| CandidateWork {
        key: s.requests[index].key.clone(),
        action: WaveAction::Prefill {
            offset,
            count: n32(count),
        },
    };
    let after_first = simulation::advance(
        &s,
        &start,
        &[row(0, 0, 8)],
        &model,
        false,
        &mut || Ok(()),
        true,
        None,
    )
    .unwrap_or_else(|_| panic!("legal pure prefill"))
    .state;
    assert!(matches!(
        after_first.requests[0].phase,
        RequestPhaseView::Decode
    ));
    assert!(simulation::advance(
        &s,
        &after_first,
        &[row(1, 0, 8)],
        &model,
        false,
        &mut || Ok(()),
        true,
        None
    )
    .is_err());
    let partial = simulation::advance(
        &s,
        &after_first,
        &[row(1, 0, 4)],
        &model,
        false,
        &mut || Ok(()),
        true,
        None,
    )
    .unwrap_or_else(|_| panic!("active capped partial"))
    .state;
    let final_state = simulation::advance(
        &s,
        &partial,
        &[row(1, 4, 4)],
        &model,
        false,
        &mut || Ok(()),
        true,
        None,
    )
    .unwrap_or_else(|_| panic!("same owner can continue past first cap"))
    .state;
    assert_eq!(final_state.requests[1].timing.committed_tokens, 1);
    let RequestPhaseView::Prefill(original) = &s.requests[1].phase else {
        panic!("source")
    };
    assert_eq!(original.executable_until, 8);
    assert_eq!(original.offset, 0);
}

#[test]
fn work_policy_cannot_authorize_an_unaligned_or_missing_reference_endpoint() {
    let mut s = snapshot(vec![decode(1), prefill(2)]);
    s.capabilities.work_policy.active_decode_prefill_chunk = Some(3);
    let work = [CandidateWork {
        key: s.requests[1].key.clone(),
        action: WaveAction::Prefill {
            offset: 0,
            count: n32(3),
        },
    }];
    assert!(
        candidates::within_work_envelope(&s.capabilities, &s.requests, &work, &mut || Ok(()))
            .unwrap()
    );
    assert!(shape::legal_rows(&s, &s.requests, &work, &mut || true).is_none());
    s.capabilities.prefill_alignment = n32(1);
    assert!(
        shape::legal_rows(&s, &s.requests, &work, &mut || true).is_none(),
        "missing reference endpoint remains unavailable"
    );
}

#[test]
fn blocked_decoder_does_not_activate_caps_or_disappear_from_timing_checks() {
    let mut d = decode(1);
    d.readiness = RequestReadiness::OutputBlocked;
    d.timing.budgets.itl_ns = n64(12); // due102
    let mut s = snapshot(vec![d, prefill(2)]);
    s.capabilities.work_policy.active_decode_prefill_chunk = Some(4);
    let context = execution::ReplayContext {
        resolver: &TestResolver,
        resources: None,
    };
    let start =
        simulation::begin(&s, &context, &mut || Ok(()), 100).unwrap_or_else(|_| panic!("begin"));
    let work = [CandidateWork {
        key: s.requests[1].key.clone(),
        action: WaveAction::Prefill {
            offset: 0,
            count: n32(8),
        },
    }];
    assert!(
        candidates::within_work_envelope(&s.capabilities, &s.requests, &work, &mut || Ok(()))
            .unwrap()
    );
    assert!(simulation::advance(
        &s,
        &start,
        &work,
        &Model(|_: &WaveExecutionShape| Some(3)),
        false,
        &mut || Ok(()),
        true,
        None
    )
    .is_err());
    assert_eq!(start.requests.len(), 2);
}
