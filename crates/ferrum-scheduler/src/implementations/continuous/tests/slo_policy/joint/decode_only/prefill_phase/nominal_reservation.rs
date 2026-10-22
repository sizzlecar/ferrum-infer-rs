//! Actual scheduling separates executed tokens from a committed policy debit.
use super::*;
use crate::implementations::continuous::prefill_budget::{PrefillPacking, PrefillPackingMode};
use std::num::NonZeroUsize;

fn reserved_selection(fixture: &PhaseFixture, max_tokens: usize) -> (BatchPlan, TimePlanningRound) {
    let mut bound = hint();
    bound.max_tokens = max_tokens;
    let mut round = fixture.round();
    let batch = fixture
        .scheduler
        .next_batch_with_token_admissions(
            fixture.scheduler.prepare_token_batch_selection(bound),
            None,
            &mut |_| {},
            Some(&mut round),
        )
        .unwrap()
        .batch
        .unwrap();
    (batch, round)
}

fn assert_actual_work(
    scheduler: &ContinuousBatchScheduler,
    batch: &BatchPlan,
    round: &TimePlanningRound,
) {
    let prefill = scheduler.prefill_queue.read();
    let decode = scheduler.decode_queue.read();
    assert_eq!(round.selected.len(), batch.requests.len());
    let mut actual_tokens = 0;
    for (scheduled, work) in batch.requests.iter().zip(&round.selected) {
        let request = prefill
            .iter()
            .chain(decode.requests.values())
            .find(|request| request.inner.request.id == scheduled.request.id)
            .unwrap();
        let tokens = scheduled.tokens_to_process.unwrap();
        actual_tokens += tokens;
        assert_eq!(work.requested.as_range().len(), tokens);
        assert_eq!(
            scheduler.cost_work(
                request,
                tokens,
                request.phase == RequestPhase::Decoding,
                round.prefill_shape_policy,
            ),
            Some((*work, true)),
        );
    }
    assert_eq!(
        batch.resource_requirements.gpu_memory,
        (actual_tokens * 16) as u64
    );
    assert_eq!(
        batch.resource_requirements.cpu_memory,
        (actual_tokens * 4) as u64
    );
    assert_eq!(
        batch.resource_requirements.kv_cache_blocks,
        actual_tokens / 16
    );
}

fn make_costs_stale(fixture: &mut PhaseFixture) {
    fixture.source = Arc::new(PhaseCosts {
        base: ConditionalSnapshot {
            at: fixture.admitted_at,
            stopping_known: true,
            completion_known: true,
        },
        unavailable: true,
    });
}

#[test]
fn nominal_reservation_is_selected_and_survives_until_the_real_prompt_fence() {
    let mut fixture = PhaseFixture::with_capacity(3);
    let before = fixture.scheduler.prefill_queue.read()[0]
        .logical_work_frontier
        .clone();
    let (first, first_round) = reserved_selection(&fixture, 5);
    assert_eq!(tokens(&first, &fixture.prefill), 1);
    assert_eq!(tokens(&first, &fixture.decoder), 1);
    assert_eq!(first_round.statistics.joint_prefill_phase_interventions, 1);
    assert_actual_work(&fixture.scheduler, &first, &first_round);
    {
        let prefill = fixture.scheduler.prefill_queue.read();
        let request = &prefill[0];
        let cap = request.joint_prefill_phase_cap.unwrap();
        assert_eq!(cap.limit.get(), 1);
        assert_eq!(
            cap.packing,
            PrefillPacking::ReservePolicyChunk { prior_limit: None }
        );
        let mut expected = before;
        expected.mark_scheduled(1);
        assert_eq!(request.logical_work_frontier, expected);
    }
    assert!(!fixture
        .scheduler
        .mark_prefill_chunk_processed(&fixture.prefill, 32, 1));
    fixture
        .scheduler
        .update_decode_progress(&fixture.decoder, 2);

    // The new request was not part of the comparison or its target set.
    // Actual admission still uses the existing executor receipt path.
    let mut newcomer = create_test_request_with_prompt_tokens(Priority::Normal, 32);
    newcomer.sampling_params.max_tokens = 64;
    newcomer.stream = true;
    let newcomer_id = newcomer.id.clone();
    enqueue_waiting(&fixture.scheduler, newcomer);
    admit(&fixture.scheduler, None);
    make_costs_stale(&mut fixture);

    for offset in 1..32 {
        let before = fixture
            .scheduler
            .prefill_queue
            .read()
            .iter()
            .find(|request| request.inner.request.id == fixture.prefill)
            .unwrap()
            .logical_work_frontier
            .clone();
        let (batch, round) = reserved_selection(&fixture, 5);
        assert_eq!(tokens(&batch, &fixture.prefill), 1);
        assert_eq!(tokens(&batch, &fixture.decoder), 1);
        assert_eq!(round.statistics.joint_prefill_phase_interventions, 0);
        assert_actual_work(&fixture.scheduler, &batch, &round);
        let newcomer_work = batch
            .requests
            .iter()
            .find(|item| item.request.id == newcomer_id);
        assert!(
            newcomer_work.is_none(),
            "the four-token policy debit survives even the final short prompt tail"
        );
        {
            let queue = fixture.scheduler.prefill_queue.read();
            let request = queue
                .iter()
                .find(|request| request.inner.request.id == fixture.prefill)
                .unwrap();
            let mut expected = before;
            expected.mark_scheduled(1);
            assert_eq!(request.logical_work_frontier, expected);
            assert_eq!(request.prefill_chunk_offset, offset);
            assert!(queue
                .iter()
                .find(|request| request.inner.request.id == newcomer_id)
                .unwrap()
                .joint_prefill_phase_cap
                .is_none());
        }
        let completed = fixture
            .scheduler
            .mark_prefill_chunk_processed(&fixture.prefill, 32, 1);
        assert_eq!(completed, offset == 31);
        fixture
            .scheduler
            .update_decode_progress(&fixture.decoder, offset + 2);
    }
    assert_eq!(
        fixture
            .scheduler
            .prefill_queue
            .read()
            .iter()
            .find(|request| request.inner.request.id == newcomer_id)
            .unwrap()
            .prefill_chunk_offset,
        0,
    );
    fixture
        .scheduler
        .update_decode_progress(&fixture.prefill, 1);
    assert!(
        fixture.scheduler.decode_queue.read().requests[&fixture.prefill]
            .joint_prefill_phase_cap
            .is_none()
    );
    let (after_fence, round) = reserved_selection(&fixture, 5);
    assert_eq!(tokens(&after_fence, &fixture.decoder), 1);
    assert_eq!(tokens(&after_fence, &fixture.prefill), 1);
    assert_eq!(tokens(&after_fence, &newcomer_id), 3);
    assert_eq!(
        after_fence
            .requests
            .iter()
            .find(|request| request.request.id == newcomer_id)
            .unwrap()
            .tokens_processed,
        0
    );
    assert_actual_work(&fixture.scheduler, &after_fence, &round);
    assert_eq!(round.statistics.joint_prefill_phase_interventions, 0);
}

#[derive(Clone, Copy, Debug)]
enum AdmissionPath {
    Legacy,
    Dynamic,
    Prepared,
}

#[test]
fn nominal_reservation_uses_prior_cap_and_one_ledger_across_both_prefill_passes() {
    for path in [
        AdmissionPath::Legacy,
        AdmissionPath::Dynamic,
        AdmissionPath::Prepared,
    ] {
        let mut fixture = PhaseFixture::with_capacity(3);
        let prefill_before = {
            let mut queue = fixture.scheduler.prefill_queue.write();
            let request = &mut queue[0];
            let prior = PrefillPhaseCap::selected(
                None,
                NonZeroUsize::new(2).unwrap(),
                32,
                PrefillPackingMode::Refill,
            );
            request.joint_prefill_phase_cap = Some(PrefillPhaseCap::selected(
                Some(prior),
                NonZeroUsize::new(1).unwrap(),
                32,
                PrefillPackingMode::ReservePolicyChunk,
            ));
            request.logical_work_frontier.clone()
        };
        let mut new_request = create_test_request_with_prompt_tokens(Priority::Normal, 32);
        new_request.sampling_params.max_tokens = 64;
        new_request.stream = true;
        let newcomer = new_request.id.clone();
        enqueue_waiting(&fixture.scheduler, new_request);
        if matches!(path, AdmissionPath::Prepared) {
            admit(&fixture.scheduler, None);
        }
        let mut newcomer_before = if matches!(path, AdmissionPath::Prepared) {
            fixture
                .scheduler
                .prefill_queue
                .read()
                .iter()
                .find(|request| request.inner.request.id == newcomer)
                .unwrap()
                .logical_work_frontier
                .clone()
        } else {
            fixture
                .scheduler
                .waiting_queue
                .read()
                .iter()
                .find(|request| request.inner.request.id == newcomer)
                .unwrap()
                .logical_work_frontier
                .clone()
        };
        if !matches!(path, AdmissionPath::Prepared) {
            newcomer_before.begin_prefill(false);
        }
        make_costs_stale(&mut fixture);
        let mut round = fixture.round();
        let mut bound = hint();
        bound.max_tokens = 5;
        let mut probes = Vec::new();
        let mut probe = |request: &InferenceRequest| {
            probes.push(request.id.clone());
            AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                request_id: request.id.clone(),
            })
        };
        let mut observer = |_| {};
        let mode = match path {
            AdmissionPath::Legacy => WaitingAdmissionMode::Legacy,
            AdmissionPath::Dynamic => WaitingAdmissionMode::Dynamic {
                wake: None,
                probe: &mut probe,
                observer: None,
            },
            AdmissionPath::Prepared => WaitingAdmissionMode::Prepared {
                wake: None,
                observer: &mut observer,
                initial_priority: Some(false),
            },
        };
        let batch = fixture
            .scheduler
            .create_iteration_batch_with_time(bound, mode, Instant::now(), Some(&mut round))
            .unwrap()
            .unwrap();
        assert_eq!(
            batch
                .requests
                .iter()
                .map(|item| (&item.request.id, item.tokens_to_process.unwrap()))
                .collect::<Vec<_>>(),
            vec![(&fixture.decoder, 1), (&fixture.prefill, 1), (&newcomer, 2)],
            "{path:?}: debit is decode1 + prior-cap2 + newcomer2, while actual work is four tokens",
        );
        assert_actual_work(&fixture.scheduler, &batch, &round);
        assert_eq!(fixture.scheduler.waiting_count(), 0);
        assert_eq!(fixture.scheduler.active_count(), 3);
        if matches!(path, AdmissionPath::Dynamic) {
            assert_eq!(probes, vec![newcomer.clone()]);
        } else {
            assert!(probes.is_empty());
        }
        let queue = fixture.scheduler.prefill_queue.read();
        let owner = queue
            .iter()
            .find(|request| request.inner.request.id == fixture.prefill)
            .unwrap();
        let mut expected = prefill_before;
        expected.mark_scheduled(1);
        assert_eq!(owner.logical_work_frontier, expected);
        assert_eq!(owner.prefill_chunk_offset, 0);
        assert_eq!(
            owner.joint_prefill_phase_cap.unwrap().packing,
            PrefillPacking::ReservePolicyChunk {
                prior_limit: NonZeroUsize::new(2)
            }
        );
        let newcomer = queue
            .iter()
            .find(|request| request.inner.request.id == newcomer)
            .unwrap();
        newcomer_before.mark_scheduled(2);
        assert_eq!(newcomer.logical_work_frontier, newcomer_before);
        assert!(newcomer.joint_prefill_phase_cap.is_none());
        assert_eq!(round.statistics.joint_interventions, 0);
    }
}

#[test]
fn nominal_reservation_apply_rejects_same_generation_cap_or_packing_change_atomically() {
    for packing_only in [false, true] {
        let mut fixture = PhaseFixture::with_capacity(3);
        fixture.config.itl_ms = 25_000.0;
        fixture.config.joint.itl_quantile = 0.99;
        fixture.scheduler.config.slo = Some(fixture.config);
        let mut other = create_test_request_with_prompt_tokens(Priority::Normal, 32);
        other.sampling_params.max_tokens = 64;
        other.stream = true;
        let other_id = other.id.clone();
        enqueue_waiting(&fixture.scheduler, other);
        admit(&fixture.scheduler, None);
        {
            let mut queue = fixture.scheduler.prefill_queue.write();
            for request in queue.iter_mut() {
                request.joint_prefill_phase_cap = Some(PrefillPhaseCap::selected(
                    None,
                    NonZeroUsize::new(4).unwrap(),
                    32,
                    PrefillPackingMode::Refill,
                ));
            }
        }
        let mut batch = select(&fixture.scheduler, None).batch.unwrap();
        let mut round = fixture.round();
        {
            let prefill = fixture.scheduler.prefill_queue.read();
            let decode = fixture.scheduler.decode_queue.read();
            for scheduled in &batch.requests {
                let request = prefill
                    .iter()
                    .chain(decode.requests.values())
                    .find(|request| request.inner.request.id == scheduled.request.id)
                    .unwrap();
                round.selected_work(fixture.scheduler.cost_work(
                    request,
                    scheduled.tokens_to_process.unwrap(),
                    request.phase == RequestPhase::Decoding,
                    round.prefill_shape_policy,
                ));
            }
        }
        let choice = fixture
            .scheduler
            .compare_joint_first_wave(&hint(), &batch.requests, &mut round, Instant::now())
            .unwrap()
            .expect("a finite reserved phase rescues this cohort");
        let generation = {
            let mut queue = fixture.scheduler.prefill_queue.write();
            let request = queue
                .iter_mut()
                .find(|request| request.inner.request.id == other_id)
                .unwrap();
            let generation = request.logical_work_frontier.progress_generation();
            request.joint_prefill_phase_cap = Some(PrefillPhaseCap::selected(
                request.joint_prefill_phase_cap,
                NonZeroUsize::new(if packing_only { 4 } else { 3 }).unwrap(),
                32,
                if packing_only {
                    PrefillPackingMode::ReservePolicyChunk
                } else {
                    PrefillPackingMode::Refill
                },
            ));
            generation
        };
        let before = fixture
            .scheduler
            .prefill_queue
            .read()
            .iter()
            .map(|request| {
                (
                    request.inner.request.id.clone(),
                    request.joint_prefill_phase_cap,
                    request.logical_work_frontier.clone(),
                )
            })
            .collect::<Vec<_>>();
        let selected_before = round.selected.clone();
        assert!(!fixture
            .scheduler
            .apply_joint_first_wave(&mut batch.requests, &mut round, choice));
        let after = fixture
            .scheduler
            .prefill_queue
            .read()
            .iter()
            .map(|request| {
                (
                    request.inner.request.id.clone(),
                    request.joint_prefill_phase_cap,
                    request.logical_work_frontier.clone(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            after, before,
            "no earlier member may acquire a partial commitment"
        );
        assert_eq!(round.selected, selected_before);
        assert_eq!(tokens(&batch, &fixture.prefill), 4);
        assert_eq!(tokens(&batch, &other_id), 4);
        assert_eq!(
            after
                .iter()
                .find(|(id, _, _)| id == &other_id)
                .unwrap()
                .2
                .progress_generation(),
            generation
        );
        assert_eq!(round.statistics.joint_interventions, 0);
    }
}

#[test]
fn nominal_reservation_preserves_rendezvous_release_and_actual_work_boundaries() {
    use ferrum_execution::model_executor::PrefixCapturePlan;
    use ferrum_runtime::CheckpointTokenSpanConstraint;
    use std::num::NonZeroU64;

    for packing in [
        PrefillPackingMode::Refill,
        PrefillPackingMode::ReservePolicyChunk,
    ] {
        for legal_capture_chunk in [true, false] {
            let mut fixture = PhaseFixture::with_capacity(4);
            fixture.scheduler.config.prefix_rendezvous_max_wait_ms = NonZeroU64::new(10_000);
            {
                let mut queue = fixture.scheduler.prefill_queue.write();
                queue[0].joint_prefill_phase_cap = Some(PrefillPhaseCap::selected(
                    None,
                    NonZeroUsize::new(3).unwrap(),
                    32,
                    packing,
                ));
            }
            let mut follower = create_test_request_with_prompt_tokens(Priority::Normal, 32);
            follower.sampling_params.max_tokens = 64;
            let follower_id = follower.id.clone();
            enqueue_waiting(&fixture.scheduler, follower);
            let mut other = create_test_request_with_prompt_tokens(Priority::Normal, 32);
            other.sampling_params.max_tokens = 64;
            let other_id = other.id.clone();
            enqueue_waiting(&fixture.scheduler, other);
            let candidates = fixture.scheduler.prefix_rendezvous_candidates();
            let source_key = &candidates
                .iter()
                .find(|item| item.key.request_id() == &fixture.prefill)
                .unwrap()
                .key;
            let follower_key = &candidates
                .iter()
                .find(|item| item.key.request_id() == &follower_id)
                .unwrap()
                .key;
            let capture = if legal_capture_chunk {
                PrefixCapturePlan {
                    boundary: 2,
                    span: CheckpointTokenSpanConstraint::any_positive(),
                }
            } else {
                PrefixCapturePlan {
                    boundary: 8,
                    span: CheckpointTokenSpanConstraint::new(
                        NonZeroU64::new(4).unwrap(),
                        NonZeroU64::new(4).unwrap(),
                    )
                    .unwrap(),
                }
            };
            let hold = fixture
                .scheduler
                .hold_prefix_followers(source_key, std::slice::from_ref(follower_key), capture)
                .unwrap();
            admit(&fixture.scheduler, None);
            assert_eq!(fixture.scheduler.active_count(), 3);
            assert_eq!(fixture.scheduler.waiting_count(), 1);
            assert!(hold.is_pending());
            make_costs_stale(&mut fixture);
            let (batch, round) = reserved_selection(&fixture, 5);
            let executed = if legal_capture_chunk { 2 } else { 3 };
            assert_eq!(tokens(&batch, &fixture.decoder), 1);
            assert_eq!(tokens(&batch, &fixture.prefill), executed);
            let other_work = batch
                .requests
                .iter()
                .find(|item| item.request.id == other_id);
            match packing {
                PrefillPackingMode::ReservePolicyChunk => assert!(other_work.is_none()),
                PrefillPackingMode::Refill => {
                    assert_eq!(other_work.unwrap().tokens_to_process, Some(4 - executed))
                }
            }
            assert_actual_work(&fixture.scheduler, &batch, &round);
            assert_eq!(hold.is_pending(), legal_capture_chunk);
            assert_eq!(
                fixture.scheduler.prefix_held_waiting_count(),
                usize::from(legal_capture_chunk)
            );
            assert_eq!(
                fixture.scheduler.prefix_request_progress(follower_key),
                Some((true, 0))
            );
            assert_eq!(
                fixture.scheduler.active_count(),
                3,
                "release is not executor admission"
            );
            assert!(!fixture.scheduler.mark_prefill_chunk_processed(
                &fixture.prefill,
                32,
                executed
            ));

            // A legal boundary remains held until the capture owner releases
            // it. An impossible span already released without enlarging work.
            hold.release();
            admit(&fixture.scheduler, None);
            assert_eq!(
                fixture.scheduler.prefix_request_progress(follower_key),
                Some((false, 0))
            );
            assert_eq!(fixture.scheduler.active_count(), 4);
            assert_eq!(fixture.scheduler.waiting_count(), 0);
            assert!(fixture
                .scheduler
                .prefill_queue
                .read()
                .iter()
                .find(|request| request.inner.request.id == follower_id)
                .unwrap()
                .joint_prefill_phase_cap
                .is_none());
        }
    }
}
