use super::*;

fn fill_policy(target: usize, active_chunk: Option<usize>) -> SchedulerConfig {
    SchedulerConfig {
        prefill_first_until_active: Some(target),
        active_decode_prefill_chunk: active_chunk,
        prefill_step_chunk: Some(8),
        ..SchedulerConfig::default()
    }
}

#[test]
fn joint_fill_first_preserves_armed_cold_wave() {
    for execution in [
        CostExecution::SingleInvocation,
        CostExecution::SequentialPrefillThenDecode,
    ] {
        let (scheduler, mut round, prefill, decoder, now) =
            fixture_with_policy(execution, true, true, fill_policy(4, Some(8)));
        let batch = next(&scheduler, &mut round, now);
        assert_eq!(tokens(&batch, &prefill), 8);
        assert!(batch.requests.iter().all(|item| item.request.id != decoder));
        assert!(scheduler
            .fill_first_initial_cohort_armed
            .load(Ordering::Acquire));
        assert_eq!(round.statistics.joint_unknown_resources, 1);
        let reasons = serde_json::to_value(round.statistics.joint_unsupported).unwrap();
        assert_eq!(reasons["cold_fill_first"], 1);
        assert_eq!(reasons["unknown_kv_capacity"], 0);
        let mut snapshot = crate::implementations::continuous::SchedulerSloSnapshot::default();
        snapshot.record(&round);
        snapshot.record(&round);
        let health = serde_json::to_value(snapshot).unwrap();
        assert_eq!(health["joint_unsupported"]["cold_fill_first"], 2);
        assert_eq!(health["joint_unknown_resources"], 2);
        assert_eq!(round.statistics.joint_cost_snapshot_queries, 0);
        assert_eq!(round.statistics.joint_interventions, 0);
    }
}

#[test]
fn joint_fill_first_actual_decode_disarms_before_comparing_first_wave() {
    for execution in [
        CostExecution::SingleInvocation,
        CostExecution::SequentialPrefillThenDecode,
    ] {
        for active_chunk in [None, Some(16)] {
            let (scheduler, mut round, prefill, decoder, now) =
                fixture_with_policy(execution, true, true, fill_policy(2, active_chunk));
            assert!(scheduler
                .fill_first_initial_cohort_armed
                .load(Ordering::Acquire));
            let batch = next(&scheduler, &mut round, now);
            assert!(!scheduler
                .fill_first_initial_cohort_armed
                .load(Ordering::Acquire));
            assert_eq!(tokens(&batch, &decoder), 1);
            assert_eq!(tokens(&batch, &prefill), 4);
            assert_eq!(round.statistics.joint_unknown_resources, 0);
            assert_eq!(round.statistics.joint_interventions, 1);
            assert!(round.statistics.joint_cost_snapshot_queries > 0);
            assert_eq!(scheduler.fill_first_dynamic_admission_limit(&hint(), 2), 0);
        }
    }
}

#[tokio::test]
async fn joint_fill_first_late_arrivals_stay_disarmed_and_full_idle_rearms() {
    let (scheduler, mut round, prefill, decoder, now) = fixture_with_policy(
        CostExecution::SequentialPrefillThenDecode,
        true,
        true,
        fill_policy(2, Some(16)),
    );
    let first = next(&scheduler, &mut round, now);
    assert_eq!(tokens(&first, &decoder), 1);
    assert!(!scheduler.mark_prefill_chunk_processed(&prefill, 16, tokens(&first, &prefill)));
    scheduler.update_decode_progress(&decoder, 3);

    let mut facts = round.request_facts().unwrap().to_vec();
    let mut clocks = round.request_clocks().unwrap().to_vec();
    let decoder_fact = &mut facts.iter_mut().find(|(id, _)| id == &decoder).unwrap().1;
    decoder_fact.generated_tokens = 3;
    decoder_fact.verified_survival_through = 3;
    let decoder_clock = &mut clocks.iter_mut().find(|(id, _)| id == &decoder).unwrap().1;
    decoder_clock.generated_tokens = 3;
    decoder_clock.nonempty_text_events = 3;

    let mut late = create_test_request_with_prompt_tokens(Priority::Normal, 16);
    late.sampling_params.max_tokens = 2;
    let late_id = late.id.clone();
    let mut late_fact = facts
        .iter()
        .find(|(id, _)| id == &prefill)
        .unwrap()
        .1
        .clone();
    late_fact.lifecycle.admission = 3;
    let late_clock = clocks.iter().find(|(id, _)| id == &prefill).unwrap().1;
    scheduler.submit(late).await.unwrap();
    assert!(!scheduler
        .fill_first_initial_cohort_armed
        .load(Ordering::Acquire));
    assert_eq!(scheduler.fill_first_dynamic_admission_limit(&hint(), 2), 0);
    facts.push((late_id.clone(), late_fact.clone()));
    clocks.push((late_id.clone(), late_clock));
    assert!(round.set_request_facts(facts));
    assert!(round.set_request_clocks(clocks));
    admit(&scheduler, None);
    let prior_queries = round.statistics.joint_cost_snapshot_queries;
    let late_batch = next(&scheduler, &mut round, now);
    assert_eq!(tokens(&late_batch, &decoder), 1);
    assert_eq!(round.statistics.joint_unknown_resources, 0);
    assert!(round.statistics.joint_cost_snapshot_queries > prior_queries);

    for id in [prefill, decoder, late_id] {
        assert!(scheduler.cancel(id).await.unwrap());
    }
    assert_eq!(scheduler.active_count() + scheduler.waiting_count(), 0);
    let mut fresh = create_test_request_with_prompt_tokens(Priority::Normal, 16);
    fresh.sampling_params.max_tokens = 2;
    let fresh_id = fresh.id.clone();
    scheduler.submit(fresh).await.unwrap();
    assert!(scheduler
        .fill_first_initial_cohort_armed
        .load(Ordering::Acquire));
    late_fact.lifecycle.admission = 4;
    assert!(round.set_request_facts([(fresh_id.clone(), late_fact)]));
    assert!(round.set_request_clocks([(fresh_id.clone(), late_clock)]));
    admit(&scheduler, None);
    let prior_queries = round.statistics.joint_cost_snapshot_queries;
    let prior_interventions = round.statistics.joint_interventions;
    let fresh_batch = next(&scheduler, &mut round, now);
    assert_eq!(tokens(&fresh_batch, &fresh_id), 8);
    assert_eq!(round.statistics.joint_unknown_resources, 1);
    assert_eq!(round.statistics.joint_cost_snapshot_queries, prior_queries);
    assert_eq!(round.statistics.joint_interventions, prior_interventions);
}
