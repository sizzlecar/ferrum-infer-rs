//! Length scaling through the actual bounded dispatch entrypoint. The owner
//! proof is synthetic and scales with each declared endpoint; no GPU allocation,
//! learned support, terminal feasibility, or latency SLO is established here.

use super::*;
use ferrum_execution::{
    model_executor::{
        PersistentCapacityMember, PersistentCapacityQuery, PersistentCapacityState,
        PersistentCohortCapacitySnapshot,
    },
    token_execution::ExecutorSequenceIssuer,
};
use ferrum_interfaces::request_observation::*;
use ferrum_types::{RequestIngressAnchor, RequestIngressBoundary};
use serde_json::{json, Value};

const LOOKAHEAD: usize = 8;
const COHORT_MEMBERS: usize = 48;

fn initial_generated(index: usize) -> usize {
    usize::from((PROFILE.initial_prefillers..PROFILE.residents).contains(&index))
}

fn run_length_case(prompt_tokens: usize, output_cap: usize) -> Value {
    // Every request remains short of its declared endpoint through K waves,
    // even at the configured maximum prefill chunk. Thus early endpoint
    // completion cannot explain less simulated work in the shorter case.
    assert!(prompt_tokens > LOOKAHEAD * PROFILE.prefill_step);
    assert!(output_cap > LOOKAHEAD + 1);
    let mut config: SchedulerSloConfig = "ttft:15684.62,tpot:110.56,itl:196.55".parse().unwrap();
    config.strategy = SloStrategy::Bounded;
    config.planning.max_lookahead_rounds = LOOKAHEAD;
    config.planning.max_elapsed_us = 10_000;
    config.planning.max_queries = 4096;
    config.planning.max_scenarios = 32;
    config.validate().unwrap();
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
        max_running_requests: PROFILE.residents,
        active_decode_prefill_chunk: Some(256),
        prefill_step_chunk: Some(PROFILE.prefill_step),
        slo: Some(config),
        ..Default::default()
    });
    let ingress = RequestIngressAnchor::capture(RequestIngressBoundary::SchedulerApi);
    let ids = (0..COHORT_MEMBERS)
        .map(|_| {
            // Geometry is explicit scheduler metadata, not tokenization of
            // this fixed text. No prompt-sized token buffer is constructed.
            let mut request = InferenceRequest::new("conditional geometry", "bounded-profile")
                .with_ingress(ingress)
                .with_metadata(PROMPT_TOKENS_METADATA_KEY, json!(prompt_tokens));
            request.sampling_params.max_tokens = output_cap;
            request.stream = true;
            scheduler.submit_request(request).unwrap()
        })
        .collect::<Vec<_>>();
    let mut scan = scheduler
        .begin_token_admission(PROFILE.residents, None)
        .unwrap();
    let mut admitted = 0;
    while let Some(candidate) = scan.next(&mut |_| {}, None).unwrap() {
        let id = candidate.request().id.clone();
        assert_eq!(ids[admitted], id);
        assert!(matches!(
            scan.apply(
                candidate,
                AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt { request_id: id })
            ),
            TokenAdmissionApply::Applied { accepted: Some(_) }
        ));
        admitted += 1;
    }
    scan.finish().unwrap();
    assert_eq!(admitted, PROFILE.residents);
    assert_eq!(
        scheduler.waiting_count(),
        COHORT_MEMBERS - PROFILE.residents
    );
    for id in &ids[PROFILE.initial_prefillers..PROFILE.residents] {
        scheduler.mark_prefill_complete(id, prompt_tokens);
        scheduler.update_decode_progress(id, 1);
    }
    let captured = Instant::now();
    let source = Arc::new(CheapKnownCosts {
        observed_at: captured,
        calls: AtomicUsize::new(0),
    });
    let mut round = TimePlanningRound::new(
        source.clone(),
        config,
        PrefillShapePolicy::Identity,
        CostExecution::SequentialPrefillThenDecode,
    );
    let lifecycle = |index: usize| RequestLifecycleId {
        producer: 1,
        admission: index as u64 + 1,
    };
    let rules = Arc::new(
        EffectiveStoppingRules::new(b"cap-only".to_vec(), StoppingMechanism::CapOnly, 16).unwrap(),
    );
    assert!(
        round.set_request_facts(ids.iter().enumerate().map(|(index, id)| {
            let generated = initial_generated(index);
            (
                id.clone(),
                RequestStoppingObservation {
                    lifecycle: lifecycle(index),
                    admitted_at: captured,
                    observed_at: captured,
                    revision: 1,
                    rules: rules.clone(),
                    cap: output_cap,
                    generated_tokens: generated,
                    verified_survival_through: generated,
                    terminal: None,
                    completion: None,
                },
            )
        }))
    );
    assert!(
        round.set_request_clocks(ids.iter().enumerate().map(|(index, id)| {
            let generated = initial_generated(index);
            (
                id.clone(),
                RequestClockFacts {
                    captured_at: captured,
                    age: Duration::ZERO,
                    first_text: (generated > 0).then_some(Duration::ZERO),
                    last_text: (generated > 0).then_some(Duration::ZERO),
                    max_text_interval: None,
                    nonempty_text_events: generated as u64,
                    violating_text_intervals: 0,
                    itl_threshold: Some(Duration::from_secs_f64(config.itl_ms / 1000.0)),
                    generated_tokens: generated,
                    streaming: true,
                },
            )
        }))
    );
    // Bind the real fixture members/lifecycles and the enlarged full execution
    // envelope in each case. A stale short-context proof would silently turn
    // this into an early AdmissionCapacity fallback instead of a scaling test.
    let through = prompt_tokens.checked_add(output_cap).unwrap() - 1;
    let issuer = ExecutorSequenceIssuer::new();
    let members = ids
        .iter()
        .enumerate()
        .map(|(index, id)| PersistentCapacityMember {
            request_id: id.clone(),
            lifecycle: lifecycle(index),
            state: if index < PROFILE.residents {
                PersistentCapacityState::Resident {
                    sequence: issuer.issue_sequence().unwrap(),
                }
            } else {
                PersistentCapacityState::Waiting
            },
            through_tokens: through,
        })
        .collect::<Vec<_>>();
    let capacity = PersistentCapacityQuery {
        members: &members,
        max_live_sequences: PROFILE.residents,
        max_wave_sequences: PROFILE.residents,
        inspection_budget: Default::default(),
    };
    let proof = PersistentCohortCapacitySnapshot::from_verified_query(&capacity).unwrap();
    round.set_persistent_capacity(&capacity, Some(proof));
    assert!(round.persistent_capacity().is_some());
    let hint = BatchHint {
        max_batch_size: PROFILE.residents,
        max_tokens: PROFILE.max_tokens,
        target_latency_ms: None,
        available_memory: None,
        resource_constraints: Default::default(),
    };
    let mut observed = scheduler.observe_planning_round(Some(round));
    let intent = scheduler.prepare_token_batch_selection(hint);
    let started = Instant::now();
    let batch = scheduler
        .next_batch_with_token_admissions(intent, None, &mut |_| {}, observed.as_mut())
        .unwrap()
        .batch
        .expect("admitted cohort retains its static resource-supported batch");
    let selection_elapsed = started.elapsed();
    let round = observed.finish().unwrap();
    let calls = source.calls.load(Ordering::Relaxed);
    assert_eq!(calls as u64, round.statistics.bounded.provider_calls);
    assert_eq!(round.query_count, calls, "no terminal forecast charge");
    assert_eq!(round.statistics.bounded.comparisons, 1);
    assert_eq!(scheduler.active_count(), PROFILE.residents);
    assert_eq!(
        scheduler.waiting_count(),
        COHORT_MEMBERS - PROFILE.residents
    );
    json!({
        "prompt_tokens": prompt_tokens, "output_cap": output_cap,
        "capacity_through_tokens_per_member": through,
        "policy": config,
        "entered_cost_evaluator": calls > 0,
        "all_ranks_scored": round.statistics.bounded.evaluated == 1,
        "original_provider_calls": calls,
        "charged_queries": round.query_count,
        "charged_planning_elapsed": round.prediction_elapsed,
        "selection_elapsed": selection_elapsed,
        "bounded": round.statistics.bounded,
        "unsupported": round.statistics.joint_unsupported,
        "cost_unsupported_detail": round.statistics.joint_cost_unsupported_detail,
        "static_fallbacks": round.statistics.static_fallbacks,
        "actual_batch_members": batch.size(),
        "actual_batch_tokens": batch.total_tokens(),
        "actual_selected_work": round.selected.iter().map(|work| json!({
            "kind": format!("{:?}", work.kind),
            "requested_tokens": work.requested.as_range().len(),
            "context_tokens": work.context_tokens,
        })).collect::<Vec<_>>(),
    })
}

#[test]
#[ignore = "CPU diagnostic: actual bounded dispatch at fixed N/ranks/K and original 10ms budget; no speed or completion gate"]
fn profile_bounded_dispatch_length_scaling_with_production_budget() {
    // All fixed metadata other than lengths/capacity coverage stays unchanged.
    // The smallest prompt already exceeds all K-wave prefill progress; cap64
    // likewise exceeds possible decode progress. Larger tails are not executed.
    let reports = [(2048, 64), (32_768, 1024), (524_288, 16_384)]
        .into_iter()
        .map(|(prompt, cap)| run_length_case(prompt, cap))
        .collect::<Vec<_>>();
    println!(
        "BOUNDED_LENGTH_PROFILE {}",
        json!({
            "pid": std::process::id(),
            "cohort_members": COHORT_MEMBERS,
            "residents": PROFILE.residents,
            "initial_prefillers": PROFILE.initial_prefillers,
            "waiting": COHORT_MEMBERS - PROFILE.residents,
            "ranks": 32, "lookahead_rounds": LOOKAHEAD,
            "candidate_policy": "Static, scalar half, scalar quarter, DecodeOnly for the same mixed first-wave geometry",
            "max_tokens": PROFILE.max_tokens, "prefill_step": PROFILE.prefill_step,
            "cases": reports,
            "scope": "actual scheduler bounded dispatch with existing CheapKnownCosts and synthetic typed owner capacity; no Engine/device execution, learned support or latency SLO evidence",
            "interpretation": "All 10ms/4096 failures are retained. Only entered_cost_evaluator=true AND all_ranks_scored=true establishes a completed bounded comparison for that length. Early fallback is not scaling success. Selection timing includes ordinary scheduler work but excludes fixture setup, health merge, JSON construction and printing; no speed/ratio assertion. This does not measure allocator peak or prove absence of every allocation."
        })
    );
}
