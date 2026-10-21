use super::*;
use crate::cost_prediction::{
    CostExecution, CostPrediction, CostQuery, CostSnapshot, CostUnknownReason,
};
use ferrum_execution::model_executor::PrefillShapePolicy;
use ferrum_types::{
    CostFallback, ModelId, RequestIngressAnchor, RequestIngressBoundary, SamplingParams,
    SchedulerSloConfig, UnknownCostPolicy,
};

struct NoCost;
impl CostSnapshot for NoCost {
    fn revision(&self) -> u64 {
        1
    }
    fn query_limits(&self) -> ferrum_types::CostQueryLimits {
        Default::default()
    }
    fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
        CostPrediction::Unknown(CostUnknownReason::NoCoverage)
    }
}

fn fixture() -> (
    ContinuousBatchScheduler,
    TimePlanningRound,
    RequestId,
    Instant,
) {
    let ingress = RequestIngressAnchor::capture(RequestIngressBoundary::SchedulerApi);
    let request = InferenceRequest {
        id: RequestId::new(),
        prompt: "test".into(),
        model_id: ModelId::new("test-model"),
        sampling_params: SamplingParams::default(),
        stream: true,
        priority: Priority::Normal,
        client_id: None,
        session_id: None,
        created_at: chrono::Utc::now(),
        ingress: Some(ingress),
        api_request: None,
        evidence_request: Default::default(),
        metadata: Default::default(),
    };
    let id = request.id.clone();
    let mut config = SchedulerSloConfig {
        ttft_ms: 1000.0,
        tpot_ms: 100.0,
        itl_ms: 100.0,
        query: Default::default(),
        planning: Default::default(),
        joint: Default::default(),
        strategy: SloStrategy::Joint,
        unknown_stopping: Default::default(),
        unknown_cost: Default::default(),
    };
    config.unknown_cost = UnknownCostPolicy::Reject;
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
        slo: Some(config),
        ..Default::default()
    });
    scheduler.submit_request(request).unwrap();
    let mut round = TimePlanningRound::new(
        Arc::new(NoCost),
        config,
        PrefillShapePolicy::Identity,
        CostExecution::SingleInvocation,
    );
    freeze(
        &scheduler,
        &mut round,
        Verdict::AllKnownInfeasible,
        ingress.instant(),
    );
    (scheduler, round, id, ingress.instant())
}

fn freeze(
    scheduler: &ContinuousBatchScheduler,
    round: &mut TimePlanningRound,
    verdict: Verdict,
    now: Instant,
) {
    round.admission_gate = Some(FrozenAdmissionGate {
        members: scheduler
            .waiting_queue
            .read()
            .iter()
            .map(|request| Member::capture(request).unwrap())
            .collect(),
        verdict,
        created_at: now,
        apply_before: now + Duration::from_secs(60),
    });
}

#[test]
fn known_infeasible_is_rejected_before_an_executor_candidate_exists() {
    let (scheduler, mut round, id, now) = fixture();
    let mut scan = scheduler.begin_token_admission(8, None).unwrap();
    assert!(scan
        .next_at(&mut |_| {}, Some(&mut round), now)
        .unwrap()
        .is_none());
    assert_eq!(round.statistics.rejected_candidates, 1);
    assert_eq!(scheduler.waiting_count(), 0);
    assert!(scheduler.admission_failed_requests.read().contains_key(&id));
}

#[test]
fn known_bad_wait_uses_original_ingress_and_then_rejects() {
    let (scheduler, mut round, _, now) = fixture();
    round.config.joint.known_infeasible_admission =
        KnownInfeasibleAdmissionPolicy::Wait { max_wait_ms: 100 };
    let waiting = scheduler.waiting_queue.read();
    let request = waiting.iter().next().unwrap();
    assert!(matches!(
        scheduler.joint_admission_time_decision(
            request,
            &mut round,
            now + Duration::from_millis(50)
        ),
        time_policy::TimeDecision::Hold
    ));
    assert_eq!(round.retry_at, Some(now + Duration::from_millis(100)));
    assert!(matches!(
        scheduler.joint_admission_time_decision(
            request,
            &mut round,
            now + Duration::from_millis(101)
        ),
        time_policy::TimeDecision::Reject(_)
    ));
}

#[test]
fn changed_cohort_or_capacity_invalidates_frozen_verdict() {
    let (scheduler, mut round, id, now) = fixture();
    let replacement = scheduler
        .waiting_queue
        .read()
        .iter()
        .next()
        .unwrap()
        .inner
        .request
        .clone();
    assert!(scheduler.cancel_request(id).unwrap());
    scheduler.submit_request(replacement).unwrap();
    scheduler.revalidate_joint_admission_gate(&scheduler.waiting_queue.read(), &mut round);
    assert!(
        round.admission_gate.is_none(),
        "same-ID replacement is a different incarnation"
    );
    freeze(&scheduler, &mut round, Verdict::StaticFeasible, now);
    round.set_capacity(None);
    assert!(round.admission_gate.is_none());
}

#[test]
fn unknown_action_uses_declared_fallback_and_expired_gate_is_not_feasible() {
    let (scheduler, mut round, _, now) = fixture();
    round.config.unknown_cost = UnknownCostPolicy::StaticFallback;
    freeze(&scheduler, &mut round, Verdict::RequiresAction, now);
    let waiting = scheduler.waiting_queue.read();
    let request = waiting.iter().next().unwrap();
    assert!(matches!(
        scheduler.joint_admission_time_decision(request, &mut round, now),
        time_policy::TimeDecision::Allow(_)
    ));
    round.config.unknown_cost = UnknownCostPolicy::Wait {
        max_wait_ms: 100,
        on_timeout: CostFallback::Reject,
    };
    round.admission_gate.as_mut().unwrap().verdict = Verdict::StaticFeasible;
    round.admission_gate.as_mut().unwrap().apply_before = now;
    assert!(matches!(
        scheduler.joint_admission_time_decision(request, &mut round, now),
        time_policy::TimeDecision::Hold
    ));
}

#[test]
fn already_admitted_zero_offset_and_recompute_are_never_stranded() {
    let (scheduler, mut round, _, now) = fixture();
    let mut waiting = scheduler.waiting_queue.write();
    let ticket = waiting
        .iter()
        .next()
        .unwrap()
        .waiting_admission_ticket
        .unwrap();
    let request = waiting.request_mut(ticket).unwrap();
    for recompute in [false, true] {
        request.logical_work_frontier.begin_prefill(recompute);
        assert_eq!(request.prefill_chunk_offset, 0);
        assert!(matches!(
            scheduler.joint_admission_time_decision(request, &mut round, now),
            time_policy::TimeDecision::Allow(_)
        ));
    }
    assert_eq!(round.statistics.rejected_candidates, 0);
}
