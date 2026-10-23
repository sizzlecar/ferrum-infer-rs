//! Admission-aware phase boundaries through the real simulation loop.
use super::*;
use ferrum_execution::{execution_attempt::AttemptPrefixRestore, token_execution::ExecutionRange};
use std::sync::Mutex;

struct Costs(Instant);
impl CostSnapshot for Costs {
    fn revision(&self) -> u64 {
        1
    }
    fn query_limits(&self) -> CostQueryLimits {
        Default::default()
    }
    fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
        CostPrediction::Known {
            central_service_time: Duration::from_millis(1),
            upper_service_time: Duration::from_millis(1),
            risk: CostRisk::EngineeringMargin,
            support: CostSupport::ExactObservedWork,
            samples: 1,
            observed_at: self.0,
            expires_at: self.0 + Duration::from_secs(60),
        }
    }
}

fn requests() -> Vec<SimulatedRequest> {
    let waiting = SimulatedRequest {
        id: RequestId::new(),
        prompt_tokens: 3,
        prefill_tokens: 0,
        output_tokens: 0,
        output_cap: 4,
        scenario_end_tokens: 4,
        age: Duration::ZERO,
        central_clock: RequestClock::default(),
        risk_clock: RequestClock::default(),
    };
    let clock = RequestClock {
        first_output: Some(Duration::ZERO),
        last_output: Some(Duration::ZERO),
        ..Default::default()
    };
    let resident = SimulatedRequest {
        id: RequestId::new(),
        prompt_tokens: 1,
        prefill_tokens: 1,
        output_tokens: 1,
        output_cap: 3,
        scenario_end_tokens: 3,
        age: Duration::ZERO,
        central_clock: clock,
        risk_clock: clock,
    };
    vec![waiting, resident]
}

struct Adapter {
    slots: AdmissionSlotPlan,
    stall_target: bool,
    premature: bool,
    // Observation only; phase/action state belongs exclusively to the core.
    waves: Mutex<Vec<(bool, usize, usize)>>,
}
impl Adapter {
    fn new(max_active: usize) -> Self {
        Self {
            slots: AdmissionSlotPlan::new(
                &[
                    InitialAdmission::Waiting { order: 0 },
                    InitialAdmission::Resident,
                ],
                max_active,
                2,
            )
            .unwrap(),
            stall_target: false,
            premature: false,
            waves: Mutex::new(Vec::new()),
        }
    }
    fn wave(
        &self,
        phased: bool,
        requests: &[SimulatedRequest],
        plan: &mut PlanBuilder,
    ) -> Result<(), Unsupported> {
        self.waves.lock().unwrap().push((
            phased,
            requests[0].prefill_tokens,
            requests[1].output_tokens,
        ));
        let active = self.slots.active_mask(requests)?;
        // The original split order is retained. A decode scheduled to retire
        // later in this wave cannot grant its slot to this prefill group.
        for kind in [AttemptWorkKind::Prefill, AttemptWorkKind::Decode] {
            for (index, request) in requests.iter().enumerate() {
                if request.finished()
                    || (request.output_tokens == 0) != (kind == AttemptWorkKind::Prefill)
                    || (!active[index] && !(self.premature && index == 0))
                    || (self.stall_target && index == 0)
                {
                    continue;
                }
                let (start, tokens, context) = if kind == AttemptWorkKind::Prefill {
                    (
                        request.prefill_tokens,
                        (request.prompt_tokens - request.prefill_tokens).min(if phased {
                            1
                        } else {
                            2
                        }),
                        request.prompt_tokens,
                    )
                } else {
                    let context = request.prompt_tokens + request.output_tokens;
                    (context - 1, 1, context)
                };
                plan.push_group(&[GroupMember {
                    request: index,
                    work: CostWork {
                        kind,
                        requested: ExecutionRange::new(start, tokens).unwrap(),
                        context_tokens: context,
                        prefix_restore: AttemptPrefixRestore::NoneObserved,
                        internally_reused_tokens: None,
                    },
                }])?;
            }
        }
        Ok(())
    }
}
impl TrajectoryAdapter<()> for Adapter {
    fn admission_slots(&self) -> Option<&AdmissionSlotPlan> {
        Some(&self.slots)
    }
    fn continuation_phase(
        &self,
        _: &(),
        _: &[SimulatedRequest],
        phase: &mut PhaseBuilder<'_>,
    ) -> Result<(), Unsupported> {
        phase.deferred_prefill_target(0)
    }
    fn plan(
        &self,
        action: Option<&()>,
        requests: &[SimulatedRequest],
        plan: &mut PlanBuilder,
    ) -> Result<(), Unsupported> {
        self.wave(action.is_some(), requests, plan)
    }
    fn plan_phase(
        &self,
        _: &(),
        _: &ContinuationPhase,
        requests: &[SimulatedRequest],
        plan: &mut PlanBuilder,
    ) -> Result<(), Unsupported> {
        self.wave(true, requests, plan)
    }
}

fn evaluate(
    adapter: &Adapter,
    requests: &[SimulatedRequest],
    max_queries: usize,
) -> Result<Comparison, Unsupported> {
    let costs = Costs(Instant::now());
    let mut config: SchedulerSloConfig = "ttft:1000,tpot:100,itl:100".parse().unwrap();
    config.planning.max_elapsed_us = 1_000_000;
    config.planning.max_queries = max_queries;
    SloPolicy::evaluate(EvaluationInput {
        snapshot: &costs,
        now: costs.0,
        config: &config,
        execution: CostExecution::SequentialPrefillThenDecode,
        requests,
        resources: ResourceSupport {
            waiting_requests: 1,
            kv_reserved_through_scenario_end: true,
        },
        adapter,
        static_action: &(),
        alternatives: &[],
        horizon: EvaluationHorizon::ScenarioCompletion,
    })
}

#[test]
fn deferred_phase_waits_for_original_slot_and_returns_to_common_continuation() {
    let adapter = Adapter::new(1);
    let result = evaluate(&adapter, &requests(), 100).unwrap();
    assert!(result
        .static_result
        .complete()
        .unwrap()
        .requests
        .iter()
        .all(|request| request.finished() && request.central_clock.completed.is_some()));
    let waves = adapter.waves.lock().unwrap();
    assert_eq!(&waves[..3], &[(true, 0, 1), (true, 0, 2), (true, 0, 3)]);
    assert_eq!(&waves[3..5], &[(true, 1, 3), (true, 2, 3)]);
    assert!(waves[5..]
        .iter()
        .all(|(phased, prefill, _)| !phased && *prefill == 3));
    assert_eq!(
        evaluate(&Adapter::new(1), &requests(), 1).unwrap_err(),
        Unsupported::QueryBudget
    );
}

#[test]
fn deferred_phase_cannot_trade_eligible_target_progress_for_other_decode_work() {
    let mut adapter = Adapter::new(2);
    adapter.stall_target = true;
    assert_eq!(
        evaluate(&adapter, &requests(), 100).unwrap_err(),
        Unsupported::NoProgress
    );
}

#[test]
fn deferred_phase_cannot_admit_waiter_inside_retiring_split_wave() {
    let mut adapter = Adapter::new(1);
    adapter.premature = true;
    let mut requests = requests();
    requests[1].output_tokens = 2; // Its next decode completes, but not before planning.
    assert_eq!(
        evaluate(&adapter, &requests, 100).unwrap_err(),
        Unsupported::WaitingResources
    );
}

#[test]
fn deferred_phase_completed_target_does_not_make_remaining_inactive_target_eligible() {
    let mut requests = requests();
    let mut second = requests[0].clone();
    second.id = RequestId::new();
    requests.push(second);
    let slots = AdmissionSlotPlan::new(
        &[
            InitialAdmission::Waiting { order: 0 },
            InitialAdmission::Resident,
            InitialAdmission::Waiting { order: 1 },
        ],
        1,
        3,
    )
    .unwrap();
    let mut builder = PhaseBuilder::new(&requests, 3, 1024);
    builder.deferred_prefill_target(0).unwrap();
    builder.deferred_prefill_target(2).unwrap();
    let mut phase = PhaseState::new(builder.finish().unwrap(), &requests, Some(&slots)).unwrap();
    let active = slots.active_mask(&requests).unwrap();
    phase.begin_wave(&requests, Some(&active)).unwrap();
    assert_eq!(phase.advance(&requests), Err(Unsupported::NoProgress));
    requests[1].output_tokens = 3;
    phase.advance(&requests).unwrap();
    let active = slots.active_mask(&requests).unwrap();
    phase.begin_wave(&requests, Some(&active)).unwrap();
    requests[0].prefill_tokens = 3;
    requests[0].output_tokens = 1;
    phase.advance(&requests).unwrap();
    let active = slots.active_mask(&requests).unwrap();
    assert!(active[0] && !active[2]);
    phase.begin_wave(&requests, Some(&active)).unwrap();
    requests[0].output_tokens += 1;
    assert!(
        !phase.advance(&requests).unwrap(),
        "completed prefill target may decode while the remaining target waits"
    );
}

#[test]
fn deferred_phase_declarations_require_original_waiters_and_cannot_mix_modes() {
    let requests = requests();
    let mut builder = PhaseBuilder::new(&requests, 2, 1024);
    builder.prefill_target(0).unwrap();
    assert_eq!(
        builder.deferred_prefill_target(0),
        Err(Unsupported::InvalidPlan)
    );
    let mut builder = PhaseBuilder::new(&requests, 2, 1024);
    builder.deferred_prefill_target(0).unwrap();
    assert_eq!(builder.prefill_target(0), Err(Unsupported::InvalidPlan));
    assert!(matches!(
        PhaseState::new(builder.finish().unwrap(), &requests, None),
        Err(Unsupported::WaitingResources)
    ));
    let mut builder = PhaseBuilder::new(&requests, 2, 1024);
    builder.deferred_prefill_target(0).unwrap();
    let residents = AdmissionSlotPlan::new(&[InitialAdmission::Resident; 2], 2, 2).unwrap();
    assert!(matches!(
        PhaseState::new(builder.finish().unwrap(), &requests, Some(&residents)),
        Err(Unsupported::InvalidPlan)
    ));
}
