use super::*;
use ferrum_execution::{execution_attempt::AttemptPrefixRestore, token_execution::ExecutionRange};

struct Snapshot(Instant);
impl CostSnapshot for Snapshot {
    fn revision(&self) -> u64 {
        1
    }
    fn query_limits(&self) -> CostQueryLimits {
        CostQueryLimits::default()
    }
    fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
        CostPrediction::Known {
            central_service_time: Duration::from_millis(10),
            upper_service_time: Duration::from_millis(10),
            risk: CostRisk::EngineeringMargin,
            support: CostSupport::ExactObservedWork,
            samples: 2,
            observed_at: self.0,
            expires_at: self.0 + Duration::from_secs(1),
        }
    }
    fn cost_scenarios(&self, _: &CostQuery<'_>, _: Instant, _: usize) -> CostScenarioPrediction {
        CostScenarioPrediction::Known {
            central_service_time: Duration::from_millis(10),
            scenarios: (0..2)
                .map(|rank| CostScenario {
                    rank,
                    weight: 0.5,
                    service_time: Duration::from_millis(10),
                })
                .collect(),
            dependence: CostScenarioDependence::SharedResidualRank,
            support: CostSupport::ExactObservedWork,
            samples: 2,
            observed_at: self.0,
            expires_at: self.0 + Duration::from_secs(1),
        }
    }
}

struct Adapter {
    slots: AdmissionSlotPlan,
    split: bool,
    over_admit: bool,
}
impl TrajectoryAdapter<()> for Adapter {
    fn admission_slots(&self) -> Option<&AdmissionSlotPlan> {
        Some(&self.slots)
    }
    fn plan(
        &self,
        _: Option<&()>,
        requests: &[SimulatedRequest],
        plan: &mut PlanBuilder,
    ) -> Result<(), Unsupported> {
        let active = self.slots.active_mask(requests)?;
        let members = requests
            .iter()
            .enumerate()
            .filter(|(index, request)| !request.finished() && (self.over_admit || active[*index]))
            .map(|(index, request)| {
                let prefill = request.output_tokens == 0;
                let context = if prefill {
                    request.prompt_tokens
                } else {
                    request.prompt_tokens + request.output_tokens
                };
                GroupMember {
                    request: index,
                    work: CostWork {
                        kind: if prefill {
                            AttemptWorkKind::Prefill
                        } else {
                            AttemptWorkKind::Decode
                        },
                        requested: ExecutionRange::new(
                            if prefill {
                                request.prefill_tokens
                            } else {
                                context - 1
                            },
                            1,
                        )
                        .unwrap(),
                        context_tokens: context,
                        prefix_restore: AttemptPrefixRestore::NoneObserved,
                        internally_reused_tokens: None,
                    },
                }
            })
            .collect::<Vec<_>>();
        if self.split {
            for kind in [AttemptWorkKind::Prefill, AttemptWorkKind::Decode] {
                let group = members
                    .iter()
                    .copied()
                    .filter(|member| member.work.kind == kind)
                    .collect::<Vec<_>>();
                if !group.is_empty() {
                    plan.push_group(&group)?;
                }
            }
            Ok(())
        } else {
            plan.push_group(&members)
        }
    }
}

fn waiting(prompt: usize, cap: usize) -> SimulatedRequest {
    SimulatedRequest {
        id: RequestId::new(),
        prompt_tokens: prompt,
        prefill_tokens: 0,
        output_tokens: 0,
        output_cap: cap,
        scenario_end_tokens: cap,
        age: Duration::ZERO,
        central_clock: RequestClock::default(),
        risk_clock: RequestClock::default(),
    }
}
fn decoder(end: usize) -> SimulatedRequest {
    let mut request = waiting(1, end);
    request.prefill_tokens = 1;
    request.output_tokens = 1;
    request.central_clock.first_output = Some(Duration::ZERO);
    request.central_clock.last_output = Some(Duration::ZERO);
    request.risk_clock = request.central_clock;
    request
}
fn config() -> SchedulerSloConfig {
    let mut config: SchedulerSloConfig = "ttft:100,tpot:100,itl:100".parse().unwrap();
    config.planning.max_elapsed_us = 1_000_000;
    config
}
fn input<'a>(
    requests: &'a [SimulatedRequest],
    adapter: &'a Adapter,
    snapshot: &'a Snapshot,
    config: &'a SchedulerSloConfig,
) -> EvaluationInput<'a, (), Adapter> {
    EvaluationInput {
        snapshot,
        now: snapshot.0,
        config,
        execution: if adapter.split {
            CostExecution::SequentialPrefillThenDecode
        } else {
            CostExecution::SingleInvocation
        },
        requests,
        resources: ResourceSupport {
            waiting_requests: adapter.slots.waiting_requests(),
            kv_reserved_through_scenario_end: true,
        },
        adapter,
        static_action: &(),
        alternatives: &[],
        static_window_rounds: 100,
    }
}

#[test]
fn natural_endpoints_release_slots_and_waiting_ttft_stays_in_the_same_cohort() {
    let adapter = Adapter {
        slots: AdmissionSlotPlan::new(
            &[
                InitialAdmission::Resident,
                InitialAdmission::Waiting { order: 0 },
            ],
            1,
            2,
        )
        .unwrap(),
        split: false,
        over_admit: false,
    };
    for (endpoint, expected_ms) in [(8, 80), (25, 250)] {
        let requests = [decoder(endpoint), waiting(1, 1)];
        let result = SloPolicy::evaluate(input(
            &requests,
            &adapter,
            &Snapshot(Instant::now()),
            &config(),
        ))
        .unwrap();
        assert_eq!(result.static_result.requests.len(), 2);
        assert_eq!(
            result.static_result.requests[1].central_clock.first_output,
            Some(Duration::from_millis(expected_ms))
        );
        assert_eq!(
            result.static_result.requests[0].central_clock.completed,
            Some(Duration::from_millis(expected_ms - 10))
        );
    }
}

#[test]
fn split_group_completion_releases_execution_slots_only_for_the_next_wave() {
    let adapter = Adapter {
        slots: AdmissionSlotPlan::new(
            &[
                InitialAdmission::Resident,
                InitialAdmission::Resident,
                InitialAdmission::Waiting { order: 0 },
            ],
            2,
            3,
        )
        .unwrap(),
        split: true,
        over_admit: false,
    };
    let requests = [waiting(1, 1), decoder(2), waiting(1, 1)];
    let result = SloPolicy::evaluate(input(
        &requests,
        &adapter,
        &Snapshot(Instant::now()),
        &config(),
    ))
    .unwrap();
    let clocks = &result.static_result.requests;
    assert_eq!(
        clocks[0].central_clock.completed,
        Some(Duration::from_millis(10))
    );
    assert_eq!(
        clocks[1].central_clock.completed,
        Some(Duration::from_millis(20))
    );
    assert_eq!(
        clocks[2].central_clock.first_output,
        Some(Duration::from_millis(30))
    );
    assert_eq!(result.static_result.queries, 3);
}

#[test]
fn incomplete_prefill_and_admitted_zero_work_keep_their_slots() {
    let adapter = Adapter {
        slots: AdmissionSlotPlan::new(
            &[
                InitialAdmission::Resident,
                InitialAdmission::Waiting { order: 0 },
            ],
            1,
            2,
        )
        .unwrap(),
        split: false,
        over_admit: false,
    };
    let requests = [waiting(3, 1), waiting(1, 1)];
    let result = SloPolicy::evaluate(input(
        &requests,
        &adapter,
        &Snapshot(Instant::now()),
        &config(),
    ))
    .unwrap();
    assert_eq!(
        result.static_result.requests[1].central_clock.first_output,
        Some(Duration::from_millis(40))
    );
    let slots = AdmissionSlotPlan::new(
        &[
            InitialAdmission::Resident,
            InitialAdmission::Waiting { order: 1 },
            InitialAdmission::Waiting { order: 0 },
        ],
        2,
        3,
    )
    .unwrap();
    let requests = [decoder(2), waiting(1, 1), waiting(1, 1)];
    assert_eq!(slots.active_mask(&requests).unwrap(), [true, false, true]);
    // Replanning without work does not skip the already selected earliest waiter.
    assert_eq!(slots.active_mask(&requests).unwrap(), [true, false, true]);
}

#[test]
fn unknown_stopping_rank_never_fills_cap_or_credits_waiting_release() {
    let adapter = Adapter {
        slots: AdmissionSlotPlan::new(
            &[
                InitialAdmission::Resident,
                InitialAdmission::Waiting { order: 0 },
            ],
            1,
            2,
        )
        .unwrap(),
        split: false,
        over_admit: false,
    };
    let requests = [decoder(25), waiting(1, 1)];
    let mut observed = Vec::new();
    let summary = SloPolicy::evaluate_scenarios_with_requests(
        input(&requests, &adapter, &Snapshot(Instant::now()), &config()),
        |rank, requests| {
            if rank.index == 1 {
                return Err(Unsupported::Cost(CostUnknownReason::NoCoverage));
            }
            requests[0].scenario_end_tokens = 8;
            Ok(())
        },
        |rank, comparison| {
            observed.push((
                rank.weight,
                comparison.static_result.requests[1]
                    .central_clock
                    .first_output,
            ))
        },
    )
    .unwrap();
    assert_eq!(observed, [(0.5, Some(Duration::from_millis(80)))]);
    assert_eq!(summary.resolved_mass, 0.5);
    assert_eq!(summary.unresolved_mass, 0.5);
}

#[test]
fn missing_queue_members_duplicate_order_and_over_admission_are_rejected() {
    assert!(AdmissionSlotPlan::new(&[InitialAdmission::Waiting { order: 1 }], 1, 1).is_err());
    assert!(AdmissionSlotPlan::new(
        &[
            InitialAdmission::Waiting { order: 0 },
            InitialAdmission::Waiting { order: 0 }
        ],
        1,
        2
    )
    .is_err());
    assert!(AdmissionSlotPlan::new(
        &[InitialAdmission::Resident, InitialAdmission::Resident],
        1,
        2
    )
    .is_err());
    let adapter = Adapter {
        slots: AdmissionSlotPlan::new(
            &[
                InitialAdmission::Resident,
                InitialAdmission::Waiting { order: 0 },
            ],
            1,
            2,
        )
        .unwrap(),
        split: false,
        over_admit: true,
    };
    let requests = [waiting(1, 1), waiting(1, 1)];
    assert!(matches!(
        SloPolicy::evaluate(input(
            &requests,
            &adapter,
            &Snapshot(Instant::now()),
            &config()
        )),
        Err(Unsupported::WaitingResources)
    ));
    let snapshot = Snapshot(Instant::now());
    let config = config();
    let mut omitted = input(&requests[..1], &adapter, &snapshot, &config);
    omitted.resources.waiting_requests = 0;
    assert!(matches!(
        SloPolicy::evaluate(omitted),
        Err(Unsupported::InvalidState)
    ));
}
