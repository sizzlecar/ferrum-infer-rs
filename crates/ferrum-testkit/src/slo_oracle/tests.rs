use super::*;
use std::num::NonZeroUsize;

fn nz(value: u64) -> NonZeroU64 {
    NonZeroU64::new(value).unwrap()
}
fn budgets(ttft: u64, tpot: u64, itl: u64) -> Budgets {
    Budgets {
        ttft: nz(ttft),
        tpot: nz(tpot),
        itl: nz(itl),
    }
}
fn request(id: u16, prompt: u16, outputs: u16, budgets: Budgets) -> Request {
    Request {
        owner: Owner(id),
        ingress_at: 0,
        prompt_tokens: prompt,
        maximum_output_tokens: outputs,
        budgets,
        milestones: Vec::new(),
    }
}
fn progress(prompt: u16, times: &[u64]) -> Progress {
    Progress {
        net_prompt_tokens: prompt,
        commit_times: times.to_vec(),
    }
}
fn decode(id: u16, previous_commits: u16) -> Work {
    Work::Decode {
        owner: Owner(id),
        previous_commits,
    }
}
fn prefill(id: u16, offset: u16, count: u16) -> Work {
    Work::Prefill {
        owner: Owner(id),
        offset,
        count,
    }
}
fn wave(work: &[Work]) -> Wave {
    Wave(work.to_vec())
}
fn costs(entries: &[(Wave, u64)]) -> CostTable {
    let mut table = CostTable::default();
    for (wave, duration) in entries {
        table.insert(wave.clone(), nz(*duration)).unwrap();
    }
    table
}
fn limits() -> EnumerationLimits {
    EnumerationLimits {
        maximum_requests: NonZeroUsize::new(4).unwrap(),
        maximum_waves: NonZeroUsize::new(16).unwrap(),
        maximum_transitions: nz(10_000),
    }
}

#[test]
fn each_owner_alone_can_finish_but_no_common_model_sequence_exists() {
    let reqs = vec![
        request(0, 0, 1, budgets(3, 3, 3)),
        request(1, 0, 1, budgets(3, 3, 3)),
    ];
    let a = wave(&[decode(0, 0)]);
    let b = wave(&[decode(1, 0)]);
    let problem = Problem {
        requests: reqs.clone(),
        initial: State {
            at: 0,
            progress: vec![progress(0, &[]), progress(0, &[])],
        },
        maximum_rows: 1,
        costs: costs(&[(a.clone(), 2), (b.clone(), 2)]),
    };
    for (index, work) in [a.clone(), b.clone()].into_iter().enumerate() {
        let solo = Problem {
            requests: vec![reqs[index].clone()],
            initial: State {
                at: 0,
                progress: vec![progress(0, &[])],
            },
            maximum_rows: 1,
            costs: costs(&[(work.clone(), 2)]),
        };
        assert_eq!(check_sequence(&solo, &[work]).unwrap().final_state.at, 2);
    }
    // Every positive-cost legal sequence was considered; the second owner
    // cannot commit before 4. No inference about other action/cost domains.
    assert!(matches!(
        enumerate(&problem, limits()).unwrap(),
        SearchResult::NoModelWitness { .. }
    ));
    assert!(matches!(
        check_sequence(&problem, &[a, b]),
        Err(Error::ModelDeadline {
            at: 4,
            due_at: 3,
            ..
        })
    ));
}

#[test]
fn early_output_cannot_bank_slack_for_a_long_adjacent_commit_gap() {
    let work = wave(&[decode(0, 1)]);
    let mut problem = Problem {
        requests: vec![request(0, 0, 2, budgets(100, 110, 20))],
        initial: State {
            at: 10,
            progress: vec![progress(0, &[10])],
        },
        maximum_rows: 1,
        costs: costs(&[(work.clone(), 110)]),
    };
    // At 120 the cumulative first+TPOT deadline is satisfied exactly. The
    // 110-tick adjacent gap still violates ITL=20. These are commit timestamps,
    // not a claim that either commit produced a visible SSE text event.
    assert_eq!(10 + 110, 120);
    assert_eq!(
        check_sequence(&problem, &[work.clone()]),
        Err(Error::ModelDeadline {
            owner: Owner(0),
            boundary: Boundary::AdjacentCommit,
            due_at: 30,
            at: 120
        })
    );
    problem.costs = costs(&[(work.clone(), 20)]);
    assert_eq!(
        check_sequence(&problem, &[work]).unwrap().final_state.at,
        30
    );
}

#[test]
fn adjacent_itl_can_pass_while_prefix_tpot_independently_fails() {
    let work = wave(&[decode(0, 1)]);
    let mut problem = Problem {
        requests: vec![request(0, 0, 2, budgets(100, 5, 100))],
        initial: State {
            at: 10,
            progress: vec![progress(0, &[10])],
        },
        maximum_rows: 1,
        costs: costs(&[(work.clone(), 6)]),
    };
    // End=16 is safely within adjacent deadline=110, but crosses first+TPOT=15.
    assert_eq!(
        check_sequence(&problem, &[work.clone()]),
        Err(Error::ModelDeadline {
            owner: Owner(0),
            boundary: Boundary::PrefixTpot,
            due_at: 15,
            at: 16,
        })
    );
    assert!(matches!(
        enumerate(&problem, limits()).unwrap(),
        SearchResult::NoModelWitness { .. }
    ));
    problem.costs = costs(&[(work.clone(), 5)]);
    assert_eq!(
        check_sequence(&problem, &[work]).unwrap().final_state.at,
        15
    );
    assert!(matches!(
        enumerate(&problem, limits()).unwrap(),
        SearchResult::Feasible { .. }
    ));
}

#[test]
fn missing_singleton_cost_does_not_hide_a_known_complete_batch() {
    let unknown_singleton = wave(&[decode(1, 0)]);
    let both = wave(&[decode(0, 0), decode(1, 0)]);
    let problem = Problem {
        requests: vec![
            request(0, 0, 1, budgets(2, 2, 2)),
            request(1, 0, 1, budgets(2, 2, 2)),
        ],
        initial: State {
            at: 0,
            progress: vec![progress(0, &[]), progress(0, &[])],
        },
        maximum_rows: 2,
        costs: costs(&[(both.clone(), 2)]),
    };
    assert_eq!(
        check_sequence(&problem, &[unknown_singleton.clone()]),
        Err(Error::MissingCost(unknown_singleton))
    );
    let SearchResult::Feasible { sequence, .. } = enumerate(&problem, limits()).unwrap() else {
        panic!("an unrelated unknown must not invalidate a known complete witness")
    };
    assert_eq!(sequence.steps.len(), 1);
    assert_eq!(sequence.steps[0].wave, both.clone());
    assert_eq!(sequence.final_state.at, 2);
    assert_eq!(check_sequence(&problem, &[both]).unwrap(), sequence);
}

#[test]
fn smaller_batch_is_slower_and_cannot_prune_the_only_joint_witness() {
    let a = wave(&[decode(0, 1)]);
    let b = wave(&[decode(1, 1)]);
    let both = wave(&[decode(0, 1), decode(1, 1)]);
    let problem = Problem {
        requests: vec![
            request(0, 0, 2, budgets(2, 2, 2)),
            request(1, 0, 2, budgets(2, 2, 2)),
        ],
        initial: State {
            at: 0,
            progress: vec![progress(0, &[0]), progress(0, &[0])],
        },
        maximum_rows: 2,
        costs: costs(&[(a.clone(), 3), (b, 3), (both.clone(), 2)]),
    };
    assert!(matches!(
        check_sequence(&problem, &[a]),
        Err(Error::ModelDeadline { .. })
    ));
    let SearchResult::Feasible { sequence, .. } = enumerate(&problem, limits()).unwrap() else {
        panic!("the faster complete batch must remain in the finite domain")
    };
    assert_eq!(sequence.steps.len(), 1);
    assert_eq!(sequence.steps[0].wave, both);
    assert_eq!(sequence.final_state.at, 2);
}

#[test]
fn peer_service_cannot_reset_prefill_milestones_or_hide_unselected_owner() {
    let mut prompt = request(0, 2, 1, budgets(8, 8, 8));
    prompt.milestones = vec![
        Milestone {
            net_prompt_tokens: 1,
            due_at: 2,
        },
        Milestone {
            net_prompt_tokens: 2,
            due_at: 5,
        },
    ];
    let p0 = wave(&[prefill(0, 0, 1)]);
    let p1 = wave(&[prefill(0, 1, 1)]);
    let d1 = wave(&[decode(1, 1)]);
    let d2 = wave(&[decode(1, 2)]);
    let d3 = wave(&[decode(1, 3)]);
    let problem = Problem {
        requests: vec![prompt, request(1, 0, 4, budgets(8, 3, 3))],
        initial: State {
            at: 0,
            progress: vec![progress(0, &[]), progress(0, &[0])],
        },
        maximum_rows: 1,
        costs: costs(&[
            (p0.clone(), 1),
            (p1.clone(), 1),
            (wave(&[prefill(0, 0, 2)]), 2),
            (d1.clone(), 1),
            (d2.clone(), 1),
            (d3.clone(), 1),
        ]),
    };
    assert_eq!(
        check_sequence(&problem, &[d1.clone(), d2.clone(), d3.clone()]),
        Err(Error::ModelDeadline {
            owner: Owner(0),
            boundary: Boundary::PrefillMilestone,
            due_at: 2,
            at: 3
        })
    );
    let valid = check_sequence(&problem, &[p0, d1, p1, d2, d3]).unwrap();
    assert_eq!(valid.final_state.progress[0].commit_times, vec![3]);
    assert!(matches!(
        enumerate(&problem, limits()).unwrap(),
        SearchResult::Feasible { .. }
    ));
    assert_eq!(problem.requests[0].milestones[0].due_at, 2);
}

#[test]
fn absent_cost_and_search_resource_limits_are_unknown_not_zero_or_impossible() {
    let mut problem = Problem {
        requests: vec![request(0, 0, 2, budgets(10, 10, 10))],
        initial: State {
            at: 0,
            progress: vec![progress(0, &[])],
        },
        maximum_rows: 1,
        costs: costs(&[(wave(&[decode(0, 0)]), 1)]),
    };
    assert!(matches!(
        enumerate(&problem, limits()).unwrap(),
        SearchResult::Inconclusive {
            cause: Uncertainty::MissingCosts,
            ..
        }
    ));
    problem.costs.insert(wave(&[decode(0, 1)]), nz(1)).unwrap();
    let mut limited = limits();
    limited.maximum_transitions = nz(1);
    assert!(matches!(
        enumerate(&problem, limited).unwrap(),
        SearchResult::Inconclusive {
            cause: Uncertainty::EnumerationBudget,
            ..
        }
    ));
    limited = limits();
    limited.maximum_waves = NonZeroUsize::new(1).unwrap();
    assert!(matches!(
        enumerate(&problem, limited).unwrap(),
        SearchResult::Inconclusive {
            cause: Uncertainty::WaveLimit,
            ..
        }
    ));
    assert!(matches!(
        enumerate(&problem, limits()).unwrap(),
        SearchResult::Feasible { .. }
    ));
}

#[test]
fn spent_time_is_not_removed_by_resetting_ingress_or_the_plan_epoch() {
    let mut problem = Problem {
        requests: vec![request(0, 0, 1, budgets(3, 3, 3))],
        initial: State {
            at: 0,
            progress: vec![progress(0, &[])],
        },
        maximum_rows: 1,
        costs: costs(&[(wave(&[decode(0, 0)]), 2)]),
    };
    assert!(matches!(
        enumerate(&problem, limits()).unwrap(),
        SearchResult::Feasible { .. }
    ));
    problem.initial.at = 2; // true elapsed time, same original ingress/deadline
    assert!(matches!(
        enumerate(&problem, limits()).unwrap(),
        SearchResult::NoModelWitness { .. }
    ));
    assert_eq!(problem.requests[0].ingress_at, 0);
}

#[test]
fn final_prefill_emits_first_commit_and_mixed_wave_advances_atomically() {
    let mixed = wave(&[prefill(0, 0, 2), decode(1, 1)]);
    let problem = Problem {
        requests: vec![
            request(0, 2, 1, budgets(2, 2, 2)),
            request(1, 0, 2, budgets(2, 2, 2)),
        ],
        initial: State {
            at: 0,
            progress: vec![progress(0, &[]), progress(0, &[0])],
        },
        maximum_rows: 2,
        costs: costs(&[(mixed.clone(), 2)]),
    };
    let before = problem.initial.clone();
    let valid = check_sequence(&problem, &[mixed]).unwrap();
    assert_eq!(valid.final_state.progress[0].commit_times, vec![2]);
    assert_eq!(valid.final_state.progress[1].commit_times, vec![0, 2]);
    let bad = wave(&[prefill(0, 0, 2), decode(1, 0)]); // stale work
    assert!(matches!(
        check_sequence(&problem, &[bad]),
        Err(Error::Invalid(_))
    ));
    assert_eq!(problem.initial, before);
    assert!(matches!(
        check_sequence(&problem, &[]),
        Err(Error::Incomplete)
    ));
}

#[test]
fn malformed_rows_and_arithmetic_overflow_never_become_model_witnesses() {
    let mut problem = Problem {
        requests: vec![request(0, 0, 1, budgets(2, 2, 2))],
        initial: State {
            at: 0,
            progress: vec![progress(0, &[])],
        },
        maximum_rows: 2,
        costs: costs(&[(wave(&[decode(0, 0)]), 2)]),
    };
    for bad in [
        wave(&[]),
        wave(&[decode(0, 0), decode(0, 0)]),
        wave(&[decode(9, 0)]),
    ] {
        assert!(matches!(
            check_sequence(&problem, &[bad]),
            Err(Error::Invalid(_))
        ));
    }
    problem.requests[0].ingress_at = u64::MAX - 2;
    problem.initial.at = u64::MAX - 1;
    assert_eq!(enumerate(&problem, limits()), Err(Error::Overflow));
}

#[test]
fn historical_failure_is_not_repaired_by_an_empty_or_later_fast_plan() {
    let problem = Problem {
        requests: vec![request(0, 0, 2, budgets(100, 100, 5))],
        initial: State {
            at: 20,
            progress: vec![progress(0, &[1, 20])],
        },
        maximum_rows: 1,
        costs: CostTable::default(),
    };
    assert!(matches!(
        check_sequence(&problem, &[]),
        Err(Error::ModelDeadline {
            boundary: Boundary::AdjacentCommit,
            due_at: 6,
            at: 20,
            ..
        })
    ));
}
