use super::*;
use ferrum_execution::model_executor::{
    PersistentCapacityMember, PersistentCapacityQuery, PersistentCapacityState,
    PersistentCohortCapacitySnapshot,
};

fn proof(
    round: &TimePlanningRound,
    slots: usize,
    through: usize,
) -> PersistentCohortCapacitySnapshot {
    // A test owner attests the resource boundary. This fixture tests scheduler
    // binding; backend tests must separately establish physical capacity.
    let members = round
        .request_facts()
        .unwrap()
        .iter()
        .map(|(id, fact)| PersistentCapacityMember {
            request_id: id.clone(),
            lifecycle: fact.lifecycle,
            state: PersistentCapacityState::Waiting,
            through_tokens: through,
        })
        .collect::<Vec<_>>();
    PersistentCohortCapacitySnapshot::from_verified_query(&PersistentCapacityQuery {
        members: &members,
        max_live_sequences: slots,
        max_wave_sequences: slots,
        inspection_budget: Default::default(),
    })
    .unwrap()
}

fn attach(round: &mut TimePlanningRound, observation: PersistentCohortCapacitySnapshot) {
    let query = PersistentCapacityQuery {
        members: observation.members(),
        max_live_sequences: observation.max_live_sequences(),
        max_wave_sequences: observation.max_wave_sequences(),
        inspection_budget: Default::default(),
    };
    round.set_persistent_capacity(&query, Some(observation.clone()));
}

#[test]
fn persistent_capacity_supports_sequential_waiting_without_fabricated_free_blocks() {
    let (scheduler, mut round, now) = waiting_fixture(3, 1, 100.0);
    round.set_capacity(None);
    let observation = proof(&round, 1, 2);
    attach(&mut round, observation);
    let decision = scheduler
        .evaluate_joint_admission(&hint(), &mut round, now, AdmissionCapMode::ExistingOnly, 0)
        .unwrap()
        .decision;
    assert_eq!(decision.choice, crate::joint_policy::PolicyChoice::Static);
    assert_eq!(decision.static_risk.request_count, 3);
    assert!(round.capacity().is_none());
    assert_eq!(
        scheduler.waiting_count(),
        3,
        "observation is not actual admission"
    );

    let (scheduler, mut round, now) = waiting_fixture(2, 1, 25.0);
    round.set_capacity(None);
    let observation = proof(&round, 1, 2);
    attach(&mut round, observation);
    let decision = scheduler
        .evaluate_joint_admission(&hint(), &mut round, now, AdmissionCapMode::ExistingOnly, 0)
        .unwrap()
        .decision;
    assert_eq!(
        decision.choice,
        crate::joint_policy::PolicyChoice::AllKnownInfeasible
    );
    assert_eq!(decision.static_risk.request_count, 2);
}

#[test]
fn persistent_capacity_rejects_insufficient_slots_or_execution_ceiling() {
    for (configured, proven, through, expected) in [
        (2, 1, 2, crate::joint_policy::Unsupported::MissingCapacity),
        (1, 1, 1, crate::joint_policy::Unsupported::AdmissionCapacity),
    ] {
        let (scheduler, mut round, now) = waiting_fixture(3, configured, 100.0);
        round.set_capacity(None);
        let observation = proof(&round, proven, through);
        attach(&mut round, observation);
        assert_eq!(
            scheduler
                .evaluate_joint_admission(
                    &hint(),
                    &mut round,
                    now,
                    AdmissionCapMode::ExistingOnly,
                    0
                )
                .unwrap_err(),
            expected
        );
    }
}

#[test]
fn persistent_capacity_is_bound_to_exact_lifetimes_and_projection_budget() {
    let (_, mut round, _) = waiting_fixture(3, 1, 100.0);
    let observation = proof(&round, 1, 2);
    let facts = round.request_facts().unwrap().to_vec();
    let clocks = round.request_clocks().unwrap().to_vec();
    attach(&mut round, observation.clone());
    assert!(round.persistent_capacity().is_some());
    assert!(round.set_request_clocks(clocks.clone()));
    assert!(
        round.persistent_capacity().is_none(),
        "recapture invalidates old proof"
    );

    let mut replacement = facts.clone();
    replacement[0].1.lifecycle.admission += 100;
    assert!(round.set_request_facts(replacement));
    assert!(round.set_request_clocks(clocks.clone()));
    attach(&mut round, observation.clone());
    assert!(
        round.persistent_capacity().is_none(),
        "same id is not same lifetime"
    );

    assert!(round.set_request_facts(facts.clone()));
    assert!(round.set_request_clocks(clocks.clone()));
    let bytes = round.request_fact_bytes
        + clocks.len() * std::mem::size_of::<(RequestId, RequestClockFacts)>()
        + observation.retained_bytes();
    round.config.planning.max_request_fact_bytes = bytes - 1;
    attach(&mut round, observation.clone());
    assert!(round.persistent_capacity().is_none());
    round.config.planning.max_request_fact_bytes = bytes;
    attach(&mut round, observation.clone());
    assert!(round.persistent_capacity().is_some());
    round.set_capacity(None);
    assert!(round.persistent_capacity().is_none());

    assert!(round.set_request_facts(facts[..2].to_vec()));
    assert!(round.set_request_clocks(clocks[..2].to_vec()));
    attach(&mut round, observation);
    assert!(
        round.persistent_capacity().is_none(),
        "proof must not outlive membership"
    );
}

#[test]
fn persistent_capacity_requires_current_incarnation_and_queue_residency() {
    let (scheduler, mut round, now) = waiting_fixture(3, 1, 100.0);
    round.set_capacity(None);
    let waiting = proof(&round, 1, 2);
    let issuer = ferrum_execution::token_execution::ExecutorSequenceIssuer::new();
    let mut members = waiting.members().to_vec();
    members[0].state = PersistentCapacityState::Resident {
        sequence: issuer.issue_sequence().unwrap(),
    };
    let query = PersistentCapacityQuery {
        members: &members,
        max_live_sequences: 1,
        max_wave_sequences: 1,
        inspection_budget: Default::default(),
    };
    let resident = PersistentCohortCapacitySnapshot::from_verified_query(&query).unwrap();
    round.set_persistent_capacity(&query, Some(waiting));
    assert!(
        round.persistent_capacity().is_none(),
        "waiting proof cannot describe a new resident"
    );
    round.set_persistent_capacity(&query, Some(resident.clone()));
    assert!(round.persistent_capacity().is_some());
    assert_eq!(
        scheduler
            .evaluate_joint_admission(&hint(), &mut round, now, AdmissionCapMode::ExistingOnly, 0)
            .unwrap_err(),
        crate::joint_policy::Unsupported::AdmissionCapacity
    );

    members[0].state = PersistentCapacityState::Resident {
        sequence: issuer.issue_sequence().unwrap(),
    };
    let replacement = PersistentCapacityQuery {
        members: &members,
        max_live_sequences: 1,
        max_wave_sequences: 1,
        inspection_budget: Default::default(),
    };
    round.set_persistent_capacity(&replacement, Some(resident));
    assert!(
        round.persistent_capacity().is_none(),
        "same request lifetime can have a different executor incarnation"
    );
}
