use super::*;
use ferrum_interfaces::execution_cost::CostProductOutput;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    OwnerPredictionValidityPolicyV1, StructuredPhaseV2, StructuredPopulationPolicyV1,
};

fn opportunities() -> Vec<CaseOpportunity> {
    let key = |product| {
        populations::classify_alternatives(
            &[populations::tests::input(1, 3, product, false, true)],
            StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1,
            true,
        )
        .unwrap()
    };
    let full = key(CostProductOutput::FullLogits);
    vec![
        CaseOpportunity {
            population: full.clone(),
            minimum_fresh_members: 1,
        },
        CaseOpportunity {
            population: full,
            minimum_fresh_members: 1,
        },
        CaseOpportunity {
            population: key(CostProductOutput::GreedyToken),
            minimum_fresh_members: 1,
        },
    ]
}

fn variable_work(index: usize) -> Result<work::CaseWork> {
    Ok(work::CaseWork {
        declared_offers_minimum: 1,
        declared_offers_upper: if index == 0 { 2 } else { 1 },
        requests: 1,
        execution_actions: 2,
        serial_declared_offer_rows: 2,
        serial_token_work: 2,
    })
}

fn plan(opportunities: &[CaseOpportunity]) -> FinitePlan {
    let mut settings = StructuredSettingsV2::default();
    // An old derived member capacity must not become a new operator limit.
    settings.max_phase_samples = settings.min_phase_samples;
    let FiniteVerification::Ready(plan) = build_with_view(
        &[vec![0, 1], vec![2]],
        |index| Ok(&opportunities[index]),
        variable_work,
        work::CaseWork {
            requests: 2,
            execution_actions: 3,
            serial_token_work: 4,
            ..Default::default()
        },
        &settings,
        usize::MAX,
        usize::MAX,
    )
    .unwrap() else {
        panic!("finite fixture was not admitted");
    };
    assert!(plan.schedule.maximum_phase_members[0] > settings.max_phase_samples);
    plan
}

// Exercise the existing ready rule over every placement of one or two
// original members in each variable cohort. The other exact family supplies
// no members. A cohort already assigned to an old phase cannot enter the next.
fn completes(plan: &FinitePlan, mut choices: usize, truncate: bool) -> bool {
    let mut phase = [0; 2];
    let mut opening = [None; 2];
    let mut members = [0usize; 2];
    let mut offered = 0u64;
    let end = plan.occurrence_case_indices.len() - usize::from(truncate);
    for (ordinal, &case) in plan.occurrence_case_indices[..end].iter().enumerate() {
        let family = usize::from(case == 2);
        let mask: &[bool] = if case == 0 {
            let mask = [&[true][..], &[true, false], &[false, true], &[true, true]][choices % 4];
            choices /= 4;
            mask
        } else {
            &[true]
        };
        let mut bound_phase = None;
        let mut accepted = None;
        for &eligible in mask {
            offered += 1;
            if eligible && phase[family] < 3 {
                if opening[family].is_none() {
                    opening[family] = Some(offered);
                } else if *bound_phase.get_or_insert(phase[family]) == phase[family] {
                    members[family] += 1;
                    accepted = Some(phase[family]);
                }
            }
            for f in 0..2 {
                let p = phase[f];
                if p < 3
                    && opening[f].is_some_and(|start| {
                        plan.schedule
                            .is_ready(
                                [
                                    StructuredPhaseV2::Fit,
                                    StructuredPhaseV2::Residual,
                                    StructuredPhaseV2::Qualification,
                                ][p],
                                offered - start,
                                members[f],
                            )
                            .unwrap()
                    })
                {
                    let cut = plan.certificate.phases[p].next_cut;
                    assert!((cut.minimum..=cut.maximum).contains(&(offered as usize)));
                    phase[f] += 1;
                    opening[f] = Some(offered);
                    members[f] = 0;
                }
            }
        }
        for (p, proof) in plan.certificate.phases.iter().enumerate() {
            if proof.representatives.contains(&ordinal) {
                assert_eq!(
                    accepted,
                    Some(p),
                    "an original anchor missed its intended independent phase"
                );
            }
        }
    }
    phase == [3, 3]
}

#[test]
fn finite_plan_proves_original_anchors_for_variable_multi_member_cohorts() {
    let opportunities = opportunities();
    let plan = plan(&opportunities);
    assert_eq!(plan.representatives, [vec![0, 1], vec![2]]);
    assert_ne!(plan.family_keys[0], plan.family_keys[1]);
    assert_eq!(plan.filler_case_indices, [1, 2]);
    for choices in 0..64 {
        assert!(completes(&plan, choices, false));
    }
    assert!(
        !completes(&plan, 0, true),
        "the minimum-offer horizon is essential"
    );
    assert_eq!(plan.work.requests, plan.occurrence_case_indices.len() + 2);
    assert_eq!(
        plan.work.execution_actions,
        2 * plan.occurrence_case_indices.len() + 3
    );
    assert_eq!(
        plan.work.serial_declared_offer_rows,
        2 * plan.occurrence_case_indices.len()
    );
    assert_eq!(
        plan.work.serial_token_work,
        2 * plan.occurrence_case_indices.len() + 4
    );
    assert!(plan.retained_payload_bytes().unwrap() <= storage_bound(2, 3, usize::MAX).unwrap());
    serde_json::to_vec(&plan).unwrap();
}

#[test]
fn finite_frozen_reproof_preserves_sequence_ttl_and_rejects_changed_authority() {
    let mut opportunities = opportunities();
    let mut original = plan(&opportunities);
    original.schedule.prediction_validity =
        Some(OwnerPredictionValidityPolicyV1::OriginalSampleAgeV1);
    let settings = StructuredSettingsV2::default();
    let verify = |original: &FinitePlan, opportunities: &[CaseOpportunity], limit| {
        verify_frozen(
            original,
            |i| Ok(&opportunities[i]),
            variable_work,
            work::CaseWork::default(),
            &settings,
            limit,
        )
    };
    let FiniteVerification::Ready(rebuilt) = verify(&original, &opportunities, usize::MAX).unwrap()
    else {
        panic!("same finite inputs failed");
    };
    assert_eq!(
        original.occurrence_case_indices,
        rebuilt.occurrence_case_indices
    );
    assert_eq!(original.certificate, rebuilt.certificate);
    assert_eq!(original.schedule, rebuilt.schedule);
    assert_eq!(
        rebuilt.work.requests + 2,
        original.work.requests,
        "already-spent setup remains outside the cold replacement"
    );
    assert_eq!(
        rebuilt.retained_heap_bytes(),
        original.retained_heap_bytes()
    );
    assert!(matches!(
        verify(&original, &opportunities, 0).unwrap(),
        FiniteVerification::Skip(FiniteRejection::RetainedCapacity)
    ));
    assert!(matches!(
        verify_frozen(
            &original,
            |i| Ok(&opportunities[i]),
            |i| {
                let mut value = variable_work(i)?;
                if i == 0 {
                    value.declared_offers_upper = 100;
                }
                Ok(value)
            },
            work::CaseWork::default(),
            &settings,
            usize::MAX
        )
        .unwrap(),
        FiniteVerification::Skip(FiniteRejection::FrozenHorizon)
    ));
    opportunities[0].minimum_fresh_members = 0;
    assert!(verify(&original, &opportunities, usize::MAX).is_err());
    opportunities[0].minimum_fresh_members = 1;
    opportunities[0].population = opportunities[2].population.clone();
    assert!(verify(&original, &opportunities, usize::MAX).is_err());
    opportunities[0].population = opportunities[1].population.clone();
    original.occurrence_case_indices[0] = 2;
    assert!(verify(&original, &opportunities, usize::MAX).is_err());
}

#[test]
fn finite_capacity_and_ambiguous_opportunities_cannot_gain_authority() {
    let mut opportunities = opportunities();
    let settings = StructuredSettingsV2::default();
    let build = |opportunities: &[CaseOpportunity], limit| {
        build_with_view(
            &[vec![0, 1], vec![2]],
            |i| Ok(&opportunities[i]),
            variable_work,
            work::CaseWork::default(),
            &settings,
            usize::MAX,
            limit,
        )
    };
    assert!(matches!(
        build(&opportunities, 0).unwrap(),
        FiniteVerification::Skip(FiniteRejection::RetainedCapacity)
    ));
    let key = guaranteed_key(&opportunities[0]).unwrap().unwrap().clone();
    opportunities[0].population = CasePopulation::Unknown {
        known_alternatives: vec![key.clone()],
    };
    assert!(matches!(
        build(&opportunities, usize::MAX).unwrap(),
        FiniteVerification::Skip(FiniteRejection::ScheduleCapacity)
    ));
    opportunities[0].population = CasePopulation::Unique(key);
    opportunities[0].minimum_fresh_members = 2;
    assert!(build(&opportunities, usize::MAX).is_err());
    opportunities[0].minimum_fresh_members = 1;
    assert!(matches!(
        build_with_view(
            &[vec![0, 1], vec![2]],
            |i| Ok(&opportunities[i]),
            |i| {
                let mut work = variable_work(i)?;
                work.declared_offers_upper = 4096;
                Ok(work)
            },
            work::CaseWork::default(),
            &settings,
            usize::MAX,
            usize::MAX
        )
        .unwrap(),
        FiniteVerification::Skip(FiniteRejection::ScheduleCapacity)
    ));
    assert!(storage_bound(usize::MAX, 1, usize::MAX).is_err());
    let mut big = variable_work(1).unwrap();
    big.requests = usize::MAX;
    big.declared_offers_minimum = 2;
    assert!(padding_precedes(0, big, 1, big).is_err());
}

#[test]
fn finite_request_bound_preserves_exact_stream_and_rejects_one_fewer_occurrence() {
    let opportunities = opportunities();
    let settings = StructuredSettingsV2::default();
    let build = |maximum_requests, maximum_retained_bytes| {
        build_with_view(
            &[vec![0, 1], vec![2]],
            |i| Ok(&opportunities[i]),
            variable_work,
            work::CaseWork::default(),
            &settings,
            maximum_requests,
            maximum_retained_bytes,
        )
        .unwrap()
    };
    let FiniteVerification::Ready(original) = build(usize::MAX, usize::MAX) else {
        panic!("original finite stream must be provable");
    };
    let count = original.occurrence_case_indices.len();
    assert_eq!(original.work.requests, count);
    assert!(original
        .certificate
        .phases
        .iter()
        .any(|phase| !phase.padding.is_empty()));
    let bound = storage_bound(2, 3, count).unwrap();
    assert!(bound < storage_bound(2, 3, usize::MAX).unwrap());
    let FiniteVerification::Ready(bounded) = build(count, bound) else {
        panic!("the actual request and storage bounds must retain the full stream");
    };
    assert_eq!(
        bounded.occurrence_case_indices,
        original.occurrence_case_indices
    );
    assert_eq!(bounded.representatives, original.representatives);
    assert_eq!(bounded.family_keys, original.family_keys);
    assert_eq!(bounded.filler_case_indices, original.filler_case_indices);
    assert_eq!(bounded.padding_case_index, original.padding_case_index);
    assert_eq!(bounded.certificate, original.certificate);
    assert_eq!(bounded.schedule, original.schedule);
    assert_eq!(bounded.work, original.work);
    assert!(bounded.retained_payload_bytes().unwrap() <= bound);
    for choices in 0..64 {
        assert!(completes(&bounded, choices, false));
    }
    // Bytes remain generous: this rejects the frozen opportunity horizon,
    // not a smaller memory allowance or a weakened independent phase floor.
    assert!(matches!(
        build(count - 1, usize::MAX),
        FiniteVerification::Skip(FiniteRejection::ScheduleCapacity)
    ));
    assert!(matches!(
        build(0, usize::MAX),
        FiniteVerification::Skip(FiniteRejection::ScheduleCapacity)
    ));
    assert_eq!(
        storage_bound(2, 3, usize::MAX).unwrap(),
        storage_bound(2, 3, MAX_COHORTS).unwrap(),
        "the request allowance cannot enlarge the original wire horizon"
    );
}
