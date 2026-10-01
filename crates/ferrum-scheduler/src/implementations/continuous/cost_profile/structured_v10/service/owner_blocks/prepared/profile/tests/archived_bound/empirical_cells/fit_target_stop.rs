use super::*;
use crate::implementations::continuous::cost_model::{CostBoundary,WaveObservationOutcome};
use crate::implementations::continuous::cost_profile::structured_v10::service::owner_blocks::collector::State as OwnerState;
use std::num::NonZeroU64;
mod numeric;

#[test]
#[ignore = "requires immutable original source7 via FERRUM_ARCHIVED_BOUND_SPEC"]
fn replay_full_family_with_original_fit_target_and_global_residual() {
    let spec: AuditSpec = serde_json::from_slice(
        &std::fs::read(std::env::var_os("FERRUM_ARCHIVED_BOUND_SPEC").unwrap()).unwrap(),
    )
    .unwrap();
    let header = seeded::verify_source_file(&spec.actual_source, spec.actual_source_sha256);
    let d = &header.declaration;
    let mut demands = BTreeMap::new();
    for line in BufReader::new(std::fs::File::open(&spec.query_journal).unwrap()).lines() {
        let value: Value = serde_json::from_str(&line.unwrap()).unwrap();
        if value["event"] != "query_constructed" {
            continue;
        }
        for pair in &spec.pairs {
            if value["transaction"] == pair.transaction
                && value["data"]["attempt"] == pair.attempt
                && value["data"]["alternative"] == pair.alternative
            {
                assert!(demands
                    .insert(pair.call_id, value["data"]["demand"].clone())
                    .is_none());
            }
        }
    }
    assert_eq!(demands.len(), spec.pairs.len());
    let mut demands_verified = BTreeSet::new();
    let mut paired_phases = BTreeMap::new();
    let mut numeric: Option<numeric::Control> = None;
    let mut physical_contract = d.nonnegative_envelope.clone().unwrap();
    if d.schedule.algorithm_universe
        == Some(OwnerAlgorithmUniversePolicyV1::SeededFirstOrdinaryDiscoveryBlockSubsetV1)
    {
        physical_contract.algorithm_universe = None;
    }
    let create = || {
        StructuredServiceCollectorV7::new_streaming(
            header.clone(),
            CostProfileLoadLimits::default(),
            NonZeroU64::new(header.maximum_file_bytes).unwrap(),
        )
        .unwrap()
    };
    let mut archive = create();
    let mut frozen = create();
    let mut selected = None;
    let mut target = None;
    let mut fit_members = 0usize;
    let mut fitted_at_block = 0;
    let mut rule = [0; 32];
    let mut opened = header.opening.monotonic_ns;
    let mut block = 0u64;
    let mut phase = 1usize;
    let mut first_phase_block = 0u64;
    let mut samples: [Vec<StructuredNumericObservationV2>; 2] = [Vec::new(), Vec::new()];
    let mut visits = [0u64; 2];
    let mut decisions = Vec::new();
    let mut boundaries = Vec::new();
    let mut failure = None::<String>;
    let mut gate: Option<Box<dyn Fn(&StructuredInputV2) -> bool>> = None;
    let mut all_rows = BTreeMap::<u32, usize>::new();
    let mut selected_rows = [BTreeMap::<u32, usize>::new(), BTreeMap::new()];
    let mut outside_fit = 0usize;
    let mut outside_residual = 0usize;
    let mut peak_input_and_readiness = 0usize;
    // Counterfactual rule only: keep original frozen-support member eligibility; use the
    // already implemented all-Fit-input-target readiness for R/Q. No new
    // schema or live replay/publication claim is made by this diagnostic.
    let mut target_schedule = d.schedule.clone();
    target_schedule.phase_support = None;
    for line in BufReader::new(std::fs::File::open(&spec.actual_source).unwrap())
        .lines()
        .skip(1)
    {
        let record: StructuredServiceRecordV7 = serde_json::from_str(&line.unwrap()).unwrap();
        archive.push(&record).unwrap();
        if selected.is_none() {
            let before: BTreeMap<_, _> =
                if matches!(record, StructuredServiceRecordV7::BlockClose { .. }) {
                    frozen
                        .owners
                        .iter()
                        .filter(|o| matches!(o.state, OwnerState::Empty) && !o.samples.is_empty())
                        .map(|o| {
                            (
                                o.contract.owner_attempt_id,
                                (
                                    OwnerInputTargetV1::from_samples(&o.samples).unwrap(),
                                    o.samples.len(),
                                ),
                            )
                        })
                        .collect()
                } else {
                    BTreeMap::new()
                };
            frozen.push(&record).unwrap();
            if let StructuredServiceRecordV7::BlockClose { block: b, .. } = &record {
                if let Some((i, o)) = frozen.owners.iter().enumerate().find(|(_, o)| {
                    o.scope.owner.role == StructuredWaveRoleV2::OrdinaryDecode
                        && o.scope.owner.product == StructuredProductV2::GreedyToken
                        && o.scope.numerical_family.is_some()
                        && matches!(o.state, OwnerState::Fitted(_))
                }) {
                    let (t, n) = before[&o.contract.owner_attempt_id].clone();
                    target = Some(t);
                    fit_members = n;
                    selected = Some(i);
                    fitted_at_block = *b;
                    first_phase_block = b + 1;
                    rule = o.contract.membership_rule;
                }
            }
        }
        match record {
            StructuredServiceRecordV7::BlockOpen {
                block: b,
                opened_at_ns,
                ..
            } => {
                block = b;
                opened = opened_at_ns;
            }
            StructuredServiceRecordV7::Completed { wave } => {
                if selected.is_none() || block <= fitted_at_block || failure.is_some() {
                    continue;
                }
                let (input, wall, observed) = physical::validate_parts(
                    &header.fingerprint,
                    header.opening.monotonic_ns,
                    Some(&physical_contract),
                    opened,
                    wave.ticket,
                    wave.fifo,
                    wave.issued_at_ns,
                    &wave.host_stages,
                    wave.independent.as_ref(),
                    &mut physical::Frontiers::default(),
                )
                .unwrap();
                let OwnerState::Fitted(fit) = &frozen.owners[selected.unwrap()].state else {
                    unreachable!()
                };
                let call = wave.host_stages.call_id;
                if phase > 2 || demands.contains_key(&call) {
                    let (prepared, offered) =
                        physical::original_prepared(&wave.host_stages, wave.independent.as_ref())
                            .unwrap();
                    let query = StructuredQueryV2::exact(
                        input_replay::project_service_actual_with_domain(
                            &prepared,
                            &offered,
                            &physical_contract.workload_domain,
                        )
                        .unwrap(),
                    );
                    if let Some(demand) = demands.get(&call) {
                        assert_eq!(
                            serde_json::to_value(query.required_coverage().unwrap()).unwrap(),
                            *demand
                        );
                        demands_verified.insert(call);
                        paired_phases.insert(
                            call,
                            match phase {
                                1 => "residual",
                                2 => "qualification",
                                _ => "future",
                            },
                        );
                    }
                    if phase > 2 {
                        numeric.as_mut().unwrap().future(
                            fit,
                            &query,
                            wave.issued_at_ns,
                            call,
                            wall,
                            demands.contains_key(&call),
                        );
                        continue;
                    }
                }
                *all_rows.entry(input.owner().rows).or_default() += 1;
                let Some(input) = fit.diagnose_declared_fit_input(&input, wave.issued_at_ns) else {
                    outside_fit += 1;
                    continue;
                };
                if phase == 2 && !gate.as_ref().unwrap()(&input) {
                    outside_residual += 1;
                    continue;
                }
                if samples[phase - 1].len() >= d.settings.max_phase_samples
                    || samples[phase - 1].len() >= d.schedule.maximum_phase_members[phase]
                {
                    failure = Some("original phase member capacity".into());
                    continue;
                }
                *selected_rows[phase - 1]
                    .entry(input.owner().rows)
                    .or_default() += 1;
                let previous = fit_members + if phase == 2 { samples[0].len() } else { 0 };
                samples[phase - 1].push(StructuredNumericObservationV2 {
                    source: header.capture_identity,
                    protocol: header.protocol,
                    ordinal: wave.fifo,
                    membership: StructuredMemberBindingV2 {
                        rule_signature: rule,
                        offered_ordinal: wave.ticket,
                        member_ordinal: (previous + samples[phase - 1].len() + 1) as u64,
                        phase: if phase == 1 {
                            StructuredPhaseV2::Residual
                        } else {
                            StructuredPhaseV2::Qualification
                        },
                    },
                    call_id: wave.host_stages.call_id,
                    fingerprint: header.fingerprint.clone().into(),
                    input,
                    boundary: CostBoundary::PreparationToHostSettledV1,
                    outcome: WaveObservationOutcome::Completed,
                    observed_at_ns: observed,
                    wall_ns: wall,
                });
            }
            StructuredServiceRecordV7::BlockClose {
                block: b, closing, ..
            } => {
                if selected.is_none() || b <= fitted_at_block || phase > 2 || failure.is_some() {
                    continue;
                }
                let OwnerState::Fitted(fit) = &frozen.owners[selected.unwrap()].state else {
                    unreachable!()
                };
                let offered = (b - first_phase_block + 1)
                    .checked_mul(d.schedule.block_offered as u64)
                    .unwrap();
                let scratch = target_schedule
                    .readiness_scratch_bytes(&samples[phase - 1])
                    .unwrap();
                let retained = samples
                    .iter()
                    .map(|v| {
                        v.capacity() * std::mem::size_of::<StructuredNumericObservationV2>()
                            + v.iter()
                                .map(|s| s.input.retained_payload_bytes().unwrap())
                                .sum::<usize>()
                    })
                    .sum::<usize>()
                    + fit.retained_payload_bytes().unwrap()
                    + target.as_ref().unwrap().retained_heap_bytes();
                peak_input_and_readiness = peak_input_and_readiness.max(retained + scratch);
                if retained + scratch > d.maximum_retained_numeric_bytes {
                    failure = Some(
                        "original numeric budget exceeded by retained inputs and readiness scratch"
                            .into(),
                    );
                    continue;
                }
                let decision = target_schedule.assess_inputs(
                    if phase == 1 {
                        StructuredPhaseV2::Residual
                    } else {
                        StructuredPhaseV2::Qualification
                    },
                    offered,
                    &samples[phase - 1],
                    target.as_ref(),
                    &d.settings,
                    &mut visits[phase - 1],
                );
                decisions.push(json!({"phase":phase,"block":b,"members":samples[phase-1].len(),
                    "offered":offered,"geometry_visits":visits[phase-1],"decision":format!("{decision:?}")}));
                match decision {
                    Ok(OwnerInputReadinessDecisionV1::Freeze) => {
                        boundaries
                            .push(json!({"phase":phase,"block":b,"at_ns":closing.monotonic_ns,
                            "members":samples[phase-1].len(),"rows":selected_rows[phase-1]}));
                        if phase == 1 {
                            gate = fit.diagnose_frozen_residual_input_gate(&samples[0]);
                            if gate.is_none() {
                                failure = Some("frozen R failed original branch validity".into());
                                continue;
                            }
                        }
                        if phase == 1 {
                            match numeric::Control::calibrate(fit, &samples[0], &d.settings) {
                                Ok(value) => numeric = Some(value),
                                Err(error) => {
                                    failure = Some(error);
                                    continue;
                                }
                            }
                        } else {
                            numeric.as_mut().unwrap().qualify(
                                fit,
                                &samples[1],
                                closing.monotonic_ns,
                            );
                        }
                        phase += 1;
                        first_phase_block = b + 1;
                    }
                    Ok(OwnerInputReadinessDecisionV1::Wait) => {}
                    other => {
                        failure = Some(format!("{other:?}"));
                    }
                }
            }
            _ => {}
        }
    }
    eprintln!(
        "ORIGINAL_FIT_TARGET_GLOBAL_RESIDUAL {}",
        json!({
            "fit_block":fitted_at_block,"fit_members":fit_members,"original_phase_support":d.schedule.phase_support,
            "counterfactual_completed_rq_input_gates":phase==3&&failure.is_none(),"failure":failure,
            "boundaries":boundaries,"decisions":decisions,"all_rows_seen_while_collecting":all_rows,
            "outside_original_fit_support":outside_fit,"outside_frozen_residual_support":outside_residual,
            "phase_geometry_visits":visits,"original_geometry_limit":d.schedule.input_readiness.as_ref().unwrap().maximum_geometry_visits,
            "peak_retained_inputs_plus_readiness_scratch":peak_input_and_readiness,
            "original_numeric_budget":d.maximum_retained_numeric_bytes,
            "numerical":numeric.as_ref().map(numeric::Control::report),
        "paired_demands_verified":demands_verified,
        "paired_original_phase":paired_phases,
        "scope":"complete original family blocks; existing Fit work/branch target input stops; unchanged original identified Fit, global independent R margin and all Q members",
            "budget_scope":"retained diagnostic phase inputs + original Fit + existing readiness scratch/visits; prototype error/point evaluation is executed, production transition reservations/publication are not proven",
            "qualification_scope":"numerical Q all selected members counted, no filtering/retry; no production publication or numeric-range authority is claimed"
        })
    );
    assert!(selected.is_some());
    assert!(all_rows.len() > 1);
    if phase == 3 && failure.is_none() {
        assert_eq!(demands_verified.len(), demands.len());
    }
}
