//! Original source8 cohorts whose ordinary decode population shrinks in width.
use super::*;
use crate::implementations::continuous::cost_model::structured_v2::StructuredPopulationPolicyV1;
use crate::implementations::continuous::cost_profile::structured_v10::wire::{
    CompletedRequest, Prepared,
};
use ferrum_interfaces::execution_cost::ActualWaveGraphState;

const POLICY: [u8; 32] = [44; 32];

fn paired(ns: u64) -> StructuredServiceClockV7 {
    StructuredServiceClockV7 {
        monotonic_ns: ns,
        wall_unix_ns: 1_000_000 + ns - 1,
    }
}
fn append(bytes: &mut Vec<u8>, record: &impl Serialize) {
    bytes.extend(record_bytes_v7(record).unwrap());
}
fn event(
    c: &mut StructuredPreparedOwnerBlockCollectorV8,
    bytes: &mut Vec<u8>,
    value: serde_json::Value,
    preparation: bool,
) {
    let record = if preparation {
        StructuredPreparedOwnerBlockRecordV8::Preparation(
            StructuredPreparationEventV8::from_diagnostic(value).unwrap(),
        )
    } else {
        StructuredPreparedOwnerBlockRecordV8::Cohort(
            StructuredCohortEventV8::from_diagnostic(value).unwrap(),
        )
    };
    c.push(&record).unwrap();
    append(bytes, &record);
}
fn open_for_offer(
    c: &mut StructuredPreparedOwnerBlockCollectorV8,
    bytes: &mut Vec<u8>,
    ticket: u64,
    block: usize,
) {
    if ticket % block as u64 == 0 {
        if ticket != 0 {
            append(bytes, &c.close_block(paired(ticket * 2000 + 1101)).unwrap());
        }
        append(
            bytes,
            &c.open_block((ticket + 1) * 2000 - 1, ticket).unwrap(),
        );
    }
}

pub(super) fn declared(block: usize, counts: [usize; 3]) -> StructuredPreparedOwnerBlockHeaderV8 {
    let mut h = header();
    let identity = ExecutorCostIdentity {
        schema_version: EXECUTOR_COST_IDENTITY_SCHEMA,
        model_weights: h.fingerprint.model_weights,
        numerical_policy: h.fingerprint.numerical_policy,
        device_runtime: h.fingerprint.device_runtime,
        execution_config: h.fingerprint.execution_config,
    };
    let envelope = h
        .declaration
        .population
        .nonnegative_envelope
        .as_mut()
        .unwrap();
    envelope.population_policy = StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1;
    envelope.workload_domain = CostWorkloadDomainV1::new_vnext(
        &identity,
        CostWorkloadLimitsV1 {
            maximum_rows: NonZeroU32::new(2).unwrap(),
            maximum_context_tokens: NonZeroU32::new(128).unwrap(),
            maximum_scheduled_tokens_per_wave: NonZeroU64::new(128).unwrap(),
            output_vocabulary_elements: NonZeroU64::new(16).unwrap(),
            repetition_slot_capacity: 4,
            fixed_state_bytes_per_row: 0,
        },
    )
    .unwrap();
    h.declaration.population.schedule =
        OwnerBlockScheduleV1::new(block, [block; 3], [8; 3]).unwrap();
    h.declaration.cohort_plan.phases = std::array::from_fn(|phase| {
        (0..counts[phase])
            .map(|repetition| CohortV2 {
                manifest_case: 0,
                repetition: repetition as u32,
                requests: [3, 4]
                    .into_iter()
                    .map(|maximum_output| CohortRequestV2 {
                        manifest_prompt: 0,
                        maximum_output,
                    })
                    .collect(),
            })
            .collect()
    });
    h.declaration.prefix_plan.phases = std::array::from_fn(|phase| {
        h.declaration.cohort_plan.phases[phase]
            .iter()
            .map(|_| {
                Some(StructuredPrefixCohortV5 {
                    release_generated: 1,
                    slots: (0..2)
                        .map(|_| StructuredPrefixSlotV5 {
                            tokenizer_policy_sha256: [33; 32],
                            token_ids: vec![ferrum_types::TokenId::new(7)],
                            token_bytes: vec![b"a".to_vec()],
                        })
                        .collect(),
                })
            })
            .collect()
    });
    StructuredPreparedOwnerBlockHeaderV8::new(
        h.capture_identity,
        h.generation,
        h.fingerprint,
        h.producer,
        h.opening,
        h.declaration,
        h.maximum_file_bytes,
    )
    .unwrap()
}

fn prepared(ids: &[&str], generated: u64, maximum_outputs: &[u64]) -> Prepared {
    old::prepared_batch_route_policy_bounds_and_future(
        ids,
        generated + 1,
        generated,
        None,
        ActualWaveGraphState::Disabled,
        Some(POLICY),
        64 + generated.saturating_sub(1) as u32,
        maximum_outputs,
        None,
    )
    .0
}

pub(in super::super::super) fn collected() -> (
    Vec<u8>,
    StructuredPreparedOwnerBlockCollectorV8,
    StructuredPreparedOwnerBlockCheckpointV8,
) {
    collected_with(16, [16, 8, 8])
}

fn collected_with(
    block: usize,
    counts: [usize; 3],
) -> (
    Vec<u8>,
    StructuredPreparedOwnerBlockCollectorV8,
    StructuredPreparedOwnerBlockCheckpointV8,
) {
    let h = declared(block, counts);
    let mut bytes = Vec::new();
    append(&mut bytes, &h);
    let mut c =
        StructuredPreparedOwnerBlockCollectorV8::new(h.clone(), CostProfileLoadLimits::default())
            .unwrap();
    let mut ticket = 0u64;
    for phase_index in 0..3 {
        let phase = match phase_index {
            0 => StructuredProfilePhaseV10::Fit,
            1 => StructuredProfilePhaseV10::Residual,
            _ => StructuredProfilePhaseV10::Qualification,
        };
        for cohort in 0..counts[phase_index] {
            let ids: Vec<_> = (0..2)
                .map(|slot| format!("family-{phase_index}-{cohort}-{slot}"))
                .collect();
            let names: Vec<_> = ids.iter().map(String::as_str).collect();
            open_for_offer(&mut c, &mut bytes, ticket, block);
            event(
                &mut c,
                &mut bytes,
                serde_json::json!({"kind":"cohort_begin","phase":phase,"cohort":cohort,"manifest_case":0,"repetition":cohort}),
                false,
            );
            for (slot, maximum) in [3, 4].into_iter().enumerate() {
                event(
                    &mut c,
                    &mut bytes,
                    serde_json::json!({"kind":"request_admitted","phase":phase,"cohort":cohort,"slot":slot,"request_id":ids[slot],"maximum_output":maximum}),
                    false,
                );
            }
            ticket += 1;
            let p = prepared(&names, 0, &[3, 4]);
            let before: Vec<_> = ids.iter().map(|id| serde_json::json!({"request_id":id,"owner_incarnation":1,"work_generation":1,"generated_tokens":0,"kv_tokens":0,"model_cache_id":null,"pending_utf8":[],"output_accepted_ordinal":0})).collect();
            let after: Vec<_> = ids.iter().map(|id| serde_json::json!({"request_id":id,"owner_incarnation":1,"work_generation":2,"generated_tokens":1,"kv_tokens":64,"model_cache_id":"cache","pending_utf8":[],"output_accepted_ordinal":1})).collect();
            let offered_rows: Vec<_> = p
                .rows
                .iter()
                .zip(&before)
                .map(|(row, before)| serde_json::json!({"before":before,"work":row.frontier.work}))
                .collect();
            event(
                &mut c,
                &mut bytes,
                serde_json::json!({"kind":"preparation_offered","offered":ticket,"phase":phase,"cohort":cohort,"rows":offered_rows}),
                true,
            );
            let mut s = old::stages(&old::header(), &p, ticket, 1000);
            s.statistical_evidence = None;
            s.structured_evidence = None;
            let rows: Vec<_> = (0..2).map(|slot| serde_json::json!({"before":before[slot],"after":after[slot],"preparation_commit":{"request_id":ids[slot],"owner_incarnation":1,"work_generation":1,"generated_before":0,"generated_after":1,"original_candidate":9,"committed_token":7,"route":"full_logits_sampler","pending_before":[],"pending_after":[]}})).collect();
            event(
                &mut c,
                &mut bytes,
                serde_json::json!({"kind":"preparation_completed","offered":ticket,"phase":phase,"cohort":cohort,"reconciled":true,"queue":{"disposition":"published","accepted_ordinal":ticket},"host_stages":s,"rows":rows,"failure":null}),
                true,
            );
            let mut hash = Sha256::new();
            hash.update(b"ferrum.calibration.generated-prefix.v1\0");
            hash.update(7u32.to_le_bytes());
            let prefix: [u8; 32] = hash.finalize().into();
            for slot in 0..2 {
                event(
                    &mut c,
                    &mut bytes,
                    serde_json::json!({"kind":"preparation_released","phase":phase,"cohort":cohort,"slot":slot,"receipt":{"frontier":after[slot],"original_policy_signature":POLICY,"original_numeric_policy":p.recipe.physical_host_rows[slot].installed_policy,"generated_prefix_sha256":prefix,"through_call_id":ticket,"through_fifo_ordinal":ticket,"actor_applied_output_ordinal":1}}),
                    true,
                );
            }
            for generated in 1..4 {
                open_for_offer(&mut c, &mut bytes, ticket, block);
                ticket += 1;
                let p = if generated < 3 {
                    prepared(&names, generated, &[3, 4])
                } else {
                    prepared(&names[1..], generated, &[4])
                };
                let s = old::stages(&old::header(), &p, ticket, 1000);
                let wave = StructuredServiceWaveV7::from_diagnostic(
                    ticket,
                    ticket * 2000,
                    ticket,
                    serde_json::to_value(&s).unwrap(),
                    None,
                )
                .unwrap();
                let record = StructuredPreparedOwnerBlockRecordV8::Population(
                    StructuredServiceRecordV7::Completed { wave },
                );
                c.push(&record).unwrap();
                append(&mut bytes, &record);
                for row in &s.rows {
                    if let Some(terminal) = row.terminal.clone() {
                        let slot = ids.iter().position(|id| id == &row.request_id).unwrap();
                        event(
                            &mut c,
                            &mut bytes,
                            serde_json::json!({"kind":"request_completed","request":CompletedRequest {phase, cohort, slot, request_id:row.request_id.clone(),owner_incarnation:1,call_id:ticket,fifo:ticket,generated_tokens:generated+1,terminal}}),
                            false,
                        );
                    }
                }
            }
            event(
                &mut c,
                &mut bytes,
                serde_json::json!({"kind":"cohort_end","phase":phase,"cohort":cohort,"admitted_count":2,"completed_count":2}),
                false,
            );
        }
    }
    assert_eq!(ticket, (counts.iter().sum::<usize>() * 4) as u64);
    assert_eq!(ticket, c.offered());
    let closing = paired(ticket * 2000 + 1101);
    if ticket % block as u64 == 0 {
        append(&mut bytes, &c.close_block(closing).unwrap());
    } else {
        append(
            &mut bytes,
            &c.seal_complete_cohorts_with_partial_tail(closing)
                .unwrap()
                .unwrap(),
        );
    }
    assert!(c.qualified_children() > 0, "{:?}", c.audit());
    let (record, checkpoint) = c.checkpoint(closing).unwrap();
    append(&mut bytes, &record);
    (bytes, c, checkpoint)
}

#[test]
fn source8_numerical_family_collects_real_widths_and_replays_all_original_offers() {
    let (bytes, c, _) = collected();
    assert_eq!(
        c.prepared_audit().cohort_phase_policy,
        StructuredPreparedCohortPhasePolicyV8::FirstEligibleNumericalFamilyPhaseV1
    );
    let owners = &c.population.owners;
    assert_eq!(owners.len(), 1);
    assert_eq!(owners[0].scope.owner.rows, 2);
    assert!(owners[0].scope.numerical_family.is_some());
    let replay = replay_structured_source_v8(&bytes, &CostProfileLoadLimits::default()).unwrap();
    assert_eq!(replay.population.source_receipt(), c.source_receipt());
    assert_eq!(
        replay.population.qualified_children(),
        c.qualified_children()
    );
    assert!(replay_structured_source_v7(&bytes, &CostProfileLoadLimits::default()).is_err());
}

#[test]
fn source8_same_cohort_shrinking_width_does_not_reenter_later_family_phase() {
    let (bytes, c, _) = collected_with(14, [14, 7, 7]);
    let audit = c.prepared_audit();
    assert!(audit.excluded_cohort_phase_attempts > 0);
    assert_eq!(
        audit
            .phase_exclusions
            .iter()
            .map(|x| x.excluded_original_offers)
            .sum::<u64>(),
        audit.excluded_cohort_phase_attempts
    );
    let mut lines = bytes.split_inclusive(|b| *b == b'\n');
    let header = serde_json::from_slice(lines.next().unwrap()).unwrap();
    let mut replay =
        StructuredPreparedOwnerBlockCollectorV8::new(header, CostProfileLoadLimits::default())
            .unwrap();
    let mut excluded_widths = Vec::new();
    for line in lines {
        let record: StructuredPreparedOwnerBlockRecordV8 = serde_json::from_slice(line).unwrap();
        let before = replay.prepared_audit().excluded_cohort_phase_attempts;
        replay.push(&record).unwrap();
        if replay.prepared_audit().excluded_cohort_phase_attempts > before {
            let StructuredPreparedOwnerBlockRecordV8::Population(
                StructuredServiceRecordV7::Completed { wave },
            ) = record
            else {
                panic!("only original completed offers can be phase-excluded")
            };
            excluded_widths.push(wave.host_stages.rows.len());
        }
    }
    assert!(excluded_widths.contains(&1));
    assert_eq!(replay.offered(), c.offered());
    assert_eq!(replay.source_receipt(), c.source_receipt());
    assert_eq!(
        replay.prepared_audit().excluded_cohort_phase_attempts,
        audit.excluded_cohort_phase_attempts
    );
}

#[test]
fn family_population_falls_back_only_for_checked_unsupported_shapes() {
    use super::super::super::discovery::PopulationKey;
    use crate::implementations::continuous::cost_model::structured_v2::StructuredUnknownV2;
    use crate::implementations::continuous::cost_profile::structured_v10::prepared::project_service_actual_with_domain;
    let h = declared(8, [1; 3]);
    let contract = h
        .declaration
        .population
        .nonnegative_envelope
        .as_ref()
        .unwrap();
    for generated in [0, 1] {
        let (p, offered, unbound, _) = old::prepared_batch_route_policy_bounds_and_future(
            &["checked-a", "checked-b"],
            generated + 1,
            generated,
            None,
            ActualWaveGraphState::Disabled,
            Some(POLICY),
            64,
            &[3, 4],
            None,
        );
        assert_eq!(
            PopulationKey::from_input(&unbound, contract.population_policy),
            Err(StructuredUnknownV2::WrongDomain)
        );
        let input = project_service_actual_with_domain(&p, &offered, &contract.workload_domain)
            .unwrap()
            .with_cost_template_policy(contract.template_policy)
            .unwrap();
        let key = PopulationKey::from_input(&input, contract.population_policy).unwrap();
        assert_eq!(matches!(key, PopulationKey::ExactOwner(_)), generated == 0);
        assert_eq!(
            matches!(key, PopulationKey::NumericalFamily(_)),
            generated == 1
        );
        let mut malformed = p;
        malformed.rows[0].frontier.maximum_output = 0;
        assert!(project_service_actual_with_domain(
            &malformed,
            &offered,
            &contract.workload_domain
        )
        .is_err());
    }
}
