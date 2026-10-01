use super::*;
use crate::implementations::continuous::cost_profile::structured_v10::wire::{
    CompletedRequest, Prepared, StructuredProfilePhaseV10,
};
use ferrum_interfaces::execution_cost::ActualWaveGraphState;
const POLICY: [u8; 32] = [44; 32];

fn paired(ns: u64) -> StructuredServiceClockV7 {
    StructuredServiceClockV7 {
        monotonic_ns: ns,
        wall_unix_ns: 1_000_000 + ns - 1,
    }
}
fn declared(block_offered: usize, counts: [usize; 3]) -> StructuredPreparedOwnerBlockHeaderV8 {
    let mut h = header();
    h.declaration.population.schedule =
        OwnerBlockScheduleV1::new(block_offered, [block_offered; 3], [8; 3]).unwrap();
    h.declaration.cohort_plan.phases = std::array::from_fn(|phase| {
        (0..counts[phase])
            .map(|repetition| CohortV2 {
                manifest_case: 0,
                repetition: repetition as u32,
                requests: vec![CohortRequestV2 {
                    manifest_prompt: 0,
                    maximum_output: 3,
                }],
            })
            .collect()
    });
    h.declaration.prefix_plan.phases = std::array::from_fn(|phase| {
        h.declaration.cohort_plan.phases[phase]
            .iter()
            .map(|_| {
                Some(StructuredPrefixCohortV5 {
                    release_generated: 1,
                    slots: vec![StructuredPrefixSlotV5 {
                        tokenizer_policy_sha256: [33; 32],
                        token_ids: vec![ferrum_types::TokenId::new(7)],
                        token_bytes: vec![b"a".to_vec()],
                    }],
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
fn append(bytes: &mut Vec<u8>, record: &impl Serialize) {
    bytes.extend(record_bytes_v7(record).unwrap());
}
fn open_for_offer(
    c: &mut StructuredPreparedOwnerBlockCollectorV8,
    bytes: &mut Vec<u8>,
    ticket: u64,
    block_offered: usize,
) {
    if ticket % block_offered as u64 != 0 {
        return;
    }
    if ticket != 0 {
        append(bytes, &c.close_block(paired(ticket * 2000 + 1101)).unwrap());
    }
    append(
        bytes,
        &c.open_block((ticket + 1) * 2000 - 1, ticket).unwrap(),
    );
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
fn prepared(id: &str, generated: u64) -> Prepared {
    prepared_on_route(id, generated, ActualWaveGraphState::Disabled)
}
fn prepared_on_route(id: &str, generated: u64, graph: ActualWaveGraphState) -> Prepared {
    old::prepared_route_policy_bounds(
        id,
        generated + 1,
        generated,
        None,
        graph,
        Some(POLICY),
        if generated < 2 { 64 } else { 65 },
        3,
    )
    .0
}
pub(in super::super::super) fn collected() -> (
    Vec<u8>,
    StructuredPreparedOwnerBlockCollectorV8,
    StructuredPreparedOwnerBlockCheckpointV8,
) {
    collected_with(24, [16, 8, 8])
}
pub(super) fn collected_with(
    block_offered: usize,
    counts: [usize; 3],
) -> (
    Vec<u8>,
    StructuredPreparedOwnerBlockCollectorV8,
    StructuredPreparedOwnerBlockCheckpointV8,
) {
    collected_with_qualification(block_offered, counts, true)
}
pub(super) fn collected_with_qualification(
    block_offered: usize,
    counts: [usize; 3],
    require_qualified: bool,
) -> (
    Vec<u8>,
    StructuredPreparedOwnerBlockCollectorV8,
    StructuredPreparedOwnerBlockCheckpointV8,
) {
    collected_with_budget(
        block_offered,
        counts,
        require_qualified,
        None,
        CostProfileLoadLimits::default(),
    )
}
pub(in super::super::super) fn collected_streaming(
    budget: std::num::NonZeroU64,
    limits: CostProfileLoadLimits,
) -> (
    Vec<u8>,
    StructuredPreparedOwnerBlockCollectorV8,
    StructuredPreparedOwnerBlockCheckpointV8,
) {
    collected_with_budget(24, [16, 8, 8], true, Some(budget), limits)
}
fn collected_with_budget(
    block_offered: usize,
    counts: [usize; 3],
    require_qualified: bool,
    budget: Option<std::num::NonZeroU64>,
    limits: CostProfileLoadLimits,
) -> (
    Vec<u8>,
    StructuredPreparedOwnerBlockCollectorV8,
    StructuredPreparedOwnerBlockCheckpointV8,
) {
    collected_with_budget_and_domain(
        block_offered,
        counts,
        require_qualified,
        budget,
        limits,
        None,
        false,
    )
}
pub(super) fn collected_with_domain(
    domain: ferrum_interfaces::execution_cost::CostMonotonicDomainV1,
) -> (
    Vec<u8>,
    StructuredPreparedOwnerBlockCollectorV8,
    StructuredPreparedOwnerBlockCheckpointV8,
) {
    collected_with_budget_and_domain(
        24,
        [16, 8, 8],
        true,
        None,
        CostProfileLoadLimits::default(),
        Some(domain),
        false,
    )
}

pub(super) fn collected_with_domain_routes(
    domain: ferrum_interfaces::execution_cost::CostMonotonicDomainV1,
) -> (
    Vec<u8>,
    StructuredPreparedOwnerBlockCollectorV8,
    StructuredPreparedOwnerBlockCheckpointV8,
) {
    collected_with_budget_and_domain(
        24,
        [32, 16, 16],
        true,
        None,
        CostProfileLoadLimits::default(),
        Some(domain),
        true,
    )
}

fn collected_with_budget_and_domain(
    block_offered: usize,
    counts: [usize; 3],
    require_qualified: bool,
    budget: Option<std::num::NonZeroU64>,
    limits: CostProfileLoadLimits,
    domain: Option<ferrum_interfaces::execution_cost::CostMonotonicDomainV1>,
    split_routes: bool,
) -> (
    Vec<u8>,
    StructuredPreparedOwnerBlockCollectorV8,
    StructuredPreparedOwnerBlockCheckpointV8,
) {
    let mut h = declared(block_offered, counts);
    if split_routes {
        h.declaration.maximum_offered_waves = counts.iter().sum::<usize>().checked_mul(3).unwrap();
        h.declaration.population.schedule.prediction_validity = Some(
            crate::implementations::continuous::cost_model::structured_v2::OwnerPredictionValidityPolicyV1::OriginalSampleAgeV1,
        );
    }
    if let Some(budget) = budget {
        h = StructuredPreparedOwnerBlockHeaderV8::new(
            h.capture_identity,
            h.generation,
            h.fingerprint,
            h.producer,
            h.opening,
            h.declaration,
            budget.get(),
        )
        .unwrap();
    }

    if let Some(domain) = domain {
        h = StructuredPreparedOwnerBlockHeaderV8::new_with_monotonic_domain(
            h.capture_identity,
            h.generation,
            h.fingerprint,
            h.producer,
            h.opening,
            h.declaration,
            h.maximum_file_bytes,
            domain,
        )
        .unwrap();
    }
    let mut bytes = Vec::new();
    append(&mut bytes, &h);
    let mut c = match budget {
        Some(budget) => {
            StructuredPreparedOwnerBlockCollectorV8::new_streaming(h.clone(), limits, budget)
        }
        None => StructuredPreparedOwnerBlockCollectorV8::new(h.clone(), limits),
    }
    .unwrap();
    let mut ticket = 0u64;
    for phase_index in 0..3 {
        let phase = match phase_index {
            0 => StructuredProfilePhaseV10::Fit,
            1 => StructuredProfilePhaseV10::Residual,
            _ => StructuredProfilePhaseV10::Qualification,
        };
        for cohort in 0..h.declaration.cohort_plan.phases[phase_index].len() {
            let graph = if split_routes && cohort % 2 == 1 {
                ActualWaveGraphState::ConfiguredEager
            } else {
                ActualWaveGraphState::Disabled
            };
            open_for_offer(&mut c, &mut bytes, ticket, block_offered);
            let id = format!("source8-{phase_index}-{cohort}");
            event(
                &mut c,
                &mut bytes,
                serde_json::json!({"kind":"cohort_begin","phase":phase,"cohort":cohort,"manifest_case":0,"repetition":cohort}),
                false,
            );
            event(
                &mut c,
                &mut bytes,
                serde_json::json!({"kind":"request_admitted","phase":phase,"cohort":cohort,"slot":0,"request_id":id,"maximum_output":3}),
                false,
            );
            ticket += 1;
            let p = prepared_on_route(&id, 0, graph);
            let before = serde_json::json!({"request_id":id,"owner_incarnation":1,"work_generation":1,"generated_tokens":0,"kv_tokens":0,"model_cache_id":null,"pending_utf8":[],"output_accepted_ordinal":0});
            event(
                &mut c,
                &mut bytes,
                serde_json::json!({"kind":"preparation_offered","offered":ticket,"phase":phase,"cohort":cohort,"rows":[{"before":before,"work":p.rows[0].frontier.work}]}),
                true,
            );
            let after = serde_json::json!({"request_id":id,"owner_incarnation":1,"work_generation":2,"generated_tokens":1,"kv_tokens":64,"model_cache_id":"cache","pending_utf8":[],"output_accepted_ordinal":1});
            let mut s = old::stages(&old::header(), &p, ticket, 1000);
            s.statistical_evidence = None;
            s.structured_evidence = None;
            event(
                &mut c,
                &mut bytes,
                serde_json::json!({"kind":"preparation_completed","offered":ticket,"phase":phase,"cohort":cohort,"reconciled":true,
                "queue":{"disposition":"published","accepted_ordinal":ticket},"host_stages":s,
                "rows":[{"before":before,"after":after,"preparation_commit":{"request_id":id,"owner_incarnation":1,"work_generation":1,"generated_before":0,"generated_after":1,"original_candidate":9,"committed_token":7,"route":"full_logits_sampler","pending_before":[],"pending_after":[]}}],"failure":null}),
                true,
            );
            let mut hash = Sha256::new();
            hash.update(b"ferrum.calibration.generated-prefix.v1\0");
            hash.update(7u32.to_le_bytes());
            let prefix: [u8; 32] = hash.finalize().into();
            event(
                &mut c,
                &mut bytes,
                serde_json::json!({"kind":"preparation_released","phase":phase,"cohort":cohort,"slot":0,
                "receipt":{"frontier":after,"original_policy_signature":POLICY,"original_numeric_policy":p.recipe.physical_host_rows[0].installed_policy,
                    "generated_prefix_sha256":prefix,"through_call_id":ticket,"through_fifo_ordinal":ticket,"actor_applied_output_ordinal":1}}),
                true,
            );
            for generated in 1..3 {
                open_for_offer(&mut c, &mut bytes, ticket, block_offered);
                ticket += 1;
                let p = prepared_on_route(&id, generated, graph);
                let s = old::stages(&old::header(), &p, ticket, 1000);
                let wave = StructuredServiceWaveV7::from_diagnostic(
                    ticket,
                    ticket * 2000,
                    ticket,
                    serde_json::to_value(&s).unwrap(),
                    None,
                )
                .unwrap();
                let r = StructuredPreparedOwnerBlockRecordV8::Population(
                    StructuredServiceRecordV7::Completed { wave },
                );
                c.push(&r).unwrap();
                append(&mut bytes, &r);
                if let Some(terminal) = s.rows[0].terminal.clone() {
                    event(
                        &mut c,
                        &mut bytes,
                        serde_json::json!({"kind":"request_completed","request":CompletedRequest {
                            phase, cohort, slot:0, request_id:id.clone(), owner_incarnation:1, call_id:ticket, fifo:ticket, generated_tokens:3, terminal,
                        }}),
                        false,
                    );
                }
            }
            event(
                &mut c,
                &mut bytes,
                serde_json::json!({"kind":"cohort_end","phase":phase,"cohort":cohort,"admitted_count":1,"completed_count":1}),
                false,
            );
        }
    }
    assert_eq!(
        (ticket, c.offered(), c.preparation_attempts()),
        (
            (counts.iter().sum::<usize>() * 3) as u64,
            (counts.iter().sum::<usize>() * 3) as u64,
            counts.iter().sum::<usize>() as u64
        )
    );
    let closing = paired(ticket * 2000 + 1101);
    if ticket % block_offered as u64 == 0 {
        let closed = c.close_block(closing).unwrap();
        append(&mut bytes, &closed);
    } else {
        let before = record_bytes_v7(&c.audit().owners).unwrap();
        let sealed = c
            .seal_complete_cohorts_with_partial_tail(closing)
            .unwrap()
            .unwrap();
        append(&mut bytes, &sealed);
        // Closing an audit-only tail must never freeze/advance an owner.
        assert_eq!(before, record_bytes_v7(&c.audit().owners).unwrap());
    }
    if require_qualified {
        assert!(c.qualified_children() > 0, "{:?}", c.audit());
    }
    let (record, checkpoint) = c.checkpoint(closing).unwrap();
    append(&mut bytes, &record);
    (bytes, c, checkpoint)
}

#[test]
fn source8_real_preparation_release_ordinary_terminal_and_three_phases_replay() {
    let (bytes, c, _) = collected();
    assert_eq!(
        c.source_receipt(),
        (bytes.len() as u64, Sha256::digest(&bytes).into())
    );
    let replayed = replay_structured_source_v8(&bytes, &CostProfileLoadLimits::default()).unwrap();
    assert_eq!(replayed.population.source_receipt(), c.source_receipt());
    assert_eq!(
        replayed.population.qualified_children(),
        c.qualified_children()
    );
    assert!(replay_structured_source_v7(&bytes, &CostProfileLoadLimits::default()).is_err());
}
#[test]
fn source8_missing_preparation_release_terminal_or_original_fifo_cannot_checkpoint() {
    let (bytes, _, _) = collected();
    for target in [
        "preparation_completed",
        "preparation_released",
        "request_completed",
        "cohort_end",
    ] {
        let mut changed = false;
        let mut bad = Vec::new();
        for line in bytes.split_inclusive(|b| *b == b'\n') {
            let value: serde_json::Value = serde_json::from_slice(line).unwrap();
            if !changed && value["kind"] == target {
                changed = true;
                continue;
            }
            bad.extend_from_slice(line);
        }
        assert!(changed);
        assert!(
            replay_structured_source_v8(&bad, &CostProfileLoadLimits::default()).is_err(),
            "{target}"
        );
    }
    let mut bad = Vec::new();
    let mut changed = false;
    for line in bytes.split_inclusive(|b| *b == b'\n') {
        let mut v: serde_json::Value = serde_json::from_slice(line).unwrap();
        if !changed && v["kind"] == "preparation_completed" {
            v["queue"]["accepted_ordinal"] = serde_json::json!(999);
            changed = true;
        }
        bad.extend(record_bytes_v7(&v).unwrap());
    }
    assert!(changed);
    assert!(replay_structured_source_v8(&bad, &CostProfileLoadLimits::default()).is_err());
}
