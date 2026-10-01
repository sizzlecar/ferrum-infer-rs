//! Real canonical commands -> original source7 blocks -> independent replay.
//! Synthetic walls exercise the protocol, not measured backend performance.
use super::*;
use crate::implementations::continuous::cost_model::structured_v2::{
    OwnerAlgorithmUniversePolicyV1, OwnerInputReadinessV1, StructuredUnknownV2,
};
use ferrum_interfaces::execution_cost::ActualWaveGraphState;
mod cold_seed;
#[cfg(test)]
mod memory_accounting;
mod phase_support;

const A: &str = "fixture.universe.a";
const B: &str = "fixture.universe.b";
const C: &str = "fixture.universe.c";

fn rebuild(h: StructuredServiceHeaderV7) -> Result<StructuredServiceHeaderV7, CostProfileError> {
    StructuredServiceHeaderV7::new(
        h.capture_identity,
        h.generation,
        h.fingerprint,
        h.producer,
        h.opening,
        h.declaration,
        h.maximum_file_bytes,
    )
}
fn header() -> StructuredServiceHeaderV7 {
    let mut h = block_header();
    h.declaration.schedule = OwnerBlockScheduleV1::new_with_input_readiness(
        8,
        [8; 3],
        [8; 3],
        OwnerInputReadinessV1::new([2; 3], 32_000_000).unwrap(),
    )
    .unwrap();
    h.declaration.settings.max_phase_samples = h.declaration.settings.max_phase_samples.max(16);
    h.declaration.schedule.algorithm_universe =
        Some(OwnerAlgorithmUniversePolicyV1::FirstOrdinaryDiscoveryBlockSubsetV1);
    h.declaration
        .nonnegative_envelope
        .as_mut()
        .unwrap()
        .population_policy = StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1;
    rebuild(h).unwrap()
}
fn record(ticket: u64, algorithms: &[&str], generated: u64) -> StructuredServiceRecordV7 {
    let id = format!("universe-{ticket}");
    let (p, _, _, _) = old::prepared_batch_algorithms_and_future(
        &[&id],
        generated + 1,
        generated,
        None,
        ActualWaveGraphState::Disabled,
        Some([44; 32]),
        64,
        &[4],
        None,
        algorithms,
    );
    let stages = old::stages(&old::header(), &p, ticket, 1_000);
    StructuredServiceRecordV7::Completed {
        wave: StructuredServiceWaveV7::from_diagnostic(
            ticket,
            ticket * 2_000,
            ticket * 3,
            serde_json::to_value(stages).unwrap(),
            None,
        )
        .unwrap(),
    }
}
fn query(algorithms: &[&str]) -> StructuredQueryV2 {
    old::prepared_batch_algorithms_and_future(
        &["independent-future"],
        2,
        1,
        None,
        ActualWaveGraphState::Disabled,
        Some([44; 32]),
        64,
        &[4],
        Some(&domain()),
        algorithms,
    )
    .3
    .unwrap()
}
fn block(
    collector: &mut StructuredServiceCollectorV7,
    bytes: &mut Vec<u8>,
    block: u64,
    roster: impl Fn(u64) -> Vec<&'static str>,
    generated: u64,
) -> StructuredServiceRecordV7 {
    let first = (block - 1) * 8 + 1;
    append(
        bytes,
        &collector
            .open_block(first * 2_000 - 1, (first - 1) * 3)
            .unwrap(),
    );
    for ticket in first..first + 8 {
        let r = record(ticket, &roster(ticket), generated);
        collector.push(&r).unwrap();
        append(bytes, &r);
    }
    let r = collector
        .close_block(paired((first + 7) * 2_000 + 1_101))
        .unwrap();
    append(bytes, &r);
    r
}
fn alternating(ticket: u64) -> Vec<&'static str> {
    if ticket % 2 == 0 {
        vec![A]
    } else {
        vec![B]
    }
}

#[test]
fn source7_algorithm_universe_discovery_a_b_heldout_ab_and_unknown_c_replay() {
    let h = header();
    let mut bytes = record_bytes_v7(&h).unwrap();
    let mut collector =
        StructuredServiceCollectorV7::new(h.clone(), CostProfileLoadLimits::default()).unwrap();
    for n in 1..=5 {
        let r = block(
            &mut collector,
            &mut bytes,
            n,
            |ticket| match n {
                1 | 2 => alternating(ticket),
                3 | 4 => {
                    if ticket % 2 == 0 {
                        vec![A, B]
                    } else {
                        vec![B, A]
                    }
                }
                _ => vec![C],
            },
            1,
        );
        let StructuredServiceRecordV7::BlockClose {
            discoveries,
            freezes,
            ..
        } = r
        else {
            unreachable!()
        };
        if n == 1 {
            assert!(freezes.is_empty(), "Discovery cannot supply Fit members");
            assert_eq!(
                discoveries.len(),
                1,
                "A and B must share the frozen numeric layout"
            );
            let discovered = &discoveries[0];
            assert_eq!(
                discovered
                    .contract
                    .nonnegative_envelope
                    .as_ref()
                    .unwrap()
                    .algorithm_universe
                    .as_ref()
                    .unwrap()
                    .algorithm_count(),
                2
            );
            assert!(discovered.contract.input_target.is_some());
            assert!(discovered.scope.numerical_family.is_some());
        } else if n <= 4 {
            assert!(discoveries.is_empty());
            assert_eq!(freezes.len(), 1);
            assert_eq!(freezes[0].failure, None, "{:?}", freezes[0]);
            assert_eq!(freezes[0].close.member_count, 8);
            assert_eq!(freezes[0].close.boundary.first_offered, (n - 1) * 8 + 1);
            assert_eq!(freezes[0].close.phase, phase_at((n - 2) as usize));
        } else {
            assert!(
                discoveries.is_empty(),
                "C cannot grow the generation universe"
            );
            assert!(freezes.is_empty());
        }
    }
    assert_eq!(
        collector.offered(),
        40,
        "unknown C still consumes every original offer"
    );
    assert_eq!(collector.audit().owners.len(), 1);
    assert_eq!(collector.qualified_children(), 1);
    let (r, original) = collector.checkpoint(paired(81_101)).unwrap();
    append(&mut bytes, &r);
    let limits = CostProfileLoadLimits::default();
    let original = original
        .activate_same_process_memory(paired(90_000), &limits)
        .unwrap();
    let replay = replay_structured_source_v7(&bytes, &limits)
        .unwrap()
        .activate_same_process_memory(paired(90_000), &limits)
        .unwrap();
    assert_eq!(original.source_sha256, replay.source_sha256);
    assert_eq!(original.total_shape_rows, 40);
    assert_eq!(
        original.children[0].parameters_signature(),
        replay.children[0].parameters_signature()
    );
    assert_eq!(
        replay.children[0]
            .provenance()
            .phases
            .each_ref()
            .map(|p| p.members),
        [8; 3]
    );
    for roster in [vec![A], vec![B], vec![A, B], vec![B, A]] {
        let q = query(&roster);
        let a = original.children[0]
            .predict_query_local(&old::fingerprint(), &q, 90_000)
            .unwrap();
        let b = replay.children[0]
            .predict_query_local(&old::fingerprint(), &q, 90_000)
            .unwrap();
        assert_eq!(a.planning_ns, b.planning_ns);
    }
    assert!(matches!(
        original.children[0].predict_query_local(&old::fingerprint(), &query(&[C]), 90_000),
        Err(StructuredUnknownV2::WrongDomain)
    ));

    // Replay derives U from the actual Discovery records before trusting this
    // close. A syntactically valid, smaller claimed subset is insufficient.
    let mut records: Vec<serde_json::Value> = bytes
        .split(|b| *b == b'\n')
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_slice(l).unwrap())
        .collect();
    let mut changed = false;
    for value in &mut records {
        if let Some(algorithms) = value
            .pointer_mut(
                "/discoveries/0/contract/nonnegative_envelope/algorithm_universe/algorithms",
            )
            .and_then(serde_json::Value::as_array_mut)
        {
            algorithms.pop();
            changed = true;
            break;
        }
    }
    assert!(changed);
    let mut altered = Vec::new();
    for value in records {
        append(&mut altered, &value);
    }
    let error = match replay_structured_source_v7(&altered, &limits) {
        Err(error) => error,
        Ok(_) => panic!("claimed U bypassed original Discovery"),
    };
    assert!(
        error
            .to_string()
            .contains("original block/freeze/certificate"),
        "{error}"
    );

    // An unknown primitive cannot conceal missing original settlement. This
    // must fail before the input-only exclusion rule can consume the offer.
    let mut c = StructuredServiceCollectorV7::new(h, limits).unwrap();
    let mut first_bytes = Vec::new();
    block(&mut c, &mut first_bytes, 1, alternating, 1);
    c.open_block(17_999, 24).unwrap();
    let mut invalid = serde_json::to_value(record(9, &[C], 1)).unwrap();
    invalid["wave"]["host_stages"]["structured_evidence"] = serde_json::Value::Null;
    let invalid = serde_json::from_value::<StructuredServiceRecordV7>(invalid).unwrap();
    assert!(c.push(&invalid).is_err());
    assert_eq!(c.offered(), 8);
    assert!(c.audit().poisoned);
}

#[test]
fn source7_algorithm_universe_prefill_then_first_ordinary_discovery_qualifies_and_replays() {
    let h = header();
    let mut bytes = record_bytes_v7(&h).unwrap();
    let mut c = StructuredServiceCollectorV7::new(h, CostProfileLoadLimits::default()).unwrap();
    let first = block(&mut c, &mut bytes, 1, |_| vec![A], 0);
    let StructuredServiceRecordV7::BlockClose { discoveries, .. } = first else {
        unreachable!()
    };
    assert_eq!(discoveries.len(), 1);
    assert!(discoveries[0].scope.numerical_family.is_none());
    assert!(discoveries[0]
        .contract
        .nonnegative_envelope
        .as_ref()
        .unwrap()
        .algorithm_universe
        .is_none());
    let second = block(&mut c, &mut bytes, 2, alternating, 1);
    let StructuredServiceRecordV7::BlockClose {
        discoveries,
        freezes,
        ..
    } = second
    else {
        unreachable!()
    };
    assert!(freezes.is_empty());
    assert_eq!(discoveries.len(), 1);
    let ordinary = &discoveries[0];
    let attempt = ordinary.owner_attempt_id;
    assert_eq!(ordinary.contract.discovery_block, 2);
    assert_eq!(ordinary.contract.discovery_offered_cutoff, 16);
    assert_eq!(
        ordinary
            .contract
            .nonnegative_envelope
            .as_ref()
            .unwrap()
            .algorithm_universe
            .as_ref()
            .unwrap()
            .algorithm_count(),
        2
    );
    for n in 3..=5 {
        let r = block(
            &mut c,
            &mut bytes,
            n,
            |ticket| {
                if n == 3 {
                    alternating(ticket)
                } else {
                    vec![A, B]
                }
            },
            1,
        );
        let StructuredServiceRecordV7::BlockClose {
            discoveries,
            freezes,
            ..
        } = r
        else {
            unreachable!()
        };
        assert!(discoveries.is_empty());
        let freeze = freezes
            .iter()
            .find(|f| f.owner_attempt_id == attempt)
            .unwrap();
        assert_eq!(freeze.failure, None, "{freeze:?}");
        assert_eq!(freeze.close.member_count, 8);
        assert_eq!(freeze.close.boundary.first_offered, (n - 1) * 8 + 1);
        assert_eq!(freeze.close.phase, phase_at((n - 3) as usize));
    }
    assert_eq!(
        c.offered(),
        40,
        "the initial Prefill quota is never reset or reused"
    );
    assert_eq!(c.audit().owners.len(), 2);
    assert_eq!(c.qualified_children(), 1);
    let (r, checkpoint) = c.checkpoint(paired(81_101)).unwrap();
    append(&mut bytes, &r);
    let limits = CostProfileLoadLimits::default();
    let live = checkpoint
        .activate_same_process_memory(paired(90_000), &limits)
        .unwrap();
    let replay = replay_structured_source_v7(&bytes, &limits)
        .unwrap()
        .activate_same_process_memory(paired(90_000), &limits)
        .unwrap();
    assert_eq!(live.source_sha256, replay.source_sha256);
    assert_eq!(
        live.children[0].parameters_signature(),
        replay.children[0].parameters_signature()
    );
    assert_eq!(
        live.children[0]
            .provenance()
            .phases
            .each_ref()
            .map(|p| p.members),
        [8; 3]
    );
    let q = query(&[B, A]);
    assert_eq!(
        live.children[0]
            .predict_query_local(&old::fingerprint(), &q, 90_000)
            .unwrap()
            .planning_ns,
        replay.children[0]
            .predict_query_local(&old::fingerprint(), &q, 90_000)
            .unwrap()
            .planning_ns
    );
}

#[test]
fn source7_algorithm_universe_policy_and_discovery_reservation_are_explicit() {
    let legacy = block_header();
    assert!(serde_json::to_value(&legacy.declaration.schedule)
        .unwrap()
        .get("algorithm_universe")
        .is_none());
    let mut invalid = header();
    invalid
        .declaration
        .nonnegative_envelope
        .as_mut()
        .unwrap()
        .population_policy = StructuredPopulationPolicyV1::ExactOwnerV1;
    assert!(rebuild(invalid).is_err());
    let h = header();
    let mut c =
        StructuredServiceCollectorV7::new(h.clone(), CostProfileLoadLimits::default()).unwrap();
    c.open_block(1_999, 0).unwrap();
    let reserved = c.audit().retained_numeric_bytes;
    assert!(reserved >= h.declaration.maximum_discovery_bytes);
    let mut bounded = h;
    bounded.declaration.maximum_retained_numeric_bytes = reserved - 1;
    let mut c = StructuredServiceCollectorV7::new(
        rebuild(bounded).unwrap(),
        CostProfileLoadLimits::default(),
    )
    .unwrap();
    assert!(
        c.open_block(1_999, 0).is_err(),
        "shared limit checked before representative cloning"
    );
    assert_eq!(c.offered(), 0);
}
