//! Counterfactual candidate using the production collector and an independently
//! replayed candidate journal. Original source, clocks and offers stay intact.
use super::*;
use crate::implementations::continuous::cost_profile::structured_v10::service::owner_blocks::collector::State as OwnerState;
use std::num::NonZeroU64;
mod source8;

fn candidate_header(original: &StructuredServiceHeaderV7) -> StructuredServiceHeaderV7 {
    let mut declaration = original.declaration.clone();
    declaration
        .nonnegative_envelope
        .as_mut()
        .unwrap()
        .planning_estimator = NonNegativePlanningEstimatorV1::IdentifiedFitGlobalResidualV1;
    let readiness = declaration.schedule.input_readiness.as_ref().unwrap();
    declaration.schedule.input_readiness = Some(
        OwnerInputReadinessV1::new_fit_target_v4(
            readiness.maximum_phase_blocks,
            readiness.maximum_geometry_visits,
        )
        .unwrap(),
    );
    let mut id = Sha256::new();
    id.update(b"ferrum.offline.production-global-residual.v1\0");
    id.update(original.capture_identity);
    let h = StructuredServiceHeaderV7::new(
        id.finalize().into(),
        original.generation,
        original.fingerprint.clone(),
        original.producer.clone(),
        original.opening,
        declaration,
        original.maximum_file_bytes,
    )
    .unwrap();
    match &original.monotonic_domain {
        Some(domain) => h.with_monotonic_domain(domain.clone()).unwrap(),
        None => h,
    }
}

#[test]
#[ignore = "requires immutable original source7 and paired query journal via FERRUM_ARCHIVED_BOUND_SPEC"]
fn replay_original_source7_through_production_global_residual() {
    let spec: AuditSpec = serde_json::from_slice(
        &std::fs::read(std::env::var_os("FERRUM_ARCHIVED_BOUND_SPEC").unwrap()).unwrap(),
    )
    .unwrap();
    let original = seeded::verify_source_file(&spec.actual_source, spec.actual_source_sha256);
    let header = candidate_header(&original);
    let budget = NonZeroU64::new(header.maximum_file_bytes).unwrap();
    let limits = CostProfileLoadLimits::default();
    let mut archive =
        StructuredServiceCollectorV7::new_streaming(original.clone(), limits.clone(), budget)
            .unwrap();
    let mut candidate =
        StructuredServiceCollectorV7::new_streaming(header.clone(), limits.clone(), budget)
            .unwrap();
    let mut replay =
        StructuredServiceCollectorV7::new_streaming(header.clone(), limits.clone(), budget)
            .unwrap();
    let mut physical_contract = original.declaration.nonnegative_envelope.clone().unwrap();
    if original.declaration.schedule.algorithm_universe
        == Some(OwnerAlgorithmUniversePolicyV1::SeededFirstOrdinaryDiscoveryBlockSubsetV1)
    {
        physical_contract.algorithm_universe = None;
    }
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
    let mut opened = original.opening.monotonic_ns;
    let mut predictions = Vec::new();
    let mut known_issued = BTreeSet::new();
    let mut verified = BTreeSet::new();
    let mut unknown = BTreeMap::<String, usize>::new();
    let mut freezes = Vec::new();
    let mut installed = None;
    let mut snapshot_verified = false;
    let mut original_fit_certificate_verified = false;
    let mut rows = BTreeMap::<u32, RowStats>::new();
    let mut paired = Vec::new();
    let mut phase_memory = Vec::new();
    for line in BufReader::new(std::fs::File::open(&spec.actual_source).unwrap())
        .lines()
        .skip(1)
    {
        let record: StructuredServiceRecordV7 = serde_json::from_str(&line.unwrap()).unwrap();
        archive.push(&record).unwrap();
        let generated = match &record {
            StructuredServiceRecordV7::BlockOpen {
                opened_at_ns,
                fifo_cutoff,
                ..
            } => {
                opened = *opened_at_ns;
                candidate.open_block(*opened_at_ns, *fifo_cutoff).unwrap()
            }
            StructuredServiceRecordV7::BlockClose { block, closing, .. } => {
                let before_bytes = candidate.audit().retained_numeric_bytes;
                let generated = candidate.close_block(*closing).unwrap();
                phase_memory.push(json!({"block":block,"before_close_numeric_bytes":before_bytes,
                    "after_close_numeric_bytes":candidate.audit().retained_numeric_bytes,
                    "remaining_phase_sample_counts":candidate.owners.iter().map(|o|o.samples.len()).collect::<Vec<_>>() }));
                if let StructuredServiceRecordV7::BlockClose {
                    freezes: values, ..
                } = &generated
                {
                    for freeze in values {
                        let owner = candidate
                            .owners
                            .iter()
                            .find(|o| o.contract.owner_attempt_id == freeze.owner_attempt_id)
                            .unwrap();
                        if let OwnerState::Fitted(model) = &owner.state {
                            if owner.scope.owner.role == StructuredWaveRoleV2::OrdinaryDecode
                                && owner.scope.owner.product == StructuredProductV2::GreedyToken
                                && owner.scope.numerical_family.is_some()
                            {
                                let old = archive
                                    .owners
                                    .iter()
                                    .find(|o| {
                                        o.contract.owner_attempt_id
                                            == owner.contract.owner_attempt_id
                                    })
                                    .unwrap();
                                let OwnerState::Fitted(old) = &old.state else {
                                    panic!("same original Fit boundary changed")
                                };
                                assert_eq!(
                                    serde_json::to_value(model.nonnegative_fit_certificate())
                                        .unwrap(),
                                    serde_json::to_value(old.nonnegative_fit_certificate())
                                        .unwrap()
                                );
                                original_fit_certificate_verified = true;
                            }
                        }
                        assert!(
                            owner.samples.is_empty(),
                            "closed phase samples must be released"
                        );
                        freezes.push(
                            json!({"block":block,"owner_attempt":freeze.owner_attempt_id,
                            "phase":freeze.close.phase,"members":freeze.close.member_count,
                            "failure":freeze.failure,"released_samples":owner.samples.is_empty()}),
                        );
                    }
                }
                if installed.is_none() {
                    installed = candidate.owners.iter().find_map(|o| match &o.state {
                        OwnerState::Qualified(m)
                            if m.owner().role == StructuredWaveRoleV2::OrdinaryDecode
                                && m.owner().product == StructuredProductV2::GreedyToken
                                && m.numerical_family_key().is_some() =>
                        {
                            Some(m.clone())
                        }
                        _ => None,
                    });
                }
                generated
            }
            StructuredServiceRecordV7::Completed { wave } => {
                let call = wave.host_stages.call_id;
                if installed.is_some() || demands.contains_key(&call) {
                    let (input, wall, _) = physical::validate_parts(
                        &original.fingerprint,
                        original.opening.monotonic_ns,
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
                        verified.insert(call);
                    }
                    if let Some(model) = &installed {
                        let stats = rows.entry(input.owner().rows).or_default();
                        stats.total += 1;
                        match model.predict_query(
                            &header.fingerprint.clone().into(),
                            &query,
                            wave.issued_at_ns,
                        ) {
                            Ok(p) => {
                                stats.known += 1;
                                stats.misses += usize::from(p.planning_ns < wall);
                                stats.maximum_under_ns = stats
                                    .maximum_under_ns
                                    .max(wall.saturating_sub(p.planning_ns));
                                predictions.push((call, p.planning_ns, wall));
                                if demands.contains_key(&call) {
                                    known_issued.insert(call);
                                    paired.push(json!({"call":call,"phase":"future","demand_verified":true,"known":true,
                                        "point_ns":p.fitted_upper_ns,"planning_ns":p.planning_ns,"wall_ns":wall,
                                        "valid_until_ns":p.valid_until_ns,"original_issue_ns":wave.issued_at_ns}));
                                }
                            }
                            Err(e) => {
                                *unknown.entry(format!("{e:?}")).or_default() += 1;
                                *stats.unknown.entry(format!("{e:?}")).or_default() += 1;
                                if demands.contains_key(&call) {
                                    paired.push(json!({"call":call,"phase":"future","demand_verified":true,"known":false,"reason":format!("{e:?}")}));
                                }
                            }
                        }
                    } else if demands.contains_key(&call) {
                        let phase = candidate
                            .owners
                            .iter()
                            .find(|o| {
                                o.scope.owner.role == StructuredWaveRoleV2::OrdinaryDecode
                                    && o.scope.owner.product == StructuredProductV2::GreedyToken
                                    && o.scope.numerical_family.is_some()
                            })
                            .and_then(|o| match &o.state {
                                OwnerState::Empty => Some(StructuredPhaseV2::Fit),
                                OwnerState::Fitted(_) => Some(StructuredPhaseV2::Residual),
                                OwnerState::Calibrated(_) => Some(StructuredPhaseV2::Qualification),
                                _ => None,
                            });
                        paired.push(
                            json!({"call":call,"phase":phase,"demand_verified":true,"future":false,
                            "reason":"not yet published; remains an original phase sample"}),
                        );
                    }
                }
                candidate.push(&record).unwrap();
                record.clone()
            }
            StructuredServiceRecordV7::OutsideDeclaredRoute { .. }
            | StructuredServiceRecordV7::NotSubmitted { .. } => {
                candidate.push(&record).unwrap();
                record.clone()
            }
            StructuredServiceRecordV7::Checkpoint { .. } => continue,
            StructuredServiceRecordV7::Failed {
                ticket,
                fifo,
                at_ns,
                reason,
                ..
            } => candidate
                .fail(*ticket, *fifo, *at_ns, reason.clone())
                .unwrap(),
            StructuredServiceRecordV7::Footer { closing, .. } => candidate.stop(*closing).unwrap(),
        };
        replay.push(&generated).unwrap();
        assert_eq!(candidate.source_receipt(), replay.source_receipt());
        if installed.is_some() && !snapshot_verified {
            if let StructuredServiceRecordV7::BlockClose { closing, .. } = generated {
                let (record, snapshot) = candidate.checkpoint(closing).unwrap();
                replay.push(&record).unwrap();
                let independent =
                    StructuredServiceCheckpointV7::from_collector(&replay, closing).unwrap();
                let a = snapshot
                    .activate_same_process_memory_streaming(closing, &limits, budget)
                    .unwrap();
                let b = independent
                    .activate_same_process_memory_streaming(closing, &limits, budget)
                    .unwrap();
                assert_eq!(a.source_sha256, b.source_sha256);
                assert_eq!(
                    a.children
                        .iter()
                        .map(|m| m.parameters_signature())
                        .collect::<Vec<_>>(),
                    b.children
                        .iter()
                        .map(|m| m.parameters_signature())
                        .collect::<Vec<_>>()
                );
                snapshot_verified = true;
            }
        }
    }
    let misses = predictions.iter().filter(|(_, p, w)| p < w).count();
    let mut bounds: Vec<_> = predictions.iter().map(|(_, p, _)| *p).collect();
    bounds.sort_unstable();
    eprintln!(
        "PRODUCTION_GLOBAL_RESIDUAL_SOURCE7 {}",
        json!({"scope":"original-source counterfactual; production collector and independent full candidate replay; no live adoption",
        "snapshot_verified":snapshot_verified,"original_identified_fit_certificate_verified":original_fit_certificate_verified,
        "known_future_all_rows":predictions.len(),"future_misses":misses,
        "future_by_rows":rows.iter().map(|(r,c)|json!({"rows":r,"total":c.total,"known":c.known,"misses":c.misses,
            "maximum_under_ns":c.maximum_under_ns,"unknown":c.unknown})).collect::<Vec<_>>(),
        "paired_actual_demands":paired,"paired_demands_verified":verified,"phase_memory":phase_memory,
        "known_verified_issued_calls":known_issued,"unknown_after_qualification":unknown,
        "planning_ns":{"min":bounds.first(),"p50":bounds.get(bounds.len()/2),"max":bounds.last()},
        "freezes":freezes,"peak_collector_reserved_bytes":candidate.reserved_peak_for_tests(),
        "audit":candidate.audit(),"memory_scope":"production collector reservation/accounting; closed phase sample vectors released; audit/replay process RSS separate"})
    );
    assert!(
        snapshot_verified,
        "production source must actually qualify and import"
    );
    assert!(
        !known_issued.is_empty(),
        "at least one original paired issued demand must be Known"
    );
    assert!(original_fit_certificate_verified);
    assert_eq!(verified.len(), demands.len());
    assert_eq!(misses, 0);
}

#[derive(Default)]
struct RowStats {
    total: usize,
    known: usize,
    misses: usize,
    maximum_under_ns: u64,
    unknown: BTreeMap<String, usize>,
}
