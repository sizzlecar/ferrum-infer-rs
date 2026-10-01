//! Same-source Fit reuse feasibility. Every fit is frozen before these R/Q
//! cells receive a timing label; the experiment does not publish live models.
use super::*;
use crate::implementations::continuous::cost_profile::structured_v10::service::owner_blocks::collector::State as OwnerState;
use std::num::NonZeroU64;

#[test]
#[ignore = "requires immutable source7 and original issued demands via FERRUM_ARCHIVED_BOUND_SPEC"]
fn replay_same_source_fit_with_frozen_global_joint_bank() {
    let spec: AuditSpec = serde_json::from_slice(
        &std::fs::read(std::env::var_os("FERRUM_ARCHIVED_BOUND_SPEC").unwrap()).unwrap(),
    )
    .unwrap();
    let header = seeded::verify_source_file(&spec.actual_source, spec.actual_source_sha256);
    let d = &header.declaration;
    let mut physical_contract = d.nonnegative_envelope.clone().unwrap();
    if d.schedule.algorithm_universe
        == Some(OwnerAlgorithmUniversePolicyV1::SeededFirstOrdinaryDiscoveryBlockSubsetV1)
    {
        physical_contract.algorithm_universe = None;
    }
    let mut collector = StructuredServiceCollectorV7::new_streaming(
        header.clone(),
        CostProfileLoadLimits::default(),
        NonZeroU64::new(header.maximum_file_bytes).unwrap(),
    )
    .unwrap();
    let mut archive_validator = StructuredServiceCollectorV7::new_streaming(
        header.clone(),
        CostProfileLoadLimits::default(),
        NonZeroU64::new(header.maximum_file_bytes).unwrap(),
    )
    .unwrap();
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
    let mut run: Option<GlobalBank> = None;
    let mut selected = None;
    let mut fitted_at_block = 0;
    let mut fitted_bytes = 0;
    let mut block = 0;
    let mut block_opened = header.opening.monotonic_ns;
    let mut closed_offered = 0;
    let mut total_b1 = 0;
    let mut compatible = 0;
    let mut unknown = 0;
    let mut calls_verified = BTreeSet::new();
    let mut history_by_call = BTreeMap::new();

    for line in BufReader::new(std::fs::File::open(&spec.actual_source).unwrap())
        .lines()
        .skip(1)
    {
        let record: StructuredServiceRecordV7 = serde_json::from_str(&line.unwrap()).unwrap();
        // The independent original collector checks every FIFO/frontier and
        // protocol transition even after the point model is frozen below.
        archive_validator.push(&record).unwrap();
        // Stop advancing at the earliest original completed Fit for ordinary
        // greedy decode. Never select a fit by its later R/Q or prediction error.
        if selected.is_none() {
            collector.push(&record).unwrap();
            if let StructuredServiceRecordV7::BlockClose { block: b, .. } = &record {
                if let Some((index, owner)) = collector.owners.iter().enumerate().find(|(_, o)| {
                    o.scope.owner.role == StructuredWaveRoleV2::OrdinaryDecode
                        && o.scope.owner.product == StructuredProductV2::GreedyToken
                        && o.scope.numerical_family.is_some()
                        && matches!(o.state, OwnerState::Fitted(_))
                }) {
                    let OwnerState::Fitted(fit) = &owner.state else {
                        unreachable!()
                    };
                    let (floor, expiry) = fit.diagnose_empirical_fit_limits();
                    fitted_bytes = fit.retained_payload_bytes().unwrap();
                    selected = Some(index);
                    fitted_at_block = *b;
                    run = Some(GlobalBank {
                        phase: 1,
                        first_phase_block: b + 1,
                        inner: Cells {
                            policy: Coordinates::AllInput,
                            slots: BTreeMap::new(),
                            original_fit_floor: floor,
                            settings: d.settings.clone(),
                            schedule: d.schedule.clone(),
                            maximum_slots: d.maximum_owners,
                            maximum_bytes: d
                                .maximum_retained_numeric_bytes
                                .checked_sub(fitted_bytes)
                                .unwrap(),
                            maximum_visits: physical_contract.settings.maximum_coordinate_visits,
                            visits: 0,
                            peak_bytes: 0,
                            predictions: Vec::new(),
                            transitions: Vec::new(),
                            refused_capacity: 0,
                        },
                    });
                    eprintln!(
                        "SAME_SOURCE_FIT {}",
                        json!({
                            "block":b,"owner_attempt":owner.contract.owner_attempt_id,
                            "parameters_sha256":fit.parameters_signature(),
                            "original_expiry_ns":expiry,"fit_floor_ns":floor,"fit_bytes":fitted_bytes
                        })
                    );
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
                block_opened = opened_at_ns;
            }
            StructuredServiceRecordV7::BlockClose {
                block: b,
                offered,
                closing,
                ..
            } => {
                assert_eq!(b, block);
                assert_eq!(offered - closed_offered, d.schedule.block_offered as u64);
                if let Some(run) = run.as_mut().filter(|_| b > fitted_at_block) {
                    run.close(b, closing.monotonic_ns);
                }
                closed_offered = offered;
            }
            StructuredServiceRecordV7::Completed { wave } => {
                let (input, wall, _) = physical::validate_parts(
                    &header.fingerprint,
                    header.opening.monotonic_ns,
                    Some(&physical_contract),
                    block_opened,
                    wave.ticket,
                    wave.fifo,
                    wave.issued_at_ns,
                    &wave.host_stages,
                    wave.independent.as_ref(),
                    &mut physical::Frontiers::default(),
                )
                .unwrap_or_else(|reason| {
                    panic!(
                        "original physical validation call={} ticket={} block={block}: {reason:?}",
                        wave.host_stages.call_id, wave.ticket
                    )
                });
                let numeric = wave
                    .host_stages
                    .actual_shape
                    .as_ref()
                    .unwrap()
                    .numeric_features
                    .as_ref()
                    .unwrap();
                if numeric.rows.len() != 1 {
                    continue;
                }
                total_b1 += 1;
                if selected.is_none() || block <= fitted_at_block {
                    continue;
                }
                // Physical validation used settled facts. Forecast equivalence
                // is checked using the original prospective input projection.
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
                let call = wave.host_stages.call_id;
                if let Some(demand) = demands.get(&call) {
                    assert_eq!(
                        serde_json::to_value(query.required_coverage().unwrap()).unwrap(),
                        *demand
                    );
                    calls_verified.insert(call);
                }
                let OwnerState::Fitted(fit) = &collector.owners[selected.unwrap()].state else {
                    unreachable!("immutable original fit cannot advance after selection");
                };
                let Some((point, raw, work)) =
                    fit.diagnose_empirical_fitted_input(&query, wave.issued_at_ns)
                else {
                    unknown += 1;
                    continue;
                };
                compatible += 1;
                let history = numeric.rows[0].sampling_history_tokens;
                history_by_call.insert(call, history);
                let run = run.as_mut().unwrap();
                let cell = run.policy.cell(&raw, &work);
                run.offer(cell, block, wave.issued_at_ns, call, history, point, wall);
                // Keep original actual shape validation in the audit; no
                // replacement of its work by a cheaper forecast is allowed.
                assert_eq!(input.owner().rows, 1);
            }
            StructuredServiceRecordV7::Failed { .. } | StructuredServiceRecordV7::Footer { .. } => {
                break
            }
            _ => {}
        }
    }
    let run = run.expect("ordinary greedy original Fit");
    let known_issued: Vec<_> = run
        .predictions
        .iter()
        .filter(|(c, _, _)| calls_verified.contains(c))
        .map(|(c, _, _)| *c)
        .collect();
    let misses = run.predictions.iter().filter(|(_, p, w)| p < w).count();
    let mut bounds: Vec<_> = run.predictions.iter().map(|(_, p, _)| *p).collect();
    bounds.sort_unstable();
    eprintln!(
        "SAME_SOURCE_GLOBAL_BANK {}",
        json!({
            "fit_block":fitted_at_block,"last_open_block":block,
            "physical_b1":total_b1,"compatible_after_fit":compatible,"unknown_after_fit":unknown,
            "known_future":run.predictions.len(),"future_misses":misses,"known_verified_issued_calls":known_issued,
            "planning_ns":if bounds.is_empty(){None}else{Some([bounds[0],bounds[bounds.len()/2],*bounds.last().unwrap()])},
            "transitions":run.transitions,"future_same_call_results":run.predictions,
            "peak_model_and_cell_bytes":fitted_bytes+run.peak_bytes,
            "capacity_refusals":run.refused_capacity,
            "scope":"same original source7 Fit and clocks; one input-stopped global Residual bank and later independent global Qualification; counterfactual only, no live enrollment",
            "budget_scope":"frozen Fit plus cells; original archive collector/projection/audit excluded",
        })
    );
    assert!(
        !known_issued.is_empty(),
        "same-source proposal must cover unchanged original issued demand"
    );
    assert_eq!(
        misses, 0,
        "independent future waves must not be underestimated"
    );
    assert_eq!(run.refused_capacity, 0);
    let histories: BTreeSet<_> = run
        .predictions
        .iter()
        .map(|(c, _, _)| history_by_call[c])
        .collect();
    assert!(histories.first() < histories.last());
}

/// One original Fit, one input-stopped R boundary, one input-stopped Q boundary.
/// Counts and keys choose both boundaries before any margin/error is examined.
struct GlobalBank {
    inner: Cells,
    phase: usize,
    first_phase_block: u64,
}
impl std::ops::Deref for GlobalBank {
    type Target = Cells;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl GlobalBank {
    fn offer(
        &mut self,
        cell: Cell,
        block: u64,
        now: u64,
        call: u64,
        history: u64,
        point: u64,
        wall: u64,
    ) {
        // R declares the entire partition before its first input. Once R closes,
        // neither Q labels nor later inputs can add another margin to this bank.
        if self.phase != 1 && !self.inner.slots.contains_key(&cell) {
            return;
        }
        self.inner
            .offer(cell.clone(), block, now, call, history, point, wall);
        if self.phase == 1 {
            if let Some(slot) = self.inner.slots.get_mut(&cell) {
                slot.phase_block = self.first_phase_block;
            }
        }
    }
    fn close(&mut self, block: u64, at_ns: u64) {
        if self.phase > 2 {
            return;
        }
        let blocks = block.checked_sub(self.first_phase_block).unwrap() + 1;
        let offered = (blocks as usize)
            .checked_mul(self.inner.schedule.block_offered)
            .unwrap();
        let maximum_blocks = self
            .inner
            .schedule
            .input_readiness
            .as_ref()
            .unwrap()
            .maximum_phase_blocks[self.phase];
        let ready = offered >= self.inner.schedule.phase_min_offered[self.phase]
            && self
                .inner
                .slots
                .values()
                .any(|slot| match (&slot.phase, self.phase) {
                    (Phase::Residual { errors }, 1) => {
                        errors.len() >= self.inner.schedule.min_members[1]
                    }
                    (Phase::Qualification { members, .. }, 2) => {
                        *members >= self.inner.schedule.min_members[2]
                    }
                    _ => false,
                });
        if !ready {
            if blocks as usize >= maximum_blocks {
                for slot in self.inner.slots.values_mut() {
                    slot.phase = Phase::Rejected;
                }
                self.phase = 3;
            }
            return;
        }
        // Existing numerical transition is reached exactly once at this
        // complete global input boundary, including when all ready cells fail.
        self.inner.close(block, at_ns);
        for slot in self.inner.slots.values_mut() {
            let keep = match self.phase {
                1 => matches!(slot.phase, Phase::Qualification { .. }),
                2 => matches!(slot.phase, Phase::Qualified { .. }),
                _ => unreachable!(),
            };
            if !keep {
                slot.phase = Phase::Rejected;
            }
        }
        self.phase += 1;
        self.first_phase_block = block + 1;
    }
}
