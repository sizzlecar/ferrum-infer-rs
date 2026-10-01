//! Same-source Fit reuse feasibility. Every fit is frozen before these R/Q
//! cells receive a timing label; the experiment does not publish live models.
use super::*;
use crate::implementations::continuous::cost_profile::structured_v10::service::owner_blocks::collector::State as OwnerState;
use std::num::NonZeroU64;

#[test]
#[ignore = "requires immutable source7 and original issued demands via FERRUM_ARCHIVED_BOUND_SPEC"]
fn replay_full_family_with_fit_row_targets() {
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
    let mut run: Option<RowTargetBank> = None;
    let mut selected = None;
    let mut fitted_at_block = 0;
    let mut fitted_bytes = 0;
    let mut block = 0;
    let mut block_opened = header.opening.monotonic_ns;
    let mut closed_offered = 0;
    let mut totals = BTreeMap::<u32, RowCounts>::new();
    let mut fit_rows = BTreeMap::<u32, usize>::new();
    let mut rows_by_call = BTreeMap::<u64, u32>::new();
    let mut compatible = 0;
    let mut unknown = 0;
    let mut calls_verified = BTreeSet::new();

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
            // Freeze target rows from admitted Fit inputs before close drains them.
            let before_fit_rows: BTreeMap<_, BTreeMap<u32, usize>> =
                if matches!(record, StructuredServiceRecordV7::BlockClose { .. }) {
                    collector
                        .owners
                        .iter()
                        .filter(|o| matches!(o.state, OwnerState::Empty))
                        .map(|o| {
                            let mut rows = BTreeMap::new();
                            for sample in &o.samples {
                                *rows.entry(sample.input.owner().rows).or_default() += 1;
                            }
                            (o.contract.owner_attempt_id, rows)
                        })
                        .collect()
                } else {
                    BTreeMap::new()
                };
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
                    fit_rows = before_fit_rows[&owner.contract.owner_attempt_id].clone();
                    assert!(!fit_rows.is_empty());
                    run = Some(RowTargetBank {
                        targets: fit_rows.keys().copied().collect(),
                        row_counts: [BTreeMap::new(), BTreeMap::new()],
                        phase_members: [0, 0],
                        boundaries: Vec::new(),
                        terminal_reason: None,
                        original_expiry_ns: expiry,
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
                            "fit_rows":fit_rows,"original_expiry_ns":expiry,"fit_floor_ns":floor,"fit_bytes":fitted_bytes
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
                let rows = u32::try_from(numeric.rows.len()).unwrap();
                assert_eq!(input.owner().rows, rows);
                totals.entry(rows).or_default().physical += 1;
                if selected.is_none() || block <= fitted_at_block {
                    continue;
                }
                totals.entry(rows).or_default().after_fit += 1;
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
                    totals.entry(rows).or_default().fit_incompatible += 1;
                    continue;
                };
                compatible += 1;
                let history = numeric.rows[0].sampling_history_tokens;
                rows_by_call.insert(call, rows);
                totals.entry(rows).or_default().fit_compatible += 1;
                let run = run.as_mut().unwrap();
                let cell = complete_key(&raw, &work);
                let before = run.inner.predictions.len();
                run.offer(
                    cell,
                    rows,
                    block,
                    wave.issued_at_ns,
                    call,
                    history,
                    point,
                    wall,
                );
                if run.inner.predictions.len() > before {
                    totals.entry(rows).or_default().known += 1;
                    totals.entry(rows).or_default().misses +=
                        usize::from(run.inner.predictions.last().unwrap().1 < wall);
                } else {
                    totals.entry(rows).or_default().unknown += 1;
                }
                // Keep original actual shape validation in the audit; no
                // replacement of its work by a cheaper forecast is allowed.
                assert_eq!(input.owner().rows, rows);
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
    let transitions: Vec<_> = run
        .transitions
        .iter()
        .map(|value| {
            let mut v = value.clone();
            v.as_object_mut().unwrap().remove("cell");
            v
        })
        .collect();
    let row_summary: Vec<_> = totals
        .iter()
        .map(|(rows, count)| {
            json!({"rows":rows,"physical":count.physical,"after_fit":count.after_fit,
            "fit_compatible":count.fit_compatible,"fit_incompatible":count.fit_incompatible,
            "known":count.known,"unknown":count.unknown+count.fit_incompatible,
            "underestimates":count.misses})
        })
        .collect();
    let paired: Vec<_> = spec
        .pairs
        .iter()
        .map(|p| {
            json!({
                "call":p.call_id,"demand_verified":calls_verified.contains(&p.call_id),
                "rows":rows_by_call.get(&p.call_id),"known":known_issued.contains(&p.call_id)
            })
        })
        .collect();
    eprintln!(
        "FULL_FAMILY_ROW_TARGETS {}",
        json!({
            "fit_block":fitted_at_block,"fit_rows":fit_rows,"last_open_block":block,
            "phase":run.phase,"completed_rq":run.phase == 3 && run.terminal_reason.is_none(),
            "terminal_reason":run.terminal_reason,"phase_boundaries":run.boundaries,
            "current_target_counts":run.target_summary(),
            "all_rows":row_summary,"known_future":run.predictions.len(),"future_misses":misses,
            "known_verified_issued_calls":known_issued,"paired":paired,
            "compatible_after_fit":compatible,"incompatible_after_fit":unknown,
            "planning_ns":if bounds.is_empty(){None}else{Some([bounds[0],bounds[bounds.len()/2],*bounds.last().unwrap()])},
            "transitions":transitions,
            "peak_fit_cells_bytes":fitted_bytes+run.peak_bytes,
            "capacity_refusals":run.refused_capacity,
            "scope":"all original family inputs; unchanged complete joint keys; Fit input rows target only; CPU feasibility, no production enrollment",
            "budget_scope":"original Fit + bounded numeric cells; input-only target diagnostic counters and original replay excluded"
        })
    );
    // Zero coverage or an incomplete phase is a result, not a positive fixture.
    assert_eq!(calls_verified.len(), spec.pairs.len());
    assert_eq!(
        totals.values().map(|v| v.fit_compatible).sum::<usize>(),
        compatible
    );
    assert_eq!(
        totals.values().map(|v| v.known).sum::<usize>(),
        run.predictions.len()
    );
    assert_eq!(
        totals.values().map(|v| v.after_fit).sum::<usize>(),
        compatible + unknown
    );
    assert!(
        totals.len() > 1,
        "full family must not silently become a B1-only stream"
    );
}

#[derive(Default)]
struct RowCounts {
    physical: usize,
    after_fit: usize,
    fit_compatible: usize,
    fit_incompatible: usize,
    known: usize,
    unknown: usize,
    misses: usize,
}

/// Same equivalence relation as production JointCellKey { input, work }.
/// Lengths retain the split. Exact rows is NOT appended to the joint key.
fn complete_key(raw: &[u64], work: &[u64]) -> Cell {
    let mut bytes = Vec::new();
    bytes.extend((raw.len() as u64).to_le_bytes());
    bytes.extend(Cell::of(raw).0);
    bytes.extend((work.len() as u64).to_le_bytes());
    bytes.extend(Cell::of(work).0);
    Cell(bytes)
}

struct RowTargetBank {
    inner: Cells,
    targets: BTreeSet<u32>,
    row_counts: [BTreeMap<(Cell, u32), usize>; 2],
    phase_members: [usize; 2],
    phase: usize,
    first_phase_block: u64,
    boundaries: Vec<Value>,
    terminal_reason: Option<&'static str>,
    original_expiry_ns: u64,
}
impl std::ops::Deref for RowTargetBank {
    type Target = Cells;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl RowTargetBank {
    fn target_summary(&self) -> Value {
        json!(self.targets.iter().map(|rows| {
            let stats = [0,1].map(|phase| {
                let counts: Vec<_> = self.row_counts[phase].iter()
                    .filter(|((_, r), _)| r == rows).map(|(_, n)| *n).collect();
                json!({"members":counts.iter().sum::<usize>(),
                    "maximum_same_key_members":counts.iter().max().copied().unwrap_or(0),
                    "ready_keys":counts.iter().filter(|n| **n >= self.schedule.min_members[phase+1]).count()})
            });
            json!({"rows":rows,"residual":stats[0],"qualification":stats[1]})
        }).collect::<Vec<_>>())
    }

    fn inputs_ready(&self, offered: usize) -> bool {
        self.phase <= 2
            && offered >= self.schedule.phase_min_offered[self.phase]
            && self.targets.iter().all(|rows| {
                self.row_counts[self.phase - 1]
                    .iter()
                    .any(|((cell, r), n)| {
                        r == rows
                            && *n >= self.schedule.min_members[self.phase]
                            && self
                                .inner
                                .slots
                                .get(cell)
                                .is_some_and(|s| match self.phase {
                                    1 => matches!(s.phase, Phase::Residual { .. }),
                                    2 => matches!(s.phase, Phase::Qualification { .. }),
                                    _ => false,
                                })
                    })
            })
    }

    #[allow(clippy::too_many_arguments)]
    fn offer(
        &mut self,
        cell: Cell,
        rows: u32,
        block: u64,
        now: u64,
        call: u64,
        history: u64,
        point: u64,
        wall: u64,
    ) {
        if now > self.original_expiry_ns || self.terminal_reason.is_some() {
            return;
        }
        if self.phase != 1 && !self.inner.slots.contains_key(&cell) {
            return;
        }
        if self.phase <= 2 {
            let eligible = match self.inner.slots.get(&cell) {
                None => self.phase == 1,
                Some(slot) => matches!(
                    (&slot.phase, self.phase),
                    (Phase::Residual { .. }, 1) | (Phase::Qualification { .. }, 2)
                ),
            };
            if !eligible {
                return;
            }
            if self.phase_members[self.phase - 1] == self.settings.max_phase_samples {
                self.terminal_reason = Some("maximum_phase_samples");
                return;
            }
        }
        let before = self.inner.refused_capacity;
        self.inner
            .offer(cell.clone(), block, now, call, history, point, wall);
        if self.phase <= 2 && self.inner.refused_capacity == before {
            self.phase_members[self.phase - 1] += 1;
            // rows changes only this input stopping count, never key membership
            // or the margin's sample population.
            *self.row_counts[self.phase - 1]
                .entry((cell.clone(), rows))
                .or_default() += 1;
            if self.phase == 1 {
                self.inner.slots.get_mut(&cell).unwrap().phase_block = self.first_phase_block;
            }
        }
    }

    fn close(&mut self, block: u64, now: u64) {
        if self.phase > 2 || self.terminal_reason.is_some() {
            return;
        }
        if now > self.original_expiry_ns {
            self.terminal_reason = Some("original_fit_expired");
            return;
        }
        let blocks = block.checked_sub(self.first_phase_block).unwrap() + 1;
        let offered = (blocks as usize)
            .checked_mul(self.schedule.block_offered)
            .unwrap();
        let maximum = self
            .schedule
            .input_readiness
            .as_ref()
            .unwrap()
            .maximum_phase_blocks[self.phase];
        if !self.inputs_ready(offered) {
            if blocks as usize >= maximum {
                self.terminal_reason = Some("missing_fit_row_target_at_maximum_phase_blocks");
            }
            return;
        }
        // This decision is complete before R residuals or Q misses are read.
        self.boundaries
            .push(json!({"phase":self.phase,"block":block,"at_ns":now,
            "members":self.phase_members[self.phase-1],"targets":self.target_summary()}));
        self.inner.close(block, now);
        for slot in self.inner.slots.values_mut() {
            let keep = match self.phase {
                1 => matches!(slot.phase, Phase::Qualification { .. }),
                2 => matches!(slot.phase, Phase::Qualified { .. }),
                _ => false,
            };
            if !keep {
                slot.phase = Phase::Rejected;
            }
        }
        self.phase += 1;
        self.first_phase_block = block + 1;
    }
}

fn fixture_row_targets(targets: &[u32]) -> RowTargetBank {
    RowTargetBank {
        inner: fixture_cells(),
        targets: targets.iter().copied().collect(),
        row_counts: [BTreeMap::new(), BTreeMap::new()],
        phase_members: [0, 0],
        phase: 1,
        first_phase_block: 1,
        boundaries: Vec::new(),
        terminal_reason: None,
        original_expiry_ns: u64::MAX,
    }
}

#[test]
fn row_targets_do_not_count_other_rows_in_the_same_complete_key() {
    let key = complete_key(&[5], &[5]);
    assert_eq!(key, complete_key(&[7], &[7])); // production dyadic equality retained
    assert_ne!(complete_key(&[1, 2], &[3]), complete_key(&[1], &[2, 3]));
    let mut bank = fixture_row_targets(&[5, 7]);
    for n in 0..8 {
        bank.offer(key.clone(), 5, 1, n, n, 1, 10, 12);
    }
    assert!(
        !bank.inputs_ready(8),
        "rows5 cannot satisfy a declared rows7 target"
    );
    for n in 8..16 {
        bank.offer(key.clone(), 7, 2, n, n, 1, 10, 12);
    }
    assert!(bank.inputs_ready(16));
    bank.close(2, 16);
    assert_eq!(bank.phase, 2);
    assert_eq!(bank.first_phase_block, 3);
}

#[test]
fn row_targets_disappearing_fit_stratum_blocks_otherwise_usable_cell() {
    let mut bank = fixture_row_targets(&[1, 7]);
    let key = complete_key(&[128], &[128]);
    for block in 1..=4 {
        for n in 0..8 {
            bank.offer(
                key.clone(),
                1,
                block,
                block * 8 + n,
                block * 8 + n,
                1,
                10,
                12,
            );
        }
        bank.close(block, block * 8 + 8);
    }
    assert_eq!(bank.phase, 1);
    assert_eq!(
        bank.terminal_reason,
        Some("missing_fit_row_target_at_maximum_phase_blocks")
    );
    assert!(bank.predictions.is_empty());
    assert!(bank.row_counts[0].values().sum::<usize>() >= 8);
}

#[test]
fn row_targets_input_cut_does_not_depend_on_current_duration() {
    let mut a = fixture_row_targets(&[1]);
    let mut b = fixture_row_targets(&[1]);
    let key = complete_key(&[128], &[128]);
    for n in 0..8 {
        a.offer(key.clone(), 1, 1, n, n, 1, 10, 12);
        b.offer(key.clone(), 1, 1, n, n, 1, 10, 999_999);
    }
    assert_eq!(a.inputs_ready(8), b.inputs_ready(8));
    a.close(1, 8);
    b.close(1, 8);
    assert_eq!(a.phase, b.phase);
    assert_eq!(a.boundaries, b.boundaries);
}
