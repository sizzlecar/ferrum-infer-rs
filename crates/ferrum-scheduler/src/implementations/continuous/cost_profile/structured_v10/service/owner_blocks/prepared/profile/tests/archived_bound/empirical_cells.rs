//! Counterfactual feasibility only: immutable original fit, input-declared cells,
//! independent original complete R/Q blocks, and untouched original expiry.
//! No live tickets, model publication, or product defaults are changed here.
use super::*;
use crate::implementations::continuous::cost_model::structured_v2::*;
use std::collections::{BTreeMap, BTreeSet};

mod fit_target_stop;
mod global_bank;
mod global_residual_control;
mod production_bank;
mod production_global;
mod row_targets;
mod same_source;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Cell(Vec<u8>);

impl Cell {
    /// Zero is exact; each nonzero coordinate stays in one predeclared [2^k,2^(k+1)).
    /// A whole vector must match; coordinate-wise unions are never constructed.
    /// This partition exists before reading any source timing. Lazy allocation
    /// at the first matching input does not select a new range from its wall.
    fn of(coordinates: &[u64]) -> Self {
        Self(
            coordinates
                .iter()
                .map(|x| {
                    if *x == 0 {
                        0
                    } else {
                        (64 - x.leading_zeros()) as u8
                    }
                })
                .collect(),
        )
    }
}

#[derive(Clone, Copy, Debug)]
enum Coordinates {
    AllInput,
    DeclaredWork,
}

impl Coordinates {
    fn cell(self, raw: &[u64], work: &[u64]) -> Cell {
        match self {
            Self::AllInput => {
                let mut out = Cell::of(raw);
                out.0.extend(Cell::of(work).0);
                out
            }
            // Compare an alternative policy explicitly; never silently drop a
            // failing raw coordinate. Work is the existing complete declared
            // envelope vector, including host work and completion/pending axes.
            Self::DeclaredWork => Cell::of(work),
        }
    }
}

#[derive(Debug)]
enum Phase {
    Residual {
        errors: Vec<i128>,
    },
    Qualification {
        margin: u64,
        members: usize,
        misses: usize,
    },
    Qualified {
        margin: u64,
        at_ns: u64,
    },
    Rejected,
}

#[derive(Debug)]
struct Slot {
    phase: Phase,
    first_block: u64,
    phase_block: u64,
    minimum_history: u64,
    maximum_history: u64,
}

struct Cells {
    policy: Coordinates,
    slots: BTreeMap<Cell, Slot>,
    original_fit_floor: u64,
    settings: StructuredSettingsV2,
    schedule: OwnerBlockScheduleV1,
    maximum_slots: usize,
    maximum_bytes: usize,
    maximum_visits: u64,
    visits: u64,
    peak_bytes: usize,
    predictions: Vec<(u64, u64, u64)>,
    transitions: Vec<Value>,
    refused_capacity: usize,
}

impl Cells {
    fn retained_bytes(&self) -> usize {
        self.slots
            .iter()
            .map(|(cell, slot)| {
                cell.0.capacity()
                    + std::mem::size_of::<Slot>()
                    + 128
                    + match &slot.phase {
                        Phase::Residual { errors } => {
                            errors.capacity() * std::mem::size_of::<i128>()
                        }
                        _ => 0,
                    }
            })
            .sum()
    }

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
        // Admission depends on input and prior state, never this sample's wall.
        let cost = cell.0.len() as u64;
        if self
            .visits
            .checked_add(cost)
            .is_none_or(|n| n > self.maximum_visits)
        {
            self.refused_capacity += 1;
            return;
        }
        self.visits += cost;
        if !self.slots.contains_key(&cell) {
            // Reserve the full bounded residual vector before accepting a cell.
            let reserve = cell.0.len()
                + std::mem::size_of::<Slot>()
                + 128
                + self.settings.max_phase_samples * std::mem::size_of::<i128>();
            if self.slots.len() >= self.maximum_slots
                || self
                    .retained_bytes()
                    .checked_add(reserve)
                    .is_none_or(|n| n > self.maximum_bytes)
            {
                self.refused_capacity += 1;
                return;
            }
            self.slots.insert(
                cell.clone(),
                Slot {
                    phase: Phase::Residual {
                        errors: Vec::with_capacity(self.settings.max_phase_samples),
                    },
                    first_block: block,
                    phase_block: block,
                    minimum_history: history,
                    maximum_history: history,
                },
            );
        }
        let slot = self.slots.get_mut(&cell).unwrap();
        slot.minimum_history = slot.minimum_history.min(history);
        slot.maximum_history = slot.maximum_history.max(history);
        match &mut slot.phase {
            Phase::Residual { errors } => {
                if errors.len() == self.settings.max_phase_samples {
                    slot.phase = Phase::Rejected;
                } else {
                    errors.push(i128::from(wall) - i128::from(point));
                }
            }
            Phase::Qualification {
                margin,
                members,
                misses,
            } => {
                *members += 1;
                *misses += usize::from(point.checked_add(*margin).is_none_or(|n| n < wall));
                if *members > self.settings.max_phase_samples {
                    slot.phase = Phase::Rejected;
                }
            }
            Phase::Qualified { margin, at_ns } if now >= *at_ns => {
                self.predictions
                    .push((call, point.checked_add(*margin).unwrap(), wall));
            }
            _ => {}
        }
        self.peak_bytes = self.peak_bytes.max(self.retained_bytes());
        assert!(self.peak_bytes <= self.maximum_bytes);
    }

    fn close(&mut self, block: u64, at_ns: u64) {
        for (cell, slot) in &mut self.slots {
            let (phase, members) = match &slot.phase {
                Phase::Residual { errors } => (1, errors.len()),
                Phase::Qualification { members, .. } => (2, *members),
                _ => continue,
            };
            let blocks = block - slot.phase_block + 1;
            // All original offers count, including offers before this cell's
            // first matching input in its initial block: the global partition
            // already classified those earlier inputs outside this cell. This
            // counterfactual ledger is NOT an original registered cohort.
            let offered = blocks as usize * self.schedule.block_offered;
            let maximum_blocks = self
                .schedule
                .input_readiness
                .as_ref()
                .unwrap()
                .maximum_phase_blocks[phase];
            if blocks as usize > maximum_blocks {
                slot.phase = Phase::Rejected;
                continue;
            }
            if members < self.schedule.min_members[phase]
                || offered < self.schedule.phase_min_offered[phase]
            {
                continue;
            }
            match &mut slot.phase {
                Phase::Residual { errors } => {
                    let min = *errors.iter().min().unwrap();
                    let max = *errors.iter().max().unwrap();
                    let span = match self.settings.learned_drift {
                        StructuredLearnedDriftV2::Disabled => 0,
                        StructuredLearnedDriftV2::ObservedResidualSpanV1 {
                            maximum_span_margin_ns,
                        } => {
                            let span = u64::try_from(max - min).unwrap();
                            if span > maximum_span_margin_ns.get() {
                                slot.phase = Phase::Rejected;
                                continue;
                            }
                            span
                        }
                    };
                    errors.sort_unstable();
                    let residual = errors[(99 * errors.len()).div_ceil(100) - 1].max(0) as u64;
                    let margin = residual
                        .max(self.original_fit_floor)
                        .checked_add(span)
                        .unwrap()
                        .checked_add(self.settings.static_margin_ns)
                        .unwrap();
                    self.transitions
                        .push(json!({"stage":"residual","block":block,
                        "cell":cell.0,"members":members,"residual_ns":residual,
                        "span_ns":span,"margin_ns":margin,
                        "history":[slot.minimum_history,slot.maximum_history]}));
                    slot.phase = Phase::Qualification {
                        margin,
                        members: 0,
                        misses: 0,
                    };
                    slot.phase_block = block + 1;
                }
                Phase::Qualification { margin, misses, .. } => {
                    self.transitions
                        .push(json!({"stage":"qualification","block":block,
                        "first_block":slot.first_block,"cell":cell.0,
                        "members":members,"misses":misses,"margin_ns":margin,
                        "history":[slot.minimum_history,slot.maximum_history]}));
                    slot.phase = if *misses == 0 {
                        Phase::Qualified {
                            margin: *margin,
                            at_ns,
                        }
                    } else {
                        Phase::Rejected
                    };
                }
                _ => unreachable!(),
            }
        }
    }
}

#[test]
fn joint_cells_keep_zero_boundaries_and_reject_unobserved_cross_products() {
    assert_ne!(Cell::of(&[0, 128]), Cell::of(&[1, 128]));
    assert_eq!(Cell::of(&[128, 128]), Cell::of(&[255, 255]));
    assert_ne!(Cell::of(&[255, 255]), Cell::of(&[256, 255]));
    let seen = BTreeSet::from([Cell::of(&[128, 1]), Cell::of(&[1, 128])]);
    assert!(!seen.contains(&Cell::of(&[128, 128])));
    assert_ne!(Cell::of(&[1, 2]), Cell::of(&[424, 456]));
}

fn fixture_cells() -> Cells {
    let settings = StructuredSettingsV2 {
        min_phase_samples: 8,
        max_phase_samples: 32,
        static_margin_ns: 0,
        ..StructuredSettingsV2::default()
    };
    let schedule = OwnerBlockScheduleV1::new_with_input_readiness(
        8,
        [8; 3],
        [8; 3],
        OwnerInputReadinessV1::new([4; 3], 32_000_000).unwrap(),
    )
    .unwrap();
    Cells {
        policy: Coordinates::DeclaredWork,
        slots: BTreeMap::new(),
        original_fit_floor: 1,
        settings,
        schedule,
        maximum_slots: 8,
        maximum_bytes: 1024 * 1024,
        maximum_visits: 32_000_000,
        visits: 0,
        peak_bytes: 0,
        predictions: Vec::new(),
        transitions: Vec::new(),
        refused_capacity: 0,
    }
}

#[test]
fn joint_cells_allow_continuous_growth_only_after_independent_complete_qualification() {
    let mut cells = fixture_cells();
    for h in 128..136 {
        cells.offer(Cell::of(&[h]), 1, h, h, h, 10, 12);
    }
    assert!(cells.predictions.is_empty());
    cells.close(1, 200);
    for h in 144..152 {
        cells.offer(Cell::of(&[h]), 2, 200 + h, h, h, 10, 12);
    }
    assert!(
        cells.predictions.is_empty(),
        "unclosed Q must never publish"
    );
    cells.close(2, 400);
    cells.offer(Cell::of(&[200]), 3, 500, 200, 200, 10, 12);
    assert_eq!(cells.predictions, vec![(200, 12, 12)]);
    cells.offer(Cell::of(&[256]), 3, 501, 256, 256, 10, 12);
    assert_eq!(
        cells.predictions.len(),
        1,
        "next joint cell requires new independent R/Q"
    );
}

#[test]
fn joint_cells_reject_entire_qualification_without_relearning_from_its_failure() {
    let mut cells = fixture_cells();
    for h in 128..136 {
        cells.offer(Cell::of(&[h]), 1, h, h, h, 10, 12);
    }
    cells.close(1, 200);
    for h in 144..152 {
        cells.offer(
            Cell::of(&[h]),
            2,
            200 + h,
            h,
            h,
            10,
            if h == 147 { 100 } else { 12 },
        );
    }
    cells.close(2, 400);
    assert!(matches!(
        cells.slots[&Cell::of(&[200])].phase,
        Phase::Rejected
    ));
    cells.offer(Cell::of(&[200]), 3, 500, 200, 200, 10, 12);
    assert!(cells.predictions.is_empty());
}

#[test]
#[ignore = "requires original source8/source7 artifacts via FERRUM_ARCHIVED_BOUND_SPEC"]
fn replay_empirical_joint_cells_from_original_complete_blocks() {
    let spec: AuditSpec = serde_json::from_slice(
        &std::fs::read(std::env::var_os("FERRUM_ARCHIVED_BOUND_SPEC").unwrap()).unwrap(),
    )
    .unwrap();
    let limits = CostProfileLoadLimits::default();
    let mut bytes = Vec::new();
    std::fs::File::open(&spec.checkpoint_source)
        .unwrap()
        .take(spec.checkpoint_bytes as u64)
        .read_to_end(&mut bytes)
        .unwrap();
    assert_eq!(
        <[u8; 32]>::from(Sha256::digest(&bytes)),
        spec.checkpoint_sha256
    );
    let h = header(&bytes);
    let checkpoint = replay_structured_source_v8(&bytes, &limits).unwrap();
    let closing = checkpoint.population.closing.clone();
    let catalog = checkpoint
        .activate_same_process_memory(closing, &limits)
        .unwrap();
    let child = catalog
        .children
        .iter()
        .find(|c| c.parameters_signature() == spec.parameters_sha256)
        .unwrap();
    let actual_header = seeded::verify_source_file(&spec.actual_source, spec.actual_source_sha256);
    assert_eq!(h.fingerprint, actual_header.fingerprint);
    let contract = h
        .declaration
        .population
        .nonnegative_envelope
        .as_ref()
        .unwrap();
    let d = &actual_header.declaration;
    // Match the original collector's seeded_input_contract. The seed declares
    // numerical membership, not physical validity of all later original work.
    // Full physical recipe/shape/settlement validation still runs first.
    let mut actual_physical_contract = d.nonnegative_envelope.clone().unwrap();
    if d.schedule.algorithm_universe
        == Some(OwnerAlgorithmUniversePolicyV1::SeededFirstOrdinaryDiscoveryBlockSubsetV1)
    {
        actual_physical_contract.algorithm_universe = None;
    }
    let frozen_model_bytes = child.retained_payload_bytes().unwrap();
    let mut original_demands = BTreeMap::new();
    for line in BufReader::new(std::fs::File::open(&spec.query_journal).unwrap()).lines() {
        let value: Value = serde_json::from_str(&line.unwrap()).unwrap();
        if value["event"] != "query_constructed" {
            continue;
        }
        if let Some(pair) = spec.pairs.iter().find(|p| {
            value["transaction"] == p.transaction
                && value["data"]["attempt"] == p.attempt
                && value["data"]["alternative"] == p.alternative
        }) {
            assert!(original_demands
                .insert(pair.call_id, value["data"]["demand"].clone())
                .is_none());
        }
    }
    assert_eq!(original_demands.len(), spec.pairs.len());
    let mut same_issued_calls = BTreeSet::new();
    let mut future_history = BTreeMap::new();
    assert_eq!(child.model.runtime_limits().1, d.settings.max_sample_age_ns);
    let mut cold_cells = [BTreeSet::new(), BTreeSet::new()];
    for line in bytes
        .split(|b| *b == b'\n')
        .skip(1)
        .filter(|s| !s.is_empty())
    {
        let v: Value = serde_json::from_slice(line).unwrap();
        if v["kind"] != "completed" {
            continue;
        }
        let StructuredServiceRecordV7::Completed { wave } = serde_json::from_slice(line).unwrap()
        else {
            unreachable!()
        };
        let (prepared, offered) =
            physical::original_prepared(&wave.host_stages, wave.independent.as_ref()).unwrap();
        let input = input_replay::project_service_actual_with_domain(
            &prepared,
            &offered,
            &contract.workload_domain,
        )
        .unwrap();
        if input.owner().rows != 1 {
            continue;
        }
        if let Some((_, raw, work)) = child
            .model
            .diagnose_empirical_cell_input(&StructuredQueryV2::exact(input))
        {
            cold_cells[0].insert(Coordinates::AllInput.cell(&raw, &work));
            cold_cells[1].insert(Coordinates::DeclaredWork.cell(&raw, &work));
        }
    }
    assert!(!cold_cells[0].is_empty());
    let mut runs: Vec<_> = [Coordinates::AllInput, Coordinates::DeclaredWork]
        .into_iter()
        .map(|policy| Cells {
            policy,
            slots: BTreeMap::new(),
            original_fit_floor: child.model.uncertainty().fit_error_floor_ns,
            settings: d.settings.clone(),
            schedule: d.schedule.clone(),
            maximum_slots: d.maximum_owners,
            maximum_bytes: d
                .maximum_retained_numeric_bytes
                .checked_sub(frozen_model_bytes)
                .unwrap(),
            maximum_visits: contract.settings.maximum_coordinate_visits,
            visits: 0,
            peak_bytes: 0,
            predictions: Vec::new(),
            transitions: Vec::new(),
            refused_capacity: 0,
        })
        .collect();
    let mut block = 0;
    let mut block_opened = actual_header.opening.monotonic_ns;
    let mut closed = 0;
    let mut closed_offered = 0;
    let mut compatible = 0;
    let mut incompatible = 0;
    let mut physical_b1 = 0;
    let mut unknown_samples = Vec::new();
    let mut far_rejected = [0; 2];
    let mut original_expiry = None;
    for line in BufReader::new(std::fs::File::open(&spec.actual_source).unwrap())
        .lines()
        .skip(1)
    {
        let record: StructuredServiceRecordV7 = serde_json::from_str(&line.unwrap()).unwrap();
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
                closing,
                offered,
                ..
            } => {
                assert_eq!(block, b);
                assert!(closed < b);
                assert_eq!(b, closed + 1);
                assert_eq!(
                    offered - closed_offered,
                    d.schedule.block_offered as u64,
                    "only the original complete offer block may freeze R/Q"
                );
                for run in &mut runs {
                    run.close(b, closing.monotonic_ns);
                }
                closed = b;
                closed_offered = offered;
            }
            StructuredServiceRecordV7::Completed { wave } => {
                // Physical validation and structural projection precede looking
                // at the timing label. This fixture deliberately studies B1.
                let shape = wave.host_stages.actual_shape.as_ref().unwrap();
                let numeric = shape.numeric_features.as_ref().unwrap();
                if numeric.rows.len() != 1 {
                    continue;
                }
                physical_b1 += 1;
                let (_, wall, _) = physical::validate_parts(
                    &actual_header.fingerprint,
                    actual_header.opening.monotonic_ns,
                    Some(&actual_physical_contract),
                    block_opened,
                    wave.ticket,
                    wave.fifo,
                    wave.issued_at_ns,
                    &wave.host_stages,
                    wave.independent.as_ref(),
                    &mut physical::Frontiers::default(),
                )
                .unwrap_or_else(|error| panic!(
                    "original physical validation failed: call={} ticket={} fifo={} block={} source_opened={} block_opened={} issued={} error={error:?}",
                    wave.host_stages.call_id, wave.ticket, wave.fifo, block,
                    actual_header.opening.monotonic_ns, block_opened, wave.issued_at_ns));
                let (prepared, offered) =
                    physical::original_prepared(&wave.host_stages, wave.independent.as_ref())
                        .unwrap();
                let input = input_replay::project_service_actual_with_domain(
                    &prepared,
                    &offered,
                    &contract.workload_domain,
                )
                .unwrap();
                let query = StructuredQueryV2::exact(input);
                if let Some(demand) = original_demands.get(&wave.host_stages.call_id) {
                    assert_eq!(
                        serde_json::to_value(query.required_coverage().unwrap()).unwrap(),
                        *demand,
                        "cannot replace an issued forecast by its cheaper settled outcome"
                    );
                    same_issued_calls.insert(wave.host_stages.call_id);
                }
                let Some((point, raw, work)) = child.model.diagnose_empirical_cell_input(&query)
                else {
                    incompatible += 1;
                    let projection_error = contract
                        .algorithm_universe
                        .as_ref()
                        .and_then(|u| query.clone().with_algorithm_universe(u).err());
                    unknown_samples.push(json!({
                        "stage":"cold_frozen_model_support",
                        "call":wave.host_stages.call_id,"ticket":wave.ticket,"fifo":wave.fifo,
                        "block":block,"issued_at_ns":wave.issued_at_ns,
                        "history":numeric.rows[0].sampling_history_tokens,
                        "reason":projection_error.map(|v|format!("{v:?}"))
                            .unwrap_or_else(||"structural_or_fitted_point_unavailable".into()),
                        "physical_valid":true,"original_offer_retained":true,
                    }));
                    continue;
                };
                compatible += 1;
                let old = child
                    .predict_query_local(child.fingerprint(), &query, wave.issued_at_ns)
                    .unwrap();
                assert_eq!(
                    *original_expiry.get_or_insert(old.valid_until_ns),
                    old.valid_until_ns
                );
                let now = child.model_now_ns(wave.issued_at_ns).unwrap();
                child.model.validate_runtime_at(now).unwrap();
                let history = numeric.rows[0].sampling_history_tokens;
                future_history.insert(wave.host_stages.call_id, history);
                for (i, run) in runs.iter_mut().enumerate() {
                    let cell = run.policy.cell(&raw, &work);
                    if history > 2 {
                        assert!(
                            !cold_cells[i].contains(&cell),
                            "short cold samples cannot cover long decode"
                        );
                        far_rejected[i] += 1;
                    }
                    run.offer(
                        cell,
                        block,
                        wave.issued_at_ns,
                        wave.host_stages.call_id,
                        history,
                        point,
                        wall,
                    );
                }
            }
            _ => {}
        }
    }
    let expiry = original_expiry.unwrap();
    assert_eq!(
        physical_b1,
        compatible + incompatible,
        "no original B1 sample silently discarded"
    );
    assert!(
        child.model.validate_runtime_at(expiry + 1).is_err(),
        "Q cannot refresh original Fit TTL"
    );
    for run in &runs {
        let misses = run.predictions.iter().filter(|(_, p, w)| p < w).count();
        let mut planning: Vec<_> = run.predictions.iter().map(|(_, p, _)| *p).collect();
        planning.sort_unstable();
        eprintln!(
            "EMPIRICAL_CELLS {}",
            json!({
                "policy":format!("{:?}", run.policy),"compatible":compatible,"incompatible":incompatible,
                "physical_b1":physical_b1,"original_b1_unknown_samples":unknown_samples,
                "short_cold_far_rejected":far_rejected,"last_open_block":block,"last_closed_block":closed,
                "cells":run.slots.len(),"known_future":run.predictions.len(),"future_misses":misses,
                "planning_ns": if planning.is_empty(){None}else{Some([planning[0],planning[planning.len()/2],*planning.last().unwrap()])},
            "original_expiry_ns":expiry,"peak_retained_numeric_bytes":run.peak_bytes,
            "frozen_model_bytes":frozen_model_bytes,
            "peak_model_and_cell_bytes":frozen_model_bytes + run.peak_bytes,
                "support_quantization_coordinate_visits":run.visits,"capacity_refusals":run.refused_capacity,
                "budget_scope":"frozen_model_plus_incremental_cells; excludes original collector, projection, prediction and audit output",
                "transitions":run.transitions,"future_same_call_results":run.predictions,
                "verified_issued_calls":same_issued_calls,
                "known_verified_issued_calls":run.predictions.iter()
                    .filter(|(call,_,_)|same_issued_calls.contains(call)).map(|(call,_,_)|*call).collect::<Vec<_>>(),
            })
        );
    }
    // A positive test must cover real future traffic; rejecting three archived
    // queries alone is not feasibility evidence. The raw-input policy is a
    // separately reported control and may prove unnecessarily restrictive.
    let work = &runs[1];
    assert!(
        work.predictions
            .iter()
            .any(|(call, _, _)| same_issued_calls.contains(call)),
        "positive evidence must include an unchanged original issued pre-execution query"
    );
    let histories: BTreeSet<_> = work
        .predictions
        .iter()
        .map(|(call, _, _)| future_history[call])
        .collect();
    assert!(
        histories.first() < histories.last(),
        "future evidence must include strictly increasing continuous decode history"
    );
    assert!(
        !work.predictions.is_empty(),
        "normal future decode must become predictable"
    );
    assert_eq!(work.predictions.iter().filter(|(_, p, w)| p < w).count(), 0);
    assert_eq!(work.refused_capacity, 0);
    assert!(far_rejected.iter().all(|n| *n > 0));
}
