//! Offline range comparison for the unchanged identified Fit/global R control.
//! Domains freeze once from all original F/R/Q inputs. Labels never select a
//! point or a cell; sparse cells retain their original counts without a quota.
use super::*;
use serde_json::{json, Value};

struct PhaseRange {
    raw: JointSupport,
    // Complete work tuple, exact zero and predeclared dyadic positive bands.
    cells: Vec<(Vec<u8>, usize)>,
    members: usize,
    raw_dimensions: usize,
    work_dimensions: usize,
}

fn work_cell(work: &[u64]) -> Vec<u8> {
    work.iter()
        .map(|&n| (64 - n.leading_zeros()) as u8)
        .collect()
}

#[derive(Clone, Copy, Default)]
struct RangeShape {
    members: usize,
    raw: usize,
    work: usize,
    projection_clone_bytes: usize,
}

/// Preflight the simultaneously live diagnostic arrays, using input dimensions
/// before any projection/allocation. Extra coordinate/container room covers
/// construction copies; actual owned capacities are checked again after freeze.
fn range_peak_bytes(
    phases: &[RangeShape; 3],
    settings: &StructuredSettingsV2,
    model_bytes: usize,
    limit: usize,
) -> Result<usize> {
    let mut total = 0usize;
    let mut peak = model_bytes
        .checked_add(std::mem::size_of::<FrozenRangeAudit>())
        .and_then(|n| n.checked_add(3 * std::mem::size_of::<Vec<(Vec<u64>, Vec<u64>)>>()))
        .ok_or(StructuredUnknown::Capacity)?;
    let mut projection = 0usize;
    for p in phases {
        if p.members < settings.min_phase_samples
            || p.members > settings.max_phase_samples
            || p.members > 4096
            || p.raw == 0
            || p.work == 0
            || p.raw > settings.max_axes
            || p.work > settings.max_axes
        {
            return Err(StructuredUnknown::Capacity);
        }
        total = total
            .checked_add(p.members)
            .ok_or(StructuredUnknown::Capacity)?;
        // Live points: 8(S+W); frozen original support: 8S; dyadic keys: W.
        // Reserve one extra raw/work copy and room for vector capacities. No
        // BTree node layout, max_axes-sized row, or cell-count guess is used.
        let row = p
            .raw
            .checked_mul(24)
            .and_then(|n| n.checked_add(p.work.checked_mul(17)?))
            .and_then(|n| n.checked_add(8 * std::mem::size_of::<Vec<u64>>()))
            .ok_or(StructuredUnknown::Capacity)?;
        peak = peak
            .checked_add(
                p.members
                    .checked_mul(row)
                    .ok_or(StructuredUnknown::Capacity)?,
            )
            .and_then(|n| n.checked_add(p.raw.checked_mul(8)?))
            .ok_or(StructuredUnknown::Capacity)?;
        projection = projection.max(p.projection_clone_bytes);
    }
    if total
        > settings
            .max_phase_samples
            .checked_mul(3)
            .ok_or(StructuredUnknown::Capacity)?
    {
        return Err(StructuredUnknown::Capacity);
    }
    peak = peak
        .checked_add(
            projection
                .checked_mul(2)
                .ok_or(StructuredUnknown::Capacity)?,
        )
        .ok_or(StructuredUnknown::Capacity)?;
    if peak > limit {
        return Err(StructuredUnknown::Capacity);
    }
    Ok(peak)
}

impl PhaseRange {
    fn new(points: &[(Vec<u64>, Vec<u64>)]) -> Result<Self> {
        let first = points
            .first()
            .ok_or(StructuredUnknown::InsufficientSamples)?;
        if points
            .iter()
            .any(|(raw, work)| raw.len() != first.0.len() || work.len() != first.1.len())
        {
            return Err(StructuredUnknown::WrongDomain);
        }
        let raw = JointSupport::new(points.iter().map(|p| p.0.as_slice()))?;
        let mut cells: Vec<_> = points
            .iter()
            .map(|(_, work)| (work_cell(work), 1usize))
            .collect();
        cells.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        cells.dedup_by(|current, previous| {
            if current.0 == previous.0 {
                previous.1 += current.1;
                true
            } else {
                false
            }
        });
        Ok(Self {
            raw,
            cells,
            members: points.len(),
            raw_dimensions: first.0.len(),
            work_dimensions: first.1.len(),
        })
    }

    fn cell_members(&self, cell: &[u8]) -> Option<usize> {
        self.cells
            .binary_search_by(|(key, _)| key.as_slice().cmp(cell))
            .ok()
            .map(|i| self.cells[i].1)
    }

    fn retained(&self) -> Option<usize> {
        let bytes = self.raw.retained_heap_bytes()?.checked_add(
            self.cells
                .capacity()
                .checked_mul(std::mem::size_of::<(Vec<u8>, usize)>())?,
        )?;
        self.cells
            .iter()
            .try_fold(bytes, |sum, (key, _)| sum.checked_add(key.capacity()))
    }

    fn summary(&self) -> Value {
        let mut histogram = BTreeMap::<usize, usize>::new();
        let cells: Vec<_> = self
            .cells
            .iter()
            .map(|(key, members)| {
                *histogram.entry(*members).or_default() += 1;
                let mut hash = Sha256::new();
                hash.update(b"ferrum.diagnostic.work-only-dyadic.v1\0");
                hash.update((key.len() as u64).to_le_bytes());
                hash.update(key);
                json!({"key_sha256":format!("{:x}",hash.finalize()),"members":members})
            })
            .collect();
        json!({"members":self.members,"raw_dimensions":self.raw_dimensions,
            "work_dimensions":self.work_dimensions,"cell_count":cells.len(),
            "members_per_cell_histogram":histogram,"all_cell_member_counts":cells,
            "retained_heap_bytes":self.retained(),"per_cell_minimum_enforced":false})
    }
}

#[derive(Default)]
struct RowCounts {
    total: usize,
    projection_unknown: BTreeMap<String, usize>,
    original_known: usize,
    a_in_phase: [usize; 3],
    b_in_phase: [usize; 3],
    a_intersection: usize,
    b_intersection: usize,
    a_known: usize,
    b_known: usize,
}

/// Test-only frozen inputs plus diagnostic counters. This is no model authority.
pub(crate) struct FrozenRangeAudit {
    phases: [PhaseRange; 3],
    original_parameters: [u8; 32],
    global_qualified: bool,
    // Disclosure only: no original R/Q sample is dropped or made a new gate.
    r_in_fit: [usize; 2],
    q_in_fit_residual: [usize; 2],
    rows: BTreeMap<u32, RowCounts>,
    paired: Vec<Value>,
    retained_bytes: usize,
    preflight_peak_bytes: usize,
}

impl QualifiedStructuredModelV2 {
    fn range_projection(&self, query: &StructuredQueryV2) -> Result<(Vec<u64>, Vec<u64>)> {
        let fitted = &self.calibrated.fitted;
        let projected = fitted.numerical_query(query)?;
        fitted.same_query_population(projected.input())?;
        projected.validate_for_prediction(&fitted.settings)?;
        // A uses the exact original raw support predicate. An unresolved query
        // needs its own raw support envelope; do not silently treat it as exact.
        if projected.pending.is_some() || projected.repetition_upper_sum.is_some() {
            return Err(StructuredUnknown::UnsupportedScope);
        }
        let physical = fitted
            .numerical
            .physical()
            .ok_or(StructuredUnknown::WrongProtocol)?;
        let input = prospective_input(projected.input())?;
        physical.contract.validate_input(&input)?;
        let raw = input.joint_support_coordinates().to_vec();
        let query = StructuredQueryV2::exact(input);
        let work = physical_envelope::envelope::prospective_query_upper(
            &query,
            &physical.contract.workload_domain,
        )?;
        Ok((raw, work))
    }

    pub(crate) fn diagnose_freeze_input_ranges(
        &self,
        populations: [&[StructuredNumericObservationV2]; 3],
        global_qualified: bool,
        maximum_numeric_bytes: usize,
    ) -> Result<FrozenRangeAudit> {
        let settings = &self.calibrated.fitted.settings;
        let physical = self
            .calibrated
            .fitted
            .numerical
            .physical()
            .ok_or(StructuredUnknown::WrongProtocol)?;
        let universe = physical.contract.algorithm_universe.as_ref();
        let mut shapes = [RangeShape::default(); 3];
        for (samples, shape) in populations.iter().zip(&mut shapes) {
            if samples.len() < settings.min_phase_samples
                || samples.len() > settings.max_phase_samples
                || samples.len() > 4096
            {
                return Err(StructuredUnknown::Capacity);
            }
            shape.members = samples.len();
            for sample in *samples {
                let input = &sample.input;
                // Collector-owned phase rows are already projected. Refuse to
                // budget a shorter local row then silently allocate the union.
                if input.algorithm_universe_signature() != universe.map(|u| u.signature()) {
                    return Err(StructuredUnknown::WrongDomain);
                }
                let (work, raw) = if let Some(universe) = universe {
                    universe.checked_axis_counts(input)?
                } else {
                    (
                        input.regression_axes().len(),
                        input.joint_support_coordinates().len(),
                    )
                };
                if raw == 0 || work == 0 || raw > settings.max_axes || work > settings.max_axes {
                    return Err(StructuredUnknown::Capacity);
                }
                shape.raw = shape.raw.max(raw);
                shape.work = shape.work.max(work);
                // The exact query and prospective normalization may coexist.
                // A completion positions vector can grow to all host rows.
                let clone = input
                    .retained_payload_bytes()
                    .and_then(|n| {
                        n.checked_add(
                            input
                                .physical_host_rows()
                                .len()
                                .checked_mul(std::mem::size_of::<u32>())?,
                        )
                    })
                    .ok_or(StructuredUnknown::Capacity)?;
                shape.projection_clone_bytes = shape.projection_clone_bytes.max(clone);
            }
        }
        let model_bytes = self
            .retained_payload_bytes()
            .ok_or(StructuredUnknown::Capacity)?;
        let preflight_peak_bytes =
            range_peak_bytes(&shapes, settings, model_bytes, maximum_numeric_bytes)?;
        let mut all_points = Vec::with_capacity(3);
        for samples in populations {
            let points = samples
                .iter()
                .map(|s| self.range_projection(&StructuredQueryV2::exact(s.input.clone())))
                .collect::<Result<Vec<_>>>()?;
            if points
                .iter()
                .any(|(raw, work)| raw.len() > settings.max_axes || work.len() > settings.max_axes)
            {
                return Err(StructuredUnknown::Capacity);
            }
            all_points.push(points);
        }
        let phases = [
            PhaseRange::new(&all_points[0])?,
            PhaseRange::new(&all_points[1])?,
            PhaseRange::new(&all_points[2])?,
        ];
        let mut r_in_fit = [0; 2];
        for (raw, work) in &all_points[1] {
            r_in_fit[0] += usize::from(phases[0].raw.contains(raw));
            r_in_fit[1] += usize::from(phases[0].cell_members(&work_cell(work)).is_some());
        }
        let mut q_in_fit_residual = [0; 2];
        for (raw, work) in &all_points[2] {
            q_in_fit_residual[0] += usize::from(phases[..2].iter().all(|p| p.raw.contains(raw)));
            let cell = work_cell(work);
            q_in_fit_residual[1] +=
                usize::from(phases[..2].iter().all(|p| p.cell_members(&cell).is_some()));
        }
        let retained_bytes = phases
            .iter()
            .try_fold(std::mem::size_of::<FrozenRangeAudit>(), |n, p| {
                n.checked_add(p.retained()?)
            })
            .ok_or(StructuredUnknown::Capacity)?;
        let temporary_base = all_points
            .capacity()
            .checked_mul(std::mem::size_of::<Vec<(Vec<u64>, Vec<u64>)>>())
            .ok_or(StructuredUnknown::Capacity)?;
        let temporary_bytes = all_points
            .iter()
            .try_fold(temporary_base, |sum, phase| {
                phase.iter().try_fold(
                    sum.checked_add(
                        phase
                            .capacity()
                            .checked_mul(std::mem::size_of::<(Vec<u64>, Vec<u64>)>())?,
                    )?,
                    |n, (raw, work)| {
                        n.checked_add(
                            raw.capacity()
                                .checked_add(work.capacity())?
                                .checked_mul(8)?,
                        )
                    },
                )
            })
            .ok_or(StructuredUnknown::Capacity)?;
        if retained_bytes
            .checked_add(model_bytes)
            .and_then(|n| n.checked_add(temporary_bytes))
            .is_none_or(|n| n > preflight_peak_bytes || n > maximum_numeric_bytes)
        {
            return Err(StructuredUnknown::Capacity);
        }
        Ok(FrozenRangeAudit {
            phases,
            original_parameters: self.parameters_signature(),
            global_qualified,
            r_in_fit,
            q_in_fit_residual,
            rows: BTreeMap::new(),
            paired: Vec::new(),
            retained_bytes,
            preflight_peak_bytes,
        })
    }
}

impl FrozenRangeAudit {
    pub(crate) fn observe_future(
        &mut self,
        model: &QualifiedStructuredModelV2,
        query: &StructuredQueryV2,
        call: u64,
        original_known: bool,
        paired: bool,
    ) {
        let qualified = self.global_qualified;
        let counts = self.rows.entry(query.owner().rows).or_default();
        counts.total += 1;
        counts.original_known += usize::from(original_known);
        let detail = match model.range_projection(query) {
            Ok((raw, work)) => {
                let cell = work_cell(&work);
                let a = self.phases.each_ref().map(|p| p.raw.contains(&raw));
                let b_members = self.phases.each_ref().map(|p| p.cell_members(&cell));
                let b = b_members.map(|n| n.is_some());
                for i in 0..3 {
                    counts.a_in_phase[i] += usize::from(a[i]);
                    counts.b_in_phase[i] += usize::from(b[i]);
                }
                let a_all = a.iter().all(|v| *v);
                let b_all = b.iter().all(|v| *v);
                counts.a_intersection += usize::from(a_all);
                counts.b_intersection += usize::from(b_all);
                counts.a_known += usize::from(qualified && original_known && a_all);
                counts.b_known += usize::from(qualified && original_known && b_all);
                if paired {
                    json!({"call":call,"rows":query.owner().rows,"original_gate_known":original_known,
                        "a_in_F_R_Q":a,"b_members_in_F_R_Q":b_members,
                        "a_joint_support_outside":self.phases.each_ref().map(|p| p.raw.diagnose(&raw).map(|e|format!("{e:?}"))),
                        "a_input_intersection":a_all,"b_input_intersection":b_all,
                        "a_known_after_all_gates":qualified && original_known && a_all,
                        "b_known_after_all_gates":qualified && original_known && b_all})
                } else {
                    Value::Null
                }
            }
            Err(error) => {
                *counts
                    .projection_unknown
                    .entry(format!("{error:?}"))
                    .or_default() += 1;
                json!({"call":call,"rows":query.owner().rows,"original_gate_known":original_known,
                    "projection_unknown":format!("{error:?}"),"a_known_after_all_gates":false,"b_known_after_all_gates":false})
            }
        };
        if paired {
            self.paired.push(detail);
        }
    }

    pub(crate) fn summary(&self) -> Value {
        let rows: Vec<_> = self
            .rows
            .iter()
            .map(|(rows, c)| {
                json!({"rows":rows,"future_queries":c.total,
            "projection_unknown":c.projection_unknown,"original_gate_known":c.original_known,
            "a_in_F_R_Q":c.a_in_phase,"b_in_F_R_Q":c.b_in_phase,
            "a_input_intersection":c.a_intersection,"b_input_intersection":c.b_intersection,
            "a_known_after_all_gates":c.a_known,"b_known_after_all_gates":c.b_known})
            })
            .collect();
        json!({"original_parameters_sha256":self.original_parameters,"global_numeric_qualified":self.global_qualified,
            "candidate_a":"original JointSupport on prospective complete raw input support; phase intersection",
            "candidate_b":"complete declared work tuple dyadic bands; zero exact; phase intersection; no per-cell fit or quota",
            "b_domain_note":"A predefined dyadic band includes unobserved magnitudes in that band; it is not an observed convex hull or a hardware guarantee.",
            "phase_order":["fit","residual","qualification"],"phases":self.phases.each_ref().map(PhaseRange::summary),
            "r_in_fit_A_B":self.r_in_fit,"q_in_fit_residual_A_B":self.q_in_fit_residual,
            "prior_phase_membership_is_diagnostic_only":true,
            "frozen_domain_owned_bytes":self.retained_bytes,"preflight_peak_bytes":self.preflight_peak_bytes,
            "budget_scope":"diagnostic temporary/frozen range arrays plus unchanged model and bounded input clone scratch; original collector, replay, audit output and allocator/RSS not counted",
            "future_rows":rows,"paired_calls":self.paired,
            "scope":"test-only input range diagnostic; all original phases kept; no production publication or mathematical guarantee"})
    }
}

#[test]
fn original_joint_range_and_work_cells_reject_unobserved_joint_combinations() {
    let points = vec![
        (vec![1, 128, 1], vec![1, 128, 1]),
        (vec![1, 1, 128], vec![1, 1, 128]),
    ];
    let range = PhaseRange::new(&points).unwrap();
    assert!(!range.raw.contains(&[1, 128, 128]));
    assert!(range.cell_members(&work_cell(&[1, 128, 128])).is_none());
    assert!(range.raw.contains(&[1, 128, 1]));
    assert_eq!(range.cell_members(&work_cell(&[1, 128, 1])), Some(1));
    assert_eq!(range.cell_members(&work_cell(&[1, 255, 1])), Some(1));
    assert!(!range.raw.contains(&[1, 255, 1]));
    assert!(range.cell_members(&work_cell(&[1, 256, 1])).is_none());
    assert!(range.cell_members(&work_cell(&[1, 0, 1])).is_none());
    let before = range.summary();
    for point in [[1, 128, 1], [1, 128, 128], [1, 255, 1]] {
        let _ = range.raw.contains(&point);
        let _ = range.cell_members(&work_cell(&point));
    }
    assert_eq!(
        range.summary(),
        before,
        "queries cannot rebuild or extend frozen domains"
    );
}

#[test]
fn range_preflight_measures_actual_raw_and_work_capacity_before_freezing() {
    let mut settings = StructuredSettingsV2 {
        min_phase_samples: 8,
        max_phase_samples: 32,
        max_axes: 1024,
        ..Default::default()
    };
    let shapes = [8, 9, 10].map(|members| RangeShape {
        members,
        raw: 67,
        work: 16,
        projection_clone_bytes: 0,
    });
    let model_bytes = 4096;
    let required = range_peak_bytes(&shapes, &settings, model_bytes, usize::MAX).unwrap();
    assert_eq!(
        range_peak_bytes(&shapes, &settings, model_bytes, required),
        Ok(required)
    );
    assert_eq!(
        range_peak_bytes(&shapes, &settings, model_bytes, required - 1),
        Err(StructuredUnknown::Capacity)
    );
    settings.max_axes = 128;
    assert_eq!(
        range_peak_bytes(&shapes, &settings, model_bytes, required),
        Ok(required),
        "a looser coordinate ceiling is not an allocation"
    );

    // Retain the actual temporary rows while the real JointSupport and cell
    // vectors coexist. Raw support is deliberately wider than declared work.
    let points: Vec<Vec<(Vec<u64>, Vec<u64>)>> = shapes
        .iter()
        .map(|s| {
            (0..s.members)
                .map(|i| (vec![i as u64; s.raw], vec![i as u64; s.work]))
                .collect()
        })
        .collect();
    let phases: Vec<_> = points.iter().map(|p| PhaseRange::new(p).unwrap()).collect();
    let mut actual = model_bytes
        + std::mem::size_of::<FrozenRangeAudit>()
        + points.capacity() * std::mem::size_of::<Vec<(Vec<u64>, Vec<u64>)>>();
    for (p, phase) in points.iter().zip(&phases) {
        actual +=
            p.capacity() * std::mem::size_of::<(Vec<u64>, Vec<u64>)>() + phase.retained().unwrap();
        actual += p
            .iter()
            .map(|(raw, work)| (raw.capacity() + work.capacity()) * 8)
            .sum::<usize>();
    }
    assert!(
        actual <= required,
        "the preflight must cover simultaneously live owned vectors"
    );
    settings.max_axes = 66;
    assert_eq!(
        range_peak_bytes(&shapes, &settings, model_bytes, usize::MAX),
        Err(StructuredUnknown::Capacity),
        "checking work=16 alone would miss raw=67"
    );
    settings.max_axes = 128;
    let mut too_many = shapes;
    too_many[2].members = settings.max_phase_samples + 1;
    assert_eq!(
        range_peak_bytes(&too_many, &settings, model_bytes, usize::MAX),
        Err(StructuredUnknown::Capacity)
    );
    let mut zero = shapes;
    zero[1].work = 0;
    assert_eq!(
        range_peak_bytes(&zero, &settings, model_bytes, usize::MAX),
        Err(StructuredUnknown::Capacity)
    );
}
