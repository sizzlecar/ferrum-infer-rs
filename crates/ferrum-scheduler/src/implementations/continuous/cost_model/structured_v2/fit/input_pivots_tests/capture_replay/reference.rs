//! Independent demand measurement after exact original-ledger replay. Every
//! reference uses the unchanged kernel, settings, anchors and scratch bound.
//! Its fresh MAX allowance is a diagnostic, never a production allowance.
use super::super::super::super::{input_geometry_core, FitRow, GeometryWork};
use super::*;

#[derive(Serialize)]
pub(super) struct ReferenceCall<'a> {
    ordinal: usize,
    population_index: usize,
    population_key: &'a serde_json::Value,
    original_case_indices: &'a [usize],
    original_complete: bool,
    original_charged_visits: u64,
    complete: bool,
    rank: Option<usize>,
    anchor_rank: Option<usize>,
    pivot_indices: Vec<usize>,
    final_selected_cases: Vec<usize>,
    reference_visits: u64,
    reference_exhausted: bool,
    error: Option<String>,
    normalization_scale_bits: Option<Vec<u64>>,
    normalized_call_sha256: Option<[u8; 32]>,
    extra_original_core_visits: u64,
}
#[derive(Serialize)]
pub(super) struct NormalizedGroup {
    signature: [u8; 32],
    ordinals: Vec<usize>,
    /// Each listed call differs in raw bits from this group's first call.
    raw_different_from_first: Vec<usize>,
    reference_visits: Vec<u64>,
}
#[derive(Serialize)]
pub(super) struct Measurement<'a> {
    diagnostic_reference_allowance_per_call: u64,
    calls: Vec<ReferenceCall<'a>>,
    successful_calls: usize,
    failed_calls: usize,
    measured_reference_visits: u128,
    /// Present only if every original call completed under its other unchanged
    /// constraints. Failed partial scans do not establish complete demand.
    full_required_visits: Option<u128>,
    full_requirement_minus_original_charge: Option<u128>,
    full_requirement_above_original_limit: Option<u128>,
    originally_incomplete_calls_full_reference_visits: Option<u128>,
    extra_original_core_passes: usize,
    extra_original_core_visits: u128,
    normalized_bit_groups: Vec<NormalizedGroup>,
    excluded_costs: [&'static str; 5],
    conclusion: &'static str,
}

struct NormalizedBody<'a> {
    ordinal: usize,
    matrix: &'a Matrix,
    bits: Vec<Vec<u64>>,
}

#[derive(Serialize)]
pub(super) struct CandidateCall {
    ordinal: usize,
    visits_before: u64,
    visits_after: u64,
    work_visits: u64,
    rank: usize,
    anchor_rank: usize,
    pivot_indices: Vec<usize>,
    final_selected_cases: Vec<usize>,
}

#[derive(Serialize)]
pub(super) struct CandidateMeasurement {
    maximum_visits: u64,
    used_visits: u64,
    exhausted: bool,
    all_results_match_complete_original_reference: bool,
    original_per_call_scratch_limits_preserved: bool,
    calls: Vec<CandidateCall>,
}

/// Exercise the actual candidate API once, in original order, against one
/// original allowance. The independent original kernel supplies every result;
/// the original exhausted run is neither restarted nor treated as a reference
/// for previously uncomputed numerical geometry.
pub(super) fn evaluate_candidate(
    verified: &Verified,
    reference: &Measurement<'_>,
) -> AuditResult<CandidateMeasurement> {
    ensure!(
        reference.failed_calls == 0 && reference.calls.len() == verified.calls.len(),
        "candidate comparison requires complete independent original references"
    );
    let mut work = StructuredInputGeometryWorkV1::new(
        NonZeroU64::new(verified.limit).context("zero original allowance")?,
    );
    let mut calls = Vec::new();
    for (call, expected) in verified.calls.iter().zip(&reference.calls) {
        let matrix = &call.matrix;
        let decoded: Vec<Vec<f64>> = matrix
            .axis_bits
            .iter()
            .map(|row| row.iter().map(|&bits| f64::from_bits(bits)).collect())
            .collect();
        let rows: Vec<_> = decoded.iter().map(Vec::as_slice).collect();
        let before = work.visits();
        let geometry = input_geometry_pivots_v1(
            &rows,
            &matrix.mandatory_anchors,
            &matrix.settings,
            &mut work,
            matrix.maximum_scratch_bytes,
        )
        .map_err(|reason| {
            anyhow::anyhow!(
                "candidate ordinal {} failed {reason:?}; visits={} exhausted={}",
                matrix.ordinal,
                work.visits(),
                work.exhausted()
            )
        })?;
        ensure!(
            Some(geometry.rank) == expected.rank
                && Some(geometry.anchor_rank) == expected.anchor_rank
                && geometry.pivot_indices == expected.pivot_indices,
            "candidate numerical result differs at {}",
            matrix.ordinal
        );
        let mut selected: Vec<_> = matrix
            .mandatory_anchors
            .iter()
            .map(|&index| matrix.cases[index])
            .collect();
        for &pivot in &geometry.pivot_indices[geometry.anchor_rank..] {
            let original = matrix.cases[pivot];
            if !selected.contains(&original) {
                selected.push(original);
            }
        }
        selected.sort_unstable();
        ensure!(
            selected == expected.final_selected_cases
                && geometry.work_visits == work.visits() - before,
            "candidate mapping or work differs at {}",
            matrix.ordinal
        );
        calls.push(CandidateCall {
            ordinal: matrix.ordinal,
            visits_before: before,
            visits_after: work.visits(),
            work_visits: geometry.work_visits,
            rank: geometry.rank,
            anchor_rank: geometry.anchor_rank,
            pivot_indices: geometry.pivot_indices,
            final_selected_cases: selected,
        });
    }
    Ok(CandidateMeasurement {
        maximum_visits: work.maximum_visits(),
        used_visits: work.visits(),
        exhausted: work.exhausted(),
        all_results_match_complete_original_reference: true,
        original_per_call_scratch_limits_preserved: true,
        calls,
    })
}
fn normalized_signature(bits: &[Vec<u64>], matrix: &Matrix) -> AuditResult<[u8; 32]> {
    let mut h = Sha256::new();
    h.update(b"cold-input-original-normalized-bits-v1\0");
    h.update(u64::try_from(bits.len())?.to_le_bytes());
    for row in bits {
        h.update(u64::try_from(row.len())?.to_le_bytes());
        for value in row {
            h.update(value.to_le_bytes());
        }
    }
    h.update(u64::try_from(matrix.mandatory_anchors.len())?.to_le_bytes());
    for &anchor in &matrix.mandatory_anchors {
        h.update(u64::try_from(anchor)?.to_le_bytes());
    }
    h.update(u64::try_from(matrix.maximum_scratch_bytes)?.to_le_bytes());
    h.update(serde_json::to_vec(&matrix.settings)?);
    Ok(h.finalize().into())
}
fn same_normalized(a: &NormalizedBody<'_>, bits: &[Vec<u64>], b: &Matrix) -> AuditResult<bool> {
    Ok(a.bits == bits
        && a.matrix.mandatory_anchors == b.mandatory_anchors
        && a.matrix.maximum_scratch_bytes == b.maximum_scratch_bytes
        && serde_json::to_vec(&a.matrix.settings)? == serde_json::to_vec(&b.settings)?)
}

pub(super) fn measure(verified: &Verified) -> AuditResult<Measurement<'_>> {
    let mut calls = Vec::new();
    let mut bodies = Vec::<NormalizedBody<'_>>::new();
    let mut groups = Vec::<NormalizedGroup>::new();
    let mut signatures = BTreeMap::<[u8; 32], Vec<usize>>::new();
    let mut measured = 0_u128;
    let mut incomplete_demand = 0_u128;
    let mut extra_visits = 0_u128;
    let mut successful = 0_usize;
    for call in &verified.calls {
        let matrix = &call.matrix;
        let decoded: Vec<Vec<f64>> = matrix
            .axis_bits
            .iter()
            .map(|row| row.iter().map(|&bits| f64::from_bits(bits)).collect())
            .collect();
        let rows: Vec<_> = decoded.iter().map(Vec::as_slice).collect();
        let mut work = StructuredInputGeometryWorkV1::new(NonZeroU64::new(u64::MAX).unwrap());
        let result = input_geometry_pivots_original_v1(
            &rows,
            &matrix.mandatory_anchors,
            &matrix.settings,
            &mut work,
            matrix.maximum_scratch_bytes,
        );
        measured = measured
            .checked_add(u128::from(work.visits()))
            .context("reference demand overflow")?;
        let mut entry = ReferenceCall {
            ordinal: matrix.ordinal,
            population_index: call.population.population_index,
            population_key: &call.population.key,
            original_case_indices: &matrix.cases,
            original_complete: call.result.audit.complete,
            original_charged_visits: call.result.audit.visits,
            complete: result.is_ok(),
            rank: None,
            anchor_rank: None,
            pivot_indices: Vec::new(),
            final_selected_cases: matrix
                .mandatory_anchors
                .iter()
                .map(|&i| matrix.cases[i])
                .collect(),
            reference_visits: work.visits(),
            reference_exhausted: work.exhausted(),
            error: None,
            normalization_scale_bits: None,
            normalized_call_sha256: None,
            extra_original_core_visits: 0,
        };
        let geometry = match result {
            Ok(value) => value,
            Err(reason) => {
                entry.error = Some(format!("{reason:?}"));
                // There is no successful original core output to group. Do
                // not invent normalized data for a failed rank/scratch scan.
                calls.push(entry);
                continue;
            }
        };
        successful += 1;
        if !call.result.audit.complete {
            incomplete_demand = incomplete_demand
                .checked_add(u128::from(work.visits()))
                .context("incomplete demand overflow")?;
        }
        entry.rank = Some(geometry.rank);
        entry.anchor_rank = Some(geometry.anchor_rank);
        for &pivot in &geometry.pivot_indices[geometry.anchor_rank..] {
            let original = matrix.cases[pivot];
            if !entry.final_selected_cases.contains(&original) {
                entry.final_selected_cases.push(original);
            }
        }
        entry.final_selected_cases.sort_unstable();
        entry.pivot_indices = geometry.pivot_indices;
        if call.result.audit.complete {
            ensure!(
                entry.rank == call.result.audit.candidate_rank
                    && entry.rank == call.result.audit.selected_rank
                    && entry.final_selected_cases == call.result.final_selected_cases
                    && entry.reference_visits == call.result.audit.visits,
                "reference changed an originally complete call {}",
                matrix.ordinal
            );
        }

        // The existing public wrapper exposes pivots, not normalized rows.
        // Run the SAME core again only for diagnostic access to its real rows;
        // retain and report this extra pass separately from reference demand.
        let borrowed: Vec<_> = rows
            .iter()
            .map(|basis| FitRow { basis, wall_ns: 0 })
            .collect();
        let mut core_work = GeometryWork {
            used: 0,
            limit: u64::MAX,
            exhausted: false,
        };
        let core = input_geometry_core(
            &borrowed,
            &matrix.settings,
            Some(&mut core_work),
            false,
            &matrix.mandatory_anchors,
        )
        .map_err(|reason| {
            anyhow::anyhow!(
                "original wrapper/core disagreement at {}: {reason:?}",
                matrix.ordinal
            )
        })?;
        ensure!(
            core.basis.len() == geometry.rank && core.pivot_indices == entry.pivot_indices,
            "original core pivots differ at {}",
            matrix.ordinal
        );
        let validation = rows
            .len()
            .checked_mul(rows[0].len())
            .context("validation count overflow")?;
        ensure!(
            !core_work.exhausted
                && core_work.used.checked_add(u64::try_from(validation)?) == Some(work.visits()),
            "original wrapper/core work differs at {}",
            matrix.ordinal
        );
        entry.extra_original_core_visits = core_work.used;
        extra_visits = extra_visits
            .checked_add(u128::from(core_work.used))
            .context("extra diagnostic work overflow")?;
        entry.normalization_scale_bits =
            Some(core.scale.iter().map(|value| value.to_bits()).collect());
        let bits: Vec<Vec<u64>> = core
            .rows
            .iter()
            .map(|row| row.iter().map(|value| value.to_bits()).collect())
            .collect();
        drop(core);
        let signature = normalized_signature(&bits, matrix)?;
        entry.normalized_call_sha256 = Some(signature);
        let candidates = signatures.entry(signature).or_default();
        let mut group = None;
        for &index in candidates.iter() {
            if same_normalized(&bodies[index], &bits, matrix)? {
                group = Some(index);
                break;
            }
        }
        let index = match group {
            Some(index) => index,
            None => {
                let index = groups.len();
                bodies.push(NormalizedBody {
                    ordinal: matrix.ordinal,
                    matrix,
                    bits,
                });
                groups.push(NormalizedGroup {
                    signature,
                    ordinals: Vec::new(),
                    raw_different_from_first: Vec::new(),
                    reference_visits: Vec::new(),
                });
                candidates.push(index);
                index
            }
        };
        if !same_call(matrix, &verified.calls[bodies[index].ordinal].matrix)? {
            groups[index].raw_different_from_first.push(matrix.ordinal);
        }
        groups[index].ordinals.push(matrix.ordinal);
        groups[index].reference_visits.push(work.visits());
        calls.push(entry);
    }
    let all_complete = successful == calls.len();
    let full_required_visits = all_complete.then_some(measured);
    let full_requirement_minus_original_charge = full_required_visits
        .map(|total| {
            total
                .checked_sub(u128::from(verified.visits))
                .context("full reference below original charge")
        })
        .transpose()?;
    Ok(Measurement {
        diagnostic_reference_allowance_per_call: u64::MAX,
        successful_calls: successful,
        failed_calls: calls.len() - successful,
        calls,
        measured_reference_visits: measured,
        full_required_visits,
        full_requirement_minus_original_charge,
        full_requirement_above_original_limit: full_required_visits.map(|total| total.saturating_sub(u128::from(verified.limit))),
        originally_incomplete_calls_full_reference_visits: all_complete.then_some(incomplete_demand),
        extra_original_core_passes: successful,
        extra_original_core_visits: extra_visits,
        normalized_bit_groups: groups.into_iter().filter(|group| group.ordinals.len() > 1).collect(),
        excluded_costs: [
            "Fresh MAX reference ledgers are diagnostic only; no original allowance is reset or refunded.",
            "The extra original-core decomposition is measured separately; it is not required demand or a proposed implementation.",
            "f64 decoding, normalized-bit copies, hashing, exact comparisons and output serialization are not charged to the original visit ledger.",
            "Offline group storage, original/normalized matrices and simultaneous temporary allocation peaks are not production cache memory accounting.",
            "A requirement-minus-charge difference is arithmetic, not a legal retry allowance or feasible saving; original failed work remains spent.",
        ],
        conclusion: "Same-kernel independent demand and bit-exact normalized-input equivalence only. Settings/scratch rejection remains rejection. No cache mechanism, numerical qualification, reduced production work allowance, startup-time or memory feasibility is established.",
    })
}
