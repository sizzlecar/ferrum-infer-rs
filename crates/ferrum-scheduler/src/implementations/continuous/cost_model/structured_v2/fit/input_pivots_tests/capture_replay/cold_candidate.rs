//! Explicit offline experiment, not a replacement for a production kernel.
//! Exact row-bit groups share only cold arithmetic; all mandatory cases remain.
use super::super::super::super::{
    dot, input_geometry_core, norm, orthogonalize, subtract, FitRow, GeometryWork, DEPENDENT,
    MIN_PIVOT,
};
use super::*;
use std::mem::size_of;

#[derive(Debug, Default, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum Mode {
    #[default]
    Disabled,
    AnchoredReadinessV2WidthCostV1,
}

struct Group {
    hash: u64,
    row: usize,
    width: usize,
    anchor: bool,
    selected: bool,
}
struct Scratch {
    groups: Vec<Group>,
    anchor_mask: Vec<u8>,
    residuals: Vec<Vec<f64>>,
    norms: Vec<f64>,
    basis: Vec<Vec<f64>>,
    scale: Vec<f64>,
    pivots: Vec<usize>,
}
struct Geometry {
    pivots: Vec<usize>,
    selected_rows: Vec<usize>,
    anchor_rank: usize,
    unique_rows: usize,
    scratch_bytes: usize,
    largest_norm: f64,
    final_maximum_residual: f64,
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or(StructuredUnknown::Capacity)
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or(StructuredUnknown::Capacity)
}
fn index_bytes(n: usize) -> Result<usize> {
    add(
        size_of::<Scratch>(),
        mul(n, size_of::<Group>() + size_of::<u8>())?,
    )
}
fn scratch_bytes(n: usize, unique: usize, d: usize, rank: usize) -> Result<usize> {
    let r = rank.min(unique).min(d);
    let coordinates = mul(add(add(unique, r)?, 3)?, d)?;
    let mut bytes = add(index_bytes(n)?, mul(coordinates, size_of::<f64>())?)?;
    bytes = add(bytes, mul(add(unique, r)?, size_of::<Vec<f64>>())?)?;
    bytes = add(bytes, mul(unique, size_of::<f64>())?)?;
    bytes = add(bytes, mul(r, size_of::<usize>())?)?;
    bytes = add(bytes, mul(n, size_of::<usize>())?)?;
    // One current pivot plus a simultaneous refresh replacement, in addition
    // to the scale/cache/basis coordinates above. Their Vec headers are live.
    bytes = add(bytes, 2 * size_of::<Vec<f64>>())?;
    // Output construction may coexist with cache teardown; reserve its header.
    add(bytes, size_of::<Geometry>())
}

fn checked_norm(row: &[f64]) -> Result<f64> {
    let value = norm(row);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(StructuredUnknown::Numerical)
    }
}
/// Same original-row two-pass residual and charge as readiness_v2. Cache
/// vectors are never the authority for accepting a new direction.
fn residual(
    row: &[f64],
    scale: &[f64],
    basis: &[Vec<f64>],
    work: &mut GeometryWork,
) -> Result<(Vec<f64>, f64)> {
    work.charge(mul(add(mul(basis.len(), 4)?, 2)?, row.len())?)?;
    let mut out: Vec<_> = row.iter().zip(scale).map(|(x, s)| x / s).collect();
    orthogonalize(&mut out, basis);
    let length = checked_norm(&out)?;
    Ok((out, length))
}

fn hash_row(bits: &[u64], work: &mut GeometryWork) -> Result<u64> {
    // Byte-wise FNV only narrows candidates. Every hash collision is followed
    // by exact original u64 comparison; no digest grants equality.
    work.charge(mul(bits.len(), 8)?)?;
    let mut hash = 0xcbf29ce484222325_u64;
    for bits in bits {
        for byte in bits.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    Ok(hash)
}
fn equal_row(a: &[u64], b: &[u64], work: &mut GeometryWork) -> Result<bool> {
    for (a, b) in a.iter().zip(b) {
        work.charge(2)?;
        if a != b {
            return Ok(false);
        }
    }
    Ok(true)
}

fn refresh(rows: &[&[f64]], s: &mut Scratch, work: &mut GeometryWork) -> Result<()> {
    for (i, group) in s.groups.iter().enumerate() {
        let (row, length) = residual(rows[group.row], &s.scale, &s.basis, work)?;
        s.residuals[i] = row;
        s.norms[i] = length;
    }
    Ok(())
}
fn pick(
    s: &Scratch,
    anchors: bool,
    largest_norm: f64,
    work: &mut GeometryWork,
) -> Result<Option<usize>> {
    work.charge(mul(s.groups.len(), 4)?)?;
    let mut best: Option<usize> = None;
    for (i, group) in s.groups.iter().enumerate() {
        if group.selected || (anchors && !group.anchor) || s.norms[i] / largest_norm < MIN_PIVOT {
            continue;
        }
        if best.is_none_or(|old| {
            let prior = &s.groups[old];
            (!anchors && group.width < prior.width)
                || ((anchors || group.width == prior.width)
                    && (s.norms[i] > s.norms[old]
                        || (s.norms[i] == s.norms[old] && group.row < prior.row)))
        }) {
            best = Some(i);
        }
    }
    Ok(best)
}
fn maximum(s: &Scratch, anchors: bool, work: &mut GeometryWork) -> Result<f64> {
    work.charge(mul(s.groups.len(), 2)?)?;
    Ok(s.groups
        .iter()
        .zip(&s.norms)
        .filter(|(g, _)| !anchors || g.anchor)
        .map(|(_, &v)| v)
        .fold(0., f64::max))
}

fn geometry(
    matrix: &Matrix,
    rows: &[&[f64]],
    widths: &[usize],
    work: &mut GeometryWork,
    maximum_scratch_bytes: usize,
) -> Result<Geometry> {
    geometry_mapped(
        matrix,
        rows,
        widths,
        None,
        &matrix.mandatory_anchors,
        work,
        maximum_scratch_bytes,
    )
}

#[allow(clippy::too_many_arguments)]
fn geometry_mapped(
    matrix: &Matrix,
    rows: &[&[f64]],
    widths: &[usize],
    source_rows: Option<&[usize]>,
    mandatory_anchors: &[usize],
    work: &mut GeometryWork,
    maximum_scratch_bytes: usize,
) -> Result<Geometry> {
    if work.exhausted {
        return Err(StructuredUnknown::Capacity);
    }
    matrix.settings.validate()?;
    let n = rows.len();
    let d = rows
        .first()
        .ok_or(StructuredUnknown::InsufficientSamples)?
        .len();
    if d == 0 || d > matrix.settings.max_axes {
        return Err(StructuredUnknown::Capacity);
    }
    if widths.len() != n
        || source_rows.map_or(matrix.axis_bits.len() != n, |indices| {
            indices.len() != n
                || indices.windows(2).any(|p| p[0] >= p[1])
                || indices.last().is_some_and(|&i| i >= matrix.axis_bits.len())
        })
    {
        return Err(StructuredUnknown::InvalidInput);
    }
    let bits = |i: usize| matrix.axis_bits[source_rows.map_or(i, |indices| indices[i])].as_slice();
    if index_bytes(n)? > maximum_scratch_bytes {
        return Err(StructuredUnknown::Capacity);
    }
    work.charge(add(mul(n, d)?, add(n, mandatory_anchors.len())?)?)?;
    if widths.contains(&0)
        || rows.iter().any(|row| {
            row.len() != d
                || row
                    .iter()
                    .any(|v| !v.is_finite() || *v < 0. || *v > (1u64 << 53) as f64)
        })
        || (0..n).any(|i| bits(i).len() != d)
        || mandatory_anchors.windows(2).any(|p| p[0] >= p[1])
        || mandatory_anchors.last().is_some_and(|&a| a >= n)
    {
        return Err(StructuredUnknown::InvalidInput);
    }
    let mut s = Scratch {
        groups: Vec::with_capacity(n),
        anchor_mask: vec![0; n],
        residuals: Vec::new(),
        norms: Vec::new(),
        basis: Vec::new(),
        scale: Vec::new(),
        pivots: Vec::new(),
    };
    for &anchor in mandatory_anchors {
        s.anchor_mask[anchor] = 1;
    }
    for i in 0..n {
        let hash = hash_row(bits(i), work)?;
        let anchor = s.anchor_mask[i] != 0;
        let mut found = None;
        for (j, group) in s.groups.iter().enumerate() {
            work.charge(2)?;
            if group.hash == hash
                && group.anchor == anchor
                && equal_row(bits(group.row), bits(i), work)?
            {
                found = Some(j);
                break;
            }
        }
        if let Some(j) = found {
            work.charge(2)?;
            if widths[i] < s.groups[j].width {
                s.groups[j].row = i;
                s.groups[j].width = widths[i];
            }
        } else {
            s.groups.push(Group {
                hash,
                row: i,
                width: widths[i],
                anchor,
                selected: false,
            });
        }
    }
    let unique = s.groups.len();
    let maximum_rank = matrix.settings.max_rank.min(unique).min(d);
    let bytes = scratch_bytes(n, unique, d, maximum_rank)?;
    if bytes > maximum_scratch_bytes {
        return Err(StructuredUnknown::Capacity);
    }
    work.charge(add(
        add(mul(n, d)?, mul(mul(unique, d)?, 3)?)?,
        add(d, unique)?,
    )?)?;
    s.scale = vec![1.; d];
    for row in rows {
        for (maximum, &v) in s.scale.iter_mut().zip(*row) {
            *maximum = maximum.max(v);
        }
    }
    s.residuals = Vec::with_capacity(unique);
    s.norms = Vec::with_capacity(unique);
    s.basis = Vec::with_capacity(maximum_rank);
    s.pivots = Vec::with_capacity(maximum_rank);
    for group in &s.groups {
        let row: Vec<_> = rows[group.row]
            .iter()
            .zip(&s.scale)
            .map(|(v, scale)| v / scale)
            .collect();
        s.norms.push(checked_norm(&row)?);
        s.residuals.push(row);
    }
    let largest_norm = s.norms.iter().copied().fold(0., f64::max);
    if largest_norm == 0. {
        return Err(StructuredUnknown::Numerical);
    }
    let mut anchors = !mandatory_anchors.is_empty();
    let mut anchor_rank = 0;
    let final_maximum_residual;
    loop {
        let choice = pick(&s, anchors, largest_norm, work)?;
        let probe = choice
            .map(|i| residual(rows[s.groups[i].row], &s.scale, &s.basis, work))
            .transpose()?;
        let (candidate, mut pivot, length) = match (choice, probe) {
            (Some(i), Some((pivot, length))) if length / largest_norm >= MIN_PIVOT => {
                (i, pivot, length)
            }
            _ => {
                // Every original row is represented in an exactly equal bit
                // group with the same anchor role. Recompute every such group
                // from its original row, including selected groups, to certify
                // phase transition or termination. No cached small norm ends it.
                refresh(rows, &mut s, work)?;
                let max = maximum(&s, anchors, work)?;
                if max / largest_norm <= DEPENDENT {
                    if anchors {
                        anchors = false;
                        anchor_rank = s.basis.len();
                        continue;
                    }
                    final_maximum_residual = max;
                    break;
                }
                if max / largest_norm < MIN_PIVOT {
                    return Err(StructuredUnknown::IllConditioned);
                }
                let i = pick(&s, anchors, largest_norm, work)?
                    .ok_or(StructuredUnknown::IllConditioned)?;
                work.charge(d)?;
                (i, s.residuals[i].clone(), s.norms[i])
            }
        };
        if s.basis.len() == maximum_rank {
            return Err(StructuredUnknown::Capacity);
        }
        work.charge(mul(d, add(s.basis.len(), 2)?)?)?;
        for v in &mut pivot {
            *v /= length;
        }
        if (dot(&pivot, &pivot) - 1.).abs() > DEPENDENT
            || s.basis.iter().any(|q| dot(&pivot, q).abs() > DEPENDENT)
        {
            return Err(StructuredUnknown::IllConditioned);
        }
        for (row, length) in s.residuals.iter_mut().zip(&mut s.norms) {
            work.charge(mul(d, 5)?)?;
            for _ in 0..2 {
                let component = dot(row, &pivot);
                subtract(row, &pivot, component);
            }
            *length = checked_norm(row)?;
        }
        s.groups[candidate].selected = true;
        s.pivots.push(s.groups[candidate].row);
        s.basis.push(pivot);
    }
    if s.basis.is_empty() {
        return Err(StructuredUnknown::Numerical);
    }
    // The original matrix's cases are sorted unique. Preserve all anchors,
    // including dependent/duplicate ones, then add only real pivot rows.
    work.charge(add(mul(n, 2)?, s.pivots.len())?)?;
    for &pivot in &s.pivots {
        s.anchor_mask[pivot] |= 2;
    }
    let mut selected_rows = Vec::with_capacity(n);
    for (i, &flag) in s.anchor_mask.iter().enumerate() {
        if flag != 0 {
            selected_rows.push(i);
        }
    }
    Ok(Geometry {
        pivots: s.pivots,
        selected_rows,
        anchor_rank,
        unique_rows: unique,
        scratch_bytes: bytes,
        largest_norm,
        final_maximum_residual,
    })
}

#[derive(Serialize)]
pub(super) struct Call {
    pub(super) ordinal: usize,
    pub(super) population_index: usize,
    original_case_indices: Vec<usize>,
    mandatory_anchor_indices: Vec<usize>,
    pub(super) complete: bool,
    pub(super) span_verified: bool,
    pub(super) rank: Option<usize>,
    pub(super) anchor_rank: Option<usize>,
    pub(super) pivot_indices: Vec<usize>,
    pub(super) final_selected_cases: Vec<usize>,
    pub(super) visits: u64,
    pub(super) visits_before: u64,
    pub(super) visits_after: u64,
    pub(super) exhausted: bool,
    pub(super) error: Option<String>,
    scratch_bytes: Option<usize>,
    unique_rows: Option<usize>,
    final_maximum_residual: Option<f64>,
    largest_original_norm: Option<f64>,
    independent_span_visits: u64,
    independent_span_error: Option<String>,
}
impl Call {
    pub(super) fn not_run(
        original: &super::Call,
        cases: &[usize],
        before: u64,
        work: &GeometryWork,
        error: StructuredUnknown,
    ) -> Self {
        Self {
            ordinal: original.matrix.ordinal,
            population_index: original.population.population_index,
            original_case_indices: cases.to_vec(),
            mandatory_anchor_indices: Vec::new(),
            complete: false,
            span_verified: false,
            rank: None,
            anchor_rank: None,
            pivot_indices: Vec::new(),
            final_selected_cases: Vec::new(),
            visits: work.used - before,
            visits_before: before,
            visits_after: work.used,
            exhausted: work.exhausted,
            error: Some(format!("{error:?}")),
            scratch_bytes: None,
            unique_rows: None,
            final_maximum_residual: None,
            largest_original_norm: None,
            independent_span_visits: 0,
            independent_span_error: None,
        }
    }
}
#[derive(Serialize)]
pub(super) struct Runs {
    maximum_visits: u64,
    used_visits: u128,
    exhausted: bool,
    complete: bool,
    span_verified: bool,
    calls: Vec<Call>,
}
#[derive(Serialize)]
pub(super) struct Measurement {
    geometry_kernel: Mode,
    independent_max: Runs,
    shared_original_budget: Runs,
    conclusion: &'static str,
}

fn span_certificate(
    rows: &[&[f64]],
    settings: &StructuredSettingsV2,
    selected: &[usize],
) -> (bool, u64, Option<String>) {
    let borrowed: Vec<_> = rows
        .iter()
        .map(|basis| FitRow { basis, wall_ns: 0 })
        .collect();
    let mut work = GeometryWork {
        used: 0,
        limit: u64::MAX,
        exhausted: false,
    };
    let result = (|| -> Result<bool> {
        let reference = input_geometry_core(&borrowed, settings, Some(&mut work), false, selected)?;
        if !reference
            .pivot_indices
            .iter()
            .all(|p| selected.binary_search(p).is_ok())
        {
            return Ok(false);
        }
        let d = reference.scale.len();
        for q in &reference.basis {
            work.charge(mul(d, add(reference.basis.len(), 1)?)?)?;
            if (dot(q, q) - 1.).abs() > DEPENDENT {
                return Ok(false);
            }
        }
        for (i, q) in reference.basis.iter().enumerate() {
            if reference.basis[..i]
                .iter()
                .any(|old| dot(q, old).abs() > DEPENDENT)
            {
                return Ok(false);
            }
        }
        work.charge(mul(rows.len(), d)?)?;
        let largest = reference
            .rows
            .iter()
            .map(|row| norm(row))
            .fold(0., f64::max);
        for row in rows {
            let (_, length) = residual(row, &reference.scale, &reference.basis, &mut work)?;
            if length / largest > DEPENDENT {
                return Ok(false);
            }
        }
        Ok(true)
    })();
    match result {
        Ok(true) => (true, work.used, None),
        Ok(false) => (
            false,
            work.used,
            Some("original full-matrix pivot/span/orthogonality certificate rejected".into()),
        ),
        Err(reason) => (false, work.used, Some(format!("{reason:?}"))),
    }
}

pub(super) fn call(
    verified: &Verified,
    call: &super::Call,
    work: &mut GeometryWork,
    certificate: bool,
) -> AuditResult<Call> {
    call_mapped(
        verified,
        call,
        None,
        &call.matrix.mandatory_anchors,
        0,
        work,
        certificate,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn call_mapped(
    verified: &Verified,
    call: &super::Call,
    source_rows: Option<&[usize]>,
    anchors: &[usize],
    metadata_bytes: usize,
    work: &mut GeometryWork,
    certificate: bool,
) -> AuditResult<Call> {
    let matrix = &call.matrix;
    let n = source_rows.map_or(matrix.cases.len(), <[usize]>::len);
    let source = |i: usize| source_rows.map_or(i, |indices| indices[i]);
    let decoded: Vec<Vec<f64>> = (0..n)
        .map(|i| &matrix.axis_bits[source(i)])
        .map(|r| r.iter().map(|&b| f64::from_bits(b)).collect())
        .collect();
    let rows: Vec<_> = decoded.iter().map(Vec::as_slice).collect();
    let before = work.used;
    // The width vector is candidate-owned metadata, not free caller scratch.
    let external = n
        .checked_mul(size_of::<usize>())
        .and_then(|b| b.checked_add(size_of::<Vec<usize>>()))
        .and_then(|b| b.checked_add(metadata_bytes))
        .context("width scratch overflow")?;
    let result = if work.exhausted || external > matrix.maximum_scratch_bytes {
        Err(StructuredUnknown::Capacity)
    } else {
        match work.charge(n) {
            Err(error) => Err(error),
            Ok(()) => {
                let widths: Vec<_> = (0..n)
                    .map(|i| verified.cases.cases[matrix.cases[source(i)]].width)
                    .collect();
                geometry_mapped(
                    matrix,
                    &rows,
                    &widths,
                    source_rows,
                    anchors,
                    work,
                    matrix.maximum_scratch_bytes - external,
                )
            }
        }
    };
    let mut record = Call {
        ordinal: matrix.ordinal,
        population_index: call.population.population_index,
        original_case_indices: (0..n).map(|i| matrix.cases[source(i)]).collect(),
        mandatory_anchor_indices: anchors.to_vec(),
        complete: result.is_ok(),
        span_verified: false,
        rank: None,
        anchor_rank: None,
        pivot_indices: Vec::new(),
        final_selected_cases: Vec::new(),
        visits: work.used - before,
        visits_before: before,
        visits_after: work.used,
        exhausted: work.exhausted,
        error: None,
        scratch_bytes: None,
        unique_rows: None,
        final_maximum_residual: None,
        largest_original_norm: None,
        independent_span_visits: 0,
        independent_span_error: None,
    };
    match result {
        Err(reason) => record.error = Some(format!("{reason:?}")),
        Ok(geometry) => {
            record.rank = Some(geometry.pivots.len());
            record.anchor_rank = Some(geometry.anchor_rank);
            record.scratch_bytes = Some(geometry.scratch_bytes + external);
            record.unique_rows = Some(geometry.unique_rows);
            record.final_maximum_residual = Some(geometry.final_maximum_residual);
            record.largest_original_norm = Some(geometry.largest_norm);
            let selected = geometry.selected_rows;
            record.final_selected_cases =
                selected.iter().map(|&i| matrix.cases[source(i)]).collect();
            record.pivot_indices = geometry.pivots;
            if certificate {
                (
                    record.span_verified,
                    record.independent_span_visits,
                    record.independent_span_error,
                ) = span_certificate(&rows, &matrix.settings, &selected);
            }
        }
    }
    Ok(record)
}

pub(super) fn evaluate(verified: &Verified, mode: Mode) -> AuditResult<Option<Measurement>> {
    if mode == Mode::Disabled {
        return Ok(None);
    }
    let mut independent = Vec::new();
    for original in &verified.calls {
        let mut work = GeometryWork {
            used: 0,
            limit: u64::MAX,
            exhausted: false,
        };
        independent.push(call(verified, original, &mut work, true)?);
    }
    let mut work = GeometryWork {
        used: 0,
        limit: verified.limit,
        exhausted: false,
    };
    let mut shared = Vec::new();
    for (original, expected) in verified.calls.iter().zip(&independent) {
        let mut actual = call(verified, original, &mut work, false)?;
        if actual.complete {
            ensure!(
                expected.complete
                    && actual.rank == expected.rank
                    && actual.anchor_rank == expected.anchor_rank
                    && actual.pivot_indices == expected.pivot_indices
                    && actual.final_selected_cases == expected.final_selected_cases,
                "cold candidate changed result with shared budget at {}",
                actual.ordinal
            );
            actual.span_verified = expected.span_verified;
        }
        shared.push(actual);
    }
    let independent_max = Runs {
        maximum_visits: u64::MAX,
        used_visits: independent.iter().map(|c| u128::from(c.visits)).sum(),
        exhausted: independent.iter().any(|c| c.exhausted),
        complete: independent.iter().all(|c| c.complete),
        span_verified: independent.iter().all(|c| c.span_verified),
        calls: independent,
    };
    let shared_original_budget = Runs {
        maximum_visits: verified.limit,
        used_visits: u128::from(work.used),
        exhausted: work.exhausted,
        complete: shared.iter().all(|c| c.complete),
        span_verified: shared.iter().all(|c| c.span_verified),
        calls: shared,
    };
    Ok(Some(Measurement { geometry_kernel: mode, independent_max, shared_original_budget,
        conclusion: "Test-only cold representative experiment. Hash/exact equality, indexing, validation, cache updates and terminal refresh are charged. Exact groups preserve anchor role; every mandatory original case remains. Independent legacy full-matrix span certification is separately charged diagnostic work. Capture decoding and report copies/serialization remain offline diagnostics, outside kernel scratch and visits. New pivots/rank are not v1-equivalence, actual F/R/Q samples, a fitted model, a feasible source schedule, or hardware SLO evidence." }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Matrix, Vec<Vec<f64>>, Vec<usize>) {
        let values: Vec<Vec<f64>> = vec![
            vec![1., 0., 0.],
            vec![1., 0., 0.],
            vec![1., 0., 0.],
            vec![1., 1., 0.],
            vec![1., 1., 0.],
            vec![1., 0., 1.],
        ];
        let matrix = Matrix {
            ordinal: 0,
            cases: (0..values.len()).collect(),
            axis_bits: values
                .iter()
                .map(|r| r.iter().map(|v| v.to_bits()).collect())
                .collect(),
            mandatory_anchors: vec![0, 1],
            settings: StructuredSettingsV2::default(),
            visits_before: 0,
            maximum_visits: u64::MAX,
            exhausted_before: false,
            maximum_scratch_bytes: 1 << 20,
        };
        (matrix, values, vec![8, 1, 1, 8, 1, 2])
    }
    fn work(limit: u64) -> GeometryWork {
        GeometryWork {
            used: 0,
            limit,
            exhausted: false,
        }
    }

    #[test]
    fn cold_candidate_exact_rows_preserve_anchor_roles_and_prefer_cheaper_real_pivots() {
        let (matrix, values, widths) = fixture();
        let rows: Vec<_> = values.iter().map(Vec::as_slice).collect();
        let mut charged = work(u64::MAX);
        let candidate = geometry(
            &matrix,
            &rows,
            &widths,
            &mut charged,
            matrix.maximum_scratch_bytes,
        )
        .unwrap();
        assert_eq!(candidate.unique_rows, 4); // Same bits with distinct anchor role stay separate.
        assert_eq!(candidate.anchor_rank, 1);
        assert_eq!(candidate.pivots, [1, 4, 5]);
        assert_eq!(candidate.selected_rows, [0, 1, 4, 5]); // Expensive mandatory row 0 stays.
        assert!(candidate.final_maximum_residual / candidate.largest_norm <= DEPENDENT);
        let (valid, audit_visits, error) =
            span_certificate(&rows, &matrix.settings, &candidate.selected_rows);
        assert!(valid && error.is_none() && audit_visits > 0);
        assert!(!span_certificate(&rows, &matrix.settings, &[0, 1, 4]).0);
        // Hash narrowing is not equality; signed-zero bits remain distinct.
        let mut compare = work(u64::MAX);
        assert!(!equal_row(&[0], &[(-0f64).to_bits()], &mut compare).unwrap());
        assert_eq!(compare.used, 2);
    }

    #[test]
    fn cold_candidate_exact_work_and_scratch_boundaries_fail_without_refunds() {
        let (matrix, values, widths) = fixture();
        let rows: Vec<_> = values.iter().map(Vec::as_slice).collect();
        let mut full = work(u64::MAX);
        let expected = geometry(
            &matrix,
            &rows,
            &widths,
            &mut full,
            matrix.maximum_scratch_bytes,
        )
        .unwrap();
        let mut exact = work(full.used);
        assert_eq!(
            geometry(&matrix, &rows, &widths, &mut exact, expected.scratch_bytes)
                .unwrap()
                .selected_rows,
            expected.selected_rows
        );
        assert_eq!(exact.used, full.used);
        let mut short = work(full.used - 1);
        assert!(matches!(
            geometry(&matrix, &rows, &widths, &mut short, expected.scratch_bytes),
            Err(StructuredUnknown::Capacity)
        ));
        assert!(short.exhausted && short.used <= short.limit);
        let spent = short.used;
        assert!(matches!(
            geometry(&matrix, &rows, &widths, &mut short, expected.scratch_bytes),
            Err(StructuredUnknown::Capacity)
        ));
        assert_eq!(short.used, spent);
        let mut scratch_short = work(u64::MAX);
        assert!(matches!(
            geometry(
                &matrix,
                &rows,
                &widths,
                &mut scratch_short,
                expected.scratch_bytes - 1
            ),
            Err(StructuredUnknown::Capacity)
        ));
        assert!(!scratch_short.exhausted);
    }

    #[test]
    fn cold_candidate_keeps_signed_bits_and_rejects_an_ambiguous_late_direction() {
        let (mut matrix, mut values, widths) = fixture();
        values[4][2] = -0.;
        matrix.axis_bits[4][2] = (-0f64).to_bits();
        let rows: Vec<_> = values.iter().map(Vec::as_slice).collect();
        let candidate = geometry(
            &matrix,
            &rows,
            &widths,
            &mut work(u64::MAX),
            matrix.maximum_scratch_bytes,
        )
        .unwrap();
        assert_eq!(candidate.unique_rows, 5);
        assert_eq!(candidate.selected_rows, [0, 1, 4, 5]);
        assert!(span_certificate(&rows, &matrix.settings, &candidate.selected_rows).0);
        values[5][2] = 1e-8;
        matrix.axis_bits[5][2] = (1e-8f64).to_bits();
        let rows: Vec<_> = values.iter().map(Vec::as_slice).collect();
        assert!(matches!(
            geometry(
                &matrix,
                &rows,
                &widths,
                &mut work(u64::MAX),
                matrix.maximum_scratch_bytes
            ),
            Err(StructuredUnknown::IllConditioned)
        ));
    }
}
