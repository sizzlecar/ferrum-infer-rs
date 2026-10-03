//! Offline audit of an original, completed geometry capture. These DTOs grant
//! no checked-input, execution, sample or publication authority.
use super::*;
use anyhow::{bail, ensure, Context};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, io::Read, path::PathBuf};

type AuditResult<T> = anyhow::Result<T>;
const METADATA_LIMIT: u64 = 64 * 1024;
const CAPTURE_LIMIT: u64 = 256 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BoundFile {
    path: PathBuf,
    sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplayCase {
    capture: BoundFile,
    report: BoundFile,
    input_sha256: [u8; 32],
    output: PathBuf,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureReport {
    path: PathBuf,
    bytes: u64,
    sha256: String,
    complete: bool,
    failure: Option<String>,
    expected_matrices: usize,
    observed_matrices: usize,
    elapsed_ns: u128,
    diagnostic_buffer_bytes: usize,
    diagnostic_retained_bytes: usize,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
enum Product {
    Prefill,
    ContinuationPrefill {
        offset: usize,
    },
    PrefillSpan {
        offset: usize,
        chunk: std::num::NonZeroU32,
    },
    Greedy,
    Full,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct OriginalCase {
    product: Product,
    template: usize,
    width: usize,
    maximum_output: std::num::NonZeroUsize,
    release_generated: usize,
    suffix_tokens: usize,
    preset: ferrum_types::SloAutomaticCostProbeSamplingPresetV1,
    prefix: serde_json::Value,
    route: serde_json::Value,
    reset: bool,
    acquisition: Option<serde_json::Value>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CaseTable {
    cases: Vec<OriginalCase>,
    prompts: Vec<usize>,
    chunk: usize,
    prefill_row_ceiling: Option<std::num::NonZeroU32>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Population {
    population_index: usize,
    key: serde_json::Value,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Matrix {
    ordinal: usize,
    cases: Vec<usize>,
    axis_bits: Vec<Vec<u64>>,
    mandatory_anchors: Vec<usize>,
    settings: StructuredSettingsV2,
    visits_before: u64,
    maximum_visits: u64,
    exhausted_before: bool,
    maximum_scratch_bytes: usize,
}
#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct GeometryAudit {
    candidate_rank: Option<usize>,
    selected_rank: Option<usize>,
    added_original_cases: usize,
    complete: bool,
    visits: u64,
}
#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
enum Gap {
    InputGeometryUnavailable {
        reason: String,
        work_exhausted: bool,
    },
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OriginalResult {
    ordinal: usize,
    audit: GeometryAudit,
    gap: Option<Gap>,
    final_selected_cases: Vec<usize>,
    visits_after: u64,
    exhausted_after: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Retirement {
    complete: bool,
    deadline_expired: bool,
    #[serde(deserialize_with = "deserialize_remaining_ns")]
    remaining_ns: u128,
    actual_requests: usize,
    actual_actions: usize,
    selected_requests: usize,
    selected_actions: usize,
}
// Serde's internally tagged content cannot deserialize u128 directly. The
// original bounded startup deadline fits u64 nanoseconds; reject larger wire
// values instead of passing through f64 or truncating them.
fn deserialize_remaining_ns<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<u128, D::Error> {
    u64::deserialize(deserializer).map(u128::from)
}

#[derive(Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
enum Record {
    #[serde(rename = "ferrum.test.cold_geometry.v1")]
    Header {
        input_sha256: [u8; 32],
        maximum_bytes: u64,
        buffer_bytes: usize,
    },
    #[serde(rename = "actual_startup_inputs")]
    Startup {
        config: Box<ferrum_types::EngineConfig>,
        templates: Vec<serde_json::Value>,
    },
    #[serde(rename = "original_cases")]
    Cases(CaseTable),
    #[serde(rename = "population")]
    Population(Population),
    #[serde(rename = "original_matrix")]
    Matrix(Matrix),
    #[serde(rename = "original_result")]
    Result(OriginalResult),
    #[serde(rename = "inventory_retired")]
    Retirement(Retirement),
    #[serde(rename = "completed_after_shutdown")]
    Completed { matrices: usize },
}
struct Call {
    population: Population,
    matrix: Matrix,
    result: OriginalResult,
}
struct Verified {
    cases: CaseTable,
    calls: Vec<Call>,
    limit: u64,
    visits: u64,
    exhausted: bool,
    retirement: Retirement,
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn read_bounded(path: &std::path::Path, maximum: u64) -> AuditResult<Vec<u8>> {
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(maximum + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= maximum,
        "bounded artifact too large: {path:?}"
    );
    Ok(bytes)
}
fn bound_file(file: &BoundFile, maximum: u64) -> AuditResult<Vec<u8>> {
    let bytes = read_bounded(&file.path, maximum)?;
    ensure!(
        digest(&bytes) == file.sha256,
        "artifact SHA256 mismatch: {:?}",
        file.path
    );
    Ok(bytes)
}

fn verify_capture(
    bytes: &[u8],
    report: &CaptureReport,
    expected_input: [u8; 32],
) -> AuditResult<Verified> {
    ensure!(
        report.complete && report.failure.is_none(),
        "capture did not attest cleanup"
    );
    ensure!(
        report.bytes == bytes.len() as u64 && report.sha256 == digest(bytes),
        "capture report does not bind these bytes"
    );
    ensure!(
        bytes.last() == Some(&b'\n'),
        "capture has no final record boundary"
    );
    let mut stream = serde_json::Deserializer::from_slice(bytes).into_iter::<Record>();
    let mut next = || -> AuditResult<Record> {
        stream
            .next()
            .context("truncated capture")?
            .context("invalid capture record")
    };
    let Record::Header {
        input_sha256,
        maximum_bytes,
        buffer_bytes,
    } = next()?
    else {
        bail!("capture header missing");
    };
    ensure!(
        input_sha256 == expected_input
            && report.bytes <= maximum_bytes
            && maximum_bytes <= CAPTURE_LIMIT,
        "capture input or byte bound mismatch"
    );
    ensure!(
        buffer_bytes == report.diagnostic_buffer_bytes
            && report.diagnostic_retained_bytes >= buffer_bytes,
        "diagnostic buffer accounting mismatch"
    );
    let Record::Startup { config, templates } = next()? else {
        bail!("original startup input missing")
    };
    ensure!(!templates.is_empty(), "original templates missing");
    let ferrum_types::SloLiveStructuredCalibration::AutomaticV1 { settings } = &config
        .scheduler
        .slo
        .cost_observation
        .live_structured_calibration
    else {
        bail!("capture is not automatic calibration");
    };
    use ferrum_types::SloAutomaticCalibrationInputReadinessV1 as Readiness;
    let limit = match settings.input_readiness {
        Readiness::WorkAxesAndBranchesV1 {
            maximum_geometry_visits,
            ..
        }
        | Readiness::WorkAxesAndBranchesV2 {
            maximum_geometry_visits,
            ..
        }
        | Readiness::WorkAxesAndBranchesV3 {
            maximum_geometry_visits,
            ..
        } => maximum_geometry_visits,
        Readiness::CountOnlyV1 {} => bail!("capture has no declared geometry allowance"),
    };
    let Record::Cases(cases) = next()? else {
        bail!("same-selection case map missing")
    };
    ensure!(
        !cases.cases.is_empty() && cases.chunk > 0,
        "empty case inventory"
    );
    for case in &cases.cases {
        ensure!(
            case.width > 0 && cases.prompts.get(case.template).is_some_and(|&n| n > 0),
            "case has invalid original width/template"
        );
    }
    let mut work = StructuredInputGeometryWorkV1::new(limit);
    let mut calls = Vec::new();
    let mut seen_populations = std::collections::BTreeSet::new();
    let retirement = loop {
        let population = match next()? {
            Record::Population(value) => value,
            Record::Retirement(value) => break value,
            _ => bail!("expected original population or retirement"),
        };
        ensure!(
            seen_populations.insert(population.population_index),
            "population repeated"
        );
        let Record::Matrix(matrix) = next()? else {
            bail!("population has no matrix")
        };
        let Record::Result(result) = next()? else {
            bail!("matrix has no original result")
        };
        let ordinal = calls.len();
        ensure!(
            matrix.ordinal == ordinal && result.ordinal == ordinal,
            "noncontiguous ordinal at {ordinal}"
        );
        replay_call(&matrix, &result, &cases, &mut work)
            .with_context(|| format!("matrix {ordinal}"))?;
        calls.push(Call {
            population,
            matrix,
            result,
        });
    };
    ensure!(
        retirement.complete,
        "original inventory was not retired completely"
    );
    let Record::Completed { matrices } = next()? else {
        bail!("shutdown footer missing")
    };
    ensure!(
        matrices > 0
            && matrices == calls.len()
            && matrices == report.expected_matrices
            && matrices == report.observed_matrices,
        "complete matrix count mismatch"
    );
    drop(next);
    ensure!(stream.next().is_none(), "records follow shutdown footer");
    Ok(Verified {
        cases,
        calls,
        limit: limit.get(),
        visits: work.visits(),
        exhausted: work.exhausted(),
        retirement,
    })
}

fn replay_call(
    matrix: &Matrix,
    result: &OriginalResult,
    cases: &CaseTable,
    work: &mut StructuredInputGeometryWorkV1,
) -> AuditResult<()> {
    ensure!(
        matrix.maximum_visits == work.maximum_visits()
            && matrix.visits_before == work.visits()
            && matrix.exhausted_before == work.exhausted(),
        "shared ledger before differs"
    );
    ensure!(
        matrix.cases.len() == matrix.axis_bits.len()
            && matrix.cases.windows(2).all(|p| p[0] < p[1])
            && matrix.cases.iter().all(|&i| i < cases.cases.len()),
        "matrix case map invalid"
    );
    ensure!(
        matrix.mandatory_anchors.windows(2).all(|p| p[0] < p[1])
            && matrix
                .mandatory_anchors
                .iter()
                .all(|&i| i < matrix.cases.len()),
        "original anchors invalid"
    );
    let decoded: Vec<Vec<f64>> = matrix
        .axis_bits
        .iter()
        .map(|row| row.iter().map(|&bits| f64::from_bits(bits)).collect())
        .collect();
    let rows: Vec<_> = decoded.iter().map(Vec::as_slice).collect();
    let mut selected: Vec<_> = matrix
        .mandatory_anchors
        .iter()
        .map(|&i| matrix.cases[i])
        .collect();
    let original_count = selected.len();
    let geometry = input_geometry_pivots_original_v1(
        &rows,
        &matrix.mandatory_anchors,
        &matrix.settings,
        work,
        matrix.maximum_scratch_bytes,
    );
    let (rank, gap) = match geometry {
        Ok(geometry) => {
            for &pivot in &geometry.pivot_indices[geometry.anchor_rank..] {
                let case = matrix.cases[pivot];
                if !selected.contains(&case) {
                    selected.push(case);
                }
            }
            (Some(geometry.rank), None)
        }
        Err(reason) => (
            None,
            Some(Gap::InputGeometryUnavailable {
                reason: format!("{reason:?}"),
                work_exhausted: work.exhausted(),
            }),
        ),
    };
    selected.sort_unstable();
    let audit = GeometryAudit {
        candidate_rank: rank,
        selected_rank: rank,
        added_original_cases: selected.len() - original_count,
        complete: gap.is_none(),
        visits: work.visits() - matrix.visits_before,
    };
    ensure!(
        audit == result.audit && gap == result.gap,
        "rank/error/charge audit differs: actual={audit:?} gap={gap:?}"
    );
    ensure!(
        selected == result.final_selected_cases,
        "original selected case set differs"
    );
    ensure!(
        work.visits() == result.visits_after && work.exhausted() == result.exhausted_after,
        "shared ledger after differs"
    );
    Ok(())
}

fn strict_signature(matrix: &Matrix) -> AuditResult<[u8; 32]> {
    let mut h = Sha256::new();
    h.update(b"cold-input-call-bits-v1\0");
    for value in [
        matrix.axis_bits.len(),
        matrix.maximum_scratch_bytes,
        matrix.mandatory_anchors.len(),
    ] {
        h.update(u64::try_from(value)?.to_le_bytes());
    }
    for row in &matrix.axis_bits {
        h.update(u64::try_from(row.len())?.to_le_bytes());
        for value in row {
            h.update(value.to_le_bytes());
        }
    }
    for &anchor in &matrix.mandatory_anchors {
        h.update(u64::try_from(anchor)?.to_le_bytes());
    }
    h.update(serde_json::to_vec(&matrix.settings)?);
    Ok(h.finalize().into())
}
fn same_call(a: &Matrix, b: &Matrix) -> AuditResult<bool> {
    Ok(a.axis_bits == b.axis_bits
        && a.mandatory_anchors == b.mandatory_anchors
        && a.maximum_scratch_bytes == b.maximum_scratch_bytes
        && serde_json::to_vec(&a.settings)? == serde_json::to_vec(&b.settings)?)
}

#[derive(Serialize)]
struct CaseDomain<'a> {
    index: usize,
    original: &'a OriginalCase,
    prompt_tokens: usize,
    decode_sequence_tokens: Option<usize>,
    prefill_offset: Option<usize>,
}
#[derive(Serialize)]
struct MatrixStatistics<'a> {
    ordinal: usize,
    population_index: usize,
    population_key: &'a serde_json::Value,
    original_case_indices: &'a [usize],
    final_selected_cases: &'a [usize],
    rows: usize,
    axes: usize,
    all_original_rows_zero_axes: Vec<usize>,
    scan_coordinates: usize,
    original_audit: &'a GeometryAudit,
    original_gap: &'a Option<Gap>,
}
#[derive(Serialize)]
struct DuplicateGroup {
    ordinals: Vec<usize>,
    original_charged_visits: Vec<u64>,
    original_visits_before: Vec<u64>,
    original_exhausted_before: Vec<bool>,
    original_complete: Vec<bool>,
}
#[derive(Serialize)]
struct Statistics<'a> {
    baseline_matched: bool,
    maximum_visits: u64,
    final_visits: u64,
    exhausted: bool,
    chunk: usize,
    prefill_row_ceiling: Option<std::num::NonZeroU32>,
    case_domains: Vec<CaseDomain<'a>>,
    matrices: Vec<MatrixStatistics<'a>>,
    strict_duplicate_groups: Vec<DuplicateGroup>,
    inventory_deadline_expired: bool,
    inventory_remaining_ns: u128,
    remaining_actual_requests: usize,
    remaining_actual_actions: usize,
    remaining_selected_requests: usize,
    remaining_selected_actions: usize,
    conclusion: &'static str,
}
fn statistics(verified: &Verified) -> AuditResult<Statistics<'_>> {
    let case_domains = verified
        .cases
        .cases
        .iter()
        .enumerate()
        .map(|(index, case)| {
            let prompt = verified.cases.prompts[case.template];
            let (decode_sequence_tokens, prefill_offset) = match case.product {
                Product::Greedy | Product::Full => (
                    Some(
                        prompt
                            .checked_add(case.release_generated)
                            .context("decode context overflow")?,
                    ),
                    None,
                ),
                Product::Prefill => (None, Some(0)),
                Product::ContinuationPrefill { offset } | Product::PrefillSpan { offset, .. } => {
                    (None, Some(offset))
                }
            };
            Ok(CaseDomain {
                index,
                original: case,
                prompt_tokens: prompt,
                decode_sequence_tokens,
                prefill_offset,
            })
        })
        .collect::<AuditResult<Vec<_>>>()?;
    let mut groups = Vec::<Vec<usize>>::new();
    let mut signatures = BTreeMap::<[u8; 32], Vec<usize>>::new();
    let mut matrices = Vec::new();
    for call in &verified.calls {
        let matrix = &call.matrix;
        let signature = strict_signature(matrix)?;
        let candidates = signatures.entry(signature).or_default();
        let mut found = None;
        for &group in candidates.iter() {
            if same_call(matrix, &verified.calls[groups[group][0]].matrix)? {
                found = Some(group);
                break;
            }
        }
        let group = match found {
            Some(group) => group,
            None => {
                let group = groups.len();
                groups.push(Vec::new());
                candidates.push(group);
                group
            }
        };
        groups[group].push(matrix.ordinal);
        let axes = matrix
            .axis_bits
            .first()
            .context("zero-axis audit has no rows")?
            .len();
        ensure!(
            axes > 0 && matrix.axis_bits.iter().all(|row| row.len() == axes),
            "invalid original row shape for zero-axis audit"
        );
        let mut zero = vec![true; axes];
        for row in &matrix.axis_bits {
            for (axis, &bits) in row.iter().enumerate() {
                let value = f64::from_bits(bits);
                ensure!(
                    value.is_finite() && value >= 0. && value <= (1u64 << 53) as f64,
                    "invalid original coordinate in zero-axis audit"
                );
                zero[axis] &= value == 0.;
            }
        }
        matrices.push(MatrixStatistics {
            ordinal: matrix.ordinal,
            population_index: call.population.population_index,
            population_key: &call.population.key,
            original_case_indices: &matrix.cases,
            final_selected_cases: &call.result.final_selected_cases,
            rows: matrix.axis_bits.len(),
            axes,
            all_original_rows_zero_axes: zero
                .iter()
                .enumerate()
                .filter_map(|(i, &zero)| zero.then_some(i))
                .collect(),
            scan_coordinates: matrix
                .axis_bits
                .len()
                .checked_mul(axes)
                .context("scan count overflow")?,
            original_audit: &call.result.audit,
            original_gap: &call.result.gap,
        });
    }
    let strict_duplicate_groups = groups
        .into_iter()
        .filter(|group| group.len() > 1)
        .map(|ordinals| DuplicateGroup {
            original_charged_visits: ordinals
                .iter()
                .map(|&i| verified.calls[i].result.audit.visits)
                .collect(),
            original_visits_before: ordinals
                .iter()
                .map(|&i| verified.calls[i].matrix.visits_before)
                .collect(),
            original_exhausted_before: ordinals
                .iter()
                .map(|&i| verified.calls[i].matrix.exhausted_before)
                .collect(),
            original_complete: ordinals
                .iter()
                .map(|&i| verified.calls[i].result.audit.complete)
                .collect(),
            ordinals,
        })
        .collect();
    Ok(Statistics {
        baseline_matched: true,
        maximum_visits: verified.limit,
        final_visits: verified.visits,
        exhausted: verified.exhausted,
        chunk: verified.cases.chunk,
        prefill_row_ceiling: verified.cases.prefill_row_ceiling,
        case_domains,
        matrices,
        strict_duplicate_groups,
        inventory_deadline_expired: verified.retirement.deadline_expired,
        inventory_remaining_ns: verified.retirement.remaining_ns,
        remaining_actual_requests: verified.retirement.actual_requests,
        remaining_actual_actions: verified.retirement.actual_actions,
        remaining_selected_requests: verified.retirement.selected_requests,
        remaining_selected_actions: verified.retirement.selected_actions,
        conclusion: concat!(
            "Descriptive original-input audit only. Strict duplicate groups compare the ordered numerical call body, not ledger state or owner authority. ",
            "Duplicate charges are not savings; zero columns are not proven-safe compaction. ",
            "No cache/hash/copy/memory cost simulation, changed geometry, source scheduling, qualification, latency or hardware feasibility is established."
        ),
    })
}

#[test]
#[ignore = "requires explicit geometry-replay-case.json beside the test executable and SHA-bound original capture/report"]
fn original_geometry_capture_replays_shared_budget_and_reports_structure() -> AuditResult<()> {
    let descriptor_path = std::env::current_exe()?.with_file_name("geometry-replay-case.json");
    let descriptor = read_bounded(&descriptor_path, METADATA_LIMIT)?;
    let case: ReplayCase = serde_json::from_slice(&descriptor)?;
    ensure!(
        case.output != case.capture.path
            && case.output != case.report.path
            && case.output != descriptor_path,
        "output overlaps input"
    );
    let report_bytes = bound_file(&case.report, METADATA_LIMIT)?;
    let report: CaptureReport = serde_json::from_slice(&report_bytes)?;
    let bytes = bound_file(&case.capture, CAPTURE_LIMIT)?;
    let verified = verify_capture(&bytes, &report, case.input_sha256)?;
    // Do not create an apparently successful audit before every original call
    // has passed. Statistics never execute an alternative kernel or ledger.
    let stats = statistics(&verified)?;
    // Independent diagnostic references follow the unchanged shared-ledger
    // proof. They never mutate or replace its allowance or outcomes.
    let reference = reference::measure(&verified)?;
    let candidate = reference::evaluate_candidate(&verified, &reference)?;
    let output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&case.output)?;
    #[derive(Serialize)]
    struct Output<'a> {
        descriptor_sha256: String,
        capture_sha256: &'a str,
        report_sha256: &'a str,
        input_sha256: [u8; 32],
        capture_original_path: &'a std::path::Path,
        capture_elapsed_ns_diagnostic_only: u128,
        audit: Statistics<'a>,
        independent_reference: reference::Measurement<'a>,
        candidate: reference::CandidateMeasurement,
    }
    serde_json::to_writer_pretty(
        output,
        &Output {
            descriptor_sha256: digest(&descriptor),
            capture_sha256: &case.capture.sha256,
            report_sha256: &case.report.sha256,
            input_sha256: case.input_sha256,
            capture_original_path: &report.path,
            capture_elapsed_ns_diagnostic_only: report.elapsed_ns,
            audit: stats,
            independent_reference: reference,
            candidate,
        },
    )?;
    Ok(())
}

#[path = "capture_replay/reference.rs"]
mod reference;

#[path = "capture_replay/tests.rs"]
mod tests;
