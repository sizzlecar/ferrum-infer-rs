//! New offline candidate, never a reinterpretation of the original archive.
//! Only the original complete block boundaries may freeze numerical phases.
use super::*;
use crate::implementations::continuous::cost_model::structured_v2::*;
use crate::implementations::continuous::cost_profile::{
    ImportedStructuredCatalogV14, ImportedStructuredModelV2,
};
use std::{collections::BTreeMap, num::NonZeroU64};
use crate::implementations::continuous::cost_profile::structured_v10::service::owner_blocks::collector::State as OwnerState;

mod prepared_seed;
mod rejected_tail;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SeedSpec {
    universe_path: PathBuf,
    universe_signature: [u8; 32],
    #[serde(default)]
    cold_log_path: Option<PathBuf>,
    #[serde(default)]
    cold_log_sha256: Option<[u8; 32]>,
    #[serde(default)]
    original_prepared_header: Option<prepared_seed::OriginalPreparedHeader>,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum CandidatePolicy {
    CheckedSeed,
    CheckedSeedFrozenPhaseIntersection,
    CheckedSeedFrozenPhaseIntersectionZeroColumns,
}

fn seed(spec: &SeedSpec, original: &StructuredServiceHeaderV7) -> DeclaredAlgorithmUniverseV1 {
    let bytes = std::fs::read(&spec.universe_path).unwrap();
    assert!(bytes.len() < 1024 * 1024);
    let u: DeclaredAlgorithmUniverseV1 = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(u.signature(), &spec.universe_signature);
    if let Some(proof) = &spec.original_prepared_header {
        assert!(
            spec.cold_log_path.is_none() && spec.cold_log_sha256.is_none(),
            "choose exactly one original cold declaration proof"
        );
        prepared_seed::verify(proof, &u, original);
        return u;
    }
    let cold_log_path = spec.cold_log_path.as_ref().expect("original cold log path");
    let cold_log_sha256 = spec.cold_log_sha256.expect("original cold log digest");
    let mut digest = Sha256::new();
    let mut declared = false;
    let signature = format!("universe={:?}", spec.universe_signature);
    for line in BufReader::new(std::fs::File::open(cold_log_path).unwrap()).split(b'\n') {
        let mut line = line.unwrap();
        line.push(b'\n');
        digest.update(&line);
        let text = std::str::from_utf8(&line).unwrap();
        declared |= text
            .contains("Automatic checked algorithm subset frozen before numerical selection")
            && text.contains(&signature);
    }
    assert_eq!(<[u8; 32]>::from(digest.finalize()), cold_log_sha256);
    assert!(
        declared,
        "candidate seed was declared in the pinned cold log before numerical selection"
    );
    u
}

fn verify_source(spec: &AuditSpec) -> StructuredServiceHeaderV7 {
    verify_source_file(&spec.actual_source, spec.actual_source_sha256)
}

pub(super) fn verify_source_file(
    path: &std::path::Path,
    expected: [u8; 32],
) -> StructuredServiceHeaderV7 {
    let mut digest = Sha256::new();
    let mut header = None;
    for (index, line) in BufReader::new(std::fs::File::open(path).unwrap())
        .split(b'\n')
        .enumerate()
    {
        let mut line = line.unwrap();
        assert!(line.len() < 8 * 1024 * 1024);
        line.push(b'\n');
        digest.update(&line);
        if index == 0 {
            let h: StructuredServiceHeaderV7 = serde_json::from_slice(&line).unwrap();
            assert_eq!(record_bytes_v7(&h).unwrap(), line);
            header = Some(h);
        } else {
            let record: StructuredServiceRecordV7 = serde_json::from_slice(&line).unwrap();
            assert_eq!(
                record_bytes_v7(&record).unwrap(),
                line,
                "canonical original line {}",
                index + 1
            );
        }
    }
    assert_eq!(<[u8; 32]>::from(digest.finalize()), expected);
    header.unwrap()
}

fn candidate_header(
    original: &StructuredServiceHeaderV7,
    u: DeclaredAlgorithmUniverseV1,
    policy: CandidatePolicy,
) -> StructuredServiceHeaderV7 {
    let mut declaration = original.declaration.clone();
    assert!(declaration.schedule.phase_support.is_none());
    if !matches!(policy, CandidatePolicy::CheckedSeed) {
        declaration.schedule.phase_support =
            Some(OwnerPhaseSupportPolicyV1::FrozenInputIntersectionV1);
    }
    if matches!(
        policy,
        CandidatePolicy::CheckedSeedFrozenPhaseIntersectionZeroColumns
    ) {
        let old = declaration
            .schedule
            .input_readiness
            .as_ref()
            .expect("original input readiness");
        declaration.schedule.input_readiness = Some(
            OwnerInputReadinessV1::new_zero_column_v3(
                old.maximum_phase_blocks,
                old.maximum_geometry_visits,
            )
            .unwrap(),
        );
    }
    // Match production BlockSession: the runtime keeps the cold seed alive
    // independently of the collector header, even when both share its Arc.
    let seed_bytes = u.retained_payload_bytes().expect("bounded checked seed");
    declaration.maximum_discovery_bytes = declaration
        .maximum_discovery_bytes
        .checked_sub(seed_bytes)
        .filter(|n| *n > 0)
        .expect("seed leaves the original discovery capacity");
    declaration.maximum_retained_numeric_bytes = declaration
        .maximum_retained_numeric_bytes
        .checked_sub(seed_bytes)
        .filter(|n| *n > 0)
        .expect("seed leaves the original numeric capacity");
    let contract = declaration
        .nonnegative_envelope
        .as_mut()
        .expect("original physical D");
    assert_eq!(
        contract.workload_domain.sha256(),
        u.workload_domain_signature()
    );
    assert!(
        contract.algorithm_universe.is_none(),
        "original first-discovery control"
    );
    contract.algorithm_universe = Some(u.clone());
    declaration.schedule.algorithm_universe =
        Some(OwnerAlgorithmUniversePolicyV1::SeededFirstOrdinaryDiscoveryBlockSubsetV1);
    let mut identity = Sha256::new();
    identity.update(b"ferrum.offline-seeded-candidate.v1\0");
    identity.update(original.capture_identity);
    identity.update(u.signature());
    if !matches!(policy, CandidatePolicy::CheckedSeed) {
        identity.update(b"\0frozen-input-intersection.v1\0");
    }
    if matches!(
        policy,
        CandidatePolicy::CheckedSeedFrozenPhaseIntersectionZeroColumns
    ) {
        identity.update(b"\0zero-column-readiness.v3\0");
    }
    let h = StructuredServiceHeaderV7::new(
        identity.finalize().into(),
        original.generation,
        original.fingerprint.clone(),
        original.producer.clone(),
        original.opening,
        declaration,
        original.maximum_file_bytes,
    )
    .unwrap();
    let h = match &original.monotonic_domain {
        Some(domain) => h.with_monotonic_domain(domain.clone()).unwrap(),
        None => h,
    };
    assert_ne!(h.capture_identity, original.capture_identity);
    assert_ne!(h.protocol, original.protocol);
    h
}

pub(super) fn gap(
    facts: Option<&OwnerInputTargetV1>,
    target: Option<&OwnerInputTargetV1>,
) -> Value {
    let (Some(facts), Some(target)) = (facts, target) else {
        return json!({"facts":facts.is_some(),"target":target.is_some()});
    };
    let f = serde_json::to_value(facts).unwrap();
    let t = serde_json::to_value(target).unwrap();
    let missing: Vec<_> = f["positive"]
        .as_array()
        .unwrap()
        .iter()
        .zip(t["positive"].as_array().unwrap())
        .enumerate()
        .filter_map(|(i, (f, t))| (t.as_bool().unwrap() && !f.as_bool().unwrap()).then_some(i))
        .collect();
    let branches: Vec<_> = f["branches"]
        .as_array()
        .unwrap()
        .iter()
        .zip(t["branches"].as_array().unwrap())
        .enumerate()
        .filter_map(|(i, (f, t))| {
            let (f, t) = (f.as_u64().unwrap(), t.as_u64().unwrap());
            (f & t != t).then_some(json!({"axis":i,"seen":f,"required":t}))
        })
        .collect();
    json!({"missing_positive_axes":missing,"missing_branches":branches})
}

// Checked universe layout is intercept followed by AlgorithmAxisV1::width
// (Kernel 4, Copy/Fill/Readback/Scratch 2, LibraryCall 3), then host/replay.
pub(super) fn algorithm_axis(
    universe: Option<&DeclaredAlgorithmUniverseV1>,
    axis: usize,
) -> Option<Value> {
    let wire = serde_json::to_value(universe?).unwrap();
    let mut start = 1;
    for algorithm in wire["algorithms"].as_array().unwrap() {
        let kind = algorithm["kind"].as_u64().unwrap();
        let width = match kind {
            0 => 4,
            1..=4 => 2,
            5 => 3,
            _ => unreachable!(),
        };
        if (start..start + width).contains(&axis) {
            return Some(
                json!({"first_basis_axis":start,"within_algorithm":axis-start,
                "signature":algorithm["signature"],"kind":kind}),
            );
        }
        start += width;
    }
    None
}

// Fixed memory regardless of original source length. Nonzero buckets have a
// ratio of 2^(1/4); reported quantiles are intervals, never exact percentiles.
struct RatioHistogram([u64; 130]);
impl Default for RatioHistogram {
    fn default() -> Self {
        Self([0; 130])
    }
}
impl RatioHistogram {
    fn record(&mut self, value: f64) {
        assert!(value.is_finite() && value >= 0.0);
        let index = if value == 0.0 {
            0
        } else {
            (1 + ((value.log2() + 16.0) * 4.0).ceil().max(0.0) as usize).min(129)
        };
        self.0[index] = self.0[index].checked_add(1).unwrap();
    }
    fn percentile(&self, percentile: u64) -> Value {
        let count: u64 = self.0.iter().sum();
        if count == 0 {
            return Value::Null;
        }
        let wanted = count.checked_mul(percentile).unwrap().div_ceil(100);
        let mut cumulative = 0;
        let index = self
            .0
            .iter()
            .position(|&n| {
                cumulative += n;
                cumulative >= wanted
            })
            .unwrap();
        if index == 0 {
            return json!({"lower":0.0,"upper":0.0});
        }
        let lower = if index == 1 {
            0.0
        } else {
            2.0f64.powf((index as f64 - 2.0) / 4.0 - 16.0)
        };
        let upper = (index < 129).then(|| 2.0f64.powf((index as f64 - 1.0) / 4.0 - 16.0));
        json!({"lower_exclusive":lower,"upper_inclusive":upper})
    }
}

#[derive(Default)]
struct ErrorGroup {
    eligible: u64,
    known: u64,
    unknown: BTreeMap<String, u64>,
    point_count: u64,
    point_absolute_error_sum_ns: u64,
    max_point_absolute_error_ns: u64,
    point_underestimates: u64,
    max_point_under_ns: u64,
    planning_underestimates: u64,
    max_planning_under_ns: u64,
    point_relative_error: RatioHistogram,
    planning_to_actual: RatioHistogram,
}
impl ErrorGroup {
    fn known(&mut self, point: Option<u64>, planning: u64, actual: u64) {
        assert!(actual > 0);
        self.known += 1;
        self.planning_underestimates += u64::from(planning < actual);
        self.max_planning_under_ns = self
            .max_planning_under_ns
            .max(actual.saturating_sub(planning));
        self.planning_to_actual
            .record(planning as f64 / actual as f64);
        if let Some(point) = point {
            self.point_count += 1;
            self.point_absolute_error_sum_ns = self
                .point_absolute_error_sum_ns
                .checked_add(point.abs_diff(actual))
                .unwrap();
            self.max_point_absolute_error_ns =
                self.max_point_absolute_error_ns.max(point.abs_diff(actual));
            self.point_underestimates += u64::from(point < actual);
            self.max_point_under_ns = self.max_point_under_ns.max(actual.saturating_sub(point));
            self.point_relative_error
                .record(point.abs_diff(actual) as f64 / actual as f64);
        }
    }
    fn summary(&self) -> Value {
        json!({"eligible":self.eligible,"known":self.known,"unknown":self.unknown,
            "point_count":self.point_count,"point_absolute_error_sum_ns":self.point_absolute_error_sum_ns,
            "max_point_absolute_error_ns":self.max_point_absolute_error_ns,
            "point_mean_absolute_error_ns":(self.point_count>0).then(||self.point_absolute_error_sum_ns as f64/self.point_count as f64),
            "point_relative_absolute_error_p50_interval":self.point_relative_error.percentile(50),
            "point_relative_absolute_error_p99_interval":self.point_relative_error.percentile(99),
            "planning_to_actual_p50_interval":self.planning_to_actual.percentile(50),
            "planning_to_actual_p99_interval":self.planning_to_actual.percentile(99),
            "planning_underestimate_count":self.planning_underestimates,"max_planning_under_ns":self.max_planning_under_ns,
            "point_underestimate_count":self.point_underestimates,"max_point_under_ns":self.max_point_under_ns})
    }
}

#[derive(Default)]
struct Evaluation {
    eligible_after_activation: usize,
    known: usize,
    reasons: BTreeMap<String, usize>,
    planning_sum_ns: u64,
    actual_sum_ns: u64,
    point_sum_ns: u64,
    point_count: usize,
    max_planning_over_ns: u64,
    max_planning_under_ns: u64,
    max_point_abs_error_ns: u64,
    maximum_history: Option<(u64, Value)>,
    examples: Vec<Value>,
    errors: ErrorGroup,
    by_rows: BTreeMap<u32, ErrorGroup>,
    history_row_max_ge_400: ErrorGroup,
}
impl Evaluation {
    fn unknown(&mut self, reason: impl Into<String>, rows: u32, row_history: u64) {
        let reason = reason.into();
        *self.reasons.entry(reason.clone()).or_default() += 1;
        *self.errors.unknown.entry(reason.clone()).or_default() += 1;
        *self
            .by_rows
            .get_mut(&rows)
            .unwrap()
            .unknown
            .entry(reason.clone())
            .or_default() += 1;
        if row_history >= 400 {
            *self
                .history_row_max_ge_400
                .unknown
                .entry(reason)
                .or_default() += 1;
        }
    }
    fn observe(
        &mut self,
        h: &StructuredServiceHeaderV7,
        catalog: &ImportedStructuredCatalogV14,
        activated: u64,
        wave: &StructuredServiceWaveV7,
    ) {
        if wave.issued_at_ns < activated {
            return;
        }
        self.eligible_after_activation += 1;
        let mut contract = h.declaration.nonnegative_envelope.clone().unwrap();
        contract.algorithm_universe = None;
        let (raw, actual_ns, _) = physical::validate_parts(
            &h.fingerprint,
            h.opening.monotonic_ns,
            Some(&contract),
            h.opening.monotonic_ns,
            wave.ticket,
            wave.fifo,
            wave.issued_at_ns,
            &wave.host_stages,
            wave.independent.as_ref(),
            &mut physical::Frontiers::default(),
        )
        .unwrap();
        let query = StructuredQueryV2::exact(raw);
        let rows = query.input().owner().rows;
        assert!(rows > 0 && rows <= contract.workload_domain.limits().maximum_rows.get());
        let (history, row_history) = online::raw_history(wave);
        self.errors.eligible += 1;
        self.by_rows.entry(rows).or_default().eligible += 1;
        if row_history >= 400 {
            self.history_row_max_ge_400.eligible += 1;
        }
        let mut lookup_reasons = Vec::new();
        let matches: Vec<&ImportedStructuredModelV2> = catalog
            .children
            .iter()
            .filter(|child| {
                let matched = match child.numerical_family_key() {
                    Some(key) => {
                        let key_result = match child.algorithm_universe() {
                            Some(u) => query.input().numerical_family_key_for_universe(u),
                            None => query.input().numerical_family_key(),
                        };
                        match key_result {
                            Ok(actual) if actual == *key => true,
                            Ok(_) => {
                                lookup_reasons.push("population_key_mismatch".to_owned());
                                false
                            }
                            Err(reason) => {
                                lookup_reasons.push(format!("population_projection:{reason:?}"));
                                false
                            }
                        }
                    }
                    None => {
                        if child.owner() == query.input().owner() {
                            true
                        } else {
                            lookup_reasons.push("exact_owner_mismatch".to_owned());
                            false
                        }
                    }
                };
                matched
            })
            .collect();
        let child = match matches.as_slice() {
            [child] => *child,
            [] => {
                lookup_reasons.sort_unstable();
                lookup_reasons.dedup();
                self.unknown(
                    format!("no_declared_population_match:{lookup_reasons:?}"),
                    rows,
                    row_history,
                );
                return;
            }
            _ => {
                self.unknown("ambiguous_declared_population_match", rows, row_history);
                return;
            }
        };
        let prediction = match child.predict_query_local_with_clock_detailed(
            child.fingerprint(),
            &query,
            wave.issued_at_ns,
        ) {
            Ok((prediction, _)) => prediction,
            Err(reason) => {
                self.unknown(format!("{reason:?}"), rows, row_history);
                return;
            }
        };
        self.known += 1;
        self.planning_sum_ns = self
            .planning_sum_ns
            .checked_add(prediction.planning_ns)
            .unwrap();
        self.actual_sum_ns = self.actual_sum_ns.checked_add(actual_ns).unwrap();
        self.max_planning_over_ns = self
            .max_planning_over_ns
            .max(prediction.planning_ns.saturating_sub(actual_ns));
        self.max_planning_under_ns = self
            .max_planning_under_ns
            .max(actual_ns.saturating_sub(prediction.planning_ns));
        let decomposition = child.model.diagnose_archived_bound(&query).unwrap();
        let point = decomposition["fitted_point_ns"].as_u64();
        if let Some(point) = point {
            self.point_count += 1;
            self.point_sum_ns = self.point_sum_ns.checked_add(point).unwrap();
            self.max_point_abs_error_ns =
                self.max_point_abs_error_ns.max(point.abs_diff(actual_ns));
        }
        self.errors.known(point, prediction.planning_ns, actual_ns);
        self.by_rows
            .get_mut(&rows)
            .unwrap()
            .known(point, prediction.planning_ns, actual_ns);
        if row_history >= 400 {
            self.history_row_max_ge_400
                .known(point, prediction.planning_ns, actual_ns);
        }
        let example = json!({"call_id":wave.host_stages.call_id,"ticket":wave.ticket,
            "issued_at_ns":wave.issued_at_ns,"rows":query.input().owner().rows,
            "history_sum":history,"history_row_max":row_history,"actual_ns":actual_ns,
            "fitted_point_ns":point,"fitted_upper_ns":prediction.fitted_upper_ns,
            "planning_ns":prediction.planning_ns,"valid_until_ns":prediction.valid_until_ns,
            "positive_certificate":decomposition["positive_certificate"],
            "identified_envelope_ns":decomposition["identified_envelope_ns"],
            "parameters_sha256":child.parameters_signature()});
        if self.examples.len() < 8 {
            self.examples.push(example.clone());
        }
        if self
            .maximum_history
            .as_ref()
            .is_none_or(|(prior, _)| history > *prior)
        {
            self.maximum_history = Some((history, example));
        }
    }
    fn summary(&self) -> Value {
        json!({"eligible_original_completed_after_activation":self.eligible_after_activation,
            "known":self.known,"unknown_or_ambiguous":self.reasons,
            "planning_sum_ns":self.planning_sum_ns,"actual_sum_ns":self.actual_sum_ns,
            "point_sum_ns":self.point_sum_ns,"point_count":self.point_count,
            "max_planning_over_ns":self.max_planning_over_ns,
            "max_planning_under_ns":self.max_planning_under_ns,
            "max_point_abs_error_ns":self.max_point_abs_error_ns,
            "prediction_error":self.errors.summary(),
            "by_actual_rows":self.by_rows.iter().map(|(rows,g)| json!({"rows":rows,"error":g.summary()})).collect::<Vec<_>>(),
            "history_row_max_ge_400":self.history_row_max_ge_400.summary(),
            "histogram":"130 fixed bins; ratio quantiles are containing intervals (quarter-octave nonzero buckets), null upper denotes overflow; grouping never selects samples or parameters",
            "first_known_examples":self.examples,"maximum_history_known":self.maximum_history,
            "boundary":"retrospective actual-shape exact query at original issued time; not the original planner-issued forecast"})
    }
}

#[test]
#[ignore = "requires pinned original source7 and cold seed spec; offline new candidate"]
fn archived_source7_seeded_candidate_preserves_original_blocks_and_future_holdout() {
    run_candidate(CandidatePolicy::CheckedSeed);
}

#[test]
#[ignore = "requires pinned original source7 and cold seed spec; new phase-support candidate"]
fn archived_source7_seeded_phase_support_candidate_preserves_original_blocks_and_future_holdout() {
    run_candidate(CandidatePolicy::CheckedSeedFrozenPhaseIntersection);
}

#[test]
#[ignore = "requires pinned original source7 and cold seed spec; zero-column readiness candidate"]
fn archived_source7_zero_column_candidate_preserves_original_blocks_and_future_holdout() {
    run_candidate(CandidatePolicy::CheckedSeedFrozenPhaseIntersectionZeroColumns);
}

#[test]
#[ignore = "requires pinned original negative-tail source; evaluates only the accepted original prefix"]
fn archived_source7_phase_support_preserves_original_clock_rejected_tail() {
    run_candidate_inner(CandidatePolicy::CheckedSeedFrozenPhaseIntersection, true);
}

#[test]
#[ignore = "requires pinned original negative-tail source; zero-column candidate retains the original rejection"]
fn archived_source7_zero_columns_preserves_original_clock_rejected_tail() {
    run_candidate_inner(
        CandidatePolicy::CheckedSeedFrozenPhaseIntersectionZeroColumns,
        true,
    );
}

fn run_candidate(policy: CandidatePolicy) {
    run_candidate_inner(policy, false);
}

fn run_candidate_inner(policy: CandidatePolicy, expect_rejected_tail: bool) {
    let spec = online::spec();
    let seed_spec: SeedSpec = serde_json::from_slice(
        &std::fs::read(
            std::env::var_os("FERRUM_ARCHIVED_SEED_SPEC").expect("cold seed audit spec"),
        )
        .unwrap(),
    )
    .unwrap();
    let original_header = verify_source(&spec);
    let u = seed(&seed_spec, &original_header);
    let h = candidate_header(&original_header, u, policy);
    let budget = NonZeroU64::new(h.maximum_file_bytes).unwrap();
    let limits = CostProfileLoadLimits::default();
    let mut original = StructuredServiceCollectorV7::new_streaming(
        original_header.clone(),
        limits.clone(),
        budget,
    )
    .unwrap();
    let mut candidate =
        StructuredServiceCollectorV7::new_streaming(h.clone(), limits.clone(), budget).unwrap();
    let mut catalog: Option<(u64, ImportedStructuredCatalogV14)> = None;
    let mut evaluation = Evaluation::default();
    let mut histories = BTreeMap::new();
    let mut targets = BTreeMap::<u64, OwnerInputTargetV1>::new();
    let mut completed_blocks = 0;
    let mut records = 0;
    let mut original_failed_tail = false;
    let mut rejected_tail = None;
    let original_header_bytes = record_bytes_v7(&original_header).unwrap();
    let mut raw_prefix = Sha256::new();
    raw_prefix.update(&original_header_bytes);
    let mut raw_prefix_bytes = original_header_bytes.len() as u64;
    let mut lines = BufReader::new(std::fs::File::open(&spec.actual_source).unwrap())
        .split(b'\n')
        .skip(1);
    while let Some(line) = lines.next() {
        let line = line.unwrap();
        let record: StructuredServiceRecordV7 = serde_json::from_slice(&line).unwrap();
        records += 1;
        if let Err(error) = original.push(&record) {
            assert!(
                expect_rejected_tail,
                "unexpected original rejection: {error:?}"
            );
            rejected_tail = Some(rejected_tail::verify_and_close_candidate(
                &original_header,
                &original,
                &mut candidate,
                &record,
                error,
                &mut lines,
                raw_prefix_bytes,
                raw_prefix.clone(),
            ));
            original_failed_tail = true;
            records += 2; // Original matching Failed and Footer, both verified.
            break;
        }
        raw_prefix.update(&line);
        raw_prefix.update(b"\n");
        raw_prefix_bytes += line.len() as u64 + 1;
        match &record {
            StructuredServiceRecordV7::BlockOpen {
                block,
                opened_at_ns,
                fifo_cutoff,
                ..
            } => {
                let fresh = candidate.open_block(*opened_at_ns, *fifo_cutoff).unwrap();
                assert!(
                    matches!(fresh, StructuredServiceRecordV7::BlockOpen { block:b,.. } if b == *block)
                );
            }
            StructuredServiceRecordV7::Completed { wave } => {
                histories.insert(wave.host_stages.call_id, online::raw_history(wave));
                candidate.push(&record).unwrap();
                if let Some((at, catalog)) = &catalog {
                    evaluation.observe(&h, catalog, *at, wave);
                }
            }
            StructuredServiceRecordV7::OutsideDeclaredRoute { .. }
            | StructuredServiceRecordV7::NotSubmitted { .. } => candidate.push(&record).unwrap(),
            StructuredServiceRecordV7::BlockClose { block, closing, .. } => {
                let before = candidate.audit();
                let mut facts = BTreeMap::new();
                for owner in &candidate.owners {
                    let id = owner.contract.owner_attempt_id;
                    if let OwnerState::Fitted(model) = &owner.state {
                        if let Some(diagnostic) =
                            model.diagnose_nonnegative_residual(&owner.samples)
                        {
                            let axis = diagnostic["first_unseen_axis"]["basis_axis"].as_u64();
                            let universe = owner
                                .contract
                                .nonnegative_envelope
                                .as_ref()
                                .and_then(|c| c.algorithm_universe.as_ref());
                            eprintln!(
                                "SEEDED_CANDIDATE_RESIDUAL_FIRST_FAILURE {}",
                                json!({
                                    "candidate_policy":policy,"block":block,"owner_attempt_id":id,"diagnostic":diagnostic,
                                    "algorithm_axis":axis.and_then(|a|algorithm_axis(universe,a as usize))
                                })
                            );
                        }
                    }
                    let fact = (!owner.samples.is_empty())
                        .then(|| OwnerInputTargetV1::from_samples(&owner.samples).unwrap());
                    eprintln!(
                        "SEEDED_CANDIDATE_OWNER {}",
                        json!({"candidate_policy":policy,"block":block,
                        "owner":before.owners.iter().find(|o|o.owner_attempt_id==id),
                        "first_call":owner.samples.first().map(|s|s.call_id),
                        "last_call":owner.samples.last().map(|s|s.call_id),
                        "first_observed_at_ns":owner.samples.first().map(|s|s.observed_at_ns),
                        "last_observed_at_ns":owner.samples.last().map(|s|s.observed_at_ns),
                        "history_sum_min":owner.samples.iter().filter_map(|s|histories.get(&s.call_id)).map(|v|v.0).min(),
                        "history_sum_max":owner.samples.iter().filter_map(|s|histories.get(&s.call_id)).map(|v|v.0).max(),
                        "input_gap":gap(fact.as_ref(),targets.get(&id).or(owner.contract.input_target.as_ref())),
                        "missing_axis_algorithms":gap(fact.as_ref(),targets.get(&id).or(owner.contract.input_target.as_ref()))["missing_positive_axes"].as_array().map(|axes| axes.iter().map(|axis|json!({"axis":axis,"algorithm":algorithm_axis(owner.contract.nonnegative_envelope.as_ref().and_then(|c|c.algorithm_universe.as_ref()),axis.as_u64().unwrap() as usize)})).collect::<Vec<_>>())})
                    );
                    if let Some(fact) = fact {
                        facts.insert(id, fact);
                    }
                }
                let close = candidate.close_block(*closing).unwrap();
                let StructuredServiceRecordV7::BlockClose {
                    freezes,
                    discoveries,
                    ..
                } = &close
                else {
                    unreachable!()
                };
                for f in freezes {
                    eprintln!(
                        "SEEDED_CANDIDATE_FREEZE {}",
                        json!({"candidate_policy":policy,"block":block,"owner_attempt_id":f.owner_attempt_id,
                        "phase":f.close.phase,"members":f.close.member_count,"failure":f.failure,"parameters":f.parameters_sha256})
                    );
                    if f.close.phase == StructuredPhaseV2::Fit && f.failure.is_none() {
                        targets.insert(
                            f.owner_attempt_id,
                            facts.remove(&f.owner_attempt_id).unwrap(),
                        );
                    }
                }
                completed_blocks += 1;
                eprintln!(
                    "SEEDED_CANDIDATE_BLOCK {}",
                    json!({"candidate_policy":policy,"audit":candidate.audit(),"discoveries":discoveries.len()})
                );
                if catalog.is_none() && candidate.qualified_children() > 0 {
                    let (_, checkpoint) = candidate.checkpoint(*closing).unwrap();
                    let imported = checkpoint
                        .activate_same_process_memory_streaming(*closing, &limits, budget)
                        .unwrap();
                    eprintln!(
                        "SEEDED_CANDIDATE_FIRST_ACTIVATION {}",
                        json!({"candidate_policy":policy,"block":block,"original_closing":closing,
                        "children":imported.children.len(),"source_sha256":imported.source_sha256})
                    );
                    catalog = Some((closing.monotonic_ns, imported));
                }
            }
            StructuredServiceRecordV7::Checkpoint { .. } => {} // Never transplant old numerical freezes.
            StructuredServiceRecordV7::Failed {
                ticket,
                fifo,
                at_ns,
                reason,
                ..
            } => {
                original_failed_tail = true;
                candidate
                    .fail(*ticket, *fifo, *at_ns, reason.clone())
                    .unwrap();
            }
            StructuredServiceRecordV7::Footer {
                closing,
                incomplete_block,
                ..
            } => {
                let footer = candidate.stop(*closing).unwrap();
                assert!(
                    matches!(footer,StructuredServiceRecordV7::Footer {incomplete_block:b,..} if b == *incomplete_block)
                );
            }
        }
    }
    assert_eq!(rejected_tail.is_some(), expect_rejected_tail);
    if rejected_tail.is_none() {
        assert_eq!(original.source_receipt().1, spec.actual_source_sha256);
        assert!(original.audit().closed);
    }
    assert!(candidate.audit().closed);
    assert_eq!(original.audit().offered, candidate.audit().offered);
    eprintln!(
        "SEEDED_CANDIDATE_FINAL {}",
        json!({"candidate_policy":policy,"original_source_sha256":spec.actual_source_sha256,
        "original_qualified_children":original.qualified_children(),"candidate_qualified_children":candidate.qualified_children(),
        "candidate_capture_identity":h.capture_identity,"candidate_protocol":h.protocol,
        "original_complete_blocks":completed_blocks,"records":records,"original_failed_tail_preserved":original_failed_tail,
        "original_clock_rejected_tail":rejected_tail,
        "first_activation_ns":catalog.as_ref().map(|v|v.0),"final_audit":candidate.audit(),"evaluation":evaluation.summary(),
        "note":"New explicit candidate; original independent records and clocks preserved, no fabricated complete tail or original archive promotion. Phase member totals are not a per-query-subdomain sample confidence claim."})
    );
}
