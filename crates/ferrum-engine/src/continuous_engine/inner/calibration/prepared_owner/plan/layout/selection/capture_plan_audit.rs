//! SHA-bound offline declaration accounting. The scheduler audit supplies
//! complete geometry results; these DTOs never recreate checked input or leases.
//! Keep the actual schedule/fresh-span/native-work arithmetic in production helpers.
use super::*;
use anyhow::{ensure, Context};
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    NumericalFamilyKeyV1, StructuredSettingsV2,
};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, io::Read, path::PathBuf};

type AuditResult<T> = anyhow::Result<T>;
const CAPTURE_LIMIT: u64 = 256 * 1024 * 1024;
const REFERENCE_LIMIT: u64 = 64 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BoundFile {
    path: PathBuf,
    sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AuditCase {
    capture: BoundFile,
    reference: BoundFile,
    output: PathBuf,
}

fn read_bounded(path: &std::path::Path, limit: u64) -> AuditResult<Vec<u8>> {
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(limit + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= limit,
        "input exceeds audit byte limit"
    );
    Ok(bytes)
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn bound_file(input: &BoundFile, limit: u64) -> AuditResult<Vec<u8>> {
    let bytes = read_bounded(&input.path, limit)?;
    ensure!(
        digest(&bytes) == input.sha256,
        "input SHA differs: {}",
        input.path.display()
    );
    Ok(bytes)
}
fn field<'a>(value: &'a Value, name: &str) -> AuditResult<&'a Value> {
    value.get(name).with_context(|| format!("missing {name}"))
}
fn decode<T: serde::de::DeserializeOwned>(value: &Value, name: &str) -> AuditResult<T> {
    Ok(serde_json::from_value(field(value, name)?.clone())?)
}

#[derive(Deserialize)]
enum Product {
    Prefill,
    ContinuationPrefill { offset: usize },
    PrefillSpan { offset: usize, chunk: NonZeroU32 },
    Greedy,
    Full,
}
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Prefix {
    Ordinary,
    Clean,
    Pending,
    Mixed { pending_rows: usize },
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CapturedAcquisition {
    template: usize,
    maximum_output: NonZeroUsize,
    preset: SloAutomaticCostProbeSamplingPresetV1,
    plan: CapturedPrefixPlan,
    input_tokens_sha256: [u8; 32],
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CapturedPrefixPlan {
    prompt_tokens: NonZeroUsize,
    boundary: NonZeroUsize,
    prefill_chunk: NonZeroU32,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CapturedCase {
    product: Product,
    template: usize,
    width: usize,
    maximum_output: NonZeroUsize,
    release_generated: usize,
    suffix_tokens: usize,
    preset: SloAutomaticCostProbeSamplingPresetV1,
    prefix: Prefix,
    route: CalibrationDecodeRoute,
    reset: bool,
    acquisition: Option<CapturedAcquisition>,
}
impl CapturedCase {
    fn restore(self) -> AuditResult<Case> {
        ensure!(self.width > 0, "zero captured width");
        let mut case = Case {
            product: match self.product {
                Product::Prefill => OpportunityProduct::Prefill,
                Product::ContinuationPrefill { offset } => {
                    OpportunityProduct::ContinuationPrefill { offset }
                }
                Product::PrefillSpan { offset, chunk } => {
                    OpportunityProduct::PrefillSpan { offset, chunk }
                }
                Product::Greedy => OpportunityProduct::Greedy,
                Product::Full => OpportunityProduct::Full,
            },
            template: self.template,
            width: self.width,
            maximum_output: self.maximum_output,
            release_generated: self.release_generated,
            suffix_tokens: self.suffix_tokens,
            preset: self.preset,
            prefix: match self.prefix {
                Prefix::Ordinary => PrefixKind::Ordinary,
                Prefix::Clean => PrefixKind::Clean,
                Prefix::Pending => PrefixKind::Pending,
                Prefix::Mixed { pending_rows } => PrefixKind::Mixed { pending_rows },
            },
            route: self.route,
            reset: self.reset,
            acquisition: None,
        };
        if let Some(key) = self.acquisition {
            ensure!(
                key.template == case.template
                    && key.maximum_output == case.maximum_output
                    && key.preset == case.preset,
                "captured acquisition/case binding differs"
            );
            case.acquisition = Some(work::acquisition_from_capture_for_test(
                &case,
                key.plan.prompt_tokens.get(),
                key.plan.boundary.get(),
                key.plan.prefill_chunk,
                key.input_tokens_sha256,
            )?);
        }
        Ok(case)
    }
}

#[derive(Deserialize)]
enum Key {
    NumericalFamily(NumericalFamilyKeyV1),
    ExactOwner(StructuredOwnerKeyV2),
}
impl Key {
    fn comparison_value(self) -> CheckedPopulationKey {
        match self {
            Self::NumericalFamily(key) => CheckedPopulationKey::NumericalFamily(key),
            Self::ExactOwner(key) => CheckedPopulationKey::ExactOwner(key),
        }
    }
}
struct CapturedPopulation {
    index: usize,
    key: CheckedPopulationKey,
    matrix: Value,
    result: Value,
}
struct Capture {
    input_sha256: Value,
    config: ferrum_types::EngineConfig,
    cases: Vec<Case>,
    prompts: Vec<usize>,
    chunk: usize,
    row_ceiling: Option<NonZeroU32>,
    populations: Vec<CapturedPopulation>,
    selection: Value,
    retirement: Value,
}

fn parse_capture(bytes: &[u8]) -> AuditResult<Capture> {
    let (mut header, mut config, mut table, mut selection, mut retirement) =
        (None, None, None, None, None);
    let mut pending = None;
    let mut populations = Vec::new();
    let mut completed = false;
    for line in bytes.split(|&b| b == b'\n').filter(|line| !line.is_empty()) {
        ensure!(!completed, "record after completed footer");
        let mut record: Value = serde_json::from_slice(line)?;
        match field(&record, "kind")?
            .as_str()
            .context("record kind is not text")?
        {
            "ferrum.test.cold_geometry.v1" => {
                ensure!(header.is_none(), "duplicate capture header");
                header = Some(field(&record, "input_sha256")?.clone());
            }
            "actual_startup_inputs" => {
                ensure!(config.is_none(), "duplicate startup inputs");
                config = Some(decode(&record, "config")?);
            }
            "original_cases" => {
                ensure!(table.is_none(), "duplicate case table");
                table = Some(record);
            }
            "population" => {
                ensure!(pending.is_none(), "unclosed population");
                let key: Key = decode(&record, "key")?;
                pending = Some(CapturedPopulation {
                    index: decode(&record, "population_index")?,
                    key: key.comparison_value(),
                    matrix: Value::Null,
                    result: Value::Null,
                });
            }
            "original_matrix" => {
                let entry = pending.as_mut().context("matrix without population")?;
                ensure!(entry.matrix.is_null(), "duplicate matrix");
                // The SHA-bound scheduler report has already replayed every bit.
                // Work accounting needs case IDs/settings/anchors, not a second matrix copy.
                record
                    .as_object_mut()
                    .context("matrix not object")?
                    .remove("axis_bits");
                entry.matrix = record;
            }
            "original_result" => {
                let mut entry = pending.take().context("result without population")?;
                ensure!(!entry.matrix.is_null(), "result before matrix");
                entry.result = record;
                populations.push(entry);
            }
            "final_selection" => {
                ensure!(selection.is_none(), "duplicate selection");
                selection = Some(field(&record, "selection")?.clone());
            }
            "inventory_retired" => {
                ensure!(retirement.is_none(), "duplicate retirement");
                ensure!(
                    decode::<bool>(&record, "complete")?,
                    "incomplete inventory retirement"
                );
                retirement = Some(record);
            }
            "completed_after_shutdown" => {
                ensure!(
                    pending.is_none() && decode::<usize>(&record, "matrices")? == populations.len(),
                    "footer matrix count differs"
                );
                completed = true;
            }
            other => anyhow::bail!("unsupported capture record {other}"),
        }
    }
    ensure!(completed, "missing shutdown footer");
    let table = table.context("missing cases")?;
    let captured: Vec<CapturedCase> = decode(&table, "cases")?;
    let cases = captured
        .into_iter()
        .map(CapturedCase::restore)
        .collect::<AuditResult<Vec<_>>>()?;
    ensure!(
        serde_json::to_value(&cases)? == *field(&table, "cases")?,
        "case scalar round trip changed the original declaration"
    );
    Ok(Capture {
        input_sha256: header.context("missing header")?,
        config: config.context("missing config")?,
        cases,
        prompts: decode(&table, "prompts")?,
        chunk: decode(&table, "chunk")?,
        row_ceiling: decode(&table, "prefill_row_ceiling")?,
        populations,
        selection: selection.context("missing final selection")?,
        retirement: retirement.context("missing retirement")?,
    })
}

fn checked_cases(ids: &[usize], matrix: &Value, case_count: usize) -> AuditResult<()> {
    let candidates: Vec<usize> = decode(matrix, "cases")?;
    let anchors: Vec<usize> = decode(matrix, "mandatory_anchors")?;
    ensure!(
        !ids.is_empty() && ids.windows(2).all(|w| w[0] < w[1]),
        "final IDs must be nonempty, sorted and unique"
    );
    ensure!(
        ids.iter()
            .all(|i| *i < case_count && candidates.contains(i)),
        "final ID outside original guaranteed matrix"
    );
    for anchor in anchors {
        ensure!(
            ids.contains(candidates.get(anchor).context("anchor outside matrix")?),
            "original mandatory anchor omitted"
        );
    }
    Ok(())
}

fn compare_original(actual: &SelectedBatch, expected: &Value) -> AuditResult<()> {
    let actual = serde_json::to_value(actual)?;
    for name in [
        "population_indices",
        "representative_case_indices",
        "input_opportunities",
        "schedule",
        "schedule_within_capacity",
        "planned_cycles",
        "maximum_anchor_span",
        "requests",
        "serial_token_work",
        "serial_wave_upper_bound",
        "declared_offer_row_bound",
        "algorithm_universe",
    ] {
        ensure!(
            actual.get(name) == expected.get(name),
            "original batch {name} differs: actual={} expected={}",
            actual.get(name).unwrap_or(&Value::Null),
            expected.get(name).unwrap_or(&Value::Null)
        );
    }
    Ok(())
}

fn audit(capture: &Capture, reference: &Value) -> AuditResult<Value> {
    ensure!(
        field(reference, "input_sha256")? == &capture.input_sha256,
        "input SHA differs"
    );
    let original_audit = field(reference, "audit")?;
    ensure!(
        decode::<bool>(original_audit, "baseline_matched")?,
        "reference lacks exact original replay"
    );
    ensure!(
        field(original_audit, "original_selection")? == &capture.selection,
        "reference selection differs"
    );
    let demand = field(reference, "independent_current_demand")?;
    ensure!(
        field(demand, "geometry_kernel")? == field(original_audit, "captured_geometry_kernel")?,
        "current demand is not captured kernel"
    );
    ensure!(
        decode::<u64>(demand, "diagnostic_allowance_per_call")? == u64::MAX,
        "reference is not complete-demand measurement"
    );
    let calls = field(demand, "calls")?
        .as_array()
        .context("reference calls not array")?;
    let source_populations = field(&capture.selection, "populations")?
        .as_array()
        .context("populations not array")?;
    let batches = field(&capture.selection, "batches")?
        .as_array()
        .context("batches not array")?;
    ensure!(
        calls.len() == capture.populations.len() && source_populations.len() == calls.len(),
        "population/reference count differs"
    );
    // This adapter measures independent populations. A merged original source
    // requires original per-case membership metadata rather than an invented split.
    ensure!(
        batches.len() == calls.len()
            && batches
                .iter()
                .all(|b| decode::<Vec<usize>>(b, "population_indices").is_ok_and(|v| v.len() == 1)),
        "only original single-population sources are supported"
    );
    let automatic = match &capture
        .config
        .scheduler
        .slo
        .cost_observation
        .live_structured_calibration
    {
        ferrum_types::SloLiveStructuredCalibration::AutomaticV1 { settings } => settings,
        _ => anyhow::bail!("capture is not explicit automatic calibration"),
    };
    let imports = &capture.config.scheduler.slo.cost_observation.profile_import;
    let available = SelectionCapacity {
        requests: decode::<usize>(&capture.retirement, "selected_requests")?
            .checked_add(decode(&capture.selection, "requests")?)
            .context("request capacity overflow")?,
        execution_actions: decode::<usize>(&capture.retirement, "selected_actions")?
            .checked_add(decode(&capture.selection, "serial_wave_upper_bound")?)
            .context("action capacity overflow")?,
        declared_offer_rows: imports
            .max_samples
            .get()
            .min(imports.max_total_shape_rows.get()),
    };
    ensure!(
        available.requests <= automatic.cost_probe.maximum_probe_requests.get()
            && available.execution_actions <= automatic.cost_probe.maximum_offered_waves.get(),
        "selection allowance exceeds original configuration"
    );
    let mut outputs = Vec::new();
    let (mut requests, mut actions, mut rows) = (0usize, 0usize, 0usize);
    for (ordinal, captured) in capture.populations.iter().enumerate() {
        let call = &calls[ordinal];
        ensure!(
            decode::<usize>(call, "ordinal")? == ordinal
                && decode::<usize>(&captured.matrix, "ordinal")? == ordinal
                && decode::<usize>(&captured.result, "ordinal")? == ordinal
                && decode::<usize>(call, "population_index")? == captured.index,
            "reference/capture ordinal or population mismatch"
        );
        ensure!(
            field(call, "original_case_indices")? == field(&captured.matrix, "cases")?
                && field(call, "mandatory_anchor_indices")?
                    == field(&captured.matrix, "mandatory_anchors")?,
            "reference matrix identity differs"
        );
        ensure!(
            decode::<bool>(call, "complete")? && field(call, "error")?.is_null(),
            "incomplete full-reference population {}",
            captured.index
        );
        let captured_population = source_populations
            .get(captured.index)
            .context("population index outside selection")?;
        ensure!(
            serde_json::to_value(&captured.key)? == *field(captured_population, "key")?,
            "captured population key differs"
        );
        let original_ids: Vec<usize> = decode(&captured.result, "final_selected_cases")?;
        ensure!(
            original_ids
                == decode::<Vec<usize>>(captured_population, "representative_case_indices")?,
            "original population representatives differ"
        );
        let complete_ids: Vec<usize> = decode(call, "final_selected_cases")?;
        checked_cases(&original_ids, &captured.matrix, capture.cases.len())?;
        checked_cases(&complete_ids, &captured.matrix, capture.cases.len())?;
        let expected: Vec<_> = batches
            .iter()
            .filter(|b| {
                decode::<Vec<usize>>(b, "population_indices").is_ok_and(|v| v == [captured.index])
            })
            .collect();
        ensure!(
            expected.len() == 1,
            "original population batch is not unique"
        );
        let expected = expected[0];
        let schedule: OwnerBlockScheduleV1 = decode(expected, "schedule")?;
        let settings: StructuredSettingsV2 = decode(&captured.matrix, "settings")?;
        // batch_plan reads ONLY settings and schedule.prediction_validity from
        // this container. Remaining fields are inert accounting metadata, not
        // a physical-domain declaration or a source that may be executed.
        let declaration = StructuredServiceDeclarationV7 {
            schedule,
            settings,
            route_population: automatic.route_population,
            domain_policy: Default::default(),
            nonnegative_envelope: None,
            maximum_window_ns: automatic.maximum_window_ns.get(),
            maximum_owners: automatic.maximum_owners.get(),
            maximum_retained_numeric_bytes: automatic.maximum_retained_numeric_bytes.get(),
            maximum_discovery_bytes: automatic.maximum_discovery_bytes.get(),
        };
        let candidates: Vec<usize> = decode(&captured.matrix, "cases")?;
        // Capture.matrix receives only member_groups' Unique/floor=1 cases,
        // after every checked alternative agrees in both axes and branches.
        // This provenance is a conditional input floor, never an observed member.
        let mut opportunities = vec![
            CaseOpportunity {
                population: CasePopulation::Unknown {
                    known_alternatives: Vec::new()
                },
                minimum_fresh_members: 0
            };
            capture.cases.len()
        ];
        for index in candidates {
            *opportunities
                .get_mut(index)
                .context("matrix case outside table")? = CaseOpportunity {
                population: CasePopulation::Unique(captured.key.clone()),
                minimum_fresh_members: 1,
            };
        }
        let make = |ids: Vec<usize>| -> AuditResult<SelectedBatch> {
            let population = SelectedPopulation {
                key: captured.key.clone(),
                representative_case_indices: ids,
                maximum_anchor_span: 0,
                scheduled: false,
                batch_index: None,
                input_geometry: None,
            };
            let mut batch = batch_plan(
                &[0],
                &[population],
                &capture.cases,
                &opportunities,
                &capture.prompts,
                capture.chunk,
                capture.row_ceiling,
                &declaration,
            )?;
            batch.population_indices = vec![captured.index];
            // Inputs already name the captured projected family. Do not replay
            // recipes or claim a new U; carry only the original wire comparison value.
            batch.algorithm_universe = expected
                .get("algorithm_universe")
                .map(|v| serde_json::from_value(v.clone()))
                .transpose()?;
            Ok(batch)
        };
        let original = make(original_ids)?;
        compare_original(&original, expected)
            .with_context(|| format!("population {} original accounting", captured.index))?;
        let complete = make(complete_ids)?;
        requests = requests
            .checked_add(complete.requests)
            .context("total requests overflow")?;
        actions = actions
            .checked_add(complete.serial_wave_upper_bound)
            .context("total actions overflow")?;
        rows = rows
            .checked_add(complete.declared_offer_row_bound)
            .context("total rows overflow")?;
        let fits_requests = complete.requests <= available.requests;
        let fits_actions = complete.serial_wave_upper_bound <= available.execution_actions;
        let fits_rows = complete.declared_offer_row_bound <= available.declared_offer_rows;
        let fits_schedule = complete.schedule_within_capacity
            && complete.maximum_anchor_span
                <= *complete.schedule.phase_min_offered.iter().min().unwrap();
        let fits = fits_requests && fits_actions && fits_rows && fits_schedule;
        let mut widths: Vec<_> = complete
            .representative_case_indices
            .iter()
            .map(|&i| capture.cases[i].width)
            .collect();
        widths.sort_unstable();
        widths.dedup();
        outputs.push(json!({ "population_index": captured.index, "population_key": captured.key,
            "original_batch_accounting_matched": true, "original_scheduled": field(expected, "scheduled")?,
            "geometry_rank": field(call, "rank")?, "geometry_anchor_rank": field(call, "anchor_rank")?,
            "geometry_full_visits": field(call, "full_visits")?, "widths": widths,
            "fits_alone_in_original_selection_allowance": fits,
            "fit_checks": { "requests": fits_requests, "execution_actions": fits_actions,
                "declared_offer_rows": fits_rows, "schedule_and_anchor_span": fits_schedule },
            "complete_input_plan": complete }));
    }
    Ok(json!({
        "scope": "Independent captured populations only; original current-kernel MAX final cases; no reselection, recipe authority, source merge or measured F/R/Q.",
        "all_original_batch_accounting_matched": true,
        "original_available_selection_capacity": {"requests": available.requests, "execution_actions": available.execution_actions, "declared_offer_rows": available.declared_offer_rows},
        "original_retirement_remaining": capture.retirement,
        "maximum_sources": automatic.maximum_retained_generations.get(),
        "geometry_full_required_visits": field(demand, "full_required_visits")?,
        "geometry_original_limit": field(original_audit, "maximum_visits")?,
        "independent_sources": outputs,
        "all_independent_sources_total": {"sources": calls.len(), "requests": requests, "execution_actions": actions, "declared_offer_rows": rows},
        "limits": ["These populations are candidates, not a requirement to collect every one.",
            "No global sorting/coalescing/preservation proof: original raw recipes and linked trajectories are absent.",
            "Input floors are conditional on fresh cohort completion; no F/R/Q samples, Known query, witness, deadline or native lease is established.",
            "MAX geometry work is a diagnostic demand; the original spent geometry ledger is never reset or refunded."]
    }))
}

#[test]
#[ignore = "requires SHA-bound geometry-plan-audit-case.json beside the engine libtest and completed scheduler MAX audit"]
fn captured_complete_geometry_reuses_original_source_work_and_phase_budget() -> AuditResult<()> {
    let descriptor_path = std::env::current_exe()?.with_file_name("geometry-plan-audit-case.json");
    let descriptor_bytes = read_bounded(&descriptor_path, 64 * 1024)?;
    let descriptor: AuditCase = serde_json::from_slice(&descriptor_bytes)?;
    ensure!(
        descriptor.output != descriptor.capture.path
            && descriptor.output != descriptor.reference.path
            && descriptor.output != descriptor_path,
        "output overlaps input"
    );
    let capture_bytes = bound_file(&descriptor.capture, CAPTURE_LIMIT)?;
    let reference_bytes = bound_file(&descriptor.reference, REFERENCE_LIMIT)?;
    let reference: Value = serde_json::from_slice(&reference_bytes)?;
    ensure!(
        field(&reference, "capture_sha256")?.as_str() == Some(descriptor.capture.sha256.as_str()),
        "reference binds a different capture"
    );
    let capture = parse_capture(&capture_bytes)?;
    let result = audit(&capture, &reference)?;
    let output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&descriptor.output)?;
    serde_json::to_writer_pretty(
        output,
        &json!({
            "descriptor_sha256": digest(&descriptor_bytes), "capture_sha256": descriptor.capture.sha256,
            "reference_sha256": descriptor.reference.sha256, "audit": result,
        }),
    )?;
    Ok(())
}

#[test]
fn capture_plan_native_scalar_restore_preserves_original_binding_and_cost() {
    let mut original = Case {
        product: OpportunityProduct::Full,
        template: 0,
        width: 2,
        maximum_output: NonZeroUsize::new(4).unwrap(),
        release_generated: 2,
        suffix_tokens: 2,
        preset: SloAutomaticCostProbeSamplingPresetV1::Configured,
        prefix: PrefixKind::Pending,
        route: CalibrationDecodeRoute::Actual,
        reset: false,
        acquisition: None,
    };
    original.acquisition = Some(
        work::acquisition_from_capture_for_test(
            &original,
            7,
            6,
            NonZeroU32::new(2).unwrap(),
            [7; 32],
        )
        .unwrap(),
    );
    let value = serde_json::to_value(&original).unwrap();
    let restored = serde_json::from_value::<CapturedCase>(value.clone())
        .unwrap()
        .restore()
        .unwrap();
    assert_eq!(serde_json::to_value(&restored).unwrap(), value);
    assert_eq!(
        work::case_work(&original, 7, 4, None).unwrap(),
        work::case_work(&restored, 7, 4, None).unwrap()
    );
    assert_eq!(
        work::setup_for_cases(&[original.clone(), original]).unwrap(),
        work::setup_for_cases(&[restored.clone(), restored]).unwrap()
    );
    let mut wrong = value.clone();
    wrong["acquisition"]["template"] = json!(1);
    assert!(serde_json::from_value::<CapturedCase>(wrong)
        .unwrap()
        .restore()
        .is_err());
    let mut wrong = value;
    wrong["prefix"] = json!("ordinary");
    assert!(serde_json::from_value::<CapturedCase>(wrong)
        .unwrap()
        .restore()
        .is_err());
}

#[test]
fn capture_plan_keeps_original_anchors_and_rejects_outside_case_ids() {
    let matrix = json!({"cases": [2, 5, 8], "mandatory_anchors": [0, 2]});
    assert!(checked_cases(&[2, 8], &matrix, 9).is_ok());
    for ids in [&[2, 5][..], &[2, 8, 9][..], &[2, 8, 8][..], &[8, 2][..]] {
        assert!(checked_cases(ids, &matrix, 9).is_err());
    }
}
