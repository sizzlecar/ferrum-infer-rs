//! SHA-bound subset experiment using historical witnesses, never new authority.
//! The current captured rows supply every numeric coordinate and span check.
use super::*;
use serde_json::{json, Value};
use std::{collections::BTreeSet, mem::size_of};

const WIDTHS: [usize; 2] = [1, 4];
const BINDING_LIMIT: u64 = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Mode {
    SymmetricWidthsOneFourV1,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Options {
    mode: Mode,
    g18_capture: BoundFile,
    g20_header: BoundFile,
    anchor_audit: BoundFile,
}

struct HistoricalCall {
    population: Population,
    cases: Vec<usize>,
    anchors: Vec<usize>,
    settings: StructuredSettingsV2,
}
struct History {
    cases: CaseTable,
    config: Box<ferrum_types::EngineConfig>,
    templates: Vec<Value>,
    calls: Vec<HistoricalCall>,
}

fn history(bytes: &[u8]) -> AuditResult<History> {
    let mut stream = serde_json::Deserializer::from_slice(bytes).into_iter::<Record>();
    let mut next = || {
        stream
            .next()
            .context("historical capture truncated")?
            .context("historical record invalid")
    };
    ensure!(
        matches!(next()?, Record::Header { .. }),
        "historical header missing"
    );
    let Record::Startup { config, templates } = next()? else {
        bail!("historical startup missing")
    };
    let Record::Cases(cases) = next()? else {
        bail!("historical cases missing")
    };
    let mut calls = Vec::new();
    let mut populations = BTreeSet::new();
    loop {
        let population = match next()? {
            Record::Population(p) => p,
            Record::Retirement(r) => {
                ensure!(r.complete, "historical inventory not retired");
                break;
            }
            _ => bail!("historical population/retirement order differs"),
        };
        let Record::Matrix(matrix) = next()? else {
            bail!("historical matrix missing")
        };
        let Record::Result(result) = next()? else {
            bail!("historical result missing")
        };
        ensure!(
            matrix.ordinal == calls.len()
                && result.ordinal == calls.len()
                && populations.insert(population.population_index),
            "historical ordinal/population differs"
        );
        ensure!(
            !matrix.cases.is_empty()
                && matrix.cases.len() == matrix.axis_bits.len()
                && matrix.cases.windows(2).all(|p| p[0] < p[1])
                && matrix.cases.iter().all(|&i| i < cases.cases.len())
                && matrix.mandatory_anchors.windows(2).all(|p| p[0] < p[1])
                && matrix
                    .mandatory_anchors
                    .iter()
                    .all(|&i| i < matrix.cases.len()),
            "historical case/anchor map invalid"
        );
        calls.push(HistoricalCall {
            population,
            cases: matrix.cases,
            anchors: matrix.mandatory_anchors,
            settings: matrix.settings,
        });
        // Old axis bits are decoded as u64, then discarded, not compared with
        // the new U or used to create current coordinates/branch flags.
    }
    let Record::Completed { matrices } = next()? else {
        bail!("historical shutdown missing")
    };
    ensure!(
        matrices == calls.len() && matrices > 0,
        "historical matrix count differs"
    );
    drop(next);
    ensure!(stream.next().is_none(), "historical trailing records");
    Ok(History {
        cases,
        config,
        templates,
        calls,
    })
}

fn field<'a>(value: &'a Value, name: &str) -> AuditResult<&'a Value> {
    value
        .get(name)
        .with_context(|| format!("missing binding field {name}"))
}
fn indices(value: &Value, name: &str) -> AuditResult<Vec<usize>> {
    Ok(serde_json::from_value(field(value, name)?.clone())?)
}
fn normalized_config(config: &ferrum_types::EngineConfig) -> AuditResult<Value> {
    let mut value = serde_json::to_value(config)?;
    for path in [
        "/scheduler/slo/cost_observation/live_structured_calibration/settings/reuse/location/path",
        "/runtime/device_memory_sampling/jsonl_path",
    ] {
        if let Some(path) = value.pointer_mut(path) {
            *path = Value::Null;
        }
    }
    Ok(value)
}
fn request_without_ephemeral(mut value: Value) -> Value {
    if let Some(object) = value.as_object_mut() {
        object.remove("id");
        object.remove("created_at");
    }
    value
}
fn templates(values: &[Value]) -> AuditResult<Vec<Value>> {
    values.iter().map(|value| {
        let bytes: Vec<u8> = serde_json::from_value(field(value, "request_bytes")?.clone())?;
        Ok(json!({"output":field(value,"output")?, "request":request_without_ephemeral(serde_json::from_slice(&bytes)?)}))
    }).collect()
}
fn same_family(a: &Value, b: &Value) -> bool {
    [
        "workload_domain",
        "host_policy",
        "product",
        "readback",
        "route",
    ]
    .iter()
    .all(|name| a.get(name).is_some() && a.get(name) == b.get(name))
}

struct Mapping {
    subset: Vec<usize>,
    historical: Vec<usize>,
    narrow: Vec<usize>,
    old_representatives: Vec<usize>,
    raw_populations: Vec<usize>,
}
fn mappings(
    verified: &Verified,
    old: &History,
    header: &Value,
    audit: &Value,
) -> AuditResult<Vec<Option<Mapping>>> {
    ensure!(
        old.cases == verified.cases,
        "historical/current case table differs"
    );
    ensure!(
        normalized_config(&old.config)? == normalized_config(&verified.startup_config)?,
        "historical/current startup config differs"
    );
    let old_templates = templates(&old.templates)?;
    ensure!(
        old_templates == templates(&verified.startup_templates)?,
        "historical/current requests differ"
    );
    let parent = header
        .pointer("/declaration/cohort_manifest_payload/parent")
        .context("source8 parent missing")?;
    let header_cases: Vec<OriginalCase> = serde_json::from_value(
        parent
            .pointer("/parent/cases")
            .context("source8 cases missing")?
            .clone(),
    )?;
    ensure!(
        header_cases == verified.cases.cases,
        "source8/current cases differ"
    );
    let header_templates = parent
        .pointer("/parent/templates")
        .and_then(Value::as_array)
        .context("source8 templates missing")?;
    for template in &old_templates {
        let matching = header_templates
            .iter()
            .filter(|entry| {
                entry.get("output") == template.get("output")
                    && entry.get("request").is_some_and(|request| {
                        request_without_ephemeral(request.clone()) == template["request"]
                    })
            })
            .count();
        ensure!(
            matching == 1,
            "original request does not have one exact source8 template"
        );
    }
    let populations = parent
        .pointer("/child/checked_selection/populations")
        .and_then(Value::as_array)
        .context("source8 populations missing")?;
    ensure!(
        old.calls.len() == populations.len(),
        "raw population inventory count differs"
    );
    for call in &old.calls {
        let raw = populations
            .get(call.population.population_index)
            .context("source8 raw population missing")?;
        ensure!(
            field(raw, "key")? == &call.population.key,
            "raw family identity differs"
        );
    }
    let audits = field(audit, "populations")?
        .as_array()
        .context("anchor audit populations not array")?;
    let mut result = Vec::new();
    let mut used_audits = BTreeSet::new();
    for call in &verified.calls {
        let Some(family) = call.population.key.get("NumericalFamily") else {
            let matching: Vec<_> = old
                .calls
                .iter()
                .filter(|raw| raw.population.key == call.population.key)
                .collect();
            ensure!(
                call.population.key.get("ExactOwner").is_some() && matching.len() == 1,
                "unchanged Prefill control lacks one exact historical owner"
            );
            let raw = matching[0];
            ensure!(
                raw.cases == call.matrix.cases
                    && serde_json::to_value(&raw.settings)?
                        == serde_json::to_value(&call.matrix.settings)?
                    && raw.anchors.iter().all(|anchor| {
                        call.matrix.mandatory_anchors.binary_search(anchor).is_ok()
                    }),
                "unchanged Prefill control lost original cases, settings or anchors"
            );
            result.push(None);
            continue;
        };
        let matching: Vec<_> = audits
            .iter()
            .enumerate()
            .filter(|(_, a)| {
                a.get("g21_population_index").and_then(Value::as_u64)
                    == Some(call.population.population_index as u64)
            })
            .collect();
        ensure!(
            matching.len() == 1,
            "current population audit is not unique"
        );
        let (audit_index, entry) = matching[0];
        ensure!(
            used_audits.insert(audit_index)
                && entry.get("g21_ordinal").and_then(Value::as_u64)
                    == Some(call.matrix.ordinal as u64)
                && field(entry, "g21_family")? == family,
            "current family/ordinal differs"
        );
        let subset: Vec<_> = call
            .matrix
            .cases
            .iter()
            .copied()
            .filter(|&i| WIDTHS.contains(&verified.cases.cases[i].width))
            .collect();
        let mut original = BTreeSet::new();
        let mut historical = BTreeSet::new();
        let mut narrow = BTreeSet::new();
        let mut old_representatives = BTreeSet::new();
        let mut raw_populations = Vec::new();
        let raw_audits = field(entry, "g18_raw_population_mapping")?
            .as_array()
            .context("raw mappings not array")?;
        for (ordinal, raw) in old.calls.iter().enumerate() {
            let Some(key) = raw.population.key.get("NumericalFamily") else {
                continue;
            };
            let width = verified.cases.cases[raw.cases[0]].width;
            if !same_family(key, family) || !WIDTHS.contains(&width) {
                continue;
            }
            ensure!(
                raw.cases
                    .iter()
                    .all(|&i| verified.cases.cases[i].width == width),
                "raw width is not homogeneous"
            );
            ensure!(
                serde_json::to_value(&raw.settings)?
                    == serde_json::to_value(&call.matrix.settings)?,
                "numerical settings differ"
            );
            let anchors: Vec<_> = raw.anchors.iter().map(|&i| raw.cases[i]).collect();
            let matching: Vec<_> = raw_audits
                .iter()
                .filter(|a| {
                    a.get("g18_population_index").and_then(Value::as_u64)
                        == Some(raw.population.population_index as u64)
                })
                .collect();
            ensure!(matching.len() == 1, "historical raw audit is not unique");
            let a = matching[0];
            ensure!(
                a.get("g18_ordinal").and_then(Value::as_u64) == Some(ordinal as u64)
                    && a.get("g20_population_index").and_then(Value::as_u64)
                        == Some(raw.population.population_index as u64)
                    && field(a, "original_raw_key")? == &raw.population.key
                    && indices(a, "candidate_case_indices")? == raw.cases
                    && indices(a, "mandatory_row_indices")? == raw.anchors
                    && indices(a, "mandatory_case_indices")? == anchors,
                "historical audit case/anchor identity differs"
            );
            original.extend(raw.cases.iter().copied());
            historical.extend(anchors);
            if width == 1 {
                narrow.extend(raw.cases.iter().copied());
                let population = &populations[raw.population.population_index];
                ensure!(
                    population.get("scheduled") == Some(&Value::Bool(true)),
                    "old narrow family was not scheduled"
                );
                let reps = indices(population, "representative_case_indices")?;
                ensure!(
                    reps.iter().all(|i| raw.cases.binary_search(i).is_ok()),
                    "old narrow representative outside raw candidates"
                );
                old_representatives.extend(reps);
            }
            raw_populations.push(raw.population.population_index);
        }
        let original: Vec<_> = original.into_iter().collect();
        let historical: Vec<_> = historical.into_iter().collect();
        ensure!(
            !subset.is_empty()
                && raw_populations.len() == raw_audits.len()
                && original == subset
                && indices(entry, "g21_subset_candidate_case_indices")? == subset
                && indices(entry, "mandatory_case_indices")? == historical
                && historical.iter().all(|i| subset.binary_search(i).is_ok()),
            "subset/raw candidate or historical anchor coverage differs"
        );
        result.push(Some(Mapping {
            subset,
            historical,
            narrow: narrow.into_iter().collect(),
            old_representatives: old_representatives.into_iter().collect(),
            raw_populations,
        }));
    }
    ensure!(used_audits.len() == audits.len(), "unused audit population");
    Ok(result)
}

struct Prepared {
    rows: Vec<usize>,
    anchors: Vec<usize>,
    endpoints: Vec<usize>,
    bytes: usize,
}
fn prepare(
    verified: &Verified,
    call: &super::Call,
    mapping: &Mapping,
    work: &mut GeometryWork,
) -> Result<Prepared> {
    if work.exhausted {
        return Err(StructuredUnknown::Capacity);
    }
    let n = call.matrix.cases.len();
    // Source row map, anchor and endpoint vectors, and one shared role mask.
    // Reserve full n before filtering, including all simultaneous backing.
    let bytes = n
        .checked_mul(3 * size_of::<usize>() + size_of::<u8>())
        .and_then(|v| v.checked_add(size_of::<Prepared>() + size_of::<Vec<u8>>()))
        .ok_or(StructuredUnknown::Capacity)?;
    if bytes > call.matrix.maximum_scratch_bytes {
        return Err(StructuredUnknown::Capacity);
    }
    let mut rows = Vec::with_capacity(n);
    for (index, &case) in call.matrix.cases.iter().enumerate() {
        work.charge(3)?;
        if WIDTHS.contains(&verified.cases.cases[case].width) {
            rows.push(index);
        }
    }
    work.charge(rows.len())?;
    let mut mask = vec![0u8; rows.len()];
    // Linear original-index walks charge all equality/index inspections.
    for &anchor in &mapping.historical {
        let mut found = false;
        for (i, &source) in rows.iter().enumerate() {
            work.charge(2)?;
            if call.matrix.cases[source] == anchor {
                mask[i] |= 1;
                found = true;
                break;
            }
        }
        if !found {
            return Err(StructuredUnknown::InvalidInput);
        }
    }
    let d = call.matrix.axis_bits[rows[0]].len();
    for axis in 0..d {
        let mut minimum = None;
        let mut maximum = None;
        let mut positive = None;
        let better_tie = |a: usize, b: usize| {
            let ca = call.matrix.cases[rows[a]];
            let cb = call.matrix.cases[rows[b]];
            (mask[a] == 0, verified.cases.cases[ca].width, ca)
                < (mask[b] == 0, verified.cases.cases[cb].width, cb)
        };
        for (i, &source) in rows.iter().enumerate() {
            // Three extremum comparisons and their bounded original-index /
            // width / existing-anchor tie breaks; no unordered hash shortcut.
            work.charge(24)?;
            let value = f64::from_bits(call.matrix.axis_bits[source][axis]);
            if !value.is_finite() || value < 0. || value > (1u64 << 53) as f64 {
                return Err(StructuredUnknown::InvalidInput);
            }
            let value_at = |j: usize| f64::from_bits(call.matrix.axis_bits[rows[j]][axis]);
            if minimum
                .is_none_or(|j| value < value_at(j) || (value == value_at(j) && better_tie(i, j)))
            {
                minimum = Some(i);
            }
            if maximum
                .is_none_or(|j| value > value_at(j) || (value == value_at(j) && better_tie(i, j)))
            {
                maximum = Some(i);
            }
            if value > 0.
                && positive.is_none_or(|j| {
                    value < value_at(j) || (value == value_at(j) && better_tie(i, j))
                })
            {
                positive = Some(i);
            }
        }
        for index in [minimum, maximum, positive].into_iter().flatten() {
            work.charge(1)?;
            mask[index] |= 2;
        }
    }
    let mut anchors = Vec::with_capacity(n);
    let mut endpoints = Vec::with_capacity(n);
    for (i, &flag) in mask.iter().enumerate() {
        work.charge(3)?;
        if flag != 0 {
            anchors.push(i);
        }
        if flag & 2 != 0 {
            endpoints.push(call.matrix.cases[rows[i]]);
        }
    }
    Ok(Prepared {
        rows,
        anchors,
        endpoints,
        bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Wire-only identities and rows: not a model projection or GPU fixture.
    fn fixture() -> (Verified, History, Value, Value) {
        let settings = StructuredSettingsV2::default();
        let cases = [1, 4, 4, 8].map(|width| {
            json!({
                "product":"Full","template":0,"width":width,"maximum_output":3,
                "release_generated":1,"suffix_tokens":2,"preset":"greedy_length",
                "prefix":"pending","route":"actual","reset":false
            })
        });
        let table = json!({"cases":cases,"prompts":[14],"chunk":8,"prefill_row_ceiling":null});
        let family = |algorithm: u8| {
            json!({"NumericalFamily":{
                "workload_domain":vec![9u8;32],"algorithm_domain":vec![algorithm;32],
                "host_policy":{"wire_fixture_policy":1},"product":"FullLogits",
                "readback":"host_synchronized","route":[0,0,2,0,0],"basis_axes":4,"support_axes":4
            }})
        };
        let rows: [[f64; 4]; 4] = [
            [1., 0., 0., 0.],
            [1., 1., 0., 0.],
            [1., 2., 1., 0.],
            [1., 9., 2., 1.],
        ];
        let config = ferrum_types::EngineConfig::default();
        let request = json!({"id":"ephemeral","created_at":"ephemeral","prompt":"wire"});
        let ts =
            vec![json!({"output":"fixture","request_bytes":serde_json::to_vec(&request).unwrap()})];
        let current = Verified {
            captured_geometry_kernel: CapturedGeometryKernel::FirstPassPrefixV1,
            cases: serde_json::from_value(table.clone()).unwrap(),
            calls: vec![super::super::Call {
                population: Population {
                    population_index: 0,
                    key: family(8),
                },
                matrix: Matrix {
                    ordinal: 0,
                    cases: vec![0, 1, 2, 3],
                    axis_bits: rows
                        .iter()
                        .map(|r| r.iter().map(|x| x.to_bits()).collect())
                        .collect(),
                    mandatory_anchors: vec![0, 3],
                    settings: settings.clone(),
                    visits_before: 0,
                    maximum_visits: 1_000_000,
                    exhausted_before: false,
                    maximum_scratch_bytes: 1 << 20,
                },
                result: OriginalResult {
                    ordinal: 0,
                    audit: GeometryAudit {
                        candidate_rank: Some(4),
                        selected_rank: Some(4),
                        added_original_cases: 2,
                        complete: true,
                        visits: 0,
                    },
                    gap: None,
                    final_selected_cases: vec![0, 1, 2, 3],
                    visits_after: 0,
                    exhausted_after: false,
                },
            }],
            limit: 1_000_000,
            visits: 123,
            exhausted: false,
            retirement: Retirement {
                complete: true,
                deadline_expired: false,
                remaining_ns: 1,
                actual_requests: 0,
                actual_actions: 0,
                selected_requests: 1,
                selected_actions: 1,
            },
            selection: None,
            startup_config: Box::new(config.clone()),
            startup_templates: ts.clone(),
        };
        let old = History {
            cases: serde_json::from_value(table.clone()).unwrap(),
            config: Box::new(config),
            templates: ts,
            calls: vec![
                HistoricalCall {
                    population: Population {
                        population_index: 0,
                        key: family(1),
                    },
                    cases: vec![0],
                    anchors: vec![0],
                    settings: settings.clone(),
                },
                HistoricalCall {
                    population: Population {
                        population_index: 1,
                        key: family(4),
                    },
                    cases: vec![1, 2],
                    anchors: vec![0],
                    settings,
                },
            ],
        };
        let header = json!({"declaration":{"cohort_manifest_payload":{"parent":{
            "parent":{"cases":table["cases"],"templates":[{"output":"fixture","request":request}]},
            "child":{"checked_selection":{"populations":[
                {"key":family(1),"scheduled":true,"representative_case_indices":[0]},
                {"key":family(4),"scheduled":false,"representative_case_indices":[1,2]}
            ]}}
        }}}});
        let raw: Vec<_> = old
            .calls
            .iter()
            .enumerate()
            .map(|(i, c)| {
                json!({
                    "g18_ordinal":i,"g18_population_index":i,"g20_population_index":i,
                    "original_raw_key":c.population.key,"candidate_case_indices":c.cases,
                    "mandatory_row_indices":c.anchors,"mandatory_case_indices":[c.cases[0]]
                })
            })
            .collect();
        let audit = json!({"populations":[{"g21_population_index":0,"g21_ordinal":0,
            "g21_family":family(8)["NumericalFamily"],"g21_subset_candidate_case_indices":[0,1,2],
            "g18_raw_population_mapping":raw,"mandatory_case_indices":[0,1]}]});
        (current, old, header, audit)
    }

    #[test]
    fn narrow_extension_binding_checks_real_case_key_and_anchor_sources() {
        let (current, mut old, mut header, mut audit) = fixture();
        let mapped = mappings(&current, &old, &header, &audit).unwrap();
        let m = mapped[0].as_ref().unwrap();
        assert_eq!(m.subset, [0, 1, 2]);
        assert_eq!(m.narrow, [0]);
        assert_eq!(m.historical, [0, 1]);
        old.cases.cases[0].width = 4;
        assert!(mappings(&current, &old, &header, &audit).is_err());
        old.cases.cases[0].width = 1;
        header["declaration"]["cohort_manifest_payload"]["parent"]["child"]["checked_selection"]
            ["populations"][0]["key"]["NumericalFamily"]["algorithm_domain"] =
            json!(vec![99u8; 32]);
        assert!(mappings(&current, &old, &header, &audit).is_err());
        let (_, _, header, _) = fixture();
        audit["populations"][0]["g18_raw_population_mapping"][1]["mandatory_case_indices"] =
            json!([2]);
        assert!(mappings(&current, &old, &header, &audit).is_err());
    }

    #[test]
    fn narrow_extension_recomputes_subset_endpoints_span_and_charges_one_budget() {
        let (current, old, header, audit) = fixture();
        let mapped = mappings(&current, &old, &header, &audit).unwrap();
        let mut work = GeometryWork {
            used: 0,
            limit: u64::MAX,
            exhausted: false,
        };
        let result = measure_call(
            &current,
            &current.calls[0],
            mapped[0].as_ref(),
            &mut work,
            true,
        )
        .unwrap();
        assert!(result.measured.complete && result.measured.span_verified);
        assert_eq!(result.measured.rank, Some(3));
        assert_eq!(result.measured.final_selected_cases, [0, 1, 2]);
        assert!(result.numeric_endpoint_case_indices.contains(&2));
        assert!(!result.measured.final_selected_cases.contains(&3)); // Old full anchor is outside the declared subset.
        assert_eq!(current.visits, 123); // Original captured ledger is untouched.
        let charge = work.used;
        let mut exact = GeometryWork {
            used: 0,
            limit: charge,
            exhausted: false,
        };
        assert!(
            measure_call(
                &current,
                &current.calls[0],
                mapped[0].as_ref(),
                &mut exact,
                false
            )
            .unwrap()
            .measured
            .complete
        );
        let mut short = GeometryWork {
            used: 0,
            limit: charge - 1,
            exhausted: false,
        };
        let failed = measure_call(
            &current,
            &current.calls[0],
            mapped[0].as_ref(),
            &mut short,
            false,
        )
        .unwrap();
        assert!(!failed.measured.complete && short.exhausted);
        let spent = short.used;
        let later = measure_call(
            &current,
            &current.calls[0],
            mapped[0].as_ref(),
            &mut short,
            false,
        )
        .unwrap();
        assert!(!later.measured.complete && later.measured.visits == 0 && short.used == spent);
        assert!(serde_json::from_value::<Mode>(json!("symmetric_widths_one_four_v1")).is_ok());
        assert!(serde_json::from_value::<Mode>(json!("all_widths")).is_err());
    }
}

#[derive(Serialize)]
struct Call {
    #[serde(flatten)]
    measured: cold_candidate::Call,
    population_key: Value,
    widths: [usize; 2],
    subset_applied: bool,
    source_original_case_indices: Vec<usize>,
    historical_mandatory_case_indices: Vec<usize>,
    numeric_endpoint_case_indices: Vec<usize>,
    original_narrow_case_indices: Vec<usize>,
    old_narrow_representative_case_indices: Vec<usize>,
    historical_raw_population_indices: Vec<usize>,
}
#[derive(Serialize)]
struct Runs {
    maximum_visits: u64,
    used_visits: u128,
    complete: bool,
    span_verified: bool,
    exhausted: bool,
    calls: Vec<Call>,
}
#[derive(Serialize)]
pub(super) struct Measurement {
    geometry_kernel: &'static str,
    mode: Mode,
    widths: [usize; 2],
    source_bindings: Value,
    all_case_and_key_bindings_verified: bool,
    current_branch_flags_recertified: bool,
    independent_max: Runs,
    shared_original_budget: Runs,
    conclusion: &'static str,
}
fn measure_call(
    verified: &Verified,
    call: &super::Call,
    mapping: Option<&Mapping>,
    work: &mut GeometryWork,
    certificate: bool,
) -> AuditResult<Call> {
    let before = work.used;
    let mut endpoints = Vec::new();
    let mut measured = match mapping {
        None => cold_candidate::call(verified, call, work, certificate)?,
        Some(mapping) => match prepare(verified, call, mapping, work) {
            Ok(prepared) => {
                let result = cold_candidate::call_mapped(
                    verified,
                    call,
                    Some(&prepared.rows),
                    &prepared.anchors,
                    prepared.bytes,
                    work,
                    certificate,
                )?;
                endpoints = prepared.endpoints;
                result
            }
            Err(reason) => {
                cold_candidate::Call::not_run(call, &mapping.subset, before, work, reason)
            }
        },
    };
    measured.visits_before = before;
    measured.visits = work.used - before;
    Ok(Call {
        measured,
        population_key: call.population.key.clone(),
        widths: WIDTHS,
        subset_applied: mapping.is_some(),
        source_original_case_indices: call.matrix.cases.clone(),
        historical_mandatory_case_indices: mapping.map_or_else(Vec::new, |m| m.historical.clone()),
        numeric_endpoint_case_indices: endpoints,
        original_narrow_case_indices: mapping.map_or_else(Vec::new, |m| m.narrow.clone()),
        old_narrow_representative_case_indices: mapping
            .map_or_else(Vec::new, |m| m.old_representatives.clone()),
        historical_raw_population_indices: mapping
            .map_or_else(Vec::new, |m| m.raw_populations.clone()),
    })
}
fn summarize(calls: Vec<Call>, maximum_visits: u64) -> Runs {
    Runs {
        maximum_visits,
        used_visits: calls.iter().map(|c| u128::from(c.measured.visits)).sum(),
        complete: calls.iter().all(|c| c.measured.complete),
        span_verified: calls.iter().all(|c| c.measured.span_verified),
        exhausted: calls.iter().any(|c| c.measured.exhausted),
        calls,
    }
}
fn binding(file: &BoundFile) -> Value {
    json!({"path":file.path,"sha256":file.sha256})
}
pub(super) fn evaluate(
    verified: &Verified,
    current: &BoundFile,
    output: &std::path::Path,
    options: &Options,
) -> AuditResult<Measurement> {
    for input in [
        &options.g18_capture,
        &options.g20_header,
        &options.anchor_audit,
    ] {
        ensure!(input.path != output, "subset output overlaps binding input");
    }
    let old = history(&bound_file(&options.g18_capture, CAPTURE_LIMIT)?)?;
    let header: Value = serde_json::from_slice(&bound_file(&options.g20_header, BINDING_LIMIT)?)?;
    let audit: Value = serde_json::from_slice(&bound_file(&options.anchor_audit, BINDING_LIMIT)?)?;
    ensure!(
        audit.get("schema").and_then(Value::as_str)
            == Some("ferrum.g22.narrow-extension-anchor-audit.v1"),
        "anchor audit schema differs"
    );
    for (name, file) in [
        ("g18_capture", &options.g18_capture),
        ("g20_source8_header", &options.g20_header),
        ("g21_capture", current),
    ] {
        ensure!(
            audit
                .pointer(&format!("/inputs/{name}/sha256"))
                .and_then(Value::as_str)
                == Some(file.sha256.as_str()),
            "anchor audit source SHA differs"
        );
    }
    let mappings = mappings(verified, &old, &header, &audit)?;
    let mut independent = Vec::new();
    for (call, mapping) in verified.calls.iter().zip(&mappings) {
        let mut work = GeometryWork {
            used: 0,
            limit: u64::MAX,
            exhausted: false,
        };
        independent.push(measure_call(
            verified,
            call,
            mapping.as_ref(),
            &mut work,
            true,
        )?);
    }
    let mut work = GeometryWork {
        used: 0,
        limit: verified.limit,
        exhausted: false,
    };
    let mut shared = Vec::new();
    for ((call, mapping), expected) in verified.calls.iter().zip(&mappings).zip(&independent) {
        let mut actual = measure_call(verified, call, mapping.as_ref(), &mut work, false)?;
        if actual.measured.complete {
            ensure!(
                expected.measured.complete
                    && actual.measured.rank == expected.measured.rank
                    && actual.measured.anchor_rank == expected.measured.anchor_rank
                    && actual.measured.pivot_indices == expected.measured.pivot_indices
                    && actual.measured.final_selected_cases
                        == expected.measured.final_selected_cases
                    && actual.numeric_endpoint_case_indices
                        == expected.numeric_endpoint_case_indices,
                "subset shared/MAX result differs"
            );
            actual.measured.span_verified = expected.measured.span_verified;
        }
        shared.push(actual);
    }
    Ok(Measurement { geometry_kernel:"anchored_readiness_v2_width_cost_subset_v1", mode:match options.mode { Mode::SymmetricWidthsOneFourV1=>Mode::SymmetricWidthsOneFourV1 },
        widths:WIDTHS, source_bindings:json!({"g18_capture":binding(&options.g18_capture),"g20_header":binding(&options.g20_header),"g21_capture":binding(current),"anchor_audit":binding(&options.anchor_audit)}),
        all_case_and_key_bindings_verified:true,current_branch_flags_recertified:false,
        independent_max:summarize(independent,u64::MAX),shared_original_budget:summarize(shared,verified.limit),
        conclusion:"Explicit offline width1/4 experiment, with unchanged Prefill controls. Historical mandatory cases witness original branches; current per-row branch flags are unavailable and are not recertified. Current subset zero/positive extrema, normalization, geometry and independent legacy subset span are recomputed. Candidate filtering/index work and endpoint scans share the original geometry budget and scratch; source-file parsing, binding verification, input decoding and report storage are separate offline diagnostics. Old width1 pivots remain input evidence, not compulsory output indices. No full-matrix span, qualification, source schedule, rows8 support, wall-time or SLO claim." })
}
