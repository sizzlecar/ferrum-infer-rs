mod source7;
use super::*;
use crate::implementations::continuous::cost_profile::structured_v10::service::owner_blocks::collector::State as OwnerState;
use std::num::NonZeroU64;

#[test]
#[ignore = "requires immutable original source8/source7/query artifacts via FERRUM_ARCHIVED_BOUND_SPEC"]
fn replay_original_identified_fit_with_global_point_residual_control() {
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
    assert_eq!(bytes.len(), spec.checkpoint_bytes);
    assert_eq!(
        <[u8; 32]>::from(Sha256::digest(&bytes)),
        spec.checkpoint_sha256
    );
    let original = header(&bytes);
    let declaration = &original.declaration.population;
    assert_eq!(
        declaration
            .nonnegative_envelope
            .as_ref()
            .unwrap()
            .planning_estimator,
        NonNegativePlanningEstimatorV1::IdentifiedEnvelopeV2
    );
    let checkpoint = replay_structured_source_v8(&bytes, &limits).unwrap();
    let closing = checkpoint.population.closing;
    let catalog = checkpoint
        .activate_same_process_memory(closing, &limits)
        .unwrap();
    let model = catalog
        .children
        .iter()
        .find(|m| m.parameters_signature() == spec.parameters_sha256)
        .expect("exact originally installed model");
    let model = &model.model;
    let parameters_before = model.parameters_signature();
    let uncertainty = model.uncertainty();

    // push(original) independently verifies original Fit certificates and cuts.
    // Every original sample is admitted by the original collector; this test
    // neither reconstructs a different sampler nor changes estimator at Fit.
    let mut collector = StructuredPreparedOwnerBlockCollectorV8::new_streaming(
        original.clone(),
        limits.clone(),
        NonZeroU64::new(original.maximum_file_bytes).unwrap(),
    )
    .unwrap();
    let mut samples = BTreeMap::<u64, [Vec<StructuredNumericObservationV2>; 3]>::new();
    let mut boundaries = BTreeMap::<u64, Vec<Value>>::new();
    let mut qualification_time = BTreeMap::<u64, u64>::new();
    let mut original_shapes = BTreeMap::<u64, Value>::new();
    use StructuredPreparedOwnerBlockRecordV8 as R;
    for line in bytes.split_inclusive(|b| *b == b'\n').skip(1) {
        let record: R = serde_json::from_slice(line).unwrap();
        if let R::Population(StructuredServiceRecordV7::BlockClose {
            block,
            closing,
            freezes,
            ..
        }) = &record
        {
            for freeze in freezes {
                let owner = collector
                    .population
                    .owners
                    .iter()
                    .find(|o| o.contract.owner_attempt_id == freeze.owner_attempt_id)
                    .unwrap();
                let phase = match freeze.close.phase {
                    StructuredPhaseV2::Fit => 0,
                    StructuredPhaseV2::Residual => 1,
                    StructuredPhaseV2::Qualification => 2,
                };
                let population = samples
                    .entry(freeze.owner_attempt_id)
                    .or_insert_with(|| std::array::from_fn(|_| Vec::new()));
                assert!(population[phase].is_empty(), "original single phase only");
                assert_eq!(owner.samples.len(), freeze.close.member_count);
                population[phase] = owner.samples.clone();
                boundaries
                    .entry(freeze.owner_attempt_id)
                    .or_default()
                    .push(json!({
                        "phase":phase,"block":block,"at_ns":closing.monotonic_ns,
                        "members":freeze.close.member_count,"failure":freeze.failure,
                        "fit_certificate_sha256":freeze.nonnegative_fit_certificate.as_ref()
                            .map(|v|<[u8;32]>::from(Sha256::digest(serde_json::to_vec(v).unwrap())))
                    }));
                if phase == 2 {
                    qualification_time.insert(freeze.owner_attempt_id, closing.monotonic_ns);
                }
            }
        }
        if let R::Population(StructuredServiceRecordV7::Completed { wave }) = &record {
            original_shapes.insert(
                wave.host_stages.call_id,
                shape_ranges(
                    &serde_json::to_value(wave.host_stages.actual_shape.as_ref().unwrap()).unwrap(),
                ),
            );
        }
        collector.push(&record).unwrap();
    }
    let original_owner = collector
        .population
        .owners
        .iter()
        .find(|o| {
            matches!(&o.state,
        OwnerState::Qualified(m) if m.parameters_signature()==spec.parameters_sha256)
        })
        .unwrap();
    let attempt = original_owner.contract.owner_attempt_id;
    let [fit, residual, qualification] = samples.remove(&attempt).unwrap();
    assert!(!fit.is_empty() && !residual.is_empty() && !qualification.is_empty());
    let q_time = qualification_time[&attempt];
    let population_rows = |values: &[StructuredNumericObservationV2]| {
        let mut count = BTreeMap::<u32, usize>::new();
        for value in values {
            *count.entry(value.input.owner().rows).or_default() += 1;
        }
        count
    };
    let fingerprint = original.fingerprint.clone().into();
    let GlobalControl {
        margin,
        qualified,
        q_errors,
        q_rows,
        q_misses,
    } = evaluate_global_control(
        model,
        &residual,
        &qualification,
        &declaration.settings,
        &fingerprint,
        q_time,
    );
    let mut range_audit = model
        .diagnose_freeze_input_ranges(
            [&fit, &residual, &qualification],
            qualified,
            declaration.maximum_retained_numeric_bytes,
        )
        .unwrap();
    let runtime = seeded::verify_source_file(&spec.actual_source, spec.actual_source_sha256);
    assert_eq!(runtime.fingerprint, original.fingerprint);
    let mut contract = runtime.declaration.nonnegative_envelope.clone().unwrap();
    if runtime.declaration.schedule.algorithm_universe
        == Some(OwnerAlgorithmUniversePolicyV1::SeededFirstOrdinaryDiscoveryBlockSubsetV1)
    {
        contract.algorithm_universe = None;
    }
    let mut archive = StructuredServiceCollectorV7::new_streaming(
        runtime.clone(),
        limits,
        NonZeroU64::new(runtime.maximum_file_bytes).unwrap(),
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
    let mut rows = BTreeMap::<u32, FutureRows>::new();
    let mut paired = Vec::new();
    let mut all_errors = Errors::default();
    let mut opened = runtime.opening.monotonic_ns;
    for line in BufReader::new(std::fs::File::open(&spec.actual_source).unwrap())
        .lines()
        .skip(1)
    {
        let record: StructuredServiceRecordV7 = serde_json::from_str(&line.unwrap()).unwrap();
        archive.push(&record).unwrap();
        match record {
            StructuredServiceRecordV7::BlockOpen { opened_at_ns, .. } => {
                opened = opened_at_ns;
            }
            StructuredServiceRecordV7::Completed { wave } => {
                let (input, wall, _) = physical::validate_parts(
                    &runtime.fingerprint,
                    runtime.opening.monotonic_ns,
                    Some(&contract),
                    opened,
                    wave.ticket,
                    wave.fifo,
                    wave.issued_at_ns,
                    &wave.host_stages,
                    wave.independent.as_ref(),
                    &mut physical::Frontiers::default(),
                )
                .unwrap();
                let (prepared, offered) =
                    physical::original_prepared(&wave.host_stages, wave.independent.as_ref())
                        .unwrap();
                let query = StructuredQueryV2::exact(
                    input_replay::project_service_actual_with_domain(
                        &prepared,
                        &offered,
                        &contract.workload_domain,
                    )
                    .unwrap(),
                );
                let call = wave.host_stages.call_id;
                if let Some(demand) = demands.get(&call) {
                    assert_eq!(
                        serde_json::to_value(query.required_coverage().unwrap()).unwrap(),
                        *demand
                    );
                }
                let row = input.owner().rows;
                let counts = rows.entry(row).or_default();
                counts.total += 1;
                let original_result = model.predict_query(&fingerprint, &query, wave.issued_at_ns);
                range_audit.observe_future(
                    model,
                    &query,
                    call,
                    original_result.is_ok(),
                    demands.contains_key(&call),
                );
                match original_result {
                    Ok(original_prediction) => {
                        counts.original_coverage_known += 1;
                        let (point, _, _) = model.diagnose_empirical_cell_input(&query).unwrap();
                        let planning = point.checked_add(margin).unwrap();
                        counts.errors.observe(point, planning, wall);
                        all_errors.observe(point, planning, wall);
                        counts
                            .old_errors
                            .observe(point, original_prediction.planning_ns, wall);
                        if demands.contains_key(&call) {
                            paired.push(json!({"call":call,"rows":row,"demand_verified":true,
                                "original_coverage_known":true,"candidate_known":qualified,
                                "original_planning_ns":original_prediction.planning_ns,
                                "observed_input":shape_ranges(&serde_json::to_value(wave.host_stages.actual_shape.as_ref().unwrap()).unwrap()),
                                "point_ns":point,"margin_ns":margin,"planning_ns":planning,"wall_ns":wall,
                                "underestimate":planning<wall}));
                        }
                    }
                    Err(reason) => {
                        *counts.unknown.entry(format!("{reason:?}")).or_default() += 1;
                        if demands.contains_key(&call) {
                            paired.push(json!({"call":call,"rows":row,"demand_verified":true,
                                "original_coverage_known":false,"candidate_known":false,
                                "observed_input":shape_ranges(&serde_json::to_value(wave.host_stages.actual_shape.as_ref().unwrap()).unwrap()),
                                "reason":format!("{reason:?}")}));
                        }
                    }
                }
            }
            _ => {}
        }
    }
    assert_eq!(paired.len(), demands.len());
    assert_eq!(model.parameters_signature(), parameters_before);
    let all_rows: Vec<_> = rows
        .iter()
        .map(|(rows, c)| {
            json!({
                "rows":rows,"physical":c.total,"original_coverage_known":c.original_coverage_known,
                "candidate_known":if qualified{c.original_coverage_known}else{0},
                "unknown_reasons":c.unknown,"point_plus_global_margin":c.errors.summary(),
                "original_identified_plan":c.old_errors.summary()
            })
        })
        .collect();
    eprintln!(
        "ORIGINAL_IDENTIFIED_GLOBAL_POINT_RESIDUAL {}",
        json!({
            "source8_checkpoint_sha256":spec.checkpoint_sha256,
            "original_parameters_sha256":parameters_before,"owner_attempt":attempt,
            "original_phase_boundaries":boundaries.remove(&attempt).unwrap(),
            "phase_rows":{"fit":population_rows(&fit),"residual":population_rows(&residual),
                "qualification":population_rows(&qualification)},
            "original_phase_actual_maxima":{
            "fit":phase_ranges(&fit,&original_shapes),
            "residual":phase_ranges(&residual,&original_shapes),
            "qualification":phase_ranges(&qualification,&original_shapes)
        },
        "original_uncertainty":uncertainty,"recomputed_original_r_equal":true,
            "margin_ns":margin,"counterfactual_qualified":qualified,
            "qualification":q_errors.summary(),
            "qualification_by_rows":q_rows.iter().map(|(r,v)|json!({"rows":r,"errors":v.summary()})).collect::<Vec<_>>(),
            "all_qualification_misses":q_misses,"all_runtime_rows":all_rows,
            "numerical_range_diagnostic":range_audit.summary(),
            "all_future_numeric_errors":all_errors.summary(),"paired_actual_demands":paired,
            "scope":"original identified Fit/certificate and unchanged complete F/R/Q; point+original global residual diagnostic only, no publication",
            "future_scope":"all original runtime prepared prospective queries; only listed paired demand was matched to the actual required-query journal",
            "interpretation":"empirical diagnostic, not a mathematical upper bound or proof for unseen directions; failed Q makes candidate unavailable even when future numbers are shown",
        "range_disclosure":"original ChallengeCoverage checks axis/branch presence; original_coverage_known does not prove numerical range support for longer contexts/history. All future numbers remain diagnostics without publication authority."
        })
    );
}

#[derive(Default)]
struct FutureRows {
    total: usize,
    original_coverage_known: usize,
    unknown: BTreeMap<String, usize>,
    errors: Errors,
    old_errors: Errors,
}
#[derive(Default)]
struct Errors {
    point_abs: Vec<u64>,
    planning_abs: Vec<u64>,
    under: Vec<u64>,
    over: Vec<u64>,
    planning: Vec<u64>,
    misses: usize,
}
impl Errors {
    fn observe(&mut self, point: u64, planning: u64, wall: u64) {
        self.point_abs.push(point.abs_diff(wall));
        self.planning_abs.push(planning.abs_diff(wall));
        self.under.push(wall.saturating_sub(planning));
        self.over.push(planning.saturating_sub(wall));
        self.planning.push(planning);
        self.misses += usize::from(planning < wall);
    }
    fn summary(&self) -> Value {
        fn quantiles(values: &[u64]) -> Option<[u64; 3]> {
            if values.is_empty() {
                return None;
            }
            let mut sorted = values.to_vec();
            sorted.sort_unstable();
            Some([
                sorted[(50 * sorted.len()).div_ceil(100) - 1],
                sorted[(99 * sorted.len()).div_ceil(100) - 1],
                *sorted.last().unwrap(),
            ])
        }
        json!({"members":self.point_abs.len(),"all_misses":self.misses,
            "point_abs_error_p50_p99_max_ns":quantiles(&self.point_abs),
            "planning_abs_error_p50_p99_max_ns":quantiles(&self.planning_abs),
            "under_p50_p99_max_ns":quantiles(&self.under),"over_p50_p99_max_ns":quantiles(&self.over),
            "planning_p50_p99_max_ns":quantiles(&self.planning)})
    }
}

fn shape_ranges(shape: &Value) -> Value {
    let rows = shape["numeric_features"]["rows"].as_array().unwrap();
    json!({
        "rows":rows.len(),
        "maximum_actual_decode_kv_tokens":shape["exact"]["decode_kv_tokens"].as_array().unwrap()
            .iter().map(|v|v.as_u64().unwrap()).max(),
        "maximum_actual_sampling_history_tokens":rows.iter()
            .map(|r|r["sampling_history_tokens"].as_u64().unwrap()).max(),
        "maximum_actual_generated_tokens_before":rows.iter()
            .map(|r|r["generated_tokens_before"].as_u64().unwrap()).max()
    })
}
fn phase_ranges(
    samples: &[StructuredNumericObservationV2],
    shapes: &BTreeMap<u64, Value>,
) -> Value {
    let mut result = serde_json::Map::new();
    for field in [
        "rows",
        "maximum_actual_decode_kv_tokens",
        "maximum_actual_sampling_history_tokens",
        "maximum_actual_generated_tokens_before",
    ] {
        result.insert(
            field.into(),
            json!(samples
                .iter()
                .filter_map(|s| shapes[&s.call_id][field].as_u64())
                .max()),
        );
    }
    Value::Object(result)
}

struct GlobalControl {
    margin: u64,
    qualified: bool,
    q_errors: Errors,
    q_rows: BTreeMap<u32, Errors>,
    q_misses: Vec<Value>,
}
fn evaluate_global_control(
    model: &QualifiedStructuredModelV2,
    residual: &[StructuredNumericObservationV2],
    qualification: &[StructuredNumericObservationV2],
    settings: &StructuredSettingsV2,
    fingerprint: &crate::implementations::continuous::cost_model::ExecutionFingerprint,
    q_time: u64,
) -> GlobalControl {
    let uncertainty = model.uncertainty();
    let mut residual_signed = Vec::with_capacity(residual.len());
    let mut residual_positive = Vec::with_capacity(residual.len());
    for sample in residual {
        let point = model
            .diagnose_original_actual_point(&sample.input)
            .expect("every original eligible R sample uses unchanged identified fitted point");
        residual_signed.push(i128::from(sample.wall_ns) - i128::from(point));
        residual_positive.push(sample.wall_ns.saturating_sub(point));
    }
    residual_positive.sort_unstable();
    let residual_ns = residual_positive[(99 * residual_positive.len()).div_ceil(100) - 1];
    let minimum = *residual_signed.iter().min().unwrap();
    let maximum = *residual_signed.iter().max().unwrap();
    let span = match settings.learned_drift {
        StructuredLearnedDriftV2::Disabled => 0,
        StructuredLearnedDriftV2::ObservedResidualSpanV1 {
            maximum_span_margin_ns,
        } => {
            let span = u64::try_from(maximum - minimum).unwrap();
            assert!(span <= maximum_span_margin_ns.get());
            span
        }
    };
    assert_eq!(residual_ns, uncertainty.residual_ns);
    assert_eq!(span, uncertainty.learned_span_margin_ns);
    let effective = residual_ns.max(uncertainty.fit_error_floor_ns);
    assert_eq!(effective, uncertainty.effective_residual_ns);
    let margin = effective
        .checked_add(span)
        .unwrap()
        .checked_add(uncertainty.static_margin_ns)
        .unwrap();

    let mut q_errors = Errors::default();
    let mut q_rows = BTreeMap::<u32, Errors>::new();
    let mut q_misses = Vec::new();
    for sample in qualification {
        let query = StructuredQueryV2::exact(sample.input.clone());
        // This is the unchanged original qualification/coverage gate. All
        // original Q members must remain eligible; no failures can be filtered.
        model.predict_query(&fingerprint, &query, q_time).unwrap();
        let (point, _, _) = model
            .diagnose_empirical_cell_input(&query)
            .expect("every original Q member keeps its original query input");
        let planning = point.checked_add(margin).unwrap();
        q_errors.observe(point, planning, sample.wall_ns);
        q_rows
            .entry(sample.input.owner().rows)
            .or_default()
            .observe(point, planning, sample.wall_ns);
        if planning < sample.wall_ns {
            q_misses.push(
                json!({"call":sample.call_id,"rows":sample.input.owner().rows,
                "wall_ns":sample.wall_ns,"point_ns":point,"planning_ns":planning,
                "under_ns":sample.wall_ns-planning}),
            );
        }
    }
    let qualified = q_errors.misses == 0;
    GlobalControl {
        margin,
        qualified,
        q_errors,
        q_rows,
        q_misses,
    }
}
