use super::*;

#[test]
#[ignore = "requires immutable original source7 and required query artifacts via FERRUM_ARCHIVED_BOUND_SPEC"]
fn replay_original_source7_identified_fit_with_global_point_residual_control() {
    let spec: AuditSpec = serde_json::from_slice(
        &std::fs::read(std::env::var_os("FERRUM_ARCHIVED_BOUND_SPEC").unwrap()).unwrap(),
    )
    .unwrap();
    let original = seeded::verify_source_file(&spec.actual_source, spec.actual_source_sha256);
    let declaration = &original.declaration;
    assert_eq!(
        declaration
            .nonnegative_envelope
            .as_ref()
            .unwrap()
            .planning_estimator,
        NonNegativePlanningEstimatorV1::IdentifiedEnvelopeV2
    );
    let limits = CostProfileLoadLimits::default();
    let mut collector = StructuredServiceCollectorV7::new_streaming(
        original.clone(),
        limits.clone(),
        NonZeroU64::new(original.maximum_file_bytes).unwrap(),
    )
    .unwrap();
    let mut samples = BTreeMap::<u64, [Vec<StructuredNumericObservationV2>; 3]>::new();
    let mut boundaries = BTreeMap::<u64, Vec<Value>>::new();
    let mut qualification_time = BTreeMap::<u64, u64>::new();
    let mut original_shapes = BTreeMap::<u64, Value>::new();
    for line in BufReader::new(std::fs::File::open(&spec.actual_source).unwrap())
        .lines()
        .skip(1)
    {
        let record: StructuredServiceRecordV7 = serde_json::from_str(&line.unwrap()).unwrap();
        if let StructuredServiceRecordV7::BlockClose {
            block,
            closing,
            freezes,
            ..
        } = &record
        {
            for freeze in freezes {
                let owner = collector
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
                assert!(population[phase].is_empty());
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
        if let StructuredServiceRecordV7::Completed { wave } = &record {
            original_shapes.insert(
                wave.host_stages.call_id,
                shape_ranges(
                    &serde_json::to_value(wave.host_stages.actual_shape.as_ref().unwrap()).unwrap(),
                ),
            );
        }
        collector.push(&record).unwrap();
    }
    let candidates: Vec<_> = collector
        .owners
        .iter()
        .filter_map(|o| match &o.state {
            OwnerState::Qualified(m) => Some((o.contract.owner_attempt_id, m.clone())),
            _ => None,
        })
        .collect();
    eprintln!(
        "ORIGINAL_SOURCE7_GLOBAL_CONTROL_CHILDREN {}",
        json!({
        "original_qualified_children":candidates.len(),"source_sha256":spec.actual_source_sha256})
    );
    for (attempt, model) in candidates {
        let parameters_before = model.parameters_signature();
        let uncertainty = model.uncertainty();
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
            &model,
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
        let runtime = original.clone();
        let mut contract = runtime.declaration.nonnegative_envelope.clone().unwrap();
        if runtime.declaration.schedule.algorithm_universe
            == Some(OwnerAlgorithmUniversePolicyV1::SeededFirstOrdinaryDiscoveryBlockSubsetV1)
        {
            contract.algorithm_universe = None;
        }
        let mut archive = StructuredServiceCollectorV7::new_streaming(
            runtime.clone(),
            limits.clone(),
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
                    if wave.issued_at_ns < q_time {
                        continue;
                    }
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
                    let original_result =
                        model.predict_query(&fingerprint, &query, wave.issued_at_ns);
                    range_audit.observe_future(
                        &model,
                        &query,
                        call,
                        original_result.is_ok(),
                        demands.contains_key(&call),
                    );
                    match original_result {
                        Ok(original_prediction) => {
                            counts.original_coverage_known += 1;
                            let (point, _, _) =
                                model.diagnose_empirical_cell_input(&query).unwrap();
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
                                "phase_coverage":model.diagnose_original_query_phase_coverage(&query),
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
            "ORIGINAL_SOURCE7_IDENTIFIED_GLOBAL_POINT_RESIDUAL {}",
            json!({
                "source7_sha256":spec.actual_source_sha256,
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
                "scope":"original source7 identified Fit/certificate and unchanged complete F/R/Q; point+original global residual diagnostic only, no publication",
                "future_scope":"all original runtime prepared prospective queries; only listed paired demand was matched to the actual required-query journal",
                "interpretation":"empirical diagnostic, not a mathematical upper bound or proof for unseen directions; failed Q makes candidate unavailable even when future numbers are shown",
            "range_disclosure":"original ChallengeCoverage checks axis/branch presence; original_coverage_known does not prove numerical range support for longer contexts/history. All future numbers remain diagnostics without publication authority."
            })
        );
    }
}
