//! CPU wall diagnostic of the complete production projection on a real Metal
//! tiny-Qwen fixture. Run identical test bytes in two immutable source trees.
//! No production switch, copied old algorithm, device timing, or SLO claim.
use super::*;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::hint::black_box;
use std::time::Duration;

const WARMUP: usize = 8;
const MEASURED: usize = 64;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Correctness,
    ReleaseScreen,
}
impl Mode {
    fn warmup(self) -> usize {
        if self == Self::Correctness {
            0
        } else {
            WARMUP
        }
    }
    fn measured(self) -> usize {
        if self == Self::Correctness {
            1
        } else {
            MEASURED
        }
    }
    fn allowance(self) -> Duration {
        if self == Self::Correctness {
            Duration::from_secs(1)
        } else {
            ferrum_types::SloPlannerConfig::default().planning_budget()
        }
    }
    fn label(self) -> &'static str {
        if self == Self::Correctness {
            "correctness_only_1s_no_performance_claim"
        } else {
            "release_screen_original_2ms"
        }
    }
}

#[derive(Debug, Serialize)]
struct Attempt {
    ordinal: usize,
    warmup: bool,
    capture_ns: u64,
    project_ns: Option<u64>,
    total_ns: u64,
    capture_polls: u64,
    project_polls: u64,
    status: String,
    deadline_live_at_end: bool,
    semantic_sha256: Option<String>,
    evidence_diagnostic: serde_json::Value,
}

fn elapsed_ns(start: Instant, end: Instant) -> u64 {
    u64::try_from(end.checked_duration_since(start).unwrap().as_nanos()).unwrap()
}

fn semantic(projection: &ExecutionCostRouteProjection) -> std::result::Result<String, String> {
    if projection.state.projected_waves() != 1 {
        return Err("projected_wave_count".into());
    }
    let statistics = projection
        .statistical_evidence
        .as_ref()
        .ok_or("missing_statistics")?;
    statistics
        .validate_exact(&projection.shape)
        .map_err(|e| format!("statistics:{e:?}"))?;
    if projection.shape.kind == ActualWaveKind::Decode {
        statistics
            .independent_attention_v2()
            .ok_or("missing_decode_independent_attention_v2")?
            .validate_exact(&projection.shape)
            .map_err(|e| format!("independent_attention:{e:?}"))?;
    }
    let structured = statistics
        .structured_capture()
        .ok_or("missing_structured_sidecar")?
        .map_err(|e| format!("structured:{e:?}"))?;
    structured
        .validate_exact(&projection.shape)
        .map_err(|e| format!("structured_binding:{e:?}"))?;
    let algorithms = structured
        .algorithm_work()
        .map_err(|e| format!("algorithm_work:{e:?}"))?;
    let bytes = serde_json::to_vec(&serde_json::json!({
        "shape": format!("{:?}", projection.shape),
        "statistics": statistics,
        "independent_attention_v2": statistics.independent_attention_v2(),
        "structured_device": structured.device(),
        "structured_algorithms": algorithms,
        "structured_complete": structured.as_ref(),
        "mask_uploads": projection.state.last_token_mask_uploads(),
        "graph": format!("{:?}", projection.state.projected_graph_state()),
        "projected_waves": projection.state.projected_waves(),
    }))
    .map_err(|e| format!("semantic_serialization:{e}"))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
fn evidence_diagnostic(projection: &ExecutionCostRouteProjection) -> serde_json::Value {
    let stats = projection.statistical_evidence.as_ref();
    serde_json::json!({"statistics_present":stats.is_some(),
        "structured_status":stats.map(|s|format!("{:?}",s.structured_capture().map(|r|r.map(|_|"present")))),
        "independent_attention_v2_present":stats.is_some_and(|s|s.independent_attention_v2().is_some()),
        "shape_numeric_present":projection.shape.numeric_features.is_some(),
        "shape_host_content_present":projection.shape.host_content_features.is_some(),
        "shape_row_multiset_present":projection.shape.row_multiset_features.is_some(),
        "statistics_error_source":"FUTURE_CPU_STATISTICAL_FAILURE is emitted by test-only plain inner before .ok()"})
}

fn attempt(
    fixture: &Fixture,
    requests: &[ExecutorResourcePlanningRequest<'_>],
    query: &FutureWaveCostQuery<'_>,
    ordinal: usize,
    allowance: Duration,
    mode: Mode,
) -> (Attempt, Option<ExecutionCostRouteProjection>) {
    let started = Instant::now();
    let deadline = started.checked_add(allowance).unwrap();
    let polls = std::cell::Cell::new(0_u64);
    let mut poll = || {
        polls.set(polls.get() + 1);
        let now = Instant::now();
        now >= started && now < deadline
    };
    // EVERY attempt captures actual current lane/registry/resources. One
    // unchanged deadline includes capture, initial_state, and projection.
    let captured = fixture.executor.execution_cost_route_view(
        black_box(requests),
        ResourcePlanningLimits::default(),
        &mut poll,
    );
    let captured_at = Instant::now();
    let capture_polls = polls.get();
    let (result, project_ns, stage, ended) = match captured {
        ExecutionCostRouteAvailability::Known(view) => {
            let view = view.with_structured_capture(true);
            let begin = Instant::now();
            let initial = view.initial_state();
            let result = black_box(fixture.executor.project_execution_cost_wave(
                black_box(&view),
                black_box(&initial),
                black_box(query),
                &mut poll,
            ));
            let ended = Instant::now();
            (result, Some(elapsed_ns(begin, ended)), "project", ended)
        }
        ExecutionCostRouteAvailability::Unknown(reason) => (
            ExecutionCostRouteAvailability::Unknown(reason),
            None,
            "capture",
            captured_at,
        ),
    };
    let (mut status, projection) = match result {
        ExecutionCostRouteAvailability::Known(projection) => {
            ("Known".to_string(), Some(projection))
        }
        ExecutionCostRouteAvailability::Unknown(reason) => (format!("{stage}:{reason:?}"), None),
    };
    let digest = projection.as_ref().map(semantic).transpose();
    let semantic_sha256 = match digest {
        Ok(value) => value,
        Err(reason) => {
            status = format!("evidence:{reason}");
            None
        }
    };
    let report = Attempt {
        ordinal,
        warmup: ordinal < mode.warmup(),
        capture_ns: elapsed_ns(started, captured_at),
        project_ns,
        total_ns: elapsed_ns(started, ended),
        capture_polls,
        project_polls: polls.get() - capture_polls,
        status,
        deadline_live_at_end: ended >= started && ended < deadline,
        // Hashing and semantic checks are outside the elapsed interval.
        semantic_sha256,
        evidence_diagnostic: projection
            .as_ref()
            .map(evidence_diagnostic)
            .unwrap_or(serde_json::Value::Null),
    };
    let complete = report.status == "Known";
    (report, projection.filter(|_| complete))
}

fn projection_boundaries(
    fixture: &Fixture,
    requests: &[ExecutorResourcePlanningRequest<'_>],
    query: &FutureWaveCostQuery<'_>,
    allowance: Duration,
) -> serde_json::Value {
    let deadline = Instant::now().checked_add(allowance).unwrap();
    let capture = fixture.executor.execution_cost_route_view(
        requests,
        ResourcePlanningLimits::default(),
        &mut || Instant::now() < deadline,
    );
    let ExecutionCostRouteAvailability::Known(view) = capture else {
        return serde_json::json!({"capture":format!("{capture:?}"),"qualified":false});
    };
    let view = view.with_structured_capture(true);
    let initial = view.initial_state();
    let expired = fixture
        .executor
        .project_execution_cost_wave(&view, &initial, query, &mut || false);
    let expired_ok = matches!(
        expired,
        ExecutionCostRouteAvailability::Unknown(ExecutionCostRouteUnknown::BudgetExhausted)
    );
    let deadline = Instant::now().checked_add(allowance).unwrap();
    let other = fixture.executor.execution_cost_route_view(
        requests,
        ResourcePlanningLimits::default(),
        &mut || Instant::now() < deadline,
    );
    let ExecutionCostRouteAvailability::Known(other) = other else {
        return serde_json::json!({"expired":format!("{expired:?}"),"foreign_capture":format!("{other:?}"),"qualified":false});
    };
    let stale = fixture
        .executor
        .project_execution_cost_wave(&other, &initial, query, &mut || Instant::now() < deadline);
    let stale_ok = matches!(
        stale,
        ExecutionCostRouteAvailability::Unknown(ExecutionCostRouteUnknown::StaleView)
    );
    let mut repeats = Vec::new();
    let mut statuses = Vec::new();
    for _ in 0..2 {
        let deadline = Instant::now().checked_add(allowance).unwrap();
        match fixture
            .executor
            .project_execution_cost_wave(&view, &initial, query, &mut || Instant::now() < deadline)
        {
            ExecutionCostRouteAvailability::Known(p) => {
                statuses.push(evidence_diagnostic(&p));
                repeats.push(p);
            }
            ExecutionCostRouteAvailability::Unknown(reason) => {
                statuses.push(serde_json::json!({"unknown":format!("{reason:?}")}))
            }
        }
    }
    let mut equality = "unavailable".to_string();
    let mut same = false;
    if repeats.len() == 2 {
        let left = semantic(&repeats[0]);
        let right = semantic(&repeats[1]);
        let deadline = Instant::now().checked_add(allowance).unwrap();
        let equal = repeats[0]
            .state
            .same_future_state(&repeats[1].state, &mut || Instant::now() < deadline);
        same = left.is_ok() && left == right && matches!(equal, Ok(true));
        equality = format!("{equal:?}");
    }
    serde_json::json!({"expired":if expired_ok {"BudgetExhausted".into()} else {format!("{expired:?}")},
        "foreign_state":format!("{stale:?}"),"repeat_status":statuses,"complete_state_equality":equality,
        "qualified":expired_ok&&stale_ok&&same})
}

fn measure(
    fixture: &Fixture,
    prefills: &[PlanRuntimePrefillInput],
    decodes: &[PlanRuntimeDecodeInput],
    probe: &Probe,
    case: &str,
    mode: Mode,
) -> (Option<ExecutionCostRouteProjection>, bool) {
    fixture.warm(prefills, decodes);
    let before = fixture.target_resource_evidence(prefills, decodes);
    let submissions = fixture.submissions();
    let masks = mask_counts(fixture);
    let rows = fixture.rows(prefills, decodes);
    let requests: Vec<_> = rows
        .iter()
        .map(|(sequence, _, _)| ExecutorResourcePlanningRequest {
            request_id: sequence.request_id(),
            cache_id: decodes
                .iter()
                .any(|d| &d.request_id == sequence.request_id())
                .then_some(sequence.cache_id.as_str()),
        })
        .collect();
    let query_rows: Vec<_> = rows
        .iter()
        .enumerate()
        .map(|(participant_index, (sequence, _, range))| {
            let host = probe
                .participants
                .iter()
                .find(|p| &p.request_id == sequence.request_id())
                .unwrap();
            let (work, output) = if let Some(input) = prefills
                .iter()
                .find(|p| &p.request_id == sequence.request_id())
            {
                (
                    ActualRowWork::Prefill {
                        offset: range.start.try_into().unwrap(),
                        count: range.len().try_into().unwrap(),
                        total_prompt_tokens: input.input_tokens.len().try_into().unwrap(),
                    },
                    FutureCostOutput::Prefill {
                        final_logits: input.chunk.is_final(),
                    },
                )
            } else {
                let input = decodes
                    .iter()
                    .find(|p| &p.request_id == sequence.request_id())
                    .unwrap();
                (
                    ActualRowWork::Decode {
                        kv_tokens: range.start.try_into().unwrap(),
                    },
                    FutureCostOutput::Decode {
                        policy: &input.logits_policy,
                    },
                )
            };
            FutureWaveCostRow {
                participant_index,
                work,
                output,
                host_policy_signature: host.output_policy_signature.unwrap(),
                host_features: host.host_features,
            }
        })
        .collect();
    let query = FutureWaveCostQuery {
        kind: if decodes.is_empty() {
            ActualWaveKind::Prefill
        } else {
            ActualWaveKind::Decode
        },
        rows: &query_rows,
    };
    let allowance = mode.allowance();
    let boundaries = projection_boundaries(fixture, &requests, &query, allowance);
    let mut reports = Vec::with_capacity(mode.warmup() + mode.measured());
    let mut expected = None::<ExecutionCostRouteProjection>;
    let mut expected_digest = None;
    for ordinal in 0..mode.warmup() + mode.measured() {
        let (mut report, projection) =
            attempt(fixture, &requests, &query, ordinal, allowance, mode);
        if let Some(projection) = projection {
            if let Some(digest) = &expected_digest {
                if report.semantic_sha256.as_ref() != Some(digest) {
                    report.status = "semantic_mismatch".into();
                }
            } else {
                expected_digest = report.semantic_sha256.clone();
                expected = Some(projection);
            }
        }
        println!(
            "FUTURE_CPU_ATTEMPT {}",
            serde_json::json!({"case":case,"mode":mode.label(),"attempt":report})
        );
        reports.push(report);
    }
    let unchanged = fixture.submissions() == submissions
        && mask_counts(fixture) == masks
        && fixture.target_resource_evidence(prefills, decodes) == before;
    let plan = fixture.executor.resolved_plan.execution_plan().payload();
    let measured: Vec<_> = reports.iter().filter(|report| !report.warmup).collect();
    let eligible = unchanged
        && boundaries["qualified"] == true
        && measured
            .iter()
            .all(|r| r.status == "Known" && r.deadline_live_at_end);
    let summary = serde_json::json!({
        "known":measured.iter().filter(|r| r.status == "Known").count(),
        "unknown":measured.iter().filter(|r| r.status != "Known").count(),
        "deadline_overruns":measured.iter().filter(|r| !r.deadline_live_at_end).count(),
        "all_attempt_total_ns":distribution(measured.iter().map(|r| r.total_ns).collect()),
        "entered_projection_ns":distribution(measured.iter().filter_map(|r| r.project_ns).collect()),
        "capture_ns":distribution(measured.iter().map(|r| r.capture_ns).collect()),
        "comparison_eligible":eligible && mode == Mode::ReleaseScreen, "correctness_eligible":eligible, "resources_unchanged":unchanged,
    });
    println!(
        "FUTURE_CPU_SCREEN {}",
        serde_json::json!({
            "schema_version":2,"case":case,"mode":mode.label(),"rows":query_rows.len(),
            "nodes":plan.nodes().len(),"dynamic_descriptors":plan.memory().dynamic_descriptors().len(),
            "warmup":mode.warmup(),"measured":mode.measured(),"budget_ns":allowance.as_nanos(),
            "capture_and_projection_share_deadline":true,"boundaries":boundaries,
            "attempts":reports,"semantic_sha256":expected_digest,"summary":summary,
            "limitation":"CPU wall for tiny real Metal Qwen eager projection; no GPU/9B/SLO inference",
        })
    );
    (expected, eligible)
}

fn distribution(mut samples: Vec<u64>) -> serde_json::Value {
    samples.sort_unstable();
    if samples.is_empty() {
        return serde_json::Value::Null;
    }
    let percentile = |numerator: usize| samples[(samples.len() * numerator).div_ceil(100) - 1];
    serde_json::json!({"samples":samples.len(),"min":samples[0],"p50":percentile(50),
        "p95":percentile(95),"max":samples[samples.len()-1]})
}

fn probe(ids: &[&RequestId], generated: u64) -> Probe {
    let mut probe = Probe::new(ids);
    probe.recorder = BoundedWaveRecorder::new(
        NonZeroU64::new(1).unwrap(),
        CostRecorderLimits {
            max_waves: 1,
            max_rows_per_wave: 8,
            max_retained_rows: ferrum_types::SloCostObservationConfig::default()
                .max_retained_rows_per_call
                .get(),
        },
    )
    .unwrap();
    declare_numeric_context(&mut probe, &vec![generated; ids.len()]);
    // Explicit host contract of this fixed plain-text greedy fixture. These are
    // declared policy facts, not tokenizer outcomes or fitted empirical data.
    for participant in &mut probe.participants {
        participant
            .host_features
            .as_mut()
            .unwrap()
            .policy
            .empirical_content_domain = Some(HostContentDomainV1::PlainTextGreedyV1);
    }

    probe
}

fn verify_actual(
    probe: &Probe,
    projection: &Option<ExecutionCostRouteProjection>,
    case: &str,
) -> bool {
    let observations = probe.recorder.observations();
    let result = (|| -> std::result::Result<(), String> {
        let projection = projection.as_ref().ok_or("no_complete_prediction")?;
        if observations.len() != 1 {
            return Err(format!("actual_wave_count:{}", observations.len()));
        }
        let wave = &observations[0];
        if wave.outcome != Some(ActualWaveOutcome::Completed)
            || wave.boundary != WaveObservationBoundary::ExecutorOnly
        {
            return Err(format!(
                "actual_outcome:{:?}, boundary:{:?}",
                wave.outcome, wave.boundary
            ));
        }
        let actual = wave.shape.as_ref().ok_or("missing_actual_shape")?;
        if actual.restore_bytes != 0
            || actual.maintenance_bytes != 0
            || actual.maintenance_units != 0
        {
            return Err("unexpected_actual_maintenance".into());
        }
        let canonical = CanonicalWaveCostShape {
            kind: actual.kind,
            path: actual.path,
            graph: actual.graph,
            row_order: actual.row_order,
            provider_signature: actual.provider_signature,
            output_policy_signature: actual.output_policy_signature,
            numeric_features: actual.numeric_features.clone(),
            host_content_features: actual.host_content_features,
            row_multiset_features: actual.row_multiset_features.clone(),
            rows: actual.rows.iter().map(|r| r.work).collect(),
            recurrent_state_bytes: actual.recurrent_state_bytes,
        };
        if canonical != projection.shape {
            return Err(format!(
                "actual_canonical_mismatch: actual={canonical:?}, predicted={:?}",
                projection.shape
            ));
        }
        if actual.rows.len() != probe.participants.len() {
            return Err("actual_participant_count".into());
        }
        let mut seen = BTreeSet::new();
        for row in &actual.rows {
            let p = probe
                .participants
                .get(row.input_index as usize)
                .ok_or("actual_participant_index")?;
            if !seen.insert(row.input_index)
                || row.request_id != p.request_id
                || row.owner_incarnation != p.owner_incarnation
                || row.work_generation != p.work_generation
            {
                return Err("actual_correlated_identity".into());
            }
        }
        let observed = actual
            .statistical_evidence
            .as_ref()
            .ok_or("missing_actual_statistics")?;
        let predicted = projection
            .statistical_evidence
            .as_ref()
            .ok_or("missing_predicted_statistics")?;
        if observed != predicted {
            return Err("actual_statistics_mismatch".into());
        }
        if observed.independent_attention_v2() != predicted.independent_attention_v2() {
            return Err("actual_independent_attention_v2_mismatch".into());
        }
        if let Some(v2) = observed.independent_attention_v2() {
            v2.validate_actual(actual)
                .map_err(|e| format!("actual_independent_binding:{e:?}"))?;
        }
        let recipe = observed
            .structured_capture()
            .ok_or("missing_actual_structure")?
            .map_err(|e| format!("actual_structure:{e:?}"))?;
        let expected = predicted
            .structured_capture()
            .ok_or("missing_predicted_structure")?
            .map_err(|e| format!("predicted_structure:{e:?}"))?;
        if recipe.device() != expected.device() {
            return Err("actual_device_recipe_mismatch".into());
        }
        if recipe
            .algorithm_work()
            .map_err(|e| format!("actual_algorithms:{e:?}"))?
            != expected
                .algorithm_work()
                .map_err(|e| format!("predicted_algorithms:{e:?}"))?
        {
            return Err("actual_algorithm_work_mismatch".into());
        }
        if recipe != expected {
            return Err("actual_complete_recipe_mismatch".into());
        }
        Ok(())
    })();
    println!(
        "FUTURE_CPU_ACTUAL {}",
        serde_json::json!({"case":case,"parity_verified":result.is_ok(),
        "observed_waves":observations.len(),"diagnostic":result.as_ref().err(),
        "actual_statistics_present":observations.first().and_then(|w|w.shape.as_ref()).is_some_and(|s|s.statistical_evidence.is_some())})
    );
    result.is_ok()
}

#[tokio::test]
#[ignore = "real Metal correctness only, explicit 1s read budget, no performance result"]
async fn future_metal_product_projection_cpu_correctness() {
    assert!(
        run(Mode::Correctness).await,
        "see all retained cell/attempt diagnostics"
    );
}
#[tokio::test]
#[ignore = "same-host paired release CPU diagnostic; real Metal fixture"]
async fn future_metal_product_projection_cpu_screen() {
    run(Mode::ReleaseScreen).await;
}
async fn run(mode: Mode) -> bool {
    let mut complete = true;
    for width in [1, 4] {
        let fixture = Fixture::with_structured_capture(32).await;
        let prefills: Vec<_> = (0..width)
            .map(|i| {
                let tokens: Vec<_> = (0..i + 1).map(|j| ((j + i) % 3) as u32).collect();
                let mut input = prompt(&tokens, tokens.len());
                input.request_id = RequestId(uuid::Uuid::from_u128(
                    0xc05f_0000_0000_0000_0000_0000_0000_0000_u128
                        | ((width as u128) << 16)
                        | i as u128 + 1,
                ));
                input
            })
            .collect();
        for input in &prefills {
            fixture.admit(input);
        }
        let ids: Vec<_> = prefills.iter().map(|p| &p.request_id).collect();
        let mut prefill_probe = probe(&ids, 0);
        let case = format!("prefill_full_b{width}");
        let (predicted, eligible) = measure(&fixture, &prefills, &[], &prefill_probe, &case, mode);
        let outputs = match executed(
            fixture
                .executor
                .plan_runtime_batch_prefill_with_capacity_observed(
                    &prefills,
                    &mut prefill_probe.context().with_structured_capture(true),
                )
                .await,
        ) {
            PlanRuntimeBatchPrefillOutcome::Completed(outputs) => outputs,
            _ => panic!("real prefill did not complete"),
        };
        complete &= verify_actual(&prefill_probe, &predicted, &case) && eligible;
        let mut decodes: Vec<_> = prefills
            .iter()
            .zip(outputs)
            .map(|(input, output)| {
                let mut input = PlanRuntimeDecodeInput::new(
                    input.request_id.clone(),
                    TokenId::new(1),
                    Arc::clone(output.output().kv_cache()),
                );
                input.logits_policy = LogitsReturnPolicy::GreedyArgmax {
                    token_mask: Some(TokenSelectionMask::new(vec![1, 0, 1])),
                    repetition_penalty: None,
                };
                input
            })
            .collect();
        for (step, label) in ["cold_mask", "resident_mask"].into_iter().enumerate() {
            let ids: Vec<_> = decodes.iter().map(|p| &p.request_id).collect();
            let mut decode_probe = probe(&ids, 1 + step as u64);
            let case = format!("decode_{label}_b{width}");
            let (predicted, eligible) =
                measure(&fixture, &[], &decodes, &decode_probe, &case, mode);
            let outputs = decode_outputs(executed(
                fixture
                    .executor
                    .plan_runtime_batch_decode_with_capacity_observed(
                        &decodes,
                        &mut decode_probe.context().with_structured_capture(true),
                    )
                    .await,
            ));
            complete &= verify_actual(&decode_probe, &predicted, &case) && eligible;
            for (input, output) in decodes.iter_mut().zip(outputs) {
                input.kv_cache = Arc::clone(&output.kv_cache);
            }
        }
        for input in decodes {
            fixture.executor.release_cache(&input.kv_cache.cache_id());
        }
    }
    println!(
        "FUTURE_CPU_PROTOCOL {}",
        serde_json::json!({"revision":2,"mode":mode.label(),
        "complete":complete,"performance_result":mode==Mode::ReleaseScreen,"budget_ns":mode.allowance().as_nanos()})
    );
    complete
}
