//! Real native executor/session/source roundtrip. The narrow nonterminal decode
//! window tests the capture pipeline, not full-horizon model coverage or speed.
use super::*;
use crate::continuous_engine::{
    HostStageEvidenceV1, HostStageWork, StructuredCalibrationOptionsV2,
};
use ferrum_interfaces::{execution_cost::HostPendingConstraintV2, output_flow::OutputCompletion};
use ferrum_scheduler::implementations::continuous::{
    cost_model::structured_v2::{windows::*, *},
    cost_profile::{
        export_structured_profile_v10, load_structured_profile_v10, CostProfileLoadLimits,
        ProfileLoadClock,
    },
};

#[path = "calibration_v2/catalog.rs"]
mod catalog;

#[path = "calibration_v2/multi.rs"]
mod multi;

const PROMPT: &str = "hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello";

struct OutputDirectory(std::path::PathBuf);
impl OutputDirectory {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "ferrum-structured-v2-live-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for OutputDirectory {
    fn drop(&mut self) {
        if std::thread::panicking() {
            eprintln!(
                "retained original structured capture at {}",
                self.0.display()
            );
        } else {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

async fn add_native_request(
    session: &mut CalibrationSession,
) -> (RequestId, tokio::task::JoinHandle<()>) {
    let mut request =
        ferrum_types::InferenceRequest::new(PROMPT, session.configuration().model.model_id.clone());
    request.stream = true;
    request.sampling_params.max_tokens = 5;
    request.sampling_params.temperature = 0.0;
    request.sampling_params.repetition_penalty = 1.0;
    request
        .metadata
        .insert("ferrum_ignore_eos".into(), true.into());
    let id = request.id.clone();
    let mut output = session
        .add_request(
            request,
            InferenceRequestContext::from_ingress(slo_clock_now()),
            Arc::new(OutputProjectionContract::cli_text()),
        )
        .await
        .unwrap();
    let consume = tokio::spawn(async move {
        let mut terminal = false;
        while let Some(frame) = output.frames.next().await {
            let end = frame.metadata().terminal;
            drop(frame);
            if end {
                terminal = true;
                break;
            }
        }
        assert!(terminal, "original credited terminal frame is required");
        let completion = output.completion.await.unwrap();
        match completion.payload() {
            OutputCompletion::Succeeded { reason, usage, .. } => {
                assert_eq!(*reason, ferrum_types::FinishReason::Length);
                assert_eq!(usage.completion_tokens, 5);
            }
            OutputCompletion::Failed(error) => {
                panic!("real output completion failed: {}", error.message())
            }
        }
    });
    (id, consume)
}

async fn complete_native_request(session: &mut CalibrationSession) -> Vec<CalibrationWaveReport> {
    complete_native_request_with_route(
        session,
        crate::continuous_engine::CalibrationDecodeRoute::Actual,
    )
    .await
}

async fn complete_native_request_with_route(
    session: &mut CalibrationSession,
    route: crate::continuous_engine::CalibrationDecodeRoute,
) -> Vec<CalibrationWaveReport> {
    let (id, consume) = add_native_request(session).await;
    let inner = session.test_engine_inner();
    ready(&inner, &id).await;
    admit(session, &id).await;
    let mut reports = Vec::new();
    while !session.frontiers().unwrap().is_empty() {
        ready(&inner, &id).await;
        let f = frontier(session, &id);
        let work = if f.prefill_progress().is_some() {
            f.prefill_work(n32(8)).unwrap()
        } else {
            f.decode_work_with_route(route).unwrap()
        };
        reports.push(wave(session, vec![work]).await);
    }
    tokio::time::timeout(Duration::from_secs(30), consume)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(reports.len(), 6);
    assert_eq!(
        reports
            .iter()
            .filter(|report| matches!(
                report.host_stages.as_ref().unwrap().rows[0].actual_work,
                HostStageWork::Prefill { .. }
            ))
            .count(),
        2
    );
    let final_stage = reports.last().unwrap().host_stages.as_ref().unwrap();
    assert_eq!(
        final_stage.rows[0].terminal.as_ref().unwrap().finish_reason,
        ferrum_types::FinishReason::Length
    );
    reports
}

#[tokio::test]
async fn structured_v2_metal_complete_cohorts_source_export_load_and_query() {
    let (mut session, model_directory) = fixture::fixture_with_structured_geometry(
        true,
        weights::CausalGeometry {
            context: 32,
            ..weights::CausalGeometry::TINY
        },
    )
    .await;
    let files = OutputDirectory::new();
    // Complete independent warmup precedes discovery. Both use real requests
    // and stay outside the three-phase source. The 16-token prompt puts the
    // adjacent decode frontiers beyond the actual <=16 SIMD-group selector
    // transitions; it does not authorize shorter-context owners.
    let warmup = complete_native_request(&mut session).await;
    let discovered = complete_native_request(&mut session).await;
    let independent = discovered[2].structured_cost_input_v2().unwrap();
    // Retain exact independent recipes on failure, including one-time uploads;
    // these serialize-only diagnostics never become collector observations.
    let discovery_recipes: Vec<_> = [(&"warmup", &warmup), (&"discovery", &discovered)].into_iter().map(|(phase, reports)| {
        serde_json::json!({"phase":phase,"recipes":reports.iter().map(|report| {
            report.host_stages.as_ref().unwrap().structured_evidence.as_ref().unwrap().as_ref().unwrap().recipe()
        }).collect::<Vec<_>>()})
    }).collect();
    std::fs::write(
        files.0.join("independent-discovery.json"),
        serde_json::to_vec(&discovery_recipes).unwrap(),
    )
    .unwrap();
    for report in &discovered[2..5] {
        assert_eq!(
            report.structured_cost_input_v2().unwrap().owner(),
            independent.owner(),
            "frozen adjacent discovery frontiers must share the actual selected owner"
        );
    }
    let owner = independent.owner().clone();
    assert_eq!(owner.rows, 1);
    assert_eq!(owner.role, StructuredWaveRoleV2::OrdinaryDecode);
    assert_eq!(owner.product, StructuredProductV2::GreedyToken);
    let fingerprint = discovered[2]
        .host_stages
        .as_ref()
        .unwrap()
        .fingerprint
        .as_ref()
        .unwrap()
        .clone();
    let initial = session
        .freeze_cost_model()
        .await
        .unwrap()
        .accepted_ordinal();
    let scope = StructuredScopeV2 {
        owner: owner.clone(),
        coverage: StructuredCoverageV2 {
            pending_eligible_positions: vec![],
            authorized_pending_constraints: vec![HostPendingConstraintV2::AnySubset],
            pending_counts: vec![0],
            length_counts: vec![0],
            pending_positions: vec![],
            length_positions: vec![],
            joint_counts: vec![(0, 0)],
        },
    };
    let rule = MembershipRuleV2 {
        owner,
        windows: vec![FrontierWindowV2 {
            rows: vec![RowWindowV2 {
                generated_before: ClosedRangeV2 {
                    minimum: 1,
                    maximum: 3,
                },
                remaining_output: ClosedRangeV2::ALL,
                context_before: ClosedRangeV2::ALL,
                work: WorkWindowV2::Decode {
                    kv_tokens: ClosedRangeV2::ALL,
                },
            }],
        }],
    };
    let plan = CohortPlanV2 {
        phases: std::array::from_fn(|_| {
            (0..3)
                .map(|repetition| CohortV2 {
                    manifest_case: 0,
                    repetition,
                    requests: vec![CohortRequestV2 {
                        manifest_prompt: 0,
                        maximum_output: 5,
                    }],
                })
                .collect()
        }),
    };
    let settings = StructuredSettingsV2 {
        static_margin_ns: 100_000_000,
        max_wave_ns: 30_000_000_000,
        max_sample_age_ns: 600_000_000_000,
        ..StructuredSettingsV2::default()
    };
    let ttl = settings.max_sample_age_ns;
    session.begin_structured_cost_calibration_v2(StructuredCalibrationOptionsV2 {
        observations_path:files.0.join("source3.jsonl"),protocol_sha256:[83;32],scope:scope.clone(),membership_rule:rule,
        cohort_plan:plan,cohort_manifest_payload:serde_json::json!({"fixture":"real tiny native Metal","prompt":PROMPT,
            "maximum_output":5,"ignore_eos":true,"phases":["fit","residual","qualification"],"cohorts_per_phase":3}),
        settings,phase_members:[9,9,9],maximum_offered_waves:NonZeroUsize::new(128).unwrap(),
        maximum_file_bytes:NonZeroU64::new(32*1024*1024).unwrap(),
    }).await.unwrap();
    let mut freezes = Vec::new();
    let mut terminal_count = 0;
    let mut original_calls =
        std::collections::BTreeMap::<u64, (usize, usize, Arc<HostStageEvidenceV1>, bool)>::new();
    let mut original_terminals = std::collections::BTreeMap::new();
    for phase in 0..3 {
        for cohort in 0..3 {
            session.begin_structured_cost_cohort_v2(cohort).unwrap();
            let reports = complete_native_request(&mut session).await;
            terminal_count += reports
                .iter()
                .filter(|r| r.host_stages.as_ref().unwrap().rows[0].terminal.is_some())
                .count();
            for report in &reports {
                let stages = Arc::clone(report.host_stages.as_ref().unwrap());
                let ordinal = report
                    .host_stage_queue
                    .as_ref()
                    .unwrap()
                    .accepted_ordinal
                    .unwrap();
                let numeric = &stages
                    .actual_shape
                    .as_ref()
                    .unwrap()
                    .numeric_features
                    .as_ref()
                    .unwrap()
                    .rows[0];
                assert_eq!(numeric.maximum_output_tokens, 5);
                let member = matches!(stages.rows[0].actual_work, HostStageWork::Decode { .. })
                    && (1..=3).contains(&numeric.generated_tokens_before);
                if member {
                    let actual = report.structured_cost_input_v2().unwrap();
                    assert_eq!(
                        actual.owner(),
                        &scope.owner,
                        "independent discovery owner drift at phase={phase} cohort={cohort} fifo={ordinal} generated={}; source={}",
                        numeric.generated_tokens_before,
                        files.0.join("source3.jsonl").display(),
                    );
                }
                if let Some(terminal) = &stages.rows[0].terminal {
                    assert_eq!(terminal.finish_reason, ferrum_types::FinishReason::Length);
                    assert_eq!(terminal.generated_tokens, 5);
                    assert!(original_terminals
                        .insert(
                            (phase, cohort),
                            (
                                ordinal,
                                stages.call_id,
                                stages.rows[0].request_id.clone(),
                                serde_json::to_value(terminal).unwrap()
                            )
                        )
                        .is_none());
                }
                assert!(original_calls
                    .insert(ordinal, (phase, cohort, stages, member))
                    .is_none());
            }
            session.end_structured_cost_cohort_v2().unwrap();
        }
        let progress = session.structured_cost_progress_v2().unwrap();
        assert!(
            progress.failure.is_none(),
            "collector failure: {:?}",
            progress.failure
        );
        assert_eq!(
            progress.reserved_members[phase],
            9,
            "original source: {}",
            files.0.join("source3.jsonl").display()
        );
        assert_eq!(progress.completed_members[phase], 9);
        assert_eq!(progress.failed_members[phase], 0);
        assert!(session.structured_cost_coverage_v2().unwrap().complete());
        let receipt = session.freeze_structured_cost_phase_v2().await.unwrap();
        assert_eq!(
            receipt.phase,
            [
                crate::continuous_engine::StructuredCapturePhase::Fit,
                crate::continuous_engine::StructuredCapturePhase::Residual,
                crate::continuous_engine::StructuredCapturePhase::Qualification
            ][phase]
        );
        assert_eq!(receipt.member_cutoff, (phase as u64 + 1) * 9);
        freezes.push(receipt);
    }
    assert_eq!(terminal_count, 9);
    let artifact = session
        .finish_structured_cost_calibration_v2()
        .await
        .unwrap();
    assert_eq!(artifact.scope_members, 27);
    assert_eq!(artifact.scope_failures, 0);
    assert!(artifact.failure.is_none());
    let live = artifact.model.as_ref().unwrap();
    let source = std::fs::read_to_string(&artifact.source_path).unwrap();
    let records: Vec<serde_json::Value> = source
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(records[0]["record"]["initial_fifo_cutoff"], initial);
    let mut reserved = std::collections::BTreeSet::new();
    let mut fifo = initial;
    let mut completed_requests = 0;
    let mut outside = 0;
    let mut phase_members = [0; 3];
    let mut phase_outside = [0; 3];
    let mut phase_terminals = [0; 3];
    for (position, wrapper) in records.iter().enumerate() {
        assert_eq!(wrapper["source_record_ordinal"], position as u64 + 1);
        let row = &wrapper["record"];
        match row["kind"].as_str() {
            Some("reserved") => {
                let offered = row["offered"].as_u64().unwrap();
                assert!(reserved.insert(offered));
                if !row["member"].is_null() {
                    let generated = row["prepared"]["rows"][0]["frontier"]["generated_before"]
                        .as_u64()
                        .unwrap();
                    assert!((1..=3).contains(&generated));
                    assert_eq!(row["window"], 0);
                }
            }
            Some("completed") => {
                assert!(reserved.remove(&row["offered"].as_u64().unwrap()));
                fifo += 1;
                assert_eq!(row["queue"]["accepted_ordinal"], fifo);
                assert!(row["conversion_error"].is_null());
                let (phase, cohort, actual, expected_member) = original_calls
                    .remove(&fifo)
                    .expect("source call must be an original live report");
                let label = ["fit", "residual", "qualification"][phase];
                assert_eq!(row["phase"], label);
                assert_eq!(row["cohort"], cohort);
                assert_eq!(!row["member"].is_null(),expected_member,"membership must be iff the original pre-execution Decode frontier is in the frozen window");
                let source_stages = if expected_member {
                    &row["host_stages"]
                } else {
                    &row["outside_settlement"]
                };
                assert_eq!(source_stages["call_id"], actual.call_id);
                assert_eq!(
                    source_stages["prepare_started_at_ns"],
                    actual.prepare_started_at_ns.unwrap()
                );
                assert_eq!(
                    source_stages["executor_returned_at_ns"],
                    actual.executor_returned_at_ns.unwrap()
                );
                assert_eq!(
                    source_stages["finalized_at_ns"],
                    actual.finalized_at_ns.unwrap()
                );
                assert_eq!(source_stages["full_wall_ns"], actual.full_wall_ns.unwrap());
                assert_eq!(
                    source_stages["rows"][0]["request_id"],
                    actual.rows[0].request_id.to_string()
                );
                assert_eq!(
                    source_stages["rows"][0]["terminal"],
                    serde_json::to_value(&actual.rows[0].terminal).unwrap()
                );
                let lower = if phase == 0 {
                    records[0]["record"]["opened_at_ns"].as_u64().unwrap()
                } else {
                    freezes[phase - 1].frozen_at_ns
                };
                assert!(actual.prepare_started_at_ns.unwrap() >= lower);
                assert!(actual.finalized_at_ns.unwrap() <= freezes[phase].frozen_at_ns);
                let previous_fifo = if phase == 0 {
                    initial
                } else {
                    freezes[phase - 1].accepted_fifo_cutoff
                };
                assert!(fifo > previous_fifo && fifo <= freezes[phase].accepted_fifo_cutoff);
                if expected_member {
                    phase_members[phase] += 1;
                    assert_eq!(row["numeric"]["call_id"], actual.call_id);
                    assert_eq!(
                        row["numeric"]["observed_at_ns"],
                        actual.finalized_at_ns.unwrap()
                    );
                } else {
                    phase_outside[phase] += 1;
                    outside += 1;
                    assert!(row["outside_settlement"].is_object());
                }
            }
            Some("request_completed") => {
                completed_requests += 1;
                let request = &row["request"];
                let phase = match request["phase"].as_str().unwrap() {
                    "fit" => 0,
                    "residual" => 1,
                    "qualification" => 2,
                    _ => panic!("unexpected request phase"),
                };
                let cohort = request["cohort"].as_u64().unwrap() as usize;
                let (ordinal, call_id, id, terminal) = original_terminals
                    .remove(&(phase, cohort))
                    .expect("only the real completed cohort can own a terminal event");
                assert_eq!(request["slot"], 0);
                assert_eq!(request["request_id"], id.to_string());
                assert_eq!(request["fifo"], ordinal);
                assert_eq!(request["call_id"], call_id);
                assert_eq!(request["generated_tokens"], 5);
                assert_eq!(request["terminal"], terminal);
                phase_terminals[phase] += 1;
            }
            _ => {}
        }
    }
    assert!(reserved.is_empty());
    assert!(
        original_calls.is_empty(),
        "no original FIFO entry may be omitted"
    );
    assert!(
        original_terminals.is_empty(),
        "every original Length must finish its declared slot"
    );
    assert_eq!(phase_members, [9; 3]);
    assert_eq!(phase_outside, [9; 3]);
    assert_eq!(phase_terminals, [3; 3]);
    assert_eq!(completed_requests, 9);
    assert_eq!(outside, 27);
    assert_eq!(freezes.last().unwrap().accepted_fifo_cutoff, fifo);
    assert!(freezes
        .windows(2)
        .all(|p| p[0].frozen_at_ns <= p[1].frozen_at_ns));
    let limits = CostProfileLoadLimits::default();
    let exported = export_structured_profile_v10(
        &artifact.source_path,
        artifact.source_sha256,
        &files.0.join("profile10.json"),
        1_000_000,
        &limits,
    )
    .unwrap();
    assert_eq!(exported.parameters_sha256, live.parameters_signature());
    let closing = records.last().unwrap()["record"]["closing"]["wall_unix_ns"]
        .as_u64()
        .unwrap();
    let loaded = load_structured_profile_v10(
        &exported.path,
        &fingerprint,
        &limits,
        ProfileLoadClock {
            wall_unix_ns: Some(closing),
            wall_max_error_ns: Some(1_000_000),
            monotonic_now_ns: 10,
        },
    )
    .unwrap();
    assert_eq!(loaded.scope(), &scope);
    assert_eq!(loaded.parameters_signature(), live.parameters_signature());
    assert_eq!(loaded.provenance().reserved_members, 27);
    let query = StructuredQueryV2::exact(independent);
    let predicted = loaded
        .predict_query_local(&fingerprint, &query, 10)
        .unwrap();
    assert!(predicted.valid_until_ns > loaded.model_now_ns(10).unwrap());
    assert!(
        matches!(
            loaded.predict_query_local(&fingerprint, &query, 10 + ttl),
            Err(StructuredUnknownV2::Stale)
        ),
        "import must not renew the original sample TTL or substitute another Unknown reason"
    );
    // Re-import through the real product startup boundary with its actual
    // paired load clock. This must retain the original source age/phase cuts.
    use crate::continuous_engine::inner::cost_observation::EngineCostRuntime;
    use ferrum_scheduler::implementations::continuous::{
        cost_model::CostBoundary,
        slo_planner::{PlanningCostEvidenceRequirement, PlanningCostModel},
    };
    let identity = session
        .test_engine_inner()
        .cost_runtime
        .as_ref()
        .unwrap()
        .identity
        .clone();
    let mut config = ferrum_types::SloCostObservationConfig::structured_whole_wave_v2();
    config.profile_import.declared_local_clock_max_error_ns = Some(1_000_000);
    let runtime = EngineCostRuntime::new(identity.clone(), &config, Some(&exported.path)).unwrap();
    let receipt = runtime.profile_receipt().unwrap();
    assert_eq!(receipt.schema_version, 10);
    assert!(receipt.selected_whole_wave.is_none());
    assert!(receipt.structured_whole_wave.is_none());
    let v2 = receipt.structured_whole_wave_v2.as_ref().unwrap();
    assert_eq!(v2.child_count, 1);
    assert_eq!(v2.children[0].parameters_sha256, exported.parameters_sha256);
    assert_eq!(v2.children[0].reserved_members, 27);
    for (before, after) in freezes.iter().zip(&v2.children[0].phases) {
        assert_eq!(before.frozen_at_ns, after.frozen_at_ns);
        assert_eq!(before.accepted_fifo_cutoff, after.accepted_fifo_cutoff);
        assert_eq!(before.parameters_sha256, after.parameters_sha256);
    }
    let snapshot = runtime.snapshot().unwrap();
    assert_eq!(
        snapshot.evidence_requirement(),
        PlanningCostEvidenceRequirement::StructuredV2
    );
    assert_eq!(
        snapshot.planning_boundary(),
        CostBoundary::PreparationToHostSettledV1
    );
    assert_eq!(snapshot.fingerprint(), &fingerprint);
    catalog::check_real_child_catalog(
        &mut session,
        &files.0,
        &exported,
        &loaded,
        &identity,
        &config,
        &runtime,
        ttl,
    )
    .await;
    tokio::time::timeout(Duration::from_secs(30), runtime.shutdown())
        .await
        .unwrap()
        .unwrap();
    session.shutdown().await.unwrap();
    drop(model_directory);
}
