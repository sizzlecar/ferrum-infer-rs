//! Same real waves, two declared owners, independent original source ledgers.
//! This is a narrow pipeline gate; FullLogits here is a real requested route,
//! not a natural pending-UTF8 challenge or complete-horizon qualification.
use super::*;
use crate::continuous_engine::inner::cost_observation::EngineCostRuntime;
use crate::continuous_engine::{
    CalibrationDecodeRoute, StructuredCalibrationGroupLimitsV2,
    StructuredCalibrationGroupOptionsV2, StructuredCapturePhase,
};
use ferrum_scheduler::implementations::continuous::cost_profile::ProfileFingerprint;
use ferrum_types::{SloCostObservationConfig, SloStructuredArtifactKindV2};

#[tokio::test]
async fn structured_group_metal_two_live_owners_sources_profiles_catalog_query() {
    let mut config = SloCostObservationConfig::structured_whole_wave_v2();
    config.profile_import.declared_local_clock_max_error_ns = Some(1_000_000);
    config.profile_import.max_file_bytes = NonZeroUsize::new(128 * 1024 * 1024).unwrap();
    let (mut session, model_directory) = fixture::fixture_with_cost_config(
        true,
        weights::CausalGeometry {
            context: 32,
            ..weights::CausalGeometry::TINY
        },
        Some(config.clone()),
    )
    .await;
    let files = OutputDirectory::new();
    let routes = [
        CalibrationDecodeRoute::Actual,
        CalibrationDecodeRoute::FullLogits,
    ];
    // Each route gets a complete independent warmup before any discovery or
    // frozen population; all requests retain the original five-token Length.
    for route in routes {
        complete_native_request_with_route(&mut session, route).await;
    }
    let mut discoveries = Vec::new();
    for route in routes {
        discoveries.push(complete_native_request_with_route(&mut session, route).await);
    }
    // Persist the actual independent evidence before assertions. These records
    // are diagnostics only; they never enter any child population or source.
    let discovery_recipes: Vec<_> = discoveries
        .iter()
        .enumerate()
        .map(|(route, reports)| {
            serde_json::json!({"route":route,"waves":reports.iter().map(|report| {
            let stages=report.host_stages.as_ref().unwrap();
            let input = report.structured_cost_input_v2().map(|input| {
                serde_json::json!({"owner":input.owner(),"domain":input.domain_signature(),
                    "basis":input.regression_axes(),"support":input.joint_support_coordinates()})
            }).map_err(|error| format!("{error:?}"));
            serde_json::json!({"stages":stages.structured_diagnostic_view(),"input":input})
        }).collect::<Vec<_>>()})
        })
        .collect();
    std::fs::write(
        files.0.join("independent-discovery.json"),
        serde_json::to_vec(&discovery_recipes).unwrap(),
    )
    .unwrap();
    // The first decode may upload a cold token mask. Declare three subsequent
    // frontiers, including the original Length boundary, before capture starts.
    // Do not infer that the first decode has this owner or drop it from FIFO.
    let inputs: Vec<_> = discoveries
        .iter()
        .map(|reports| reports[3].structured_cost_input_v2().unwrap())
        .collect();
    for (reports, input) in discoveries.iter().zip(&inputs) {
        for (position, report) in reports[3..6].iter().enumerate() {
            let stages = report.host_stages.as_ref().unwrap();
            assert_eq!(
                stages
                    .actual_shape
                    .as_ref()
                    .unwrap()
                    .numeric_features
                    .as_ref()
                    .unwrap()
                    .rows[0]
                    .generated_tokens_before,
                position as u64 + 2
            );
            assert_eq!(
                report.structured_cost_input_v2().unwrap().owner(),
                input.owner(),
                "independent three adjacent frontiers must share a declared owner"
            );
            assert_eq!(
                stages.rows[0].terminal.is_some(),
                position == 2,
                "the third declared frontier is the original Length boundary"
            );
        }
    }
    assert_ne!(inputs[0].owner(), inputs[1].owner());
    assert_eq!(inputs[0].owner().product, StructuredProductV2::GreedyToken);
    assert_eq!(inputs[1].owner().product, StructuredProductV2::FullLogits);
    let fingerprint = discoveries[0][2]
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
    let plan = CohortPlanV2 {
        phases: std::array::from_fn(|_| {
            (0..6)
                .map(|cohort| CohortV2 {
                    manifest_case: cohort / 3,
                    repetition: cohort % 3,
                    requests: vec![CohortRequestV2 {
                        manifest_prompt: 0,
                        maximum_output: 5,
                    }],
                })
                .collect()
        }),
    };
    plan.validate().unwrap();
    let settings = StructuredSettingsV2 {
        static_margin_ns: 100_000_000,
        max_wave_ns: 30_000_000_000,
        ..Default::default()
    };
    assert_eq!(
        settings.max_sample_age_ns, 300_000_000_000,
        "original default TTL stays unchanged"
    );
    let children=inputs.iter().enumerate().map(|(child,input)| {
        let owner=input.owner().clone();
        StructuredCalibrationOptionsV2 {
            observations_path:files.0.join(format!("source-{child}.jsonl")),protocol_sha256:[91;32],
            scope:StructuredScopeV2 {owner:owner.clone(),coverage:StructuredCoverageV2 {
                pending_eligible_positions:vec![],authorized_pending_constraints:vec![HostPendingConstraintV2::AnySubset],
                pending_counts:vec![0],length_counts:vec![0,1],pending_positions:vec![],length_positions:vec![0],joint_counts:vec![(0,0),(0,1)]}},
            membership_rule:MembershipRuleV2 {owner,windows:vec![FrontierWindowV2 {rows:vec![RowWindowV2 {
                generated_before:ClosedRangeV2 {minimum:2,maximum:4},remaining_output:ClosedRangeV2::ALL,
                context_before:ClosedRangeV2::ALL,work:WorkWindowV2::Decode {kv_tokens:ClosedRangeV2::ALL}}]}]},
            cohort_plan:plan.clone(),cohort_manifest_payload:serde_json::json!({"fixture":"real native Metal multi owner", "prompt":PROMPT,
                "maximum_output":5,"ignore_eos":true,"routes":["actual","full_logits"],"cohorts_per_phase":6,
                "declared_generated_before":[2,4],"length_counts":[0,1]}),
            settings:settings.clone(),phase_members:[9;3],maximum_offered_waves:NonZeroUsize::new(128).unwrap(),
            maximum_file_bytes:NonZeroU64::new(32*1024*1024).unwrap(),
        }
    }).collect();
    session
        .begin_structured_cost_group_v2(StructuredCalibrationGroupOptionsV2 {
            shared_source: None,
            children,
            limits: StructuredCalibrationGroupLimitsV2::default(),
        })
        .await
        .unwrap();
    let mut originals = std::collections::BTreeMap::<
        u64,
        (usize, usize, Arc<HostStageEvidenceV1>, Option<usize>),
    >::new();
    let mut freezes = Vec::new();
    for phase in 0..3 {
        for cohort in 0..6 {
            session
                .begin_structured_cost_group_cohort_v2(cohort)
                .unwrap();
            let route_index = cohort / 3;
            for report in
                complete_native_request_with_route(&mut session, routes[route_index]).await
            {
                let stages = Arc::clone(report.host_stages.as_ref().unwrap());
                let numeric = &stages
                    .actual_shape
                    .as_ref()
                    .unwrap()
                    .numeric_features
                    .as_ref()
                    .unwrap()
                    .rows[0];
                let member = matches!(stages.rows[0].actual_work, HostStageWork::Decode { .. })
                    && (2..=4).contains(&numeric.generated_tokens_before);
                if member {
                    assert_eq!(
                        report.structured_cost_input_v2().unwrap().owner(),
                        inputs[route_index].owner()
                    );
                }
                let fifo = report
                    .host_stage_queue
                    .as_ref()
                    .unwrap()
                    .accepted_ordinal
                    .unwrap();
                assert!(originals
                    .insert(fifo, (phase, cohort, stages, member.then_some(route_index)))
                    .is_none());
            }
            session.end_structured_cost_group_cohort_v2().unwrap();
        }
        for progress in session.structured_cost_group_progress_v2().unwrap() {
            assert!(progress.failure.is_none(), "{:?}", progress.failure);
            assert_eq!(progress.reserved_members[phase], 9);
            assert_eq!(progress.completed_members[phase], 9);
            assert_eq!(progress.failed_members[phase], 0);
        }
        assert!(session
            .structured_cost_group_coverage_v2()
            .unwrap()
            .iter()
            .all(|c| c.complete()));
        let cut = session
            .freeze_structured_cost_group_phase_v2()
            .await
            .unwrap();
        assert_eq!(cut.len(), 2);
        assert_eq!(cut[0].frozen_at_ns, cut[1].frozen_at_ns);
        assert_eq!(cut[0].accepted_fifo_cutoff, cut[1].accepted_fifo_cutoff);
        assert_eq!(cut[0].member_cutoff, (phase as u64 + 1) * 9);
        assert_eq!(cut[1].member_cutoff, cut[0].member_cutoff);
        freezes.push(cut);
    }
    assert_eq!(originals.len(), 108);
    let group = session.finish_structured_cost_group_v2().await.unwrap();
    assert!(group.failure.is_none(), "{:?}", group.failure);
    assert_eq!(group.fingerprint, fingerprint);
    assert_eq!(group.children.len(), 2);
    let mut closing = None;
    let mut exports = Vec::new();
    for (child, source) in group.children.iter().enumerate() {
        assert_eq!(source.phase, StructuredCapturePhase::Qualified);
        assert!(source.failure.is_none());
        assert_eq!(source.scope_failures, 0);
        assert_eq!(source.scope_members, 27);
        let records: Vec<serde_json::Value> = std::fs::read_to_string(&source.source_path)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(records[0]["record"]["initial_fifo_cutoff"], initial);
        let mut seen = std::collections::BTreeSet::new();
        let mut members = [0; 3];
        let mut outside = [0; 3];
        let mut member_terminals = [0; 3];
        let mut terminal = 0;
        for (ordinal, wrapper) in records.iter().enumerate() {
            assert_eq!(wrapper["source_record_ordinal"], ordinal as u64 + 1);
            let row = &wrapper["record"];
            if row["kind"] == "request_completed" {
                terminal += 1;
                assert_eq!(row["request"]["generated_tokens"], 5);
            }
            if row["kind"] != "completed" {
                continue;
            }
            let fifo = row["queue"]["accepted_ordinal"].as_u64().unwrap();
            assert!(seen.insert(fifo));
            let (phase, cohort, actual, member) = &originals[&fifo];
            let selected = *member == Some(child);
            assert_eq!(!row["member"].is_null(), selected);
            assert_eq!(row["phase"], ["fit", "residual", "qualification"][*phase]);
            assert_eq!(row["cohort"], *cohort);
            assert!(row["conversion_error"].is_null());
            let stages = if selected {
                members[*phase] += 1;
                if actual.rows[0].terminal.is_some() {
                    member_terminals[*phase] += 1;
                    assert_eq!(
                        actual.rows[0].terminal.as_ref().unwrap().finish_reason,
                        ferrum_types::FinishReason::Length
                    );
                }
                &row["host_stages"]
            } else {
                outside[*phase] += 1;
                &row["outside_settlement"]
            };
            for (field, value) in [
                (
                    "prepare_started_at_ns",
                    actual.prepare_started_at_ns.unwrap(),
                ),
                (
                    "executor_returned_at_ns",
                    actual.executor_returned_at_ns.unwrap(),
                ),
                ("finalized_at_ns", actual.finalized_at_ns.unwrap()),
                ("full_wall_ns", actual.full_wall_ns.unwrap()),
            ] {
                assert_eq!(stages[field], value);
            }
            assert_eq!(stages["call_id"], actual.call_id);
            assert_eq!(
                stages["rows"][0]["request_id"],
                actual.rows[0].request_id.to_string()
            );
            assert_eq!(
                stages["rows"][0]["terminal"],
                serde_json::to_value(&actual.rows[0].terminal).unwrap()
            );
            let lower = if *phase == 0 {
                records[0]["record"]["opened_at_ns"].as_u64().unwrap()
            } else {
                freezes[*phase - 1][child].frozen_at_ns
            };
            assert!(actual.prepare_started_at_ns.unwrap() >= lower);
            assert!(actual.finalized_at_ns.unwrap() <= freezes[*phase][child].frozen_at_ns);
            if selected {
                assert_eq!(
                    row["numeric"]["observed_at_ns"],
                    actual.finalized_at_ns.unwrap()
                );
            }
        }
        assert_eq!(seen, originals.keys().copied().collect());
        assert_eq!(members, [9; 3]);
        assert_eq!(member_terminals, [3; 3]);
        assert_eq!(outside, [27; 3]);
        assert_eq!(terminal, 18);
        let end = records.last().unwrap()["record"]["closing"].clone();
        if let Some(before) = &closing {
            assert_eq!(before, &end);
        } else {
            closing = Some(end);
        }
        let export = export_structured_profile_v10(
            &source.source_path,
            source.source_sha256,
            &files.0.join(format!("profile-{child}.json")),
            1_000_000,
            &CostProfileLoadLimits::default(),
        )
        .unwrap();
        assert_eq!(
            export.parameters_sha256,
            source.model.as_ref().unwrap().parameters_signature()
        );
        exports.push(export);
    }
    let catalog = serde_json::json!({"artifact_type":"ferrum.structured-v2-catalog","schema_version":1,"model_revision":MODEL_REVISION_V2,
        "fingerprint":ProfileFingerprint::from(&group.fingerprint),"children":exports.iter().zip(&group.children).map(|(e,s)| {
            let model=s.model.as_ref().unwrap();serde_json::json!({"profile_path":e.path,"profile_sha256":e.file_sha256,"owner":model.owner(),"domain_signature":model.domain_signature()})
        }).collect::<Vec<_>>()});
    let path = files.0.join("catalog.json");
    std::fs::write(&path, serde_json::to_vec(&catalog).unwrap()).unwrap();
    let inner = session.test_engine_inner();
    let original_runtime = inner.cost_runtime.as_ref().unwrap();
    let before_version = original_runtime.snapshot().map(|s| s.model_version());
    let inspected = session.inspect_structured_cost_profile_v2(&path).unwrap();
    assert_eq!(
        original_runtime.snapshot().map(|s| s.model_version()),
        before_version,
        "inspection must not install a model"
    );
    let receipt = inspected.structured_whole_wave_v2.as_ref().unwrap();
    assert_eq!(receipt.child_count, 2);
    assert_eq!(
        receipt.artifact_kind,
        SloStructuredArtifactKindV2::CatalogV1
    );
    for child in &receipt.children {
        assert_eq!(child.reserved_members, 27);
        for (phase, cut) in child.phases.iter().enumerate() {
            assert_eq!(cut.frozen_at_ns, freezes[phase][0].frozen_at_ns);
        }
    }
    let runtime =
        EngineCostRuntime::new(original_runtime.identity.clone(), &config, Some(&path)).unwrap();
    let snapshot = runtime.snapshot().unwrap();
    for input in inputs {
        let query = StructuredQueryV2::exact(input);
        let now = runtime.clock.now_ns().unwrap();
        assert!(
            snapshot.audit_structured_query_v2(&query, now).is_ok(),
            "each original owner must yield a Known imported query"
        );
        assert!(
            matches!(
                snapshot.audit_structured_query_v2(
                    &query,
                    now.checked_add(settings.max_sample_age_ns).unwrap()
                ),
                Err(StructuredUnknownV2::Stale)
            ),
            "catalog import must not refresh either child's original sample TTL"
        );
    }
    runtime.shutdown().await.unwrap();
    drop(inner);
    session.shutdown().await.unwrap();
    drop(model_directory);
}
