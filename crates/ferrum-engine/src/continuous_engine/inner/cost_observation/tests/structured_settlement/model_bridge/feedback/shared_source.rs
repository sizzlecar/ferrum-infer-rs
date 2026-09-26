//! Real private recorder -> common source4 -> profile11 -> product runtime.
use super::*;
use crate::continuous_engine::inner::cost_observation::{
    StructuredCalibrationGroupOptionsV2, StructuredCalibrationGroupV2,
};

#[test]
fn shared_source4_real_receipts_roundtrip_both_independent_owners_into_product_runtime() {
    run_shared_case(false);
}
#[test]
fn shared_source4_failed_private_settlement_keeps_original_receipt_and_child_failure() {
    run_shared_case(true);
}
fn run_shared_case(fail_after_receipt: bool) {
    struct Directory(PathBuf);
    impl Directory {
        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    let directory =
        Directory(std::env::temp_dir().join(format!("ferrum-source4-{}", uuid::Uuid::new_v4())));
    fs::create_dir(directory.path()).unwrap();
    let path = directory
        .path()
        .canonicalize()
        .unwrap()
        .join("shared.jsonl");
    let algorithms = ["fixture.shared.a", "fixture.shared.b"];
    let mut queries = Vec::new();
    let children = algorithms.iter().map(|&algorithm| {
        let w = wave(algorithm);
        let input = StructuredInputV2::from_actual(&w.prepared.exact, &w.prepared.selected, &w.prepared.recipe).unwrap();
        let owner = input.owner().clone();
        queries.push(StructuredQueryV2::exact(input));
        StructuredCalibrationOptionsV2 {
            observations_path: path.clone(), protocol_sha256: [81; 32],
            scope: StructuredScopeV2 { owner: owner.clone(), coverage: StructuredCoverageV2 {
                pending_eligible_positions: vec![], authorized_pending_constraints: vec![HostPendingConstraintV2::AnySubset],
                pending_counts: vec![0], length_counts: vec![1], pending_positions: vec![], length_positions: vec![0], joint_counts: vec![(0,1)],
            }},
            membership_rule: MembershipRuleV2 { owner, windows: vec![FrontierWindowV2 { rows: vec![RowWindowV2 {
                generated_before: ClosedRangeV2 { minimum: 0, maximum: 0 },
                remaining_output: ClosedRangeV2 { minimum: 1, maximum: 1 },
                context_before: ClosedRangeV2 { minimum: 7, maximum: 7 },
                work: WorkWindowV2::Decode { kv_tokens: ClosedRangeV2 { minimum: 7, maximum: 7 } },
            }]}]},
            cohort_plan: CohortPlanV2 { phases: std::array::from_fn(|_| (0..16).map(|repetition| CohortV2 {
                manifest_case: repetition, repetition: 0, requests: vec![CohortRequestV2 { manifest_prompt: 0, maximum_output: 1 }],
            }).collect()) },
            cohort_manifest_payload: serde_json::json!({"fixture":"two real recorder owners, complete independent phases"}),
            settings: StructuredSettingsV2 { max_phase_samples: 8, max_axes: 256, max_rank: 8, static_margin_ns: 1, ..Default::default() },
            phase_members: [8; 3], maximum_offered_waves: NonZeroUsize::new(48).unwrap(),
            maximum_file_bytes: NonZeroU64::new(16 * 1024 * 1024).unwrap(),
        }
    }).collect();
    let clock = Arc::new(VirtualClock(AtomicU64::new(1)));
    let queue = sink(64, 8192);
    let ids = EngineCostIds::default();
    let mut group = StructuredCalibrationGroupV2::new(
        StructuredCalibrationGroupOptionsV2 {
            shared_source: Some(path.clone()),
            children,
            limits: Default::default(),
        },
        session().fingerprint().clone(),
        clock.clone(),
        0,
    )
    .unwrap();
    let mut cutoff = 0;
    for _phase in 0..3 {
        for cohort in 0..16 {
            group.begin_cohort(cohort).unwrap();
            let w = wave(algorithms[cohort % 2]);
            let row = &w.actual.rows[0];
            group.admitted(row.request_id.clone(), 1).unwrap();
            let frontier = CalibrationFrontier {
                session: Arc::new(()),
                request_id: row.request_id.clone(),
                owner: NonZeroU64::new(row.owner_incarnation).unwrap(),
                generation: NonZeroU64::new(row.work_generation).unwrap(),
                request_evidence: CalibrationRequestEvidence {
                    original_input_tokens: 7,
                    original_input_tokens_sha256: [31; 32],
                },
                generated: 0,
                prefill: None,
                kv_tokens: 7,
            };
            group.offer(&[frontier.decode_work().unwrap()]).unwrap();
            group.reserve_prepared(w.prepared).unwrap();
            let capture = group.pending_capture().unwrap();
            record(
                &ids,
                &queue,
                &clock,
                w.actual,
                w.host,
                Some(capture.clone()),
                0,
            );
            if fail_after_receipt {
                assert!(group.complete(&capture, false).is_err());
                let cutoff = capture
                    .host_stage_queue()
                    .unwrap()
                    .accepted_ordinal
                    .unwrap();
                let failed = group.finish(cutoff).unwrap();
                assert!(failed.failure.is_some());
                assert!(failed.children.iter().all(|c| c.model.is_none()));
                let raw = fs::read_to_string(&path).unwrap();
                let records = raw
                    .lines()
                    .map(|line| {
                        serde_json::from_str::<serde_json::Value>(line).unwrap()["record"].clone()
                    })
                    .collect::<Vec<_>>();
                let completed = records
                    .iter()
                    .filter(|r| r["kind"] == "completed")
                    .collect::<Vec<_>>();
                assert_eq!(completed.len(), 1);
                assert!(!completed[0]["host_stages"].is_null());
                assert!(!completed[0]["conversion_error"].is_null());
                assert_eq!(completed[0]["reconciled"], false);
                assert!(records.iter().any(|r| r["kind"] == "phase_failed"
                    && r["child_failures"]
                        .as_array()
                        .is_some_and(|c| c.len() == algorithms.len())));
                assert!(!records.iter().any(|r| r["kind"] == "phase_freeze"));
                return;
            }
            group.complete(&capture, true).unwrap();
            cutoff = capture
                .host_stage_queue()
                .unwrap()
                .accepted_ordinal
                .unwrap();
            group.end_cohort().unwrap();
        }
        clock.set(clock.now_ns().unwrap() + 1);
        let receipts = group.freeze(cutoff).unwrap();
        assert_eq!(
            receipts[0].source_prefix_sha256,
            receipts[1].source_prefix_sha256
        );
        assert_eq!(
            receipts[0].source_prefix_bytes,
            receipts[1].source_prefix_bytes
        );
    }
    let artifact = group.finish(cutoff).unwrap();
    assert!(artifact.failure.is_none());
    assert_eq!(artifact.children.len(), 2);
    assert!(artifact
        .children
        .iter()
        .all(|c| c.source_path == path && c.model.is_some() && c.scope_members == 24));
    let raw = fs::read_to_string(&path).unwrap();
    let events = raw
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap()["record"].clone())
        .collect::<Vec<_>>();
    assert_eq!(events[0]["schema_version"], 4);
    assert_eq!(
        events.iter().filter(|r| r["kind"] == "reserved").count(),
        cutoff as usize
    );
    assert_eq!(
        events.iter().filter(|r| r["kind"] == "completed").count(),
        cutoff as usize
    );
    let mut config = SloCostObservationConfig::structured_whole_wave_v2();
    config.profile_import.declared_local_clock_max_error_ns = Some(1_000_000_000);
    config.profile_import.max_clock_error_ns = 2_000_000_000;
    let import = &config.profile_import;
    let limits = file::CostProfileLoadLimits {
        max_file_bytes: import.max_file_bytes,
        max_samples: import.max_samples,
        max_total_shape_rows: import.max_total_shape_rows,
        max_source_field_bytes: import.max_source_field_bytes,
        max_profile_age_ns: import.max_profile_age_ns,
        max_clock_error_ns: import.max_clock_error_ns,
    };
    let catalog = directory.path().join("profile11.json");
    let exported = file::export_structured_profile_v11(
        &path,
        artifact.children[0].source_sha256,
        &catalog,
        &[1_000_000_000; 2],
        &limits,
    )
    .unwrap();
    for (child, original) in exported.children.iter().zip(&artifact.children) {
        assert_eq!(
            child.parameters_sha256,
            original.model.as_ref().unwrap().parameters_signature()
        );
    }
    let load = profile::read_load_clock(clock.as_ref(), &config.profile_import).unwrap();
    let runtime = EngineCostRuntime::build_with_profile(
        identity(),
        clock.clone(),
        &config,
        false,
        Some(&catalog),
        Some(load),
    )
    .unwrap();
    let snapshot = runtime.snapshot().unwrap();
    for query in queries {
        assert!(snapshot
            .audit_structured_query_v2(&query, clock.now_ns().unwrap())
            .is_ok());
    }
    let receipt = runtime.profile_receipt().unwrap();
    let v2 = receipt.structured_whole_wave_v2.as_ref().unwrap();
    assert_eq!(
        v2.artifact_kind,
        ferrum_types::SloStructuredArtifactKindV2::SharedCatalogV11
    );
    assert_eq!(
        v2.total_imported_bytes,
        exported.file_bytes + exported.source_bytes
    );
    assert_eq!(v2.total_shape_rows, cutoff);
    assert_eq!(receipt.offered_samples, cutoff as usize);
}
