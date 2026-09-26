//! CPU recorder -> original source3 -> product profile10 loader -> live worker.
//! Synthetic clock intervals exercise contracts; they are not backend timings.
use super::*;
use crate::continuous_engine::inner::{
    calibration::{CalibrationFrontier, CalibrationRequestEvidence, CalibrationWork},
    cost_observation::{
        profile, profile_export::structured_v2::StructuredCalibrationCollectorV2,
        PreparedRowBindingV2, PreparedStructuredFactsV2, StructuredCalibrationOptionsV2,
    },
};
use ferrum_interfaces::execution_cost::HostPendingConstraintV2;
use ferrum_scheduler::implementations::continuous::{
    cost_model::structured_v2::{windows::*, *},
    cost_profile::{self as file, export_structured_profile_v10, ProfileFingerprint},
};
use ferrum_types::{
    SloCostObservationConfig, SloSelectedFeedbackSettingsV1, SloSelectedFeedbackStorageV1,
    SloStructuredFeedbackPolicy,
};
use std::{fs, path::PathBuf};

struct Wave {
    actual: ActualWaveShape,
    host: HostCostFeaturesV1,
    prepared: PreparedStructuredFactsV2,
}

fn wave(algorithm: &'static str) -> Wave {
    let (_, mut hosts) = selected_shape(&[3]);
    let mut host = hosts.remove(0);
    host.state.generated_tokens_before = 0;
    host.state.maximum_output_tokens = 1;
    host.state.sampling_history_tokens = 0;
    let mut selected = SelectedCommandCostBuilderV1::new_with_algorithm_work(1);
    selected
        .kernel_with_replay_geometry(
            SelectedAlgorithmClassV1::new(algorithm, 1, [1; 32], [2; 32]).unwrap(),
            KernelNumericWorkV1 {
                logical_units: 32,
                padded_units: 32,
                inner_units_per_logical_unit: 32,
                grid: [1, 1, 1],
                scratch_bytes: 0,
                staged_weight_bytes: 0,
            },
            KernelReplayGeometryV1 {
                block: [32, 1, 1],
                dynamic_shared_bytes: 0,
                fixed_parameters: &[32],
            },
        )
        .unwrap();
    let selected = selected.finish().unwrap();
    let mut builder =
        CanonicalWaveCostBuilder::new_with_structured_statistics(0, CostProductOutput::FullLogits);
    builder
        .physical_command(CostPhysicalCommand {
            native_op_id: "fixture.feedback",
            command_index: 0,
            node_index: None,
            command_phase: DeviceCommandPhase::Compute,
            provider: None,
            path: CostCommandPath::Eager,
            participant_start: 0,
            participant_count: 1,
            token_count: 1,
            batching_form: "packed",
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            reusable_graph_node_count: None,
            statistical_evidence: Some(&selected),
        })
        .unwrap();
    builder
        .core_readback_route(CoreReadbackRoute::HostSynchronized)
        .unwrap();
    builder
        .row(CanonicalCostRow {
            work: ActualRowWork::Decode { kv_tokens: 7 },
            host_policy_signature: [6; 32],
            mask_upload_required: false,
            host_features: Some(host),
            output: CostRowOutput::Decode {
                requires_full_logits: true,
                repetition_tokens: 0,
                repetition_penalty_bits: 1f32.to_bits(),
            },
        })
        .unwrap();
    let built = builder
        .finish_with_captured_structure(
            ActualWaveKind::Decode,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Disabled,
            ActualWaveRowOrder::Ordered,
            64,
        )
        .unwrap();
    let selected = built.statistical.unwrap();
    let recipe = Arc::clone(selected.structured_capture().unwrap().unwrap());
    let owner = StructuredOwnerFactsV2::from_prepared(&built.exact, &selected, &recipe).unwrap();
    let mut actual = shape(&[ActualRowWork::Decode { kv_tokens: 7 }]);
    actual.provider_signature = built.exact.provider_signature;
    actual.output_policy_signature = built.exact.output_policy_signature;
    actual.numeric_features = built.exact.numeric_features.clone();
    actual.host_content_features = built.exact.host_content_features;
    actual.row_multiset_features = built.exact.row_multiset_features.clone();
    actual.statistical_evidence = Some(selected.clone());
    let row = &actual.rows[0];
    let prepared = PreparedStructuredFactsV2 {
        exact: built.exact,
        selected,
        recipe,
        owner,
        rows: vec![PreparedRowBindingV2 {
            request_id: row.request_id.clone(),
            owner_incarnation: row.owner_incarnation,
            work_generation: row.work_generation,
            frontier: PreparedRowFactsV2 {
                physical_position: 0,
                work: PreparedWorkV2::Decode { kv_tokens: 7 },
                generated_before: 0,
                maximum_output: 1,
                context_before: 7,
            },
        }],
    };
    prepared.validate().unwrap();
    Wave {
        actual,
        host,
        prepared,
    }
}

fn record(
    ids: &EngineCostIds,
    queue: &Arc<BoundedCostSampleSink>,
    clock: &Arc<VirtualClock>,
    actual: ActualWaveShape,
    host: HostCostFeaturesV1,
    capture: Option<Arc<CostCalibrationCapture>>,
    extra_wall: u64,
) -> Arc<HostStageEvidenceV1> {
    record_with_hook(
        ids,
        queue,
        clock,
        actual,
        host,
        capture,
        extra_wall,
        None,
        |_| {},
    )
}

fn record_with_hook(
    ids: &EngineCostIds,
    queue: &Arc<BoundedCostSampleSink>,
    clock: &Arc<VirtualClock>,
    actual: ActualWaveShape,
    host: HostCostFeaturesV1,
    capture: Option<Arc<CostCalibrationCapture>>,
    extra_wall: u64,
    terminal_reason: Option<ferrum_types::FinishReason>,
    hook: impl FnOnce(&mut EngineCostCall),
) -> Arc<HostStageEvidenceV1> {
    let at = clock.now_ns().unwrap() + 100;
    clock.set(at + 2);
    let row = &actual.rows[0];
    let mut call = EngineCostCall::begin(
        ids,
        clock.clone(),
        queue.clone(),
        EngineCostCallSpec {
            identity: identity(),
            participants: vec![CostObservationParticipant {
                request_id: row.request_id.clone(),
                owner_incarnation: row.owner_incarnation,
                work_generation: row.work_generation,
                input_index: row.input_index,
                output_policy_signature: Some([6; 32]),
                host_features: Some(host),
            }],
            prepare_started_at_ns: Some(at + 1),
            boundary: WaveObservationBoundary::IsolatedPreparationToCommit,
            recorder_limits: CostRecorderLimits {
                max_waves: 4,
                max_rows_per_wave: 8,
                max_retained_rows: 128,
            },
        },
    )
    .unwrap()
    .with_structured_capture(true);
    if let Some(capture) = &capture {
        call.attach_calibration_capture(capture.clone());
    }
    hook(&mut call);
    let mut context = call.context().unwrap();
    context.physical_wave(Ok(actual.clone()), Some(at + 3));
    clock.set(at + 6);
    context.terminal(ActualWaveOutcome::Completed, None);
    context.finish_call(ObservedCallOutcome::Completed);
    drop(context);
    let settled_at = at + 10 + extra_wall;
    clock.set(settled_at);
    call.begin_host_row(&row.request_id);
    clock.set(settled_at + 1);
    let commit = HostCommitEvidence {
        request_id: row.request_id.clone(),
        owner_incarnation: row.owner_incarnation,
        work_generation: row.work_generation,
        input_index: row.input_index,
        outcome: HostCommitOutcome::Committed(HostCommittedWork::Decode {
            kv_tokens_before: 7,
            kv_tokens_after: 8,
            generated_tokens_before: 0,
            generated_tokens_after: 1,
        }),
        committed_at_ns: Some(settled_at + 1),
    };
    call.note_host_token_commit(&commit);
    let mut pending = call.host_publication(&commit, true, true).unwrap();
    clock.set(settled_at + 2);
    pending.terminal_handed_off();
    clock.set(settled_at + 3);
    let mut terminal = terminal();
    if let Some(reason) = terminal_reason {
        terminal.finish_reason = reason;
    }
    terminal.generated_tokens = 1;
    terminal.through_output_ordinal = 1;
    call.record_settled(pending.settle(owner(row), terminal));
    let expected_rejection = call.rejection.unwrap_or(CostCallRejection::Composite);
    call.reject(CostCallRejection::Composite);
    clock.set(settled_at + 10);
    let stages = call.make_host_stages().unwrap();
    assert_eq!(
        call.finish(),
        CostCallDisposition::Rejected(expected_rejection)
    );
    capture.map_or(stages, |capture| capture.host_stages().unwrap())
}

struct Fixture {
    directory: PathBuf,
    path: PathBuf,
    config: SloCostObservationConfig,
    clock: Arc<VirtualClock>,
    queries: Vec<StructuredQueryV2>,
    originals: Vec<(PathBuf, Vec<u8>)>,
}
impl Fixture {
    fn new() -> Self {
        let directory = std::env::temp_dir().join(format!(
            "ferrum-structured-feedback-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir(&directory).unwrap();
        let mut config = SloCostObservationConfig::structured_whole_wave_v2();
        config.profile_import.declared_local_clock_max_error_ns = Some(1_000_000_000);
        config.profile_import.max_clock_error_ns = 2_000_000_000;
        config.structured_feedback = SloStructuredFeedbackPolicy::RetrospectiveOwnerMarginV1 {
            policy: SloSelectedFeedbackSettingsV1 {
                window_samples: NonZeroUsize::new(2).unwrap(),
                minimum_underestimates: NonZeroUsize::new(2).unwrap(),
                minimum_consecutive_underestimates: NonZeroUsize::new(2).unwrap(),
                trigger_excess_ns: NonZeroU64::new(2).unwrap(),
                correction_padding_ns: 1,
                maximum_family_margin_ns: NonZeroU64::new(1000).unwrap(),
                maximum_consumption_lag_ns: NonZeroU64::new(10_000).unwrap(),
                maximum_uncomparable_observations: 0,
                maximum_failed_or_partial: 0,
                maximum_queue_drops: 0,
                maximum_state_bytes: NonZeroUsize::new(64 * 1024).unwrap(),
            },
            storage: SloSelectedFeedbackStorageV1::CreateNew {
                path: directory.join("feedback.json"),
            },
        };
        let import = &config.profile_import;
        let limits = file::CostProfileLoadLimits {
            max_file_bytes: import.max_file_bytes,
            max_samples: import.max_samples,
            max_total_shape_rows: import.max_total_shape_rows,
            max_source_field_bytes: import.max_source_field_bytes,
            max_profile_age_ns: import.max_profile_age_ns,
            max_clock_error_ns: import.max_clock_error_ns,
        };
        let mut children = Vec::new();
        let mut queries = Vec::new();
        let mut originals = Vec::new();
        for (index, algorithm) in ["fixture.feedback.a", "fixture.feedback.b"]
            .into_iter()
            .enumerate()
        {
            let probe = wave(algorithm);
            let input = StructuredInputV2::from_actual(
                &probe.prepared.exact,
                &probe.prepared.selected,
                &probe.prepared.recipe,
            )
            .unwrap();
            let owner = input.owner().clone();
            queries.push(StructuredQueryV2::exact(input));
            let clock = Arc::new(VirtualClock(AtomicU64::new(1)));
            let queue = sink(64, 8192);
            let ids = EngineCostIds::default();
            let options = StructuredCalibrationOptionsV2 {
                observations_path: directory.join(format!("source-{index}.jsonl")),
                protocol_sha256: [91; 32],
                scope: StructuredScopeV2 {
                    owner: owner.clone(),
                    coverage: StructuredCoverageV2 {
                        pending_eligible_positions: vec![],
                        authorized_pending_constraints: vec![HostPendingConstraintV2::AnySubset],
                        pending_counts: vec![0],
                        length_counts: vec![1],
                        pending_positions: vec![],
                        length_positions: vec![0],
                        joint_counts: vec![(0, 1)],
                    },
                },
                membership_rule: MembershipRuleV2 {
                    owner,
                    windows: vec![FrontierWindowV2 {
                        rows: vec![RowWindowV2 {
                            generated_before: ClosedRangeV2 {
                                minimum: 0,
                                maximum: 0,
                            },
                            remaining_output: ClosedRangeV2 {
                                minimum: 1,
                                maximum: 1,
                            },
                            context_before: ClosedRangeV2 {
                                minimum: 7,
                                maximum: 7,
                            },
                            work: WorkWindowV2::Decode {
                                kv_tokens: ClosedRangeV2 {
                                    minimum: 7,
                                    maximum: 7,
                                },
                            },
                        }],
                    }],
                },
                cohort_plan: CohortPlanV2 {
                    phases: std::array::from_fn(|_| {
                        (0..8)
                            .map(|repetition| CohortV2 {
                                manifest_case: 0,
                                repetition,
                                requests: vec![CohortRequestV2 {
                                    manifest_prompt: 0,
                                    maximum_output: 1,
                                }],
                            })
                            .collect()
                    }),
                },
                cohort_manifest_payload: serde_json::json!({"fixture":"CPU original recorder complete one-output cohorts","algorithm":algorithm}),
                settings: StructuredSettingsV2 {
                    static_margin_ns: 1,
                    ..Default::default()
                },
                phase_members: [8; 3],
                maximum_offered_waves: NonZeroUsize::new(24).unwrap(),
                maximum_file_bytes: NonZeroU64::new(16 * 1024 * 1024).unwrap(),
            };
            let fingerprint = session().fingerprint().clone();
            let mut collector = StructuredCalibrationCollectorV2::new(
                options,
                fingerprint.clone(),
                clock.clone(),
                0,
            )
            .unwrap();
            let mut cutoff = 0;
            for _phase in 0..3 {
                for cohort in 0..8 {
                    collector.begin_cohort(cohort).unwrap();
                    let w = wave(algorithm);
                    let row = &w.actual.rows[0];
                    collector.admitted(row.request_id.clone(), 1).unwrap();
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
                    let work: CalibrationWork = frontier.decode_work().unwrap();
                    collector.offer(&[work]).unwrap();
                    collector.reserve_prepared(w.prepared).unwrap();
                    let capture = collector.pending_capture().unwrap();
                    record(
                        &ids,
                        &queue,
                        &clock,
                        w.actual,
                        w.host,
                        Some(capture.clone()),
                        0,
                    );
                    collector.complete(&capture, true).unwrap();
                    cutoff = capture
                        .host_stage_queue()
                        .unwrap()
                        .accepted_ordinal
                        .unwrap();
                    collector.end_cohort().unwrap();
                }
                clock.set(clock.now_ns().unwrap() + 1);
                collector.freeze(cutoff).unwrap();
            }
            let artifact = collector.finish(cutoff).unwrap();
            assert!(artifact.failure.is_none());
            assert!(artifact.model.is_some());
            let exported = export_structured_profile_v10(
                &artifact.source_path,
                artifact.source_sha256,
                &directory.join(format!("profile-{index}.json")),
                1_000_000_000,
                &limits,
            )
            .unwrap();
            children.push(serde_json::json!({"profile_path":exported.path,"profile_sha256":exported.file_sha256,
                "owner":queries[index].owner(),"domain_signature":queries[index].domain_signature()}));
            for path in [artifact.source_path, exported.path] {
                originals.push((path.clone(), fs::read(path).unwrap()));
            }
        }
        let path = directory.join("catalog.json");
        fs::write(&path, serde_json::to_vec(&serde_json::json!({
            "artifact_type":"ferrum.structured-v2-catalog","schema_version":1,"model_revision":MODEL_REVISION_V2,
            "fingerprint":ProfileFingerprint::from(session().fingerprint()),"children":children,
        })).unwrap()).unwrap();
        originals.push((path.clone(), fs::read(&path).unwrap()));
        Self {
            directory,
            path,
            config,
            clock: Arc::new(VirtualClock(AtomicU64::new(10_000))),
            queries,
            originals,
        }
    }
    fn build(&self) -> EngineCostRuntime {
        let load =
            profile::read_load_clock(self.clock.as_ref(), &self.config.profile_import).unwrap();
        EngineCostRuntime::build_with_profile(
            identity(),
            self.clock.clone(),
            &self.config,
            false,
            Some(&self.path),
            Some(load),
        )
        .unwrap()
    }
    fn observe(&self, runtime: &EngineCostRuntime, extra_wall: u64) {
        let w = wave("fixture.feedback.a");
        record(
            &runtime.ids,
            &runtime.sink,
            &self.clock,
            w.actual,
            w.host,
            None,
            extra_wall,
        );
        runtime.consume_samples();
    }
    fn unchanged(&self) {
        for (path, bytes) in &self.originals {
            assert_eq!(&fs::read(path).unwrap(), bytes);
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

#[tokio::test]
async fn structured_feedback_real_source_preserves_other_owner_base_and_original_ttl() {
    let f = Fixture::new();
    let runtime = f.build();
    let old = runtime.snapshot().unwrap();
    let at = f.clock.now_ns().unwrap();
    let a = old.audit_structured_query_v2(&f.queries[0], at).unwrap();
    let b = old.audit_structured_query_v2(&f.queries[1], at).unwrap();
    f.observe(&runtime, 30);
    assert_eq!(
        runtime.snapshot().unwrap().model_version(),
        old.model_version()
    );
    f.observe(&runtime, 30);
    let current = runtime.snapshot().unwrap();
    let now = f.clock.now_ns().unwrap();
    assert!(current.model_version() > old.model_version());
    assert!(
        old.audit_structured_query_v2(&f.queries[1], now).is_err(),
        "one changed owner invalidates the entire old catalog epoch"
    );
    let updated = current
        .audit_structured_query_v2(&f.queries[0], now)
        .unwrap();
    let unchanged = current
        .audit_structured_query_v2(&f.queries[1], now)
        .unwrap();
    assert!(updated.planning_ns > a.planning_ns);
    assert_eq!(unchanged.planning_ns, b.planning_ns);
    assert_eq!(updated.typical_ns, a.typical_ns);
    assert_eq!(now + updated.valid_for_ns, at + a.valid_for_ns);
    for _ in 0..4 {
        f.observe(&runtime, 30);
    }
    let after = runtime.snapshot().unwrap();
    assert_eq!(
        after.model_version(),
        current.model_version(),
        "a correction is relative to the immutable base, not compounded"
    );
    assert!(after
        .audit_structured_query_v2(&f.queries[0], at + a.valid_for_ns + 1)
        .is_err());
    f.unchanged();
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn structured_feedback_correction_stops_drain_before_queued_samples_use_new_epoch() {
    let f = Fixture::new();
    let runtime = f.build();
    f.observe(&runtime, 30);
    let old = runtime.snapshot().unwrap();
    for _ in 0..3 {
        let w = wave("fixture.feedback.a");
        record(
            &runtime.ids,
            &runtime.sink,
            &f.clock,
            w.actual,
            w.host,
            None,
            30,
        );
    }
    runtime.consume_samples();
    let next = runtime.snapshot().unwrap();
    let first = runtime.audit_snapshot().structured_feedback.unwrap();
    assert_eq!(
        first.compared, 2,
        "the first queued row triggers the correction and ends this drain"
    );
    assert_eq!(first.corrections, 1);
    assert!(first.revoked.is_none());
    assert!(next.model_version() > old.model_version());
    assert!(old
        .audit_structured_query_v2(&f.queries[0], f.clock.now_ns().unwrap())
        .is_err());
    runtime.consume_samples();
    let second = runtime.audit_snapshot().structured_feedback.unwrap();
    assert_eq!(second.compared, 4);
    assert_eq!(second.corrections, 1);
    assert_eq!(second.uncomparable_observations, 0);
    assert!(second.revoked.is_none());
    assert_eq!(
        runtime.snapshot().unwrap().model_version(),
        next.model_version()
    );
    f.unchanged();
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn structured_feedback_complete_unknown_owner_cannot_acquire_qualification() {
    let f = Fixture::new();
    let runtime = f.build();
    let old = runtime.snapshot().unwrap();
    let w = wave("fixture.feedback.undeclared");
    let input =
        StructuredInputV2::from_actual(&w.prepared.exact, &w.prepared.selected, &w.prepared.recipe)
            .unwrap();
    let query = StructuredQueryV2::exact(input);
    assert!(old
        .audit_structured_query_v2(&query, f.clock.now_ns().unwrap())
        .is_err());
    record(
        &runtime.ids,
        &runtime.sink,
        &f.clock,
        w.actual,
        w.host,
        None,
        30,
    );
    runtime.consume_samples();
    let audit = runtime.audit_snapshot().structured_feedback.unwrap();
    assert_eq!(audit.compared, 0);
    assert_eq!(audit.corrections, 0);
    assert_eq!(audit.uncomparable_observations, 1);
    assert!(runtime.snapshot().is_none());
    assert!(old
        .audit_structured_query_v2(&query, f.clock.now_ns().unwrap())
        .is_err());
    f.unchanged();
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn structured_feedback_same_source_resumes_across_new_local_clock_without_ttl_renewal() {
    let mut f = Fixture::new();
    let runtime = f.build();
    f.observe(&runtime, 30);
    f.observe(&runtime, 30);
    let before = runtime.snapshot().unwrap();
    let at = f.clock.now_ns().unwrap();
    let prediction = before.audit_structured_query_v2(&f.queries[0], at).unwrap();
    let first_receipt = runtime.profile_receipt().unwrap().clone();
    runtime.shutdown().await.unwrap();
    assert!(before.audit_structured_query_v2(&f.queries[0], at).is_err());
    drop(runtime);
    let SloStructuredFeedbackPolicy::RetrospectiveOwnerMarginV1 { storage, .. } =
        &mut f.config.structured_feedback
    else {
        unreachable!()
    };
    *storage = SloSelectedFeedbackStorageV1::Resume {
        path: f.directory.join("feedback.json"),
    };
    f.clock = Arc::new(VirtualClock(AtomicU64::new(500_000)));
    let resumed = f.build();
    let current = resumed.snapshot().unwrap();
    let now = f.clock.now_ns().unwrap();
    let next = current
        .audit_structured_query_v2(&f.queries[0], now)
        .unwrap();
    assert_eq!(current.model_version(), before.model_version());
    assert_eq!(next.planning_ns, prediction.planning_ns);
    assert_eq!(next.typical_ns, prediction.typical_ns);
    assert!(next.valid_for_ns <= prediction.valid_for_ns);
    let old = first_receipt.structured_whole_wave_v2.as_ref().unwrap();
    let new = resumed
        .profile_receipt()
        .unwrap()
        .structured_whole_wave_v2
        .as_ref()
        .unwrap();
    for (old, new) in old.children.iter().zip(&new.children) {
        assert_eq!(old.profile_sha256, new.profile_sha256);
        assert_eq!(old.source_sha256, new.source_sha256);
        assert_eq!(old.parameters_sha256, new.parameters_sha256);
        assert_eq!(old.capture_identity_sha256, new.capture_identity_sha256);
        assert_eq!(old.protocol_sha256, new.protocol_sha256);
        assert!(new.oldest_imported_age_ns >= old.oldest_imported_age_ns);
        assert_ne!(
            old.source_monotonic_anchor_ns,
            new.source_monotonic_anchor_ns
        );
    }
    assert!(current
        .audit_structured_query_v2(&f.queries[0], now + next.valid_for_ns + 1)
        .is_err());
    f.unchanged();
    resumed.shutdown().await.unwrap();
}

#[tokio::test]
async fn structured_feedback_real_worker_rejects_missing_private_settlement_and_queue_loss() {
    let f = Fixture::new();
    let runtime = f.build();
    let old = runtime.snapshot().unwrap();
    let w = wave("fixture.feedback.a");
    let stages = record(
        &runtime.ids,
        &sink(8, 256),
        &f.clock,
        w.actual,
        w.host,
        None,
        30,
    );
    let mut invalid = stages.as_ref().clone();
    invalid.structured_evidence = None;
    runtime
        .sink
        .offer_evidence_numbered(entry(Arc::new(invalid)))
        .unwrap();
    runtime.consume_samples();
    assert!(runtime.snapshot().is_none());
    assert!(old
        .audit_structured_query_v2(&f.queries[0], f.clock.now_ns().unwrap())
        .is_err());
    f.unchanged();
    runtime.shutdown().await.unwrap();

    let mut f = Fixture::new();
    f.config.max_queued_samples = NonZeroUsize::new(1).unwrap();
    f.config.max_samples_per_update = NonZeroUsize::new(1).unwrap();
    let runtime = f.build();
    let old = runtime.snapshot().unwrap();
    for _ in 0..2 {
        let w = wave("fixture.feedback.a");
        record(
            &runtime.ids,
            &runtime.sink,
            &f.clock,
            w.actual,
            w.host,
            None,
            30,
        );
    }
    runtime.consume_samples();
    assert!(runtime.snapshot().is_none());
    assert!(old
        .audit_structured_query_v2(&f.queries[0], f.clock.now_ns().unwrap())
        .is_err());
    f.unchanged();
    runtime.shutdown().await.unwrap();
}

// Private prospective protocol is tested on this original recorder/source3/
// profile10 fixture; constructing diagnostic rows cannot qualify the model.
#[path = "prospective.rs"]
mod prospective;
