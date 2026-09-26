use super::*;
use ferrum_interfaces::execution_cost::{CoreReadbackRoute, HostPendingConstraintV2};
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::windows::*;
use std::sync::atomic::{AtomicU64, Ordering};

fn prepared_unsubmitted(id: RequestId) -> PreparedStructuredFactsV2 {
    use ferrum_interfaces::execution_cost::*;
    use ferrum_interfaces::vnext::DeviceCommandPhase;
    let mut selected = SelectedCommandCostBuilderV1::new_with_algorithm_work(1);
    selected
        .kernel(
            SelectedAlgorithmClassV1::new("fixture.unsubmitted", 1, [2; 32], [3; 32]).unwrap(),
            KernelNumericWorkV1 {
                logical_units: 32,
                padded_units: 32,
                inner_units_per_logical_unit: 8,
                grid: [1, 1, 1],
                scratch_bytes: 0,
                staged_weight_bytes: 0,
            },
        )
        .unwrap();
    let selected = selected.finish().unwrap();
    let mut builder =
        CanonicalWaveCostBuilder::new_with_structured_statistics(0, CostProductOutput::FullLogits);
    builder
        .physical_command(CostPhysicalCommand {
            native_op_id: "fixture.unsubmitted",
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
            host_features: Some(HostCostFeaturesV1 {
                policy: HostCostPolicyV2 {
                    empirical_content_domain: Some(HostContentDomainV1::PlainTextGreedyV1),
                    categorical_signature: [9; 32],
                    decoder_text_bytes_per_token: 4,
                    decoder_scratch_bytes_per_token: 8,
                    raw_token_bytes_bound: 4,
                },
                state: HostCostStateV1 {
                    generated_tokens_before: 0,
                    maximum_output_tokens: 10,
                    sampling_history_tokens: 0,
                    sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
                    pending_decoded_utf8: false,
                    completion_state_signature: satisfied_completion_cost_signature(),
                },
            }),
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
    PreparedStructuredFactsV2 {
        exact: built.exact,
        selected,
        recipe,
        owner,
        rows: vec![PreparedRowBindingV2 {
            request_id: id,
            owner_incarnation: 1,
            work_generation: 1,
            frontier: PreparedRowFactsV2 {
                physical_position: 0,
                work: PreparedWorkV2::Decode { kv_tokens: 7 },
                generated_before: 0,
                maximum_output: 10,
                context_before: 7,
            },
        }],
    }
}

struct Clock(AtomicU64);
impl CostObservationClock for Clock {
    fn now_ns(&self) -> Option<u64> {
        Some(self.0.load(Ordering::SeqCst)).filter(|n| *n != u64::MAX)
    }
}
struct Files(PathBuf);
impl Files {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("ferrum-group-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Files {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn fingerprint() -> model::ExecutionFingerprint {
    model::ExecutionFingerprint {
        model_weights: [1; 32],
        numerical_policy: [2; 32],
        device_runtime: [3; 32],
        execution_config: [4; 32],
    }
}
fn options(files: &Files) -> StructuredCalibrationGroupOptionsV2 {
    let children = (1..=2)
        .map(|algorithm| {
            // Pure configuration is not a receipt or qualification. These two
            // predeclared owners differ before any observation is available.
            let owner = StructuredOwnerKeyV2 {
                rows: 1,
                role: StructuredWaveRoleV2::OrdinaryDecode,
                product: StructuredProductV2::GreedyToken,
                readback: CoreReadbackRoute::HostSynchronized,
                provider_template: StructuredTemplateV2::Ordered([5; 32]),
                algorithm_domain: [algorithm; 32],
                installed_policy: [6; 32],
            };
            StructuredCalibrationOptionsV2 {
                observations_path: files.0.join(format!("child-{algorithm}.jsonl")),
                protocol_sha256: [9; 32],
                scope: StructuredScopeV2 {
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
                },
                membership_rule: MembershipRuleV2 {
                    owner,
                    windows: vec![FrontierWindowV2 {
                        rows: vec![RowWindowV2 {
                            generated_before: ClosedRangeV2::ALL,
                            remaining_output: ClosedRangeV2::ALL,
                            context_before: ClosedRangeV2::ALL,
                            work: WorkWindowV2::Decode {
                                kv_tokens: ClosedRangeV2::ALL,
                            },
                        }],
                    }],
                },
                cohort_plan: CohortPlanV2 {
                    phases: std::array::from_fn(|_| {
                        vec![CohortV2 {
                            manifest_case: 0,
                            repetition: 0,
                            requests: vec![CohortRequestV2 {
                                manifest_prompt: 0,
                                maximum_output: 10,
                            }],
                        }]
                    }),
                },
                cohort_manifest_payload: serde_json::json!({"requests":[{"max_tokens":10}]}),
                settings: StructuredSettingsV2 {
                    max_phase_samples: 8,
                    max_axes: 64,
                    max_rank: 64,
                    ..Default::default()
                },
                phase_members: [8; 3],
                maximum_offered_waves: NonZeroUsize::new(64).unwrap(),
                maximum_file_bytes: NonZeroU64::new(1024 * 1024).unwrap(),
            }
        })
        .collect();
    StructuredCalibrationGroupOptionsV2 {
        shared_source: None,
        children,
        limits: Default::default(),
    }
}
fn records(path: &std::path::Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn structured_group_declared_aggregate_limits_and_owner_cohorts_fail_before_writes() {
    let files = Files::new();
    let original = options(&files);
    original.validate().unwrap();
    for failure in 0..7 {
        let mut changed = original.clone();
        match failure {
            0 => changed.children[1].scope.owner = changed.children[0].scope.owner.clone(),
            1 => changed.children[1].cohort_plan.phases[1][0].requests[0].maximum_output = 11,
            2 => changed.limits.maximum_children = NonZeroUsize::MIN,
            3 => changed.limits.maximum_total_file_bytes = NonZeroU64::new(1024 * 1024).unwrap(),
            4 => changed.limits.maximum_retained_numeric_bytes = NonZeroUsize::MIN,
            5 => changed.limits.maximum_retained_coordinates = NonZeroUsize::MIN,
            _ => changed.children[1].observations_path = files.0.join(".").join("child-1.jsonl"),
        }
        assert!(
            StructuredCalibrationGroupV2::new(
                changed,
                fingerprint(),
                Arc::new(Clock(AtomicU64::new(7))),
                0
            )
            .is_err(),
            "case {failure}"
        );
        assert_eq!(std::fs::read_dir(&files.0).unwrap().count(), 0);
    }
}

#[test]
fn structured_group_closing_clock_failure_is_common_and_never_a_successful_group() {
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    struct ClosingClock {
        fail: AtomicBool,
        reads: AtomicUsize,
    }
    impl CostObservationClock for ClosingClock {
        fn now_ns(&self) -> Option<u64> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            (!self.fail.load(Ordering::SeqCst)).then_some(23)
        }
    }
    let files = Files::new();
    let clock = Arc::new(ClosingClock {
        fail: AtomicBool::new(false),
        reads: AtomicUsize::new(0),
    });
    let group = StructuredCalibrationGroupV2::new(options(&files), fingerprint(), clock.clone(), 0)
        .unwrap();
    clock.reads.store(0, Ordering::SeqCst);
    clock.fail.store(true, Ordering::SeqCst);
    let artifact = group.finish(0).unwrap();
    // The test deliberately does not manufacture qualified phase state. Its
    // independent regression is that every child uses ONE closing read; the
    // previous per-child finish implementation would read twice here.
    assert_eq!(clock.reads.load(Ordering::SeqCst), 1);
    assert!(artifact.failure.is_some());
    for child in artifact.children {
        assert_eq!(child.phase, StructuredCapturePhase::Failed);
        assert!(child.model.is_none());
        let raw = records(&child.source_path);
        assert!(raw.last().unwrap()["record"]["closing"].is_null());
    }
}

#[test]
fn structured_group_unsubmitted_attempts_remain_in_every_source_and_cannot_freeze() {
    let files = Files::new();
    let options = options(&files);
    let paths = options
        .children
        .iter()
        .map(|c| c.observations_path.clone())
        .collect::<Vec<_>>();
    let mut group = StructuredCalibrationGroupV2::new(
        options,
        fingerprint(),
        Arc::new(Clock(AtomicU64::new(7))),
        0,
    )
    .unwrap();
    group.begin_cohort(0).unwrap();
    group.admitted(RequestId::new(), 10).unwrap();
    // A preparation failure has an offered population slot and no physical
    // call/member. It remains in both raw sources, without inventing a wall.
    group.offer(&[]).unwrap();
    group
        .complete_unsubmitted("typed resource unavailable")
        .unwrap();
    assert!(group
        .progress()
        .iter()
        .all(|p| p.offered_attempts == 1 && p.reserved_members == [0; 3]));
    assert!(group.freeze(0).is_err()); // real request was not completed
    let artifact = group.finish(0).unwrap();
    assert!(artifact.failure.is_some());
    assert!(artifact
        .children
        .iter()
        .all(|c| c.model.is_none() && c.phase == StructuredCapturePhase::Failed));
    for path in paths {
        let raw = records(&path);
        assert!(raw.iter().any(
            |r| r["record"]["kind"] == "preparation_unavailable" && r["record"]["offered"] == 1
        ));
        assert!(!raw.iter().any(|r| r["record"]["kind"] == "phase_freeze"));
    }
}

#[test]
fn structured_group_reserved_unsubmitted_member_keeps_denominator_and_original_outside() {
    let files = Files::new();
    let id = RequestId::new();
    let prepared = prepared_unsubmitted(id.clone());
    let owner = prepared.validate().unwrap();
    let mut opts = options(&files);
    opts.children[0].scope.owner = owner.clone();
    opts.children[0].membership_rule.owner = owner;
    let paths = opts
        .children
        .iter()
        .map(|c| c.observations_path.clone())
        .collect::<Vec<_>>();
    let mut group = StructuredCalibrationGroupV2::new(
        opts,
        fingerprint(),
        Arc::new(Clock(AtomicU64::new(7))),
        0,
    )
    .unwrap();
    group.begin_cohort(0).unwrap();
    group.admitted(id, 10).unwrap();
    group.offer(&[]).unwrap();
    group.reserve_prepared(prepared).unwrap();
    let captures = group
        .children
        .iter()
        .map(|c| c.pending_capture().unwrap())
        .collect::<Vec<_>>();
    assert!(Arc::ptr_eq(&captures[0], &captures[1]));
    assert_eq!(captures[0].structured_sessions().len(), 2);
    assert_eq!(group.progress()[0].reserved_members, [1, 0, 0]);
    assert_eq!(group.progress()[1].reserved_members, [0, 0, 0]);
    // No writer records Prepared inside the original measurement window.
    assert!(paths
        .iter()
        .all(|p| !records(p).iter().any(|r| r["record"]["kind"] == "reserved")));
    group
        .complete_unsubmitted("guard rejected before device submission")
        .unwrap();
    assert_eq!(group.progress()[0].failed_members, [1, 0, 0]);
    assert_eq!(group.progress()[0].completed_members, [0; 3]);
    assert!(group.end_cohort().is_err()); // failure did not advance a token
    let artifact = group.finish(0).unwrap();
    assert!(artifact.failure.is_some());
    for (index, path) in paths.iter().enumerate() {
        let raw = records(path);
        let reserved = raw
            .iter()
            .find(|r| r["record"]["kind"] == "reserved")
            .unwrap();
        assert_eq!(
            reserved["record"]["member"],
            if index == 0 {
                serde_json::json!(1)
            } else {
                serde_json::Value::Null
            }
        );
        assert!(raw.iter().any(|r| r["record"]["kind"] == "unsubmitted"));
        assert!(!raw
            .iter()
            .any(|r| r["record"]["kind"] == "request_completed"));
    }
}

#[test]
fn structured_group_a_child_io_limit_poison_invalidates_all_without_partial_success() {
    let files = Files::new();
    let mut group = StructuredCalibrationGroupV2::new(
        options(&files),
        fingerprint(),
        Arc::new(Clock(AtomicU64::new(7))),
        0,
    )
    .unwrap();
    let paths = group
        .children
        .iter()
        .map(|c| c.options.observations_path.clone())
        .collect::<Vec<_>>();
    // Exercise the real bounded writer: the oversized value is borrowed and
    // rejected while streaming, never retained as a per-child JSON tree.
    let oversized = "x".repeat(2 * 1024 * 1024);
    assert!(group.children[0]
        .source
        .record_borrowed(&oversized)
        .is_err());
    assert!(group.begin_cohort(0).is_err());
    assert!(group
        .progress()
        .iter()
        .all(|p| p.phase == StructuredCapturePhase::Failed && p.failure.is_some()));
    assert!(group.finish(0).is_err());
    assert!(std::fs::metadata(&paths[0]).unwrap().len() <= 1024 * 1024);
    let survivor = records(&paths[1]);
    assert_eq!(survivor.last().unwrap()["record"]["phase"], "failed");
    assert!(survivor
        .iter()
        .any(|r| r["record"]["kind"] == "phase_failed"));
}

#[test]
fn structured_group_original_fifo_gap_and_opening_clock_are_not_rebased() {
    let files = Files::new();
    let clock = Arc::new(Clock(AtomicU64::new(23)));
    let mut group =
        StructuredCalibrationGroupV2::new(options(&files), fingerprint(), clock.clone(), 41)
            .unwrap();
    clock.0.store(55, Ordering::SeqCst);
    for child in &mut group.children {
        assert_eq!(child.binding.opened_at_ns(), 23);
        child
            .ledger
            .accept_fifo(Some(HostStageQueueReceipt {
                disposition: HostStageQueueDisposition::Published,
                accepted_ordinal: Some(42),
            }))
            .unwrap();
    }
    assert!(group.children[1]
        .ledger
        .accept_fifo(Some(HostStageQueueReceipt {
            disposition: HostStageQueueDisposition::Published,
            accepted_ordinal: Some(44)
        }))
        .is_err());
    assert!(group.freeze(44).is_err());
    let artifact = group.finish(44).unwrap();
    assert!(artifact
        .children
        .iter()
        .all(|c| c.phase == StructuredCapturePhase::Failed && c.model.is_none()));
    for child in artifact.children {
        let raw = records(&child.source_path);
        assert_eq!(raw[0]["record"]["opened_at_ns"], 23);
        assert_eq!(raw.last().unwrap()["record"]["last_captured_fifo"], 42);
        assert_eq!(raw.last().unwrap()["record"]["accepted_fifo_cutoff"], 44);
        assert_eq!(raw.last().unwrap()["record"]["fifo_audit_complete"], false);
    }
}

#[test]
fn shared_source4_records_unavailable_once_and_failed_population_cannot_publish() {
    let files = Files::new();
    let mut configured = options(&files);
    let path = files.0.canonicalize().unwrap().join("shared.jsonl");
    configured.shared_source = Some(path.clone());
    for child in &mut configured.children {
        child.observations_path = path.clone();
    }
    let mut group = StructuredCalibrationGroupV2::new(
        configured,
        fingerprint(),
        Arc::new(Clock(AtomicU64::new(7))),
        0,
    )
    .unwrap();
    group.begin_cohort(0).unwrap();
    group.admitted(RequestId::new(), 10).unwrap();
    group.offer(&[]).unwrap();
    group
        .complete_unsubmitted("original preparation unavailable")
        .unwrap();
    assert!(group.freeze(0).is_err());
    let artifact = group.finish(0).unwrap();
    assert!(artifact.failure.is_some());
    assert!(artifact
        .children
        .iter()
        .all(|c| c.model.is_none() && c.source_path == path));
    let raw = records(&path);
    assert_eq!(raw[0]["record"]["schema_version"], 4);
    assert_eq!(
        raw.iter()
            .filter(|r| r["record"]["record"]["kind"] == "preparation_unavailable")
            .count(),
        1
    );
    assert!(!raw.iter().any(|r| r["record"]["kind"] == "phase_freeze"));
    assert_eq!(std::fs::read_dir(&files.0).unwrap().count(), 1);
}

#[test]
fn shared_source4_header_capacity_fails_during_group_construction_before_offers() {
    for oversized_record in [false, true] {
        let files = Files::new();
        let mut configured = options(&files);
        let path = files.0.canonicalize().unwrap().join("shared.jsonl");
        configured.shared_source = Some(path.clone());
        for child in &mut configured.children {
            child.observations_path = path.clone();
            if oversized_record {
                // File budget permits this declaration; the unchanged 8 MiB
                // line bound must still reject it before any cohort or offer.
                child.maximum_file_bytes = NonZeroU64::new(32 * 1024 * 1024).unwrap();
                child.cohort_manifest_payload =
                    serde_json::json!({"declaration":"a".repeat(8 * 1024 * 1024 - 128)});
            } else {
                child.maximum_file_bytes = NonZeroU64::new(1).unwrap();
            }
        }
        configured.validate().unwrap();
        let result = StructuredCalibrationGroupV2::new(
            configured,
            fingerprint(),
            Arc::new(Clock(AtomicU64::new(7))),
            0,
        );
        assert!(matches!(result, Err(ExportError::Config(_))));
        // A failed capture may leave its original empty diagnostic file, but
        // cannot write a header the loader would reject or begin inference.
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);
        assert_eq!(std::fs::read_dir(&files.0).unwrap().count(), 1);
    }
}
