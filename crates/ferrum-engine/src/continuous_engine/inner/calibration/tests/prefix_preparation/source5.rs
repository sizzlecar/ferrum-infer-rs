//! Actual CPU manual execution writes source5 preparation and release. This
//! executor deliberately has no qualified Structured future projector, so its
//! ordinary measurement must fail closed while all original output drains.
use super::*;
use ferrum_interfaces::execution_cost::{CoreReadbackRoute, HostPendingConstraintV2};
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    prefixes::*, windows::*, *,
};
use std::num::NonZeroU64;

struct Source(std::path::PathBuf);
impl Source {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "ferrum-source5-live-{}.jsonl",
            uuid::Uuid::new_v4()
        )))
    }
}
impl Drop for Source {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn declaration(
    session: &CalibrationSession,
    source: &Source,
) -> (StructuredCalibrationGroupOptionsV2, StructuredPrefixPlanV5) {
    // This predeclared test owner is intentionally not a qualified producer.
    // It cannot turn the controlled executor's unsupported projection Known.
    let owner = StructuredOwnerKeyV2 {
        rows: 1,
        role: StructuredWaveRoleV2::OrdinaryDecode,
        product: StructuredProductV2::GreedyToken,
        readback: CoreReadbackRoute::HostSynchronized,
        provider_template: StructuredTemplateV2::Ordered([1; 32]),
        algorithm_domain: [2; 32],
        installed_policy: [3; 32],
    };
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
    let cohort_plan = CohortPlanV2 {
        phases: std::array::from_fn(|_| {
            vec![CohortV2 {
                manifest_case: 0,
                repetition: 0,
                requests: vec![CohortRequestV2 {
                    manifest_prompt: 0,
                    maximum_output: 3,
                }],
            }]
        }),
    };
    let options = StructuredCalibrationGroupOptionsV2 {
        shared_source: Some(source.0.clone()),
        limits: Default::default(),
        children: vec![StructuredCalibrationOptionsV2 {
            observations_path: source.0.clone(),
            protocol_sha256: [4; 32],
            scope,
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
            cohort_plan,
            cohort_manifest_payload: serde_json::json!({"protocol_fixture":"real-source5-prefix","full_output":3}),
            settings: StructuredSettingsV2 {
                max_phase_samples: 8,
                max_axes: 64,
                ..Default::default()
            },
            phase_members: [8; 3],
            maximum_offered_waves: NonZeroUsize::new(64).unwrap(),
            maximum_file_bytes: NonZeroU64::new(1 << 20).unwrap(),
        }],
    };
    let prefixes = StructuredPrefixPlanV5 {
        phases: std::array::from_fn(|_| {
            vec![Some(StructuredPrefixCohortV5 {
                release_generated: 1,
                slots: vec![StructuredPrefixSlotV5 {
                    tokenizer_policy_sha256: plan(session, &[11]).tokenizer_policy_sha256,
                    token_ids: vec![TokenId::new(11)],
                    token_bytes: vec![vec![0xc3]],
                }],
            })]
        }),
    };
    (options, prefixes)
}

#[tokio::test]
async fn prefix_source5_actual_writer_releases_pending_and_preserves_failed_full_length() {
    let (mut session, executor) = prepared_session().await;
    let source = Source::new();
    let (options, prefixes) = declaration(&session, &source);
    session
        .begin_structured_prefix_cost_group_v5(options, prefixes)
        .await
        .unwrap();
    session.begin_structured_cost_group_cohort_v2(0).unwrap();
    let req = request(&session, 3);
    let id = req.id.clone();
    let output = session
        .add_request(
            req,
            InferenceRequestContext::from_ingress(slo_clock_now()),
            Arc::new(OutputProjectionContract::cli_text()),
        )
        .await
        .unwrap();
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    legacy_v1_begin_rejects_prefix_session(&mut session).await;
    let consumer = tokio::spawn(async move {
        let mut output = output;
        let mut frames = Vec::new();
        while let Some(frame) = output.frames.next().await {
            frames.push(frame.metadata().clone());
            drop(frame);
        }
        let completion = output.completion.await.unwrap();
        let OutputCompletion::Succeeded { reason, usage, .. } = completion.payload() else {
            panic!("source5 must keep the real Length output");
        };
        assert_eq!(*reason, ferrum_types::FinishReason::Length);
        assert_eq!(usage.completion_tokens, 3);
        frames
    });
    ready(&session, &id, false).await;
    admit(&mut session).await;
    let first = frontier(&session, &id)
        .prefill_work(NonZeroU32::MIN)
        .unwrap();
    let report = wave(&mut session, &executor, vec![first]).await;
    assert!(report.error.is_none(), "{:?}", report.error);
    assert!(
        report.structured_prepared_projection.is_none(),
        "preparation is not a duplicate numerical projection"
    );
    assert!(
        session.take_prefix_wave_evidence().is_none(),
        "source5 owns the private actual event"
    );
    ready(&session, &id, false).await;
    let PrefixReleaseProgressV5::Released { receipts } =
        session.advance_structured_prefix_release_v5().unwrap()
    else {
        panic!("actual actor must permit the fixed release");
    };
    assert_eq!(receipts.len(), 1);
    assert_eq!(receipts[0].frontier.pending_utf8, [0xc3]);
    assert_eq!(receipts[0].frontier.generated_tokens, 1);
    assert!(session.engine.inner.sequences.read()[&id]
        .calibration_prefix
        .is_none());
    assert!(
        session.release_prefix_preparation(&id).is_err(),
        "public DTO path cannot bypass source5 writer"
    );
    for generation in 2..=3 {
        let work = frontier(&session, &id).decode_work().unwrap();
        let report = wave(&mut session, &executor, vec![work]).await;
        assert!(report.error.is_none(), "{:?}", report.error);
        if generation == 2 {
            assert!(report
                .structured_prepared_projection
                .as_ref()
                .unwrap()
                .error
                .is_some());
            assert!(session
                .structured_cost_group_progress_v2()
                .unwrap()
                .iter()
                .all(|p| p.phase == StructuredCapturePhase::Failed));
            assert!(matches!(
                session.advance_structured_prefix_release_v5().unwrap(),
                PrefixReleaseProgressV5::Inactive
            ), "the CLI release probe must preserve the already-released suffix after numerical capture fails");
            assert!(session.engine.inner.sequences.read()[&id]
                .calibration_prefix
                .is_none());
            ready(&session, &id, false).await;
        } else {
            assert!(
                report.structured_prepared_projection.is_none(),
                "closed capture cannot retry its unsupported projection"
            );
        }
    }
    let frames = bounded(consumer).await.unwrap();
    assert!(frames.last().unwrap().terminal);
    assert_eq!(frames.last().unwrap().generated_tokens, 3);
    let artifact = session.finish_structured_cost_group_v2().await.unwrap();
    assert!(artifact.failure.is_some());
    assert!(artifact.children.iter().all(|c| c.model.is_none()));
    let records: Vec<serde_json::Value> = std::fs::read_to_string(&source.0)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap()["record"].clone())
        .collect();
    assert_eq!(records[0]["schema_version"], 5);
    let prepared = records
        .iter()
        .filter(|r| r["kind"] == "preparation_completed")
        .collect::<Vec<_>>();
    assert_eq!(prepared.len(), 1);
    assert_eq!(prepared[0]["failure"], serde_json::Value::Null);
    assert_eq!(
        prepared[0]["rows"][0]["preparation_commit"]["original_candidate"],
        6
    );
    assert_eq!(
        prepared[0]["rows"][0]["preparation_commit"]["committed_token"],
        11
    );
    assert_eq!(
        records
            .iter()
            .filter(|r| r["kind"] == "preparation_released")
            .count(),
        1
    );
    assert!(records.iter().any(|r| r["kind"] == "phase_failed"));
    assert!(!records.iter().any(|r| r["kind"] == "phase_freeze"));
    legacy_v1_begin_rejects_prefix_session(&mut session).await;
    assert!(session.selected_phase_boundary().is_err());
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn prefix_source5_rejects_invented_token_bytes_before_source_creation() {
    let (mut session, executor) = prepared_session().await;
    let source = Source::new();
    let (options, mut prefixes) = declaration(&session, &source);
    prefixes.phases[0][0].as_mut().unwrap().slots[0].token_bytes[0] = b"a".to_vec();
    let error = session
        .begin_structured_prefix_cost_group_v5(options, prefixes)
        .await
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("declared bytes differ from actual tokenizer"));
    assert!(!source.0.exists());
    assert!(session.frontiers().unwrap().is_empty());
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    session.shutdown().await.unwrap();
}
