//! Real session/output lifecycle with the explicitly unsupported controlled
//! projector: group failure must not block Length drain or create a model.
use super::*;
use crate::continuous_engine::inner::cost_observation::StructuredCalibrationGroupV2;
use ferrum_interfaces::execution_cost::{CoreReadbackRoute, HostPendingConstraintV2};
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{windows::*, *};

fn group_options(paths: &[SourcePath; 2]) -> StructuredCalibrationGroupOptionsV2 {
    StructuredCalibrationGroupOptionsV2 { shared_source: None, limits:Default::default(), children:paths.iter().enumerate().map(|(i,path)| {
        let owner = StructuredOwnerKeyV2 {rows:1, role:StructuredWaveRoleV2::OrdinaryDecode,
            product:StructuredProductV2::GreedyToken, readback:CoreReadbackRoute::HostSynchronized,
            provider_template:StructuredTemplateV2::Ordered([1;32]), algorithm_domain:[i as u8+1;32], installed_policy:[3;32]};
        StructuredCalibrationOptionsV2 {
            observations_path:path.0.clone(), protocol_sha256:[4;32],
            scope:StructuredScopeV2 {owner:owner.clone(), coverage:StructuredCoverageV2 {
                pending_eligible_positions:vec![], authorized_pending_constraints:vec![HostPendingConstraintV2::AnySubset],
                pending_counts:vec![0], length_counts:vec![0], pending_positions:vec![], length_positions:vec![], joint_counts:vec![(0,0)],
            }},
            membership_rule:MembershipRuleV2 {owner, windows:vec![FrontierWindowV2 {rows:vec![RowWindowV2 {
                generated_before:ClosedRangeV2::ALL, remaining_output:ClosedRangeV2::ALL, context_before:ClosedRangeV2::ALL,
                work:WorkWindowV2::Decode {kv_tokens:ClosedRangeV2::ALL},
            }]}]},
            cohort_plan:CohortPlanV2 {phases:std::array::from_fn(|_|vec![CohortV2 {manifest_case:0, repetition:0,
                requests:vec![CohortRequestV2 {manifest_prompt:0,maximum_output:4}]}])},
            cohort_manifest_payload:serde_json::json!({"controlled_protocol_fixture":true,"maximum_output":4}),
            settings:StructuredSettingsV2 {max_phase_samples:8,max_axes:64,..Default::default()},phase_members:[8;3],
            maximum_offered_waves:NonZeroUsize::new(64).unwrap(),maximum_file_bytes:NonZeroU64::new(1<<20).unwrap(),
        }
    }).collect()}
}

#[tokio::test]
async fn structured_group_session_unsupported_projector_fails_capture_but_completes_length() {
    unsupported_projector_keeps_diagnostic_and_drains(false).await;
}

#[tokio::test]
async fn structured_single_v2_unsupported_projector_fails_capture_but_completes_length() {
    unsupported_projector_keeps_diagnostic_and_drains(true).await;
}

async fn unsupported_projector_keeps_diagnostic_and_drains(single: bool) {
    let (mut session, executor) = collector_fixture().await;
    let paths = [SourcePath::new(), SourcePath::new()];
    session.limits = session
        .limits
        .with_structured_prepared_projection_budget(Some(
            StructuredPreparedProjectionBudgetV2::new(1_000_000).unwrap(),
        ));
    // Public API cannot enable immutable capture after engine creation.
    assert!(session
        .begin_structured_cost_group_v2(group_options(&paths))
        .await
        .is_err());
    assert!(session.structured_group_v2.is_none());
    let cutoff = session
        .freeze_cost_model()
        .await
        .unwrap()
        .accepted_ordinal();
    let clock = Arc::clone(&session.engine.inner.cost_runtime.as_ref().unwrap().clock);
    // As in the existing single-collector isolation test, direct installation
    // tests the failure path. It does not grant this mock a qualified producer
    // or implement a fake successor-state projector.
    if single {
        session.structured_capture_v2 = Some(
            crate::continuous_engine::inner::cost_observation::StructuredCalibrationCollectorV2::new(
                group_options(&paths).children.remove(0), fingerprint(), clock, cutoff,
            ).unwrap(),
        );
        session.begin_structured_cost_cohort_v2(0).unwrap();
    } else {
        session.structured_group_v2 = Some(
            StructuredCalibrationGroupV2::new(group_options(&paths), fingerprint(), clock, cutoff)
                .unwrap(),
        );
        session.begin_structured_cost_group_cohort_v2(0).unwrap();
    }
    let (id, mut output) = add(&mut session, 4).await;
    admit(&mut session).await;
    for turn in 0..4 {
        let row = if turn == 0 {
            frontier(&session, &id)
                .prefill_work(NonZeroU32::new(4).unwrap())
                .unwrap()
        } else {
            frontier(&session, &id).decode_work().unwrap()
        };
        let report = wave(&mut session, &executor, vec![row]).await;
        assert!(report.error.is_none(), "{:?}", report.error);
        if turn == 0 {
            let diagnostic = report.structured_prepared_projection.as_ref().unwrap();
            assert_eq!(diagnostic.configured_budget_us, Some(1_000_000));
            assert_eq!(diagnostic.budget_ns, 1_000_000_000);
            assert!(diagnostic.wall_ns.is_some());
            assert!(diagnostic.error.is_some());
        } else {
            assert!(
                report.structured_prepared_projection.is_none(),
                "failed capture must not retry its projection"
            );
        }
        assert_eq!(
            report.submission,
            CalibrationSubmissionState::HostReconciled
        );
        if turn < 3 {
            drop(bounded(output.frames.next()).await.unwrap());
            ready(&session, &id, false).await;
        }
    }
    bounded(async {
        while let Some(frame) = output.frames.next().await {
            drop(frame);
        }
    })
    .await;
    assert!(session.frontiers().unwrap().is_empty());
    assert_eq!(executor.entries.load(Ordering::Acquire), 4);
    let children = if single {
        vec![session
            .finish_structured_cost_calibration_v2()
            .await
            .unwrap()]
    } else {
        let artifact = session.finish_structured_cost_group_v2().await.unwrap();
        assert!(artifact.failure.is_some());
        artifact.children
    };
    assert!(children
        .iter()
        .all(|c| c.phase == StructuredCapturePhase::Failed
            && c.model.is_none()
            && c.failure.is_some()));
    for path in paths.iter().take(children.len()) {
        let text = std::fs::read_to_string(&path.0).unwrap();
        let raw: Vec<serde_json::Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert!(raw.iter().any(|r| r["record"]["kind"] == "offered"));
        assert!(!raw.iter().any(|r| r["record"]["kind"] == "phase_freeze"));
    }
    session.shutdown().await.unwrap();
}
