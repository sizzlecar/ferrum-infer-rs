//! Synthetic clocks, real installed policy and private settlement writer.
//! JSONL is emitted only from original completed diagnostics; no stage digest
//! or settlement is repaired by the fixture before strict source6 replay.
use super::*;
use ferrum_scheduler::implementations::continuous::{
    cost_model::structured_v2::{
        StructuredCoverageV2, StructuredPhaseV2, StructuredQueryV2, StructuredScopeV2,
        StructuredServiceDomainPolicyV1, StructuredSettingsV2,
    },
    cost_profile::{
        export_structured_profile_v13, load_structured_profile_v13, CostProfileLoadLimits,
        ImportedStructuredCatalogV13, ProfileFingerprint, ProfileLoadClock,
        StructuredServiceClockV6, StructuredServiceCollectorV6, StructuredServiceDeclarationV6,
        StructuredServiceHeaderV6, StructuredServiceRecordV6, StructuredServiceWaveV6,
    },
};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
#[path = "source6_installed/feedback.rs"]
mod feedback;
#[path = "source6_installed/fixture.rs"]
mod fixture;
use fixture::{Fixture, Template};

const ELIGIBLE: usize = 16;
const OFFERS: usize = ELIGIBLE + 1;
const STRIDE: u64 = 100_000;
const WALL_EPOCH: u64 = 1_000_000_000;

#[derive(Clone, Copy, Default)]
struct Variation {
    missing_early_phase: Option<usize>,
    slow_qualification: bool,
}
fn append(bytes: &mut Vec<u8>, value: &impl serde::Serialize) {
    bytes.extend(serde_json::to_vec(value).unwrap());
    bytes.push(b'\n');
}
fn fingerprint() -> model::ExecutionFingerprint {
    let ExecutorCostIdentityAvailability::Known(identity) = identity() else {
        unreachable!()
    };
    model::ExecutionFingerprint {
        model_weights: identity.model_weights,
        numerical_policy: identity.numerical_policy,
        device_runtime: identity.device_runtime,
        execution_config: identity.execution_config,
    }
}
fn header(fixture: &Fixture) -> StructuredServiceHeaderV6 {
    let query = fixture.continuing.query();
    StructuredServiceHeaderV6::new(
        Sha256::digest(uuid::Uuid::new_v4().as_bytes()).into(),
        1,
        ProfileFingerprint::from(&fingerprint()),
        serde_json::json!({"executable_path": std::env::current_exe().unwrap()}),
        StructuredServiceClockV6 {
            wall_unix_ns: WALL_EPOCH,
            monotonic_ns: 1,
        },
        StructuredServiceDeclarationV6 {
            route_population: ferrum_types::SloCalibrationRoutePopulationV1::AllAttempts,
            domain_policy: StructuredServiceDomainPolicyV1::AllOffered,
            nonnegative_envelope: None,
            phase_offered_waves: [OFFERS; 3],
            maximum_window_ns: 1_000_000_000,
            settings: StructuredSettingsV2 {
                max_sample_age_ns: 1_000_000_000,
                static_margin_ns: 50,
                ..Default::default()
            },
            scopes: vec![StructuredScopeV2 {
                numerical_family: None,
                owner: query.owner().clone(),
                coverage: StructuredCoverageV2 {
                    pending_counts: vec![0],
                    length_counts: vec![0, 1],
                    pending_positions: vec![],
                    length_positions: vec![0],
                    pending_eligible_positions: vec![],
                    authorized_pending_constraints: vec![],
                    joint_counts: vec![(0, 0), (0, 1)],
                },
            }],
            maximum_retained_numeric_bytes: 32 * 1024 * 1024,
        },
        16 * 1024 * 1024,
    )
    .unwrap()
}

fn record(
    ids: &EngineCostIds,
    queue: &Arc<BoundedCostSampleSink>,
    template: &Template,
    ticket: u64,
    phase: StructuredPhaseV2,
    reason: Option<FinishReason>,
    wall: u64,
) -> StructuredServiceWaveV6 {
    let (fifo, stages) = record_stages(ids, queue, template, ticket, reason, wall);
    // The legacy Serialize view deliberately skips the private structured
    // sidecar. Use the same explicit borrowed diagnostic view as the live
    // source6 publisher; never reconstruct or repair its settlement binding.
    let diagnostic = serde_json::to_value(stages.structured_diagnostic_view()).unwrap();
    assert_eq!(
        diagnostic["structured_evidence"],
        serde_json::to_value(&stages.structured_evidence).unwrap(),
        "source6 must retain the exact original qualified settlement"
    );
    // Like the production publisher, carry the original independent sidecar
    // separately: the legacy statistical Serialize view intentionally omits
    // it, while the private settlement binding includes its exact contents.
    let independent = stages
        .statistical_evidence
        .as_ref()
        .and_then(|value| value.independent_attention_v2())
        .map(|value| value.to_wire_v2());
    assert!(
        independent.is_some(),
        "this fixture exercises a real independent sidecar in the settlement binding"
    );
    StructuredServiceWaveV6::from_diagnostic(
        ticket,
        phase,
        ticket * STRIDE,
        fifo,
        diagnostic,
        independent,
    )
    .unwrap()
}

fn record_stages(
    ids: &EngineCostIds,
    queue: &Arc<BoundedCostSampleSink>,
    template: &Template,
    ticket: u64,
    reason: Option<FinishReason>,
    wall: u64,
) -> (u64, Arc<HostStageEvidenceV1>) {
    let start = ticket * STRIDE;
    let clock = Arc::new(VirtualClock(AtomicU64::new(start + 1)));
    let mut actual = template.actual.clone();
    // Fresh requests make every open frontier honest; there is no reuse of a
    // terminated owner or unexplained context reset inside a phase.
    actual.rows[0].request_id = RequestId::new();
    actual.rows[0].owner_incarnation = ticket;
    let row = &actual.rows[0];
    let mut call = EngineCostCall::begin(
        ids,
        clock.clone(),
        Arc::clone(queue),
        EngineCostCallSpec {
            identity: identity(),
            participants: vec![CostObservationParticipant {
                request_id: row.request_id.clone(),
                owner_incarnation: row.owner_incarnation,
                work_generation: row.work_generation,
                input_index: row.input_index,
                output_policy_signature: Some(template.policy_signature),
                host_features: Some(template.host),
            }],
            prepare_started_at_ns: Some(start),
            boundary: WaveObservationBoundary::IsolatedPreparationToCommit,
            recorder_limits: CostRecorderLimits {
                max_waves: 1,
                max_rows_per_wave: 8,
                max_retained_rows: 128,
            },
        },
    )
    .unwrap()
    .with_structured_capture(true);
    let mut context = call.context().unwrap();
    context.physical_wave(Ok(actual.clone()), Some(start + 2));
    clock.set(start + 5);
    context.terminal(ActualWaveOutcome::Completed, None);
    context.finish_call(ObservedCallOutcome::Completed);
    drop(context);
    settle(
        &mut call,
        &clock,
        row,
        start + wall - 3,
        reason.map(|reason| HostTerminalStageV1 {
            finish_reason: reason,
            ..terminal()
        }),
    );
    call.reject(CostCallRejection::Composite);
    clock.set(start + wall + 1);
    assert_eq!(call.finish(), CostCallDisposition::Queued);
    let (fifo, entry, ticket_receipt) = queue.pop_numbered_with_ticket().unwrap();
    assert!(ticket_receipt.is_none());
    let CostEvidenceEntry::StagesOnly {
        stages,
        legacy_rejection: CostCallRejection::Composite,
    } = entry
    else {
        panic!("original private host stages are retained")
    };
    assert_eq!(stages.full_wall_ns, Some(wall));
    assert_eq!(
        stages.rows[0].terminal.as_ref().map(|t| t.finish_reason),
        reason
    );
    stages
        .structured_evidence
        .as_ref()
        .unwrap()
        .as_ref()
        .unwrap()
        .validate_host_stages(&stages)
        .unwrap();
    (fifo, stages)
}

struct Captured {
    bytes: Vec<u8>,
    qualified: usize,
    freezes: Vec<StructuredServiceRecordV6>,
    closing: StructuredServiceClockV6,
}
fn collect(fixture: &Fixture, variation: Variation) -> Captured {
    let header = header(fixture);
    let mut bytes = Vec::new();
    append(&mut bytes, &header);
    let mut collector =
        StructuredServiceCollectorV6::new(header, CostProfileLoadLimits::default()).unwrap();
    let ids = EngineCostIds::default();
    let queue = sink(2, 512);
    let mut freezes = Vec::new();
    for (phase_index, phase) in [
        StructuredPhaseV2::Fit,
        StructuredPhaseV2::Residual,
        StructuredPhaseV2::Qualification,
    ]
    .into_iter()
    .enumerate()
    {
        let first = (phase_index * OFFERS + 1) as u64;
        let open = StructuredServiceRecordV6::PhaseOpen {
            phase,
            opened_at_ns: first * STRIDE - 1,
            fifo_cutoff: first - 1,
        };
        collector.push(&open).unwrap();
        append(&mut bytes, &open);
        for offset in 0..OFFERS {
            let nonmember = offset == ELIGIBLE;
            let at_length = !nonmember && offset % 4 == 3;
            let reason = if at_length {
                Some(FinishReason::Length)
            } else if nonmember || variation.missing_early_phase == Some(phase_index) {
                None
            } else {
                match offset % 4 {
                    1 => Some(FinishReason::EOS),
                    2 => Some(FinishReason::Stop),
                    _ => None,
                }
            };
            let template = if nonmember {
                &fixture.nonmember
            } else if at_length {
                &fixture.at_length
            } else {
                &fixture.continuing
            };
            let slow = variation.slow_qualification && phase_index == 2 && offset == 1;
            let wall = 1000
                + u64::from(reason.is_some()) * 100
                + u64::from(at_length) * 30
                + u64::from(slow || nonmember) * 5000;
            let event = StructuredServiceRecordV6::Completed {
                wave: record(
                    &ids,
                    &queue,
                    template,
                    first + offset as u64,
                    phase,
                    reason,
                    wall,
                ),
            };
            collector.push(&event).unwrap();
            append(&mut bytes, &event);
        }
        let freeze = collector
            .freeze((first + OFFERS as u64 - 1) * STRIDE + STRIDE / 2)
            .unwrap();
        let StructuredServiceRecordV6::PhaseFreeze { children, .. } = &freeze else {
            unreachable!()
        };
        assert_eq!(
            children[0].members, ELIGIBLE,
            "failed/slow eligible waves stay in their denominator"
        );
        append(&mut bytes, &freeze);
        freezes.push(freeze);
    }
    let closing = StructuredServiceClockV6 {
        monotonic_ns: (3 * OFFERS + 1) as u64 * STRIDE,
        wall_unix_ns: WALL_EPOCH + (3 * OFFERS + 1) as u64 * STRIDE - 1,
    };
    let footer = StructuredServiceRecordV6::Footer {
        offered: (3 * OFFERS) as u64,
        accepted_fifo_cutoff: (3 * OFFERS) as u64,
        closing,
        failure: None,
    };
    collector.push(&footer).unwrap();
    append(&mut bytes, &footer);
    assert_eq!(collector.offered(), (3 * OFFERS) as u64);
    Captured {
        bytes,
        qualified: collector.qualified_children(),
        freezes,
        closing,
    }
}
struct Files(PathBuf);
impl Files {
    fn new() -> Self {
        let dir =
            std::env::temp_dir().join(format!("ferrum-installed-source6-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&dir).unwrap();
        Self(dir)
    }
}
impl Drop for Files {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn replay(captured: &Captured) -> Result<ImportedStructuredCatalogV13, String> {
    let files = Files::new();
    let source = files.0.join("source6.jsonl");
    let profile = files.0.join("profile13.json");
    std::fs::write(&source, &captured.bytes).unwrap();
    let limits = CostProfileLoadLimits::default();
    let exported = export_structured_profile_v13(
        &source,
        Sha256::digest(&captured.bytes).into(),
        &profile,
        0,
        &limits,
    )
    .map_err(|e| format!("{e:?}"))?;
    assert_eq!(exported.children.len(), captured.qualified);
    load_structured_profile_v13(
        &profile,
        &fingerprint(),
        &limits,
        ProfileLoadClock {
            wall_unix_ns: Some(captured.closing.wall_unix_ns + 100),
            wall_max_error_ns: Some(0),
            monotonic_now_ns: 500,
        },
    )
    .map_err(|e| format!("{e:?}"))
}

#[tokio::test]
async fn installed_source6_real_settlement_roundtrip_predicts_future_and_retains_population() {
    let fixture = Fixture::new().await;
    let captured = collect(&fixture, Variation::default());
    assert_eq!(captured.qualified, 1, "{:?}", captured.freezes);
    let imported = replay(&captured).unwrap();
    assert_eq!(
        (imported.offered_attempts, imported.total_shape_rows),
        ((3 * OFFERS) as u64, (3 * OFFERS) as u64)
    );
    assert_eq!(
        imported.children[0]
            .provenance()
            .phases
            .each_ref()
            .map(|p| p.members),
        [ELIGIBLE; 3]
    );
    let prediction = imported.children[0]
        .predict_query_local(&fingerprint(), &fixture.continuing.query(), 500)
        .unwrap();
    assert!(prediction.fitted_lower_ns.abs_diff(1000) <= 2);
    assert!(prediction.fitted_upper_ns.abs_diff(1100) <= 2);
    assert!(prediction.planning_ns >= 1150);
}

#[tokio::test]
async fn installed_source6_missing_early_in_each_independent_phase_stays_unknown() {
    let fixture = Fixture::new().await;
    for phase in 0..3 {
        let captured = collect(
            &fixture,
            Variation {
                missing_early_phase: Some(phase),
                ..Default::default()
            },
        );
        if captured.qualified == 0 {
            assert!(replay(&captured).is_err());
        } else {
            let imported = replay(&captured).unwrap();
            assert!(
                imported.children[0]
                    .predict_query_local(&fingerprint(), &fixture.continuing.query(), 500)
                    .is_err(),
                "phase {phase} must independently cover early completion"
            );
        }
    }
}

#[tokio::test]
async fn installed_source6_slow_eligible_qualification_is_not_removed() {
    let fixture = Fixture::new().await;
    let captured = collect(
        &fixture,
        Variation {
            slow_qualification: true,
            ..Default::default()
        },
    );
    assert_eq!(captured.qualified, 0);
    let StructuredServiceRecordV6::PhaseFreeze { children, .. } = &captured.freezes[2] else {
        unreachable!()
    };
    assert_eq!(
        children[0].failure.as_deref(),
        Some("QualificationUnderestimate")
    );
    assert_eq!(children[0].members, ELIGIBLE);
    assert!(replay(&captured).is_err());
}

#[tokio::test]
async fn installed_source6_below_capacity_length_is_rejected_without_rebinding_receipt() {
    let fixture = Fixture::new().await;
    // Positive controls use the same frontier and export path. Missing private
    // evidence must not make the Length rejection pass for an unrelated reason.
    for reason in [
        None,
        Some(FinishReason::EOS),
        Some(FinishReason::Stop),
        Some(FinishReason::Length),
    ] {
        let mut collector =
            StructuredServiceCollectorV6::new(header(&fixture), CostProfileLoadLimits::default())
                .unwrap();
        collector
            .push(&StructuredServiceRecordV6::PhaseOpen {
                phase: StructuredPhaseV2::Fit,
                opened_at_ns: 1,
                fifo_cutoff: 0,
            })
            .unwrap();
        let event = StructuredServiceRecordV6::Completed {
            wave: record(
                &EngineCostIds::default(),
                &sink(2, 512),
                &fixture.continuing,
                1,
                StructuredPhaseV2::Fit,
                reason,
                1100,
            ),
        };
        let result = collector.push(&event);
        assert_eq!(
            result.is_ok(),
            reason != Some(FinishReason::Length),
            "{reason:?}: {result:?}"
        );
        assert_eq!(collector.qualified_children(), 0);
    }
}
