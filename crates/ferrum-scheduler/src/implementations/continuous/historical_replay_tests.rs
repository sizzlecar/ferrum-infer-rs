use super::*;

use ferrum_interfaces::model_executor::ExecutorPrefillAdmissionReceipt;
use ferrum_interfaces::vnext::{CapacityDomainId, DeferredAction};
use ferrum_types::{ModelId, SamplingParams};
use serde::Deserialize;

const HISTORICAL_REPLAY_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/da9c1ee8_cross_phase_capacity_replay.json"
));

#[derive(Debug, Deserialize)]
struct HistoricalReplayFixture {
    schema_version: u32,
    name: String,
    source: HistoricalReplaySource,
    coordinator_id: u64,
    wake_epochs: HistoricalWakeEpochs,
    requests: Vec<HistoricalRequest>,
    wait_sources: Vec<HistoricalWaitSource>,
    rejected_terminal: HistoricalRejectedTerminal,
    steps: Vec<HistoricalReplayStep>,
}

#[derive(Debug, Deserialize)]
struct HistoricalReplaySource {
    git_sha: String,
    binary_sha256: String,
    trace_sha256: String,
    artifact: String,
    failure_class: String,
    terminal_event_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
struct HistoricalWakeEpochs {
    release: u64,
    capacity: u64,
    policy: u64,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
enum HistoricalRequestRole {
    RecomputePrefill,
    Decode,
}

#[derive(Debug, Deserialize)]
struct HistoricalRequest {
    role: HistoricalRequestRole,
    request_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum HistoricalWaitSource {
    Domain { domain: u32, epoch: u64 },
    PlanDeviceBudget { epoch: u64 },
}

impl HistoricalWaitSource {
    fn availability(&self) -> CapacityAvailabilityEpoch {
        match *self {
            Self::Domain { domain, epoch } => CapacityAvailabilityEpoch::new(
                CapacityAvailabilitySource::Domain(CapacityDomainId::new(domain).unwrap()),
                epoch,
            )
            .unwrap(),
            Self::PlanDeviceBudget { epoch } => {
                CapacityAvailabilityEpoch::new(CapacityAvailabilitySource::PlanDeviceBudget, epoch)
                    .unwrap()
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum HistoricalExpectedAction {
    Deferred,
    YieldPlanned,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum HistoricalReplayStep {
    SubmitAndAdmit {
        max_batch_size: usize,
    },
    PromoteAllToDecode {
        tokens: usize,
    },
    EvictDecodeForRecompute {
        role: HistoricalRequestRole,
        attempted_decode_width: usize,
    },
    ReadmitRecompute {
        max_batch_size: usize,
    },
    DeferForCapacity {
        role: HistoricalRequestRole,
        expected: HistoricalExpectedAction,
    },
    ArmAndCompleteYield {
        attempted_decode_width: usize,
        observed_free_blocks: Option<usize>,
    },
    CancelAll,
}

impl HistoricalReplayStep {
    const fn label(&self) -> &'static str {
        match self {
            Self::SubmitAndAdmit { .. } => "submit_and_admit",
            Self::PromoteAllToDecode { .. } => "promote_all_to_decode",
            Self::EvictDecodeForRecompute { .. } => "evict_decode_for_recompute",
            Self::ReadmitRecompute { .. } => "readmit_recompute",
            Self::DeferForCapacity { .. } => "defer_for_capacity",
            Self::ArmAndCompleteYield { .. } => "arm_and_complete_yield",
            Self::CancelAll => "cancel_all",
        }
    }
}

#[derive(Debug, Deserialize)]
struct HistoricalRejectedTerminal {
    active: usize,
    prefill: usize,
    decode: usize,
    blocked_prefill: usize,
    blocked_decode: usize,
    pending_release_fences: usize,
}

impl HistoricalRejectedTerminal {
    fn matches(&self, snapshot: &ContinuousSchedulerTraceSnapshot) -> bool {
        snapshot.active_len == self.active
            && snapshot.prefill_queue_len == self.prefill
            && snapshot.decode_queue_len == self.decode
            && snapshot.execution_capacity_blocked_prefill_len == self.blocked_prefill
            && snapshot.execution_capacity_blocked_decode_len == self.blocked_decode
            && snapshot.pressure_pending_release_fences == self.pending_release_fences
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HistoricalYieldProjection {
    kind: PressureYieldKind,
    progress_owner: HistoricalRequestRole,
    victim: HistoricalRequestRole,
    planned_ordinal: u64,
    armed_ordinal: Option<u64>,
    release_ordinal: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HistoricalReplayProjection {
    source_git_sha: String,
    source_trace_sha256: String,
    batches: Vec<Vec<HistoricalRequestRole>>,
    snapshot_labels: Vec<&'static str>,
    snapshots: Vec<ContinuousSchedulerTraceSnapshot>,
    yield_projection: HistoricalYieldProjection,
    journal: Vec<PressureTransition>,
    old_terminal_matches: usize,
    passive_park_prevented: bool,
}

fn request(_role: HistoricalRequestRole, request_id: RequestId) -> InferenceRequest {
    InferenceRequest {
        id: request_id,
        prompt: "historical capacity replay".to_string(),
        model_id: ModelId::new("historical-capacity-replay"),
        sampling_params: SamplingParams::default(),
        stream: false,
        priority: Priority::Normal,
        client_id: None,
        session_id: None,
        created_at: chrono::Utc::now(),
        api_request: None,
        evidence_request: Default::default(),
        metadata: std::collections::HashMap::new(),
    }
}

fn role_for_request(
    requests: &[(HistoricalRequestRole, RequestId)],
    request_id: &RequestId,
) -> HistoricalRequestRole {
    requests
        .iter()
        .find_map(|(role, id)| (id == request_id).then_some(*role))
        .unwrap_or_else(|| panic!("historical replay observed unknown request {request_id}"))
}

fn request_id_for_role(
    requests: &[(HistoricalRequestRole, RequestId)],
    role: HistoricalRequestRole,
) -> &RequestId {
    requests
        .iter()
        .find_map(|(candidate, id)| (*candidate == role).then_some(id))
        .unwrap_or_else(|| panic!("historical replay fixture is missing role {role:?}"))
}

fn project_batch(
    batch: &BatchPlan,
    requests: &[(HistoricalRequestRole, RequestId)],
) -> Vec<HistoricalRequestRole> {
    batch
        .requests
        .iter()
        .map(|scheduled| role_for_request(requests, &scheduled.request.id))
        .collect()
}

async fn replay_historical_cross_phase_fixture() -> HistoricalReplayProjection {
    let fixture: HistoricalReplayFixture = serde_json::from_str(HISTORICAL_REPLAY_JSON).unwrap();
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.name, "da9c1ee8_cross_phase_capacity_deadlock");
    assert_eq!(
        fixture.source.git_sha,
        "da9c1ee8363c686e71420fd5df8042c496e69757"
    );
    assert_eq!(
        fixture.source.failure_class,
        "cross_phase_capacity_progress_deadlock"
    );
    assert_eq!(fixture.source.terminal_event_ids.len(), 7);
    assert_eq!(
        fixture.source.terminal_event_ids.first().unwrap(),
        "evt-engine-vnext-admission-531"
    );
    assert_eq!(
        fixture.source.terminal_event_ids.last().unwrap(),
        "evt-engine-vnext-admission-537"
    );
    assert_eq!(fixture.source.binary_sha256.len(), 64);
    assert_eq!(fixture.source.trace_sha256.len(), 64);
    assert!(fixture.source.artifact.ends_with("scheduler-trace.jsonl"));

    let coordinator = NonZeroU64::new(fixture.coordinator_id).unwrap();
    let wake = AdmissionWakeEpochs::new(
        coordinator,
        fixture.wake_epochs.release,
        fixture.wake_epochs.capacity,
        fixture.wake_epochs.policy,
    );
    let availability = fixture
        .wait_sources
        .iter()
        .map(HistoricalWaitSource::availability)
        .collect::<Vec<_>>();
    let wait_condition =
        CapacityWaitCondition::from_observation(fixture.coordinator_id, availability.clone())
            .unwrap();
    let requests = fixture
        .requests
        .iter()
        .map(|request| {
            (
                request.role,
                RequestId(uuid::Uuid::parse_str(&request.request_id).unwrap()),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(requests.len(), 2);
    assert_ne!(requests[0].0, requests[1].0);

    let release_sources = wait_condition
        .observed()
        .iter()
        .map(|observed| observed.source())
        .collect::<Vec<_>>();
    let release_snapshot = ExecutionCapacityReleaseSnapshot::new(
        requests
            .iter()
            .map(|(_, request_id)| (request_id.clone(), release_sources.clone())),
    );
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
        max_running_requests: requests.len(),
        prompt_token_estimate: true,
        ..SchedulerConfig::default()
    });
    let mut batches = Vec::new();
    let mut snapshot_labels = Vec::new();
    let mut snapshots = Vec::new();
    let mut pending_yield = None;
    let mut yield_projection = None;
    let mut passive_park_prevented = false;

    for step in &fixture.steps {
        match *step {
            HistoricalReplayStep::SubmitAndAdmit { max_batch_size } => {
                for (role, request_id) in &requests {
                    scheduler
                        .submit(request(*role, request_id.clone()))
                        .await
                        .unwrap();
                }
                let admitted = scheduler
                    .next_batch_with_dynamic_admission(
                        BatchHint::simple(max_batch_size),
                        AdmissionWakeSnapshot::new(wake, &availability),
                        &mut |request| {
                            AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                                request_id: request.id.clone(),
                            })
                        },
                    )
                    .unwrap()
                    .expect("historical requests must be admitted");
                assert_eq!(admitted.requests.len(), requests.len());
                batches.push(project_batch(&admitted, &requests));
            }
            HistoricalReplayStep::PromoteAllToDecode { tokens } => {
                for (_, request_id) in &requests {
                    scheduler.mark_prefill_complete(request_id, tokens);
                }
                assert!(requests.iter().all(|(_, request_id)| {
                    scheduler.trace_phase(request_id) == Some(RequestPhase::Decoding)
                }));
            }
            HistoricalReplayStep::EvictDecodeForRecompute {
                role,
                attempted_decode_width,
            } => {
                assert!(scheduler.defer_decode_to_waiting_for_capacity(
                    request_id_for_role(&requests, role),
                    attempted_decode_width,
                ));
            }
            HistoricalReplayStep::ReadmitRecompute { max_batch_size } => {
                let batch = scheduler
                    .next_batch_with_dynamic_admission(
                        BatchHint::simple(max_batch_size),
                        AdmissionWakeSnapshot::new(wake, &availability),
                        &mut |request| {
                            AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                                request_id: request.id.clone(),
                            })
                        },
                    )
                    .unwrap()
                    .expect("recompute prefill must rejoin the active decode");
                assert_eq!(batch.requests.len(), requests.len());
                batches.push(project_batch(&batch, &requests));
                assert_eq!(
                    scheduler.trace_phase(request_id_for_role(
                        &requests,
                        HistoricalRequestRole::RecomputePrefill,
                    )),
                    Some(RequestPhase::Prefilling)
                );
                assert_eq!(
                    scheduler.trace_phase(request_id_for_role(
                        &requests,
                        HistoricalRequestRole::Decode,
                    )),
                    Some(RequestPhase::Decoding)
                );
            }
            HistoricalReplayStep::DeferForCapacity { role, expected } => {
                let request_id = request_id_for_role(&requests, role);
                let deferral = AdmissionDeferral::new(
                    DeferredAction::WaitForRelease,
                    wake,
                    wait_condition.clone(),
                );
                let action = match scheduler.trace_phase(request_id) {
                    Some(RequestPhase::Prefilling) => scheduler
                        .defer_prefill_for_execution_capacity(
                            request_id,
                            deferral,
                            &release_snapshot,
                        )
                        .unwrap(),
                    Some(RequestPhase::Decoding) => scheduler
                        .defer_decode_for_execution_capacity(
                            std::slice::from_ref(request_id),
                            deferral,
                            &release_snapshot,
                        )
                        .unwrap(),
                    phase => panic!("historical capacity failure reached invalid phase {phase:?}"),
                };
                match (expected, action) {
                    (
                        HistoricalExpectedAction::Deferred,
                        ExecutionCapacityAction::Deferred { count: 1 },
                    ) => {}
                    (
                        HistoricalExpectedAction::YieldPlanned,
                        ExecutionCapacityAction::YieldPlanned { transaction },
                    ) => {
                        let progress_owner =
                            role_for_request(&requests, transaction.progress_owner_id());
                        let victim = role_for_request(&requests, transaction.victim_request_id());
                        yield_projection = Some(HistoricalYieldProjection {
                            kind: transaction.kind(),
                            progress_owner,
                            victim,
                            planned_ordinal: transaction.planned_ordinal().get(),
                            armed_ordinal: None,
                            release_ordinal: None,
                        });
                        pending_yield = Some(transaction);
                        let planned = scheduler.trace_snapshot();
                        assert_eq!(planned.pressure_pending_release_fences, 1);
                        assert!(!fixture.rejected_terminal.matches(&planned));
                        assert_eq!(
                            scheduler.passive_capacity_wait_condition().unwrap(),
                            None,
                            "a planned release fence must prevent passive parking"
                        );
                        passive_park_prevented = true;
                    }
                    (expected, actual) => {
                        panic!("historical replay expected {expected:?}, observed {actual:?}")
                    }
                }
            }
            HistoricalReplayStep::ArmAndCompleteYield {
                attempted_decode_width,
                observed_free_blocks,
            } => {
                let transaction = pending_yield
                    .take()
                    .expect("yield completion requires a planned transaction");
                let armed = scheduler
                    .arm_execution_capacity_yield(&transaction)
                    .unwrap();
                let completion = scheduler
                    .complete_execution_capacity_yield(
                        &transaction,
                        attempted_decode_width,
                        observed_free_blocks,
                    )
                    .unwrap();
                assert!(completion.victim_requeued());
                let projection = yield_projection.as_mut().unwrap();
                projection.armed_ordinal = Some(armed.get());
                projection.release_ordinal = Some(completion.release_transition_ordinal().get());
            }
            HistoricalReplayStep::CancelAll => {
                for (_, request_id) in &requests {
                    assert!(scheduler.cancel(request_id.clone()).await.unwrap());
                }
            }
        }
        snapshot_labels.push(step.label());
        snapshots.push(scheduler.trace_snapshot());
    }

    assert!(pending_yield.is_none());
    let final_snapshot = snapshots.last().unwrap();
    assert_eq!(final_snapshot.active_len, 0);
    assert_eq!(final_snapshot.waiting_queue_len, 0);
    assert_eq!(final_snapshot.pressure_active_episodes, 0);
    assert_eq!(final_snapshot.pressure_pending_release_fences, 0);
    let old_terminal_matches = snapshots
        .iter()
        .filter(|snapshot| fixture.rejected_terminal.matches(snapshot))
        .count();
    assert_eq!(old_terminal_matches, 0);

    HistoricalReplayProjection {
        source_git_sha: fixture.source.git_sha,
        source_trace_sha256: fixture.source.trace_sha256,
        batches,
        snapshot_labels,
        snapshots,
        yield_projection: yield_projection.expect("fixture must plan one typed yield"),
        journal: scheduler.pressure_transition_journal(),
        old_terminal_matches,
        passive_park_prevented,
    }
}

#[tokio::test]
async fn da9c1ee8_cross_phase_capacity_replay_rejects_old_terminal_one_hundred_of_one_hundred() {
    const REPLAY_COUNT: usize = 100;
    let expected = replay_historical_cross_phase_fixture().await;
    assert_eq!(expected.old_terminal_matches, 0);
    assert!(expected.passive_park_prevented);
    assert_eq!(expected.snapshot_labels.len(), 8);
    assert_eq!(expected.batches.len(), 2);
    assert!(expected.yield_projection.planned_ordinal > 0);
    assert!(
        expected.yield_projection.planned_ordinal
            < expected.yield_projection.armed_ordinal.unwrap()
    );
    assert!(
        expected.yield_projection.armed_ordinal.unwrap()
            < expected.yield_projection.release_ordinal.unwrap()
    );
    assert!(expected
        .journal
        .windows(2)
        .all(|pair| pair[0].ordinal() < pair[1].ordinal()));

    for ordinal in 1..REPLAY_COUNT {
        assert_eq!(
            replay_historical_cross_phase_fixture().await,
            expected,
            "historical scheduler replay {ordinal} diverged from replay 0"
        );
    }
    println!(
        "FERRUM G04 HISTORICAL CAPACITY REPLAY KEEP: source_commit={} trace_sha256={} deterministic_replays={REPLAY_COUNT}/{REPLAY_COUNT} yield_planned={REPLAY_COUNT}/{REPLAY_COUNT} passive_park_prevented={REPLAY_COUNT}/{REPLAY_COUNT} old_terminal=0/{REPLAY_COUNT} snapshots={} journal_transitions={}",
        expected.source_git_sha,
        expected.source_trace_sha256,
        expected.snapshots.len(),
        expected.journal.len(),
    );
}
