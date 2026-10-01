//! Optional metadata on the existing resource-profile writer. These records
//! are observations, never execution authority. No live owner/lease is retained.
use super::{
    EngineConfig, FerrumProfileEvent, ProfileEntrypoint, ProfileEventKind, ProfileStatus,
    SchedulerTraceJournal, ENGINE_RUNTIME_TRACE_PRESET_HASH, OBSERVABILITY_PROFILE_SCHEMA_VERSION,
};
use ferrum_interfaces::execution_cost::GuardedNotSubmittedReason;
use ferrum_interfaces::vnext::{
    CheckpointAuthorityId, CompletionSlotId, NativeCheckpointObservationSink,
    NativeCheckpointTransferIdentity, NativeCheckpointTransferKind,
    NativeCheckpointTransferObservation, RequestAuthorityId, SequenceAuthorityId,
};
use ferrum_scheduler::implementations::continuous::slo_planner::{
    PlanningTimeError, PlanningUnknownReason, RequestWorkKey,
};
use ferrum_types::{ObservabilityProfileDetail, RequestId};
use serde::Serialize;
use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

#[derive(Clone, Serialize)]
pub(in crate::continuous_engine) struct Owner {
    request_id: RequestId,
    incarnation: u64,
    work_generation: u64,
    generation_source: GenerationSource,
}
#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum GenerationSource {
    Scheduler,
    Engine,
}
impl Owner {
    pub(in crate::continuous_engine) fn new(
        request_id: &RequestId,
        incarnation: u64,
        work_generation: u64,
    ) -> Self {
        Self {
            request_id: request_id.clone(),
            incarnation,
            work_generation,
            generation_source: GenerationSource::Scheduler,
        }
    }
    pub(in crate::continuous_engine) fn from_engine(
        request_id: &RequestId,
        incarnation: u64,
        work_generation: u64,
    ) -> Self {
        Self {
            request_id: request_id.clone(),
            incarnation,
            work_generation,
            generation_source: GenerationSource::Engine,
        }
    }
}
impl From<&RequestWorkKey> for Owner {
    fn from(key: &RequestWorkKey) -> Self {
        Self {
            request_id: key.request_id.clone(),
            incarnation: key.incarnation,
            work_generation: key.work_generation.get(),
            generation_source: GenerationSource::Scheduler,
        }
    }
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::continuous_engine) enum Route {
    CacheCapture,
    ReadyRestore,
    RendezvousComparison,
    RendezvousContinuation,
}

#[derive(Clone, Copy)]
pub(in crate::continuous_engine) enum Decision {
    Known,
    PreferDirect,
    HoldRecommended,
    Unknown(PlanningUnknownReason),
    ClockError(PlanningTimeError),
}
impl Serialize for Decision {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Only the journal worker formats the typed scheduler reason.
        use serde::ser::SerializeStruct;
        let mut value = serializer.serialize_struct("PrefixDecision", 2)?;
        match self {
            Self::Known => value.serialize_field("outcome", "known")?,
            Self::PreferDirect => value.serialize_field("outcome", "prefer_direct")?,
            Self::HoldRecommended => value.serialize_field("outcome", "hold_recommended")?,
            Self::Unknown(reason) => {
                value.serialize_field("outcome", "unknown")?;
                value.serialize_field("reason", &format!("{reason:?}"))?;
            }
            Self::ClockError(reason) => {
                value.serialize_field("outcome", "clock_error")?;
                value.serialize_field("reason", &format!("{reason:?}"))?;
            }
        }
        value.end()
    }
}

#[derive(Serialize)]
pub(in crate::continuous_engine) struct NativeIdentity {
    slot: CompletionSlotId,
    checkpoint: CheckpointAuthorityId,
    sequence: SequenceAuthorityId,
    request: RequestAuthorityId,
    boundary_tokens: u64,
    kind: &'static str,
}
impl From<&NativeCheckpointTransferIdentity> for NativeIdentity {
    fn from(identity: &NativeCheckpointTransferIdentity) -> Self {
        Self {
            slot: identity.slot_id(),
            checkpoint: identity.checkpoint_authority(),
            sequence: identity.sequence_authority(),
            request: identity.request_authority(),
            boundary_tokens: identity.boundary_tokens(),
            kind: match identity.kind() {
                NativeCheckpointTransferKind::Capture => "capture",
                NativeCheckpointTransferKind::Restore => "restore",
            },
        }
    }
}

#[derive(Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub(in crate::continuous_engine) enum Event {
    Decision {
        route: Route,
        owner: Owner,
        peer: Option<Owner>,
        boundary_tokens: u32,
        snapshot_generation: u64,
        inference_epoch: u64,
        maintenance_epoch: u64,
        decision: Decision,
    },
    NativeGuard {
        owner: Owner,
        identity: NativeIdentity,
        source_capture: Option<NativeIdentity>,
        inference_epoch: Option<u64>,
        maintenance_epoch: Option<u64>,
        rejection: Option<GuardedNotSubmittedReason>,
    },
    /// Capture: model publication acknowledged. Restore: target acknowledged.
    /// Epochs/engine owner join through NativeGuard's exact transfer identity.
    /// Startup may lack a predictive guard, so this record invents no epoch.
    NativeAcknowledged {
        identity: NativeIdentity,
        source_capture: Option<NativeIdentity>,
        host_prefix_tokens: Option<u64>,
        host_full_input_tokens: Option<u64>,
        measured_product_wall_ns: u128,
        learning_accepted: bool,
    },
    Wave {
        owner: Owner,
        stage: WaveStage,
        inference_epoch: Option<u64>,
        maintenance_epoch: u64,
        prefill_start: Option<usize>,
        prefill_tokens: Option<usize>,
        final_prefill: Option<bool>,
    },
    RecordingSummary {
        offered: u64,
        accepted: u64,
        dropped: u64,
    },
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::continuous_engine) enum WaveStage {
    BackendSubmitted,
    HostReconciled,
}

struct Context {
    entrypoint: ProfileEntrypoint,
    model: String,
}
pub(super) struct Record {
    ordinal: u64,
    timestamp: chrono::DateTime<chrono::Utc>,
    context: Arc<Context>,
    event: Event,
}
impl Serialize for Record {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Match the existing profile envelope. JSON construction and formatting
        // happen only on its writer thread, never in a native final guard.
        let request_id = match &self.event {
            Event::Decision { owner, .. }
            | Event::NativeGuard { owner, .. }
            | Event::Wave { owner, .. } => owner.request_id.to_string(),
            Event::NativeAcknowledged { identity, .. } => format!(
                "native-checkpoint-{}-{}",
                identity.checkpoint.coordinator_id().get(),
                identity.checkpoint.serial()
            ),
            Event::RecordingSummary { .. } => "prefix-resource-recording".to_owned(),
        };
        let timestamp_nanos = self
            .timestamp
            .timestamp_nanos_opt()
            .unwrap_or_else(|| self.timestamp.timestamp_micros() * 1_000);
        let attributes = BTreeMap::from([
            (
                "execution_trace_source".to_owned(),
                serde_json::json!("prefix_resource"),
            ),
            ("prefix_resource_schema".to_owned(), serde_json::json!(1)),
            ("ordinal".to_owned(), serde_json::json!(self.ordinal)),
            (
                "prefix".to_owned(),
                serde_json::to_value(&self.event).map_err(serde::ser::Error::custom)?,
            ),
        ]);
        let shape = match &self.event {
            Event::Decision {
                route,
                boundary_tokens,
                ..
            } => BTreeMap::from([
                (
                    "route".to_owned(),
                    serde_json::to_value(route).map_err(serde::ser::Error::custom)?,
                ),
                (
                    "candidate_boundary_tokens".to_owned(),
                    serde_json::json!(boundary_tokens),
                ),
            ]),
            Event::NativeGuard { identity, .. } | Event::NativeAcknowledged { identity, .. } => {
                BTreeMap::from([
                    ("transfer_kind".to_owned(), serde_json::json!(identity.kind)),
                    (
                        "actual_boundary_tokens".to_owned(),
                        serde_json::json!(identity.boundary_tokens),
                    ),
                ])
            }
            Event::Wave {
                stage,
                prefill_start,
                prefill_tokens,
                final_prefill,
                ..
            } => BTreeMap::from([
                (
                    "stage".to_owned(),
                    serde_json::to_value(stage).map_err(serde::ser::Error::custom)?,
                ),
                (
                    "wave_row_kind".to_owned(),
                    serde_json::json!(if prefill_tokens.is_some() {
                        "prefill"
                    } else {
                        "decode"
                    }),
                ),
                ("prefill_start".to_owned(), serde_json::json!(prefill_start)),
                (
                    "prefill_tokens".to_owned(),
                    serde_json::json!(prefill_tokens),
                ),
                ("final_prefill".to_owned(), serde_json::json!(final_prefill)),
            ]),
            Event::RecordingSummary {
                offered,
                accepted,
                dropped,
            } => BTreeMap::from([
                ("offered".to_owned(), serde_json::json!(offered)),
                ("accepted".to_owned(), serde_json::json!(accepted)),
                ("dropped".to_owned(), serde_json::json!(dropped)),
            ]),
        };
        let event = FerrumProfileEvent {
            schema_version: OBSERVABILITY_PROFILE_SCHEMA_VERSION,
            ts_unix_nanos: timestamp_nanos,
            event_id: format!("evt-prefix-resource-{}-{timestamp_nanos}", self.ordinal),
            correlation_id: Some(request_id.clone()),
            request_id,
            entrypoint: self.context.entrypoint,
            backend: "actual".to_owned(),
            runtime_preset_hash: ENGINE_RUNTIME_TRACE_PRESET_HASH.to_owned(),
            phase: "slo.prefix_resource".to_owned(),
            event_kind: ProfileEventKind::Instant,
            timestamp: self.timestamp,
            status: ProfileStatus::Ok,
            model: Some(self.context.model.clone()),
            duration_us: None,
            memory: None,
            resource: None,
            error: None,
            replay: None,
            shape,
            backend_detail: None,
            attributes,
        };
        event.validate().map_err(serde::ser::Error::custom)?;
        event.serialize(serializer)
    }
}
#[derive(Default)]
struct Counters {
    offered: AtomicU64,
    accepted: AtomicU64,
    dropped: AtomicU64,
}

#[derive(Clone)]
pub(in crate::continuous_engine) struct Recorder {
    journal: SchedulerTraceJournal,
    counters: Arc<Counters>,
    context: Arc<Context>,
}
impl Recorder {
    pub(in crate::continuous_engine) fn for_detail(
        config: &EngineConfig,
        entrypoint: ProfileEntrypoint,
        journal: Option<&SchedulerTraceJournal>,
    ) -> Option<Self> {
        if !matches!(
            config.runtime.profile_detail,
            ObservabilityProfileDetail::Resource
                | ObservabilityProfileDetail::Debug
                | ObservabilityProfileDetail::Full
        ) {
            return None;
        }
        Some(Self {
            journal: journal?.clone(),
            counters: Arc::new(Counters::default()),
            context: Arc::new(Context {
                entrypoint,
                model: config.model.model_id.to_string(),
            }),
        })
    }
    pub(in crate::continuous_engine) fn record(&self, event: Event) {
        let ordinal = self.counters.offered.fetch_add(1, Ordering::Relaxed);
        let record = Record {
            ordinal,
            timestamp: chrono::Utc::now(),
            context: self.context.clone(),
            event,
        };
        if self
            .journal
            .inner
            .try_enqueue(super::SchedulerTraceRecord::Prefix(record))
            .is_ok()
        {
            self.counters.accepted.fetch_add(1, Ordering::Relaxed);
        } else {
            self.counters.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
    /// Called after native/cost shutdown on the existing blocking close task.
    /// A missing summary means this recording is incomplete, not zero activity.
    pub(in crate::continuous_engine) fn finish(
        &self,
    ) -> Result<(), ferrum_bench_core::jsonl_journal::JsonlJournalError> {
        let offered = self.counters.offered.load(Ordering::Acquire);
        self.journal
            .inner
            .enqueue(super::SchedulerTraceRecord::Prefix(Record {
                ordinal: offered,
                timestamp: chrono::Utc::now(),
                context: self.context.clone(),
                event: Event::RecordingSummary {
                    offered,
                    accepted: self.counters.accepted.load(Ordering::Acquire),
                    dropped: self.counters.dropped.load(Ordering::Acquire),
                },
            }))
    }
}

/// Profile-only adapter: the original learning sink and its accounted memory
/// are unchanged. The engine pins this adapter while native code holds a Weak.
pub(in crate::continuous_engine) struct NativeSink {
    inner: Arc<dyn NativeCheckpointObservationSink>,
    recorder: Recorder,
}
impl NativeSink {
    pub(in crate::continuous_engine) fn wrap(
        inner: Arc<dyn NativeCheckpointObservationSink>,
        recorder: Recorder,
    ) -> Arc<dyn NativeCheckpointObservationSink> {
        Arc::new(Self { inner, recorder })
    }
}
impl NativeCheckpointObservationSink for NativeSink {
    fn try_record(&self, receipt: NativeCheckpointTransferObservation) -> bool {
        // Normalize scalars before moving the unique genuine receipt into the
        // original FIFO. No source checkpoint or allocation is retained.
        let mut event = Event::NativeAcknowledged {
            identity: receipt.identity().into(),
            source_capture: receipt.source_capture_identity().map(Into::into),
            host_prefix_tokens: receipt.host_work().map(|work| work.prefix_tokens()),
            host_full_input_tokens: receipt.host_work().map(|work| work.full_input_tokens()),
            measured_product_wall_ns: receipt.wall_elapsed().as_nanos(),
            learning_accepted: false,
        };
        let accepted = self.inner.try_record(receipt);
        if let Event::NativeAcknowledged {
            learning_accepted, ..
        } = &mut event
        {
            *learning_accepted = accepted;
        }
        self.recorder.record(event);
        accepted
    }
}

#[cfg(test)]
mod tests;
