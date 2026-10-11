use super::*;
use ferrum_interfaces::{
    DecodeCycleCallContext, DecodeCycleDrainOwner, DecodeCycleOutcome,
    DecodeCycleParticipantWitness, DecodeCycleScope, DecodeCycleStage, DecodeCycleTrace,
    DecodeCycleTraceError,
};
use std::num::NonZeroUsize;

// Bounded diagnostic storage. Saturation is reported as incomplete coverage,
// never as a successful capture and never as inference backpressure.
const CYCLE_EVENT_CAPACITY: usize = 1_048_576;

pub(super) struct CycleDiagnostic {
    trace: DecodeCycleTrace,
    drain: Mutex<Option<DecodeCycleDrainOwner>>,
    journal: SchedulerTraceJournal,
    next_call: AtomicU64,
}

impl CycleDiagnostic {
    pub(super) fn new(path: &Path) -> Result<Self> {
        let capacity = NonZeroUsize::new(CYCLE_EVENT_CAPACITY)
            .ok_or_else(|| FerrumError::config("decode cycle capacity must be positive"))?;
        let (trace, drain) = DecodeCycleTrace::new(capacity)
            .map_err(|error| FerrumError::config(error.to_string()))?;
        let journal = SchedulerTraceJournal::create(path).map_err(|error| {
            FerrumError::io(format!(
                "open decode cycle JSONL {}: {error}",
                path.display()
            ))
        })?;
        Ok(Self {
            trace,
            drain: Mutex::new(Some(drain)),
            journal,
            next_call: AtomicU64::new(0),
        })
    }

    pub(super) fn sink(&self) -> Arc<dyn ExecutionEventSink> {
        Arc::new(CycleExecutionEventSink(self.trace.clone()))
    }
}

struct CycleExecutionEventSink(DecodeCycleTrace);

impl ExecutionEventSink for CycleExecutionEventSink {
    fn decode_cycle_trace(&self) -> Option<DecodeCycleTrace> {
        Some(self.0.clone())
    }

    fn enablement(&self) -> ferrum_interfaces::vnext::ExecutionEventSinkEnablement {
        ferrum_interfaces::vnext::ExecutionEventSinkEnablement::None
    }

    fn is_enabled(&self, _: VNextExecutionEventKind) -> bool {
        false
    }

    fn device_timing_mode(&self) -> ferrum_interfaces::vnext::DeviceTimingMode {
        ferrum_interfaces::vnext::DeviceTimingMode::Off
    }

    fn record(&self, _: EventEmissionPermit) -> std::result::Result<(), ExecutionEventSinkError> {
        Ok(())
    }
}

impl EngineInner {
    pub(super) fn decode_cycle_iteration_mark(&self, iteration: u64, stage: DecodeCycleStage) {
        let Some(cycle) = self.cycle_diagnostic.as_ref() else {
            return;
        };
        cycle
            .trace
            .mark(stage, DecodeCycleScope::Iteration(iteration));
    }

    pub(super) fn decode_cycle_call_contexts(
        &self,
        inputs: &mut [PlanRuntimeDecodeInput],
    ) -> Option<DecodeCycleCallContext> {
        let cycle = self.cycle_diagnostic.as_ref()?;
        let raw_call = cycle.next_call.fetch_add(1, Ordering::Relaxed);
        let Ok(call) = u32::try_from(raw_call) else {
            cycle
                .trace
                .invalidate(DecodeCycleTraceError::ClockExhausted);
            return None;
        };
        let iteration = self
            .iteration_count
            .load(Ordering::Relaxed)
            .saturating_sub(1);
        for (ordinal, input) in inputs.iter_mut().enumerate() {
            let Ok(ordinal) = u32::try_from(ordinal) else {
                cycle
                    .trace
                    .invalidate(DecodeCycleTraceError::InvalidCallerMap);
                return None;
            };
            input.decode_cycle_context = Some(cycle.trace.call_context(iteration, call, ordinal));
        }
        inputs.first().and_then(|input| input.decode_cycle_context)
    }

    pub(super) fn decode_cycle_call_mark(
        &self,
        context: Option<DecodeCycleCallContext>,
        stage: DecodeCycleStage,
        outcome: Option<DecodeCycleOutcome>,
    ) {
        let (Some(cycle), Some(context)) = (self.cycle_diagnostic.as_ref(), context) else {
            return;
        };
        let scope = DecodeCycleScope::Call(context);
        match outcome {
            Some(outcome) => {
                cycle.trace.mark_with_outcome(stage, scope, outcome);
            }
            None => {
                cycle.trace.mark(stage, scope);
            }
        }
    }

    pub(super) fn decode_cycle_participant_mark(
        &self,
        witness: Option<&DecodeCycleParticipantWitness>,
        stage: DecodeCycleStage,
        outcome: Option<DecodeCycleOutcome>,
    ) {
        let (Some(cycle), Some(witness)) = (self.cycle_diagnostic.as_ref(), witness) else {
            return;
        };
        let scope = DecodeCycleScope::Participant(witness.clone());
        match outcome {
            Some(outcome) => {
                cycle.trace.mark_with_outcome(stage, scope, outcome);
            }
            None => {
                cycle.trace.mark(stage, scope);
            }
        }
    }

    /// Export only after the engine driver has joined. Worker producer guards
    /// make a canceled/detached completion tail explicitly incomplete.
    pub(super) async fn close_decode_cycle_capture(&self) -> Result<()> {
        let Some(cycle) = self.cycle_diagnostic.as_ref() else {
            return Ok(());
        };
        let Some(drain) = cycle.drain.lock().take() else {
            return Ok(());
        };
        let journal = cycle.journal.clone();
        let entrypoint = self
            .runtime_config
            .profile_entrypoint
            .unwrap_or(ProfileEntrypoint::Synthetic);
        let model = self.config.model.model_id.to_string();
        let backend = format!("{:?}", self.config.backend.device);
        tokio::task::spawn_blocking(move || {
            let capture = drain.finish();
            let collector_complete = capture.has_complete_coverage();
            let shape = BTreeMap::from([
                (
                    "collected_events".to_string(),
                    serde_json::json!(capture.events.len()),
                ),
                (
                    "event_capacity".to_string(),
                    serde_json::json!(capture.capacity),
                ),
            ]);
            let timestamp = chrono::Utc::now();
            let mut attributes = BTreeMap::new();
            attributes.insert(
                "clock_source".to_string(),
                serde_json::json!(DecodeCycleTrace::clock_source()),
            );
            attributes.insert(
                "collector_complete".to_string(),
                serde_json::json!(collector_complete),
            );
            attributes.insert(
                "boundary_coverage_requires_validation".to_string(),
                serde_json::json!(true),
            );
            attributes.insert(
                "capture".to_string(),
                serde_json::to_value(capture).map_err(|error| {
                    FerrumError::internal(format!("serialize decode cycle capture: {error}"))
                })?,
            );
            let event = FerrumProfileEvent {
                schema_version: OBSERVABILITY_PROFILE_SCHEMA_VERSION,
                ts_unix_nanos: timestamp.timestamp_nanos_opt().unwrap_or(i64::MAX),
                event_id: "evt-decode-cycle-capture".to_string(),
                request_id: "engine".to_string(),
                correlation_id: Some("decode-cycle".to_string()),
                entrypoint,
                backend,
                runtime_preset_hash: ENGINE_RUNTIME_TRACE_PRESET_HASH.to_string(),
                phase: "engine.decode_cycle_capture".to_string(),
                event_kind: ProfileEventKind::Instant,
                timestamp,
                status: ProfileStatus::DiagnosticOnly,
                model: Some(model),
                duration_us: None,
                memory: None,
                resource: None,
                error: None,
                replay: None,
                shape,
                backend_detail: None,
                attributes,
            };
            let enqueue = journal
                .enqueue(event)
                .map_err(|error| FerrumError::io(format!("write decode cycle capture: {error}")));
            let close = journal
                .close()
                .map_err(|error| FerrumError::io(format!("close decode cycle capture: {error}")));
            enqueue?;
            close
        })
        .await
        .map_err(|error| {
            FerrumError::internal(format!("decode cycle drain task failed: {error}"))
        })?
    }
}
