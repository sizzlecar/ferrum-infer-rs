//! Host observations use the original Engine clocks and instance journal.
use super::*;

pub(in crate::continuous_engine) enum EngineHostObservation<'a> {
    Iteration {
        iteration: u64,
        batch_size: usize,
        started: Instant,
        scheduled: Instant,
        completed: Instant,
        error: Option<&'a FerrumError>,
    },
    FirstStreamText {
        request_id: &'a RequestId,
        prompt_tokens: usize,
        generated_tokens: usize,
        ingress: RequestIngressAnchor,
        committed_at: Instant,
    },
}

impl EngineInner {
    pub(in crate::continuous_engine) fn record_host_observation(
        &self,
        observation: EngineHostObservation<'_>,
    ) {
        let Some(sink) = &self.host_profile_sink else {
            return;
        };
        #[cfg(test)]
        self.trace_payload_builds.fetch_add(1, Ordering::Relaxed);
        let id = self
            .resource_trace_event_counter
            .fetch_add(1, Ordering::Relaxed);
        if let Err(error) = sink.record_host_observation(id, observation) {
            self.report_observation_failure(error);
        }
    }
}

impl VNextProfileExecutionEventSink {
    fn record_host_observation(
        &self,
        id: u64,
        observation: EngineHostObservation<'_>,
    ) -> std::result::Result<(), ExecutionEventSinkError> {
        let micros = |duration: Duration| duration.as_micros().min(u64::MAX as u128) as u64;
        let (phase, request_id, duration_us, shape, mut attributes, error) = match observation {
            EngineHostObservation::Iteration {
                iteration,
                batch_size,
                started,
                scheduled,
                completed,
                error,
            } => (
                "iter_prof",
                format!("engine-iteration-{iteration}"),
                micros(completed.saturating_duration_since(started)),
                BTreeMap::from([
                    ("iteration".into(), serde_json::json!(iteration)),
                    ("batch_size".into(), serde_json::json!(batch_size)),
                ]),
                BTreeMap::from([
                    (
                        "schedule_us".into(),
                        serde_json::json!(micros(scheduled.saturating_duration_since(started))),
                    ),
                    (
                        "process_us".into(),
                        serde_json::json!(micros(completed.saturating_duration_since(scheduled))),
                    ),
                    (
                        "measurement_boundary".into(),
                        serde_json::json!("engine_iteration_host"),
                    ),
                ]),
                error.map(|error| ProfileError {
                    kind: "engine_iteration".into(),
                    message: error.to_string(),
                    blocking: false,
                }),
            ),
            EngineHostObservation::FirstStreamText {
                request_id,
                prompt_tokens,
                generated_tokens,
                ingress,
                committed_at,
            } => (
                "stream_ttft_prof",
                request_id.to_string(),
                micros(ingress.elapsed_at(committed_at)),
                BTreeMap::from([
                    ("prompt_tokens".into(), serde_json::json!(prompt_tokens)),
                    (
                        "generated_tokens".into(),
                        serde_json::json!(generated_tokens),
                    ),
                ]),
                BTreeMap::from([
                    (
                        "request_start_boundary".into(),
                        serde_json::json!(ingress.boundary().as_str()),
                    ),
                    (
                        "measurement_boundary".into(),
                        serde_json::json!("engine_first_nonempty_text_committed_to_output"),
                    ),
                    ("client_visible_latency".into(), serde_json::json!(false)),
                ]),
                None,
            ),
        };
        attributes.insert(
            "profile_detail".into(),
            serde_json::json!(self.context.profile_detail.as_str()),
        );
        let timestamp = chrono::Utc::now();
        self.enqueue_profile_batch(vec![FerrumProfileEvent {
            schema_version: OBSERVABILITY_PROFILE_SCHEMA_VERSION,
            ts_unix_nanos: timestamp
                .timestamp_nanos_opt()
                .unwrap_or_else(|| timestamp.timestamp_micros() * 1_000),
            event_id: format!("evt-engine-host-{id}"),
            correlation_id: Some(request_id.clone()),
            request_id,
            entrypoint: self.context.entrypoint,
            backend: "actual".into(),
            runtime_preset_hash: ENGINE_RUNTIME_TRACE_PRESET_HASH.into(),
            phase: phase.into(),
            event_kind: ProfileEventKind::TimedSpan,
            timestamp,
            status: if error.is_some() {
                ProfileStatus::Failure
            } else {
                ProfileStatus::Ok
            },
            model: Some(self.context.model.clone()),
            duration_us: Some(duration_us),
            memory: None,
            resource: None,
            error,
            replay: None,
            shape,
            backend_detail: None,
            attributes,
        }])
    }
}
