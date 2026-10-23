//! Public scheduler journal projection. A selected plan and its planning
//! statistics share an iteration with the observed Engine process result.
//! These host spans are neither GPU kernel timings nor client-visible latency.

use super::*;

pub(super) struct IterationTrace<'a> {
    pub(super) iteration: u64,
    pub(super) started_ns: Option<u64>,
    pub(super) batch: Option<&'a ferrum_interfaces::BatchPlan>,
    pub(super) plan: Option<&'a SchedulerTracePlanStats>,
    pub(super) planning: Option<&'a TimePlanningRound>,
    pub(super) schedule_elapsed: Duration,
    pub(super) process_started_ns: Option<u64>,
    pub(super) process_completed_ns: Option<u64>,
    pub(super) process_elapsed: Option<Duration>,
    pub(super) process_result: Option<&'a Result<()>>,
    pub(super) post_process_result: &'a Result<()>,
    pub(super) prefill_tokens_before: Option<u64>,
    pub(super) decode_tokens_before: Option<u64>,
}

impl EngineInner {
    pub(super) fn trace_iteration(&self, trace: IterationTrace<'_>) {
        // Do not inspect the plan, clone statistics or build JSON when the
        // public trace is disabled. Legacy-only tracing keeps its old writer.
        if self.scheduler_trace_jsonl.is_none() {
            return;
        }
        let no_request = RequestId(uuid::Uuid::nil());
        let request_id = trace
            .batch
            .and_then(|batch| batch.requests.first())
            .map_or(&no_request, |request| &request.request.id);
        let process_error = trace
            .process_result
            .and_then(|result| result.as_ref().err());
        let post_process_error = trace.post_process_result.as_ref().err();
        let error = post_process_error.or(process_error);
        let result = match (trace.batch.is_some(), error.is_some()) {
            (true, false) => "some_ok",
            (true, true) => "some_error",
            (false, false) => "none",
            (false, true) => "none_error",
        };
        self.write_executor_scheduler_profile_event(
            request_id,
            "scheduler.iteration",
            if error.is_some() {
                ProfileEventKind::Error
            } else if trace.batch.is_some() {
                ProfileEventKind::TimedSpan
            } else {
                ProfileEventKind::Instant
            },
            if error.is_some() {
                ProfileStatus::Failure
            } else {
                ProfileStatus::Ok
            },
            trace.process_elapsed.map(duration_to_us),
            || {
                let prefill_after = self.total_prefill_tokens.load(Ordering::Relaxed);
                let decode_after = self.total_decode_tokens.load(Ordering::Relaxed);
                SchedulerProfileFields {
                    shape: BTreeMap::from([(
                        "batch_size".to_string(),
                        serde_json::json!(trace.batch.map_or(0, |batch| batch.size())),
                    )]),
                    attributes: BTreeMap::from([
                        ("iteration".to_string(), serde_json::json!(trace.iteration)),
                        ("result".to_string(), serde_json::json!(result)),
                        (
                            "batch_id".to_string(),
                            serde_json::json!(trace.batch.map(|batch| &batch.batch_id)),
                        ),
                        ("plan".to_string(), serde_json::json!(trace.plan)),
                        (
                            "slo_planning".to_string(),
                            serde_json::json!(trace.planning.map(TimePlanningRound::diagnostics)),
                        ),
                        (
                            "process_error".to_string(),
                            serde_json::json!(process_error.map(ToString::to_string)),
                        ),
                        (
                            "post_process_error".to_string(),
                            serde_json::json!(post_process_error.map(ToString::to_string)),
                        ),
                        (
                            "timing_us".to_string(),
                            serde_json::json!({
                                "schedule": duration_to_us(trace.schedule_elapsed),
                                "process": trace.process_elapsed.map(duration_to_us),
                            }),
                        ),
                        (
                            "boundaries".to_string(),
                            serde_json::json!({
                                "iteration_started_monotonic_ns": trace.started_ns,
                                "process_started_monotonic_ns": trace.process_started_ns,
                                "process_completed_monotonic_ns": trace.process_completed_ns,
                            }),
                        ),
                        (
                            "engine_counters".to_string(),
                            serde_json::json!({
                                "prefill_tokens_before": trace.prefill_tokens_before,
                                "prefill_tokens_after": prefill_after,
                                "prefill_tokens_delta": trace.prefill_tokens_before
                                    .map(|before| prefill_after.saturating_sub(before)),
                                "decode_tokens_before": trace.decode_tokens_before,
                                "decode_tokens_after": decode_after,
                                "decode_tokens_delta": trace.decode_tokens_before
                                    .map(|before| decode_after.saturating_sub(before)),
                            }),
                        ),
                        (
                            "measurement_scope".to_string(),
                            serde_json::json!("Final scheduled input ranges and Engine host process result; not proof that every selected request executed successfully, GPU kernel timing, or client-visible causality. A no-batch event uses the nil request ID."),
                        ),
                    ]),
                    error: error.map(|error| ProfileError {
                        kind: "scheduler_iteration_process".to_string(),
                        message: error.to_string(),
                        blocking: true,
                    }),
                }
            },
        );
    }
}
