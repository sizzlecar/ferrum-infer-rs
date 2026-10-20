//! Projection of the original submission's work and observed terminal result.
//! Work is kept in node/participant order; it is not a list of successful GPU
//! launches. Device timestamps remain in their original timing events.
use super::*;
use ferrum_runtime::{OperationCompletionDisposition, TerminalExecutionFacts};

impl VNextProfileEventContext {
    pub(super) fn execution_facts_event(
        &self,
        facts: TerminalExecutionFacts<'_>,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) -> std::result::Result<FerrumProfileEvent, ExecutionEventSinkError> {
        let completion = facts.completion();
        let submission = completion.submission();
        let batch = submission.batch_identity();
        let mut nodes = Vec::with_capacity(facts.work().nodes().len());
        let mut first_request = None;
        for work in facts.work().nodes() {
            let node = usize::try_from(work.node_index())
                .ok()
                .and_then(|index| batch.nodes().get(index))
                .ok_or_else(|| {
                    ExecutionEventSinkError::contract("execution work references an absent node")
                })?;
            if work.participants().len() != node.participants().len() {
                return Err(ExecutionEventSinkError::contract(
                    "execution work participant count differs from its original node",
                ));
            }
            let participants = work
                .participants()
                .iter()
                .zip(node.participants())
                .enumerate()
                .map(|(index, (range, participant))| {
                    let identity = participant.identity().parts();
                    first_request.get_or_insert_with(|| identity.request_id.to_string());
                    serde_json::json!({
                        "participant_index": index,
                        "run_id": identity.run_id,
                        "request_id": identity.request_id,
                        "frame_id": identity.frame_id,
                        "node_invocation_id": identity.node_invocation_id,
                        "active_sequence_slot": identity.active_sequence_slot,
                        "admission_generation": identity.admission_generation,
                        "activation_epoch": identity.activation_epoch,
                        "source_tokens": range.source_token_range(),
                        "packed_tokens": range.immediate_token_range(),
                        "full_input_tokens": range.full_input_tokens(),
                    })
                })
                .collect::<Vec<_>>();
            nodes.push(serde_json::json!({
                "node_index": work.node_index(),
                "node_id": node.node_id(),
                "operation_id": node.operation_id(),
                "provider_id": node.provider_id(),
                "work_shape_fingerprint": node.work_shape_fingerprint(),
                "participants": participants,
            }));
        }
        let request_id = first_request.ok_or_else(|| {
            ExecutionEventSinkError::contract("terminal execution facts have no participant")
        })?;
        let succeeded = matches!(
            completion.disposition(),
            OperationCompletionDisposition::Succeeded
        );
        // Borrow the pure command descriptions and resolved current-work rows.
        // Serializing the bound receipt would repeat every participant identity.
        // This scope excludes transfers, initialization, and terminal readback;
        // a terminal success is not evidence of complete device-leaf coverage.
        let device_commands = facts.device_facts().map(|facts| {
            serde_json::json!({
                "scope": "compute_commands",
                "batch_command_count": facts.device().batch_command_count(),
                "commands": facts.device().compute_commands(),
                "projection_rows": facts.projection_rows(),
            })
        });
        let event = FerrumProfileEvent {
            schema_version: OBSERVABILITY_PROFILE_SCHEMA_VERSION,
            ts_unix_nanos: timestamp
                .timestamp_nanos_opt()
                .unwrap_or_else(|| timestamp.timestamp_micros() * 1_000),
            event_id: format!("evt-execution-facts-{}", completion.fingerprint()),
            request_id,
            correlation_id: Some(submission.fingerprint().to_owned()),
            entrypoint: self.entrypoint,
            backend: "actual".to_owned(),
            runtime_preset_hash: ENGINE_RUNTIME_TRACE_PRESET_HASH.to_owned(),
            phase: "vnext.execution_facts".to_owned(),
            event_kind: if succeeded {
                ProfileEventKind::Instant
            } else {
                ProfileEventKind::Error
            },
            timestamp,
            status: if succeeded {
                ProfileStatus::Ok
            } else {
                ProfileStatus::Failure
            },
            model: Some(self.model.clone()),
            duration_us: None,
            memory: None,
            resource: None,
            error: (!succeeded).then(|| ProfileError {
                kind: "execution_completion_failure".to_owned(),
                message: format!("{:?}", completion.disposition()),
                blocking: true,
            }),
            replay: None,
            shape: BTreeMap::from([("node_count".to_owned(), serde_json::json!(nodes.len()))]),
            backend_detail: Some(BTreeMap::from([
                (
                    "backend_device".to_owned(),
                    serde_json::json!(self.backend_device),
                ),
                (
                    "backend_type".to_owned(),
                    serde_json::json!(self.backend_type),
                ),
            ])),
            attributes: BTreeMap::from([
                (
                    "profile_detail".to_owned(),
                    serde_json::json!(self.profile_detail.as_str()),
                ),
                (
                    "diagnostic_only".to_owned(),
                    serde_json::json!(self.profile_detail.diagnostic_only()),
                ),
                ("plan_id".to_owned(), serde_json::json!(batch.plan_id())),
                ("plan_hash".to_owned(), serde_json::json!(batch.plan_hash())),
                (
                    "execution_device_id".to_owned(),
                    serde_json::json!(batch.device_id()),
                ),
                (
                    "runtime_implementation_fingerprint".to_owned(),
                    serde_json::json!(batch.runtime_implementation_fingerprint()),
                ),
                (
                    "submission_fingerprint".to_owned(),
                    serde_json::json!(submission.fingerprint()),
                ),
                (
                    "completion_fingerprint".to_owned(),
                    serde_json::json!(completion.fingerprint()),
                ),
                (
                    "completion_disposition".to_owned(),
                    serde_json::json!(completion.disposition()),
                ),
                ("submitted_node_work".to_owned(), serde_json::json!(nodes)),
                (
                    "device_facts_coverage".to_owned(),
                    serde_json::json!(facts.device_coverage()),
                ),
                (
                    "device_execution".to_owned(),
                    serde_json::json!(device_commands),
                ),
                (
                    "has_device_attribution".to_owned(),
                    serde_json::json!(facts.attribution().is_some()),
                ),
            ]),
        };
        event
            .validate()
            .map_err(ExecutionEventSinkError::contract)?;
        Ok(event)
    }
}

#[cfg(test)]
mod tests;
