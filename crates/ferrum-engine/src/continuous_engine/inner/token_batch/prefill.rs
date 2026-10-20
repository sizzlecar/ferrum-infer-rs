use super::*;

impl EngineInner {
    pub(in crate::continuous_engine) fn plan_prompt_tail_prefill_chunk(
        &self,
        request_id: &RequestId,
        chunk: ferrum_execution::model_executor::PrefillChunk,
        original_prompt: bool,
        policy: PrefillShapePolicy,
        already_projected: bool,
    ) -> Result<ferrum_execution::model_executor::PrefillChunk> {
        let planned = policy.project(chunk, original_prompt)?.chunk();
        // The same immutable policy priced this selected range. A second
        // projection must not silently change work after time admission.
        if already_projected && planned != chunk {
            return Err(FerrumError::internal(
                "selected prefill range differs from its projected execution shape",
            ));
        }
        if planned == chunk {
            return Ok(chunk);
        }
        if self.scheduler_trace_jsonl.is_some() {
            self.write_executor_scheduler_profile_event(
                request_id,
                "vnext.prefill_prompt_tail_boundary_planned",
                ProfileEventKind::Instant,
                ProfileStatus::Ok,
                None,
                || SchedulerProfileFields {
                    shape: BTreeMap::from([
                        (
                            "tokens_processed".to_owned(),
                            serde_json::json!(chunk.tokens_processed()),
                        ),
                        (
                            "scheduled_tokens".to_owned(),
                            serde_json::json!(chunk.tokens_to_process()),
                        ),
                        (
                            "planned_tokens".to_owned(),
                            serde_json::json!(planned.tokens_to_process()),
                        ),
                        (
                            "capture_boundary".to_owned(),
                            serde_json::json!(planned.end()),
                        ),
                        (
                            "capture_span".to_owned(),
                            serde_json::json!(match policy {
                                PrefillShapePolicy::PromptTail(boundary) => Some(boundary.prefix()),
                                _ => None,
                            }),
                        ),
                    ]),
                    attributes: BTreeMap::new(),
                    error: None,
                },
            );
        }
        Ok(planned)
    }

    pub(in crate::continuous_engine) fn trace_plan_runtime_prefill_completion(
        &self,
        request_id: &RequestId,
        chunk: ferrum_execution::model_executor::PrefillChunk,
    ) {
        if self.scheduler_trace_jsonl.is_none() {
            return;
        }
        // Emit only after the executor result and both outer progress owners
        // have committed. Imported tokens are recorded by prefix_restore;
        // these ranges account exclusively for work actually executed here.
        self.write_executor_scheduler_profile_event(
            request_id,
            "vnext.prefill_chunk_completed",
            ProfileEventKind::Instant,
            ProfileStatus::Ok,
            None,
            || SchedulerProfileFields {
                shape: BTreeMap::from([
                    (
                        "start_token".to_owned(),
                        serde_json::json!(chunk.tokens_processed()),
                    ),
                    ("end_token".to_owned(), serde_json::json!(chunk.end())),
                    (
                        "computed_tokens".to_owned(),
                        serde_json::json!(chunk.tokens_to_process()),
                    ),
                    (
                        "prompt_tokens".to_owned(),
                        serde_json::json!(chunk.total_prompt_tokens()),
                    ),
                ]),
                attributes: BTreeMap::new(),
                error: None,
            },
        );
    }
}
