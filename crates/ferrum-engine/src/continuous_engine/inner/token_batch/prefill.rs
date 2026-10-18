use super::*;

impl EngineInner {
    pub(in crate::continuous_engine) fn plan_prompt_tail_prefill_chunk(
        &self,
        request_id: &RequestId,
        chunk: ferrum_execution::model_executor::PrefillChunk,
        original_prompt: bool,
    ) -> Result<ferrum_execution::model_executor::PrefillChunk> {
        if !original_prompt {
            return Ok(chunk);
        }
        let Some(plan) = self.model_executor.plan_prompt_tail_capture_boundary(chunk) else {
            return Ok(chunk);
        };
        let span = plan.boundary.checked_sub(chunk.tokens_processed());
        if plan.boundary >= chunk.total_prompt_tokens()
            || plan.boundary > chunk.end()
            || !span.is_some_and(|span| plan.span.permits(span as u64))
        {
            return Err(FerrumError::backend(
                "invalid prompt-tail capture boundary for scheduled prefill",
            ));
        }
        if plan.boundary == chunk.end() {
            return Ok(chunk);
        }
        let planned = ferrum_execution::model_executor::PrefillChunk::new(
            chunk.tokens_processed(),
            span.expect("validated positive capture span"),
            chunk.total_prompt_tokens(),
        )?;
        if self.scheduler_trace_jsonl.is_some() {
            self.write_executor_scheduler_profile_event(
                request_id,
                "vnext.prefill_prompt_tail_boundary_planned",
                ProfileEventKind::Instant,
                ProfileStatus::Ok,
                None,
                BTreeMap::from([
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
                        serde_json::json!(plan.boundary),
                    ),
                    ("capture_span".to_owned(), serde_json::json!(plan.span)),
                ]),
                BTreeMap::new(),
                None,
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
            BTreeMap::from([
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
            BTreeMap::new(),
            None,
        );
    }
}
