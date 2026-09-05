use super::*;

impl SequencePhysicalResources {
    pub(super) fn is_empty(&self) -> bool {
        self.legacy_kv_allocation.is_none()
            && self.legacy_draft_kv_allocation.is_none()
            && self.recurrent_state_allocation.is_none()
            && self.model_cache_id.is_none()
    }
}

impl SequenceState {
    /// Return the first matched stop, or hold the longest suffix that could
    /// still become a stop. Character boundaries keep Unicode stops safe.
    fn visible_text_end(&self, text: &str, terminal: bool) -> usize {
        if let Some(end) = self
            .stop_text_seqs
            .iter()
            .filter(|stop| !stop.is_empty())
            .filter_map(|stop| text.find(stop.as_str()))
            .min()
        {
            return end;
        }
        if terminal {
            return text.len();
        }
        let held = self
            .stop_text_seqs
            .iter()
            .flat_map(|stop| {
                stop.char_indices()
                    .skip(1)
                    .map(move |(len, _)| &stop[..len])
            })
            .filter(|prefix| text.ends_with(prefix))
            .map(str::len)
            .max()
            .unwrap_or(0);
        text.len() - held
    }

    fn decoded_output_text(
        &self,
        tokenizer: &(dyn Tokenizer + Send + Sync),
        terminal: Option<FinishReason>,
    ) -> Result<String> {
        let mut text = tokenizer.decode(&self.generated_tokens, true)?;
        let end = self.visible_text_end(&text, terminal.is_some());
        if end < text.len() {
            text.truncate(end);
        } else if let (Some(reason), Some(&last)) = (terminal, self.generated_tokens.last()) {
            // A tokenizer may retain a model terminal even with skip_special.
            // Keep Harmony's typed terminators, but hide ordinary EOS/stop IDs.
            if self.stop_token_ids.contains(&last.get())
                && !self.should_stream_generated_token(Some(tokenizer), last, Some(reason))
            {
                text = tokenizer.decode(
                    &self.generated_tokens[..self.generated_tokens.len() - 1],
                    true,
                )?;
            }
        }
        Ok(text)
    }

    fn validate_structured_stop_boundary(
        &self,
        tokenizer: &(dyn Tokenizer + Send + Sync),
    ) -> Result<()> {
        let Some(processor) = self.structured_output_processor.as_ref() else {
            return Ok(());
        };
        if self.stop_text_seqs.is_empty() {
            return Ok(());
        }
        let full = tokenizer.decode(&self.generated_tokens, true)?;
        let stop_end = self.visible_text_end(&full, true);
        if stop_end == full.len() {
            return Ok(());
        }
        // The grammar already validated the generated value. A text stop must
        // not cut *inside* that value, even if the shorter prefix also parses
        // (for example, cutting the final digit from a schema-constrained 123).
        // Use its typed activation boundary, so stops in reasoning are caught
        // without guessing a model-specific reasoning delimiter.
        let progress =
            processor.progress_with_terminals(&self.generated_tokens, &self.stop_token_ids)?;
        let start = self.generated_tokens.len() - progress.grammar_token_count;
        let prefix_len = tokenizer
            .decode(&self.generated_tokens[..start], true)?
            .len();
        // The grammar accepts engine-owned terminal IDs as framing after the
        // value. Some tokenizers still render them with skip_special enabled.
        let end = self.generated_tokens.len()
            - usize::from(
                self.generated_tokens
                    .last()
                    .is_some_and(|token| self.stop_token_ids.contains(&token.get())),
            );
        let grammar_text = tokenizer.decode(&self.generated_tokens[start..end], true)?;
        let mut values =
            serde_json::Deserializer::from_str(&grammar_text).into_iter::<serde_json::Value>();
        values
            .next()
            .ok_or_else(|| FerrumError::model("structured output has no JSON value"))?
            .map_err(|_| {
                FerrumError::model("structured output did not contain a complete JSON value")
            })?;
        if stop_end < prefix_len + values.byte_offset() {
            return Err(FerrumError::model(
                "stop sequence truncated the structured output before its complete JSON value",
            ));
        }
        Ok(())
    }

    pub(in crate::continuous_engine) fn client_receiver_closed(&self) -> bool {
        self.response_sender
            .as_ref()
            .is_some_and(tokio::sync::oneshot::Sender::is_closed)
            || self
                .stream_sender
                .as_ref()
                .is_some_and(tokio::sync::mpsc::Sender::is_closed)
    }

    pub(in crate::continuous_engine) fn structured_output_terminal_error(
        &self,
        finish_reason: FinishReason,
    ) -> Option<FerrumError> {
        let processor = self.structured_output_processor.as_ref()?;
        match processor
            .progress_with_terminals(&self.generated_tokens, &self.stop_token_ids)
        {
            Ok(progress) if progress.accepting && finish_reason != FinishReason::Error => None,
            Ok(progress) if progress.accepting => Some(FerrumError::model(
                "structured-output generation failed after reaching an accepting state",
            )),
            Ok(progress) => Some(FerrumError::model(format!(
                "structured-output generation ended with {finish_reason:?} before a complete valid value: phase={:?}, generated_tokens={}, consumed_tokens={}, delimiter_tokens={:?}, delimiter_prefix_tokens={}, reasoning_tokens={:?}, boundary_forced={}, budget={:?}, grammar_tokens={}, sampling_history_scope={:?}, sampling_history_tokens={}, sampling_history_unique_tokens={}, trailing_class={:?}, trailing_class_tokens={}, trailing_token_id={:?}, trailing_identical_tokens={}, liveness_identical_token_limit={}, liveness_interventions={}",
                progress.phase,
                progress.generated_token_count,
                progress.consumed_token_count,
                progress.delimiter_token_count,
                progress.delimiter_prefix_token_count,
                progress.reasoning_token_count,
                progress.boundary_forced,
                progress.budget,
                progress.grammar_token_count,
                self.sampling_history.scope(),
                self.sampling_history.token_count(&self.generated_tokens),
                self.sampling_history.unique_token_count(),
                progress.trailing_token_class,
                progress.trailing_token_class_count,
                progress.trailing_token_id,
                progress.trailing_identical_token_count,
                progress.liveness_identical_token_limit,
                progress.liveness_intervention_count,
            ))),
            Err(error) => Some(error),
        }
    }
}

impl EngineInner {
    fn capture_sequence_terminal_token_trace(
        &self,
        sequence: &SequenceState,
    ) -> Option<SequenceTokenTraceEvidence> {
        (self.scheduler_trace_jsonl.is_some()
            && self.config.runtime.profile_detail.diagnostic_only())
        .then(|| SequenceTokenTraceEvidence::capture(sequence))
    }

    fn write_sequence_terminal_token_trace(
        &self,
        request_id: &RequestId,
        termination: &str,
        finish_reason: Option<FinishReason>,
        evidence: Option<&SequenceTokenTraceEvidence>,
    ) {
        let Some(evidence) = evidence else {
            return;
        };
        let mut attributes = BTreeMap::from([
            ("termination".to_string(), serde_json::json!(termination)),
            (
                "token_trace".to_string(),
                serde_json::to_value(evidence).expect("typed sequence token trace must serialize"),
            ),
        ]);
        if let Some(finish_reason) = finish_reason {
            attributes.insert(
                "finish_reason".to_string(),
                serde_json::json!(finish_reason_trace_name(finish_reason)),
            );
        }
        self.write_executor_scheduler_profile_event(
            request_id,
            "engine_sequence_terminal_evidence",
            ProfileEventKind::Instant,
            ProfileStatus::DiagnosticOnly,
            None,
            BTreeMap::from([
                (
                    "prompt_token_count".to_string(),
                    serde_json::json!(evidence.prompt_token_count),
                ),
                (
                    "generated_token_count".to_string(),
                    serde_json::json!(evidence.generated_token_count),
                ),
            ]),
            attributes,
            None,
        );
    }

    // ── stream helper ──────────────────────────────────────────────────

    pub(super) fn stop_reason_for_request(&self, request_id: &RequestId) -> Option<FinishReason> {
        let sequences = self.sequences.read();
        match sequences.get(request_id) {
            Some(seq) => seq.stop_reason(Some(self.tokenizer.as_ref())),
            None => Some(FinishReason::Error),
        }
    }

    pub(super) fn should_stream_generated_token(
        &self,
        request_id: &RequestId,
        token: TokenId,
        stop_reason: Option<FinishReason>,
    ) -> bool {
        if !matches!(
            stop_reason,
            Some(FinishReason::Stop) | Some(FinishReason::EOS)
        ) {
            return !matches!(stop_reason, Some(FinishReason::Error));
        }
        self.sequences
            .read()
            .get(request_id)
            .is_some_and(|sequence| {
                sequence.should_stream_generated_token(
                    Some(self.tokenizer.as_ref()),
                    token,
                    stop_reason,
                )
            })
    }

    pub(super) async fn send_stream_update(&self, request_id: &RequestId, token: TokenId) {
        self.send_stream_text(request_id, Some(token), None).await;
    }

    async fn send_stream_text(
        &self,
        request_id: &RequestId,
        token: Option<TokenId>,
        terminal: Option<FinishReason>,
    ) {
        // Decode the full generated-token history (skip_special=true matches
        // the final-response decode in `complete_request`) and emit only
        // the delta that hasn't been streamed yet. Per-token decode is
        // wrong for any model whose vocab can split a multi-byte UTF-8
        // sequence across BPE pieces — Qwen3 / Qwen2.5 routinely do this
        // for Chinese chars and emoji, and the single-token decode then
        // returns a `\u{FFFD}` replacement char that renders as a square /
        // `?` glyph in the terminal.
        //
        // Algorithm: hold the write lock once to (a) clone sender, (b)
        // decode current full history, (c) if the decoded text ends in
        // `\u{FFFD}` defer the emit (a later token will complete the
        // multi-byte sequence), (d) otherwise carve off the substring
        // past `streamed_text_len` and bump the watermark. Possible stop
        // prefixes are held until they match, diverge, or generation ends.
        let (sender, delta, ttft_s, itl_s, first_emit_prof) = {
            let mut sequences = self.sequences.write();
            let Some(seq) = sequences.get_mut(request_id) else {
                return;
            };
            let sender = seq.stream_sender.clone();
            let Ok(mut full) = self.tokenizer.decode(&seq.generated_tokens, true) else {
                return;
            };
            let incomplete_utf8 = full.ends_with('\u{FFFD}');
            if incomplete_utf8 && terminal.is_none() {
                // Partial multi-byte UTF-8 at the tail; wait for the next
                // token. Do NOT advance streamed_text_len so the bytes get
                // re-considered once the sequence completes.
                return;
            }
            if !incomplete_utf8 {
                seq.decoded_text_len = full.len();
            }
            let visible = if terminal.is_some() {
                let Ok(text) = seq.decoded_output_text(self.tokenizer.as_ref(), terminal) else {
                    return;
                };
                text
            } else {
                let end = seq.visible_text_end(&full, false);
                full.truncate(end);
                full
            };
            if visible.ends_with('\u{FFFD}') {
                return;
            }
            let Some(delta) = visible.get(seq.streamed_text_len..) else {
                return;
            };
            let delta = delta.to_string();
            seq.streamed_text_len = visible.len();

            // Latency-metric tracking (PLAYBOOK § 7 definitions).
            // We capture timestamps in the critical section so the
            // first-emit point matches the moment we commit to streaming
            // the delta — not the moment the chunk actually crosses the
            // socket, which the engine can't observe.
            let mut ttft_s: Option<f64> = None;
            let mut itl_s: Option<f64> = None;
            let mut first_emit_prof: Option<(usize, usize, u64)> = None;
            if !delta.is_empty() {
                let now = Instant::now();
                match seq.first_emit_at {
                    None => {
                        let ttft = now.duration_since(seq.start_time);
                        ttft_s = Some(ttft.as_secs_f64());
                        if self.runtime_config.batch_decode_prof {
                            first_emit_prof = Some((
                                seq.input_tokens.len(),
                                seq.generated_tokens.len(),
                                ttft.as_micros() as u64,
                            ));
                        }
                        seq.first_emit_at = Some(now);
                    }
                    Some(_) => {
                        if let Some(prev) = seq.last_emit_at {
                            itl_s = Some(now.duration_since(prev).as_secs_f64());
                        }
                    }
                }
                seq.last_emit_at = Some(now);
                seq.emitted_chunks = seq.emitted_chunks.saturating_add(1);
            }

            (sender, delta, ttft_s, itl_s, first_emit_prof)
        };

        if let Some(t) = ttft_s {
            histogram!("ferrum.engine.ttft_seconds").record(t);
        }
        if let Some(t) = itl_s {
            histogram!("ferrum.engine.itl_seconds").record(t);
        }
        if let Some((prompt_tokens, generated_tokens, ttft_us)) = first_emit_prof {
            eprintln!(
                "[stream-ttft-prof] req={} prompt_tokens={} generated_tokens={} ttft={}us",
                request_id, prompt_tokens, generated_tokens, ttft_us,
            );
            let profile = global_profile();
            if profile.is_enabled() {
                let _ = profile.push_event(
                    "stream_ttft_prof",
                    profile_fields_from_json(serde_json::json!({
                        "request_id": request_id.to_string(),
                        "prompt_tokens": prompt_tokens,
                        "generated_tokens": generated_tokens,
                    })),
                    profile_fields_from_json(serde_json::json!({
                        "ttft": ttft_us,
                    })),
                    false,
                );
            }
        }

        if let Some(tx) = sender {
            if delta.is_empty() {
                return;
            }
            let chunk = StreamChunk {
                request_id: request_id.clone(),
                text: delta,
                token,
                finish_reason: None,
                usage: None,
                created_at: chrono::Utc::now(),
                metadata: HashMap::new(),
                api_response: None,
                execution_evidence: None,
            };
            if tx.send(Ok(chunk)).await.is_ok() {
                // A bounded channel send often completes immediately, so a
                // hot decode loop can keep running without giving the CLI /
                // HTTP stream receiver a chance to flush visible output. Yield
                // after successful streaming sends to preserve token-level UX;
                // non-streaming requests do not enter this branch.
                tokio::task::yield_now().await;
            }
        }
    }

    // ── completion ─────────────────────────────────────────────────────

    pub(super) async fn cancel_abandoned_requests(&self) -> Result<()> {
        let abandoned: Vec<_> = self
            .sequences
            .read()
            .iter()
            .filter_map(|(request_id, sequence)| {
                sequence
                    .client_receiver_closed()
                    .then(|| request_id.clone())
            })
            .collect();

        for request_id in abandoned {
            self.cancel_abandoned_request(&request_id).await?;
        }
        Ok(())
    }

    async fn cancel_abandoned_request(&self, request_id: &RequestId) -> Result<()> {
        let detected_scheduler_iteration = self.scheduler.trace_snapshot().current_iteration;
        let (completion_resources, terminal_token_trace) = {
            let mut sequences = self.sequences.write();
            let Some(mut sequence) = sequences.remove(request_id) else {
                return Ok(());
            };
            let terminal_token_trace = self.capture_sequence_terminal_token_trace(&sequence);
            (sequence.take_completion_resources(), terminal_token_trace)
        };
        self.write_sequence_terminal_token_trace(
            request_id,
            "client_disconnected",
            None,
            terminal_token_trace.as_ref(),
        );
        let physical_resources_present = !completion_resources.physical.is_empty();
        let request_slot_present = completion_resources.request_slot.is_some();
        self.write_executor_scheduler_profile_event(
            request_id,
            "engine_client_disconnect_detected",
            ProfileEventKind::Instant,
            ProfileStatus::Ok,
            None,
            BTreeMap::from([(
                "scheduler_iteration".to_string(),
                serde_json::json!(detected_scheduler_iteration),
            )]),
            BTreeMap::from([
                (
                    "disconnect_reason".to_string(),
                    serde_json::json!("client_receiver_closed"),
                ),
                (
                    "terminal_state".to_string(),
                    serde_json::json!("pending_release"),
                ),
            ]),
            None,
        );

        if self.model_executor.execution_resource_authority()
            == ExecutionResourceAuthority::PlanRuntime
        {
            self.model_executor.cancel_prefill_admission(request_id);
        }

        let released_waiting_capacity = self.scheduler.trace_phase(request_id)
            == Some(RequestPhase::Waiting)
            && !completion_resources.physical.is_empty();
        self.release_sequence_physical_resources(request_id, completion_resources.physical)
            .await;
        let scheduler_cancel = self.scheduler.cancel(request_id.clone()).await;
        if released_waiting_capacity {
            self.scheduler.record_external_capacity_release();
        }
        if let Some(request_slot) = completion_resources.request_slot {
            request_slot.close(self);
        }

        let terminal_scheduler_iteration = self.scheduler.trace_snapshot().current_iteration;
        let scheduler_tick_delta =
            terminal_scheduler_iteration.saturating_sub(detected_scheduler_iteration);
        let (phase, status, scheduler_cancel_result, terminal_state, profile_error) =
            match &scheduler_cancel {
                Ok(true) => (
                    "engine_client_disconnect_released",
                    ProfileStatus::Ok,
                    "cancelled",
                    "released",
                    None,
                ),
                Ok(false) => (
                    "engine_client_disconnect_released",
                    ProfileStatus::Ok,
                    "already_absent",
                    "released",
                    None,
                ),
                Err(error) => (
                    "engine_client_disconnect_release_failed",
                    ProfileStatus::Failure,
                    "error",
                    "release_failed",
                    Some(ProfileError {
                        kind: "scheduler_cancel_failed".to_string(),
                        message: error.to_string(),
                        blocking: true,
                    }),
                ),
            };
        self.write_executor_scheduler_profile_event(
            request_id,
            phase,
            ProfileEventKind::Instant,
            status,
            None,
            BTreeMap::from([
                (
                    "detected_scheduler_iteration".to_string(),
                    serde_json::json!(detected_scheduler_iteration),
                ),
                (
                    "terminal_scheduler_iteration".to_string(),
                    serde_json::json!(terminal_scheduler_iteration),
                ),
                (
                    "scheduler_tick_delta".to_string(),
                    serde_json::json!(scheduler_tick_delta),
                ),
            ]),
            BTreeMap::from([
                (
                    "disconnect_reason".to_string(),
                    serde_json::json!("client_receiver_closed"),
                ),
                (
                    "scheduler_cancel_result".to_string(),
                    serde_json::json!(scheduler_cancel_result),
                ),
                (
                    "terminal_state".to_string(),
                    serde_json::json!(terminal_state),
                ),
                (
                    "physical_resources_present".to_string(),
                    serde_json::json!(physical_resources_present),
                ),
                (
                    "request_slot_present".to_string(),
                    serde_json::json!(request_slot_present),
                ),
            ]),
            profile_error,
        );

        match scheduler_cancel {
            Ok(true) => {
                debug!(request_id = %request_id, "Cancelled request after client disconnected");
                Ok(())
            }
            Ok(false) => {
                warn!(
                    request_id = %request_id,
                    "Client-disconnected sequence was absent from scheduler during cancellation"
                );
                Ok(())
            }
            Err(error) => Err(error),
        }
    }

    pub(super) async fn complete_request(
        &self,
        request_id: &RequestId,
        finish_reason: FinishReason,
    ) -> Result<()> {
        self.complete_request_inner(request_id, finish_reason, None)
            .await
    }

    pub(super) async fn complete_request_with_error(
        &self,
        request_id: &RequestId,
        error: FerrumError,
    ) -> Result<()> {
        self.complete_request_inner(request_id, FinishReason::Error, Some(error))
            .await
    }

    async fn complete_request_inner(
        &self,
        request_id: &RequestId,
        finish_reason: FinishReason,
        mut explicit_terminal_error: Option<FerrumError>,
    ) -> Result<()> {
        if explicit_terminal_error.is_none() {
            explicit_terminal_error = self.sequences.read().get(request_id).and_then(|seq| {
                seq.structured_output_terminal_error(finish_reason)
                    .or_else(|| {
                        seq.validate_structured_stop_boundary(self.tokenizer.as_ref())
                            .err()
                    })
            });
        }
        if explicit_terminal_error.is_none() && finish_reason != FinishReason::Error {
            // Flush held, unmatched prefixes and legal text before an in-token
            // stop as a separate text chunk. Consumers may defer a text delta;
            // they must still receive the independent terminal/usage event.
            self.send_stream_text(request_id, None, Some(finish_reason))
                .await;
        }
        let (
            response,
            stream_sender,
            response_sender,
            completion_resources,
            terminal_error,
            terminal_token_trace,
        ) = {
            let mut sequences = self.sequences.write();
            if let Some(mut seq) = sequences.remove(request_id) {
                let terminal_error = explicit_terminal_error.take();
                let finish_reason = if terminal_error.is_some() {
                    FinishReason::Error
                } else {
                    finish_reason
                };
                let terminal_token_trace = self.capture_sequence_terminal_token_trace(&seq);
                let text = seq
                    .decoded_output_text(self.tokenizer.as_ref(), Some(finish_reason))
                    .unwrap_or_default();
                let api_response = ferrum_types::api_response_from_generated_text(
                    &seq.original_request,
                    &text,
                    finish_reason,
                );
                let prompt_token_count = seq.input_tokens.len();
                let execution_evidence = seq.take_execution_evidence()?;

                // TPOT histogram (PLAYBOOK § 7 definition):
                //   tpot = (e2e − ttft) / (output_tokens − 1)
                // Only meaningful when first_emit_at is set (i.e. at
                // least one stream chunk landed) and ≥ 2 chunks were
                // emitted to give a non-degenerate decode window.
                if let (Some(first), Some(last)) = (seq.first_emit_at, seq.last_emit_at) {
                    if seq.emitted_chunks >= 2 {
                        let decode_s = last.duration_since(first).as_secs_f64();
                        let tpot_s = decode_s / (seq.emitted_chunks - 1) as f64;
                        histogram!("ferrum.engine.tpot_seconds").record(tpot_s);
                    }
                }

                let response = InferenceResponse {
                    request_id: request_id.clone(),
                    text,
                    tokens: seq.generated_tokens.clone(),
                    finish_reason,
                    usage: TokenUsage::new(prompt_token_count, seq.generated_tokens.len()),
                    latency_ms: seq.start_time.elapsed().as_millis() as u64,
                    created_at: chrono::Utc::now(),
                    metadata: HashMap::new(),
                    api_response,
                    execution_evidence,
                };

                let completion_resources = seq.take_completion_resources();
                (
                    response,
                    seq.stream_sender.take(),
                    seq.response_sender.take(),
                    completion_resources,
                    terminal_error,
                    terminal_token_trace,
                )
            } else {
                return Ok(());
            }
        };
        self.write_sequence_terminal_token_trace(
            request_id,
            "completed",
            Some(response.finish_reason),
            terminal_token_trace.as_ref(),
        );

        if self.model_executor.execution_resource_authority()
            == ExecutionResourceAuthority::PlanRuntime
        {
            self.model_executor.cancel_prefill_admission(request_id);
        }

        let finish_reason = response.finish_reason;
        if finish_reason == FinishReason::Error {
            self.release_sequence_physical_resources(request_id, completion_resources.physical)
                .await;
        } else {
            self.complete_sequence_physical_resources(
                request_id,
                completion_resources.physical,
                &response.usage,
            )
            .await?;
        }

        let scheduler_complete = self.scheduler.complete(request_id.clone(), &response).await;
        if let Some(request_slot) = completion_resources.request_slot {
            request_slot.close(self);
        }
        scheduler_complete?;

        if let Some(tx) = response_sender {
            let response_result = terminal_error
                .as_ref()
                .map_or_else(|| Ok(response.clone()), |error| Err(error.clone()));
            let _ = tx.send(response_result);
        }

        if let Some(tx) = stream_sender {
            if let Some(error) = terminal_error {
                let _ = tx.send(Err(error)).await;
            } else {
                let final_chunk = StreamChunk {
                    request_id: request_id.clone(),
                    text: String::new(),
                    token: None,
                    finish_reason: Some(finish_reason),
                    usage: Some(response.usage.clone()),
                    created_at: chrono::Utc::now(),
                    metadata: HashMap::new(),
                    api_response: response.api_response.clone(),
                    execution_evidence: response.execution_evidence.clone(),
                };
                let _ = tx.send(Ok(final_chunk)).await;
            }
        }

        debug!(
            "Request {} completed: {} tokens, {:?}",
            request_id,
            response.tokens.len(),
            finish_reason
        );

        Ok(())
    }
}

fn finish_reason_trace_name(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Length => "length",
        FinishReason::Stop => "stop",
        FinishReason::EOS => "eos",
        FinishReason::Cancelled => "cancelled",
        FinishReason::Error => "error",
        FinishReason::ContentFilter => "content_filter",
    }
}
