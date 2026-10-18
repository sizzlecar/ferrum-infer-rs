use super::*;

// Restoring the original outer state on unwind preserves owners that have not
// reached the close registry yet. The guard never synthesizes terminal facts.
struct RemovedCompletionSequence<'a> {
    sequences: &'a mut HashMap<RequestId, SequenceState>,
    request_id: RequestId,
    sequence: Option<SequenceState>,
}

impl Drop for RemovedCompletionSequence<'_> {
    fn drop(&mut self) {
        if let Some(sequence) = self.sequence.take() {
            self.sequences.insert(self.request_id.clone(), sequence);
        }
    }
}

struct CompletionHandoff {
    response: InferenceResponse,
    stream_sender: Option<mpsc::Sender<Result<StreamChunk>>>,
    response_sender: Option<tokio::sync::oneshot::Sender<Result<InferenceResponse>>>,
    terminal_error: Option<FerrumError>,
    completion_error: Option<FerrumError>,
    terminal_token_trace: Option<SequenceTokenTraceEvidence>,
    token_close: Option<Result<execution_close::ExecutionCloseTicket>>,
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
        if let Some(root_end) = processor
            .complete_root_text_len_with_terminals(&self.generated_tokens, &self.stop_token_ids)?
        {
            // The composed grammar owns the entire wire result, including
            // tool and native protocol terminators. A stop may trim trailing
            // whitespace, but cannot cut a complete result inside a merged
            // token merely because the untrimmed token was grammar-valid.
            if stop_end < root_end {
                return Err(FerrumError::model(
                    "stop sequence truncated the structured output before its complete tool or final result",
                ));
            }
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

    fn classified_structured_api_response(
        &self,
        finish_reason: FinishReason,
    ) -> Result<Option<ferrum_types::ApiResponse>> {
        if !matches!(finish_reason, FinishReason::Stop | FinishReason::EOS) {
            return Ok(None);
        }
        let Some(processor) = self.structured_output_processor.as_ref() else {
            return Ok(None);
        };
        let Some((branch, payload)) = processor
            .classified_result_with_terminals(&self.generated_tokens, &self.stop_token_ids)?
        else {
            return Ok(None);
        };
        ferrum_types::api_response_from_classified_generated_text(
            &self.original_request,
            &payload,
            finish_reason,
            branch,
        )
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
        self.discard_pending_prefix_restore(request_id);
        let detected_scheduler_iteration = self.scheduler.trace_snapshot().current_iteration;
        let (completion_resources, terminal_token_trace, token_close, request_slot_present) = {
            let mut sequences = self.sequences.write();
            let Some(mut sequence) = sequences.remove(request_id) else {
                return Ok(());
            };
            let terminal_token_trace = self.capture_sequence_terminal_token_trace(&sequence);
            let mut resources = sequence.take_completion_resources();
            let request_slot_present = resources.request_slot.is_some();
            // Keep the outer handoff atomic with terminal consumption by the
            // close registry. No await may separate removal and registration.
            let token_close = resources.execution_frontier.take().map(|frontier| {
                let cancelled = self.scheduler.cancel_request(request_id.clone());
                let ticket = match resources.execution_close.take() {
                    Some(outcome) => self.begin_observed_close(
                        request_id.clone(),
                        frontier.sequence().clone(),
                        resources.request_slot.take(),
                        outcome,
                    ),
                    None => self.begin_execution_close(
                        request_id.clone(),
                        frontier.sequence().clone(),
                        resources.request_slot.take(),
                        ferrum_execution::token_execution::ExecutionEndReason::Cancelled,
                    ),
                };
                (frontier, cancelled, ticket)
            });
            (
                resources,
                terminal_token_trace,
                token_close,
                request_slot_present,
            )
        };
        self.write_sequence_terminal_token_trace(
            request_id,
            "client_disconnected",
            None,
            terminal_token_trace.as_ref(),
        );
        let execution_incarnation_present = token_close.is_some();
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

        let mut close_error = None;
        let mut token_close_pending = false;
        let scheduler_cancel = if let Some((frontier, scheduler_cancel, ticket)) = token_close {
            let close_result = match ticket {
                Ok(ticket) => ticket.initial_result().await,
                Err(error) => Err(error),
            };
            let observation_result = self.drain_execution_closes();
            close_error = close_result.and(observation_result).err();
            token_close_pending = frontier.sequence().observed_end().is_none();
            scheduler_cancel
        } else {
            // No token incarnation was admitted. Cancel only the retained
            // admission obligation; no physical release is inferred here.
            self.model_executor.cancel_prefill_admission(request_id);
            let scheduler_cancel = self.scheduler.cancel_request(request_id.clone());
            if let Some(request_slot) = completion_resources.request_slot {
                request_slot.close(self);
            }
            scheduler_cancel
        };

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
                    serde_json::json!(if token_close_pending {
                        "close_pending"
                    } else {
                        terminal_state
                    }),
                ),
                (
                    "execution_incarnation_present".to_string(),
                    serde_json::json!(execution_incarnation_present),
                ),
                (
                    "request_slot_present".to_string(),
                    serde_json::json!(request_slot_present),
                ),
            ]),
            profile_error,
        );

        let scheduler_result = match scheduler_cancel {
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
        };
        scheduler_result.and(close_error.map_or(Ok(()), Err))
    }

    pub(super) async fn complete_request(
        &self,
        request_id: &RequestId,
        finish_reason: FinishReason,
    ) -> Result<()> {
        self.complete_request_inner(request_id, finish_reason, None, None)
            .await
    }

    pub(in crate::continuous_engine) async fn complete_request_with_error(
        &self,
        request_id: &RequestId,
        error: FerrumError,
    ) -> Result<()> {
        self.complete_request_inner(request_id, FinishReason::Error, Some(error), None)
            .await
    }

    #[cfg(test)]
    pub(in crate::continuous_engine) async fn complete_request_for_test(
        &self,
        request_id: &RequestId,
        finish_reason: FinishReason,
    ) -> Result<()> {
        self.complete_request_inner(request_id, finish_reason, None, None)
            .await
    }

    pub(super) async fn complete_unpublished_admission(
        &self,
        key: &ferrum_scheduler::implementations::continuous::PrefixRequestKey,
        error: FerrumError,
    ) -> Result<()> {
        self.complete_request_inner(
            key.request_id(),
            FinishReason::Error,
            Some(error),
            Some(key),
        )
        .await
    }

    async fn complete_request_inner(
        &self,
        request_id: &RequestId,
        finish_reason: FinishReason,
        mut explicit_terminal_error: Option<FerrumError>,
        expected_admission: Option<
            &ferrum_scheduler::implementations::continuous::PrefixRequestKey,
        >,
    ) -> Result<()> {
        // An unpublished admission has no prefix restore, and its original
        // scheduler ticket has already been removed. Neither operation may
        // touch a replacement merely because it reuses the RequestId.
        if expected_admission.is_none() {
            self.discard_pending_prefix_restore(request_id);
        }
        let mut classified_api_response = None;
        if explicit_terminal_error.is_none() {
            explicit_terminal_error = self.sequences.read().get(request_id).and_then(|seq| {
                seq.structured_output_terminal_error(finish_reason)
                    .or_else(|| {
                        seq.validate_structured_stop_boundary(self.tokenizer.as_ref())
                            .err()
                    })
                    .or_else(
                        || match seq.classified_structured_api_response(finish_reason) {
                            Ok(response) => {
                                classified_api_response = response;
                                None
                            }
                            Err(error) => Some(error),
                        },
                    )
            });
        }
        if explicit_terminal_error.is_none() && finish_reason != FinishReason::Error {
            // Flush held, unmatched prefixes and legal text before an in-token
            // stop as a separate text chunk. Consumers may defer a text delta;
            // they must still receive the independent terminal/usage event.
            self.send_stream_text(request_id, None, Some(finish_reason))
                .await;
        }
        let Some(handoff) = self.handoff_request_completion(
            request_id,
            finish_reason,
            explicit_terminal_error,
            expected_admission,
            classified_api_response,
        ) else {
            return Ok(());
        };
        self.finish_request_completion(request_id, handoff).await
    }

    fn handoff_request_completion(
        &self,
        request_id: &RequestId,
        finish_reason: FinishReason,
        mut explicit_terminal_error: Option<FerrumError>,
        expected_admission: Option<
            &ferrum_scheduler::implementations::continuous::PrefixRequestKey,
        >,
        mut classified_api_response: Option<ferrum_types::ApiResponse>,
    ) -> Option<CompletionHandoff> {
        // A terminal failure is not permission to abandon the other owners.
        // Keep the first control-plane error, while independently closing the
        // resources and client channels whose disposition is still known.
        let mut completion_error = None;
        let (
            response,
            mut stream_sender,
            mut response_sender,
            terminal_error,
            terminal_token_trace,
            token_close,
        ) = {
            let mut sequences = self.sequences.write();
            if let Some(key) = expected_admission {
                if !sequences
                    .get(request_id)
                    .is_some_and(|sequence| sequence.pending_token_admission.as_ref() == Some(key))
                {
                    return None;
                }
            }
            if let Some(sequence) = sequences.remove(request_id) {
                let mut removed = RemovedCompletionSequence {
                    sequences: &mut sequences,
                    request_id: request_id.clone(),
                    sequence: Some(sequence),
                };
                let seq = removed
                    .sequence
                    .as_mut()
                    .expect("removed sequence is retained");
                let mut terminal_error = explicit_terminal_error.take();
                let mut finish_reason = if terminal_error.is_some() {
                    FinishReason::Error
                } else {
                    finish_reason
                };
                let terminal_token_trace = self.capture_sequence_terminal_token_trace(seq);
                let text = seq
                    .decoded_output_text(self.tokenizer.as_ref(), Some(finish_reason))
                    .unwrap_or_default();
                let api_response = classified_api_response.take().or_else(|| {
                    // Declared native chat uses the same output projection as
                    // HTTP streaming; an authoritative grammar result wins.
                    ferrum_types::api_response_from_generated_text(
                        &seq.original_request,
                        &text,
                        finish_reason,
                    )
                });
                let prompt_token_count = seq.input_tokens.len();
                let execution_evidence = match seq.take_execution_evidence() {
                    Ok(evidence) => evidence,
                    Err(error) => {
                        warn!(request_id = %request_id, error = %error, "Request completion evidence failed");
                        terminal_error.get_or_insert_with(|| error.clone());
                        completion_error = Some(error);
                        finish_reason = FinishReason::Error;
                        None
                    }
                };

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

                let mut response = InferenceResponse {
                    request_id: request_id.clone(),
                    text,
                    tokens: seq.generated_tokens.clone(),
                    finish_reason,
                    usage: TokenUsage::new(prompt_token_count, seq.generated_tokens.len()),
                    latency_ms: seq.start_time.elapsed().as_millis() as u64,
                    created_at: chrono::Utc::now(),
                    metadata: HashMap::new(),
                    api_response: if terminal_error.is_none() {
                        api_response
                    } else {
                        None
                    },
                    execution_evidence,
                };

                // The same outer lock excludes a recompute drain until this
                // exact incarnation and its slot have reached the registry.
                let scheduler_completion = if expected_admission.is_none() {
                    self.scheduler
                        .complete_request(request_id.clone(), &response)
                } else {
                    Ok(())
                };
                if let Err(error) = scheduler_completion {
                    warn!(request_id = %request_id, error = %error, "Scheduler completion failed");
                    terminal_error.get_or_insert_with(|| error.clone());
                    completion_error.get_or_insert(error);
                    response.finish_reason = FinishReason::Error;
                    response.api_response = None;
                }
                if seq.execution_frontier.is_none() && expected_admission.is_none() {
                    self.model_executor.cancel_prefill_admission(request_id);
                }
                let mut completion_resources = seq.take_completion_resources();
                let token_close = completion_resources
                    .execution_frontier
                    .take()
                    .map(|frontier| {
                        use ferrum_execution::token_execution::ExecutionEndReason;
                        let reason = if response.finish_reason == FinishReason::Error {
                            ExecutionEndReason::Cancelled
                        } else {
                            ExecutionEndReason::Completed {
                                input_tokens: response.usage.prompt_tokens,
                                output_tokens: response.usage.completion_tokens,
                            }
                        };
                        match completion_resources.execution_close.take() {
                            Some(outcome) => self.begin_observed_close(
                                request_id.clone(),
                                frontier.sequence().clone(),
                                completion_resources.request_slot.take(),
                                outcome,
                            ),
                            None => self.begin_execution_close(
                                request_id.clone(),
                                frontier.sequence().clone(),
                                completion_resources.request_slot.take(),
                                reason,
                            ),
                        }
                    });
                // Non-materialized requests own no executor sequence; their
                // product slot can close synchronously before any await.
                if let Some(slot) = completion_resources.request_slot.take() {
                    slot.close(self);
                }
                let senders = (seq.stream_sender.take(), seq.response_sender.take());
                removed.sequence.take();
                (
                    response,
                    senders.0,
                    senders.1,
                    terminal_error,
                    terminal_token_trace,
                    token_close,
                )
            } else {
                return None;
            }
        };
        // This error is already final for the client, independently of the
        // physical close result. Transfer publication before the first await:
        // cancellation must not drop the only senders after removing Sequence.
        if let Some(error) = terminal_error.as_ref() {
            if let Some(tx) = response_sender.take() {
                let _ = tx.send(Err(error.clone()));
            }
            if let Some(tx) = stream_sender.take() {
                match tx.try_send(Err(error.clone())) {
                    Ok(()) | Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {}
                    Err(tokio::sync::mpsc::error::TrySendError::Full(terminal)) => {
                        // One bounded terminal delivery, only on the error
                        // path. The close registry retains all physical owners;
                        // this task owns only the client channel and its error.
                        drop(tokio::spawn(async move {
                            let _ = tx.send(terminal).await;
                        }));
                    }
                }
            }
        }
        Some(CompletionHandoff {
            response,
            stream_sender,
            response_sender,
            terminal_error,
            completion_error,
            terminal_token_trace,
            token_close,
        })
    }

    /// All original outer owners and known errors are handed off synchronously
    /// before shutdown waits for any executor's close result.
    pub(in crate::continuous_engine) fn handoff_shutdown_requests(&self) -> Vec<FerrumError> {
        // These are optional source-checkpoint/capture pins, not restored
        // target publications. Relinquish all coordination (including stale
        // entries without an outer request) before waiting for physical close.
        self.release_pending_prefix_restores_for_capacity_pressure();
        let cohorts = std::mem::take(&mut *self.prefix_rendezvous.lock());
        drop(cohorts);
        let request_ids: Vec<_> = self.sequences.read().keys().cloned().collect();
        let mut errors = Vec::new();
        for request_id in request_ids {
            self.discard_pending_prefix_restore(&request_id);
            if let Some(handoff) = self.handoff_request_completion(
                &request_id,
                FinishReason::Error,
                Some(FerrumError::scheduler(
                    "engine shut down before request completion",
                )),
                None,
                None,
            ) {
                self.write_sequence_terminal_token_trace(
                    &request_id,
                    "completed",
                    Some(handoff.response.finish_reason),
                    handoff.terminal_token_trace.as_ref(),
                );
                if let Some(error) = handoff.completion_error {
                    errors.push(error);
                }
                if let Some(Err(error)) = handoff.token_close {
                    errors.push(error);
                }
                // The ticket is observation only. Registry tasks retain all
                // actual owners; shutdown joins them after the entire handoff.
            }
        }
        errors
    }

    async fn finish_request_completion(
        &self,
        request_id: &RequestId,
        handoff: CompletionHandoff,
    ) -> Result<()> {
        let CompletionHandoff {
            mut response,
            stream_sender,
            response_sender,
            mut terminal_error,
            mut completion_error,
            terminal_token_trace,
            token_close,
        } = handoff;
        // Physical closure may remain held after logical completion. Awaiting
        // the initial result cannot cancel the registry's independent owner.
        if let Some(ticket) = token_close {
            let close_result = match ticket {
                Ok(ticket) => ticket.initial_result().await,
                Err(error) => Err(error),
            };
            // Drain all terminal owners even if the operation itself failed.
            let observation_result = self.drain_execution_closes();
            if let Err(error) = close_result.and(observation_result) {
                terminal_error.get_or_insert_with(|| error.clone());
                completion_error.get_or_insert(error);
                response.finish_reason = FinishReason::Error;
                response.api_response = None;
            }
        }
        self.write_sequence_terminal_token_trace(
            request_id,
            "completed",
            Some(response.finish_reason),
            terminal_token_trace.as_ref(),
        );

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
                    finish_reason: Some(response.finish_reason),
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
            response.finish_reason
        );

        completion_error.map_or(Ok(()), Err)
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

#[cfg(test)]
mod cold_error_tests;
