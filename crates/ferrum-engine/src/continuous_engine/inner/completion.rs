use super::*;

use ferrum_execution::token_execution::{ExecutionFrontier, ExecutorSequence};
use ferrum_scheduler::implementations::continuous::PrefixRequestKey;

#[derive(Clone, Copy)]
enum CompletionExpectation<'a> {
    Current,
    Admission(&'a PrefixRequestKey),
    Frontier(&'a ExecutorSequence),
    Recompute(&'a PrefixRequestKey, Option<&'a ExecutionFrontier>),
}

impl CompletionExpectation<'_> {
    fn matches(&self, inner: &EngineInner, sequence: &SequenceState) -> bool {
        match self {
            Self::Current => true,
            Self::Admission(key) => sequence.pending_token_admission.as_ref() == Some(*key),
            Self::Frontier(expected) => sequence
                .execution_frontier
                .as_ref()
                .is_some_and(|frontier| frontier.sequence() == *expected),
            Self::Recompute(key, frontier) => {
                inner.scheduler.execution_capacity_request_is_current(key)
                    && sequence.execution_frontier.as_ref() == *frontier
            }
        }
    }
}

pub(in crate::continuous_engine) struct CompletionHandoff {
    inner: Weak<EngineInner>,
    pub(in crate::continuous_engine) done: Option<tokio::sync::oneshot::Sender<Result<()>>>,
    pub(in crate::continuous_engine) response: InferenceResponse,
    pub(in crate::continuous_engine) stream_sender: Option<OutputSender>,
    response_sender: Option<tokio::sync::oneshot::Sender<Result<InferenceResponse>>>,
    terminal_error: Option<FerrumError>,
    terminal_error_published: bool,
    terminal_error_is_delivery: bool,
    completion_error: Option<FerrumError>,
    terminal_token_trace: Option<SequenceTokenTraceEvidence>,
    token_close: Option<Result<execution_close::ExecutionCloseTicket>>,
}

impl Drop for CompletionHandoff {
    fn drop(&mut self) {
        // An output stop may drop a not-yet-polled completion or interrupt a
        // queue/close wait. Its physical ticket is only an observer; preserve
        // the client terminal and trace without awaiting that observer.
        if !self.terminal_error_published && !self.terminal_error_is_delivery {
            if let Some(error) = self.terminal_error.take() {
                if let Some(inner) = self.inner.upgrade() {
                    inner.output_tasks.record_interrupted_error(error);
                    inner.work_notify.notify_one();
                }
            }
        }
        if let Some(trace) = self.terminal_token_trace.take() {
            if let Some(inner) = self.inner.upgrade() {
                inner.write_sequence_terminal_token_trace(
                    &self.response.request_id,
                    "completed",
                    Some(FinishReason::Error),
                    Some(&trace),
                );
            }
        }
        if let Some(sender) = self.response_sender.take() {
            let _ = sender.send(Err(FerrumError::scheduler(
                "engine stopped before output completion",
            )));
        }
        if let Some(sender) = self.stream_sender.take() {
            sender.interrupt_delivery();
        }
        if let Some(done) = self.done.take() {
            let _ = done.send(Err(FerrumError::scheduler(
                "engine stopped before output completion",
            )));
        }
    }
}

impl CompletionHandoff {
    pub(in crate::continuous_engine) fn retain_delivery_error(&mut self, error: FerrumError) {
        if self.terminal_error.is_none() {
            self.terminal_error_is_delivery = true;
            self.terminal_error = Some(error.clone());
        }
        self.completion_error.get_or_insert(error);
        self.response.finish_reason = FinishReason::Error;
        self.response.api_response = None;
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
            // Wire terminators belong to the generated text independently of
            // the later client outcome. Keep other finish-reason semantics.
            if self.stop_token_ids.contains(&last.get())
                && matches!(
                    reason,
                    FinishReason::Stop | FinishReason::EOS | FinishReason::Error
                )
                && !self.is_declared_wire_terminator(Some(tokenizer), last)
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
                .is_some_and(OutputSender::is_closed)
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

    pub(super) fn stop_reason_for_sequence(
        &self,
        handle: &Arc<SequenceHandle>,
    ) -> Option<FinishReason> {
        match handle.lock().as_ref() {
            Some(seq) if self.sequences.is_current(handle) => {
                seq.stop_reason(Some(self.tokenizer.as_ref()))
            }
            Some(_) => Some(FinishReason::Error),
            None => Some(FinishReason::Error),
        }
    }

    pub(super) fn should_stream_sequence_token(
        &self,
        handle: &Arc<SequenceHandle>,
        token: TokenId,
        stop_reason: Option<FinishReason>,
    ) -> bool {
        if !matches!(
            stop_reason,
            Some(FinishReason::Stop) | Some(FinishReason::EOS)
        ) {
            return !matches!(stop_reason, Some(FinishReason::Error));
        }
        handle.lock().as_ref().is_some_and(|sequence| {
            self.sequences.is_current(handle)
                && sequence.should_stream_generated_token(
                    Some(self.tokenizer.as_ref()),
                    token,
                    stop_reason,
                )
        })
    }

    #[cfg(test)]
    pub(in crate::continuous_engine) async fn send_stream_update(
        &self,
        request_id: &RequestId,
        token: TokenId,
    ) {
        if let Some(handle) = self.sequences.get(request_id) {
            self.send_stream_update_for_sequence(&handle, token).await;
        }
    }

    pub(super) async fn send_stream_update_for_sequence(
        &self,
        handle: &Arc<SequenceHandle>,
        token: TokenId,
    ) {
        {
            let mut state = handle.lock();
            if let Some(seq) = state.as_mut().filter(|_| self.sequences.is_current(handle)) {
                // Incremental decoding can still be retried from the complete
                // history. Terminal decoding below must report its failure.
                let _ = self.send_stream_text_locked(handle.request_id(), seq, Some(token), None);
            }
        }
        tokio::task::yield_now().await;
    }

    fn send_stream_text_locked(
        &self,
        request_id: &RequestId,
        seq: &mut SequenceState,
        token: Option<TokenId>,
        terminal: Option<FinishReason>,
    ) -> Result<()> {
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
        let (ttft_s, itl_s, first_emit_prof) = {
            let sender = seq.stream_sender.clone();
            let mut full = self.tokenizer.decode(&seq.generated_tokens, true)?;
            let incomplete_utf8 = full.ends_with('\u{FFFD}');
            if incomplete_utf8 && terminal.is_none() {
                // Partial multi-byte UTF-8 at the tail; wait for the next
                // token. Do NOT advance streamed_text_len so the bytes get
                // re-considered once the sequence completes.
                return Ok(());
            }
            if !incomplete_utf8 {
                seq.decoded_text_len = full.len();
            }
            let visible = if terminal.is_some() {
                seq.decoded_output_text(self.tokenizer.as_ref(), terminal)?
            } else {
                let end = seq.visible_text_end(&full, false);
                full.truncate(end);
                full
            };
            if visible.ends_with('\u{FFFD}') {
                return Ok(());
            }
            let Some(delta) = visible.get(seq.streamed_text_len..) else {
                return Ok(());
            };
            let delta = delta.to_string();
            let range = seq.streamed_text_len..visible.len();
            if !delta.is_empty() {
                if let Some(sender) = sender {
                    let chunk = StreamChunk {
                        request_id: request_id.clone(),
                        text: delta.clone(),
                        token,
                        finish_reason: None,
                        usage: None,
                        created_at: chrono::Utc::now(),
                        metadata: HashMap::new(),
                        api_response: None,
                        execution_evidence: None,
                    };
                    if let Err(error) = sender.publish_text(self, chunk, range, terminal) {
                        sender.record_failure(error);
                        self.work_notify.notify_one();
                        return Ok(());
                    }
                }
            }
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

            (ttft_s, itl_s, first_emit_prof)
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
        Ok(())
    }

    pub(in crate::continuous_engine) fn rebuild_output_span(
        &self,
        sender: &OutputSender,
        request_id: &RequestId,
        span: &output_delivery::TextSpan,
    ) -> Result<Option<StreamChunk>> {
        // Delivery keeps only a weak association with the original request;
        // no index lookup can redirect a pending span to a replacement.
        let handle = sender.source_sequence();
        let state = handle.as_ref().map(|handle| handle.lock());
        let text = match sender.completed_text(span.bytes.clone())? {
            Some(text) => text,
            None => {
                let Some(sequence) =
                    state
                        .as_ref()
                        .and_then(|state| state.as_ref())
                        .filter(|sequence| {
                            sequence
                                .stream_sender
                                .as_ref()
                                .is_some_and(|actual| actual.same_owner(sender))
                        })
                else {
                    return Ok(None);
                };
                let visible =
                    sequence.decoded_output_text(self.tokenizer.as_ref(), span.terminal)?;
                visible
                    .get(span.bytes.clone())
                    .ok_or_else(|| {
                        FerrumError::internal(
                            "pending output span disagrees with original token history",
                        )
                    })?
                    .to_owned()
            }
        };
        Ok(Some(StreamChunk {
            request_id: request_id.clone(),
            text,
            token: span.token,
            finish_reason: None,
            usage: None,
            created_at: span.created_at,
            metadata: HashMap::new(),
            api_response: None,
            execution_evidence: None,
        }))
    }

    // ── completion ─────────────────────────────────────────────────────

    pub(super) async fn cancel_abandoned_requests(&self) -> Result<()> {
        for handle in self.sequences.snapshot() {
            let (failure, abandoned) = {
                let state = handle.lock();
                let Some(sequence) = state.as_ref() else {
                    continue;
                };
                (
                    sequence
                        .stream_sender
                        .as_ref()
                        .and_then(OutputSender::failure),
                    sequence.client_receiver_closed(),
                )
            };
            if let Some(error) = failure {
                self.complete_request_exact_detached(&handle, FinishReason::Error, Some(error))?;
            } else if abandoned {
                self.cancel_abandoned_sequence(&handle).await?;
            }
        }
        Ok(())
    }

    async fn cancel_abandoned_sequence(&self, handle: &Arc<SequenceHandle>) -> Result<()> {
        let request_id = handle.request_id();
        let detected_scheduler_iteration = self.scheduler.trace_snapshot().current_iteration;
        let (terminal_token_trace, token_close, request_slot_present, scheduler_cancel) = {
            let mut state = handle.lock();
            let Some(sequence) = state.as_mut() else {
                return Ok(());
            };
            if !self.sequences.is_current(handle) {
                return Ok(());
            }
            self.discard_pending_prefix_restore(request_id);
            let scheduler_cancel = self.scheduler.cancel_request(request_id.clone());
            let terminal_token_trace = self.capture_sequence_terminal_token_trace(&sequence);
            let mut resources = sequence.take_completion_resources();
            let request_slot_present = resources.request_slot.is_some();
            // Keep the outer handoff atomic with terminal consumption by the
            // close registry. No await may separate removal and registration.
            let token_close = resources.execution_frontier.take().map(|frontier| {
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
                (frontier, ticket)
            });
            if token_close.is_none() {
                self.model_executor.cancel_prefill_admission(request_id);
            }
            if let Some(slot) = resources.request_slot.take() {
                slot.close(self);
            }
            // Keep the index occupied until scheduler termination and exact
            // physical handoff are complete. Old Arcs subsequently see Taken.
            state.take();
            self.sequences.remove_exact(handle);
            (
                terminal_token_trace,
                token_close,
                request_slot_present,
                scheduler_cancel,
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
        if let Some((frontier, ticket)) = token_close {
            let close_result = match ticket {
                Ok(ticket) => ticket.initial_result().await,
                Err(error) => Err(error),
            };
            let observation_result = self.drain_execution_closes();
            close_error = close_result.and(observation_result).err();
            token_close_pending = frontier.sequence().observed_end().is_none();
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

    #[cfg(test)]
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
        handle: &Arc<SequenceHandle>,
        key: &ferrum_scheduler::implementations::continuous::PrefixRequestKey,
        error: FerrumError,
    ) -> Result<()> {
        self.register_request_completion(
            handle,
            FinishReason::Error,
            Some(error),
            CompletionExpectation::Admission(key),
        )
        .map(drop)
    }

    #[cfg(test)]
    async fn complete_request_inner(
        &self,
        request_id: &RequestId,
        finish_reason: FinishReason,
        explicit_terminal_error: Option<FerrumError>,
        expected_admission: Option<
            &ferrum_scheduler::implementations::continuous::PrefixRequestKey,
        >,
    ) -> Result<()> {
        let Some(handle) = self.sequences.get(request_id) else {
            return Ok(());
        };
        let Some(receiver) = self.register_request_completion(
            &handle,
            finish_reason,
            explicit_terminal_error,
            expected_admission.map_or(
                CompletionExpectation::Current,
                CompletionExpectation::Admission,
            ),
        )?
        else {
            return Ok(());
        };
        receiver
            .await
            .map_err(|_| FerrumError::internal("completion delivery stopped"))?
    }

    fn register_request_completion(
        &self,
        handle: &Arc<SequenceHandle>,
        finish_reason: FinishReason,
        mut explicit_terminal_error: Option<FerrumError>,
        expected: CompletionExpectation<'_>,
    ) -> Result<Option<tokio::sync::oneshot::Receiver<Result<()>>>> {
        let request_id = handle.request_id();
        let handoff = {
            let mut state = handle.lock();
            let Some(seq) = state.as_mut() else {
                return Ok(None);
            };
            if !self.sequences.is_current(handle) || !expected.matches(self, seq) {
                return Ok(None);
            }
            if !matches!(expected, CompletionExpectation::Admission(_)) {
                self.discard_pending_prefix_restore(request_id);
            }
            let mut classified_api_response = None;
            let mut terminal_error_is_delivery = false;
            if explicit_terminal_error.is_none() {
                let delivery_error = seq.stream_sender.as_ref().and_then(OutputSender::failure);
                terminal_error_is_delivery = delivery_error.is_some();
                explicit_terminal_error = delivery_error.or_else(|| {
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
            let mut completion_error = None;
            if explicit_terminal_error.is_none() && finish_reason != FinishReason::Error {
                if let Err(error) =
                    self.send_stream_text_locked(request_id, seq, None, Some(finish_reason))
                {
                    explicit_terminal_error = Some(error.clone());
                    completion_error = Some(error);
                }
            }
            self.handoff_request_completion_locked(
                handle,
                &mut state,
                finish_reason,
                explicit_terminal_error,
                terminal_error_is_delivery,
                completion_error,
                expected,
                classified_api_response,
            )
        };
        let Some(handoff) = handoff else {
            return Ok(None);
        };
        self.schedule_output_completion(request_id, handoff)
            .map(Some)
    }

    pub(in crate::continuous_engine) fn complete_request_detached(
        &self,
        request_id: &RequestId,
        finish_reason: FinishReason,
        error: Option<FerrumError>,
    ) -> Result<()> {
        let Some(handle) = self.sequences.get(request_id) else {
            return Ok(());
        };
        self.complete_request_exact_detached(&handle, finish_reason, error)
    }

    pub(in crate::continuous_engine) fn complete_request_exact_detached(
        &self,
        handle: &Arc<SequenceHandle>,
        finish_reason: FinishReason,
        error: Option<FerrumError>,
    ) -> Result<()> {
        self.register_request_completion(
            handle,
            finish_reason,
            error,
            CompletionExpectation::Current,
        )
        .map(drop)
    }

    pub(in crate::continuous_engine) fn complete_request_exact_frontier_detached(
        &self,
        handle: &Arc<SequenceHandle>,
        expected: &ExecutorSequence,
        finish_reason: FinishReason,
        error: Option<FerrumError>,
    ) -> Result<()> {
        self.register_request_completion(
            handle,
            finish_reason,
            error,
            CompletionExpectation::Frontier(expected),
        )
        .map(drop)
    }

    pub(in crate::continuous_engine) fn complete_request_exact_recompute_detached(
        &self,
        handle: &Arc<SequenceHandle>,
        key: &PrefixRequestKey,
        frontier: Option<&ExecutionFrontier>,
        error: FerrumError,
    ) -> Result<()> {
        self.register_request_completion(
            handle,
            FinishReason::Error,
            Some(error),
            CompletionExpectation::Recompute(key, frontier),
        )
        .map(drop)
    }

    fn schedule_output_completion(
        &self,
        request_id: &RequestId,
        mut handoff: CompletionHandoff,
    ) -> Result<tokio::sync::oneshot::Receiver<Result<()>>> {
        if let Some(error) = handoff.completion_error.as_ref().or_else(|| {
            handoff
                .token_close
                .as_ref()
                .and_then(|result| result.as_ref().err())
        }) {
            self.output_tasks.record_control_error(error.clone());
            self.work_notify.notify_one();
        }
        let (done, receiver) = tokio::sync::oneshot::channel();
        handoff.done = Some(done);
        if let Some(sender) = handoff.stream_sender.take() {
            sender.complete(self, handoff)?;
        } else {
            let inner = self
                .self_weak
                .upgrade()
                .ok_or_else(|| FerrumError::internal("completion engine unavailable"))?;
            let request_id = request_id.clone();
            let mut stop = self.output_tasks.stop_receiver();
            drop(self.output_tasks.spawn(async move {
                tokio::select! {
                    biased;
                    _ = output_delivery::wait_for_stop(&mut stop) => {},
                    _ = inner.finish_output_completion(&request_id, handoff, None) => {},
                }
                Ok(())
            })?);
        }
        Ok(receiver)
    }

    fn handoff_request_completion_locked(
        &self,
        handle: &Arc<SequenceHandle>,
        state: &mut Option<SequenceState>,
        finish_reason: FinishReason,
        mut explicit_terminal_error: Option<FerrumError>,
        terminal_error_is_delivery: bool,
        mut completion_error: Option<FerrumError>,
        expected: CompletionExpectation<'_>,
        mut classified_api_response: Option<ferrum_types::ApiResponse>,
    ) -> Option<CompletionHandoff> {
        let request_id = handle.request_id();
        // A terminal failure is not permission to abandon the other owners.
        // Keep the first control-plane error, while independently closing the
        // resources and client channels whose disposition is still known.
        let (
            response,
            stream_sender,
            mut response_sender,
            terminal_error,
            terminal_token_trace,
            token_close,
        ) = {
            if let Some(seq) = state.as_mut() {
                if !self.sequences.is_current(handle) || !expected.matches(self, seq) {
                    return None;
                }
                let mut terminal_error = explicit_terminal_error.take();
                let mut finish_reason = if terminal_error.is_some() {
                    FinishReason::Error
                } else {
                    finish_reason
                };
                let terminal_token_trace = self.capture_sequence_terminal_token_trace(seq);
                let text = match seq
                    .decoded_output_text(self.tokenizer.as_ref(), Some(finish_reason))
                {
                    Ok(text) => text,
                    Err(error) => {
                        warn!(request_id = %request_id, error = %error, "Request completion decoding failed");
                        terminal_error.get_or_insert_with(|| error.clone());
                        completion_error.get_or_insert(error);
                        finish_reason = FinishReason::Error;
                        // This failed DTO is only used for accounting/closure;
                        // both client channels receive the retained error.
                        String::new()
                    }
                };
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
                        completion_error.get_or_insert(error);
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
                let scheduler_completion =
                    if !matches!(expected, CompletionExpectation::Admission(_)) {
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
                if seq.execution_frontier.is_none()
                    && !matches!(expected, CompletionExpectation::Admission(_))
                {
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
                // The index remains occupied until synchronous scheduler
                // termination and exact owner transfer finish. Unwinding
                // earlier leaves the remaining state in its original handle.
                state.take();
                self.sequences.remove_exact(handle);
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
        }

        Some(CompletionHandoff {
            inner: self.self_weak.clone(),
            done: None,
            response,
            stream_sender,
            response_sender,
            terminal_error,
            terminal_error_published: false,
            terminal_error_is_delivery,
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
        let mut errors = Vec::new();
        for handle in self.sequences.snapshot() {
            let request_id = handle.request_id();
            let handoff = {
                let mut state = handle.lock();
                if !self.sequences.is_current(&handle) {
                    continue;
                }
                self.discard_pending_prefix_restore(request_id);
                self.handoff_request_completion_locked(
                    &handle,
                    &mut state,
                    FinishReason::Error,
                    Some(FerrumError::scheduler(
                        "engine shut down before request completion",
                    )),
                    false,
                    None,
                    CompletionExpectation::Current,
                    None,
                )
            };
            if let Some(mut handoff) = handoff {
                // Shutdown cannot wait for a full client queue. Publish one
                // bounded fallback before relinquishing the last sender; the
                // receiver emits it only after already queued data.
                if let Some(sender) = handoff.stream_sender.take() {
                    sender.interrupt_delivery();
                }
                self.write_sequence_terminal_token_trace(
                    &request_id,
                    "completed",
                    Some(handoff.response.finish_reason),
                    handoff.terminal_token_trace.take().as_ref(),
                );
                if let Some(error) = handoff.completion_error.take() {
                    errors.push(error);
                }
                if let Some(Err(error)) = handoff.token_close.take() {
                    errors.push(error);
                }
                handoff.terminal_error.take();
                // Physical ownership already resides in execution_closes.
                // The ticket is an observer, never the release authority.
            }
        }
        errors
    }

    pub(in crate::continuous_engine) async fn finish_output_completion(
        &self,
        request_id: &RequestId,
        mut handoff: CompletionHandoff,
        stream_sender: Option<&OutputSender>,
    ) {
        let result = self
            .finish_request_completion(request_id, &mut handoff, stream_sender)
            .await;
        if let Some(done) = handoff.done.take() {
            let _ = done.send(result);
        }
    }

    async fn finish_request_completion(
        &self,
        request_id: &RequestId,
        handoff: &mut CompletionHandoff,
        stream_sender: Option<&OutputSender>,
    ) -> Result<()> {
        let terminal_error = &mut handoff.terminal_error;
        let mut completion_error = handoff.completion_error.take();
        let token_close = handoff.token_close.take();
        let response = &mut handoff.response;
        // An already-known product error precedes physical-close observation,
        // but follows every earlier text span in this same delivery owner.
        let known_stream_error = terminal_error.is_some();
        if let (Some(sender), Some(error)) = (stream_sender, terminal_error.as_ref()) {
            if sender.send(Err(error.clone())).await.is_ok() {
                handoff.terminal_error_published = true;
            }
        }
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
                self.output_tasks.record_control_error(error.clone());
                self.work_notify.notify_one();
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
            handoff.terminal_token_trace.take().as_ref(),
        );

        if let Some(tx) = handoff.response_sender.take() {
            let response_result = terminal_error
                .as_ref()
                .map_or_else(|| Ok(response.clone()), |error| Err(error.clone()));
            let _ = tx.send(response_result);
        }

        if let Some(tx) = stream_sender.filter(|_| !known_stream_error) {
            if let Some(error) = terminal_error.as_ref() {
                if tx.send(Err(error.clone())).await.is_ok() {
                    handoff.terminal_error_published = true;
                }
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
                if let Err(error) = tx.send(Ok(final_chunk)).await {
                    if !tx.is_closed() {
                        // A terminal payload can exceed the queue byte limit
                        // (for example structured output/evidence). EOF is not
                        // a valid substitute for that explicit delivery error.
                        let _ = tx.send(Err(error.clone())).await;
                        completion_error.get_or_insert(error);
                    }
                }
            }
        }

        debug!(
            "Request {} completed: {} tokens, {:?}",
            request_id,
            response.tokens.len(),
            response.finish_reason
        );

        // All requested publication attempts have reached their disposition.
        // Only cancellation before this point uses the shutdown fallback.
        handoff.terminal_error.take();
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
