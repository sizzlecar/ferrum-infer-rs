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
    pub(in crate::continuous_engine) fn client_receiver_closed(&self) -> bool {
        self.response_sender
            .as_ref()
            .is_some_and(tokio::sync::oneshot::Sender::is_closed)
            || self
                .stream_sender
                .as_ref()
                .is_some_and(tokio::sync::mpsc::Sender::is_closed)
    }

    fn structured_output_terminal_error(&self, finish_reason: FinishReason) -> Option<FerrumError> {
        let processor = self.structured_output_processor.as_ref()?;
        match processor
            .progress_with_terminals(&self.generated_tokens, &self.stop_token_ids)
        {
            Ok(progress) if progress.accepting && finish_reason != FinishReason::Error => None,
            Ok(progress) if progress.accepting => Some(FerrumError::model(
                "structured-output generation failed after reaching an accepting state",
            )),
            Ok(progress) => Some(FerrumError::model(format!(
                "structured-output generation ended with {finish_reason:?} before a complete valid value: phase={:?}, generated_tokens={}, consumed_tokens={}, delimiter_tokens={:?}, delimiter_prefix_tokens={}, reasoning_tokens={:?}, boundary_forced={}, budget={:?}",
                progress.phase,
                progress.generated_token_count,
                progress.consumed_token_count,
                progress.delimiter_token_count,
                progress.delimiter_prefix_token_count,
                progress.reasoning_token_count,
                progress.boundary_forced,
                progress.budget,
            ))),
            Err(error) => Some(error),
        }
    }
}

impl EngineInner {
    // ── stream helper ──────────────────────────────────────────────────

    pub(super) fn stop_reason_for_request(&self, request_id: &RequestId) -> Option<FinishReason> {
        let sequences = self.sequences.read();
        match sequences.get(request_id) {
            Some(seq) => seq.stop_reason(Some(self.tokenizer.as_ref())),
            None => Some(FinishReason::Error),
        }
    }

    pub(super) fn should_stream_generated_token(&self, stop_reason: Option<FinishReason>) -> bool {
        !matches!(
            stop_reason,
            Some(FinishReason::Stop) | Some(FinishReason::EOS) | Some(FinishReason::Error)
        )
    }

    pub(super) async fn send_stream_update(&self, request_id: &RequestId, token: TokenId) {
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
        // past `streamed_text_len` and bump the watermark. Buffering is
        // bounded — the longest multi-byte sequence is 4 bytes, so at
        // most one or two tokens get deferred before flushing.
        let (sender, delta, ttft_s, itl_s, first_emit_prof) = {
            let mut sequences = self.sequences.write();
            let Some(seq) = sequences.get_mut(request_id) else {
                return;
            };
            let sender = seq.stream_sender.clone();
            let full = self
                .tokenizer
                .decode(&seq.generated_tokens, true)
                .unwrap_or_else(|_| format!("token_{}", token.get()));
            if full.ends_with('\u{FFFD}') {
                // Partial multi-byte UTF-8 at the tail; wait for the next
                // token. Do NOT advance streamed_text_len so the bytes get
                // re-considered once the sequence completes.
                return;
            }
            let delta = full[seq.streamed_text_len..].to_string();
            seq.streamed_text_len = full.len();

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
                token: Some(token),
                finish_reason: None,
                usage: None,
                created_at: chrono::Utc::now(),
                metadata: HashMap::new(),
                api_response: None,
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
        let completion_resources = {
            let mut sequences = self.sequences.write();
            let Some(mut sequence) = sequences.remove(request_id) else {
                return Ok(());
            };
            sequence.take_completion_resources()
        };

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
        let (response, stream_sender, response_sender, completion_resources, terminal_error) = {
            let mut sequences = self.sequences.write();
            if let Some(mut seq) = sequences.remove(request_id) {
                let terminal_error = explicit_terminal_error
                    .take()
                    .or_else(|| seq.structured_output_terminal_error(finish_reason));
                let finish_reason = if terminal_error.is_some() {
                    FinishReason::Error
                } else {
                    finish_reason
                };
                let text = self
                    .tokenizer
                    .decode(&seq.generated_tokens, true)
                    .unwrap_or_default();
                let api_response =
                    ferrum_types::api_response_from_generated_text(&seq.original_request, &text);

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
                    usage: TokenUsage::new(seq.input_tokens.len(), seq.generated_tokens.len()),
                    latency_ms: seq.start_time.elapsed().as_millis() as u64,
                    created_at: chrono::Utc::now(),
                    metadata: HashMap::new(),
                    api_response,
                };

                let completion_resources = seq.take_completion_resources();
                (
                    response,
                    seq.stream_sender.take(),
                    seq.response_sender.take(),
                    completion_resources,
                    terminal_error,
                )
            } else {
                return Ok(());
            }
        };

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
