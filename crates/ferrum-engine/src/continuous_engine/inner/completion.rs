use super::*;

impl EngineInner {
    // ── stream helper ──────────────────────────────────────────────────

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
        let (sender, delta, ttft_s, itl_s) = {
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
            if !delta.is_empty() {
                let now = Instant::now();
                match seq.first_emit_at {
                    None => {
                        ttft_s = Some(now.duration_since(seq.start_time).as_secs_f64());
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

            (sender, delta, ttft_s, itl_s)
        };

        if let Some(t) = ttft_s {
            histogram!("ferrum.engine.ttft_seconds").record(t);
        }
        if let Some(t) = itl_s {
            histogram!("ferrum.engine.itl_seconds").record(t);
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
            };
            let _ = tx.send(Ok(chunk)).await;
        }
    }

    // ── completion ─────────────────────────────────────────────────────

    pub(super) async fn complete_request(
        &self,
        request_id: &RequestId,
        finish_reason: FinishReason,
    ) -> Result<()> {
        let (response, stream_sender, response_sender, has_kv_cache, model_cache_id) = {
            let mut sequences = self.sequences.write();
            if let Some(seq) = sequences.remove(request_id) {
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

                let has_kv = seq.kv_cache.is_some();
                let cache_id = seq.model_cache_id.clone();
                (
                    response,
                    seq.stream_sender,
                    seq.response_sender,
                    has_kv,
                    cache_id,
                )
            } else {
                return Ok(());
            }
        };

        // Release model executor's KV cache for this sequence (frees GPU memory).
        if let Some(ref cache_id) = model_cache_id {
            self.model_executor.release_cache(cache_id);
        }

        if has_kv_cache {
            let _ = self.kv_cache.deallocate(request_id.clone()).await;
        }

        self.scheduler
            .complete(request_id.clone(), &response)
            .await?;

        if let Some(tx) = response_sender {
            let _ = tx.send(response.clone());
        }

        if let Some(tx) = stream_sender {
            let final_chunk = StreamChunk {
                request_id: request_id.clone(),
                text: String::new(),
                token: None,
                finish_reason: Some(finish_reason),
                usage: Some(response.usage.clone()),
                created_at: chrono::Utc::now(),
                metadata: HashMap::new(),
            };
            let _ = tx.send(Ok(final_chunk)).await;
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
