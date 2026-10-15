//! Explicit, CPU-only diagnostic of the real token commit/text publication transaction.
//! Inputs and results stay outside the repository. This is not a serving benchmark.
use super::*;
use ferrum_tokenizer::implementations::HuggingFaceTokenizer;
use ferrum_types::SpecialTokens;
use serde::{Deserialize, Serialize};
use std::{path::PathBuf, sync::Mutex};

#[derive(Deserialize)]
struct ProbeInput {
    tokenizer_path: PathBuf,
    requests: Vec<RequestInput>,
    repeats: usize,
    #[serde(default)]
    provenance: serde_json::Value,
}

#[derive(Deserialize)]
struct RequestInput {
    #[serde(default)]
    input_token_ids: Option<Vec<u32>>,
    #[serde(default)]
    output_token_ids: Option<Vec<u32>>,
    #[serde(default)]
    input_text: Option<String>,
    #[serde(default)]
    output_text: Option<String>,
}

#[derive(Default, Serialize)]
struct DecodeCounts {
    calls: u64,
    elapsed_ns: u64,
    target_history_calls: u64,
    target_history_ns: u64,
    stream_target_calls: u64,
    stream_target_ns: u64,
    incremental_calls: u64,
    incremental_ns: u64,
}

#[derive(Default)]
struct CounterState {
    target: Vec<TokenId>,
    phase: Phase,
    counts: DecodeCounts,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum Phase {
    #[default]
    Unmeasured,
    Commit,
    Stop,
    Stream,
    Terminal,
}

struct MeasuredTokenizer {
    inner: HuggingFaceTokenizer,
    state: Mutex<CounterState>,
}

fn ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

impl MeasuredTokenizer {
    fn begin(&self, target: Vec<TokenId>) {
        *self.state.lock().unwrap() = CounterState {
            target,
            phase: Phase::Commit,
            counts: DecodeCounts::default(),
        };
    }

    fn phase(&self, phase: Phase) {
        self.state.lock().unwrap().phase = phase;
    }

    fn finish(&self) -> DecodeCounts {
        let mut state = self.state.lock().unwrap();
        state.phase = Phase::Unmeasured;
        std::mem::take(&mut state.counts)
    }
}

impl Tokenizer for MeasuredTokenizer {
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>> {
        self.inner.encode(text, add_special)
    }

    fn decode(&self, tokens: &[TokenId], skip_special: bool) -> Result<String> {
        let started = Instant::now();
        let result = self.inner.decode(tokens, skip_special);
        let elapsed = ns(started.elapsed());
        let mut state = self.state.lock().unwrap();
        if state.phase != Phase::Unmeasured && skip_special {
            state.counts.calls += 1;
            state.counts.elapsed_ns += elapsed;
            if tokens == state.target {
                state.counts.target_history_calls += 1;
                state.counts.target_history_ns += elapsed;
                if state.phase == Phase::Stream {
                    state.counts.stream_target_calls += 1;
                    state.counts.stream_target_ns += elapsed;
                }
            }
        }
        result
    }

    fn decode_incremental(&self, prev: &[TokenId], next: TokenId) -> Result<String> {
        let started = Instant::now();
        let result = self.inner.decode_incremental(prev, next);
        let elapsed = ns(started.elapsed());
        let mut state = self.state.lock().unwrap();
        if state.phase != Phase::Unmeasured {
            // Internal calls made by HF's incremental implementation do not
            // reenter this wrapper. Report their encompassing time separately.
            state.counts.incremental_calls += 1;
            state.counts.incremental_ns += elapsed;
        }
        result
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn special_tokens(&self) -> &SpecialTokens {
        self.inner.special_tokens()
    }
    fn token_id(&self, text: &str) -> Option<TokenId> {
        self.inner.token_id(text)
    }
    fn token_text(&self, token: TokenId) -> Option<&str> {
        self.inner.token_text(token)
    }
    fn token_bytes(&self, token: TokenId) -> Option<Vec<u8>> {
        self.inner.token_bytes(token)
    }
    fn apply_chat_template(
        &self,
        messages: &[ferrum_interfaces::tokenizer::ChatMessage],
    ) -> Result<String> {
        self.inner.apply_chat_template(messages)
    }
    fn info(&self) -> TokenizerInfo {
        self.inner.info()
    }
}

#[derive(Serialize)]
struct Transaction {
    repeat: usize,
    request: usize,
    step: usize,
    active_requests: usize,
    greedy: bool,
    terminal: bool,
    whole_ns: u64,
    visible_chunks: usize,
    visible_bytes: usize,
    decode: DecodeCounts,
    /// Only the ordinary streaming call is counted, and only if an earlier
    /// call decoded the exact same token sequence. This is an optimistic
    /// removed-work bound, not an implemented speedup or a GPU-gap estimate.
    one_duplicate_decode_upper_ns: u64,
}

fn resolve_tokens(
    tokenizer: &dyn Tokenizer,
    ids: &Option<Vec<u32>>,
    text: &Option<String>,
) -> Vec<TokenId> {
    match (ids, text) {
        (Some(ids), None) => ids.iter().copied().map(TokenId::new).collect(),
        (None, Some(text)) => tokenizer.encode(text, false).unwrap(),
        _ => panic!("provide exactly one of token IDs or text for each input/output"),
    }
}

#[tokio::test]
#[ignore = "CPU diagnostic: requires FERRUM_OUTPUT_DECODE_BUDGET_INPUT and a real local HF tokenizer"]
async fn real_hf_complete_token_text_transaction_budget() {
    let config_path = std::env::var_os("FERRUM_OUTPUT_DECODE_BUDGET_INPUT")
        .expect("set the explicit diagnostic input JSON path");
    let config: ProbeInput = serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
    assert!(!config.requests.is_empty() && config.repeats > 0);
    let tokenizer_file = if config.tokenizer_path.is_dir() {
        config.tokenizer_path.join("tokenizer.json")
    } else {
        config.tokenizer_path.clone()
    };
    let tokenizer = Arc::new(MeasuredTokenizer {
        inner: HuggingFaceTokenizer::from_file(tokenizer_file.to_str().unwrap())
            .await
            .unwrap(),
        state: Mutex::new(CounterState::default()),
    });
    let tokens = config
        .requests
        .iter()
        .map(|request| {
            let input = resolve_tokens(
                tokenizer.as_ref(),
                &request.input_token_ids,
                &request.input_text,
            );
            let output = resolve_tokens(
                tokenizer.as_ref(),
                &request.output_token_ids,
                &request.output_text,
            );
            assert!(!input.is_empty() && !output.is_empty());
            (input, output)
        })
        .collect::<Vec<_>>();
    let mut transactions = Vec::new();
    for repeat in 0..config.repeats {
        let mut engine_config = EngineConfig::default();
        engine_config.scheduler.max_running_requests = engine_config
            .scheduler
            .max_running_requests
            .max(tokens.len());
        engine_config.scheduler.max_waiting_requests = engine_config
            .scheduler
            .max_waiting_requests
            .max(tokens.len());
        let scheduler = Arc::new(ContinuousBatchScheduler::new(
            engine_config.scheduler.clone(),
        ));
        let engine = ContinuousBatchEngine::new(
            engine_config,
            scheduler.clone(),
            tokenizer.clone(),
            Arc::new(crate::registry::GreedySampler),
            Arc::new(MockKvCacheManager::new(256)),
            Arc::new(MockModelExecutor::instant(tokenizer.vocab_size())),
            Arc::new(MockTensorFactory),
        )
        .unwrap();
        let mut requests = Vec::new();
        for (input, output) in &tokens {
            let mut request = policy_request();
            request.stream = true;
            request.sampling_params.max_tokens = output.len();
            let rid = request.id.clone();
            scheduler.submit(request.clone()).await.unwrap();
            let mut sequence =
                SequenceState::new_with_tokenizer(request, input.clone(), Some(tokenizer.clone()));
            let (sender, receiver) = mpsc::channel(4);
            let (response_sender, response_receiver) = tokio::sync::oneshot::channel();
            sequence.stream_sender = Some(sender);
            sequence.response_sender = Some(response_sender);
            engine.inner.sequences.write().insert(rid.clone(), sequence);
            requests.push((rid, receiver, Some(response_receiver), String::new(), false));
        }
        // Put every registered request in an actual scheduler active queue so
        // the real terminal completion can retire it. No model runs here.
        for _ in 0..requests.len() {
            if scheduler.waiting_count() == 0 {
                break;
            }
            let batch = scheduler
                .next_batch(ferrum_interfaces::BatchHint::simple(requests.len()))
                .await
                .expect("diagnostic requests must admit to the scheduler");
            for scheduled in batch.requests {
                scheduler.mark_prefill_complete(&scheduled.request.id, 1);
            }
        }
        assert_eq!(scheduler.waiting_count(), 0);
        let mut logits = Vec::new();
        for step in 0..tokens.iter().map(|(_, output)| output.len()).max().unwrap() {
            let active = tokens
                .iter()
                .filter(|(_, output)| step < output.len())
                .count();
            for (request_index, (_, output)) in tokens.iter().enumerate() {
                let Some(&token) = output.get(step) else {
                    continue;
                };
                let (rid, receiver, response_receiver, visible, completed) =
                    &mut requests[request_index];
                assert!(
                    !*completed,
                    "frozen history continued after a real stop boundary"
                );
                let target = output[..=step].to_vec();
                tokenizer.begin(target);
                let started = Instant::now();
                let greedy = {
                    let mut sequences = engine.inner.sequences.write();
                    let sequence = sequences.get_mut(rid).unwrap();
                    let greedy = sequence.can_use_model_greedy_argmax();
                    if greedy {
                        sequence
                            .validate_and_commit_model_greedy_argmax_token(
                                Some(tokenizer.as_ref()),
                                token,
                            )
                            .unwrap();
                    } else {
                        // Only needed for a real split UTF-8 continuation. No
                        // alternate token is silently accepted by this replay.
                        // Its synthetic-logits construction is included in
                        // this separately identified fallback transaction.
                        logits.resize(tokenizer.vocab_size(), f32::NEG_INFINITY);
                        logits.fill(f32::NEG_INFINITY);
                        logits[token.get() as usize] = 0.0;
                        let actual = sequence
                            .sample_and_commit_with_processors_and_tokenizer(
                                &mut logits,
                                Some(tokenizer.as_ref()),
                            )
                            .unwrap();
                        assert_eq!(actual, token);
                    }
                    greedy
                };
                tokenizer.phase(Phase::Stop);
                let stop = engine.inner.stop_reason_for_request(rid);
                if engine.inner.should_stream_generated_token(rid, token, stop) {
                    tokenizer.phase(Phase::Stream);
                    engine.inner.send_stream_update(rid, token).await;
                }
                if let Some(reason) = stop {
                    tokenizer.phase(Phase::Terminal);
                    engine.inner.complete_request(rid, reason).await.unwrap();
                    *completed = true;
                }
                let mut visible_chunks = 0;
                let mut visible_bytes = 0;
                while let Ok(chunk) = receiver.try_recv() {
                    let chunk = chunk.unwrap();
                    if !chunk.text.is_empty() {
                        visible_chunks += 1;
                        visible_bytes += chunk.text.len();
                        visible.push_str(&chunk.text);
                    }
                }
                let whole_ns = ns(started.elapsed());
                let decode = tokenizer.finish();
                let duplicate = if stop.is_none()
                    && decode.stream_target_calls == 1
                    && decode.target_history_calls >= 2
                {
                    decode.stream_target_ns
                } else {
                    0
                };
                if *completed {
                    assert_eq!(
                        step + 1,
                        output.len(),
                        "frozen history contains tokens after stop"
                    );
                    let response = response_receiver.take().unwrap().await.unwrap().unwrap();
                    assert_eq!(&response.tokens, output);
                    assert_eq!(
                        response.text.as_str(),
                        visible.as_str(),
                        "channel text differs from final response"
                    );
                }
                transactions.push(Transaction {
                    repeat,
                    request: request_index,
                    step,
                    active_requests: active,
                    greedy,
                    terminal: stop.is_some(),
                    whole_ns,
                    visible_chunks,
                    visible_bytes,
                    decode,
                    one_duplicate_decode_upper_ns: duplicate,
                });
            }
        }
        assert!(requests.iter().all(|request| request.4));
        assert!(engine.inner.sequences.read().is_empty());
        engine.shutdown().await.unwrap();
    }
    let ordinary = transactions
        .iter()
        .filter(|t| t.greedy && !t.terminal)
        .collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::json!({
            "diagnostic": "real_hf_complete_token_text_transaction_budget",
            "provenance": config.provenance,
            "tokenizer_path": tokenizer_file,
            "requests": tokens.iter().map(|(i,o)| serde_json::json!({"input_tokens":i.len(),"output_tokens":o.len()})).collect::<Vec<_>>(),
            "repeats": config.repeats,
            "ordinary_greedy_transactions": ordinary.len(),
            "ordinary_whole_ns": ordinary.iter().map(|t|u128::from(t.whole_ns)).sum::<u128>().to_string(),
            "ordinary_one_duplicate_decode_upper_ns": ordinary.iter().map(|t|u128::from(t.one_duplicate_decode_upper_ns)).sum::<u128>().to_string(),
            "limitations": ["CPU text-publication transaction replay; no model/GPU/network throughput claim", "Production KV-resource commit, scheduler decode-progress update and decode-token counters are excluded; this is not the full readback-to-next-launch interval", "All requests start together; round-robin replay is not the recorded server schedule", "Tokenization, state construction and target-history instrumentation setup are outside timing; one-hot logits construction is included only in separately identified fallback transactions", "Wrapper clocks, matching, counters, channel drain and existing yield are inside whole timing", "Incremental decoder time is separate; its nested HF calls do not reenter the counting wrapper", "Duplicate bound excludes terminal transactions and assumes removal without replacement cost"],
            "transactions": transactions,
        })
    );
}
