use super::*;
use futures::StreamExt;

fn stop_boundary_engine(pieces: &[&str], end_with_eos: bool) -> ContinuousBatchEngine {
    let mut pairs = vec![("test", 5), ("", 6)];
    pairs.extend(
        pieces
            .iter()
            .enumerate()
            .map(|(i, text)| (*text, i as u32 + 7)),
    );
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(128, &pairs));
    // The reusable executor repeats its first token at prefill and first decode.
    // An empty decoded piece leaves the visible sequence exactly as specified.
    let mut tokens = (6..7 + pieces.len() as u32).collect::<Vec<_>>();
    if end_with_eos {
        tokens.push(3); // PolicyTokenizer's ordinary EOS has no decoded text.
    }
    let config = EngineConfig::default();
    ContinuousBatchEngine::new(
        config.clone(),
        Arc::new(ContinuousBatchScheduler::new(config.scheduler)),
        tokenizer,
        Arc::new(crate::registry::GreedySampler),
        Arc::new(MockKvCacheManager::new(256)),
        Arc::new(ferrum_testkit::ConfigurableModelExecutor::with_token_sequence(128, tokens)),
        Arc::new(MockTensorFactory),
    )
    .unwrap()
}

async fn stop_boundary_outputs(
    pieces: &[&str],
    stops: &[&str],
) -> (InferenceResponse, Vec<StreamChunk>) {
    stop_boundary_outputs_with_eos(pieces, stops, false).await
}

async fn stop_boundary_outputs_with_eos(
    pieces: &[&str],
    stops: &[&str],
    end_with_eos: bool,
) -> (InferenceResponse, Vec<StreamChunk>) {
    let mut request = policy_request();
    request.sampling_params.stop_sequences = stops.iter().map(|s| (*s).to_owned()).collect();
    request.sampling_params.max_tokens = pieces.len() + 2 + usize::from(end_with_eos);

    let engine = stop_boundary_engine(pieces, end_with_eos);
    let response = tokio::time::timeout(Duration::from_secs(2), engine.infer(request.clone()))
        .await
        .expect("fixed token sequence must complete")
        .unwrap();
    engine.shutdown().await.unwrap();

    let engine = stop_boundary_engine(pieces, end_with_eos);
    request.id = RequestId::new();
    request.stream = true;
    let mut stream = engine.infer_stream(request).await.unwrap();
    let chunks = tokio::time::timeout(Duration::from_secs(2), async {
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap());
        }
        chunks
    })
    .await
    .expect("fixed token stream must terminate");
    engine.shutdown().await.unwrap();
    (response, chunks)
}

#[tokio::test]
async fn stop_boundary_react_marker_never_leaks_into_stream() {
    let (response, chunks) = stop_boundary_outputs(
        &["Action Input: wiki", "\n", "Observ", "ation"],
        &["\nObservation"],
    )
    .await;
    let text = chunks
        .iter()
        .map(|chunk| chunk.text.as_str())
        .collect::<String>();
    assert_eq!(
        text, "Action Input: wiki",
        "stop prefixes must remain buffered"
    );
    assert_eq!(
        response.text, text,
        "sync and stream must have the same boundary"
    );
    assert_eq!(response.finish_reason, FinishReason::Stop);
    let terminal = chunks.last().unwrap();
    assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
    let usage = terminal.usage.as_ref().unwrap();
    assert_eq!(usage.prompt_tokens, response.usage.prompt_tokens);
    assert_eq!(usage.completion_tokens, response.usage.completion_tokens);
    assert_eq!(usage.total_tokens, response.usage.total_tokens);
    assert_eq!(
        response.usage.completion_tokens, 6,
        "stop tokens still consumed model work"
    );
}

#[tokio::test]
async fn stop_boundary_inside_token_preserves_preceding_text() {
    let (response, chunks) = stop_boundary_outputs(&["OK \n\n"], &["\n"]).await;
    let text = chunks
        .iter()
        .map(|chunk| chunk.text.as_str())
        .collect::<String>();
    assert_eq!(
        text, "OK ",
        "a stop must not discard the valid text before it"
    );
    assert_eq!(response.text, text);
    assert_eq!(response.finish_reason, FinishReason::Stop);
}

#[tokio::test]
async fn stop_boundary_flushes_and_selects_the_earliest_match() {
    struct Case<'a> {
        pieces: &'a [&'a str],
        stops: &'a [&'a str],
        expected: &'a str,
        finish: FinishReason,
        end_with_eos: bool,
    }
    for case in [
        Case {
            pieces: &["answer", "ab"],
            stops: &["abc"],
            expected: "answerab",
            finish: FinishReason::Length,
            end_with_eos: false,
        },
        Case {
            pieces: &["answer", "ab"],
            stops: &["abc"],
            expected: "answerab",
            finish: FinishReason::Stop,
            end_with_eos: true,
        },
        Case {
            pieces: &["keep:", "ab", "cSTOPtail"],
            stops: &["", "STOP", "bc", "abc", "abcd"],
            expected: "keep:",
            finish: FinishReason::Stop,
            end_with_eos: false,
        },
        Case {
            pieces: &["答案：", "终", "止后续"],
            stops: &["终止"],
            expected: "答案：",
            finish: FinishReason::Stop,
            end_with_eos: false,
        },
    ] {
        let (response, chunks) =
            stop_boundary_outputs_with_eos(case.pieces, case.stops, case.end_with_eos).await;
        let text = chunks
            .iter()
            .map(|chunk| chunk.text.as_str())
            .collect::<String>();
        assert_eq!(
            text, case.expected,
            "pieces={:?}, stops={:?}",
            case.pieces, case.stops
        );
        assert_eq!(response.text, text);
        assert_eq!(response.finish_reason, case.finish);
        assert_eq!(chunks.last().unwrap().finish_reason, Some(case.finish));
    }
}

#[tokio::test]
async fn stop_boundary_mismatch_releases_the_prefix_before_later_output() {
    let (response, chunks) = stop_boundary_outputs(&["ab", "x", " tail"], &["abc"]).await;
    let mut text = String::new();
    let mut released_after_mismatch = false;
    for chunk in chunks {
        text.push_str(&chunk.text);
        released_after_mismatch |= text == "abx" && chunk.finish_reason.is_none();
    }
    assert!(
        released_after_mismatch,
        "a disproven stop prefix must become visible before subsequent output"
    );
    assert_eq!(text, "abx tail");
    assert_eq!(response.text, text);
    assert_eq!(response.finish_reason, FinishReason::Length);
}

#[test]
fn stop_boundary_buffer_does_not_hide_a_non_stop_control_token() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        6,
        &[("test", 0), ("ab", 4), ("x", 5), ("<think>", 7)],
    ));
    let mut request = policy_request();
    request.sampling_params.stop_sequences = vec!["abc".to_owned()];
    let mut state =
        SequenceState::new_with_tokenizer(request, vec![TokenId::new(0)], Some(tokenizer.clone()));
    state.generated_tokens.push(TokenId::new(4));
    // "ab" is valid decoded text, withheld solely because "abc" may follow.
    state.streamed_text_len = 0;
    state.decoded_text_len = 2;
    assert!(state.allowed_extended_token_ids.contains(&7));
    assert!(!state.stop_token_ids.contains(&7));

    let mut logits = vec![f32::NEG_INFINITY; 8];
    logits[5] = 1.0;
    logits[7] = 100.0;
    let sampled = state
        .sample_and_commit_with_processors_and_tokenizer(&mut logits, Some(tokenizer.as_ref()))
        .unwrap();

    assert_eq!(
        sampled,
        TokenId::new(5),
        "hidden non-stop control must be rejected even while visible text is buffered"
    );
    assert_eq!(logits[7], f32::NEG_INFINITY);
}

#[tokio::test]
async fn stop_boundary_rejects_strict_json_truncated_inside_a_token() {
    let make_engine = |generated: &str, stop: &str, terminal_token: Option<u32>| {
        let mut pairs = vec![("test", 5), (generated, 7), (stop, 8)];
        if terminal_token == Some(3) {
            pairs.push(("</s>", 3));
        }
        let mut tokenizer = PolicyTokenizer::new(16, &pairs);
        if terminal_token == Some(8) {
            // After grammar acceptance the fixture's tied fallback logits
            // must choose the explicit whitespace stop before ordinary EOS.
            tokenizer.special.eos_token = Some(TokenId::new(15));
        }
        let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(tokenizer);
        let config = EngineConfig::default();
        ContinuousBatchEngine::new(
            config.clone(),
            Arc::new(ContinuousBatchScheduler::new(config.scheduler)),
            tokenizer,
            Arc::new(crate::registry::GreedySampler),
            Arc::new(MockKvCacheManager::new(256)),
            Arc::new(ferrum_testkit::ConfigurableModelExecutor::with_token_sequence(16, vec![7])),
            Arc::new(MockTensorFactory),
        )
        .unwrap()
    };
    let object_schema = r#"{"type":"object","properties":{"value":{"type":"integer"}},"required":["value"],"additionalProperties":false}"#;
    for (generated, schema, stop, expected, terminal_token) in [
        (r#"{"value":1}"#, object_schema, "}", None, None),
        ("123", r#"{"enum":[123]}"#, "3", None, None),
        (
            r#"{"value":1}"#,
            object_schema,
            "\n",
            Some(r#"{"value":1}"#),
            Some(8),
        ),
        ("123", r#"{"enum":[123]}"#, "</s>", Some("123"), Some(3)),
    ] {
        let mut request = policy_request();
        request.sampling_params.max_tokens = 1;
        request.sampling_params.response_format =
            ferrum_types::ResponseFormat::JsonSchema(schema.to_string());

        // The same ordinary token must produce valid structured output without
        // a stop. A failure setting up the grammar is not evidence of this bug.
        let engine = make_engine(generated, stop, terminal_token);
        let response = engine.infer(request.clone()).await.unwrap_or_else(|error| {
            panic!("no-stop control failed: generated={generated:?}, stop={stop:?}: {error}")
        });
        assert_eq!(response.text, generated);
        engine.shutdown().await.unwrap();

        // Stop token 8 does not mask ordinary token 7. A stop inside its JSON
        // value must fail, while a stop in trailing whitespace is harmless.
        request.id = RequestId::new();
        request.sampling_params.stop_sequences = vec![stop.to_string()];
        request.sampling_params.max_tokens += usize::from(terminal_token.is_some());
        let engine = make_engine(generated, stop, terminal_token);
        let result = engine.infer(request.clone()).await;
        engine.shutdown().await.unwrap();
        if let Some(expected) = expected {
            let response = result.unwrap_or_else(|error| {
                panic!("stop after complete value failed: generated={generated:?}, stop={stop:?}: {error}")
            });
            assert_eq!(response.text, expected);
            assert_eq!(response.finish_reason, FinishReason::Stop);
            if let Some(terminal_token) = terminal_token {
                assert_eq!(response.tokens.last(), Some(&TokenId::new(terminal_token)));
            }
        } else {
            assert!(
                result.is_err(),
                "stop {stop:?} must not turn {generated:?} into successful strict output: {result:?}"
            );
        }

        let engine = make_engine(generated, stop, terminal_token);
        request.id = RequestId::new();
        request.stream = true;
        let mut stream = engine.infer_stream(request).await.unwrap();
        let chunks = tokio::time::timeout(Duration::from_secs(2), async {
            let mut chunks = Vec::new();
            while let Some(chunk) = stream.next().await {
                chunks.push(chunk);
            }
            chunks
        })
        .await
        .expect("structured output must terminate");
        engine.shutdown().await.unwrap();
        if let Some(expected) = expected {
            let chunks = chunks.into_iter().collect::<Result<Vec<_>>>().unwrap();
            let text = chunks
                .iter()
                .map(|chunk| chunk.text.as_str())
                .collect::<String>();
            assert_eq!(text, expected);
            assert_eq!(
                chunks.last().unwrap().finish_reason,
                Some(FinishReason::Stop)
            );
        } else {
            assert!(chunks.iter().any(Result::is_err), "chunks: {chunks:?}");
            assert!(
                chunks
                    .iter()
                    .filter_map(|chunk| chunk.as_ref().ok())
                    .all(|chunk| {
                        chunk.text.is_empty()
                            && chunk.finish_reason.is_none()
                            && chunk.usage.is_none()
                    }),
                "invalid structured tail must not be flushed as successful output: {chunks:?}"
            );
        }
    }
}
