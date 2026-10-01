//! CPU cohort primitives, not backend mask residency or numerical qualification.
//! Source5 and actual output actors retain their original failure/credit rules.
use super::*;
use ferrum_interfaces::model_executor::LogitsReturnPolicy;

struct DrainedOutput {
    wire: Vec<u8>,
    tokens: Vec<(usize, TokenId)>,
}

async fn consume_original_output(mut output: CreditedOutputSession) -> DrainedOutput {
    let mut wire = Vec::new();
    let mut tokens = Vec::new();
    let mut terminal = false;
    while let Some(frame) = output.frames.next().await {
        assert!(frame.wire().credit().events > 0);
        assert!(frame.wire().payload().capacity() <= frame.wire().credit().bytes);
        let metadata = frame.metadata();
        if let Some(token) = metadata.token {
            tokens.push((metadata.generated_tokens, token));
        }
        wire.extend_from_slice(frame.wire().payload());
        if metadata.terminal {
            assert!(!terminal, "one actual terminal handoff per owner");
            assert_eq!(metadata.generated_tokens, 4);
            terminal = true;
        }
        // Return every real frame lease before waiting for further work.
        drop(frame);
    }
    assert!(
        terminal,
        "closed output is not a successful terminal receipt"
    );
    let completion = output.completion.await.unwrap();
    let OutputCompletion::Succeeded { reason, usage, .. } = completion.payload() else {
        panic!("the complete original request must succeed");
    };
    assert_eq!(*reason, ferrum_types::FinishReason::Length);
    assert_eq!(usage.completion_tokens, 4);
    // Joining this consumer means the completion lease was returned as well.
    drop(completion);
    DrainedOutput { wire, tokens }
}

pub(super) fn mixed_requests(
    session: &CalibrationSession,
    chat: bool,
) -> Vec<(ferrum_types::InferenceRequest, OutputProjectionContract)> {
    use ferrum_types::{ApiChatRequest, ApiCompletionRequest, ApiRequest, ApiStreamOptions};
    let cli = request(session, 4);
    let mut sse = request(session, 4);
    let model = session.configuration().model.model_id.to_string();
    let contract = if chat {
        sse.metadata.insert(
            ferrum_types::PROMPT_OPENED_REASONING_METADATA_KEY.into(),
            false.into(),
        );
        sse.api_request = Some(ApiRequest::Chat(ApiChatRequest {
            messages: Vec::new(),
            tools: Vec::new(),
            tool_choice: None,
            tool_call_protocol: Default::default(),
            legacy_functions: Vec::new(),
            legacy_function_call: None,
            response_format: None,
            stream_options: Some(ApiStreamOptions {
                include_usage: Some(true),
            }),
        }));
        OutputProjectionContract::chat_sse("probe".into(), model, true)
    } else {
        sse.api_request = Some(ApiRequest::Completion(ApiCompletionRequest {
            prompt: sse.prompt.clone(),
            response_format: None,
        }));
        OutputProjectionContract::completions_sse("probe".into(), model, true)
    };
    vec![(cli, OutputProjectionContract::cli_text()), (sse, contract)]
}

async fn mixed_codec_probe(chat: bool) {
    let (mut session, executor, source) = joint_unstarted_cohort(4).await;
    let mut ids = Vec::new();
    let mut outputs = Vec::new();
    for (request, contract) in mixed_requests(&session, chat) {
        ids.push(request.id.clone());
        outputs.push(
            session
                .add_request(
                    request,
                    InferenceRequestContext::from_ingress(slo_clock_now()),
                    Arc::new(contract),
                )
                .await
                .unwrap(),
        );
    }
    for id in &ids {
        ready(&session, id, false).await;
    }
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    let original: Vec<_> = {
        ids.iter()
            .map(|id| {
                let owner = frontier(&session, id).owner_incarnation();
                let sequences = session.engine.inner.sequences.read();
                let sequence = &sequences[id];
                (
                    serde_json::to_value(&sequence.sampling_params).unwrap(),
                    owner,
                )
            })
            .collect()
    };
    let consumers: Vec<_> = outputs
        .into_iter()
        .map(|output| tokio::spawn(consume_original_output(output)))
        .collect();

    // Each original admission is a separate real turn. No joint offer exists
    // while one declared owner is still waiting for admission.
    for remaining in (0..ids.len()).rev() {
        admit(&mut session).await;
        assert_eq!(session.engine.inner.scheduler.waiting_count(), remaining);
    }
    let mut last_call = 0;
    let mut last_fifo = 0;
    for generated in 1..=4 {
        for id in &ids {
            ready(&session, id, false).await;
        }
        let rows = if generated == 1 {
            joint_prefill(&session, &ids)
        } else {
            ids.iter()
                .map(|id| {
                    frontier(&session, id)
                        .decode_work_with_route(if generated >= 3 {
                            CalibrationDecodeRoute::FullLogits
                        } else {
                            CalibrationDecodeRoute::Actual
                        })
                        .unwrap()
                })
                .collect()
        };
        let report = wave(&mut session, &executor, rows).await;
        assert!(report.error.is_none(), "{:?}", report.error);
        assert_eq!(
            report.submission,
            CalibrationSubmissionState::HostReconciled
        );
        let stages = report.host_stages.as_ref().expect("actual host receipt");
        assert_eq!(stages.rows.len(), 2);
        assert!(stages.call_id > last_call);
        last_call = stages.call_id;
        let ordinal = report.host_stage_queue.unwrap().accepted_ordinal.unwrap();
        assert_eq!(ordinal, last_fifo + 1, "all actual waves retain their FIFO");
        last_fifo = ordinal;
        for (index, row) in stages.rows.iter().enumerate() {
            assert_eq!(row.request_id, ids[index]);
            assert_eq!(row.owner_incarnation, original[index].1.get());
            assert!(row.token_committed_at_ns.is_some());
            assert!(row.settled_at_ns.is_some());
            if generated == 4 {
                let terminal = row.terminal.as_ref().expect("actual Length settlement");
                assert_eq!(terminal.finish_reason, ferrum_types::FinishReason::Length);
                assert_eq!(terminal.generated_tokens, 4);
                assert!(terminal.terminal_handoff_succeeded && terminal.owner_matched);
                assert!(terminal.request_slot_closed);
                assert!(!terminal.output_failed && !terminal.physical_failed);
            } else {
                assert!(
                    row.terminal.is_none(),
                    "ordinary work cannot stand in for Length"
                );
            }
        }
        if generated >= 3 {
            assert_eq!(report.ordered_work.participants().len(), 2);
            assert!(report
                .ordered_work
                .participants()
                .iter()
                .all(|participant| {
                    matches!(
                        participant.selection().decode_policy,
                        Some(LogitsReturnPolicy::FullLogits)
                    )
                }));
        }
        if generated == 2 {
            for id in &ids {
                ready(&session, id, false).await;
            }
            let PrefixReleaseProgressV5::Released { receipts } =
                session.advance_structured_prefix_release_v5().unwrap()
            else {
                panic!("the complete G2 cohort must release through real actors");
            };
            assert_eq!(receipts.len(), 2);
            for (slot, id) in ids.iter().enumerate() {
                let release = receipts
                    .iter()
                    .find(|r| &r.frontier.request_id == id)
                    .unwrap();
                assert_eq!(
                    release.frontier.pending_utf8,
                    if slot == 0 { vec![] } else { vec![0xc3] }
                );
                assert_eq!(release.through_fifo_ordinal, last_fifo);
                assert_eq!(
                    release.actor_applied_output_ordinal,
                    release.frontier.output_accepted_ordinal
                );
                let sequences = session.engine.inner.sequences.read();
                let sequence = &sequences[id];
                assert!(sequence.calibration_prefix.is_none());
                assert_eq!(
                    sequence.cost_policy_signature,
                    Some(release.original_policy_signature)
                );
                assert_eq!(
                    sequence.cost_numeric_policy,
                    Some(release.original_numeric_policy)
                );
                assert_eq!(
                    serde_json::to_value(&sequence.sampling_params).unwrap(),
                    original[slot].0
                );
            }
        }
        if generated == 3 {
            // This is a real original-policy nonterminal suffix. The CPU
            // logits favor token6, but pending C3 constrains slot1 to A9.
            let sequences = session.engine.inner.sequences.read();
            for (slot, id) in ids.iter().enumerate() {
                let sequence = &sequences[id];
                assert_eq!(
                    sequence.generated_tokens[2],
                    TokenId::new(if slot == 0 { 6 } else { 12 })
                );
                assert!(sequence.pending_decoded_utf8_bytes.is_empty());
                assert!(sequence.calibration_prefix.is_none());
                assert_eq!(
                    serde_json::to_value(&sequence.sampling_params).unwrap(),
                    original[slot].0
                );
            }
        }
    }
    let mut drained = Vec::new();
    for consumer in consumers {
        drained.push(bounded(consumer).await.unwrap());
    }
    assert_eq!(drained[0].wire, b"aaokok");
    assert!(drained[1].tokens.contains(&(3, TokenId::new(12))));
    let wire = std::str::from_utf8(&drained[1].wire).unwrap();
    assert_eq!(
        wire.lines().filter(|line| *line == "data: [DONE]").count(),
        1
    );
    let events: Vec<serde_json::Value> = wire
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|line| *line != "[DONE]")
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let text: String = events
        .iter()
        .filter_map(|event| {
            if chat {
                event["choices"][0]["delta"]["content"].as_str()
            } else {
                event["choices"][0]["text"].as_str()
            }
        })
        .collect();
    assert_eq!(text, "aéok");
    assert!(events
        .iter()
        .any(|event| event["choices"][0]["finish_reason"] == "length"));
    assert!(events
        .iter()
        .any(|event| event["usage"]["completion_tokens"] == 4));
    assert_eq!(executor.physical.load(Ordering::Acquire), 4);
    assert_eq!(executor.completion_calls.load(Ordering::Acquire), 2);
    assert!(session.frontiers().unwrap().is_empty());
    let artifact = session.finish_structured_cost_group_v2().await.unwrap();
    assert!(
        artifact.failure.is_some(),
        "unsupported numerical projection stays failed"
    );
    assert!(artifact.children.iter().all(|child| child.model.is_none()));
    let source_records = records(&source);
    assert_eq!(
        source_records
            .iter()
            .filter(|r| r["kind"] == "preparation_completed")
            .count(),
        2
    );
    assert_eq!(
        source_records
            .iter()
            .filter(|r| r["kind"] == "preparation_released")
            .count(),
        2
    );
    assert!(!source_records.iter().any(|r| r["kind"] == "phase_freeze"));
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn probe_primitives_joint_chat_codec_suffix_precedes_length() {
    mixed_codec_probe(true).await;
}

#[tokio::test]
async fn probe_primitives_joint_completions_codec_suffix_precedes_length() {
    mixed_codec_probe(false).await;
}
