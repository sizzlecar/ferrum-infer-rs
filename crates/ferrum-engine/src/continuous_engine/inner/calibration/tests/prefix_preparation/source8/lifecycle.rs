use super::*;

async fn collect(early_stop: bool) {
    let maximum = if early_stop { 4 } else { 3 };
    let expected_reason = if early_stop {
        ferrum_types::FinishReason::Stop
    } else {
        ferrum_types::FinishReason::Length
    };
    let (mut session, executor) = source8_session(maximum, false).await;
    let (ids, outputs) = admit_cohort(&mut session, maximum, early_stop).await;
    let consumers: Vec<_> = outputs
        .into_iter()
        .map(|mut output| {
            tokio::spawn(async move {
                let mut wire = Vec::new();
                let mut terminal = false;
                while let Some(frame) = output.frames.next().await {
                    assert!(frame.wire().credit().events > 0);
                    assert!(frame.wire().payload().capacity() <= frame.wire().credit().bytes);
                    wire.extend_from_slice(frame.wire().payload());
                    if frame.metadata().terminal {
                        assert!(!terminal);
                        assert_eq!(frame.metadata().generated_tokens, 3);
                        terminal = true;
                    }
                    drop(frame);
                }
                assert!(terminal);
                let completion = output.completion.await.unwrap();
                let OutputCompletion::Succeeded { reason, usage, .. } = completion.payload() else {
                    panic!("original output failed");
                };
                assert_eq!(*reason, expected_reason);
                assert_eq!(usage.completion_tokens, 3);
                wire
            })
        })
        .collect();

    let mut previous_call = 0;
    for generated in 1..=3 {
        for id in &ids {
            ready(&session, id, false).await;
        }
        let rows = ids
            .iter()
            .map(|id| {
                let f = frontier(&session, id);
                if generated == 1 {
                    f.prefill_work(NonZeroU32::MIN).unwrap()
                } else {
                    f.decode_work_with_route(CalibrationDecodeRoute::FullLogits)
                        .unwrap()
                }
            })
            .collect();
        let report = wave(&mut session, &executor, rows).await;
        assert!(report.error.is_none(), "{report:?}");
        assert_eq!(
            report.submission,
            CalibrationSubmissionState::HostReconciled
        );
        assert!(report.no_submission_proof().is_none());
        let stages = report
            .host_stages
            .as_ref()
            .expect("original actual settlement");
        assert!(stages.call_id > previous_call);
        previous_call = stages.call_id;
        assert_eq!(
            report.host_stage_queue.unwrap().accepted_ordinal,
            Some(generated)
        );
        assert_eq!(stages.rows.len(), 2);
        for (index, row) in stages.rows.iter().enumerate() {
            assert_eq!(row.request_id, ids[index]);
            assert!(row.token_committed_at_ns.is_some() && row.settled_at_ns.is_some());
            if generated == 3 {
                let terminal = row.terminal.as_ref().expect("original terminal receipt");
                assert_eq!(terminal.finish_reason, expected_reason);
                assert_eq!(terminal.generated_tokens, 3);
                assert!(
                    terminal.owner_matched
                        && terminal.terminal_handoff_succeeded
                        && terminal.request_slot_closed
                );
                assert!(
                    !terminal.output_failed
                        && !terminal.physical_failed
                        && !terminal.scheduler_failed
                );
            } else {
                assert!(row.terminal.is_none());
            }
        }
        assert_collecting(&session, generated, 1);
        if generated == 1 {
            release(&mut session, &ids, generated).await;
        } else {
            let original = stages
                .structured_evidence
                .as_ref()
                .expect("original structured CPU producer")
                .as_ref()
                .unwrap_or_else(|e| panic!("original settlement rejected: {e:?}; {stages:?}"));
            original.validate_host_stages(stages).unwrap();
            assert!(original
                .recipe()
                .physical_host_rows()
                .iter()
                .all(|row| matches!(
                    row.installed_policy.empirical_content_domain,
                    Some(HostContentDomainV1::PlainTextInstalledV2(_))
                )));
        }
        if generated == 2 {
            let sequences = session.engine.inner.sequences.read();
            for id in &ids {
                assert_eq!(sequences[id].generated_tokens[1], TokenId::new(12));
                assert!(sequences[id].pending_decoded_utf8_bytes.is_empty());
                assert!(sequences[id].calibration_prefix.is_none());
            }
        }
    }
    let mut wire = Vec::new();
    for consumer in consumers {
        wire.push(bounded(consumer).await.unwrap());
    }
    let sse = std::str::from_utf8(&wire[1]).unwrap();
    assert_eq!(
        sse.lines().filter(|line| *line == "data: [DONE]").count(),
        1
    );
    assert!(sse.contains(if early_stop {
        "\"finish_reason\":\"stop\""
    } else {
        "\"finish_reason\":\"length\""
    }));
    assert!(std::str::from_utf8(&wire[0]).unwrap().starts_with('é'));
    assert!(session.frontiers().unwrap().is_empty());
    session.end_prepared_owner_cohort().unwrap();
    assert_collecting(&session, 3, 1);
    assert_eq!(executor.physical.load(Ordering::Acquire), 3);
    assert_eq!(executor.completion_calls.load(Ordering::Acquire), 2);
    // A completed request is not a qualified calibration population. Other
    // declared cohorts and independent model phases remain unexecuted.
    assert!(session
        .engine
        .inner
        .cost_runtime
        .as_ref()
        .unwrap()
        .snapshot()
        .is_none());
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn source8_engine_actual_cli_sse_preparation_release_continuation_and_length() {
    collect(false).await;
}

#[tokio::test]
async fn source8_engine_actual_cli_sse_installed_stop_keeps_original_terminal() {
    collect(true).await;
}
