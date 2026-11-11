use super::*;

#[test]
fn real_comfy_float_input_history_requires_explicit_float_workflow_literals() {
    // Reduced from the first real Comfy history: validate_inputs coerces FLOAT
    // fields to Python float and writes them back into the original graph.
    // Keep the raw JSON so serde preserves the distinction between 1 and 1.0.
    let history: Value = serde_json::from_str(
        r#"{"27":{"class_type":"MiniMaxH3SigmaShift","inputs":{"model":["26",0],"shift_video":1.0,"shift_audio":1.0},"_meta":{"title":"MiniMaxH3SigmaShift"}}}"#,
    )
    .unwrap();
    let mut submitted = history.clone();
    for field in ["shift_video", "shift_audio"] {
        assert!(history["27"]["inputs"][field].is_f64());
        submitted["27"]["inputs"][field] = json!(1);
    }
    assert_ne!(
        execution_audit::execution_graph_sha(&submitted).unwrap(),
        execution_audit::execution_graph_sha(&history).unwrap()
    );

    // Fix the workflow's declared FLOAT literals, not the binding validator.
    for field in ["shift_video", "shift_audio"] {
        submitted["27"]["inputs"][field] = json!(1.0);
    }
    submitted["27"].as_object_mut().unwrap().remove("_meta");
    assert_eq!(
        execution_audit::execution_graph_sha(&submitted).unwrap(),
        execution_audit::execution_graph_sha(&history).unwrap()
    );

    // Integer seed identity must remain exact; a generic as_f64 comparison
    // would collapse these adjacent u64 values.
    let mut first = json!({"10":{"class_type":"RandomNoise","inputs":{"noise_seed":u64::MAX - 1}}});
    let second = json!({"10":{"class_type":"RandomNoise","inputs":{"noise_seed":u64::MAX}}});
    assert_ne!(
        execution_audit::execution_graph_sha(&first).unwrap(),
        execution_audit::execution_graph_sha(&second).unwrap()
    );
    first["10"]["inputs"]["noise_seed"] = json!(u64::MAX.to_string());
    assert_ne!(
        execution_audit::execution_graph_sha(&first).unwrap(),
        execution_audit::execution_graph_sha(&second).unwrap()
    );
}

fn audited(origin: String, directory: &std::path::Path) -> ImageBenchmarkConfig {
    let mut config = comfy_config(origin, directory);
    let ImageEndpointProtocol::Comfy(protocol) = &mut config.endpoints[0].protocol else {
        unreachable!()
    };
    protocol.graph_cache_disabled_declared = false;
    protocol.execution_audit = Some(ComfyExecutionAuditConfig {
        limits: Default::default(),
    });
    for workflow in [&mut protocol.generation, &mut protocol.edit]
        .into_iter()
        .flatten()
    {
        workflow.required_execution_nodes = vec!["input".into(), "out".into()];
    }
    config
}

#[test]
fn node_event_observation_keeps_receive_order_without_foreign_or_reconnect_events() {
    use execution_audit::ComfyExecutionEventKind as Kind;
    let directory = tempfile::tempdir().unwrap();
    let mut spec = template(directory.path(), false);
    spec.required_execution_nodes = vec!["input".into(), "out".into()];
    let workflow = PreparedWorkflow::load(&spec, false, &ComfyProtocolLimits::default()).unwrap();
    let mut audit = ComfyExecutionAuditEvidence::default();
    let events = [
        json!({"type":"execution_start","data":{"prompt_id":"owned","timestamp":4242}}),
        json!({"type":"execution_cached","data":{"prompt_id":"owned","nodes":[]}}),
        json!({"type":"executing","data":{"prompt_id":"owned","node":"input"}}),
        json!({"type":"executing","data":{"prompt_id":"foreign","node":"out"}}),
        json!({"type":"executed","data":{"prompt_id":"owned","node":"input","output":{"preview":"not retained"}}}),
        json!({"type":"executing","data":{"prompt_id":"owned","node":"dynamic:child","display_node":"input"}}),
        json!({"type":"executing","data":{"prompt_id":"owned","node":"input"}}),
        json!({"type":"executing","data":{"node":"out"}}),
        json!({"type":"executing","data":{"prompt_id":"owned","node":"out"}}),
        json!({"type":"execution_success","data":{"prompt_id":"foreign"}}),
        json!({"type":"execution_success","data":{"prompt_id":"owned"}}),
    ];
    for (index, event) in events.iter().enumerate() {
        audit.messages += 1;
        let done = execution_audit::consume(
            event,
            "owned",
            &workflow,
            &mut audit,
            Duration::from_millis(10 + index as u64),
        )
        .unwrap();
        assert_eq!(done, index == events.len() - 1);
    }
    let observed = audit
        .events
        .iter()
        .map(|event| {
            (
                event.message_ordinal,
                event.client_received_elapsed_ns,
                event.kind,
                event.node.as_deref(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        observed,
        [
            (1, 10_000_000, Kind::ExecutionStart, None),
            (2, 11_000_000, Kind::ExecutionCached, None),
            (3, 12_000_000, Kind::Executing, Some("input")),
            (5, 14_000_000, Kind::Executed, Some("input")),
            (6, 15_000_000, Kind::Executing, Some("dynamic:child")),
            (7, 16_000_000, Kind::Executing, Some("input")),
            (9, 18_000_000, Kind::Executing, Some("out")),
            (11, 20_000_000, Kind::ExecutionSuccess, None),
        ]
    );
    assert_eq!(audit.events[0].server_timestamp_ms, Some(4242));
    assert_eq!(audit.events[4].display_node.as_deref(), Some("input"));
    assert!(!serde_json::to_string(&audit.events)
        .unwrap()
        .contains("not retained"));
    assert_eq!(
        audit.executed_required_nodes,
        ["input".into(), "out".into()].into()
    );
}

#[test]
fn ui_executed_events_do_not_replace_required_execution_or_success() {
    let directory = tempfile::tempdir().unwrap();
    let mut spec = template(directory.path(), false);
    spec.required_execution_nodes = vec!["input".into(), "out".into()];
    let workflow = PreparedWorkflow::load(&spec, false, &ComfyProtocolLimits::default()).unwrap();
    let mut audit = ComfyExecutionAuditEvidence::default();
    for event in [
        json!({"type":"execution_start","data":{"prompt_id":"owned"}}),
        json!({"type":"execution_cached","data":{"prompt_id":"owned","nodes":[]}}),
        json!({"type":"executed","data":{"prompt_id":"owned","node":"input"}}),
        json!({"type":"executed","data":{"prompt_id":"owned","node":"out"}}),
    ] {
        audit.messages += 1;
        assert!(!execution_audit::consume(
            &event,
            "owned",
            &workflow,
            &mut audit,
            Duration::from_millis(1),
        )
        .unwrap());
    }
    assert!(audit.executed_required_nodes.is_empty());
    assert!(!audit.execution_success_observed && !audit.complete);
    audit.messages += 1;
    assert!(execution_audit::consume(
        &json!({"type":"execution_success","data":{"prompt_id":"owned"}}),
        "owned",
        &workflow,
        &mut audit,
        Duration::from_millis(2),
    )
    .is_err());
    assert!(!audit.complete);
}

#[tokio::test]
async fn websocket_audit_requires_original_execution_and_accepts_server_graph_metadata() {
    for mode in [Mode::Normal, Mode::MetadataHistory] {
        let state = Fixture::new(mode);
        let (origin, _server) = server(&state).await;
        let directory = tempfile::tempdir().unwrap();
        let report = run_benchmark(audited(origin, directory.path()), dataset(1))
            .await
            .unwrap();
        assert_eq!(report.endpoints[0].measured.succeeded, 1);
        let evidence = &report.endpoints[0].protocol_requests[0];
        let audit = evidence.execution_audit.as_ref().unwrap();
        assert!(audit.complete && audit.history_graph_verified && audit.execution_success_observed);
        assert_eq!(
            audit.executed_required_nodes,
            ["input".into(), "out".into()].into()
        );
        assert_eq!(state.submitted.lock().unwrap().len(), 1);
        assert!(evidence.body_complete_elapsed_ms.is_some());
        use execution_audit::ComfyExecutionEventKind as Kind;
        assert_eq!(
            audit
                .events
                .iter()
                .map(|event| (event.kind, event.node.as_deref()))
                .collect::<Vec<_>>(),
            [
                (Kind::ExecutionStart, None),
                (Kind::ExecutionCached, None),
                (Kind::Executing, Some("input")),
                (Kind::Executing, Some("out")),
                (Kind::ExecutionSuccess, None),
            ]
        );
        assert!(audit.events.windows(2).all(|pair| {
            pair[0].message_ordinal < pair[1].message_ordinal
                && pair[0].client_received_elapsed_ns <= pair[1].client_received_elapsed_ns
        }));
        assert!(audit.events[0].message_ordinal > 1); // Status/reconnect are not node evidence.
        assert!(
            audit.events[0].client_received_elapsed_ns as f64 / 1_000_000.0 >= audit.handshake_ms
        );
        assert!(
            audit.events.last().unwrap().client_received_elapsed_ns as f64 / 1_000_000.0
                <= evidence.protocol_elapsed_ms
        );
    }
}

#[tokio::test]
async fn websocket_audit_fails_closed_for_cached_missing_foreign_disconnected_and_changed_work() {
    for mode in [
        Mode::CachedRequired,
        Mode::MissingExecution,
        Mode::ForeignExecution,
        Mode::AuditDisconnect,
        Mode::MutatedHistory,
    ] {
        let state = Fixture::new(mode);
        let (origin, _server) = server(&state).await;
        let directory = tempfile::tempdir().unwrap();
        let report = run_benchmark(audited(origin, directory.path()), dataset(1))
            .await
            .unwrap();
        assert_eq!(report.endpoints[0].measured.succeeded, 0);
        assert!(report.endpoints[0].requests[0].failure.is_some());
        let evidence = &report.endpoints[0].protocol_requests[0];
        assert!(!evidence.execution_audit.as_ref().unwrap().complete);
        let audit = evidence.execution_audit.as_ref().unwrap();
        if mode == Mode::ForeignExecution {
            assert!(audit.events.iter().all(|event| event.node.is_none()));
        }
        if mode == Mode::AuditDisconnect {
            assert!(!audit.execution_success_observed);
            assert!(audit.events.iter().all(|event| {
                event.kind != execution_audit::ComfyExecutionEventKind::ExecutionSuccess
            }));
        }
        assert!(matches!(
            evidence.cleanup,
            ComfyCleanupOutcome::Terminal | ComfyCleanupOutcome::CancelledQuiescent
        ));
        assert_eq!(state.submitted.lock().unwrap().len(), 1);
    }
}

#[tokio::test]
async fn websocket_audit_budgets_frames_messages_and_total_bytes_without_resubmission() {
    for limit in 0..3 {
        let state = Fixture::new(if limit == 0 {
            Mode::OversizedAudit
        } else {
            Mode::Normal
        });
        let (origin, _server) = server(&state).await;
        let directory = tempfile::tempdir().unwrap();
        let mut config = audited(origin, directory.path());
        let ImageEndpointProtocol::Comfy(protocol) = &mut config.endpoints[0].protocol else {
            unreachable!()
        };
        let limits = &mut protocol.execution_audit.as_mut().unwrap().limits;
        match limit {
            0 => limits.maximum_message_bytes = 256,
            1 => limits.maximum_messages = 2,
            _ => {
                limits.maximum_message_bytes = 256;
                limits.maximum_total_bytes = 256;
            }
        }
        let maximum_messages = limits.maximum_messages;
        let report = run_benchmark(config, dataset(1)).await.unwrap();
        assert_eq!(
            report.endpoints[0].requests[0].failure,
            Some(ImageFailureKind::ResponseLimit)
        );
        assert!(state.submitted.lock().unwrap().len() <= 1);
        let audit = report.endpoints[0].protocol_requests[0]
            .execution_audit
            .as_ref()
            .unwrap();
        assert!(!audit.complete);
        assert!(audit.events.len() <= maximum_messages as usize);
        assert!(audit
            .events
            .iter()
            .all(|event| event.message_ordinal <= maximum_messages));
        if limit == 1 {
            // The two allowed status/reconnect messages exhaust the receive
            // budget; no later execution event may be retained after failure.
            assert!(audit.events.is_empty());
        }
    }
}

#[tokio::test]
async fn websocket_audit_tail_is_excluded_from_png_latency_but_retains_closed_loop_slot() {
    let state = Fixture::new(Mode::AuditTail);
    let (origin, _server) = server(&state).await;
    let directory = tempfile::tempdir().unwrap();
    let report = run_benchmark(audited(origin, directory.path()), dataset(2))
        .await
        .unwrap();
    assert_eq!(report.endpoints[0].measured.succeeded, 2);
    for (request, protocol) in report.endpoints[0]
        .requests
        .iter()
        .zip(&report.endpoints[0].protocol_requests)
    {
        let body = protocol.body_complete_elapsed_ms.unwrap();
        assert!((protocol.protocol_elapsed_ms - protocol.audit_tail_ms - body).abs() < 0.001);
        assert!(request.e2e_ms >= body);
        assert!(request.e2e_ms < protocol.protocol_elapsed_ms);
        assert!(protocol.execution_audit.as_ref().unwrap().complete);
    }
    let endpoint = &report.endpoints[0];
    assert!(
        endpoint.arrivals[1].dispatched_offset_ms.unwrap()
            >= endpoint.protocol_requests[0].protocol_elapsed_ms
    );
}

#[tokio::test]
async fn websocket_audit_validates_mode_and_required_graph_nodes_before_network() {
    let state = Fixture::new(Mode::Normal);
    let (origin, _server) = server(&state).await;
    let directory = tempfile::tempdir().unwrap();
    let good = audited(origin, directory.path());
    for invalid in 0..4 {
        let mut config = good.clone();
        let ImageEndpointProtocol::Comfy(protocol) = &mut config.endpoints[0].protocol else {
            unreachable!()
        };
        match invalid {
            0 => protocol.graph_cache_disabled_declared = true,
            1 => protocol
                .generation
                .as_mut()
                .unwrap()
                .required_execution_nodes
                .clear(),
            2 => protocol
                .generation
                .as_mut()
                .unwrap()
                .required_execution_nodes
                .push("input".into()),
            _ => {
                protocol
                    .generation
                    .as_mut()
                    .unwrap()
                    .required_execution_nodes = vec!["absent".into()]
            }
        }
        assert!(run_benchmark(config, dataset(1)).await.is_err());
    }
    assert!(state.submitted.lock().unwrap().is_empty());
    assert!(state.audit_clients.lock().unwrap().is_empty());
}
