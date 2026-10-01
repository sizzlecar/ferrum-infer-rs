//! Ordinary Enforce input/output and constructor-owned asynchronous learning.
//! This is the cold sampling gate; qualified hold/replay is a separate gate.
use super::*;
use ferrum_interfaces::engine::InferenceEngine;
use ferrum_interfaces::output_flow::{CreditedOutputSession, OutputCompletion};
use futures::{FutureExt, StreamExt};

async fn engine() -> (ContinuousBatchEngine, Arc<ControlledExecutor>) {
    let (tokenizer, executor) = startup_checkpoint_components(2).await;
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    executor
        .emit_structured_cost_observations
        .store(true, Ordering::Release);
    let mut config = ferrum_types::EngineConfig::default();
    // Parse the same shared product default; no manual cost/reference artifact.
    config.scheduler.slo = serde_json::from_value(serde_json::json!({"mode":"enforce"})).unwrap();
    let slo = &mut config.scheduler.slo;
    let ferrum_types::SloLiveStructuredCalibration::AutomaticV1 { settings } =
        &mut slo.cost_observation.live_structured_calibration
    else {
        panic!("ordinary Enforce must select automatic calibration")
    };
    // Each fixture begins cold and writes no process-shared cache.
    settings.reuse = ferrum_types::SloAutomaticCalibrationReuseV1::Disabled {};
    slo.default_service_class = Some("native-prefix-cpu".into());
    slo.services.push(ferrum_types::ServiceSloConfig {
        id: "native-prefix-cpu".into(),
        server_token_commit: ferrum_types::SloLatencyBudgets {
            ttft_ms: NonZeroU64::new(30_000).unwrap(),
            tpot_ms: NonZeroU64::new(30_000).unwrap(),
            itl_ms: NonZeroU64::new(30_000).unwrap(),
        },
        client_visible: None,
        attainment: Default::default(),
    });
    // Portable ownership/function gate, not a 2 ms performance measurement.
    slo.planner.max_planning_us = NonZeroU64::new(1_000_000).unwrap();
    config.scheduler.max_running_requests = 2;
    config.scheduler.prefill_step_chunk = Some(2);
    config.scheduler.prefix_rendezvous_max_wait_ms = NonZeroU64::new(30_000);
    config.batching.max_batch_size = 2;
    config.batching.max_num_batched_tokens = 4;
    config.runtime.max_model_len = Some(8);
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let engine = ContinuousBatchEngine::new_plan_runtime(
        config,
        scheduler,
        tokenizer,
        Arc::new(crate::registry::GreedySampler),
        executor.clone(),
        Arc::new(MockTensorFactory),
    )
    .unwrap();
    // Drive the real iteration explicitly; its actual learner stays threaded.
    engine.inner.bg_loop_spawned.store(true, Ordering::Release);
    assert!(engine.inner.config.scheduler.slo.cost_profile.is_none());
    assert!(engine.inner.prefill_reference_runtime.is_none());
    let runtime = engine.inner.cost_runtime.as_ref().unwrap();
    assert!(runtime.snapshot().is_none());
    assert!(runtime.prefix_cost_snapshot().is_none());
    assert!(runtime.prefix_cost_sink().is_some());
    (engine, executor)
}

async fn submit(engine: &ContinuousBatchEngine) -> (RequestId, CreditedOutputSession) {
    let mut request = ferrum_types::InferenceRequest::new(
        "test test test test",
        engine.inner.config.model.model_id.clone(),
    );
    request.stream = true;
    request.sampling_params.max_tokens = 2;
    request.sampling_params.temperature = 0.0;
    request.sampling_params.repetition_penalty = 1.0;
    request
        .metadata
        .insert("ferrum_ignore_eos".into(), true.into());
    let id = request.id.clone();
    let session = engine
        .infer_credited_stream(
            request,
            InferenceRequestContext::from_ingress(slo_clock_now()),
            Arc::new(OutputProjectionContract::cli_text()),
        )
        .await
        .unwrap();
    (id, session)
}

#[derive(Default)]
struct OutputAudit {
    tokens: usize,
    bytes: usize,
    terminal: bool,
}
impl OutputAudit {
    fn drain(&mut self, session: &mut CreditedOutputSession) {
        while let Some(Some(frame)) = session.frames.next().now_or_never() {
            self.tokens += usize::from(frame.metadata().token.is_some());
            self.bytes += frame.wire().payload().len();
            self.terminal |= frame.metadata().terminal;
            drop(frame);
        }
    }
    async fn finish(mut self, mut session: CreditedOutputSession) {
        while let Some(frame) = bounded(session.frames.next()).await {
            self.tokens += usize::from(frame.metadata().token.is_some());
            self.bytes += frame.wire().payload().len();
            self.terminal |= frame.metadata().terminal;
            drop(frame);
        }
        let completion = bounded(session.completion).await.unwrap();
        match completion.payload() {
            OutputCompletion::Succeeded { usage, reason, .. } => {
                assert_eq!(usage.prompt_tokens, 4);
                assert_eq!(usage.completion_tokens, 2);
                assert_ne!(*reason, ferrum_types::FinishReason::Error);
            }
            OutputCompletion::Failed(error) => panic!("ordinary native output failed: {error:?}"),
        }
        assert!(self.terminal && self.bytes > 0);
        assert_eq!(self.tokens, 2);
    }
}

async fn tick(engine: &ContinuousBatchEngine) {
    bounded(engine.inner.run_iteration()).await.unwrap();
    // The real output actor and CPU learner own their respective queues.
    tokio::time::sleep(Duration::from_millis(1)).await;
}

async fn progress<T>(
    engine: &ContinuousBatchEngine,
    executor: &ControlledExecutor,
    phase: &str,
    future: impl std::future::Future<Output = T>,
) -> T {
    tokio::time::timeout(Duration::from_secs(10), future)
        .await
        .unwrap_or_else(|_| {
            let owners = engine.inner.sequences.read().iter().map(|(id, s)| {
                (id.clone(), s.prefill_tokens_processed, s.prefill_complete,
                    s.generated_tokens.len(), s.cost_frontier)
            }).collect::<Vec<_>>();
            panic!("ordinary prefix stalled at {phase}: owners={owners:?}, native={:?}, copies={}, maintenance_snapshot={}, queue={:?}, audit={:?}",
                executor.native_structured_counts(), copies(executor).len(),
                engine.inner.cost_runtime.as_ref().unwrap().prefix_cost_snapshot().is_some(),
                engine.inner.cost_runtime.as_ref().unwrap().sink.stats(),
                engine.inner.slo_controller.lock().last_audit);
        })
}

fn maintenance_known(
    runtime: &EngineCostRuntime,
    executor: &ControlledExecutor,
    domains: &[(Domain, Option<vnext::NativeCheckpointTransferHostWork>); 2],
) -> bool {
    let Some(snapshot) = runtime.prefix_cost_snapshot() else {
        return false;
    };
    let Some(now) = runtime.clock.now_ns() else {
        return false;
    };
    let mut offer = offer(&RequestId::new(), &RequestId::new());
    offer.expires_at_ns = 1_000_000_000;
    let anchored = snapshot.anchored(PlanningCostClockAnchor::exact(0, now), &offer);
    domains.iter().all(|(domain, host_work)| {
        anchored
            .predict(
                &fingerprint(executor),
                &offer,
                match domain.kind() {
                    Kind::Capture => PrefixMaintenanceStage::Capture,
                    Kind::Restore => PrefixMaintenanceStage::Restore,
                },
                &prefix_cost_shape(domain, host_work.as_ref()).unwrap(),
                0,
            )
            .is_some()
    })
}

#[tokio::test]
async fn native_prefix_cpu_ordinary_loop_cold_receipts_learn_and_complete_output() {
    let (engine, executor) = engine().await;
    let runtime = engine.inner.cost_runtime.as_ref().unwrap().clone();
    let minimum = engine
        .inner
        .config
        .scheduler
        .slo
        .cost_observation
        .model
        .min_samples
        .get();
    let mut domains = None;
    let mut restored_pairs = 0;
    // A real cold growth attempt is not a sample. Stop on actual default-minimum
    // qualification under one elapsed deadline, never an arbitrary owner count.
    let mut pairs = 0usize;
    let learned = tokio::time::timeout(Duration::from_secs(10), async {
    while restored_pairs < minimum {
        pairs += 1;
        let (source, mut source_output) = submit(&engine).await;
        progress(&engine, &executor, "source partial prefill", async {
            loop {
                if engine
                    .inner
                    .sequences
                    .read()
                    .get(&source)
                    .is_some_and(|s| s.prefill_tokens_processed == 2)
                {
                    break;
                }
                tick(&engine).await;
            }
        })
        .await;
        let (target, mut target_output) = submit(&engine).await;
        let before_copies = copies(&executor).len();
        let mut source_audit = OutputAudit::default();
        let mut target_audit = OutputAudit::default();
        let mut controller_decisions = Vec::new();
        progress(&engine, &executor, "paired requests and output", async {
            loop {
                if !engine.inner.sequences.read().contains_key(&source)
                    && !engine.inner.sequences.read().contains_key(&target)
                {
                    break;
                }
                tick(&engine).await;
                let controller = engine.inner.slo_controller.lock();
                controller_decisions.push((
                    controller.last_observation,
                    controller.last_resource_unavailable,
                ));
                drop(controller);
                source_audit.drain(&mut source_output);
                target_audit.drain(&mut target_output);
            }
        })
        .await;
        source_audit.finish(source_output).await;
        target_audit.finish(target_output).await;
        let actual = copies(&executor);
        let appended = &actual[before_copies..];
        assert!(appended.iter().all(|bytes| bytes == &2u32.to_le_bytes()));
        let attempts = executor
            .evidence
            .prefix
            .as_ref()
            .unwrap()
            .attempts
            .lock()
            .drain(..)
            .collect::<Vec<_>>();
        eprintln!("ordinary cold pair={pairs} source={source} target={target} copies={} attempts={attempts:?} controller={controller_decisions:?}", appended.len());
        let published = executor.evidence.prefix.as_ref().unwrap().publications.lock().drain(..).collect::<Vec<_>>();
        match appended.len() {
            0 => { assert!(published.is_empty()); }
            2 => {
                assert_eq!(published.len(), 2, "both copies need native full-publication ack");
                assert_eq!(published[0].domain.kind(), Kind::Capture);
                assert_eq!(published[1].domain.kind(), Kind::Restore);
                assert!(published[1].source.as_ref().is_some_and(|source| source.same_transfer(&published[0].identity)));
                let observed = [
                    (published[0].domain.clone(), published[0].host_work),
                    (published[1].domain.clone(), published[1].host_work),
                ];
                if let Some(old) = &domains { assert_eq!(old, &observed); }
                domains = Some(observed);
                restored_pairs += 1;
            },
            other => panic!("unpaired native checkpoint copies: {other}"),
        }
        assert!(
            engine.inner.slo_controller.lock().prefix.is_none(),
            "cold sampling is not a learned hold"
        );
        if restored_pairs >= minimum {
            let domains = domains
                .as_ref()
                .expect("actual full-publication checkpoint domains");
            progress(
                &engine,
                &executor,
                "maintenance learner publication",
                async {
                    while !maintenance_known(&runtime, &executor, domains) {
                        tokio::time::sleep(Duration::from_millis(1)).await;
                    }
                },
            )
            .await;
            break;
        }
    }
    }).await;
    assert!(learned.is_ok(),
        "cold learning deadline: pairs={pairs}, full_ack_pairs={restored_pairs}/{minimum}, native={:?}, queue={:?}, audit={:?}",
        executor.evidence.prefix.as_ref().unwrap().attempts.lock(), runtime.sink.stats(), engine.inner.slo_controller.lock().last_audit);
    assert!(
        restored_pairs >= minimum,
        "ordinary loop never supplied enough real full-ack transfers: {restored_pairs}/{minimum}"
    );
    assert!(maintenance_known(
        &runtime,
        &executor,
        domains.as_ref().unwrap()
    ));
    // The shared constructor and original asynchronous worker did all wiring.
    // No inference model/qualified hold claim is inferred from this milestone.
    engine.shutdown().await.unwrap();
}
