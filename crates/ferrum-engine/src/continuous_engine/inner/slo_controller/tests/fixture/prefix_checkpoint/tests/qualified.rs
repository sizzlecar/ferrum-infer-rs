//! Full product startup followed by real user-traffic prefix decisions.
//! No runtime/model replacement, manual transfer or sample/receipt injection.
use super::*;
use crate::{AutomaticCostProbeOutput, AutomaticCostProbeTemplate};
use ferrum_interfaces::engine::InferenceEngine;
use ferrum_interfaces::output_flow::{CreditedOutputSession, OutputCompletion};
use futures::StreamExt;
mod diagnostics;
mod maintenance;
mod ready;

const PROMPT: usize = 24;
const BOUNDARY: usize = PROMPT - 1;

fn request(engine: &ContinuousBatchEngine, follower: bool) -> ferrum_types::InferenceRequest {
    let mut tokens = vec!["ok"; PROMPT];
    if follower {
        tokens[BOUNDARY..].fill("test");
    }
    let mut request = ferrum_types::InferenceRequest::new(
        tokens.join(" "),
        engine.inner.config.model.model_id.clone(),
    );
    request.stream = true;
    request.sampling_params.max_tokens = 3;
    request.sampling_params.temperature = 0.0;
    request.sampling_params.repetition_penalty = 1.0;
    request.sampling_params.top_k = Some(64);
    request
        .metadata
        .insert("ferrum_ignore_eos".into(), true.into());
    request
}

async fn startup() -> (ContinuousBatchEngine, Arc<ControlledExecutor>) {
    startup_with_wait(NonZeroU64::new(30_000)).await
}

fn prefix_journal_tail(engine: &ContinuousBatchEngine) -> String {
    use std::io::BufRead;
    let Some(path) = engine.inner.config.runtime.profile_jsonl.as_ref() else {
        return "prefix journal not configured".into();
    };
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(error) => return format!("prefix journal {path:?}: {error}"),
    };
    let mut recent = std::collections::VecDeque::with_capacity(64);
    let mut malformed = 0_usize;
    for line in std::io::BufReader::new(file).lines() {
        let value = match line
            .ok()
            .and_then(|line| serde_json::from_str::<serde_json::Value>(&line).ok())
        {
            Some(value) => value,
            None => {
                malformed += 1;
                continue;
            }
        };
        if value.get("phase").and_then(serde_json::Value::as_str) != Some("slo.prefix_resource") {
            continue;
        }
        if let Some(prefix) = value.get("attributes").and_then(|a| a.get("prefix")) {
            if recent.len() == 64 {
                recent.pop_front();
            }
            recent.push_back(prefix.clone());
        }
    }
    // An open writer can still have outstanding records. Reading this file
    // neither drains samples nor queries/replays any planning state.
    format!("prefix_journal={path:?}, malformed_or_partial_lines={malformed}, last64={recent:?}")
}

// Only evaluated by failed assertions. Keep publication/currentness and the
// original route failure alongside the controller's summarized decision.
fn failure_state(
    engine: &ContinuousBatchEngine,
    executor: &ControlledExecutor,
    request: &RequestId,
) -> String {
    let runtime = engine.inner.cost_runtime.as_ref().unwrap();
    let inference = runtime
        .try_snapshot()
        // The runtime already filters snapshots through its private current
        // publication gate. Preserve outer contention / inner absence here.
        .map(|model| model.map(|model| model.model_version()));
    let maintenance = runtime
        .prefix_cost_snapshot()
        .map(|model| (model.model_version(), model.current()));
    let sequence = engine.inner.sequences.read().get(request).map(|sequence| {
        (
            sequence.cost_frontier,
            sequence.prefill_tokens_processed,
            sequence.generated_tokens.len(),
            sequence
                .time_admission
                .as_ref()
                .is_some_and(|state| state.has_current_time_witness(slo_clock_now())),
        )
    });
    let controller = {
        let state = engine.inner.slo_controller.lock();
        (state.last_observation, state.last_audit)
    };
    let route_unknown = *executor.cost_route_unknown.lock();
    format!(
        "inference={inference:?}, maintenance={maintenance:?}, sequence=(cost_frontier,prefill_processed,generated,current_time_witness)={sequence:?}, route_unknown={route_unknown:?}, controller={controller:?}, inference_funnel={:?}; {}",
        runtime.audit_snapshot(),
        prefix_journal_tail(engine)
    )
}

async fn startup_with_wait(
    wait: Option<NonZeroU64>,
) -> (ContinuousBatchEngine, Arc<ControlledExecutor>) {
    startup_with_wait_and_natural_eos(wait, false, true).await
}

async fn startup_with_wait_and_natural_eos(
    wait: Option<NonZeroU64>,
    natural_eos: bool,
    prefix_enabled: bool,
) -> (ContinuousBatchEngine, Arc<ControlledExecutor>) {
    startup_with_probes(
        wait,
        natural_eos,
        prefix_enabled,
        &[PROMPT],
        ferrum_types::SloAutomaticCalibrationDiagnosticsV1::MemoryOnly,
        |_, _| {},
    )
    .await
}

async fn startup_with_probes(
    wait: Option<NonZeroU64>,
    natural_eos: bool,
    prefix_enabled: bool,
    prompt_lengths: &[usize],
    diagnostics: ferrum_types::SloAutomaticCalibrationDiagnosticsV1,
    configure: impl FnOnce(&ContinuousBatchEngine, &Arc<ControlledExecutor>),
) -> (ContinuousBatchEngine, Arc<ControlledExecutor>) {
    diagnostics::install();
    // Use the product tokenizer's source-config parser. The original helper
    // has no EOS vocabulary declaration; removing ignore-EOS alone would not
    // exercise an EOS-enabled installed policy.
    let (tokenizer, executor) = if natural_eos {
        let components = startup_checkpoint_components_with_generation_config(
            2,
            Some(br#"{"eos_token_id":63}"#),
        )
        .await;
        assert_eq!(
            components.0.special_tokens().eos_token,
            Some(ferrum_types::TokenId::new(63))
        );
        components
    } else {
        startup_checkpoint_components(2).await
    };
    executor
        .declare_checkpoint_cost_domain(NonZeroU32::new(64).unwrap(), NonZeroU64::new(4).unwrap());
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    executor
        .emit_structured_cost_observations
        .store(true, Ordering::Release);
    executor
        .emit_cost_observations
        .store(true, Ordering::Release);
    executor
        .completion_work_known
        .store(true, Ordering::Release);
    executor.prepare_structured_query_resources();
    let mut config = ferrum_types::EngineConfig::default();
    config.scheduler.slo = serde_json::from_value(serde_json::json!({"mode":"enforce"})).unwrap();
    let slo = &mut config.scheduler.slo;
    let ferrum_types::SloLiveStructuredCalibration::AutomaticV1 { settings } =
        &mut slo.cost_observation.live_structured_calibration
    else {
        unreachable!()
    };
    settings.reuse = ferrum_types::SloAutomaticCalibrationReuseV1::Disabled {};
    settings.diagnostics = diagnostics;
    settings.cost_probe.maximum_concurrent_requests = NonZeroUsize::new(2).unwrap();
    // Original sample/rank/population/qualification requirements stay default.
    slo.default_service_class = Some("native-prefix-qualified".into());
    slo.services.push(ferrum_types::ServiceSloConfig {
        id: "native-prefix-qualified".into(),
        server_token_commit: ferrum_types::SloLatencyBudgets {
            ttft_ms: NonZeroU64::new(120_000).unwrap(),
            tpot_ms: NonZeroU64::new(120_000).unwrap(),
            itl_ms: NonZeroU64::new(120_000).unwrap(),
        },
        client_visible: None,
        attainment: Default::default(),
    });
    // Explicit portable protocol workload: both direct and shared trajectories
    // fit this same H. Neither this 1 s allowance nor its latency is GPU evidence.
    slo.planner.max_planning_us = NonZeroU64::new(1_000_000).unwrap();
    slo.planner.lookahead_waves = NonZeroUsize::new(16).unwrap();
    config.scheduler.max_running_requests = 2;
    config.scheduler.prefill_step_chunk = Some(2);
    config.scheduler.prefix_rendezvous_max_wait_ms = wait;
    config.batching.max_batch_size = 2;
    config.batching.max_num_batched_tokens = 4;
    config.runtime.max_model_len = Some(64);
    config.runtime.prefix_state_cache_enabled = prefix_enabled;
    config.runtime.profile_detail = ferrum_types::ObservabilityProfileDetail::Resource;
    config.runtime.profile_jsonl = Some(std::env::temp_dir().join(format!(
        "ferrum-native-prefix-{}-{}-{}.jsonl",
        std::process::id(),
        if wait.is_some() { "wait" } else { "no-wait" },
        RequestId::new()
    )));
    config.scheduler.slo.validate().unwrap();
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
    let runtime = engine.inner.cost_runtime.as_ref().unwrap().clone();
    assert!(runtime.snapshot().is_none());
    assert!(runtime.prefix_cost_snapshot().is_none());
    configure(&engine, &executor);
    let mut probe = request(&engine, false);
    if natural_eos {
        probe.metadata.remove("ferrum_ignore_eos");
        assert!(!probe.metadata.contains_key("ferrum_ignore_eos"));
    }
    // Same declared program/geometry, different contents: users cannot inherit
    // the startup source's checkpoint by matching its cached token prefix.
    let templates = prompt_lengths
        .iter()
        .map(|&length| {
            let mut probe = probe.clone();
            probe.id = RequestId::new();
            probe.prompt = vec!["test"; length].join(" ");
            AutomaticCostProbeTemplate::new(probe, AutomaticCostProbeOutput::CliText).unwrap()
        })
        .collect();
    let engine = tokio::time::timeout(
        Duration::from_secs(180),
        engine.finish_automatic_startup_with_probes(templates),
    )
    .await
    .expect("original automatic startup exceeded its declared probe bounds")
    .unwrap();
    eprintln!("startup native evidence: counts={:?}, copies={}, prefix_epoch={:?}, attempts={:?}, full_publications={:?}",
        executor.native_structured_counts(), copies(&executor).len(),
        runtime.prefix_cost_snapshot().map(|model| (model.model_version(), model.current())),
        executor.evidence.prefix.as_ref().unwrap().attempts.lock(),
        executor.evidence.prefix.as_ref().unwrap().publications.lock());
    assert_eq!(
        engine.inner.config.runtime.prefix_state_cache_enabled,
        prefix_enabled
    );
    if prefix_enabled {
        assert_startup_maintenance_known(
            &runtime,
            &executor,
            engine
                .inner
                .config
                .scheduler
                .slo
                .cost_observation
                .model
                .min_samples
                .get(),
        );
    } else {
        // collect_startup_prefix_cost skips shared maintenance sampling when
        // the cache is off. Private source ACKs still must exist, but cannot
        // establish that every shared maintenance domain has been qualified.
        let (captures, restores) = executor.native_prefix_terminal_totals();
        assert!(
            captures > 0 && restores > 0,
            "automatic private checkpoint work is missing"
        );
        assert_eq!(
            executor.native_prefix_live_lease_counts(),
            (0, 0),
            "completed startup must retire private leases without shared index entries"
        );
    }
    assert!(
        runtime.snapshot().is_some(),
        "actual startup did not qualify inference: {:?}",
        runtime.audit_snapshot()
    );
    assert!(
        engine.inner.prefill_reference_runtime.is_some(),
        "actual reference startup remained unknown"
    );
    let receipt = runtime.profile_receipt().unwrap();
    assert!(receipt.path.is_none());
    assert_eq!(receipt.storage, ferrum_types::SloCostProfileStorage::Memory);
    assert!(receipt.offered_samples > 0);
    assert!(executor.native_structured_counts().1 > 0);
    engine.inner.bg_loop_spawned.store(true, Ordering::Release);
    (engine, executor)
}

fn assert_startup_maintenance_known(
    runtime: &EngineCostRuntime,
    executor: &ControlledExecutor,
    minimum: usize,
) {
    use ferrum_scheduler::implementations::continuous::cost_model::CostPrediction;

    let (published, incomplete) = executor.native_prefix_publication_summary();
    assert!(
        !incomplete,
        "actual publication summary lost a receipt or domain"
    );
    let mut totals = (0usize, 0usize);
    for receipt in &published {
        let total = match receipt.domain.kind() {
            Kind::Capture => &mut totals.0,
            Kind::Restore => &mut totals.1,
        };
        *total = total.checked_add(receipt.published).unwrap();
    }
    assert_eq!(
        totals,
        executor.native_prefix_terminal_totals(),
        "every terminal startup transfer must have its actual publication receipt"
    );
    // The original shared maintenance probe declares this exact input and
    // boundary. Additional private sources may publish other host lengths;
    // their presence cannot substitute for either original shared domain.
    let domains: Vec<_> = published
        .iter()
        .filter(|receipt| {
            receipt.host_work.is_some_and(|work| {
                work.prefix_tokens() == BOUNDARY as u64 && work.full_input_tokens() == PROMPT as u64
            })
        })
        .collect();
    assert_eq!(
        domains.len(),
        2,
        "startup must publish the original Capture and Restore domains: {published:?}"
    );
    for kind in [Kind::Capture, Kind::Restore] {
        assert_eq!(
            domains
                .iter()
                .filter(|receipt| receipt.domain.kind() == kind)
                .count(),
            1
        );
    }
    let snapshot = runtime
        .prefix_cost_snapshot()
        .expect("actual startup maintenance model");
    let now = runtime.clock.now_ns().expect("current cost clock");
    let mut offer = super::offer(&RequestId::new(), &RequestId::new());
    offer.boundary_tokens = NonZeroU32::new(BOUNDARY as u32).unwrap();
    offer.expires_at_ns = 1_000_000_000;
    let anchored = snapshot.anchored(PlanningCostClockAnchor::exact(0, now), &offer);
    for receipt in domains {
        let (domain, host_work, count) = (&receipt.domain, receipt.host_work, receipt.published);
        assert!(
            count >= minimum,
            "actual fullack count {count}/{minimum}: {domain:?}"
        );
        let shape = prefix_cost_shape(domain, host_work.as_ref()).unwrap();
        let actual_prediction = snapshot.diagnostic_prediction(&fingerprint(executor), &shape, now);
        let CostPrediction::Known(known) = &actual_prediction else {
            panic!("original maintenance domain did not qualify: {actual_prediction:?}, domain={domain:?}");
        };
        assert!(known.sample_count >= minimum && known.sample_count <= count,
            "model must retain the original minimum from actual published receipts: retained={} published={count} minimum={minimum}", known.sample_count);
        let prediction = anchored.predict(
            &fingerprint(executor),
            &offer,
            match domain.kind() {
                Kind::Capture => PrefixMaintenanceStage::Capture,
                Kind::Restore => PrefixMaintenanceStage::Restore,
            },
            &shape,
            0,
        );
        assert!(prediction.is_some(), "startup exact domain remains Unknown: count={count}, current={}, epoch={}, prediction={:?}, now={now}, shape={shape:?}, domain={domain:?}",
            snapshot.current(), snapshot.model_version(), snapshot.diagnostic_prediction(&fingerprint(executor), &shape, now));
        eprintln!("startup actual maintenance Known: kind={:?}, fullack_count={count}, epoch={}, shape={shape:?}", domain.kind(), snapshot.model_version());
    }
}

async fn submit(
    engine: &ContinuousBatchEngine,
    follower: bool,
) -> (RequestId, tokio::task::JoinHandle<()>) {
    let request = request(engine, follower);
    submit_request(engine, request, false).await
}

async fn submit_request(
    engine: &ContinuousBatchEngine,
    request: ferrum_types::InferenceRequest,
    natural_eos: bool,
) -> (RequestId, tokio::task::JoinHandle<()>) {
    let id = request.id.clone();
    let session = engine
        .infer_credited_stream(
            request,
            InferenceRequestContext::from_ingress(slo_clock_now()),
            Arc::new(OutputProjectionContract::cli_text()),
        )
        .await
        .unwrap();
    let output = if natural_eos {
        tokio::spawn(consume_with_eos(session, true))
    } else {
        tokio::spawn(consume(session))
    };
    (id, output)
}

async fn consume(session: CreditedOutputSession) {
    consume_with_eos(session, false).await
}

async fn consume_with_eos(mut session: CreditedOutputSession, natural_eos: bool) {
    let mut tokens = 0;
    let mut bytes = 0;
    let mut terminal = false;
    while let Some(frame) = session.frames.next().await {
        tokens += usize::from(frame.metadata().token.is_some());
        bytes += frame.wire().payload().len();
        terminal |= frame.metadata().terminal;
        drop(frame);
    }
    let completion = session.completion.await.unwrap();
    match completion.payload() {
        OutputCompletion::Succeeded { usage, reason, .. } => {
            assert_eq!(usage.prompt_tokens, PROMPT);
            if natural_eos {
                assert!(usage.completion_tokens <= 3);
                assert!(matches!(
                    reason,
                    ferrum_types::FinishReason::EOS | ferrum_types::FinishReason::Length
                ));
                if *reason == ferrum_types::FinishReason::Length {
                    assert_eq!(usage.completion_tokens, 3);
                }
            } else {
                assert_eq!(usage.completion_tokens, 3);
            }
        }
        OutputCompletion::Failed(error) => panic!("ordinary qualified output failed: {error:?}"),
    }
    assert!(terminal);
    if natural_eos {
        // An immediate model EOS may legally expose no text event.
        assert!(tokens <= 3);
    } else {
        assert!(bytes > 0);
        assert_eq!(tokens, 3);
    }
}

async fn tick(engine: &ContinuousBatchEngine) {
    tokio::time::timeout(Duration::from_secs(3), engine.inner.run_iteration())
        .await
        .unwrap()
        .unwrap();
    tokio::task::yield_now().await;
}

fn partial(engine: &ContinuousBatchEngine, source: &RequestId) -> usize {
    engine
        .inner
        .sequences
        .read()
        .get(source)
        .map_or(usize::MAX, |s| s.prefill_tokens_processed)
}

#[tokio::test]
async fn native_prefix_cpu_guarded_recapture_is_new_full_ack() {
    let (_, executor) = startup_checkpoint_components(1).await;
    let tap = Tap::new(None);
    install(&executor, &tap);
    let source = super::input();
    let _output = super::partial(&executor, &source);
    let guard = Guard::default();
    let lease = capture(&executor, &source, &guard);
    let first = tap.records.lock()[0].clone();
    let before = copies(&executor).len();
    guard.reject.store(true, Ordering::Release);
    for _ in 0..4 {
        let old_calls = guard.calls.load(Ordering::Acquire);
        assert!(!executor
            .prefix_capture(super::request(&source), &guard)
            .unwrap());
        if guard.calls.load(Ordering::Acquire) > old_calls {
            break;
        }
    }
    assert_eq!(lease.status(), PrefixCaptureStatus::Ready);
    assert_eq!(copies(&executor).len(), before);
    assert_eq!(tap.records.lock().len(), 1);
    guard.reject.store(false, Ordering::Release);
    let replacement = capture(&executor, &source, &guard);
    assert!(Arc::ptr_eq(&lease, &replacement));
    let observations = tap.records.lock();
    assert_eq!(observations.len(), 2);
    assert!(!first.identity.same_transfer(&observations[1].identity));
    assert_eq!(first.domain, observations[1].domain);
    assert_eq!(copies(&executor).len(), before + 1);
}

#[tokio::test]
async fn native_prefix_cpu_automatic_startup_qualifies_inference() {
    assert_original_policy_startup(false, true).await;
}

#[tokio::test]
async fn native_prefix_cpu_automatic_startup_natural_eos_qualifies_before_user_prefill() {
    assert_original_policy_startup(true, true).await;
}

#[tokio::test]
async fn native_prefix_cpu_automatic_startup_cache_off_releases_private_checkpoints_before_user_prefill(
) {
    assert_original_policy_startup(false, false).await;
}

async fn assert_original_policy_startup(natural_eos: bool, prefix_enabled: bool) {
    let (engine, executor) =
        startup_with_wait_and_natural_eos(NonZeroU64::new(30_000), natural_eos, prefix_enabled)
            .await;
    let startup_native_totals = executor.native_prefix_terminal_totals();
    // The original Configured request policy must be usable after the real
    // collector qualifies startup. A published auxiliary GreedyLength model
    // alone is insufficient: execute ordinary admission and require its real
    // finite time witness before the first completed user prefill.
    let mut user_request = request(&engine, false);
    assert_eq!(user_request.prompt, vec!["ok"; PROMPT].join(" "));
    if natural_eos {
        user_request.metadata.remove("ferrum_ignore_eos");
        assert!(!user_request.metadata.contains_key("ferrum_ignore_eos"));
    }
    let (source, output) = submit_request(&engine, user_request, natural_eos).await;
    let mut configured_witness = false;
    tokio::time::timeout(Duration::from_secs(20), async {
        while engine.inner.sequences.read().contains_key(&source) {
            tick(&engine).await;
            configured_witness |=
                engine
                    .inner
                    .sequences
                    .read()
                    .get(&source)
                    .is_some_and(|sequence| {
                        sequence.prefill_tokens_processed > 0
                            && sequence.generated_tokens.is_empty()
                            && sequence.time_admission.as_ref().is_some_and(|state| {
                                state.has_current_time_witness(slo_clock_now())
                            })
                    });
        }
    })
    .await
    .expect("original Configured request failed to make bounded progress");
    assert!(
        configured_witness,
        "published startup source did not cover the original Configured policy: {}",
        failure_state(&engine, &executor, &source)
    );
    output.await.unwrap();
    if !prefix_enabled {
        assert_eq!(
            executor.native_prefix_terminal_totals(),
            startup_native_totals,
            "ordinary cache-off inference must not capture or restore startup checkpoints"
        );
        assert_eq!(
            executor.native_prefix_live_lease_counts(),
            (0, 0),
            "ordinary cache-off inference must leave private and shared ownership empty"
        );
    }
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn native_prefix_cpu_ordinary_qualified_hold_restore_and_first_commit() {
    let (engine, executor) = startup().await;
    let runtime = engine.inner.cost_runtime.as_ref().unwrap().clone();
    assert!(runtime.prefix_cost_snapshot().is_some(),
        "startup produced no maintenance snapshot; finite user witnesses cannot be cleared to bootstrap it");
    let plan = executor
        .plan_prefix_capture_boundary(PrefixCaptureBoundary {
            processed_tokens: 0,
            source_prompt_tokens: PROMPT,
            common_prefix_tokens: PROMPT - 1,
            follower_prompt_tokens: &[PROMPT],
        })
        .unwrap();
    assert_eq!(plan.boundary, BOUNDARY);
    let (source, source_output) = submit(&engine, false).await;
    tokio::time::timeout(Duration::from_secs(20), async {
        while partial(&engine, &source) < BOUNDARY - 1 {
            tick(&engine).await;
        }
    })
    .await
    .expect("known source did not reach the pre-boundary partial span");
    assert_eq!(partial(&engine, &source), BOUNDARY - 1);
    assert!(
        engine.inner.sequences.read()[&source]
            .time_admission
            .as_ref()
            .is_some_and(|state| state.has_current_time_witness(slo_clock_now())),
        "qualified source must retain its actual finite witness: {}",
        failure_state(&engine, &executor, &source)
    );
    let (target, target_output) = submit(&engine, true).await;
    let before = copies(&executor).len();
    let mut hold_seen = false;
    let mut original_hold_deadline = None;
    let configured_wait = Duration::from_millis(
        engine
            .inner
            .config
            .scheduler
            .prefix_rendezvous_max_wait_ms
            .unwrap()
            .get(),
    );
    let mut restored_before_token = false;
    let mut first_commit_witness = false;
    let mut decisions = Vec::new();
    let completed = tokio::time::timeout(Duration::from_secs(20), async {
        while engine.inner.sequences.read().contains_key(&source)
            || engine.inner.sequences.read().contains_key(&target)
        {
            let before_tokens = engine
                .inner
                .sequences
                .read()
                .get(&target)
                .map(|s| s.generated_tokens.len());
            let before_turn = slo_clock_now();
            tick(&engine).await;
            let after_turn = slo_clock_now();
            let controller = engine.inner.slo_controller.lock();
            let has_hold = controller.prefix.is_some();
            hold_seen |= has_hold;
            let audit = controller.last_audit;
            decisions.push((controller.last_observation, audit));
            drop(controller);
            if has_hold {
                let deadline = engine.inner.slo_prefix_deadline().unwrap();
                if let Some(original) = original_hold_deadline {
                    assert_eq!(deadline, original, "fresh replay must not renew the hold");
                } else {
                    assert!(
                        deadline >= before_turn + configured_wait,
                        "cohort retention was shortened to the comparison's start slack"
                    );
                    assert!(
                        deadline <= after_turn + configured_wait,
                        "cohort retention exceeded its original configured bound"
                    );
                    original_hold_deadline = Some(deadline);
                }
            }
            let sequences = engine.inner.sequences.read();
            if let Some(sequence) = sequences.get(&target) {
                if sequence.prefill_tokens_processed == BOUNDARY
                    && sequence.generated_tokens.is_empty()
                {
                    restored_before_token = true;
                    assert_eq!(copies(&executor).len(), before + 2);
                    assert_eq!(
                        executor.native_structured_history.lock()[&target].len(),
                        BOUNDARY,
                        "native ack must commit the real restored history"
                    );
                }
                if before_tokens == Some(0) && !sequence.generated_tokens.is_empty() {
                    first_commit_witness = audit.is_some_and(|a| {
                        a.witness.is_some() && a.backend_submitted && a.host_reconciled
                    });
                }
            }
        }
    })
    .await;
    assert!(completed.is_ok(), "qualified loop stalled: hold={hold_seen}, restored={restored_before_token}, first_witness={first_commit_witness}, decisions={decisions:?}; {}", prefix_journal_tail(&engine));
    source_output.await.unwrap();
    target_output.await.unwrap();
    let copies = copies(&executor);
    engine.shutdown().await.unwrap();
    eprintln!("qualified actual prefix: hold={hold_seen}, restored={restored_before_token}, first_witness={first_commit_witness}, copied={}, decisions={decisions:?}; {}", copies.len() - before, prefix_journal_tail(&engine));
    assert!(
        hold_seen,
        "real learned comparison never adopted the admitted hold"
    );
    assert!(
        restored_before_token,
        "target never committed the native restore before its first token"
    );
    assert!(
        first_commit_witness,
        "restored target first commit lacked the ordinary fresh trajectory witness"
    );
    assert_eq!(copies.len(), before + 2);
    assert!(copies[before..]
        .iter()
        .all(|bytes| bytes == &(BOUNDARY as u32).to_le_bytes()));
}
