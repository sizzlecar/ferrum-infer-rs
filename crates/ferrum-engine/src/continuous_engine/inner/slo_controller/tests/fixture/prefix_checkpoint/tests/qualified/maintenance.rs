//! One real backend ticket during automatic private-prefix preparation. This
//! exercises a reachable origin of the pending state seen by the next loop;
//! it does not assert that hardware's previously pending ticket had this origin.
use super::*;
use crate::continuous_engine::inner::slo_controller::SloIterationPlan;
mod journal;

#[derive(Debug, Clone)]
struct DeferredSeed {
    request: RequestId,
    prior_source: RequestId,
    prior_boundary: usize,
    observed_at_ns: u64,
}

#[tokio::test]
async fn native_prefix_cpu_cache_off_maintenance_preserves_frozen_startup_and_user_witness() {
    let deferred = Arc::new(Mutex::new(None::<DeferredSeed>));
    let captured_seed = Arc::new(Mutex::new(None));
    let observed = Arc::clone(&deferred);
    let captured = Arc::clone(&captured_seed);
    // Original optional source journals enable identity/complete-source replay.
    // Their IO is protocol evidence only, never a performance measurement.
    let directory = journal::Directory::new();
    // A second original template supplies another actual acquisition key.
    // Source grouping and all reservation/F/R/Q decisions remain production.
    let (engine, executor) = startup_with_probes(
        NonZeroU64::new(30_000),
        false,
        false,
        &[PROMPT - 2, PROMPT],
        directory.policy(),
        move |engine, executor| {
            let weak = Arc::downgrade(executor);
            let clock = Arc::clone(&engine.inner.cost_runtime.as_ref().unwrap().clock);
            let boundary = Arc::clone(&observed);
            *executor.deferrals.private_seed_arm.lock() = Some(Box::new(move |inputs| {
                let executor = weak.upgrade().unwrap();
                let prefix = executor.evidence.prefix.as_ref().unwrap();
                if let Some(seed) = boundary.lock().clone() {
                    // A later real target prefill can observe the seed's
                    // captured backing while the original source owns it.
                    let authority = prefix
                        .private_interests
                        .lock()
                        .iter()
                        .filter_map(std::sync::Weak::upgrade)
                        .find_map(|lease| {
                            if lease.source != seed.request
                                || lease.status() != PrefixCaptureStatus::Ready
                            {
                                return None;
                            }
                            let authority = lease.checkpoint.lock().as_ref().map(|v| v.authority());
                            authority
                        });
                    if let Some(authority) = authority {
                        assert_eq!(
                            executor.deferrals.maintenance_calls.load(Ordering::Acquire),
                            1
                        );
                        let mut captured = captured.lock();
                        if captured.is_none() {
                            // Match the live backing to its actual published
                            // transfer, before the bounded diagnostic ring rolls.
                            *captured = Some(
                                prefix
                                    .publications
                                    .lock()
                                    .iter()
                                    .find(|publication| {
                                        publication.identity.kind()
                                            == vnext::NativeCheckpointTransferKind::Capture
                                            && publication.identity.checkpoint_authority()
                                                == authority
                                    })
                                    .expect(
                                        "ready deferred checkpoint lacks its original publication",
                                    )
                                    .identity
                                    .clone(),
                            );
                        }
                    }
                    return false;
                }
                let [input] = inputs else { return false };
                if input.chunk.tokens_processed() != 0 || input.chunk.is_final() {
                    return false;
                }
                // Require a live Ready lease that still owns its actual native
                // checkpoint. A historical capture counter is insufficient.
                let prior = prefix
                    .private_interests
                    .lock()
                    .iter()
                    .filter_map(std::sync::Weak::upgrade)
                    .find_map(|lease| {
                        if lease.source == input.request_id
                            || lease.purpose != PrefixCapturePurpose::PrivateCalibration
                            || lease.status() != PrefixCaptureStatus::Ready
                        {
                            return None;
                        }
                        let checkpoint = lease.checkpoint.lock();
                        let checkpoint = checkpoint.as_ref()?;
                        assert_eq!(checkpoint.completed_tokens(), lease.boundary);
                        Some((lease.source.clone(), lease.boundary))
                    });
                let Some((prior_source, prior_boundary)) = prior else {
                    return false;
                };
                assert!(boundary
                    .lock()
                    .replace(DeferredSeed {
                        request: input.request_id.clone(),
                        prior_source,
                        prior_boundary,
                        observed_at_ns: clock.now_ns().unwrap(),
                    })
                    .is_none());
                true
            }));
        },
    )
    .await;
    let seed = deferred.lock().clone().expect(
        "the actual frozen source must reach a second seed while its first checkpoint lives",
    );
    assert_ne!(seed.request, seed.prior_source);
    assert!(seed.prior_boundary > 0);
    let captured_seed = captured_seed.lock().take().unwrap_or_else(|| {
        panic!("deferred seed was not captured after its original maintenance ticket: {seed:?}")
    });
    assert!(executor.deferrals.private_seed_arm.lock().take().is_some());
    assert_eq!(
        executor.deferrals.maintenance_calls.load(Ordering::Acquire),
        1
    );
    assert_eq!(executor.deferrals.capacity_epoch.load(Ordering::Acquire), 1);
    assert!(engine
        .inner
        .slo_controller
        .lock()
        .pending_maintenance
        .is_none());

    let runtime = engine.inner.cost_runtime.as_ref().unwrap();
    let ferrum_types::SloLiveStructuredCalibration::AutomaticV1 { settings } = &engine
        .inner
        .config
        .scheduler
        .slo
        .cost_observation
        .live_structured_calibration
    else {
        unreachable!()
    };
    let defaults = ferrum_types::SloAutomaticCalibrationSettingsV1::default();
    assert_eq!(
        settings.cost_probe.maximum_probe_requests,
        defaults.cost_probe.maximum_probe_requests
    );
    assert_eq!(
        settings.cost_probe.maximum_offered_waves,
        defaults.cost_probe.maximum_offered_waves
    );
    assert_eq!(
        settings.cost_probe.maximum_input_projection_requests,
        defaults.cost_probe.maximum_input_projection_requests
    );
    assert_eq!(
        settings.cost_probe.maximum_duration_ms,
        defaults.cost_probe.maximum_duration_ms
    );
    let numerical =
        crate::continuous_engine::inner::cost_observation::automatic_numerical_settings(settings);
    let children = runtime.startup_series_children_for_test().unwrap();
    let source = directory.completed_source(&captured_seed);
    let after_maintenance: Vec<_> = children
        .iter()
        .filter(|child| {
            child.provenance().source_sha256 == source.source_sha256
                && child.provenance().capture_identity == source.capture_identity
        })
        .collect();
    assert!(
        !after_maintenance.is_empty(),
        "startup kept only publication from before the deferred source: {seed:?}"
    );
    for child in after_maintenance {
        let phases = &child.provenance().phases;
        assert!(phases[0].frozen_at_ns > seed.observed_at_ns);
        assert!(
            phases[0].members
                >= numerical
                    .min_phase_samples
                    .max(numerical.max_rank + numerical.min_fit_redundancy)
        );
        assert!(phases[1..]
            .iter()
            .all(|phase| phase.members >= numerical.min_phase_samples));
        assert!(phases
            .windows(2)
            .all(|pair| pair[0].accepted_fifo_cutoff < pair[1].accepted_fifo_cutoff));
    }
    assert!(engine.inner.sequences.read().is_empty());
    assert_eq!(engine.inner.scheduler.active_count(), 0);
    assert_eq!(engine.inner.scheduler.waiting_count(), 0);
    assert_eq!(executor.native_prefix_live_lease_counts(), (0, 0));
    let startup_native = executor.native_prefix_terminal_totals();
    let epoch = runtime.snapshot().unwrap().model_version();

    let (request, output) = submit_request(&engine, request(&engine, false), false).await;
    let mut prefill_witness = false;
    let mut decode_witness = false;
    let mut maintained_source_witness = false;
    let mut awaiting_receipt = false;
    tokio::time::timeout(Duration::from_secs(20), async {
        while engine.inner.sequences.read().contains_key(&request) {
            let generated = engine.inner.sequences.read()[&request]
                .generated_tokens
                .len();
            let processed = engine.inner.sequences.read()[&request].prefill_tokens_processed;
            if processed == 0 {
                // Preserve normal admission and the first real prefill turn.
                tick(&engine).await;
            } else {
                // Same product controller entry and declared hint as run_iteration.
                // Retain its existing private receipt to identify the actually
                // adopted source, without replaying or replacing planner input.
                engine.inner.refresh_credited_output_readiness();
                let hint = ferrum_interfaces::BatchHint {
                    max_batch_size: engine.inner.config.batching.max_batch_size,
                    max_tokens: engine.inner.config.batching.max_num_batched_tokens,
                    target_latency_ms: None,
                    available_memory: None,
                    resource_constraints: Default::default(),
                };
                let before = engine.inner.controller_timing_snapshot().unwrap();
                match engine.inner.prepare_slo_controller(&hint).unwrap() {
                    SloIterationPlan::Selected(prepared) => {
                        let capture = engine
                            .inner
                            .slo_controller
                            .lock()
                            .pending_execution
                            .as_ref()
                            .unwrap()
                            .work
                            .prospective_capture
                            .clone();
                        engine
                            .inner
                            .execute_slo_controller_wave(prepared)
                            .await
                            .unwrap();
                        runtime.drain_calibration_fixture();
                        let after = engine.inner.controller_timing_snapshot().unwrap();
                        if after.witnesses.decisions.samples > before.witnesses.decisions.samples {
                            assert_eq!(
                                after.witnesses.decisions.samples,
                                before.witnesses.decisions.samples + 1
                            );
                            assert_eq!(
                                after.witnesses.backend_submitted.samples,
                                before.witnesses.backend_submitted.samples + 1
                            );
                            assert_eq!(
                                after.witnesses.host_reconciled.samples,
                                before.witnesses.host_reconciled.samples + 1
                            );
                            assert_eq!(runtime.snapshot().unwrap().model_version(), epoch);
                            let capture = capture.as_ref().expect(
                                "normal witness lacks its original prospective capture",
                            );
                            // Host ACKs precede asynchronous observation resolution.
                            // Wait for this same original receipt inside the existing
                            // whole-request timeout; never mint a checkpoint or retry
                            // the wave to replace missing evidence.
                            awaiting_receipt = true;
                            while capture.settled_receipt_for_test().is_none() {
                                runtime.drain_calibration_fixture();
                                tokio::task::yield_now().await;
                            }
                            awaiting_receipt = false;
                            maintained_source_witness |= source.matches_adopted(
                                capture,
                                &children,
                                epoch,
                            );
                        }
                    }
                    SloIterationPlan::Idle | SloIterationPlan::Progressed => {
                        tokio::task::yield_now().await;
                    }
                    _ => panic!("cache-off ordinary inference selected non-controller work"),
                }
            }
            prefill_witness |=
                engine
                    .inner
                    .sequences
                    .read()
                    .get(&request)
                    .is_some_and(|sequence| {
                        sequence.prefill_tokens_processed > 0
                            && sequence.generated_tokens.is_empty()
                            && sequence.time_admission.as_ref().is_some_and(|state| {
                                state.has_current_time_witness(slo_clock_now())
                            })
                    });
            decode_witness |= generated > 0
                && engine
                    .inner
                    .slo_controller
                    .lock()
                    .last_audit
                    .is_some_and(|audit| {
                        audit.witness.is_some() && audit.backend_submitted && audit.host_reconciled
                    });
        }
    })
    .await
    .unwrap_or_else(|_| {
        let audit = runtime.audit_snapshot();
        panic!(
            "ordinary request timed out after maintenance: awaiting_receipt={awaiting_receipt}, raw_accepted={}, raw_resolved={}, raw_lost={}, raw_resolution_failed={}",
            audit.sink.raw_accepted,
            audit.sink.raw_resolved,
            audit.sink.raw_lost,
            audit.sink.raw_resolution_failed,
        );
    });
    output.await.unwrap();
    assert!(
        prefill_witness && decode_witness,
        "ordinary execution needs original prefill and submitted/reconciled decode witnesses: {}",
        failure_state(&engine, &executor, &request)
    );
    assert!(
        maintained_source_witness,
        "ordinary adopted no witness from the completed source containing the deferred seed"
    );
    assert_eq!(executor.native_prefix_terminal_totals(), startup_native);
    assert_eq!(executor.native_prefix_live_lease_counts(), (0, 0));
    assert!(engine
        .inner
        .slo_controller
        .lock()
        .pending_maintenance
        .is_none());
    engine.shutdown().await.unwrap();
    assert_eq!(executor.native_prefix_live_lease_counts(), (0, 0));
}
