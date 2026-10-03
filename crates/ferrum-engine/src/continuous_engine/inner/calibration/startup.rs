//! Private automatic reference bootstrap at the shared product readiness cut.
//! It owns the unpublished engine and never grants the public manual session
//! an exception to its Observe-only, no-live-collector contract.
use super::*;
use ferrum_types::{SloAutomaticReferenceProbeSettingsV1, SloLiveStructuredCalibration};
use std::{
    future::Future,
    num::{NonZeroU32, NonZeroU64, NonZeroUsize},
    pin::Pin,
    time::Duration,
};

mod budget;
mod plan;
mod prefix;
pub(in crate::continuous_engine::inner) use prefix::AcknowledgedProbePrefixRestore;
pub(in crate::continuous_engine::inner::calibration) use prefix::{
    AcquiredProbePrefix, ProbePrefixAcquisition, ProbePrefixAcquisitionPlan, ProbePrefixFallback,
    ProbePrefixRestore,
};
mod probes;
mod progress;
#[cfg(test)]
mod tests;

fn invalid(message: impl Into<String>) -> FerrumError {
    FerrumError::invalid_request(message)
}

impl ContinuousBatchEngine {
    pub(crate) fn finish_automatic_startup(
        self,
    ) -> Pin<Box<impl Future<Output = Result<Self>> + Send + 'static>> {
        self.finish_automatic_startup_with_probes(Vec::new())
    }

    // Allocate the orchestration state at its producer boundary. Callers,
    // including reference-only startup, retain one pointer rather than a copy
    // of every optional phase's maximum async frame. Keep construction out of
    // callers' poll frames even when the optional phases are never entered.
    #[inline(never)]
    pub(crate) fn finish_automatic_startup_with_probes(
        self,
        probes: Vec<crate::automatic_cost_probe::AutomaticCostProbeTemplate>,
    ) -> Pin<Box<impl Future<Output = Result<Self>> + Send + 'static>> {
        Box::pin(self.finish_automatic_startup_with_probes_inner(probes))
    }

    async fn finish_automatic_startup_with_probes_inner(
        mut self,
        probes: Vec<crate::automatic_cost_probe::AutomaticCostProbeTemplate>,
    ) -> Result<Self> {
        if !crate::continuous_engine::slo_startup::automatic_reference_enabled(&self.inner.config) {
            #[cfg(any(test, feature = "test-support"))]
            if crate::geometry_capture::armed() {
                let _ = self.shutdown().await;
                return Err(invalid(
                    "geometry capture requires original automatic startup",
                ));
            }
            return Ok(self);
        }
        let SloLiveStructuredCalibration::AutomaticV1 { settings } = &self
            .inner
            .config
            .scheduler
            .slo
            .cost_observation
            .live_structured_calibration
        else {
            unreachable!()
        };
        let automatic_settings = settings.clone();
        let settings = automatic_settings.reference_probe.clone();
        let started = std::time::Instant::now();
        // An explicitly loaded reference is already immutable; its loading
        // errors are never rescued by an automatic replacement.
        let reference_loaded = self.inner.prefill_reference_runtime.is_some();
        let reused_cost = self.inner.cost_runtime.as_ref().is_some_and(|runtime| {
            runtime.reused_cost_is_fresh() && runtime.startup_algorithm_seed().is_some()
        });
        #[cfg(any(test, feature = "test-support"))]
        if crate::geometry_capture::armed() {
            if reused_cost
                || probes.is_empty()
                || self.inner.config.scheduler.slo.mode != ferrum_types::SloMode::Enforce
            {
                let _ = self.shutdown().await;
                return Err(invalid(
                    "geometry capture requires fresh Enforce product templates",
                ));
            }
            crate::geometry_capture::startup_inputs(&self.inner.config, &probes);
        }
        if reference_loaded && probes.is_empty() {
            if let Err(error) = self.begin_automatic_collection() {
                let _ = self.shutdown().await;
                return Err(error);
            }
            return Ok(self);
        }
        if let Err(error) = Self::check_startup_session(&mut self) {
            let _ = self.shutdown().await;
            return Err(error);
        }
        let mut session = CalibrationSession::new_driver_session(
            self,
            CalibrationLimits::new(if probes.is_empty() {
                NonZeroUsize::MIN
            } else {
                automatic_settings.cost_probe.maximum_concurrent_requests
            })?,
        );
        // Freeze product templates and tokenizer-derived context variants before
        // reference execution. Live route projection and final input selection
        // follow reference retirement, reusing these same prepared inputs.
        let mut prepared_cost = if probes.is_empty() || reused_cost {
            None
        } else {
            let prepared = session
                .prepare_startup_cost(&automatic_settings, &probes)
                .await;
            if let Err(error) = &prepared {
                tracing::warn!(%error, "Automatic cost input preflight failed before measured probes");
            }
            Some(prepared)
        };
        // The future may be dropped by timeout while a durable wave is still
        // executing. Keep its numeric diagnostic state outside that future.
        let mut progress = progress::StartupProgress::default();
        let reference_started = std::time::Instant::now();
        let measured = if reference_loaded {
            Ok(Ok(None))
        } else {
            tokio::time::timeout(
                Duration::from_millis(settings.maximum_duration_ms.get()),
                session.collect_startup_reference_tracked(&settings, &mut progress),
            )
            .await
            .map(|r| r.map(Some))
        };
        let probe_elapsed_ms = reference_started.elapsed().as_secs_f64() * 1_000.0;
        let reference = match measured {
            Ok(Ok(reference)) => reference,
            Ok(Err(error)) => {
                progress.log(
                    session.engine.inner.controller_timing_snapshot(),
                    "startup_failed",
                );
                tracing::warn!(%error, probe_elapsed_ms, "Automatic reference remains unknown after incomplete probes");
                None
            }
            Err(_) => {
                progress.log(
                    session.engine.inner.controller_timing_snapshot(),
                    "duration_budget_expired",
                );
                tracing::warn!(
                    probe_elapsed_ms,
                    "Automatic reference remains unknown after probe duration budget expired"
                );
                None
            }
        };
        // A cancelled waiter may leave a real execution in flight. Complete
        // its reconciliation and cancelled-owner cleanup before user traffic.
        let drain_started = std::time::Instant::now();
        progress.phase = "drain";
        progress.stage(progress::Stage::Drain);
        if let Err(error) = session.drain_startup_tracked(Some(&mut progress)).await {
            progress.log(
                session.engine.inner.controller_timing_snapshot(),
                "drain_failed",
            );
            tracing::error!(%error, probe_elapsed_ms,
                drain_elapsed_ms = drain_started.elapsed().as_secs_f64() * 1_000.0,
                "Automatic reference could not retire its private probes");
            let _ = session.shutdown().await;
            return Err(error);
        }
        progress.stage(progress::Stage::Drained);
        progress.log(session.engine.inner.controller_timing_snapshot(), "drained");
        let drain_elapsed_ms = drain_started.elapsed().as_secs_f64() * 1_000.0;
        // Reference work is independent of cost reuse. Its elapsed time cannot
        // refresh an old cost sample; recheck before skipping actual probes.
        let reused_cost = reused_cost
            && session
                .engine
                .inner
                .cost_runtime
                .as_ref()
                .is_some_and(|runtime| {
                    runtime.reused_cost_is_fresh() && runtime.startup_algorithm_seed().is_some()
                });
        if prepared_cost.is_none() && !probes.is_empty() && !reused_cost {
            prepared_cost = Some(
                session
                    .prepare_startup_cost(&automatic_settings, &probes)
                    .await,
            );
        }
        let mut cost_epoch = reused_cost
            .then(|| {
                session
                    .engine
                    .inner
                    .cost_runtime
                    .as_ref()
                    .and_then(|runtime| runtime.snapshot())
                    .map(|snapshot| snapshot.model_version())
            })
            .flatten();
        if let Some(prepared) = prepared_cost {
            let collected = match prepared {
                Ok(prepared) => {
                    session
                        .collect_prepared_startup_cost_with_prefix(
                            &automatic_settings,
                            prepared,
                            &probes,
                        )
                        .await
                }
                Err(error) => Err(error),
            };
            #[cfg(any(test, feature = "test-support"))]
            if crate::geometry_capture::armed() {
                // Ordinary collection errors are normally swallowed below.
                // The explicit private stop must instead retire the engine.
                let drained = session.drain_startup_tracked(None).await;
                let shutdown = session.shutdown().await;
                crate::geometry_capture::shutdown_finished(drained.is_ok() && shutdown.is_ok());
                drained?;
                shutdown?;
                return Err(invalid(
                    "test geometry capture finished without exposing an engine",
                ));
            }
            match collected {
                Ok(epoch) => cost_epoch = Some(epoch),
                Err(error) => {
                    // No unknown population is promoted. User requests retain
                    // CompleteRequests behavior and may collect fresh evidence.
                    tracing::warn!(%error, "Automatic cost probes did not publish a qualified catalog");
                    if let Some(runtime) = &session.engine.inner.cost_runtime {
                        runtime.record_prepared_owner_failure(&error.to_string());
                    }
                }
            }
            if let Err(error) = session.drain_startup_tracked(None).await {
                let _ = session.shutdown().await;
                return Err(error);
            }
        } else if reused_cost && !probes.is_empty() {
            // Reusing inference costs does not invent a maintenance model.
            // This is the sole cost-probe budget on the reuse branch.
            let mut budget = cohort_driver::ProbeExecutionBudget::new_with_input_projection_limit(
                tokio::time::Instant::now()
                    + Duration::from_millis(
                        automatic_settings.cost_probe.maximum_duration_ms.get(),
                    ),
                automatic_settings.cost_probe.maximum_probe_requests,
                automatic_settings.cost_probe.maximum_offered_waves,
                automatic_settings
                    .cost_probe
                    .maximum_input_projection_requests,
            );
            if let Err(error) = session
                .collect_startup_prefix_cost(&automatic_settings, &probes, &mut budget)
                .await
            {
                tracing::warn!(%error, "Automatic checkpoint maintenance costs remain unknown after private probes");
            }
            if let Err(error) = session.drain_startup_tracked(None).await {
                let _ = session.shutdown().await;
                return Err(error);
            }
        }
        #[cfg(any(test, feature = "test-support"))]
        if crate::geometry_capture::armed() {
            let drained = session.drain_startup_tracked(None).await;
            let shutdown = session.shutdown().await;
            crate::geometry_capture::shutdown_finished(drained.is_ok() && shutdown.is_ok());
            drained?;
            shutdown?;
            return Err(invalid(
                "geometry capture did not complete the original inventory",
            ));
        }
        let mut engine = session.engine;
        let Some(inner) = Arc::get_mut(&mut engine.inner) else {
            let _ = engine.shutdown().await;
            return Err(invalid(
                "automatic bootstrap lost exclusive engine ownership",
            ));
        };
        if let Some((artifact, source)) = reference {
            tracing::info!(
                domain = ?artifact.loaded.piecewise_domain(),
                source_sha256 = ?artifact.source_sha256,
                reference_sha256 = ?artifact.loaded.identity().artifact_sha256,
                "Installed immutable measured automatic reference"
            );
            inner.prefill_reference_runtime = Some(
                super::super::prefill_reference_runtime::EnginePrefillReferenceRuntime::from_verified(
                    artifact.loaded, source, artifact.bytes.into(),
                ),
            );
        }
        inner.manual_calibration_driver = false;
        inner.automatic_reference_bootstrap = false;
        if let Err(error) = engine.begin_automatic_collection() {
            let _ = engine.shutdown().await;
            return Err(error);
        }
        tracing::info!(
            probe_elapsed_ms,
            drain_elapsed_ms,
            startup_elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0,
            reference_known = engine.inner.prefill_reference_runtime.is_some(),
            startup_cost_epoch = cost_epoch,
            "Automatic reference bootstrap finished and live collection started"
        );
        Ok(engine)
    }

    pub(in crate::continuous_engine::inner::calibration) fn check_startup_session(
        engine: &mut Self,
    ) -> Result<()> {
        let inner = Arc::get_mut(&mut engine.inner)
            .ok_or_else(|| invalid("automatic bootstrap requires an exclusive fresh engine"))?;
        if inner.model_executor.execution_resource_authority()
            != ExecutionResourceAuthority::PlanRuntime
            || !crate::continuous_engine::slo_startup::automatic_reference_enabled(&inner.config)
            || !matches!(inner.model_executor.slo_execution_capability(),
                ferrum_interfaces::model_executor::ExecutorSloCapability::GuardedEagerWaves
                | ferrum_interfaces::model_executor::ExecutorSloCapability::GuardedOnDemandWaves)
            || inner.spec_config.is_some()
            || inner.bg_loop_spawned.load(Ordering::Acquire)
            || inner.is_running.load(Ordering::Acquire)
            || inner.shutdown_started.load(Ordering::Acquire)
            || inner.manual_calibration_driver
            || !inner.sequences.read().is_empty()
            || inner.scheduler.active_count() != 0
            || inner.scheduler.waiting_count() != 0
            || inner.cost_runtime.is_none()
        {
            return Err(invalid("automatic bootstrap requires guarded unused PlanRuntime with original observation runtime"));
        }
        inner.manual_calibration_driver = true;
        inner.automatic_reference_bootstrap = true;
        Ok(())
    }

    fn begin_automatic_collection(&self) -> Result<()> {
        let runtime = self
            .inner
            .cost_runtime
            .as_ref()
            .ok_or_else(|| invalid("automatic calibration observation runtime is absent"))?;
        if let Some((source, reference)) = self
            .inner
            .prefill_reference_runtime
            .as_ref()
            .and_then(|reference| reference.startup_evidence())
        {
            // This optional storage policy uses the same cross-restart quota
            // as live generations. Optional persistence failure is audited
            // separately; the original qualified memory reference remains valid.
            let started = std::time::Instant::now();
            runtime.persist_automatic_reference(source, reference)?;
            tracing::debug!(
                diagnostic_elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0,
                "Automatic reference diagnostic policy completed"
            );
        }
        runtime.begin_automatic_calibration()
    }
}

impl CalibrationSession {
    #[cfg(test)]
    async fn drain_startup(&mut self) -> Result<()> {
        self.drain_startup_tracked(None).await
    }

    pub(in crate::continuous_engine::inner::calibration) async fn drain_startup_geometry(
        &mut self,
    ) -> Result<()> {
        self.drain_startup_tracked(None).await
    }

    async fn drain_startup_tracked(
        &mut self,
        progress: Option<&mut progress::StartupProgress>,
    ) -> Result<()> {
        if self.pending.is_some() {
            let turn = self.step(CalibrationAction::Reap).await?;
            if let (Some(progress), CalibrationTurn::Reaped(report)) = (progress, &turn) {
                progress.end_wave(
                    Some(report),
                    self.engine.inner.controller_timing_snapshot(),
                    true,
                );
            }
        }
        self.engine.inner.drain_slo_execution().await?;
        if self.indeterminate {
            return Err(invalid("automatic probe execution remained indeterminate"));
        }
        let inner = &self.engine.inner;
        let _iteration = inner.iteration_lock.lock().await;
        inner.cancel_abandoned_requests().await?;
        inner.complete_credited_output_failures().await?;
        inner.complete_execution_readiness_failures().await?;
        if !inner.sequences.read().is_empty()
            || inner.scheduler.active_count() != 0
            || inner.scheduler.waiting_count() != 0
        {
            return Err(invalid("automatic reference probes did not fully drain"));
        }
        // Actor cancellation follows native and SequenceState retirement. A
        // vacant scheduler slot does not prove that its output account closed.
        // Do not retain the iteration lock while the original actors settle.
        drop(_iteration);
        self.drain_startup_output_capacity().await
    }

    /// Only the exclusive private driver may require the entire pool to drain.
    /// Its caller has dropped or consumed every output session before this cut.
    async fn drain_startup_output_capacity(&self) -> Result<()> {
        let inner = &self.engine.inner;
        if !inner.manual_calibration_driver
            || inner.bg_loop_spawned.load(Ordering::Acquire)
            || inner.is_running.load(Ordering::Acquire)
        {
            return Err(invalid(
                "output settlement requires the exclusive private driver",
            ));
        }
        let Some(pool) = inner.output_credit_pool.get() else {
            return Ok(());
        };
        let pool = pool
            .as_ref()
            .map_err(|error| FerrumError::config(error.clone()))?;
        // Subscribe before observing the ledger. Account closure can precede
        // final frame/history release; retained_accounts is the real slot gate
        // used by open_request, so open_accounts alone is insufficient.
        let mut wake = pool.subscribe();
        while pool.snapshot().retained_accounts != 0 {
            wake.changed().await.map_err(|error| {
                invalid(format!(
                    "private output settlement notification failed: {error}"
                ))
            })?;
        }
        Ok(())
    }

    #[cfg(test)]
    async fn collect_startup_reference(
        &mut self,
        settings: &SloAutomaticReferenceProbeSettingsV1,
    ) -> Result<(reference::MemoryReferenceArtifact, Arc<[u8]>)> {
        self.collect_startup_reference_tracked(settings, &mut progress::StartupProgress::default())
            .await
    }

    async fn collect_startup_reference_tracked(
        &mut self,
        settings: &SloAutomaticReferenceProbeSettingsV1,
        progress: &mut progress::StartupProgress,
    ) -> Result<(reference::MemoryReferenceArtifact, Arc<[u8]>)> {
        let started = tokio::time::Instant::now();
        let deadline = started
            .checked_add(Duration::from_millis(settings.maximum_duration_ms.get()))
            .ok_or_else(|| invalid("automatic reference deadline overflow"))?;
        progress.phase = "prepare_plan";
        let mut probes = plan::ProbePlan::new(self, settings)?;
        let mut budget = plan::ProbeBudget::new(settings.maximum_probe_requests);
        let mut work = budget::ProbeWorkPlanner::new(settings)?;
        let formal_requests = |prefill_anchors: usize| {
            prefill_anchors
                .checked_add(1)
                .and_then(|n| n.checked_mul(settings.fresh_trials_per_anchor.get()))
                .ok_or_else(|| invalid("automatic reference formal request count overflow"))
        };
        let mut observations = Vec::new();
        let mut curves = Vec::new();
        // Establish the smallest prefill and decode anchors first. A cold
        // retry may use only requests beyond this complete minimum protocol.
        budget.reserve(
            formal_requests(1)?
                .checked_add(2)
                .ok_or_else(|| invalid("automatic reference preparation count overflow"))?,
        )?;
        let prompt = &probes.prompts[0];
        let measured = tokio::time::Instant::now();
        self.startup_probe_tracked(
            prompt,
            &probes.partition,
            probes::Capture::Warmup,
            &mut budget,
            progress,
        )
        .await?;
        let warmup = measured.elapsed();
        let measured = tokio::time::Instant::now();
        let samples = self
            .startup_discovery_tracked(
                prompt,
                &probes.partition,
                probes::Discovery::Prefill,
                &mut budget,
                progress,
            )
            .await?;
        work.record_prefill(prompt.tokens.get(), warmup.max(measured.elapsed()))?;
        append_prefill_discovery(prompt, samples, &mut curves, &mut observations)?;

        budget.reserve(formal_requests(1)?)?;
        let measured = tokio::time::Instant::now();
        self.startup_probe_tracked(
            prompt,
            &probes.partition,
            probes::Capture::Warmup,
            &mut budget,
            progress,
        )
        .await?;
        let warmup = measured.elapsed();
        let measured = tokio::time::Instant::now();
        let decode = self
            .startup_discovery_tracked(
                prompt,
                &probes.partition,
                probes::Discovery::Decode,
                &mut budget,
                progress,
            )
            .await?
            .pop()
            .ok_or_else(|| invalid("automatic decode discovery is empty"))?;
        work.record_decode(warmup.max(measured.elapsed()))?;

        // Admit an ascending prefix before spending any work on its next
        // anchor. The estimate reserves every selected formal repetition;
        // it is not a reference measurement or a statistical latency bound.
        let mut stopped = "candidate_domain_complete";
        for prompt in probes.prompts.iter().skip(1) {
            let formal = formal_requests(curves.len() + 1)?;
            let required = work.next_required(prompt.tokens.get(), 2, None)?;
            if !budget.can_fit(2, formal) {
                stopped = "request_budget";
                break;
            }
            if !work.fits(started.elapsed(), required) {
                stopped = "estimated_work_budget";
                break;
            }
            budget.reserve(
                formal
                    .checked_add(1)
                    .ok_or_else(|| invalid("automatic reference preparation count overflow"))?,
            )?;
            budget.reserve_time(deadline, required);
            let measured = tokio::time::Instant::now();
            self.startup_probe_tracked(
                prompt,
                &probes.partition,
                probes::Capture::Warmup,
                &mut budget,
                progress,
            )
            .await?;
            let warmup = measured.elapsed();
            budget.reserve(formal)?;
            budget.reserve_time(
                deadline,
                work.next_required(prompt.tokens.get(), 1, Some(warmup))?,
            );
            let measured = tokio::time::Instant::now();
            let samples = self
                .startup_discovery_tracked(
                    prompt,
                    &probes.partition,
                    probes::Discovery::Prefill,
                    &mut budget,
                    progress,
                )
                .await?;
            work.record_prefill(prompt.tokens.get(), warmup.max(measured.elapsed()))?;
            append_prefill_discovery(prompt, samples, &mut curves, &mut observations)?;
        }
        let required = work.complete_required()?;
        if !work.fits(started.elapsed(), required) {
            return Err(invalid("automatic reference measured preparation leaves insufficient complete-trial budget"));
        }
        probes.retain_prefix(curves.len())?;
        tracing::info!(
            selected_prompt_anchors = probes.prompts.len(),
            maximum_prompt_tokens = probes.prompts.last().expect("selected prefix").tokens.get(),
            completed_preparation_requests = budget.used(),
            formal_probe_requests = formal_requests(curves.len())?,
            elapsed_ms = started.elapsed().as_millis(),
            estimated_remaining_with_reserve_ms = required.as_millis(),
            work_estimate_margin_percent = settings.work_estimate_margin_percent,
            finalization_reserve_percent = settings.finalization_reserve_percent,
            stopped,
            "automatic reference selected complete protocol before freezing"
        );
        progress.phase = "freeze_protocol";
        let frozen = probes.freeze(self, settings, curves, &observations, &decode)?;
        observations.push(decode);
        let mut collector = self.freeze_reference_plan(frozen, observations).await?;
        // No optional work or retry remains after freezing. Every originally
        // selected formal trial is required under the unchanged hard timeout.
        budget.release_reservations();
        for repetition in 0..settings.fresh_trials_per_anchor.get() {
            for (curve, prompt) in probes.prompts.iter().enumerate() {
                self.startup_probe_tracked(
                    prompt,
                    &probes.partition,
                    probes::Capture::Trial {
                        collector: &mut collector,
                        key: CalibrationReferenceTrial::Prefill { curve, repetition },
                    },
                    &mut budget,
                    progress,
                )
                .await?;
            }
            self.startup_probe_tracked(
                &probes.prompts[0],
                &probes.partition,
                probes::Capture::Trial {
                    collector: &mut collector,
                    key: CalibrationReferenceTrial::Decode { repetition },
                },
                &mut budget,
                progress,
            )
            .await?;
        }
        progress.phase = "capture_source";
        let source = self.capture_reference_source_memory_v1(&collector).await?;
        progress.phase = "finish_artifact";
        let artifact = collector.finish_from_memory_source_v1(&source)?;
        Ok((artifact, source.into_bytes()))
    }
}

fn append_prefill_discovery(
    prompt: &plan::ProbePrompt,
    samples: Vec<CalibrationReferenceDiscoverySample>,
    curves: &mut Vec<CalibrationReferenceCurve>,
    observations: &mut Vec<CalibrationReferenceDiscoverySample>,
) -> Result<()> {
    let first = samples
        .first()
        .ok_or_else(|| invalid("automatic prefill discovery is empty"))?;
    curves.push(CalibrationReferenceCurve {
        total_prompt_tokens: prompt.tokens,
        input_tokens_sha256: first.request_evidence().original_input_tokens_sha256,
        partition: samples.iter().map(|row| row.shape()).collect(),
    });
    observations.extend(samples);
    Ok(())
}
