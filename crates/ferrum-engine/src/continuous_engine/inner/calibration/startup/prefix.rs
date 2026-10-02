//! Private maintenance observations before public admission. Source prefill is
//! executed once; every sample still copies real state into a fresh target and
//! reaches the ordinary model/scheduler acknowledgement boundary.
use super::super::cohort_driver::ProbeExecutionBudget;
use super::*;
use crate::automatic_cost_probe::AutomaticCostProbeTemplate;
use ferrum_interfaces::execution_cost::{GuardedNotSubmittedReason, HostSubmissionRejection};
use ferrum_interfaces::model_executor::{
    PlanRuntimePrefixRestoreInput, PlanRuntimePrefixRestoreOutcome, PrefixCaptureBoundary,
    PrefixCapturePurpose, PrefixCaptureRequest,
};
use ferrum_interfaces::vnext::{
    CheckpointTransferSubmissionGuard, NativeCheckpointTransferIdentity,
    NativeCheckpointTransferKind, PreparedCheckpointTransfer,
};
use ferrum_types::{SloAutomaticCalibrationSettingsV1, SloAutomaticCostProbeSamplingPresetV1};

mod acquisition;
pub(in crate::continuous_engine::inner::calibration) use acquisition::{
    AcquiredProbePrefix, ProbePrefixAcquisition, ProbePrefixAcquisitionPlan, ProbePrefixFallback,
    ProbePrefixRestore,
};

struct StartupOwner {
    id: RequestId,
    identity: Arc<()>,
    frontier: super::super::super::cost_observation::CostFrontier,
    offset: usize,
}

pub(in crate::continuous_engine::inner) use acquisition::AcknowledgedProbePrefixRestore;

struct StartupPrefixGuard {
    engine: std::sync::Weak<EngineInner>,
    owners: Vec<StartupOwner>,
    action_owner: usize,
    deadline: std::time::Instant,
    restore: bool,
    expected_source_capture: Option<NativeCheckpointTransferIdentity>,
    observed_transfer: parking_lot::Mutex<Option<NativeCheckpointTransferIdentity>>,
}

impl CheckpointTransferSubmissionGuard for StartupPrefixGuard {
    fn check(
        &self,
        actual: &PreparedCheckpointTransfer<'_>,
    ) -> std::result::Result<(), GuardedNotSubmittedReason> {
        let result = (|| {
            if (actual.cost_domain().kind() == NativeCheckpointTransferKind::Restore)
                != self.restore
            {
                return Err(GuardedNotSubmittedReason::ActualRouteMismatch);
            }
            if self
                .expected_source_capture
                .as_ref()
                .is_some_and(|expected| {
                    actual
                        .source_capture_identity()
                        .is_none_or(|source| !source.same_transfer(expected))
                })
            {
                return Err(GuardedNotSubmittedReason::ActualRouteMismatch);
            }
            use HostSubmissionRejection::*;
            let check = || {
                let engine = self.engine.upgrade().ok_or(Cancelled)?;
                if !engine.manual_calibration_driver
                    || !engine.automatic_reference_bootstrap
                    || engine.bg_loop_spawned.load(Ordering::Acquire)
                    || engine.is_running.load(Ordering::Acquire)
                    || engine.shutdown_started.load(Ordering::Acquire)
                {
                    return Err(Cancelled);
                }
                if std::time::Instant::now() >= self.deadline {
                    return Err(WitnessExpired);
                }
                let sequences = engine.sequences.try_read().ok_or(Busy)?;
                if sequences.len() != self.owners.len() {
                    return Err(FrontierChanged);
                }
                for owner in &self.owners {
                    let sequence = sequences.get(&owner.id).ok_or(FrontierChanged)?;
                    if !Arc::ptr_eq(&sequence.stream_projection_identity, &owner.identity)
                        || sequence.cost_frontier != Some(owner.frontier)
                        || sequence.prefill_tokens_processed != owner.offset
                        || sequence.prefill_complete
                        || !sequence.generated_tokens.is_empty()
                        || sequence.time_admission.as_ref().is_some_and(|state| {
                            state.has_current_time_witness(std::time::Instant::now())
                        })
                    {
                        return Err(FrontierChanged);
                    }
                    let output = sequence.credited_output.as_ref().ok_or(OutputRevoked)?;
                    if output.failure.is_some()
                        || output.port.consumer_closed()
                        || output.grant.is_some()
                    {
                        return Err(OutputRevoked);
                    }
                }
                if std::time::Instant::now() >= self.deadline {
                    return Err(WitnessExpired);
                }
                Ok(())
            };
            check().map_err(GuardedNotSubmittedReason::HostRejected)
        })();
        if result.is_ok() {
            *self.observed_transfer.lock() = Some(actual.identity().clone());
        }
        if let Some(engine) = self.engine.upgrade() {
            if let Some(recorder) = &engine.prefix_resource_recorder {
                use crate::continuous_engine::profile::prefix::{Event, Owner};
                let owner = &self.owners[self.action_owner];
                recorder.record(Event::NativeGuard {
                    owner: Owner::from_engine(
                        &owner.id,
                        owner.frontier.owner_incarnation.get(),
                        owner.frontier.work_generation.get(),
                    ),
                    identity: actual.identity().into(),
                    source_capture: actual.source_capture_identity().map(Into::into),
                    // This is unprotected sampling, not a predictive witness.
                    inference_epoch: None,
                    maintenance_epoch: None,
                    rejection: result.as_ref().err().copied(),
                });
            }
        }
        result
    }
}

impl CalibrationSession {
    pub(in crate::continuous_engine::inner::calibration) async fn collect_startup_prefix_cost(
        &mut self,
        settings: &SloAutomaticCalibrationSettingsV1,
        templates: &[AutomaticCostProbeTemplate],
        budget: &mut ProbeExecutionBudget,
    ) -> Result<()> {
        let inner = &self.engine.inner;
        if templates.is_empty()
            || !inner.config.runtime.prefix_state_cache_enabled
            || !inner.model_executor.supports_guarded_prefix_maintenance()
            || inner
                .cost_runtime
                .as_ref()
                .and_then(|runtime| runtime.prefix_cost_sink())
                .is_none()
        {
            return Ok(());
        }
        self.completed_owner_boundary()?;
        if !inner.manual_calibration_driver
            || !inner.automatic_reference_bootstrap
            || inner.bg_loop_spawned.load(Ordering::Acquire)
            || inner.is_running.load(Ordering::Acquire)
            || self.prepared_owner_capture.is_some()
            || self.prefix_source5
            || self.prefix_source8
        {
            return Err(invalid(
                "checkpoint bootstrap requires the exclusive unused automatic session",
            ));
        }
        if self.limits.maximum_requests().get() < 2 {
            return Err(invalid(
                "checkpoint bootstrap needs two actual concurrent owners",
            ));
        }
        let repetitions = inner
            .config
            .scheduler
            .slo
            .cost_observation
            .model
            .min_samples
            .get();
        self.startup_checkpoint_sampling = true;
        let result = self
            .collect_startup_prefix_templates(settings, templates, repetitions, budget)
            .await;
        self.startup_checkpoint_sampling = false;
        result
    }

    async fn collect_startup_prefix_templates(
        &mut self,
        settings: &SloAutomaticCalibrationSettingsV1,
        templates: &[AutomaticCostProbeTemplate],
        repetitions: usize,
        budget: &mut ProbeExecutionBudget,
    ) -> Result<()> {
        for (template_index, template) in templates.iter().enumerate() {
            budget.require_time()?;
            let maximum = NonZeroUsize::new(
                template
                    .resolved_request()?
                    .sampling_params
                    .max_tokens
                    .min(settings.cost_probe.maximum_output_tokens.get()),
            )
            .ok_or_else(|| invalid("checkpoint probe maximum output is empty"))?;
            let (request, contract) = template.instantiate(
                maximum,
                0,
                SloAutomaticCostProbeSamplingPresetV1::Configured,
            )?;
            budget.claim_checkpoint_owner()?;
            let source = request.id.clone();
            let source_output = self
                .add_request(request, InferenceRequestContext::capture(), contract)
                .await?;
            let result = self
                .collect_startup_prefix_template(template, maximum, &source, repetitions, budget)
                .await;
            drop(source_output);
            // Dropping a private output consumer uses normal cancellation.
            // Never release live native work just because its deadline passed.
            self.drain_startup_geometry().await?;
            tracing::info!(
                template_index,
                requested_maintenance_samples = repetitions,
                succeeded = result.is_ok(),
                requests_remaining = budget.requests_remaining(),
                actions_remaining = budget.attempts_remaining(),
                "Automatic private checkpoint template retired"
            );
            result?;
        }
        if let Some(runtime) = &self.engine.inner.cost_runtime {
            if let Ok(checkpoint) = runtime.request_checkpoint() {
                let _ = tokio::time::timeout_at(budget.deadline(), checkpoint.wait()).await;
            }
        }
        Ok(())
    }

    async fn collect_startup_prefix_template(
        &mut self,
        template: &AutomaticCostProbeTemplate,
        maximum: NonZeroUsize,
        source: &RequestId,
        repetitions: usize,
        budget: &mut ProbeExecutionBudget,
    ) -> Result<()> {
        self.admit_startup_checkpoint_owners(budget).await?;
        let (tokens, maximum_sequence_tokens) = {
            let sequences = self.engine.inner.sequences.read();
            let sequence = &sequences[source];
            (
                sequence.prefill_context_tokens(),
                sequence.model_maximum_sequence_tokens(),
            )
        };
        let common = tokens
            .len()
            .checked_sub(1)
            .filter(|n| *n > 0)
            .ok_or_else(|| invalid("checkpoint probe needs a nonempty prefix and suffix"))?;
        let plan = self
            .engine
            .inner
            .model_executor
            .plan_prefix_capture_boundary(PrefixCaptureBoundary {
                processed_tokens: 0,
                source_prompt_tokens: tokens.len(),
                common_prefix_tokens: common,
                follower_prompt_tokens: &[tokens.len()],
            })
            .ok_or_else(|| invalid("checkpoint probe has no model-declared partial boundary"))?;
        tracing::info!(
            prompt_tokens = tokens.len(),
            boundary = plan.boundary,
            requested_samples = repetitions,
            "Automatic private checkpoint boundary selected"
        );
        let chunk = self
            .engine
            .inner
            .config
            .scheduler
            .prefill_step_chunk
            .unwrap_or(self.engine.inner.config.batching.max_num_batched_tokens)
            .max(1);
        loop {
            budget.require_time()?;
            let frontier = self
                .frontiers()?
                .into_iter()
                .find(|f| f.request_id() == source)
                .ok_or_else(|| invalid("checkpoint source frontier disappeared"))?;
            let (offset, _) = frontier
                .prefill_progress()
                .ok_or_else(|| invalid("checkpoint source unexpectedly decoded"))?;
            if offset == plan.boundary {
                break;
            }
            let count = plan
                .boundary
                .checked_sub(offset)
                .filter(|n| *n > 0)
                .and_then(|n| u32::try_from(n.min(chunk)).ok())
                .and_then(NonZeroU32::new)
                .ok_or_else(|| invalid("checkpoint boundary is outside source progress"))?;
            tokio::time::timeout_at(budget.deadline(), self.startup_output_ready(source))
                .await
                .map_err(|_| invalid("checkpoint source output readiness expired"))??;
            budget.claim_checkpoint_action()?;
            match self
                .step(CalibrationAction::Wave(vec![frontier.prefill_work(count)?]))
                .await?
            {
                CalibrationTurn::Wave(report) | CalibrationTurn::Reaped(report) => {
                    if let Some(error) = report.error {
                        return Err(error);
                    }
                    if report.submission == CalibrationSubmissionState::InFlightUnknown {
                        return Err(invalid("checkpoint prefill submission is indeterminate"));
                    }
                }
                CalibrationTurn::Blocked(_) => {
                    tokio::task::yield_now().await;
                }
                _ => {
                    return Err(invalid(
                        "checkpoint source prefill returned an unexpected turn",
                    ))
                }
            }
        }
        for sample in 0..repetitions {
            budget.require_time()?;
            let (request, contract) = template.instantiate(
                maximum,
                sample as u64,
                SloAutomaticCostProbeSamplingPresetV1::Configured,
            )?;
            budget.claim_checkpoint_owner()?;
            let target = request.id.clone();
            let output = self
                .add_request(request, InferenceRequestContext::capture(), contract)
                .await?;
            let result = self
                .collect_startup_checkpoint_pair(
                    source,
                    &target,
                    &tokens,
                    maximum_sequence_tokens,
                    plan.boundary,
                    budget,
                )
                .await;
            let CreditedOutputSession { frames, completion } = output;
            drop(frames);
            // Retire only this target. The same actual source stays Open for
            // the next distinct native Capture; no prefill is recomputed.
            while self.engine.inner.sequences.read().contains_key(&target) {
                let _ = self.step(CalibrationAction::Maintenance).await?;
                tokio::task::yield_now().await;
            }
            // The source still owns an output account. Wait for this exact
            // target's actor acknowledgement, then release its projection
            // lease, instead of waiting for the whole pool or retrying admission.
            let settled = completion.await.map_err(|_| {
                invalid("checkpoint target lost its output settlement acknowledgement")
            });
            drop(settled?);
            result?;
        }
        Ok(())
    }

    async fn admit_startup_checkpoint_owners(
        &mut self,
        budget: &ProbeExecutionBudget,
    ) -> Result<()> {
        while self.engine.inner.scheduler.waiting_count() > 0 {
            budget.require_time()?;
            let _ = self.step(CalibrationAction::AdmitOne).await?;
            tokio::task::yield_now().await;
        }
        Ok(())
    }

    async fn collect_startup_checkpoint_pair(
        &mut self,
        source: &RequestId,
        target: &RequestId,
        source_tokens: &[TokenId],
        source_maximum: usize,
        boundary: usize,
        budget: &mut ProbeExecutionBudget,
    ) -> Result<()> {
        self.admit_startup_checkpoint_owners(budget).await?;
        for id in [source, target] {
            tokio::time::timeout_at(budget.deadline(), self.startup_output_ready(id))
                .await
                .map_err(|_| invalid("checkpoint pair output readiness expired"))??;
        }
        let guard = |restore| -> Result<Arc<dyn CheckpointTransferSubmissionGuard>> {
            let sequences = self.engine.inner.sequences.read();
            let owner = |id: &RequestId| -> Result<StartupOwner> {
                let sequence = sequences
                    .get(id)
                    .ok_or_else(|| invalid("checkpoint private owner disappeared"))?;
                Ok(StartupOwner {
                    id: id.clone(),
                    identity: Arc::clone(&sequence.stream_projection_identity),
                    frontier: sequence
                        .cost_frontier
                        .ok_or_else(|| invalid("checkpoint owner frontier missing"))?,
                    offset: sequence.prefill_tokens_processed,
                })
            };
            Ok(Arc::new(StartupPrefixGuard {
                engine: Arc::downgrade(&self.engine.inner),
                owners: vec![owner(source)?, owner(target)?],
                action_owner: usize::from(restore),
                deadline: budget.deadline().into_std(),
                restore,
                expected_source_capture: None,
                observed_transfer: parking_lot::Mutex::new(None),
            }))
        };
        let capture_guard = guard(false)?;
        let restore_guard = guard(true)?;
        budget.claim_checkpoint_action()?;
        if !self
            .engine
            .inner
            .model_executor
            .try_capture_plan_runtime_prefix_guarded(
                PrefixCaptureRequest {
                    purpose: PrefixCapturePurpose::SharedCache,
                    source_request_id: source,
                    source_tokens,
                    maximum_sequence_tokens: source_maximum,
                    boundary,
                    expires_at: budget.deadline().into_std(),
                },
                capture_guard,
            )
            .await?
        {
            return Err(invalid(
                "checkpoint capture did not reach model publication",
            ));
        }
        let (tokens, maximum_sequence_tokens) = {
            let sequences = self.engine.inner.sequences.read();
            let sequence = &sequences[target];
            (
                sequence.prefill_context_tokens(),
                sequence.model_maximum_sequence_tokens(),
            )
        };
        let prepared = self
            .engine
            .inner
            .scheduler
            .prepare_prefix_restore(target, 0, tokens.len())?
            .ok_or_else(|| invalid("checkpoint target restore admission unavailable"))?;
        budget.claim_checkpoint_action()?;
        let restored = self
            .engine
            .inner
            .model_executor
            .try_restore_plan_runtime_prefix_guarded(
                PlanRuntimePrefixRestoreInput {
                    request_id: target,
                    input_tokens: &tokens,
                    maximum_sequence_tokens,
                    checkpoint: None,
                    retry: None,
                },
                restore_guard,
            )
            .await?;
        match restored {
            PlanRuntimePrefixRestoreOutcome::Restored(output)
                if output.restored_tokens() == boundary =>
            {
                self.engine
                    .inner
                    .commit_prefix_restore_output(target, prepared, &tokens, output)?;
            }
            _ => {
                return Err(invalid(
                    "checkpoint restore did not acknowledge the declared boundary",
                ))
            }
        }
        Ok(())
    }
}
