//! A bounded series of independently complete fresh-request sources. The sole
//! worker enforces the original activation deadline and retains installation
//! receipts even when the caller's cancellable collection future is dropped.
use super::super::cohort_driver::PreparedProbeExecutionBudget;
use super::*;
use crate::automatic_cost_probe::AutomaticCostProbeTemplate;
use ferrum_types::SloAutomaticCalibrationSettingsV1;
use std::{future::Future, pin::Pin, time::Duration};

pub(in crate::continuous_engine::inner::calibration) struct PreparedStartupCost {
    inputs: super::plan::PreparedProbeInputs,
    preparation_elapsed: Duration,
    budget: PreparedProbeExecutionBudget,
}

impl CalibrationSession {
    /// Tokenizer and product-template inspection before the independent
    /// reference. Live route inventory follows reference retirement because
    /// on-demand programs may not exist at this static boundary.
    #[inline(never)]
    pub(in crate::continuous_engine::inner::calibration) fn prepare_startup_cost<'a>(
        &'a mut self,
        settings: &'a SloAutomaticCalibrationSettingsV1,
        templates: &'a [AutomaticCostProbeTemplate],
    ) -> Pin<Box<impl Future<Output = Result<PreparedStartupCost>> + Send + 'a>> {
        // Construct and allocate the optional phase here, so its large state
        // never becomes a temporary in the shared startup poll frame.
        Box::pin(self.prepare_startup_cost_inner(settings, templates))
    }

    async fn prepare_startup_cost_inner(
        &mut self,
        settings: &SloAutomaticCalibrationSettingsV1,
        templates: &[AutomaticCostProbeTemplate],
    ) -> Result<PreparedStartupCost> {
        let started = tokio::time::Instant::now();
        let maximum = Duration::from_millis(settings.cost_probe.maximum_duration_ms.get());
        let planned = tokio::time::timeout(
            maximum,
            super::plan::PreparedProbeInputs::new(self, settings, templates),
        )
        .await
        .map_err(|_| FerrumError::resource_exhausted("automatic cost preflight duration expired"))
        .and_then(|plan| plan);
        let inputs = planned?;
        let preflight = super::super::cohort_driver::ProbePreflightCharge::default();
        tracing::info!(
            original_templates = templates.len(),
            planning_elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0,
            "Automatic cost static input preparation before reference"
        );
        // Formatting and writing the cold audit also consume this allowance.
        let preparation_elapsed = started.elapsed();
        let remaining = maximum
            .checked_sub(preparation_elapsed)
            .filter(|n| !n.is_zero())
            .ok_or_else(|| {
                FerrumError::resource_exhausted("automatic cost preflight duration expired")
            })?;
        let budget = PreparedProbeExecutionBudget::new(
            remaining,
            settings.cost_probe.maximum_probe_requests,
            settings.cost_probe.maximum_offered_waves,
            settings.cost_probe.maximum_input_projection_requests,
            preflight,
        )?;
        Ok(PreparedStartupCost {
            inputs,
            preparation_elapsed,
            budget,
        })
    }

    #[inline(never)]
    pub(in crate::continuous_engine::inner::calibration) fn collect_prepared_startup_cost<'a>(
        &'a mut self,
        settings: &'a SloAutomaticCalibrationSettingsV1,
        prepared: PreparedStartupCost,
    ) -> Pin<Box<impl Future<Output = Result<u64>> + Send + 'a>> {
        Box::pin(self.collect_prepared_startup_cost_inner(settings, prepared, &[]))
    }

    #[inline(never)]
    pub(in crate::continuous_engine::inner::calibration) fn collect_prepared_startup_cost_with_prefix<
        'a,
    >(
        &'a mut self,
        settings: &'a SloAutomaticCalibrationSettingsV1,
        prepared: PreparedStartupCost,
        templates: &'a [AutomaticCostProbeTemplate],
    ) -> Pin<Box<impl Future<Output = Result<u64>> + Send + 'a>> {
        Box::pin(self.collect_prepared_startup_cost_inner(settings, prepared, templates))
    }

    async fn collect_prepared_startup_cost_inner(
        &mut self,
        settings: &SloAutomaticCalibrationSettingsV1,
        prepared: PreparedStartupCost,
        templates: &[AutomaticCostProbeTemplate],
    ) -> Result<u64> {
        let PreparedStartupCost {
            inputs,
            preparation_elapsed,
            budget,
        } = prepared;
        let started = tokio::time::Instant::now();
        // Reference collection has its own time allowance. Resume the cost
        // allowance exactly once, preserving every preflight request charge.
        let mut budget = budget.resume()?;
        let deadline = budget.deadline();
        // This unprotected population runs before source8 declaration. It
        // shares the original request/action/deadline ledger with that source;
        // no user request or numerical cohort is silently modified.
        if let Err(error) = self
            .collect_startup_prefix_cost(settings, templates, &mut budget)
            .await
        {
            tracing::warn!(%error, "Automatic checkpoint maintenance costs remain unknown after private probes");
        }
        self.drain_startup_geometry().await?;
        let mut cursor = inputs.into_cursor()?;
        self.begin_startup_owner_series(cursor.maximum_sources(), deadline)?;
        let mut last_epoch = None;
        let mut last_error = None;
        let mut completed_sources = 0usize;
        let mut completed_cohorts = 0usize;
        let mut declared_sources = 0usize;
        let mut planned_cohorts = 0usize;
        let mut source_index = 0usize;
        let mut inventory_elapsed = Duration::ZERO;
        'units: while cursor.pending_input_units() > 0 {
            if tokio::time::Instant::now() >= deadline {
                last_error = Some(FerrumError::resource_exhausted(
                    "automatic cost probe duration budget expired before next input unit",
                ));
                break;
            }
            if let Err(error) = self.begin_startup_inventory() {
                last_error = Some(error);
                break;
            }
            let inventory_started = tokio::time::Instant::now();
            let planned = Box::pin(cursor.next(self, &mut budget)).await;
            // Both ordinary readiness and pure capture use the same isolated
            // owner boundary; retire it before a numerical collector opens.
            let drained = self.drain_startup_geometry().await;
            let closed = self.end_startup_inventory().await;
            inventory_elapsed += inventory_started.elapsed();
            let planned = planned.and_then(|plan| {
                drained?;
                closed?;
                Ok(plan)
            });
            let series = match planned {
                Ok(Some(series)) => series,
                Ok(None) => break,
                Err(error) => {
                    tracing::warn!(%error, preflight = ?budget.preflight_charge(),
                        pending_inventory_templates = cursor.pending_templates(),
                        pending_input_units = cursor.pending_input_units(),
                        actual_requests_remaining = budget.requests_remaining(),
                        actual_waves_remaining = budget.attempts_remaining(),
                        declared_requests_remaining = budget.selection_requests_remaining(),
                        declared_waves_remaining = budget.selection_attempts_remaining(),
                        published_epoch = ?last_epoch,
                        "Automatic input unit unavailable; prior source receipt and pending obligations retained");
                    last_error = Some(error);
                    break;
                }
            };
            declared_sources += series.len();
            planned_cohorts += series.planned_cohorts();
            tracing::info!(
                unit_sources = series.len(), declared_sources, planned_cohorts,
                pending_inventory_templates = cursor.pending_templates(),
                pending_input_units = cursor.pending_input_units(),
                collection_requests_remaining = budget.requests_remaining(),
                actual_wave_attempts_remaining = budget.attempts_remaining(),
                declared_requests_remaining = budget.selection_requests_remaining(),
                declared_waves_remaining = budget.selection_attempts_remaining(),
                preflight = ?budget.preflight_charge(),
                input_preparation_ms = inventory_elapsed.as_secs_f64() * 1_000.0,
                "Frozen automatic input-unit source series"
            );
            for local_source in 0..series.len() {
                if tokio::time::Instant::now() >= deadline {
                    last_error = Some(FerrumError::resource_exhausted(
                        "automatic cost probe duration budget expired before next source",
                    ));
                    break 'units;
                }
                let mut active_cohort = None;
                let collect = async {
                    let source = series.source(local_source)?;
                    self.begin_prepared_owner_source(source.declaration, source.limits)
                        .await?;
                    self.prepared_owner_capture
                        .as_mut()
                        .ok_or_else(|| {
                            FerrumError::internal("automatic cost probe collector absent")
                        })?
                        .retain_execution_plan(source.external_retained_bytes)?;
                    for (local, cohort) in source.cohorts.iter().enumerate() {
                        active_cohort = Some((cohort.pass, cohort.ordinal));
                        self.begin_prepared_owner_cohort(cohort.pass, cohort.ordinal)?;
                        let (requests, options) = series.requests_for(local_source, local)?;
                        self.run_probe_cohort(requests, options, &mut budget)
                            .await?;
                        self.end_prepared_owner_cohort()?;
                        completed_cohorts += 1;
                        active_cohort = None;
                    }
                    // A complete source can legitimately fail numerical qualification.
                    // Keep that distinct from an interrupted/incomplete protocol.
                    let activation = self.activate_prepared_owner_source().await;
                    Ok::<_, FerrumError>(activation)
                };
                // Deadline admission is enforced before each new cohort/action.
                // Do not drop the output consumers of an already started wave:
                // its original host settlement is mandatory retirement, even when
                // it crosses the deadline. Activation retains its own strict gate.
                let collected = Box::pin(collect).await;
                #[cfg(test)]
                if let Some(capture) = self.prepared_owner_capture.as_ref() {
                    eprintln!(
                        "automatic completed-source audit: source={source_index} result={collected:?} population={:?} prepared={:?}",
                        capture.audit(), capture.prepared_audit()
                    );
                }
                match collected {
                    Ok(activation) => {
                        completed_sources += 1;
                        match activation {
                            Ok(epoch) => {
                                last_epoch = Some(epoch);
                                tracing::info!(
                                    source_index,
                                    epoch,
                                    completed_sources,
                                    completed_cohorts,
                                    elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0,
                                    "Automatic cost source activated at original worker barrier"
                                );
                            }
                            Err(error) => {
                                tracing::warn!(source_index, %error, completed_sources,
                                "Complete automatic cost source did not qualify; later original sources remain eligible");
                                last_error = Some(error);
                            }
                        }
                        // Retirement is mandatory ownership cleanup, not new
                        // measured work. Keep an acknowledged installation even
                        // if cleanup crosses the original collection deadline.
                        if let Err(error) = self.retire_startup_owner_source().await {
                            last_error = Some(error);
                            break 'units;
                        }
                    }
                    Err(error) => {
                        let audit = self.prepared_owner_capture.as_ref().map(|c| c.audit());
                        let accepted_fifo =
                            self.prepared_owner_capture.as_ref().map(|c| c.last_fifo());
                        tracing::warn!(
                            source_index, %error, ?active_cohort, ?accepted_fifo,
                            completed_sources, completed_cohorts, planned_cohorts,
                            published_epoch = ?last_epoch,
                            offered = audit.as_ref().map(|a| a.offered),
                            block = audit.as_ref().map(|a| a.block),
                            qualified = audit.as_ref().map(|a| a.owners.iter().filter(|o| o.qualified).count()),
                            failed_owners = audit.as_ref().map(|a| a.owners.iter().filter(|o| o.failure.is_some()).count()),
                            actual_requests_remaining = budget.requests_remaining(),
                            actual_waves_remaining = budget.attempts_remaining(),
                            declared_requests_remaining = budget.selection_requests_remaining(),
                            declared_waves_remaining = budget.selection_attempts_remaining(),
                            elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0,
                            "Automatic cost source interrupted; prior installation receipt retained pending final drain and live-state check"
                        );
                        last_error = Some(error);
                        break 'units;
                    }
                }
                source_index += 1;
            }
        }
        // Cancellation may leave a submitted wave or preparation roots alive.
        // Reconcile them before revoking the private series capability. Neither
        // retirement nor a later failure removes an already installed catalog.
        let drained = self.drain_startup_geometry().await;
        let finished = self.finish_startup_owner_series();
        // A timeout can drop an activation waiter just after the sole worker
        // installed its source. Revoke first, then cross that worker's barrier
        // and read only this series' success receipt, never an arbitrary model.
        let settled = self.freeze_cost_model().await;
        last_epoch = self.startup_owner_series_installed_epoch().or(last_epoch);
        drained?;
        finished?;
        settled?;
        let live_state = self
            .engine
            .inner
            .cost_runtime
            .as_ref()
            .and_then(|runtime| runtime.catalog_live_state());
        tracing::info!(
            declared_sources, completed_sources, completed_cohorts,
            pending_inventory_templates = cursor.pending_templates(),
            pending_input_units = cursor.pending_input_units(),
            actual_requests_remaining = budget.requests_remaining(),
            actual_waves_remaining = budget.attempts_remaining(),
            declared_requests_remaining = budget.selection_requests_remaining(),
            declared_waves_remaining = budget.selection_attempts_remaining(),
            input_preparation_ms = inventory_elapsed.as_secs_f64() * 1_000.0,
            published_epoch = ?last_epoch,
            settled_catalog_epoch = live_state.map(|(epoch, _)| epoch),
            settled_catalog_current = live_state.map(|(_, current)| current),
            preparation_elapsed_ms = preparation_elapsed.as_secs_f64() * 1_000.0,
            elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0,
            "Automatic cost startup series finished"
        );
        // Even an input unit with no affordable numerical source contributed
        // checked algorithms. Move that declaration before online generation 1;
        // never reconstruct it from the smaller installed catalogue.
        if let Some(seed) = cursor.take_algorithm_seed() {
            self.engine
                .inner
                .cost_runtime
                .as_ref()
                .ok_or_else(|| FerrumError::config("cold algorithm seed requires cost runtime"))?
                .install_startup_algorithm_seed(seed)?;
        }
        last_epoch.ok_or_else(|| {
            last_error.unwrap_or_else(|| {
                FerrumError::invalid_request(
                    "automatic cost source series has no qualified publication",
                )
            })
        })
    }
}
