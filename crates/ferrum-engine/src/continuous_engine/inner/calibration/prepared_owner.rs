//! Private source8 driver boundary. A declaration authorizes collection only;
//! original request, execution and worker receipts establish each sample.
use super::structured::capture_error;
use super::*;
use ferrum_scheduler::implementations::continuous::cost_profile::{
    CostProfileLoadLimits, StructuredPreparedOwnerBlockDeclarationV8,
    StructuredPreparedOwnerBlockHeaderV8, StructuredServiceClockV7,
};

mod plan;
mod startup;
#[cfg(test)]
pub(super) use plan::PreparedProbeInputs;
#[cfg(test)]
pub(super) use startup::StartupSourcePreparation;

pub(super) struct StartupOwnerSeries {
    authority: super::super::cost_observation::StartupOwnerSeries,
    maximum_sources: usize,
    started_sources: usize,
    checkpointed: bool,
    finished: bool,
    inventory_open: bool,
}

impl CalibrationSession {
    pub(super) fn startup_owner_series_installed_epoch(&self) -> Option<u64> {
        self.startup_owner_series
            .as_ref()
            .and_then(|series| series.authority.installed_epoch())
    }
    /// Private authorization for a pre-frozen sequence of complete source8
    /// manifests. The caller retains one execution budget for the entire run.
    pub(super) fn begin_startup_owner_series(
        &mut self,
        maximum_sources: usize,
        deadline: tokio::time::Instant,
    ) -> Result<()> {
        self.completed_owner_boundary()?;
        if self.startup_owner_series.is_some()
            || self.prefix_source8
            || self.prefix_source5
            || self.prefix_preparation.is_some()
            || self.prepared_owner_capture.is_some()
            || self.selected_capture_identity.is_some()
            || self.structured_capture.is_some()
            || self.structured_capture_v2.is_some()
            || self.structured_group_v2.is_some()
            || !self.engine.inner.manual_calibration_driver
            || self.engine.inner.bg_loop_spawned.load(Ordering::Acquire)
            || self.engine.inner.is_running.load(Ordering::Acquire)
            || self.engine.inner.shutdown_started.load(Ordering::Acquire)
        {
            return Err(FerrumError::config(
                "startup source series requires unused isolated capture",
            ));
        }
        let authority = self
            .engine
            .inner
            .cost_runtime
            .as_ref()
            .ok_or_else(|| FerrumError::internal("source8 original runtime absent"))?
            .begin_prepared_owner_series(maximum_sources, deadline)?;
        self.startup_owner_series = Some(StartupOwnerSeries {
            authority,
            maximum_sources,
            started_sources: 0,
            checkpointed: false,
            finished: false,
            inventory_open: false,
        });
        Ok(())
    }

    /// Only the drained private startup series may run ordinary readiness
    /// between sources. The permanent source8 isolation flag is never cleared.
    pub(super) fn startup_inventory_active(&self) -> bool {
        self.startup_owner_series.as_ref().is_some_and(|series| {
            series.inventory_open
                && !series.finished
                && !series.checkpointed
                && self.prepared_owner_capture.is_none()
        })
    }

    pub(super) fn prepared_owner_source_active(&self) -> bool {
        self.prefix_source8 && !self.startup_inventory_active()
    }

    pub(super) fn begin_startup_inventory(&mut self) -> Result<()> {
        self.completed_owner_boundary()?;
        let series = self.startup_owner_series.as_mut().ok_or_else(|| {
            FerrumError::invalid_request("startup inventory requires its private series")
        })?;
        if series.finished
            || series.inventory_open
            || series.checkpointed
            || self.prepared_owner_capture.is_some()
            || self.prefix_preparation.is_some()
        {
            return Err(FerrumError::invalid_request(
                "startup inventory requires a retired source",
            ));
        }
        series.authority.ensure_open()?;
        series.inventory_open = true;
        Ok(())
    }

    pub(super) async fn end_startup_inventory(&mut self) -> Result<()> {
        self.completed_owner_boundary()?;
        if !self.startup_inventory_active() {
            return Err(FerrumError::invalid_request(
                "startup inventory is not open",
            ));
        }
        self.freeze_cost_model().await?;
        self.startup_owner_series.as_mut().unwrap().inventory_open = false;
        Ok(())
    }

    pub(super) async fn retire_startup_owner_source(&mut self) -> Result<()> {
        self.completed_owner_boundary()?;
        self.finish_prepared_owner_prefix()?;
        if self
            .startup_owner_series
            .as_ref()
            .is_none_or(|series| series.finished || !series.checkpointed)
            || self.prepared_owner_capture.is_none()
        {
            return Err(FerrumError::invalid_request(
                "retire requires a complete checkpointed startup source",
            ));
        }
        // A dropped activation waiter does not cancel installation. Drain the
        // worker control barrier before releasing this source's driver state.
        self.freeze_cost_model().await?;
        if let Some(source) = &mut self.prepared_owner_capture {
            source.finish_journal();
        }
        self.prepared_owner_capture = None;
        self.startup_owner_series.as_mut().unwrap().checkpointed = false;
        // prefix_source8 remains permanent isolation from all manual protocols.
        Ok(())
    }

    pub(super) fn finish_startup_owner_series(&mut self) -> Result<()> {
        let series = self
            .startup_owner_series
            .as_mut()
            .ok_or_else(|| FerrumError::invalid_request("startup source series absent"))?;
        self.engine
            .inner
            .cost_runtime
            .as_ref()
            .ok_or_else(|| FerrumError::internal("source8 original runtime absent"))?
            .finish_prepared_owner_series(&series.authority)?;
        series.finished = true;
        series.inventory_open = false;
        // Revocation happens even when a caller still needs to drain owners.
        // Temporary evidence is released only after that real safe boundary.
        self.completed_owner_boundary()?;
        self.prefix_preparation = None;
        if let Some(source) = &mut self.prepared_owner_capture {
            source.finish_journal();
        }
        self.prepared_owner_capture = None;
        Ok(())
    }
    pub(super) fn check_prepared_owner_capture(&self) -> Result<()> {
        if !self.prepared_owner_source_active() {
            return Ok(());
        }
        let collector = self.prepared_owner_capture.as_ref().ok_or_else(|| {
            FerrumError::invalid_request("source8 collector closed before probe completion")
        })?;
        if !collector.collecting() {
            return Err(FerrumError::invalid_request(format!(
                "source8 collection failed: {}",
                collector.failure().unwrap_or("source stopped")
            )));
        }
        Ok(())
    }
    pub(super) async fn begin_prepared_owner_source(
        &mut self,
        declaration: StructuredPreparedOwnerBlockDeclarationV8,
        limits: CostProfileLoadLimits,
    ) -> Result<()> {
        self.begin_prepared_owner_source_with_journal(declaration, limits, None)
            .await
    }
    pub(super) async fn begin_prepared_owner_source_with_journal(
        &mut self,
        mut declaration: StructuredPreparedOwnerBlockDeclarationV8,
        limits: CostProfileLoadLimits,
        mut journal: Option<super::super::cost_observation::PreparedSourceJournal>,
    ) -> Result<()> {
        self.completed_owner_boundary()?;
        let in_series = if let Some(series) = &self.startup_owner_series {
            if series.finished
                || series.inventory_open
                || series.started_sources >= series.maximum_sources
                || self.prepared_owner_capture.is_some()
            {
                return Err(FerrumError::invalid_request(
                    "startup source series is exhausted or source not retired",
                ));
            }
            series.authority.ensure_open()?;
            true
        } else {
            false
        };
        if !self.engine.inner.manual_calibration_driver
            || self.engine.inner.bg_loop_spawned.load(Ordering::Acquire)
            || self.engine.inner.is_running.load(Ordering::Acquire)
            || self.engine.inner.shutdown_started.load(Ordering::Acquire)
            || self.prefix_source5
            || (self.prefix_source8 && !in_series)
            || self.prefix_preparation.is_some()
            || self.selected_capture_identity.is_some()
            || self.structured_capture.is_some()
            || self.structured_capture_v2.is_some()
            || self.structured_group_v2.is_some()
            || self
                .configuration()
                .scheduler
                .slo
                .cost_observation
                .structured_capture
                != ferrum_types::SloStructuredCostCapture::HostSettledV1
        {
            return Err(FerrumError::config(
                "source8 requires unused isolated prepared capture",
            ));
        }
        self.validate_declared_prefix_plan(&declaration.cohort_plan, &declaration.prefix_plan)?;
        let cutoff = self.freeze_cost_model().await?.accepted_ordinal();
        let runtime = self
            .engine
            .inner
            .cost_runtime
            .as_ref()
            .ok_or_else(|| FerrumError::internal("source8 original runtime absent"))?;
        let ferrum_interfaces::execution_cost::ExecutorCostIdentityAvailability::Known(identity) =
            &runtime.identity
        else {
            return Err(FerrumError::unsupported(
                "source8 requires actual executor identity",
            ));
        };
        if identity.schema_version
            != ferrum_interfaces::execution_cost::EXECUTOR_COST_IDENTITY_SCHEMA
        {
            return Err(FerrumError::unsupported(
                "source8 executor identity schema differs",
            ));
        }
        let fingerprint =
            ferrum_scheduler::implementations::continuous::cost_model::ExecutionFingerprint {
                model_weights: identity.model_weights,
                numerical_policy: identity.numerical_policy,
                device_runtime: identity.device_runtime,
                execution_config: identity.execution_config,
            };
        // The order matches the original opening clock mapping: any delay
        // makes persisted observations conservatively older.
        let wall_unix_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()
            .and_then(|d| u64::try_from(d.as_nanos()).ok())
            .filter(|v| *v > 0)
            .ok_or_else(|| FerrumError::internal("source8 wall clock unavailable"))?;
        let monotonic_ns = runtime
            .clock
            .now_ns()
            .ok_or_else(|| FerrumError::internal("source8 monotonic clock unavailable"))?;
        use sha2::{Digest, Sha256};
        let mut identity_hash = Sha256::new();
        identity_hash.update(b"ferrum.automatic-prepared-owner.v1\0");
        identity_hash.update(wall_unix_ns.to_le_bytes());
        identity_hash.update(monotonic_ns.to_le_bytes());
        identity_hash.update(declaration.signature().map_err(capture_error)?);
        let executable_path = std::env::current_exe()
            .map_err(|e| FerrumError::internal(format!("source8 executable identity: {e}")))?;
        let maximum_encoded_source_bytes = match &self
            .configuration()
            .scheduler
            .slo
            .cost_observation
            .live_structured_calibration
        {
            ferrum_types::SloLiveStructuredCalibration::AutomaticV1 { settings } => {
                Some(settings.maximum_encoded_source_bytes)
            }
            _ => None,
        };
        let original_numeric = declaration.population.maximum_retained_numeric_bytes;
        let cache_numeric = runtime
            .prepared_reuse_writer_retained_bytes()
            .and_then(|bytes| original_numeric.checked_sub(bytes))
            .filter(|n| *n > 0);
        let cache_reserved = cache_numeric.is_some_and(|numeric| {
            declaration.population.maximum_retained_numeric_bytes = numeric;
            declaration.validate().is_ok()
        });
        if !cache_reserved {
            declaration.population.maximum_retained_numeric_bytes = original_numeric;
        }
        let header = StructuredPreparedOwnerBlockHeaderV8::new(
            identity_hash.finalize().into(), 1, (&fingerprint).into(),
            serde_json::json!({"executable_path": executable_path, "implementation": "ferrum", "source": "automatic-prepared-owner.v1"}),
            StructuredServiceClockV7 { wall_unix_ns, monotonic_ns },
            declaration, maximum_encoded_source_bytes.map_or(limits.max_file_bytes.get() as u64, std::num::NonZeroU64::get),
        ).map_err(capture_error)?;
        let header = match runtime.clock.monotonic_domain() {
            Some(domain) => header
                .with_monotonic_domain(domain.clone())
                .map_err(capture_error)?,
            None => header,
        };
        let reuse_journal = cache_reserved
            .then(|| runtime.open_reuse_prepared_source(header.capture_identity, header.protocol))
            .flatten();
        // Cache admission may fail independently of ordinary calibration. No
        // record has been emitted yet: restore the full allowance and rebuild
        // the original signature without cloning the large cohort inventory.
        let header = if cache_reserved && reuse_journal.is_none() {
            let mut declaration = header.declaration;
            declaration.population.maximum_retained_numeric_bytes = original_numeric;
            let restored = StructuredPreparedOwnerBlockHeaderV8::new(
                header.capture_identity,
                header.generation,
                header.fingerprint,
                header.producer,
                header.opening,
                declaration,
                header.maximum_file_bytes,
            )
            .map_err(capture_error)?;
            match header.monotonic_domain {
                Some(domain) => restored
                    .with_monotonic_domain(domain)
                    .map_err(capture_error)?,
                None => restored,
            }
        } else {
            header
        };
        if journal.is_none() {
            if let ferrum_types::SloLiveStructuredCalibration::AutomaticV1 { settings } = &self
                .configuration()
                .scheduler
                .slo
                .cost_observation
                .live_structured_calibration
            {
                journal = runtime
                    .open_prepared_source_journal(&settings.diagnostics, header.maximum_file_bytes);
            }
        }
        let failed_reuse = reuse_journal.clone();
        self.prepared_owner_capture = Some(
            super::super::cost_observation::PreparedOwnerCalibration::new_with_journals(
                header,
                limits,
                Arc::clone(&runtime.clock),
                cutoff,
                maximum_encoded_source_bytes,
                journal,
                reuse_journal,
            )
            .map_err(|reason| {
                if let Some(journal) = &failed_reuse {
                    journal.abandon();
                }
                capture_error(reason)
            })?,
        );
        self.prefix_source8 = true;
        if let Some(series) = &mut self.startup_owner_series {
            series.started_sources += 1;
            series.checkpointed = false;
        }
        Ok(())
    }

    pub(super) fn begin_prepared_owner_cohort(
        &mut self,
        pass: usize,
        ordinal: usize,
    ) -> Result<()> {
        self.completed_owner_boundary()?;
        self.prepared_owner_capture
            .as_mut()
            .ok_or_else(|| FerrumError::invalid_request("source8 collector absent"))?
            .begin_cohort(pass, ordinal)
            .map_err(capture_error)
    }

    pub(super) fn end_prepared_owner_cohort(&mut self) -> Result<()> {
        self.completed_owner_boundary()?;
        self.finish_prepared_owner_prefix()?;
        self.prepared_owner_capture
            .as_mut()
            .ok_or_else(|| FerrumError::invalid_request("source8 collector absent"))?
            .end_cohort()
            .map_err(capture_error)
    }

    // Called once after a durable original receipt has settled, including a
    // Reap after the caller was cancelled. The source owns offer accounting.
    pub(super) fn record_prepared_owner_settlement(&mut self, report: &CalibrationWaveReport) {
        if !self.prepared_owner_source_active() {
            self.record_prefix_wave(report);
            return;
        }
        let result = self.record_prepared_owner_settlement_inner(report);
        if let Err(error) = result {
            if let Some(c) = &mut self.prepared_owner_capture {
                c.invalidate(error.to_string());
            }
            self.fail_prepared_owner_prefix(&error.to_string());
        }
    }

    fn record_prepared_owner_settlement_inner(
        &mut self,
        report: &CalibrationWaveReport,
    ) -> Result<()> {
        let preparing = self
            .prepared_owner_capture
            .as_ref()
            .ok_or_else(|| FerrumError::invalid_request("source8 collector closed"))?
            .prefix_pending();
        if report.no_submission_proof().is_some() {
            self.record_prepared_owner_no_submission(report)?;
            return self
                .prepared_owner_capture
                .as_mut()
                .unwrap()
                .no_submission(report)
                .map_err(capture_error);
        }
        self.record_prefix_wave(report);
        self.consume_prepared_owner_prefix_wave(preparing)?;
        if !preparing {
            self.prepared_owner_capture
                .as_mut()
                .unwrap()
                .ordinary(report)
                .map_err(capture_error)?;
        }
        Ok(())
    }

    pub(super) async fn activate_prepared_owner_source(&mut self) -> Result<u64> {
        self.completed_owner_boundary()?;
        self.finish_prepared_owner_prefix()?;
        let cutoff = self.freeze_cost_model().await?.accepted_ordinal();
        let checkpoint = self
            .prepared_owner_capture
            .as_mut()
            .ok_or_else(|| FerrumError::invalid_request("source8 collector absent"))?
            .checkpoint()
            .map_err(capture_error)?;
        if checkpoint.accepted_fifo_cutoff() != cutoff {
            return Err(FerrumError::invalid_request(
                "source8 checkpoint lost original FIFO boundary",
            ));
        }
        let runtime = self.engine.inner.cost_runtime.as_ref().unwrap();
        let waiter = match self.startup_owner_series.as_mut() {
            Some(series) => {
                if series.finished || series.checkpointed {
                    return Err(FerrumError::invalid_request(
                        "startup source already checkpointed or series closed",
                    ));
                }
                // This is set only after the original all-cohort checkpoint
                // succeeds. No partial manifest can authorize retirement.
                series.checkpointed = true;
                runtime.activate_prepared_owner_checkpoint_in_series(
                    checkpoint,
                    &mut series.authority,
                )?
            }
            None => runtime.activate_prepared_owner_checkpoint(checkpoint)?,
        };
        let installed = waiter
            .wait()
            .await
            .map_err(|error| FerrumError::internal(format!("source8 activation: {error}")))?;
        match installed.catalog_activation {
            Some(Ok(epoch)) => Ok(epoch),
            Some(Err(error)) => Err(FerrumError::invalid_request(error)),
            None => Err(FerrumError::internal(
                "source8 worker lost activation result",
            )),
        }
    }
}
