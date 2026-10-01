//! Engine-facing observation state. Training runs on one separate CPU thread.
use super::checkpoint::{CheckpointRequestError, CostCheckpointWaiter};
use super::profile::EngineCostSnapshot;
use super::{trainer::CostTrainingState, worker::CostTrainingWorker, *};
use ferrum_scheduler::implementations::continuous::cost_profile::ProfileLoadClock;
use ferrum_types::{FerrumError, SloCostObservationConfig};
use std::path::Path;

mod calibration_profile;
pub(in crate::continuous_engine) use calibration_profile::LoadedCalibrationProfile;

pub(in crate::continuous_engine) struct EngineCostRuntime {
    pub clock: Arc<dyn CostObservationClock>,
    pub ids: EngineCostIds,
    pub sink: Arc<BoundedCostSampleSink>,
    pub identity: ExecutorCostIdentityAvailability,
    workload_domain: Option<CostWorkloadDomainV1>,
    pub recorder_limits: CostRecorderLimits,
    pub structured_capture: bool,
    pub(super) prospective_capture: Option<Arc<super::prospective_capture::CaptureAudit>>,
    pub(super) issued_structured_prediction:
        Option<Arc<super::prospective_capture::IssuedPredictionAudit>>,
    // The worker owns another training Arc, but never an EngineCostRuntime.
    // Thus no self-cycle or join-from-worker is possible during Drop.
    pub(super) training: Arc<CostTrainingState>,
    worker: Option<CostTrainingWorker>,
    identity_only: bool,
    pub(super) reuse: Option<Arc<super::automatic_reuse::AutomaticReuse>>,
    reuse_hit: bool,
    #[cfg(test)]
    snapshot_reads: std::sync::atomic::AtomicU64,
}

// Keep identity-only construction free of cache I/O. The deterministic cache
// fixture shares all cache lifecycle code while driving the worker itself.
#[derive(Clone, Copy)]
enum RuntimeDriver {
    Background,
    Manual,
    #[cfg(test)]
    ManualWithReuse,
}
impl RuntimeDriver {
    fn opens_reuse(self) -> bool {
        match self {
            Self::Background => true,
            Self::Manual => false,
            #[cfg(test)]
            Self::ManualWithReuse => true,
        }
    }
    fn spawns_worker(self) -> bool {
        matches!(self, Self::Background)
    }
}

impl EngineCostRuntime {
    /// Install this exact sink on the executor's product checkpoint boundary.
    /// Identity-only experiment runtimes do not collect or publish costs.
    pub(in crate::continuous_engine) fn prefix_cost_sink(
        &self,
    ) -> Option<Arc<dyn ferrum_interfaces::vnext::NativeCheckpointObservationSink>> {
        if self.identity_only
            || !matches!(&self.identity, ExecutorCostIdentityAvailability::Known(value) if value.schema_version == EXECUTOR_COST_IDENTITY_SCHEMA)
        {
            return None;
        }
        Some(self.training.prefix.as_ref()?.sink())
    }

    pub(in crate::continuous_engine) fn prefix_cost_snapshot(
        &self,
    ) -> Option<Arc<PrefixCostSnapshot>> {
        self.try_prefix_cost_snapshot().flatten()
    }

    /// Outer None is contention/unsupported; inner None is absent evidence.
    pub(in crate::continuous_engine) fn try_prefix_cost_snapshot(
        &self,
    ) -> Option<Option<Arc<PrefixCostSnapshot>>> {
        if self.identity_only
            || !matches!(&self.identity, ExecutorCostIdentityAvailability::Known(value) if value.schema_version == EXECUTOR_COST_IDENTITY_SCHEMA)
        {
            return None;
        }
        self.training.prefix.as_ref()?.try_snapshot()
    }
    /// Keep the existing checked incarnation/generation allocator for resource
    /// publication. Pre-cost experiments do not import, capture, train or query
    /// a cost model, and own no background training thread.
    pub fn new_identity_only(
        identity: ExecutorCostIdentityAvailability,
        config: &SloCostObservationConfig,
    ) -> Result<Self, FerrumError> {
        let mut runtime = Self::build_with_profile(
            identity,
            Arc::new(EngineCostClock::default()),
            config,
            false,
            None,
            None,
        )?;
        runtime.identity_only = true;
        Ok(runtime)
    }

    pub fn new(
        identity: ExecutorCostIdentityAvailability,
        config: &SloCostObservationConfig,
        profile_path: Option<&Path>,
    ) -> Result<Self, FerrumError> {
        Self::new_with_domain(identity, config, profile_path, None)
    }

    pub fn new_with_domain(
        identity: ExecutorCostIdentityAvailability,
        config: &SloCostObservationConfig,
        profile_path: Option<&Path>,
        workload_domain: Option<CostWorkloadDomainV1>,
    ) -> Result<Self, FerrumError> {
        let clock: Arc<dyn CostObservationClock> = Arc::new(EngineCostClock::default());
        let load_clock = profile_path
            .map(|_| profile::read_load_clock(clock.as_ref(), &config.profile_import))
            .transpose()?;
        Self::build_with_profile_and_domain(
            identity,
            clock,
            config,
            true,
            profile_path,
            load_clock,
            workload_domain,
        )
    }

    #[cfg(test)]
    pub(in crate::continuous_engine) fn with_clock(
        identity: ExecutorCostIdentityAvailability,
        clock: Arc<dyn CostObservationClock>,
    ) -> Result<Self, FerrumError> {
        Self::build(identity, clock, &SloCostObservationConfig::default(), false)
    }

    #[cfg(test)]
    pub(in crate::continuous_engine) fn build(
        identity: ExecutorCostIdentityAvailability,
        clock: Arc<dyn CostObservationClock>,
        config: &SloCostObservationConfig,
        background: bool,
    ) -> Result<Self, FerrumError> {
        Self::build_with_profile(identity, clock, config, background, None, None)
    }

    #[cfg(test)]
    pub(in crate::continuous_engine) fn build_with_clock_and_domain(
        identity: ExecutorCostIdentityAvailability,
        clock: Arc<dyn CostObservationClock>,
        config: &SloCostObservationConfig,
        background: bool,
        workload_domain: CostWorkloadDomainV1,
    ) -> Result<Self, FerrumError> {
        Self::build_with_profile_and_domain(
            identity,
            clock,
            config,
            background,
            None,
            None,
            Some(workload_domain),
        )
    }

    pub(super) fn build_with_profile(
        identity: ExecutorCostIdentityAvailability,
        clock: Arc<dyn CostObservationClock>,
        config: &SloCostObservationConfig,
        background: bool,
        profile_path: Option<&Path>,
        load_clock: Option<ProfileLoadClock>,
    ) -> Result<Self, FerrumError> {
        Self::build_with_profile_and_domain(
            identity,
            clock,
            config,
            background,
            profile_path,
            load_clock,
            None,
        )
    }

    #[cfg(test)]
    pub(super) fn build_with_manual_worker_and_reuse(
        identity: ExecutorCostIdentityAvailability,
        clock: Arc<dyn CostObservationClock>,
        config: &SloCostObservationConfig,
        workload_domain: CostWorkloadDomainV1,
    ) -> Result<Self, FerrumError> {
        Self::build_with_driver(
            identity,
            clock,
            config,
            RuntimeDriver::ManualWithReuse,
            None,
            None,
            Some(workload_domain),
        )
    }

    pub(super) fn build_with_profile_and_domain(
        identity: ExecutorCostIdentityAvailability,
        clock: Arc<dyn CostObservationClock>,
        config: &SloCostObservationConfig,
        background: bool,
        profile_path: Option<&Path>,
        load_clock: Option<ProfileLoadClock>,
        workload_domain: Option<CostWorkloadDomainV1>,
    ) -> Result<Self, FerrumError> {
        Self::build_with_driver(
            identity,
            clock,
            config,
            if background {
                RuntimeDriver::Background
            } else {
                RuntimeDriver::Manual
            },
            profile_path,
            load_clock,
            workload_domain,
        )
    }

    fn build_with_driver(
        identity: ExecutorCostIdentityAvailability,
        clock: Arc<dyn CostObservationClock>,
        config: &SloCostObservationConfig,
        driver: RuntimeDriver,
        profile_path: Option<&Path>,
        load_clock: Option<ProfileLoadClock>,
        workload_domain: Option<CostWorkloadDomainV1>,
    ) -> Result<Self, FerrumError> {
        if let Some(domain) = &workload_domain {
            if !matches!(&identity, ExecutorCostIdentityAvailability::Known(identity) if domain.matches_execution_identity(identity))
            {
                return Err(FerrumError::config(
                    "workload domain differs from actual executor identity",
                ));
            }
        }
        config.validate().map_err(FerrumError::config)?;
        let recorder_limits = CostRecorderLimits {
            max_waves: config.max_waves_per_call.get(),
            max_rows_per_wave: config.max_rows_per_wave.get(),
            max_retained_rows: config.max_retained_rows_per_call.get(),
        };
        recorder_limits
            .validate()
            .map_err(|error| FerrumError::config(error.to_string()))?;
        let mut seed = profile::load_seed_with_domain(
            &identity,
            config,
            profile_path,
            load_clock,
            workload_domain.as_ref(),
        )?;
        let mut restored_monitor = None;
        let mut restored_startup_seed = None;
        let mut reuse = None;
        let mut reuse_hit = false;
        // Shared by both product entrypoints. Explicit profile imports retain
        // their original strict errors; identity-only experiments do no I/O.
        if driver.opens_reuse()
            && profile_path.is_none()
            && config.profile_export.is_none()
            && config.predictor == ferrum_types::SloCostPredictor::StructuredWholeWaveV2
        {
            if let (
                ferrum_types::SloLiveStructuredCalibration::AutomaticV1 { settings },
                ExecutorCostIdentityAvailability::Known(actual),
                Some(domain),
            ) = (
                &config.live_structured_calibration,
                &identity,
                workload_domain.as_ref(),
            ) {
                match super::automatic_reuse::AutomaticReuse::open(
                    config, settings,
                    ferrum_scheduler::implementations::continuous::cost_model::ExecutionFingerprint {
                        model_weights: actual.model_weights, numerical_policy: actual.numerical_policy,
                        device_runtime: actual.device_runtime, execution_config: actual.execution_config,
                    }, domain.clone(), clock.clone(),
                ) {
                    Ok(opened) => {
                        if let Some(restored) = opened.restored {
                            seed = restored.seed;
                            restored_monitor = Some(restored.monitor);
                            restored_startup_seed = restored.startup_seed;
                            reuse_hit = true;
                        }
                        tracing::info!(cache_hit = reuse_hit, cache_miss = ?opened.miss, "Automatic cost restart cache opened");
                        reuse = Some(opened.coordinator);
                    }
                    Err(reason) => tracing::info!(?reason, "Automatic cost restart cache unavailable; ordinary calibration remains enabled"),
                }
            }
        }
        let export = config
            .profile_export
            .as_ref()
            .map(|options| {
                let fingerprint = seed
                    .trainer
                    .as_ref()
                    .map(|trainer| trainer.fingerprint())
                    .ok_or_else(|| {
                        FerrumError::config(
                            "cost profile export requires a known executor fingerprint",
                        )
                    })?;
                let opening = profile_export::ExportClockReading::opening(clock.as_ref())
                    .map_err(|error| FerrumError::config(error.to_string()))?;
                profile_export::ExportPlan::new(
                    options,
                    fingerprint,
                    &profile::model_settings(&config.model),
                    opening,
                )
                .map_err(|error| FerrumError::config(error.to_string()))
            })
            .transpose()?;
        let live = super::live_calibration::LiveCalibration::open_with_domain(
            &config.live_structured_calibration,
            &identity,
            clock.as_ref(),
            &config.profile_import,
            workload_domain.clone(),
        )?;
        if driver.spawns_worker() {
            if let Some(live) = &live {
                // Runs before any worker/feedback publication is exposed to
                // requests, including imported and restart-reused models that
                // skip bootstrap. Identity-only/manual runtimes retain no I/O.
                live.prepare_producer_identity();
            }
        }
        if let (Some(live), Some(reuse)) = (&live, &reuse) {
            live.install_automatic_reuse(reuse.clone());
            if let Some(seed) = restored_startup_seed {
                // Before worker start and the original source7 generation.
                // Cached coordinates alone grant no numerical qualification.
                match live.install_startup_algorithm_seed(seed.clone()) {
                    Ok(()) => {
                        if let Err(reason) = reuse.note_startup_seed(&seed) {
                            tracing::warn!(
                                ?reason,
                                "Restored algorithm seed cannot be persisted again"
                            );
                        }
                    }
                    Err(reason) => {
                        tracing::warn!(%reason, "Cached algorithm seed rejected; cold inventory remains required")
                    }
                }
            }
        }
        let training = Arc::new(CostTrainingState::new_with_reuse(
            config,
            seed,
            export,
            clock.clone(),
            live,
            match &identity {
                ExecutorCostIdentityAvailability::Known(v) => Some(ferrum_scheduler::implementations::continuous::cost_model::ExecutionFingerprint { model_weights:v.model_weights, numerical_policy:v.numerical_policy, device_runtime:v.device_runtime, execution_config:v.execution_config }),
                _ => None,
            },
            restored_monitor,
            reuse.clone(),
        )?);
        if reuse_hit {
            // The restored view/epoch is active now, but no worker or original
            // enrollment exists yet. Charge every actually held child; expiry
            // removal must go through the normal catalog transaction later.
            let snapshot = training.snapshot().ok_or_else(|| {
                FerrumError::internal("restored cost catalog missing before worker start")
            })?;
            let live = training.live.as_ref().ok_or_else(|| {
                FerrumError::internal("restored automatic catalog has no live coordinator")
            })?;
            live.note_automatic_catalog_origins(&snapshot.retained_catalog_children()?)?;
        }
        if config.predictor == ferrum_types::SloCostPredictor::StructuredWholeWaveV2 {
            if let Some(domain) = &workload_domain {
                training.sink.install_workload_domain(domain.clone())?;
            }
        }
        let worker = if driver.spawns_worker() {
            let state = super::trainer::TrainingWorkerOwner(training.clone());
            Some(
                CostTrainingWorker::spawn_with_progress(
                    move || state.consume_progress(),
                    training
                        .live
                        .as_ref()
                        .map(|_| std::time::Duration::from_millis(100)),
                )
                .map_err(|error| {
                    FerrumError::backend(format!("start cost training worker: {error}"))
                })?,
            )
        } else {
            None
        };
        if let Some(worker) = &worker {
            training.sink.attach_worker(worker.notification_thread())?;
            if let Some(live) = &training.live {
                live.attach_worker(worker.notification_thread());
            }
        }
        Ok(Self {
            clock,
            ids: EngineCostIds::default(),
            sink: training.sink.clone(),
            identity,
            workload_domain,
            recorder_limits,
            structured_capture: !config.structured_capture.is_disabled(),
            issued_structured_prediction: (config.predictor
                == ferrum_types::SloCostPredictor::StructuredWholeWaveV2
                && !config.structured_capture.is_disabled())
            .then(|| Arc::new(super::prospective_capture::IssuedPredictionAudit::default())),
            prospective_capture: (!config.prospective_structured_capture.is_disabled())
                .then(|| Arc::new(super::prospective_capture::CaptureAudit::default())),
            training,
            worker,
            identity_only: false,
            reuse,
            reuse_hit,
            #[cfg(test)]
            snapshot_reads: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// A coalescing signal only: never locks, drains, calibrates or publishes
    /// on the inference thread. The bounded sink drops and counts excess work.
    pub(super) fn reserve_live_ticket(
        &self,
        now: Option<u64>,
    ) -> Option<super::live_calibration::Ticket> {
        self.training.live.as_ref()?.reserve(now)
    }

    pub fn workload_domain(&self) -> Option<&CostWorkloadDomainV1> {
        self.workload_domain.as_ref()
    }
    pub fn reused_cost_is_fresh(&self) -> bool {
        self.reuse_hit
            && self.clock.now_ns().is_some_and(|now| {
                self.snapshot().is_some_and(|snapshot| {
                    snapshot.validate_live_freshness(now).is_ok()
                        && snapshot
                            .live_children(now)
                            .is_ok_and(|children| !children.is_empty())
                })
            })
    }
    pub fn acknowledge_restart_shutdown(&self) {
        if let Some(reuse) = &self.reuse {
            match reuse.acknowledge_clean_shutdown() {
                Ok(()) => tracing::info!(
                    "Automatic cost restart cache committed after clean engine shutdown"
                ),
                Err(reason) => {
                    tracing::info!(?reason, reuse_audit = ?reuse.audit(),
                        "Automatic cost restart cache remains unavailable")
                }
            }
        }
    }
    pub fn prepared_reuse_writer_retained_bytes(&self) -> Option<usize> {
        self.reuse
            .as_ref()
            .map(|_| super::automatic_reuse::OriginalSourceJournal::writer_retained_bytes())
    }
    pub fn open_reuse_prepared_source(
        &self,
        capture: [u8; 32],
        protocol: [u8; 32],
    ) -> Option<super::automatic_reuse::OriginalSourceJournal> {
        self.reuse.as_ref().and_then(|reuse| {
            match reuse.open_source(
                super::automatic_reuse::SourceKind::PreparedOwnersV8,
                capture,
                protocol,
            ) {
                Ok(source) => Some(source),
                Err(reason) => {
                    tracing::info!(?reason, "Automatic source8 restart journal unavailable");
                    None
                }
            }
        })
    }

    pub fn wake_trainer(&self) {
        if let Some(worker) = &self.worker {
            worker.wake();
        }
    }

    /// Called after reference bootstrap has released its driver and drained all
    /// probe execution. Only automatic live capture starts here; the original
    /// observation FIFO and checkpoints remain active throughout bootstrap.
    pub fn begin_automatic_calibration(&self) -> Result<(), FerrumError> {
        if let Some(live) = &self.training.live {
            live.begin_automatic_calibration()?;
            self.wake_trainer();
        }
        Ok(())
    }

    /// Optional cold-start evidence uses the same owned disk quota as live
    /// generations. MemoryOnly keeps its original in-memory evidence chain.
    pub fn persist_automatic_reference(
        &self,
        source: &[u8],
        reference: &[u8],
    ) -> Result<(), FerrumError> {
        if let Some(live) = &self.training.live {
            live.persist_automatic_reference(source, reference)?;
        }
        Ok(())
    }

    /// A precise accepted-sample barrier; it never publishes wall-now, changes
    /// TTL, closes export files, or waits for the trainer on the caller thread.
    pub fn request_checkpoint(&self) -> Result<CostCheckpointWaiter, CheckpointRequestError> {
        self.sink.request_checkpoint()
    }

    pub fn request_profile_cut(
        &self,
        paths: super::profile_export::CostProfileCutPaths,
    ) -> Result<CostCheckpointWaiter, CheckpointRequestError> {
        self.sink.request_checkpoint_with_export(Some(paths))
    }

    pub async fn shutdown(&self) -> Result<(), FerrumError> {
        if self.identity_only {
            return Ok(());
        }
        // The same live barrier is also useful at shutdown. A concurrent
        // checkpoint owns the sole slot; joining the worker still completes it.
        let checkpoint = self.request_checkpoint().ok();
        // Drop may request worker drain too, but only this acknowledged
        // shutdown path may clear the automatic cache's dirty session marker.
        self.training.authorize_restart_shutdown();
        self.training.request_export_finalization();
        if let Some(worker) = &self.worker {
            if !worker.shutdown().await {
                return Err(FerrumError::internal("cost training worker panicked"));
            }
        } else {
            // Only deterministic test runtimes omit the dedicated worker.
            while self.training.consume_batch() {}
        }
        if let Some(checkpoint) = checkpoint {
            let checkpoint = checkpoint
                .wait()
                .await
                .map_err(|error| FerrumError::backend(error.to_string()))?;
            tracing::debug!(accepted_ordinal = checkpoint.accepted_ordinal,
                model_version = checkpoint.snapshot.as_ref().map(|snapshot| snapshot.model_version()),
                training = ?checkpoint.training, export = ?checkpoint.export,
                "cost shutdown checkpoint completed");
        }
        tracing::debug!(
            samples = self.trained_samples(),
            observation_funnel = ?self.audit_snapshot(),
            model_version = self.snapshot().map(|snapshot| snapshot.model_version()),
            "cost training worker shutdown complete"
        );
        self.training.close_structured_epoch();
        self.training.export_result()?;
        Ok(())
    }

    #[cfg(test)]
    pub fn consume_samples(&self) {
        assert!(
            self.worker.is_none(),
            "manual drain requires a deterministic test runtime"
        );
        self.training.consume_batch();
    }

    /// The workerless deterministic harness uses the very same consumer and
    /// resolver. Production never drains observations on an inference task.
    #[cfg(test)]
    pub(in crate::continuous_engine) fn drain_calibration_fixture(&self) {
        if self.worker.is_none() {
            while self.training.consume_batch() {}
        }
    }

    #[cfg(test)]
    pub fn with_training_paused<T>(&self, action: impl FnOnce() -> T) -> T {
        self.training.with_training_paused(action)
    }

    #[cfg(test)]
    pub async fn with_training_paused_async<F: std::future::Future>(&self, action: F) -> F::Output {
        self.training.with_training_paused_async(action).await
    }

    #[cfg(test)]
    pub fn shutdown_started(&self) -> bool {
        self.worker
            .as_ref()
            .is_some_and(CostTrainingWorker::shutdown_started)
    }

    pub fn snapshot(&self) -> Option<Arc<EngineCostSnapshot>> {
        #[cfg(test)]
        self.snapshot_reads
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.training.snapshot()
    }

    /// Cold lifecycle diagnostic after the original worker barrier. The
    /// installation receipt alone does not prove the current feedback gate.
    pub(in crate::continuous_engine::inner) fn catalog_live_state(&self) -> Option<(u64, bool)> {
        self.snapshot()
            .map(|model| (model.model_version(), model.current()))
    }

    /// Outer None is lock contention; inner None is no published cost model.
    pub fn try_snapshot(&self) -> Option<Option<Arc<EngineCostSnapshot>>> {
        #[cfg(test)]
        self.snapshot_reads
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.training.try_snapshot()
    }

    /// The same nonblocking version gate is used during publication and by the
    /// final HostGuard. None means a concurrent publication owns the lock.
    pub(in crate::continuous_engine) fn try_model_version_current(
        &self,
        expected: u64,
    ) -> Option<bool> {
        self.try_snapshot()
            .map(|value| value.is_some_and(|model| model.model_version() == expected))
    }

    /// Owned provenance for the most recently published calibration. Reading
    /// diagnostics must not retain the worker's receipt lock across a later
    /// publication, and the startup import cannot mask a newer live receipt.
    pub fn profile_receipt(&self) -> Option<ferrum_types::SloCostProfileReceipt> {
        self.training
            .published_catalog_receipt()
            .or_else(|| self.training.receipt.clone())
    }

    pub fn trained_samples(&self) -> u64 {
        self.training.trained_samples()
    }

    #[cfg(test)]
    pub(in crate::continuous_engine) fn test_activity(&self) -> (usize, u64) {
        (
            usize::from(self.worker.is_some()),
            self.snapshot_reads
                .load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    pub fn audit_snapshot(&self) -> audit::ObservationFunnelSnapshot {
        let mut snapshot = self.training.audit_snapshot();
        snapshot.prospective_capture = self
            .prospective_capture
            .as_ref()
            .map(|audit| audit.snapshot());
        snapshot.issued_structured_prediction = self
            .issued_structured_prediction
            .as_ref()
            .map(|audit| audit.snapshot());
        snapshot
    }
}

impl Drop for EngineCostRuntime {
    fn drop(&mut self) {
        // The existing worker Drop joins after this signal; it owns training
        // state and its clock independently. No IO or fallible wait is added
        // here, and no worker can reach this runtime to join itself.
        self.training.request_export_finalization();
        if let Some(worker) = &self.worker {
            worker.wake();
        }
    }
}

/// A leaf-local guard publishes on every return path. The evidence core still
/// rejects incomplete/cancelled work; dropping this guard never creates facts.
/// Training is delegated to the CPU worker and never runs in Drop.
pub(in crate::continuous_engine) struct ObservedCostCall(Option<EngineCostCall>);
impl ObservedCostCall {
    pub(super) fn new(call: EngineCostCall) -> Self {
        Self(Some(call))
    }
}
impl Deref for ObservedCostCall {
    type Target = EngineCostCall;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref().expect("owned until drop")
    }
}
impl DerefMut for ObservedCostCall {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut().expect("owned until drop")
    }
}
impl Drop for ObservedCostCall {
    fn drop(&mut self) {
        if let Some(call) = self.0.take() {
            let _ = call.finish();
        }
    }
}
