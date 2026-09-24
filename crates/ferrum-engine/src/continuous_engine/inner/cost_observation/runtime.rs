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
    pub recorder_limits: CostRecorderLimits,
    // The worker owns another training Arc, but never an EngineCostRuntime.
    // Thus no self-cycle or join-from-worker is possible during Drop.
    training: Arc<CostTrainingState>,
    worker: Option<CostTrainingWorker>,
}

impl EngineCostRuntime {
    pub fn new(
        identity: ExecutorCostIdentityAvailability,
        config: &SloCostObservationConfig,
        profile_path: Option<&Path>,
    ) -> Result<Self, FerrumError> {
        let clock: Arc<dyn CostObservationClock> = Arc::new(EngineCostClock::default());
        let load_clock = profile_path
            .map(|_| profile::read_load_clock(clock.as_ref(), &config.profile_import))
            .transpose()?;
        Self::build_with_profile(identity, clock, config, true, profile_path, load_clock)
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

    pub(super) fn build_with_profile(
        identity: ExecutorCostIdentityAvailability,
        clock: Arc<dyn CostObservationClock>,
        config: &SloCostObservationConfig,
        background: bool,
        profile_path: Option<&Path>,
        load_clock: Option<ProfileLoadClock>,
    ) -> Result<Self, FerrumError> {
        config.validate().map_err(FerrumError::config)?;
        let recorder_limits = CostRecorderLimits {
            max_waves: config.max_waves_per_call.get(),
            max_rows_per_wave: config.max_rows_per_wave.get(),
            max_retained_rows: config.max_retained_rows_per_call.get(),
        };
        recorder_limits
            .validate()
            .map_err(|error| FerrumError::config(error.to_string()))?;
        let seed = profile::load_seed(&identity, config, profile_path, load_clock)?;
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
        let training = Arc::new(CostTrainingState::new(config, seed, export, clock.clone())?);
        let worker = if background {
            let state = super::trainer::TrainingWorkerOwner(training.clone());
            Some(
                CostTrainingWorker::spawn(move || state.consume_batch()).map_err(|error| {
                    FerrumError::backend(format!("start cost training worker: {error}"))
                })?,
            )
        } else {
            None
        };
        if let Some(worker) = &worker {
            training.sink.attach_worker(worker.notification_thread())?;
        }
        Ok(Self {
            clock,
            ids: EngineCostIds::default(),
            sink: training.sink.clone(),
            identity,
            recorder_limits,
            training,
            worker,
        })
    }

    /// A coalescing signal only: never locks, drains, calibrates or publishes
    /// on the inference thread. The bounded sink drops and counts excess work.
    pub fn wake_trainer(&self) {
        if let Some(worker) = &self.worker {
            worker.wake();
        }
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
        // The same live barrier is also useful at shutdown. A concurrent
        // checkpoint owns the sole slot; joining the worker still completes it.
        let checkpoint = self.request_checkpoint().ok();
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

    #[cfg(test)]
    pub fn with_training_paused<T>(&self, action: impl FnOnce() -> T) -> T {
        self.training.with_training_paused(action)
    }

    #[cfg(test)]
    pub fn shutdown_started(&self) -> bool {
        self.worker
            .as_ref()
            .is_some_and(CostTrainingWorker::shutdown_started)
    }

    pub fn snapshot(&self) -> Option<Arc<EngineCostSnapshot>> {
        self.training.snapshot()
    }

    /// Outer None is lock contention; inner None is no published cost model.
    pub fn try_snapshot(&self) -> Option<Option<Arc<EngineCostSnapshot>>> {
        self.training.try_snapshot()
    }

    pub fn profile_receipt(&self) -> Option<&ferrum_types::SloCostProfileReceipt> {
        self.training.receipt.as_ref()
    }

    pub fn trained_samples(&self) -> u64 {
        self.training.trained_samples()
    }

    pub fn audit_snapshot(&self) -> audit::ObservationFunnelSnapshot {
        self.training.audit_snapshot()
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
