//! An optional, bounded reference estimator for one immutable execution configuration.
//!
//! The assembly must create a separate instance for each model/device/numerical
//! configuration. Product token work is not physical kernel geometry. Predictions
//! are window maxima with an engineering margin, not latency quantiles or SLO proofs.
//! Queue accounting bounds retained logical payload, not allocator overhead or RSS;
//! readers own any old immutable snapshots they retain.

mod queue;
mod snapshot;
mod worker;

pub use ferrum_types::ReferenceEstimatorConfig;
pub use queue::ReferenceEstimatorStats;

use ferrum_execution::execution_attempt::ExecutionAttemptObserver;
use ferrum_scheduler::cost_prediction::CostSnapshotSource;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::watch;

#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum EstimatorError {
    #[error("invalid reference estimator configuration: {0}")]
    Configuration(String),
    #[error("reference estimator requires a live Tokio runtime")]
    RuntimeUnavailable,
    #[error("reference estimator worker panicked")]
    WorkerPanicked,
    #[error("reference estimator worker was cancelled")]
    WorkerCancelled,
    #[error("reference estimator completion service disconnected")]
    CompletionDisconnected,
}

/// Returned only after the original worker has actually exited and been joined.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub struct WorkerExit {
    pub stats: ReferenceEstimatorStats,
    pub revision: u64,
}

pub struct ReferenceEstimator;

/// The sole lifecycle owner. Observer/readers carry no Engine or execution owner.
/// Dropping a shutdown wait never cancels the original worker or its join service.
pub struct ReferenceEstimatorOwner {
    shared: Arc<queue::Shared>,
    observer: Arc<queue::Observer>,
    source: Arc<snapshot::Source>,
    completion: watch::Receiver<Option<Result<WorkerExit, EstimatorError>>>,
}

impl ReferenceEstimator {
    pub fn spawn(
        config: ReferenceEstimatorConfig,
    ) -> Result<ReferenceEstimatorOwner, EstimatorError> {
        Self::spawn_inner(config, worker::Hooks::default())
    }

    fn spawn_inner(
        config: ReferenceEstimatorConfig,
        hooks: worker::Hooks,
    ) -> Result<ReferenceEstimatorOwner, EstimatorError> {
        validate(&config)?;
        let runtime = tokio::runtime::Handle::try_current()
            .map_err(|_| EstimatorError::RuntimeUnavailable)?;
        let (sender, receiver) = crossbeam_channel::bounded(config.queue_capacity);
        let (wake, wake_receiver) = crossbeam_channel::bounded(1);
        let shared = Arc::new(queue::Shared::new(config, wake));
        let observer = Arc::new(queue::Observer::new(Arc::clone(&shared), sender));
        let (source, revisions) = snapshot::Source::new(config.max_participants);
        let source = Arc::new(source);
        let (completion_sender, completion) = watch::channel(None);
        let worker_shared = Arc::clone(&shared);
        let worker_source = Arc::clone(&source);
        let task = runtime.spawn_blocking(move || {
            worker::run(
                config,
                worker_shared,
                worker_source,
                revisions,
                receiver,
                wake_receiver,
                hooks,
            )
        });
        // This service owns the only JoinHandle, including after the public
        // owner and every shutdown future have been dropped.
        let joined_shared = Arc::clone(&shared);
        let joined_source = Arc::clone(&source);
        runtime.spawn(async move {
            let result = match task.await {
                Ok(()) => Ok(WorkerExit {
                    stats: joined_shared.stats(),
                    revision: joined_source.revision(),
                }),
                Err(error) => {
                    joined_shared.worker_failed();
                    Err(if error.is_panic() {
                        EstimatorError::WorkerPanicked
                    } else {
                        EstimatorError::WorkerCancelled
                    })
                }
            };
            completion_sender.send_replace(Some(result));
        });
        Ok(ReferenceEstimatorOwner {
            shared,
            observer,
            source,
            completion,
        })
    }
}

impl ReferenceEstimatorOwner {
    pub fn observer(&self) -> Arc<dyn ExecutionAttemptObserver> {
        self.observer.clone()
    }
    pub fn source(&self) -> Arc<dyn CostSnapshotSource> {
        self.source.clone()
    }
    pub fn stats(&self) -> ReferenceEstimatorStats {
        self.shared.stats()
    }

    /// Stop accepting new samples, drain all accepted samples, then observe
    /// the original worker's join. Cancellation is observation-only.
    pub async fn shutdown(&self) -> Result<WorkerExit, EstimatorError> {
        self.shared.stop();
        let mut completion = self.completion.clone();
        loop {
            if let Some(result) = completion.borrow_and_update().clone() {
                return result;
            }
            completion
                .changed()
                .await
                .map_err(|_| EstimatorError::CompletionDisconnected)?;
        }
    }
}

impl Drop for ReferenceEstimatorOwner {
    fn drop(&mut self) {
        self.shared.stop();
    }
}

fn validate(config: &ReferenceEstimatorConfig) -> Result<(), EstimatorError> {
    config.validate().map_err(EstimatorError::Configuration)?;
    let invalid = |message: &str| EstimatorError::Configuration(message.to_owned());
    if config.queue_capacity == 0
        || config.max_queue_bytes == 0
        || config.max_participants == 0
        || config.max_summary_bytes == 0
        || config.max_domains == 0
        || config.window_samples == 0
        || config.min_samples == 0
        || config.snapshot_ttl_ms == 0
    {
        return Err(invalid(
            "all capacity, sample and TTL limits must be positive",
        ));
    }
    if config.min_samples > config.window_samples {
        return Err(invalid("min_samples exceeds window_samples"));
    }
    if !config.safety_margin.is_finite() || config.safety_margin < 0.0 {
        return Err(invalid("safety_margin must be finite and nonnegative"));
    }
    // The bounded channel needs a representable lap/mark stamp as well as
    // addressable message storage. Reject impossible capacities before the
    // channel constructor can panic; this is not an available-RAM guarantee.
    config
        .queue_capacity
        .checked_add(1)
        .and_then(usize::checked_next_power_of_two)
        .and_then(|mark| mark.checked_mul(2))
        .ok_or_else(|| invalid("queue capacity exceeds bounded channel index range"))?;
    std::alloc::Layout::array::<queue::Envelope>(config.queue_capacity)
        .map_err(|_| invalid("queue message storage exceeds addressable allocation range"))?;
    let one = ferrum_execution::execution_attempt::ExecutionAttemptSummary::projected_bytes(1)
        .ok_or_else(|| invalid("summary extent overflow"))?;
    if one > config.max_summary_bytes || one > config.max_queue_bytes {
        return Err(invalid("byte limits cannot hold one participant"));
    }
    ferrum_execution::execution_attempt::ExecutionAttemptSummary::projected_bytes(
        config.max_participants,
    )
    .ok_or_else(|| invalid("participant extent overflow"))?;
    let members = config
        .max_domains
        .checked_mul(config.max_participants)
        .ok_or_else(|| invalid("domain extent overflow"))?;
    std::alloc::Layout::array::<snapshot::Member>(members)
        .map_err(|_| invalid("domain storage exceeds addressable allocation range"))?;
    let samples = config
        .max_domains
        .checked_mul(config.window_samples)
        .ok_or_else(|| invalid("window extent overflow"))?;
    std::alloc::Layout::array::<Duration>(samples)
        .map_err(|_| invalid("window storage exceeds addressable allocation range"))?;
    Instant::now()
        .checked_add(Duration::from_millis(config.snapshot_ttl_ms))
        .ok_or_else(|| invalid("snapshot TTL exceeds monotonic clock range"))?;
    Ok(())
}

#[cfg(test)]
mod tests;
