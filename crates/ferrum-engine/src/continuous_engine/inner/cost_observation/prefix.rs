//! Exact maintenance calibration from acknowledged product checkpoint receipts.
//! This population shares the inference FIFO and byte ledgers, but never its
//! feedback monitor, structured learner, model epoch, or sample denominator.
use super::memory::ObservationBytePermit;
use super::*;
use ferrum_interfaces::vnext::{
    NativeCheckpointObservationSink, NativeCheckpointTransferObservation,
};
use ferrum_scheduler::implementations::continuous::cost_model as model;
use parking_lot::{Mutex, RwLock};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Weak;

mod shape;
pub(in crate::continuous_engine) use shape::prefix_cost_shape;
mod snapshot;
pub(in crate::continuous_engine) use snapshot::PrefixCostSnapshot;
#[cfg(test)]
mod tests;

/// Fixed-size normalized input; only the typed receipt consumer constructs it
/// in production. No checkpoint, sequence owner, or GPU allocation is retained.
pub(super) struct PrefixSample {
    observation: model::WaveCostObservation,
    loss_epoch: Option<Arc<Epoch>>,
}
impl PrefixSample {
    pub(super) fn abandon(&self) {
        if let Some(epoch) = &self.loss_epoch {
            epoch.next();
        }
    }
}

/// A new observation turn and any lost receipt both advance this authority.
/// A publication can expose only the revision captured before its own update;
/// it cannot reactivate after a concurrent producer-side loss.
struct Epoch {
    revision: AtomicU64,
    exhausted: AtomicBool,
    _base_memory: Arc<ObservationBytePermit>,
}
impl Epoch {
    fn new(base_memory: Arc<ObservationBytePermit>) -> Arc<Self> {
        Arc::new(Self {
            revision: AtomicU64::new(0),
            exhausted: AtomicBool::new(false),
            _base_memory: base_memory,
        })
    }
    fn next(&self) -> Option<u64> {
        if self.exhausted.load(Ordering::Acquire) {
            return None;
        }
        match self
            .revision
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
        {
            Ok(old) => Some(old + 1),
            Err(_) => {
                self.exhausted.store(true, Ordering::Release);
                None
            }
        }
    }
    fn current(&self, revision: u64) -> bool {
        revision != 0
            && !self.exhausted.load(Ordering::Acquire)
            && self.revision.load(Ordering::Acquire) == revision
    }
}

struct PrefixSink {
    queue: Weak<BoundedCostSampleSink>,
    fingerprint: model::ExecutionFingerprint,
    clock: Arc<dyn CostObservationClock>,
    epoch: Arc<Epoch>,
}
impl NativeCheckpointObservationSink for PrefixSink {
    fn try_record(&self, observation: NativeCheckpointTransferObservation) -> bool {
        // M2 emits Capture only after actual model cache/rendezvous publication
        // acknowledgement, and Restore only after target acknowledgement.
        let converted = (|| {
            let shape =
                prefix_cost_shape(observation.cost_domain(), observation.host_work()).ok()?;
            let wall_total_ns = u64::try_from(observation.wall_elapsed().as_nanos()).ok()?;
            if wall_total_ns == 0 {
                return None;
            }
            let observed_at_ns = self.clock.now_ns()?;
            Some(PrefixSample {
                loss_epoch: Some(self.epoch.clone()),
                observation: model::WaveCostObservation {
                    fingerprint: self.fingerprint.clone(),
                    actual_shape: shape,
                    boundary: model::CostBoundary::PreparationToCommit,
                    outcome: model::WaveObservationOutcome::Completed,
                    timing: model::WaveTiming {
                        wall_total_ns,
                        // Complete product wall is the training target. Optional
                        // device timing overlaps it and cannot extend this sample.
                        device_elapsed_ns: None,
                        stages: model::WaveStageTimings::default(),
                    },
                    observed_at_ns,
                },
            })
        })();
        let accepted = converted
            .and_then(|sample| self.queue.upgrade()?.offer_prefix(sample).ok())
            .is_some();
        if !accepted {
            // Lost maintenance evidence cannot silently leave an older
            // permission current; a later successful publication may recover.
            self.epoch.next();
        }
        accepted
    }
}

struct Mutable {
    trainer: model::CostModelTrainer,
    retained: ObservationBytePermit,
    // The trainer itself retains its published Arc. This lease must outlive
    // that Arc even when no planner holds the external snapshot wrapper.
    published_memory: Option<Arc<ObservationBytePermit>>,
}

pub(super) struct PrefixCostTraining {
    sink: Arc<PrefixSink>,
    queue: Arc<BoundedCostSampleSink>,
    state: Mutex<Mutable>,
    snapshot: RwLock<Option<Arc<PrefixCostSnapshot>>>,
    epoch: Arc<Epoch>,
    _base_memory: Arc<ObservationBytePermit>,
}
impl PrefixCostTraining {
    pub(super) fn new(
        queue: Arc<BoundedCostSampleSink>,
        fingerprint: model::ExecutionFingerprint,
        clock: Arc<dyn CostObservationClock>,
        config: &ferrum_types::SloCostModelConfig,
    ) -> Option<Self> {
        let base_bytes = std::mem::size_of::<Self>()
            .checked_add(std::mem::size_of::<PrefixSink>())?
            .checked_add(std::mem::size_of::<Epoch>())?
            .checked_add(6 * std::mem::size_of::<usize>())?;
        let base_memory = Arc::new(queue.reserve_prefix_working_bytes(base_bytes.checked_add(
            std::mem::size_of::<ObservationBytePermit>() + 2 * std::mem::size_of::<usize>(),
        )?)?);
        let retained = queue.reserve_prefix_working_bytes(0)?;
        let mut settings = profile::model_settings(config);
        // Exact maintenance is deliberately independent of inference's
        // StructuredV2/host feature selection; all original limits stay intact.
        settings.feature_model = model::CostFeatureModel::ExactV1 {};
        let trainer = model::CostModelTrainer::new(fingerprint.clone(), settings).ok()?;
        let epoch = Epoch::new(base_memory.clone());
        let sink = Arc::new(PrefixSink {
            queue: Arc::downgrade(&queue),
            fingerprint,
            clock,
            epoch: epoch.clone(),
        });
        Some(Self {
            sink,
            queue,
            state: Mutex::new(Mutable {
                trainer,
                retained,
                published_memory: None,
            }),
            snapshot: RwLock::new(None),
            epoch,
            _base_memory: base_memory,
        })
    }

    pub(super) fn sink(&self) -> Arc<dyn NativeCheckpointObservationSink> {
        self.sink.clone()
    }

    pub(super) fn snapshot(&self) -> Option<Arc<PrefixCostSnapshot>> {
        self.try_snapshot().flatten()
    }

    pub(super) fn try_snapshot(&self) -> Option<Option<Arc<PrefixCostSnapshot>>> {
        self.snapshot
            .try_read()
            .map(|snapshot| snapshot.as_ref().filter(|s| s.current()).cloned())
    }

    /// Sole worker, before any inference feedback handling. Receipt time is
    /// preserved even when this FIFO entry waited behind a long training turn.
    pub(super) fn consume(&self, sample: PrefixSample) {
        let revision = self.epoch.next();
        *self.snapshot.write() = None;
        if let Some(revision) = revision {
            let _ = self.consume_inner(sample, revision);
        }
    }

    fn consume_inner(&self, sample: PrefixSample, revision: u64) -> Option<()> {
        let mut state = self.state.lock();
        let now = sample.observation.observed_at_ns;
        let memory = state
            .trainer
            .exact_maintenance_memory_requirements(&sample.observation.actual_shape, now)
            .ok()?;
        let previous_retained = state.retained.bytes();
        if !state.retained.grow_to(memory.retained_bytes) {
            return None;
        }
        // Reserve before observe/publish creates keys, samples, cloned floor
        // trees, quantile scratch, or a fresh immutable snapshot.
        let wrapper_bytes = std::mem::size_of::<PrefixCostSnapshot>()
            .checked_add(std::mem::size_of::<ObservationBytePermit>())?
            .checked_add(4 * std::mem::size_of::<usize>())?;
        let snapshot_bytes = memory.snapshot_bytes.checked_add(wrapper_bytes)?;
        let Some(reservation) = self.queue.reserve_prefix_working_bytes(
            memory.publication_peak_bytes.checked_add(wrapper_bytes)?,
        ) else {
            let _ = state.retained.shrink_to(previous_retained);
            return None;
        };
        if !matches!(
            state.trainer.observe(sample.observation),
            Ok(model::ObservationDisposition::Recorded)
        ) {
            let _ = state.retained.shrink_to(previous_retained);
            return None;
        }
        let model = state.trainer.publish(now).ok()?;
        // If an accounting invariant ever prevents shrinking, retain the full
        // construction charge. A published trainer Arc must never lose it.
        let _ = reservation.shrink_to(snapshot_bytes);
        let memory = Arc::new(reservation);
        let snapshot = Arc::new(PrefixCostSnapshot::new(
            model,
            self.epoch.clone(),
            revision,
            memory.clone(),
        ));
        // Swap the trainer's matching lease only after its new Arc exists.
        state.published_memory = Some(memory);
        // An intervening lost sample advanced the authority and cannot be
        // overwritten by this otherwise successful publication.
        if snapshot.current() {
            *self.snapshot.write() = Some(snapshot);
        }
        Some(())
    }
}
