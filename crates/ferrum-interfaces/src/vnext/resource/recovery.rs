use super::{
    invalid_resource, sequence_slot_active, sequence_slot_poisoned_drained,
    sequence_slot_poisoned_undrained, Arc, AtomicU64, BTreeMap, DeviceRuntime, Mutex, Ordering,
    PlanRuntimeResources, RequestIdentity, RunId, SequenceAuthorityId, Serialize, StreamState,
    TrustedPlanRuntimeEvidence, VNextError, SEQUENCE_DISPATCH_POISONED_BIT,
};

/// Terminal resource disposition produced by an explicit sequence abort.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ActiveSequenceAbortDisposition {
    SynchronizedAndPoisoned,
    SequenceSessionTerminalized,
}

/// Core-signed evidence that the exact active slot epoch was atomically
/// poisoned. This type is trusted output and has no deserialization or public
/// construction path.
#[derive(Debug, Serialize)]
#[must_use = "sequence abort evidence must be recorded by execution"]
pub struct ActiveSequenceAbortReceipt {
    pub(super) plan: TrustedPlanRuntimeEvidence,
    pub(super) sequence_authority: SequenceAuthorityId,
    pub(super) run_id: RunId,
    pub(super) request_id: RequestIdentity,
    pub(super) activation_epoch: u64,
    pub(super) runtime_implementation_fingerprint: String,
    pub(super) disposition: ActiveSequenceAbortDisposition,
}

impl ActiveSequenceAbortReceipt {
    pub fn plan(&self) -> &TrustedPlanRuntimeEvidence {
        &self.plan
    }

    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    pub fn request_id(&self) -> &RequestIdentity {
        &self.request_id
    }

    pub const fn sequence_authority(&self) -> SequenceAuthorityId {
        self.sequence_authority
    }

    pub const fn activation_epoch(&self) -> u64 {
        self.activation_epoch
    }

    pub fn runtime_implementation_fingerprint(&self) -> &str {
        &self.runtime_implementation_fingerprint
    }

    pub const fn disposition(&self) -> ActiveSequenceAbortDisposition {
        self.disposition
    }
}

#[derive(Debug)]
pub enum AbandonedSequenceRecoveryError<E> {
    Contract(VNextError),
    Runtime(E),
    StreamStillOwned { slot: u32, activation_epoch: u64 },
}

#[derive(Clone)]
pub(super) struct AbandonedSequenceMetadata {
    pub(super) plan: TrustedPlanRuntimeEvidence,
    pub(super) sequence_authority: SequenceAuthorityId,
    pub(super) run_id: RunId,
    pub(super) request_id: RequestIdentity,
    pub(super) slot: u32,
    pub(super) activation_epoch: u64,
    pub(super) runtime_implementation_fingerprint: String,
    pub(super) state: Arc<AtomicU64>,
    pub(super) sequence_dispatch_gate: Arc<AtomicU64>,
    pub(super) drained: bool,
}

impl AbandonedSequenceMetadata {
    pub(super) fn key(&self) -> (u32, u64) {
        (self.slot, self.activation_epoch)
    }

    fn abort_receipt(&self) -> ActiveSequenceAbortReceipt {
        ActiveSequenceAbortReceipt {
            plan: self.plan.clone(),
            sequence_authority: self.sequence_authority,
            run_id: self.run_id.clone(),
            request_id: self.request_id.clone(),
            activation_epoch: self.activation_epoch,
            runtime_implementation_fingerprint: self.runtime_implementation_fingerprint.clone(),
            disposition: ActiveSequenceAbortDisposition::SynchronizedAndPoisoned,
        }
    }
}

struct AbandonedSequenceRecord<R>
where
    R: DeviceRuntime,
{
    metadata: AbandonedSequenceMetadata,
    stream: AbandonedSequenceStream<R::Stream>,
}

enum AbandonedSequenceStream<S> {
    ExternallyOwned,
    Attached(S),
    Recovering,
}

pub(super) struct SequenceRecoveryRegistry<R>
where
    R: DeviceRuntime,
{
    // Records (and their raw streams) must drop before the owning root.
    records: Mutex<BTreeMap<(u32, u64), AbandonedSequenceRecord<R>>>,
    _resources: Arc<PlanRuntimeResources<R>>,
}

impl<R> SequenceRecoveryRegistry<R>
where
    R: DeviceRuntime,
{
    pub(super) fn new(resources: Arc<PlanRuntimeResources<R>>) -> Self {
        Self {
            records: Mutex::new(BTreeMap::new()),
            _resources: resources,
        }
    }

    fn lock_records(
        &self,
    ) -> std::sync::MutexGuard<'_, BTreeMap<(u32, u64), AbandonedSequenceRecord<R>>> {
        self.records
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(super) fn is_empty(&self) -> bool {
        self.lock_records().is_empty()
    }

    pub(super) fn register(&self, metadata: AbandonedSequenceMetadata) {
        let key = metadata.key();
        let mut records = self.lock_records();
        if records.contains_key(&key) {
            debug_assert!(false, "sequence recovery epoch registered twice");
            return;
        }
        records.insert(
            key,
            AbandonedSequenceRecord {
                metadata,
                stream: AbandonedSequenceStream::ExternallyOwned,
            },
        );
    }

    pub(super) fn attach_stream(&self, key: (u32, u64), stream: R::Stream) {
        let mut records = self.lock_records();
        let Some(record) = records.get_mut(&key) else {
            std::mem::forget(stream);
            return;
        };
        record
            .metadata
            .sequence_dispatch_gate
            .fetch_or(SEQUENCE_DISPATCH_POISONED_BIT, Ordering::AcqRel);
        match &record.stream {
            AbandonedSequenceStream::ExternallyOwned => {
                record.stream = AbandonedSequenceStream::Attached(stream);
            }
            AbandonedSequenceStream::Attached(_) | AbandonedSequenceStream::Recovering => {
                std::mem::forget(stream);
            }
        }
    }

    pub(super) fn set_drained(&self, key: (u32, u64), drained: bool) {
        let mut records = self.lock_records();
        let record = records
            .get_mut(&key)
            .expect("active sequence recovery metadata remains registered");
        record.metadata.drained = drained;
    }

    pub(super) fn clear(&self, key: (u32, u64)) {
        let removed = self.lock_records().remove(&key);
        debug_assert!(
            removed.is_some(),
            "terminal sequence lost recovery metadata"
        );
    }

    pub(super) fn recover(
        &self,
        runtime: &Arc<R>,
        slot: u32,
    ) -> Result<ActiveSequenceAbortReceipt, AbandonedSequenceRecoveryError<R::Error>> {
        // Move the raw stream into an explicit Recovering state, then release
        // the registry before invoking backend code. Concurrent recovery sees
        // the state and fails closed without blocking on the backend call.
        let (key, mut stream, was_drained) = {
            let mut records = self.lock_records();
            let matching = records
                .keys()
                .filter(|(candidate, _)| *candidate == slot)
                .copied()
                .collect::<Vec<_>>();
            if matching.len() != 1 {
                return Err(AbandonedSequenceRecoveryError::Contract(invalid_resource(
                    "abandoned sequence recovery requires one exact registered slot epoch",
                )));
            }
            let key = matching[0];
            let record = records
                .get_mut(&key)
                .expect("matching recovery key remains registered");
            let stream =
                match std::mem::replace(&mut record.stream, AbandonedSequenceStream::Recovering) {
                    AbandonedSequenceStream::Attached(stream) => stream,
                    AbandonedSequenceStream::ExternallyOwned => {
                        record.stream = AbandonedSequenceStream::ExternallyOwned;
                        return Err(AbandonedSequenceRecoveryError::StreamStillOwned {
                            slot,
                            activation_epoch: record.metadata.activation_epoch,
                        });
                    }
                    AbandonedSequenceStream::Recovering => {
                        record.stream = AbandonedSequenceStream::Recovering;
                        return Err(AbandonedSequenceRecoveryError::Contract(invalid_resource(
                        "abandoned sequence recovery is already in progress for this slot epoch",
                    )));
                    }
                };
            (key, stream, record.metadata.drained)
        };

        let backend_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if !was_drained {
                runtime.synchronize(&mut stream)?;
            }
            Ok(runtime.stream_state(&stream) == StreamState::Ready)
        }));
        let stream_ready = match backend_result {
            Ok(Ok(stream_ready)) => stream_ready,
            Ok(Err(error)) => {
                self.restore_recovery_stream(key, stream, false);
                return Err(AbandonedSequenceRecoveryError::Runtime(error));
            }
            Err(payload) => {
                self.restore_recovery_stream(key, stream, false);
                std::panic::resume_unwind(payload);
            }
        };
        if !stream_ready {
            self.restore_recovery_stream(key, stream, false);
            return Err(AbandonedSequenceRecoveryError::Contract(invalid_resource(
                "abandoned sequence synchronization did not drain its stream",
            )));
        }

        let mut records = self.lock_records();
        let Some(record) = records.get_mut(&key) else {
            std::mem::forget(stream);
            return Err(AbandonedSequenceRecoveryError::Contract(invalid_resource(
                "abandoned sequence recovery registration disappeared while synchronization was in progress",
            )));
        };
        if !matches!(record.stream, AbandonedSequenceStream::Recovering) {
            std::mem::forget(stream);
            return Err(AbandonedSequenceRecoveryError::Contract(invalid_resource(
                "abandoned sequence recovery ownership changed while synchronization was in progress",
            )));
        }
        record.metadata.drained = true;
        let expected_active = sequence_slot_active(record.metadata.activation_epoch);
        let expected_undrained = sequence_slot_poisoned_undrained(record.metadata.activation_epoch);
        let expected_drained = sequence_slot_poisoned_drained(record.metadata.activation_epoch);
        let actual = record.metadata.state.load(Ordering::Acquire);
        if actual == expected_active || actual == expected_undrained {
            if record
                .metadata
                .state
                .compare_exchange(
                    actual,
                    expected_drained,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_err()
            {
                record.stream = AbandonedSequenceStream::Attached(stream);
                return Err(AbandonedSequenceRecoveryError::Contract(invalid_resource(
                    "abandoned sequence epoch changed during recovery",
                )));
            }
        } else if actual != expected_drained {
            record.stream = AbandonedSequenceStream::Attached(stream);
            return Err(AbandonedSequenceRecoveryError::Contract(invalid_resource(
                "abandoned sequence recovery does not own an active or poisoned registered slot epoch",
            )));
        }
        record
            .metadata
            .sequence_dispatch_gate
            .fetch_or(SEQUENCE_DISPATCH_POISONED_BIT, Ordering::AcqRel);

        let receipt = record.metadata.abort_receipt();
        let removed = records.remove(&key);
        drop(records);
        drop(removed);
        drop(stream);
        Ok(receipt)
    }

    fn restore_recovery_stream(&self, key: (u32, u64), stream: R::Stream, drained: bool) {
        let mut records = self.lock_records();
        let Some(record) = records.get_mut(&key) else {
            std::mem::forget(stream);
            return;
        };
        if matches!(record.stream, AbandonedSequenceStream::Recovering) {
            record.metadata.drained = drained;
            record.stream = AbandonedSequenceStream::Attached(stream);
        } else {
            std::mem::forget(stream);
        }
    }

    pub(super) fn recover_all_for_owner_drop(
        &self,
        runtime: &Arc<R>,
    ) -> Result<(), AbandonedSequenceRecoveryError<R::Error>> {
        loop {
            let slot = {
                let records = self.lock_records();
                records.keys().next().map(|(slot, _)| *slot)
            };
            let Some(slot) = slot else {
                return Ok(());
            };
            let _ = self.recover(runtime, slot)?;
        }
    }
}
