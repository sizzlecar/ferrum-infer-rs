use super::*;

pub(super) struct OwnedLeaseSlot<B> {
    pub(super) entry: ResourceLeaseEntry,
    pub(super) actual_resource_id: Option<ResourceId>,
    pub(super) actual_generation: Option<u64>,
    pub(super) descriptor: Option<BufferDescriptor>,
    pub(super) buffer: Option<B>,
}

impl<B> OwnedLeaseSlot<B> {
    pub(super) fn new(reservation: &ResourceReservation) -> Self {
        Self {
            entry: ResourceLeaseEntry::from_reservation(reservation, ResourceLeaseState::Active),
            actual_resource_id: None,
            actual_generation: None,
            descriptor: None,
            buffer: None,
        }
    }

    pub(super) fn install(&mut self, allocation: CoreOwnedAllocation<B>) {
        self.actual_resource_id = Some(allocation.resource_id);
        self.actual_generation = Some(allocation.generation);
        self.descriptor = Some(allocation.descriptor);
        self.buffer = Some(allocation.buffer);
    }

    pub(super) fn clear(&mut self) {
        drop(self.buffer.take());
        self.descriptor.take();
        self.actual_resource_id.take();
        self.actual_generation.take();
    }

    pub(super) fn take_allocation(&mut self) -> Option<CoreOwnedAllocation<B>> {
        Some(CoreOwnedAllocation {
            resource_id: self.actual_resource_id.take()?,
            generation: self.actual_generation.take()?,
            descriptor: self.descriptor.take()?,
            buffer: self.buffer.take()?,
        })
    }

    pub(super) fn restore_allocation(&mut self, allocation: CoreOwnedAllocation<B>) {
        debug_assert!(self.buffer.is_none());
        self.install(allocation);
    }
}

/// Borrowed access to a live, active, generation-bound committed buffer.
pub struct LeasedBufferView<'a, B> {
    pub(super) identity: &'a ResourceTransactionIdentity,
    pub(super) admission: &'a StaticProvisioningBinding,
    pub(super) resource_id: &'a ResourceId,
    pub(super) generation: u64,
    pub(super) descriptor: &'a BufferDescriptor,
    pub(super) buffer: &'a B,
}

impl<'a, B> LeasedBufferView<'a, B> {
    pub fn identity(&self) -> &ResourceTransactionIdentity {
        self.identity
    }

    pub fn admission(&self) -> &StaticProvisioningBinding {
        self.admission
    }

    pub fn resource_id(&self) -> &ResourceId {
        self.resource_id
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub fn committed_descriptor(&self) -> &BufferDescriptor {
        self.descriptor
    }

    pub fn buffer(&self) -> &B {
        self.buffer
    }
}

#[derive(Debug)]
pub enum ExecutionStreamCreationError<E> {
    Contract(VNextError),
    Runtime(E),
}

/// A stream created and owned by one exact admitted runtime instance. Its
/// runtime and raw stream are private so execution can only proceed through an
/// active sequence permit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BoundExecutionStreamState {
    Ready,
    InUse,
    Poisoned,
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

    fn attach_stream(&self, key: (u32, u64), stream: R::Stream) {
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

#[must_use = "an execution stream must be activated through its resource lease"]
pub struct BoundExecutionStream<R>
where
    R: DeviceRuntime,
{
    // The raw stream drops or transfers into the recovery registry before this
    // owning sequence hold can release logical capacity or backing extents.
    pub(super) runtime: Arc<R>,
    pub(super) coordinator_id: LogicalAdmissionCoordinatorId,
    pub(super) sequence_authority: SequenceAuthorityId,
    pub(super) stream: Option<R::Stream>,
    pub(super) state: BoundExecutionStreamState,
    pub(super) sequence_recovery: Arc<SequenceRecoveryRegistry<R>>,
    pub(super) sequence_dispatch_gate: Arc<AtomicU64>,
    pub(super) abandoned_sequence: Option<(u32, u64)>,
    pub(super) resources: Arc<AdmittedSequenceResources<R>>,
}

impl<R> BoundExecutionStream<R>
where
    R: DeviceRuntime,
{
    pub(super) fn stream(&self) -> &R::Stream {
        self.stream
            .as_ref()
            .expect("bound execution stream retains its raw stream")
    }

    pub(super) fn stream_mut(&mut self) -> &mut R::Stream {
        self.stream
            .as_mut()
            .expect("bound execution stream retains its raw stream")
    }
}

impl<R> Drop for BoundExecutionStream<R>
where
    R: DeviceRuntime,
{
    fn drop(&mut self) {
        let Some(key) = self.abandoned_sequence.take() else {
            return;
        };
        self.sequence_dispatch_gate
            .fetch_or(SEQUENCE_DISPATCH_POISONED_BIT, Ordering::AcqRel);
        if let Some(stream) = self.stream.take() {
            self.sequence_recovery.attach_stream(key, stream);
        }
    }
}
