use super::{
    invalid_resource, validate_runtime_descriptor_for_admission, Arc, AtomicBool, BufferDescriptor,
    BufferRequest, DeviceCapacityClaim, DeviceId, DeviceRuntime, FailureDomain, FailureEnvelope,
    Ordering, PhantomData, RefCell, RequestIdentity, ResourceAbandonSignal, ResourceId,
    ResourcePoolId, ResourcePoolIdentity, ResourceReservation, ResourceReservationBatch,
    ResourceTransactionAction, ResourceTransactionState, RunId, Serialize,
    StaticProvisioningBinding, TransactionId, VNextError,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourceTransactionIdentity {
    pub(super) pool_id: ResourcePoolId,
    pub(super) run_id: RunId,
    pub(super) transaction_id: TransactionId,
    pub(super) request_id: RequestIdentity,
}

impl ResourceTransactionIdentity {
    pub fn for_admission(
        admission: &StaticProvisioningBinding,
        run_id: RunId,
        transaction_id: TransactionId,
    ) -> Self {
        Self {
            pool_id: admission.pool_id(),
            run_id,
            transaction_id,
            request_id: admission.request_id().clone(),
        }
    }

    pub const fn pool_id(&self) -> ResourcePoolId {
        self.pool_id
    }

    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    pub fn transaction_id(&self) -> &TransactionId {
        &self.transaction_id
    }

    pub fn request_id(&self) -> &RequestIdentity {
        &self.request_id
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ResourceActionCursor {
    pub(super) order: usize,
    pub(super) action: ResourceTransactionAction,
    pub(super) before: ResourceTransactionState,
    pub(super) allocation_authorized: bool,
}

pub struct ResourceTransactionContext<'a, R>
where
    R: DeviceRuntime,
{
    pub(super) runtime: &'a Arc<R>,
    pub(super) identity: &'a ResourceTransactionIdentity,
    pub(super) binding: &'a StaticProvisioningBinding,
    pub(super) reservations: &'a ResourceReservationBatch,
    pub(super) cursor: Option<ResourceActionCursor>,
    pub(super) allocation_authority: Option<&'a AtomicBool>,
    pub(super) pending_allocation: Option<&'a RefCell<Option<CoreOwnedAllocation<R::Buffer>>>>,
}

impl<'a, R> ResourceTransactionContext<'a, R>
where
    R: DeviceRuntime,
{
    pub fn identity(&self) -> &ResourceTransactionIdentity {
        self.identity
    }

    pub fn admission(&self) -> &StaticProvisioningBinding {
        self.binding
    }

    pub fn reservations(&self) -> &ResourceReservationBatch {
        self.reservations
    }

    fn allocation_permit<'permit>(
        &'permit self,
        request: &'permit BufferRequest,
    ) -> Result<DeviceAllocationPermit<'permit>, VNextError> {
        let cursor = self
            .cursor
            .filter(|cursor| {
                cursor.action == ResourceTransactionAction::Commit
                    && cursor.before == ResourceTransactionState::Reserved
                    && cursor.allocation_authorized
            })
            .ok_or_else(|| {
                invalid_resource("device allocation is authorized only during an exact commit")
            })?;
        let reservation = &self.reservations.reservations[cursor.order];
        if request.resource_id() != reservation.resource_id()
            || request.size_bytes() != reservation.size_bytes()
            || request.alignment_bytes() != reservation.alignment_bytes()
            || request.usage() != reservation.usage()
            || request.element_type() != reservation.element_type()
        {
            return Err(invalid_resource(
                "buffer request differs from the active admitted allocation",
            ));
        }
        self.allocation_authority
            .ok_or_else(|| invalid_resource("commit allocation authority is unavailable"))?
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| {
                invalid_resource(
                    "the active resource action already consumed its allocation permit",
                )
            })?;
        Ok(DeviceAllocationPermit {
            identity: self.identity,
            binding: self.binding,
            reservation,
            request,
            seal: AllocationSeal,
        })
    }

    /// The only allocation path exposed to a transaction driver. A successful
    /// runtime allocation is installed in core-owned pending storage before
    /// this method returns. The receipt contains metadata only and is tied to
    /// this commit call, so dropping it cannot lose buffer ownership.
    pub fn allocate<'commit>(
        &'commit self,
        request: &BufferRequest,
    ) -> Result<DeviceAllocationReceipt<'commit>, DeviceAllocationError<R::Error>> {
        validate_runtime_descriptor_for_admission(
            self.runtime.descriptor(),
            self.binding,
            "allocation preflight",
        )
        .map_err(DeviceAllocationError::Contract)?;
        let permit = self
            .allocation_permit(request)
            .map_err(DeviceAllocationError::Contract)?;
        let resource_id = permit.resource_id().clone();
        let generation = permit.generation();
        let allocation = self
            .runtime
            .allocate(permit)
            .map_err(DeviceAllocationError::Runtime)?;
        let reservation = &self.reservations.reservations[self
            .cursor
            .expect("allocation permit requires an action cursor")
            .order];
        let pending = self.pending_allocation.ok_or_else(|| {
            DeviceAllocationError::Contract(invalid_resource(
                "core pending allocation storage is unavailable",
            ))
        })?;
        if pending.borrow().is_some() {
            return Err(DeviceAllocationError::Contract(invalid_resource(
                "core pending allocation storage is already occupied",
            )));
        }
        let expected_descriptor = BufferDescriptor {
            resource_id: reservation.resource_id.clone(),
            size_bytes: reservation.size_bytes,
            alignment_bytes: reservation.alignment_bytes,
            usage: reservation.usage,
            element_type: reservation.element_type,
        };
        pending.replace(Some(CoreOwnedAllocation {
            resource_id: resource_id.clone(),
            generation,
            descriptor: expected_descriptor,
            buffer: allocation,
        }));
        let descriptor = {
            let pending = pending.borrow();
            self.runtime.buffer_descriptor(
                &pending
                    .as_ref()
                    .expect("allocation was installed before descriptor inspection")
                    .buffer,
            )
        };
        pending
            .borrow_mut()
            .as_mut()
            .expect("allocation remains core-owned during descriptor inspection")
            .descriptor = descriptor.clone();
        validate_runtime_descriptor_for_admission(
            self.runtime.descriptor(),
            self.binding,
            "allocation completion",
        )
        .map_err(DeviceAllocationError::Contract)?;
        Ok(DeviceAllocationReceipt {
            resource_id,
            generation,
            descriptor,
            scope: PhantomData,
        })
    }
}

pub(super) struct AllocationSeal;

#[must_use = "a device allocation permit must be consumed by DeviceRuntime::allocate"]
pub struct DeviceAllocationPermit<'a> {
    pub(super) identity: &'a ResourceTransactionIdentity,
    pub(super) binding: &'a StaticProvisioningBinding,
    pub(super) reservation: &'a ResourceReservation,
    pub(super) request: &'a BufferRequest,
    pub(super) seal: AllocationSeal,
}

impl<'a> DeviceAllocationPermit<'a> {
    pub fn identity(&self) -> &ResourceTransactionIdentity {
        self.identity
    }

    pub fn admission(&self) -> &StaticProvisioningBinding {
        self.binding
    }

    pub fn reservation(&self) -> &ResourceReservation {
        self.reservation
    }

    pub fn request(&self) -> &BufferRequest {
        self.request
    }

    pub fn resource_id(&self) -> &ResourceId {
        self.reservation.resource_id()
    }

    pub const fn generation(&self) -> u64 {
        self.reservation.generation()
    }

    pub fn into_request(self) -> &'a BufferRequest {
        let _ = self.seal;
        self.request
    }
}

#[derive(Debug)]
pub enum DeviceAllocationError<E> {
    Contract(VNextError),
    Runtime(E),
}

impl<E> DeviceAllocationError<E> {
    pub fn contract_error(&self) -> Option<&VNextError> {
        match self {
            Self::Contract(error) => Some(error),
            Self::Runtime(_) => None,
        }
    }

    pub fn runtime_error(&self) -> Option<&E> {
        match self {
            Self::Contract(_) => None,
            Self::Runtime(error) => Some(error),
        }
    }
}

#[must_use = "an allocation receipt must be returned by the active commit call"]
pub struct DeviceAllocationReceipt<'commit> {
    resource_id: ResourceId,
    generation: u64,
    descriptor: BufferDescriptor,
    scope: PhantomData<&'commit mut ()>,
}

impl DeviceAllocationReceipt<'_> {
    pub fn resource_id(&self) -> &ResourceId {
        &self.resource_id
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub fn descriptor(&self) -> &BufferDescriptor {
        &self.descriptor
    }
}

pub(super) struct DriverCommitAcknowledgement {
    resource_id: ResourceId,
    generation: u64,
    descriptor: BufferDescriptor,
}

impl DriverCommitAcknowledgement {
    pub(super) fn from_receipt(receipt: &DeviceAllocationReceipt<'_>) -> Self {
        Self {
            resource_id: receipt.resource_id.clone(),
            generation: receipt.generation,
            descriptor: receipt.descriptor.clone(),
        }
    }

    pub(super) fn matches<B>(&self, allocation: &CoreOwnedAllocation<B>) -> bool {
        self.resource_id == allocation.resource_id
            && self.generation == allocation.generation
            && self.descriptor == allocation.descriptor
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceDriverFailure {
    failure: FailureEnvelope,
}

impl ResourceDriverFailure {
    pub fn new(failure: FailureEnvelope) -> Result<Self, VNextError> {
        failure.validate()?;
        if failure.domain() != FailureDomain::Resource {
            return Err(invalid_resource(
                "resource driver failure must use the resource failure domain",
            ));
        }
        Ok(Self { failure })
    }

    pub fn failure(&self) -> &FailureEnvelope {
        &self.failure
    }

    pub fn into_failure(self) -> FailureEnvelope {
        self.failure
    }
}

pub(super) struct CoreOwnedAllocation<B> {
    pub(super) resource_id: ResourceId,
    pub(super) generation: u64,
    pub(super) descriptor: BufferDescriptor,
    pub(super) buffer: B,
}

impl<B> CoreOwnedAllocation<B> {
    pub(super) fn matches(&self, reservation: &ResourceReservation) -> bool {
        self.resource_id == reservation.resource_id
            && self.generation == reservation.generation
            && reservation.matches_descriptor(&self.descriptor)
    }
}

/// Borrowed view used to reconcile an invalid allocation. Core retains the
/// actual buffer regardless of the driver's return value.
pub struct ResourceCommitView<'a, B> {
    pub(super) resource_id: &'a ResourceId,
    pub(super) generation: u64,
    pub(super) descriptor: &'a BufferDescriptor,
    pub(super) buffer: &'a B,
}

impl<'a, B> ResourceCommitView<'a, B> {
    pub fn resource_id(&self) -> &ResourceId {
        self.resource_id
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub fn descriptor(&self) -> &BufferDescriptor {
        self.descriptor
    }

    pub fn buffer(&self) -> &B {
        self.buffer
    }
}

#[must_use = "owned quarantine buffers must be retained until backend cleanup"]
pub struct ResourceOwnedBuffer<B> {
    pub(super) order: usize,
    pub(super) expected_resource_id: ResourceId,
    pub(super) actual_resource_id: ResourceId,
    pub(super) expected_generation: u64,
    pub(super) actual_generation: u64,
    pub(super) expected_descriptor: BufferDescriptor,
    pub(super) actual_descriptor: BufferDescriptor,
    pub(super) buffer: B,
}

impl<B> ResourceOwnedBuffer<B> {
    pub fn resource_id(&self) -> &ResourceId {
        &self.expected_resource_id
    }

    pub const fn generation(&self) -> u64 {
        self.expected_generation
    }

    pub fn actual_resource_id(&self) -> &ResourceId {
        &self.actual_resource_id
    }

    pub const fn actual_generation(&self) -> u64 {
        self.actual_generation
    }

    pub fn expected_descriptor(&self) -> &BufferDescriptor {
        &self.expected_descriptor
    }

    pub fn actual_descriptor(&self) -> &BufferDescriptor {
        &self.actual_descriptor
    }

    pub fn buffer(&self) -> &B {
        &self.buffer
    }

    pub(super) fn into_allocation(self) -> (usize, CoreOwnedAllocation<B>) {
        (
            self.order,
            CoreOwnedAllocation {
                resource_id: self.actual_resource_id,
                generation: self.actual_generation,
                descriptor: self.actual_descriptor,
                buffer: self.buffer,
            },
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceOwnershipReason {
    Quarantine,
    Abandon,
}

/// Ownership transferred out of core when normal cleanup cannot prove that
/// buffers and their device-capacity claim are gone. Dropping this object is
/// the durable owner's explicit cleanup point.
#[must_use = "resource ownership must remain durable until cleanup completes"]
pub struct ResourcePoolOwnership<R>
where
    R: DeviceRuntime,
{
    // Declaration order is the normal cleanup order. Device buffers must be
    // gone before their capacity becomes reusable, and both must precede the
    // backend runtime/context teardown.
    pub(super) buffers: Vec<ResourceOwnedBuffer<R::Buffer>>,
    pub(super) capacity_claim: Option<DeviceCapacityClaim>,
    pub(super) pool_identity: ResourcePoolIdentity,
    pub(super) reason: ResourceOwnershipReason,
    pub(super) signal: Option<ResourceAbandonSignal>,
    pub(super) runtime: Arc<R>,
}

impl<R> ResourcePoolOwnership<R>
where
    R: DeviceRuntime,
{
    pub fn runtime(&self) -> &R {
        &self.runtime
    }

    pub fn pool_identity(&self) -> &ResourcePoolIdentity {
        &self.pool_identity
    }

    pub const fn reason(&self) -> ResourceOwnershipReason {
        self.reason
    }

    pub fn abandon_signal(&self) -> Option<&ResourceAbandonSignal> {
        self.signal.as_ref()
    }

    pub fn buffers(&self) -> &[ResourceOwnedBuffer<R::Buffer>] {
        &self.buffers
    }

    pub fn claimed_bytes(&self) -> u64 {
        self.capacity_claim
            .as_ref()
            .map_or(0, DeviceCapacityClaim::bytes)
    }

    fn must_retain_on_drop(&self) -> bool {
        std::thread::panicking()
    }
}

impl<R> Drop for ResourcePoolOwnership<R>
where
    R: DeviceRuntime,
{
    fn drop(&mut self) {
        if !self.must_retain_on_drop() {
            return;
        }

        // A driver that drops unresolved ownership has violated the durable
        // cleanup contract. Preserve memory safety by retaining device buffers
        // and the capacity claim; the abandon signal and recovery registry made
        // the leak observable before this last-resort path.
        for buffer in std::mem::take(&mut self.buffers) {
            std::mem::forget(buffer);
        }
        if let Some(claim) = self.capacity_claim.take() {
            std::mem::forget(claim);
        }
        // Buffer and stream handles may borrow backend-owned device/context
        // state without expressing that lifetime in their Rust types. Retain
        // both owners as one unit; dropping either after intentionally
        // retaining an in-flight handle would invalidate the safety fallback.
        std::mem::forget(Arc::clone(&self.runtime));
    }
}

#[must_use = "a failed ownership transfer must be returned to core"]
pub struct ResourceOwnershipTransferFailure<R>
where
    R: DeviceRuntime,
{
    failure: ResourceDriverFailure,
    ownership: ResourcePoolOwnership<R>,
}

impl<R> ResourceOwnershipTransferFailure<R>
where
    R: DeviceRuntime,
{
    pub fn new(failure: ResourceDriverFailure, ownership: ResourcePoolOwnership<R>) -> Self {
        Self { failure, ownership }
    }

    pub fn failure(&self) -> &ResourceDriverFailure {
        &self.failure
    }

    pub fn ownership(&self) -> &ResourcePoolOwnership<R> {
        &self.ownership
    }

    pub(super) fn into_parts(self) -> (ResourceDriverFailure, ResourcePoolOwnership<R>) {
        (self.failure, self.ownership)
    }
}

/// Backend adapter for one resource at a time. Core owns action ordering,
/// actual state, all buffers, receipts, and recovery progress. Methods must be
/// idempotent for the full identity/action/resource/generation key.
pub trait ResourceTransactionDriver: Send {
    type Buffer: Send + Sync + 'static;
    type Runtime: DeviceRuntime<Buffer = Self::Buffer>;

    fn runtime(&self) -> &Arc<Self::Runtime>;

    fn device_id(&self) -> &DeviceId;

    fn device_runtime_implementation_fingerprint(&self) -> &str;

    fn device_capacity_bytes(&self) -> u64;

    fn reserve_resource(
        &mut self,
        context: &ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
    ) -> Result<(), ResourceDriverFailure>;

    fn commit_resource<'commit>(
        &mut self,
        context: &'commit ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
    ) -> Result<DeviceAllocationReceipt<'commit>, ResourceDriverFailure>;

    fn compensate_reserve_resource(
        &mut self,
        context: &ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
    ) -> Result<(), ResourceDriverFailure>;

    fn compensate_commit_resource(
        &mut self,
        context: &ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
        buffer: &Self::Buffer,
    ) -> Result<(), ResourceDriverFailure>;

    fn rollback_resource(
        &mut self,
        context: &ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
    ) -> Result<(), ResourceDriverFailure>;

    fn release_resource(
        &mut self,
        context: &ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
        buffer: &Self::Buffer,
    ) -> Result<(), ResourceDriverFailure>;

    fn reconcile_commit_outcome(
        &mut self,
        context: &ResourceTransactionContext<'_, Self::Runtime>,
        expected: &ResourceReservation,
        actual: ResourceCommitView<'_, Self::Buffer>,
    ) -> Result<(), ResourceDriverFailure>;

    fn quarantine_transaction(
        &mut self,
        context: &ResourceTransactionContext<'_, Self::Runtime>,
        ownership: ResourcePoolOwnership<Self::Runtime>,
    ) -> Result<(), ResourceOwnershipTransferFailure<Self::Runtime>>;

    fn abandon_transaction(&mut self, ownership: ResourcePoolOwnership<Self::Runtime>);
}
