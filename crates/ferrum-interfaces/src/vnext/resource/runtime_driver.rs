use super::{
    Arc, BufferRequest, DeviceAllocationError, DeviceAllocationReceipt, DeviceId, DeviceRuntime,
    FailureDomain, FailureEnvelope, ResourceCommitView, ResourceDriverFailure,
    ResourceOwnershipTransferFailure, ResourcePoolOwnership, ResourceReservation,
    ResourceTransactionContext, ResourceTransactionDriver, VNextError,
};

/// Production transaction adapter for a concrete [`DeviceRuntime`].
///
/// Core owns reservation ordering, allocation authority, buffers, and capacity
/// claims. This adapter therefore has no parallel allocator ledger: reserve,
/// rollback, and release are acknowledgements, while commit consumes the exact
/// core-issued allocation permit.
pub struct RuntimeResourceDriver<R>
where
    R: DeviceRuntime,
{
    runtime: Arc<R>,
    retained_ownership: Vec<ResourcePoolOwnership<R>>,
}

impl<R> RuntimeResourceDriver<R>
where
    R: DeviceRuntime,
{
    pub fn new(runtime: Arc<R>) -> Result<Self, VNextError> {
        runtime.descriptor().validate()?;
        Ok(Self {
            runtime,
            retained_ownership: Vec::new(),
        })
    }

    pub fn runtime(&self) -> &Arc<R> {
        &self.runtime
    }

    /// Number of pools retained after an indeterminate transaction outcome.
    /// Normal provisioning and shutdown leave this at zero.
    pub fn retained_pool_count(&self) -> usize {
        self.retained_ownership.len()
    }

    fn failure(
        code: &'static str,
        message: impl std::fmt::Display,
        retryable: bool,
    ) -> ResourceDriverFailure {
        let message = message
            .to_string()
            .chars()
            .filter(|character| !character.is_control() || matches!(character, '\n' | '\t'))
            .take(1024)
            .collect::<String>();
        ResourceDriverFailure::new(
            FailureEnvelope::new(FailureDomain::Resource, code, message, retryable)
                .expect("runtime resource driver failures use bounded static metadata"),
        )
        .expect("runtime resource driver failures use the resource domain")
    }

    fn allocation_failure(&self, error: DeviceAllocationError<R::Error>) -> ResourceDriverFailure {
        match error {
            DeviceAllocationError::Contract(error) => {
                Self::failure("allocation_contract", error, false)
            }
            DeviceAllocationError::Runtime(error) => match self.runtime.describe_error(&error) {
                Ok(report) => {
                    Self::failure("device_allocation", report.message(), report.retryable())
                }
                Err(classification_error) => Self::failure(
                    "device_allocation_unclassified",
                    format!("{error}; classification failed: {classification_error}"),
                    false,
                ),
            },
        }
    }
}

impl<R> ResourceTransactionDriver for RuntimeResourceDriver<R>
where
    R: DeviceRuntime,
{
    type Buffer = R::Buffer;
    type Runtime = R;

    fn runtime(&self) -> &Arc<Self::Runtime> {
        &self.runtime
    }

    fn device_id(&self) -> &DeviceId {
        &self.runtime.descriptor().id
    }

    fn device_runtime_implementation_fingerprint(&self) -> &str {
        &self.runtime.descriptor().runtime_implementation_fingerprint
    }

    fn device_capacity_bytes(&self) -> u64 {
        self.runtime.descriptor().total_memory_bytes
    }

    fn reserve_resource(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        _reservation: &ResourceReservation,
    ) -> Result<(), ResourceDriverFailure> {
        Ok(())
    }

    fn commit_resource<'commit>(
        &mut self,
        context: &'commit ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
    ) -> Result<DeviceAllocationReceipt<'commit>, ResourceDriverFailure> {
        let request = BufferRequest::new(
            reservation.resource_id().clone(),
            reservation.size_bytes(),
            reservation.alignment_bytes(),
            reservation.usage(),
            reservation.element_type(),
        )
        .map_err(|error| Self::failure("buffer_request", error, false))?;
        context
            .allocate(&request)
            .map_err(|error| self.allocation_failure(error))
    }

    fn compensate_reserve_resource(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        _reservation: &ResourceReservation,
    ) -> Result<(), ResourceDriverFailure> {
        Ok(())
    }

    fn compensate_commit_resource(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        _reservation: &ResourceReservation,
        _buffer: &Self::Buffer,
    ) -> Result<(), ResourceDriverFailure> {
        Ok(())
    }

    fn rollback_resource(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        _reservation: &ResourceReservation,
    ) -> Result<(), ResourceDriverFailure> {
        Ok(())
    }

    fn release_resource(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        _reservation: &ResourceReservation,
        _buffer: &Self::Buffer,
    ) -> Result<(), ResourceDriverFailure> {
        Ok(())
    }

    fn reconcile_commit_outcome(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        _expected: &ResourceReservation,
        _actual: ResourceCommitView<'_, Self::Buffer>,
    ) -> Result<(), ResourceDriverFailure> {
        Ok(())
    }

    fn quarantine_transaction(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        ownership: ResourcePoolOwnership<Self::Runtime>,
    ) -> Result<(), ResourceOwnershipTransferFailure<Self::Runtime>> {
        self.retained_ownership.push(ownership);
        Ok(())
    }

    fn abandon_transaction(&mut self, ownership: ResourcePoolOwnership<Self::Runtime>) {
        self.retained_ownership.push(ownership);
    }
}
