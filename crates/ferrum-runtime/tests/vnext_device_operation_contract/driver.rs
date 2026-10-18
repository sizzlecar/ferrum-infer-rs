use super::*;

#[derive(Default)]
pub(crate) struct DriverTrace {
    pub(crate) calls: u64,
}

pub(crate) struct TestDriver {
    pub(crate) runtime: Arc<TestRuntime>,
    pub(crate) trace: Arc<Mutex<DriverTrace>>,
}

impl fmt::Debug for TestDriver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TestDriver")
            .field("device", &self.runtime.descriptor().id)
            .finish_non_exhaustive()
    }
}

impl ResourceTransactionDriver for TestDriver {
    type Buffer = TestBuffer;
    type Runtime = TestRuntime;

    fn runtime(&self) -> &Arc<Self::Runtime> {
        &self.runtime
    }

    fn device_id(&self) -> &DeviceId {
        &self.runtime.descriptor.id
    }

    fn device_runtime_implementation_fingerprint(&self) -> &str {
        &self.runtime.descriptor.runtime_implementation_fingerprint
    }

    fn device_capacity_bytes(&self) -> u64 {
        self.runtime.descriptor.total_memory_bytes
    }

    fn reserve_resource(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        _reservation: &ResourceReservation,
    ) -> Result<(), ResourceDriverFailure> {
        self.trace.lock().unwrap().calls += 1;
        Ok(())
    }

    fn commit_resource<'commit>(
        &mut self,
        context: &'commit ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
    ) -> Result<DeviceAllocationReceipt<'commit>, ResourceDriverFailure> {
        self.trace.lock().unwrap().calls += 1;
        let request = BufferRequest::new(
            reservation.resource_id().clone(),
            reservation.size_bytes(),
            reservation.alignment_bytes(),
            reservation.usage(),
            reservation.element_type(),
        )
        .unwrap();
        context
            .allocate(&request)
            .map_err(|_| resource_failure("allocation"))
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
        drop(ownership);
        Ok(())
    }

    fn abandon_transaction(&mut self, ownership: ResourcePoolOwnership<Self::Runtime>) {
        drop(ownership);
    }
}

pub(crate) fn resource_failure(code: &str) -> ResourceDriverFailure {
    ResourceDriverFailure::new(
        FailureEnvelope::new(FailureDomain::Resource, code, "resource failure", false).unwrap(),
    )
    .unwrap()
}

pub(crate) fn runtime(catalog: &CapabilityCatalog) -> (Arc<TestRuntime>, Arc<Mutex<RuntimeTrace>>) {
    let trace = Arc::new(Mutex::new(RuntimeTrace::default()));
    let descriptor = catalog.device().clone();
    let mut alternate_descriptor = descriptor.clone();
    alternate_descriptor.runtime_implementation_fingerprint = sha('f');
    (
        Arc::new(TestRuntime {
            descriptor,
            alternate_descriptor,
            use_alternate_descriptor: AtomicBool::new(false),
            descriptor_reads_until_drift: AtomicU64::new(0),
            trace: Arc::clone(&trace),
        }),
        trace,
    )
}
