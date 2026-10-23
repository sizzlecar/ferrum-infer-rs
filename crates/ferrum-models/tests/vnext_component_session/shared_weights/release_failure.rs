//! Inject only the existing transaction driver's fallible release acknowledgement;
//! allocation, upload, ownership and device storage use the real Reference runtime.
use super::*;

struct ReleaseDriver {
    inner: RuntimeResourceDriver<ReferenceDeviceRuntime>,
    fail: Arc<AtomicBool>,
}
impl ResourceTransactionDriver for ReleaseDriver {
    type Runtime = ReferenceDeviceRuntime;
    type Buffer = <ReferenceDeviceRuntime as DeviceRuntime>::Buffer;
    fn runtime(&self) -> &Arc<Self::Runtime> {
        self.inner.runtime()
    }
    fn device_id(&self) -> &DeviceId {
        self.inner.device_id()
    }
    fn device_runtime_implementation_fingerprint(&self) -> &str {
        self.inner.device_runtime_implementation_fingerprint()
    }
    fn device_capacity_bytes(&self) -> u64 {
        self.inner.device_capacity_bytes()
    }
    fn reserve_resource(
        &mut self,
        context: &ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
    ) -> Result<(), ResourceDriverFailure> {
        self.inner.reserve_resource(context, reservation)
    }
    fn commit_resource<'c>(
        &mut self,
        context: &'c ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
    ) -> Result<DeviceAllocationReceipt<'c>, ResourceDriverFailure> {
        self.inner.commit_resource(context, reservation)
    }
    fn compensate_reserve_resource(
        &mut self,
        context: &ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
    ) -> Result<(), ResourceDriverFailure> {
        self.inner.compensate_reserve_resource(context, reservation)
    }
    fn compensate_commit_resource(
        &mut self,
        context: &ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
        buffer: &Self::Buffer,
    ) -> Result<(), ResourceDriverFailure> {
        self.inner
            .compensate_commit_resource(context, reservation, buffer)
    }
    fn rollback_resource(
        &mut self,
        context: &ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
    ) -> Result<(), ResourceDriverFailure> {
        self.inner.rollback_resource(context, reservation)
    }
    fn release_resource(
        &mut self,
        context: &ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
        buffer: &Self::Buffer,
    ) -> Result<(), ResourceDriverFailure> {
        if self.fail.load(Ordering::Relaxed) {
            return Err(ResourceDriverFailure::new(
                FailureEnvelope::new(
                    FailureDomain::Resource,
                    "shared_release_failure",
                    "injected release acknowledgement failure",
                    true,
                )
                .unwrap(),
            )
            .unwrap());
        }
        self.inner.release_resource(context, reservation, buffer)
    }
    fn reconcile_commit_outcome(
        &mut self,
        context: &ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
        actual: ResourceCommitView<'_, Self::Buffer>,
    ) -> Result<(), ResourceDriverFailure> {
        self.inner
            .reconcile_commit_outcome(context, reservation, actual)
    }
    fn quarantine_transaction(
        &mut self,
        context: &ResourceTransactionContext<'_, Self::Runtime>,
        ownership: ResourcePoolOwnership<Self::Runtime>,
    ) -> Result<(), ResourceOwnershipTransferFailure<Self::Runtime>> {
        self.inner.quarantine_transaction(context, ownership)
    }
    fn abandon_transaction(&mut self, ownership: ResourcePoolOwnership<Self::Runtime>) {
        self.inner.abandon_transaction(ownership);
    }
}

#[test]
fn failed_last_weight_release_retains_physical_claim_until_successful_retry() {
    let fixture = Fixture::new("device.component.shared-release-failure");
    let compiled = fixture.compile_rows(2, &fixture.family);
    let plan = compiled.executable().execution_plan();
    let runtime = fixture.composition.runtime().clone();
    let StaticProvisioning::Required(permit) = plan
        .provision_static(runtime.clone(), id("request.shared.failure"))
        .unwrap()
        .into_provisioning()
    else {
        panic!("Stored fixture needs static admission")
    };
    let identity = ResourceTransactionIdentity::for_admission(
        permit.binding(),
        id("run.shared.failure"),
        id("transaction.shared.failure"),
    );
    let fail = Arc::new(AtomicBool::new(true));
    let driver = ReleaseDriver {
        inner: RuntimeResourceDriver::new(runtime.clone()).unwrap(),
        fail: fail.clone(),
    };
    let committed = ResourceTransaction::begin(driver, identity, permit)
        .unwrap()
        .reserve()
        .unwrap()
        .commit()
        .unwrap_or_else(|_| panic!("real allocation failed"));
    let initialized = committed
        .initialize_static_managed(
            &fixture.family,
            plan,
            &fixture.source,
            &fixture.host,
            StaticInitializationPolicy::new(WEIGHT_BYTES as u64, 8).unwrap(),
        )
        .unwrap_or_else(|error| panic!("initialization: {error}"));
    let (root, weights) = initialized
        .into_shared_plan_runtime(plan)
        .unwrap_or_else(|error| panic!("handoff: {error}"));
    match PlanRuntimeResources::close(root) {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => {
            assert_eq!(receipt.released_static_resources(), 0)
        }
        _ => panic!("cache owns weight, root close must not attempt native release"),
    }
    let failure = match weights.close() {
        Err(failure) => failure,
        Ok(_) => panic!("release fault must remain observable"),
    };
    assert_eq!(failure.failure().code(), "shared_release_failure");
    assert_eq!(runtime.snapshot().live_allocations, 1);
    assert!(fixture
        .capacity
        .reserve(DeviceOpaqueMemoryKind::Bootstrap, DEVICE_BYTES)
        .is_err());
    fail.store(false, Ordering::Relaxed);
    let receipt = failure
        .retry()
        .unwrap_or_else(|error| panic!("retry: {:?}", error.failure()));
    assert_eq!(receipt.released_static_resources(), 1);
    fixture.assert_device_reusable();
    assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 0);
}
