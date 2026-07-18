use super::*;

#[derive(Debug)]
pub(crate) struct TestBuffer {
    pub(crate) descriptor: BufferDescriptor,
}

#[derive(Debug)]
pub(crate) struct TestStream {
    pub(crate) synchronizations: u64,
    pub(crate) failed: Arc<AtomicBool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TestFence {
    Succeeded,
    Failed,
    Indeterminate,
    ContractFailed,
}

#[derive(Debug)]
pub(crate) struct TestRuntimeError;

impl fmt::Display for TestRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("test runtime failure")
    }
}

impl Error for TestRuntimeError {}

#[derive(Debug, Clone)]
pub(crate) struct TestRuntime {
    pub(crate) descriptor: DeviceDescriptor,
    pub(crate) fail_next_fence: Arc<AtomicBool>,
    pub(crate) indeterminate_next_fence: Arc<AtomicBool>,
    pub(crate) contract_fail_next_fence: Arc<AtomicBool>,
    pub(crate) panic_next_submit: Arc<AtomicBool>,
    pub(crate) synchronize_fails: Arc<AtomicBool>,
    pub(crate) stream_failed: Arc<AtomicBool>,
}

impl TestRuntime {
    pub(crate) fn fail_next_fence(&self) {
        assert!(!self.fail_next_fence.swap(true, Ordering::SeqCst));
    }

    pub(crate) fn make_next_fence_indeterminate(&self) {
        assert!(!self.indeterminate_next_fence.swap(true, Ordering::SeqCst));
    }

    pub(crate) fn contract_fail_next_fence(&self) {
        assert!(!self.contract_fail_next_fence.swap(true, Ordering::SeqCst));
    }

    pub(crate) fn panic_next_submit(&self) {
        assert!(!self.panic_next_submit.swap(true, Ordering::SeqCst));
    }

    pub(crate) fn set_synchronize_fails(&self, fails: bool) {
        self.synchronize_fails.store(fails, Ordering::SeqCst);
    }

    pub(crate) fn reset_stream_failure(&self) {
        self.stream_failed.store(false, Ordering::SeqCst);
    }
}

impl DeviceRuntime for TestRuntime {
    type Buffer = TestBuffer;
    type Stream = TestStream;
    type Command = ();
    type Fence = TestFence;
    type Error = TestRuntimeError;

    fn descriptor(&self) -> &DeviceDescriptor {
        &self.descriptor
    }

    fn allocate(&self, permit: DeviceAllocationPermit<'_>) -> Result<Self::Buffer, Self::Error> {
        let request = permit.into_request();
        Ok(TestBuffer {
            descriptor: BufferDescriptor {
                resource_id: request.resource_id().clone(),
                size_bytes: request.size_bytes(),
                alignment_bytes: request.alignment_bytes(),
                usage: request.usage(),
                element_type: request.element_type(),
            },
        })
    }

    fn buffer_descriptor(&self, buffer: &Self::Buffer) -> BufferDescriptor {
        buffer.descriptor.clone()
    }

    fn create_stream(&self) -> Result<Self::Stream, Self::Error> {
        Ok(TestStream {
            synchronizations: 0,
            failed: Arc::clone(&self.stream_failed),
        })
    }

    fn stream_state(&self, stream: &Self::Stream) -> StreamState {
        if stream.failed.load(Ordering::SeqCst) {
            StreamState::Failed
        } else {
            StreamState::Ready
        }
    }

    fn encode_copy(
        &self,
        _source: &Self::Buffer,
        _destination: &Self::Buffer,
        _region: CopyRegion,
    ) -> Result<Self::Command, Self::Error> {
        Ok(())
    }

    fn encode_upload(
        &self,
        _source: &[u8],
        _source_layout: HostTransferLayout,
        _destination: &Self::Buffer,
        _destination_offset_bytes: u64,
    ) -> Result<Self::Command, Self::Error> {
        Ok(())
    }

    fn encode_zero(
        &self,
        _destination: &Self::Buffer,
        _destination_offset_bytes: u64,
        _length_bytes: u64,
    ) -> Result<Self::Command, Self::Error> {
        Ok(())
    }

    fn submit(
        &self,
        _stream: &mut Self::Stream,
        commands: DeviceCommandBatch<Self::Command>,
    ) -> Result<Self::Fence, DefinitelyNotSubmitted<Self::Error>> {
        assert!(!commands.is_empty(), "core must not submit an empty batch");
        assert!(
            !self.panic_next_submit.swap(false, Ordering::SeqCst),
            "injected event submit panic"
        );
        Ok(
            if self.indeterminate_next_fence.swap(false, Ordering::SeqCst) {
                TestFence::Indeterminate
            } else if self.contract_fail_next_fence.swap(false, Ordering::SeqCst) {
                TestFence::ContractFailed
            } else if self.fail_next_fence.swap(false, Ordering::SeqCst) {
                TestFence::Failed
            } else {
                TestFence::Succeeded
            },
        )
    }

    fn query_fence(&self, fence: &Self::Fence) -> FenceQuery<Self::Error> {
        match fence {
            TestFence::Succeeded => {
                FenceQuery::Terminal(DeviceTerminalReceipt::unprofiled(DeviceTerminal::Succeeded))
            }
            TestFence::Failed => FenceQuery::Terminal(DeviceTerminalReceipt::unprofiled(
                DeviceTerminal::FailedButQuiescent(TestRuntimeError),
            )),
            TestFence::Indeterminate => FenceQuery::Indeterminate(TestRuntimeError),
            TestFence::ContractFailed => {
                self.stream_failed.store(true, Ordering::SeqCst);
                FenceQuery::Terminal(DeviceTerminalReceipt::unprofiled(DeviceTerminal::Succeeded))
            }
        }
    }

    fn wait_fence(
        &self,
        fence: &Self::Fence,
    ) -> Result<DeviceTerminalReceipt<Self::Error>, FenceIndeterminate<Self::Error>> {
        match fence {
            TestFence::Succeeded => {
                Ok(DeviceTerminalReceipt::unprofiled(DeviceTerminal::Succeeded))
            }
            TestFence::Failed => Ok(DeviceTerminalReceipt::unprofiled(
                DeviceTerminal::FailedButQuiescent(TestRuntimeError),
            )),
            TestFence::Indeterminate => Err(FenceIndeterminate::new(TestRuntimeError)),
            TestFence::ContractFailed => {
                self.stream_failed.store(true, Ordering::SeqCst);
                Ok(DeviceTerminalReceipt::unprofiled(DeviceTerminal::Succeeded))
            }
        }
    }

    fn synchronize(&self, stream: &mut Self::Stream) -> Result<(), Self::Error> {
        stream.synchronizations += 1;
        if self.synchronize_fails.load(Ordering::SeqCst) {
            return Err(TestRuntimeError);
        }
        stream.failed.store(false, Ordering::SeqCst);
        Ok(())
    }

    fn readback(
        &self,
        _stream: &mut Self::Stream,
        _source: &Self::Buffer,
        _region: CopyRegion,
        output_layout: HostTransferLayout,
    ) -> Result<Vec<u8>, Self::Error> {
        Ok(vec![0; output_layout.byte_len().unwrap() as usize])
    }

    fn describe_error(&self, error: &Self::Error) -> Result<DeviceErrorReport, VNextError> {
        DeviceErrorReport::new("test_runtime", error.to_string(), false)
    }
}

pub(crate) struct TestExecutionProvider {
    pub(crate) descriptor: OperationProviderDescriptor,
}

impl TestExecutionProvider {
    pub(crate) fn new(catalog: &CapabilityCatalog) -> Self {
        Self {
            descriptor: catalog.providers_for(&id("operation.main")).unwrap()[0].clone(),
        }
    }
}

impl OperationResourceEstimator for TestExecutionProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            16,
            None,
            None,
        ))
    }
}

impl OperationProvider<TestRuntime> for TestExecutionProvider {
    fn encode_selected(
        &self,
        _invocation: BatchedOperationInvocation<'_, TestBuffer>,
    ) -> Result<(), OperationFailure> {
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_and_submit_single(
    provider: &BoundOperationProvider<'_, TestRuntime>,
    resolved: &ResolvedModelPlan,
    identity: &ExecutionIdentityEnvelope,
    frame_id: &ExecutionFrameId,
    node_invocation_id: &NodeInvocationId,
    node_id: &NodeId,
    active: &TrustedActiveSequenceBinding,
    invocation: InvocationResourceLease<TestRuntime>,
    lane: &Arc<ExecutionLane<TestRuntime>>,
    reaper: &Arc<CompletionReaper<TestRuntime>>,
) -> Result<CompletionHandle<TestRuntime>, OperationDispatchError<TestRuntime>> {
    let parts = identity.parts();
    if parts.frame_id != Some(*frame_id)
        || parts.node_invocation_id != Some(*node_invocation_id)
        || parts.node_id.as_ref() != Some(node_id)
    {
        return Err(OperationDispatchError::Contract(
            VNextError::InvalidExecutionPlan {
                reason: "single-participant dispatch arguments disagree".to_owned(),
            },
        ));
    }
    let active_bindings = std::slice::from_ref(active);
    let batch_identity = OperationDispatch::bind_batch_identity(
        resolved,
        vec![identity.clone()],
        active_bindings,
        &invocation,
        lane,
    )
    .map_err(OperationDispatchError::Contract)?;
    OperationDispatch::encode_and_submit(
        provider,
        resolved,
        &batch_identity,
        active_bindings,
        invocation,
        lane,
        reaper,
    )
}

#[derive(Debug, Default)]
pub(crate) struct DriverTrace {
    pub(crate) reconciles: usize,
    pub(crate) quarantines: usize,
    pub(crate) abandon: Vec<ResourceAbandonSignal>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommitBehavior {
    Valid,
    InvalidFirst,
}

#[derive(Debug)]
pub(crate) struct TestDriver {
    pub(crate) device_id: DeviceId,
    pub(crate) device_runtime_implementation_fingerprint: String,
    pub(crate) device_capacity_bytes: u64,
    pub(crate) behavior: CommitBehavior,
    pub(crate) invalid_returned: bool,
    pub(crate) trace: Arc<Mutex<DriverTrace>>,
    pub(crate) runtime: Arc<TestRuntime>,
}

impl ResourceTransactionDriver for TestDriver {
    type Buffer = TestBuffer;
    type Runtime = TestRuntime;

    fn runtime(&self) -> &Arc<Self::Runtime> {
        &self.runtime
    }

    fn device_id(&self) -> &DeviceId {
        &self.device_id
    }

    fn device_runtime_implementation_fingerprint(&self) -> &str {
        &self.device_runtime_implementation_fingerprint
    }

    fn device_capacity_bytes(&self) -> u64 {
        self.device_capacity_bytes
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
        if self.behavior == CommitBehavior::InvalidFirst && !self.invalid_returned {
            self.invalid_returned = true;
            return Err(ResourceDriverFailure::new(
                FailureEnvelope::new(
                    FailureDomain::Resource,
                    "commit_injected_failure",
                    "injected commit failure",
                    true,
                )
                .unwrap(),
            )
            .unwrap());
        }
        let request = BufferRequest::new(
            reservation.resource_id().clone(),
            reservation.size_bytes(),
            reservation.alignment_bytes(),
            reservation.usage(),
            reservation.element_type(),
        )
        .unwrap();
        context.allocate(&request).map_err(|error| {
            ResourceDriverFailure::new(
                FailureEnvelope::new(
                    FailureDomain::Resource,
                    "commit_allocation_failed",
                    format!("{error:?}"),
                    false,
                )
                .unwrap(),
            )
            .unwrap()
        })
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
        reservation: &ResourceReservation,
        buffer: &Self::Buffer,
    ) -> Result<(), ResourceDriverFailure> {
        assert_eq!(buffer.descriptor.resource_id, *reservation.resource_id());
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
        reservation: &ResourceReservation,
        buffer: &Self::Buffer,
    ) -> Result<(), ResourceDriverFailure> {
        assert_eq!(buffer.descriptor.resource_id, *reservation.resource_id());
        Ok(())
    }

    fn reconcile_commit_outcome(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        _expected: &ResourceReservation,
        _actual: ResourceCommitView<'_, Self::Buffer>,
    ) -> Result<(), ResourceDriverFailure> {
        self.trace.lock().unwrap().reconciles += 1;
        Ok(())
    }

    fn quarantine_transaction(
        &mut self,
        _context: &ResourceTransactionContext<'_, Self::Runtime>,
        ownership: ResourcePoolOwnership<Self::Runtime>,
    ) -> Result<(), ResourceOwnershipTransferFailure<Self::Runtime>> {
        assert!(!ownership.buffers().is_empty());
        self.trace.lock().unwrap().quarantines += 1;
        drop(ownership);
        Ok(())
    }

    fn abandon_transaction(&mut self, ownership: ResourcePoolOwnership<Self::Runtime>) {
        self.trace.lock().unwrap().abandon.push(
            ownership
                .abandon_signal()
                .expect("abandon must carry signal")
                .clone(),
        );
        drop(ownership);
    }
}

pub(crate) fn driver(
    plan: &ExecutionPlan,
    behavior: CommitBehavior,
) -> (TestDriver, Arc<Mutex<DriverTrace>>) {
    let trace = Arc::new(Mutex::new(DriverTrace::default()));
    let runtime = Arc::new(TestRuntime {
        descriptor: DeviceDescriptor {
            id: plan.payload().device_id().clone(),
            class: DeviceClass::Reference,
            ordinal: 0,
            total_memory_bytes: plan.payload().memory().device_capacity_bytes(),
            runtime_implementation_fingerprint: plan
                .payload()
                .device_runtime_implementation_fingerprint()
                .to_owned(),
            capabilities: BTreeSet::from([id("capability.compute")]),
            dynamic_storage_profiles: BTreeSet::from([contiguous_storage_profile()]),
        },
        fail_next_fence: Arc::new(AtomicBool::new(false)),
        indeterminate_next_fence: Arc::new(AtomicBool::new(false)),
        contract_fail_next_fence: Arc::new(AtomicBool::new(false)),
        panic_next_submit: Arc::new(AtomicBool::new(false)),
        synchronize_fails: Arc::new(AtomicBool::new(false)),
        stream_failed: Arc::new(AtomicBool::new(false)),
    });
    (
        TestDriver {
            device_id: plan.payload().device_id().clone(),
            device_runtime_implementation_fingerprint: plan
                .payload()
                .device_runtime_implementation_fingerprint()
                .to_owned(),
            device_capacity_bytes: plan.payload().memory().device_capacity_bytes(),
            behavior,
            invalid_returned: false,
            trace: trace.clone(),
            runtime,
        },
        trace,
    )
}

pub(crate) fn transaction(
    plan: &ExecutionPlan,
    run_id: &str,
    transaction_id: &str,
    provisioning_request: &str,
    behavior: CommitBehavior,
) -> (
    ResourceTransaction<TestDriver, TransactionNew>,
    Arc<Mutex<DriverTrace>>,
    Arc<TestRuntime>,
) {
    let request: RequestIdentity = id(provisioning_request);
    let (driver, trace) = driver(plan, behavior);
    let runtime = Arc::clone(driver.runtime());
    let ProvisionedPlanParts { provisioning } = plan
        .provision_static(Arc::clone(driver.runtime()), request.clone())
        .unwrap()
        .into_parts();
    let permit = match provisioning {
        StaticProvisioning::Required(permit) => permit,
        StaticProvisioning::NoStatic(_) => {
            panic!("event fixture unexpectedly has no physical provisioning")
        }
    };
    let pool_ids = permit
        .maintenance_controller()
        .pool_ids()
        .cloned()
        .collect::<Vec<_>>();
    permit
        .maintenance_controller()
        .initialize_pools(&pool_ids)
        .unwrap();
    let identity = ResourceTransactionIdentity::for_admission(
        permit.binding(),
        id(run_id),
        id(transaction_id),
    );
    (
        ResourceTransaction::begin(driver, identity, permit).unwrap(),
        trace,
        runtime,
    )
}
