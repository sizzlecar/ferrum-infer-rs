mod vnext_device_operation_contract;

use vnext_device_operation_contract::*;

struct ZeroWeightSource;

impl WeightComponentSource for ZeroWeightSource {
    fn component<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        let byte_len = usize::try_from(component.physical_bytes()?).map_err(|_| {
            VNextError::InvalidExecutionPlan {
                reason: format!(
                    "test weight component `{}` exceeds the host address space",
                    component.id
                ),
            }
        })?;
        WeightComponentPayload::new(
            component,
            component
                .external_names
                .first()
                .expect("test component has an external name")
                .clone(),
            "model.safetensors",
            component.dimensions.clone(),
            component.physical_element_type(),
            vec![0_u8; byte_len],
        )
    }
}

struct StrictSourceWeightSource {
    requested: Arc<Mutex<Vec<WeightId>>>,
}

impl WeightComponentSource for StrictSourceWeightSource {
    fn component<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        if !matches!(
            component.id.as_str(),
            "weight.component.left" | "weight.component.right"
        ) {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!(
                    "strict source rejects derived execution component `{}`",
                    component.id
                ),
            });
        }
        self.requested.lock().unwrap().push(component.id.clone());
        ZeroWeightSource.component(component)
    }
}

struct DerivedComponentMaterializer {
    descriptor: WeightMaterializerDescriptor,
}

impl DerivedComponentMaterializer {
    fn new() -> Self {
        Self {
            descriptor: WeightMaterializerDescriptor::new(
                id("weight-materializer.test.derived-components"),
                ContractVersion::new(1, 0),
                sha('8'),
                WeightMaterializationFidelity::Exact,
                BTreeSet::from([id("capability.compute")]),
            )
            .unwrap(),
        }
    }
}

impl WeightMaterializer for DerivedComponentMaterializer {
    fn descriptor(&self) -> &WeightMaterializerDescriptor {
        &self.descriptor
    }

    fn execution_schema(
        &self,
        family: &PreparedModelFamily,
        _device: &DeviceDescriptor,
    ) -> Result<WeightSchema, VNextError> {
        let mut schema = family.weight_schema().clone();
        schema.layout_id = id("weight-layout.device-operation-derived");
        schema.components[0].id = id("weight.execution.left");
        schema.components[0].external_names = vec![
            "derived.left.primary.bin".to_owned(),
            "derived.left.secondary.bin".to_owned(),
        ];
        schema.components[1].id = id("weight.execution.right");
        schema.components[1].external_names = vec![
            "derived.right.primary.bin".to_owned(),
            "derived.right.secondary.bin".to_owned(),
        ];
        let PhysicalWeightLayout::Composite { parts } = &mut schema.tensors[0].physical_layout
        else {
            panic!("fixture weight must remain composite");
        };
        let PhysicalWeightLayout::Dense { component_id } = parts[0].layout.as_mut() else {
            panic!("fixture left weight must remain dense");
        };
        *component_id = id("weight.execution.left");
        let PhysicalWeightLayout::Dense { component_id } = parts[1].layout.as_mut() else {
            panic!("fixture right weight must remain dense");
        };
        *component_id = id("weight.execution.right");
        Ok(schema)
    }

    fn component_sources(
        &self,
        _family: &PreparedModelFamily,
        _execution_schema: &WeightSchema,
    ) -> Result<BTreeMap<WeightId, Vec<WeightId>>, VNextError> {
        Ok(BTreeMap::from([
            (
                id("weight.execution.left"),
                vec![id("weight.component.left"), id("weight.component.right")],
            ),
            (
                id("weight.execution.right"),
                vec![id("weight.component.left"), id("weight.component.right")],
            ),
        ]))
    }

    fn materialize_component<'source>(
        &self,
        source: &'source dyn WeightComponentSource,
        source_components: &[&WeightComponentSpec],
        execution_component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        if source_components.is_empty() {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "derived test materializer requires source components".to_owned(),
            });
        }
        let source_payloads = source_components
            .iter()
            .map(|source_component| source.component(source_component))
            .collect::<Result<Vec<_>, _>>()?;
        let fill = match execution_component.id.as_str() {
            "weight.execution.left" => 0xa1,
            "weight.execution.right" => 0xb2,
            other => {
                return Err(VNextError::InvalidExecutionPlan {
                    reason: format!("unexpected derived execution component `{other}`"),
                })
            }
        };
        let byte_len = usize::try_from(execution_component.physical_bytes()?).map_err(|_| {
            VNextError::InvalidExecutionPlan {
                reason: "derived test component exceeds host address space".to_owned(),
            }
        })?;
        WeightComponentPayload::from_ordered_sources(
            execution_component,
            execution_component.external_names.clone(),
            source_payloads
                .iter()
                .flat_map(|payload| payload.source_files().iter().cloned())
                .collect(),
            execution_component.dimensions.clone(),
            execution_component.physical_element_type(),
            vec![fill; byte_len],
        )
    }

    fn materialize_components<'source>(
        &self,
        source: &'source dyn WeightComponentSource,
        source_components: &[&WeightComponentSpec],
        execution_components: &[&WeightComponentSpec],
    ) -> Result<Vec<WeightComponentPayload<'source>>, VNextError> {
        if source_components.is_empty() {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "derived test materializer requires shared source components".to_owned(),
            });
        }
        let source_payloads = source_components
            .iter()
            .map(|source_component| source.component(source_component))
            .collect::<Result<Vec<_>, _>>()?;
        let source_files = source_payloads
            .iter()
            .flat_map(|payload| payload.source_files().iter().cloned())
            .collect::<Vec<_>>();
        execution_components
            .iter()
            .map(|execution_component| {
                let fill = match execution_component.id.as_str() {
                    "weight.execution.left" => 0xa1,
                    "weight.execution.right" => 0xb2,
                    other => {
                        return Err(VNextError::InvalidExecutionPlan {
                            reason: format!("unexpected derived execution component `{other}`"),
                        })
                    }
                };
                let byte_len =
                    usize::try_from(execution_component.physical_bytes()?).map_err(|_| {
                        VNextError::InvalidExecutionPlan {
                            reason: "derived test component exceeds host address space".to_owned(),
                        }
                    })?;
                WeightComponentPayload::from_ordered_sources(
                    execution_component,
                    execution_component.external_names.clone(),
                    source_files.clone(),
                    execution_component.dimensions.clone(),
                    execution_component.physical_element_type(),
                    vec![fill; byte_len],
                )
            })
            .collect()
    }
}

struct OrderedTransformSource;

impl WeightComponentSource for OrderedTransformSource {
    fn component<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        ZeroWeightSource.component(component)
    }

    fn component_segments<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentSegments<'source>, VNextError> {
        let (first, second) = match component.id.as_str() {
            "weight.component.left" => (0x11, 0x12),
            "weight.component.right" => (0x21, 0x22),
            other => panic!("unexpected transform source component `{other}`"),
        };
        WeightComponentSegments::from_ordered_segments(
            component,
            component.external_names.clone(),
            vec!["model.safetensors".to_owned()],
            component.dimensions.clone(),
            component.physical_element_type(),
            vec![
                WeightComponentSegment::new(vec![first; 3]),
                WeightComponentSegment::new(vec![second; 5]),
            ],
        )
    }
}

struct StaticTransformMaterializer {
    descriptor: WeightMaterializerDescriptor,
    transform_count: usize,
    host_materialization_calls: Arc<AtomicU64>,
}

impl StaticTransformMaterializer {
    fn new(transform_count: usize, host_materialization_calls: Arc<AtomicU64>) -> Self {
        assert!(matches!(transform_count, 1 | 2));
        Self {
            descriptor: WeightMaterializerDescriptor::new(
                id(format!(
                    "weight-materializer.test.static-transform-{transform_count}"
                )),
                ContractVersion::new(1, 0),
                sha(if transform_count == 1 { '6' } else { '7' }),
                WeightMaterializationFidelity::Exact,
                BTreeSet::from([id("capability.compute")]),
            )
            .unwrap(),
            transform_count,
            host_materialization_calls,
        }
    }

    fn transform_plans(&self) -> Vec<StaticWeightTransformPlan> {
        let mut plans = vec![StaticWeightTransformPlan::BlockFp8ToMarlinFp8Group128 {
            source_values_id: id("weight.component.left"),
            source_scales_id: id("weight.component.right"),
            packed_values_id: id("weight.execution.transform0.packed"),
            scales_id: id("weight.execution.transform0.scales"),
            logical_dimensions: vec![128, 128],
            matrices_per_output: 1,
        }];
        if self.transform_count == 2 {
            plans.push(StaticWeightTransformPlan::BlockFp8ToMarlinFp8Group128 {
                source_values_id: id("weight.component.right"),
                source_scales_id: id("weight.component.left"),
                packed_values_id: id("weight.execution.transform1.packed"),
                scales_id: id("weight.execution.transform1.scales"),
                logical_dimensions: vec![256, 128],
                matrices_per_output: 1,
            });
        }
        plans
    }
}

impl WeightMaterializer for StaticTransformMaterializer {
    fn descriptor(&self) -> &WeightMaterializerDescriptor {
        &self.descriptor
    }

    fn execution_schema(
        &self,
        family: &PreparedModelFamily,
        _device: &DeviceDescriptor,
    ) -> Result<WeightSchema, VNextError> {
        let mut schema = family.weight_schema().clone();
        schema.layout_id = id(format!(
            "weight-layout.device-operation-static-transform-{}",
            self.transform_count
        ));
        let component_count = self.transform_count * 2;
        schema.components = (0..self.transform_count)
            .flat_map(|index| {
                ["packed", "scales"].map(move |suffix| WeightComponentSpec {
                    id: id(format!("weight.execution.transform{index}.{suffix}")),
                    role: WeightComponentRole::Values,
                    external_names: vec![format!("transform{index}.{suffix}.bin")],
                    dimensions: vec![u64::try_from(4 / component_count).unwrap()],
                    encoding: WeightEncoding::Dense {
                        element_type: ElementType::F32,
                    },
                    required: true,
                })
            })
            .collect();
        schema.tensors[0].physical_layout = PhysicalWeightLayout::Composite {
            parts: schema
                .components
                .iter()
                .enumerate()
                .map(|(index, component)| CompositeWeightPart {
                    layout: Box::new(PhysicalWeightLayout::Dense {
                        component_id: component.id.clone(),
                    }),
                    logical_offsets: vec![u64::try_from(index * (4 / component_count)).unwrap()],
                    extents: vec![u64::try_from(4 / component_count).unwrap()],
                })
                .collect(),
        };
        Ok(schema)
    }

    fn component_sources(
        &self,
        _family: &PreparedModelFamily,
        _execution_schema: &WeightSchema,
    ) -> Result<BTreeMap<WeightId, Vec<WeightId>>, VNextError> {
        Ok(self
            .transform_plans()
            .into_iter()
            .flat_map(|plan| {
                let sources = plan
                    .source_component_ids()
                    .into_iter()
                    .cloned()
                    .collect::<Vec<_>>();
                plan.execution_component_ids()
                    .into_iter()
                    .cloned()
                    .map(move |execution_id| (execution_id, sources.clone()))
                    .collect::<Vec<_>>()
            })
            .collect())
    }

    fn static_weight_transforms(
        &self,
        _family: &PreparedModelFamily,
        _execution_schema: &WeightSchema,
    ) -> Result<Vec<StaticWeightTransformPlan>, VNextError> {
        Ok(self.transform_plans())
    }

    fn materialize_component<'source>(
        &self,
        _source: &'source dyn WeightComponentSource,
        _source_components: &[&WeightComponentSpec],
        _execution_component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        self.host_materialization_calls
            .fetch_add(1, Ordering::SeqCst);
        Err(VNextError::InvalidExecutionPlan {
            reason: "device transform must not fall back to host materialization".to_owned(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TransformCall {
    source_ids: Vec<WeightId>,
    source_segment_bytes: Vec<Vec<Vec<u8>>>,
    destination_ids: Vec<WeightId>,
    destination_buffers: Vec<BufferDescriptor>,
    destination_offsets: Vec<u64>,
    scratch: BufferDescriptor,
}

struct TransformRuntime {
    base: Arc<TestRuntime>,
    calls: Arc<Mutex<Vec<TransformCall>>>,
}

impl DeviceRuntime for TransformRuntime {
    type Buffer = TestBuffer;
    type Stream = TestStream;
    type Command = TestCommand;
    type Fence = TestFence;
    type Error = TestRuntimeError;

    fn descriptor(&self) -> &DeviceDescriptor {
        self.base.descriptor()
    }

    fn attention_execution_policy(&self) -> ferrum_types::AttentionExecutionPolicy {
        self.base.attention_execution_policy()
    }

    fn allocate(&self, permit: DeviceAllocationPermit<'_>) -> Result<Self::Buffer, Self::Error> {
        self.base.allocate(permit)
    }

    fn buffer_descriptor(&self, buffer: &Self::Buffer) -> BufferDescriptor {
        self.base.buffer_descriptor(buffer)
    }

    fn begin_static_weight_import(
        &self,
    ) -> Option<
        Result<Box<dyn StaticWeightImportSession<Self::Buffer, Self::Error> + '_>, Self::Error>,
    > {
        self.base.begin_static_weight_import()
    }

    fn encode_static_weight_transform(
        &self,
        request: StaticWeightTransformRequest<'_, '_, Self::Buffer>,
    ) -> Option<Result<Self::Command, Self::Error>> {
        self.calls.lock().unwrap().push(TransformCall {
            source_ids: request
                .sources()
                .iter()
                .map(|source| source.component_id().clone())
                .collect(),
            source_segment_bytes: request
                .sources()
                .iter()
                .map(|source| {
                    source
                        .segments()
                        .iter()
                        .map(|segment| segment.bytes().to_vec())
                        .collect()
                })
                .collect(),
            destination_ids: request
                .destinations()
                .iter()
                .map(|destination| destination.component().id.clone())
                .collect(),
            destination_buffers: request
                .destinations()
                .iter()
                .map(|destination| self.buffer_descriptor(destination.buffer()))
                .collect(),
            destination_offsets: request
                .destinations()
                .iter()
                .map(StaticWeightTransformDestination::destination_offset_bytes)
                .collect(),
            scratch: self.buffer_descriptor(request.scratch()),
        });
        Some(Ok(TestCommand::Copy))
    }

    fn create_stream(&self) -> Result<Self::Stream, Self::Error> {
        self.base.create_stream()
    }

    fn stream_state(&self, stream: &Self::Stream) -> StreamState {
        self.base.stream_state(stream)
    }

    fn encode_copy(
        &self,
        source: &Self::Buffer,
        destination: &Self::Buffer,
        region: CopyRegion,
    ) -> Result<Self::Command, Self::Error> {
        self.base.encode_copy(source, destination, region)
    }

    fn encode_upload(
        &self,
        source: &[u8],
        source_layout: HostTransferLayout,
        destination: &Self::Buffer,
        destination_offset_bytes: u64,
    ) -> Result<Self::Command, Self::Error> {
        self.base
            .encode_upload(source, source_layout, destination, destination_offset_bytes)
    }

    fn encode_zero(
        &self,
        destination: &Self::Buffer,
        destination_offset_bytes: u64,
        length_bytes: u64,
    ) -> Result<Self::Command, Self::Error> {
        self.base
            .encode_zero(destination, destination_offset_bytes, length_bytes)
    }

    fn submit(
        &self,
        stream: &mut Self::Stream,
        commands: DeviceCommandBatch<Self::Command>,
    ) -> Result<Self::Fence, DefinitelyNotSubmitted<Self::Error>> {
        self.base.submit(stream, commands)
    }

    fn submission_attribution(&self, fence: &Self::Fence) -> Option<DeviceSubmissionAttribution> {
        self.base.submission_attribution(fence)
    }

    fn query_fence(&self, fence: &Self::Fence) -> FenceQuery<Self::Error> {
        self.base.query_fence(fence)
    }

    fn wait_fence(
        &self,
        fence: &Self::Fence,
    ) -> Result<DeviceTerminalReceipt<Self::Error>, FenceIndeterminate<Self::Error>> {
        self.base.wait_fence(fence)
    }

    fn synchronize(&self, stream: &mut Self::Stream) -> Result<(), Self::Error> {
        self.base.synchronize(stream)
    }

    fn readback(
        &self,
        stream: &mut Self::Stream,
        source: &Self::Buffer,
        region: CopyRegion,
        output_layout: HostTransferLayout,
    ) -> Result<Vec<u8>, Self::Error> {
        self.base.readback(stream, source, region, output_layout)
    }

    fn describe_error(&self, error: &Self::Error) -> Result<DeviceErrorReport, VNextError> {
        self.base.describe_error(error)
    }
}

struct TransformDriver {
    runtime: Arc<TransformRuntime>,
}

impl fmt::Debug for TransformDriver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TransformDriver")
            .field("device", &self.runtime.descriptor().id)
            .finish_non_exhaustive()
    }
}

impl ResourceTransactionDriver for TransformDriver {
    type Buffer = TestBuffer;
    type Runtime = TransformRuntime;

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
        .unwrap();
        context
            .allocate(&request)
            .map_err(|_| resource_failure("transform-allocation"))
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

fn committed_transaction(
    plan: &ExecutionPlan,
    runtime: Arc<TestRuntime>,
    suffix: &str,
) -> ResourceTransaction<TestDriver, TransactionCommitted> {
    let ProvisionedPlanParts { provisioning } = plan
        .provision_static(
            Arc::clone(&runtime),
            id(format!("request.static-initialization.{suffix}")),
        )
        .unwrap()
        .into_parts();
    let admission = match provisioning {
        StaticProvisioning::Required(admission) => admission,
        StaticProvisioning::NoStatic(_) => {
            panic!("static initialization fixture requires plan-static resources")
        }
    };
    let identity = ResourceTransactionIdentity::for_admission(
        admission.binding(),
        id(format!("run.static-initialization.{suffix}")),
        id(format!("transaction.static-initialization.{suffix}")),
    );
    ResourceTransaction::begin(
        TestDriver {
            runtime,
            trace: Arc::new(Mutex::new(DriverTrace::default())),
        },
        identity,
        admission,
    )
    .unwrap()
    .reserve()
    .unwrap()
    .commit()
    .unwrap()
}

fn test_plan() -> (
    ResolvedModelPlan,
    ExecutionPlan,
    Arc<TestRuntime>,
    Arc<Mutex<RuntimeTrace>>,
) {
    let catalog = catalog();
    let registry = operation_registry(
        &catalog,
        Arc::new(Mutex::new(ProviderBehavior::Success)),
        Arc::new(Mutex::new(ProviderTrace::default())),
    );
    let (resolved, plan) = resolved_model_plan(&registry);
    let (runtime, trace) = runtime(&catalog);
    (resolved, plan, runtime, trace)
}

fn derived_component_test_plan() -> (
    PreparedModelFamily,
    ExecutionPlan,
    Arc<TestRuntime>,
    Arc<Mutex<RuntimeTrace>>,
) {
    let materializers =
        WeightMaterializerRegistry::new(vec![Box::new(DerivedComponentMaterializer::new())])
            .unwrap();
    let catalog = materializers.augment_catalog(catalog()).unwrap();
    let registry = operation_registry(
        &catalog,
        Arc::new(Mutex::new(ProviderBehavior::Success)),
        Arc::new(Mutex::new(ProviderTrace::default())),
    );
    let family = TestModelRegistry::new()
        .registration
        .prepare(&json!({"width": 4}))
        .unwrap();
    let runtime_policy = policy();
    let materializer_id = id("weight-materializer.test.derived-components");
    let mut options = ProgramPlanCompileOptions::new(BTreeMap::from([(
        id("value.input"),
        ProgramTensorSpec {
            dimensions: vec![4],
            element_type: ElementType::F32,
            layout: ResolvedTensorLayout::Contiguous,
        },
    )]))
    .unwrap();
    options.require_weight_materializer(materializer_id);
    let plan = ProgramPlanCompiler::compile_with_weight_materializers(
        &family,
        &catalog,
        &runtime_policy,
        &registry.planning(),
        &materializers,
        &options,
    )
    .unwrap()
    .executable()
    .execution_plan()
    .clone();
    let (runtime, trace) = runtime(&catalog);
    (family, plan, runtime, trace)
}

fn static_transform_test_plan(
    transform_count: usize,
) -> (
    PreparedModelFamily,
    ExecutionPlan,
    Arc<TestRuntime>,
    Arc<Mutex<RuntimeTrace>>,
    Arc<AtomicU64>,
) {
    let host_materialization_calls = Arc::new(AtomicU64::new(0));
    let materializer =
        StaticTransformMaterializer::new(transform_count, Arc::clone(&host_materialization_calls));
    let materializer_id = materializer.descriptor().id().clone();
    let materializers = WeightMaterializerRegistry::new(vec![Box::new(materializer)]).unwrap();
    let catalog = materializers.augment_catalog(catalog()).unwrap();
    let registry = operation_registry(
        &catalog,
        Arc::new(Mutex::new(ProviderBehavior::Success)),
        Arc::new(Mutex::new(ProviderTrace::default())),
    );
    let family = TestModelRegistry::new()
        .registration
        .prepare(&json!({"width": 4}))
        .unwrap();
    let mut options = ProgramPlanCompileOptions::new(BTreeMap::from([(
        id("value.input"),
        ProgramTensorSpec {
            dimensions: vec![4],
            element_type: ElementType::F32,
            layout: ResolvedTensorLayout::Contiguous,
        },
    )]))
    .unwrap();
    options.require_weight_materializer(materializer_id);
    let plan = ProgramPlanCompiler::compile_with_weight_materializers(
        &family,
        &catalog,
        &policy(),
        &registry.planning(),
        &materializers,
        &options,
    )
    .unwrap()
    .executable()
    .execution_plan()
    .clone();
    let (runtime, trace) = runtime(&catalog);
    (family, plan, runtime, trace, host_materialization_calls)
}

fn committed_transform_transaction(
    plan: &ExecutionPlan,
    runtime: Arc<TransformRuntime>,
    suffix: &str,
) -> ResourceTransaction<TransformDriver, TransactionCommitted> {
    let ProvisionedPlanParts { provisioning } = plan
        .provision_static(
            Arc::clone(&runtime),
            id(format!("request.static-transform.{suffix}")),
        )
        .unwrap()
        .into_parts();
    let admission = match provisioning {
        StaticProvisioning::Required(admission) => admission,
        StaticProvisioning::NoStatic(_) => panic!("transform fixture requires static resources"),
    };
    let identity = ResourceTransactionIdentity::for_admission(
        admission.binding(),
        id(format!("run.static-transform.{suffix}")),
        id(format!("transaction.static-transform.{suffix}")),
    );
    ResourceTransaction::begin(TransformDriver { runtime }, identity, admission)
        .unwrap()
        .reserve()
        .unwrap()
        .commit()
        .unwrap()
}

fn close_transform(resources: Arc<PlanRuntimeResources<TransformRuntime>>, expected: usize) {
    match PlanRuntimeResources::close(resources) {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => {
            assert_eq!(receipt.released_static_resources(), expected)
        }
        Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => {
            panic!("transform plan runtime close retained {strong_count} references")
        }
        Err(failure) => panic!(
            "transform plan runtime close failed: {:?}",
            failure.failure()
        ),
    }
}

fn close(resources: Arc<PlanRuntimeResources<TestRuntime>>) {
    close_with_expected_static_resources(resources, 2);
}

fn close_with_expected_static_resources(
    resources: Arc<PlanRuntimeResources<TestRuntime>>,
    expected_static_resources: usize,
) {
    match PlanRuntimeResources::close(resources) {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => assert_eq!(
            receipt.released_static_resources(),
            expected_static_resources
        ),
        Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => {
            panic!("plan runtime close retained {strong_count} references")
        }
        Err(failure) => panic!("plan runtime close failed: {:?}", failure.failure()),
    }
}

fn handoff(
    initialized: InitializedResourceTransaction<TestDriver>,
) -> Arc<PlanRuntimeResources<TestRuntime>> {
    match initialized.into_plan_runtime() {
        Ok(resources) => resources,
        Err(failure) => panic!("plan runtime handoff failed: {}", failure.error()),
    }
}

#[test]
fn static_initialization_uploads_schema_components_before_handoff() {
    let (resolved, plan, runtime, trace) = test_plan();
    let family = &resolved.parts().prepared_family;
    let execution_weight_schema = plan.payload().execution_weights().schema();
    assert_eq!(execution_weight_schema, family.weight_schema());
    let expected_uploaded_bytes = execution_weight_schema
        .components
        .iter()
        .map(WeightComponentSpec::physical_bytes)
        .try_fold(0_u64, |total, bytes| {
            total
                .checked_add(bytes?)
                .ok_or(VNextError::InvalidExecutionPlan {
                    reason: "test schema byte length overflow".to_owned(),
                })
        })
        .unwrap();
    let initialized = committed_transaction(&plan, Arc::clone(&runtime), "success")
        .initialize_static(
            family,
            &plan,
            &ZeroWeightSource,
            StaticInitializationPolicy::new(8, 2).unwrap(),
        )
        .unwrap();
    let receipt = initialized.receipt();
    assert_eq!(
        receipt.initialized_resource_count(),
        plan.payload().memory().static_allocations().len()
    );
    assert_eq!(
        receipt.uploaded_component_count(),
        execution_weight_schema.components.len()
    );
    assert_eq!(receipt.uploaded_bytes(), expected_uploaded_bytes);
    assert_eq!(receipt.upload_command_count(), 2);
    assert_eq!(receipt.imported_component_count(), 0);
    assert_eq!(receipt.transformed_component_count(), 0);
    assert_eq!(receipt.transformed_bytes(), 0);
    assert_eq!(receipt.transform_command_count(), 0);
    assert_eq!(receipt.device_transform_encode_duration_us(), 0);
    assert_eq!(receipt.device_import_duration_us(), 0);
    assert_eq!(receipt.import_seal_duration_us(), 0);
    assert!(receipt.total_duration_us() >= receipt.source_materialization_duration_us());
    assert!(receipt.total_duration_us() >= receipt.device_encode_duration_us());
    assert!(receipt.total_duration_us() >= receipt.submission_wait_duration_us());
    assert!(receipt.slowest_component_id().is_some());
    assert!(
        receipt.source_materialization_duration_us()
            >= receipt.slowest_component_materialization_duration_us()
    );
    assert_eq!(
        receipt.source_files(),
        &BTreeSet::from(["model.safetensors".to_owned()])
    );
    {
        let trace = trace.lock().unwrap();
        assert_eq!(
            trace.submit_calls as usize,
            receipt.submission_batch_count()
        );
        assert_eq!(trace.wait_fence_calls, trace.submit_calls);
        assert!(trace
            .submitted_command_counts
            .iter()
            .all(|count| *count <= 2));
        assert_eq!(trace.synchronize_calls, 0);
    }
    close(handoff(initialized));
}

#[test]
fn static_initialization_materializes_derived_components_through_trusted_plan_authority() {
    let (family, plan, runtime, trace) = derived_component_test_plan();
    assert_eq!(
        plan.payload().execution_weights().component_sources(),
        &BTreeMap::from([
            (
                id("weight.execution.left"),
                vec![id("weight.component.left"), id("weight.component.right")],
            ),
            (
                id("weight.execution.right"),
                vec![id("weight.component.left"), id("weight.component.right")],
            ),
        ])
    );
    let requested = Arc::new(Mutex::new(Vec::new()));
    let source = StrictSourceWeightSource {
        requested: Arc::clone(&requested),
    };
    let initialized = committed_transaction(&plan, Arc::clone(&runtime), "derived-materializer")
        .initialize_static(
            &family,
            &plan,
            &source,
            StaticInitializationPolicy::new(64, 8).unwrap(),
        )
        .unwrap();
    assert_eq!(
        *requested.lock().unwrap(),
        vec![id("weight.component.left"), id("weight.component.right")]
    );
    assert_eq!(
        trace.lock().unwrap().uploaded_payloads,
        vec![vec![0xa1; 8], vec![0xb2; 8]]
    );
    let expected_static_resources = plan.payload().memory().static_allocations().len();
    close_with_expected_static_resources(handoff(initialized), expected_static_resources);
}

#[test]
fn required_static_transform_fails_closed_when_runtime_is_unsupported() {
    let (family, plan, runtime, trace, host_materialization_calls) = static_transform_test_plan(1);
    trace.lock().unwrap().static_weight_import_enabled = true;
    let failed = committed_transaction(&plan, Arc::clone(&runtime), "transform-unsupported")
        .initialize_static(
            &family,
            &plan,
            &OrderedTransformSource,
            StaticInitializationPolicy::new(64, 8).unwrap(),
        );
    let failure = match failed {
        Ok(_) => panic!("required transform must fail when the runtime has no encoder"),
        Err(failure) => failure,
    };
    assert!(!failure.is_indeterminate());
    assert_eq!(failure.failure().code(), "static_weight_transform_encode");
    assert!(failure
        .failure()
        .message()
        .contains("does not support the required static weight transform"));
    assert_eq!(host_materialization_calls.load(Ordering::SeqCst), 0);
    let trace = trace.lock().unwrap();
    assert!(trace.uploaded_payloads.is_empty());
    assert_eq!(trace.static_weight_import_begin_calls, 0);
    assert_eq!(trace.imported_component_count, 0);
}

#[test]
fn required_static_transforms_preserve_sources_destinations_and_reuse_maximum_scratch() {
    let (family, plan, base_runtime, trace, host_materialization_calls) =
        static_transform_test_plan(2);
    trace.lock().unwrap().static_weight_import_enabled = true;
    let calls = Arc::new(Mutex::new(Vec::new()));
    let runtime = Arc::new(TransformRuntime {
        base: base_runtime,
        calls: Arc::clone(&calls),
    });
    let execution_weights = plan.payload().execution_weights();
    let scratch_allocations = plan
        .payload()
        .memory()
        .static_allocations()
        .iter()
        .filter(|allocation| matches!(allocation.kind(), AllocationKind::InitializationScratch))
        .collect::<Vec<_>>();
    let [scratch_allocation] = scratch_allocations.as_slice() else {
        panic!("multiple transforms must admit exactly one initialization scratch allocation")
    };
    assert_eq!(
        scratch_allocation.per_instance_bytes(),
        execution_weights
            .maximum_static_weight_transform_scratch_bytes()
            .unwrap()
    );
    assert_eq!(scratch_allocation.per_instance_bytes(), 256 * 128);
    assert_eq!(scratch_allocation.usage(), BufferUsage::Scratch);
    assert_eq!(scratch_allocation.element_type(), ElementType::U8);
    assert_eq!(
        Some(scratch_allocation.resource_id().clone()),
        execution_weights
            .static_weight_transform_scratch_resource_id()
            .unwrap()
    );

    let initialized =
        committed_transform_transaction(&plan, Arc::clone(&runtime), "transform-enabled")
            .initialize_static(
                &family,
                &plan,
                &OrderedTransformSource,
                StaticInitializationPolicy::new(64, 8).unwrap(),
            )
            .unwrap();
    let receipt = initialized.receipt();
    let expected_transformed_bytes = execution_weights
        .schema()
        .components
        .iter()
        .map(|component| component.physical_bytes().unwrap())
        .sum::<u64>();
    assert_eq!(receipt.transformed_component_count(), 4);
    assert_eq!(receipt.transformed_bytes(), expected_transformed_bytes);
    assert_eq!(receipt.transform_command_count(), 2);
    assert_eq!(receipt.uploaded_component_count(), 0);
    assert_eq!(receipt.uploaded_bytes(), 0);
    assert_eq!(receipt.upload_command_count(), 0);
    assert_eq!(receipt.imported_component_count(), 0);
    assert_eq!(receipt.imported_bytes(), 0);
    assert_eq!(host_materialization_calls.load(Ordering::SeqCst), 0);

    let calls = calls.lock().unwrap();
    assert_eq!(calls.len(), 2);
    for (call, transform) in calls
        .iter()
        .zip(execution_weights.static_weight_transforms())
    {
        assert_eq!(
            call.source_ids,
            transform
                .source_component_ids()
                .into_iter()
                .cloned()
                .collect::<Vec<_>>()
        );
        assert_eq!(
            call.destination_ids,
            transform
                .execution_component_ids()
                .into_iter()
                .cloned()
                .collect::<Vec<_>>()
        );
        assert_eq!(call.destination_buffers.len(), 2);
        assert_eq!(call.destination_offsets.len(), 2);
        assert!(call
            .destination_buffers
            .iter()
            .all(|descriptor| descriptor.usage == BufferUsage::Weights));
        let expected_segments = call
            .source_ids
            .iter()
            .map(|source_id| match source_id.as_str() {
                "weight.component.left" => vec![vec![0x11; 3], vec![0x12; 5]],
                "weight.component.right" => vec![vec![0x21; 3], vec![0x22; 5]],
                other => panic!("unexpected recorded source `{other}`"),
            })
            .collect::<Vec<_>>();
        assert_eq!(call.source_segment_bytes, expected_segments);
        assert_eq!(call.scratch.resource_id, *scratch_allocation.resource_id());
        assert_eq!(call.scratch.usage, BufferUsage::Scratch);
        assert_eq!(call.scratch.element_type, ElementType::U8);
        assert_eq!(call.scratch.size_bytes, scratch_allocation.size_bytes());
    }
    assert_eq!(calls[0].scratch, calls[1].scratch);
    drop(calls);

    let trace = trace.lock().unwrap();
    assert!(trace.uploaded_payloads.is_empty());
    assert_eq!(trace.static_weight_import_begin_calls, 0);
    assert_eq!(trace.imported_component_count, 0);
    drop(trace);
    let expected_static_resources = plan.payload().memory().static_allocations().len();
    let resources = match initialized.into_plan_runtime() {
        Ok(resources) => resources,
        Err(failure) => panic!("transform plan runtime handoff failed: {}", failure.error()),
    };
    close_transform(resources, expected_static_resources);
}

#[test]
fn static_initialization_seals_backend_weight_import_without_upload_commands() {
    let (resolved, plan, runtime, trace) = test_plan();
    trace.lock().unwrap().static_weight_import_enabled = true;
    let family = &resolved.parts().prepared_family;
    let execution_weight_schema = plan.payload().execution_weights().schema();
    let expected_imported_bytes = execution_weight_schema
        .components
        .iter()
        .map(|component| component.physical_bytes().unwrap())
        .try_fold(0_u64, u64::checked_add)
        .unwrap();

    let initialized = committed_transaction(&plan, Arc::clone(&runtime), "import")
        .initialize_static(
            family,
            &plan,
            &ZeroWeightSource,
            StaticInitializationPolicy::new(8, 2).unwrap(),
        )
        .unwrap();
    let receipt = initialized.receipt();
    assert_eq!(receipt.uploaded_component_count(), 0);
    assert_eq!(receipt.uploaded_bytes(), 0);
    assert_eq!(receipt.upload_command_count(), 0);
    assert_eq!(receipt.transformed_component_count(), 0);
    assert_eq!(receipt.transformed_bytes(), 0);
    assert_eq!(receipt.transform_command_count(), 0);
    assert_eq!(receipt.device_transform_encode_duration_us(), 0);
    assert_eq!(
        receipt.imported_component_count(),
        execution_weight_schema.components.len()
    );
    assert_eq!(receipt.imported_bytes(), expected_imported_bytes);
    assert_eq!(receipt.submission_batch_count(), 0);
    assert_eq!(receipt.submission_wait_duration_us(), 0);
    assert!(receipt.slowest_component_id().is_some());
    assert!(receipt.total_duration_us() >= receipt.device_import_duration_us());
    assert!(receipt.total_duration_us() >= receipt.import_seal_duration_us());
    {
        let trace = trace.lock().unwrap();
        assert_eq!(trace.static_weight_import_begin_calls, 1);
        assert_eq!(trace.static_weight_import_seal_calls, 1);
        assert_eq!(
            trace.imported_component_count,
            execution_weight_schema.components.len()
        );
        assert_eq!(trace.imported_bytes, expected_imported_bytes);
        assert_eq!(trace.submit_calls, 0);
    }
    close(handoff(initialized));
}

#[test]
fn fence_wait_panic_requires_stream_recovery_before_retry() {
    let (resolved, plan, runtime, trace) = test_plan();
    trace.lock().unwrap().fence_behavior = FenceBehavior::Panic;
    let failed = suppress_expected_panic_hook(|| {
        committed_transaction(&plan, Arc::clone(&runtime), "recovery").initialize_static(
            &resolved.parts().prepared_family,
            &plan,
            &ZeroWeightSource,
            StaticInitializationPolicy::new(8, 2).unwrap(),
        )
    });
    let failure = match failed {
        Ok(_) => panic!("fence wait panic must not permit runtime handoff"),
        Err(failure) => failure,
    };
    assert!(failure.is_indeterminate());
    assert_eq!(failure.failure().code(), "static_fence_wait_panic");
    assert_eq!(trace.lock().unwrap().synchronize_calls, 0);

    trace.lock().unwrap().fence_behavior = FenceBehavior::Succeeded;
    let committed = match failure.recover() {
        Ok(transaction) => transaction,
        Err(failure) => panic!("stream recovery failed: {failure}"),
    };
    assert_eq!(trace.lock().unwrap().synchronize_calls, 1);
    let initialized = committed
        .initialize_static(
            &resolved.parts().prepared_family,
            &plan,
            &ZeroWeightSource,
            StaticInitializationPolicy::new(8, 2).unwrap(),
        )
        .unwrap();
    close(handoff(initialized));
}

#[test]
fn error_classifier_panic_preserves_indeterminate_recovery_ownership() {
    let (resolved, plan, runtime, trace) = test_plan();
    {
        let mut trace = trace.lock().unwrap();
        trace.fence_behavior = FenceBehavior::Indeterminate;
        trace.describe_error_panics = true;
    }
    let failed = suppress_expected_panic_hook(|| {
        committed_transaction(&plan, Arc::clone(&runtime), "classifier-panic").initialize_static(
            &resolved.parts().prepared_family,
            &plan,
            &ZeroWeightSource,
            StaticInitializationPolicy::new(8, 2).unwrap(),
        )
    });
    let failure = match failed {
        Ok(_) => panic!("an indeterminate fence must not permit runtime handoff"),
        Err(failure) => failure,
    };
    assert!(failure.is_indeterminate());
    assert_eq!(failure.failure().code(), "static_fence_indeterminate");
    assert!(failure
        .failure()
        .message()
        .contains("classification panicked"));

    {
        let mut trace = trace.lock().unwrap();
        trace.fence_behavior = FenceBehavior::Succeeded;
        trace.describe_error_panics = false;
    }
    let committed = match failure.recover() {
        Ok(transaction) => transaction,
        Err(failure) => panic!("stream recovery failed: {failure}"),
    };
    assert_eq!(trace.lock().unwrap().synchronize_calls, 1);
    let initialized = committed
        .initialize_static(
            &resolved.parts().prepared_family,
            &plan,
            &ZeroWeightSource,
            StaticInitializationPolicy::new(8, 2).unwrap(),
        )
        .unwrap();
    close(handoff(initialized));
}

#[test]
fn dropped_indeterminate_failure_defers_backend_cleanup_until_maintenance() {
    let (resolved, plan, runtime, trace) = test_plan();
    trace.lock().unwrap().fence_behavior = FenceBehavior::Indeterminate;
    let status_before = static_initialization_cleanup_status();
    let failed = committed_transaction(&plan, Arc::clone(&runtime), "deferred-drop")
        .initialize_static(
            &resolved.parts().prepared_family,
            &plan,
            &ZeroWeightSource,
            StaticInitializationPolicy::new(8, 2).unwrap(),
        );
    let failure = match failed {
        Ok(_) => panic!("an indeterminate fence must retain cleanup ownership"),
        Err(failure) => failure,
    };
    assert!(failure.is_indeterminate());
    drop(failure);

    assert_eq!(trace.lock().unwrap().synchronize_calls, 0);
    let status_deferred = static_initialization_cleanup_status();
    assert_eq!(
        status_deferred.submitted_total(),
        status_before.submitted_total() + 1
    );
    assert_eq!(status_deferred.pending(), status_before.pending() + 1);

    trace.lock().unwrap().fence_behavior = FenceBehavior::Succeeded;
    let maintenance = maintain_static_initialization_cleanups(1).unwrap();
    assert_eq!(maintenance.attempted(), 1);
    assert_eq!(maintenance.completed(), 1);
    assert_eq!(
        maintenance.status_after().pending(),
        status_before.pending()
    );
    assert_eq!(trace.lock().unwrap().synchronize_calls, 1);
}
