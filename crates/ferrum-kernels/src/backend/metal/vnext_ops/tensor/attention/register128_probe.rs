//! Independent diagnostic semantics: F32 request tag -> U32 SIMD coordinates.
//! This observes the shared calibration helper, not register pressure, spills,
//! occupancy, or branch counts in the complete attention shader.
use super::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_runtime::*;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[path = "register128_probe/family.rs"]
mod family;

const OP: &str = "operation.diagnostic.metal.simd_coordinates";
const CAPABILITY: &str = "capability.diagnostic.metal.simd_coordinates";
const PROFILE: &str = "metal.simd_coordinates.u32.v1";
const INPUT: &str = "value.probe.tag";
const OUTPUT: &str = "value.probe.coordinates";
const TAG_ECHO: &str = "value.probe.tag_echo";
const LANES: usize = 128;
const FIELDS: usize = 13;
const OUTPUT_BYTES: u64 = (LANES * FIELDS * 4) as u64;
const DEVICE_BYTES: u64 = 64 << 20;

fn id<T: TryFrom<String>>(value: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(value.into()).unwrap()
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
fn input_spec() -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: vec![1],
        element_type: ElementType::F32,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn output_spec() -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: vec![LANES as u64, FIELDS as u64],
        element_type: ElementType::U32,
        layout: ResolvedTensorLayout::Contiguous,
    }
}

struct Contract(OperationDescriptor);
impl Contract {
    fn new() -> Self {
        let tensor = |dimensions: &[u64], element, access| {
            TensorContract::new(
                dimensions
                    .iter()
                    .copied()
                    .map(DimensionConstraint::Exact)
                    .collect(),
                BTreeSet::from([element]),
                vec![LayoutConstraint::Contiguous],
                access,
                AliasPolicy::NoAlias,
            )
            .unwrap()
        };
        let descriptor = OperationDescriptor {
            id: id(OP),
            version: ContractVersion::new(1, 0),
            inputs: vec![tensor(&[1], ElementType::F32, TensorAccess::Read)],
            outputs: vec![
                tensor(
                    &[LANES as u64, FIELDS as u64],
                    ElementType::U32,
                    TensorAccess::Write,
                ),
                tensor(&[1], ElementType::F32, TensorAccess::Write),
            ],
            attributes: AttributeSchema::empty(),
            resources: ResourceRequirements {
                minimum_value_alignment_bytes: 16,
                scratch: ResourcePresenceRequirement::Forbidden,
                binding: ResourcePresenceRequirement::Forbidden,
                persistent: ResourcePresenceRequirement::Forbidden,
            },
            oracle: OracleSpec::OperationDefined,
            provider: ProviderRequirement {
                minimum_version: ContractVersion::new(1, 0),
                required_capabilities: BTreeSet::from([id(CAPABILITY)]),
            },
            profile_phase: ProfilePhase::Forward,
        };
        descriptor.validate().unwrap();
        Self(descriptor)
    }
}
impl OperationContract for Contract {
    fn descriptor(&self) -> &OperationDescriptor {
        &self.0
    }
    fn validate_signature(
        &self,
        inputs: &[TensorContract],
        outputs: &[TensorContract],
    ) -> Result<(), VNextError> {
        if inputs != self.0.inputs || outputs != self.0.outputs {
            return Err(invalid("coordinate probe signature differs"));
        }
        Ok(())
    }
}

struct Pipelines {
    probe: ComputePipelineState,
    production: Option<ComputePipelineState>,
    // Drop the actual PSOs before their admitted native ownership.
    _retention: MetalTensorProviderRetention,
}
struct Provider {
    descriptor: OperationProviderDescriptor,
    pipelines: Arc<Pipelines>,
}
impl OperationResourceEstimator for Provider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }
    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        estimate_without_workspace(&self.descriptor, &request, OP)
    }
}
impl OperationProvider<MetalDeviceRuntime> for Provider {
    fn reusable_execution_topology(
        &self,
        _: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        Ok(ReusableExecutionTopology::EagerBoundary)
    }
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<MetalDeviceCommand>, OperationFailure> {
        Err(provider_failure(
            invocation.participants()[0].identity().clone(),
            OP,
            "coordinate probe requires fixed tensor work".into(),
        ))
    }
    fn encode_tensor_selected(
        &self,
        invocation: TensorOperationInvocation<'_, MetalDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<MetalDeviceCommand>>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        let encode = || -> Result<MetalDeviceCommand, String> {
            let [participant] = invocation.participants() else {
                return Err("coordinate probe requires one participant".into());
            };
            if invocation.operation().id.as_str() != OP
                || !matches!(participant.work(), NodeWorkContract::Fixed)
                || !invocation.resource_work_shape().resource_work().is_tensor()
            {
                return Err("coordinate probe operation/work mismatch".into());
            }
            let input = tensor_port_region(
                participant,
                ResolvedValueRole::Input,
                0,
                ElementType::F32,
                4,
            )?;
            let output = tensor_port_region(
                participant,
                ResolvedValueRole::Output,
                0,
                ElementType::U32,
                OUTPUT_BYTES,
            )?;
            let tag_echo = tensor_port_region(
                participant,
                ResolvedValueRole::Output,
                1,
                ElementType::F32,
                4,
            )?;
            disjoint(&input, &output)?;
            disjoint(&input, &tag_echo)?;
            disjoint(&output, &tag_echo)?;
            let pipelines = Arc::clone(&self.pipelines);
            MetalDeviceCommand::tensor_operation(
                OP,
                vec![input, output, tag_echo],
                DeviceCommandLogicalWork::tensor((LANES * FIELDS + 1) as u64, 0)
                    .map_err(|e| e.to_string())?,
                move |encoder, regions| {
                    let compute = encoder.compute_encoder();
                    compute.set_compute_pipeline_state(&pipelines.probe);
                    set_region(compute, 0, &regions[0]);
                    set_region(compute, 1, &regions[1]);
                    set_region(compute, 2, &regions[2]);
                    compute.set_threadgroup_memory_length(0, SHARED_BYTES);
                    compute.dispatch_thread_groups(
                        MTLSize::new(1, 1, 1),
                        MTLSize::new(LANES as u64, 1, 1),
                    );
                    Ok(())
                },
            )
            .map_err(|e| e.to_string())
        };
        encode()
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|reason| provider_failure(identity, OP, reason))
    }
}

fn pipeline_properties(pso: &ComputePipelineState) -> Value {
    serde_json::json!({"thread_execution_width": pso.thread_execution_width(), "maximum_threads": pso.max_total_threads_per_threadgroup(), "static_threadgroup_bytes": pso.static_threadgroup_memory_length(), "dynamic_threadgroup_bytes": SHARED_BYTES})
}

// Reconstruct every relation from observed coordinates; no Apple lane layout
// or requirement that either fast path be available is built into the oracle.
fn verify_mapping(words: &[u32], tag: u32) -> Vec<Vec<u32>> {
    assert_eq!(words.len(), LANES * FIELDS);
    let rows: Vec<_> = words.chunks_exact(FIELDS).map(|row| row.to_vec()).collect();
    for (tid, row) in rows.iter().enumerate() {
        assert_eq!(row[0], tag);
        assert_eq!(row[12], tid as u32);
        assert!(row[1] < 32 && row[2] < 4);
        assert!(row[3..7].iter().all(|&coordinate| coordinate < 8));
    }
    for sid in 0..4 {
        let lanes: BTreeMap<_, _> = rows
            .iter()
            .filter(|r| r[2] == sid)
            .map(|r| (r[1], r))
            .collect();
        assert_eq!(lanes.len(), 32);
        let coordinates: BTreeSet<_> = lanes
            .values()
            .flat_map(|r| [(r[3], r[4]), (r[5], r[6])])
            .collect();
        assert_eq!(
            coordinates.len(),
            64,
            "SIMD matrix elements must cover the 8x8 tile exactly"
        );
        let same_row = lanes.values().all(|r| r[3] == r[5]);
        let expected: Vec<_> = lanes
            .iter()
            .map(|(&lane, row)| {
                let peers: Vec<_> = lanes.iter().filter(|(_, peer)| peer[3] == row[3]).collect();
                let mask = peers
                    .iter()
                    .fold(0u32, |mask, (&peer, _)| mask | (1u32 << peer));
                let columns = peers.iter().fold(0u32, |mask, (_, peer)| {
                    mask | (1u32 << peer[4]) | (1u32 << peer[6])
                });
                assert_eq!(row[7], peers.len() as u32);
                assert_eq!(row[8], mask);
                assert_eq!(row[9], columns);
                (
                    same_row && peers.len() == 4 && columns == 255,
                    mask == [lane, lane ^ 1, lane ^ 8, lane ^ 9]
                        .into_iter()
                        .fold(0u32, |mask, peer| mask | (1u32 << peer)),
                )
            })
            .collect();
        let four = expected.iter().all(|e| e.0);
        let xor = four && expected.iter().all(|e| e.1);
        for row in lanes.values() {
            assert_eq!(row[10], u32::from(four));
            assert_eq!(row[11], u32::from(xor));
        }
    }
    rows
}

#[test]
#[ignore = "real managed Metal SIMD-coordinate diagnostic; requires exclusive GPU access"]
fn managed_metal_attention_coordinates_and_pso_resources() {
    metal::objc::rc::autoreleasepool(|| {
        let source = concat!(
            include_str!("../bf16.metal"),
            "\n#include <metal_simdgroup_matrix>\n",
            include_str!("register128_calibration.metal"),
            "\n",
            include_str!("register128_probe.metal")
        );
        let fingerprint =
            implementation_fingerprint(&[source.as_bytes(), FINGERPRINT_SOURCE.as_bytes()]);
        let mut device = metal_tensor_runtime_config(id("device.metal.0")).unwrap();
        device.capabilities.insert(id(CAPABILITY));
        device.runtime_implementation_fingerprint = fingerprint.clone();
        let runtime = Arc::new(
            MetalDeviceRuntime::new_tensor_eager(
                crate::backend::metal::vnext_runtime::MetalTensorRuntimeConfig {
                    device,
                    usable_memory_bytes: DEVICE_BYTES,
                    bootstrap_memory_bytes: 16 << 20,
                    stream_memory_bytes: 1 << 20,
                    maximum_streams: 2,
                    maximum_in_flight_per_stream: 1,
                    maximum_commands_per_submission: 8,
                    opaque_memory_profile: "metal.coordinate_probe.declared.v1".into(),
                },
            )
            .unwrap(),
        );
        let capacity = DeviceCapacityOwner::open(runtime.descriptor(), DEVICE_BYTES).unwrap();
        let retention = runtime.retain_tensor_provider().unwrap();
        let production = compile(&runtime).unwrap();
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = runtime
            .device()
            .new_library_with_source(source, &options)
            .unwrap();
        let function = library
            .get_function("ferrum_attention_coordinate_probe", None)
            .unwrap();
        let probe = runtime
            .device()
            .new_compute_pipeline_state_with_function(&function)
            .unwrap();
        drop((function, library));
        assert_eq!(
            probe.thread_execution_width(),
            32,
            "the diagnostic requires 32-wide SIMD support"
        );
        assert!(probe.max_total_threads_per_threadgroup() >= LANES as u64);
        assert!(probe
            .static_threadgroup_memory_length()
            .checked_add(SHARED_BYTES)
            .is_some_and(|bytes| bytes <= runtime.device().max_threadgroup_memory_length()));
        let pipelines = Arc::new(Pipelines {
            probe,
            production,
            _retention: retention,
        });
        let properties = serde_json::json!({"probe": pipeline_properties(&pipelines.probe), "production": pipelines.production.as_ref().map(pipeline_properties)});
        let contract = Contract::new();
        let mut bindings = contiguous_bindings(1);
        bindings.push(ProviderStorageBindingRequirement::new(
            ResolvedValueRole::Output,
            1,
            DynamicStorageRequirement::contiguous(),
        ));
        let descriptor = provider_descriptor(
            &runtime,
            &contract,
            "provider.metal.coordinate_probe",
            CAPABILITY,
            "resource-estimator.metal.coordinate_probe",
            bindings,
            &[],
            &[],
            fingerprint.clone(),
        )
        .unwrap()
        .with_fixed_tensor_work(true);
        let registry = OperationRuntimeRegistry::new(
            vec![Box::new(contract)],
            vec![Box::new(Provider {
                descriptor,
                pipelines,
            })],
        )
        .unwrap();
        let engine = EngineProviderDescriptor::new(
            id("provider.engine.metal.coordinate_probe"),
            ContractVersion::new(1, 0),
            fingerprint.clone(),
            runtime.descriptor().id.clone(),
            runtime.descriptor().capabilities.clone(),
        )
        .unwrap();
        let catalog = registry
            .capability_catalog(runtime.descriptor().clone(), vec![engine])
            .unwrap();
        let prepared = TypedFamilyRegistration::new(family::Family)
            .prepare_with_profile(&serde_json::json!({}), &id(PROFILE))
            .unwrap();
        let mut options =
            ProgramPlanCompileOptions::new(BTreeMap::from([(id(INPUT), input_spec())])).unwrap();
        options
            .insert_tensor_spec(id(OUTPUT), output_spec())
            .unwrap();
        options
            .insert_tensor_spec(id(TAG_ECHO), input_spec())
            .unwrap();
        let policy = TensorComponentPolicy::new(
            DEVICE_BYTES,
            runtime
                .descriptor()
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect(),
        )
        .unwrap();
        let component = CompiledTensorComponent::compile(
            prepared,
            &registry,
            &catalog,
            &policy,
            &WeightMaterializerRegistry::identity_only().unwrap(),
            &options,
        )
        .unwrap();
        let host = runtime
            .tensor_host_account(
                HostMemoryLimits::new(
                    1 << 20,
                    HostMemoryClassLimits {
                        input_preprocess_bytes: 1 << 19,
                        output_bytes: 1 << 19,
                        transfer_staging_bytes: 0,
                        weight_backing_bytes: 0,
                    },
                )
                .unwrap(),
            )
            .unwrap();
        let mut session = component
            .initialize(
                runtime.clone(),
                host.clone(),
                ComponentWeightSource::None,
                ComponentInitializationConfig {
                    request_id: id("request.coordinate_probe.init"),
                    run_id: id("run.coordinate_probe.init"),
                    transaction_id: id("transaction.coordinate_probe.init"),
                    static_policy: StaticInitializationPolicy::new(1 << 20, 16).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap();
        let mut observed = Vec::new();
        for (index, tag) in [1.0_f32, 2.0_f32].into_iter().enumerate() {
            let mut bytes = host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, 4)
                .unwrap();
            bytes.as_mut().copy_from_slice(&tag.to_bits().to_le_bytes());
            let input = ManagedHostTensor::new(
                id(INPUT),
                ResolvedTensorSpec::new(
                    vec![1],
                    ElementType::F32,
                    ResolvedTensorLayout::Contiguous,
                )
                .unwrap(),
                bytes.freeze(),
            )
            .unwrap();
            let output = session
                .execute(
                    id(format!("run.coordinate_probe.{index}")),
                    id(format!("request.coordinate_probe.{index}")),
                    vec![input],
                )
                .unwrap();
            assert_eq!(output.len(), 2);
            let tag_echo = output
                .iter()
                .find(|value| value.value_id().as_str() == TAG_ECHO)
                .unwrap();
            assert_eq!(tag_echo.tensor().element_type(), ElementType::F32);
            assert_eq!(tag_echo.bytes().bytes(), &tag.to_bits().to_le_bytes());
            let coordinates = output
                .iter()
                .find(|value| value.value_id().as_str() == OUTPUT)
                .unwrap();
            assert_eq!(coordinates.tensor().element_type(), ElementType::U32);
            let words: Vec<_> = coordinates
                .bytes()
                .bytes()
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            observed.push(verify_mapping(&words, tag.to_bits()));
            drop(output);
            assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        }
        assert!(matches!(
            session
                .close()
                .unwrap_or_else(|e| panic!("close: {:?}", e.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        drop((component, registry));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        // The provider lease is reusable only after all command/plan owners retire.
        drop(runtime.retain_tensor_provider().unwrap());
        let native = runtime.native_memory_snapshot();
        let runtime_fingerprint = runtime
            .descriptor()
            .runtime_implementation_fingerprint
            .clone();
        drop(runtime);
        let all = capacity
            .reserve(DeviceOpaqueMemoryKind::Bootstrap, DEVICE_BYTES)
            .unwrap();
        assert!(capacity.reserve(DeviceOpaqueMemoryKind::Stream, 1).is_err());
        drop(all);
        eprintln!(
            "metal_attention_coordinate_probe={}",
            serde_json::json!({
                "source_fingerprint": fingerprint, "runtime_fingerprint": runtime_fingerprint, "pso": properties, "columns": ["tag_bits", "lane", "simdgroup", "row0", "col0", "row1", "col1", "peer_count", "peer_mask", "column_mask", "four_peers", "xor_peers", "thread_in_group"],
                "observations": observed, "native_memory": native, "host_committed_after_close": host.usage().unwrap().committed_bytes(), "device_full_capacity_reclaimed": true,
                "scope": "shared coordinate calibration only; not full shader register pressure, spills, occupancy, branch counts or timing"
            })
        );
    });
}
