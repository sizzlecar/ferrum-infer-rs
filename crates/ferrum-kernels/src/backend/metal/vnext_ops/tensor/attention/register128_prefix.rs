//! Managed QK -> online-softmax state diagnostic, not an Attention replacement.
//! Its output/register liveness/IO differ from full attention; subtracting the
//! durations does not measure pure PV time or pure GPU timestamps.
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
use std::time::Instant;

#[path = "register128_prefix/family.rs"]
mod family;
#[path = "register128_prefix/oracle.rs"]
mod oracle;

const OP: &str = "operation.diagnostic.metal.qk_softmax_state";
const CAPABILITY: &str = "capability.diagnostic.metal.qk_softmax_state";
const PROFILE: &str = "metal.diagnostic.qk_softmax_state.f32.v1";
const QUERY: &str = "value.prefix.query";
const KEY: &str = "value.prefix.key";
const OUTPUT: &str = "value.prefix.row_state";
const DEVICE_BYTES: u64 = 512 << 20;
const SOURCE: &str = concat!(
    include_str!("../bf16.metal"),
    "\n",
    include_str!("../attention.metal"),
    "\n",
    include_str!("register128_calibration.metal"),
    "\n",
    include_str!("register128.metal"),
    "\n",
    include_str!("register128_prefix.metal")
);

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
fn resolved(spec: ProgramTensorSpec) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(spec.dimensions, spec.element_type, spec.layout).unwrap()
}

struct Contract(OperationDescriptor);
impl Contract {
    fn new(config: family::Config) -> Self {
        let tensor = |spec: ProgramTensorSpec, access| {
            TensorContract::new(
                spec.dimensions
                    .into_iter()
                    .map(DimensionConstraint::Exact)
                    .collect(),
                BTreeSet::from([spec.element_type]),
                vec![LayoutConstraint::Contiguous],
                access,
                AliasPolicy::NoAlias,
            )
            .unwrap()
        };
        let descriptor = OperationDescriptor {
            id: id(OP),
            version: ContractVersion::new(1, 0),
            inputs: vec![
                tensor(config.input_spec(), TensorAccess::Read),
                tensor(config.input_spec(), TensorAccess::Read),
            ],
            outputs: vec![tensor(config.output_spec(), TensorAccess::Write)],
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
            return Err(invalid("prefix diagnostic signature differs"));
        }
        Ok(())
    }
}

struct Pipelines {
    prefix: ComputePipelineState,
    // PSO drops before its admitted native owner.
    _retention: MetalTensorProviderRetention,
}
struct Provider {
    descriptor: OperationProviderDescriptor,
    config: family::Config,
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
            "prefix requires fixed tensor work".into(),
        ))
    }
    fn encode_tensor_selected(
        &self,
        invocation: TensorOperationInvocation<'_, MetalDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<MetalDeviceCommand>>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        let encode = || -> Result<MetalDeviceCommand, String> {
            let [participant] = invocation.participants() else {
                return Err("prefix requires one participant".into());
            };
            if invocation.operation().id.as_str() != OP
                || !matches!(participant.work(), NodeWorkContract::Fixed)
                || !invocation.resource_work_shape().resource_work().is_tensor()
            {
                return Err("prefix operation/work mismatch".into());
            }
            let c = self.config;
            let input_bytes = resolved(c.input_spec())
                .minimum_storage_bytes()
                .map_err(|e| e.to_string())?;
            let output_bytes = resolved(c.output_spec())
                .minimum_storage_bytes()
                .map_err(|e| e.to_string())?;
            let q = tensor_port_region(
                participant,
                ResolvedValueRole::Input,
                0,
                ElementType::Bf16,
                input_bytes,
            )?;
            let k = tensor_port_region(
                participant,
                ResolvedValueRole::Input,
                1,
                ElementType::Bf16,
                input_bytes,
            )?;
            let output = tensor_port_region(
                participant,
                ResolvedValueRole::Output,
                0,
                ElementType::F32,
                output_bytes,
            )?;
            disjoint(&q, &k)?;
            disjoint(&q, &output)?;
            disjoint(&k, &output)?;
            let params = RegisterParams::new(
                c.batch * c.heads,
                c.sequence,
                c.sequence,
                c.heads,
                c.heads,
                0.125,
            )?;
            // Typed diagnostic supports exactly all-valid/full-bidirectional
            // Self attention. No public Attention contract is repurposed.
            let mut mask = super::super::NativeMask {
                isolated_segments: 0,
                segment_count: 1,
                segment_ends: [0; 32],
                bidirectional_mask: 1,
                key_run_counts: [0; 16],
                key_run_ends: [0; 512],
                key_valid_masks: [0; 16],
                first_valid_keys: [0; 16],
            };
            mask.segment_ends[0] = c.sequence as u32;
            for batch in 0..c.batch as usize {
                mask.key_run_counts[batch] = 1;
                mask.key_run_ends[batch * 32] = c.sequence as u32;
                mask.key_valid_masks[batch] = 1;
            }
            let operations = c
                .batch
                .checked_mul(c.heads)
                .and_then(|n| n.checked_mul(c.sequence))
                .and_then(|n| n.checked_mul(c.sequence))
                .and_then(|n| n.checked_mul(2 * 128 + 5))
                .ok_or("prefix logical work overflows")?;
            let pipelines = Arc::clone(&self.pipelines);
            MetalDeviceCommand::tensor_operation(
                OP,
                vec![q, k, output],
                DeviceCommandLogicalWork::tensor(operations, 0).map_err(|e| e.to_string())?,
                move |encoder, regions| {
                    let compute = encoder.compute_encoder();
                    compute.set_compute_pipeline_state(&pipelines.prefix);
                    set_region(compute, 0, &regions[0]);
                    set_region(compute, 1, &regions[1]);
                    set_region(compute, 3, &regions[2]);
                    set_params(compute, 4, &params);
                    set_params(compute, 5, &mask);
                    compute.set_threadgroup_memory_length(0, SHARED_BYTES);
                    compute.dispatch_thread_groups(
                        MTLSize::new(params.grid_groups, 1, 1),
                        MTLSize::new(128, 1, 1),
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

fn run_case(
    runtime: &Arc<MetalDeviceRuntime>,
    pipelines: &Arc<Pipelines>,
    host: &HostMemoryAccount,
    fingerprint: &str,
    config: family::Config,
    query_phase: u64,
    key_phase: u64,
    timed: bool,
) -> Value {
    let config = config.validate().unwrap();
    let contract = Contract::new(config);
    let descriptor = provider_descriptor(
        runtime,
        &contract,
        "provider.metal.qk_prefix",
        CAPABILITY,
        "resource-estimator.metal.qk_prefix",
        contiguous_bindings(2),
        &[],
        &[],
        fingerprint.to_owned(),
    )
    .unwrap()
    .with_fixed_tensor_work(true);
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(contract)],
        vec![Box::new(Provider {
            descriptor,
            config,
            pipelines: Arc::clone(pipelines),
        })],
    )
    .unwrap();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.metal.qk_prefix"),
        ContractVersion::new(1, 0),
        fingerprint.to_owned(),
        runtime.descriptor().id.clone(),
        runtime.descriptor().capabilities.clone(),
    )
    .unwrap();
    let catalog = registry
        .capability_catalog(runtime.descriptor().clone(), vec![engine])
        .unwrap();
    let prepared = TypedFamilyRegistration::new(family::Family)
        .prepare_with_profile(&serde_json::to_value(config).unwrap(), &id(PROFILE))
        .unwrap();
    let mut options = ProgramPlanCompileOptions::new(BTreeMap::from([
        (id(QUERY), config.input_spec()),
        (id(KEY), config.input_spec()),
    ]))
    .unwrap();
    options
        .insert_tensor_spec(id(OUTPUT), config.output_spec())
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
    let mut session = component
        .initialize(
            runtime.clone(),
            host.clone(),
            ComponentWeightSource::None,
            ComponentInitializationConfig {
                request_id: id("request.qk_prefix.init"),
                run_id: id("run.qk_prefix.init"),
                transaction_id: id("transaction.qk_prefix.init"),
                static_policy: StaticInitializationPolicy::new(1 << 20, 16).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    let input_bytes = resolved(config.input_spec())
        .minimum_storage_bytes()
        .unwrap();
    let input_allocation_bytes = usize::try_from(input_bytes).unwrap();
    let inputs: Vec<_> = [(QUERY, query_phase), (KEY, key_phase)]
        .into_iter()
        .map(|(name, phase)| {
            let mut bytes = host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, input_allocation_bytes)
                .unwrap();
            for (i, word) in bytes.as_mut().chunks_exact_mut(2).enumerate() {
                let value = oracle::value(i as u64, phase) as f32;
                // Fixture values are exactly representable; no rounding is hidden.
                assert_eq!(value.to_bits() & 0xffff, 0);
                word.copy_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
            }
            ManagedHostTensor::new(id(name), resolved(config.input_spec()), bytes.freeze()).unwrap()
        })
        .collect();
    // Metal no-copy backing is page-aligned; the real claim can exceed the
    // logical payload for the small tail fixture.
    let input_committed_bytes = host.usage().unwrap().committed_bytes();
    let reference = oracle::Reference::new(query_phase, key_phase);
    let mut samples = Vec::new();
    let mut errors = Vec::new();
    let repeats = if timed { 4 } else { 2 };
    for repetition in 0..repeats {
        let input = inputs.clone();
        let started = Instant::now();
        let output = session
            .execute(
                id(format!(
                    "run.qk_prefix.{}.{}.{repetition}",
                    config.sequence, config.heads
                )),
                id(format!(
                    "request.qk_prefix.{}.{}.{repetition}",
                    config.sequence, config.heads
                )),
                input,
            )
            .unwrap();
        samples.push(started.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].value_id().as_str(), OUTPUT);
        assert_eq!(output[0].tensor().element_type(), ElementType::F32);
        errors.push(reference.verify(config, output[0].bytes().bytes()));
        drop(output);
        assert_eq!(
            host.usage().unwrap().committed_bytes(),
            input_committed_bytes
        );
    }
    drop(inputs);
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|e| panic!("close: {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    drop((component, registry));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    serde_json::json!({"config":config,"query_phase":query_phase,"key_phase":key_phase,"timed":timed,
        "samples_ms":samples,"warmup_ms":if timed {Some(samples[0])}else{None},
        "measured_ms":if timed {&samples[1..]}else{&[]},"max_relative_sum_errors":errors,
        "sum_relative_bound":oracle::Reference::relative_bound(config.sequence),"row_max":"all output rows exact F32 bits versus independent FP64",
        "output_bytes":resolved(config.output_spec()).minimum_storage_bytes().unwrap(),"full_output_checks":repeats,
        "input_logical_bytes":input_bytes*2,"input_committed_bytes":input_committed_bytes,
        "host_committed_after_close":host.usage().unwrap().committed_bytes(),"native_memory":runtime.native_memory_snapshot()})
}

#[test]
#[ignore = "real managed Metal QK/softmax prefix diagnostic; requires exclusive GPU access"]
fn managed_metal_attention_qk_softmax_prefix_diagnostic() {
    metal::objc::rc::autoreleasepool(|| {
        let fingerprint = implementation_fingerprint(&[
            SOURCE.as_bytes(),
            include_str!("register128_prefix.rs").as_bytes(),
            include_str!("register128_prefix/family.rs").as_bytes(),
            include_str!("register128_prefix/oracle.rs").as_bytes(),
        ]);
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
                    opaque_memory_profile: "metal.qk_prefix.declared.v1".into(),
                },
            )
            .unwrap(),
        );
        let capacity = DeviceCapacityOwner::open(runtime.descriptor(), DEVICE_BYTES).unwrap();
        let retention = runtime.retain_tensor_provider().unwrap();
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = runtime
            .device()
            .new_library_with_source(SOURCE, &options)
            .unwrap();
        let constants = metal::FunctionConstantValues::new();
        constants.set_constant_value_at_index(
            (&1_u32 as *const u32).cast(),
            metal::MTLDataType::UInt,
            104,
        );
        constants.set_constant_value_at_index(
            (&0_u32 as *const u32).cast(),
            metal::MTLDataType::UInt,
            105,
        );
        let function = library
            .get_function("ferrum_tensor_attention_qk_prefix_f32", Some(constants))
            .unwrap();
        let prefix = runtime
            .device()
            .new_compute_pipeline_state_with_function(&function)
            .unwrap();
        drop((function, library));
        assert_eq!(prefix.thread_execution_width(), 32);
        assert!(prefix.max_total_threads_per_threadgroup() >= 128);
        assert!(prefix
            .static_threadgroup_memory_length()
            .checked_add(SHARED_BYTES)
            .is_some_and(|n| n <= runtime.device().max_threadgroup_memory_length()));
        let properties = serde_json::json!({"thread_execution_width":prefix.thread_execution_width(),
            "maximum_threads":prefix.max_total_threads_per_threadgroup(),"static_threadgroup_bytes":prefix.static_threadgroup_memory_length(),
            "dynamic_threadgroup_bytes":SHARED_BYTES});
        let pipelines = Arc::new(Pipelines {
            prefix,
            _retention: retention,
        });
        let host = runtime
            .tensor_host_account(
                HostMemoryLimits::new(
                    128 << 20,
                    HostMemoryClassLimits {
                        input_preprocess_bytes: 96 << 20,
                        output_bytes: 16 << 20,
                        transfer_staging_bytes: 0,
                        weight_backing_bytes: 0,
                    },
                )
                .unwrap(),
            )
            .unwrap();
        let small = run_case(
            &runtime,
            &pipelines,
            &host,
            &fingerprint,
            family::Config {
                batch: 2,
                sequence: 33,
                heads: 2,
                visibility: family::Visibility::AllValidBidirectional,
            },
            1,
            3,
            false,
        );
        let large = run_case(
            &runtime,
            &pipelines,
            &host,
            &fingerprint,
            family::Config {
                batch: 1,
                sequence: 4096,
                heads: 32,
                visibility: family::Visibility::AllValidBidirectional,
            },
            0,
            0,
            true,
        );
        drop(pipelines);
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
            "metal_attention_qk_prefix={}",
            serde_json::json!({"source_fingerprint":fingerprint,
            "runtime_fingerprint":runtime_fingerprint,"pso":properties,"cases":[small,large],"native_memory":native,
            "host_committed_after_close":host.usage().unwrap().committed_bytes(),"device_full_capacity_reclaimed":true,
            "timing_scope":"managed_execute_admission_encode_wait_readback; input preparation and independent FP64 full-output checks excluded; DeviceTimingOff",
            "limits":"QK/online-softmax prefix only; removing V/PV/O changes register liveness and IO. Source synchronization points retained, machine barriers not counted. Full minus prefix is not pure PV time or a GPU timestamp."})
        );
    });
}
