//! Managed attention stage/output-width diagnostics, not Attention replacements.
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

#[path = "register128_diagnostic/family.rs"]
mod family;
#[path = "register128_diagnostic/oracle.rs"]
mod oracle;
#[path = "register128_diagnostic/steel.rs"]
mod steel;

const OP: &str = "operation.diagnostic.metal.qk_softmax_state";
const CAPABILITY: &str = "capability.diagnostic.metal.attention_diagnostic";
const PROFILE: &str = "metal.diagnostic.qk_softmax_state.f32.v1";
const BF16_OP: &str = "operation.diagnostic.metal.sdpa_bf16_comparison";
const BF16_OUTPUT: &str = "value.diagnostic.sdpa_bf16";
const BF16_PROFILE: &str = "metal.diagnostic.production_sdpa.bf16.natural_exp.v1";
const NATIVE_BF16_PROFILE: &str =
    "metal.diagnostic.production_sdpa.native_bfloat_decode.finite_fixture.v1";
const STEEL_PROFILE: &str = "metal.diagnostic.official_steel_sdpa.bf16.log2e_fast_exp2_division.v1";
const VALUE_OP: &str = "operation.diagnostic.metal.attention_value_width";
const VALUE_PROFILE: &str = "metal.diagnostic.attention_value_width.f32.v1";
const VALUE_OUTPUT: &str = "value.diagnostic.attention_values";
const VALUE: &str = "value.diagnostic.values";
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
    include_str!("register128_prefix.metal"),
    "\n",
    include_str!("register128_diagnostic/value_width.metal")
);
// Only the input decode helper changes. The original rounded_bf16 helper
// retains its shift-based decode, and encode_bf16 is included verbatim.
// All subsequent kernel bodies are the same sources as SOURCE.
const NATIVE_BF16_SOURCE: &str = concat!(
    "#define decode_bf16 decode_bf16_shift\n",
    include_str!("../bf16.metal"),
    "\n#undef decode_bf16\ninline float decode_bf16(ushort bits) { return float(as_type<bfloat>(bits)); }\n",
    include_str!("../attention.metal"),
    "\n",
    include_str!("register128_calibration.metal"),
    "\n",
    include_str!("register128.metal"),
    "\n",
    include_str!("register128_prefix.metal"),
    "\n",
    include_str!("register128_diagnostic/value_width.metal")
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
            id: id(config.mode.operation()),
            version: ContractVersion::new(1, 0),
            inputs: config
                .mode
                .inputs()
                .iter()
                .map(|_| tensor(config.input_spec(), TensorAccess::Read))
                .collect(),
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
            return Err(invalid("attention diagnostic signature differs"));
        }
        Ok(())
    }
}

struct Pipelines {
    states: Vec<Pipeline>,
    // PSO drops before its admitted native owner.
    _retention: MetalTensorProviderRetention,
}
struct Pipeline {
    mode: family::Mode,
    aligned: [bool; 2],
    state: ComputePipelineState,
}
impl Pipelines {
    fn select(&self, c: family::Config) -> &ComputePipelineState {
        let aligned = if c.mode == family::Mode::SteelReference {
            [c.sequence.is_multiple_of(32), c.sequence.is_multiple_of(16)]
        } else {
            [false, false]
        };
        &self
            .states
            .iter()
            .find(|candidate| candidate.mode == c.mode && candidate.aligned == aligned)
            .expect("diagnostic PSO compiled")
            .state
    }
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
        estimate_without_workspace(&self.descriptor, &request, self.config.mode.operation())
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
            self.config.mode.operation(),
            "diagnostic requires fixed tensor work".into(),
        ))
    }
    fn encode_tensor_selected(
        &self,
        invocation: TensorOperationInvocation<'_, MetalDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<MetalDeviceCommand>>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        let encode = || -> Result<MetalDeviceCommand, String> {
            let [participant] = invocation.participants() else {
                return Err("attention diagnostic requires one participant".into());
            };
            if invocation.operation().id.as_str() != self.config.mode.operation()
                || !matches!(participant.work(), NodeWorkContract::Fixed)
                || !invocation.resource_work_shape().resource_work().is_tensor()
            {
                return Err("attention diagnostic operation/work mismatch".into());
            }
            let c = self.config;
            let input_bytes = resolved(c.input_spec())
                .minimum_storage_bytes()
                .map_err(|e| e.to_string())?;
            let output_bytes = resolved(c.output_spec())
                .minimum_storage_bytes()
                .map_err(|e| e.to_string())?;
            let mut regions = Vec::new();
            for port in 0..c.mode.inputs().len() {
                regions.push(tensor_port_region(
                    participant,
                    ResolvedValueRole::Input,
                    u32::try_from(port).unwrap(),
                    ElementType::Bf16,
                    input_bytes,
                )?);
            }
            regions.push(tensor_port_region(
                participant,
                ResolvedValueRole::Output,
                0,
                c.mode.output_element(),
                output_bytes,
            )?);
            for (i, region) in regions.iter().enumerate() {
                for other in &regions[i + 1..] {
                    disjoint(region, other)?;
                }
            }
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
                .and_then(|n| {
                    n.checked_mul(
                        2 * 128
                            + 5
                            + if c.mode.is_qk() {
                                0
                            } else {
                                2 * c.mode.columns()
                            },
                    )
                })
                .ok_or("attention diagnostic logical work overflows")?;
            let steel_params = (c.mode == family::Mode::SteelReference)
                .then(|| steel::Params::new(c))
                .transpose()
                .map_err(|e| e.to_string())?;
            let pipelines = Arc::clone(&self.pipelines);
            MetalDeviceCommand::tensor_operation(
                c.mode.operation(),
                regions,
                DeviceCommandLogicalWork::tensor(input_bytes / 2, operations)
                    .map_err(|e| e.to_string())?,
                move |encoder, regions| {
                    let compute = encoder.compute_encoder();
                    compute.set_compute_pipeline_state(pipelines.select(c));
                    for (slot, region) in regions[..c.mode.inputs().len()].iter().enumerate() {
                        set_region(compute, slot as u64, region);
                    }
                    set_region(compute, 3, &regions[c.mode.inputs().len()]);
                    compute.set_threadgroup_memory_length(0, c.mode.dynamic_bytes());
                    if let Some(steel_params) = &steel_params {
                        set_params(compute, 4, steel_params);
                        compute.dispatch_thread_groups(
                            MTLSize::new(c.sequence.div_ceil(32), c.heads, c.batch),
                            MTLSize::new(32, 4, 1),
                        );
                    } else {
                        set_params(compute, 4, &params);
                        set_params(compute, 5, &mask);
                        compute.dispatch_thread_groups(
                            MTLSize::new(params.grid_groups, 1, 1),
                            MTLSize::new(128, 1, 1),
                        );
                    }
                    Ok(())
                },
            )
            .map_err(|e| e.to_string())
        };
        encode()
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|reason| provider_failure(identity, self.config.mode.operation(), reason))
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
    trial: usize,
) -> Value {
    let config = config.validate().unwrap();
    let contract = Contract::new(config);
    let descriptor = provider_descriptor(
        runtime,
        &contract,
        "provider.metal.attention_diagnostic",
        CAPABILITY,
        "resource-estimator.metal.attention_diagnostic",
        contiguous_bindings(u32::try_from(config.mode.inputs().len()).unwrap()),
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
        id("provider.engine.metal.attention_diagnostic"),
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
        .prepare_with_profile(
            &serde_json::to_value(config).unwrap(),
            &id(config.mode.profile()),
        )
        .unwrap();
    let mut options = ProgramPlanCompileOptions::new(
        config
            .mode
            .inputs()
            .iter()
            .map(|name| (id(*name), config.input_spec()))
            .collect(),
    )
    .unwrap();
    options
        .insert_tensor_spec(id(config.mode.output()), config.output_spec())
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
                request_id: id(format!("request.attention_diagnostic.{trial}.init")),
                run_id: id(format!("run.attention_diagnostic.{trial}.init")),
                transaction_id: id(format!("transaction.attention_diagnostic.{trial}.init")),
                static_policy: StaticInitializationPolicy::new(1 << 20, 16).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    let input_bytes = resolved(config.input_spec())
        .minimum_storage_bytes()
        .unwrap();
    let input_allocation_bytes = usize::try_from(input_bytes).unwrap();
    let inputs: Vec<_> = config
        .mode
        .inputs()
        .iter()
        .map(|&name| {
            let mut bytes = host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, input_allocation_bytes)
                .unwrap();
            for (i, word) in bytes.as_mut().chunks_exact_mut(2).enumerate() {
                let i = i as u64;
                let value = if name == VALUE {
                    let column = i % 128;
                    let head = (i / 128) % config.heads;
                    let key = (i / (128 * config.heads)) % config.sequence;
                    let batch = i / (128 * config.heads * config.sequence);
                    match config.value_fixture {
                        family::ValueFixture::PositivePeriodic => {
                            oracle::positive_value(batch, key, head, column)
                        }
                        family::ValueFixture::ChannelConstant => {
                            oracle::constant_value(head, column)
                        }
                    }
                } else {
                    oracle::value(
                        i,
                        if name == QUERY {
                            query_phase
                        } else {
                            key_phase
                        },
                    )
                } as f32;
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
    let value_reference =
        (!config.mode.is_qk()).then(|| oracle::ValueReference::new(config, query_phase, key_phase));
    let mut samples = Vec::new();
    let mut errors = Vec::new();
    let repeats = if timed { 4 } else { 2 };
    for repetition in 0..repeats {
        let input = inputs.clone();
        let started = Instant::now();
        let output = session
            .execute(
                id(format!(
                    "run.attention_diagnostic.{trial}.{}.{}.{}.{repetition}",
                    config.mode.columns(),
                    config.sequence,
                    config.heads
                )),
                id(format!(
                    "request.attention_diagnostic.{trial}.{}.{}.{}.{repetition}",
                    config.mode.columns(),
                    config.sequence,
                    config.heads
                )),
                input,
            )
            .unwrap();
        samples.push(started.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].value_id().as_str(), config.mode.output());
        assert_eq!(
            output[0].tensor().element_type(),
            config.mode.output_element()
        );
        errors.push(if config.mode.is_bf16() {
            value_reference
                .as_ref()
                .unwrap()
                .verify_bf16(output[0].bytes().bytes())
        } else if let Some(value_reference) = &value_reference {
            value_reference.verify(output[0].bytes().bytes())
        } else {
            reference.verify(config, output[0].bytes().bytes())
        });
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
    let mut result = serde_json::json!({"config":config,"trial":trial,"query_phase":query_phase,"key_phase":key_phase,"timed":timed,
        "samples_ms":samples,"warmup_ms":if timed {Some(samples[0])}else{None},
        "measured_ms":if timed {&samples[1..]}else{&[]},
        "output_bytes":resolved(config.output_spec()).minimum_storage_bytes().unwrap(),"full_output_checks":repeats,
        "input_logical_bytes":input_bytes*config.mode.inputs().len() as u64,"input_committed_bytes":input_committed_bytes,
        "host_committed_after_close":host.usage().unwrap().committed_bytes(),"native_memory":runtime.native_memory_snapshot()});
    if config.mode.is_qk() {
        result["max_relative_sum_errors"] = serde_json::json!(errors);
        result["sum_relative_bound"] =
            serde_json::json!(oracle::Reference::relative_bound(config.sequence));
        result["row_max"] =
            serde_json::json!("all output rows exact F32 bits versus independent FP64");
    } else if config.mode.is_bf16() {
        result["max_relative_output_errors"] = serde_json::json!(errors);
        result["reference"] = if config.value_fixture == family::ValueFixture::ChannelConstant {
            serde_json::json!(
                "all BF16 output bits equal channel-constant V; Q/K are nonuniform periodic"
            )
        } else {
            serde_json::json!({"independent_fp64_positive_values":true,"relative_bound":oracle::ValueReference::bf16_relative_bound(config),"profile":config.mode.profile()})
        };
    } else {
        result["max_relative_output_errors"] = serde_json::json!(errors);
        result["output_relative_bound"] =
            serde_json::json!(oracle::ValueReference::relative_bound(config.sequence));
        result["reference"] = serde_json::json!("independent FP64 phase-count weighted positive V; every F32 output checked; no BF16 narrowing");
    }
    result
}

fn run_diagnostic(modes: &[family::Mode]) -> Value {
    metal::objc::rc::autoreleasepool(|| {
        let steel_source = modes
            .contains(&family::Mode::SteelReference)
            .then(steel::Source::from_locator)
            .transpose()
            .unwrap();
        let requested_language = if modes.iter().any(|mode| mode.is_bf16()) {
            "MSL3.1 explicit, both comparison modes"
        } else {
            "system_default"
        };
        let fingerprint = implementation_fingerprint(&[
            SOURCE.as_bytes(),
            NATIVE_BF16_SOURCE.as_bytes(),
            include_str!("register128_diagnostic.rs").as_bytes(),
            include_str!("register128_diagnostic/family.rs").as_bytes(),
            include_str!("register128_diagnostic/oracle.rs").as_bytes(),
            include_str!("register128_diagnostic/steel.rs").as_bytes(),
            requested_language.as_bytes(),
            steel_source
                .as_ref()
                .map(|s| s.fingerprint.as_bytes())
                .unwrap_or(&[]),
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
                    opaque_memory_profile: "metal.attention_diagnostic.declared.v1".into(),
                },
            )
            .unwrap(),
        );
        let capacity = DeviceCapacityOwner::open(runtime.descriptor(), DEVICE_BYTES).unwrap();
        let retention = runtime.retain_tensor_provider().unwrap();
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        if modes.iter().any(|mode| mode.is_bf16()) {
            options.set_language_version(metal::MTLLanguageVersion::V3_1);
        }
        let library = runtime
            .device()
            .new_library_with_source(SOURCE, &options)
            .unwrap();
        let native_bf16_library = modes
            .contains(&family::Mode::ProductionNativeBf16)
            .then(|| {
                runtime
                    .device()
                    .new_library_with_source(NATIVE_BF16_SOURCE, &options)
                    .unwrap()
            });
        // Separate library: official source never receives Ferrum BF16 helpers.
        let steel_library = steel_source.as_ref().map(|source| {
            runtime
                .device()
                .new_library_with_source(&source.source, &options)
                .unwrap()
        });
        let mut states = Vec::new();
        let mut properties = Vec::new();
        for &mode in modes {
            let alignments: &[[bool; 2]] = if mode == family::Mode::SteelReference {
                &[[false, false], [false, true], [true, false], [true, true]]
            } else {
                &[[false, false]]
            };
            for &aligned in alignments {
                let constants = metal::FunctionConstantValues::new();
                if mode == family::Mode::SteelReference {
                    for (index, value) in [
                        (200, aligned[0]),
                        (201, aligned[1]),
                        (300, false),
                        (301, false),
                        (302, false),
                    ] {
                        constants.set_constant_value_at_index(
                            (&value as *const bool).cast(),
                            metal::MTLDataType::Bool,
                            index,
                        );
                    }
                } else {
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
                }
                let selected_library = if mode == family::Mode::SteelReference {
                    steel_library.as_ref().unwrap()
                } else if mode == family::Mode::ProductionNativeBf16 {
                    native_bf16_library.as_ref().unwrap()
                } else {
                    &library
                };
                let function = selected_library
                    .get_function(mode.kernel(), Some(constants))
                    .unwrap();
                let state = runtime
                    .device()
                    .new_compute_pipeline_state_with_function(&function)
                    .unwrap();
                assert_eq!(state.thread_execution_width(), 32);
                assert!(state.max_total_threads_per_threadgroup() >= 128);
                assert!(state
                    .static_threadgroup_memory_length()
                    .checked_add(mode.dynamic_bytes())
                    .is_some_and(|n| n <= runtime.device().max_threadgroup_memory_length()));
                properties.push(serde_json::json!({"mode":mode,"kernel":mode.kernel(),"aligned_q":aligned[0],"aligned_k":aligned[1],
                    "requested_language":requested_language,"fast_math_enabled":false,
                    "input_decode":if mode == family::Mode::ProductionNativeBf16 {"float(as_type<bfloat>(ushort_bits))"} else if mode == family::Mode::SteelReference {"official native bfloat input"} else {"as_type<float>(uint(ushort_bits)<<16)"},
                    "thread_execution_width":state.thread_execution_width(),"maximum_threads":state.max_total_threads_per_threadgroup(),
                    "static_threadgroup_bytes":state.static_threadgroup_memory_length(),"dynamic_threadgroup_bytes":mode.dynamic_bytes()}));
                states.push(Pipeline {
                    mode,
                    aligned,
                    state,
                });
            }
        }
        drop((library, steel_library, native_bf16_library));
        let pipelines = Arc::new(Pipelines {
            states,
            _retention: retention,
        });
        let host = runtime
            .tensor_host_account(
                HostMemoryLimits::new(
                    256 << 20,
                    HostMemoryClassLimits {
                        input_preprocess_bytes: 128 << 20,
                        output_bytes: 96 << 20,
                        transfer_staging_bytes: 0,
                        weight_backing_bytes: 0,
                    },
                )
                .unwrap(),
            )
            .unwrap();
        // Logical large inputs total 96MiB; no-copy alignment costs additional
        // pages per owner. The explicit limits above admit these real claims
        // and a single 64MiB F32 output (64 columns use 32MiB).
        let mut cases = Vec::new();
        for (trial, &mode) in modes.iter().enumerate() {
            cases.push(run_case(
                &runtime,
                &pipelines,
                &host,
                &fingerprint,
                family::Config {
                    batch: 2,
                    sequence: 33,
                    heads: 2,
                    visibility: family::Visibility::AllValidBidirectional,
                    mode,
                    value_fixture: family::ValueFixture::PositivePeriodic,
                },
                1,
                3,
                false,
                trial,
            ));
        }
        // One process balances the two compared modes in A -> B -> B -> A order.
        // Each group owns fresh inputs/session and keeps warmup + all samples.
        // The single QK-state mode retains its original one-large-case scope.
        let large_modes: Vec<_> = modes
            .iter()
            .copied()
            .chain(modes.iter().rev().copied().filter(|mode| !mode.is_qk()))
            .collect();
        for (index, &mode) in large_modes.iter().enumerate() {
            cases.push(run_case(
                &runtime,
                &pipelines,
                &host,
                &fingerprint,
                family::Config {
                    batch: 1,
                    sequence: 4096,
                    heads: 32,
                    visibility: family::Visibility::AllValidBidirectional,
                    mode,
                    value_fixture: if mode.is_bf16() {
                        family::ValueFixture::ChannelConstant
                    } else {
                        family::ValueFixture::PositivePeriodic
                    },
                },
                0,
                0,
                true,
                modes.len() + index,
            ));
        }
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
        serde_json::json!({"source_fingerprint":fingerprint,
            "runtime_fingerprint":runtime_fingerprint,"pso":properties,"reference_source":steel_source.map(|source|source.report),"requested_language":requested_language,"large_case_order":large_modes,"cases":cases,"native_memory":native,
            "host_committed_after_close":host.usage().unwrap().committed_bytes(),"device_full_capacity_reclaimed":true,
            "timing_scope":"managed_execute_admission_encode_wait_readback; input preparation and independent full-output checks excluded; DeviceTimingOff",
            "bf16_comparison":if modes.iter().any(|mode|mode.is_bf16()) {serde_json::json!({"both_arms":"BF16 contiguous BSHD inputs and output, explicit MSL3.1 fast_math=false", "baseline":"production shader source at explicit MSL3.1; historical production measurements used system-default language and a different binary", "reference":if modes.contains(&family::Mode::SteelReference) {"official Steel native bfloat, F32 fragments, original log2e/fast::exp2/division; independent numerical profile, not Ferrum IEEE equivalence"} else {"same production kernel bodies, only decode_bf16 uses float(as_type<bfloat>(bits)); original encode and rounded_bf16 unchanged; finite fixture qualification only"}, "r263_difference":if modes.contains(&family::Mode::SteelReference) {Some("R263 MLX API exposed BHSD output and copied to BSHD; this reference writes BSHD directly")} else {None}, "qualification":"small positive nonconstant V, independent FP64; large channel-constant V, every BF16 output bit checked"})} else {serde_json::Value::Null},
            "limits":if modes.iter().any(|mode|mode.is_bf16()) {"Test-only self attention comparison with all-valid keys, qualified only for the declared finite fixtures, not full production masked/nonfinite/subnormal semantics. Both arms request MSL3.1; neither timing is a pure GPU timestamp, nor does it reproduce the earlier MLX API or system-default production binary. No machine shader/register/spill observation."} else {"Test-only QK-state or PV/output-width diagnostics. F32 output IO and register liveness differ from production BF16. Both value widths retain all D128 K/V loading, IEEE guards for their computed columns and source synchronization points; the positive fixture keeps range guards inactive. Widths need not choose identical overflow fallback for arbitrary extreme values; no machine barrier/register/spill observation. Timing differences are not pure PV or GPU timestamps."}})
    })
}

#[test]
#[ignore = "real managed Metal QK/softmax prefix diagnostic; requires exclusive GPU access"]
fn managed_metal_attention_qk_softmax_prefix_diagnostic() {
    eprintln!(
        "metal_attention_qk_prefix={}",
        run_diagnostic(&[family::Mode::QkSoftmaxState])
    );
}

#[test]
#[ignore = "real managed Metal F32 output-width diagnostic; requires exclusive GPU access"]
fn managed_metal_attention_value_width_diagnostic() {
    eprintln!(
        "metal_attention_value_width={}",
        run_diagnostic(&[family::Mode::ValueWidth64, family::Mode::ValueWidth128])
    );
}

#[test]
#[ignore = "official Steel source versus production-source BF16, explicit manifest and exclusive GPU required"]
fn managed_metal_official_steel_reference_diagnostic() {
    eprintln!(
        "metal_attention_steel_reference={}",
        run_diagnostic(&[family::Mode::ProductionBf16, family::Mode::SteelReference])
    );
}

#[test]
#[ignore = "production-source BF16 decode comparison, explicit MSL3.1; requires exclusive GPU access"]
fn managed_metal_native_bf16_decode_diagnostic() {
    eprintln!(
        "metal_attention_native_bf16_decode={}",
        run_diagnostic(&[
            family::Mode::ProductionBf16,
            family::Mode::ProductionNativeBf16
        ])
    );
}
