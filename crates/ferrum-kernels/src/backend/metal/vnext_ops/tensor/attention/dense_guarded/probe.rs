//! Test-only observation of real production dispatch flags, with admitted IO.
//! The fill and snapshot blits change this workload; this is not a benchmark.
use super::super::*;
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

#[path = "probe/family.rs"]
mod family;

const OP: &str = "operation.diagnostic.metal.dense_attention_flags";
const CAPABILITY: &str = "capability.diagnostic.metal.dense_attention_flags";
const PROFILE: &str = "metal.diagnostic.dense_attention_flags.bf16_u32.v1";
const INPUTS: [&str; 3] = [
    "value.dense_probe.q",
    "value.dense_probe.k",
    "value.dense_probe.v",
];
const OUTPUTS: [&str; 2] = ["value.dense_probe.y", "value.dense_probe.flags"];
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

#[derive(Clone, Copy, Debug)]
enum ProbeShape {
    AllValid,
    Holes,
    MixedEmpty,
}
impl ProbeShape {
    const BATCH: u64 = 2;
    const HEADS: u64 = 2;
    const SEQUENCE: u64 = 65;
    fn key_runs(self) -> Vec<Vec<(i64, bool)>> {
        match self {
            Self::AllValid => vec![vec![(65, true)]; 2],
            Self::Holes => vec![
                vec![(16, false), (15, true), (2, false), (15, true), (17, false)],
                vec![(17, true), (16, false), (32, true)],
            ],
            Self::MixedEmpty => vec![
                vec![(32, false), (16, true), (16, false), (1, true)],
                vec![(65, false)],
            ],
        }
    }
    fn key_valid(self, batch: u64, key: u64) -> bool {
        let mut end = 0;
        for (length, valid) in &self.key_runs()[batch as usize] {
            end += *length as u64;
            if key < end {
                return *valid;
            }
        }
        panic!("probe key outside physical tensor");
    }
    fn row_has_key(self, batch: u64, query: u64) -> bool {
        (0..Self::SEQUENCE).any(|key| self.key_valid(batch, key) && (query >= 17 || key <= query))
    }
    fn constant_value(self, index: u64) -> f32 {
        let channel = index % 128;
        let head = (index / 128) % Self::HEADS;
        let batch = index / (Self::SEQUENCE * Self::HEADS * 128);
        1.0 + ((batch * Self::HEADS + head + channel) % 8) as f32 / 8.0
    }
    fn tensor(self) -> ProgramTensorSpec {
        ProgramTensorSpec {
            dimensions: vec![Self::BATCH * Self::SEQUENCE, Self::HEADS * 128],
            element_type: ElementType::Bf16,
            layout: ResolvedTensorLayout::Contiguous,
        }
    }
    fn resolved(self) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(
            self.tensor().dimensions,
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap()
    }
    fn checked(self) -> StructuredAttentionBf16Geometry {
        let tensor = self.resolved();
        let mut attributes = BTreeMap::from([
            (
                id("sequence_length"),
                SemanticValue::Unsigned(Self::SEQUENCE),
            ),
            (id("head_dim"), SemanticValue::Unsigned(128)),
            (
                id("scale"),
                SemanticValue::Rational(CanonicalRational::new(1, 8).unwrap()),
            ),
        ]);
        let runs = self.key_runs();
        for (name, values) in [
            ("segment_lengths", vec![17, 48]),
            ("segment_bidirectional", vec![0, 1]),
            (
                "key_run_counts",
                runs.iter().map(|run| run.len() as i64).collect(),
            ),
            (
                "key_run_lengths",
                runs.iter().flatten().map(|run| run.0).collect(),
            ),
            (
                "key_run_valid",
                runs.iter().flatten().map(|run| i64::from(run.1)).collect(),
            ),
        ] {
            attributes.insert(id(name), SemanticValue::Integers(values));
        }
        StructuredAttentionBf16Geometry::new(&tensor, &tensor, &tensor, &tensor, &attributes)
            .unwrap()
    }
    fn mask(self) -> NativeMask {
        NativeMask::new(self.checked().mask(), Self::SEQUENCE as u32)
    }
    fn plan(self) -> Schedule {
        Schedule::new(GuardGeometry {
            batch_heads: Self::BATCH * Self::HEADS,
            query: Self::SEQUENCE,
            key: Self::SEQUENCE,
            query_heads: Self::HEADS,
            key_value_heads: Self::HEADS,
            head_dim: 128,
            scale: 0.125,
            all_keys_valid: matches!(self, Self::AllValid),
            dense_start: dense_start(&self.mask(), Self::SEQUENCE),
        })
        .unwrap()
        .expect("typed probe has a dense suffix")
    }
    fn flags(self) -> ProgramTensorSpec {
        ProgramTensorSpec {
            dimensions: vec![self.plan().strict.jobs],
            element_type: ElementType::U32,
            layout: ResolvedTensorLayout::Contiguous,
        }
    }
    fn tensor_bytes(self) -> u64 {
        Self::BATCH * Self::SEQUENCE * Self::HEADS * 128 * 2
    }
    fn value(self, case: Case, port: usize, index: u64) -> f32 {
        let channel = index % 128;
        let head = (index / 128) % Self::HEADS;
        let row = (index / (Self::HEADS * 128)) % Self::SEQUENCE;
        let batch = index / (Self::SEQUENCE * Self::HEADS * 128);
        if port > 0 && !self.key_valid(batch, row) {
            // A masked nonfinite payload must not reject an otherwise admitted
            // head. The actual flag assertion observes that qualification.
            return if port == 1 {
                f32::NAN
            } else {
                f32::NEG_INFINITY
            };
        }
        if port == 0 && !(0..Self::SEQUENCE).any(|key| self.key_valid(batch, key)) {
            return f32::NAN;
        }
        if batch == 0 && row == 33 {
            if case == Case::VisibleKeyNan && port == 1 && head == 0 && channel == 0 {
                return f32::NAN;
            }
            if case == Case::VisibleValueInf && port == 2 && head == 1 && channel == 127 {
                return f32::INFINITY;
            }
        }
        if port == 2 {
            // Constant across sequence: exact independent attention answer for
            // all finite Q/K, including strict prefix and huge finite scores.
            return self.constant_value(index);
        }
        if port == 0 && row == 32 && channel == 0 {
            if case == Case::AllReject {
                return 2_097_152.0;
            }
            if case == Case::OneHeadNan && batch == 1 && head == 0 {
                return f32::NAN;
            }
        }
        ((index % 7) as i32 - 3) as f32 / 2.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Case {
    Finite,
    AllReject,
    OneHeadNan,
    VisibleKeyNan,
    VisibleValueInf,
}

struct Contract(OperationDescriptor);
impl Contract {
    fn new(shape: ProbeShape) -> Self {
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
            inputs: (0..3)
                .map(|_| tensor(shape.tensor(), TensorAccess::Read))
                .collect(),
            outputs: vec![
                tensor(shape.tensor(), TensorAccess::Write),
                tensor(shape.flags(), TensorAccess::Write),
            ],
            attributes: AttributeSchema::empty(),
            resources: ResourceRequirements {
                minimum_value_alignment_bytes: 16,
                scratch: ResourcePresenceRequirement::Optional,
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
            return Err(invalid("dense flag probe signature differs"));
        }
        Ok(())
    }
}

struct ProbePipelines {
    actual: DensePipelines,
    // Native pipelines retire before their single admitted provider owner.
    _retention: MetalTensorProviderRetention,
}
struct Provider {
    descriptor: OperationProviderDescriptor,
    pipelines: Arc<ProbePipelines>,
    shape: ProbeShape,
}
impl OperationResourceEstimator for Provider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }
    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        if request.operation().id.as_str() != OP
            || request.operation().fingerprint()? != self.descriptor.operation_fingerprint()
        {
            return Err(invalid("dense flag probe received another operation"));
        }
        estimate_workspace(&self.descriptor, &request, self.shape.plan().flag_bytes())
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
            "dense flag probe requires fixed tensors".into(),
        ))
    }
    fn encode_tensor_selected(
        &self,
        invocation: TensorOperationInvocation<'_, MetalDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<MetalDeviceCommand>>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        let encode_command = || -> Result<MetalDeviceCommand, String> {
            let [participant] = invocation.participants() else {
                return Err("dense flag probe requires one participant".into());
            };
            if invocation.operation().id.as_str() != OP
                || !matches!(participant.work(), NodeWorkContract::Fixed)
                || !invocation.resource_work_shape().resource_work().is_tensor()
            {
                return Err("dense flag probe operation/work mismatch".into());
            }
            let plan = self.shape.plan();
            let mut regions = Vec::with_capacity(6);
            for ordinal in 0..3 {
                regions.push(tensor_port_region(
                    participant,
                    ResolvedValueRole::Input,
                    ordinal,
                    ElementType::Bf16,
                    self.shape.tensor_bytes(),
                )?);
            }
            regions.push(tensor_port_region(
                participant,
                ResolvedValueRole::Output,
                0,
                ElementType::Bf16,
                self.shape.tensor_bytes(),
            )?);
            for input in &regions[..3] {
                disjoint(input, &regions[3])?;
            }
            append_flags(participant, &plan, &mut regions)?;
            let snapshot = tensor_port_region(
                participant,
                ResolvedValueRole::Output,
                1,
                ElementType::U32,
                plan.strict.jobs * 4,
            )?;
            for region in &regions {
                disjoint(region, &snapshot)?;
            }
            regions.push(snapshot);
            let mask = self.shape.mask();
            let pipelines = Arc::clone(&self.pipelines);
            let operations = self.shape.checked().logical_arithmetic_operations();
            MetalDeviceCommand::tensor_operation(
                OP,
                regions,
                DeviceCommandLogicalWork::tensor(
                    self.shape.tensor_bytes() / 2 + plan.strict.jobs,
                    operations,
                )
                .map_err(|e| e.to_string())?,
                move |encoder, regions| {
                    let flags = &regions[4];
                    // Force nonzero stale flags in this very allocation before
                    // production prepass. No dependence on pool address reuse.
                    encoder.with_blit(|blit| {
                        blit.fill_buffer(
                            flags.buffer(),
                            metal::NSRange::new(flags.offset_bytes(), plan.flag_bytes()),
                            0xff,
                        )
                    });
                    // Exact production schedule/PSOs/dispatch composition, not a
                    // test reimplementation of the three kernels.
                    super::super::encode(
                        encoder,
                        &regions[..5],
                        &plan,
                        &mask,
                        &pipelines.actual,
                        false,
                    );
                    let snapshot = &regions[5];
                    encoder.with_blit(|blit| {
                        blit.copy_from_buffer(
                            flags.buffer(),
                            flags.offset_bytes(),
                            snapshot.buffer(),
                            snapshot.offset_bytes(),
                            plan.strict.jobs * 4,
                        )
                    });
                    Ok(())
                },
            )
            .map_err(|e| e.to_string())
        };
        encode_command()
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|reason| provider_failure(identity, OP, reason))
    }
}

#[test]
fn dense_flag_probe_declares_real_outputs_and_distinct_guard_cases() {
    let shape = ProbeShape::AllValid;
    Contract::new(shape);
    TypedFamilyRegistration::new(family::Family { shape })
        .prepare_with_profile(&serde_json::json!({}), &id(PROFILE))
        .unwrap();
    assert_eq!(shape.plan().strict.jobs, 12);
    for case in [Case::Finite, Case::AllReject, Case::OneHeadNan] {
        for batch in 0..ProbeShape::BATCH {
            for head in 0..ProbeShape::HEADS {
                let index = ((batch * ProbeShape::SEQUENCE + 32) * ProbeShape::HEADS + head) * 128;
                let q = shape.value(case, 0, index);
                assert_eq!(
                    !q.is_finite() || q.abs() > 1_048_576.0,
                    case == Case::AllReject
                        || (case == Case::OneHeadNan && batch == 1 && head == 0)
                );
            }
        }
    }
}

#[test]
fn masked_flag_probe_uses_validated_runs_and_poisoned_invisible_payloads() {
    for shape in [ProbeShape::Holes, ProbeShape::MixedEmpty] {
        Contract::new(shape);
        TypedFamilyRegistration::new(family::Family { shape })
            .prepare_with_profile(&serde_json::json!({}), &id(PROFILE))
            .unwrap();
        assert!(shape.plan().masked_keys);
        for batch in 0..ProbeShape::BATCH {
            for key in 0..ProbeShape::SEQUENCE {
                let index = (batch * ProbeShape::SEQUENCE + key) * ProbeShape::HEADS * 128;
                for port in [1, 2] {
                    assert_eq!(
                        shape.value(Case::Finite, port, index).is_finite(),
                        shape.key_valid(batch, key)
                    );
                }
            }
        }
        assert!(shape.key_valid(0, 33));
        assert!(!shape.row_has_key(0, 0));
    }
    assert!(!ProbeShape::MixedEmpty.row_has_key(1, 64));
    assert!(ProbeShape::Holes.row_has_key(1, 0));
}

#[test]
#[ignore = "real production dense PSO flags through managed scratch/outputs; requires exclusive Metal GPU"]
fn managed_metal_dense_attention_actual_fast_flags_and_rejection() {
    run_probe(ProbeShape::AllValid);
}

#[test]
#[ignore = "masked production three-phase flags through managed scratch/outputs; requires exclusive Metal GPU"]
fn managed_metal_masked_dense_attention_actual_fast_flags_and_rejection() {
    for shape in [ProbeShape::Holes, ProbeShape::MixedEmpty] {
        run_probe(shape);
    }
}

fn run_probe(shape: ProbeShape) {
    metal::objc::rc::autoreleasepool(|| {
        let shape_identity = format!("{shape:?}");
        let fingerprint = implementation_fingerprint(&[
            FINGERPRINT_SOURCE.as_bytes(),
            register128::FINGERPRINT_SOURCE.as_bytes(),
            include_bytes!("probe.rs"),
            include_bytes!("probe/family.rs"),
            shape_identity.as_bytes(),
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
                    opaque_memory_profile: "metal.dense_flags_probe.declared.v1".into(),
                },
            )
            .unwrap(),
        );
        let capacity = DeviceCapacityOwner::open(runtime.descriptor(), DEVICE_BYTES).unwrap();
        let retention = runtime.retain_tensor_provider().unwrap();
        let actual = compile(&runtime)
            .expect("production dense pipelines must be available for this diagnostic");
        assert!(
            actual.supports(&shape.plan()),
            "selected production mask PSOs must be available"
        );
        let selected_fast = if shape.plan().masked_keys {
            &actual.masked.as_ref().expect("masked PSOs required").fast
        } else {
            &actual.fast
        };
        let pso = selected_fast.iter().map(|p| serde_json::json!({
            "simd_width":p.thread_execution_width(), "maximum_threads":p.max_total_threads_per_threadgroup(),
            "static_threadgroup_bytes":p.static_threadgroup_memory_length()})).collect::<Vec<_>>();
        let selected_strict = if shape.plan().masked_keys {
            &actual.masked.as_ref().expect("masked PSOs required").strict
        } else {
            &actual.strict
        };
        let strict_pso = selected_strict
            .iter()
            .zip(["Structured", "Cross"])
            .map(|(p, domain)| {
                serde_json::json!({
            "domain":domain, "simd_width":p.thread_execution_width(),
            "maximum_threads":p.max_total_threads_per_threadgroup(),
            "static_threadgroup_bytes":p.static_threadgroup_memory_length()})
            })
            .collect::<Vec<_>>();
        let pipelines = Arc::new(ProbePipelines {
            actual,
            _retention: retention,
        });
        let contract = Contract::new(shape);
        let mut bindings = contiguous_bindings(3);
        bindings.push(ProviderStorageBindingRequirement::new(
            ResolvedValueRole::Output,
            1,
            DynamicStorageRequirement::contiguous(),
        ));
        let descriptor = provider_descriptor(
            &runtime,
            &contract,
            "provider.metal.dense_flags_probe",
            CAPABILITY,
            "resource-estimator.metal.dense_flags_probe",
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
                shape,
            })],
        )
        .unwrap();
        let engine = EngineProviderDescriptor::new(
            id("provider.engine.metal.dense_flags_probe"),
            ContractVersion::new(1, 0),
            fingerprint.clone(),
            runtime.descriptor().id.clone(),
            runtime.descriptor().capabilities.clone(),
        )
        .unwrap();
        let catalog = registry
            .capability_catalog(runtime.descriptor().clone(), vec![engine])
            .unwrap();
        let prepared = TypedFamilyRegistration::new(family::Family { shape })
            .prepare_with_profile(&serde_json::json!({}), &id(PROFILE))
            .unwrap();
        let mut options = ProgramPlanCompileOptions::new(
            INPUTS
                .into_iter()
                .map(|name| (id(name), shape.tensor()))
                .collect(),
        )
        .unwrap();
        options
            .insert_tensor_spec(id(OUTPUTS[0]), shape.tensor())
            .unwrap();
        options
            .insert_tensor_spec(id(OUTPUTS[1]), shape.flags())
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
                    request_id: id("request.dense_flags.init"),
                    run_id: id("run.dense_flags.init"),
                    transaction_id: id("transaction.dense_flags.init"),
                    static_policy: StaticInitializationPolicy::new(1 << 20, 16).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap();
        let mut observations = Vec::new();
        for (request, case) in [
            Case::Finite,
            Case::AllReject,
            Case::OneHeadNan,
            Case::Finite,
            Case::VisibleKeyNan,
            Case::Finite,
            Case::VisibleValueInf,
            Case::Finite,
        ]
        .into_iter()
        .enumerate()
        {
            let inputs = INPUTS
                .into_iter()
                .enumerate()
                .map(|(port, name)| {
                    let mut bytes = host
                        .allocate_zeroed(
                            HostMemoryClass::InputPreprocess,
                            shape.tensor_bytes() as usize,
                        )
                        .unwrap();
                    for (index, chunk) in bytes.as_mut().chunks_exact_mut(2).enumerate() {
                        chunk.copy_from_slice(
                            &half::bf16::from_f32(shape.value(case, port, index as u64))
                                .to_bits()
                                .to_le_bytes(),
                        );
                    }
                    ManagedHostTensor::new(id(name), shape.resolved(), bytes.freeze()).unwrap()
                })
                .collect();
            let outputs = session
                .execute(
                    id(format!("run.dense_flags.{request}")),
                    id(format!("request.dense_flags.{request}")),
                    inputs,
                )
                .unwrap();
            assert_eq!(outputs.len(), 2);
            let flags = outputs
                .iter()
                .find(|o| o.value_id().as_str() == OUTPUTS[1])
                .unwrap();
            assert_eq!(flags.tensor().element_type(), ElementType::U32);
            let words = flags
                .bytes()
                .bytes()
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            let plan = shape.plan();
            assert_eq!(words.len() as u64, plan.strict.jobs);
            for (job, &flag) in words.iter().enumerate() {
                let head = job as u64 / plan.strict.query_tiles;
                let batch = head / ProbeShape::HEADS;
                let tile = job as u64 % plan.strict.query_tiles;
                let expected = tile >= plan.gate.first_fast_tile
                    && (0..ProbeShape::SEQUENCE).any(|key| shape.key_valid(batch, key))
                    && case != Case::AllReject
                    && !(case == Case::OneHeadNan && head == ProbeShape::HEADS)
                    && !(case == Case::VisibleKeyNan && head == 0)
                    && !(case == Case::VisibleValueInf && head == 1);
                assert_eq!(
                    flag,
                    u32::from(expected),
                    "shape={shape:?} case={case:?} request={request} batch={batch} job={job} tile={tile}: actual production flags={words:?}"
                );
            }
            if case == Case::Finite {
                assert!(
                    words.iter().any(|&flag| flag == 1),
                    "actual fast execution must be observed"
                );
            }
            let y = outputs
                .iter()
                .find(|o| o.value_id().as_str() == OUTPUTS[0])
                .unwrap();
            for (index, bytes) in y.bytes().bytes().chunks_exact(2).enumerate() {
                let index = index as u64;
                let row = (index / (ProbeShape::HEADS * 128)) % ProbeShape::SEQUENCE;
                let batch_head = index / (ProbeShape::SEQUENCE * ProbeShape::HEADS * 128)
                    * ProbeShape::HEADS
                    + (index / 128) % ProbeShape::HEADS;
                let actual = half::bf16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap()));
                let batch = batch_head / ProbeShape::HEADS;
                let head = batch_head % ProbeShape::HEADS;
                let channel = index % 128;
                let row_has_key = shape.row_has_key(batch, row);
                let own_job = batch_head * plan.strict.query_tiles + row / 32;
                let own_job_flag = words[own_job as usize];
                if !row_has_key {
                    assert_eq!(
                        actual.to_bits(),
                        0,
                        "all masked row must be +0 despite poison: shape={shape:?} case={case:?} request={request} batch={batch} head={head} query={row} channel={channel} row_has_key={row_has_key} own_job={own_job} own_job_flag={own_job_flag} actual_flags={words:?}"
                    );
                } else if (case == Case::OneHeadNan && row == 32 && batch_head == ProbeShape::HEADS)
                    || (case == Case::VisibleKeyNan && batch_head == 0 && row >= 17)
                {
                    assert!(
                        actual.is_nan(),
                        "strict rejected query must preserve NaN classification: shape={shape:?} case={case:?} request={request} batch={batch} head={head} query={row} channel={channel} row_has_key={row_has_key} own_job={own_job} own_job_flag={own_job_flag} actual_bits={:#06x} actual_flags={words:?}",
                        actual.to_bits()
                    );
                } else if case == Case::VisibleValueInf
                    && batch_head == 1
                    && row >= 17
                    && index % 128 == 127
                {
                    assert_eq!(
                        actual.to_f32(),
                        f32::INFINITY,
                        "visible nonfinite V uses strict fallback: shape={shape:?} case={case:?} request={request} batch={batch} head={head} query={row} channel={channel} row_has_key={row_has_key} own_job={own_job} own_job_flag={own_job_flag} actual_flags={words:?}"
                    );
                } else {
                    assert_eq!(
                        actual.to_bits(),
                        half::bf16::from_f32(shape.constant_value(index)).to_bits(),
                        "independent constant-V oracle: shape={shape:?} case={case:?} request={request} batch={batch} head={head} query={row} channel={channel} index={index} row_has_key={row_has_key} own_job={own_job} own_job_flag={own_job_flag} actual_flags={words:?}"
                    );
                }
            }
            observations.push(
                serde_json::json!({"case":format!("{case:?}"), "flags":words,
                "output_elements_checked":shape.tensor_bytes()/2}),
            );
            drop(outputs);
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
        drop(runtime.retain_tensor_provider().unwrap());
        let runtime_fingerprint = runtime
            .descriptor()
            .runtime_implementation_fingerprint
            .clone();
        drop(runtime);
        let reclaimed = capacity
            .reserve(DeviceOpaqueMemoryKind::Bootstrap, DEVICE_BYTES)
            .unwrap();
        assert!(capacity.reserve(DeviceOpaqueMemoryKind::Stream, 1).is_err());
        drop(reclaimed);
        eprintln!(
            "metal_dense_attention_actual_flags={}",
            serde_json::json!({
                "source_fingerprint":fingerprint, "runtime_fingerprint":runtime_fingerprint,
                "mask_case":format!("{shape:?}"), "key_runs":shape.key_runs(),
                "fast_pso":pso, "strict_pso":strict_pso,
                "selected_strict_domain":"Structured",
                "selected_key_visibility":if shape.plan().masked_keys { "MaskRuns" } else { "AllValid" },
                "observations":observations, "scratch_bytes":shape.plan().flag_bytes(),
                "snapshot_bytes":shape.plan().strict.jobs*4, "host_committed_after_close":0,
                "device_full_capacity_reclaimed":true,
                "scope":"production compile/encode with two additional managed diagnostic blits; not performance evidence"
            })
        );
    });
}
