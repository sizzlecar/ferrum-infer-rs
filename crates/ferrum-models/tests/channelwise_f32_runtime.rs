//! Actual managed execution and owner retirement for the four F32 channel
//! operations. CUDA uses the same closed bit/numeric oracles as Reference.
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutionOutput, TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, OnceLock},
};
const OUTPUT: &str = "value.channelwise.fixture.output";
const PROFILE: &str = "fixture.channelwise.f32.v1";
fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
fn input_id(index: usize) -> ProgramValueId {
    id(format!("value.channelwise.fixture.input{index}"))
}
fn tensor(shape: Vec<u64>) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: shape,
        element_type: ElementType::F32,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
enum Kind {
    Slice,
    Bias,
    Residual,
    Nan,
}
impl Kind {
    fn operation(self) -> F32ChannelwiseOp {
        match self {
            Self::Slice => F32ChannelwiseOp::GroupedColumnSlice,
            Self::Bias => F32ChannelwiseOp::BiasAdd,
            Self::Residual => F32ChannelwiseOp::ScaledResidual,
            Self::Nan => F32ChannelwiseOp::NanToNum,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    kind: Kind,
}
impl Config {
    fn inputs(&self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        let kind = self.kind.operation();
        let mut specs = vec![tensor(vec![
            2,
            if kind == F32ChannelwiseOp::GroupedColumnSlice {
                12
            } else {
                4
            },
        ])];
        if kind == F32ChannelwiseOp::ScaledResidual {
            specs.push(tensor(vec![2, 4]));
        }
        if matches!(
            kind,
            F32ChannelwiseOp::BiasAdd | F32ChannelwiseOp::ScaledResidual
        ) {
            specs.push(tensor(vec![4]));
        }
        specs
            .into_iter()
            .enumerate()
            .map(|(i, s)| (input_id(i), s))
            .collect()
    }
    fn attrs(&self) -> BTreeMap<AttributeId, SemanticValue> {
        if matches!(self.kind, Kind::Slice) {
            [("group_width", 6), ("column_start", 2), ("column_count", 2)]
                .into_iter()
                .map(|(k, v)| (id(k), SemanticValue::Unsigned(v)))
                .collect()
        } else {
            BTreeMap::new()
        }
    }
    fn geometry(&self) -> Result<F32ChannelwiseGeometry, VNextError> {
        let specs = self
            .inputs()
            .into_values()
            .map(|s| ResolvedTensorSpec::new(s.dimensions, s.element_type, s.layout))
            .collect::<Result<Vec<_>, _>>()?;
        let output = ResolvedTensorSpec::new(
            vec![2, 4],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )?;
        F32ChannelwiseGeometry::new(
            self.kind.operation(),
            &specs.iter().collect::<Vec<_>>(),
            &output,
            &self.attrs(),
        )
    }
}
struct Family;
impl ModelFamilyProvider for Family {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.fixture.channelwise_f32"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.fixture.channelwise_f32")])
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Config, VNextError> {
        let c = serde_json::from_value(raw.clone()).map_err(|e| invalid(format!("{e}")))?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(c).unwrap() != *raw {
            return Err(invalid("requires canonical fixture"));
        }
        c.geometry()?;
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(id("metadata.fixture.channelwise_f32"))
    }
    fn weights(&self, _: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(&self, c: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let kind = c.kind.operation();
        let arithmetic = matches!(
            kind,
            F32ChannelwiseOp::BiasAdd | F32ChannelwiseOp::ScaledResidual
        );
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: id(PROFILE),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(OUTPUT),
                boundaries: BTreeMap::from([(id(OUTPUT), ElementType::F32)]),
                states: vec![],
                kv_storage: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(kind.operation_id()),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: (kind == F32ChannelwiseOp::ScaledResidual)
                        .then_some(ElementType::F32),
                    accumulation_type: arithmetic.then_some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }],
            }],
            vec![id(PROFILE)],
        )
    }
    fn semantic_program(
        &self,
        c: &Config,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let inputs: Vec<_> = (0..c.kind.operation().input_count())
            .map(input_id)
            .collect();
        ModelProgram::new(
            self.family_id().clone(),
            inputs.clone(),
            vec![ProgramBlock {
                id: "block.fixture.channelwise_f32".into(),
                nodes: vec![ProgramNode {
                    id: id("node.fixture.channelwise_f32"),
                    operation_id: id(c.kind.operation().operation_id()),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs,
                    outputs: vec![id(OUTPUT)],
                    attributes: c.attrs(),
                }],
            }],
            vec![],
            vec![],
            vec![id(OUTPUT)],
        )?
        .with_input_tensor_specs(c.inputs())
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
fn fixture(c: &Config, trial: usize) -> (Vec<Vec<u32>>, Vec<u32>) {
    let (mut inputs, mut expected) = match c.kind {
        Kind::Slice => {
            let mut x: Vec<_> = (0..24).map(|i| 0x3f800000 + i).collect();
            for (i, bits) in [
                (2, 0x80000000),
                (3, 0x7f800001),
                (8, 1),
                (9, 0xffc12345),
                (14, 0x7f800000),
                (15, 0xff800000),
            ] {
                x[i] = bits;
            }
            let y = [2, 3, 8, 9, 14, 15, 20, 21]
                .into_iter()
                .map(|i| x[i])
                .collect();
            (vec![x], y)
        }
        Kind::Bias => {
            let x = [
                f32::from_bits(0x3f800008),
                -0.0,
                f32::from_bits(1),
                f32::MAX,
                -1.0,
                2.0,
                -f32::from_bits(1),
                -f32::MAX,
            ];
            let b = [
                f32::from_bits(0x35800000),
                -0.0,
                f32::from_bits(1),
                f32::MAX,
            ];
            let y = [
                0x3f800010,
                0x80000000,
                2,
                0x7f800000,
                (-1.0_f32 + b[0]).to_bits(),
                2.0_f32.to_bits(),
                0,
                0,
            ];
            (
                vec![
                    x.into_iter().map(f32::to_bits).collect(),
                    b.into_iter().map(f32::to_bits).collect(),
                ],
                y.to_vec(),
            )
        }
        Kind::Residual => {
            let a = f32::from_bits(0x3f800001);
            let b = f32::from_bits(0x3f7ffffe);
            let base = [-1.0, -f32::MAX, 0.0, 1.0, -1.0, -f32::MAX, 0.0, 1.0];
            let branch = [
                a,
                f32::MAX,
                f32::from_bits(1),
                f32::INFINITY,
                a,
                f32::MAX,
                f32::from_bits(1),
                f32::INFINITY,
            ];
            let scale = [b, 2.0, 1.0, 0.0];
            let expected = [
                (-2.0_f32.powi(-46)).to_bits(),
                f32::MAX.to_bits(),
                1,
                f32::NAN.to_bits(),
            ]
            .repeat(2);
            assert_eq!(a * b - 1.0, 0.0);
            assert!((f32::MAX * 2.0 - f32::MAX).is_infinite());
            (
                vec![
                    base.into_iter().map(f32::to_bits).collect(),
                    branch.into_iter().map(f32::to_bits).collect(),
                    scale.into_iter().map(f32::to_bits).collect(),
                ],
                expected,
            )
        }
        Kind::Nan => {
            let x = vec![
                0, 0x80000000, 1, 0x807fffff, 0x7f800000, 0xff800000, 0x7f800001, 0xffc12345,
            ];
            let y = vec![0, 0x80000000, 1, 0x807fffff, 0x7f7fffff, 0xff7fffff, 0, 0];
            (vec![x], y)
        }
    };
    if trial != 0 {
        for input in &mut inputs {
            input.fill(0);
        }
        expected.fill(0);
    }
    (inputs, expected)
}
fn assert_output(c: &Config, bytes: &[u8], expected: &[u32]) {
    assert_eq!(bytes.len(), expected.len() * 4);
    let actual = bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    for (got, &expected) in actual.zip(expected) {
        if matches!(c.kind, Kind::Residual) && f32::from_bits(expected).is_nan() {
            assert!(f32::from_bits(got).is_nan());
        } else {
            assert_eq!(got, expected);
        }
    }
}
fn run<R: DeviceRuntime>(
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    catalog: &CapabilityCatalog,
    c: &Config,
) {
    let prepared = TypedFamilyRegistration::new(Family)
        .prepare_with_profile(&serde_json::to_value(c).unwrap(), &id(PROFILE))
        .unwrap();
    let d = runtime.descriptor();
    let materializers = WeightMaterializerRegistry::new(vec![]).unwrap();
    let catalog = materializers.augment_catalog(catalog.clone()).unwrap();
    let mut options = ProgramPlanCompileOptions::new(c.inputs()).unwrap();
    options
        .insert_tensor_spec(id(OUTPUT), tensor(vec![2, 4]))
        .unwrap();
    let component = CompiledTensorComponent::compile(
        prepared,
        registry,
        &catalog,
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect(),
        )
        .unwrap(),
        &materializers,
        &options,
    )
    .unwrap();
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            1 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 18,
                transfer_staging_bytes: 1 << 18,
                output_bytes: 1 << 18,
                weight_backing_bytes: 1 << 18,
            },
        )
        .unwrap(),
    );
    let mut session = component
        .initialize(
            runtime,
            host.clone(),
            ComponentWeightSource::None,
            ComponentInitializationConfig {
                request_id: id("request.channelwise.init"),
                run_id: id("run.channelwise.init"),
                transaction_id: id("transaction.channelwise.init"),
                static_policy: StaticInitializationPolicy::new(1 << 18, 16).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    let mut retained: Option<Vec<TensorExecutionOutput>> = None;
    for trial in 0..2 {
        let (values, expected) = fixture(c, trial);
        let specs = c.inputs();
        let inputs = values
            .iter()
            .enumerate()
            .map(|(i, bits)| {
                let mut allocation = host
                    .allocate_zeroed(HostMemoryClass::InputPreprocess, bits.len() * 4)
                    .unwrap();
                for (dst, bits) in allocation.as_mut().chunks_exact_mut(4).zip(bits) {
                    dst.copy_from_slice(&bits.to_le_bytes());
                }
                let name = input_id(i);
                let s = &specs[&name];
                ManagedHostTensor::new(
                    name,
                    ResolvedTensorSpec::new(s.dimensions.clone(), s.element_type, s.layout.clone())
                        .unwrap(),
                    allocation.freeze(),
                )
                .unwrap()
            })
            .collect();
        let outputs = session
            .execute(
                id(format!("run.channelwise.{trial}")),
                id(format!("request.channelwise.{trial}")),
                inputs,
            )
            .unwrap();
        assert_eq!(outputs[0].tensor().element_type(), ElementType::F32);
        assert!(host.owns(outputs[0].bytes()));
        assert_output(c, outputs[0].bytes().bytes(), &expected);
        if let Some(previous) = retained.as_ref() {
            let (_, prior_expected) = fixture(c, 0);
            assert_output(c, previous[0].bytes().bytes(), &prior_expected);
        }
        retained = Some(outputs);
    }
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|f| panic!("channelwise close: {:?}", f.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert!(host.usage().unwrap().committed_bytes() > 0);
    let retained = retained.unwrap();
    assert!(host.owns(retained[0].bytes()));
    assert_eq!(retained[0].bytes().bytes(), [0; 32]);
    drop(retained);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
#[test]
fn managed_reference_channelwise_profiles_match_closed_oracles_and_release_owners() {
    let c = ReferenceVNextComposition::create(id("device.reference.channelwise")).unwrap();
    for kind in [Kind::Slice, Kind::Bias, Kind::Residual, Kind::Nan] {
        run(
            c.runtime().clone(),
            c.registry(),
            c.catalog(),
            &Config { kind },
        );
        assert_eq!(c.runtime().snapshot().live_allocations, 0);
    }
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires configured CUDA host and pinned native operator-set lock"]
fn managed_cuda_channelwise_profiles_match_closed_oracles_and_release_owners() {
    use ferrum_kernels::backend::cuda::{
        vnext_ops::{cuda_tensor_runtime_config, cuda_vnext_tensor_providers, CudaTensorOperation},
        vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
    };
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 2,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "unqualified-channelwise-f32-fixture-v1".into(),
        })
        .unwrap(),
    );
    let registry = OperationRuntimeRegistry::new(
        F32ChannelwiseOp::ALL
            .into_iter()
            .map(|kind| Box::new(kind.contract().unwrap()) as Box<dyn OperationContract>)
            .collect(),
        cuda_vnext_tensor_providers(
            &runtime,
            &[
                CudaTensorOperation::GroupedColumnSliceF32,
                CudaTensorOperation::ChannelBiasAddF32,
                CudaTensorOperation::ChannelScaledResidualF32,
                CudaTensorOperation::NanToNumF32,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.channelwise.cuda"),
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )
    .unwrap();
    let catalog = registry
        .capability_catalog(d.clone(), vec![engine])
        .unwrap();
    for kind in [Kind::Slice, Kind::Bias, Kind::Residual, Kind::Nan] {
        run(runtime.clone(), &registry, &catalog, &Config { kind });
    }
}
