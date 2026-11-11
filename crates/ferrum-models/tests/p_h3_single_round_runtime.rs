//! Real Stored gamma/multiplier execution for independent single-round ops.
//! Numerical fixtures do not prove Comfy/Torch binary parity or performance.
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::*;
use half::bf16;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, OnceLock},
};
const X: &str = "value.single_round.x";
const U: &str = "value.single_round.update";
const P: &str = "value.single_round.parameter";
const Y: &str = "value.single_round.y";
const PROFILE: &str = "fixture.single_round.bf16.v1";
fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
fn spec(dimensions: Vec<u64>) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum Kind {
    RowRms,
    AddcmulRow,
    AddcmulElementwise,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    rows: u64,
    width: u64,
    kind: Kind,
}
impl Config {
    fn operation(&self) -> &'static str {
        if self.kind == Kind::RowRms {
            ROW_RMS_NORM_SINGLE_ROUND_BF16_OPERATION_ID
        } else {
            ADDCMUL_BF16_OPERATION_ID
        }
    }
    fn parameter_shape(&self) -> Vec<u64> {
        match self.kind {
            Kind::RowRms => vec![self.width],
            Kind::AddcmulRow => vec![1, self.width],
            Kind::AddcmulElementwise => vec![self.rows, self.width],
        }
    }
    fn inputs(&self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        let mut inputs = BTreeMap::from([(id(X), spec(vec![self.rows, self.width]))]);
        if self.kind != Kind::RowRms {
            inputs.insert(id(U), spec(vec![self.rows, self.width]));
        }
        inputs
    }
    fn attrs(&self) -> BTreeMap<AttributeId, SemanticValue> {
        if self.kind == Kind::RowRms {
            BTreeMap::from([(
                id("epsilon"),
                SemanticValue::Rational(CanonicalRational::new(1, 100000).unwrap()),
            )])
        } else {
            BTreeMap::from([(
                id("broadcast"),
                SemanticValue::Text(
                    if self.kind == Kind::AddcmulRow {
                        "row"
                    } else {
                        "elementwise"
                    }
                    .into(),
                ),
            )])
        }
    }
}
struct Family;
impl ModelFamilyProvider for Family {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.fixture.single_round"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.fixture.single_round")])
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(c).unwrap() != *raw {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "noncanonical fixture config".into(),
            });
        }
        let x = ResolvedTensorSpec::new(
            vec![c.rows, c.width],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        let p = ResolvedTensorSpec::new(
            c.parameter_shape(),
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        if c.kind == Kind::RowRms {
            RowRmsNormSingleRoundBf16Geometry::new(&x, &p, &x, CanonicalRational::new(1, 100000)?)?;
        } else {
            AddcmulBf16Geometry::new(
                &x,
                &p,
                &x,
                &x,
                if c.kind == Kind::AddcmulRow {
                    RowBroadcast::Row
                } else {
                    RowBroadcast::Elementwise
                },
            )?;
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(id("metadata.fixture.single_round"))
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Config, VNextError> {
        let c =
            serde_json::from_value(raw.clone()).map_err(|e| VNextError::InvalidExecutionPlan {
                reason: format!("{e}"),
            })?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn weights(&self, c: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::Stored(WeightSchema {
            format_id: id("weight-format.safetensors.dense"),
            layout_id: id("weight-layout.fixture.single_round"),
            version: ContractVersion::new(1, 0),
            components: vec![WeightComponentSpec {
                id: id("component.single_round.parameter"),
                role: WeightComponentRole::Values,
                external_names: vec!["parameter".into()],
                dimensions: c.parameter_shape(),
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::Bf16,
                },
                required: true,
            }],
            tensors: vec![WeightTensorSpec {
                id: id("weight.single_round.parameter"),
                dimensions: c.parameter_shape(),
                logical_element_type: ElementType::Bf16,
                physical_layout: PhysicalWeightLayout::Dense {
                    component_id: id("component.single_round.parameter"),
                },
                required: true,
            }],
        }))
    }
    fn numerical_profiles(&self, c: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: id(PROFILE),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(Y),
                boundaries: BTreeMap::from([(id(Y), ElementType::Bf16)]),
                states: vec![],
                kv_storage: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(c.operation()),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: Some(ElementType::F32),
                    accumulation_type: Some(ElementType::F32),
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
        ModelProgram::new(
            self.family_id().clone(),
            c.inputs().keys().cloned().collect(),
            vec![ProgramBlock {
                id: "block.single_round".into(),
                nodes: vec![ProgramNode {
                    id: id("node.single_round"),
                    operation_id: id(c.operation()),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: if c.kind == Kind::RowRms {
                        vec![id(X), id(P)]
                    } else {
                        vec![id(X), id(P), id(U)]
                    },
                    outputs: vec![id(Y)],
                    attributes: c.attrs(),
                }],
            }],
            vec![],
            vec![WeightReference {
                weight_id: id("weight.single_round.parameter"),
                value_id: id(P),
                tensor: spec(c.parameter_shape()),
            }],
            vec![id(Y)],
        )?
        .with_input_tensor_specs(c.inputs())
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn host() -> HostMemoryAccount {
    host_with_weight_capacity(1 << 20)
}
fn host_with_weight_capacity(weight_backing_bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            1 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 20,
                transfer_staging_bytes: 1 << 20,
                output_bytes: 1 << 20,
                weight_backing_bytes,
            },
        )
        .unwrap(),
    )
}
fn bits(x: f32) -> u16 {
    bf16::from_f32(x).to_bits()
}
fn decode(x: u16) -> f32 {
    bf16::from_bits(x).to_f32()
}
fn managed(h: &HostMemoryAccount, name: &str, c: &Config, values: &[u16]) -> ManagedHostTensor {
    let mut a = h
        .allocate_zeroed(HostMemoryClass::InputPreprocess, values.len() * 2)
        .unwrap();
    for (d, v) in a.as_mut().chunks_exact_mut(2).zip(values) {
        d.copy_from_slice(&v.to_le_bytes())
    }
    ManagedHostTensor::new(
        id(name),
        ResolvedTensorSpec::new(
            vec![c.rows, c.width],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        a.freeze(),
    )
    .unwrap()
}
fn values(output: &ferrum_execution::TensorExecutionOutput) -> Vec<u16> {
    output
        .bytes()
        .bytes()
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn fixture(c: &Config) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
    let elements = (c.rows * c.width) as usize;
    let mut x = vec![bits(1.); elements];
    let mut parameter =
        vec![bits(129. / 128.); c.parameter_shape().iter().product::<u64>() as usize];
    let update = vec![bits(132. / 128.); elements];
    let expected = if c.kind == Kind::RowRms {
        for (i, v) in x.iter_mut().enumerate() {
            if c.width == 5376 && i % 128 >= 120 {
                *v = bits(1.5)
            }
            if i >= c.width as usize {
                *v = bits(-decode(*v))
            }
        }
        parameter.fill(bits(1.));
        parameter[0] = bits(129. / 128.);
        // F64 sum, then declared FP32 rsqrt/normalization. These dyadic fixture
        // squares sum exactly, independently of the provider reduction tree.
        let mut expected = Vec::new();
        for row in x.chunks_exact(c.width as usize) {
            let sum: f64 = row.iter().map(|&v| f64::from(decode(v)).powi(2)).sum();
            let mean = (sum / f64::from(u32::try_from(c.width).unwrap())) as f32;
            let inverse = 1.0_f32 / (mean + 1e-5_f32).sqrt();
            for (&v, &g) in row.iter().zip(&parameter) {
                expected.push(bits((decode(v) * inverse) * decode(g)))
            }
        }
        if c.width == 5376 {
            assert_eq!(decode(expected[0]), 0.96875);
            let old = bits(
                decode(bits((1.0_f32 / (1.078125_f32 + 1e-5_f32).sqrt()) * 1.0)) * 129. / 128.,
            );
            assert_eq!(decode(old), 0.97265625);
            assert_ne!(expected[0], old);
        }
        expected
    } else {
        if c.kind == Kind::AddcmulElementwise {
            parameter[c.width as usize..].fill(bits(-0.5));
        }
        x[c.width as usize..].fill(bits(-1.));
        let expected: Vec<_> = x
            .iter()
            .zip(&update)
            .enumerate()
            .map(|(i, (&x, &u))| {
                let g = parameter[if c.kind == Kind::AddcmulRow {
                    i % c.width as usize
                } else {
                    i
                }];
                let exact = f64::from(decode(x)) + f64::from(decode(g)) * f64::from(decode(u));
                bits(exact as f32)
            })
            .collect();
        assert_eq!(decode(expected[0]), 2.046875);
        assert_eq!(
            decode(bits(1.0 + decode(bits((129. / 128.) * (132. / 128.))))),
            2.03125
        );
        expected
    };
    (x, parameter, update, expected)
}
fn init(capacity: u64) -> ComponentInitializationConfig {
    ComponentInitializationConfig {
        run_id: id("run.single_round.init"),
        request_id: id("request.single_round.init"),
        transaction_id: id("transaction.single_round.init"),
        static_policy: StaticInitializationPolicy::new(capacity, 8).unwrap(),
        executor: TensorExecutorConfig::default(),
    }
}
fn run<R: DeviceRuntime>(
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    catalog: &CapabilityCatalog,
) {
    for c in [
        Config {
            rows: 2,
            width: 5376,
            kind: Kind::RowRms,
        },
        Config {
            rows: 2,
            width: 7,
            kind: Kind::RowRms,
        },
        Config {
            rows: 2,
            width: 3,
            kind: Kind::AddcmulRow,
        },
        Config {
            rows: 2,
            width: 3,
            kind: Kind::AddcmulElementwise,
        },
    ] {
        let (x, parameter, update, expected) = fixture(&c);
        let bytes: Vec<_> = parameter.iter().flat_map(|v| v.to_le_bytes()).collect();
        let directory = tempfile::tempdir().unwrap();
        safetensors::tensor::serialize_to_file(
            [(
                "parameter",
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::BF16,
                    c.parameter_shape().iter().map(|v| *v as usize).collect(),
                    &bytes,
                )
                .unwrap(),
            )],
            &None,
            &directory.path().join("model.safetensors"),
        )
        .unwrap();
        let archive = SafetensorsArchive::open(directory.path()).unwrap();
        let family = TypedFamilyRegistration::new(Family)
            .prepare_with_profile(&serde_json::to_value(&c).unwrap(), &id(PROFILE))
            .unwrap();
        let materializers = WeightMaterializerRegistry::identity_only().unwrap();
        let catalog = materializers.augment_catalog(catalog.clone()).unwrap();
        let policy = TensorComponentPolicy::new(
            runtime.descriptor().total_memory_bytes,
            runtime
                .descriptor()
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect(),
        )
        .unwrap();
        let component = CompiledTensorComponent::compile(
            family,
            registry,
            &catalog,
            &policy,
            &materializers,
            &ProgramPlanCompileOptions::new(c.inputs()).unwrap(),
        )
        .unwrap();
        let account = host();
        let mut session = component
            .initialize(
                runtime.clone(),
                account.clone(),
                ComponentWeightSource::Stored(&archive),
                init(1 << 16),
            )
            .unwrap();
        let mut inputs = vec![managed(&account, X, &c, &x)];
        if c.kind != Kind::RowRms {
            inputs.push(managed(&account, U, &c, &update));
        }
        let output = session
            .execute(
                id("run.single_round.execute"),
                id("request.single_round.execute"),
                inputs,
            )
            .unwrap();
        assert_eq!(values(&output[0]), expected, "{:?},H={}", c.kind, c.width);
        assert!(account.owns(output[0].bytes()));
        // Perturb only row 1; row 0 is an independent request's row.
        let mut changed = x.clone();
        changed[c.width as usize..].fill(bits(0.25));
        let mut inputs = vec![managed(&account, X, &c, &changed)];
        if c.kind != Kind::RowRms {
            inputs.push(managed(&account, U, &c, &update));
        }
        let changed_output = session
            .execute(
                id("run.single_round.changed"),
                id("request.single_round.changed"),
                inputs,
            )
            .unwrap();
        assert_eq!(
            &values(&changed_output[0])[..c.width as usize],
            &expected[..c.width as usize]
        );
        assert!(matches!(
            session
                .close()
                .unwrap_or_else(|e| panic!("close failed: {:?}", e.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        drop((output, changed_output));
        assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    }
}
#[test]
fn stored_reference_single_round_row_rms_and_addcmul_match_independent_oracles() {
    let c = ReferenceVNextComposition::create(id("device.reference.single_round")).unwrap();
    run(c.runtime().clone(), c.registry(), c.catalog());
    assert_eq!(c.runtime().snapshot().live_allocations, 0);
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires configured CUDA host with admitted fixture capacity"]
fn stored_cuda_single_round_row_rms_and_addcmul_match_independent_oracles() {
    use ferrum_kernels::backend::cuda::{
        vnext_ops::{cuda_tensor_runtime_config, cuda_vnext_tensor_providers, CudaTensorOperation},
        vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
    };
    let r = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "single-round-fixture-capacity".into(),
        })
        .unwrap(),
    );
    let providers = cuda_vnext_tensor_providers(
        &r,
        &[
            CudaTensorOperation::RowRmsNormSingleRound,
            CudaTensorOperation::Addcmul,
        ],
    )
    .unwrap();
    for provider in &providers {
        assert_eq!(
            provider.descriptor().accepted_weight_formats(),
            &BTreeSet::from([id::<WeightFormatId>("weight-format.safetensors.dense")])
        );
    }
    let registry = OperationRuntimeRegistry::new(
        vec![
            Box::new(row_rms_norm_single_round_bf16_contract().unwrap()),
            Box::new(addcmul_bf16_contract().unwrap()),
        ],
        providers,
    )
    .unwrap();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.single_round.fixture"),
        ContractVersion::new(1, 0),
        r.descriptor().runtime_implementation_fingerprint.clone(),
        r.descriptor().id.clone(),
        r.descriptor().capabilities.clone(),
    )
    .unwrap();
    let catalog = registry
        .capability_catalog(r.descriptor().clone(), vec![engine])
        .unwrap();
    run(r, &registry, &catalog);
}
