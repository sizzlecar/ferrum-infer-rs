//! Actual Stored component checks for the independent fused RMS/partial RoPE
//! numerical profile. These fixtures do not establish upstream binary parity.
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
const X: &str = "value.fused_rms_rope.x";
const C: &str = "value.fused_rms_rope.cos";
const S: &str = "value.fused_rms_rope.sin";
const G: &str = "value.fused_rms_rope.gamma";
const Y: &str = "value.fused_rms_rope.y";
const PROFILE: &str = "fixture.fused_rms_rope.bf16.single_norm.v1";
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
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    rows: u64,
    heads: u64,
    head: u64,
    rot: u64,
}
impl Config {
    fn inputs(&self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        BTreeMap::from([
            (id(X), spec(vec![self.rows, self.heads * self.head])),
            (id(C), spec(vec![self.rows, self.rot])),
            (id(S), spec(vec![self.rows, self.rot])),
        ])
    }
}
struct Family;
impl ModelFamilyProvider for Family {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.fixture.fused_rms_rope"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.fixture.fused_rms_rope")])
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        config: &Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(config).unwrap() != *raw
            || config.heads == 0
            || config.heads.checked_mul(config.head).is_none()
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "invalid fused RMS/RoPE fixture config".into(),
            });
        }
        let x = ResolvedTensorSpec::new(
            vec![config.rows, config.heads * config.head],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        let g = ResolvedTensorSpec::new(
            vec![config.head],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        let c = ResolvedTensorSpec::new(
            vec![config.rows, config.rot],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        FusedHeadRmsPartialRopeBf16Geometry::new(
            &x,
            &g,
            &c,
            &c,
            &x,
            config.head,
            config.rot,
            CanonicalRational::new(1, 100000)?,
        )?;
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        config: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.fixture.fused_rms_rope"))
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Config, VNextError> {
        let config =
            serde_json::from_value(raw.clone()).map_err(|e| VNextError::InvalidExecutionPlan {
                reason: format!("{e}"),
            })?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }
    fn weights(&self, c: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::Stored(WeightSchema {
            format_id: id("weight-format.safetensors.dense"),
            layout_id: id("weight-layout.fixture.fused_rms_rope"),
            version: ContractVersion::new(1, 0),
            components: vec![WeightComponentSpec {
                id: id("component.fixture.gamma"),
                role: WeightComponentRole::Values,
                external_names: vec!["gamma".into()],
                dimensions: vec![c.head],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::Bf16,
                },
                required: true,
            }],
            tensors: vec![WeightTensorSpec {
                id: id("weight.fixture.gamma"),
                dimensions: vec![c.head],
                logical_element_type: ElementType::Bf16,
                physical_layout: PhysicalWeightLayout::Dense {
                    component_id: id("component.fixture.gamma"),
                },
                required: true,
            }],
        }))
    }
    fn numerical_profiles(&self, _: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
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
                    operation_id: id(FUSED_HEAD_RMS_PARTIAL_ROPE_BF16_OPERATION_ID),
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
            vec![id(X), id(C), id(S)],
            vec![ProgramBlock {
                id: "block.fixture.fused_rms_rope".into(),
                nodes: vec![ProgramNode {
                    id: id("node.fixture.fused_rms_rope"),
                    operation_id: id(FUSED_HEAD_RMS_PARTIAL_ROPE_BF16_OPERATION_ID),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![id(X), id(G), id(C), id(S)],
                    outputs: vec![id(Y)],
                    attributes: BTreeMap::from([
                        (id("head_dim"), SemanticValue::Unsigned(c.head)),
                        (id("rotary_dim"), SemanticValue::Unsigned(c.rot)),
                        (
                            id("epsilon"),
                            SemanticValue::Rational(CanonicalRational::new(1, 100000)?),
                        ),
                    ]),
                }],
            }],
            vec![],
            vec![WeightReference {
                weight_id: id("weight.fixture.gamma"),
                value_id: id(G),
                tensor: spec(vec![c.head]),
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
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            1 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 20,
                transfer_staging_bytes: 1 << 20,
                output_bytes: 1 << 20,
                weight_backing_bytes: 1 << 20,
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
fn managed(
    host: &HostMemoryAccount,
    name: &str,
    spec: &ProgramTensorSpec,
    values: &[u16],
) -> ManagedHostTensor {
    let mut a = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, values.len() * 2)
        .unwrap();
    for (target, value) in a.as_mut().chunks_exact_mut(2).zip(values) {
        target.copy_from_slice(&value.to_le_bytes());
    }
    ManagedHostTensor::new(
        id(name),
        ResolvedTensorSpec::new(
            spec.dimensions.clone(),
            spec.element_type,
            spec.layout.clone(),
        )
        .unwrap(),
        a.freeze(),
    )
    .unwrap()
}

/// Independent full-head sum in F64, then explicit declared F32 normalization
/// and BF16 boundaries. Fixture dyadic squares sum exactly in both reductions.
fn oracle(config: &Config, x: &[u16], gamma: &[u16], cos: &[u16], sin: &[u16]) -> Vec<u16> {
    let (head, rot, columns) = (
        config.head as usize,
        config.rot as usize,
        (config.head * config.heads) as usize,
    );
    let mut output = vec![0; x.len()];
    for (ordinal, source) in x.chunks_exact(head).enumerate() {
        let square_sum: f64 = source.iter().map(|&v| f64::from(decode(v)).powi(2)).sum();
        let mean = (square_sum / f64::from(u32::try_from(config.head).unwrap())) as f32;
        let inverse = 1.0_f32 / (mean + 1e-5_f32).sqrt();
        let n: Vec<_> = source
            .iter()
            .zip(gamma)
            .map(|(&v, &g)| decode(bits((decode(v) * inverse) * decode(g))))
            .collect();
        let row = ordinal * head / columns;
        for d in 0..head {
            let value = if d < rot {
                let partner = if d < rot / 2 {
                    -n[d + rot / 2]
                } else {
                    n[d - rot / 2]
                };
                let p = n[d] * decode(cos[row * rot + d]);
                let q = partner * decode(sin[row * rot + d]);
                p + q
            } else {
                n[d]
            };
            output[ordinal * head + d] = bits(value);
        }
    }
    output
}

fn run<R: DeviceRuntime>(
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    catalog: &CapabilityCatalog,
) {
    for (head, rot) in [(8, 6), (128, 96), (8, 8)] {
        // Ragged request documents [2,4] rows; this operation is row-local.
        let c = Config {
            rows: 6,
            heads: 2,
            head,
            rot,
        };
        let columns = (head * 2) as usize;
        let mut gamma = vec![bits(1.); head as usize];
        gamma[0] = bits(0.78125);
        gamma[(rot / 2) as usize] = bits(0.7890625);
        if head > rot {
            gamma[rot as usize] = bits(129. / 128.);
        }
        let mut x = vec![bits(1.); 6 * columns];
        if head == 128 {
            for row in [1, 3, 5] {
                for h in 0..2 {
                    for d in 0..8 {
                        x[row * columns + h * 128 + d] = bits(1.5);
                    }
                }
            }
        }
        // Second document has independent contents and frequencies.
        x[2 * columns..3 * columns].fill(bits(-1.));
        let mut cos = vec![bits(1.); 6 * rot as usize];
        let mut sin = vec![bits(0.); cos.len()];
        cos[..rot as usize].fill(bits(0.70703125));
        sin[..rot as usize].fill(bits(0.70703125));
        let expected = oracle(&c, &x, &gamma, &cos, &sin);
        assert_eq!(decode(expected[0]), -0.005523681640625);
        if head == 128 {
            assert_eq!(decode(expected[columns + 96]), 0.96875);
            assert_ne!(decode(expected[columns + 96]), decode(x[columns + 96]));
        }
        let directory = tempfile::tempdir().unwrap();
        let gamma_bytes: Vec<_> = gamma.iter().flat_map(|v| v.to_le_bytes()).collect();
        safetensors::tensor::serialize_to_file(
            [(
                "gamma",
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::BF16,
                    vec![head as usize],
                    &gamma_bytes,
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
                ComponentInitializationConfig {
                    run_id: id("run.fused.init"),
                    request_id: id("request.fused.init"),
                    transaction_id: id("transaction.fused.init"),
                    static_policy: StaticInitializationPolicy::new(4096, 8).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap();
        let specs = c.inputs();
        let output = session
            .execute(
                id("run.fused.execute"),
                id("request.fused.execute"),
                vec![
                    managed(&account, X, &specs[&id::<ProgramValueId>(X)], &x),
                    managed(&account, C, &specs[&id::<ProgramValueId>(C)], &cos),
                    managed(&account, S, &specs[&id::<ProgramValueId>(S)], &sin),
                ],
            )
            .unwrap();
        let actual: Vec<_> = output[0]
            .bytes()
            .bytes()
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(actual, expected, "D={head},rot={rot}");
        assert!(account.owns(output[0].bytes()));
        // Perturb only document 2 and execute through the same stored session.
        x[2 * columns..].fill(bits(0.5));
        let altered_expected = oracle(&c, &x, &gamma, &cos, &sin);
        let altered = session
            .execute(
                id("run.fused.altered"),
                id("request.fused.altered"),
                vec![
                    managed(&account, X, &specs[&id::<ProgramValueId>(X)], &x),
                    managed(&account, C, &specs[&id::<ProgramValueId>(C)], &cos),
                    managed(&account, S, &specs[&id::<ProgramValueId>(S)], &sin),
                ],
            )
            .unwrap();
        let altered_values: Vec<_> = altered[0]
            .bytes()
            .bytes()
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(altered_values, altered_expected);
        assert_eq!(&actual[..2 * columns], &altered_values[..2 * columns]);
        assert!(matches!(
            session
                .close()
                .unwrap_or_else(|e| panic!("close failed: {:?}", e.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        drop((output, altered));
        assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    }
}
#[test]
fn stored_reference_fused_rms_rope_preserves_single_norm_rotation_and_request_isolation() {
    let c = ReferenceVNextComposition::create(id("device.reference.fused_rms_rope")).unwrap();
    run(c.runtime().clone(), c.registry(), c.catalog());
    assert_eq!(c.runtime().snapshot().live_allocations, 0);
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires configured CUDA host with admitted fixture capacity"]
fn stored_cuda_fused_rms_rope_matches_independent_boundary_fixtures() {
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
            opaque_memory_profile: "fused-rms-rope-fixture-capacity".into(),
        })
        .unwrap(),
    );
    let providers =
        cuda_vnext_tensor_providers(&r, &[CudaTensorOperation::FusedHeadRmsPartialRope]).unwrap();
    let provider = &providers[0];
    assert_eq!(
        provider.descriptor().accepted_weight_formats(),
        &BTreeSet::from([id::<WeightFormatId>("weight-format.safetensors.dense")])
    );
    assert!(provider
        .descriptor()
        .accepted_quantization_formats()
        .is_empty());
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(
            fused_head_rms_partial_rope_bf16_contract().unwrap(),
        )],
        providers,
    )
    .unwrap();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.fused_rms_rope.fixture"),
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
