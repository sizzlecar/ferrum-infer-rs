//! One small Stored family used by real Reference and CUDA executions.
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
pub fn id<T>(s: impl Into<String>) -> T
where
    T: TryFrom<String>,
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
const X: &str = "value.layer_norm.fixture.x";
const Y: &str = "value.layer_norm.fixture.y";
const G: &str = "value.layer_norm.fixture.gamma";
const B: &str = "value.layer_norm.fixture.beta";
const PROFILE: &str = "fixture.layer_norm.f32.two_pass.separate_affine.v1";
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    width: u64,
}
struct Family {
    id: ModelFamilyId,
}
impl Family {
    fn new() -> Self {
        Self {
            id: id("family.fixture.layer_norm.f32"),
        }
    }
}
fn tensor(dimensions: Vec<u64>) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type: ElementType::F32,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn weight_value(name: &str) -> &'static str {
    if name == "gamma" {
        G
    } else {
        B
    }
}
impl ModelFamilyProvider for Family {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.fixture.layer_norm.f32")])
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Config, VNextError> {
        let c: Config = serde_json::from_value(raw.clone()).map_err(|e| invalid(e.to_string()))?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(c).unwrap() != *raw || c.width == 0 || c.width > 1 << 24 {
            return Err(invalid("invalid fixture width/config"));
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(id("metadata.fixture.layer_norm.f32"))
    }
    fn weights(&self, c: &Config) -> Result<ModelWeights, VNextError> {
        let mut components = Vec::new();
        let mut tensors = Vec::new();
        for name in ["gamma", "beta"] {
            let component: WeightId = id(format!("component.layer_norm.fixture.{name}"));
            components.push(WeightComponentSpec {
                id: component.clone(),
                role: WeightComponentRole::Values,
                external_names: vec![name.into()],
                dimensions: vec![c.width],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::F32,
                },
                required: true,
            });
            tensors.push(WeightTensorSpec {
                id: id(format!("weight.layer_norm.fixture.{name}")),
                dimensions: vec![c.width],
                logical_element_type: ElementType::F32,
                physical_layout: PhysicalWeightLayout::Dense {
                    component_id: component,
                },
                required: true,
            });
        }
        Ok(ModelWeights::Stored(WeightSchema {
            format_id: id("weight-format.safetensors.dense"),
            layout_id: id("weight-layout.fixture.layer_norm.f32"),
            version: ContractVersion::new(1, 0),
            components,
            tensors,
        }))
    }
    fn numerical_profiles(&self, _: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: id(PROFILE),
                family_id: self.id.clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(Y),
                boundaries: BTreeMap::from([(id(Y), ElementType::F32)]),
                states: vec![],
                kv_storage: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(LAYER_NORM_F32_OPERATION_ID),
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
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(c)?.resolve(&p.id)? != p {
            return Err(invalid("fixture profile differs"));
        }
        ModelProgram::new(
            self.id.clone(),
            vec![id(X)],
            vec![ProgramBlock {
                id: "block.layer_norm.fixture".into(),
                nodes: vec![ProgramNode {
                    id: id("node.layer_norm.fixture"),
                    operation_id: id(LAYER_NORM_F32_OPERATION_ID),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![id(X), id(G), id(B)],
                    outputs: vec![id(Y)],
                    attributes: BTreeMap::from([(
                        id("epsilon"),
                        SemanticValue::Rational(CanonicalRational::new(1, 4)?),
                    )]),
                }],
            }],
            vec![],
            ["gamma", "beta"]
                .into_iter()
                .map(|name| WeightReference {
                    weight_id: id(format!("weight.layer_norm.fixture.{name}")),
                    value_id: id(weight_value(name)),
                    tensor: tensor(vec![c.width]),
                })
                .collect(),
            vec![id(Y)],
        )?
        .with_input_tensor_specs(BTreeMap::from([(id(X), tensor(vec![2, c.width]))]))
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
fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            1 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 256 << 10,
                output_bytes: 256 << 10,
                transfer_staging_bytes: 256 << 10,
                weight_backing_bytes: 256 << 10,
            },
        )
        .unwrap(),
    )
}
fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
struct Parameters {
    gamma: Vec<u8>,
    beta: Vec<u8>,
}
impl WeightComponentSource for Parameters {
    fn component<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        let [name] = component.external_names.as_slice() else {
            return Err(invalid("one original affine parameter per component"));
        };
        let bytes = match name.as_str() {
            "gamma" => self.gamma.as_slice(),
            "beta" => self.beta.as_slice(),
            _ => return Err(invalid("unknown affine parameter")),
        };
        WeightComponentPayload::new(
            component,
            name.clone(),
            "layer_norm.typed_fixture",
            component.dimensions.clone(),
            ElementType::F32,
            bytes,
        )
    }
}
fn parameters(width: usize) -> (Vec<f32>, Vec<f32>) {
    let mut gamma = (0..width)
        .map(|i| 1.000244140625 + (i % 7) as f32 / 64.0)
        .collect::<Vec<_>>();
    let mut beta = (0..width)
        .map(|i| if i % 2 == 0 { 0.125 } else { -0.125 })
        .collect::<Vec<_>>();
    if width >= 4 && width % 4 == 0 {
        gamma[width - 1] = f32::from_bits(0x3f800001);
        beta[width - 1] = -(1.5 + f32::from_bits(0x34800000));
    }
    (gamma, beta)
}
fn input(width: usize, nonfinite: bool) -> Vec<f32> {
    let mut x = (0..width)
        .map(|c| {
            if width == 3 {
                [1.0, 2.0, 4.0][c]
            } else {
                [1.0, 1.0, 1.0, 3.0][c % 4]
            }
        })
        .collect::<Vec<_>>();
    if nonfinite {
        x[0] = f32::NAN;
    }
    x.extend(std::iter::repeat_n(3.0, width));
    x
}
/// Independent F64 statistics; the declared affine rounds the normalized value,
/// then gamma product, then beta sum separately in F32.
fn oracle(x: &[f32], gamma: &[f32], beta: &[f32]) -> Vec<f32> {
    x.chunks_exact(gamma.len())
        .flat_map(|row| {
            let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / row.len() as f64;
            let var = row
                .iter()
                .map(|&v| (f64::from(v) - mean).powi(2))
                .sum::<f64>()
                / row.len() as f64;
            let inverse = 1.0 / (var + 0.25).sqrt();
            row.iter().zip(gamma).zip(beta).map(move |((&x, &g), &b)| {
                let normalized = ((f64::from(x) - mean) * inverse) as f32;
                let scaled = normalized * g;
                scaled + b
            })
        })
        .collect()
}
pub fn run<R: DeviceRuntime>(
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    catalog: &CapabilityCatalog,
    materializers: &WeightMaterializerRegistry,
) {
    for width in [1, 3, 128, 2560] {
        let c = Config { width };
        let (gamma, beta) = parameters(width as usize);
        let source = Parameters {
            gamma: bytes(&gamma),
            beta: bytes(&beta),
        };
        let family = TypedFamilyRegistration::new(Family::new())
            .prepare_with_profile(&serde_json::to_value(&c).unwrap(), &id(PROFILE))
            .unwrap();
        let d = runtime.descriptor();
        let storage = d
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let options =
            ProgramPlanCompileOptions::new(BTreeMap::from([(id(X), tensor(vec![2, width]))]))
                .unwrap();
        // Less than the two original parameter vectors cannot admit this real
        // Stored program, before allocating device storage or reading inputs.
        assert!(CompiledTensorComponent::compile(
            family.clone(),
            registry,
            catalog,
            &TensorComponentPolicy::new(width * 8 - 1, storage.clone()).unwrap(),
            materializers,
            &options
        )
        .is_err());
        let compiled = CompiledTensorComponent::compile(
            family,
            registry,
            catalog,
            &TensorComponentPolicy::new(d.total_memory_bytes, storage).unwrap(),
            materializers,
            &options,
        )
        .unwrap();
        let account = host();
        let mut session = compiled
            .initialize(
                runtime.clone(),
                account.clone(),
                ComponentWeightSource::Stored(&source),
                ComponentInitializationConfig {
                    run_id: id(format!("run.layer_norm.init.{width}")),
                    request_id: id(format!("request.layer_norm.init.{width}")),
                    transaction_id: id(format!("transaction.layer_norm.init.{width}")),
                    static_policy: StaticInitializationPolicy::new(128 << 10, 8).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap_or_else(|e| panic!("LayerNorm init: {e:?}"));
        for nonfinite in [false, true] {
            let x = input(width as usize, nonfinite);
            let expected = oracle(&x, &gamma, &beta);
            let raw = bytes(&x);
            let mut allocation = account
                .allocate_zeroed(HostMemoryClass::InputPreprocess, raw.len())
                .unwrap();
            allocation.as_mut().copy_from_slice(&raw);
            let input = ManagedHostTensor::new(
                id(X),
                ResolvedTensorSpec::new(
                    vec![2, width],
                    ElementType::F32,
                    ResolvedTensorLayout::Contiguous,
                )
                .unwrap(),
                allocation.freeze(),
            )
            .unwrap();
            let output = session
                .execute(
                    id(format!("run.layer_norm.{width}.{nonfinite}")),
                    id(format!("request.layer_norm.{width}.{nonfinite}")),
                    vec![input],
                )
                .unwrap_or_else(|e| panic!("LayerNorm execute: {e:?}"));
            assert_eq!(output[0].tensor().dimensions(), [2, width]);
            assert_eq!(output[0].tensor().element_type(), ElementType::F32);
            let actual = output[0]
                .bytes()
                .bytes()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            for (i, (&got, &expected)) in actual.iter().zip(&expected).enumerate() {
                if expected.is_nan() {
                    assert!(got.is_nan());
                } else {
                    assert!(
                        (got - expected).abs() <= 3.0e-6 * expected.abs().max(1.0),
                        "width{width} c{i}: {got} != {expected}"
                    );
                }
            }
            if !nonfinite && width % 4 == 0 {
                // 1.5*(1+2^-23) rounds to 1.5+2^-22, then cancels.
                // A fused affine instead returns -2^-24, and must fail here.
                assert_eq!(actual[width as usize - 1].to_bits(), 0.0f32.to_bits());
                assert_eq!(
                    1.5f32.mul_add(gamma[width as usize - 1], beta[width as usize - 1]),
                    -f32::from_bits(0x33800000)
                );
                assert!(actual
                    .iter()
                    .any(|v| v.is_finite() && half::bf16::from_f32(*v).to_f32() != *v));
            }
        }
        assert!(matches!(
            session
                .close()
                .unwrap_or_else(|f| panic!("LayerNorm close: {:?}", f.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    }
}
