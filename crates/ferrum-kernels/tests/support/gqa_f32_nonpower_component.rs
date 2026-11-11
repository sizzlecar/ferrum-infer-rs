//! Actual F32 GQA execution on original compact KV, shared by Reference/CUDA.
//! Oracle scores and softmax are independent F64 mathematics on dyadic inputs;
//! the explicit D72 cancellation also distinguishes adjacent from midpoint trees.
//! The 2e-6 relative/absolute bound applies only to these finite fixtures, not
//! to arbitrary inputs, exponential libraries, or Torch binary parity.
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

pub fn id<T: TryFrom<String>>(name: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(name.into()).unwrap()
}
const Q: &str = "value.fixture.gqa_f32.query";
const K: &str = "value.fixture.gqa_f32.key";
const V: &str = "value.fixture.gqa_f32.value";
const Y: &str = "value.fixture.gqa_f32.output";
const PROFILE: &str = "fixture.gqa_f32.general_width.v1";
const BATCH: usize = 2;
const SEQUENCE: usize = 3;
const Q_HEADS: usize = 4;
const KV_HEADS: usize = 2;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    width: u64,
}
fn spec(width: u64) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: vec![(BATCH * SEQUENCE) as u64, width],
        element_type: ElementType::F32,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
impl Config {
    fn specs(&self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        BTreeMap::from([
            (id(Q), spec(self.width * Q_HEADS as u64)),
            (id(K), spec(self.width * KV_HEADS as u64)),
            (id(V), spec(self.width * KV_HEADS as u64)),
        ])
    }
    fn attrs(&self) -> BTreeMap<AttributeId, SemanticValue> {
        let mut a = BTreeMap::from([
            (id("head_dim"), SemanticValue::Unsigned(self.width)),
            (
                id("sequence_length"),
                SemanticValue::Unsigned(SEQUENCE as u64),
            ),
            (
                id("scale"),
                SemanticValue::Rational(CanonicalRational::new(1, 1).unwrap()),
            ),
        ]);
        for (key, values) in [
            ("segment_lengths", vec![3]),
            ("segment_bidirectional", vec![1]),
            ("key_run_counts", vec![3, 1]),
            ("key_run_lengths", vec![1, 1, 1, 3]),
            ("key_run_valid", vec![1, 0, 1, 1]),
        ] {
            a.insert(id(key), SemanticValue::Integers(values));
        }
        a
    }
}
struct Family {
    id: ModelFamilyId,
}
impl ModelFamilyProvider for Family {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.fixture.gqa_f32")])
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
        if serde_json::to_value(c).unwrap() != *raw || !(2..=256).contains(&c.width) {
            return Err(invalid("invalid fixture width"));
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(id("metadata.fixture.gqa_f32"))
    }
    fn weights(&self, _: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
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
                boundaries: BTreeMap::from([(id(Y), ElementType::F32)]),
                states: vec![],
                kv_storage: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(STRUCTURED_GQA_ATTENTION_F32_OPERATION_ID),
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
            vec![id(Q), id(K), id(V)],
            vec![ProgramBlock {
                id: "block.fixture.gqa_f32".into(),
                nodes: vec![ProgramNode {
                    id: id("node.fixture.gqa_f32"),
                    operation_id: id(STRUCTURED_GQA_ATTENTION_F32_OPERATION_ID),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![id(Q), id(K), id(V)],
                    outputs: vec![id(Y)],
                    attributes: c.attrs(),
                }],
            }],
            vec![],
            vec![],
            vec![id(Y)],
        )?
        .with_input_tensor_specs(c.specs())
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
fn data(width: usize, trial: usize) -> [Vec<f32>; 3] {
    let mut q = vec![0.; BATCH * SEQUENCE * Q_HEADS * width];
    let mut k = vec![0.; BATCH * SEQUENCE * KV_HEADS * width];
    let mut v = k.clone();
    for row in 0..BATCH * SEQUENCE {
        for head in 0..Q_HEADS {
            for c in 0..width {
                q[(row * Q_HEADS + head) * width + c] =
                    ((c * 5 + row * 3 + head * 7) % 9) as f32 / 32. - 4. / 32.;
            }
        }
        for head in 0..KV_HEADS {
            for c in 0..width {
                let i = (row * KV_HEADS + head) * width + c;
                k[i] = ((c * 7 + row * 3 + head * 11) % 11) as f32 / 32. - 5. / 32.;
                v[i] = (row * 3 + head * 7 + c % 7) as f32 / 16.
                    + 1. / 8192.
                    + if trial == 1 && row >= SEQUENCE {
                        4.
                    } else {
                        0.
                    };
            }
        }
    }
    // Masked request-0/key-1 must never be read, including its tail columns.
    k[KV_HEADS * width..2 * KV_HEADS * width].fill(f32::NAN);
    v[KV_HEADS * width..2 * KV_HEADS * width].fill(f32::NAN);
    if width == 72 {
        // All first-request/head-0 queries see score0=0, score2=1. A midpoint
        // tree loses the final 1; an out-of-bounds padded tree is also invalid.
        for row in 0..SEQUENCE {
            q[(row * Q_HEADS) * width..(row * Q_HEADS + 1) * width].fill(1.);
        }
        k[..width].fill(0.);
        k[2 * KV_HEADS * width..(2 * KV_HEADS + 1) * width].fill(0.);
        k[2 * KV_HEADS * width + 64..2 * KV_HEADS * width + 67]
            .copy_from_slice(&[33554432., -33554432., 1.]);
    }
    [q, k, v]
}
fn oracle(width: usize, inputs: &[Vec<f32>; 3]) -> Vec<f64> {
    let [q, k, v] = inputs;
    let mut out = vec![0.; q.len()];
    for batch in 0..BATCH {
        for row in 0..SEQUENCE {
            for head in 0..Q_HEADS {
                let kv = head / (Q_HEADS / KV_HEADS);
                let qi = ((batch * SEQUENCE + row) * Q_HEADS + head) * width;
                let keys: Vec<_> = (0..SEQUENCE)
                    .filter(|key| batch != 0 || *key != 1)
                    .collect();
                let scores: Vec<_> = keys
                    .iter()
                    .map(|&key| {
                        let ki = ((batch * SEQUENCE + key) * KV_HEADS + kv) * width;
                        (0..width)
                            .map(|c| f64::from(q[qi + c]) * f64::from(k[ki + c]))
                            .sum::<f64>()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum = weights.iter().sum::<f64>();
                for c in 0..width {
                    out[qi + c] = keys
                        .iter()
                        .zip(&weights)
                        .map(|(&key, &w)| {
                            w * f64::from(v[((batch * SEQUENCE + key) * KV_HEADS + kv) * width + c])
                        })
                        .sum::<f64>()
                        / sum;
                }
            }
        }
    }
    out
}
pub fn run<R: DeviceRuntime>(
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    catalog: &CapabilityCatalog,
    materializers: &WeightMaterializerRegistry,
) {
    // Include one- and two-lane final warps after a full-warp subtree.
    for width in [2u64, 3, 33, 64, 65, 72, 80, 96, 128, 130, 255, 256] {
        let c = Config { width };
        let raw = serde_json::to_value(&c).unwrap();
        let family = TypedFamilyRegistration::new(Family {
            id: id("family.fixture.gqa_f32"),
        })
        .prepare_with_profile(&raw, &id(PROFILE))
        .unwrap();
        let mut options = ProgramPlanCompileOptions::new(c.specs()).unwrap();
        options
            .insert_tensor_spec(id(Y), spec(width * Q_HEADS as u64))
            .unwrap();
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
            catalog,
            &policy,
            materializers,
            &options,
        )
        .unwrap();
        let host = HostMemoryAccount::with_allocation_alignment(
            HostMemoryLimits::new(
                4 << 20,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 1 << 20,
                    transfer_staging_bytes: 1 << 20,
                    output_bytes: 1 << 20,
                    weight_backing_bytes: 0,
                },
            )
            .unwrap(),
            64,
        )
        .unwrap();
        let mut session = component
            .initialize(
                Arc::clone(&runtime),
                host.clone(),
                ComponentWeightSource::None,
                ComponentInitializationConfig {
                    run_id: id("run.gqa_f32.init"),
                    request_id: id("request.gqa_f32.init"),
                    transaction_id: id("transaction.gqa_f32.init"),
                    static_policy: StaticInitializationPolicy::new(1 << 20, 8).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap();
        let mut previous: Option<Vec<f32>> = None;
        let mut held = None;
        for trial in 0..2 {
            let inputs = data(width as usize, trial);
            let expected = oracle(width as usize, &inputs);
            let managed = [Q, K, V]
                .into_iter()
                .zip(&inputs)
                .map(|(name, values)| {
                    let mut bytes = host
                        .allocate_zeroed(HostMemoryClass::InputPreprocess, values.len() * 4)
                        .unwrap();
                    for (out, value) in bytes.as_mut().chunks_exact_mut(4).zip(values) {
                        out.copy_from_slice(&value.to_le_bytes());
                    }
                    let s = c.specs().remove(&id(name)).unwrap();
                    ManagedHostTensor::new(
                        id(name),
                        ResolvedTensorSpec::new(s.dimensions, s.element_type, s.layout).unwrap(),
                        bytes.freeze(),
                    )
                    .unwrap()
                })
                .collect();
            let outputs = session
                .execute(
                    id(format!("run.gqa_f32.{width}.{trial}")),
                    id(format!("request.gqa_f32.{width}.{trial}")),
                    managed,
                )
                .unwrap();
            let [output] = outputs.as_slice() else {
                panic!("one GQA output")
            };
            assert!(host.owns(output.bytes()));
            assert_eq!(output.tensor().dimensions(), [6, 4 * width]);
            assert_eq!(output.tensor().element_type(), ElementType::F32);
            let actual: Vec<_> = output
                .bytes()
                .bytes()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            for (i, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
                let bound = 2e-6 * f64::max(1., want.abs());
                assert!(
                    got.is_finite() && (f64::from(got) - want).abs() <= bound,
                    "D{width} trial{trial} index{i}: {got} vs F64 {want}, bound{bound}"
                );
            }
            assert!(
                actual
                    .iter()
                    .any(|x| half::bf16::from_f32(*x).to_f32() != *x),
                "F32 output retains low bits"
            );
            if let Some(old) = &previous {
                let split = SEQUENCE * Q_HEADS * width as usize;
                assert!(
                    actual[..split]
                        .iter()
                        .zip(&old[..split])
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "other request unaffected"
                );
                assert_ne!(&actual[split..], &old[split..]);
            }
            previous = Some(actual);
            if trial == 1 {
                held = Some(output.bytes().clone());
            }
            drop(outputs);
            assert_eq!(
                host.usage()
                    .unwrap()
                    .class_allocated_bytes(HostMemoryClass::InputPreprocess),
                0
            );
        }
        assert!(matches!(
            session
                .close()
                .unwrap_or_else(|e| panic!("GQA close: {:?}", e.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        assert_eq!(
            host.usage().unwrap().committed_bytes(),
            host.backing_bytes(BATCH * SEQUENCE * Q_HEADS * width as usize * 4)
                .unwrap()
        );
        drop(held);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}
