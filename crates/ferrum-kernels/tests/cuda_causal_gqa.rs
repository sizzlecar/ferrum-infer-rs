#![cfg(feature = "cuda")]
//! Real vendor causal GQA against an independent F64 oracle on finite BF16
//! inputs. The 0.04 absolute bound is declared for this fixture only.
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_tensor_runtime_config, cuda_vnext_tensor_providers_with_cudnn_causal_gqa},
    vnext_runtime::{
        CudaCudnnConfig, CudaDeviceRuntime, CudaTensorRuntimeConfig, CudnnSdpaPlanPolicy,
    },
};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    sync::Arc,
};
const Q: &str = "value.test.q";
const K: &str = "value.test.k";
const V: &str = "value.test.v";
const Y: &str = "value.test.y";
const PROFILE: &str = "test.cudnn.causal_gqa";
const B: usize = 2;
const S: usize = 129;
const HQ: usize = 16;
const HK: usize = 2;
const D: usize = 128;
fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Config {}
struct Family {
    id: ModelFamilyId,
}
fn specs() -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
    [Q, K, V]
        .into_iter()
        .map(|name| {
            (
                id(name),
                ProgramTensorSpec {
                    dimensions: vec![(B * S) as u64, (if name == Q { HQ } else { HK } * D) as u64],
                    element_type: ElementType::Bf16,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            )
        })
        .collect()
}
fn attrs() -> BTreeMap<AttributeId, SemanticValue> {
    let mut a = BTreeMap::from([
        (id("sequence_length"), SemanticValue::Unsigned(S as u64)),
        (id("head_dim"), SemanticValue::Unsigned(D as u64)),
        (
            id("scale"),
            SemanticValue::Rational(CanonicalRational::new(1, 8).unwrap()),
        ),
    ]);
    for (name, values) in [
        ("segment_lengths", vec![S as i64]),
        ("segment_bidirectional", vec![0]),
        ("key_run_counts", vec![1; B]),
        ("key_run_lengths", vec![S as i64; B]),
        ("key_run_valid", vec![1; B]),
    ] {
        a.insert(id(name), SemanticValue::Integers(values));
    }
    a
}
impl ModelFamilyProvider for Family {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.test.gqa")])
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Config, VNextError> {
        let c = Config {};
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        _: &Config,
    ) -> Result<(), VNextError> {
        if raw != &serde_json::json!({}) {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "fixture config differs".into(),
            });
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(id("metadata.test.gqa"))
    }
    fn weights(&self, _: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
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
                boundaries: BTreeMap::from([(id(Y), ElementType::Bf16)]),
                states: vec![],
                kv_storage: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(STRUCTURED_GQA_ATTENTION_FUSED_BF16_OPERATION_ID),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: None,
                    accumulation_type: None,
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }],
            }],
            vec![id(PROFILE)],
        )
    }
    fn semantic_program(
        &self,
        _: &Config,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        ModelProgram::new(
            self.id.clone(),
            vec![id(Q), id(K), id(V)],
            vec![ProgramBlock {
                id: "block.test.gqa".into(),
                nodes: vec![ProgramNode {
                    id: id("node.test.gqa"),
                    operation_id: id(STRUCTURED_GQA_ATTENTION_FUSED_BF16_OPERATION_ID),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![id(Q), id(K), id(V)],
                    outputs: vec![id(Y)],
                    attributes: attrs(),
                }],
            }],
            vec![],
            vec![],
            vec![id(Y)],
        )?
        .with_input_tensor_specs(specs())
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn data(trial: usize) -> [Vec<f32>; 3] {
    let mut q = vec![0.; B * S * HQ * D];
    let mut k = vec![0.; B * S * HK * D];
    let mut v = k.clone();
    for b in 0..B {
        for r in 0..S {
            for h in 0..HQ {
                for d in 0..D {
                    q[((b * S + r) * HQ + h) * D + d] =
                        ((r + 3 * h + d) % 7) as f32 / 32. - 0.09375;
                }
            }
            for h in 0..HK {
                for d in 0..D {
                    let i = ((b * S + r) * HK + h) * D + d;
                    k[i] = ((r * 3 + h + d) % 9) as f32 / 32. - 0.125;
                    v[i] = ((r * 5 + h + d) % 13) as f32 / 16.
                        + h as f32
                        + b as f32 * 2.
                        + if b == 1 { trial as f32 } else { 0. };
                }
            }
        }
    }
    [q, k, v]
}
fn oracle(x: &[Vec<f32>; 3]) -> Vec<f64> {
    let [q, k, v] = x;
    let mut out = vec![0.; q.len()];
    for b in 0..B {
        for r in 0..S {
            for h in 0..HQ {
                let kh = h / (HQ / HK);
                let qi = ((b * S + r) * HQ + h) * D;
                let scores: Vec<_> = (0..=r)
                    .map(|j| {
                        let ki = ((b * S + j) * HK + kh) * D;
                        (0..D)
                            .map(|d| f64::from(q[qi + d]) * f64::from(k[ki + d]))
                            .sum::<f64>()
                            / 8.
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let p: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
                let z = p.iter().sum::<f64>();
                for d in 0..D {
                    out[qi + d] = (0..=r)
                        .map(|j| p[j] * f64::from(v[((b * S + j) * HK + kh) * D + d]))
                        .sum::<f64>()
                        / z;
                }
            }
        }
    }
    out
}

#[test]
#[ignore = "requires CUDA and explicit cuDNN9.20 library; actual managed causal GQA"]
fn compact_causal_gqa_matches_independent_oracle_and_retires() {
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "test.cudnn.gqa.explicit".into(),
        })
        .unwrap(),
    );
    let config = CudaCudnnConfig {
        library_path: PathBuf::from(
            std::env::var_os("FERRUM_TEST_CUDNN_LIBRARY").expect("cuDNN library required"),
        ),
        dependency_libraries: std::env::var_os("FERRUM_TEST_CUDNN_DEPENDENCIES")
            .map(|p| std::env::split_paths(&p).collect())
            .unwrap_or_default(),
        opaque_memory_bytes: 128 << 20,
        workspace_limit_bytes: 64 << 20,
        opaque_memory_profile: "test.cudnn.gqa.owner".into(),
    };
    let providers = cuda_vnext_tensor_providers_with_cudnn_causal_gqa(
        &runtime,
        &config,
        CudnnSdpaPlanPolicy {
            maximum_plans: 4,
            maximum_live_bindings: 4,
            opaque_memory_bytes: 32 << 20,
            workspace_limit_bytes: 64 << 20,
        },
        &[],
    )
    .unwrap();
    let registry = OperationRuntimeRegistry::new(
        vec![
            Box::new(structured_gqa_attention_fused_bf16_contract().unwrap()),
            Box::new(structured_attention_fused_bf16_contract().unwrap()),
            Box::new(conv2d_bf16_contract().unwrap()),
        ],
        providers,
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.test.gqa"),
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )
    .unwrap();
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(d.clone(), vec![engine])
                .unwrap(),
        )
        .unwrap();
    let family = TypedFamilyRegistration::new(Family {
        id: id("family.test.cudnn_gqa"),
    })
    .prepare_with_profile(&serde_json::json!({}), &id(PROFILE))
    .unwrap();
    let mut options = ProgramPlanCompileOptions::new(specs()).unwrap();
    options
        .insert_tensor_spec(id(Y), specs().remove(&id(Q)).unwrap())
        .unwrap();
    let policy = TensorComponentPolicy::new(
        1 << 30,
        d.dynamic_storage_profiles.iter().copied().collect(),
    )
    .unwrap();
    let component = CompiledTensorComponent::compile(
        family,
        &registry,
        &catalog,
        &policy,
        &materializers,
        &options,
    )
    .unwrap();
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            32 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 16 << 20,
                transfer_staging_bytes: 8 << 20,
                output_bytes: 8 << 20,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let mut session = component
        .initialize(
            runtime.clone(),
            host.clone(),
            ComponentWeightSource::None,
            ComponentInitializationConfig {
                run_id: id("run.gqa.init"),
                request_id: id("request.gqa.init"),
                transaction_id: id("transaction.gqa.init"),
                static_policy: StaticInitializationPolicy::new(1 << 20, 8).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    let mut old: Option<Vec<f32>> = None;
    let mut held = None;
    let mut max_abs = 0f64;
    for trial in 0..2 {
        let x = data(trial);
        let want = oracle(&x);
        let inputs = [Q, K, V]
            .into_iter()
            .zip(&x)
            .map(|(name, values)| {
                let mut bytes = host
                    .allocate_zeroed(HostMemoryClass::InputPreprocess, values.len() * 2)
                    .unwrap();
                for (dst, value) in bytes.as_mut().chunks_exact_mut(2).zip(values) {
                    dst.copy_from_slice(&half::bf16::from_f32(*value).to_bits().to_le_bytes());
                }
                let s = specs().remove(&id(name)).unwrap();
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
                id(format!("run.gqa.{trial}")),
                id(format!("request.gqa.{trial}")),
                inputs,
            )
            .unwrap();
        let output = &outputs[0];
        assert!(host.owns(output.bytes()));
        let got: Vec<_> = output
            .bytes()
            .bytes()
            .chunks_exact(2)
            .map(|b| half::bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect();
        for (i, (a, e)) in got.iter().zip(want).enumerate() {
            let err = (f64::from(*a) - e).abs();
            max_abs = max_abs.max(err);
            assert!(
                a.is_finite() && err <= 0.04,
                "trial{trial} index{i}: {a} vs{e} err{err}"
            );
        }
        if let Some(previous) = old {
            assert_eq!(&got[..S * HQ * D], &previous[..S * HQ * D]);
            assert_ne!(&got[S * HQ * D..], &previous[S * HQ * D..]);
        }
        old = Some(got);
        held = Some(output.bytes().clone());
        drop(outputs);
    }
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|e| panic!("close {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert!(held.as_ref().unwrap().bytes().len() > 0);
    drop(held);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    eprintln!("causal GQA B2 S129 Hq16 Hkv2 D128 two runs: max_abs={max_abs}; Host retired=0");
}
