//! Real component execution against an independent F64 dense correlation.
//! Dyadic finite operands make these small sums exactly representable in F32;
//! equality here does not establish arbitrary Torch/cuDNN reduction parity.
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

const X: &str = "value.conv3d.input";
const W: &str = "value.conv3d.weight";
const B: &str = "value.conv3d.bias";
const Y: &str = "value.conv3d.output";
const GAMMA: &str = "value.conv3d.norm.gamma";
const BETA: &str = "value.conv3d.norm.beta";
const NORMALIZED: &str = "value.conv3d.normalized";
const PROFILE: &str = "numerical.conv3d.dyadic_fixture.v1";
fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
fn spec(d: &[u64]) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: d.to_vec(),
        element_type: ElementType::F32,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    input: [u64; 5],
    weight: [u64; 5],
    output: [u64; 5],
    padding: [i64; 6],
    stride: [i64; 3],
    dilation: [i64; 3],
    modes: [i64; 3],
    groups: u64,
    frame_norm_groups: Option<u64>,
}
impl Case {
    fn input_ids(&self) -> Vec<&'static str> {
        let mut ids = vec![X, W, B];
        if self.frame_norm_groups.is_some() {
            ids.extend([GAMMA, BETA]);
        }
        ids
    }
    fn inputs(&self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        let mut inputs = BTreeMap::from([
            (id(X), spec(&self.input)),
            (id(W), spec(&self.weight)),
            (id(B), spec(&[self.output[1]])),
        ]);
        if self.frame_norm_groups.is_some() {
            inputs.insert(id(GAMMA), spec(&[self.input[1]]));
            inputs.insert(id(BETA), spec(&[self.input[1]]));
        }
        inputs
    }
    fn attributes(&self) -> BTreeMap<AttributeId, SemanticValue> {
        let mut a = BTreeMap::from([(id("groups"), SemanticValue::Unsigned(self.groups))]);
        for (k, v) in [
            ("padding", self.padding.to_vec()),
            ("stride", self.stride.to_vec()),
            ("dilation", self.dilation.to_vec()),
            ("padding_modes", self.modes.to_vec()),
        ] {
            a.insert(id(k), SemanticValue::Integers(v));
        }
        a
    }
    fn geometry(&self) -> Result<Conv3dF32Geometry, VNextError> {
        let t = |d: &[u64]| {
            ResolvedTensorSpec::new(
                d.to_vec(),
                ElementType::F32,
                ResolvedTensorLayout::Contiguous,
            )
        };
        if let Some(groups) = self.frame_norm_groups {
            FrameGroupNormSiluNcthwF32Geometry::new(
                &t(&self.input)?,
                &t(&[self.input[1]])?,
                &t(&[self.input[1]])?,
                &t(&self.input)?,
                groups,
                CanonicalRational::new(1, 1_000_000)?,
            )?;
        }
        Conv3dF32Geometry::new(
            &t(&self.input)?,
            &t(&self.weight)?,
            &t(&[self.output[1]])?,
            &t(&self.output)?,
            &self.attributes(),
        )
    }
}
struct Family(ModelFamilyId);
impl ModelFamilyProvider for Family {
    type Config = Case;
    fn family_id(&self) -> &ModelFamilyId {
        &self.0
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.conv3d.fixture")])
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Case, VNextError> {
        let c = serde_json::from_value(raw.clone()).map_err(|e| invalid(e.to_string()))?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Case,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(c).unwrap() != *raw {
            return Err(invalid("config differs"));
        }
        c.geometry()?;
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Case,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(id("metadata.conv3d.fixture"))
    }
    fn weights(&self, _: &Case) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(&self, c: &Case) -> Result<FamilyNumericalProfiles, VNextError> {
        let mut boundaries = BTreeMap::from([(id(Y), ElementType::F32)]);
        let mut operations = vec![CONV3D_F32_OPERATION_ID];
        if c.frame_norm_groups.is_some() {
            boundaries.insert(id(NORMALIZED), ElementType::F32);
            operations.push(FRAME_GROUP_NORM_SILU_NCTHW_F32_OPERATION_ID);
        }
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: id(PROFILE),
                family_id: self.0.clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(Y),
                boundaries,
                states: vec![],
                kv_storage: vec![],
                operations: operations
                    .into_iter()
                    .map(|op| NumericalOperationContract {
                        operation_id: id(op),
                        version: ContractVersion::new(1, 0),
                        multiplication_type: Some(ElementType::F32),
                        accumulation_type: Some(ElementType::F32),
                        staged_arithmetic: None,
                        composite_arithmetic: None,
                    })
                    .collect(),
            }],
            vec![id(PROFILE)],
        )
    }
    fn semantic_program(
        &self,
        c: &Case,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let mut nodes = Vec::new();
        if let Some(groups) = c.frame_norm_groups {
            nodes.push(ProgramNode {
                id: id("node.conv3d.frame_norm"),
                operation_id: id(FRAME_GROUP_NORM_SILU_NCTHW_F32_OPERATION_ID),
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::Fixed,
                inputs: vec![id(X), id(GAMMA), id(BETA)],
                outputs: vec![id(NORMALIZED)],
                attributes: BTreeMap::from([
                    (id("groups"), SemanticValue::Unsigned(groups)),
                    (
                        id("epsilon"),
                        SemanticValue::Rational(CanonicalRational::new(1, 1_000_000)?),
                    ),
                ]),
            });
        }
        nodes.push(ProgramNode {
            id: id("node.conv3d"),
            operation_id: id(CONV3D_F32_OPERATION_ID),
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs: vec![
                id(if c.frame_norm_groups.is_some() {
                    NORMALIZED
                } else {
                    X
                }),
                id(W),
                id(B),
            ],
            outputs: vec![id(Y)],
            attributes: c.attributes(),
        });
        ModelProgram::new(
            self.0.clone(),
            c.input_ids().into_iter().map(id).collect(),
            vec![ProgramBlock {
                id: "block.conv3d".into(),
                nodes,
            }],
            vec![],
            vec![],
            vec![id(Y)],
        )?
        .with_input_tensor_specs(c.inputs())
    }
    fn semantic_metadata(&self, _: &Case) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn invalid(s: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan { reason: s.into() }
}
fn cases() -> Vec<Case> {
    let mut cases = vec![
        Case {
            input: [2, 4, 5, 4, 6],
            weight: [6, 2, 3, 3, 3],
            output: [2, 6, 5, 4, 6],
            padding: [2, 0, 1, 1, 1, 1],
            stride: [1; 3],
            dilation: [1; 3],
            modes: [0, 1, 1],
            groups: 2,
            frame_norm_groups: None,
        },
        Case {
            input: [1, 2, 5, 6, 8],
            weight: [3, 2, 3, 3, 3],
            output: [1, 3, 3, 3, 4],
            padding: [2, 0, 0, 1, 0, 1],
            stride: [2; 3],
            dilation: [1; 3],
            modes: [0, 1, 1],
            groups: 1,
            frame_norm_groups: None,
        },
        Case {
            input: [1, 2, 6, 4, 5],
            weight: [2, 1, 2, 1, 2],
            output: [1, 2, 6, 4, 2],
            padding: [2, 0, 0, 0, 1, 0],
            stride: [1, 1, 2],
            dilation: [2, 1, 2],
            modes: [0, 0, 1],
            groups: 2,
            frame_norm_groups: None,
        },
        Case {
            input: [1, 2, 2, 2, 2],
            weight: [4, 2, 1, 1, 1],
            output: [1, 4, 2, 2, 2],
            padding: [0; 6],
            stride: [1; 3],
            dilation: [1; 3],
            modes: [0; 3],
            groups: 1,
            frame_norm_groups: None,
        },
    ];
    cases.push(Case {
        frame_norm_groups: Some(2),
        ..cases[0].clone()
    });
    cases
}
fn operands(c: &Case, trial: u64) -> Vec<Vec<f32>> {
    let mut sizes = vec![
        c.input.iter().product(),
        c.weight.iter().product(),
        c.output[1],
    ];
    if c.frame_norm_groups.is_some() {
        sizes.extend([c.input[1]; 2]);
    }
    sizes
        .into_iter()
        .enumerate()
        .map(|(port, size)| {
            (0..size)
                .map(|i| ((i * 7 + port as u64 * 3 + trial * 5) % 17) as f32 / 16. - 0.5)
                .collect()
        })
        .collect()
}
/// The oracle expands padding into an explicit dense volume, then computes
/// output-index/flattened-filter dots in F64. It never calls provider geometry
/// coordinate mapping, CUDA padding, or Reference arithmetic helpers.
fn oracle(c: &Case, v: &[Vec<f32>]) -> Vec<f32> {
    let normalized = c
        .frame_norm_groups
        .map(|groups| normalized_frames(c, v, groups));
    let input = normalized.as_ref().unwrap_or(&v[0]);
    let dims: [usize; 5] = c.input.map(|x| x as usize);
    let padded: [usize; 3] = std::array::from_fn(|a| {
        dims[a + 2] + c.padding[2 * a] as usize + c.padding[2 * a + 1] as usize
    });
    let volume: usize = padded.iter().product();
    let mut dense = vec![0f64; dims[0] * dims[1] * volume];
    for (i, out) in dense.iter_mut().enumerate() {
        let spatial = [
            i / (padded[1] * padded[2]) % padded[0],
            i / padded[2] % padded[1],
            i % padded[2],
        ];
        let mut source = [0usize; 3];
        let mut valid = true;
        for a in 0..3 {
            let position = spatial[a] as i64 - c.padding[2 * a];
            let n = dims[a + 2] as i64;
            if position >= 0 && position < n {
                source[a] = position as usize;
            } else if c.modes[a] == 1 {
                let p = position.rem_euclid(2 * (n - 1));
                source[a] = if p < n { p } else { 2 * (n - 1) - p } as usize;
            } else {
                valid = false;
            }
        }
        if valid {
            let channel = i / volume;
            *out = f64::from(
                input
                    [((channel * dims[2] + source[0]) * dims[3] + source[1]) * dims[4] + source[2]],
            );
        }
    }
    let [batch, outputs, od, oh, ow] = c.output.map(|x| x as usize);
    let [_, channels, kd, kh, kw] = c.weight.map(|x| x as usize);
    let output_spatial = od * oh * ow;
    let dot = channels * kd * kh * kw;
    (0..batch * outputs * output_spatial)
        .map(|index| {
            let o = index / output_spatial % outputs;
            let n = index / (outputs * output_spatial);
            let base = [index / (oh * ow) % od, index / ow % oh, index % ow];
            let mut sum = 0f64;
            for k in 0..dot {
                let channel = o / (outputs / c.groups as usize) * channels + k / (kd * kh * kw);
                let kernel = [k / (kh * kw) % kd, k / kw % kh, k % kw];
                let p: [usize; 3] = std::array::from_fn(|a| {
                    base[a] * c.stride[a] as usize + kernel[a] * c.dilation[a] as usize
                });
                sum += dense[((n * dims[1] + channel) * padded[0] + p[0]) * padded[1] * padded[2]
                    + p[1] * padded[2]
                    + p[2]]
                    * f64::from(v[1][o * dot + k]);
            }
            (f64::from(sum as f32) + f64::from(v[2][o])) as f32
        })
        .collect()
}
fn normalized_frames(c: &Case, v: &[Vec<f32>], groups: u64) -> Vec<f32> {
    let [batches, channels, frames, height, width] = c.input.map(|x| x as usize);
    let groups = groups as usize;
    let spatial = height * width;
    let per_group = channels / groups;
    let mut output = vec![0.; v[0].len()];
    for n in 0..batches {
        for t in 0..frames {
            for g in 0..groups {
                let indices: Vec<_> = (g * per_group..(g + 1) * per_group)
                    .flat_map(|channel| {
                        (0..spatial).map(move |pixel| {
                            ((n * channels + channel) * frames + t) * spatial + pixel
                        })
                    })
                    .collect();
                let mean =
                    indices.iter().map(|&i| f64::from(v[0][i])).sum::<f64>() / indices.len() as f64;
                let variance = indices
                    .iter()
                    .map(|&i| (f64::from(v[0][i]) - mean).powi(2))
                    .sum::<f64>()
                    / indices.len() as f64;
                for i in indices {
                    let channel = i / (frames * spatial) % channels;
                    let affine = (f64::from(v[0][i]) - mean) / (variance + 1e-6).sqrt()
                        * f64::from(v[3][channel])
                        + f64::from(v[4][channel]);
                    output[i] = (affine / (1. + (-affine).exp())) as f32;
                }
            }
        }
    }
    output
}
fn qualify<R: DeviceRuntime>(
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    catalog: &CapabilityCatalog,
    materializers: &WeightMaterializerRegistry,
) {
    for (case_index, c) in cases().into_iter().enumerate() {
        let family = TypedFamilyRegistration::new(Family(id("family.conv3d.fixture")))
            .prepare_with_profile(&serde_json::to_value(&c).unwrap(), &id(PROFILE))
            .unwrap();
        let mut options = ProgramPlanCompileOptions::new(c.inputs()).unwrap();
        options.insert_tensor_spec(id(Y), spec(&c.output)).unwrap();
        if c.frame_norm_groups.is_some() {
            options
                .insert_tensor_spec(id(NORMALIZED), spec(&c.input))
                .unwrap();
        }
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
        let host = HostMemoryAccount::new(
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
        );
        let mut session = component
            .initialize(
                runtime.clone(),
                host.clone(),
                ComponentWeightSource::None,
                ComponentInitializationConfig {
                    run_id: id("run.conv3d.init"),
                    request_id: id("request.conv3d.init"),
                    transaction_id: id("transaction.conv3d.init"),
                    static_policy: StaticInitializationPolicy::new(1 << 20, 8).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap();
        let mut retained = None;
        let mut previous = Vec::new();
        for trial in 0..3 {
            let mut values = operands(&c, if trial == 2 { 1 } else { trial });
            if trial == 2 {
                let [_, _, depth, height, width] = c.input;
                for (i, x) in values[0].iter_mut().enumerate() {
                    if (i as u64 / (height * width)) % depth == depth - 1 && i % 3 == 0 {
                        *x += 4.;
                    }
                }
            }
            let expected = oracle(&c, &values);
            let inputs = c
                .input_ids()
                .into_iter()
                .zip(&values)
                .map(|(name, v)| {
                    let mut bytes = host
                        .allocate_zeroed(HostMemoryClass::InputPreprocess, v.len() * 4)
                        .unwrap();
                    for (dst, x) in bytes.as_mut().chunks_exact_mut(4).zip(v) {
                        dst.copy_from_slice(&x.to_le_bytes());
                    }
                    let t = c.inputs().remove(&id(name)).unwrap();
                    ManagedHostTensor::new(
                        id(name),
                        ResolvedTensorSpec::new(t.dimensions, t.element_type, t.layout).unwrap(),
                        bytes.freeze(),
                    )
                    .unwrap()
                })
                .collect();
            let outputs = session
                .execute(
                    id(format!("run.conv3d.{case_index}.{trial}")),
                    id(format!("request.conv3d.{case_index}.{trial}")),
                    inputs,
                )
                .unwrap();
            assert_eq!(outputs.len(), 1);
            assert!(host.owns(outputs[0].bytes()));
            assert_eq!(outputs[0].tensor().dimensions(), c.output);
            let actual: Vec<_> = outputs[0]
                .bytes()
                .bytes()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            if c.frame_norm_groups.is_none() {
                assert_eq!(actual, expected, "case {case_index}, trial {trial}");
            } else {
                // Declared fixture tolerance includes F32 statistics/exp and
                // convolution rounding versus the independent F64 pipeline.
                for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
                    assert!(
                        a.is_finite() && (a - e).abs() <= 3e-5 * (1. + e.abs()),
                        "normalized case {case_index}/{trial}/{i}: {a} vs {e}"
                    );
                }
            }
            if trial == 2 {
                let [_, _, _, h, w] = c.output;
                for (i, &v) in actual.iter().enumerate() {
                    let time = i as u64 / (h * w) % c.output[2];
                    if time * (c.stride[0] as u64) < c.input[2] - 1 {
                        assert_eq!(v, previous[i], "future leaked: {case_index}/{i}");
                    }
                }
                assert_ne!(actual, previous, "last frame was not consumed");
            }
            previous = actual;
            retained = Some(outputs);
        }
        assert!(matches!(
            session
                .close()
                .unwrap_or_else(|e| panic!("close: {:?}", e.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        assert!(!retained.as_ref().unwrap()[0].bytes().bytes().is_empty());
        drop((retained, component));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        eprintln!(
            "Conv3D {case_index}: {:?} -> {:?}, independent F64 correlation and causality, closed",
            c.input, c.output
        );
    }
}

#[test]
fn managed_reference_conv3d_matches_independent_volume_and_keeps_causality() {
    use ferrum_kernels::backend::reference::ReferenceVNextComposition;
    let c = ReferenceVNextComposition::create(id("device.reference.conv3d.fixture")).unwrap();
    qualify(
        c.runtime().clone(),
        c.registry(),
        c.catalog(),
        c.weight_materializers(),
    );
    assert_eq!(c.runtime().snapshot().live_allocations, 0);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires exclusive CUDA and explicit cuDNN library/dependencies"]
fn managed_cuda_conv3d_matches_independent_volume_and_keeps_causality() {
    use ferrum_kernels::backend::cuda::{
        vnext_ops::{
            cuda_tensor_runtime_config, cuda_vnext_tensor_providers_with_cudnn_selection,
            CudaCudnnOperation, CudaTensorOperation,
        },
        vnext_runtime::{CudaCudnnConfig, CudaDeviceRuntime, CudaTensorRuntimeConfig},
    };
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 2,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 32,
            opaque_memory_profile: "unqualified-conv3d-fixture-v1".into(),
        })
        .unwrap(),
    );
    let config = CudaCudnnConfig {
        library_path: std::env::var_os("FERRUM_TEST_CUDNN_LIBRARY")
            .expect("explicit library")
            .into(),
        dependency_libraries: std::env::var_os("FERRUM_TEST_CUDNN_DEPENDENCIES")
            .map(|p| std::env::split_paths(&p).collect())
            .unwrap_or_default(),
        opaque_memory_bytes: 256 << 20,
        workspace_limit_bytes: 64 << 20,
        opaque_memory_profile: "fixture.conv3d.f32.fma.v1".into(),
    };
    let registry = OperationRuntimeRegistry::new(
        vec![
            Box::new(conv3d_f32_contract().unwrap()),
            Box::new(frame_group_norm_silu_ncthw_f32_contract().unwrap()),
        ],
        cuda_vnext_tensor_providers_with_cudnn_selection(
            &runtime,
            &config,
            None,
            &[CudaCudnnOperation::Conv3dF32],
            &[CudaTensorOperation::FrameGroupNormSiluNcthwF32],
        )
        .unwrap(),
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let device = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.conv3d.fixture"),
        ContractVersion::new(1, 0),
        device.runtime_implementation_fingerprint.clone(),
        device.id.clone(),
        device.capabilities.clone(),
    )
    .unwrap();
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(device.clone(), vec![engine])
                .unwrap(),
        )
        .unwrap();
    qualify(runtime, &registry, &catalog, &materializers);
}
