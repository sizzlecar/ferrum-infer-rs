#![cfg(feature = "cuda")]
//! Real managed cuDNN backward-data, compared with independent F64 adjoint scatter.
//! Small dyadic operands make all sums exact here; this is not general Torch parity.
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_tensor_runtime_config, cuda_vnext_cnn_f32_with_transpose_providers},
    vnext_runtime::{CudaCudnnConfig, CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

const X: &str = "value.transpose.input";
const W: &str = "value.transpose.weight";
const B: &str = "value.transpose.bias";
const Y: &str = "value.transpose.output";
const PROFILE: &str = "numerical.transpose.scatter_fixture.v1";
fn id<T: TryFrom<String>>(value: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(value.into()).unwrap()
}
fn spec(dimensions: &[u64]) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: dimensions.to_vec(),
        element_type: ElementType::F32,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    input: [u64; 4],
    weight: [u64; 4],
    stride: [i64; 2],
    padding: [i64; 2],
    dilation: [i64; 2],
    output_padding: [i64; 2],
    groups: u64,
}
impl Case {
    fn output(&self) -> [u64; 4] {
        let mut result = [self.input[0], self.weight[1] * self.groups, 0, 0];
        for axis in 0..2 {
            result[axis + 2] = ((self.input[axis + 2] as i64 - 1) * self.stride[axis]
                - 2 * self.padding[axis]
                + (self.weight[axis + 2] as i64 - 1) * self.dilation[axis]
                + self.output_padding[axis]
                + 1) as u64;
        }
        result
    }
    fn inputs(&self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        BTreeMap::from([
            (id(X), spec(&self.input)),
            (id(W), spec(&self.weight)),
            (id(B), spec(&[self.output()[1]])),
        ])
    }
    fn attributes(&self) -> BTreeMap<AttributeId, SemanticValue> {
        let mut attrs = BTreeMap::from([(id("groups"), SemanticValue::Unsigned(self.groups))]);
        for (name, values) in [
            ("stride", self.stride),
            ("padding", self.padding),
            ("dilation", self.dilation),
            ("output_padding", self.output_padding),
        ] {
            attrs.insert(id(name), SemanticValue::Integers(values.to_vec()));
        }
        attrs
    }
    fn geometry(&self) -> Result<ConvTranspose2dF32Geometry, VNextError> {
        let resolved = |dims: &[u64]| {
            ResolvedTensorSpec::new(
                dims.to_vec(),
                ElementType::F32,
                ResolvedTensorLayout::Contiguous,
            )
        };
        ConvTranspose2dF32Geometry::new(
            &resolved(&self.input)?,
            &resolved(&self.weight)?,
            &resolved(&[self.output()[1]])?,
            &resolved(&self.output())?,
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
        BTreeSet::from([id("metadata.transpose.fixture")])
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Case, VNextError> {
        let c: Case = serde_json::from_value(raw.clone()).map_err(|e| invalid(e.to_string()))?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Case,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(c).unwrap() != *raw {
            return Err(invalid("configuration differs"));
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
        Ok(id("metadata.transpose.fixture"))
    }
    fn weights(&self, _: &Case) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(&self, _: &Case) -> Result<FamilyNumericalProfiles, VNextError> {
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
                    operation_id: id(CONV_TRANSPOSE2D_F32_OPERATION_ID),
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
        c: &Case,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        ModelProgram::new(
            self.family_id().clone(),
            vec![id(X), id(W), id(B)],
            vec![ProgramBlock {
                id: "block.transpose".into(),
                nodes: vec![ProgramNode {
                    id: id("node.transpose"),
                    operation_id: id(CONV_TRANSPOSE2D_F32_OPERATION_ID),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![id(X), id(W), id(B)],
                    outputs: vec![id(Y)],
                    attributes: c.attributes(),
                }],
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
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
fn data(g: ConvTranspose2dF32Geometry, trial: u64) -> [Vec<f32>; 3] {
    std::array::from_fn(|port| {
        (0..g.input_elements(port).unwrap())
            .map(|i| ((i * 7 + port as u64 * 3 + trial * 5) % 17) as f32 / 16. - 0.5)
            .collect()
    })
}
fn scatter(g: ConvTranspose2dF32Geometry, values: &[Vec<f32>; 3]) -> Vec<f32> {
    let [n, ci, ih, iw] = g.input();
    let [_, co, oh, ow] = g.output();
    let [_, cog, kh, kw] = g.weight();
    let [ph, pw] = g.padding();
    let [sh, sw] = g.stride();
    let [dh, dw] = g.dilation();
    let mut output = vec![0f64; g.elements() as usize];
    for batch in 0..n {
        for channel in 0..ci {
            for y in 0..ih {
                for x in 0..iw {
                    let value =
                        f64::from(values[0][(((batch * ci + channel) * ih + y) * iw + x) as usize]);
                    for oc in 0..cog {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let oy = (y * sh + ky * dh) as i64 - ph as i64;
                                let ox = (x * sw + kx * dw) as i64 - pw as i64;
                                if oy >= 0 && ox >= 0 && oy < oh as i64 && ox < ow as i64 {
                                    let o = channel / (ci / g.groups()) * cog + oc;
                                    output[(((batch * co + o) * oh + oy as u64) * ow + ox as u64)
                                        as usize] += value
                                        * f64::from(
                                            values[1][(((channel * cog + oc) * kh + ky) * kw + kx)
                                                as usize],
                                        );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    output
        .into_iter()
        .enumerate()
        .map(|(i, sum)| {
            // cuDNN returns F32 before the separately launched F32 bias add.
            (f64::from(sum as f32) + f64::from(values[2][i / (oh * ow) as usize % co as usize]))
                as f32
        })
        .collect()
}

#[test]
#[ignore = "requires exclusive CUDA and explicit cuDNN library/dependencies"]
fn managed_transpose_matches_adjoint_scatter_for_audio_strides_groups_and_output_padding() {
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 2,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 32,
            opaque_memory_profile: "unqualified-transpose-scatter-fixture-v1".into(),
        })
        .unwrap(),
    );
    let config = CudaCudnnConfig {
        library_path: std::env::var_os("FERRUM_TEST_CUDNN_LIBRARY")
            .expect("explicit cuDNN library")
            .into(),
        dependency_libraries: std::env::var_os("FERRUM_TEST_CUDNN_DEPENDENCIES")
            .map(|paths| std::env::split_paths(&paths).collect())
            .unwrap_or_default(),
        opaque_memory_bytes: 256 << 20,
        workspace_limit_bytes: 64 << 20,
        opaque_memory_profile: "fixture.conv_transpose.f32.fma.v1".into(),
    };
    let registry = OperationRuntimeRegistry::new(
        vec![
            Box::new(conv2d_f32_contract().unwrap()),
            Box::new(conv_transpose2d_f32_contract().unwrap()),
        ],
        cuda_vnext_cnn_f32_with_transpose_providers(&runtime, &config, &[]).unwrap(),
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let device = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.transpose.fixture"),
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
    let cases = [
        Case {
            input: [2, 4, 1, 3],
            weight: [4, 3, 1, 9],
            stride: [1, 5],
            padding: [0, 2],
            dilation: [1, 1],
            output_padding: [0, 1],
            groups: 2,
        },
        Case {
            input: [1, 2, 1, 4],
            weight: [2, 3, 1, 4],
            stride: [1, 2],
            padding: [0, 1],
            dilation: [1, 1],
            output_padding: [0, 0],
            groups: 1,
        },
        Case {
            input: [2, 4, 3, 2],
            weight: [4, 2, 3, 2],
            stride: [2, 3],
            padding: [1, 1],
            dilation: [2, 1],
            output_padding: [1, 2],
            groups: 2,
        },
    ];
    for (case_index, c) in cases.into_iter().enumerate() {
        let g = c.geometry().unwrap();
        let family = TypedFamilyRegistration::new(Family(id("family.transpose.fixture")))
            .prepare_with_profile(&serde_json::to_value(&c).unwrap(), &id(PROFILE))
            .unwrap();
        let mut options = ProgramPlanCompileOptions::new(c.inputs()).unwrap();
        options
            .insert_tensor_spec(id(Y), spec(&c.output()))
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
            &registry,
            &catalog,
            &policy,
            &materializers,
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
                    run_id: id("run.transpose.init"),
                    request_id: id("request.transpose.init"),
                    transaction_id: id("transaction.transpose.init"),
                    static_policy: StaticInitializationPolicy::new(1 << 20, 8).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap();
        let mut retained = None;
        for trial in 0..2 {
            let values = data(g, trial);
            let expected = scatter(g, &values);
            let inputs = [X, W, B]
                .into_iter()
                .zip(&values)
                .map(|(name, values)| {
                    let mut bytes = host
                        .allocate_zeroed(HostMemoryClass::InputPreprocess, values.len() * 4)
                        .unwrap();
                    for (out, value) in bytes.as_mut().chunks_exact_mut(4).zip(values) {
                        out.copy_from_slice(&value.to_le_bytes());
                    }
                    let tensor = c.inputs().remove(&id(name)).unwrap();
                    ManagedHostTensor::new(
                        id(name),
                        ResolvedTensorSpec::new(
                            tensor.dimensions,
                            tensor.element_type,
                            tensor.layout,
                        )
                        .unwrap(),
                        bytes.freeze(),
                    )
                    .unwrap()
                })
                .collect();
            let outputs = session
                .execute(
                    id(format!("run.transpose.{case_index}.{trial}")),
                    id(format!("request.transpose.{case_index}.{trial}")),
                    inputs,
                )
                .unwrap();
            assert_eq!(outputs.len(), 1);
            assert!(host.owns(outputs[0].bytes()));
            assert_eq!(outputs[0].tensor().dimensions(), c.output());
            let actual = outputs[0]
                .bytes()
                .bytes()
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "case {case_index}, trial {trial}");
            retained = Some(outputs);
        }
        assert!(matches!(
            session
                .close()
                .unwrap_or_else(|e| panic!("close: {:?}", e.failure())),
            PlanRuntimeCloseOutcome::Closed(_)
        ));
        assert!(retained.as_ref().unwrap()[0].bytes().bytes().len() > 0);
        drop((retained, component));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        eprintln!(
            "transpose case {case_index}: {:?} -> {:?}, exact independent scatter, closed",
            g.input(),
            g.output()
        );
    }
}
