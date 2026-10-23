use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::*;

fn hf_config() -> Value {
    json!({
        "_class_name": "QwenImage21Transformer2DModel",
        "num_layers": 1,
        "num_attention_heads": 2,
        "attention_head_dim": 6,
        "context_in_dim": 8,
        "in_channels": 3,
        "out_channels": 4,
        "patch_size": 1,
        "mlp_ratio": 3,
        "axes_dims_rope": [2, 2, 2],
        "causal_condition": true,
        "eps": 0.000001
    })
}

fn config_from(raw: &Value) -> QwenImage21Config {
    QwenImage21Config::from_hf_json(
        &serde_json::to_vec(raw).unwrap(),
        StructureLimits::default(),
    )
    .unwrap()
}

fn registration() -> TypedFamilyRegistration<QwenImage21ImageProjectionFamily> {
    TypedFamilyRegistration::new(QwenImage21ImageProjectionFamily::new().unwrap())
}

fn profile_id() -> NumericalProfileId {
    NumericalProfileId::new(IMAGE_PROJECTION_PROFILE_ID).unwrap()
}

fn prepare(config: &QwenImage21Config) -> PreparedModelFamily {
    registration()
        .prepare_with_profile(&serde_json::to_value(config).unwrap(), &profile_id())
        .unwrap()
}

#[test]
fn projection_definition_keeps_only_the_required_img_in_weight() {
    let mut raw = hf_config();
    raw["num_layers"] = json!(32);
    raw["num_attention_heads"] = json!(32);
    raw["attention_head_dim"] = json!(128);
    raw["context_in_dim"] = json!(4096);
    raw["in_channels"] = json!(64);
    raw["out_channels"] = json!(64);
    raw["axes_dims_rope"] = json!([16, 56, 56]);
    let config = config_from(&raw);
    let registration = registration();
    let definition = registration
        .define(&serde_json::to_value(&config).unwrap())
        .unwrap();
    let full = ExpectedWeights::from_config(&config, StructureLimits::default())
        .unwrap()
        .weight_schema()
        .unwrap();
    assert_ne!(definition.family_id().as_str(), super::super::FAMILY_ID);
    assert_eq!(
        definition.metadata(),
        &ModelSemanticMetadata::TensorComponent
    );
    let schema = definition.stored_weight_schema().unwrap();
    let [component] = schema.components.as_slice() else {
        panic!("projection must retain exactly its physical weight component")
    };
    let [tensor] = schema.tensors.as_slice() else {
        panic!("projection must retain exactly its logical weight tensor")
    };
    assert!(component.required && tensor.required);
    assert_eq!(component.external_names, ["img_in.weight"]);
    assert_eq!(component.dimensions, [4096, 64]);
    assert_eq!(component.physical_element_type(), ElementType::Bf16);
    assert_eq!(component.physical_bytes().unwrap(), 524_288);
    assert_eq!(tensor, full.tensor(&tensor.id).unwrap());
    assert_eq!(
        component,
        full.physical_component_refs(&tensor.id).unwrap()[0]
    );
    assert!(full.tensors.iter().all(|tensor| tensor.required));
    assert!(full.components.iter().all(|component| component.required));
    assert!(full.components.len() > schema.components.len());
    registration.prepare(&definition, &profile_id()).unwrap();
}

#[test]
fn prepared_projection_has_fixed_tensor_work_and_explicit_bf16_boundaries() {
    let config = config_from(&hf_config());
    let family = prepare(&config);
    let program = family.program();
    assert_eq!(family.metadata(), &ModelSemanticMetadata::TensorComponent);
    assert!(program.states().is_empty());
    assert!(program.checkpoint_inputs().is_none());
    assert_eq!(program.inputs().len(), 1);
    assert_eq!(program.inputs()[0].as_str(), IMAGE_PROJECTION_INPUT_ID);
    assert_eq!(program.outputs().len(), 1);
    assert_eq!(program.outputs()[0].as_str(), IMAGE_PROJECTION_OUTPUT_ID);
    let [block] = program.blocks() else {
        panic!("projection has one block")
    };
    let [node] = block.nodes.as_slice() else {
        panic!("projection has one operation")
    };
    assert_eq!(node.id.as_str(), IMAGE_PROJECTION_NODE_ID);
    assert_eq!(node.work, ProgramNodeWorkSpec::Fixed);
    assert_eq!(node.operation_id.as_str(), DENSE_LINEAR_BF16_OPERATION_ID);
    assert_eq!(node.required_version, ContractVersion::new(1, 0));
    assert_eq!(node.inputs.len(), 2);
    let [weight] = program.weights() else {
        panic!("projection has one weight reference")
    };
    assert_eq!(weight.weight_id.as_str(), IMAGE_PROJECTION_WEIGHT_ID);
    assert_eq!(weight.tensor.dimensions, [12, 3]);
    assert_eq!(weight.tensor.element_type, ElementType::Bf16);
    let profile = family.numerical_profile();
    assert_eq!(profile.primary_activation, program.outputs()[0]);
    assert_eq!(profile.activation_type().unwrap(), ElementType::Bf16);
    assert!(profile.states.is_empty() && profile.kv_storage.is_empty());
    let [operation] = profile.operations.as_slice() else {
        panic!("projection declares one arithmetic contract")
    };
    assert_eq!(operation.multiplication_type, Some(ElementType::F32));
    assert_eq!(operation.accumulation_type, Some(ElementType::F32));
    assert!(operation.staged_arithmetic.is_none());
    assert!(operation.composite_arithmetic.is_none());
    assert!(family.canonical_config().get("rows").is_none());
}

#[test]
fn registration_requires_checked_canonical_config_and_declared_profile() {
    let config = config_from(&hf_config());
    let registration = registration();
    let canonical = serde_json::to_value(&config).unwrap();
    assert!(registration.define(&hf_config()).is_err());
    let mut unknown = canonical.clone();
    unknown["rows"] = json!(4);
    assert!(registration.define(&unknown).is_err());
    let mut invalid = canonical.clone();
    invalid["in_channels"] = json!(0);
    assert!(registration.define(&invalid).is_err());
    let definition = registration.define(&canonical).unwrap();
    assert!(registration
        .prepare(
            &definition,
            &NumericalProfileId::new("unknown.profile").unwrap()
        )
        .is_err());
    let prepared = registration.prepare(&definition, &profile_id()).unwrap();
    let mut altered = prepared.numerical_profile().clone();
    altered.operations[0].accumulation_type = Some(ElementType::Bf16);
    assert!(QwenImage21ImageProjectionFamily::new()
        .unwrap()
        .semantic_program(&config, &altered)
        .is_err());
}

#[test]
fn cuda_profile_is_explicit_and_preserves_the_same_no_bias_program() {
    let config = config_from(&hf_config());
    let provider = QwenImage21ImageProjectionFamily::new().unwrap();
    let profiles = provider.numerical_profiles(&config).unwrap();
    assert_eq!(profiles.auto_preference(), &[profile_id()]);
    let cuda_id =
        NumericalProfileId::new(QwenImage21ImageProjectionFamily::CUDA_PROFILE_ID).unwrap();
    assert_ne!(cuda_id, profile_id());
    let registration = registration();
    let definition = registration
        .define(&serde_json::to_value(&config).unwrap())
        .unwrap();
    let cuda = registration.prepare(&definition, &cuda_id).unwrap();
    let reference = registration.prepare(&definition, &profile_id()).unwrap();
    assert_eq!(cuda.program(), reference.program());
    assert_eq!(cuda.weights(), reference.weights());
    assert_eq!(
        cuda.numerical_profile().operations,
        reference.numerical_profile().operations
    );
    assert_eq!(cuda.numerical_profile().id, cuda_id);
    assert_ne!(
        cuda.fingerprint().unwrap(),
        reference.fingerprint().unwrap()
    );
    assert_eq!(cuda.program().weights().len(), 1);
    assert_eq!(cuda.program().blocks()[0].nodes[0].inputs.len(), 2);
}

/// Deliberately carries every required transformer weight into a projection.
/// Preparation must reject this without weakening the core schema contract.
struct FullSchemaProjection(QwenImage21ImageProjectionFamily);

impl ModelFamilyProvider for FullSchemaProjection {
    type Config = QwenImage21Config;

    fn family_id(&self) -> &ModelFamilyId {
        self.0.family_id()
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        self.0.external_metadata_ids()
    }
    fn validate_config_identity(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<(), VNextError> {
        self.0.validate_config_identity(raw, config)
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.0.validated_external_metadata_id(raw, config)
    }
    fn parse_config(&self, raw: &Value) -> Result<Self::Config, VNextError> {
        self.0.parse_config(raw)
    }
    fn weights(&self, config: &Self::Config) -> Result<ModelWeights, VNextError> {
        ExpectedWeights::from_config(config, StructureLimits::default())
            .and_then(|expected| expected.weight_schema())
            .map_err(|error| invalid_config("weight_schema", error.to_string()))
            .map(ModelWeights::Stored)
    }
    fn numerical_profiles(
        &self,
        config: &Self::Config,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        self.0.numerical_profiles(config)
    }
    fn semantic_program(
        &self,
        config: &Self::Config,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        self.0.semantic_program(config, profile)
    }
    fn semantic_metadata(
        &self,
        config: &Self::Config,
    ) -> Result<ModelSemanticMetadata, VNextError> {
        self.0.semantic_metadata(config)
    }
}

#[test]
fn complete_required_schema_cannot_prepare_a_projection_only_program() {
    let config = config_from(&hf_config());
    let registration = TypedFamilyRegistration::new(FullSchemaProjection(
        QwenImage21ImageProjectionFamily::new().unwrap(),
    ));
    let definition = registration
        .define(&serde_json::to_value(config).unwrap())
        .unwrap();
    let error = registration
        .prepare(&definition, &profile_id())
        .unwrap_err();
    assert!(matches!(
        error,
        VNextError::InvalidModelConfig { field, reason, .. }
            if field == "program.weights" && reason.contains("required weight")
    ));
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TensorPolicy {
    capacity: u64,
    profiles: Vec<DynamicStorageProfile>,
}

impl RuntimePolicy for TensorPolicy {
    fn version(&self) -> ContractVersion {
        ContractVersion::new(1, 0)
    }
    fn memory_capacity_bytes(&self) -> u64 {
        self.capacity
    }
    fn memory_reserve_bytes(&self) -> u64 {
        0
    }
    fn maximum_active_sequences(&self) -> u32 {
        1
    }
    fn work_limits(&self) -> RuntimeWorkLimits {
        RuntimeWorkLimits::FixedTensor
    }
    fn attention_execution_policy(&self) -> ferrum_types::AttentionExecutionPolicy {
        ferrum_types::AttentionExecutionPolicy::Portable
    }
    fn execution_determinism_requirement(&self) -> ExecutionDeterminismRequirement {
        ExecutionDeterminismRequirement::BitwiseSameRuntime
    }
    fn dynamic_storage_profile_order(&self) -> &[DynamicStorageProfile] {
        &self.profiles
    }
    fn reusable_execution_policy(&self) -> Option<&ReusableExecutionPolicy> {
        None
    }
    fn validate(&self) -> Result<(), VNextError> {
        if self.capacity == 0 || self.profiles.is_empty() {
            return Err(invalid_config(
                "policy",
                "tensor test policy has no capacity or storage",
            ));
        }
        Ok(())
    }
}

fn compile_input(
    dimensions: Vec<u64>,
    element_type: ElementType,
) -> Result<ProgramPlanCompilation, VNextError> {
    let family = prepare(&config_from(&hf_config()));
    let composition = ReferenceVNextComposition::create(
        DeviceId::new("device.reference.qwen_image21.projection").unwrap(),
    )
    .unwrap();
    let descriptor = composition.runtime().descriptor();
    let policy = TensorPolicy {
        capacity: descriptor.total_memory_bytes,
        profiles: descriptor
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    };
    let options = ProgramPlanCompileOptions::new(BTreeMap::from([(
        ProgramValueId::new(IMAGE_PROJECTION_INPUT_ID)?,
        ProgramTensorSpec {
            dimensions,
            element_type,
            layout: ResolvedTensorLayout::Contiguous,
        },
    )]))?;
    let compiled = ProgramPlanCompiler::compile_with_weight_materializers(
        &family,
        composition.catalog(),
        &policy,
        &composition.registry().planning(),
        composition.weight_materializers(),
        &options,
    );
    assert_eq!(composition.runtime().snapshot().live_allocations, 0);
    compiled
}

#[test]
fn latent_rows_are_inferred_and_sealed_by_each_fixed_tensor_compilation() {
    let one = compile_input(vec![1, 3], ElementType::Bf16).unwrap();
    let four = compile_input(vec![4, 3], ElementType::Bf16).unwrap();
    let output_id = ProgramValueId::new(IMAGE_PROJECTION_OUTPUT_ID).unwrap();
    assert_eq!(one.value_tensors()[&output_id].dimensions(), [1, 12]);
    assert_eq!(four.value_tensors()[&output_id].dimensions(), [4, 12]);
    let plan = four.executable().execution_plan();
    assert_eq!(plan.payload().work_limits(), RuntimeWorkLimits::FixedTensor);
    assert!(matches!(
        plan.payload().nodes()[0].work(),
        NodeWorkContract::Fixed
    ));
    let ports = plan.payload().tensor_ports().unwrap();
    assert_eq!(ports.inputs()[0].tensor().dimensions(), [4, 3]);
    assert_eq!(ports.outputs()[0].tensor().dimensions(), [4, 12]);
    assert_eq!(
        ports.outputs()[0].tensor().element_type(),
        ElementType::Bf16
    );
    assert_ne!(
        one.executable().execution_plan().plan_hash(),
        plan.plan_hash()
    );
}

#[test]
fn compilation_rejects_wrong_dtype_rank_width_and_invalid_row_capacity() {
    for (dimensions, dtype) in [
        (vec![4, 3], ElementType::F16),
        (vec![4, 3], ElementType::F32),
        (vec![3, 4], ElementType::Bf16),
        (vec![1, 4, 3], ElementType::Bf16),
        (vec![0, 3], ElementType::Bf16),
        (vec![u64::MAX, 3], ElementType::Bf16),
    ] {
        assert!(compile_input(dimensions, dtype).is_err());
    }
}
