use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use ferrum_interfaces::vnext::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[path = "vnext_numerical_profile/compiler.rs"]
mod compiler;

fn id<T: TryFrom<String>>(value: &str) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(value.to_owned()).unwrap()
}

fn tensor(dtype: ElementType, dimensions: Vec<u64>) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type: dtype,
        layout: ResolvedTensorLayout::Contiguous,
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    width: u64,
}

struct Family {
    family_id: ModelFamilyId,
    program_calls: Arc<AtomicUsize>,
    corrupt_physical_identity: bool,
}

impl Family {
    fn profile(&self, profile_id: &str, dtype: ElementType) -> NumericalExecutionProfile {
        NumericalExecutionProfile {
            id: id(profile_id),
            family_id: self.family_id.clone(),
            version: ContractVersion::new(1, 0),
            primary_activation: id("value.output"),
            boundaries: BTreeMap::from([(id("value.output"), dtype)]),
            states: vec![StateSpec {
                id: id("state.recurrent"),
                value_id: id("value.state"),
                tensor: tensor(dtype, vec![4]),
                lifetime: StateLifetime::Sequence,
                capacity_demand: StateCapacityDemand::FixedPerScope,
                initialization: StateInitialization::Zero,
                checkpoint: StateCheckpointCapability::Unsupported,
            }],
            operations: vec![NumericalOperationContract {
                operation_id: id(if dtype == ElementType::F16 {
                    "operation.fixture.f16"
                } else {
                    "operation.fixture.f32"
                }),
                version: ContractVersion::new(1, 0),
                multiplication_type: Some(ElementType::F32),
                accumulation_type: Some(ElementType::F32),
            }],
        }
    }
}

impl ModelFamilyProvider for Family {
    type Config = Config;

    fn family_id(&self) -> &ModelFamilyId {
        &self.family_id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.numerical-fixture")])
    }
    fn validate_config_identity(&self, _raw: &Value, _config: &Config) -> Result<(), VNextError> {
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        _raw: &Value,
        _config: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        Ok(id("metadata.numerical-fixture"))
    }
    fn parse_config(&self, raw: &Value) -> Result<Config, VNextError> {
        let config: Config = serde_json::from_value(raw.clone()).map_err(|error| {
            VNextError::InvalidModelConfig {
                family_id: self.family_id.to_string(),
                field: "config".into(),
                reason: error.to_string(),
            }
        })?;
        if config.width == 0 || !config.width.is_multiple_of(256) {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id.to_string(),
                field: "width".into(),
                reason: "Q4_K rows must contain complete blocks".into(),
            });
        }
        Ok(config)
    }
    fn weight_schema(&self, config: &Config) -> Result<WeightSchema, VNextError> {
        Ok(WeightSchema {
            format_id: id("weight-format.gguf"),
            layout_id: id("weight-layout.gguf-q4-k"),
            version: ContractVersion::new(1, 0),
            components: vec![WeightComponentSpec {
                id: id("component.blocks"),
                role: WeightComponentRole::PackedValues,
                external_names: vec!["projection.weight".into()],
                dimensions: vec![1, config.width / 256],
                encoding: WeightEncoding::BlockQuantized(BlockQuantizationSpec {
                    format_id: id("quantization.gguf.q4-k"),
                    logical_values_per_block: 256,
                    bytes_per_block: 144,
                }),
                required: true,
            }],
            tensors: vec![WeightTensorSpec {
                id: id("weight.projection"),
                dimensions: vec![1, config.width],
                logical_element_type: ElementType::F32,
                physical_layout: PhysicalWeightLayout::BlockQuantized {
                    blocks: PhysicalWeightComponentBinding::exact_contiguous(id(
                        "component.blocks",
                    )),
                    block_axis: 1,
                    block_padding: PhysicalWeightPadding::Exact,
                },
                required: true,
            }],
        })
    }
    fn numerical_profiles(&self, _config: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        FamilyNumericalProfiles::new(
            &self.family_id,
            ContractVersion::new(1, 0),
            vec![
                self.profile("fixture.f16", ElementType::F16),
                self.profile("fixture.f32", ElementType::F32),
            ],
            vec![id("fixture.f32")],
        )
    }
    fn specialize_weight_schema(
        &self,
        _config: &Config,
        source: &WeightSchema,
        profile: &NumericalExecutionProfile,
    ) -> Result<WeightSchema, VNextError> {
        let mut schema = source.clone();
        schema.tensors[0].logical_element_type = profile.activation_type()?;
        if self.corrupt_physical_identity {
            schema.components[0].dimensions[0] += 1;
        }
        Ok(schema)
    }
    fn semantic_program(
        &self,
        config: &Config,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        self.program_calls.fetch_add(1, Ordering::SeqCst);
        ModelProgram::new(
            self.family_id.clone(),
            vec![id("value.input")],
            vec![ProgramBlock {
                id: "main".into(),
                nodes: vec![ProgramNode {
                    id: id("node.projection"),
                    operation_id: profile.operations[0].operation_id.clone(),
                    required_version: profile.operations[0].version,
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![id("value.input"), id("value.weight"), id("value.state")],
                    outputs: vec![id("value.output")],
                    attributes: BTreeMap::new(),
                }],
            }],
            profile.states.clone(),
            vec![WeightReference {
                weight_id: id("weight.projection"),
                value_id: id("value.weight"),
                tensor: tensor(profile.activation_type()?, vec![1, config.width]),
            }],
            vec![id("value.output")],
        )
    }
    fn semantic_metadata(&self, _config: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        let template = "{{ messages }}";
        Ok(ModelSemanticMetadata {
            template: TemplateMetadata {
                template: template.into(),
                source_file: "chat_template.jinja".into(),
                sha256: format!("{:x}", Sha256::digest(template.as_bytes())),
            },
            special_tokens: SpecialTokenMetadata {
                bos_token_id: None,
                eos_token_ids: BTreeSet::from([2]),
                pad_token_id: None,
                collision_policy: SpecialTokenCollisionPolicy::require_distinct(),
            },
        })
    }
}

struct Registry {
    registration: TypedFamilyRegistration<Family>,
}
impl Registry {
    fn new(calls: Arc<AtomicUsize>, corrupt_physical_identity: bool) -> Self {
        Self {
            registration: TypedFamilyRegistration::new(Family {
                family_id: id("family.numerical-fixture"),
                program_calls: calls,
                corrupt_physical_identity,
            }),
        }
    }
    fn definition(&self) -> ModelFamilyDefinition {
        self.registration.define(&json!({"width": 256})).unwrap()
    }
}
impl ModelFamilyRegistry for Registry {
    fn registrations(&self) -> Vec<&dyn ModelFamilyRegistration> {
        vec![&self.registration]
    }
}

#[test]
fn definition_retains_its_typed_preparation_and_original_registration() {
    let calls = Arc::new(AtomicUsize::new(0));
    let original = Registry::new(calls.clone(), false);
    let other_calls = Arc::new(AtomicUsize::new(0));
    let other = Registry::new(other_calls.clone(), false);
    let definition = original.definition();
    assert_eq!(
        definition.fingerprint().unwrap(),
        other.definition().fingerprint().unwrap()
    );
    assert!(other
        .registration
        .prepare(&definition, &id("fixture.f32"))
        .is_err());
    assert_eq!(other_calls.load(Ordering::SeqCst), 0);
    let prepared = original
        .registration
        .prepare(&definition.clone(), &id("fixture.f32"))
        .unwrap();
    assert_eq!(prepared.numerical_profile().id, id("fixture.f32"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn definition_has_no_program_and_same_gguf_source_prepares_two_numerical_abis() {
    let calls = Arc::new(AtomicUsize::new(0));
    let registry = Registry::new(calls.clone(), false);
    let definition = registry.definition();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let f16 = registry
        .registration
        .prepare(&definition, &id("fixture.f16"))
        .unwrap();
    let f32 = registry
        .registration
        .prepare(&definition, &id("fixture.f32"))
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        f16.weight_schema().components,
        definition.weight_schema().components
    );
    assert_eq!(
        f16.weight_schema().components,
        f32.weight_schema().components
    );
    assert_ne!(
        f16.weight_schema().tensors[0].logical_element_type,
        f32.weight_schema().tensors[0].logical_element_type
    );
    assert_eq!(f16.config_fingerprint(), f32.config_fingerprint());
    assert_ne!(f16.fingerprint().unwrap(), f32.fingerprint().unwrap());
    assert_ne!(
        f16.program().fingerprint().unwrap(),
        f32.program().fingerprint().unwrap()
    );
}

#[test]
fn unqualified_explicit_profile_does_not_silently_enter_auto_preferences() {
    let registry = Registry::new(Arc::default(), false);
    let definition = registry.definition();
    let profiles = definition.numerical_profiles();
    assert_eq!(
        profiles
            .candidates(&NumericalExecutionPolicy::Auto)
            .unwrap()
            .iter()
            .map(|p| p.id.as_str())
            .collect::<Vec<_>>(),
        ["fixture.f32"]
    );
    assert_eq!(
        profiles
            .candidates(&NumericalExecutionPolicy::Require(id("fixture.f16")))
            .unwrap()[0]
            .id,
        id("fixture.f16")
    );
    assert!(profiles
        .candidates(&NumericalExecutionPolicy::Require(id("fixture.unknown")))
        .is_err());
}

#[test]
fn physical_source_drift_is_rejected_before_program_construction() {
    let calls = Arc::new(AtomicUsize::new(0));
    let registry = Registry::new(calls.clone(), true);
    assert!(registry
        .registration
        .prepare(&registry.definition(), &id("fixture.f16"))
        .is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn inferred_boundaries_and_declared_state_or_operation_drift_are_rejected() {
    let registry = Registry::new(Arc::default(), false);
    let prepared = registry
        .registration
        .prepare(&registry.definition(), &id("fixture.f16"))
        .unwrap();
    let profile = prepared.numerical_profile();
    let resolved =
        |dtype| ResolvedTensorSpec::new(vec![1], dtype, ResolvedTensorLayout::Contiguous).unwrap();
    assert!(profile
        .validate_inferred_boundaries(&BTreeMap::from([(
            id("value.output"),
            resolved(ElementType::F32)
        )]))
        .is_err());
    profile
        .validate_inferred_boundaries(&BTreeMap::from([(
            id("value.output"),
            resolved(ElementType::F16),
        )]))
        .unwrap();
    let mut operation = profile.clone();
    operation.operations[0].version = ContractVersion::new(2, 0);
    assert!(operation.validate_program(prepared.program()).is_err());
    let mut state = profile.clone();
    state.states[0].tensor.element_type = ElementType::F32;
    assert!(state.validate_program(prepared.program()).is_err());
    let mut capacity = profile.clone();
    capacity.states[0].capacity_demand = StateCapacityDemand::TokenScaled {
        bytes_per_token: 1,
        maximum_tokens: 8,
    };
    assert!(capacity.validate().is_err());
}

#[test]
fn wire_rebuilds_selected_profile_and_rejects_forged_arithmetic_or_old_semantics() {
    let registry = Registry::new(Arc::default(), false);
    let prepared = registry
        .registration
        .prepare(&registry.definition(), &id("fixture.f16"))
        .unwrap();
    let wire = serde_json::to_value(&prepared).unwrap();
    assert_eq!(
        PreparedModelFamily::from_json_validated(&serde_json::to_vec(&wire).unwrap(), &registry)
            .unwrap(),
        prepared
    );
    for field in ["numerical_profile", "wire_version"] {
        let mut old = wire.clone();
        old.as_object_mut().unwrap().remove(field);
        assert!(PreparedModelFamily::decode_untrusted(&serde_json::to_vec(&old).unwrap()).is_err());
    }
    let mut old = wire.clone();
    old["wire_version"] = json!(1);
    assert!(PreparedModelFamily::decode_untrusted(&serde_json::to_vec(&old).unwrap()).is_err());
    let mut arithmetic = wire.clone();
    arithmetic["numerical_profile"]["operations"][0]["accumulation_type"] =
        serde_json::to_value(ElementType::F16).unwrap();
    assert!(PreparedModelFamily::from_json_validated(
        &serde_json::to_vec(&arithmetic).unwrap(),
        &registry
    )
    .is_err());
    let mut identity = wire;
    identity["numerical_profile"]["id"] = json!("fixture.f32");
    assert!(PreparedModelFamily::from_json_validated(
        &serde_json::to_vec(&identity).unwrap(),
        &registry
    )
    .is_err());
}
