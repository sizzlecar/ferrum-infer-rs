use super::*;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Config {
    width: u64,
}

pub(super) struct Family(ModelFamilyId, bool);
impl Default for Family {
    fn default() -> Self {
        Self(id("family.cuda-guard-fixture"), false)
    }
}
impl Family {
    pub(super) fn with_program_binding() -> Self {
        Self(id("family.cuda-guard-fixture"), true)
    }
    fn operation_id(&self) -> OperationId {
        if self.1 {
            id("operation.cuda-guard-fixture.scale-with-binding")
        } else {
            id(CONSTANT_SCALE_OPERATION_ID)
        }
    }
}

/// The binding fixture has a distinct contract: the production scale operation
/// forbids auxiliary resources and is never weakened for a test provider.
struct BindingScaleContract(OperationDescriptor);
impl OperationContract for BindingScaleContract {
    fn descriptor(&self) -> &OperationDescriptor {
        &self.0
    }
    fn validate_signature(
        &self,
        inputs: &[TensorContract],
        outputs: &[TensorContract],
    ) -> Result<(), VNextError> {
        constant_scale_contract()?.validate_signature(inputs, outputs)
    }
}
pub(super) fn scale_contract(program_binding: bool) -> Box<dyn OperationContract> {
    let standard = constant_scale_contract().unwrap();
    if !program_binding {
        return Box::new(standard);
    }
    let mut descriptor = standard.descriptor().clone();
    descriptor.id = Family::with_program_binding().operation_id();
    descriptor.resources.binding = ResourcePresenceRequirement::Required;
    descriptor.validate().unwrap();
    Box::new(BindingScaleContract(descriptor))
}

impl ModelFamilyProvider for Family {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        &self.0
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.cuda-guard-fixture")])
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        config: &Config,
    ) -> Result<(), VNextError> {
        if raw.get("width").and_then(serde_json::Value::as_u64) != Some(config.width)
            || config.width != 4
        {
            return Err(invalid("fixture requires four F16 elements"));
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        config: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.cuda-guard-fixture"))
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Config, VNextError> {
        serde_json::from_value(raw.clone())
            .map_err(|error| invalid(&format!("invalid fixture configuration: {error}")))
    }
    fn weight_schema(&self, _: &Config) -> Result<WeightSchema, VNextError> {
        // The family wire requires a schema. This optional unused component
        // never enters this weightless program or acquires an allocation.
        Ok(WeightSchema {
            format_id: id("weight-format.cuda-guard-fixture"),
            layout_id: id("weight-layout.cuda-guard-fixture"),
            version: ContractVersion::new(1, 0),
            components: vec![WeightComponentSpec {
                id: id("component.unused"),
                role: WeightComponentRole::Values,
                external_names: vec!["unused".into()],
                dimensions: vec![1],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::F16,
                },
                required: false,
            }],
            tensors: vec![WeightTensorSpec {
                id: id("weight.unused"),
                dimensions: vec![1],
                logical_element_type: ElementType::F16,
                physical_layout: PhysicalWeightLayout::Dense {
                    component_id: id("component.unused"),
                },
                required: false,
            }],
        })
    }
    fn numerical_profiles(&self, _: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        FamilyNumericalProfiles::new(
            &self.0,
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: id("fixture.f16"),
                version: ContractVersion::new(1, 0),
                family_id: self.0.clone(),
                primary_activation: id("value.output"),
                boundaries: BTreeMap::from([(id("value.output"), ElementType::F16)]),
                states: vec![],
                kv_storage: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: self.operation_id(),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: Some(ElementType::F32),
                    accumulation_type: None,
                }],
            }],
            vec![id("fixture.f16")],
        )
    }
    fn semantic_program(
        &self,
        _: &Config,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        ModelProgram::new(
            self.0.clone(),
            vec![id("value.input")],
            vec![ProgramBlock {
                id: "block.scale".into(),
                nodes: vec![ProgramNode {
                    id: id("node.scale"),
                    operation_id: self.operation_id(),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::tokens(id("value.input"), 0),
                    inputs: vec![id("value.input")],
                    outputs: vec![id("value.output")],
                    attributes: BTreeMap::from([
                        (id("hidden_size"), SemanticValue::Unsigned(4)),
                        (
                            id("scale"),
                            SemanticValue::Rational(CanonicalRational::new(2, 1)?),
                        ),
                    ]),
                }],
            }],
            vec![],
            vec![],
            vec![id("value.output")],
        )
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata {
            template: TemplateMetadata {
                template: "{{ messages }}".into(),
                source_file: "fixture".into(),
                sha256: digest(b"{{ messages }}"),
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
