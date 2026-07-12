use ferrum_interfaces::vnext::*;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn id<T>(value: impl Into<String>) -> T
where
    T: TryFrom<String, Error = VNextError>,
{
    T::try_from(value.into()).unwrap()
}

fn sha(character: char) -> String {
    std::iter::repeat_n(character, 64).collect()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct TestConfig {
    #[serde(default = "default_model_type")]
    model_type: String,
    width: u64,
    #[serde(default = "default_block_size")]
    block_size: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    quantization_config: Option<TestQuantizationConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    weight_layout: Option<TestWeightLayoutConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct TestQuantizationConfig {
    method: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct TestWeightLayoutConfig {
    kind: String,
}

fn default_model_type() -> String {
    "metadata.model-wire-test".to_owned()
}

const fn default_block_size() -> u64 {
    128
}

struct TestFamily;

impl ModelFamilyProvider for TestFamily {
    type Config = TestConfig;

    fn family_id(&self) -> &ModelFamilyId {
        static FAMILY_ID: std::sync::OnceLock<ModelFamilyId> = std::sync::OnceLock::new();
        FAMILY_ID.get_or_init(|| id("family.model-wire-test"))
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.model-wire-test")])
    }

    fn validate_config_identity(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<(), VNextError> {
        let identity = ExternalModelMetadataId::try_from(config.model_type.clone())?;
        if raw.get("model_type").and_then(Value::as_str) != Some(config.model_type.as_str())
            || !self.external_metadata_ids().contains(&identity)
        {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "model_type".to_owned(),
                reason: "model_type must be explicitly present, preserved, and declared".to_owned(),
            });
        }
        Ok(())
    }

    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        ExternalModelMetadataId::try_from(config.model_type.clone())
    }

    fn parse_config(&self, raw: &Value) -> Result<Self::Config, VNextError> {
        let config: TestConfig = serde_json::from_value(raw.clone()).map_err(|error| {
            VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "config".to_owned(),
                reason: error.to_string(),
            }
        })?;
        if config.width != 4 {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "width".to_owned(),
                reason: "test family requires width 4".to_owned(),
            });
        }
        Ok(config)
    }

    fn weight_schema(&self, config: &Self::Config) -> Result<WeightSchema, VNextError> {
        Ok(WeightSchema {
            format_id: id("weight-format.model-wire-test"),
            layout_id: id("weight-layout.model-wire-test"),
            version: ContractVersion::new(1, 0),
            components: vec![WeightComponentSpec {
                id: id("weight.component.model-wire-test"),
                role: WeightComponentRole::Values,
                external_names: vec!["weight.bin".to_owned()],
                dimensions: vec![config.width],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::F32,
                },
                required: true,
            }],
            tensors: vec![WeightTensorSpec {
                id: id("weight.tensor.model-wire-test"),
                dimensions: vec![config.width],
                logical_element_type: ElementType::F32,
                physical_layout: PhysicalWeightLayout::Dense {
                    component_id: id("weight.component.model-wire-test"),
                },
                required: true,
            }],
        })
    }

    fn semantic_program(&self, config: &Self::Config) -> Result<ModelProgram, VNextError> {
        ModelProgram::new(
            self.family_id().clone(),
            vec![id("value.input")],
            vec![ProgramBlock {
                id: "block.main".to_owned(),
                nodes: vec![ProgramNode {
                    id: id("node.main"),
                    operation_id: id("operation.model-wire-test"),
                    required_version: ContractVersion::new(1, 0),
                    inputs: vec![id("value.input"), id("value.weight"), id("value.state")],
                    outputs: vec![id("value.output")],
                    attributes: BTreeMap::new(),
                }],
            }],
            vec![StateSpec {
                id: id("state.cache"),
                value_id: id("value.state"),
                tensor: ProgramTensorSpec {
                    dimensions: vec![config.width],
                    element_type: ElementType::U8,
                    layout: ResolvedTensorLayout::Contiguous,
                },
                lifetime: StateLifetime::Sequence,
                capacity_demand: StateCapacityDemand::FixedPerScope,
            }],
            vec![WeightReference {
                weight_id: id("weight.tensor.model-wire-test"),
                value_id: id("value.weight"),
                tensor: ProgramTensorSpec {
                    dimensions: vec![config.width],
                    element_type: ElementType::F32,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            }],
            vec![id("value.output")],
        )
    }

    fn semantic_metadata(
        &self,
        _config: &Self::Config,
    ) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata {
            template: TemplateMetadata {
                template: "{{ messages }}".to_owned(),
                source_file: "template.json".to_owned(),
                sha256: sha('a'),
            },
            special_tokens: SpecialTokenMetadata {
                bos_token_id: Some(1),
                eos_token_ids: BTreeSet::from([2]),
                pad_token_id: Some(0),
                collision_policy: SpecialTokenCollisionPolicy::require_distinct(),
            },
        })
    }
}

struct TestRegistry {
    registration: TypedFamilyRegistration<TestFamily>,
}

impl TestRegistry {
    fn new() -> Self {
        Self {
            registration: TypedFamilyRegistration::new(TestFamily),
        }
    }

    fn prepare(&self) -> PreparedModelFamily {
        self.registration
            .prepare(&json!({
                "model_type": "metadata.model-wire-test",
                "width": 4
            }))
            .unwrap()
    }
}

impl ModelFamilyRegistry for TestRegistry {
    fn registrations(&self) -> Vec<&dyn ModelFamilyRegistration> {
        vec![&self.registration]
    }
}

struct RejectingRegistry;

impl ModelFamilyRegistry for RejectingRegistry {
    fn registrations(&self) -> Vec<&dyn ModelFamilyRegistration> {
        Vec::new()
    }
}

struct DuplicateExternalFamily;

impl ModelFamilyProvider for DuplicateExternalFamily {
    type Config = TestConfig;

    fn family_id(&self) -> &ModelFamilyId {
        static FAMILY_ID: std::sync::OnceLock<ModelFamilyId> = std::sync::OnceLock::new();
        FAMILY_ID.get_or_init(|| id("family.duplicate-external"))
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.model-wire-test")])
    }

    fn validate_config_identity(
        &self,
        _raw: &Value,
        _config: &Self::Config,
    ) -> Result<(), VNextError> {
        Ok(())
    }

    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.model-wire-test"))
    }

    fn parse_config(&self, _raw: &Value) -> Result<Self::Config, VNextError> {
        Err(VNextError::InvalidModelConfig {
            family_id: self.family_id().to_string(),
            field: "config".to_owned(),
            reason: "identity-only adversarial registration".to_owned(),
        })
    }

    fn weight_schema(&self, _config: &Self::Config) -> Result<WeightSchema, VNextError> {
        unreachable!("identity-only adversarial registration")
    }

    fn semantic_program(&self, _config: &Self::Config) -> Result<ModelProgram, VNextError> {
        unreachable!("identity-only adversarial registration")
    }

    fn semantic_metadata(
        &self,
        _config: &Self::Config,
    ) -> Result<ModelSemanticMetadata, VNextError> {
        unreachable!("identity-only adversarial registration")
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct ChangingConfig {
    value: u64,
}

struct ChangingFamily;

impl ModelFamilyProvider for ChangingFamily {
    type Config = ChangingConfig;

    fn family_id(&self) -> &ModelFamilyId {
        static FAMILY_ID: std::sync::OnceLock<ModelFamilyId> = std::sync::OnceLock::new();
        FAMILY_ID.get_or_init(|| id("family.changing-config"))
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.changing-config")])
    }

    fn validate_config_identity(
        &self,
        _raw: &Value,
        _config: &Self::Config,
    ) -> Result<(), VNextError> {
        Ok(())
    }

    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.changing-config"))
    }

    fn parse_config(&self, raw: &Value) -> Result<Self::Config, VNextError> {
        let mut config: ChangingConfig = serde_json::from_value(raw.clone()).unwrap();
        config.value += 1;
        Ok(config)
    }

    fn weight_schema(&self, _config: &Self::Config) -> Result<WeightSchema, VNextError> {
        unreachable!("changed raw input must fail before schema construction")
    }

    fn semantic_program(&self, _config: &Self::Config) -> Result<ModelProgram, VNextError> {
        unreachable!("changed raw input must fail before program construction")
    }

    fn semantic_metadata(
        &self,
        _config: &Self::Config,
    ) -> Result<ModelSemanticMetadata, VNextError> {
        unreachable!("changed raw input must fail before metadata construction")
    }
}

#[derive(Clone)]
struct OneShotConfig {
    model_type: String,
    width: u64,
    serialize_calls: Arc<AtomicUsize>,
}

#[derive(Deserialize)]
struct OneShotConfigWire {
    model_type: String,
    width: u64,
}

impl<'de> Deserialize<'de> for OneShotConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = OneShotConfigWire::deserialize(deserializer)?;
        Ok(Self {
            model_type: wire.model_type,
            width: wire.width,
            serialize_calls: Arc::new(AtomicUsize::new(0)),
        })
    }
}

impl Serialize for OneShotConfig {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let call = self.serialize_calls.fetch_add(1, Ordering::SeqCst);
        let mut state = serializer.serialize_struct("OneShotConfig", 2)?;
        state.serialize_field("model_type", &self.model_type)?;
        state.serialize_field("width", &(self.width + call as u64))?;
        state.end()
    }
}

struct OneShotFamily {
    serialize_calls: Arc<AtomicUsize>,
}

impl ModelFamilyProvider for OneShotFamily {
    type Config = OneShotConfig;

    fn family_id(&self) -> &ModelFamilyId {
        TestFamily.family_id()
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        TestFamily.external_metadata_ids()
    }

    fn validate_config_identity(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<(), VNextError> {
        if raw.get("model_type").and_then(Value::as_str) != Some(config.model_type.as_str()) {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "model_type".to_owned(),
                reason: "model_type must be explicitly present and preserved".to_owned(),
            });
        }
        Ok(())
    }

    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        ExternalModelMetadataId::try_from(config.model_type.clone())
    }

    fn parse_config(&self, raw: &Value) -> Result<Self::Config, VNextError> {
        let wire: OneShotConfigWire = serde_json::from_value(raw.clone()).map_err(|error| {
            VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "config".to_owned(),
                reason: error.to_string(),
            }
        })?;
        Ok(OneShotConfig {
            model_type: wire.model_type,
            width: wire.width,
            serialize_calls: Arc::clone(&self.serialize_calls),
        })
    }

    fn weight_schema(&self, config: &Self::Config) -> Result<WeightSchema, VNextError> {
        TestFamily.weight_schema(&TestConfig {
            model_type: config.model_type.clone(),
            width: config.width,
            block_size: default_block_size(),
            quantization_config: None,
            weight_layout: None,
        })
    }

    fn semantic_program(&self, config: &Self::Config) -> Result<ModelProgram, VNextError> {
        TestFamily.semantic_program(&TestConfig {
            model_type: config.model_type.clone(),
            width: config.width,
            block_size: default_block_size(),
            quantization_config: None,
            weight_layout: None,
        })
    }

    fn semantic_metadata(
        &self,
        config: &Self::Config,
    ) -> Result<ModelSemanticMetadata, VNextError> {
        TestFamily.semantic_metadata(&TestConfig {
            model_type: config.model_type.clone(),
            width: config.width,
            block_size: default_block_size(),
            quantization_config: None,
            weight_layout: None,
        })
    }
}

struct AliasedFamily;

impl ModelFamilyProvider for AliasedFamily {
    type Config = TestConfig;

    fn family_id(&self) -> &ModelFamilyId {
        TestFamily.family_id()
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([
            id("metadata.model-wire-test"),
            id("metadata.model-wire-test-alias"),
        ])
    }

    fn validate_config_identity(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<(), VNextError> {
        let actual = raw.get("model_type").and_then(Value::as_str);
        if actual != Some(config.model_type.as_str()) {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "model_type".to_owned(),
                reason: "model_type must be explicitly present and preserved".to_owned(),
            });
        }
        Ok(())
    }

    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        ExternalModelMetadataId::try_from(config.model_type.clone())
    }

    fn parse_config(&self, raw: &Value) -> Result<Self::Config, VNextError> {
        TestFamily.parse_config(raw)
    }

    fn weight_schema(&self, config: &Self::Config) -> Result<WeightSchema, VNextError> {
        TestFamily.weight_schema(config)
    }

    fn semantic_program(&self, config: &Self::Config) -> Result<ModelProgram, VNextError> {
        TestFamily.semantic_program(config)
    }

    fn semantic_metadata(
        &self,
        config: &Self::Config,
    ) -> Result<ModelSemanticMetadata, VNextError> {
        TestFamily.semantic_metadata(config)
    }
}

struct AliasedRegistry {
    registration: TypedFamilyRegistration<AliasedFamily>,
}

impl AliasedRegistry {
    fn new() -> Self {
        Self {
            registration: TypedFamilyRegistration::new(AliasedFamily),
        }
    }
}

impl ModelFamilyRegistry for AliasedRegistry {
    fn registrations(&self) -> Vec<&dyn ModelFamilyRegistration> {
        vec![&self.registration]
    }
}

struct DuplicateRegistry {
    first: TypedFamilyRegistration<TestFamily>,
    duplicate_internal: TypedFamilyRegistration<TestFamily>,
    duplicate_external: TypedFamilyRegistration<DuplicateExternalFamily>,
}

impl DuplicateRegistry {
    fn new() -> Self {
        Self {
            first: TypedFamilyRegistration::new(TestFamily),
            duplicate_internal: TypedFamilyRegistration::new(TestFamily),
            duplicate_external: TypedFamilyRegistration::new(DuplicateExternalFamily),
        }
    }
}

impl ModelFamilyRegistry for DuplicateRegistry {
    fn registrations(&self) -> Vec<&dyn ModelFamilyRegistration> {
        vec![
            &self.first,
            &self.duplicate_internal,
            &self.duplicate_external,
        ]
    }
}

#[test]
fn typed_family_config_and_registry_identity_fail_closed() {
    let registration = TypedFamilyRegistration::new(TestFamily);
    let defaulted = registration
        .prepare(&json!({
            "model_type": "metadata.model-wire-test",
            "width": 4
        }))
        .unwrap();
    assert_eq!(defaulted.canonical_config()["block_size"], json!(128));

    let explicit = registration
        .prepare(&json!({
            "model_type": "metadata.model-wire-test",
            "width": 4,
            "block_size": 64
        }))
        .unwrap();
    assert_eq!(explicit.canonical_config()["block_size"], json!(64));

    assert!(registration.prepare(&json!({"width": 4})).is_err());
    assert!(registration
        .prepare(&json!({
            "model_type": "metadata.some-other-family",
            "width": 4
        }))
        .is_err());
    assert!(TypedFamilyRegistration::new(ChangingFamily)
        .prepare(&json!({"value": 1}))
        .is_err());
    assert!(registration
        .prepare(&json!({
            "model_type": "metadata.model-wire-test",
            "width": 4,
            "caller_override": true
        }))
        .is_err());
    assert!(registration
        .prepare(&json!({
            "model_type": "metadata.model-wire-test",
            "width": 4,
            "quantization_config": {"method": "int4", "caller_override": true}
        }))
        .is_err());
    assert!(registration
        .prepare(&json!({
            "model_type": "metadata.model-wire-test",
            "width": 4,
            "weight_layout": {"kind": "dense", "caller_override": true}
        }))
        .is_err());

    let registry = TestRegistry::new();
    let registry: &dyn ModelFamilyRegistry = &registry;
    assert!(matches!(
        registry.resolve_external(&id("metadata.unknown")),
        Err(VNextError::UnknownExternalModelMetadata { .. })
    ));

    let duplicates = DuplicateRegistry::new();
    let duplicates: &dyn ModelFamilyRegistry = &duplicates;
    assert!(matches!(
        duplicates.resolve(&id("family.model-wire-test")),
        Err(VNextError::AmbiguousModelFamilyRegistration {
            identity_kind: "internal family",
            matches: 2,
            ..
        })
    ));
    assert!(matches!(
        duplicates.resolve_external(&id("metadata.model-wire-test")),
        Err(VNextError::AmbiguousModelFamilyRegistration {
            identity_kind: "external metadata",
            matches: 3,
            ..
        })
    ));
}

#[test]
fn typed_config_is_serialized_once_and_signed_external_identity_is_replayed() {
    let serialize_calls = Arc::new(AtomicUsize::new(0));
    let prepared = TypedFamilyRegistration::new(OneShotFamily {
        serialize_calls: Arc::clone(&serialize_calls),
    })
    .prepare(&json!({
        "model_type": "metadata.model-wire-test",
        "width": 4
    }))
    .unwrap();
    assert_eq!(serialize_calls.load(Ordering::SeqCst), 1);
    assert_eq!(prepared.canonical_config()["width"], json!(4));

    let registry = AliasedRegistry::new();
    let prepared = registry
        .registration
        .prepare(&json!({
            "model_type": "metadata.model-wire-test",
            "width": 4
        }))
        .unwrap();
    assert_eq!(
        prepared.external_metadata_id(),
        &id("metadata.model-wire-test")
    );
    let mut forged = serde_json::to_value(&prepared).unwrap();
    forged["external_metadata_id"] = json!("metadata.model-wire-test-alias");
    let unvalidated =
        PreparedModelFamily::decode_untrusted(&serde_json::to_vec(&forged).unwrap()).unwrap();
    assert!(unvalidated.revalidate(&registry).is_err());
}

#[test]
fn prepared_family_wire_round_trip_requires_external_typed_registry() {
    let registry = TestRegistry::new();
    let prepared = registry.prepare();
    let bytes = serde_json::to_vec(&prepared).unwrap();

    let unvalidated = PreparedModelFamily::decode_untrusted(&bytes).unwrap();
    assert_eq!(unvalidated.revalidate(&registry).unwrap(), prepared);
    assert_eq!(
        PreparedModelFamily::from_json_validated(&bytes, &registry).unwrap(),
        prepared
    );
}

#[test]
fn prepared_family_wire_rejects_unknown_fields_and_typed_drift() {
    let registry = TestRegistry::new();
    let prepared = registry.prepare();

    let mut unknown = serde_json::to_value(&prepared).unwrap();
    unknown["caller_selected_provider"] = json!("provider.impostor");
    assert!(PreparedModelFamily::decode_untrusted(&serde_json::to_vec(&unknown).unwrap()).is_err());

    let mut nested_unknown = serde_json::to_value(&prepared).unwrap();
    nested_unknown["metadata"]["template"]["caller_hint"] = json!("ignored");
    assert!(
        PreparedModelFamily::decode_untrusted(&serde_json::to_vec(&nested_unknown).unwrap())
            .is_err()
    );

    let mut config_drift = serde_json::to_value(&prepared).unwrap();
    config_drift["canonical_config"]["width"] = json!(8);
    let unvalidated =
        PreparedModelFamily::decode_untrusted(&serde_json::to_vec(&config_drift).unwrap()).unwrap();
    assert!(unvalidated.revalidate(&registry).is_err());

    let mut fingerprint_drift = serde_json::to_value(&prepared).unwrap();
    fingerprint_drift["config_fingerprint"] = json!(sha('f'));
    assert!(PreparedModelFamily::from_json_validated(
        &serde_json::to_vec(&fingerprint_drift).unwrap(),
        &registry,
    )
    .is_err());
}

#[test]
fn prepared_family_wire_accepts_max_and_rejects_max_plus_one_before_serde() {
    let registry = TestRegistry::new();
    let prepared = registry.prepare();
    let mut bytes = serde_json::to_vec(&prepared).unwrap();
    bytes.resize(MAX_PREPARED_MODEL_FAMILY_WIRE_BYTES, b' ');
    assert_eq!(
        PreparedModelFamily::from_json_validated(&bytes, &registry).unwrap(),
        prepared
    );

    bytes.push(b' ');
    match PreparedModelFamily::decode_untrusted(&bytes) {
        Err(VNextError::Serialization { context, message }) => {
            assert_eq!(context, "decode untrusted prepared model family");
            assert!(message.contains("maximum is 16777216"));
        }
        Err(error) => panic!("wrong max+1 error: {error}"),
        Ok(_) => panic!("max+1 prepared family wire was accepted"),
    }
}

#[test]
fn prepared_model_family_wire_proof_line() {
    const EXPECTED: usize = 24;
    let mut passed = 0usize;
    macro_rules! check {
        ($condition:expr) => {{
            assert!($condition);
            passed += 1;
        }};
    }

    let registry = TestRegistry::new();
    let prepared = registry.prepare();
    let bytes = serde_json::to_vec(&prepared).unwrap();
    let unvalidated = PreparedModelFamily::decode_untrusted(&bytes).unwrap();
    passed += 1;
    check!(unvalidated.revalidate(&registry).unwrap() == prepared);
    check!(PreparedModelFamily::from_json_validated(&bytes, &registry).unwrap() == prepared);
    check!(PreparedModelFamily::from_json_validated(&bytes, &RejectingRegistry).is_err());

    let mut top_unknown = serde_json::to_value(&prepared).unwrap();
    top_unknown["caller_override"] = json!(true);
    check!(
        PreparedModelFamily::decode_untrusted(&serde_json::to_vec(&top_unknown).unwrap()).is_err()
    );
    let mut nested_unknown = serde_json::to_value(&prepared).unwrap();
    nested_unknown["metadata"]["template"]["caller_override"] = json!(true);
    check!(
        PreparedModelFamily::decode_untrusted(&serde_json::to_vec(&nested_unknown).unwrap())
            .is_err()
    );

    let mut config_drift = serde_json::to_value(&prepared).unwrap();
    config_drift["canonical_config"]["width"] = json!(8);
    check!(
        PreparedModelFamily::decode_untrusted(&serde_json::to_vec(&config_drift).unwrap())
            .unwrap()
            .revalidate(&registry)
            .is_err()
    );
    let mut fingerprint_drift = serde_json::to_value(&prepared).unwrap();
    fingerprint_drift["config_fingerprint"] = json!(sha('f'));
    check!(PreparedModelFamily::from_json_validated(
        &serde_json::to_vec(&fingerprint_drift).unwrap(),
        &registry,
    )
    .is_err());

    let mut at_limit = bytes;
    at_limit.resize(MAX_PREPARED_MODEL_FAMILY_WIRE_BYTES, b' ');
    check!(PreparedModelFamily::from_json_validated(&at_limit, &registry).unwrap() == prepared);
    at_limit.push(b' ');
    check!(matches!(
        PreparedModelFamily::decode_untrusted(&at_limit),
        Err(VNextError::Serialization {
            context: "decode untrusted prepared model family",
            ..
        })
    ));

    let registration = TypedFamilyRegistration::new(TestFamily);
    let defaulted = registration
        .prepare(&json!({
            "model_type": "metadata.model-wire-test",
            "width": 4
        }))
        .unwrap();
    check!(defaulted.canonical_config()["block_size"] == json!(128));
    check!(registration.prepare(&json!({"width": 4})).is_err());
    check!(registration
        .prepare(&json!({
            "model_type": "metadata.some-other-family",
            "width": 4
        }))
        .is_err());
    check!(registration
        .prepare(&json!({
            "model_type": "metadata.model-wire-test",
            "width": 4,
            "caller_override": true
        }))
        .is_err());
    check!(registration
        .prepare(&json!({
            "model_type": "metadata.model-wire-test",
            "width": 4,
            "quantization_config": {"method": "int4", "caller_override": true}
        }))
        .is_err());
    check!(registration
        .prepare(&json!({
            "model_type": "metadata.model-wire-test",
            "width": 4,
            "weight_layout": {"kind": "dense", "caller_override": true}
        }))
        .is_err());

    let registry: &dyn ModelFamilyRegistry = &registry;
    check!(matches!(
        registry.resolve_external(&id("metadata.unknown")),
        Err(VNextError::UnknownExternalModelMetadata { .. })
    ));
    let duplicates = DuplicateRegistry::new();
    let duplicates: &dyn ModelFamilyRegistry = &duplicates;
    check!(matches!(
        duplicates.resolve(&id("family.model-wire-test")),
        Err(VNextError::AmbiguousModelFamilyRegistration { .. })
    ));
    check!(matches!(
        duplicates.resolve_external(&id("metadata.model-wire-test")),
        Err(VNextError::AmbiguousModelFamilyRegistration { .. })
    ));
    check!(registration
        .prepare(&json!({
            "model_type": "metadata.model-wire-test",
            "width": 4,
            "block_size": 64
        }))
        .is_ok());
    check!(TypedFamilyRegistration::new(ChangingFamily)
        .prepare(&json!({"value": 1}))
        .is_err());

    let serialize_calls = Arc::new(AtomicUsize::new(0));
    let one_shot = TypedFamilyRegistration::new(OneShotFamily {
        serialize_calls: Arc::clone(&serialize_calls),
    })
    .prepare(&json!({
        "model_type": "metadata.model-wire-test",
        "width": 4
    }))
    .unwrap();
    check!(
        serialize_calls.load(Ordering::SeqCst) == 1
            && one_shot.canonical_config()["width"] == json!(4)
    );

    let aliased_registry = AliasedRegistry::new();
    let aliased = aliased_registry
        .registration
        .prepare(&json!({
            "model_type": "metadata.model-wire-test",
            "width": 4
        }))
        .unwrap();
    check!(aliased.external_metadata_id() == &id("metadata.model-wire-test"));
    let mut forged_identity = serde_json::to_value(&aliased).unwrap();
    forged_identity["external_metadata_id"] = json!("metadata.model-wire-test-alias");
    check!(
        PreparedModelFamily::decode_untrusted(&serde_json::to_vec(&forged_identity).unwrap())
            .unwrap()
            .revalidate(&aliased_registry)
            .is_err()
    );

    assert_eq!(passed, EXPECTED);
    println!("VNEXT MODEL WIRE PASS: {passed}/{EXPECTED}");
}
