//! Production model-family packages consumed by the vNext planner and runtime.

use std::collections::BTreeSet;
use std::fs;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::sync::Arc;

use ferrum_interfaces::vnext::{
    AttributeId, ElementType, ExternalModelMetadataId, ModelFamilyRegistration,
    ModelFamilyRegistry, PreparedModelFamily, ProgramNode, SemanticValue, StateCapacityDemand,
    StateLifetime, TypedFamilyRegistration, WeightComponentSource,
    ROUTED_SHARED_SWIGLU_MOE_OPERATION_ID,
};
use ferrum_types::{
    DataType, Device, ModelCapabilities, ModelId, ModelInfo, ModelType, MoeCapabilities,
};
use serde_json::Value;

mod hf_metadata;
pub mod qwen35;
pub mod source;
mod weight_layout;

pub use source::{
    huggingface_snapshot_identity, HuggingFaceSnapshotIdentity, ProductionModelSourceBundle,
    ProductionWeightArtifact,
};

type PrepareModel =
    fn(Arc<ProductionModelSourceBundle>) -> ferrum_types::Result<PreparedProductionModel>;
type CreateFamilyRegistration = fn() -> ferrum_types::Result<Box<dyn ModelFamilyRegistration>>;

struct ModelLoaderRegistration {
    external_metadata_ids: &'static [&'static str],
    gguf_architectures: &'static [&'static str],
    execution_kind: ProductionExecutionKind,
    allows_legacy_reference: bool,
    prepare: PrepareModel,
    create_family_registration: CreateFamilyRegistration,
}

const MODEL_LOADERS: &[ModelLoaderRegistration] = &[ModelLoaderRegistration {
    external_metadata_ids: &[
        qwen35::EXTERNAL_METADATA_ID,
        qwen35::MOE_EXTERNAL_METADATA_ID,
    ],
    gguf_architectures: &["qwen35", "qwen35moe"],
    execution_kind: ProductionExecutionKind::CausalLanguage,
    allows_legacy_reference: cfg!(any(test, feature = "test-support")),
    prepare: qwen35::prepare_from_sources,
    create_family_registration: qwen35_family_registration,
}];

/// Returns whether a GGUF architecture belongs to a family whose product
/// execution has migrated to vNext. Direct files for these architectures must
/// provide typed semantic and tokenizer sources instead of silently falling
/// back to a legacy executor.
pub fn gguf_architecture_requires_typed_product_sources(architecture: &str) -> bool {
    MODEL_LOADERS
        .iter()
        .any(|registration| registration.gguf_architectures.contains(&architecture))
}

fn qwen35_family_registration() -> ferrum_types::Result<Box<dyn ModelFamilyRegistration>> {
    let provider = qwen35::Qwen35FamilyProvider::new()
        .map_err(|error| ferrum_types::FerrumError::model(error.to_string()))?;
    Ok(Box::new(TypedFamilyRegistration::new(provider)))
}

/// Complete typed family registry for production vNext registrations.
/// It is derived from `MODEL_LOADERS`, so family preparation and resolved-plan
/// revalidation cannot drift into separate model-name switch statements.
pub struct ProductionModelFamilyRegistry {
    registrations: Vec<Box<dyn ModelFamilyRegistration>>,
}

impl ProductionModelFamilyRegistry {
    pub fn new() -> ferrum_types::Result<Self> {
        let registrations = MODEL_LOADERS
            .iter()
            .map(|registration| (registration.create_family_registration)())
            .collect::<ferrum_types::Result<Vec<_>>>()?;
        Ok(Self { registrations })
    }
}

impl ModelFamilyRegistry for ProductionModelFamilyRegistry {
    fn registrations(&self) -> Vec<&dyn ModelFamilyRegistration> {
        self.registrations
            .iter()
            .map(|registration| registration.as_ref())
            .collect()
    }
}

struct LegacyModelRegistration {
    external_metadata_id: &'static str,
    allows_legacy_reference: bool,
}

/// Explicit migration ledger for safetensors families that still use the
/// legacy executor registry. Moving a family to vNext requires deleting its
/// row here and adding one `MODEL_LOADERS` row; unknown metadata never gains a
/// legacy fallback implicitly.
const LEGACY_MODELS: &[LegacyModelRegistration] = &[
    LegacyModelRegistration {
        external_metadata_id: "hf.architecture.LlamaForCausalLM",
        allows_legacy_reference: false,
    },
    LegacyModelRegistration {
        external_metadata_id: "hf.architecture.Qwen2ForCausalLM",
        allows_legacy_reference: false,
    },
    LegacyModelRegistration {
        external_metadata_id: "hf.architecture.Qwen3ForCausalLM",
        allows_legacy_reference: false,
    },
    LegacyModelRegistration {
        external_metadata_id: "hf.architecture.Qwen3MoeForCausalLM",
        allows_legacy_reference: false,
    },
    LegacyModelRegistration {
        external_metadata_id: "hf.architecture.Gemma3ForCausalLM",
        allows_legacy_reference: false,
    },
    LegacyModelRegistration {
        external_metadata_id: "hf.architecture.Gemma3ForConditionalGeneration",
        allows_legacy_reference: false,
    },
    LegacyModelRegistration {
        external_metadata_id: "hf.architecture.MistralForCausalLM",
        allows_legacy_reference: false,
    },
    LegacyModelRegistration {
        external_metadata_id: "hf.architecture.PhiForCausalLM",
        allows_legacy_reference: false,
    },
    LegacyModelRegistration {
        external_metadata_id: "hf.architecture.GPT2LMHeadModel",
        allows_legacy_reference: false,
    },
    LegacyModelRegistration {
        external_metadata_id: "hf.architecture.BertModel",
        allows_legacy_reference: false,
    },
    LegacyModelRegistration {
        external_metadata_id: "hf.architecture.BertForMaskedLM",
        allows_legacy_reference: false,
    },
    LegacyModelRegistration {
        external_metadata_id: "hf.architecture.BertForSequenceClassification",
        allows_legacy_reference: false,
    },
    LegacyModelRegistration {
        external_metadata_id: "hf.architecture.CLIPModel",
        allows_legacy_reference: false,
    },
    LegacyModelRegistration {
        external_metadata_id: "hf.architecture.ChineseCLIPModel",
        allows_legacy_reference: false,
    },
    LegacyModelRegistration {
        external_metadata_id: "hf.architecture.SiglipModel",
        allows_legacy_reference: false,
    },
    LegacyModelRegistration {
        external_metadata_id: "hf.architecture.WhisperForConditionalGeneration",
        allows_legacy_reference: false,
    },
    LegacyModelRegistration {
        external_metadata_id: "hf.architecture.Qwen3TTSForConditionalGeneration",
        allows_legacy_reference: false,
    },
];

/// Product executor category selected by model semantics, never by a family
/// name. New families using existing operations register against an existing
/// kind without changing the engine composition root.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductionExecutionKind {
    CausalLanguage,
}

/// Runtime-facing language-model facts produced by a typed family package.
///
/// These values replace the legacy engine's second parse through
/// `ModelDefinition`. Non-zero fields and head compatibility are checked once
/// while the package is prepared, before an execution plan is compiled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CausalLanguageModelDescriptor {
    architecture: String,
    parameter_count: NonZeroU64,
    hidden_size: NonZeroUsize,
    layer_count: NonZeroUsize,
    attention_head_count: NonZeroUsize,
    kv_head_count: NonZeroUsize,
    attention_head_dimension: NonZeroUsize,
    vocabulary_size: NonZeroUsize,
    maximum_sequence_tokens: NonZeroUsize,
    execution_dtype: DataType,
}

impl CausalLanguageModelDescriptor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        architecture: impl Into<String>,
        parameter_count: u64,
        hidden_size: usize,
        layer_count: usize,
        attention_head_count: usize,
        kv_head_count: usize,
        attention_head_dimension: usize,
        vocabulary_size: usize,
        maximum_sequence_tokens: usize,
        execution_dtype: DataType,
    ) -> ferrum_types::Result<Self> {
        let architecture = architecture.into();
        if architecture.trim().is_empty() || architecture.trim() != architecture {
            return Err(ferrum_types::FerrumError::model(
                "architecture capability id must be non-empty and canonical",
            ));
        }
        let parameter_count = NonZeroU64::new(parameter_count)
            .ok_or_else(|| ferrum_types::FerrumError::model("parameter_count must be positive"))?;
        let hidden_size = NonZeroUsize::new(hidden_size)
            .ok_or_else(|| ferrum_types::FerrumError::model("hidden_size must be positive"))?;
        let layer_count = NonZeroUsize::new(layer_count)
            .ok_or_else(|| ferrum_types::FerrumError::model("layer_count must be positive"))?;
        let attention_head_count = NonZeroUsize::new(attention_head_count).ok_or_else(|| {
            ferrum_types::FerrumError::model("attention_head_count must be positive")
        })?;
        let kv_head_count = NonZeroUsize::new(kv_head_count)
            .ok_or_else(|| ferrum_types::FerrumError::model("kv_head_count must be positive"))?;
        let attention_head_dimension =
            NonZeroUsize::new(attention_head_dimension).ok_or_else(|| {
                ferrum_types::FerrumError::model("attention_head_dimension must be positive")
            })?;
        let vocabulary_size = NonZeroUsize::new(vocabulary_size)
            .ok_or_else(|| ferrum_types::FerrumError::model("vocabulary_size must be positive"))?;
        let maximum_sequence_tokens =
            NonZeroUsize::new(maximum_sequence_tokens).ok_or_else(|| {
                ferrum_types::FerrumError::model("maximum_sequence_tokens must be positive")
            })?;
        if kv_head_count.get() > attention_head_count.get() {
            return Err(ferrum_types::FerrumError::model(format!(
                "kv_head_count {} exceeds attention_head_count {}",
                kv_head_count, attention_head_count
            )));
        }
        if attention_head_count.get() % kv_head_count.get() != 0 {
            return Err(ferrum_types::FerrumError::model(format!(
                "attention_head_count {} is not divisible by kv_head_count {}",
                attention_head_count, kv_head_count
            )));
        }
        attention_head_count
            .get()
            .checked_mul(attention_head_dimension.get())
            .ok_or_else(|| {
                ferrum_types::FerrumError::model("attention projection width overflows usize")
            })?;
        if !execution_dtype.is_float() {
            return Err(ferrum_types::FerrumError::model(format!(
                "causal language execution dtype must be floating point, got {execution_dtype:?}"
            )));
        }
        Ok(Self {
            architecture,
            parameter_count,
            hidden_size,
            layer_count,
            attention_head_count,
            kv_head_count,
            attention_head_dimension,
            vocabulary_size,
            maximum_sequence_tokens,
            execution_dtype,
        })
    }

    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    pub const fn parameter_count(&self) -> u64 {
        self.parameter_count.get()
    }

    pub const fn hidden_size(&self) -> usize {
        self.hidden_size.get()
    }

    pub const fn layer_count(&self) -> usize {
        self.layer_count.get()
    }

    pub const fn attention_head_count(&self) -> usize {
        self.attention_head_count.get()
    }

    pub const fn kv_head_count(&self) -> usize {
        self.kv_head_count.get()
    }

    pub const fn attention_head_dimension(&self) -> usize {
        self.attention_head_dimension.get()
    }

    pub const fn vocabulary_size(&self) -> usize {
        self.vocabulary_size.get()
    }

    pub const fn maximum_sequence_tokens(&self) -> usize {
        self.maximum_sequence_tokens.get()
    }

    pub const fn execution_dtype(&self) -> DataType {
        self.execution_dtype
    }
}

/// One immutable semantic family and the exact indexed weight source used to
/// initialize its execution plan. Keeping them together prevents product code
/// from resolving a family from one model directory and loading bytes from
/// another.
pub struct PreparedProductionModel {
    family: PreparedModelFamily,
    weights: Arc<dyn WeightComponentSource>,
    descriptor: CausalLanguageModelDescriptor,
    sources: Arc<ProductionModelSourceBundle>,
}

impl std::fmt::Debug for PreparedProductionModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedProductionModel")
            .field("family_id", self.family.family_id())
            .field("descriptor", &self.descriptor)
            .field("sources", &self.sources)
            .finish_non_exhaustive()
    }
}

impl PreparedProductionModel {
    pub(super) fn new(
        family: PreparedModelFamily,
        weights: impl WeightComponentSource + 'static,
        descriptor: CausalLanguageModelDescriptor,
        sources: Arc<ProductionModelSourceBundle>,
    ) -> Self {
        Self {
            family,
            weights: Arc::new(weights),
            descriptor,
            sources,
        }
    }

    pub fn family(&self) -> &PreparedModelFamily {
        &self.family
    }

    pub fn weights(&self) -> &dyn WeightComponentSource {
        self.weights.as_ref()
    }

    pub fn weight_source(&self) -> &Arc<dyn WeightComponentSource> {
        &self.weights
    }

    pub fn descriptor(&self) -> &CausalLanguageModelDescriptor {
        &self.descriptor
    }

    pub fn sources(&self) -> &Arc<ProductionModelSourceBundle> {
        &self.sources
    }

    pub fn product_source_identity(
        &self,
        requested_model: impl Into<String>,
        resolved_model: impl Into<String>,
    ) -> ferrum_types::Result<ferrum_interfaces::vnext::ProductModelSourceIdentity> {
        let template = &self.family.metadata().template;
        self.sources.product_source_identity(
            requested_model,
            resolved_model,
            &template.source_file,
            &template.template,
        )
    }

    pub const fn execution_kind(&self) -> ProductionExecutionKind {
        ProductionExecutionKind::CausalLanguage
    }

    /// Projects the immutable typed package into startup auto-configuration.
    /// Fixed sequence state comes directly from the semantic program; token-
    /// scaled state (for example KV) remains under the plan/runtime capacity
    /// policy and is deliberately not double-counted here.
    pub fn model_capabilities(&self) -> ferrum_types::Result<ModelCapabilities> {
        let estimated_weight_bytes = self.sources.weight_payload_bytes()?;
        let recurrent_state_bytes_per_sequence = self
            .family
            .program()
            .states()
            .iter()
            .filter(|state| {
                state.lifetime == StateLifetime::Sequence
                    && state.capacity_demand == StateCapacityDemand::FixedPerScope
            })
            .try_fold(0_u64, |total, state| {
                let bytes = state.tensor.byte_len().map_err(|error| {
                    ferrum_types::FerrumError::model(format!(
                        "fixed per-sequence state {} has invalid storage size: {error}",
                        state.id
                    ))
                })?;
                total.checked_add(bytes).ok_or_else(|| {
                    ferrum_types::FerrumError::model(
                        "fixed per-sequence state byte size overflows u64",
                    )
                })
            })?;
        let quantization_formats = self.family.weight_schema().quantization_formats();
        let quantization = (!quantization_formats.is_empty()).then(|| {
            quantization_formats
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("+")
        });
        let mut supported_dtypes =
            BTreeSet::from([data_type_label(self.descriptor.execution_dtype())]);
        supported_dtypes.extend(
            self.family
                .program()
                .states()
                .iter()
                .filter_map(|state| element_type_label(state.tensor.element_type)),
        );
        let moe = moe_capabilities_from_program(&self.family)?;

        Ok(ModelCapabilities {
            architecture: self.descriptor.architecture().to_owned(),
            quantization,
            moe,
            max_context_len: Some(self.descriptor.maximum_sequence_tokens()),
            num_hidden_layers: Some(self.descriptor.layer_count()),
            head_dim: Some(self.descriptor.attention_head_dimension()),
            kv_heads: Some(self.descriptor.kv_head_count()),
            estimated_weight_bytes: (estimated_weight_bytes > 0).then_some(estimated_weight_bytes),
            recurrent_state_bytes_per_sequence: (recurrent_state_bytes_per_sequence > 0)
                .then_some(recurrent_state_bytes_per_sequence),
            supported_dtypes: supported_dtypes.into_iter().collect(),
            graph_safe_moe: false,
        })
    }

    /// Projects typed package facts into the transitional `ModelExecutor`
    /// metadata contract. The family id is descriptive only and never drives
    /// executor selection.
    pub fn model_info(&self, model_id: ModelId, device: Device) -> ModelInfo {
        ModelInfo {
            model_id,
            model_type: ModelType::Custom(self.family.family_id().to_string()),
            num_parameters: self.descriptor.parameter_count(),
            hidden_size: self.descriptor.hidden_size(),
            num_layers: self.descriptor.layer_count(),
            num_heads: self.descriptor.attention_head_count(),
            num_kv_heads: self.descriptor.kv_head_count(),
            vocab_size: self.descriptor.vocabulary_size(),
            max_sequence_length: self.descriptor.maximum_sequence_tokens(),
            dtype: self.descriptor.execution_dtype(),
            device,
            version: None,
            license: None,
            metadata: Default::default(),
        }
    }
}

fn moe_capabilities_from_program(
    family: &PreparedModelFamily,
) -> ferrum_types::Result<Option<MoeCapabilities>> {
    let mut capabilities = None;
    for node in family
        .program()
        .blocks()
        .iter()
        .flat_map(|block| &block.nodes)
        .filter(|node| node.operation_id.as_str() == ROUTED_SHARED_SWIGLU_MOE_OPERATION_ID)
    {
        let current = MoeCapabilities {
            num_experts: required_positive_node_attribute(node, "expert_count")?,
            experts_per_token: required_positive_node_attribute(node, "experts_per_token")?,
            moe_intermediate_size: Some(required_positive_node_attribute(
                node,
                "routed_intermediate_size",
            )?),
        };
        if current.experts_per_token > current.num_experts {
            return Err(ferrum_types::FerrumError::model(format!(
                "MoE program node {} routes {} experts per token from only {} experts",
                node.id, current.experts_per_token, current.num_experts
            )));
        }
        if capabilities
            .as_ref()
            .is_some_and(|expected| expected != &current)
        {
            return Err(ferrum_types::FerrumError::model(format!(
                "MoE program node {} has capabilities inconsistent with earlier layers",
                node.id
            )));
        }
        capabilities = Some(current);
    }
    Ok(capabilities)
}

fn required_positive_node_attribute(
    node: &ProgramNode,
    attribute: &str,
) -> ferrum_types::Result<usize> {
    let attribute_id = AttributeId::new(attribute).map_err(|error| {
        ferrum_types::FerrumError::internal(format!(
            "standard MoE attribute id {attribute:?} is invalid: {error}"
        ))
    })?;
    let Some(SemanticValue::Unsigned(value)) = node.attributes.get(&attribute_id) else {
        return Err(ferrum_types::FerrumError::model(format!(
            "MoE program node {} lacks unsigned attribute {attribute:?}",
            node.id
        )));
    };
    let value = usize::try_from(*value).map_err(|_| {
        ferrum_types::FerrumError::model(format!(
            "MoE program node {} attribute {attribute:?} exceeds usize",
            node.id
        ))
    })?;
    if value == 0 {
        return Err(ferrum_types::FerrumError::model(format!(
            "MoE program node {} attribute {attribute:?} must be positive",
            node.id
        )));
    }
    Ok(value)
}

fn data_type_label(data_type: DataType) -> String {
    data_type.to_string().to_ascii_lowercase()
}

fn element_type_label(element_type: ElementType) -> Option<String> {
    match element_type {
        ElementType::F16 => Some("fp16".to_owned()),
        ElementType::Bf16 => Some("bf16".to_owned()),
        ElementType::F32 => Some("fp32".to_owned()),
        ElementType::Bool
        | ElementType::U8
        | ElementType::U32
        | ElementType::I8
        | ElementType::I32 => None,
    }
}

/// A resolved static registration. Resolution reads only `config.json`; model
/// weights are not opened until [`RegisteredProductionModel::prepare`] after
/// the engine has selected a compatible backend composition.
pub struct RegisteredProductionModel {
    registration: &'static ModelLoaderRegistration,
    external_metadata_id: ExternalModelMetadataId,
}

impl RegisteredProductionModel {
    pub fn external_metadata_id(&self) -> &ExternalModelMetadataId {
        &self.external_metadata_id
    }

    pub const fn execution_kind(&self) -> ProductionExecutionKind {
        self.registration.execution_kind
    }

    pub fn prepare(&self, model_dir: &Path) -> ferrum_types::Result<PreparedProductionModel> {
        let current = external_metadata_id_from_model_dir(model_dir)?;
        if current != self.external_metadata_id {
            return Err(ferrum_types::FerrumError::model(format!(
                "model metadata identity changed between registration and preparation: expected {} got {}",
                self.external_metadata_id, current
            )));
        }
        let sources = Arc::new(ProductionModelSourceBundle::open_colocated_safetensors(
            model_dir,
        )?);
        self.prepare_from_sources(sources)
    }

    pub fn prepare_from_sources(
        &self,
        sources: Arc<ProductionModelSourceBundle>,
    ) -> ferrum_types::Result<PreparedProductionModel> {
        let current = external_metadata_id_from_bytes(sources.config_json(), "config.json")?;
        if current != self.external_metadata_id {
            return Err(ferrum_types::FerrumError::model(format!(
                "model metadata identity changed between registration and preparation: expected {} got {}",
                self.external_metadata_id, current
            )));
        }
        let prepared = (self.registration.prepare)(sources)?;
        if prepared.family().external_metadata_id() != &self.external_metadata_id {
            return Err(ferrum_types::FerrumError::model(format!(
                "registered model loader returned metadata identity {} for resolved identity {}",
                prepared.family().external_metadata_id(),
                self.external_metadata_id
            )));
        }
        if prepared.execution_kind() != self.registration.execution_kind {
            return Err(ferrum_types::FerrumError::model(format!(
                "registered model loader returned execution kind {:?} for registered kind {:?}",
                prepared.execution_kind(),
                self.registration.execution_kind
            )));
        }
        Ok(prepared)
    }
}

/// Explicit migration result keyed only by external model metadata.
///
/// Product paths that require vNext must consume this through
/// [`ProductionModelRegistration::into_required`]. A legacy result exists only
/// for an explicit registry row; unknown metadata is rejected during
/// resolution and can never enter the old architecture cascade implicitly.
pub enum ProductionModelRegistration {
    Registered(RegisteredProductionModel),
    LegacyRegistered {
        external_metadata_id: ExternalModelMetadataId,
        allows_legacy_reference: bool,
    },
}

impl ProductionModelRegistration {
    pub fn external_metadata_id(&self) -> &ExternalModelMetadataId {
        match self {
            Self::Registered(registration) => registration.external_metadata_id(),
            Self::LegacyRegistered {
                external_metadata_id,
                ..
            } => external_metadata_id,
        }
    }

    pub const fn allows_legacy_reference(&self) -> bool {
        match self {
            Self::Registered(registration) => registration.registration.allows_legacy_reference,
            Self::LegacyRegistered {
                allows_legacy_reference,
                ..
            } => *allows_legacy_reference,
        }
    }

    /// Requires a registered vNext production package and fails closed instead
    /// of allowing a product caller to fall back to a legacy executor.
    pub fn into_required(self) -> ferrum_types::Result<RegisteredProductionModel> {
        match self {
            Self::Registered(registration) => Ok(registration),
            Self::LegacyRegistered {
                external_metadata_id,
                ..
            } => Err(ferrum_types::FerrumError::unsupported(format!(
                "model family metadata {external_metadata_id} is registered for the legacy runtime only; vNext product fallback is forbidden"
            ))),
        }
    }
}

pub fn resolve_registered_model_from_dir(
    model_dir: &Path,
) -> ferrum_types::Result<ProductionModelRegistration> {
    let external_metadata_id = external_metadata_id_from_model_dir(model_dir)?;
    resolve_registered_model(external_metadata_id)
}

pub fn resolve_registered_model_from_sources(
    sources: &ProductionModelSourceBundle,
) -> ferrum_types::Result<ProductionModelRegistration> {
    let external_metadata_id =
        external_metadata_id_from_bytes(sources.config_json(), "config.json")?;
    resolve_registered_model(external_metadata_id)
}

fn resolve_registered_model(
    external_metadata_id: ExternalModelMetadataId,
) -> ferrum_types::Result<ProductionModelRegistration> {
    let mut loaders = MODEL_LOADERS.iter().filter(|registration| {
        registration
            .external_metadata_ids
            .contains(&external_metadata_id.as_str())
    });
    let loader = loaders.next();
    if loaders.next().is_some() {
        return Err(ferrum_types::FerrumError::internal(format!(
            "model metadata {external_metadata_id} has duplicate vNext production registrations"
        )));
    }
    let mut legacy_rows = LEGACY_MODELS
        .iter()
        .filter(|registration| registration.external_metadata_id == external_metadata_id.as_str());
    let legacy = legacy_rows.next();
    if legacy_rows.next().is_some() {
        return Err(ferrum_types::FerrumError::internal(format!(
            "model metadata {external_metadata_id} has duplicate legacy registrations"
        )));
    }

    match (loader, legacy) {
        (Some(_), Some(_)) => Err(ferrum_types::FerrumError::internal(format!(
            "model metadata {external_metadata_id} is registered in both vNext and legacy registries"
        ))),
        (Some(registration), None) => Ok(ProductionModelRegistration::Registered(
            RegisteredProductionModel {
                registration,
                external_metadata_id,
            },
        )),
        (None, Some(registration)) => Ok(ProductionModelRegistration::LegacyRegistered {
            external_metadata_id,
            allows_legacy_reference: registration.allows_legacy_reference,
        }),
        (None, None) => Err(ferrum_types::FerrumError::unsupported(format!(
            "model metadata {external_metadata_id} is absent from both the vNext and explicit legacy registries; implicit architecture fallback is forbidden"
        ))),
    }
}

fn external_metadata_id_from_model_dir(
    model_dir: &Path,
) -> ferrum_types::Result<ExternalModelMetadataId> {
    let path = model_dir.join("config.json");
    let raw = fs::read(&path)
        .map_err(|error| ferrum_types::FerrumError::model(format!("read {path:?}: {error}")))?;
    external_metadata_id_from_bytes(&raw, &path.display().to_string())
}

fn external_metadata_id_from_bytes(
    raw: &[u8],
    source: &str,
) -> ferrum_types::Result<ExternalModelMetadataId> {
    let config: Value = serde_json::from_slice(raw)
        .map_err(|error| ferrum_types::FerrumError::model(format!("parse {source}: {error}")))?;
    let architectures = config
        .get("architectures")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ferrum_types::FerrumError::model(
                "config.json must declare exactly one architectures entry for typed family resolution",
            )
        })?;
    let architecture = match architectures.as_slice() {
        [value] => value.as_str().filter(|value| !value.is_empty()),
        _ => None,
    }
    .ok_or_else(|| {
        ferrum_types::FerrumError::model(
            "config.json must declare exactly one non-empty architecture identity",
        )
    })?;
    ExternalModelMetadataId::new(format!("hf.architecture.{architecture}"))
        .map_err(|error| ferrum_types::FerrumError::model(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_production_selection_rejects_legacy_family() {
        let metadata = ExternalModelMetadataId::new("hf.architecture.LlamaForCausalLM").unwrap();
        let selection = ProductionModelRegistration::LegacyRegistered {
            external_metadata_id: metadata,
            allows_legacy_reference: false,
        };

        let error = match selection.into_required() {
            Ok(_) => panic!("legacy family unexpectedly entered the vNext product path"),
            Err(error) => error.to_string(),
        };

        assert!(
            error.contains("hf.architecture.LlamaForCausalLM"),
            "{error}"
        );
        assert!(
            error.contains("vNext product fallback is forbidden"),
            "{error}"
        );
    }

    #[test]
    fn registry_ids_are_unique_and_disjoint() {
        let mut ids = std::collections::BTreeSet::new();
        let mut gguf_architectures = std::collections::BTreeSet::new();
        for registration in MODEL_LOADERS {
            for external_metadata_id in registration.external_metadata_ids {
                assert!(
                    ids.insert(*external_metadata_id),
                    "duplicate vNext registration {external_metadata_id}"
                );
            }
            for architecture in registration.gguf_architectures {
                assert!(
                    gguf_architectures.insert(*architecture),
                    "GGUF architecture {architecture} maps to more than one vNext registration"
                );
            }
        }
        for registration in LEGACY_MODELS {
            assert!(
                ids.insert(registration.external_metadata_id),
                "registration {} appears in both or more than once",
                registration.external_metadata_id
            );
        }
    }

    #[test]
    fn migrated_gguf_architecture_requires_typed_product_sources() {
        assert!(gguf_architecture_requires_typed_product_sources("qwen35"));
        assert!(gguf_architecture_requires_typed_product_sources(
            "qwen35moe"
        ));
        assert!(!gguf_architecture_requires_typed_product_sources("qwen3"));
    }

    #[test]
    fn production_family_registry_is_derived_from_loader_rows() {
        let registry = ProductionModelFamilyRegistry::new().unwrap();
        assert_eq!(registry.registrations().len(), MODEL_LOADERS.len());
        let metadata = ExternalModelMetadataId::new(qwen35::EXTERNAL_METADATA_ID).unwrap();
        let registration = (&registry as &dyn ModelFamilyRegistry)
            .resolve_external(&metadata)
            .unwrap();
        assert_eq!(registration.family_id().as_str(), qwen35::FAMILY_ID);
        let moe_metadata = ExternalModelMetadataId::new(qwen35::MOE_EXTERNAL_METADATA_ID).unwrap();
        let moe_registration = (&registry as &dyn ModelFamilyRegistry)
            .resolve_external(&moe_metadata)
            .unwrap();
        assert_eq!(moe_registration.family_id().as_str(), qwen35::FAMILY_ID);
    }

    #[test]
    fn qwen35_moe_metadata_resolves_to_vnext_product_loader() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(
            directory.path().join("config.json"),
            r#"{"architectures":["Qwen3_5MoeForConditionalGeneration"]}"#,
        )
        .unwrap();

        let registration = resolve_registered_model_from_dir(directory.path())
            .unwrap()
            .into_required()
            .unwrap();
        assert_eq!(
            registration.external_metadata_id().as_str(),
            qwen35::MOE_EXTERNAL_METADATA_ID
        );
    }

    #[test]
    fn unknown_metadata_cannot_enter_an_implicit_legacy_path() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(
            directory.path().join("config.json"),
            r#"{"architectures":["UnregisteredQwenForConditionalGeneration"]}"#,
        )
        .unwrap();

        let error = resolve_registered_model_from_dir(directory.path())
            .err()
            .expect("unknown architecture must fail closed")
            .to_string();
        assert!(
            error.contains("hf.architecture.UnregisteredQwenForConditionalGeneration"),
            "{error}"
        );
        assert!(
            error.contains("implicit architecture fallback is forbidden"),
            "{error}"
        );
    }

    #[test]
    fn registration_resolution_does_not_open_weights_and_rechecks_identity() {
        let directory = tempfile::tempdir().unwrap();
        let config_path = directory.path().join("config.json");
        fs::write(
            &config_path,
            r#"{"architectures":["Qwen3_5ForConditionalGeneration"]}"#,
        )
        .unwrap();

        let registration = resolve_registered_model_from_dir(directory.path())
            .unwrap()
            .into_required()
            .unwrap();
        assert_eq!(
            registration.external_metadata_id().as_str(),
            qwen35::EXTERNAL_METADATA_ID
        );
        assert_eq!(
            registration.execution_kind(),
            ProductionExecutionKind::CausalLanguage
        );

        fs::write(
            config_path,
            r#"{"architectures":["DifferentForConditionalGeneration"]}"#,
        )
        .unwrap();
        let error = match registration.prepare(directory.path()) {
            Ok(_) => panic!("changed metadata unexpectedly prepared"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("metadata identity changed"), "{error}");
        assert!(error.contains(qwen35::EXTERNAL_METADATA_ID), "{error}");
    }

    #[test]
    fn causal_language_descriptor_rejects_invalid_runtime_facts() {
        assert!(CausalLanguageModelDescriptor::new(
            "test",
            0,
            16,
            2,
            2,
            1,
            4,
            32,
            128,
            DataType::FP16,
        )
        .is_err());
        assert!(CausalLanguageModelDescriptor::new(
            "test",
            1,
            16,
            2,
            2,
            1,
            0,
            32,
            128,
            DataType::FP16,
        )
        .is_err());
        assert!(CausalLanguageModelDescriptor::new(
            "test",
            1,
            16,
            2,
            2,
            3,
            4,
            32,
            128,
            DataType::FP16,
        )
        .is_err());
        assert!(CausalLanguageModelDescriptor::new(
            "test",
            1,
            16,
            2,
            3,
            2,
            4,
            32,
            128,
            DataType::FP16,
        )
        .is_err());
        assert!(CausalLanguageModelDescriptor::new(
            "test",
            1,
            16,
            2,
            2,
            1,
            4,
            32,
            128,
            DataType::INT8,
        )
        .is_err());
        assert!(
            CausalLanguageModelDescriptor::new("test", 1, 15, 2, 2, 1, 4, 32, 128, DataType::FP16,)
                .is_ok(),
            "hidden width and explicit attention projection width are independent facts"
        );
    }
}
