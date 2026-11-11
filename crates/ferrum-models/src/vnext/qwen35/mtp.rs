//! Opt-in, single-draft Qwen3.5 MTP graphs over the original source archive.
//!
//! The draft consumes an already accepted, final-normalized main hidden row.
//! Its paged KV cursor is independent of the supplied main-position RoPE.
//! This module does not authorize dynamic hidden inputs or implement speculative
//! acceptance, scheduling or rollback. InstallUnion is an installation graph,
//! not a language generation entrypoint.

use super::*;
use ferrum_runtime::{KvStateStorage, NumericalOperationContract, NumericalProfileId};

mod graph;
mod verify;
mod weights;

pub const FAMILY_ID: &str = "family.qwen3_5.mtp.one_draft";
pub const TOKEN_IDS: &str = "value.mtp.input.token_ids";
pub const ACCEPTED_HIDDEN: &str = "value.mtp.input.accepted_hidden";
pub const ROPE_COS: &str = "value.mtp.input.rope_cos";
pub const ROPE_SIN: &str = "value.mtp.input.rope_sin";
pub const GREEDY_TOKENS: &str = "value.mtp.output.greedy_tokens";
pub const HIDDEN_TAIL: &str = "value.mtp.output.accepted_hidden";
pub const MAIN_GREEDY_TOKENS: &str = "value.output.greedy_tokens";
pub const MAIN_HIDDEN_TAIL: &str = "value.output.accepted_hidden";

/// Supply the sole internal shape that needs arithmetic (H+H). The generic
/// Core deliberately does not interpret operation attributes as dimension
/// equations. Call this after declaring actual input capacities and before
/// compiling OneDraft or InstallUnion. Existing conflicting shapes reject.
pub fn configure_compile_options(
    family: &PreparedModelFamily,
    options: &mut ferrum_runtime::ProgramPlanCompileOptions,
) -> Result<(), VNextError> {
    if family.family_id().as_str() != FAMILY_ID {
        return Err(invalid_config("mtp.compile", "foreign model family"));
    }
    let config = Qwen35MtpFamilyProvider::new()?.parse_config(family.canonical_config())?;
    if matches!(
        config.role,
        Qwen35MtpRole::MainWithDraftWeights | Qwen35MtpRole::MainVerify
    ) {
        return Ok(());
    }
    let token = options
        .tensor_specs()
        .get(&value_id(TOKEN_IDS)?)
        .ok_or_else(|| invalid_config("mtp.compile", "draft token capacity is missing"))?;
    let [rows] = token.dimensions.as_slice() else {
        return Err(invalid_config(
            "mtp.compile",
            "draft token IDs must have rank one",
        ));
    };
    if *rows == 0
        || token.element_type != ElementType::U32
        || token.layout != ResolvedTensorLayout::Contiguous
    {
        return Err(invalid_config(
            "mtp.compile",
            "draft token capacity must be contiguous U32",
        ));
    }
    let hidden = Qwen35FamilyProvider::text_config(&config.main)?.hidden_size as u64;
    let shape = tensor_spec(
        vec![
            *rows,
            hidden
                .checked_mul(2)
                .ok_or_else(|| invalid_config("mtp.compile", "concat width overflows"))?,
        ],
        ElementType::F16,
    );
    let value = value_id("value.mtp.hidden.concat")?;
    if options
        .tensor_specs()
        .get(&value)
        .is_some_and(|existing| existing != &shape)
    {
        return Err(invalid_config(
            "mtp.compile",
            "explicit concat shape conflicts with typed inputs",
        ));
    }
    options.insert_tensor_spec(value, shape)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen35MtpRole {
    MainWithDraftWeights,
    /// Greedy-only main scan; every step requires an explicit selected-prefix
    /// commit. The two terminal slots are q0/q1 and their final-norm h0/h1.
    MainVerify,
    OneDraft,
    InstallUnion,
}

impl Qwen35MtpRole {
    pub const fn numerical_profile(self) -> &'static str {
        match self {
            Self::MainWithDraftWeights => "qwen3_5.mtp.main_visual_f16_v1",
            Self::MainVerify => "qwen3_5.mtp.main_verify_f16_v1",
            Self::OneDraft => "qwen3_5.mtp.one_draft_f16_v1",
            Self::InstallUnion => "qwen3_5.mtp.install_union_f16_v1",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen35MtpFamilyConfig {
    main: Qwen35FamilyConfig,
    role: Qwen35MtpRole,
    /// Actual header identities. Payload conversion stays in the ordinary
    /// safetensors materializer, including the original norm +1 rule.
    mtp_tensors: Vec<weights::MtpTensor>,
}

impl Qwen35MtpFamilyConfig {
    pub const fn role(&self) -> Qwen35MtpRole {
        self.role
    }
    pub fn with_role(&self, role: Qwen35MtpRole) -> Self {
        Self {
            role,
            ..self.clone()
        }
    }
}

pub struct Qwen35MtpFamilyProvider {
    family_id: ModelFamilyId,
}

impl Qwen35MtpFamilyProvider {
    pub fn new() -> Result<Self, VNextError> {
        Ok(Self {
            family_id: ModelFamilyId::new(FAMILY_ID)?,
        })
    }

    fn validate(&self, c: &Qwen35MtpFamilyConfig) -> Result<(), VNextError> {
        Qwen35FamilyProvider::new()?.validate_typed_config(&c.main)?;
        let text = Qwen35FamilyProvider::text_config(&c.main)?;
        if c.main.weight_format != FamilyWeightFormat::SafetensorsDense
            || text.is_moe()
            || !text.attn_output_gate
        {
            return Err(invalid_config(
                "mtp",
                "one-draft supports the declared dense gated-attention safetensors source only",
            ));
        }
        weights::validate(&text, &c.mtp_tensors)
    }
}

impl ModelFamilyProvider for Qwen35MtpFamilyProvider {
    type Config = Qwen35MtpFamilyConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.family_id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([
            ExternalModelMetadataId::new(EXTERNAL_METADATA_ID).expect("static metadata id")
        ])
    }
    fn parse_config(&self, raw: &Value) -> Result<Self::Config, VNextError> {
        let c = serde_json::from_value(raw.clone())
            .map_err(|e| invalid_config("mtp.config", e.to_string()))?;
        self.validate(&c)?;
        Ok(c)
    }
    fn validate_config_identity(&self, raw: &Value, c: &Self::Config) -> Result<(), VNextError> {
        self.validate(c)?;
        let typed =
            serde_json::to_value(c).map_err(|e| invalid_config("mtp.config", e.to_string()))?;
        if *raw != typed {
            return Err(invalid_config(
                "mtp.config",
                "raw and typed configuration differ",
            ));
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        c: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        ExternalModelMetadataId::new(EXTERNAL_METADATA_ID)
    }
    fn weights(&self, c: &Self::Config) -> Result<ModelWeights, VNextError> {
        weights::union_schema(c).map(ModelWeights::Stored)
    }
    fn numerical_profiles(&self, c: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let (_, profile) = graph::build(c)?;
        FamilyNumericalProfiles::new(
            &self.family_id,
            ContractVersion::new(1, 0),
            vec![profile.clone()],
            vec![profile.id],
        )
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let (program, declared) = graph::build(c)?;
        if *profile != declared {
            return Err(invalid_config(
                "mtp.profile",
                "profile differs from the declared role",
            ));
        }
        Ok(program)
    }
    fn semantic_metadata(&self, c: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(c.main.metadata.clone().into())
    }
}

/// Metadata-only preparation from the same verified source bundle used by the
/// main model. Each role declares the identical complete physical weight plan.
pub fn define_from_sources(
    sources: Arc<ProductionModelSourceBundle>,
    role: Qwen35MtpRole,
) -> ferrum_types::Result<DefinedProductionModel> {
    let error = |e: String| ferrum_types::FerrumError::model(e);
    let archive = sources.weights().open_safetensors_archive()?;
    let main = load_safetensors_family_config(&sources, &archive).map_err(error)?;
    let text = Qwen35FamilyProvider::text_config(&main).map_err(|e| error(e.to_string()))?;
    let mtp_tensors = weights::read_header(&text, &archive).map_err(|e| error(e.to_string()))?;
    let original = production_descriptor(&main).map_err(error)?;
    let parameters = mtp_tensors
        .iter()
        .try_fold(original.parameter_count(), |total, tensor| {
            tensor
                .dimensions
                .iter()
                .try_fold(1u64, |n, d| n.checked_mul(*d))
                .and_then(|n| total.checked_add(n))
                .ok_or_else(|| error("MTP union parameter count overflows".into()))
        })?;
    let layers = match role {
        Qwen35MtpRole::OneDraft => 1,
        Qwen35MtpRole::MainWithDraftWeights | Qwen35MtpRole::MainVerify => text.num_hidden_layers,
        Qwen35MtpRole::InstallUnion => text
            .num_hidden_layers
            .checked_add(1)
            .ok_or_else(|| error("MTP layer count overflows".into()))?,
    };
    let descriptor = CausalLanguageModelDescriptor::new(
        "qwen3_5_mtp",
        parameters,
        text.hidden_size,
        layers,
        text.num_attention_heads,
        text.num_key_value_heads,
        text.head_dim,
        main.vocab_size as usize,
        main.max_position_embeddings as usize,
        DataType::FP16,
    )?;
    let config = Qwen35MtpFamilyConfig {
        main,
        role,
        mtp_tensors,
    };
    let provider = Qwen35MtpFamilyProvider::new().map_err(|e| error(e.to_string()))?;
    provider
        .validate(&config)
        .map_err(|e| error(e.to_string()))?;
    let raw = serde_json::to_value(config).map_err(|e| error(e.to_string()))?;
    DefinedProductionModel::new(
        TypedFamilyRegistration::new(provider),
        &raw,
        archive,
        descriptor,
        sources,
    )
}
