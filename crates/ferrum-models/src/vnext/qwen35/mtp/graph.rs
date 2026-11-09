use super::*;
use ferrum_runtime::{
    CAUSAL_PAGED_ATTENTION_EXPLICIT_ROPE_F16_OPERATION_ID, CONCAT_COLUMNS_F16_OPERATION_ID,
    DENSE_LINEAR_OPERATION_ID, TAIL_TWO_GREEDY_F16_OPERATION_ID,
};

fn node(
    role: &str,
    operation: &str,
    version: ContractVersion,
    inputs: Vec<ProgramValueId>,
    outputs: Vec<ProgramValueId>,
    attributes: BTreeMap<AttributeId, SemanticValue>,
) -> Result<ProgramNode, VNextError> {
    Ok(ProgramNode {
        id: node_id(format!("node.mtp.{role}"))?,
        operation_id: operation_id(operation)?,
        required_version: version,
        work: ProgramNodeWorkSpec::tokens(inputs[0].clone(), 0),
        inputs,
        outputs,
        attributes,
    })
}

fn hidden(role: &str) -> Result<ProgramValueId, VNextError> {
    value_id(format!("value.mtp.hidden.{role}"))
}

fn draft(
    c: &Qwen35MtpFamilyConfig,
    references: Vec<WeightReference>,
) -> Result<(ModelProgram, NumericalExecutionProfile), VNextError> {
    let t = Qwen35FamilyProvider::text_config(&c.main)?;
    let h = t.hidden_size as u64;
    let i = t
        .dense_intermediate_size
        .ok_or_else(|| invalid_config("mtp", "dense FFN width missing"))? as u64;
    let two_h = h
        .checked_mul(2)
        .ok_or_else(|| invalid_config("mtp", "concat width overflows"))?;
    let w = weights::value;
    let version = ContractVersion::new(1, 0);
    let transport = ContractVersion::new(1, 0);
    let norm_attrs = BTreeMap::from([
        attribute("hidden_size", h)?,
        attribute("epsilon", c.main.rms_norm_epsilon)?,
    ]);
    let mut nodes = vec![node(
        "embedding",
        TOKEN_EMBEDDING_OPERATION_ID,
        version,
        vec![
            value_id(TOKEN_IDS)?,
            weight_value_id(required_weight(&c.main, None, "embed_tokens")?)?,
        ],
        vec![hidden("embedding")?],
        BTreeMap::from([
            attribute("hidden_size", h)?,
            attribute("vocab_size", c.main.vocab_size)?,
        ]),
    )?];
    for (role, input, weight) in [
        (
            "norm_embedding",
            hidden("embedding")?,
            "pre_fc_norm_embedding",
        ),
        (
            "norm_hidden",
            value_id(ACCEPTED_HIDDEN)?,
            "pre_fc_norm_hidden",
        ),
    ] {
        nodes.push(node(
            role,
            RMS_NORM_OPERATION_ID,
            version,
            vec![input, w(weight)?],
            vec![hidden(role)?],
            norm_attrs.clone(),
        )?);
    }
    nodes.push(node(
        "concat",
        CONCAT_COLUMNS_F16_OPERATION_ID,
        transport,
        vec![hidden("norm_embedding")?, hidden("norm_hidden")?],
        vec![hidden("concat")?],
        BTreeMap::from([
            attribute("left_features", h)?,
            attribute("right_features", h)?,
            attribute("out_features", two_h)?,
        ]),
    )?);
    nodes.push(node(
        "fc",
        DENSE_LINEAR_OPERATION_ID,
        ContractVersion::new(1, 1),
        vec![hidden("concat")?, w("fc")?],
        vec![hidden("projected")?],
        BTreeMap::from([
            attribute("in_features", two_h)?,
            attribute("out_features", h)?,
        ]),
    )?);
    let mut state = crate::vnext::numerical::kv_state(
        0,
        t.num_key_value_heads as u64,
        t.head_dim as u64,
        c.main.max_position_embeddings,
    )?;
    state.id = StateId::new("state.mtp.layer.0.kv")?;
    state.value_id = value_id("value.state.mtp.layer.0.kv")?;
    let mut attention = vec![hidden("projected")?, w("layers.0.input_layernorm")?];
    for role in ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"] {
        attention.push(w(&format!("layers.0.self_attn.{role}"))?);
    }
    attention.extend([
        state.value_id.clone(),
        value_id(ROPE_COS)?,
        value_id(ROPE_SIN)?,
    ]);
    nodes.push(node(
        "attention",
        CAUSAL_PAGED_ATTENTION_EXPLICIT_ROPE_F16_OPERATION_ID,
        transport,
        attention,
        vec![hidden("attention")?],
        BTreeMap::from([
            attribute("query_heads", t.num_attention_heads as u64)?,
            attribute("key_value_heads", t.num_key_value_heads as u64)?,
            attribute("head_dim", t.head_dim as u64)?,
            attribute("hidden_size", h)?,
            attribute("query_features", t.full_attention_query_total_dim() as u64)?,
            attribute(
                "query_projection_features",
                t.full_attention_q_proj_total_dim() as u64,
            )?,
            attribute("kv_features", t.full_attention_kv_total_dim() as u64)?,
            attribute("rope_dim", t.full_attention_rope_dim() as u64)?,
            attribute("rope_pairs", (t.full_attention_rope_dim() / 2) as u64)?,
            attribute("maximum_context_tokens", c.main.max_position_embeddings)?,
            attribute(
                "rope_theta",
                canonical_positive_f64(t.rope_parameters.rope_theta)?,
            )?,
            attribute("rope_interleaved", false)?,
            attribute("output_gate", true)?,
            attribute("causal", true)?,
            attribute("epsilon", c.main.rms_norm_epsilon)?,
            attribute("layer_index", 0u64)?,
        ]),
    )?);
    nodes.push(node(
        "post_attention_norm",
        RMS_NORM_OPERATION_ID,
        version,
        vec![
            hidden("attention")?,
            w("layers.0.post_attention_layernorm")?,
        ],
        vec![hidden("post_attention_norm")?],
        norm_attrs.clone(),
    )?);
    nodes.push(node(
        "mlp",
        DENSE_SWIGLU_OPERATION_ID,
        version,
        vec![
            hidden("post_attention_norm")?,
            w("layers.0.mlp.gate_up")?,
            w("layers.0.mlp.down_proj")?,
        ],
        vec![hidden("mlp")?],
        BTreeMap::from([
            attribute("hidden_size", h)?,
            attribute("intermediate_size", i)?,
        ]),
    )?);
    nodes.push(node(
        "residual",
        RESIDUAL_ADD_OPERATION_ID,
        version,
        vec![hidden("attention")?, hidden("mlp")?],
        vec![hidden("residual")?],
        BTreeMap::from([attribute("hidden_size", h)?]),
    )?);
    nodes.push(node(
        "final_norm",
        RMS_NORM_OPERATION_ID,
        version,
        vec![hidden("residual")?, w("norm")?],
        vec![hidden("final_norm")?],
        norm_attrs,
    )?);
    let projection = c
        .main
        .weights
        .iter()
        .find(|w| w.layer_index.is_none() && w.role == "lm_head")
        .unwrap_or(required_weight(&c.main, None, "embed_tokens")?);
    nodes.push(node(
        "head",
        TAIL_TWO_GREEDY_F16_OPERATION_ID,
        transport,
        vec![hidden("final_norm")?, weight_value_id(projection)?],
        vec![value_id(GREEDY_TOKENS)?, value_id(HIDDEN_TAIL)?],
        BTreeMap::from([
            attribute("hidden_size", h)?,
            attribute("vocab_size", c.main.vocab_size)?,
        ]),
    )?);
    let mut operations = BTreeMap::new();
    let mut boundaries = BTreeMap::new();
    for n in &nodes {
        for output in &n.outputs {
            boundaries.insert(
                output.clone(),
                if output.as_str() == GREEDY_TOKENS {
                    ElementType::U32
                } else {
                    ElementType::F16
                },
            );
        }
        let pure = matches!(
            n.operation_id.as_str(),
            TOKEN_EMBEDDING_OPERATION_ID | CONCAT_COLUMNS_F16_OPERATION_ID
        );
        operations.insert(
            n.operation_id.clone(),
            NumericalOperationContract {
                operation_id: n.operation_id.clone(),
                version: n.required_version,
                multiplication_type: (!pure
                    && n.operation_id.as_str() != RESIDUAL_ADD_OPERATION_ID)
                    .then_some(ElementType::F32),
                accumulation_type: (!pure).then_some(ElementType::F32),
                staged_arithmetic: None,
                composite_arithmetic: None,
            },
        );
    }
    let family_id = ModelFamilyId::new(FAMILY_ID)?;
    let profile = NumericalExecutionProfile {
        id: NumericalProfileId::new(Qwen35MtpRole::OneDraft.numerical_profile())
            .map_err(|e| invalid_config("mtp.profile", e))?,
        version: transport,
        family_id: family_id.clone(),
        primary_activation: hidden("embedding")?,
        boundaries,
        states: vec![state.clone()],
        kv_storage: vec![KvStateStorage::F16 {
            state: state.id.clone(),
        }],
        operations: operations.into_values().collect(),
    };
    let program = ModelProgram::new(
        family_id,
        [TOKEN_IDS, ACCEPTED_HIDDEN, ROPE_COS, ROPE_SIN]
            .into_iter()
            .map(value_id)
            .collect::<Result<_, _>>()?,
        vec![ProgramBlock {
            id: "block.mtp.one_draft".into(),
            nodes,
        }],
        vec![state],
        references,
        vec![value_id(GREEDY_TOKENS)?, value_id(HIDDEN_TAIL)?],
    )?;
    // No immutable conditioning declaration: accepted hidden is dynamically
    // authorized per wave by the future driver, not a known full sequence table.
    Ok((program, profile))
}

pub(super) fn build(
    c: &Qwen35MtpFamilyConfig,
) -> Result<(ModelProgram, NumericalExecutionProfile), VNextError> {
    let original = Qwen35FamilyProvider::new()?;
    let profiles = original.numerical_profiles(&c.main)?;
    let main_profile = profiles.resolve(
        &NumericalProfileId::new(F16_VISUAL_PROMPT_NUMERICAL_PROFILE_ID)
            .map_err(|e| invalid_config("mtp.profile", e))?,
    )?;
    let main = original.semantic_program(&c.main, main_profile)?;
    let references = weights::union_references(c, &main)?;
    if c.role == Qwen35MtpRole::MainVerify {
        return super::verify::build(c, &main, main_profile, references);
    }
    let (draft, draft_profile) = draft(c, references.clone())?;
    if c.role == Qwen35MtpRole::OneDraft {
        return Ok((draft, draft_profile));
    }
    let mut profile = main_profile.clone();
    profile.family_id = ModelFamilyId::new(FAMILY_ID)?;
    profile.id = NumericalProfileId::new(c.role.numerical_profile())
        .map_err(|e| invalid_config("mtp.profile", e))?;
    let mut inputs = main.inputs().to_vec();
    let mut blocks = main.blocks().to_vec();
    let mut states = main.states().to_vec();
    let mut outputs = main.outputs().to_vec();
    if c.role == Qwen35MtpRole::InstallUnion {
        inputs.extend_from_slice(draft.inputs());
        blocks.extend_from_slice(draft.blocks());
        states.extend_from_slice(draft.states());
        outputs.extend_from_slice(draft.outputs());
        profile.boundaries.extend(draft_profile.boundaries);
        profile.states.extend(draft_profile.states);
        profile.kv_storage.extend(draft_profile.kv_storage);
        for operation in draft_profile.operations {
            if let Some(existing) = profile
                .operations
                .iter()
                .find(|o| o.operation_id == operation.operation_id)
            {
                if existing != &operation {
                    return Err(invalid_config(
                        "mtp.profile",
                        "main/draft numerical operation contracts disagree",
                    ));
                }
            } else {
                profile.operations.push(operation);
            }
        }
    }
    profile.states.sort_by(|a, b| a.id.cmp(&b.id));
    profile
        .kv_storage
        .sort_by(|a, b| a.payload_state().cmp(b.payload_state()));
    profile
        .operations
        .sort_by(|a, b| a.operation_id.cmp(&b.operation_id));
    let program = ModelProgram::new(
        profile.family_id.clone(),
        inputs,
        blocks,
        states,
        references,
        outputs,
    )?;
    let program = if c.role == Qwen35MtpRole::MainWithDraftWeights {
        program.with_checkpoint_inputs(
            main.checkpoint_inputs()
                .expect("original main binding")
                .clone(),
        )?
    } else {
        program
    };
    Ok((program, profile))
}
