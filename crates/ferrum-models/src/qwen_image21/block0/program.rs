use super::*;

pub(super) fn build(
    family: &QwenImage21Block0Family,
    config: &QwenImage21Block0Config,
    profile: &NumericalExecutionProfile,
) -> Result<ModelProgram, VNextError> {
    type F = QwenImage21Block0Family;
    let (_, attention_profile, mlp_profile) = profile_sources()
        .into_iter()
        .find(|(name, _, _)| *name == profile.id.as_str())
        .ok_or_else(|| invalid("profile", "unknown full-block profile"))?;
    let attention = Attention::new()?;
    let attention_profiles = attention.numerical_profiles(config.attention())?;
    let attention = attention.semantic_program(
        config.attention(),
        attention_profiles.resolve(&profile_id(attention_profile)?)?,
    )?;
    let mlp = Mlp::new()?;
    let mlp_profiles = mlp.numerical_profiles(config.transformer())?;
    let mlp = mlp.semantic_program(
        config.transformer(),
        mlp_profiles.resolve(&profile_id(mlp_profile)?)?,
    )?;
    if !attention.states().is_empty() || !mlp.states().is_empty() {
        return Err(invalid(
            "program",
            "block branches must not introduce persistent states",
        ));
    }
    let norm_attributes = BTreeMap::from([
        (
            AttributeId::new("hidden_size")?,
            SemanticValue::Unsigned(config.transformer().hidden_size()),
        ),
        (
            AttributeId::new("epsilon")?,
            SemanticValue::Rational(config.transformer().epsilon()),
        ),
    ]);
    let broadcast = BTreeMap::from([(
        AttributeId::new("broadcast")?,
        SemanticValue::Text(config.broadcast().as_str().into()),
    )]);
    let mut nodes = vec![
        node(
            "norm1",
            LAYER_NORM_BF16_OPERATION_ID,
            &[F::INPUT_ID],
            F::NORMALIZED1_ID,
            norm_attributes.clone(),
        )?,
        node(
            "modulation1",
            MODULATION_SCALE_BF16_OPERATION_ID,
            &[F::NORMALIZED1_ID, F::SCALE1_ID],
            F::MODULATED1_ID,
            broadcast.clone(),
        )?,
        node(
            "tanh_gate1",
            TANH_BF16_OPERATION_ID,
            &[F::GATE1_ID],
            F::TANH_GATE1_ID,
            BTreeMap::new(),
        )?,
    ];
    append_branch(
        &mut nodes,
        &attention,
        "attention",
        &[
            (Attention::INPUT_ID, F::MODULATED1_ID),
            (Attention::FREQUENCIES_ID, F::FREQUENCIES_ID),
        ],
    )?;
    nodes.extend([
        node(
            "residual1",
            GATED_RESIDUAL_BF16_OPERATION_ID,
            &[F::INPUT_ID, F::TANH_GATE1_ID, Attention::OUTPUT_ID],
            F::RESIDUAL1_ID,
            broadcast.clone(),
        )?,
        node(
            "norm2",
            LAYER_NORM_BF16_OPERATION_ID,
            &[F::RESIDUAL1_ID],
            F::NORMALIZED2_ID,
            norm_attributes,
        )?,
        node(
            "modulation2",
            MODULATION_SCALE_BF16_OPERATION_ID,
            &[F::NORMALIZED2_ID, F::SCALE2_ID],
            F::MODULATED2_ID,
            broadcast.clone(),
        )?,
        node(
            "tanh_gate2",
            TANH_BF16_OPERATION_ID,
            &[F::GATE2_ID],
            F::TANH_GATE2_ID,
            BTreeMap::new(),
        )?,
    ]);
    append_branch(
        &mut nodes,
        &mlp,
        "mlp",
        &[(Mlp::INPUT_ID, F::MODULATED2_ID)],
    )?;
    nodes.push(node(
        "residual2",
        GATED_RESIDUAL_BF16_OPERATION_ID,
        &[F::RESIDUAL1_ID, F::TANH_GATE2_ID, Mlp::OUTPUT_ID],
        F::OUTPUT_ID,
        broadcast,
    )?);
    let mut weights = attention.weights().to_vec();
    weights.extend_from_slice(mlp.weights());
    ModelProgram::new(
        family.id.clone(),
        F::INPUT_IDS
            .into_iter()
            .map(ProgramValueId::new)
            .collect::<Result<_, _>>()?,
        vec![ProgramBlock {
            id: "block.qwen_image21.block0".into(),
            nodes,
        }],
        vec![],
        weights,
        F::OUTPUT_IDS
            .into_iter()
            .map(ProgramValueId::new)
            .collect::<Result<_, _>>()?,
    )?
    .with_input_tensor_specs(config.input_tensor_specs()?)
}

fn node(
    name: &str,
    operation: &str,
    inputs: &[&str],
    output: &str,
    attributes: BTreeMap<AttributeId, SemanticValue>,
) -> Result<ProgramNode, VNextError> {
    Ok(ProgramNode {
        id: NodeId::new(format!("node.qwen_image21.block0.{name}"))?,
        operation_id: OperationId::new(operation)?,
        required_version: ContractVersion::new(1, 0),
        work: ProgramNodeWorkSpec::Fixed,
        inputs: inputs
            .iter()
            .map(|name| ProgramValueId::new(*name))
            .collect::<Result<_, _>>()?,
        outputs: vec![ProgramValueId::new(output)?],
        attributes,
    })
}

/// Reuse exact branch algorithms and weights. Only entry values and node
/// identities change; operation attributes, ordering and BF16 values survive.
fn append_branch(
    nodes: &mut Vec<ProgramNode>,
    branch: &ModelProgram,
    name: &str,
    entries: &[(&str, &str)],
) -> Result<(), VNextError> {
    let mapping = entries
        .iter()
        .map(|(from, to)| Ok((ProgramValueId::new(*from)?, ProgramValueId::new(*to)?)))
        .collect::<Result<BTreeMap<_, _>, VNextError>>()?;
    if mapping.keys().cloned().collect::<BTreeSet<_>>() != branch.inputs().iter().cloned().collect()
    {
        return Err(invalid(
            "program",
            "branch remapping must cover every input exactly",
        ));
    }
    for (index, source) in branch
        .blocks()
        .iter()
        .flat_map(|block| &block.nodes)
        .enumerate()
    {
        if source.work != ProgramNodeWorkSpec::Fixed {
            return Err(invalid(
                "program",
                "full block requires fixed tensor branch work",
            ));
        }
        let mut node = source.clone();
        node.id = NodeId::new(format!("node.qwen_image21.block0.{name}.{index}"))?;
        for input in &mut node.inputs {
            if let Some(replacement) = mapping.get(input) {
                *input = replacement.clone();
            }
        }
        nodes.push(node);
    }
    Ok(())
}
