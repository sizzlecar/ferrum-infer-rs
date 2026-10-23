use super::*;

pub(super) const JOINT: &str = "value.qwen_image21.transformer.joint";
pub(super) const COMBINED_LATENTS: &str = "value.qwen_image21.transformer.combined_latents";
pub(super) const UNORDERED_JOINT: &str = "value.qwen_image21.transformer.unordered_joint";
pub(super) const MODULATION: &str = "value.qwen_image21.transformer.modulation_table";
pub(super) const CONDITIONING: [&str; 4] = [
    "value.qwen_image21.transformer.scale1",
    "value.qwen_image21.transformer.gate1",
    "value.qwen_image21.transformer.scale2",
    "value.qwen_image21.transformer.gate2",
];
pub(super) const TANH_GATES: [&str; 2] = [
    "value.qwen_image21.transformer.tanh_gate1",
    "value.qwen_image21.transformer.tanh_gate2",
];
const TEXT_NORM: &str = "value.qwen_image21.transformer.text_norm";
const TEXT_IN: &str = "value.qwen_image21.transformer.text_in";
const TEXT_GELU: &str = "value.qwen_image21.transformer.text_gelu";
const TEXT_OUT: &str = "value.qwen_image21.transformer.text_projected";
const IMAGE_OUT: &str = "value.qwen_image21.transformer.image_projected";
const TIME_IN: &str = "value.qwen_image21.transformer.time_in";
const TIME_SILU: &str = "value.qwen_image21.transformer.time_silu";
const TEMB: &str = "value.qwen_image21.transformer.time_embedding";
const MOD_SILU: &str = "value.qwen_image21.transformer.modulation_silu";
pub(super) const TARGET_HIDDEN: &str = "value.qwen_image21.transformer.target_hidden";
pub(super) const FINAL_SCALE: &str = "value.qwen_image21.transformer.final_scale";
const FINAL_SILU: &str = "value.qwen_image21.transformer.final_silu";
const FINAL_SCALE_TABLE: &str = "value.qwen_image21.transformer.final_scale_table";
const FINAL_NORM: &str = "value.qwen_image21.transformer.final_norm";
const FINAL_MOD: &str = "value.qwen_image21.transformer.final_modulation";

pub(super) fn weight_value(name: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.qwen_image21.transformer.weight.{name}"))
}

pub(super) fn build(
    family: &QwenImage21TransformerFamily,
    config: &QwenImage21TransformerConfig,
    block_profile: &str,
) -> Result<ModelProgram, VNextError> {
    type F = QwenImage21TransformerFamily;
    let t = config.transformer();
    let l = config.layout();
    let h = t.hidden_size();
    let full = ExpectedWeights::from_config(t, StructureLimits::default())
        .map_err(|e| invalid("weights", e.to_string()))?;
    let mut weights = full
        .tensors()
        .values()
        .map(|tensor| {
            Ok(WeightReference {
                weight_id: WeightId::new(format!(
                    "weight.qwen_image21.{}",
                    tensor.external_name()
                ))?,
                value_id: weight_value(tensor.external_name())?,
                tensor: ProgramTensorSpec {
                    dimensions: tensor.dimensions().to_vec(),
                    element_type: ElementType::Bf16,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            })
        })
        .collect::<Result<Vec<_>, VNextError>>()?;
    let epsilon = BTreeMap::from([(
        AttributeId::new("epsilon")?,
        SemanticValue::Rational(t.epsilon()),
    )]);
    let mut nodes = Vec::new();
    if l.reference_image().is_some() && !config.is_decode() {
        nodes.push(node(
            "combined_latents",
            BATCHED_ROW_CONCAT_BF16_OPERATION_ID,
            vec![id(F::REFERENCE_LATENTS_ID)?, id(F::LATENTS_ID)?],
            COMBINED_LATENTS,
            BTreeMap::from([(
                AttributeId::new("batch")?,
                SemanticValue::Unsigned(l.batch()),
            )]),
        )?);
    }
    nodes.extend([
        node(
            "text_norm",
            ZERO_CENTERED_RMS_NORM_BF16_OPERATION_ID,
            vec![id(F::TEXT_ID)?, weight_value("txt_in.text_norm.weight")?],
            TEXT_NORM,
            epsilon.clone(),
        )?,
        dense(
            "text_in",
            TEXT_NORM,
            "txt_in.in_layer.weight",
            TEXT_IN,
            t.context_dim(),
            h,
        )?,
        unary("text_gelu", GELU_TANH_BF16_OPERATION_ID, TEXT_IN, TEXT_GELU)?,
        dense(
            "text_out",
            TEXT_GELU,
            "txt_in.out_layer.weight",
            TEXT_OUT,
            h,
            h,
        )?,
        dense(
            "image_in",
            if l.reference_image().is_some() && !config.is_decode() {
                COMBINED_LATENTS
            } else {
                F::LATENTS_ID
            },
            "img_in.weight",
            IMAGE_OUT,
            t.in_channels(),
            h,
        )?,
        node(
            "joint",
            BATCHED_ROW_CONCAT_BF16_OPERATION_ID,
            vec![id(TEXT_OUT)?, id(IMAGE_OUT)?],
            if l.reference_image().is_some() {
                UNORDERED_JOINT
            } else {
                JOINT
            },
            BTreeMap::from([(
                AttributeId::new("batch")?,
                SemanticValue::Unsigned(l.batch()),
            )]),
        )?,
        dense(
            "time_in",
            F::TIME_FEATURES_ID,
            "time_text_embed.timestep_embedder.linear_1.weight",
            TIME_IN,
            256,
            h,
        )?,
        unary("time_silu", SILU_BF16_OPERATION_ID, TIME_IN, TIME_SILU)?,
        dense(
            "time_out",
            TIME_SILU,
            "time_text_embed.timestep_embedder.linear_2.weight",
            TEMB,
            h,
            h,
        )?,
        unary("modulation_silu", SILU_BF16_OPERATION_ID, TEMB, MOD_SILU)?,
        dense(
            "modulation_table",
            MOD_SILU,
            "modulation.1.weight",
            MODULATION,
            h,
            4 * h,
        )?,
    ]);
    if config.is_decode() {
        nodes.retain(|node| {
            ![
                TEXT_NORM,
                TEXT_IN,
                TEXT_GELU,
                TEXT_OUT,
                JOINT,
                UNORDERED_JOINT,
            ]
            .contains(&node.outputs[0].as_str())
        });
        let image = nodes
            .iter_mut()
            .find(|node| node.outputs[0].as_str() == IMAGE_OUT)
            .ok_or_else(|| invalid("program", "decode lacks image projection"))?;
        image.outputs[0] = id(JOINT)?;
    }
    if let Some(reference) = l.reference_image().filter(|_| !config.is_decode()) {
        let text = u64::from(l.text_tokens());
        let start = u64::from(reference.image_slot_start());
        let slots = u64::from(reference.image_slots());
        let refs = reference.latent_tokens();
        let stride = text + refs + l.target_tokens();
        nodes.push(node(
            "interleaved_joint",
            ROW_GATHER_BF16_OPERATION_ID,
            vec![id(UNORDERED_JOINT)?],
            JOINT,
            gather_attributes(
                l.batch(),
                0,
                h,
                &[
                    (start, 0, stride, 1),
                    (refs, text, stride, 1),
                    (text - start - slots, start + slots, stride, 1),
                    (l.target_tokens(), text + refs, stride, 1),
                ],
            )?,
        )?);
    }
    for (index, output) in CONDITIONING.into_iter().enumerate() {
        nodes.push(node(
            &format!("conditioning{index}"),
            ROW_GATHER_BF16_OPERATION_ID,
            vec![id(MODULATION)?],
            output,
            conditioning_attributes(config, index as u64 * h)?,
        )?);
    }
    // Every layer consumes the same raw timestep gates. Preserve the unary
    // BF16 boundary once and share its immutable result across the layer graph.
    // Recomputing this independent branch per layer prevents dependency-based
    // memory reuse even though the arithmetic has identical inputs.
    for (index, raw) in [CONDITIONING[1], CONDITIONING[3]].into_iter().enumerate() {
        nodes.push(unary(
            &format!("tanh_gate{}", index + 1),
            TANH_BF16_OPERATION_ID,
            raw,
            TANH_GATES[index],
        )?);
    }
    let block = Block::new()?;
    let block_config = config.block_config()?;
    let profiles = block.numerical_profiles(&block_config)?;
    let source = block.semantic_program(
        &block_config,
        profiles.resolve(&profile_id(block_profile)?)?,
    )?;
    let mut previous = id(JOINT)?;
    for layer in 0..t.layers() {
        previous = remap::append(&mut nodes, &source, &weights, layer, previous, config)?;
    }
    // Final normalization/projection are row-local. Select target rows first,
    // as the upstream SP target-output path does, and retain no prefix output.
    nodes.extend([
        node(
            "target_hidden",
            ROW_GATHER_BF16_OPERATION_ID,
            vec![previous],
            TARGET_HIDDEN,
            gather_attributes(
                l.batch(),
                0,
                h,
                &[(
                    l.target_tokens(),
                    if config.is_decode() {
                        0
                    } else {
                        l.target_start()
                    },
                    if config.is_decode() {
                        l.target_tokens()
                    } else {
                        l.sequence_length()
                    },
                    1,
                )],
            )?,
        )?,
        unary("final_silu", SILU_BF16_OPERATION_ID, TEMB, FINAL_SILU)?,
        dense(
            "final_scale_table",
            FINAL_SILU,
            "norm_out.linear.weight",
            FINAL_SCALE_TABLE,
            h,
            h,
        )?,
        node(
            "final_scale",
            ROW_GATHER_BF16_OPERATION_ID,
            vec![id(FINAL_SCALE_TABLE)?],
            FINAL_SCALE,
            gather_attributes(l.batch(), 0, h, &[(l.target_tokens(), 0, 1, 0)])?,
        )?,
        node(
            "final_norm",
            LAYER_NORM_BF16_OPERATION_ID,
            vec![id(TARGET_HIDDEN)?],
            FINAL_NORM,
            {
                let mut attrs = epsilon;
                attrs.insert(AttributeId::new("hidden_size")?, SemanticValue::Unsigned(h));
                attrs
            },
        )?,
        node(
            "final_modulation",
            MODULATION_SCALE_BF16_OPERATION_ID,
            vec![id(FINAL_NORM)?, id(FINAL_SCALE)?],
            FINAL_MOD,
            BTreeMap::from([(
                AttributeId::new("broadcast")?,
                SemanticValue::Text("elementwise".into()),
            )]),
        )?,
        dense(
            "velocity",
            FINAL_MOD,
            "proj_out.weight",
            F::VELOCITY_ID,
            h,
            t.out_channels(),
        )?,
    ]);
    let mut inputs = F::INPUT_IDS
        .into_iter()
        .map(id)
        .collect::<Result<Vec<_>, _>>()?;
    if config.is_decode() {
        inputs.retain(|input| input.as_str() != F::TEXT_ID);
    } else if l.reference_image().is_some() {
        inputs.push(id(F::REFERENCE_LATENTS_ID)?);
    }
    let mut outputs = vec![id(F::VELOCITY_ID)?];
    if config.execution_phase() == QwenImage21TransformerPhase::PrefixPrefill {
        outputs.extend(
            config
                .prefix_cache_ports()?
                .into_iter()
                .map(|port| port.value_id().clone()),
        );
    }
    if config.is_decode() {
        // The checkpoint/schema stays complete, while this phase declares only
        // values it actually consumes. Shared installation admission verifies
        // their unchanged physical placements against the complete snapshot.
        let consumed = nodes
            .iter()
            .flat_map(|node| &node.inputs)
            .collect::<BTreeSet<_>>();
        weights.retain(|weight| consumed.contains(&weight.value_id));
    }
    ModelProgram::new(
        family.id.clone(),
        inputs,
        vec![ProgramBlock {
            id: match config.execution_phase() {
                QwenImage21TransformerPhase::PrefixPrefill => {
                    "block.qwen_image21.transformer.prefix_prefill"
                }
                QwenImage21TransformerPhase::PrefixDecode => {
                    "block.qwen_image21.transformer.prefix_decode"
                }
                QwenImage21TransformerPhase::Full if l.reference_image().is_some() => {
                    "block.qwen_image21.transformer.edit"
                }
                QwenImage21TransformerPhase::Full => "block.qwen_image21.transformer.t2i",
            }
            .into(),
            nodes,
        }],
        config.prefix_states()?,
        weights,
        outputs,
    )?
    .with_input_tensor_specs(config.input_tensor_specs()?)
}

pub(super) fn conditioning_attributes(
    config: &QwenImage21TransformerConfig,
    column_start: u64,
) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
    let l = config.layout();
    let runs = if config.is_decode() {
        vec![(l.target_tokens(), 0, 1, 0)]
    } else if config.transformer().causal_condition() {
        vec![
            (l.target_start(), l.batch(), 0, 0),
            (l.target_tokens(), 0, 1, 0),
        ]
    } else {
        vec![(l.sequence_length(), 0, 1, 0)]
    };
    gather_attributes(
        l.batch(),
        column_start,
        config.transformer().hidden_size(),
        &runs,
    )
}

/// Tuple fields are (output length, source start, batch stride, row stride).
pub(super) fn gather_attributes(
    batch: u64,
    column_start: u64,
    column_count: u64,
    runs: &[(u64, u64, u64, u64)],
) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
    let mut attrs = BTreeMap::from([
        (AttributeId::new("batch")?, SemanticValue::Unsigned(batch)),
        (
            AttributeId::new("column_start")?,
            SemanticValue::Unsigned(column_start),
        ),
        (
            AttributeId::new("column_count")?,
            SemanticValue::Unsigned(column_count),
        ),
    ]);
    for (index, name) in [
        "run_lengths",
        "source_starts",
        "source_batch_strides",
        "source_row_strides",
    ]
    .into_iter()
    .enumerate()
    {
        let values = runs
            .iter()
            .map(|run| {
                i64::try_from([run.0, run.1, run.2, run.3][index])
                    .map_err(|_| invalid("layout", "row-map metadata exceeds i64"))
            })
            .collect::<Result<_, _>>()?;
        attrs.insert(AttributeId::new(name)?, SemanticValue::Integers(values));
    }
    Ok(attrs)
}
fn id(value: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(value)
}
fn unary(
    name: &str,
    operation: &str,
    input: &str,
    output: &str,
) -> Result<ProgramNode, VNextError> {
    node(name, operation, vec![id(input)?], output, BTreeMap::new())
}
fn dense(
    name: &str,
    input: &str,
    weight: &str,
    output: &str,
    columns: u64,
    outputs: u64,
) -> Result<ProgramNode, VNextError> {
    node(
        name,
        DENSE_LINEAR_BF16_OPERATION_ID,
        vec![id(input)?, weight_value(weight)?],
        output,
        BTreeMap::from([
            (
                AttributeId::new("in_features")?,
                SemanticValue::Unsigned(columns),
            ),
            (
                AttributeId::new("out_features")?,
                SemanticValue::Unsigned(outputs),
            ),
        ]),
    )
}
fn node(
    name: &str,
    operation: &str,
    inputs: Vec<ProgramValueId>,
    output: &str,
    attributes: BTreeMap<AttributeId, SemanticValue>,
) -> Result<ProgramNode, VNextError> {
    Ok(ProgramNode {
        id: NodeId::new(format!("node.qwen_image21.transformer.{name}"))?,
        operation_id: OperationId::new(operation)?,
        required_version: ContractVersion::new(1, 0),
        work: ProgramNodeWorkSpec::Fixed,
        inputs,
        outputs: vec![id(output)?],
        attributes,
    })
}
