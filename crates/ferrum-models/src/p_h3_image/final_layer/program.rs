use super::*;
pub(super) const PARAMETERS: &str = "value.p_h3_image.final_layer.parameters";
#[derive(Clone, Copy)]
pub(super) struct Route {
    pub source: u64,
    pub length: u64,
    pub time: u64,
}
pub(super) fn routes(c: &PH3ImageFinalLayerConfig, image: bool) -> Result<Vec<Route>, VNextError> {
    let mut result = Vec::new();
    let mut base = 0u64;
    for request in &c.requests {
        let mut source = base
            .checked_add(request.prefix_rows())
            .and_then(|n| {
                if image {
                    n.checked_add(request.audio_rows().ok()?)
                } else {
                    Some(n)
                }
            })
            .ok_or_else(|| invalid("target source offset overflows"))?;
        for run in if image {
            request.image_runs()
        } else {
            request.audio_runs()
        } {
            result.push(Route {
                source,
                length: run.length(),
                time: run.time_index(),
            });
            source = source
                .checked_add(run.length())
                .ok_or_else(|| invalid("source run overflows"))?;
        }
        base = base
            .checked_add(request.input_rows()?)
            .ok_or_else(|| invalid("request source offset overflows"))?;
    }
    Ok(result)
}
fn value(name: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.p_h3_image.final_layer.{name}"))
}
fn node(
    name: &str,
    operation: &str,
    inputs: Vec<ProgramValueId>,
    output: ProgramValueId,
    attributes: BTreeMap<AttributeId, SemanticValue>,
) -> Result<ProgramNode, VNextError> {
    Ok(ProgramNode {
        id: NodeId::new(format!("node.p_h3_image.final_layer.{name}"))?,
        operation_id: OperationId::new(operation)?,
        required_version: ContractVersion::new(1, 0),
        work: ProgramNodeWorkSpec::Fixed,
        inputs,
        outputs: vec![output],
        attributes,
    })
}
fn affine(
    name: &str,
    module: &str,
    input: ProgramValueId,
    output: ProgramValueId,
    k: u64,
    n: u64,
) -> Result<ProgramNode, VNextError> {
    node(
        name,
        AFFINE_LINEAR_F32_OPERATION_ID,
        vec![
            input,
            weights::value(&format!("{module}.weight"))?,
            weights::value(&format!("{module}.bias"))?,
        ],
        output,
        BTreeMap::from([
            (AttributeId::new("in_features")?, SemanticValue::Unsigned(k)),
            (
                AttributeId::new("out_features")?,
                SemanticValue::Unsigned(n),
            ),
        ]),
    )
}
fn list(
    name: &str,
    values: impl IntoIterator<Item = u64>,
) -> Result<(AttributeId, SemanticValue), VNextError> {
    Ok((
        AttributeId::new(name)?,
        SemanticValue::Integers(
            values
                .into_iter()
                .map(|v| {
                    i64::try_from(v).map_err(|_| invalid("routing exceeds signed attribute extent"))
                })
                .collect::<Result<_, _>>()?,
        ),
    ))
}
pub(super) fn build(
    c: &PH3ImageFinalLayerConfig,
    family: &ModelFamilyId,
) -> Result<ModelProgram, VNextError> {
    c.validate()?;
    let mut nodes = vec![affine(
        "curve_affine",
        "final_layer.adaln_proj.linear",
        ProgramValueId::new(PH3ImageFinalLayerFamily::BASIS_ID)?,
        ProgramValueId::new(PARAMETERS)?,
        c.basis,
        2 * c.hidden,
    )?];
    for image in [true, false] {
        let stream = if image { "image" } else { "audio" };
        let module = if image {
            "final_layer.video_out"
        } else {
            "final_layer.audio_out"
        };
        let final_id = ProgramValueId::new(if image {
            PH3ImageFinalLayerFamily::IMAGE_VELOCITY_ID
        } else {
            PH3ImageFinalLayerFamily::AUDIO_VELOCITY_ID
        })?;
        let width = if image { c.image_width } else { c.audio_width };
        let routes = routes(c, image)?;
        let chunks = routes.chunks(ROW_GATHER_BF16_MAX_RUNS).collect::<Vec<_>>();
        let mut previous = None;
        for (index, chunk) in chunks.iter().enumerate() {
            let prefix = format!("{stream}.chunk{index}");
            let gathered = value(&format!("{prefix}.gathered"))?;
            let gather_attrs = BTreeMap::from([
                (AttributeId::new("batch")?, SemanticValue::Unsigned(1)),
                (
                    AttributeId::new("column_start")?,
                    SemanticValue::Unsigned(0),
                ),
                (
                    AttributeId::new("column_count")?,
                    SemanticValue::Unsigned(c.hidden),
                ),
                list("run_lengths", chunk.iter().map(|r| r.length))?,
                list("source_starts", chunk.iter().map(|r| r.source))?,
                list("source_batch_strides", std::iter::repeat_n(0, chunk.len()))?,
                list("source_row_strides", std::iter::repeat_n(1, chunk.len()))?,
            ]);
            nodes.push(node(
                &format!("{prefix}.gather"),
                ROW_GATHER_BF16_OPERATION_ID,
                vec![ProgramValueId::new(PH3ImageFinalLayerFamily::INPUT_ID)?],
                gathered.clone(),
                gather_attrs,
            )?);
            let normalized = value(&format!("{prefix}.normalized"))?;
            nodes.push(node(
                &format!("{prefix}.norm"),
                ROW_RMS_NORM_SINGLE_ROUND_BF16_OPERATION_ID,
                vec![gathered, weights::value("final_layer.norm.weight")?],
                normalized.clone(),
                BTreeMap::from([(
                    AttributeId::new("epsilon")?,
                    SemanticValue::Rational(c.epsilon),
                )]),
            )?);
            let modulated = value(&format!("{prefix}.modulated"))?;
            nodes.push(node(
                &format!("{prefix}.modulate"),
                ROW_SELECTED_MODULATION_BF16_F32_OPERATION_ID,
                vec![normalized, ProgramValueId::new(PARAMETERS)?],
                modulated.clone(),
                BTreeMap::from([
                    list("run_lengths", chunk.iter().map(|r| r.length))?,
                    list("parameter_rows", chunk.iter().map(|r| r.time))?,
                ]),
            )?);
            let head = value(&format!("{prefix}.head_f32"))?;
            nodes.push(affine(
                &format!("{prefix}.head"),
                module,
                modulated,
                head.clone(),
                c.hidden,
                width,
            )?);
            let velocity = if chunks.len() == 1 {
                final_id.clone()
            } else {
                value(&format!("{prefix}.velocity"))?
            };
            nodes.push(node(
                &format!("{prefix}.round_velocity"),
                CAST_F32_TO_BF16_OPERATION_ID,
                vec![head],
                velocity.clone(),
                BTreeMap::new(),
            )?);
            previous = Some(if let Some(left) = previous {
                let output = if index + 1 == chunks.len() {
                    final_id.clone()
                } else {
                    value(&format!("{stream}.combined{index}"))?
                };
                nodes.push(node(
                    &format!("{stream}.concat{index}"),
                    BATCHED_ROW_CONCAT_BF16_OPERATION_ID,
                    vec![left, velocity],
                    output.clone(),
                    BTreeMap::from([(AttributeId::new("batch")?, SemanticValue::Unsigned(1))]),
                )?);
                output
            } else {
                velocity
            });
        }
    }
    ModelProgram::new(
        family.clone(),
        vec![
            ProgramValueId::new(PH3ImageFinalLayerFamily::INPUT_ID)?,
            ProgramValueId::new(PH3ImageFinalLayerFamily::BASIS_ID)?,
        ],
        vec![ProgramBlock {
            id: "block.p_h3_image.final_layer".into(),
            nodes,
        }],
        vec![],
        weights::references(c)?,
        vec![
            ProgramValueId::new(PH3ImageFinalLayerFamily::IMAGE_VELOCITY_ID)?,
            ProgramValueId::new(PH3ImageFinalLayerFamily::AUDIO_VELOCITY_ID)?,
        ],
    )?
    .with_input_tensor_specs(c.input_tensor_specs()?)
}
pub(super) fn tensor_specs(
    c: &PH3ImageFinalLayerConfig,
    graph: &ModelProgram,
) -> Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
    let mut specs = c.input_tensor_specs()?;
    specs.extend(
        graph
            .weights()
            .iter()
            .map(|w| (w.value_id.clone(), w.tensor.clone())),
    );
    for node in graph.blocks().iter().flat_map(|b| &b.nodes) {
        let mut output = specs[&node.inputs[0]].clone();
        let read = |name: &str| -> Result<u64, VNextError> {
            match node.attributes.get(&AttributeId::new(name)?) {
                Some(SemanticValue::Unsigned(v)) => Ok(*v),
                _ => Err(invalid("missing unsigned shape attribute")),
            }
        };
        match node.operation_id.as_str() {
            AFFINE_LINEAR_F32_OPERATION_ID => {
                output.dimensions[1] = read("out_features")?;
                output.element_type = ElementType::F32;
            }
            ROW_GATHER_BF16_OPERATION_ID => {
                let Some(SemanticValue::Integers(lengths)) =
                    node.attributes.get(&AttributeId::new("run_lengths")?)
                else {
                    return Err(invalid("missing gather lengths"));
                };
                let rows = lengths
                    .iter()
                    .try_fold(0u64, |n, l| {
                        u64::try_from(*l).ok().and_then(|l| n.checked_add(l))
                    })
                    .ok_or_else(|| invalid("gather output rows overflow"))?;
                output.dimensions = vec![
                    rows.checked_mul(read("batch")?)
                        .ok_or_else(|| invalid("gather batch rows overflow"))?,
                    read("column_count")?,
                ];
            }
            ROW_RMS_NORM_SINGLE_ROUND_BF16_OPERATION_ID => {}
            ROW_SELECTED_MODULATION_BF16_F32_OPERATION_ID => output.element_type = ElementType::F32,
            CAST_F32_TO_BF16_OPERATION_ID => output.element_type = ElementType::Bf16,
            BATCHED_ROW_CONCAT_BF16_OPERATION_ID => {
                output.dimensions[0] = output.dimensions[0]
                    .checked_add(specs[&node.inputs[1]].dimensions[0])
                    .ok_or_else(|| invalid("concat output rows overflow"))?
            }
            _ => {
                return Err(invalid(
                    "new final operation requires explicit output geometry",
                ))
            }
        }
        output.validate("final node output")?;
        for value in &node.outputs {
            specs.insert(value.clone(), output.clone());
        }
    }
    Ok(specs)
}
pub(super) fn compile_options(
    c: &PH3ImageFinalLayerConfig,
) -> Result<ProgramPlanCompileOptions, VNextError> {
    let graph = build(c, &ModelFamilyId::new(PH3ImageFinalLayerFamily::FAMILY_ID)?)?;
    let specs = tensor_specs(c, &graph)?;
    let mut options = ProgramPlanCompileOptions::new(c.input_tensor_specs()?)?;
    for value in graph
        .blocks()
        .iter()
        .flat_map(|b| &b.nodes)
        .flat_map(|n| &n.outputs)
    {
        options.insert_tensor_spec(value.clone(), specs[value].clone())?;
    }
    Ok(options)
}
