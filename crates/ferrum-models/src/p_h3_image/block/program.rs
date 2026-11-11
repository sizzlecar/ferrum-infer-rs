use super::*;
const TABLE_F32: &str = "value.p_h3_image.curve_block.adaln_f32";
const TABLE_BF16: &str = "value.p_h3_image.curve_block.adaln_bf16";
const TABLE_ROWS: &str = "value.p_h3_image.curve_block.adaln_rows";
pub(super) const PARAMETERS: [&str; 6] = [
    "value.p_h3_image.curve_block.shift1",
    "value.p_h3_image.curve_block.scale1",
    "value.p_h3_image.curve_block.gate1",
    "value.p_h3_image.curve_block.shift2",
    "value.p_h3_image.curve_block.scale2",
    "value.p_h3_image.curve_block.gate2",
];
pub(super) const NORM1: &str = "value.p_h3_image.curve_block.norm1";
pub(super) const MOD1: &str = "value.p_h3_image.curve_block.mod1";
pub(super) const SCALED1: &str = "value.p_h3_image.curve_block.scaled1";
pub(super) const QKV: &str = "value.p_h3_image.curve_block.qkv";
pub(super) const Q: &str = "value.p_h3_image.curve_block.q";
pub(super) const K: &str = "value.p_h3_image.curve_block.k";
pub(super) const V: &str = "value.p_h3_image.curve_block.v";
pub(super) const QR: &str = "value.p_h3_image.curve_block.q_rotary";
pub(super) const KR: &str = "value.p_h3_image.curve_block.k_rotary";
pub(super) const ATTENTION: &str = "value.p_h3_image.curve_block.attention";
pub(super) const ATTENTION_OUT: &str = "value.p_h3_image.curve_block.attention_out";
pub(super) const RESIDUAL1: &str = "value.p_h3_image.curve_block.residual1";
pub(super) const NORM2: &str = "value.p_h3_image.curve_block.norm2";
pub(super) const SCALED2: &str = "value.p_h3_image.curve_block.scaled2";
pub(super) const MOD2: &str = "value.p_h3_image.curve_block.mod2";
pub(super) const GATE_UP: &str = "value.p_h3_image.curve_block.gate_up";
pub(super) const MLP: &str = "value.p_h3_image.curve_block.mlp";
pub(super) fn compact_parameters(c: &PH3ImageCurveBlockConfig) -> bool {
    c.requests
        .iter()
        .try_fold(0usize, |n, r| n.checked_add(r.runs().len()))
        .is_some_and(|runs| runs <= ROW_SELECTED_BF16_MAX_RUNS)
}
fn selected_attributes(
    c: &PH3ImageCurveBlockConfig,
    columns: &[(&str, u64)],
) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
    let runs = c.requests.iter().flat_map(|r| r.runs()).collect::<Vec<_>>();
    let list = |values: Vec<u64>| -> Result<SemanticValue, VNextError> {
        Ok(SemanticValue::Integers(
            values
                .into_iter()
                .map(|v| {
                    i64::try_from(v).map_err(|_| invalid("selected routing exceeds signed extent"))
                })
                .collect::<Result<_, _>>()?,
        ))
    };
    let mut a = BTreeMap::from([
        (
            attr("run_lengths")?,
            list(runs.iter().map(|r| r.length()).collect())?,
        ),
        (
            attr("parameter_rows")?,
            list(
                runs.iter()
                    .map(|r| 3 * r.time_index() + r.modality() as u64)
                    .collect(),
            )?,
        ),
    ]);
    for &(name, column) in columns {
        a.insert(attr(name)?, SemanticValue::Unsigned(column));
    }
    Ok(a)
}
fn id(s: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(s)
}
fn attr(s: &str) -> Result<AttributeId, VNextError> {
    AttributeId::new(s)
}
fn node(
    name: &str,
    operation: &str,
    inputs: Vec<ProgramValueId>,
    output: &str,
    attributes: BTreeMap<AttributeId, SemanticValue>,
) -> Result<ProgramNode, VNextError> {
    Ok(ProgramNode {
        id: NodeId::new(format!("node.p_h3_image.curve_block.{name}"))?,
        operation_id: OperationId::new(operation)?,
        required_version: ContractVersion::new(1, 0),
        work: ProgramNodeWorkSpec::Fixed,
        inputs,
        outputs: vec![id(output)?],
        attributes,
    })
}
fn gather_attributes(
    column: u64,
    width: u64,
    runs: &[(u64, u64, u64)],
) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
    let integers =
        |field: &str, values: Vec<u64>| -> Result<(AttributeId, SemanticValue), VNextError> {
            Ok((
                attr(field)?,
                SemanticValue::Integers(
                    values
                        .into_iter()
                        .map(|v| i64::try_from(v).map_err(|_| invalid("row map exceeds i64")))
                        .collect::<Result<_, _>>()?,
                ),
            ))
        };
    Ok(BTreeMap::from([
        (attr("batch")?, SemanticValue::Unsigned(1)),
        (attr("column_start")?, SemanticValue::Unsigned(column)),
        (attr("column_count")?, SemanticValue::Unsigned(width)),
        integers("run_lengths", runs.iter().map(|r| r.0).collect())?,
        integers("source_starts", runs.iter().map(|r| r.1).collect())?,
        integers("source_batch_strides", vec![0; runs.len()])?,
        integers("source_row_strides", runs.iter().map(|r| r.2).collect())?,
    ]))
}
fn projection(
    c: &PH3ImageCurveBlockConfig,
    name: &str,
    input: &str,
    output: &str,
    k: u64,
    n: u64,
    fused_swiglu: bool,
) -> Result<ProgramNode, VNextError> {
    let mut inputs = vec![id(input)?];
    let storage = c
        .projection_storage
        .module(name)
        .map_err(|e| invalid(e.to_string()))?;
    let suffixes: &[&str] = match storage {
        PH3ImageProjectionStorage::CodebookW4A8 => &[
            "weight",
            "weight_s_rel",
            "weight_s_channel",
            "weight_codebook",
        ],
        PH3ImageProjectionStorage::Int8W8A8 => &["weight", "weight_scale"],
    };
    for suffix in suffixes {
        inputs.push(weights::value(&format!("{name}.{suffix}"))?);
    }
    node(
        &format!("{name}.projection"),
        match (storage, fused_swiglu) {
            (PH3ImageProjectionStorage::CodebookW4A8, false) => {
                COMFY_CONVROT_W4A8_BF16_OPERATION_ID
            }
            (PH3ImageProjectionStorage::CodebookW4A8, true) => {
                COMFY_CONVROT_SWIGLU_F32_W4A8_BF16_OPERATION_ID
            }
            (PH3ImageProjectionStorage::Int8W8A8, false) => COMFY_CONVROT_W8A8_BF16_OPERATION_ID,
            (PH3ImageProjectionStorage::Int8W8A8, true) => {
                COMFY_CONVROT_SWIGLU_F32_W8A8_BF16_OPERATION_ID
            }
        },
        inputs,
        output,
        BTreeMap::from([
            (attr("in_features")?, SemanticValue::Unsigned(k)),
            (attr("out_features")?, SemanticValue::Unsigned(n)),
        ]),
    )
}
pub(super) fn boundaries(
    program: &ModelProgram,
) -> Result<BTreeMap<ProgramValueId, ElementType>, VNextError> {
    let mut result = BTreeMap::new();
    for output in program
        .blocks()
        .iter()
        .flat_map(|block| &block.nodes)
        .flat_map(|node| &node.outputs)
    {
        result.insert(
            output.clone(),
            if output.as_str() == TABLE_F32 {
                ElementType::F32
            } else {
                ElementType::Bf16
            },
        );
    }
    Ok(result)
}
pub(super) fn compile_options(
    c: &PH3ImageCurveBlockConfig,
) -> Result<ProgramPlanCompileOptions, VNextError> {
    let mut options = ProgramPlanCompileOptions::new(c.input_tensor_specs()?)?;
    let mut insert = |name: &str, rows: u64, width: u64, dtype: ElementType| {
        options.insert_tensor_spec(id(name)?, config::tensor(vec![rows, width], dtype))
    };
    let h = c.hidden;
    let inner = c.attention_inner_dim();
    let rows = c.rows()?;
    insert(TABLE_F32, c.basis_rows, 18 * h, ElementType::F32)?;
    insert(TABLE_BF16, c.basis_rows, 18 * h, ElementType::Bf16)?;
    insert(TABLE_ROWS, 3 * c.basis_rows, 6 * h, ElementType::Bf16)?;
    let runs = c.requests.iter().flat_map(|r| r.runs()).collect::<Vec<_>>();
    let chunks = runs.chunks(ROW_GATHER_BF16_MAX_RUNS).collect::<Vec<_>>();
    for output in PARAMETERS.into_iter().filter(|_| !compact_parameters(c)) {
        let mut combined_rows = 0u64;
        for (index, chunk) in chunks.iter().enumerate() {
            let chunk_rows = chunk.iter().try_fold(0u64, |sum, run| {
                sum.checked_add(run.length())
                    .ok_or_else(|| invalid("routing chunk rows overflow"))
            })?;
            let name = if chunks.len() == 1 {
                output.to_owned()
            } else {
                format!("{output}.chunk{index}")
            };
            insert(&name, chunk_rows, h, ElementType::Bf16)?;
            combined_rows = combined_rows
                .checked_add(chunk_rows)
                .ok_or_else(|| invalid("combined routing rows overflow"))?;
            if index > 0 {
                let name = if index + 1 == chunks.len() {
                    output.to_owned()
                } else {
                    format!("{output}.combined{index}")
                };
                insert(&name, combined_rows, h, ElementType::Bf16)?;
            }
        }
    }
    for name in [
        NORM1,
        MOD1,
        ATTENTION_OUT,
        RESIDUAL1,
        NORM2,
        MOD2,
        MLP,
        PH3ImageCurveBlockFamily::OUTPUT_ID,
    ] {
        insert(name, rows, h, ElementType::Bf16)?;
    }
    if !compact_parameters(c) {
        for name in [SCALED1, SCALED2] {
            insert(name, rows, h, ElementType::Bf16)?;
        }
    }
    insert(QKV, rows, 3 * inner, ElementType::Bf16)?;
    for name in [Q, K, V, QR, KR, ATTENTION] {
        insert(name, rows, inner, ElementType::Bf16)?;
    }
    insert(GATE_UP, rows, 2 * c.ffn, ElementType::Bf16)?;
    Ok(options)
}
pub(super) fn build(
    family: &PH3ImageCurveBlockFamily,
    c: &PH3ImageCurveBlockConfig,
) -> Result<ModelProgram, VNextError> {
    type F = PH3ImageCurveBlockFamily;
    let h = c.hidden;
    let inner = c.attention_inner_dim();
    let rows = c.rows()?;
    let mut nodes = Vec::new();
    nodes.push(node(
        "adaln_projection",
        AFFINE_LINEAR_F32_OPERATION_ID,
        vec![
            id(F::BASIS_ID)?,
            weights::value("adaln_proj.linear.weight")?,
            weights::value("adaln_proj.linear.bias")?,
        ],
        TABLE_F32,
        BTreeMap::from([
            (attr("in_features")?, SemanticValue::Unsigned(c.basis)),
            (attr("out_features")?, SemanticValue::Unsigned(18 * h)),
        ]),
    )?);
    nodes.push(node(
        "adaln_cast",
        CAST_F32_TO_BF16_OPERATION_ID,
        vec![id(TABLE_F32)?],
        TABLE_BF16,
        BTreeMap::new(),
    )?);
    nodes.push(node(
        "adaln_reshape",
        RESHAPE_COPY_BF16_OPERATION_ID,
        vec![id(TABLE_BF16)?],
        TABLE_ROWS,
        BTreeMap::from([
            (attr("rows")?, SemanticValue::Unsigned(3 * c.basis_rows)),
            (attr("columns")?, SemanticValue::Unsigned(6 * h)),
        ]),
    )?);
    let runs = c
        .requests
        .iter()
        .flat_map(|request| request.runs())
        .map(|run| {
            (
                run.length(),
                3 * run.time_index() + run.modality() as u64,
                0,
            )
        })
        .collect::<Vec<_>>();
    for (parameter, &output) in PARAMETERS
        .iter()
        .enumerate()
        .filter(|_| !compact_parameters(c))
    {
        let chunks = runs.chunks(ROW_GATHER_BF16_MAX_RUNS).collect::<Vec<_>>();
        let mut previous: Option<String> = None;
        for (chunk, runs) in chunks.iter().enumerate() {
            let current = if chunks.len() == 1 {
                output.to_owned()
            } else {
                format!("{output}.chunk{chunk}")
            };
            nodes.push(node(
                &format!("parameter{parameter}.chunk{chunk}"),
                ROW_GATHER_BF16_OPERATION_ID,
                vec![id(TABLE_ROWS)?],
                &current,
                gather_attributes(parameter as u64 * h, h, runs)?,
            )?);
            previous = Some(if let Some(old) = previous {
                let combined = if chunk + 1 == chunks.len() {
                    output.to_owned()
                } else {
                    format!("{output}.combined{chunk}")
                };
                nodes.push(node(
                    &format!("parameter{parameter}.concat{chunk}"),
                    BATCHED_ROW_CONCAT_BF16_OPERATION_ID,
                    vec![id(&old)?, id(&current)?],
                    &combined,
                    BTreeMap::from([(attr("batch")?, SemanticValue::Unsigned(1))]),
                )?);
                combined
            } else {
                current
            });
        }
    }
    let broadcast = BTreeMap::from([(
        attr("broadcast")?,
        SemanticValue::Text("elementwise".into()),
    )]);
    let norm = BTreeMap::from([(attr("epsilon")?, SemanticValue::Rational(c.norm_eps))]);
    nodes.push(node(
        "norm1",
        ROW_RMS_NORM_SINGLE_ROUND_BF16_OPERATION_ID,
        vec![id(F::INPUT_ID)?, weights::value("norm1.weight")?],
        NORM1,
        norm.clone(),
    )?);
    if compact_parameters(c) {
        nodes.push(node(
            "modulation1",
            ROW_SELECTED_MODULATION_BF16_OPERATION_ID,
            vec![id(NORM1)?, id(TABLE_ROWS)?],
            MOD1,
            selected_attributes(c, &[("shift_column", 0), ("scale_column", h)])?,
        )?);
    } else {
        nodes.push(node(
            "scale1",
            MODULATION_SCALE_BF16_OPERATION_ID,
            vec![id(NORM1)?, id(PARAMETERS[1])?],
            SCALED1,
            broadcast.clone(),
        )?);
        nodes.push(node(
            "shift1",
            PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID,
            vec![id(SCALED1)?, id(PARAMETERS[0])?],
            MOD1,
            BTreeMap::new(),
        )?);
    }
    nodes.push(projection(
        c,
        "attn.qkv_proj",
        MOD1,
        QKV,
        h,
        3 * inner,
        false,
    )?);
    for (index, output) in [Q, K, V].into_iter().enumerate() {
        nodes.push(node(
            &format!("qkv_slice{index}"),
            ROW_GATHER_BF16_OPERATION_ID,
            vec![id(QKV)?],
            output,
            gather_attributes(index as u64 * inner, inner, &[(rows, 0, 1)])?,
        )?);
    }
    let rotary = BTreeMap::from([
        (attr("head_dim")?, SemanticValue::Unsigned(c.head_dim)),
        (attr("rotary_dim")?, SemanticValue::Unsigned(c.rotary)),
        (attr("epsilon")?, SemanticValue::Rational(c.qk_eps)),
    ]);
    for (name, input, gamma, output) in [
        ("q_rotary", Q, "attn.q_norm.weight", QR),
        ("k_rotary", K, "attn.k_norm.weight", KR),
    ] {
        nodes.push(node(
            name,
            FUSED_HEAD_RMS_PARTIAL_ROPE_BF16_OPERATION_ID,
            vec![
                id(input)?,
                weights::value(gamma)?,
                id(F::COSINE_ID)?,
                id(F::SINE_ID)?,
            ],
            output,
            rotary.clone(),
        )?);
    }
    nodes.push(node(
        "attention",
        SEGMENTED_ATTENTION_BF16_OPERATION_ID,
        vec![id(QR)?, id(KR)?, id(V)?],
        ATTENTION,
        BTreeMap::from([
            (attr("head_dim")?, SemanticValue::Unsigned(c.head_dim)),
            (attr("scale")?, SemanticValue::Rational(c.attention_scale)),
            (
                attr("segment_lengths")?,
                SemanticValue::Integers(
                    c.requests
                        .iter()
                        .map(|request| {
                            i64::try_from(request.rows()?)
                                .map_err(|_| invalid("attention segment exceeds i64"))
                        })
                        .collect::<Result<_, _>>()?,
                ),
            ),
        ]),
    )?);
    nodes.push(projection(
        c,
        "attn.out_proj",
        ATTENTION,
        ATTENTION_OUT,
        inner,
        h,
        false,
    )?);
    if compact_parameters(c) {
        nodes.push(node(
            "residual1",
            ROW_SELECTED_ADDCMUL_BF16_OPERATION_ID,
            vec![id(F::INPUT_ID)?, id(TABLE_ROWS)?, id(ATTENTION_OUT)?],
            RESIDUAL1,
            selected_attributes(c, &[("gate_column", 2 * h)])?,
        )?);
    } else {
        nodes.push(node(
            "residual1",
            ADDCMUL_BF16_OPERATION_ID,
            vec![id(F::INPUT_ID)?, id(PARAMETERS[2])?, id(ATTENTION_OUT)?],
            RESIDUAL1,
            broadcast.clone(),
        )?);
    }
    nodes.push(node(
        "norm2",
        ROW_RMS_NORM_SINGLE_ROUND_BF16_OPERATION_ID,
        vec![id(RESIDUAL1)?, weights::value("norm2.weight")?],
        NORM2,
        norm,
    )?);
    if compact_parameters(c) {
        nodes.push(node(
            "modulation2",
            ROW_SELECTED_MODULATION_BF16_OPERATION_ID,
            vec![id(NORM2)?, id(TABLE_ROWS)?],
            MOD2,
            selected_attributes(c, &[("shift_column", 3 * h), ("scale_column", 4 * h)])?,
        )?);
    } else {
        nodes.push(node(
            "scale2",
            MODULATION_SCALE_BF16_OPERATION_ID,
            vec![id(NORM2)?, id(PARAMETERS[4])?],
            SCALED2,
            broadcast.clone(),
        )?);
        nodes.push(node(
            "shift2",
            PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID,
            vec![id(SCALED2)?, id(PARAMETERS[3])?],
            MOD2,
            BTreeMap::new(),
        )?);
    }
    nodes.push(projection(
        c,
        "mlp.fc1",
        MOD2,
        GATE_UP,
        h,
        2 * c.ffn,
        false,
    )?);
    nodes.push(projection(c, "mlp.fc2", GATE_UP, MLP, c.ffn, h, true)?);
    if compact_parameters(c) {
        nodes.push(node(
            "residual2",
            ROW_SELECTED_ADDCMUL_BF16_OPERATION_ID,
            vec![id(RESIDUAL1)?, id(TABLE_ROWS)?, id(MLP)?],
            F::OUTPUT_ID,
            selected_attributes(c, &[("gate_column", 5 * h)])?,
        )?);
    } else {
        nodes.push(node(
            "residual2",
            ADDCMUL_BF16_OPERATION_ID,
            vec![id(RESIDUAL1)?, id(PARAMETERS[5])?, id(MLP)?],
            F::OUTPUT_ID,
            broadcast,
        )?);
    }
    if nodes.len() > family.limits.max_plan_nodes {
        return Err(invalid("block program exceeds declared node capacity"));
    }
    ModelProgram::new(
        family.id.clone(),
        c.input_tensor_specs()?.keys().cloned().collect(),
        vec![ProgramBlock {
            id: "block.p_h3_image.curve".into(),
            nodes,
        }],
        vec![],
        weights::references(c)?,
        vec![id(F::OUTPUT_ID)?],
    )?
    .with_input_tensor_specs(c.input_tensor_specs()?)
}
