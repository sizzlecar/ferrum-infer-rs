use super::{weights, Qwen3VLVisionConfig, Qwen3VLVisionFamily as Family};
use ferrum_runtime::*;
use std::collections::BTreeMap;
pub(super) const AFFINE: &str = "operation.affine_linear.bf16";
pub(super) const PATCH: &str = "operation.patch_projection.bf16";
pub(super) const NORM: &str = "operation.affine_layer_norm.bf16";
pub(super) const GELU: &str = "operation.gelu_erf.bf16";
pub(super) const ROTARY: &str = "operation.half_rotary.bf16";
pub(super) const ATTENTION: &str = "operation.segmented_attention.bf16";
pub(super) const RESHAPE: &str = "operation.reshape_copy.bf16";
type Attributes = BTreeMap<AttributeId, SemanticValue>;
fn attrs(
    values: impl IntoIterator<Item = (&'static str, SemanticValue)>,
) -> Result<Attributes, VNextError> {
    values
        .into_iter()
        .map(|(k, v)| Ok((AttributeId::new(k)?, v)))
        .collect()
}
fn id(name: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.qwen3_vl.vision.{name}"))
}

struct Builder {
    nodes: Vec<ProgramNode>,
    specs: BTreeMap<ProgramValueId, ProgramTensorSpec>,
}
impl Builder {
    fn push(
        &mut self,
        name: &str,
        op: &str,
        inputs: Vec<ProgramValueId>,
        out: ProgramValueId,
        rows: u64,
        cols: u64,
        attributes: Attributes,
    ) -> Result<ProgramValueId, VNextError> {
        ResolvedTensorSpec::new(
            vec![rows, cols],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        self.specs.insert(
            out.clone(),
            ProgramTensorSpec {
                dimensions: vec![rows, cols],
                element_type: ElementType::Bf16,
                layout: ResolvedTensorLayout::Contiguous,
            },
        );
        self.nodes.push(ProgramNode {
            id: NodeId::new(format!("node.qwen3_vl.vision.{name}"))?,
            operation_id: OperationId::new(op)?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs,
            outputs: vec![out.clone()],
            attributes,
        });
        Ok(out)
    }
    fn affine(
        &mut self,
        name: &str,
        input: ProgramValueId,
        weight: &str,
        rows: u64,
        k: u64,
        n: u64,
    ) -> Result<ProgramValueId, VNextError> {
        self.push(
            name,
            AFFINE,
            vec![
                input,
                weights::value_id(&format!("{weight}.weight"))?,
                weights::value_id(&format!("{weight}.bias"))?,
            ],
            id(name)?,
            rows,
            n,
            attrs([
                ("in_features", SemanticValue::Unsigned(k)),
                ("out_features", SemanticValue::Unsigned(n)),
            ])?,
        )
    }
    fn norm(
        &mut self,
        name: &str,
        input: ProgramValueId,
        weight: &str,
        rows: u64,
        h: u64,
        eps: CanonicalRational,
    ) -> Result<ProgramValueId, VNextError> {
        self.push(
            name,
            NORM,
            vec![
                input,
                weights::value_id(&format!("{weight}.weight"))?,
                weights::value_id(&format!("{weight}.bias"))?,
            ],
            id(name)?,
            rows,
            h,
            attrs([
                ("hidden_size", SemanticValue::Unsigned(h)),
                ("epsilon", SemanticValue::Rational(eps)),
            ])?,
        )
    }
    fn reshape(
        &mut self,
        name: &str,
        input: ProgramValueId,
        rows: u64,
        h: u64,
    ) -> Result<ProgramValueId, VNextError> {
        self.push(
            name,
            RESHAPE,
            vec![input],
            id(name)?,
            rows,
            h,
            attrs([
                ("rows", SemanticValue::Unsigned(rows)),
                ("columns", SemanticValue::Unsigned(h)),
            ])?,
        )
    }
    fn merger(
        &mut self,
        c: &Qwen3VLVisionConfig,
        index: Option<usize>,
        input: ProgramValueId,
    ) -> Result<ProgramValueId, VNextError> {
        let a = c.architecture();
        let rows = c.merged_rows()?;
        let h = a.hidden_size();
        let merged = a.merger_width()?;
        let (name, weight, output) = match index {
            None => (
                "merger".to_string(),
                "model.visual.merger".to_string(),
                Family::OUTPUT_ID,
            ),
            Some(i) => (
                format!("deepstack_merger{i}"),
                format!("model.visual.deepstack_merger_list.{i}"),
                Family::DEEPSTACK_IDS[i],
            ),
        };
        let x = if index.is_some() {
            let x = self.reshape(&format!("{name}.shuffle"), input, rows, merged)?;
            self.norm(
                &format!("{name}.norm"),
                x,
                &format!("{weight}.norm"),
                rows,
                merged,
                a.epsilon(),
            )?
        } else {
            let x = self.norm(
                &format!("{name}.norm"),
                input,
                &format!("{weight}.norm"),
                c.rows()?,
                h,
                a.epsilon(),
            )?;
            self.reshape(&format!("{name}.shuffle"), x, rows, merged)?
        };
        let x = self.affine(
            &format!("{name}.fc1"),
            x,
            &format!("{weight}.linear_fc1"),
            rows,
            merged,
            merged,
        )?;
        let x = self.push(
            &format!("{name}.gelu"),
            GELU,
            vec![x],
            id(&format!("{name}.gelu"))?,
            rows,
            merged,
            Attributes::new(),
        )?;
        self.push(
            &format!("{name}.fc2"),
            AFFINE,
            vec![
                x,
                weights::value_id(&format!("{weight}.linear_fc2.weight"))?,
                weights::value_id(&format!("{weight}.linear_fc2.bias"))?,
            ],
            ProgramValueId::new(output)?,
            rows,
            a.output_size(),
            attrs([
                ("in_features", SemanticValue::Unsigned(merged)),
                ("out_features", SemanticValue::Unsigned(a.output_size())),
            ])?,
        )
    }
}
pub(super) fn build(
    c: &Qwen3VLVisionConfig,
    family: &ModelFamilyId,
) -> Result<(ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>), VNextError> {
    c.validate()?;
    let a = c.architecture();
    let rows = c.rows()?;
    let h = a.hidden_size();
    let m = a.intermediate_size();
    let mut b = Builder {
        nodes: Vec::new(),
        specs: c.input_tensor_specs()?,
    };
    let patches = ProgramValueId::new(Family::PATCHES_ID)?;
    let positions = ProgramValueId::new(Family::POSITIONS_ID)?;
    let cos = ProgramValueId::new(Family::COSINE_ID)?;
    let sin = ProgramValueId::new(Family::SINE_ID)?;
    let patch = b.push(
        "patch_projection",
        PATCH,
        vec![
            patches.clone(),
            weights::value_id("model.visual.patch_embed.proj.weight")?,
            weights::value_id("model.visual.patch_embed.proj.bias")?,
        ],
        id("patch_projection")?,
        rows,
        h,
        Attributes::new(),
    )?;
    let mut x = b.push(
        "position_add",
        PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID,
        vec![patch, positions.clone()],
        id("position_add")?,
        rows,
        h,
        Attributes::new(),
    )?;
    let scale = CanonicalRational::from_decimal_str(
        &((1.0 / (a.head_dim() as f64).sqrt()) as f32).to_string(),
    )?;
    let attention = attrs([
        ("head_dim", SemanticValue::Unsigned(a.head_dim())),
        ("scale", SemanticValue::Rational(scale)),
        (
            "segment_lengths",
            SemanticValue::Integers(
                c.grids()
                    .iter()
                    .map(|g| Ok(g.rows()? as i64))
                    .collect::<Result<Vec<_>, VNextError>>()?,
            ),
        ),
    ])?;
    for layer in 0..a.depth() {
        let prefix = format!("layer{layer}");
        let weight = format!("model.visual.blocks.{layer}");
        let n = b.norm(
            &format!("{prefix}.norm1"),
            x.clone(),
            &format!("{weight}.norm1"),
            rows,
            h,
            a.epsilon(),
        )?;
        let qkv = b.affine(
            &format!("{prefix}.qkv"),
            n,
            &format!("{weight}.attn.qkv"),
            rows,
            h,
            h * 3,
        )?;
        let mut parts = Vec::new();
        for (part, name) in ["q", "k", "v"].into_iter().enumerate() {
            let name = format!("{prefix}.{name}");
            parts.push(b.push(
                &name,
                ROW_GATHER_BF16_OPERATION_ID,
                vec![qkv.clone()],
                id(&name)?,
                rows,
                h,
                attrs([
                    ("batch", SemanticValue::Unsigned(1)),
                    ("run_lengths", SemanticValue::Integers(vec![rows as i64])),
                    ("source_starts", SemanticValue::Integers(vec![0])),
                    ("source_batch_strides", SemanticValue::Integers(vec![0])),
                    ("source_row_strides", SemanticValue::Integers(vec![1])),
                    ("column_start", SemanticValue::Unsigned(part as u64 * h)),
                    ("column_count", SemanticValue::Unsigned(h)),
                ])?,
            )?);
        }
        for (part, name) in ["q_rotary", "k_rotary"].into_iter().enumerate() {
            let name = format!("{prefix}.{name}");
            parts[part] = b.push(
                &name,
                ROTARY,
                vec![parts[part].clone(), cos.clone(), sin.clone()],
                id(&name)?,
                rows,
                h,
                attrs([("head_dim", SemanticValue::Unsigned(a.head_dim()))])?,
            )?;
        }
        let n = b.push(
            &format!("{prefix}.attention"),
            ATTENTION,
            parts,
            id(&format!("{prefix}.attention"))?,
            rows,
            h,
            attention.clone(),
        )?;
        let n = b.affine(
            &format!("{prefix}.project"),
            n,
            &format!("{weight}.attn.proj"),
            rows,
            h,
            h,
        )?;
        let residual = b.push(
            &format!("{prefix}.attention_residual"),
            PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID,
            vec![x, n],
            id(&format!("{prefix}.attention_residual"))?,
            rows,
            h,
            Attributes::new(),
        )?;
        let n = b.norm(
            &format!("{prefix}.norm2"),
            residual.clone(),
            &format!("{weight}.norm2"),
            rows,
            h,
            a.epsilon(),
        )?;
        let n = b.affine(
            &format!("{prefix}.fc1"),
            n,
            &format!("{weight}.mlp.linear_fc1"),
            rows,
            h,
            m,
        )?;
        let n = b.push(
            &format!("{prefix}.gelu"),
            GELU_TANH_BF16_OPERATION_ID,
            vec![n],
            id(&format!("{prefix}.gelu"))?,
            rows,
            m,
            Attributes::new(),
        )?;
        let n = b.affine(
            &format!("{prefix}.fc2"),
            n,
            &format!("{weight}.mlp.linear_fc2"),
            rows,
            m,
            h,
        )?;
        x = b.push(
            &format!("{prefix}.output"),
            PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID,
            vec![residual, n],
            id(&format!("{prefix}.output"))?,
            rows,
            h,
            Attributes::new(),
        )?;
        if let Some(index) = a.deepstack_layers().iter().position(|&l| l == layer) {
            b.merger(c, Some(index), x.clone())?;
        }
    }
    b.merger(c, None, x)?;
    let weights = c
        .expected_weights()?
        .into_iter()
        .map(|w| {
            Ok(WeightReference {
                weight_id: w.weight_id()?,
                value_id: w.value_id()?,
                tensor: ProgramTensorSpec {
                    dimensions: w.dimensions().to_vec(),
                    element_type: ElementType::Bf16,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            })
        })
        .collect::<Result<Vec<_>, VNextError>>()?;
    let outputs = std::iter::once(Family::OUTPUT_ID)
        .chain(Family::DEEPSTACK_IDS)
        .map(ProgramValueId::new)
        .collect::<Result<Vec<_>, _>>()?;
    let graph = ModelProgram::new(
        family.clone(),
        vec![patches, positions, cos, sin],
        vec![ProgramBlock {
            id: "block.qwen3_vl.vision".into(),
            nodes: b.nodes,
        }],
        vec![],
        weights,
        outputs,
    )?
    .with_input_tensor_specs(c.input_tensor_specs()?)?;
    Ok((graph, b.specs))
}
