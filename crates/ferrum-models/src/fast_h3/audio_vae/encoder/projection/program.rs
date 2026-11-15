use super::*;
use crate::fast_h3::audio_vae::program::Graph;

struct ProjectionGraph {
    graph: Graph,
    rows: u64,
}
impl ProjectionGraph {
    fn emit(
        &mut self,
        name: &str,
        op: &str,
        inputs: Vec<ProgramValueId>,
        width: u64,
        attrs: BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<ProgramValueId, VNextError> {
        self.graph
            .emit_shape(name, op, inputs, vec![self.rows, width], attrs)
    }
    fn affine(
        &mut self,
        name: &str,
        source: &str,
        x: ProgramValueId,
        inputs: u64,
        outputs: u64,
    ) -> Result<ProgramValueId, VNextError> {
        self.emit(
            name,
            AFFINE_LINEAR_F32_OPERATION_ID,
            vec![
                x,
                weight_value(&format!("{source}.weight"))?,
                weight_value(&format!("{source}.bias"))?,
            ],
            outputs,
            BTreeMap::from([
                (id("in_features")?, SemanticValue::Unsigned(inputs)),
                (id("out_features")?, SemanticValue::Unsigned(outputs)),
            ]),
        )
    }
    fn norm(
        &mut self,
        name: &str,
        x: ProgramValueId,
        width: u64,
    ) -> Result<ProgramValueId, VNextError> {
        self.emit(
            name,
            LAYER_NORM_F32_OPERATION_ID,
            vec![
                x,
                weight_value(&format!("{name}.weight"))?,
                weight_value(&format!("{name}.bias"))?,
            ],
            width,
            BTreeMap::from([(
                id("epsilon")?,
                SemanticValue::Rational(CanonicalRational::new(1, 100_000)?),
            )]),
        )
    }
}
pub(super) fn build(
    c: &ProjectionConfig,
    limits: FastH3AudioComponentLimits,
) -> Result<(ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>), VNextError> {
    let a = &c.architecture;
    let d = a.latent_dim;
    let channels = a.latent_channels;
    let head = d / a.num_attention_heads;
    let rows = c
        .mono_batch
        .checked_mul(c.latent_samples)
        .ok_or_else(|| invalid("audio projection rows overflow"))?;
    let input: ProgramValueId = id(PreparedFastH3AudioEncoderProjection::INPUT_ID)?;
    let inputs = BTreeMap::from([(input.clone(), tensor(vec![rows, d]))]);
    if inputs[&input].byte_len()? > limits.maximum_activation_bytes {
        return Err(invalid("audio projection input exceeds capacity"));
    }
    let mut g = ProjectionGraph {
        graph: Graph::new(c.mono_batch, limits, inputs.clone()),
        rows,
    };
    let direct = g.norm("pre_block.norm3", input.clone(), d)?;
    let direct = g.affine("pre_block.proj", "pre_block.proj", direct, d, channels)?;
    let qkv = g.norm("pre_block.norm1", input, d)?;
    let qkv = g.affine("pre_block.attn.qkv", "pre_block.attn.qkv", qkv, d, d * 3)?;
    let mut parts = Vec::new();
    for (i, name) in ["query", "key", "value"].into_iter().enumerate() {
        parts.push(g.emit(
            &format!("pre_block.attn.{name}"),
            GROUPED_COLUMN_SLICE_F32_OPERATION_ID,
            vec![qkv.clone()],
            d,
            BTreeMap::from([
                (id("group_width")?, SemanticValue::Unsigned(d * 3)),
                (id("column_start")?, SemanticValue::Unsigned(i as u64 * d)),
                (id("column_count")?, SemanticValue::Unsigned(d)),
            ]),
        )?);
    }
    // The supported head range fits this exact dyadic representation of the
    // F32 inverse square-root scale, including non-square head widths.
    let scale = (1.0f64 / (head as f64).sqrt()) as f32;
    let scale = CanonicalRational::new((f64::from(scale) * (1u64 << 30) as f64) as i64, 1 << 30)?;
    let attention = g.emit(
        "pre_block.attn.causal",
        STRUCTURED_GQA_ATTENTION_F32_OPERATION_ID,
        parts,
        d,
        BTreeMap::from([
            (
                id("sequence_length")?,
                SemanticValue::Unsigned(c.latent_samples),
            ),
            (id("head_dim")?, SemanticValue::Unsigned(head)),
            (id("scale")?, SemanticValue::Rational(scale)),
            (
                id("segment_lengths")?,
                SemanticValue::Integers(vec![c.latent_samples as i64]),
            ),
            (
                id("segment_bidirectional")?,
                SemanticValue::Integers(vec![0]),
            ),
            (
                id("key_run_counts")?,
                SemanticValue::Integers(vec![1; c.mono_batch as usize]),
            ),
            (
                id("key_run_lengths")?,
                SemanticValue::Integers(vec![c.latent_samples as i64; c.mono_batch as usize]),
            ),
            (
                id("key_run_valid")?,
                SemanticValue::Integers(vec![1; c.mono_batch as usize]),
            ),
        ]),
    )?;
    let pooled = g.emit(
        "pre_block.attn.head_mean",
        GROUPED_COLUMN_MEAN_F32_OPERATION_ID,
        vec![attention],
        head,
        BTreeMap::from([(id("group_width")?, SemanticValue::Unsigned(head))]),
    )?;
    let pooled = g.emit(
        "pre_block.attn.adaptive_pool",
        ADAPTIVE_AVG_POOL_ROWS_F32_OPERATION_ID,
        vec![pooled],
        channels,
        BTreeMap::new(),
    )?;
    let attention = g.affine(
        "pre_block.attn.proj",
        "pre_block.attn.proj",
        pooled,
        channels,
        channels,
    )?;
    let residual = g.emit(
        "pre_block.attn.residual",
        PLAIN_RESIDUAL_ADD_F32_OPERATION_ID,
        vec![direct, attention],
        channels,
        BTreeMap::new(),
    )?;
    let normalized = g.norm("pre_block.norm2", residual.clone(), channels)?;
    let normalized = g.norm("pre_block.mlp.norm", normalized, channels)?;
    let gate = g.affine(
        "pre_block.mlp.w0",
        "pre_block.mlp.w0",
        normalized.clone(),
        channels,
        channels * 2,
    )?;
    let gate = g.emit(
        "pre_block.mlp.gelu",
        GELU_TANH_F32_OPERATION_ID,
        vec![gate],
        channels * 2,
        BTreeMap::new(),
    )?;
    let up = g.affine(
        "pre_block.mlp.w1",
        "pre_block.mlp.w1",
        normalized,
        channels,
        channels * 2,
    )?;
    let product = g.emit(
        "pre_block.mlp.product",
        MULTIPLY_F32_OPERATION_ID,
        vec![gate, up],
        channels * 2,
        BTreeMap::new(),
    )?;
    let mlp = g.affine(
        "pre_block.mlp.w2",
        "pre_block.mlp.w2",
        product,
        channels * 2,
        channels,
    )?;
    let latent = g.emit(
        "pre_block.mlp.residual",
        PLAIN_RESIDUAL_ADD_F32_OPERATION_ID,
        vec![residual, mlp],
        channels,
        BTreeMap::new(),
    )?;
    let mean = g.affine("encoder.mode", "mean_proj", latent, channels, channels)?;
    g.graph.finish(
        ProjectionConfig::FAMILY_ID,
        "block.fast_h3.audio.encoder.projection",
        inputs,
        &c.weights,
        mean,
    )
}
