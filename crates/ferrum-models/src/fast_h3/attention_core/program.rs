use super::*;
use crate::tensor_program_fragment::{StoredCompositionIdentity, StoredCompositionLimits};
type F = FastH3AttentionCoreFamily;
fn tensor(dimensions: Vec<u64>, element_type: ElementType) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn bf16(d: Vec<u64>) -> ProgramTensorSpec {
    tensor(d, ElementType::Bf16)
}
fn value(label: &str) -> Result<ProgramValueId, VNextError> {
    id(&format!("value.fast_h3.attention_core.{label}"))
}
fn scoped(name: &str, instance: &str) -> Result<ProgramValueId, VNextError> {
    fragment::scoped_value(&id(name)?, Some(instance))
}
fn one(
    label: &str,
    operation: &str,
    inputs: Vec<(ProgramValueId, ProgramTensorSpec)>,
    output: ProgramValueId,
    shape: Vec<u64>,
    attributes: BTreeMap<AttributeId, SemanticValue>,
    arithmetic: bool,
) -> Result<Fragment, VNextError> {
    let mut specs = inputs.iter().cloned().collect::<BTreeMap<_, _>>();
    specs.insert(output.clone(), bf16(shape));
    let op: OperationId = id(operation)?;
    Ok(Fragment {
        schema: None,
        weights: vec![],
        specs,
        boundaries: BTreeMap::from([(output.clone(), ElementType::Bf16)]),
        operations: vec![NumericalOperationContract {
            operation_id: op.clone(),
            version: ContractVersion::new(1, 0),
            multiplication_type: arithmetic.then_some(ElementType::F32),
            accumulation_type: arithmetic.then_some(ElementType::F32),
            staged_arithmetic: None,
            composite_arithmetic: None,
        }],
        nodes: vec![ProgramNode {
            id: id(&format!("node.fast_h3.attention_core.{label}"))?,
            operation_id: op,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs: inputs.into_iter().map(|(id, _)| id).collect(),
            outputs: vec![output],
            attributes,
        }],
    })
}
fn bridge(
    c: &FastH3AttentionCoreConfig,
    label: &str,
    input: ProgramValueId,
    output: ProgramValueId,
    rows: u64,
    kind: LeadingBatchBf16Kind,
) -> Result<Fragment, VNextError> {
    let b = c.batches();
    let width = c.columns()?;
    let batched = vec![b, rows, width];
    let flat = vec![
        b.checked_mul(rows)
            .ok_or_else(|| invalid("batch rows overflow"))?,
        width,
    ];
    let (x, y) = if kind == LeadingBatchBf16Kind::Merge {
        (batched, flat)
    } else {
        (flat, batched)
    };
    LeadingBatchBf16Geometry::new(
        kind,
        &ResolvedTensorSpec::new(
            x.clone(),
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?,
        &ResolvedTensorSpec::new(
            y.clone(),
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?,
    )?;
    one(
        label,
        kind.view_operation_id(),
        vec![(input, bf16(x))],
        output,
        y,
        BTreeMap::new(),
        false,
    )
}
pub(super) fn compose(
    c: &FastH3AttentionCoreConfig,
    l: FastH3AttentionCoreLimits,
) -> Result<StoredComposition, VNextError> {
    c.validate(l)?;
    let mut parts = vec![];
    parts.push(Fragment::family(
        &FastH3PooledAttentionFamily::new(l.pooled)?,
        &c.pooled,
        FastH3PooledAttentionFamily::PROFILE_ID,
        c.pooled.compile_options(l.pooled)?.tensor_specs(),
        fragment::mapping(&[
            (FastH3PooledAttentionFamily::QUERY_ID, F::POOLED_QUERY_ID),
            (FastH3PooledAttentionFamily::KEY_ID, F::POOLED_KEY_ID),
            (FastH3PooledAttentionFamily::VALUE_ID, F::POOLED_VALUE_ID),
        ])?,
    )?);
    parts.push(Fragment::family(
        &FastH3PrefixAttentionFamily::new(l.prefix)?,
        &c.prefix,
        c.prefix.profile_id(),
        c.prefix.compile_options(l.prefix)?.tensor_specs(),
        fragment::mapping(&[
            (FastH3PrefixAttentionFamily::QUERY_ID, F::QUERY_ID),
            (FastH3PrefixAttentionFamily::KEY_ID, F::KEY_ID),
            (FastH3PrefixAttentionFamily::VALUE_ID, F::VALUE_ID),
        ])?,
    )?);
    parts.push(Fragment::family_instance(
        &FastH3VsaRowTransportFamily::new(l.transport)?,
        &c.video_query,
        FastH3VsaRowTransportFamily::PROFILE_ID,
        c.video_query.compile_options(l.transport)?.tensor_specs(),
        fragment::mapping(&[(FastH3VsaRowTransportFamily::INPUT_ID, F::QUERY_ID)])?,
        Some("video_query"),
    )?);
    parts.push(Fragment::family(
        &FastH3VsaQkvQuantizeFamily::new(l.quantize)?,
        &c.quantize,
        FastH3VsaQkvQuantizeFamily::PROFILE_ID,
        c.quantize.compile_options(l.quantize)?.tensor_specs(),
        BTreeMap::from([
            (
                id(FastH3VsaQkvQuantizeFamily::QUERY_ID)?,
                scoped(FastH3VsaRowTransportFamily::OUTPUT_ID, "video_query")?,
            ),
            (
                id(FastH3VsaQkvQuantizeFamily::KEY_ID)?,
                id(FastH3PrefixAttentionFamily::CENTERED_KEY_ID)?,
            ),
            (id(FastH3VsaQkvQuantizeFamily::VALUE_ID)?, id(F::VALUE_ID)?),
        ]),
    )?);
    let names = [
        FastH3VsaQkvQuantizeFamily::PACKED_QUERY_ID,
        FastH3VsaQkvQuantizeFamily::PACKED_KEY_ID,
        FastH3VsaQkvQuantizeFamily::PACKED_VALUE_ID,
        FastH3VsaQkvQuantizeFamily::QUERY_SCALES_ID,
        FastH3VsaQkvQuantizeFamily::KEY_SCALES_ID,
        FastH3VsaQkvQuantizeFamily::VALUE_SCALES_ID,
    ];
    let mut inputs = names
        .into_iter()
        .zip(c.quantize.output_shapes())
        .map(|(n, d)| Ok((id(n)?, tensor(d, ElementType::U8))))
        .collect::<Result<Vec<_>, VNextError>>()?;
    inputs.push((
        id(FastH3PooledAttentionFamily::SCORES_ID)?,
        tensor(c.pooled.score_shape(), ElementType::F32),
    ));
    let resolved = inputs
        .iter()
        .map(|(_, t)| {
            ResolvedTensorSpec::new(t.dimensions.clone(), t.element_type, t.layout.clone())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let output = ResolvedTensorSpec::new(
        c.fine_pack.input_shape(),
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )?;
    SparseFp4AttentionGeometry::with_region_scores(
        [
            &resolved[0],
            &resolved[1],
            &resolved[2],
            &resolved[3],
            &resolved[4],
            &resolved[5],
            &resolved[6],
        ],
        [&output],
        &c.attention_attributes()?,
    )?;
    parts.push(one(
        "dynamic_region_attention",
        SPARSE_FP4_ATTENTION_REGION_TOPK_OPERATION_ID,
        inputs,
        value("video_heads")?,
        c.fine_pack.input_shape(),
        c.attention_attributes()?,
        true,
    )?);
    parts.push(Fragment::family_instance(
        &FastH3HeadAxisPackFamily::new(l.head_pack)?,
        &c.fine_pack,
        c.fine_pack.profile_id(),
        c.fine_pack.compile_options(l.head_pack)?.tensor_specs(),
        BTreeMap::from([(
            id(FastH3HeadAxisPackFamily::INPUT_ID)?,
            value("video_heads")?,
        )]),
        Some("video"),
    )?);
    for (label, input, rows) in [
        (
            "prefix",
            id(FastH3PrefixAttentionFamily::OUTPUT_ID)?,
            c.prefix.prefix_rows(),
        ),
        (
            "video",
            scoped(FastH3HeadAxisPackFamily::OUTPUT_ID, "video")?,
            c.video_rows(),
        ),
    ] {
        parts.push(bridge(
            c,
            &format!("{label}_flat"),
            input,
            value(&format!("{label}_flat"))?,
            rows,
            LeadingBatchBf16Kind::Merge,
        )?);
    }
    parts.push(one(
        "prefix_video_concat",
        BATCHED_ROW_CONCAT_BF16_OPERATION_ID,
        vec![
            (
                value("prefix_flat")?,
                bf16(vec![c.batches() * c.prefix.prefix_rows(), c.columns()?]),
            ),
            (
                value("video_flat")?,
                bf16(vec![c.batches() * c.video_rows(), c.columns()?]),
            ),
        ],
        value("fine_flat")?,
        vec![c.batches() * c.padded_rows(), c.columns()?],
        BTreeMap::from([(id("batch")?, SemanticValue::Unsigned(c.batches()))]),
        false,
    )?);
    parts.push(bridge(
        c,
        "fine_split",
        value("fine_flat")?,
        value("fine_tiled")?,
        c.padded_rows(),
        LeadingBatchBf16Kind::Split,
    )?);
    parts.push(Fragment::family_instance(
        &FastH3VsaRowTransportFamily::new(l.transport)?,
        &c.untile,
        FastH3VsaRowTransportFamily::PROFILE_ID,
        c.untile.compile_options(l.transport)?.tensor_specs(),
        BTreeMap::from([(
            id(FastH3VsaRowTransportFamily::INPUT_ID)?,
            value("fine_tiled")?,
        )]),
        Some("fine"),
    )?);
    parts.push(Fragment::family_instance(
        &FastH3HeadAxisPackFamily::new(l.head_pack)?,
        &c.pooled_pack,
        c.pooled_pack.profile_id(),
        c.pooled_pack.compile_options(l.head_pack)?.tensor_specs(),
        fragment::mapping(&[(
            FastH3HeadAxisPackFamily::INPUT_ID,
            FastH3PooledAttentionFamily::OUTPUT_ID,
        )])?,
        Some("pooled"),
    )?);
    parts.push(Fragment::family_instance(
        &FastH3VsaRowTransportFamily::new(l.transport)?,
        &c.expand,
        FastH3VsaRowTransportFamily::PROFILE_ID,
        c.expand.compile_options(l.transport)?.tensor_specs(),
        BTreeMap::from([(
            id(FastH3VsaRowTransportFamily::INPUT_ID)?,
            scoped(FastH3HeadAxisPackFamily::OUTPUT_ID, "pooled")?,
        )]),
        Some("pooled"),
    )?);
    for (label, input) in [
        (
            "fine",
            scoped(FastH3VsaRowTransportFamily::OUTPUT_ID, "fine")?,
        ),
        (
            "pooled",
            scoped(FastH3VsaRowTransportFamily::OUTPUT_ID, "pooled")?,
        ),
        ("gate", id(F::GATE_ID)?),
    ] {
        parts.push(bridge(
            c,
            &format!("{label}_packed_flat"),
            input,
            value(&format!("{label}_packed_flat"))?,
            c.packed_rows(),
            LeadingBatchBf16Kind::Merge,
        )?);
    }
    parts.push(Fragment::family(
        &FastH3AttentionOutputFamily::new(l.projection)?,
        &c.projection,
        FastH3AttentionOutputFamily::PROFILE_ID,
        c.projection.compile_options()?.tensor_specs(),
        BTreeMap::from([
            (
                id(FastH3AttentionOutputFamily::FINE_ID)?,
                value("fine_packed_flat")?,
            ),
            (
                id(FastH3AttentionOutputFamily::POOLED_ID)?,
                value("pooled_packed_flat")?,
            ),
            (
                id(FastH3AttentionOutputFamily::GATE_ID)?,
                value("gate_packed_flat")?,
            ),
        ]),
    )?);
    StoredComposition::join(
        StoredCompositionIdentity {
            family: id(F::FAMILY_ID)?,
            profile: NumericalProfileId::new(c.profile_id()).map_err(invalid)?,
            format: id("weight-format.safetensors.dense")?,
            layout: id("weight-layout.fast_h3.attention_core.v1")?,
            block: "block.fast_h3.attention_core".into(),
        },
        parts,
        [
            F::QUERY_ID,
            F::KEY_ID,
            F::VALUE_ID,
            F::GATE_ID,
            F::POOLED_QUERY_ID,
            F::POOLED_KEY_ID,
            F::POOLED_VALUE_ID,
        ]
        .into_iter()
        .map(id)
        .collect::<Result<_, _>>()?,
        vec![id(F::OUTPUT_ID)?],
        id(F::OUTPUT_ID)?,
        StoredCompositionLimits {
            maximum_nodes: l.maximum_plan_nodes,
            maximum_stored_bytes: l.maximum_stored_bytes,
            maximum_metadata_bytes: l.maximum_metadata_bytes,
        },
    )
}
