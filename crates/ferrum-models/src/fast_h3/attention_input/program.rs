use super::*;
use crate::tensor_program_fragment::{StoredCompositionIdentity, StoredCompositionLimits};
type F = FastH3AttentionInputFamily;
pub(super) fn compose(
    c: &FastH3AttentionInputConfig,
    l: FastH3AttentionInputLimits,
) -> Result<StoredComposition, VNextError> {
    c.validate(l)?;
    let mut parts = Vec::new();
    for (label, input, cfg) in [
        ("hidden", F::HIDDEN_ID, &c.tile_hidden),
        ("cosine", F::COSINE_ID, &c.tile_rotary),
        ("sine", F::SINE_ID, &c.tile_rotary),
    ] {
        parts.push(Fragment::family_instance(
            &FastH3VsaRowTransportFamily::new(l.transport)?,
            cfg,
            FastH3VsaRowTransportFamily::PROFILE_ID,
            cfg.compile_options(l.transport)?.tensor_specs(),
            fragment::mapping(&[(FastH3VsaRowTransportFamily::INPUT_ID, input)])?,
            Some(label),
        )?);
        parts.push(bridge(
            LeadingBatchBf16Kind::Merge,
            label,
            scoped(FastH3VsaRowTransportFamily::OUTPUT_ID, label)?,
            value(&format!("flat_{label}"))?,
            c.batches(),
            c.tiled_rows(),
            cfg.columns(),
        )?);
    }
    parts.push(bridge(
        LeadingBatchBf16Kind::Merge,
        "packed_hidden",
        id(F::HIDDEN_ID)?,
        value("flat_packed_hidden")?,
        c.batches(),
        c.packed_rows(),
        c.hidden_size(),
    )?);
    for (label, config, input) in [
        ("qkv", &c.qkv, "flat_hidden"),
        ("gate", &c.gate, "flat_packed_hidden"),
    ] {
        parts.push(Fragment::family_instance(
            &FastH3AttentionProjectionFamily::new(l.projection)?,
            config,
            FastH3AttentionProjectionFamily::PROFILE_ID,
            config.compile_options()?.tensor_specs(),
            BTreeMap::from([(
                id(FastH3AttentionProjectionFamily::INPUT_ID)?,
                value(input)?,
            )]),
            Some(label),
        )?);
    }
    parts.push(Fragment::family(
        &FastH3QkNormRopeFamily::new(l.qk)?,
        &c.qk,
        FastH3QkNormRopeFamily::PROFILE_ID,
        c.qk.compile_options()?.tensor_specs(),
        BTreeMap::from([
            (
                id(FastH3QkNormRopeFamily::QUERY_ID)?,
                scoped(FastH3AttentionProjection::Query.output_id(), "qkv")?,
            ),
            (
                id(FastH3QkNormRopeFamily::KEY_ID)?,
                scoped(FastH3AttentionProjection::Key.output_id(), "qkv")?,
            ),
            (
                id(FastH3QkNormRopeFamily::COSINE_ID)?,
                value("flat_cosine")?,
            ),
            (id(FastH3QkNormRopeFamily::SINE_ID)?, value("flat_sine")?),
        ]),
    )?);
    for (label, input, output, rows) in [
        (
            "query",
            id(FastH3QkNormRopeFamily::QUERY_OUTPUT_ID)?,
            F::QUERY_ID,
            c.tiled_rows(),
        ),
        (
            "key",
            id(FastH3QkNormRopeFamily::KEY_OUTPUT_ID)?,
            F::KEY_ID,
            c.tiled_rows(),
        ),
        (
            "value",
            scoped(FastH3AttentionProjection::Value.output_id(), "qkv")?,
            F::VALUE_ID,
            c.tiled_rows(),
        ),
        (
            "gate",
            scoped(
                FastH3AttentionProjection::CompressionGate.output_id(),
                "gate",
            )?,
            F::GATE_ID,
            c.packed_rows(),
        ),
    ] {
        parts.push(bridge(
            LeadingBatchBf16Kind::Split,
            label,
            input,
            id(output)?,
            c.batches(),
            rows,
            c.output_width()?,
        )?);
    }
    for (label, input) in [
        ("query", F::QUERY_ID),
        ("key", F::KEY_ID),
        ("value", F::VALUE_ID),
    ] {
        parts.push(Fragment::family_instance(
            &FastH3VsaPoolFamily::new(l.pooling)?,
            &c.pooling,
            FastH3VsaPoolFamily::PROFILE_ID,
            c.pooling.compile_options(l.pooling)?.tensor_specs(),
            fragment::mapping(&[(FastH3VsaPoolFamily::INPUT_ID, input)])?,
            Some(label),
        )?);
    }
    StoredComposition::join(
        StoredCompositionIdentity {
            family: id(F::FAMILY_ID)?,
            profile: NumericalProfileId::new(F::PROFILE_ID).map_err(invalid)?,
            format: id("weight-format.safetensors.dense")?,
            layout: id("weight-layout.fast_h3.attention_input.v1")?,
            block: "block.fast_h3.attention_input".into(),
        },
        parts,
        [F::HIDDEN_ID, F::COSINE_ID, F::SINE_ID]
            .into_iter()
            .map(id)
            .collect::<Result<_, _>>()?,
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
        id(F::QUERY_ID)?,
        StoredCompositionLimits {
            maximum_nodes: l.maximum_plan_nodes,
            maximum_stored_bytes: l.maximum_stored_bytes,
            maximum_metadata_bytes: l.maximum_metadata_bytes,
        },
    )
}
fn scoped(name: &str, label: &str) -> Result<ProgramValueId, VNextError> {
    fragment::scoped_value(&id(name)?, Some(label))
}
fn value(label: &str) -> Result<ProgramValueId, VNextError> {
    id(&format!("value.fast_h3.attention_input.{label}"))
}
fn tensor(dimensions: Vec<u64>) -> Result<ProgramTensorSpec, VNextError> {
    let t = ProgramTensorSpec {
        dimensions,
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    };
    t.byte_len()?;
    Ok(t)
}
fn bridge(
    kind: LeadingBatchBf16Kind,
    label: &str,
    input: ProgramValueId,
    output: ProgramValueId,
    batches: u64,
    rows: u64,
    columns: u64,
) -> Result<Fragment, VNextError> {
    let batched = tensor(vec![batches, rows, columns])?;
    let flat = tensor(vec![
        batches
            .checked_mul(rows)
            .ok_or_else(|| invalid("flat rows overflow"))?,
        columns,
    ])?;
    let (x, y) = match kind {
        LeadingBatchBf16Kind::Merge => (batched, flat),
        LeadingBatchBf16Kind::Split => (flat, batched),
    };
    LeadingBatchBf16Geometry::new(
        kind,
        &ResolvedTensorSpec::new(x.dimensions.clone(), x.element_type, x.layout.clone())?,
        &ResolvedTensorSpec::new(y.dimensions.clone(), y.element_type, y.layout.clone())?,
    )?;
    let operation_id: OperationId = id(kind.view_operation_id())?;
    Ok(Fragment {
        schema: None,
        weights: vec![],
        specs: BTreeMap::from([(input.clone(), x), (output.clone(), y)]),
        boundaries: BTreeMap::from([(output.clone(), ElementType::Bf16)]),
        operations: vec![NumericalOperationContract {
            operation_id: operation_id.clone(),
            version: ContractVersion::new(1, 0),
            multiplication_type: None,
            accumulation_type: None,
            staged_arithmetic: None,
            composite_arithmetic: None,
        }],
        nodes: vec![ProgramNode {
            id: id(&format!("node.fast_h3.attention_input.{kind:?}.{label}"))?,
            operation_id,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs: vec![input],
            outputs: vec![output],
            attributes: BTreeMap::new(),
        }],
    })
}
