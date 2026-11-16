use super::*;
use crate::tensor_program_fragment::{StoredCompositionIdentity, StoredCompositionLimits};
type F = FastH3PrefixAttentionFamily;
fn value(s: &str) -> Result<ProgramValueId, VNextError> {
    id(&format!("value.fast_h3.vsa.prefix_attention.{s}"))
}
fn one(
    label: &str,
    operation: &str,
    inputs: Vec<(ProgramValueId, ProgramTensorSpec)>,
    output: ProgramValueId,
    shape: Vec<u64>,
    attrs: BTreeMap<AttributeId, SemanticValue>,
    arithmetic: bool,
) -> Result<Fragment, VNextError> {
    let mut specs = inputs.iter().cloned().collect::<BTreeMap<_, _>>();
    specs.insert(output.clone(), tensor(shape));
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
            id: id(&format!("node.fast_h3.vsa.prefix_attention.{label}"))?,
            operation_id: op,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs: inputs.into_iter().map(|(id, _)| id).collect(),
            outputs: vec![output],
            attributes: attrs,
        }],
    })
}
fn bridge(
    c: &FastH3PrefixAttentionConfig,
    label: &str,
    input: ProgramValueId,
    output: ProgramValueId,
    rows: u64,
    kind: LeadingBatchBf16Kind,
) -> Result<Fragment, VNextError> {
    let rank3 = vec![c.batches(), rows, c.columns()];
    let rank2 = vec![c.batches() * rows, c.columns()];
    let (x, y) = if kind == LeadingBatchBf16Kind::Merge {
        (rank3, rank2)
    } else {
        (rank2, rank3)
    };
    LeadingBatchBf16Geometry::new(kind, &resolved(x.clone())?, &resolved(y.clone())?)?;
    one(
        label,
        kind.view_operation_id(),
        vec![(input, tensor(x))],
        output,
        y,
        BTreeMap::new(),
        false,
    )
}
pub(super) fn compose(
    c: &FastH3PrefixAttentionConfig,
    l: FastH3PrefixAttentionLimits,
) -> Result<StoredComposition, VNextError> {
    c.validate(l)?;
    let mut parts = vec![];
    parts.push(Fragment::family(
        &FastH3KeyCenterFamily::new(l.center)?,
        &c.center,
        FastH3KeyCenterFamily::PROFILE_ID,
        c.center.compile_options(l.center)?.tensor_specs(),
        fragment::mapping(&[(FastH3KeyCenterFamily::INPUT_ID, F::KEY_ID)])?,
    )?);
    parts.push(Fragment::family_instance(
        &FastH3VsaRowTransportFamily::new(l.transport)?,
        &c.query,
        FastH3VsaRowTransportFamily::PROFILE_ID,
        c.query.compile_options(l.transport)?.tensor_specs(),
        BTreeMap::from([(id(FastH3VsaRowTransportFamily::INPUT_ID)?, id(F::QUERY_ID)?)]),
        Some("query"),
    )?);
    let prefix_query =
        fragment::scoped_value(&id(FastH3VsaRowTransportFamily::OUTPUT_ID)?, Some("query"))?;
    let qflat = vec![c.batches() * c.prefix_rows, c.columns()];
    let prefix_shape = vec![c.batches(), c.prefix_rows, c.columns()];
    parts.push(one(
        "query_fp8",
        HEAD_FP8_ROUNDTRIP_BF16_OPERATION_ID,
        vec![(prefix_query, tensor(prefix_shape.clone()))],
        value("query_fp8")?,
        prefix_shape,
        HeadFp8RoundtripBf16Geometry::attributes(c.head_dim)?,
        true,
    )?);
    parts.push(bridge(
        c,
        "query_quantized_flat",
        value("query_fp8")?,
        value("query_quantized_flat")?,
        c.prefix_rows,
        LeadingBatchBf16Kind::Merge,
    )?);
    for (label, input) in [
        ("key", id(FastH3KeyCenterFamily::OUTPUT_ID)?),
        ("value", id(F::VALUE_ID)?),
    ] {
        let mut transport = Fragment::family_instance(
            &FastH3VsaRowTransportFamily::new(l.transport)?,
            &c.valid,
            FastH3VsaRowTransportFamily::PROFILE_ID,
            c.valid.compile_options(l.transport)?.tensor_specs(),
            BTreeMap::from([(id(FastH3VsaRowTransportFamily::INPUT_ID)?, input)]),
            Some(label),
        )?;
        if label == "key" {
            // Preserve the checked map and its stored identity. Per-head
            // conversion commutes with selecting complete rows, so evaluate
            // only the selected heads through the explicit indexed contract.
            let [node] = transport.nodes.as_mut_slice() else {
                return Err(invalid("row transport requires one indexed operation"));
            };
            node.operation_id = id(INDEXED_HEAD_FP8_ROUNDTRIP_BF16_OPERATION_ID)?;
            node.attributes = HeadFp8RoundtripBf16Geometry::attributes(c.head_dim)?;
            transport.operations = vec![NumericalOperationContract {
                operation_id: node.operation_id.clone(),
                version: ContractVersion::new(1, 0),
                multiplication_type: Some(ElementType::F32),
                accumulation_type: None,
                staged_arithmetic: None,
                composite_arithmetic: None,
            }];
        }
        parts.push(transport);
        parts.push(bridge(
            c,
            &format!("{label}_valid_flat"),
            fragment::scoped_value(&id(FastH3VsaRowTransportFamily::OUTPUT_ID)?, Some(label))?,
            value(&format!("{label}_valid_flat"))?,
            c.valid_key_rows(),
            LeadingBatchBf16Kind::Merge,
        )?);
    }
    let kvflat = vec![c.batches() * c.valid_key_rows(), c.columns()];
    parts.push(one(
        "dense",
        c.arithmetic.operation_id(),
        vec![
            (value("query_quantized_flat")?, tensor(qflat.clone())),
            (value("key_valid_flat")?, tensor(kvflat.clone())),
            (value("value_valid_flat")?, tensor(kvflat)),
        ],
        value("attended_flat")?,
        qflat,
        c.attention_attributes()?,
        true,
    )?);
    parts.push(bridge(
        c,
        "output_split",
        value("attended_flat")?,
        id(F::OUTPUT_ID)?,
        c.prefix_rows,
        LeadingBatchBf16Kind::Split,
    )?);
    StoredComposition::join(
        StoredCompositionIdentity {
            family: id(F::FAMILY_ID)?,
            profile: NumericalProfileId::new(c.profile_id()).map_err(invalid)?,
            format: id("weight-format.safetensors.dense")?,
            layout: id("weight-layout.fast_h3.vsa.prefix_attention.v1")?,
            block: "block.fast_h3.vsa.prefix_attention".into(),
        },
        parts,
        [F::QUERY_ID, F::KEY_ID, F::VALUE_ID]
            .into_iter()
            .map(id)
            .collect::<Result<_, _>>()?,
        vec![id(F::OUTPUT_ID)?, id(F::CENTERED_KEY_ID)?],
        id(F::OUTPUT_ID)?,
        StoredCompositionLimits {
            maximum_nodes: l.maximum_plan_nodes,
            maximum_stored_bytes: l.maximum_stored_bytes,
            maximum_metadata_bytes: l.maximum_metadata_bytes,
        },
    )
}
