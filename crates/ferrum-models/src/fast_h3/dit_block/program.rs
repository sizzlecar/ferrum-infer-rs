use super::*;
use crate::tensor_program_fragment::{StoredCompositionIdentity, StoredCompositionLimits};
type F = FastH3DitBlockFamily;
fn tensor(dimensions: Vec<u64>) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn scoped(name: &str, instance: &str) -> Result<ProgramValueId, VNextError> {
    fragment::scoped_value(&id(name)?, Some(instance))
}
fn value(name: &str) -> Result<ProgramValueId, VNextError> {
    id(&format!("value.fast_h3.dit_block.{name}"))
}
pub(super) fn compose(
    c: &FastH3DitBlockConfig,
    l: FastH3DitBlockLimits,
) -> Result<StoredComposition, VNextError> {
    c.validate(l)?;
    let norm = FastH3ModulationFamily::new(l.modulation)?;
    let norm1 = Fragment::family_instance(
        &norm,
        &c.norm1,
        FastH3ModulationFamily::PROFILE_ID,
        c.norm1.compile_options()?.tensor_specs(),
        fragment::mapping(&[
            (FastH3ModulationFamily::INPUT_ID, F::HIDDEN_ID),
            (FastH3ModulationFamily::TABLE_ID, F::TABLE_ID),
        ])?,
        Some("attention_norm"),
    )?;
    let flat = tensor(vec![c.rows()?, c.hidden_size()]);
    let batched = tensor(vec![c.batches(), c.packed_rows(), c.hidden_size()]);
    let norm1_output = scoped(FastH3ModulationFamily::OUTPUT_ID, "attention_norm")?;
    let split_output = value("normalized_batched")?;
    LeadingBatchBf16Geometry::new(
        LeadingBatchBf16Kind::Split,
        &ResolvedTensorSpec::new(
            flat.dimensions.clone(),
            flat.element_type,
            flat.layout.clone(),
        )?,
        &ResolvedTensorSpec::new(
            batched.dimensions.clone(),
            batched.element_type,
            batched.layout.clone(),
        )?,
    )?;
    let split = Fragment {
        schema: None,
        weights: vec![],
        specs: BTreeMap::from([
            (norm1_output.clone(), flat.clone()),
            (split_output.clone(), batched),
        ]),
        boundaries: BTreeMap::from([(split_output.clone(), ElementType::Bf16)]),
        operations: vec![NumericalOperationContract {
            operation_id: id(SPLIT_LEADING_BATCH_BF16_VIEW_OPERATION_ID)?,
            version: ContractVersion::new(1, 0),
            multiplication_type: None,
            accumulation_type: None,
            staged_arithmetic: None,
            composite_arithmetic: None,
        }],
        nodes: vec![ProgramNode {
            id: id("node.fast_h3.dit_block.normalized_split")?,
            operation_id: id(SPLIT_LEADING_BATCH_BF16_VIEW_OPERATION_ID)?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs: vec![norm1_output],
            outputs: vec![split_output.clone()],
            attributes: BTreeMap::new(),
        }],
    };
    let attention_input = Fragment::family(
        &FastH3AttentionInputFamily::new(l.attention_input)?,
        &c.attention_input,
        FastH3AttentionInputFamily::PROFILE_ID,
        c.attention_input
            .compile_options(l.attention_input)?
            .tensor_specs(),
        BTreeMap::from([
            (id(FastH3AttentionInputFamily::HIDDEN_ID)?, split_output),
            (
                id(FastH3AttentionInputFamily::COSINE_ID)?,
                id(F::COSINE_ID)?,
            ),
            (id(FastH3AttentionInputFamily::SINE_ID)?, id(F::SINE_ID)?),
        ]),
    )?;
    let attention_core = Fragment::family(
        &FastH3AttentionCoreFamily::new(l.attention_core)?,
        &c.attention_core,
        c.attention_core.profile_id(),
        c.attention_core
            .compile_options(l.attention_core)?
            .tensor_specs(),
        [
            FastH3AttentionCoreFamily::QUERY_ID,
            FastH3AttentionCoreFamily::KEY_ID,
            FastH3AttentionCoreFamily::VALUE_ID,
            FastH3AttentionCoreFamily::GATE_ID,
            FastH3AttentionCoreFamily::POOLED_QUERY_ID,
            FastH3AttentionCoreFamily::POOLED_KEY_ID,
            FastH3AttentionCoreFamily::POOLED_VALUE_ID,
        ]
        .into_iter()
        .map(|name| Ok((id(name)?, id(name)?)))
        .collect::<Result<_, VNextError>>()?,
    )?;
    let norm2 = Fragment::family_instance(
        &norm,
        &c.norm2,
        FastH3ModulationFamily::PROFILE_ID,
        c.norm2.compile_options()?.tensor_specs(),
        fragment::mapping(&[
            (FastH3ModulationFamily::INPUT_ID, F::HIDDEN_ID),
            (FastH3ModulationFamily::TABLE_ID, F::TABLE_ID),
            (
                FastH3ModulationFamily::BRANCH_ID,
                FastH3AttentionCoreFamily::OUTPUT_ID,
            ),
        ])?,
        Some("ffn_norm"),
    )?;
    let ffn = Fragment::family(
        &FastH3FfnFamily::new(l.ffn)?,
        &c.ffn,
        FastH3FfnFamily::PROFILE_ID,
        c.ffn.compile_options()?.tensor_specs(),
        BTreeMap::from([(
            id(FastH3FfnFamily::INPUT_ID)?,
            scoped(FastH3ModulationFamily::OUTPUT_ID, "ffn_norm")?,
        )]),
    )?;
    let hidden = scoped(FastH3ModulationFamily::HIDDEN_ID, "ffn_norm")?;
    let table =
        c.norm2.compile_options()?.tensor_specs()[&id(FastH3ModulationFamily::TABLE_ID)?].clone();
    let attrs = c.final_attributes()?;
    let x = ResolvedTensorSpec::new(
        flat.dimensions.clone(),
        flat.element_type,
        flat.layout.clone(),
    )?;
    let t = ResolvedTensorSpec::new(
        table.dimensions.clone(),
        table.element_type,
        table.layout.clone(),
    )?;
    RowSelectedBf16Geometry::new(RowSelectedBf16Kind::AddcmulFma, &[&x, &t, &x], &x, &attrs)?;
    let final_residual = Fragment {
        schema: None,
        weights: vec![],
        specs: BTreeMap::from([
            (hidden.clone(), flat.clone()),
            (id(F::TABLE_ID)?, table),
            (id(FastH3FfnFamily::OUTPUT_ID)?, flat.clone()),
            (id(F::OUTPUT_ID)?, flat),
        ]),
        boundaries: BTreeMap::from([(id(F::OUTPUT_ID)?, ElementType::Bf16)]),
        operations: vec![NumericalOperationContract {
            operation_id: id(ROW_SELECTED_ADDCMUL_FMA_BF16_OPERATION_ID)?,
            version: ContractVersion::new(1, 0),
            multiplication_type: Some(ElementType::F32),
            accumulation_type: Some(ElementType::F32),
            staged_arithmetic: None,
            composite_arithmetic: None,
        }],
        nodes: vec![ProgramNode {
            id: id("node.fast_h3.dit_block.ffn_residual")?,
            operation_id: id(ROW_SELECTED_ADDCMUL_FMA_BF16_OPERATION_ID)?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs: vec![hidden, id(F::TABLE_ID)?, id(FastH3FfnFamily::OUTPUT_ID)?],
            outputs: vec![id(F::OUTPUT_ID)?],
            attributes: attrs,
        }],
    };
    StoredComposition::join(
        StoredCompositionIdentity {
            family: id(F::FAMILY_ID)?,
            profile: NumericalProfileId::new(c.profile_id()).map_err(invalid)?,
            format: id("weight-format.safetensors.dense")?,
            layout: id("weight-layout.fast_h3.dit_block.v1")?,
            block: "block.fast_h3.dit_block".into(),
        },
        vec![
            norm1,
            split,
            attention_input,
            attention_core,
            norm2,
            ffn,
            final_residual,
        ],
        [F::HIDDEN_ID, F::COSINE_ID, F::SINE_ID, F::TABLE_ID]
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
