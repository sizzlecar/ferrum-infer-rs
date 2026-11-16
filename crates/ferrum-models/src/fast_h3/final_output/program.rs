use super::*;
use crate::tensor_program_fragment::{
    StoredComposition, StoredCompositionIdentity, StoredCompositionLimits,
};
pub(super) type Composed = StoredComposition;
pub(super) fn compose(
    c: &FastH3FinalOutputConfig,
    limits: FastH3FinalOutputLimits,
) -> Result<Composed, VNextError> {
    c.validate(limits)?;
    type F = FastH3FinalOutputFamily;
    let norm = Fragment::family(
        &FastH3OutputNormFamily::new(limits.conditioning)?,
        &c.norm,
        FastH3OutputNormFamily::PROFILE_ID,
        c.norm.compile_options()?.tensor_specs(),
        fragment::mapping(&[
            (FastH3OutputNormFamily::INPUT_ID, F::INPUT_ID),
            (FastH3OutputNormFamily::TABLE_ID, F::TABLE_ID),
        ])?,
    )?;
    let mut fragments = vec![norm];
    for (label, modality, projection) in [
        ("video", FastH3OutputModality::Video, &c.video),
        ("audio", FastH3OutputModality::Audio, &c.audio),
    ] {
        let input = value(&format!("{label}.f32"))?;
        fragments.push(select(
            c,
            label,
            &runs(&c.segments, modality)?,
            input.clone(),
        )?);
        fragments.push(Fragment::family_instance(
            &FastH3PddProjectionFamily::new(limits.projection)?,
            projection,
            FastH3PddProjectionFamily::PROFILE_ID,
            projection.compile_options()?.tensor_specs(),
            BTreeMap::from([(id(FastH3PddProjectionFamily::INPUT_ID)?, input)]),
            Some(label),
        )?);
    }
    let mut built = StoredComposition::join(
        StoredCompositionIdentity {
            family: id(F::FAMILY_ID)?,
            profile: NumericalProfileId::new(F::PROFILE_ID).map_err(invalid)?,
            format: id("weight-format.safetensors.dense")?,
            layout: id("weight-layout.fast_h3.final_output.v1")?,
            block: "block.fast_h3.final_output".into(),
        },
        fragments,
        vec![id(F::INPUT_ID)?, id(F::TABLE_ID)?],
        vec![id(F::VIDEO_OUTPUT_ID)?, id(F::AUDIO_OUTPUT_ID)?],
        id(F::VIDEO_OUTPUT_ID)?,
        StoredCompositionLimits {
            maximum_nodes: limits.maximum_plan_nodes,
            maximum_stored_bytes: limits.maximum_stored_bytes,
            maximum_metadata_bytes: limits.maximum_metadata_bytes,
        },
    )?;
    if c.input_residency == FastH3PredictionInput::RequestState {
        let hidden: ProgramValueId = id(F::INPUT_ID)?;
        let mut specs = built.options.tensor_specs().clone();
        let tensor = specs
            .remove(&hidden)
            .ok_or_else(|| invalid("missing hidden input geometry"))?;
        let states = vec![StateSpec {
            id: id("state.fast_h3.final_output.hidden")?,
            value_id: hidden.clone(),
            tensor,
            lifetime: StateLifetime::Request,
            capacity_demand: StateCapacityDemand::FixedPerScope,
            initialization: StateInitialization::None,
            checkpoint: StateCheckpointCapability::Unsupported,
        }];
        let inputs = built
            .program
            .inputs()
            .iter()
            .filter(|v| *v != &hidden)
            .cloned()
            .collect();
        let input_specs = built
            .program
            .input_tensor_specs()
            .iter()
            .filter(|(v, _)| *v != &hidden)
            .map(|(v, s)| (v.clone(), s.clone()))
            .collect();
        built.program = ModelProgram::new(
            built.program.family_id().clone(),
            inputs,
            built.program.blocks().to_vec(),
            states.clone(),
            built.program.weights().to_vec(),
            built.program.outputs().to_vec(),
        )?
        .with_input_tensor_specs(input_specs)?;
        built.profile.states = states;
        built.profile.validate_program(&built.program)?;
        built
            .schema
            .validate_program_references(&built.profile.family_id, &built.program)?;
        built.options = ProgramPlanCompileOptions::new(specs)?;
    }
    Ok(built)
}

fn value(label: &str) -> Result<ProgramValueId, VNextError> {
    id(&format!("value.fast_h3.final_output.{label}"))
}
fn tensor(rows: u64, columns: u64, element_type: ElementType) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: vec![rows, columns],
        element_type,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn list(
    name: &str,
    values: impl IntoIterator<Item = u64>,
) -> Result<(AttributeId, SemanticValue), VNextError> {
    Ok((
        id(name)?,
        SemanticValue::Integers(
            values
                .into_iter()
                .map(|v| i64::try_from(v).map_err(invalid))
                .collect::<Result<_, _>>()?,
        ),
    ))
}
fn select(
    c: &FastH3FinalOutputConfig,
    label: &str,
    runs: &[RowRun],
    output: ProgramValueId,
) -> Result<Fragment, VNextError> {
    let input: ProgramValueId = id(FastH3OutputNormFamily::OUTPUT_ID)?;
    let h = c.norm.hidden_size();
    let mut nodes = vec![];
    let mut specs = BTreeMap::from([(input.clone(), tensor(c.norm.rows(), h, ElementType::Bf16))]);
    let mut boundaries = BTreeMap::new();
    let mut add = |name: &str,
                   op: &str,
                   inputs: Vec<ProgramValueId>,
                   output: ProgramValueId,
                   rows,
                   dtype,
                   attributes|
     -> Result<(), VNextError> {
        nodes.push(ProgramNode {
            id: id(&format!("node.fast_h3.final_output.{label}.{name}"))?,
            operation_id: id(op)?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs,
            outputs: vec![output.clone()],
            attributes,
        });
        specs.insert(output.clone(), tensor(rows, h, dtype));
        boundaries.insert(output, dtype);
        Ok(())
    };
    let mut previous = None;
    let mut total = 0u64;
    for (index, chunk) in runs.chunks(ROW_GATHER_BF16_MAX_RUNS).enumerate() {
        let rows = chunk.iter().try_fold(0u64, |n, r| {
            n.checked_add(r.rows)
                .ok_or_else(|| invalid("selection extent overflow"))
        })?;
        let part = value(&format!("{label}.part{index}"))?;
        let attrs = BTreeMap::from([
            (id("batch")?, SemanticValue::Unsigned(1)),
            (id("column_start")?, SemanticValue::Unsigned(0)),
            (id("column_count")?, SemanticValue::Unsigned(h)),
            list("run_lengths", chunk.iter().map(|r| r.rows))?,
            list("source_starts", chunk.iter().map(|r| r.start))?,
            list("source_batch_strides", std::iter::repeat_n(0, chunk.len()))?,
            list("source_row_strides", std::iter::repeat_n(1, chunk.len()))?,
        ]);
        add(
            &format!("gather{index}"),
            ROW_GATHER_BF16_OPERATION_ID,
            vec![input.clone()],
            part.clone(),
            rows,
            ElementType::Bf16,
            attrs,
        )?;
        total = total
            .checked_add(rows)
            .ok_or_else(|| invalid("selected total overflow"))?;
        previous = Some(if let Some(left) = previous {
            let combined = value(&format!("{label}.combined{index}"))?;
            add(
                &format!("concat{index}"),
                BATCHED_ROW_CONCAT_BF16_OPERATION_ID,
                vec![left, part],
                combined.clone(),
                total,
                ElementType::Bf16,
                BTreeMap::from([(id("batch")?, SemanticValue::Unsigned(1))]),
            )?;
            combined
        } else {
            part
        });
    }
    add(
        "promote",
        CAST_BF16_TO_F32_OPERATION_ID,
        vec![previous.ok_or_else(|| invalid("empty selection"))?],
        output,
        total,
        ElementType::F32,
        BTreeMap::new(),
    )?;
    let operations = nodes
        .iter()
        .map(|n| n.operation_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|operation_id| NumericalOperationContract {
            operation_id,
            version: ContractVersion::new(1, 0),
            multiplication_type: None,
            accumulation_type: None,
            staged_arithmetic: None,
            composite_arithmetic: None,
        })
        .collect();
    Ok(Fragment {
        schema: None,
        nodes,
        weights: vec![],
        specs,
        operations,
        boundaries,
    })
}
