use super::*;
pub(super) struct Composed {
    pub schema: WeightSchema,
    pub program: ModelProgram,
    pub profile: NumericalExecutionProfile,
    pub options: ProgramPlanCompileOptions,
}
pub(super) fn compose(
    c: &PH3ImageDiTConfig,
    limits: PH3ImageDiTLimits,
) -> Result<Composed, VNextError> {
    limits.validate()?;
    c.validate()?;
    if serde_json::to_vec(c)
        .map_err(|e| invalid(e.to_string()))?
        .len()
        > limits.max_metadata_bytes
    {
        return Err(invalid("configuration exceeds declared metadata capacity"));
    }
    let refiner = PH3ImageTokenRefinerFamily::new(limits.refiner())?;
    let stack = PH3ImageCurveStackFamily::new(limits.stack())?;
    let final_layer = PH3ImageFinalLayerFamily::new(limits.final_layer())?;
    let refine = fragment::Fragment::family(
        &refiner,
        &c.refiner,
        PH3ImageTokenRefinerFamily::CUDA_PROFILE_ID,
        c.refiner.compile_options()?,
        fragment::mapping(&[(
            PH3ImageTokenRefinerFamily::INPUT_ID,
            PH3ImageDiTFamily::RAW_TEXT_ID,
        )])?,
    )?;
    let main = fragment::Fragment::family(
        &stack,
        &c.stack,
        PH3ImageCurveStackFamily::cuda_profile_id(&c.stack),
        stack.compile_options(&c.stack)?,
        fragment::mapping(&[
            (PH3ImageCurveStackFamily::INPUT_ID, PACKED),
            (
                PH3ImageCurveStackFamily::BASIS_ID,
                PH3ImageDiTFamily::BASIS_ID,
            ),
            (
                PH3ImageCurveStackFamily::COSINE_ID,
                PH3ImageDiTFamily::COSINE_ID,
            ),
            (
                PH3ImageCurveStackFamily::SINE_ID,
                PH3ImageDiTFamily::SINE_ID,
            ),
        ])?,
    )?;
    let final_head = fragment::Fragment::family(
        &final_layer,
        &c.final_layer,
        PH3ImageFinalLayerFamily::CUDA_PROFILE_ID,
        c.final_layer.compile_options()?,
        fragment::mapping(&[
            (
                PH3ImageFinalLayerFamily::INPUT_ID,
                PH3ImageCurveStackFamily::OUTPUT_ID,
            ),
            (
                PH3ImageFinalLayerFamily::BASIS_ID,
                PH3ImageDiTFamily::BASIS_ID,
            ),
        ])?,
    )?;
    let patch = projection::build(c)?;
    let packing = pack(c)?;
    let family = ModelFamilyId::new(PH3ImageDiTFamily::FAMILY_ID)?;
    let mut schema = WeightSchema {
        format_id: WeightFormatId::new("weight-format.safetensors.dense")?,
        layout_id: WeightLayoutId::new(
            "weight-layout.p_h3_image.dit_forward.complete_canonical_mixed.v1",
        )?,
        version: ContractVersion::new(1, 0),
        components: vec![],
        tensors: vec![],
    };
    let mut weights = vec![];
    let mut nodes = vec![];
    let mut specs = c.input_tensor_specs()?;
    let mut boundaries = BTreeMap::new();
    let mut operations = BTreeMap::new();
    for (fragment, active) in [
        (refine, c.phase != PH3ImageDiTPhase::RefinedText),
        (patch, c.phase != PH3ImageDiTPhase::RefineOnly),
        (packing, c.phase != PH3ImageDiTPhase::RefineOnly),
        (main, c.phase != PH3ImageDiTPhase::RefineOnly),
        (final_head, c.phase != PH3ImageDiTPhase::RefineOnly),
    ] {
        // The complete immutable weight declaration deliberately survives the
        // steady phase. SharedComponentWeights validates its physical plan.
        schema.components.extend(fragment.schema.components);
        schema.tensors.extend(fragment.schema.tensors);
        weights.extend(fragment.weights);
        if !active {
            continue;
        }
        nodes.extend(fragment.nodes);
        for (id, spec) in fragment.specs {
            if specs.get(&id).is_some_and(|old| old != &spec) {
                return Err(invalid(
                    "joined fragments disagree about activation geometry",
                ));
            }
            specs.insert(id, spec);
        }
        for (id, dtype) in fragment.boundaries {
            if boundaries.insert(id, dtype).is_some() {
                return Err(invalid("fragments produce the same activation"));
            }
        }
        for op in fragment.operations {
            if operations
                .get(&op.operation_id)
                .is_some_and(|old| old != &op)
            {
                return Err(invalid("fragments disagree about numerical arithmetic"));
            }
            operations.insert(op.operation_id.clone(), op);
        }
    }
    if nodes.len() > limits.max_plan_nodes {
        return Err(invalid("complete DiT exceeds node capacity"));
    }
    let stored = schema.components.iter().try_fold(0u64, |n, v| {
        n.checked_add(v.physical_bytes()?)
            .ok_or_else(|| invalid("complete weight bytes overflow"))
    })?;
    if stored > limits.max_stored_bytes {
        return Err(invalid("complete DiT exceeds Stored weight capacity"));
    }
    schema.validate(&family)?;
    let mut outputs = if c.phase == PH3ImageDiTPhase::RefineOnly {
        vec![ProgramValueId::new(PH3ImageDiTFamily::REFINED_TEXT_ID)?]
    } else {
        vec![
            ProgramValueId::new(PH3ImageDiTFamily::IMAGE_VELOCITY_ID)?,
            ProgramValueId::new(PH3ImageDiTFamily::AUDIO_VELOCITY_ID)?,
        ]
    };
    if c.phase == PH3ImageDiTPhase::WithTextRefiner {
        outputs.push(ProgramValueId::new(PH3ImageDiTFamily::REFINED_TEXT_ID)?);
    }
    let inputs = c.input_tensor_specs()?;
    let program = ModelProgram::new(
        family.clone(),
        inputs.keys().cloned().collect(),
        vec![ProgramBlock {
            id: "block.p_h3_image.dit_forward".into(),
            nodes,
        }],
        vec![],
        weights,
        outputs,
    )?
    .with_input_tensor_specs(inputs)?;
    schema.validate_program_references(&family, &program)?;
    if c.attention_policy == PH3ImageDenoiserAttentionPolicy::CudnnSdpaExperimentalV1 {
        // Internal probability/reduction rounding belongs to the selected
        // vendor implementation. Do not inherit the sequential all-F32 claim.
        if let Some(attention) =
            operations.get_mut(&OperationId::new(SEGMENTED_ATTENTION_BF16_OPERATION_ID)?)
        {
            attention.multiplication_type = None;
            attention.accumulation_type = None;
        }
    }
    let profile = NumericalExecutionProfile {
        id: NumericalProfileId::new(PH3ImageDiTFamily::cuda_profile_id(c)).map_err(invalid)?,
        family_id: family,
        version: ContractVersion::new(1, 0),
        primary_activation: ProgramValueId::new(if c.phase == PH3ImageDiTPhase::RefineOnly {
            PH3ImageDiTFamily::REFINED_TEXT_ID
        } else {
            PH3ImageDiTFamily::IMAGE_VELOCITY_ID
        })?,
        boundaries,
        states: vec![],
        kv_storage: vec![],
        operations: operations.into_values().collect(),
    };
    profile.validate_program(&program)?;
    let required = program
        .inputs()
        .iter()
        .chain(
            program
                .blocks()
                .iter()
                .flat_map(|b| &b.nodes)
                .flat_map(|n| &n.outputs),
        )
        .cloned()
        .collect::<BTreeSet<_>>();
    if specs.keys().cloned().collect::<BTreeSet<_>>() != required {
        return Err(invalid(
            "complete DiT has missing or inactive activation shapes",
        ));
    }
    let encoded = serde_json::to_vec(&(&schema, &program, &profile, &specs))
        .map_err(|e| invalid(e.to_string()))?;
    if encoded.len() > limits.max_metadata_bytes {
        return Err(invalid("complete DiT exceeds metadata capacity"));
    }
    Ok(Composed {
        schema,
        program,
        profile,
        options: ProgramPlanCompileOptions::new(specs)?,
    })
}
pub(super) const PACKED: &str = "value.p_h3_image.dit_forward.packed";
fn value(name: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.p_h3_image.dit_forward.{name}"))
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
                .map(|v| i64::try_from(v).map_err(|_| invalid("packing run exceeds signed extent")))
                .collect::<Result<_, _>>()?,
        ),
    ))
}
fn pack(c: &PH3ImageDiTConfig) -> Result<fragment::Fragment, VNextError> {
    let mut nodes = vec![];
    let mut specs = BTreeMap::new();
    let mut boundaries = BTreeMap::new();
    let text = ProgramValueId::new(PH3ImageDiTFamily::REFINED_TEXT_ID)?;
    let image = ProgramValueId::new(projection::IMAGE_EMBEDDED)?;
    let audio = ProgramValueId::new(projection::AUDIO_EMBEDDED)?;
    let text_rows = c.text_rows()?;
    let image_rows = c.image_rows()?;
    let audio_rows = c.audio_rows()?;
    for (id, rows) in [
        (text.clone(), text_rows),
        (image.clone(), image_rows),
        (audio.clone(), audio_rows),
    ] {
        specs.insert(id, config::tensor(vec![rows, c.hidden], ElementType::Bf16));
    }
    let mut add = |name: &str,
                   op: &str,
                   inputs: Vec<ProgramValueId>,
                   out: ProgramValueId,
                   rows: u64,
                   attrs|
     -> Result<(), VNextError> {
        nodes.push(projection::node(name, op, inputs, out.clone(), attrs)?);
        specs.insert(
            out.clone(),
            config::tensor(vec![rows, c.hidden], ElementType::Bf16),
        );
        boundaries.insert(out, ElementType::Bf16);
        Ok(())
    };
    let concat_attrs = || {
        BTreeMap::from([(
            AttributeId::new("batch").unwrap(),
            SemanticValue::Unsigned(1),
        )])
    };
    let text_image = value("text_image_group")?;
    let all = value("source_groups")?;
    let first_rows = text_rows
        .checked_add(image_rows)
        .ok_or_else(|| invalid("packing rows overflow"))?;
    let total = first_rows
        .checked_add(audio_rows)
        .ok_or_else(|| invalid("packing rows overflow"))?;
    add(
        "group_text_image",
        BATCHED_ROW_CONCAT_BF16_OPERATION_ID,
        vec![text, image],
        text_image.clone(),
        first_rows,
        concat_attrs(),
    )?;
    add(
        "group_audio",
        BATCHED_ROW_CONCAT_BF16_OPERATION_ID,
        vec![text_image, audio],
        all.clone(),
        total,
        concat_attrs(),
    )?;
    let runs = packing_runs(c)?;
    let chunks = runs.chunks(ROW_GATHER_BF16_MAX_RUNS).collect::<Vec<_>>();
    let mut previous = None;
    let mut combined = 0u64;
    for (index, chunk) in chunks.iter().enumerate() {
        let rows = chunk.iter().try_fold(0u64, |n, r| {
            n.checked_add(r.1)
                .ok_or_else(|| invalid("packing chunk overflow"))
        })?;
        let out = if chunks.len() == 1 {
            ProgramValueId::new(PACKED)?
        } else {
            value(&format!("packed_chunk{index}"))?
        };
        let attrs = BTreeMap::from([
            (AttributeId::new("batch")?, SemanticValue::Unsigned(1)),
            (
                AttributeId::new("column_start")?,
                SemanticValue::Unsigned(0),
            ),
            (
                AttributeId::new("column_count")?,
                SemanticValue::Unsigned(c.hidden),
            ),
            list("run_lengths", chunk.iter().map(|r| r.1))?,
            list("source_starts", chunk.iter().map(|r| r.0))?,
            list("source_batch_strides", std::iter::repeat_n(0, chunk.len()))?,
            list("source_row_strides", std::iter::repeat_n(1, chunk.len()))?,
        ]);
        add(
            &format!("pack{index}"),
            ROW_GATHER_BF16_OPERATION_ID,
            vec![all.clone()],
            out.clone(),
            rows,
            attrs,
        )?;
        combined = combined
            .checked_add(rows)
            .ok_or_else(|| invalid("packed rows overflow"))?;
        previous = Some(if let Some(left) = previous {
            let merged = if index + 1 == chunks.len() {
                ProgramValueId::new(PACKED)?
            } else {
                value(&format!("packed_combined{index}"))?
            };
            add(
                &format!("pack_concat{index}"),
                BATCHED_ROW_CONCAT_BF16_OPERATION_ID,
                vec![left, out],
                merged.clone(),
                combined,
                concat_attrs(),
            )?;
            merged
        } else {
            out
        });
    }
    let operations = [
        ROW_GATHER_BF16_OPERATION_ID,
        BATCHED_ROW_CONCAT_BF16_OPERATION_ID,
    ]
    .into_iter()
    .map(|id| {
        Ok(NumericalOperationContract {
            operation_id: OperationId::new(id)?,
            version: ContractVersion::new(1, 0),
            multiplication_type: None,
            accumulation_type: None,
            staged_arithmetic: None,
            composite_arithmetic: None,
        })
    })
    .collect::<Result<_, VNextError>>()?;
    Ok(fragment::Fragment {
        schema: WeightSchema {
            format_id: WeightFormatId::new("weight-format.safetensors.dense")?,
            layout_id: WeightLayoutId::new("weight-layout.p_h3_image.dit_forward.transport.v1")?,
            version: ContractVersion::new(1, 0),
            components: vec![],
            tensors: vec![],
        },
        nodes,
        weights: vec![],
        specs,
        operations,
        boundaries,
    })
}
/// Source matrices are grouped by kind, while attention documents are grouped
/// by request. Explicit runs preserve text/ref/audio/target order for each one.
pub(super) fn packing_runs(c: &PH3ImageDiTConfig) -> Result<Vec<(u64, u64)>, VNextError> {
    let mut runs = Vec::new();
    let mut text = 0u64;
    let mut image = c.text_rows()?;
    let mut audio = image
        .checked_add(c.image_rows()?)
        .ok_or_else(|| invalid("audio group offset overflow"))?;
    for r in &c.requests {
        let reference = r.reference_rows(c)?;
        let target = r.target_rows(c)?;
        runs.push((text, r.text_rows));
        if reference > 0 {
            runs.push((image, reference));
        }
        runs.push((audio, 4));
        runs.push((
            image
                .checked_add(reference)
                .ok_or_else(|| invalid("target offset overflow"))?,
            target,
        ));
        text = text
            .checked_add(r.text_rows)
            .ok_or_else(|| invalid("text offset overflow"))?;
        image = image
            .checked_add(reference)
            .and_then(|n| n.checked_add(target))
            .ok_or_else(|| invalid("image offset overflow"))?;
        audio = audio
            .checked_add(4)
            .ok_or_else(|| invalid("audio offset overflow"))?;
    }
    Ok(runs)
}
