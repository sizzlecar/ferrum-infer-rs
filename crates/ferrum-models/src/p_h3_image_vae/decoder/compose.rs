use super::*;
pub(super) struct Template {
    pub schema: WeightSchema,
    pub program: ModelProgram,
    pub profile: NumericalExecutionProfile,
    pub options: ProgramPlanCompileOptions,
}
pub(super) struct Composed {
    pub schema: WeightSchema,
    pub program: ModelProgram,
    pub profile: NumericalExecutionProfile,
    pub options: ProgramPlanCompileOptions,
}
fn strings(v: &serde_json::Value) -> Result<usize, VNextError> {
    match v {
        serde_json::Value::String(_) => Ok(1),
        serde_json::Value::Array(a) => a.iter().try_fold(0usize, |n, v| {
            n.checked_add(strings(v)?)
                .ok_or_else(|| invalid("metadata string count overflows"))
        }),
        serde_json::Value::Object(o) => o.values().try_fold(o.len(), |n, v| {
            n.checked_add(strings(v)?)
                .ok_or_else(|| invalid("metadata string count overflows"))
        }),
        _ => Ok(0),
    }
}
pub(super) fn admit(
    c: &PH3ImageVaeDecoderExecutionConfig,
    limits: PH3ImageVaeDecoderLimits,
) -> Result<Template, VNextError> {
    admit_with_fused_attention(c, limits, false)
}
fn admit_with_fused_attention(
    c: &PH3ImageVaeDecoderExecutionConfig,
    limits: PH3ImageVaeDecoderLimits,
    fused_attention: bool,
) -> Result<Template, VNextError> {
    limits.validate()?;
    c.validate()?;
    let family = PH3ImageVaeDecoderBlockFamily::new(limits.block())?;
    let block = c.block(0)?;
    let ModelWeights::Stored(schema) = family.weights(&block)? else {
        return Err(invalid("decoder block requires Stored weights"));
    };
    let profile = family
        .numerical_profiles(&block)?
        .resolve(
            &NumericalProfileId::new(if fused_attention {
                PH3ImageVaeDecoderBlockFamily::CUDA_FUSED_F32_PROFILE_ID
            } else {
                PH3ImageVaeDecoderBlockFamily::CUDA_F32_PROFILE_ID
            })
            .map_err(invalid)?,
        )?
        .clone();
    let program = family.semantic_program(&block, &profile)?;
    let options = block.compile_options()?;
    let layers = c.architecture().layers();
    let (outer_components, outer_tensors, outer_weights) = outer::schema(c)?;
    let mut outer_nodes = outer::prefix(c)?;
    outer_nodes.extend(outer::suffix(c, remap::output(layers - 1)?)?);
    let count = program.blocks().iter().try_fold(0usize, |n, b| {
        n.checked_add(b.nodes.len())
            .ok_or_else(|| invalid("block node count overflows"))
    })?;
    if count
        .checked_mul(
            usize::try_from(layers).map_err(|_| invalid("layer count exceeds address space"))?,
        )
        .and_then(|n| n.checked_add(outer_nodes.len()))
        .is_none_or(|n| n > limits.max_plan_nodes)
    {
        return Err(invalid("decoder exceeds declared plan-node capacity"));
    }
    let block_bytes = schema.components.iter().try_fold(0u64, |n, c| {
        n.checked_add(c.physical_bytes()?)
            .ok_or_else(|| invalid("block stored bytes overflow"))
    })?;
    let outer_bytes = outer_components.iter().try_fold(0u64, |n, c| {
        n.checked_add(c.physical_bytes()?)
            .ok_or_else(|| invalid("outer stored bytes overflow"))
    })?;
    if block_bytes
        .checked_mul(layers)
        .and_then(|n| n.checked_add(outer_bytes))
        .is_none_or(|n| n > limits.max_stored_bytes)
    {
        return Err(invalid("decoder exceeds declared Stored-weight capacity"));
    }
    // Bound identifier growth and every serialized template occurrence before
    // allocating the expanded 36-layer graph. This overcounts shared inputs.
    let raw = serde_json::to_value((&schema, &program, &profile, options.tensor_specs()))
        .map_err(|e| invalid(e.to_string()))?;
    let prefix = format!("weight.p_h3_image.vae.decoder.layer{}. ", layers - 1)
        .len()
        .max(format!("value.p_h3_image.vae.decoder.layer{}. ", layers - 1).len())
        .max(format!("node.p_h3_image.vae.decoder.layer{}. ", layers - 1).len());
    let template_bytes = serde_json::to_vec(&raw)
        .map_err(|e| invalid(e.to_string()))?
        .len()
        .checked_add(
            strings(&raw)?
                .checked_mul(prefix)
                .ok_or_else(|| invalid("metadata growth overflows"))?,
        )
        .ok_or_else(|| invalid("template metadata overflows"))?;
    let outer_bytes = serde_json::to_vec(&(
        c,
        outer_components,
        outer_tensors,
        outer_weights,
        outer_nodes,
        c.input_tensor_specs()?,
    ))
    .map_err(|e| invalid(e.to_string()))?
    .len();
    if template_bytes
        .checked_mul(layers as usize)
        .and_then(|n| n.checked_add(outer_bytes))
        .is_none_or(|n| n > limits.max_metadata_bytes)
    {
        return Err(invalid(
            "decoder exceeds declared metadata serialization bound",
        ));
    }
    let expected: BTreeSet<_> = program
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
        .collect();
    if expected != options.tensor_specs().keys().cloned().collect() {
        return Err(invalid(
            "block options do not cover exactly real graph inputs/outputs",
        ));
    }
    Ok(Template {
        schema,
        program,
        profile,
        options,
    })
}
pub(super) fn build(
    c: &PH3ImageVaeDecoderExecutionConfig,
    limits: PH3ImageVaeDecoderLimits,
) -> Result<Composed, VNextError> {
    build_with_fused_attention(c, limits, false)
}
pub(super) fn build_with_fused_attention(
    c: &PH3ImageVaeDecoderExecutionConfig,
    limits: PH3ImageVaeDecoderLimits,
    fused_attention: bool,
) -> Result<Composed, VNextError> {
    let template = admit_with_fused_attention(c, limits, fused_attention)?;
    let (mut components, mut tensors, mut weights) = outer::schema(c)?;
    let mut specs = c.input_tensor_specs()?;
    specs.extend(
        weights
            .iter()
            .map(|w| (w.value_id.clone(), w.tensor.clone())),
    );
    let mut nodes = outer::prefix(c)?;
    outer::specs(c, &nodes, &mut specs)?;
    let mut boundaries: BTreeMap<_, _> = nodes
        .iter()
        .flat_map(|n| &n.outputs)
        .map(|v| (v.clone(), ElementType::F32))
        .collect();
    for layer in 0..c.architecture().layers() {
        let schema = remap::schema(layer, &template.schema)?;
        components.extend(schema.components);
        tensors.extend(schema.tensors);
        let body = remap::layer(layer, &template)?;
        nodes.extend(body.nodes);
        weights.extend(body.weights);
        for (id, spec) in body.specs {
            if specs.get(&id).is_some_and(|old| old != &spec) {
                return Err(invalid("layer changes shared activation shape"));
            }
            specs.insert(id, spec);
        }
        for (id, dtype) in body.boundaries {
            if boundaries.insert(id, dtype).is_some() {
                return Err(invalid("multiple producers for decoder boundary"));
            }
        }
    }
    let suffix = outer::suffix(c, remap::output(c.architecture().layers() - 1)?)?;
    outer::specs(c, &suffix, &mut specs)?;
    boundaries.extend(
        suffix
            .iter()
            .flat_map(|n| &n.outputs)
            .map(|v| (v.clone(), ElementType::F32)),
    );
    nodes.extend(suffix);
    // Shape lookup may include static outer weight ports; the compiler's
    // explicit metadata is only the executable graph's inputs and outputs.
    for w in &weights {
        specs.remove(&w.value_id);
    }
    let family = ModelFamilyId::new(PH3ImageVaeDecoderFamily::FAMILY_ID)?;
    let schema = WeightSchema {
        format_id: template.schema.format_id,
        layout_id: WeightLayoutId::new(
            "weight-layout.p_h3_image.vae.decoder.comfy_int8.f32_promoted.v1",
        )?,
        version: ContractVersion::new(1, 0),
        components,
        tensors,
    };
    schema.validate(&family)?;
    let program = ModelProgram::new(
        family.clone(),
        c.input_tensor_specs()?.keys().cloned().collect(),
        vec![ProgramBlock {
            id: "block.p_h3_image.vae.decoder".into(),
            nodes,
        }],
        vec![],
        weights,
        vec![ProgramValueId::new(PH3ImageVaeDecoderFamily::OUTPUT_ID)?],
    )?
    .with_input_tensor_specs(c.input_tensor_specs()?)?;
    schema.validate_program_references(&family, &program)?;
    let mut operations: BTreeMap<_, _> = template
        .profile
        .operations
        .into_iter()
        .map(|o| (o.operation_id.clone(), o))
        .collect();
    for id in [
        AFFINE_LINEAR_F32_OPERATION_ID,
        LAYER_NORM_F32_OPERATION_ID,
        BATCHED_SUFFIX_APPEND_F32_OPERATION_ID,
    ] {
        let arithmetic = id != BATCHED_SUFFIX_APPEND_F32_OPERATION_ID;
        let operation_id = OperationId::new(id)?;
        operations.insert(
            operation_id.clone(),
            NumericalOperationContract {
                operation_id,
                version: ContractVersion::new(1, 0),
                multiplication_type: arithmetic.then_some(ElementType::F32),
                accumulation_type: arithmetic.then_some(ElementType::F32),
                staged_arithmetic: None,
                composite_arithmetic: None,
            },
        );
    }
    let profile = NumericalExecutionProfile {
        id: NumericalProfileId::new(if fused_attention {
            PH3ImageVaeDecoderFamily::CUDA_FUSED_F32_PROFILE_ID
        } else {
            PH3ImageVaeDecoderFamily::CUDA_PROFILE_ID
        })
        .map_err(invalid)?,
        family_id: family,
        version: ContractVersion::new(1, 0),
        primary_activation: ProgramValueId::new(PH3ImageVaeDecoderFamily::OUTPUT_ID)?,
        boundaries,
        states: vec![],
        kv_storage: vec![],
        operations: operations.into_values().collect(),
    };
    profile.validate_program(&program)?;
    let options = ProgramPlanCompileOptions::new(specs)?;
    Ok(Composed {
        schema,
        program,
        profile,
        options,
    })
}
