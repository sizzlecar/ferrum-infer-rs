use super::*;
pub(super) struct Template {
    pub schema: WeightSchema,
    pub program: ModelProgram,
    pub profile: NumericalExecutionProfile,
    pub options: ProgramPlanCompileOptions,
}
pub(super) struct Templates {
    variants: BTreeMap<PH3ImageBlockProjectionStorage, Template>,
}
impl Templates {
    pub fn layer(&self, c: &PH3ImageCurveStackConfig, index: u64) -> Result<&Template, VNextError> {
        self.variants
            .get(&c.storage_for_layer(index)?)
            .ok_or_else(|| invalid("missing declared projection template"))
    }
}
pub(super) struct Composed {
    pub schema: WeightSchema,
    pub program: ModelProgram,
    pub profile: NumericalExecutionProfile,
    pub options: ProgramPlanCompileOptions,
}
fn strings(value: &serde_json::Value) -> Result<usize, VNextError> {
    match value {
        serde_json::Value::String(_) => Ok(1),
        serde_json::Value::Array(a) => a.iter().try_fold(0usize, |n, v| {
            n.checked_add(strings(v)?)
                .ok_or_else(|| invalid("metadata strings overflow"))
        }),
        serde_json::Value::Object(o) => o.values().try_fold(o.len(), |n, v| {
            n.checked_add(strings(v)?)
                .ok_or_else(|| invalid("metadata strings overflow"))
        }),
        _ => Ok(0),
    }
}
/// Upper-bound every string occurrence by the maximum new ID prefix, even
/// strings that do not change. Repeated template roots/shared ports/contracts
/// further overestimate the merged graph. The bound is derived from the real
/// serialized template and layer count, not a fixed node/token-size guess.
pub(super) fn metadata_bound(
    c: &PH3ImageCurveStackConfig,
    templates: &Templates,
) -> Result<usize, VNextError> {
    c.validate()?;
    let layer = c.layers - 1;
    let prefix = format!("weight.p_h3_image.curve_stack.layer{layer}.")
        .len()
        .max(format!("value.p_h3_image.curve_stack.layer{layer}.").len())
        .max(format!("node.p_h3_image.curve_stack.layer{layer}.").len());
    let sizes = templates
        .variants
        .iter()
        .map(|(storage, t)| {
            let value =
                serde_json::to_value((&t.schema, &t.program, &t.profile, t.options.tensor_specs()))
                    .map_err(|e| invalid(e.to_string()))?;
            let encoded = serde_json::to_vec(&value)
                .map_err(|e| invalid(e.to_string()))?
                .len();
            let scoped = encoded
                .checked_add(
                    strings(&value)?
                        .checked_mul(prefix)
                        .ok_or_else(|| invalid("metadata identifier growth overflows"))?,
                )
                .ok_or_else(|| invalid("metadata template bound overflows"))?;
            Ok((*storage, scoped))
        })
        .collect::<Result<BTreeMap<_, _>, VNextError>>()?;
    match &c.projection_storage {
        PH3ImageStackProjectionStorage::Uniform(storage) => sizes[storage]
            .checked_mul(
                usize::try_from(c.layers).map_err(|_| invalid("layers exceed address space"))?,
            )
            .and_then(|n| n.checked_add(c.metadata_bytes().ok()?))
            .ok_or_else(|| invalid("complete metadata bound overflows")),
        PH3ImageStackProjectionStorage::PerLayer(storage) => {
            storage.iter().try_fold(c.metadata_bytes()?, |sum, p| {
                sum.checked_add(sizes[p])
                    .ok_or_else(|| invalid("complete metadata bound overflows"))
            })
        }
    }
}
pub(super) fn admit(
    c: &PH3ImageCurveStackConfig,
    limits: PH3ImageCurveStackLimits,
) -> Result<Templates, VNextError> {
    limits.validate()?;
    c.validate()?;
    if c.metadata_bytes()? > limits.max_metadata_bytes {
        return Err(invalid("configuration exceeds declared metadata capacity"));
    }
    let family = PH3ImageCurveBlockFamily::new(limits.block())?;
    let mut variants = BTreeMap::new();
    for storage in c.projection_storage.unique() {
        let block = c.block.clone().with_projection_storage(storage)?;
        let ModelWeights::Stored(schema) = family.weights(&block)? else {
            return Err(invalid("Curve block template must have Stored weights"));
        };
        let profiles = family.numerical_profiles(&block)?;
        let profile = profiles
            .resolve(
                &NumericalProfileId::new(PH3ImageCurveBlockFamily::cuda_profile_id(&block))
                    .map_err(invalid)?,
            )?
            .clone();
        let program = family.semantic_program(&block, &profile)?;
        let options = block.compile_options()?;
        profile.validate_program(&program)?;
        let expected = program
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
        if expected != options.tensor_specs().keys().cloned().collect() {
            return Err(invalid(
            "block template explicit geometry does not cover exactly inputs and produced values",
        ));
        }
        let nodes = program.blocks().iter().try_fold(0usize, |n, b| {
            n.checked_add(b.nodes.len())
                .ok_or_else(|| invalid("template node count overflows"))
        })?;
        if nodes > limits.max_plan_nodes {
            return Err(invalid("stack exceeds declared plan-node capacity"));
        }
        let bytes = schema.components.iter().try_fold(0u64, |n, c| {
            n.checked_add(c.physical_bytes()?)
                .ok_or_else(|| invalid("template stored bytes overflow"))
        })?;
        if bytes > limits.max_stored_bytes {
            return Err(invalid("stack exceeds declared Stored-weight capacity"));
        }
        variants.insert(
            storage,
            Template {
                schema,
                program,
                profile,
                options,
            },
        );
    }
    let t = Templates { variants };
    let costs = t
        .variants
        .iter()
        .map(|(storage, t)| {
            let nodes = t
                .program
                .blocks()
                .iter()
                .map(|b| b.nodes.len())
                .sum::<usize>();
            let bytes = t.schema.components.iter().try_fold(0u64, |n, c| {
                n.checked_add(c.physical_bytes()?)
                    .ok_or_else(|| invalid("template stored bytes overflow"))
            })?;
            Ok((*storage, (nodes, bytes)))
        })
        .collect::<Result<BTreeMap<_, _>, VNextError>>()?;
    let (nodes, bytes) = match &c.projection_storage {
        PH3ImageStackProjectionStorage::Uniform(p) => {
            let (n, b) = costs[p];
            (
                n.checked_mul(
                    usize::try_from(c.layers)
                        .map_err(|_| invalid("layers exceed address space"))?,
                )
                .ok_or_else(|| invalid("stack node count overflow"))?,
                b.checked_mul(c.layers)
                    .ok_or_else(|| invalid("stack stored bytes overflow"))?,
            )
        }
        PH3ImageStackProjectionStorage::PerLayer(p) => {
            p.iter().try_fold((0usize, 0u64), |(n, b), p| {
                let (dn, db) = costs[p];
                Ok::<_, VNextError>((
                    n.checked_add(dn)
                        .ok_or_else(|| invalid("stack node count overflow"))?,
                    b.checked_add(db)
                        .ok_or_else(|| invalid("stack stored bytes overflow"))?,
                ))
            })?
        }
    };
    if nodes > limits.max_plan_nodes || bytes > limits.max_stored_bytes {
        return Err(invalid(
            "stack exceeds declared node or Stored-weight capacity",
        ));
    }
    if metadata_bound(c, &t)? > limits.max_metadata_bytes {
        return Err(invalid(
            "stack exceeds declared metadata serialization bound",
        ));
    }
    Ok(t)
}
pub(super) fn build(
    c: &PH3ImageCurveStackConfig,
    limits: PH3ImageCurveStackLimits,
) -> Result<Composed, VNextError> {
    let templates = admit(c, limits)?;
    let mut components = Vec::new();
    let mut tensors = Vec::new();
    let mut nodes = Vec::new();
    let mut weights = Vec::new();
    let mut specs = BTreeMap::new();
    let mut boundaries = BTreeMap::new();
    let mut operations = BTreeMap::new();
    for layer in 0..c.layers {
        let template = templates.layer(c, layer)?;
        let schema = remap::schema(layer, &template.schema)?;
        components.extend(schema.components);
        tensors.extend(schema.tensors);
        let body = remap::layer(c, layer, template)?;
        nodes.extend(body.nodes);
        weights.extend(body.weights);
        for (id, spec) in body.specs {
            if specs.get(&id).is_some_and(|old| old != &spec) {
                return Err(invalid(
                    "shared activation changes tensor structure between layers",
                ));
            }
            specs.insert(id, spec);
        }
        for (id, dtype) in body.boundaries {
            if boundaries.insert(id, dtype).is_some() {
                return Err(invalid("multiple layer output boundaries"));
            }
        }
        for op in &template.profile.operations {
            if operations
                .get(&op.operation_id)
                .is_some_and(|old| old != op)
            {
                return Err(invalid(
                    "layers disagree about numerical operation contract",
                ));
            }
            operations.insert(op.operation_id.clone(), op.clone());
        }
    }
    let template = templates.layer(c, 0)?;
    let schema = WeightSchema {
        format_id: template.schema.format_id.clone(),
        layout_id: WeightLayoutId::new(if c.projection_storage.all_w4() {
            "weight-layout.p_h3_image.curve_stack.canonical_w4_f32.v1"
        } else {
            "weight-layout.p_h3_image.curve_stack.canonical_mixed_convrot_f32.v1"
        })?,
        version: template.schema.version,
        components,
        tensors,
    };
    let family = ModelFamilyId::new(PH3ImageCurveStackFamily::FAMILY_ID)?;
    schema.validate(&family)?;
    let program = ModelProgram::new(
        family.clone(),
        c.input_tensor_specs()?.keys().cloned().collect(),
        vec![ProgramBlock {
            id: "block.p_h3_image.curve_stack".into(),
            nodes,
        }],
        vec![],
        weights,
        vec![ProgramValueId::new(PH3ImageCurveStackFamily::OUTPUT_ID)?],
    )?
    .with_input_tensor_specs(c.input_tensor_specs()?)?;
    schema.validate_program_references(&family, &program)?;
    let profile = NumericalExecutionProfile {
        id: NumericalProfileId::new(PH3ImageCurveStackFamily::cuda_profile_id(c))
            .map_err(invalid)?,
        family_id: family,
        version: ContractVersion::new(1, 0),
        primary_activation: ProgramValueId::new(PH3ImageCurveStackFamily::OUTPUT_ID)?,
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
