use super::*;

pub(super) struct Prepared {
    pub schema: WeightSchema,
    pub sources: BTreeMap<WeightId, Vec<WeightId>>,
    pub maximum_group_bytes: u64,
}

impl Prepared {
    pub fn execution_weight_bytes(&self) -> Result<u64, VNextError> {
        self.schema
            .components
            .iter()
            .filter(|component| component.required)
            .try_fold(0_u64, |total, component| {
                total
                    .checked_add(component.physical_bytes()?)
                    .ok_or_else(|| invalid("required execution weight bytes overflow"))
            })
    }
}

pub(super) fn prepare(
    family: &PreparedModelFamily,
    groups: &[Vec<WeightId>],
) -> Result<Prepared, VNextError> {
    prepare_program(family.stored_weight_schema()?, family.program(), groups)
}

pub(super) fn prepare_program(
    original: &WeightSchema,
    program: &ModelProgram,
    groups: &[Vec<WeightId>],
) -> Result<Prepared, VNextError> {
    original.validate(program.family_id())?;
    if original.format_id.as_str() != "weight-format.safetensors.dense" {
        return Err(invalid(
            "requires the explicit dense safetensors source format",
        ));
    }
    let selected: BTreeSet<_> = groups.iter().flatten().cloned().collect();
    let weights: BTreeMap<_, _> = program
        .weights()
        .iter()
        .map(|reference| (&reference.value_id, &reference.weight_id))
        .collect();
    let mut consumed = BTreeSet::new();
    for node in program.blocks().iter().flat_map(|block| &block.nodes) {
        for (ordinal, input) in node.inputs.iter().enumerate() {
            let Some(weight) = weights.get(input) else {
                continue;
            };
            let fp8_weight_port = node.operation_id.as_str() == DENSE_LINEAR_FP8_BF16_OPERATION_ID
                && node.required_version == VERSION
                && ordinal == 1;
            if selected.contains(*weight) {
                if !fp8_weight_port {
                    return Err(invalid(format!(
                        "selected weight `{weight}` has a non-FP8 consumer"
                    )));
                }
                consumed.insert((*weight).clone());
            } else if fp8_weight_port {
                return Err(invalid("FP8 dense weight is absent from explicit groups"));
            }
        }
    }
    if consumed != selected {
        return Err(invalid(
            "selected weights must all have actual FP8 dense consumers",
        ));
    }
    let mut schema = original.clone();
    let mut source_map: BTreeMap<_, _> = original
        .components
        .iter()
        .map(|component| (component.id.clone(), vec![component.id.clone()]))
        .collect();
    let mut used_sources = BTreeSet::new();
    let mut maximum_group_bytes = 0;
    for group in groups {
        let mut source_specs = Vec::with_capacity(group.len());
        for weight in group {
            let tensor = original
                .tensors
                .iter()
                .find(|tensor| &tensor.id == weight)
                .ok_or_else(|| invalid("selected logical weight is absent"))?;
            let PhysicalWeightLayout::Dense { component_id } = &tensor.physical_layout else {
                return Err(invalid("conversion requires an exact dense source leaf"));
            };
            let component = original
                .components
                .iter()
                .find(|component| &component.id == component_id)
                .ok_or_else(|| invalid("selected source component is absent"))?;
            validate_source(component)?;
            if tensor.logical_element_type != ElementType::Bf16
                || tensor.dimensions != component.dimensions
                || !used_sources.insert(component.id.clone())
            {
                return Err(invalid(
                    "selected source shape/type differs or source is repeated",
                ));
            }
            source_specs.push(component);
        }
        let derived = derive_group(&source_specs)?;
        let source_ids: Vec<_> = source_specs.iter().map(|spec| spec.id.clone()).collect();
        let group_bytes = derived.iter().try_fold(0_u64, |total, component| {
            total
                .checked_add(component.physical_bytes()?)
                .ok_or_else(|| invalid("group bytes overflow"))
        })?;
        if group_bytes > isize::MAX as u64 {
            return Err(invalid("group backing exceeds address space"));
        }
        maximum_group_bytes = maximum_group_bytes.max(group_bytes);
        for (index, weight) in group.iter().enumerate() {
            let tensor = schema
                .tensors
                .iter_mut()
                .find(|tensor| &tensor.id == weight)
                .unwrap();
            tensor.physical_layout = PhysicalWeightLayout::QuantizedTensor {
                packed_values: PhysicalWeightComponentBinding::exact_contiguous(
                    derived[index].id.clone(),
                ),
                packed_dimensions: tensor.dimensions.clone(),
                scales: PhysicalWeightComponentBinding::exact_contiguous(
                    derived[source_specs.len() + index].id.clone(),
                ),
            };
        }
        for component in derived {
            if source_map
                .insert(component.id.clone(), source_ids.clone())
                .is_some()
            {
                return Err(invalid("derived component identity collides with source"));
            }
            schema.components.push(component);
        }
    }
    let mut referenced = BTreeSet::new();
    for tensor in &schema.tensors {
        referenced.extend(
            schema
                .physical_component_refs(&tensor.id)?
                .into_iter()
                .map(|c| c.id.clone()),
        );
    }
    schema
        .components
        .retain(|component| referenced.contains(&component.id));
    source_map.retain(|id, _| referenced.contains(id));
    schema.components.sort_by(|a, b| a.id.cmp(&b.id));
    schema.tensors.sort_by(|a, b| a.id.cmp(&b.id));
    // The container remains safetensors-compatible; exact component encodings,
    // layout identity and quantization format declare the mixed execution ABI.
    schema.layout_id = WeightLayoutId::new(BF16_WHOLE_TENSOR_FP8_LAYOUT_ID)?;
    schema.version = VERSION;
    schema.validate(program.family_id())?;
    Ok(Prepared {
        schema,
        sources: source_map,
        maximum_group_bytes,
    })
}

pub(super) fn validate_source(component: &WeightComponentSpec) -> Result<(), VNextError> {
    if component.role != WeightComponentRole::Values
        || component.encoding
            != (WeightEncoding::Dense {
                element_type: ElementType::Bf16,
            })
        || component.dimensions.len() != 2
        || component.dimensions.contains(&0)
        || component.external_names.len() != 1
        || component.physical_bytes()? > isize::MAX as u64
    {
        return Err(invalid(
            "source requires one exact contiguous BF16 matrix per component",
        ));
    }
    Ok(())
}

/// Every derived component carries synthetic, unique names, while the trusted
/// source map retains the complete actual ordered checkpoint component IDs.
pub(super) fn derive_group(
    sources: &[&WeightComponentSpec],
) -> Result<Vec<WeightComponentSpec>, VNextError> {
    if sources.is_empty() || sources.len() > 32 {
        return Err(invalid("invalid source group length"));
    }
    let mut unique = BTreeSet::new();
    let width = sources[0].dimensions.get(1).copied();
    for source in sources {
        validate_source(source)?;
        if !unique.insert(source.id.clone()) || source.dimensions.get(1).copied() != width {
            return Err(invalid(
                "ordered source group repeats components or has different input widths",
            ));
        }
    }
    let encoded = serde_json::to_vec(sources).map_err(|e| invalid(e.to_string()))?;
    let group = fingerprint(&[b"bf16-fp8-component-group-v1", &encoded]);
    let component =
        |suffix: &str, dimensions, role, encoding| -> Result<WeightComponentSpec, VNextError> {
            let identity = format!("component.fp8.{group}.{suffix}");
            Ok(WeightComponentSpec {
                id: WeightId::new(&identity)?,
                role,
                external_names: (0..sources.len())
                    .map(|i| format!("{identity}.source.{i}"))
                    .collect(),
                dimensions,
                encoding,
                required: sources.iter().any(|source| source.required),
            })
        };
    let mut outputs = sources
        .iter()
        .enumerate()
        .map(|(index, source)| {
            component(
                &format!("values.{index}"),
                source.dimensions.clone(),
                WeightComponentRole::PackedValues,
                WeightEncoding::Quantized(QuantizationSpec::fp8_e4m3fn_whole_tensor_f32()),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    // Logical weight trees own distinct physical leaves. The group shares the
    // numerical scale, while each matrix stores its own identical F32 scalar.
    for index in 0..sources.len() {
        outputs.push(component(
            &format!("scale.{index}"),
            vec![1],
            WeightComponentRole::Scales,
            WeightEncoding::Dense {
                element_type: ElementType::F32,
            },
        )?);
    }
    Ok(outputs)
}
