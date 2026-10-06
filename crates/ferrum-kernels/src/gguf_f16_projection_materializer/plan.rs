use super::*;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct GgufF16ProjectionWeightInventoryV1 {
    pub logical_weight: WeightId,
    pub consumers: Vec<GgufF16ProjectionConsumerV1>,
    pub logical_dimensions: Vec<u64>,
    pub source_components: Vec<WeightId>,
    pub execution_components: Vec<WeightId>,
    pub conversion: &'static str,
}
#[derive(Debug, Clone, Serialize)]
pub struct GgufF16ProjectionConsumerV1 {
    pub operation: OperationId,
    pub input_ordinal: u32,
    pub role: Option<GgufF16ProjectionRoleV1>,
}
#[derive(Debug, Clone, Serialize)]
pub struct GgufF16ProjectionInventoryV1 {
    pub source_schema_fingerprint: String,
    pub execution_schema_fingerprint: String,
    pub weights: Vec<GgufF16ProjectionWeightInventoryV1>,
    /// Unique components referenced by actual program weights, not all GGUF tensors.
    pub source_consumed_bytes: u64,
    pub converted_f16_bytes: u64,
    pub retained_consumed_bytes: u64,
    pub unique_consumed_execution_bytes: u64,
    /// Alignment and placement high-water marks belong to the compiled resource
    /// plan. These byte sums are not device allocation or peak-memory evidence.
    pub includes_placement_alignment: bool,
}
pub(crate) struct Prepared {
    pub schema: WeightSchema,
    pub sources: BTreeMap<WeightId, Vec<WeightId>>,
    pub inventory: GgufF16ProjectionInventoryV1,
}

pub(super) fn prepare(family: &PreparedModelFamily) -> Result<Prepared, VNextError> {
    prepare_program(family.weight_schema(), family.program())
}

pub(super) fn prepare_program(
    original: &WeightSchema,
    program: &ModelProgram,
) -> Result<Prepared, VNextError> {
    prepare_program_with_roles(original, program, gguf_f16_projection_role_v1)
}

/// Reuse the original consumer/source/storage checks. The legacy entrypoint
/// retains its original resolver, schema IDs and materialization behavior.
pub(crate) fn prepare_program_with_roles(
    original: &WeightSchema,
    program: &ModelProgram,
    role_for: fn(&OperationId, u32) -> Option<GgufF16ProjectionRoleV1>,
) -> Result<Prepared, VNextError> {
    original.validate(program.family_id())?;
    if original.format_id.as_str() != SOURCE_FORMAT {
        return Err(invalid(
            "GGUF RN-F16 materializer requires the declared native GGUF source schema",
        ));
    }
    let by_value = program
        .weights()
        .iter()
        .map(|r| (&r.value_id, &r.weight_id))
        .collect::<BTreeMap<_, _>>();
    let mut consumers: BTreeMap<WeightId, Vec<GgufF16ProjectionConsumerV1>> = BTreeMap::new();
    for node in program.blocks().iter().flat_map(|b| &b.nodes) {
        for (ordinal, value) in node.inputs.iter().enumerate() {
            let ordinal =
                u32::try_from(ordinal).map_err(|_| invalid("projection input ordinal overflow"))?;
            let role = role_for(&node.operation_id, ordinal);
            let weight = by_value.get(value);
            if role.is_some() && (node.required_version != VERSION || weight.is_none()) {
                return Err(invalid(
                    "GGUF RN-F16 operation version or typed weight input differs",
                ));
            }
            if let Some(weight) = weight {
                consumers
                    .entry((*weight).clone())
                    .or_default()
                    .push(GgufF16ProjectionConsumerV1 {
                        operation: node.operation_id.clone(),
                        input_ordinal: ordinal,
                        role,
                    });
            }
        }
    }
    let originals = original
        .components
        .iter()
        .map(|c| (&c.id, c))
        .collect::<BTreeMap<_, _>>();
    let mut derived = BTreeMap::<WeightId, (WeightComponentSpec, Vec<WeightId>)>::new();
    let mut schema = original.clone();
    let mut authorized = BTreeSet::new();
    for (weight, uses) in &consumers {
        let any = uses.iter().any(|u| u.role.is_some());
        if any && uses.iter().any(|u| u.role.is_none()) {
            return Err(invalid(format!(
                "GGUF RN-F16 logical weight `{weight}` has an unauthorized consumer"
            )));
        }
        if !any {
            continue;
        }
        authorized.insert(weight.clone());
        let tensor = schema
            .tensors
            .iter_mut()
            .find(|t| &t.id == weight)
            .ok_or_else(|| invalid("projection program weight is absent from schema"))?;
        if tensor.logical_element_type != ElementType::F16
            || !(2..=3).contains(&tensor.dimensions.len())
        {
            return Err(invalid(
                "GGUF RN-F16 projection logical dtype/rank is unsupported",
            ));
        }
        tensor.physical_layout = rewrite(
            &tensor.physical_layout,
            &tensor.dimensions,
            &originals,
            &mut derived,
        )?;
    }
    if authorized.is_empty() {
        return Err(invalid(
            "GGUF RN-F16 requires an explicit authorized projection operation",
        ));
    }
    schema.format_id = WeightFormatId::new(GGUF_F16_PROJECTION_FORMAT_ID)?;
    schema.layout_id = WeightLayoutId::new(GGUF_F16_PROJECTION_LAYOUT_ID)?;
    schema.version = VERSION;
    schema
        .components
        .extend(derived.values().map(|(component, _)| component.clone()));
    // Keep source components when any unchanged tensor still references them
    // (e.g. tied output/embedding). Otherwise the compressed device copy is
    // absent from the execution plan, not uploaded then expanded on device.
    let mut referenced = BTreeSet::new();
    for tensor in &schema.tensors {
        referenced.extend(
            schema
                .physical_component_refs(&tensor.id)?
                .iter()
                .map(|c| c.id.clone()),
        );
    }
    schema.components.retain(|c| referenced.contains(&c.id));
    schema.components.sort_by(|a, b| a.id.cmp(&b.id));
    schema.validate(program.family_id())?;
    let sources = schema
        .components
        .iter()
        .map(|c| {
            (
                c.id.clone(),
                derived
                    .get(&c.id)
                    .map(|(_, sources)| sources.clone())
                    .unwrap_or_else(|| vec![c.id.clone()]),
            )
        })
        .collect();
    let mut source_used = BTreeMap::new();
    let mut execution_used = BTreeMap::new();
    let mut weights = Vec::new();
    for (weight, uses) in consumers {
        let tensor = original
            .tensor(&weight)
            .ok_or_else(|| invalid("consumed weight absent"))?;
        let source_components = original.physical_component_refs(&weight)?;
        let execution_components = schema.physical_component_refs(&weight)?;
        for c in &source_components {
            source_used.insert(c.id.clone(), c.physical_bytes()?);
        }
        for c in &execution_components {
            execution_used.insert(c.id.clone(), c.physical_bytes()?);
        }
        weights.push(GgufF16ProjectionWeightInventoryV1 {
            logical_weight: weight.clone(),
            consumers: uses,
            logical_dimensions: tensor.dimensions.clone(),
            source_components: source_components.iter().map(|c| c.id.clone()).collect(),
            execution_components: execution_components.iter().map(|c| c.id.clone()).collect(),
            conversion: if authorized.contains(&weight) {
                "authorized_gguf_rn_f16_projection"
            } else {
                "retained_unauthorized_consumer"
            },
        });
    }
    let sum = |values: Vec<u64>| {
        values.into_iter().try_fold(0u64, |a, b| {
            a.checked_add(b)
                .ok_or_else(|| invalid("GGUF RN-F16 inventory bytes overflow"))
        })
    };
    let converted_f16_bytes = sum(execution_used
        .iter()
        .filter(|(id, _)| derived.contains_key(*id))
        .map(|(_, b)| *b)
        .collect())?;
    let retained_consumed_bytes = sum(execution_used
        .iter()
        .filter(|(id, _)| !derived.contains_key(*id))
        .map(|(_, b)| *b)
        .collect())?;
    let inventory =
        GgufF16ProjectionInventoryV1 {
            source_schema_fingerprint: original.fingerprint()?,
            execution_schema_fingerprint: schema.fingerprint()?,
            weights,
            source_consumed_bytes: sum(source_used.values().copied().collect())?,
            converted_f16_bytes,
            retained_consumed_bytes,
            unique_consumed_execution_bytes: converted_f16_bytes
                .checked_add(retained_consumed_bytes)
                .ok_or_else(|| invalid("GGUF RN-F16 total bytes overflow"))?,
            includes_placement_alignment: false,
        };
    Ok(Prepared {
        schema,
        sources,
        inventory,
    })
}

fn strides(shape: &[u64]) -> Result<Vec<u64>, VNextError> {
    let mut stride = 1u64;
    let mut result = vec![0; shape.len()];
    for (index, dim) in shape.iter().enumerate().rev() {
        result[index] = stride;
        stride = stride
            .checked_mul(*dim)
            .ok_or_else(|| invalid("GGUF RN-F16 row-major span overflows"))?;
    }
    Ok(result)
}
fn product(shape: &[u64]) -> Result<u64, VNextError> {
    shape.iter().try_fold(1u64, |a, b| {
        a.checked_mul(*b)
            .ok_or_else(|| invalid("GGUF RN-F16 shape overflows"))
    })
}
fn validate_storage(
    binding: &PhysicalWeightComponentBinding,
    physical: &[u64],
    logical: &[u64],
) -> Result<(), VNextError> {
    if product(physical)? != product(logical)? {
        return Err(invalid("GGUF RN-F16 storage span differs"));
    }
    match &binding.storage {
        PhysicalStorageLayout::Contiguous {
            padding: PhysicalWeightPadding::Exact,
        } if physical == logical => Ok(()),
        PhysicalStorageLayout::Strided {
            strides_in_elements,
            padding: PhysicalWeightPadding::Exact,
        } if *strides_in_elements == strides(logical)? => Ok(()),
        _ => Err(invalid(
            "GGUF RN-F16 requires exact row-major physical storage",
        )),
    }
}
fn rewrite(
    layout: &PhysicalWeightLayout,
    logical: &[u64],
    originals: &BTreeMap<&WeightId, &WeightComponentSpec>,
    derived: &mut BTreeMap<WeightId, (WeightComponentSpec, Vec<WeightId>)>,
) -> Result<PhysicalWeightLayout, VNextError> {
    if let PhysicalWeightLayout::Dense { component_id } = layout {
        let source = *originals
            .get(component_id)
            .ok_or_else(|| invalid("dense projection component absent"))?;
        if source.encoding
            == (WeightEncoding::Dense {
                element_type: ElementType::F16,
            })
            && source.dimensions == logical
        {
            return Ok(layout.clone());
        }
    }
    let mut sources = Vec::new();
    ordered_sources(layout, logical, originals, &mut sources)?;
    let target = conversion::derived_component(&sources, logical)?;
    let ids = sources.iter().map(|s| s.id.clone()).collect::<Vec<_>>();
    // Registry groups by exact ordered source set. One group has one payload;
    // aliases with a different physical shape need a future explicit protocol.
    if derived
        .values()
        .any(|(component, group)| group == &ids && component != &target)
    {
        return Err(invalid(
            "GGUF RN-F16 source group has incompatible target shapes",
        ));
    }
    let id = target.id.clone();
    derived.entry(id.clone()).or_insert((target, ids));
    Ok(PhysicalWeightLayout::Dense { component_id: id })
}
fn ordered_sources<'a>(
    layout: &PhysicalWeightLayout,
    logical: &[u64],
    originals: &BTreeMap<&WeightId, &'a WeightComponentSpec>,
    output: &mut Vec<&'a WeightComponentSpec>,
) -> Result<(), VNextError> {
    match layout {
        PhysicalWeightLayout::Composite { parts } => {
            let mut parts = parts.iter().collect::<Vec<_>>();
            parts.sort_by(|a, b| a.logical_offsets.cmp(&b.logical_offsets));
            let mut cursor = 0u64;
            for part in parts {
                if part.extents.len() != logical.len()
                    || part.logical_offsets.len() != logical.len()
                    || part.logical_offsets[0] != cursor
                    || part.logical_offsets[1..].iter().any(|n| *n != 0)
                    || part.extents[1..] != logical[1..]
                {
                    return Err(invalid(
                        "GGUF RN-F16 composite is not a consecutive row-major stack",
                    ));
                }
                cursor = cursor
                    .checked_add(part.extents[0])
                    .ok_or_else(|| invalid("GGUF RN-F16 stacked rows overflow"))?;
                ordered_sources(&part.layout, &part.extents, originals, output)?;
            }
            if cursor != logical[0] {
                return Err(invalid(
                    "GGUF RN-F16 stack does not cover the whole logical projection",
                ));
            }
        }
        PhysicalWeightLayout::BlockQuantized {
            blocks,
            block_axis,
            block_padding,
        } => {
            if *block_axis as usize != logical.len() - 1
                || *block_padding != PhysicalWeightPadding::Exact
            {
                return Err(invalid("GGUF RN-F16 block axis/padding unsupported"));
            }
            let source = *originals
                .get(&blocks.component_id)
                .ok_or_else(|| invalid("GGUF RN-F16 source component absent"))?;
            let WeightEncoding::BlockQuantized(spec) = &source.encoding else {
                return Err(invalid("GGUF RN-F16 block leaf encoding differs"));
            };
            conversion::supported_format(spec)?;
            let mut shape = logical.to_vec();
            let width = shape
                .last_mut()
                .ok_or_else(|| invalid("empty projection shape"))?;
            if !width.is_multiple_of(u64::from(spec.logical_values_per_block)) {
                return Err(invalid("GGUF RN-F16 partial block"));
            }
            *width /= u64::from(spec.logical_values_per_block);
            validate_storage(blocks, &source.dimensions, &shape)?;
            output.push(source);
        }
        PhysicalWeightLayout::Dense { component_id } => {
            let source = *originals
                .get(component_id)
                .ok_or_else(|| invalid("dense projection component absent"))?;
            if source.dimensions != logical
                || source.encoding
                    != (WeightEncoding::Dense {
                        element_type: ElementType::F16,
                    })
            {
                return Err(invalid("retained dense projection must be exact F16"));
            }
            output.push(source);
        }
        PhysicalWeightLayout::Stored { component } => {
            let source = *originals
                .get(&component.component_id)
                .ok_or_else(|| invalid("stored projection component absent"))?;
            if source.encoding
                != (WeightEncoding::Dense {
                    element_type: ElementType::F16,
                })
            {
                return Err(invalid("retained stored projection must be exact F16"));
            }
            validate_storage(component, &source.dimensions, logical)?;
            output.push(source);
        }
        _ => {
            return Err(invalid(
                "GGUF RN-F16 does not authorize transformed/indexed/tiled projection layouts",
            ))
        }
    }
    Ok(())
}
