//! Exhaustive typed identifier relocation, following the existing Qwen block
//! composition pattern. IDs are opaque; arithmetic and attributes are copied
//! unchanged, and an unknown source port never survives implicitly.
use super::*;
pub(super) fn value(layer: u64, source: &ProgramValueId) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!(
        "value.p_h3_image.curve_stack.layer{layer}.{source}"
    ))
}
pub(super) fn weight(layer: u64, source: &WeightId) -> Result<WeightId, VNextError> {
    WeightId::new(format!(
        "weight.p_h3_image.curve_stack.layer{layer}.{source}"
    ))
}
pub(super) fn node(layer: u64, source: &NodeId) -> Result<NodeId, VNextError> {
    NodeId::new(format!("node.p_h3_image.curve_stack.layer{layer}.{source}"))
}
pub(super) fn output(layer: u64, layers: u64) -> Result<ProgramValueId, VNextError> {
    if layers == 0 || layer >= layers {
        return Err(invalid("layer output is outside the declared stack"));
    }
    if layer + 1 == layers {
        ProgramValueId::new(PH3ImageCurveStackFamily::OUTPUT_ID)
    } else {
        value(
            layer,
            &ProgramValueId::new(PH3ImageCurveBlockFamily::OUTPUT_ID)?,
        )
    }
}
pub(super) fn external(layer: u64, name: &str) -> Result<String, VNextError> {
    let suffix = name
        .strip_prefix("blocks.0.")
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            invalid("source component is outside the canonical block-zero checkpoint scope")
        })?;
    Ok(format!("blocks.{layer}.{suffix}"))
}
pub(super) fn schema(layer: u64, source: &WeightSchema) -> Result<WeightSchema, VNextError> {
    let components = source
        .components
        .iter()
        .map(|c| {
            let mut c = c.clone();
            c.id = weight(layer, &c.id)?;
            c.external_names = c
                .external_names
                .iter()
                .map(|n| external(layer, n))
                .collect::<Result<_, _>>()?;
            Ok(c)
        })
        .collect::<Result<Vec<_>, VNextError>>()?;
    let tensors = source
        .tensors
        .iter()
        .map(|t| {
            let mut t = t.clone();
            t.id = weight(layer, &t.id)?;
            t.physical_layout = match &t.physical_layout {
                PhysicalWeightLayout::Dense { component_id } => PhysicalWeightLayout::Dense {
                    component_id: weight(layer, component_id)?,
                },
                _ => {
                    return Err(invalid(
                        "block template contains an unsupported non-Dense physical port",
                    ))
                }
            };
            Ok(t)
        })
        .collect::<Result<Vec<_>, VNextError>>()?;
    Ok(WeightSchema {
        format_id: source.format_id.clone(),
        layout_id: WeightLayoutId::new(
            "weight-layout.p_h3_image.curve_stack.canonical_convrot_f32.v1",
        )?,
        version: source.version,
        components,
        tensors,
    })
}
pub(super) struct Layer {
    pub nodes: Vec<ProgramNode>,
    pub weights: Vec<WeightReference>,
    pub specs: BTreeMap<ProgramValueId, ProgramTensorSpec>,
    pub boundaries: BTreeMap<ProgramValueId, ElementType>,
}
pub(super) fn layer(
    c: &PH3ImageCurveStackConfig,
    layer: u64,
    source: &compose::Template,
) -> Result<Layer, VNextError> {
    if !source.program.states().is_empty()
        || !source.profile.states.is_empty()
        || !source.profile.kv_storage.is_empty()
    {
        return Err(invalid(
            "Curve block composition requires a stateless template",
        ));
    }
    let mut map = BTreeMap::new();
    for input in source.program.inputs() {
        let destination = match input.as_str() {
            PH3ImageCurveBlockFamily::INPUT_ID => {
                if layer == 0 {
                    ProgramValueId::new(PH3ImageCurveStackFamily::INPUT_ID)?
                } else {
                    output(layer - 1, c.layers)?
                }
            }
            PH3ImageCurveBlockFamily::BASIS_ID => {
                ProgramValueId::new(PH3ImageCurveStackFamily::BASIS_ID)?
            }
            PH3ImageCurveBlockFamily::COSINE_ID => {
                ProgramValueId::new(PH3ImageCurveStackFamily::COSINE_ID)?
            }
            PH3ImageCurveBlockFamily::SINE_ID => {
                ProgramValueId::new(PH3ImageCurveStackFamily::SINE_ID)?
            }
            _ => return Err(invalid("block template introduced an unmapped input")),
        };
        if map.insert(input.clone(), destination).is_some() {
            return Err(invalid("duplicate template input"));
        }
    }
    let weights = source
        .program
        .weights()
        .iter()
        .map(|w| {
            let target = value(layer, &w.value_id)?;
            if map.insert(w.value_id.clone(), target.clone()).is_some() {
                return Err(invalid("weight conflicts with an input"));
            }
            Ok(WeightReference {
                weight_id: weight(layer, &w.weight_id)?,
                value_id: target,
                tensor: w.tensor.clone(),
            })
        })
        .collect::<Result<Vec<_>, VNextError>>()?;
    for n in source.program.blocks().iter().flat_map(|b| &b.nodes) {
        for produced in &n.outputs {
            let target = if produced.as_str() == PH3ImageCurveBlockFamily::OUTPUT_ID {
                output(layer, c.layers)?
            } else {
                value(layer, produced)?
            };
            if map.insert(produced.clone(), target).is_some() {
                return Err(invalid("template output has another producer"));
            }
        }
    }
    let lookup = |id: &ProgramValueId| {
        map.get(id)
            .cloned()
            .ok_or_else(|| invalid("unmapped source activation"))
    };
    let nodes = source
        .program
        .blocks()
        .iter()
        .flat_map(|b| &b.nodes)
        .map(|n| {
            if n.work != ProgramNodeWorkSpec::Fixed {
                return Err(invalid("Curve block introduced unsupported token work"));
            }
            Ok(ProgramNode {
                id: node(layer, &n.id)?,
                operation_id: n.operation_id.clone(),
                required_version: n.required_version,
                work: n.work.clone(),
                inputs: n.inputs.iter().map(&lookup).collect::<Result<_, _>>()?,
                outputs: n.outputs.iter().map(&lookup).collect::<Result<_, _>>()?,
                attributes: n.attributes.clone(),
            })
        })
        .collect::<Result<_, VNextError>>()?;
    let specs = source
        .options
        .tensor_specs()
        .iter()
        .map(|(id, s)| Ok((lookup(id)?, s.clone())))
        .collect::<Result<_, VNextError>>()?;
    let boundaries = source
        .profile
        .boundaries
        .iter()
        .map(|(id, &dtype)| Ok((lookup(id)?, dtype)))
        .collect::<Result<_, VNextError>>()?;
    Ok(Layer {
        nodes,
        weights,
        specs,
        boundaries,
    })
}
