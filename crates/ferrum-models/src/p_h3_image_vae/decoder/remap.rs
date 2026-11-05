//! Relocate the actual block graph, retaining operation attributes/arithmetic.
use super::*;
fn value(layer: u64, id: &ProgramValueId) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.p_h3_image.vae.decoder.layer{layer}.{id}"))
}
fn weight(layer: u64, id: &WeightId) -> Result<WeightId, VNextError> {
    WeightId::new(format!("weight.p_h3_image.vae.decoder.layer{layer}.{id}"))
}
pub(super) fn output(layer: u64) -> Result<ProgramValueId, VNextError> {
    value(
        layer,
        &ProgramValueId::new(PH3ImageVaeDecoderBlockFamily::OUTPUT_ID)?,
    )
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
                .map(|name| {
                    let tail = name
                        .strip_prefix("decoder.transformer_blocks.0.")
                        .filter(|s| !s.is_empty())
                        .ok_or_else(|| {
                            invalid("block template source escaped declared layer-zero scope")
                        })?;
                    Ok(format!("decoder.transformer_blocks.{layer}.{tail}"))
                })
                .collect::<Result<_, VNextError>>()?;
            Ok(c)
        })
        .collect::<Result<_, VNextError>>()?;
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
                        "block template introduced unsupported non-Dense physical port",
                    ))
                }
            };
            Ok(t)
        })
        .collect::<Result<_, VNextError>>()?;
    Ok(WeightSchema {
        format_id: source.format_id.clone(),
        layout_id: source.layout_id.clone(),
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
pub(super) fn layer(layer: u64, t: &compose::Template) -> Result<Layer, VNextError> {
    if !t.program.states().is_empty()
        || !t.profile.states.is_empty()
        || !t.profile.kv_storage.is_empty()
    {
        return Err(invalid("requires a stateless decoder block template"));
    }
    let mut map = BTreeMap::new();
    for input in t.program.inputs() {
        let out = match input.as_str() {
            PH3ImageVaeDecoderBlockFamily::INPUT_ID => {
                if layer == 0 {
                    ProgramValueId::new(outer::EMBEDDED)?
                } else {
                    output(layer - 1)?
                }
            }
            PH3ImageVaeDecoderBlockFamily::COSINE_ID => {
                ProgramValueId::new(PH3ImageVaeDecoderFamily::COSINE_ID)?
            }
            PH3ImageVaeDecoderBlockFamily::SINE_ID => {
                ProgramValueId::new(PH3ImageVaeDecoderFamily::SINE_ID)?
            }
            _ => return Err(invalid("unmapped decoder block input")),
        };
        if map.insert(input.clone(), out).is_some() {
            return Err(invalid("duplicate block input"));
        }
    }
    let weights = t
        .program
        .weights()
        .iter()
        .map(|w| {
            let target = value(layer, &w.value_id)?;
            if map.insert(w.value_id.clone(), target.clone()).is_some() {
                return Err(invalid("block weight conflicts with input"));
            }
            Ok(WeightReference {
                weight_id: weight(layer, &w.weight_id)?,
                value_id: target,
                tensor: w.tensor.clone(),
            })
        })
        .collect::<Result<_, VNextError>>()?;
    for n in t.program.blocks().iter().flat_map(|b| &b.nodes) {
        for id in &n.outputs {
            if map.insert(id.clone(), value(layer, id)?).is_some() {
                return Err(invalid("duplicate block producer"));
            }
        }
    }
    let lookup = |id: &ProgramValueId| {
        map.get(id)
            .cloned()
            .ok_or_else(|| invalid("unmapped block activation"))
    };
    let nodes = t
        .program
        .blocks()
        .iter()
        .flat_map(|b| &b.nodes)
        .map(|n| {
            if n.work != ProgramNodeWorkSpec::Fixed {
                return Err(invalid("block introduced token work"));
            }
            Ok(ProgramNode {
                id: NodeId::new(format!("node.p_h3_image.vae.decoder.layer{layer}.{}", n.id))?,
                operation_id: n.operation_id.clone(),
                required_version: n.required_version,
                work: n.work.clone(),
                inputs: n.inputs.iter().map(&lookup).collect::<Result<_, _>>()?,
                outputs: n.outputs.iter().map(&lookup).collect::<Result<_, _>>()?,
                attributes: n.attributes.clone(),
            })
        })
        .collect::<Result<_, VNextError>>()?;
    let specs = t
        .options
        .tensor_specs()
        .iter()
        .map(|(id, s)| Ok((lookup(id)?, s.clone())))
        .collect::<Result<_, VNextError>>()?;
    let boundaries = t
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
