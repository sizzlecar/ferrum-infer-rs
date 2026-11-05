use super::*;
use config::tensor;
pub(super) const EMBEDDED: &str = "value.p_h3_image.vae.decoder.embedded";
pub(super) fn entries(
    c: &PH3ImageVaeDecoderExecutionConfig,
) -> Result<Vec<(&'static str, Vec<u64>, Vec<u64>)>, VNextError> {
    let a = c.architecture();
    let h = a.hidden_size()?;
    let z = a.latent_channels();
    let p = a.patch_output_channels()?;
    Ok(vec![
        ("post_quant_conv.weight", vec![z, z, 1, 1, 1], vec![z, z]),
        ("post_quant_conv.bias", vec![z], vec![z]),
        ("decoder.x_embedder.weight", vec![h, z], vec![h, z]),
        ("decoder.x_embedder.bias", vec![h], vec![h]),
        (
            "decoder.register_tokens",
            vec![1, a.register_tokens(), h],
            vec![a.register_tokens(), h],
        ),
        ("decoder.norm_out.weight", vec![h], vec![h]),
        ("decoder.norm_out.bias", vec![h], vec![h]),
        ("decoder.proj_out.weight", vec![p, h], vec![p, h]),
        ("decoder.proj_out.bias", vec![p], vec![p]),
    ])
}
fn value(name: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.p_h3_image.vae.decoder.{name}"))
}
fn weight(name: &str) -> Result<WeightId, VNextError> {
    WeightId::new(format!("weight.p_h3_image.vae.decoder.{name}"))
}
pub(super) fn weight_value(name: &str) -> Result<ProgramValueId, VNextError> {
    value(&format!("weight.{name}"))
}
pub(super) fn schema(
    c: &PH3ImageVaeDecoderExecutionConfig,
) -> Result<
    (
        Vec<WeightComponentSpec>,
        Vec<WeightTensorSpec>,
        Vec<WeightReference>,
    ),
    VNextError,
> {
    let mut components = Vec::new();
    let mut tensors = Vec::new();
    let mut references = Vec::new();
    for (name, _, dimensions) in entries(c)? {
        let component_id = WeightId::new(format!("component.p_h3_image.vae.decoder.{name}"))?;
        let id = weight(name)?;
        components.push(WeightComponentSpec {
            id: component_id.clone(),
            role: WeightComponentRole::Values,
            external_names: vec![name.into()],
            dimensions: dimensions.clone(),
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F32,
            },
            required: true,
        });
        tensors.push(WeightTensorSpec {
            id: id.clone(),
            dimensions: dimensions.clone(),
            logical_element_type: ElementType::F32,
            physical_layout: PhysicalWeightLayout::Dense { component_id },
            required: true,
        });
        references.push(WeightReference {
            weight_id: id,
            value_id: weight_value(name)?,
            tensor: tensor(dimensions),
        });
    }
    Ok((components, tensors, references))
}
fn node(
    name: &str,
    operation: &str,
    inputs: Vec<ProgramValueId>,
    output: ProgramValueId,
    attributes: BTreeMap<AttributeId, SemanticValue>,
) -> Result<ProgramNode, VNextError> {
    Ok(ProgramNode {
        id: NodeId::new(format!("node.p_h3_image.vae.decoder.{name}"))?,
        operation_id: OperationId::new(operation)?,
        required_version: ContractVersion::new(1, 0),
        work: ProgramNodeWorkSpec::Fixed,
        inputs,
        outputs: vec![output],
        attributes,
    })
}
fn affine(
    name: &str,
    module: &str,
    input: ProgramValueId,
    output: ProgramValueId,
    k: u64,
    n: u64,
) -> Result<ProgramNode, VNextError> {
    node(
        name,
        AFFINE_LINEAR_F32_OPERATION_ID,
        vec![
            input,
            weight_value(&format!("{module}.weight"))?,
            weight_value(&format!("{module}.bias"))?,
        ],
        output,
        BTreeMap::from([
            (AttributeId::new("in_features")?, SemanticValue::Unsigned(k)),
            (
                AttributeId::new("out_features")?,
                SemanticValue::Unsigned(n),
            ),
        ]),
    )
}
pub(super) fn prefix(
    c: &PH3ImageVaeDecoderExecutionConfig,
) -> Result<Vec<ProgramNode>, VNextError> {
    let a = c.architecture();
    let z = a.latent_channels();
    let h = a.hidden_size()?;
    let post = value("post_quant")?;
    let patch = value("patch_embeddings")?;
    Ok(vec![
        affine(
            "post_quant",
            "post_quant_conv",
            ProgramValueId::new(PH3ImageVaeDecoderFamily::INPUT_ID)?,
            post.clone(),
            z,
            z,
        )?,
        affine("embed", "decoder.x_embedder", post, patch.clone(), z, h)?,
        node(
            "append_suffix",
            BATCHED_SUFFIX_APPEND_F32_OPERATION_ID,
            vec![patch, weight_value("decoder.register_tokens")?],
            ProgramValueId::new(EMBEDDED)?,
            BTreeMap::from([
                (
                    AttributeId::new("batch")?,
                    SemanticValue::Unsigned(c.batch()),
                ),
                (AttributeId::new("zero_rows")?, SemanticValue::Unsigned(1)),
            ]),
        )?,
    ])
}
pub(super) fn suffix(
    c: &PH3ImageVaeDecoderExecutionConfig,
    input: ProgramValueId,
) -> Result<Vec<ProgramNode>, VNextError> {
    let a = c.architecture();
    let normalized = value("normalized")?;
    Ok(vec![
        node(
            "norm_out",
            LAYER_NORM_F32_OPERATION_ID,
            vec![
                input,
                weight_value("decoder.norm_out.weight")?,
                weight_value("decoder.norm_out.bias")?,
            ],
            normalized.clone(),
            BTreeMap::from([(
                AttributeId::new("epsilon")?,
                SemanticValue::Rational(a.norm_epsilon()),
            )]),
        )?,
        affine(
            "proj_out",
            "decoder.proj_out",
            normalized,
            ProgramValueId::new(PH3ImageVaeDecoderFamily::OUTPUT_ID)?,
            a.hidden_size()?,
            a.patch_output_channels()?,
        )?,
    ])
}
/// Follow actual prefix/suffix nodes; every independent output dimension is
/// inserted explicitly, including the broadcast suffix's changed row count.
pub(super) fn specs(
    c: &PH3ImageVaeDecoderExecutionConfig,
    nodes: &[ProgramNode],
    specs: &mut BTreeMap<ProgramValueId, ProgramTensorSpec>,
) -> Result<(), VNextError> {
    for n in nodes {
        let mut out = specs
            .get(&n.inputs[0])
            .cloned()
            .ok_or_else(|| invalid("outer source tensor missing"))?;
        match n.operation_id.as_str() {
            AFFINE_LINEAR_F32_OPERATION_ID => {
                let Some(SemanticValue::Unsigned(columns)) =
                    n.attributes.get(&AttributeId::new("out_features")?)
                else {
                    return Err(invalid("affine output shape missing"));
                };
                out.dimensions[1] = *columns;
            }
            BATCHED_SUFFIX_APPEND_F32_OPERATION_ID => out.dimensions[0] = c.rows()?,
            LAYER_NORM_F32_OPERATION_ID => {}
            _ => {
                return Err(invalid(
                    "new outer operation needs explicit shape propagation",
                ))
            }
        }
        out.validate("VAE outer output")?;
        for id in &n.outputs {
            specs.insert(id.clone(), out.clone());
        }
    }
    Ok(())
}
