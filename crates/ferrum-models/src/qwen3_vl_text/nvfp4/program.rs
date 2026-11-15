use super::*;
use config::{tensor, value};

/// Reuse the existing decoder graph and its complete output/mask/RoPE
/// semantics. Only declared dense projection sites are replaced.
pub(super) fn build(
    c: &Qwen3VLTextNvfp4Config,
    family: &ModelFamilyId,
) -> Result<ModelProgram, VNextError> {
    Ok(build_with_options(c, family)?.0)
}
pub(super) fn build_with_options(
    c: &Qwen3VLTextNvfp4Config,
    family: &ModelFamilyId,
) -> Result<(ModelProgram, ProgramPlanCompileOptions), VNextError> {
    c.validate()?;
    let base = super::super::program::build(c.text(), family)?;
    let mut options = c.text().compile_options()?;
    let selected: BTreeMap<_, _> = c
        .projections
        .iter()
        .map(|p| {
            Ok((
                super::super::weights::value_id(&format!("{}.weight", p.module))?,
                p,
            ))
        })
        .collect::<Result<_, VNextError>>()?;
    let mut blocks = base.blocks().to_vec();
    let mut consumed = BTreeSet::new();
    for block in &mut blocks {
        let mut nodes = Vec::new();
        let mut quantized =
            BTreeMap::<ProgramValueId, (u64, ProgramValueId, ProgramValueId)>::new();
        for n in &block.nodes {
            let p = n
                .inputs
                .iter()
                .enumerate()
                .find_map(|(i, x)| selected.get(x).map(|p| (i, *p)));
            let Some((index, p)) = p else {
                let mut node = n.clone();
                if node.operation_id.as_str() == STRUCTURED_GQA_ATTENTION_BF16_OPERATION_ID
                    && c.attention == config::Qwen3VLTextNvfp4Attention::VendorFusedCausal
                {
                    node.operation_id =
                        OperationId::new(STRUCTURED_GQA_ATTENTION_FUSED_BF16_OPERATION_ID)?;
                }
                nodes.push(node);
                continue;
            };
            if index != 1
                || n.operation_id.as_str() != DENSE_LINEAR_BF16_OPERATION_ID
                || n.inputs.len() != 2
                || n.outputs.len() != 1
            {
                return Err(invalid(
                    "selected projection must be an ordinary BF16 dense weight-port consumer",
                ));
            }
            consumed.insert(p.module.clone());
            // A single immutable activation, unit SF and arithmetic profile
            // define one quantization domain. Q/K/V and gate/up share their
            // actual producer outputs instead of allocating duplicate work.
            let (packed_a, scale_a) =
                if let Some((columns, packed, scales)) = quantized.get(&n.inputs[0]) {
                    if *columns != p.columns {
                        return Err(invalid(
                            "shared activation has conflicting projection widths",
                        ));
                    }
                    (packed.clone(), scales.clone())
                } else {
                    let packed_a = ProgramValueId::new(format!("{}.nvfp4_packed", n.outputs[0]))?;
                    let scale_a = ProgramValueId::new(format!("{}.nvfp4_scales", n.outputs[0]))?;
                    let layout = Nvfp4MatrixLayout::new(c.text().layout().rows(), p.columns)?;
                    for (id, dimensions) in [
                        (packed_a.clone(), layout.packed_shape().to_vec()),
                        (scale_a.clone(), layout.scale_shape().to_vec()),
                    ] {
                        options.insert_tensor_spec(
                            id,
                            ProgramTensorSpec {
                                dimensions,
                                element_type: ElementType::U8,
                                layout: ResolvedTensorLayout::Contiguous,
                            },
                        )?;
                    }
                    nodes.push(ProgramNode {
                        id: NodeId::new(format!("{}.nvfp4_quantize", n.id))?,
                        operation_id: OperationId::new(NVFP4_QUANTIZE_BF16_OPERATION_ID)?,
                        required_version: ContractVersion::new(1, 0),
                        work: ProgramNodeWorkSpec::Fixed,
                        inputs: vec![n.inputs[0].clone(), value(&config::unit_component()?)?],
                        outputs: vec![packed_a.clone(), scale_a.clone()],
                        attributes: Nvfp4QuantizeProfile::CudaRcpApproxFtzV1.attributes()?,
                    });
                    quantized.insert(
                        n.inputs[0].clone(),
                        (p.columns, packed_a.clone(), scale_a.clone()),
                    );
                    (packed_a, scale_a)
                };
            nodes.push(ProgramNode {
                id: n.id.clone(),
                operation_id: OperationId::new(NVFP4_MATMUL_BF16_OPERATION_ID)?,
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::Fixed,
                inputs: vec![
                    packed_a,
                    value(&p.packed()?)?,
                    scale_a,
                    value(&p.scales()?)?,
                    value(&p.alpha_component()?)?,
                ],
                outputs: n.outputs.clone(),
                attributes: BTreeMap::new(),
            });
        }
        block.nodes = nodes;
    }
    if consumed.len() != c.projections.len() {
        return Err(invalid("projection selection contains unconsumed weights"));
    }
    let mut weights: Vec<_> = base
        .weights()
        .iter()
        .filter(|w| !selected.contains_key(&w.value_id))
        .cloned()
        .collect();
    for p in &c.projections {
        for (component, physical) in [
            (p.packed()?, true),
            (p.scales()?, true),
            (p.alpha_component()?, false),
        ] {
            weights.push(reference(&component, physical)?);
        }
    }
    weights.push(reference(&config::unit_component()?, false)?);
    let program = ModelProgram::new(
        family.clone(),
        base.inputs().to_vec(),
        blocks,
        base.states().to_vec(),
        weights,
        base.outputs().to_vec(),
    )?
    .with_input_tensor_specs(base.input_tensor_specs().clone())?;
    Ok((program, options))
}
fn reference(c: &WeightComponentSpec, physical: bool) -> Result<WeightReference, VNextError> {
    let t = tensor(c, physical)?;
    Ok(WeightReference {
        weight_id: t.id,
        value_id: value(c)?,
        tensor: ProgramTensorSpec {
            dimensions: t.dimensions,
            element_type: t.logical_element_type,
            layout: ResolvedTensorLayout::Contiguous,
        },
    })
}
