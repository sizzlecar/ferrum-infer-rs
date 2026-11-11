use super::*;
struct Emitter {
    nodes: Vec<ProgramNode>,
    specs: BTreeMap<ProgramValueId, ProgramTensorSpec>,
}
fn value(name: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.p_h3_image.vae.encoder_vertical.{name}"))
}
impl Emitter {
    fn emit(
        &mut self,
        name: &str,
        kind: F32CnnOp,
        inputs: Vec<ProgramValueId>,
        shape: Vec<u64>,
        attributes: BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<ProgramValueId, VNextError> {
        let output = if name == "downsample" {
            ProgramValueId::new(PH3ImageVaeEncoderVerticalFamily::OUTPUT_ID)?
        } else {
            value(name)?
        };
        let tensor = config::tensor(shape);
        let input_specs = inputs
            .iter()
            .map(|v| {
                let t = &self.specs[v];
                ResolvedTensorSpec::new(t.dimensions.clone(), t.element_type, t.layout.clone())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output_spec = ResolvedTensorSpec::new(
            tensor.dimensions.clone(),
            tensor.element_type,
            tensor.layout.clone(),
        )?;
        F32CnnGeometry::new(
            kind,
            &input_specs.iter().collect::<Vec<_>>(),
            &output_spec,
            &attributes,
        )?;
        self.nodes.push(ProgramNode {
            id: NodeId::new(format!("node.p_h3_image.vae.encoder_vertical.{name}"))?,
            operation_id: OperationId::new(kind.operation_id())?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs,
            outputs: vec![output.clone()],
            attributes,
        });
        self.specs.insert(output.clone(), tensor);
        Ok(output)
    }
    fn conv(
        &mut self,
        name: &str,
        module: &str,
        input: ProgramValueId,
        stride: u64,
        padding: [i64; 4],
    ) -> Result<ProgramValueId, VNextError> {
        let x = &self.specs[&input];
        let weight = weights::value(&format!("{module}.weight"))?;
        let w = &self.specs[&weight];
        let height = (x.dimensions[2] + padding[0] as u64 + padding[1] as u64 - w.dimensions[2])
            / stride
            + 1;
        let width = (x.dimensions[3] + padding[2] as u64 + padding[3] as u64 - w.dimensions[3])
            / stride
            + 1;
        let shape = vec![x.dimensions[0], w.dimensions[0], height, width];
        self.emit(
            name,
            F32CnnOp::Conv2d,
            vec![input, weight, weights::value(&format!("{module}.bias"))?],
            shape,
            BTreeMap::from([
                (
                    AttributeId::new("padding")?,
                    SemanticValue::Integers(padding.to_vec()),
                ),
                (
                    AttributeId::new("stride")?,
                    SemanticValue::Integers(vec![stride as i64; 2]),
                ),
                (
                    AttributeId::new("dilation")?,
                    SemanticValue::Integers(vec![1; 2]),
                ),
                (AttributeId::new("groups")?, SemanticValue::Unsigned(1)),
                (
                    AttributeId::new("padding_mode")?,
                    SemanticValue::Text(Conv2dF32PaddingMode::Reflect.as_str().into()),
                ),
            ]),
        )
    }
    fn norm(
        &mut self,
        name: &str,
        module: &str,
        input: ProgramValueId,
        a: &PH3ImageVaeEncoderArchitecture,
    ) -> Result<ProgramValueId, VNextError> {
        let shape = self.specs[&input].dimensions.clone();
        self.emit(
            name,
            F32CnnOp::GroupNormSilu,
            vec![
                input,
                weights::value(&format!("{module}.weight"))?,
                weights::value(&format!("{module}.bias"))?,
            ],
            shape,
            BTreeMap::from([
                (
                    AttributeId::new("groups")?,
                    SemanticValue::Unsigned(a.norm_groups()),
                ),
                (
                    AttributeId::new("epsilon")?,
                    SemanticValue::Rational(a.epsilon()),
                ),
            ]),
        )
    }
}
fn emitted(
    c: &PH3ImageVaeEncoderVerticalConfig,
    family: &ModelFamilyId,
) -> Result<(ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>), VNextError> {
    c.validate()?;
    let refs = weights::references(c)?;
    let mut e = Emitter {
        nodes: vec![],
        specs: c.input_tensor_specs()?,
    };
    e.specs
        .extend(refs.iter().map(|w| (w.value_id.clone(), w.tensor.clone())));
    let stem = e.conv(
        "stem",
        "encoder.conv_in",
        ProgramValueId::new(PH3ImageVaeEncoderVerticalFamily::INPUT_ID)?,
        1,
        [1; 4],
    )?;
    let prefix = format!("encoder.down.{}.block.0", c.block_level());
    let norm1 = e.norm(
        "norm1",
        &format!("{prefix}.norm1"),
        stem.clone(),
        c.architecture(),
    )?;
    let conv1 = e.conv("conv1", &format!("{prefix}.conv1"), norm1, 1, [1; 4])?;
    let shortcut = e.conv(
        "shortcut",
        &format!("{prefix}.nin_shortcut"),
        stem,
        1,
        [0; 4],
    )?;
    let norm2 = e.norm("norm2", &format!("{prefix}.norm2"), conv1, c.architecture())?;
    let conv2 = e.conv("conv2", &format!("{prefix}.conv2"), norm2, 1, [1; 4])?;
    let shape = e.specs[&conv2].dimensions.clone();
    let residual = e.emit(
        "residual",
        F32CnnOp::ResidualAdd,
        vec![conv2, shortcut],
        shape,
        BTreeMap::new(),
    )?;
    e.conv(
        "downsample",
        &format!("encoder.down.{}.downsample.conv", c.block_level()),
        residual,
        2,
        [0, 1, 0, 1],
    )?;
    for w in &refs {
        e.specs.remove(&w.value_id);
    }
    let graph = ModelProgram::new(
        family.clone(),
        vec![ProgramValueId::new(
            PH3ImageVaeEncoderVerticalFamily::INPUT_ID,
        )?],
        vec![ProgramBlock {
            id: "block.p_h3_image.vae.encoder_vertical".into(),
            nodes: e.nodes,
        }],
        vec![],
        refs,
        vec![ProgramValueId::new(
            PH3ImageVaeEncoderVerticalFamily::OUTPUT_ID,
        )?],
    )?
    .with_input_tensor_specs(c.input_tensor_specs()?)?;
    Ok((graph, e.specs))
}
pub(super) fn build(
    c: &PH3ImageVaeEncoderVerticalConfig,
    family: &ModelFamilyId,
) -> Result<ModelProgram, VNextError> {
    Ok(emitted(c, family)?.0)
}
pub(super) fn compile_options(
    c: &PH3ImageVaeEncoderVerticalConfig,
) -> Result<ProgramPlanCompileOptions, VNextError> {
    ProgramPlanCompileOptions::new(
        emitted(
            c,
            &ModelFamilyId::new(PH3ImageVaeEncoderVerticalFamily::FAMILY_ID)?,
        )?
        .1,
    )
}
