use super::*;
struct Emitter {
    nodes: Vec<ProgramNode>,
    specs: BTreeMap<ProgramValueId, ProgramTensorSpec>,
}
fn value(name: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.p_h3_image.vae.encoder.{name}"))
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
        let output = if name == "quant_conv" {
            ProgramValueId::new(PH3ImageVaeEncoderFamily::OUTPUT_ID)?
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
        let out = ResolvedTensorSpec::new(
            tensor.dimensions.clone(),
            tensor.element_type,
            tensor.layout.clone(),
        )?;
        F32CnnGeometry::new(
            kind,
            &input_specs.iter().collect::<Vec<_>>(),
            &out,
            &attributes,
        )?;
        self.nodes.push(ProgramNode {
            id: NodeId::new(format!("node.p_h3_image.vae.encoder.{name}"))?,
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
        let extent = |original: u64, front: i64, back: i64, kernel: u64| {
            original
                .checked_add(front as u64)
                .and_then(|n| n.checked_add(back as u64))
                .and_then(|n| n.checked_sub(kernel))
                .map(|n| n / stride + 1)
                .ok_or_else(|| invalid("single-frame convolution kernel exceeds padded image"))
        };
        let shape = vec![
            x.dimensions[0],
            w.dimensions[0],
            extent(x.dimensions[2], padding[0], padding[1], w.dimensions[2])?,
            extent(x.dimensions[3], padding[2], padding[3], w.dimensions[3])?,
        ];
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
    fn residual(
        &mut self,
        prefix: &str,
        input: ProgramValueId,
        output_channels: u64,
        a: &PH3ImageVaeEncoderArchitecture,
    ) -> Result<ProgramValueId, VNextError> {
        let in_channels = self.specs[&input].dimensions[1];
        let norm1 = self.norm(
            &format!("{prefix}.norm1"),
            &format!("{prefix}.norm1"),
            input.clone(),
            a,
        )?;
        let h = self.conv(
            &format!("{prefix}.conv1"),
            &format!("{prefix}.conv1"),
            norm1,
            1,
            [1; 4],
        )?;
        let shortcut = if in_channels != output_channels {
            self.conv(
                &format!("{prefix}.nin_shortcut"),
                &format!("{prefix}.nin_shortcut"),
                input,
                1,
                [0; 4],
            )?
        } else {
            input
        };
        let norm2 = self.norm(&format!("{prefix}.norm2"), &format!("{prefix}.norm2"), h, a)?;
        let h = self.conv(
            &format!("{prefix}.conv2"),
            &format!("{prefix}.conv2"),
            norm2,
            1,
            [1; 4],
        )?;
        let shape = self.specs[&h].dimensions.clone();
        self.emit(
            &format!("{prefix}.residual"),
            F32CnnOp::ResidualAdd,
            vec![h, shortcut],
            shape,
            BTreeMap::new(),
        )
    }
}
fn emitted(
    c: &PH3ImageVaeEncoderConfig,
    family: &ModelFamilyId,
) -> Result<(ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>), VNextError> {
    c.validate()?;
    let a = c.architecture();
    let refs = weights::references(c)?;
    let mut e = Emitter {
        nodes: vec![],
        specs: c.input_tensor_specs()?,
    };
    e.specs
        .extend(refs.iter().map(|r| (r.value_id.clone(), r.tensor.clone())));
    let mut hidden = e.conv(
        "encoder.conv_in",
        "encoder.conv_in",
        ProgramValueId::new(PH3ImageVaeEncoderFamily::INPUT_ID)?,
        1,
        [1; 4],
    )?;
    for (level, &channels) in a.channels().iter().enumerate() {
        for block in 0..a.layers_per_block() {
            hidden = e.residual(
                &format!("encoder.down.{level}.block.{block}"),
                hidden,
                channels,
                a,
            )?;
        }
        if a.spatial_downsample_factors()[level] > 1 || a.temporal_downsample_factors()[level] > 1 {
            let stride = a.spatial_downsample_factors()[level];
            let pad = if stride == 2 { [0, 1, 0, 1] } else { [0; 4] };
            let name = format!("encoder.down.{level}.downsample.conv");
            hidden = e.conv(&name, &name, hidden, stride, pad)?;
        }
    }
    hidden = e.norm("encoder.norm_out", "encoder.norm_out", hidden, a)?;
    hidden = e.conv("encoder.conv_out", "encoder.conv_out", hidden, 1, [1; 4])?;
    e.conv("quant_conv", "quant_conv", hidden, 1, [0; 4])?;
    if e.nodes.len() != c.plan_nodes()? {
        return Err(invalid(
            "complete node admission differs from emitted full CNN graph",
        ));
    }
    for w in &refs {
        e.specs.remove(&w.value_id);
    }
    let graph = ModelProgram::new(
        family.clone(),
        vec![ProgramValueId::new(PH3ImageVaeEncoderFamily::INPUT_ID)?],
        vec![ProgramBlock {
            id: "block.p_h3_image.vae.encoder.full_cnn".into(),
            nodes: e.nodes,
        }],
        vec![],
        refs,
        vec![ProgramValueId::new(PH3ImageVaeEncoderFamily::OUTPUT_ID)?],
    )?
    .with_input_tensor_specs(c.input_tensor_specs()?)?;
    Ok((graph, e.specs))
}
pub(super) fn build(
    c: &PH3ImageVaeEncoderConfig,
    family: &ModelFamilyId,
) -> Result<ModelProgram, VNextError> {
    Ok(emitted(c, family)?.0)
}
pub(super) fn compile_options(
    c: &PH3ImageVaeEncoderConfig,
) -> Result<ProgramPlanCompileOptions, VNextError> {
    ProgramPlanCompileOptions::new(
        emitted(c, &ModelFamilyId::new(PH3ImageVaeEncoderFamily::FAMILY_ID)?)?.1,
    )
}
