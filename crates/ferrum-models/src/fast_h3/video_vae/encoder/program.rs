use super::*;
pub(super) struct Graph {
    pub program: ModelProgram,
    pub specs: BTreeMap<ProgramValueId, ProgramTensorSpec>,
    pub schema: WeightSchema,
}
struct Emitter {
    nodes: Vec<ProgramNode>,
    specs: BTreeMap<ProgramValueId, ProgramTensorSpec>,
    references: Vec<WeightReference>,
    components: Vec<WeightComponentSpec>,
    weights: Vec<WeightTensorSpec>,
    stored_bytes: u64,
    limits: FastH3VideoEncoderLimits,
}
impl Emitter {
    fn weight(&mut self, name: &str, shape: Vec<u64>) -> Result<ProgramValueId, VNextError> {
        let t = tensor(shape.clone());
        self.stored_bytes = self
            .stored_bytes
            .checked_add(t.byte_len()?)
            .filter(|v| *v <= self.limits.maximum_stored_bytes)
            .ok_or_else(|| invalid("original encoder weights exceed capacity"))?;
        let value: ProgramValueId = id(&format!("value.fast_h3.video_encoder.weight.{name}"))?;
        let weight: WeightId = id(&format!("weight.fast_h3.video_encoder.{name}"))?;
        let component: WeightId = id(&format!("component.fast_h3.video_encoder.{name}"))?;
        if self.specs.insert(value.clone(), t.clone()).is_some() {
            return Err(invalid("duplicate encoder weight"));
        }
        self.components.push(WeightComponentSpec {
            id: component.clone(),
            role: WeightComponentRole::Values,
            external_names: vec![name.into()],
            dimensions: shape.clone(),
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F32,
            },
            required: true,
        });
        self.weights.push(WeightTensorSpec {
            id: weight.clone(),
            dimensions: shape,
            logical_element_type: ElementType::F32,
            physical_layout: PhysicalWeightLayout::Dense {
                component_id: component,
            },
            required: true,
        });
        self.references.push(WeightReference {
            weight_id: weight,
            value_id: value.clone(),
            tensor: t,
        });
        Ok(value)
    }
    fn emit(
        &mut self,
        name: &str,
        kind: F32CnnOp,
        inputs: Vec<ProgramValueId>,
        shape: Vec<u64>,
        attributes: BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<ProgramValueId, VNextError> {
        let output: ProgramValueId = if name == "quant_conv" {
            id(PreparedFastH3VideoEncoder::OUTPUT_ID)?
        } else {
            id(&format!("value.fast_h3.video_encoder.{name}"))?
        };
        let t = tensor(shape);
        if t.byte_len()? > self.limits.maximum_activation_bytes
            || self.nodes.len() >= self.limits.maximum_plan_nodes
        {
            return Err(invalid("encoder activation or plan exceeds capacity"));
        }
        let inputs_resolved = inputs
            .iter()
            .map(|v| resolved(&self.specs[v]))
            .collect::<Result<Vec<_>, _>>()?;
        F32CnnGeometry::new(
            kind,
            &inputs_resolved.iter().collect::<Vec<_>>(),
            &resolved(&t)?,
            &attributes,
        )?;
        self.nodes.push(ProgramNode {
            id: id(&format!("node.fast_h3.video_encoder.{name}"))?,
            operation_id: id(kind.operation_id())?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs,
            outputs: vec![output.clone()],
            attributes,
        });
        self.specs.insert(output.clone(), t);
        Ok(output)
    }
    fn conv(
        &mut self,
        name: &str,
        input: ProgramValueId,
        channels: u64,
        kernel: u64,
        stride: [u64; 3],
        padding: [u64; 6],
    ) -> Result<ProgramValueId, VNextError> {
        let x = self.specs[&input].dimensions.clone();
        let w = self.weight(
            &format!("{name}.weight"),
            vec![channels, x[1], kernel, kernel, kernel],
        )?;
        let b = self.weight(&format!("{name}.bias"), vec![channels])?;
        let mut shape = vec![x[0], channels];
        for axis in 0..3 {
            shape.push(
                x[axis + 2]
                    .checked_add(padding[2 * axis])
                    .and_then(|v| v.checked_add(padding[2 * axis + 1]))
                    .and_then(|v| v.checked_sub(kernel))
                    .map(|v| v / stride[axis] + 1)
                    .ok_or_else(|| invalid("causal kernel exceeds padded input"))?,
            );
        }
        self.emit(
            name,
            F32CnnOp::Conv3d,
            vec![input, w, b],
            shape,
            BTreeMap::from([
                (
                    id("padding")?,
                    SemanticValue::Integers(padding.iter().map(|v| *v as i64).collect()),
                ),
                (id("padding_modes")?, SemanticValue::Integers(vec![0, 1, 1])),
                (
                    id("stride")?,
                    SemanticValue::Integers(stride.iter().map(|v| *v as i64).collect()),
                ),
                (id("dilation")?, SemanticValue::Integers(vec![1; 3])),
                (id("groups")?, SemanticValue::Unsigned(1)),
            ]),
        )
    }
    fn norm(
        &mut self,
        name: &str,
        input: ProgramValueId,
        a: &PH3ImageVaeEncoderArchitecture,
    ) -> Result<ProgramValueId, VNextError> {
        let shape = self.specs[&input].dimensions.clone();
        let w = self.weight(&format!("{name}.weight"), vec![shape[1]])?;
        let b = self.weight(&format!("{name}.bias"), vec![shape[1]])?;
        self.emit(
            name,
            F32CnnOp::FrameGroupNormSilu,
            vec![input, w, b],
            shape,
            BTreeMap::from([
                (id("groups")?, SemanticValue::Unsigned(a.norm_groups())),
                (id("epsilon")?, SemanticValue::Rational(a.epsilon())),
            ]),
        )
    }
    fn residual(
        &mut self,
        name: &str,
        input: ProgramValueId,
        channels: u64,
        a: &PH3ImageVaeEncoderArchitecture,
    ) -> Result<ProgramValueId, VNextError> {
        let normalized = self.norm(&format!("{name}.norm1"), input.clone(), a)?;
        let h = self.conv(
            &format!("{name}.conv1"),
            normalized,
            channels,
            3,
            [1; 3],
            [2, 0, 1, 1, 1, 1],
        )?;
        let normalized = self.norm(&format!("{name}.norm2"), h, a)?;
        let h = self.conv(
            &format!("{name}.conv2"),
            normalized,
            channels,
            3,
            [1; 3],
            [2, 0, 1, 1, 1, 1],
        )?;
        let shortcut = if self.specs[&input].dimensions[1] == channels {
            input
        } else {
            self.conv(
                &format!("{name}.conv_shortcut"),
                input,
                channels,
                1,
                [1; 3],
                [0; 6],
            )?
        };
        self.emit(
            &format!("{name}.residual"),
            F32CnnOp::VolumeResidualAdd,
            vec![shortcut, h.clone()],
            self.specs[&h].dimensions.clone(),
            BTreeMap::new(),
        )
    }
}
pub(super) fn build(c: &Config, limits: FastH3VideoEncoderLimits) -> Result<Graph, VNextError> {
    c.validate(limits)?;
    let a = &c.architecture;
    let g = c.geometry;
    let input_id = id(PreparedFastH3VideoEncoder::INPUT_ID)?;
    let input = tensor(vec![g.batch, a.in_channels(), g.frames, g.height, g.width]);
    if input.byte_len()? > limits.maximum_activation_bytes {
        return Err(invalid("encoder input exceeds activation capacity"));
    }
    let input_specs = BTreeMap::from([(input_id, input)]);
    let mut e = Emitter {
        nodes: vec![],
        specs: input_specs.clone(),
        references: vec![],
        components: vec![],
        weights: vec![],
        stored_bytes: 0,
        limits,
    };
    let mut h = e.conv(
        "encoder.conv_in",
        id(PreparedFastH3VideoEncoder::INPUT_ID)?,
        a.channels()[0],
        3,
        [1; 3],
        [2, 0, 1, 1, 1, 1],
    )?;
    for (level, &channels) in a.channels().iter().enumerate() {
        for block in 0..a.layers_per_block() {
            h = e.residual(
                &format!("encoder.down_blocks.{level}.resnets.{block}"),
                h,
                channels,
                a,
            )?;
        }
        let s = a.spatial_downsample_factors()[level];
        let t = a.temporal_downsample_factors()[level];
        if s > 1 || t > 1 {
            let p = u64::from(s == 2);
            h = e.conv(
                &format!("encoder.down_blocks.{level}.downsamplers.0.conv"),
                h,
                channels,
                3,
                [t, s, s],
                [2, 0, 0, p, 0, p],
            )?;
        }
    }
    h = e.norm("encoder.norm_out", h, a)?;
    let moments = a
        .latent_channels()
        .checked_mul(2)
        .ok_or_else(|| invalid("moment width overflows"))?;
    h = e.conv(
        "encoder.conv_out",
        h,
        moments,
        3,
        [1; 3],
        [2, 0, 1, 1, 1, 1],
    )?;
    e.conv("quant_conv", h, moments, 1, [1; 3], [0; 6])?;
    for w in &e.references {
        e.specs.remove(&w.value_id);
    }
    let schema = WeightSchema {
        format_id: id("weight-format.safetensors.dense")?,
        layout_id: id("weight-layout.fast_h3.video_encoder.original_f32.full_time.v1")?,
        version: ContractVersion::new(1, 0),
        components: e.components,
        tensors: e.weights,
    };
    schema.validate(&id(family::Family::ID)?)?;
    let program = ModelProgram::new(
        id(family::Family::ID)?,
        vec![id(PreparedFastH3VideoEncoder::INPUT_ID)?],
        vec![ProgramBlock {
            id: "block.fast_h3.video_encoder.full_causal".into(),
            nodes: e.nodes,
        }],
        vec![],
        e.references,
        vec![id(PreparedFastH3VideoEncoder::OUTPUT_ID)?],
    )?
    .with_input_tensor_specs(input_specs)?;
    Ok(Graph {
        program,
        specs: e.specs,
        schema,
    })
}
