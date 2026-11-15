use super::*;
pub(super) struct Graph {
    nodes: Vec<ProgramNode>,
    specs: BTreeMap<ProgramValueId, ProgramTensorSpec>,
    batch: u64,
    limits: FastH3AudioDecoderLimits,
}
impl Graph {
    pub(super) fn new(
        batch: u64,
        limits: FastH3AudioComponentLimits,
        inputs: BTreeMap<ProgramValueId, ProgramTensorSpec>,
    ) -> Self {
        Self {
            nodes: Vec::new(),
            specs: inputs,
            batch,
            limits,
        }
    }
    pub(super) fn emit(
        &mut self,
        name: &str,
        op: &str,
        inputs: Vec<ProgramValueId>,
        channels: u64,
        length: u64,
        attrs: BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<ProgramValueId, VNextError> {
        self.emit_shape(
            name,
            op,
            inputs,
            vec![self.batch, channels, 1, length],
            attrs,
        )
    }
    pub(super) fn emit_shape(
        &mut self,
        name: &str,
        op: &str,
        inputs: Vec<ProgramValueId>,
        shape: Vec<u64>,
        attrs: BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<ProgramValueId, VNextError> {
        if self.nodes.len() >= self.limits.maximum_plan_nodes {
            return Err(invalid("audio node capacity exceeded"));
        }
        let output: ProgramValueId = id(format!("value.fast_h3.audio.{name}"))?;
        let spec = tensor(shape);
        spec.validate("audio activation")?;
        if spec.byte_len()? > self.limits.maximum_activation_bytes {
            return Err(invalid("audio activation capacity exceeded"));
        }
        if self.specs.insert(output.clone(), spec).is_some() {
            return Err(invalid("duplicate audio value"));
        }
        self.nodes.push(ProgramNode {
            id: id(format!("node.fast_h3.audio.{name}"))?,
            operation_id: id(op)?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs,
            outputs: vec![output.clone()],
            attributes: attrs,
        });
        Ok(output)
    }
    pub(super) fn conv(
        &mut self,
        name: &str,
        x: ProgramValueId,
        channels: u64,
        length: u64,
        kernel: u64,
        dilation: u64,
    ) -> Result<ProgramValueId, VNextError> {
        let pad = (kernel - 1)
            .checked_mul(dilation)
            .ok_or_else(|| invalid("audio dilation overflow"))?
            / 2;
        self.convolution(name, name, x, channels, length, dilation, 1, pad)
    }
    /// Source names keep their checkpoint identity when a component gives its
    /// final output a stable public value ID. Shapes remain explicit NCHW.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn convolution(
        &mut self,
        name: &str,
        source: &str,
        x: ProgramValueId,
        channels: u64,
        length: u64,
        dilation: u64,
        stride: u64,
        pad: u64,
    ) -> Result<ProgramValueId, VNextError> {
        let integers = |v: Vec<u64>| -> Result<SemanticValue, VNextError> {
            Ok(SemanticValue::Integers(
                v.into_iter()
                    .map(|x| i64::try_from(x).map_err(invalid))
                    .collect::<Result<_, _>>()?,
            ))
        };
        self.emit(
            name,
            CONV2D_F32_OPERATION_ID,
            vec![
                x,
                weight_value(&format!("{source}.weight"))?,
                weight_value(&format!("{source}.bias"))?,
            ],
            channels,
            length,
            BTreeMap::from([
                (id("padding")?, integers(vec![0, 0, pad, pad])?),
                (id("stride")?, integers(vec![1, stride])?),
                (id("dilation")?, integers(vec![1, dilation])?),
                (id("groups")?, SemanticValue::Unsigned(1)),
                (
                    id("padding_mode")?,
                    SemanticValue::Text("explicit_zero".into()),
                ),
            ]),
        )
    }
    fn upsample(
        &mut self,
        name: &str,
        x: ProgramValueId,
        channels: u64,
        length: u64,
        kernel: u64,
        stride: u64,
    ) -> Result<ProgramValueId, VNextError> {
        let integers = |v: Vec<u64>| -> Result<SemanticValue, VNextError> {
            Ok(SemanticValue::Integers(
                v.into_iter()
                    .map(|x| i64::try_from(x).map_err(invalid))
                    .collect::<Result<_, _>>()?,
            ))
        };
        self.emit(
            name,
            CONV_TRANSPOSE2D_F32_OPERATION_ID,
            vec![
                x,
                weight_value(&format!("{name}.weight"))?,
                weight_value(&format!("{name}.bias"))?,
            ],
            channels,
            length,
            BTreeMap::from([
                (id("padding")?, integers(vec![0, (kernel - stride) / 2])?),
                (id("stride")?, integers(vec![1, stride])?),
                (id("dilation")?, integers(vec![1, 1])?),
                (id("output_padding")?, integers(vec![0, 0])?),
                (id("groups")?, SemanticValue::Unsigned(1)),
            ]),
        )
    }
    fn activation(
        &mut self,
        name: &str,
        x: ProgramValueId,
        channels: u64,
        length: u64,
    ) -> Result<ProgramValueId, VNextError> {
        let doubled = length
            .checked_mul(2)
            .ok_or_else(|| invalid("audio upsample extent overflow"))?;
        let x = self.emit(
            &format!("{name}.up_filter"),
            FIR_RESAMPLE1D_F32_OPERATION_ID,
            vec![x, weight_value(&format!("{name}.upsample.filter"))?],
            channels,
            doubled,
            SignalF32Geometry::fir_attributes(true, 2, [5, 5], [15, 15])?,
        )?;
        let x = self.emit(
            &format!("{name}.up_scale"),
            SCALE_RANK4_F32_OPERATION_ID,
            vec![x],
            channels,
            doubled,
            BatchedF32Geometry::scale_attributes(2.)?,
        )?;
        let x = self.emit(
            &format!("{name}.snake"),
            SNAKE_BETA_LOG_NCHW_F32_OPERATION_ID,
            vec![
                x,
                weight_value(&format!("{name}.act.alpha"))?,
                weight_value(&format!("{name}.act.beta"))?,
            ],
            channels,
            doubled,
            BTreeMap::new(),
        )?;
        self.emit(
            &format!("{name}.down_filter"),
            FIR_RESAMPLE1D_F32_OPERATION_ID,
            vec![
                x,
                weight_value(&format!("{name}.downsample.lowpass.filter"))?,
            ],
            channels,
            length,
            SignalF32Geometry::fir_attributes(false, 2, [5, 6], [0, 0])?,
        )
    }
    pub(super) fn add(
        &mut self,
        name: &str,
        left: ProgramValueId,
        right: ProgramValueId,
        channels: u64,
        length: u64,
    ) -> Result<ProgramValueId, VNextError> {
        self.emit(
            name,
            PLAIN_RESIDUAL_ADD_NCHW_F32_OPERATION_ID,
            vec![left, right],
            channels,
            length,
            BTreeMap::new(),
        )
    }
    pub(super) fn finish(
        self,
        family: &str,
        block: &str,
        inputs: BTreeMap<ProgramValueId, ProgramTensorSpec>,
        weights: &[AudioWeight],
        output: ProgramValueId,
    ) -> Result<(ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>), VNextError> {
        let weights = weights
            .iter()
            .map(|w| {
                Ok(WeightReference {
                    weight_id: weight_id(&w.name)?,
                    value_id: weight_value(&w.name)?,
                    tensor: tensor(w.component.dimensions.clone()),
                })
            })
            .collect::<Result<_, VNextError>>()?;
        Ok((
            ModelProgram::new(
                id(family)?,
                inputs.keys().cloned().collect(),
                vec![ProgramBlock {
                    id: block.into(),
                    nodes: self.nodes,
                }],
                vec![],
                weights,
                vec![output],
            )?
            .with_input_tensor_specs(inputs)?,
            self.specs,
        ))
    }
}
pub(super) fn build(
    c: &DecoderConfig,
    limits: FastH3AudioDecoderLimits,
) -> Result<(ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>), VNextError> {
    let a = &c.architecture;
    let batch = c.batch()?;
    let input: ProgramValueId = id(PreparedFastH3AudioDecoder::INPUT_ID)?;
    let inputs = BTreeMap::from([(
        input.clone(),
        tensor(vec![batch, a.latent_channels, 1, c.latent_samples]),
    )]);
    if inputs[&input].byte_len()? > limits.maximum_activation_bytes {
        return Err(invalid("audio input capacity exceeded"));
    }
    let mut g = Graph::new(batch, limits, inputs.clone());
    let mut length = c.latent_samples;
    let x = g.conv("dec_in_proj", input.clone(), a.latent_dim, length, 1, 1)?;
    let mut x = g.conv("decoder.conv_pre", x, a.decoder_dim, length, 7, 1)?;
    let mut channels = a.decoder_dim;
    for stage in 0..a.decoder_rates.len() {
        length = length
            .checked_mul(a.decoder_rates[stage])
            .ok_or_else(|| invalid("waveform length overflow"))?;
        channels /= 2;
        x = g.upsample(
            &format!("decoder.ups.{stage}.0"),
            x,
            channels,
            length,
            a.decoder_kernel_sizes[stage],
            a.decoder_rates[stage],
        )?;
        let mut sum = None;
        for branch in 0..a.resblock_kernel_sizes.len() {
            let prefix = format!(
                "decoder.resblocks.{}",
                stage * a.resblock_kernel_sizes.len() + branch
            );
            let mut block = x.clone();
            for (step, &dilation) in a.resblock_dilation_sizes[branch].iter().enumerate() {
                let y = g.activation(
                    &format!("{prefix}.activations.{}", 2 * step),
                    block.clone(),
                    channels,
                    length,
                )?;
                let y = g.conv(
                    &format!("{prefix}.convs1.{step}"),
                    y,
                    channels,
                    length,
                    a.resblock_kernel_sizes[branch],
                    dilation,
                )?;
                let y = g.activation(
                    &format!("{prefix}.activations.{}", 2 * step + 1),
                    y,
                    channels,
                    length,
                )?;
                let y = g.conv(
                    &format!("{prefix}.convs2.{step}"),
                    y,
                    channels,
                    length,
                    a.resblock_kernel_sizes[branch],
                    1,
                )?;
                block = g.add(
                    &format!("{prefix}.residual.{step}"),
                    y,
                    block,
                    channels,
                    length,
                )?;
            }
            sum = Some(match sum {
                None => block,
                Some(previous) => g.add(
                    &format!("decoder.stage{stage}.sum{branch}"),
                    previous,
                    block,
                    channels,
                    length,
                )?,
            });
        }
        x = g.emit(
            &format!("decoder.stage{stage}.mean"),
            SCALE_RANK4_F32_OPERATION_ID,
            vec![sum.ok_or_else(|| invalid("empty residual branches"))?],
            channels,
            length,
            // Torch's scalar true-division computes its reciprocal in double,
            // casts once to opmath F32, then multiplies each F32 element.
            BatchedF32Geometry::scale_attributes(
                (1.0f64 / a.resblock_kernel_sizes.len() as f64) as f32,
            )?,
        )?;
    }
    x = g.activation("decoder.activation_post", x, channels, length)?;
    x = g.conv("decoder.conv_post", x, 1, length, 7, 1)?;
    let output = g.emit(
        "waveform",
        CLAMP_RANK4_F32_OPERATION_ID,
        vec![x],
        1,
        length,
        SignalF32Geometry::clamp_attributes(-1., 1.)?,
    )?;
    g.finish(
        family::Family::FAMILY_ID,
        "block.fast_h3.audio.decoder",
        inputs,
        &c.weights,
        output,
    )
}
