use super::*;
type Attributes = BTreeMap<AttributeId, SemanticValue>;
#[derive(Clone)]
struct ValueTensor {
    id: ProgramValueId,
    shape: [u64; 4],
}
pub(super) struct Graph {
    pub nodes: Vec<ProgramNode>,
    pub weights: Vec<QwenImage21VaeDecoderWeight>,
    pub specs: BTreeMap<ProgramValueId, ProgramTensorSpec>,
}
impl Graph {
    fn weight(&mut self, name: &str, dimensions: &[u64]) -> Result<ProgramValueId, VNextError> {
        let weight = QwenImage21VaeDecoderWeight::new(name, dimensions)?;
        if self.weights.iter().any(|w| w.external_name() == name) {
            return Err(invalid("weights", format!("duplicate graph weight {name}")));
        }
        let id = weight.value_id()?;
        self.weights.push(weight);
        Ok(id)
    }
    fn node(
        &mut self,
        name: &str,
        operation: &str,
        inputs: Vec<ProgramValueId>,
        shape: [u64; 4],
        attributes: Attributes,
    ) -> Result<ValueTensor, VNextError> {
        tensor(&shape)?.minimum_storage_bytes()?;
        let id = ProgramValueId::new(format!("value.qwen_image21.vae.decoder.{name}"))?;
        if self.specs.insert(id.clone(), spec(&shape)).is_some() {
            return Err(invalid("program", "duplicate node output"));
        }
        self.nodes.push(ProgramNode {
            id: NodeId::new(format!("node.qwen_image21.vae.decoder.{name}"))?,
            operation_id: OperationId::new(operation)?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs,
            outputs: vec![id.clone()],
            attributes,
        });
        Ok(ValueTensor { id, shape })
    }
    fn conv(
        &mut self,
        name: &str,
        x: &ValueTensor,
        out_channels: u64,
        kernel: u64,
        mode: Conv2dPaddingMode,
    ) -> Result<ValueTensor, VNextError> {
        let shape = [x.shape[0], out_channels, x.shape[2], x.shape[3]];
        let weight_shape = [out_channels, x.shape[1], kernel, kernel];
        let pad = if kernel == 1 { 0 } else { 1 };
        let attrs = BTreeMap::from([
            (
                AttributeId::new("padding")?,
                SemanticValue::Integers(vec![pad; 4]),
            ),
            (
                AttributeId::new("stride")?,
                SemanticValue::Integers(vec![1, 1]),
            ),
            (
                AttributeId::new("dilation")?,
                SemanticValue::Integers(vec![1, 1]),
            ),
            (AttributeId::new("groups")?, SemanticValue::Unsigned(1)),
            (
                AttributeId::new("padding_mode")?,
                SemanticValue::Text(mode.as_str().into()),
            ),
        ]);
        Conv2dBf16Geometry::new(
            &tensor(&x.shape)?,
            &tensor(&weight_shape)?,
            &tensor(&[out_channels])?,
            &tensor(&shape)?,
            &attrs,
        )?;
        let w = self.weight(&format!("{name}.weight"), &weight_shape)?;
        let b = self.weight(&format!("{name}.bias"), &[out_channels])?;
        self.node(
            name,
            CONV2D_BF16_OPERATION_ID,
            vec![x.id.clone(), w, b],
            shape,
            attrs,
        )
    }
    fn norm(
        &mut self,
        name: &str,
        x: &ValueTensor,
        image_gamma: bool,
    ) -> Result<ValueTensor, VNextError> {
        let channels = x.shape[1];
        let gamma_shape = if image_gamma {
            vec![channels, 1, 1]
        } else {
            vec![channels, 1, 1, 1]
        };
        let attrs = BTreeMap::from([
            (
                AttributeId::new("epsilon")?,
                SemanticValue::Rational(CanonicalRational::new(1, 1_000_000_000_000)?),
            ),
            (
                AttributeId::new("scale")?,
                SemanticValue::Rational(f32_rational((channels as f64).sqrt() as f32)?),
            ),
        ]);
        let t = tensor(&x.shape)?;
        let gamma = tensor(&gamma_shape)?;
        let operation = if image_gamma {
            ChannelL2NormImageBf16Geometry::new(&t, &gamma, &t, &attrs)?;
            CHANNEL_L2_NORM_IMAGE_BF16_OPERATION_ID
        } else {
            ChannelL2NormBf16Geometry::new(&t, &gamma, &t, &attrs)?;
            CHANNEL_L2_NORM_BF16_OPERATION_ID
        };
        let weight = self.weight(&format!("{name}.gamma"), &gamma_shape)?;
        self.node(name, operation, vec![x.id.clone(), weight], x.shape, attrs)
    }
    fn silu(&mut self, name: &str, x: &ValueTensor) -> Result<ValueTensor, VNextError> {
        Bf16NchwGeometry::new(&tensor(&x.shape)?, &tensor(&x.shape)?)?;
        self.node(
            name,
            SILU_NCHW_BF16_OPERATION_ID,
            vec![x.id.clone()],
            x.shape,
            BTreeMap::new(),
        )
    }
    fn add(
        &mut self,
        name: &str,
        x: &ValueTensor,
        y: &ValueTensor,
    ) -> Result<ValueTensor, VNextError> {
        if x.shape != y.shape {
            return Err(invalid("program", "residual shapes differ"));
        }
        self.node(
            name,
            PLAIN_RESIDUAL_ADD_NCHW_BF16_OPERATION_ID,
            vec![x.id.clone(), y.id.clone()],
            x.shape,
            BTreeMap::new(),
        )
    }
    fn residual(
        &mut self,
        name: &str,
        x: &ValueTensor,
        channels: u64,
    ) -> Result<ValueTensor, VNextError> {
        let shortcut = if x.shape[1] != channels {
            self.conv(
                &format!("{name}.conv_shortcut"),
                x,
                channels,
                1,
                Conv2dPaddingMode::ExplicitZero,
            )?
        } else {
            x.clone()
        };
        let norm1 = self.norm(&format!("{name}.norm1"), x, false)?;
        let silu1 = self.silu(&format!("{name}.silu1"), &norm1)?;
        let conv1 = self.conv(
            &format!("{name}.conv1"),
            &silu1,
            channels,
            3,
            Conv2dPaddingMode::ExplicitZero,
        )?;
        let norm2 = self.norm(&format!("{name}.norm2"), &conv1, false)?;
        let silu2 = self.silu(&format!("{name}.silu2"), &norm2)?;
        let conv2 = self.conv(
            &format!("{name}.conv2"),
            &silu2,
            channels,
            3,
            Conv2dPaddingMode::ExplicitZero,
        )?;
        self.add(&format!("{name}.residual"), &conv2, &shortcut)
    }
    fn attention(&mut self, x: &ValueTensor) -> Result<ValueTensor, VNextError> {
        let prefix = "decoder.mid_block.attentions.0";
        let normalized = self.norm(&format!("{prefix}.norm"), x, true)?;
        let channels = x.shape[1];
        let qkv = self.conv(
            &format!("{prefix}.to_qkv"),
            &normalized,
            channels
                .checked_mul(3)
                .ok_or_else(|| invalid("attention", "packed channels overflow"))?,
            1,
            Conv2dPaddingMode::ImplicitZero,
        )?;
        let attrs = BTreeMap::from([(
            AttributeId::new("scale")?,
            SemanticValue::Rational(f32_rational((1.0 / (channels as f64).sqrt()) as f32)?),
        )]);
        SpatialAttentionBf16Geometry::new(&tensor(&qkv.shape)?, &tensor(&x.shape)?, &attrs)?;
        let attended = self.node(
            &format!("{prefix}.attention"),
            SPATIAL_ATTENTION_BF16_OPERATION_ID,
            vec![qkv.id],
            x.shape,
            attrs,
        )?;
        let projected = self.conv(
            &format!("{prefix}.proj"),
            &attended,
            channels,
            1,
            Conv2dPaddingMode::ImplicitZero,
        )?;
        self.add(&format!("{prefix}.residual"), &projected, x)
    }
    fn gather(
        &mut self,
        name: &str,
        x: &ValueTensor,
        out_channels: u64,
        stride: u64,
        offsets: Vec<i64>,
    ) -> Result<ValueTensor, VNextError> {
        let shape = [
            x.shape[0],
            out_channels,
            x.shape[2]
                .checked_mul(2)
                .ok_or_else(|| invalid("height", "upsample overflows"))?,
            x.shape[3]
                .checked_mul(2)
                .ok_or_else(|| invalid("width", "upsample overflows"))?,
        ];
        let attrs = BTreeMap::from([
            (AttributeId::new("scale_h")?, SemanticValue::Unsigned(2)),
            (AttributeId::new("scale_w")?, SemanticValue::Unsigned(2)),
            (
                AttributeId::new("channel_stride")?,
                SemanticValue::Unsigned(stride),
            ),
            (
                AttributeId::new("channel_offsets")?,
                SemanticValue::Integers(offsets),
            ),
        ]);
        SpatialUpsampleGatherBf16Geometry::new(&tensor(&x.shape)?, &tensor(&shape)?, &attrs)?;
        self.node(
            name,
            SPATIAL_UPSAMPLE_GATHER_BF16_OPERATION_ID,
            vec![x.id.clone()],
            shape,
            attrs,
        )
    }
}

pub(super) fn build(config: &QwenImage21VaeDecoderConfig) -> Result<Graph, VNextError> {
    config.validate()?;
    let mut graph = Graph {
        nodes: vec![],
        weights: vec![],
        specs: BTreeMap::new(),
    };
    let channels = config.decoder_channels()?;
    let input = ValueTensor {
        id: ProgramValueId::new(QwenImage21VaeDecoderFamily::INPUT_ID)?,
        shape: [
            config.batch(),
            config.latent_channels(),
            config.height(),
            config.width(),
        ],
    };
    let post = graph.conv(
        "post_quant_conv",
        &input,
        config.latent_channels(),
        1,
        Conv2dPaddingMode::ExplicitZero,
    )?;
    let mut x = graph.conv(
        "decoder.conv_in",
        &post,
        channels[0],
        3,
        Conv2dPaddingMode::ExplicitZero,
    )?;
    x = graph.residual("decoder.mid_block.resnets.0", &x, channels[0])?;
    x = graph.attention(&x)?;
    x = graph.residual("decoder.mid_block.resnets.1", &x, channels[0])?;
    for (i, pair) in channels.windows(2).enumerate() {
        let original = x.clone();
        let out_channels = pair[1];
        let prefix = format!("decoder.up_blocks.{i}");
        for j in 0..=config.num_res_blocks() {
            x = graph.residual(&format!("{prefix}.resnets.{j}"), &x, out_channels)?;
        }
        if i < config.temporal_upsample().len() {
            x = graph.gather(
                &format!("{prefix}.upsampler.nearest"),
                &x,
                out_channels,
                1,
                vec![0; 4],
            )?;
            // This is ordinary nn.Conv2d(padding=1), not a CausalConv3d:
            // preserve its native implicit padding in the plan and profile.
            x = graph.conv(
                &format!("{prefix}.upsampler.resample.1"),
                &x,
                out_channels,
                3,
                Conv2dPaddingMode::ImplicitZero,
            )?;
            let stride = pair[0] / pair[1];
            let factor = if config.temporal_upsample()[i] { 8 } else { 4 };
            let repeats = factor / stride;
            let temporal_offset = if config.temporal_upsample()[i] { 4 } else { 0 };
            let offsets = (0..4)
                .map(|phase| {
                    i64::try_from((temporal_offset + phase) / repeats)
                        .map_err(|_| invalid("shortcut", "offset overflows"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let shortcut = graph.gather(
                &format!("{prefix}.first_frame_shortcut"),
                &original,
                out_channels,
                stride,
                offsets,
            )?;
            x = graph.add(&format!("{prefix}.residual"), &x, &shortcut)?;
        }
    }
    x = graph.norm("decoder.norm_out", &x, false)?;
    x = graph.silu("decoder.silu_out", &x)?;
    let output = graph.conv(
        "decoder.conv_out",
        &x,
        4,
        3,
        Conv2dPaddingMode::ExplicitZero,
    )?;
    if tensor(&output.shape)? != config.output_tensor()? {
        return Err(invalid(
            "output",
            "decoder output differs from configured RGBA dimensions",
        ));
    }
    Ok(graph)
}

/// Exact binary rational for the effective F32 scale, independently chosen
/// from the model's real-valued sqrt(C) / 1/sqrt(C) formula.
fn f32_rational(value: f32) -> Result<CanonicalRational, VNextError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid("scale", "requires positive finite F32"));
    }
    let bits = value.to_bits();
    let shift = 150_i32 - ((bits >> 23) & 255) as i32;
    if !(0..64).contains(&shift) {
        return Err(invalid(
            "scale",
            "effective scale is outside exact rational representation",
        ));
    }
    CanonicalRational::new(i64::from((bits & 0x7fffff) | (1 << 23)), 1_u64 << shift)
}
