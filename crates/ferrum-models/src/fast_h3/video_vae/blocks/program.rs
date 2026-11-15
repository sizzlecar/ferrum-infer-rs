use super::*;
struct Graph {
    rows: u64,
    nodes: Vec<ProgramNode>,
    specs: BTreeMap<ProgramValueId, ProgramTensorSpec>,
}
impl Graph {
    fn emit(
        &mut self,
        name: &str,
        operation: &str,
        inputs: Vec<ProgramValueId>,
        dtype: ElementType,
        width: u64,
        attributes: BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<ProgramValueId, VNextError> {
        let output = value(name)?;
        self.specs
            .insert(output.clone(), tensor(vec![self.rows, width], dtype));
        self.nodes.push(ProgramNode {
            id: NodeId::new(format!("node.fast_h3.video_decoder.{name}"))?,
            operation_id: OperationId::new(operation)?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs,
            outputs: vec![output.clone()],
            attributes,
        });
        Ok(output)
    }
    fn cast(
        &mut self,
        name: &str,
        x: ProgramValueId,
        dtype: ElementType,
        width: u64,
    ) -> Result<ProgramValueId, VNextError> {
        self.emit(
            name,
            if dtype == ElementType::F32 {
                CAST_BF16_TO_F32_OPERATION_ID
            } else {
                CAST_F32_TO_BF16_OPERATION_ID
            },
            vec![x],
            dtype,
            width,
            BTreeMap::new(),
        )
    }
    fn norm(
        &mut self,
        name: &str,
        x: ProgramValueId,
        gamma: ProgramValueId,
        width: u64,
        head: u64,
        dtype: ElementType,
        eps: CanonicalRational,
    ) -> Result<ProgramValueId, VNextError> {
        self.emit(
            name,
            if dtype == ElementType::F32 {
                HEAD_RMS_NORM_F32_OPERATION_ID
            } else {
                HEAD_RMS_NORM_BF16_OPERATION_ID
            },
            vec![x, gamma],
            dtype,
            width,
            BTreeMap::from([
                (AttributeId::new("head_dim")?, SemanticValue::Unsigned(head)),
                (AttributeId::new("epsilon")?, SemanticValue::Rational(eps)),
            ]),
        )
    }
    fn quantize(
        &mut self,
        name: &str,
        x: ProgramValueId,
        width: u64,
    ) -> Result<[ProgramValueId; 2], VNextError> {
        let outputs = [
            value(&format!("{name}.packed"))?,
            value(&format!("{name}.scales"))?,
        ];
        let g = Nvfp4MatrixLayout::new(self.rows, width)?;
        for (v, s) in outputs.iter().zip([g.packed_shape(), g.scale_shape()]) {
            self.specs
                .insert(v.clone(), tensor(s.to_vec(), ElementType::U8));
        }
        self.nodes.push(ProgramNode {
            id: NodeId::new(format!("node.fast_h3.video_decoder.{name}"))?,
            operation_id: OperationId::new(NVFP4_QUANTIZE_BF16_OPERATION_ID)?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs: vec![x, wvalue("unit_sf")?],
            outputs: outputs.to_vec(),
            attributes: Nvfp4QuantizeProfile::CudaRcpApproxFtzV1.attributes()?,
        });
        Ok(outputs)
    }
    fn projection(
        &mut self,
        name: &str,
        w: &str,
        activation: &[ProgramValueId; 2],
        width: u64,
    ) -> Result<ProgramValueId, VNextError> {
        let raw = self.emit(
            &format!("{name}.gemm"),
            NVFP4_MATMUL_BF16_OPERATION_ID,
            vec![
                activation[0].clone(),
                wvalue(&format!("{w}.packed"))?,
                activation[1].clone(),
                wvalue(&format!("{w}.scales"))?,
                wvalue(&format!("{w}.alpha"))?,
            ],
            ElementType::Bf16,
            width,
            BTreeMap::new(),
        )?;
        self.emit(
            &format!("{name}.bias"),
            CHANNEL_BIAS_ADD_BF16_OPERATION_ID,
            vec![raw, wvalue(&format!("{w}.bias"))?],
            ElementType::Bf16,
            width,
            BTreeMap::new(),
        )
    }
}
pub(super) fn build(
    c: &FastH3VideoDecoderBlocksConfig,
) -> Result<(ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>), VNextError> {
    type F = FastH3VideoDecoderBlocksFamily;
    let input = ProgramValueId::new(F::INPUT_ID)?;
    let cosine = ProgramValueId::new(F::COSINE_ID)?;
    let sine = ProgramValueId::new(F::SINE_ID)?;
    let rows = c.rows()?;
    let inputs = BTreeMap::from([
        (
            input.clone(),
            tensor(vec![rows, c.hidden], ElementType::F32),
        ),
        (
            cosine.clone(),
            tensor(vec![rows, c.rotary_dim], ElementType::Bf16),
        ),
        (
            sine.clone(),
            tensor(vec![rows, c.rotary_dim], ElementType::Bf16),
        ),
    ]);
    let mut g = Graph {
        rows,
        nodes: Vec::new(),
        specs: inputs.clone(),
    };
    let mut hidden = input.clone();
    for b in &c.blocks {
        let name = |s: &str| format!("block{}.{}", b.index, s);
        let normalized = g.norm(
            &name("norm1"),
            hidden.clone(),
            wvalue(&name("norm1.weight"))?,
            c.hidden,
            c.hidden,
            ElementType::F32,
            c.epsilon,
        )?;
        let normalized = g.cast(&name("norm1_bf16"), normalized, ElementType::Bf16, c.hidden)?;
        let quantized = g.quantize(&name("qkv_quant"), normalized, c.hidden)?;
        let mut qkv = Vec::new();
        for p in &PROJECTIONS[..3] {
            qkv.push(g.projection(&name(p), &name(p), &quantized, c.hidden)?);
        }
        for (i, label) in [(0, "query"), (1, "key")] {
            let normalized = g.norm(
                &name(&format!("{label}_norm")),
                qkv[i].clone(),
                wvalue("unit_gamma")?,
                c.hidden,
                c.head_dim,
                ElementType::Bf16,
                c.epsilon,
            )?;
            qkv[i] = g.emit(
                &name(&format!("{label}_rope")),
                PARTIAL_HALF_SPLIT_ROPE_BF16_OPERATION_ID,
                vec![normalized, cosine.clone(), sine.clone()],
                ElementType::Bf16,
                c.hidden,
                BTreeMap::from([
                    (
                        AttributeId::new("head_dim")?,
                        SemanticValue::Unsigned(c.head_dim),
                    ),
                    (
                        AttributeId::new("rotary_dim")?,
                        SemanticValue::Unsigned(c.rotary_dim),
                    ),
                ]),
            )?;
        }
        // Actual quality decoder D=64 gives exact 1/sqrt(D)=1/8. Keep other
        // dimensions out of this original numeric profile until qualified.
        let attention = g.emit(
            &name("attention"),
            SEGMENTED_ATTENTION_BF16_OPERATION_ID,
            qkv,
            ElementType::Bf16,
            c.hidden,
            BTreeMap::from([
                (
                    AttributeId::new("head_dim")?,
                    SemanticValue::Unsigned(c.head_dim),
                ),
                (
                    AttributeId::new("scale")?,
                    SemanticValue::Rational(CanonicalRational::new(1, 8)?),
                ),
                (
                    AttributeId::new("segment_lengths")?,
                    SemanticValue::Integers(c.sequences.iter().map(|n| *n as i64).collect()),
                ),
            ]),
        )?;
        let quantized = g.quantize(&name("out_quant"), attention, c.hidden)?;
        let out = g.projection(&name("out"), &name(PROJECTIONS[3]), &quantized, c.hidden)?;
        let out = g.cast(&name("out_f32"), out, ElementType::F32, c.hidden)?;
        hidden = g.emit(
            &name("residual1"),
            CHANNEL_SCALED_RESIDUAL_SEPARATE_F32_OPERATION_ID,
            vec![hidden, out, wvalue(&name("scale1"))?],
            ElementType::F32,
            c.hidden,
            BTreeMap::new(),
        )?;
        let normalized = g.norm(
            &name("norm2"),
            hidden.clone(),
            wvalue(&name("norm2.weight"))?,
            c.hidden,
            c.hidden,
            ElementType::F32,
            c.epsilon,
        )?;
        let normalized = g.cast(&name("norm2_bf16"), normalized, ElementType::Bf16, c.hidden)?;
        let quantized = g.quantize(&name("ff_in_quant"), normalized, c.hidden)?;
        let packed = g.projection(&name("ff_in"), &name(PROJECTIONS[4]), &quantized, 2 * c.ffn)?;
        let ff = g.emit(
            &name("ff_swiglu"),
            PACKED_SWIGLU_BF16_OPERATION_ID,
            vec![packed],
            ElementType::Bf16,
            c.ffn,
            PackedSwiGluBf16Profile::SiluBf16ThenMultiply.attributes()?,
        )?;
        let quantized = g.quantize(&name("ff_out_quant"), ff, c.ffn)?;
        let out = g.projection(&name("ff_out"), &name(PROJECTIONS[5]), &quantized, c.hidden)?;
        let out = g.cast(&name("ff_out_f32"), out, ElementType::F32, c.hidden)?;
        hidden = g.emit(
            &name("residual2"),
            CHANNEL_SCALED_RESIDUAL_SEPARATE_F32_OPERATION_ID,
            vec![hidden, out, wvalue(&name("scale2"))?],
            ElementType::F32,
            c.hidden,
            BTreeMap::new(),
        )?;
    }
    let weights = c
        .schema()?
        .tensors
        .iter()
        .map(|w| {
            let name =
                w.id.as_str()
                    .strip_prefix("weight.fast_h3.video_decoder.")
                    .ok_or_else(|| invalid("weight ID namespace"))?;
            Ok(WeightReference {
                weight_id: w.id.clone(),
                value_id: wvalue(name)?,
                tensor: tensor(w.dimensions.clone(), w.logical_element_type),
            })
        })
        .collect::<Result<Vec<_>, VNextError>>()?;
    Ok((
        ModelProgram::new(
            ModelFamilyId::new(F::FAMILY_ID)?,
            vec![input, cosine, sine],
            vec![ProgramBlock {
                id: "block.fast_h3.video_decoder".into(),
                nodes: g.nodes,
            }],
            vec![],
            weights,
            vec![hidden],
        )?
        .with_input_tensor_specs(inputs)?,
        g.specs,
    ))
}
