use super::*;
use crate::tensor_program_fragment::{
    Fragment, StoredComposition, StoredCompositionIdentity, StoredCompositionLimits,
};

pub(super) fn compose(
    c: &FastH3VideoClipConfig,
    l: FastH3VideoClipLimits,
) -> Result<StoredComposition, VNextError> {
    c.validate(l)?;
    type F = FastH3VideoClipFamily;
    type B = FastH3VideoDecoderBlocksFamily;
    let prefix = outer(c, true)?;
    let blocks = Fragment::family(
        &B::new(l.blocks)?,
        &c.blocks,
        B::PROFILE_ID,
        c.blocks.compile_options(l.blocks)?.tensor_specs(),
        BTreeMap::from([
            (ProgramValueId::new(B::INPUT_ID)?, value("embedded")?),
            (
                ProgramValueId::new(B::COSINE_ID)?,
                ProgramValueId::new(F::COSINE_ID)?,
            ),
            (
                ProgramValueId::new(B::SINE_ID)?,
                ProgramValueId::new(F::SINE_ID)?,
            ),
        ]),
    )?;
    let result = StoredComposition::join(
        StoredCompositionIdentity {
            family: ModelFamilyId::new(F::FAMILY_ID)?,
            profile: NumericalProfileId::new(F::PROFILE_ID).map_err(invalid)?,
            format: WeightFormatId::new("weight-format.safetensors.dense")?,
            layout: WeightLayoutId::new("weight-layout.fast_h3.video_clip.v1")?,
            block: "block.fast_h3.video_clip".into(),
        },
        vec![prefix, blocks, outer(c, false)?],
        vec![
            ProgramValueId::new(F::INPUT_ID)?,
            ProgramValueId::new(F::COSINE_ID)?,
            ProgramValueId::new(F::SINE_ID)?,
        ],
        vec![ProgramValueId::new(F::OUTPUT_ID)?],
        ProgramValueId::new(F::OUTPUT_ID)?,
        StoredCompositionLimits {
            maximum_nodes: l.maximum_plan_nodes,
            maximum_stored_bytes: l.blocks.maximum_stored_bytes,
            maximum_metadata_bytes: l.blocks.maximum_metadata_bytes,
        },
    )?;
    for t in result.options.tensor_specs().values() {
        if t.byte_len()? > l.maximum_activation_bytes {
            return Err(invalid("clip activation exceeds declared capacity"));
        }
    }
    Ok(result)
}
fn weight_value(name: &str) -> Result<ProgramValueId, VNextError> {
    value(&format!("weight.{name}"))
}
fn attrs(
    pairs: &[(&str, SemanticValue)],
) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
    pairs
        .iter()
        .map(|(k, v)| Ok((AttributeId::new(*k)?, v.clone())))
        .collect()
}

/// The outer schema is carried once by the prefix; the suffix references the
/// same logical tensors after block composition, without another source owner.
fn outer(c: &FastH3VideoClipConfig, prefix: bool) -> Result<Fragment, VNextError> {
    type F = FastH3VideoClipFamily;
    let a = &c.architecture;
    let z = a.latent_channels() as u64;
    let h = a.hidden_size().map_err(invalid)? as u64;
    let p = a.output_patch_width().map_err(invalid)? as u64;
    let rows = c.patch_rows()?;
    let all = c.rows()?;
    let mut tensors = vec![];
    let mut components = vec![];
    let mut weights = vec![];
    if prefix {
        for o in &c.outer {
            let name = &o.original.external_names[0];
            let id = WeightId::new(format!("weight.fast_h3.video_clip.{name}"))?;
            tensors.push(WeightTensorSpec {
                id: id.clone(),
                dimensions: o.component.dimensions.clone(),
                logical_element_type: o.component.physical_element_type(),
                required: true,
                physical_layout: PhysicalWeightLayout::Dense {
                    component_id: o.component.id.clone(),
                },
            });
            weights.push(WeightReference {
                weight_id: id,
                value_id: weight_value(name)?,
                tensor: ProgramTensorSpec {
                    dimensions: o.component.dimensions.clone(),
                    element_type: o.component.physical_element_type(),
                    layout: ResolvedTensorLayout::Contiguous,
                },
            });
            components.push(o.component.clone());
        }
    }
    let schema = prefix.then(|| WeightSchema {
        format_id: WeightFormatId::new("weight-format.safetensors.dense").expect("static ID"),
        layout_id: WeightLayoutId::new("weight-layout.fast_h3.video_clip.outer.v1")
            .expect("static ID"),
        version: ContractVersion::new(1, 0),
        tensors,
        components,
    });
    let mut nodes = vec![];
    let mut specs = BTreeMap::new();
    let mut boundaries = BTreeMap::new();
    let mut add = |label: &str,
                   op: &str,
                   inputs: Vec<ProgramValueId>,
                   output: ProgramValueId,
                   shape: ProgramTensorSpec,
                   attributes|
     -> Result<(), VNextError> {
        nodes.push(ProgramNode {
            id: NodeId::new(format!("node.fast_h3.video_clip.{label}"))?,
            operation_id: OperationId::new(op)?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs,
            outputs: vec![output.clone()],
            attributes,
        });
        boundaries.insert(output.clone(), shape.element_type);
        specs.insert(output, shape);
        Ok(())
    };
    if prefix {
        add(
            "autocast_input",
            CAST_F32_TO_BF16_OPERATION_ID,
            vec![ProgramValueId::new(F::INPUT_ID)?],
            value("latent_bf16")?,
            tensor(rows, z, ElementType::Bf16),
            BTreeMap::new(),
        )?;
        add(
            "post_quant_dot",
            DENSE_LINEAR_BF16_OPERATION_ID,
            vec![
                value("latent_bf16")?,
                weight_value("post_quant_conv.weight")?,
            ],
            value("post_quant_dot")?,
            tensor(rows, z, ElementType::Bf16),
            attrs(&[
                ("in_features", SemanticValue::Unsigned(z)),
                ("out_features", SemanticValue::Unsigned(z)),
            ])?,
        )?;
        add(
            "post_quant_bias",
            CHANNEL_BIAS_ADD_BF16_OPERATION_ID,
            vec![
                value("post_quant_dot")?,
                weight_value("post_quant_conv.bias")?,
            ],
            value("post_quant")?,
            tensor(rows, z, ElementType::Bf16),
            BTreeMap::new(),
        )?;
        add(
            "proj_in",
            AFFINE_LINEAR_BF16_OPERATION_ID,
            vec![
                value("post_quant")?,
                weight_value("decoder.proj_in.weight")?,
                weight_value("decoder.proj_in.bias")?,
            ],
            value("patches_bf16")?,
            tensor(rows, h, ElementType::Bf16),
            attrs(&[
                ("in_features", SemanticValue::Unsigned(z)),
                ("out_features", SemanticValue::Unsigned(h)),
            ])?,
        )?;
        add(
            "promote_register_input",
            CAST_BF16_TO_F32_OPERATION_ID,
            vec![value("patches_bf16")?],
            value("patches_f32")?,
            tensor(rows, h, ElementType::F32),
            BTreeMap::new(),
        )?;
        add(
            "append_registers",
            BATCHED_SUFFIX_APPEND_F32_OPERATION_ID,
            vec![
                value("patches_f32")?,
                weight_value("decoder.register_tokens")?,
            ],
            value("embedded")?,
            tensor(all, h, ElementType::F32),
            attrs(&[
                ("batch", SemanticValue::Unsigned(c.grid[0])),
                ("zero_rows", SemanticValue::Unsigned(1)),
            ])?,
        )?;
    } else {
        let epsilon =
            CanonicalRational::new((a.norm_epsilon() * 1e12).round() as i64, 1_000_000_000_000)?;
        add(
            "norm_out",
            LAYER_NORM_F32_OPERATION_ID,
            vec![
                c.blocks.output_id()?,
                weight_value("decoder.norm_out.weight")?,
                weight_value("decoder.norm_out.bias")?,
            ],
            value("normalized")?,
            tensor(all, h, ElementType::F32),
            attrs(&[("epsilon", SemanticValue::Rational(epsilon))])?,
        )?;
        add(
            "autocast_proj_out",
            CAST_F32_TO_BF16_OPERATION_ID,
            vec![value("normalized")?],
            value("normalized_bf16")?,
            tensor(all, h, ElementType::Bf16),
            BTreeMap::new(),
        )?;
        add(
            "proj_out",
            AFFINE_LINEAR_BF16_OPERATION_ID,
            vec![
                value("normalized_bf16")?,
                weight_value("decoder.proj_out.weight")?,
                weight_value("decoder.proj_out.bias")?,
            ],
            value("all_patches")?,
            tensor(all, p, ElementType::Bf16),
            attrs(&[
                ("in_features", SemanticValue::Unsigned(h)),
                ("out_features", SemanticValue::Unsigned(p)),
            ])?,
        )?;
        add(
            "drop_registers",
            ROW_GATHER_BF16_OPERATION_ID,
            vec![value("all_patches")?],
            ProgramValueId::new(F::OUTPUT_ID)?,
            tensor(rows, p, ElementType::Bf16),
            attrs(&[
                ("batch", SemanticValue::Unsigned(c.grid[0])),
                ("column_start", SemanticValue::Unsigned(0)),
                ("column_count", SemanticValue::Unsigned(p)),
                (
                    "run_lengths",
                    SemanticValue::Integers(vec![
                        i64::try_from(c.patch_rows_per_clip()?).map_err(invalid)?
                    ]),
                ),
                ("source_starts", SemanticValue::Integers(vec![0])),
                (
                    "source_batch_strides",
                    SemanticValue::Integers(vec![
                        i64::try_from(c.sequence_rows()?).map_err(invalid)?
                    ]),
                ),
                ("source_row_strides", SemanticValue::Integers(vec![1])),
            ])?,
        )?;
    }
    if prefix {
        specs.insert(
            ProgramValueId::new(F::INPUT_ID)?,
            tensor(rows, z, ElementType::F32),
        );
    }
    let operations = nodes
        .iter()
        .map(|n| n.operation_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|operation_id| {
            let arithmetic = matches!(
                operation_id.as_str(),
                DENSE_LINEAR_BF16_OPERATION_ID
                    | AFFINE_LINEAR_BF16_OPERATION_ID
                    | LAYER_NORM_F32_OPERATION_ID
                    | CHANNEL_BIAS_ADD_BF16_OPERATION_ID
            );
            NumericalOperationContract {
                operation_id,
                version: ContractVersion::new(1, 0),
                multiplication_type: arithmetic.then_some(ElementType::F32),
                accumulation_type: arithmetic.then_some(ElementType::F32),
                staged_arithmetic: None,
                composite_arithmetic: None,
            }
        })
        .collect();
    Ok(Fragment {
        schema,
        nodes,
        weights,
        specs,
        operations,
        boundaries,
    })
}
