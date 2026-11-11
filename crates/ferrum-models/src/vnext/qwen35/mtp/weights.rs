use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct MtpTensor {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub element_type: ElementType,
}

/// The checkpoint's single MTP block; external names are model-format roles,
/// never inferred from a repository name or selected hardware.
fn inventory(t: &Qwen35TextConfig) -> Vec<(&'static str, Vec<u64>, bool)> {
    let h = t.hidden_size as u64;
    let i = t.dense_intermediate_size.unwrap_or(0) as u64;
    vec![
        ("fc", vec![h, 2 * h], false),
        ("pre_fc_norm_embedding", vec![h], true),
        ("pre_fc_norm_hidden", vec![h], true),
        ("norm", vec![h], true),
        ("layers.0.input_layernorm", vec![h], true),
        ("layers.0.post_attention_layernorm", vec![h], true),
        (
            "layers.0.self_attn.q_proj",
            vec![t.full_attention_q_proj_total_dim() as u64, h],
            false,
        ),
        (
            "layers.0.self_attn.k_proj",
            vec![t.full_attention_kv_total_dim() as u64, h],
            false,
        ),
        (
            "layers.0.self_attn.v_proj",
            vec![t.full_attention_kv_total_dim() as u64, h],
            false,
        ),
        (
            "layers.0.self_attn.o_proj",
            vec![h, t.full_attention_query_total_dim() as u64],
            false,
        ),
        ("layers.0.self_attn.q_norm", vec![t.head_dim as u64], true),
        ("layers.0.self_attn.k_norm", vec![t.head_dim as u64], true),
        ("layers.0.mlp.gate_proj", vec![i, h], false),
        ("layers.0.mlp.up_proj", vec![i, h], false),
        ("layers.0.mlp.down_proj", vec![h, i], false),
    ]
}

pub(super) fn read_header(
    t: &Qwen35TextConfig,
    archive: &SafetensorsArchive,
) -> Result<Vec<MtpTensor>, VNextError> {
    let mut tensors = Vec::new();
    for (role, _, _) in inventory(t) {
        let name = format!("mtp.{role}.weight");
        let tensor = archive
            .tensor(&name)
            .map_err(|e| invalid_config("mtp.weights", e.to_string()))?;
        tensors.push(MtpTensor {
            name,
            dimensions: tensor.shape().to_vec(),
            element_type: tensor
                .element_type()
                .ok_or_else(|| invalid_config("mtp.weights", "unsupported source dtype"))?,
        });
    }
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    validate(t, &tensors)?;
    Ok(tensors)
}

pub(super) fn validate(t: &Qwen35TextConfig, tensors: &[MtpTensor]) -> Result<(), VNextError> {
    let expected = inventory(t);
    if tensors.len() != expected.len() || tensors.windows(2).any(|w| w[0].name >= w[1].name) {
        return Err(invalid_config(
            "mtp.weights",
            "MTP requires exactly the canonical sorted 15 tensor identities",
        ));
    }
    for (role, dimensions, _) in expected {
        let name = format!("mtp.{role}.weight");
        let source = tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| invalid_config("mtp.weights", format!("missing {name}")))?;
        if source.dimensions != dimensions
            || !matches!(
                source.element_type,
                ElementType::F16 | ElementType::Bf16 | ElementType::F32
            )
        {
            return Err(invalid_config(
                "mtp.weights",
                format!("invalid dense source shape/type for {name}"),
            ));
        }
    }
    Ok(())
}

pub(super) fn value(role: &str) -> Result<ProgramValueId, VNextError> {
    value_id(format!("value.weight.mtp.{role}"))
}

pub(super) fn union_schema(c: &Qwen35MtpFamilyConfig) -> Result<WeightSchema, VNextError> {
    let mut schema = safetensors_weight_schema(&c.main)?;
    schema.layout_id = WeightLayoutId::new("weight-layout.qwen3_5.mtp.union_dense_v1")?;
    let t = Qwen35FamilyProvider::text_config(&c.main)?;
    for (role, mut dimensions, norm) in inventory(&t) {
        if role == "layers.0.mlp.up_proj" {
            continue;
        }
        let (logical, names) = if role == "layers.0.mlp.gate_proj" {
            dimensions.insert(0, 2);
            (
                "layers.0.mlp.gate_up",
                vec![
                    "mtp.layers.0.mlp.gate_proj.weight".to_owned(),
                    "mtp.layers.0.mlp.up_proj.weight".to_owned(),
                ],
            )
        } else {
            (role, vec![format!("mtp.{role}.weight")])
        };
        let component = WeightId::new(format!("component.mtp.{logical}"))?;
        schema.components.push(WeightComponentSpec {
            id: component.clone(),
            role: WeightComponentRole::Values,
            external_names: names,
            dimensions: dimensions.clone(),
            required: true,
            encoding: if norm {
                dense_weight_encoding("final_norm")?
            } else {
                WeightEncoding::Dense {
                    element_type: ElementType::F16,
                }
            },
        });
        schema.tensors.push(WeightTensorSpec {
            id: WeightId::new(format!("weight.mtp.{logical}"))?,
            dimensions,
            logical_element_type: ElementType::F16,
            physical_layout: PhysicalWeightLayout::Dense {
                component_id: component,
            },
            required: true,
        });
    }
    Ok(schema)
}

pub(super) fn union_references(
    c: &Qwen35MtpFamilyConfig,
    main: &ModelProgram,
) -> Result<Vec<WeightReference>, VNextError> {
    let mut references = main.weights().to_vec();
    for tensor in union_schema(c)?
        .tensors
        .into_iter()
        .filter(|w| w.id.as_str().starts_with("weight.mtp."))
    {
        references.push(WeightReference {
            value_id: value(
                tensor
                    .id
                    .as_str()
                    .strip_prefix("weight.mtp.")
                    .expect("filtered"),
            )?,
            weight_id: tensor.id,
            tensor: tensor_spec(tensor.dimensions, tensor.logical_element_type),
        });
    }
    references.sort_by(|a, b| a.weight_id.cmp(&b.weight_id));
    Ok(references)
}
