//! Bind source-declared activation bases to each physical projection leaf.

use ferrum_quantization::gguf::source::{hadamard_sign_component, hadamard_transform_spec};
use ferrum_quantization::gguf::{GgufHadamardDirection, GgufHadamardSigns};

use super::*;

#[cfg(test)]
mod real_source;

pub(super) fn validate(
    config: &Qwen35FamilyConfig,
    text: &Qwen35TextConfig,
) -> Result<(), VNextError> {
    let Some(metadata) = &config.gguf_hadamard else {
        return Ok(());
    };
    if config.weight_format != FamilyWeightFormat::GgufNative
        || config.recurrent_weight_abi != RecurrentWeightAbi::NegativeRateInterleaved
    {
        return Err(invalid_config(
            "gguf_hadamard",
            "activation transforms require the native GGUF recurrent ABI",
        ));
    }
    if text.is_moe() {
        return Err(invalid_config(
            "gguf_hadamard",
            "Hadamard-routed expert execution is not yet supported",
        ));
    }
    let key_heads = u32::try_from(text.linear_attention.num_key_heads)
        .map_err(|_| invalid_config("gguf_hadamard", "key head count exceeds u32"))?;
    let value_heads = u32::try_from(text.linear_attention.num_value_heads)
        .map_err(|_| invalid_config("gguf_hadamard", "value head count exceeds u32"))?;
    metadata
        .validate_for_tensors(
            gguf_architecture(text),
            config
                .weights
                .iter()
                .map(|weight| (weight.external_name.as_str(), weight.dimensions.as_slice())),
            Some((key_heads, value_heads)),
        )
        .map_err(|error| invalid_config("gguf_hadamard", error.to_string()))?;
    for (name, transform) in metadata.weights() {
        let weight = config
            .weights
            .iter()
            .find(|weight| &weight.external_name == name)
            .ok_or_else(|| {
                invalid_config(
                    "gguf_hadamard",
                    format!("declared transform {name:?} has no consumed family weight"),
                )
            })?;
        let valid_role = match transform.direction() {
            GgufHadamardDirection::AfterEmbeddingLookup => weight.role == "embed_tokens",
            GgufHadamardDirection::BeforeMatmul => matches!(
                weight.role.as_str(),
                "lm_head"
                    | "linear_attn_qkv"
                    | "linear_attn_z"
                    | "linear_attn_out"
                    | "self_attn_q"
                    | "self_attn_k"
                    | "self_attn_v"
                    | "self_attn_o"
                    | "mlp_gate"
                    | "mlp_up"
                    | "mlp_down"
            ),
        };
        if !valid_role {
            return Err(invalid_config(
                "gguf_hadamard",
                format!(
                    "role {:?} cannot consume the declared transform",
                    weight.role
                ),
            ));
        }
        if text.tie_word_embeddings && weight.role == "embed_tokens" {
            return Err(invalid_config("gguf_hadamard", "inverse latent embeddings cannot also serve as a tied output projection; independent output weights are required"));
        }
    }
    Ok(())
}

pub(super) fn apply_schema(
    config: &Qwen35FamilyConfig,
    schema: &mut WeightSchema,
) -> Result<(), VNextError> {
    let Some(metadata) = &config.gguf_hadamard else {
        return Ok(());
    };
    let text = Qwen35FamilyProvider::text_config(config)?;
    validate(config, &text)?;
    let names: BTreeMap<_, _> = schema
        .components
        .iter()
        .map(|component| {
            let [name] = component.external_names.as_slice() else {
                return Err(invalid_config(
                    "gguf_hadamard",
                    "GGUF projection components must name exactly one source tensor",
                ));
            };
            Ok((component.id.clone(), name.clone()))
        })
        .collect::<Result<_, VNextError>>()?;
    let mut consumed = BTreeSet::new();
    for tensor in &mut schema.tensors {
        wrap_leaves(&mut tensor.physical_layout, metadata, &names, &mut consumed)?;
    }
    if consumed != metadata.weights().keys().cloned().collect() {
        return Err(invalid_config(
            "gguf_hadamard",
            "not every source transform was consumed by a logical weight",
        ));
    }
    if matches!(metadata.signs(), GgufHadamardSigns::Explicit(_)) {
        let widths: BTreeSet<_> = metadata
            .weights()
            .values()
            .map(|weight| weight.input_width())
            .collect();
        for width in widths {
            schema.components.push(hadamard_sign_component(width)?);
        }
    }
    // The recursive layout now expresses additional physical source semantics.
    // Unrotated models retain their existing version and identity unchanged.
    schema.version = ContractVersion::new(1, 3);
    Ok(())
}

fn wrap_leaves(
    layout: &mut PhysicalWeightLayout,
    metadata: &GgufHadamard,
    names: &BTreeMap<WeightId, String>,
    consumed: &mut BTreeSet<String>,
) -> Result<(), VNextError> {
    let component_id = match layout {
        PhysicalWeightLayout::Composite { parts } => {
            for part in parts {
                wrap_leaves(&mut part.layout, metadata, names, consumed)?;
            }
            return Ok(());
        }
        PhysicalWeightLayout::Dense { component_id } => component_id,
        PhysicalWeightLayout::Stored { component } => &component.component_id,
        PhysicalWeightLayout::BlockQuantized { blocks, .. } => &blocks.component_id,
        _ => {
            return Err(invalid_config(
                "gguf_hadamard",
                "GGUF transform requires a native projection leaf",
            ))
        }
    };
    let name = names
        .get(component_id)
        .ok_or_else(|| invalid_config("gguf_hadamard", "missing projection source name"))?;
    if let Some(transform) = hadamard_transform_spec(metadata, name)? {
        if !consumed.insert(name.clone()) {
            return Err(invalid_config(
                "gguf_hadamard",
                "source transform would be applied more than once",
            ));
        }
        *layout = PhysicalWeightLayout::Hadamard {
            values: Box::new(layout.clone()),
            transform,
        };
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::vnext::{HadamardApplication, NumericalExecutionPolicy};

    fn config() -> Qwen35FamilyConfig {
        let mut config = super::super::tests::test_dense_gguf_config();
        // This fixture transforms only independent projection components.
        let mut weights = serde_json::Map::new();
        for weight in &config.weights {
            if weight.layer_index == Some(0)
                && matches!(
                    weight.role.as_str(),
                    "linear_attn_qkv" | "linear_attn_z" | "linear_attn_out" | "mlp_gate" | "mlp_up"
                )
            {
                let width = *weight.dimensions.last().unwrap();
                weights.insert(
                    weight.external_name.clone(),
                    serde_json::json!({
                        "direction": "before_matmul", "input_width": width,
                        "gdn_permutation": if weight.role == "linear_attn_out" {
                            serde_json::json!({"head_dim":4,"key_heads":2,"repeats":1})
                        } else { Value::Null }
                    }),
                );
            }
        }
        config.gguf_hadamard = Some(serde_json::from_value(serde_json::json!({
            "version":1,"block_size":4,"signs":{"mode":"explicit","by_width":{"8":vec![1_i8;8],"16":vec![-1_i8;16]}},
            "gdn_v_grouped":true,"weights":weights
        })).unwrap());
        config
    }

    #[test]
    fn hadamard_packed_gdn_rotates_only_qkv_and_z_and_reuses_signs() {
        let config = config();
        let provider = Qwen35FamilyProvider::new().unwrap();
        provider.validate_typed_config(&config).unwrap();
        let schema = provider.weight_schema(&config).unwrap();
        schema.validate(provider.family_id()).unwrap();
        let packed = schema
            .tensors
            .iter()
            .find(|tensor| {
                tensor.id
                    == packed_linear_attention_weight_id(0, PACKED_LINEAR_ATTN_QKVZBA_ROLE).unwrap()
            })
            .unwrap();
        let PhysicalWeightLayout::Composite { parts } = &packed.physical_layout else {
            panic!("expected per-source projections");
        };
        assert!(matches!(
            parts[0].layout.as_ref(),
            PhysicalWeightLayout::Hadamard { .. }
        ));
        assert!(matches!(
            parts[1].layout.as_ref(),
            PhysicalWeightLayout::Hadamard { .. }
        ));
        assert!(!matches!(
            parts[2].layout.as_ref(),
            PhysicalWeightLayout::Hadamard { .. }
        ));
        assert!(!matches!(
            parts[3].layout.as_ref(),
            PhysicalWeightLayout::Hadamard { .. }
        ));
        let signs: Vec<_> = schema
            .components
            .iter()
            .filter(|component| component.role == WeightComponentRole::TransformSigns)
            .collect();
        assert_eq!(signs.len(), 2);
        assert_eq!(
            signs
                .iter()
                .map(|component| component.physical_bytes().unwrap())
                .sum::<u64>(),
            (8 + 16) * 4
        );
        let output = required_weight(&config, Some(0), "linear_attn_out").unwrap();
        let output = schema
            .tensors
            .iter()
            .find(|tensor| tensor.id == weight_id(output).unwrap())
            .unwrap();
        let PhysicalWeightLayout::Hadamard { transform, .. } = &output.physical_layout else {
            panic!("missing GDN output transform");
        };
        assert!(matches!(
            transform.application,
            HadamardApplication::BeforeMatmul {
                input_permutation: Some(_)
            }
        ));
    }

    #[test]
    fn hadamard_family_roundtrip_binds_sign_identity_and_rejects_tampered_geometry() {
        let config = config();
        let provider = Qwen35FamilyProvider::new().unwrap();
        let raw = serde_json::to_value(&config).unwrap();
        assert_eq!(provider.parse_config(&raw).unwrap(), config);
        let registration = TypedFamilyRegistration::new(provider);
        let before = registration.define(&raw).unwrap();
        let mut changed_sign = raw.clone();
        changed_sign["gguf_hadamard"]["signs"]["by_width"]["16"][0] = serde_json::json!(1);
        let after = registration.define(&changed_sign).unwrap();
        assert_ne!(before.fingerprint().unwrap(), after.fingerprint().unwrap());
        let mut malformed = raw;
        malformed["gguf_hadamard"]["weights"]["blk.0.ssm_out.weight"]["gdn_permutation"]
            ["repeats"] = serde_json::json!(2);
        assert!(registration.define(&malformed).is_err());
    }

    #[test]
    fn hadamard_family_rejects_unconsumed_declarations_and_tied_latent_embedding() {
        let provider = Qwen35FamilyProvider::new().unwrap();
        let mut raw = serde_json::to_value(config()).unwrap();
        raw["gguf_hadamard"]["weights"]["blk.999.attn_q.weight"] = serde_json::json!({"direction":"before_matmul","input_width":16,"gdn_permutation":null});
        assert!(provider.parse_config(&raw).is_err());
        let mut raw = serde_json::to_value(config()).unwrap();
        raw["gguf_hadamard"]["weights"]["token_embd.weight"] = serde_json::json!({"direction":"after_embedding_lookup","input_width":16,"gdn_permutation":null});
        assert!(provider
            .parse_config(&raw)
            .unwrap_err()
            .to_string()
            .contains("tied output projection"));
    }

    #[test]
    fn hadamard_family_offers_only_f16_kv_and_cannot_reuse_an_int8_profile() {
        let config = config();
        let provider = Qwen35FamilyProvider::new().unwrap();
        let profiles = provider.numerical_profiles(&config).unwrap();
        assert!(profiles
            .candidates(&NumericalExecutionPolicy::Auto, KvStorageFormat::F16)
            .is_ok());
        assert!(profiles
            .candidates(
                &NumericalExecutionPolicy::Auto,
                KvStorageFormat::Int8PerTokenHeadF32ScaleV1
            )
            .is_err());
        assert!(profiles
            .candidates(
                &NumericalExecutionPolicy::Require(
                    F32_MASTER_INT8_KV_NUMERICAL_PROFILE_ID
                        .to_owned()
                        .try_into()
                        .unwrap()
                ),
                KvStorageFormat::Int8PerTokenHeadF32ScaleV1
            )
            .is_err());
        let mut plain = config.clone();
        plain.gguf_hadamard = None;
        let plain_profiles = provider.numerical_profiles(&plain).unwrap();
        let int8 = plain_profiles
            .resolve(
                &F32_MASTER_INT8_KV_NUMERICAL_PROFILE_ID
                    .to_owned()
                    .try_into()
                    .unwrap(),
            )
            .unwrap();
        assert!(provider
            .semantic_program(&config, int8)
            .unwrap_err()
            .to_string()
            .contains("requires F16 KV"));
        assert!(plain_profiles
            .candidates(
                &NumericalExecutionPolicy::Auto,
                KvStorageFormat::Int8PerTokenHeadF32ScaleV1
            )
            .is_ok());
    }
}
