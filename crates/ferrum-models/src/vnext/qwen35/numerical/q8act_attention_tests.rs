use super::*;
use ferrum_interfaces::vnext::{NumericalExecutionPolicy, TypedFamilyRegistration};

fn source() -> Qwen35FamilyConfig {
    let mut config = super::super::tests::test_dense_gguf_config();
    config.hf_config["text_config"]["hidden_size"] = 256.into();
    let text = Qwen35FamilyProvider::text_config(&config).unwrap();
    for weight in &mut config.weights {
        // All original K=hidden and hidden-vector dimensions are updated. The
        // attention heads/recurrent state dimensions intentionally stay small.
        match weight.role.as_str() {
            "embed_tokens" | "lm_head" | "mlp_gate" | "mlp_up" | "linear_attn_qkv"
            | "linear_attn_z" | "linear_attn_a" | "linear_attn_b" | "self_attn_q"
            | "self_attn_k" | "self_attn_v" => weight.dimensions[1] = 256,
            "mlp_down"
            | "linear_attn_out"
            | "self_attn_o"
            | "final_norm"
            | "input_layernorm"
            | "post_attention_layernorm" => weight.dimensions[0] = 256,
            _ => {}
        }
        let (format, bytes) = match weight.role.as_str() {
            "linear_attn_qkv" => ("quantization.gguf.q4-k", 144),
            "linear_attn_z" => ("quantization.gguf.q5-k", 176),
            "self_attn_q" => ("quantization.gguf.iq4-xs", 136),
            _ => continue,
        };
        weight.source_encoding =
            FamilyWeightSourceEncoding::BlockQuantized(BlockQuantizationSpec {
                format_id: QuantizationFormatId::new(format).unwrap(),
                logical_values_per_block: 256,
                bytes_per_block: bytes,
            });
    }
    assert!(q8act_attention_eligible(&config, &text));
    config
}

#[test]
fn q8act_attention_require_preserves_auto_and_only_overrides_declared_operations() {
    let registration = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap());
    let config = source();
    let definition = registration
        .define(&serde_json::to_value(&config).unwrap())
        .unwrap();
    let catalog = definition.numerical_profiles();
    let strict_id = NumericalProfileId::new(F32_MASTER_NUMERICAL_PROFILE_ID).unwrap();
    let candidate_id = NumericalProfileId::new(
        F32_MASTER_FFN_ATTENTION_Q4K_Q5K_IQ4XS_Q8ACT_G32_NUMERICAL_PROFILE_ID,
    )
    .unwrap();
    let strict = catalog.resolve(&strict_id).unwrap();
    let selected = catalog
        .candidates(
            &NumericalExecutionPolicy::Require(candidate_id.clone()),
            KvStorageFormat::F16,
        )
        .unwrap()[0];
    assert_eq!(
        catalog
            .candidates(&NumericalExecutionPolicy::Auto, KvStorageFormat::F16)
            .unwrap(),
        [strict]
    );
    assert!(catalog
        .candidates(
            &NumericalExecutionPolicy::Require(candidate_id.clone()),
            KvStorageFormat::Int8PerTokenHeadF32ScaleV1
        )
        .is_err());
    let mut restored = selected.clone();
    restored.id = strict.id.clone();
    for (new, old) in [
        (
            Q8ActAttentionProfile::GatedDelta.operation_id(),
            GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID,
        ),
        (
            Q8ActAttentionProfile::Causal.operation_id(),
            CAUSAL_PAGED_ATTENTION_F32_MASTER_OPERATION_ID,
        ),
        (
            Q8ActSwiGluProfile::Q4KQ5KIq4Xs.operation_id(),
            DENSE_SWIGLU_OPERATION_ID,
        ),
    ] {
        let target = restored
            .operations
            .iter_mut()
            .find(|op| op.operation_id.as_str() == new)
            .unwrap();
        *target = strict
            .operations
            .iter()
            .find(|op| op.operation_id.as_str() == old)
            .unwrap()
            .clone();
    }
    restored
        .operations
        .sort_by(|a, b| a.operation_id.cmp(&b.operation_id));
    assert_eq!(
        &restored, strict,
        "states, boundaries, head and nonprojection operations remain strict"
    );
    let prepared = registration.prepare(&definition, &candidate_id).unwrap();
    let baseline = registration.prepare(&definition, &strict_id).unwrap();
    assert_eq!(prepared.weight_schema(), baseline.weight_schema());
    assert_eq!(prepared.program().states(), baseline.program().states());
    selected.validate_program(prepared.program()).unwrap();
    let mut missing_kv = selected.clone();
    missing_kv.kv_storage.clear();
    assert!(missing_kv
        .validate_program(prepared.program())
        .unwrap_err()
        .to_string()
        .contains("is missing from the numerical storage contract"));
}

#[test]
fn q8act_attention_rejects_profile_mutation_and_unconsumed_source_roles() {
    let provider = Qwen35FamilyProvider::new().unwrap();
    let mut config = source();
    let id = NumericalProfileId::new(
        F32_MASTER_FFN_ATTENTION_Q4K_Q5K_IQ4XS_Q8ACT_G32_NUMERICAL_PROFILE_ID,
    )
    .unwrap();
    let mut profile = provider
        .numerical_profiles(&config)
        .unwrap()
        .resolve(&id)
        .unwrap()
        .clone();
    let operation = profile
        .operations
        .iter_mut()
        .find(|op| op.operation_id.as_str() == Q8ActAttentionProfile::Causal.operation_id())
        .unwrap();
    operation.composite_arithmetic.as_mut().unwrap().projections[0].leaves[0]
        .shape
        .minimum_output_features = 2;
    profile.validate().unwrap();
    assert!(provider.semantic_program(&config, &profile).is_err());
    for weight in &mut config.weights {
        weight.source_encoding = FamilyWeightSourceEncoding::Dense {
            element_type: ElementType::F16,
        };
    }
    let text = Qwen35FamilyProvider::text_config(&config).unwrap();
    assert!(!q8act_attention_eligible(&config, &text));
    let mut unused = config
        .weights
        .iter()
        .find(|w| w.role == "self_attn_q")
        .unwrap()
        .clone();
    unused.source_encoding = FamilyWeightSourceEncoding::BlockQuantized(BlockQuantizationSpec {
        format_id: QuantizationFormatId::new("quantization.gguf.q4-k").unwrap(),
        logical_values_per_block: 256,
        bytes_per_block: 144,
    });
    unused.layer_index = Some(0); // layer0 is recurrent, this causal role is not consumed.
    config.weights.push(unused.clone());
    assert!(!q8act_attention_eligible(&config, &text));
    config.weights.last_mut().unwrap().layer_index = Some(text.num_hidden_layers as u32);
    assert!(!q8act_attention_eligible(&config, &text));
    assert!(provider
        .numerical_profiles(&config)
        .unwrap()
        .resolve(&id)
        .is_err());
}
