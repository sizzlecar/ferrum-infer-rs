use super::*;
use ferrum_interfaces::vnext::{NumericalExecutionPolicy, TypedFamilyRegistration};

fn source() -> Qwen35FamilyConfig {
    let mut c = super::super::q8act_attention_tests::source();
    c.hf_config["text_config"]["tie_word_embeddings"] = false.into();
    let mut head = c
        .weights
        .iter()
        .find(|w| w.role == "embed_tokens")
        .unwrap()
        .clone();
    head.role = "lm_head".into();
    head.external_name = "output.weight".into();
    head.source_encoding = FamilyWeightSourceEncoding::BlockQuantized(BlockQuantizationSpec {
        format_id: QuantizationFormatId::new("quantization.gguf.q6-k").unwrap(),
        logical_values_per_block: 256,
        bytes_per_block: 210,
    });
    c.weights.push(head);
    c
}

fn id() -> NumericalProfileId {
    NumericalProfileId::new(F32_MASTER_GGUF_Q8_HEAD_V1_NUMERICAL_PROFILE_ID).unwrap()
}

#[test]
fn q6_head_profile_require_changes_only_head_and_keeps_auto_body_and_kv_exact() {
    let registration = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap());
    let config = source();
    let definition = registration
        .define(&serde_json::to_value(&config).unwrap())
        .unwrap();
    let catalog = definition.numerical_profiles();
    let candidate = catalog
        .candidates(
            &NumericalExecutionPolicy::Require(id()),
            KvStorageFormat::F16,
        )
        .unwrap()[0];
    let body_id = NumericalProfileId::new(F32_MASTER_FFN_ATTENTION_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_ALL_ROWS_NUMERICAL_PROFILE_ID).unwrap();
    let body = catalog.resolve(&body_id).unwrap();
    let strict_id = NumericalProfileId::new(F32_MASTER_NUMERICAL_PROFILE_ID).unwrap();
    assert_eq!(
        catalog
            .candidates(&NumericalExecutionPolicy::Auto, KvStorageFormat::F16)
            .unwrap(),
        [catalog.resolve(&strict_id).unwrap()]
    );
    assert!(catalog
        .candidates(
            &NumericalExecutionPolicy::Require(id()),
            KvStorageFormat::Int8PerTokenHeadF32ScaleV1
        )
        .is_err());
    let mut restored = candidate.clone();
    restored.id = body.id.clone();
    let head = restored
        .operations
        .iter_mut()
        .find(|o| o.operation_id.as_str() == LAST_TOKEN_DENSE_LINEAR_Q6_MMQ_F32_OPERATION_ID)
        .unwrap();
    assert_eq!(head.staged_arithmetic, Some(Q6MmqF32Policy::new().staged()));
    *head = body
        .operations
        .iter()
        .find(|o| o.operation_id.as_str() == LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID)
        .unwrap()
        .clone();
    restored
        .operations
        .sort_by(|a, b| a.operation_id.cmp(&b.operation_id));
    assert_eq!(
        serde_json::to_vec(&restored).unwrap(),
        serde_json::to_vec(body).unwrap(),
        "every body operation, state, KV and output boundary is identical AA"
    );
    let prepared = registration.prepare(&definition, &id()).unwrap();
    let baseline = registration.prepare(&definition, &body_id).unwrap();
    assert_eq!(prepared.weight_schema(), baseline.weight_schema());
    assert_eq!(prepared.program().states(), baseline.program().states());
    candidate.validate_program(prepared.program()).unwrap();
    let nodes: Vec<_> = prepared
        .program()
        .blocks()
        .iter()
        .flat_map(|b| &b.nodes)
        .collect();
    let old: Vec<_> = baseline
        .program()
        .blocks()
        .iter()
        .flat_map(|b| &b.nodes)
        .collect();
    assert_eq!(nodes.len(), old.len());
    for (new, old) in nodes.into_iter().zip(old) {
        if old.operation_id.as_str() == LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID {
            assert_eq!(
                new.operation_id.as_str(),
                LAST_TOKEN_DENSE_LINEAR_Q6_MMQ_F32_OPERATION_ID
            );
            let mut restored = new.clone();
            restored.operation_id = old.operation_id.clone();
            assert_eq!(
                serde_json::to_value(restored).unwrap(),
                serde_json::to_value(old).unwrap()
            );
        } else {
            assert_eq!(
                serde_json::to_value(new).unwrap(),
                serde_json::to_value(old).unwrap()
            );
        }
    }
}

#[test]
fn q6_head_profile_eligibility_follows_consumed_head_and_exact_block_abi() {
    let mut c = source();
    let text = Qwen35FamilyProvider::text_config(&c).unwrap();
    assert!(eligible(&c, &text));
    let i = c.weights.iter().position(|w| w.role == "lm_head").unwrap();
    let saved = c.weights[i].clone();
    c.weights[i].source_encoding = FamilyWeightSourceEncoding::Dense {
        element_type: ElementType::F16,
    };
    assert!(!eligible(&c, &text));
    c.weights[i] = saved.clone();
    let FamilyWeightSourceEncoding::BlockQuantized(b) = &mut c.weights[i].source_encoding else {
        unreachable!()
    };
    b.bytes_per_block = 209;
    assert!(!eligible(&c, &text));
    c.weights[i] = saved;
    c.weights[i].dimensions[1] = 255;
    assert!(!eligible(&c, &text));
    // An eligible but unconsumed embedding must not qualify a dense lm_head.
    c = source();
    let q6 = c
        .weights
        .iter()
        .find(|w| w.role == "lm_head")
        .unwrap()
        .source_encoding
        .clone();
    c.weights
        .iter_mut()
        .find(|w| w.role == "embed_tokens")
        .unwrap()
        .source_encoding = q6;
    c.weights
        .iter_mut()
        .find(|w| w.role == "lm_head")
        .unwrap()
        .source_encoding = FamilyWeightSourceEncoding::Dense {
        element_type: ElementType::F16,
    };
    assert!(!eligible(&c, &text));
    c.weights.retain(|w| w.role != "lm_head");
    c.hf_config["text_config"]["tie_word_embeddings"] = true.into();
    let text = Qwen35FamilyProvider::text_config(&c).unwrap();
    assert!(eligible(&c, &text), "actual tied embedding head");
}

#[test]
fn q6_head_profile_rejects_forged_policy_and_preserves_aa_under_old_identity() {
    let c = source();
    let provider = Qwen35FamilyProvider::new().unwrap();
    let catalog = provider.numerical_profiles(&c).unwrap();
    let selected = catalog.resolve(&id()).unwrap();
    let mut tampered = selected.clone();
    tampered
        .operations
        .iter_mut()
        .find(|o| o.operation_id.as_str() == LAST_TOKEN_DENSE_LINEAR_Q6_MMQ_F32_OPERATION_ID)
        .unwrap()
        .staged_arithmetic = None;
    assert!(provider.semantic_program(&c, &tampered).is_err());
    let mut forged_old = selected.clone();
    forged_old.id = NumericalProfileId::new(F32_MASTER_FFN_ATTENTION_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_ALL_ROWS_NUMERICAL_PROFILE_ID).unwrap();
    assert!(provider.semantic_program(&c, &forged_old).is_err());
    let mut missing_kv = selected.clone();
    missing_kv.kv_storage.clear();
    assert!(provider.semantic_program(&c, &missing_kv).is_err());
}
