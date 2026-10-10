use super::*;
use ferrum_interfaces::vnext::{NumericalExecutionPolicy, TypedFamilyRegistration};

fn source() -> Qwen35FamilyConfig {
    let mut config = super::super::q8act_attention_tests::source();
    config.hf_config["text_config"]["tie_word_embeddings"] = false.into();
    let encoding = FamilyWeightSourceEncoding::BlockQuantized(BlockQuantizationSpec {
        format_id: QuantizationFormatId::new("quantization.gguf.q6-k").unwrap(),
        logical_values_per_block: 256,
        bytes_per_block: 210,
    });
    let mut head = config
        .weights
        .iter()
        .find(|w| w.role == "embed_tokens")
        .unwrap()
        .clone();
    head.role = "lm_head".into();
    head.external_name = "output.weight".into();
    head.source_encoding = encoding.clone();
    config.weights.push(head);
    config
        .weights
        .iter_mut()
        .find(|w| w.role == "self_attn_k")
        .unwrap()
        .source_encoding = encoding;
    config
}

fn id() -> NumericalProfileId {
    NumericalProfileId::new(F32_MASTER_Q6_HEAD_ATTENTION_M8_Q6_F16_MMQ_V1_NUMERICAL_PROFILE_ID)
        .unwrap()
}

#[test]
fn q6_f16_require_preserves_auto_head_weights_state_and_program_structure() {
    let registration = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap());
    let config = source();
    let definition = registration
        .define(&serde_json::to_value(&config).unwrap())
        .unwrap();
    let catalog = definition.numerical_profiles();
    let old_id = NumericalProfileId::new(
        F32_MASTER_Q6_HEAD_ATTENTION_M8_MMQ_GEOMETRY_V1_NUMERICAL_PROFILE_ID,
    )
    .unwrap();
    let old = catalog.resolve(&old_id).unwrap();
    let new = catalog
        .candidates(
            &NumericalExecutionPolicy::Require(id()),
            KvStorageFormat::F16,
        )
        .unwrap()[0];
    assert_eq!(
        catalog
            .candidates(&NumericalExecutionPolicy::Auto, KvStorageFormat::F16)
            .unwrap(),
        [catalog
            .resolve(&NumericalProfileId::new(F32_MASTER_NUMERICAL_PROFILE_ID).unwrap())
            .unwrap()]
    );
    assert!(catalog
        .candidates(
            &NumericalExecutionPolicy::Require(id()),
            KvStorageFormat::Int8PerTokenHeadF32ScaleV1
        )
        .is_err());
    assert_ne!(new.fingerprint().unwrap(), old.fingerprint().unwrap());
    let pairs = [
        (
            UpstreamMarkerV2Profile::SwiGluQ6F16,
            UpstreamMarkerV2Profile::SwiGluExtraAllRows,
        ),
        (
            UpstreamMarkerV2Profile::GatedDeltaQ6F16,
            UpstreamMarkerV2Profile::GatedDeltaM8Geometry,
        ),
        (
            UpstreamMarkerV2Profile::CausalQ6F16,
            UpstreamMarkerV2Profile::CausalM8Geometry,
        ),
    ];
    let mut normalized = new.clone();
    normalized.id = old.id.clone();
    for (candidate, prior) in pairs {
        let target = normalized
            .operations
            .iter_mut()
            .find(|op| op.operation_id.as_str() == candidate.operation_id())
            .unwrap();
        *target = old
            .operations
            .iter()
            .find(|op| op.operation_id.as_str() == prior.operation_id())
            .unwrap()
            .clone();
    }
    normalized
        .operations
        .sort_by(|a, b| a.operation_id.cmp(&b.operation_id));
    assert_eq!(
        normalized, *old,
        "head, KV, recurrent state and unrelated numerical contracts remain exact"
    );
    let prepared = registration.prepare(&definition, &id()).unwrap();
    let baseline = registration.prepare(&definition, &old_id).unwrap();
    assert_eq!(prepared.weight_schema(), baseline.weight_schema());
    assert_eq!(prepared.program().states(), baseline.program().states());
    new.validate_program(prepared.program()).unwrap();
    let nodes: Vec<_> = prepared
        .program()
        .blocks()
        .iter()
        .flat_map(|block| &block.nodes)
        .collect();
    let old_nodes: Vec<_> = baseline
        .program()
        .blocks()
        .iter()
        .flat_map(|block| &block.nodes)
        .collect();
    assert_eq!(nodes.len(), old_nodes.len());
    for (node, prior) in nodes.into_iter().zip(old_nodes) {
        let mut normalized = node.clone();
        if let Some((_, old)) = pairs
            .iter()
            .find(|(new, _)| node.operation_id.as_str() == new.operation_id())
        {
            normalized.operation_id = operation_id(old.operation_id()).unwrap();
        }
        assert_eq!(
            serde_json::to_value(normalized).unwrap(),
            serde_json::to_value(prior).unwrap()
        );
    }
}

#[test]
fn q6_f16_profile_rejects_substituted_body_and_missing_head_contract() {
    let provider = Qwen35FamilyProvider::new().unwrap();
    let config = source();
    let catalog = provider.numerical_profiles(&config).unwrap();
    let new = catalog.resolve(&id()).unwrap();
    let old_id = NumericalProfileId::new(
        F32_MASTER_Q6_HEAD_ATTENTION_M8_MMQ_GEOMETRY_V1_NUMERICAL_PROFILE_ID,
    )
    .unwrap();
    let mut old_body = catalog.resolve(&old_id).unwrap().clone();
    old_body.id = id();
    assert!(provider.semantic_program(&config, &old_body).is_err());
    let mut old_identity = new.clone();
    old_identity.id = old_id;
    assert!(provider.semantic_program(&config, &old_identity).is_err());
    let mut missing_head = new.clone();
    missing_head
        .operations
        .iter_mut()
        .find(|op| {
            op.operation_id.as_str()
                == ferrum_interfaces::vnext::LAST_TOKEN_DENSE_LINEAR_Q6_MMQ_F32_OPERATION_ID
        })
        .unwrap()
        .staged_arithmetic = None;
    assert!(provider.semantic_program(&config, &missing_head).is_err());
}
