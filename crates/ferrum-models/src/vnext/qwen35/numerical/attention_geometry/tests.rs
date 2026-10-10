use super::*;
use ferrum_interfaces::vnext::{
    NumericalArithmeticStage, NumericalExecutionPolicy, TypedFamilyRegistration,
    UpstreamProjectionGeometrySelection, LAST_TOKEN_DENSE_LINEAR_Q6_MMQ_F32_OPERATION_ID,
};

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
    NumericalProfileId::new(F32_MASTER_Q6_HEAD_ATTENTION_M8_MMQ_GEOMETRY_V1_NUMERICAL_PROFILE_ID)
        .unwrap()
}

fn pairs() -> [(UpstreamMarkerV2Profile, UpstreamMarkerV2Profile); 2] {
    [
        (
            UpstreamMarkerV2Profile::GatedDeltaM8Geometry,
            UpstreamMarkerV2Profile::GatedDeltaExtraAllRows,
        ),
        (
            UpstreamMarkerV2Profile::CausalM8Geometry,
            UpstreamMarkerV2Profile::CausalExtraAllRows,
        ),
    ]
}

#[test]
fn attention_geometry_profile_require_preserves_auto_ffn_head_state_and_weight_contracts() {
    let registration = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap());
    let config = source();
    let definition = registration
        .define(&serde_json::to_value(&config).unwrap())
        .unwrap();
    let catalog = definition.numerical_profiles();
    let old_id = NumericalProfileId::new(F32_MASTER_GGUF_Q8_HEAD_V1_NUMERICAL_PROFILE_ID).unwrap();
    let old = catalog.resolve(&old_id).unwrap();
    let new = catalog
        .candidates(
            &NumericalExecutionPolicy::Require(id()),
            KvStorageFormat::F16,
        )
        .unwrap()[0];
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
    assert_ne!(new.fingerprint().unwrap(), old.fingerprint().unwrap());
    assert_eq!(
        serde_json::from_str::<NumericalExecutionProfile>(&serde_json::to_string(new).unwrap())
            .unwrap(),
        *new
    );
    let mut restored = new.clone();
    restored.id = old.id.clone();
    for (new_kind, old_kind) in pairs() {
        let target = restored
            .operations
            .iter_mut()
            .find(|o| o.operation_id.as_str() == new_kind.operation_id())
            .unwrap();
        *target = old
            .operations
            .iter()
            .find(|o| o.operation_id.as_str() == old_kind.operation_id())
            .unwrap()
            .clone();
    }
    restored
        .operations
        .sort_by(|a, b| a.operation_id.cmp(&b.operation_id));
    assert_eq!(
        serde_json::to_vec(&restored).unwrap(),
        serde_json::to_vec(old).unwrap(),
        "only two attention declarations change; FFN/Q6 head/state/KV/output remain exact"
    );
    let prepared = registration.prepare(&definition, &id()).unwrap();
    let baseline = registration.prepare(&definition, &old_id).unwrap();
    assert_eq!(prepared.weight_schema(), baseline.weight_schema());
    assert_eq!(prepared.program().states(), baseline.program().states());
    new.validate_program(prepared.program()).unwrap();
    let nodes = prepared
        .program()
        .blocks()
        .iter()
        .flat_map(|b| &b.nodes)
        .collect::<Vec<_>>();
    let old_nodes = baseline
        .program()
        .blocks()
        .iter()
        .flat_map(|b| &b.nodes)
        .collect::<Vec<_>>();
    assert_eq!(nodes.len(), old_nodes.len());
    for (node, original) in nodes.into_iter().zip(old_nodes) {
        let mut normalized = node.clone();
        if let Some((new_kind, _)) = pairs()
            .into_iter()
            .find(|(_, old_kind)| original.operation_id.as_str() == old_kind.operation_id())
        {
            assert_eq!(node.operation_id.as_str(), new_kind.operation_id());
            normalized.operation_id = original.operation_id.clone();
        }
        assert_eq!(
            serde_json::to_value(normalized).unwrap(),
            serde_json::to_value(original).unwrap()
        );
    }
}

#[test]
fn attention_geometry_profile_rejects_forged_bounds_old_id_and_missing_head_or_kv() {
    let config = source();
    let provider = Qwen35FamilyProvider::new().unwrap();
    let catalog = provider.numerical_profiles(&config).unwrap();
    let selected = catalog.resolve(&id()).unwrap();
    let mut wrong_bound = selected.clone();
    let composite = wrong_bound.operations.iter_mut().find_map(|o| o.composite_arithmetic.as_mut().filter(|c| c.schema_version == ferrum_interfaces::vnext::COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_GEOMETRY)).unwrap();
    let leaf = composite.projections[0]
        .leaves
        .iter_mut()
        .find(|l| {
            l.arithmetic
                .upstream_policy()
                .is_some_and(|p| p.geometry_selection.is_some())
        })
        .unwrap();
    let [NumericalArithmeticStage::UpstreamProjection { policy }] =
        leaf.arithmetic.stages.as_mut_slice()
    else {
        panic!("upstream")
    };
    policy.geometry_selection = Some(UpstreamProjectionGeometrySelection::M8Q4Q5Mmq {
        minimum_input_features: 4096,
        minimum_output_features: 5120,
    });
    wrong_bound.validate().unwrap();
    assert_ne!(
        wrong_bound.fingerprint().unwrap(),
        selected.fingerprint().unwrap()
    );
    assert!(
        provider.semantic_program(&config, &wrong_bound).is_err(),
        "valid generic declaration cannot masquerade as the closed family profile"
    );
    let mut old_id = selected.clone();
    old_id.id = NumericalProfileId::new(F32_MASTER_GGUF_Q8_HEAD_V1_NUMERICAL_PROFILE_ID).unwrap();
    assert!(provider.semantic_program(&config, &old_id).is_err());
    let mut old_body = catalog.resolve(&old_id.id).unwrap().clone();
    old_body.id = id();
    assert!(provider.semantic_program(&config, &old_body).is_err());
    let mut missing = selected.clone();
    missing
        .operations
        .iter_mut()
        .find(|o| o.operation_id.as_str() == LAST_TOKEN_DENSE_LINEAR_Q6_MMQ_F32_OPERATION_ID)
        .unwrap()
        .staged_arithmetic = None;
    assert!(provider.semantic_program(&config, &missing).is_err());
    missing = selected.clone();
    missing.kv_storage.clear();
    assert!(provider.semantic_program(&config, &missing).is_err());
}

#[test]
fn attention_geometry_profile_eligibility_requires_the_actual_q6_head_and_old_aa_domain() {
    let mut config = source();
    let provider = Qwen35FamilyProvider::new().unwrap();
    let selected = provider
        .numerical_profiles(&config)
        .unwrap()
        .resolve(&id())
        .unwrap()
        .clone();
    config
        .weights
        .iter_mut()
        .find(|w| w.role == "lm_head")
        .unwrap()
        .source_encoding = FamilyWeightSourceEncoding::Dense {
        element_type: ElementType::F16,
    };
    assert!(provider
        .numerical_profiles(&config)
        .unwrap()
        .resolve(&id())
        .is_err());
    assert!(provider.semantic_program(&config, &selected).is_err());
    config = source();
    config.recurrent_weight_abi = RecurrentWeightAbi::LogRateGrouped;
    assert!(provider
        .numerical_profiles(&config)
        .unwrap()
        .resolve(&id())
        .is_err());
    assert!(provider.semantic_program(&config, &selected).is_err());
}
