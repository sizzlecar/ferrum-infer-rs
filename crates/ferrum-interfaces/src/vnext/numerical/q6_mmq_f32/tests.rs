use super::*;
use crate::vnext::*;

fn block() -> BlockQuantizationSpec {
    BlockQuantizationSpec {
        format_id: QuantizationFormatId::new("quantization.gguf.q6-k").unwrap(),
        logical_values_per_block: 256,
        bytes_per_block: 210,
    }
}

#[test]
fn q6_head_f32_schema_is_closed_and_does_not_enter_old_f16_schemas() {
    let staged = Q6MmqF32Policy::new().staged();
    staged.validate().unwrap();
    assert_eq!(staged.output_type(), Some(ElementType::F32));
    let wire = serde_json::to_value(&staged).unwrap();
    assert_eq!(
        serde_json::from_value::<StagedNumericalArithmetic>(wire.clone()).unwrap(),
        staged
    );
    for schema in [1, 2, 3, 4, 5, 6] {
        let mut old = staged.clone();
        old.schema_version = schema;
        assert!(old.validate().is_err());
    }
    let mut old = crate::vnext::Q8ActSwiGluProfile::Iq4Xs
        .arithmetic()
        .projections[0]
        .leaves[0]
        .arithmetic
        .clone();
    let original = serde_json::to_vec(&old).unwrap();
    old.validate().unwrap();
    assert_eq!(serde_json::to_vec(&old).unwrap(), original);
    old.schema_version = NUMERICAL_ARITHMETIC_SCHEMA_VERSION_Q6_MMQ_F32;
    assert!(old.validate().is_err());
    let mut foreign = wire.clone();
    foreign["stages"][0]["policy"]["arithmetic"] = "mmq_d4_marker_v2".into();
    assert!(serde_json::from_value::<StagedNumericalArithmetic>(foreign).is_err());
    let mut extra = wire;
    extra["stages"][0]["policy"]["silent_fallback_on_error"] = true.into();
    assert!(serde_json::from_value::<StagedNumericalArithmetic>(extra).is_err());
}

#[test]
fn q6_head_f32_routes_use_actual_rows_and_exact_block_abi() {
    let p = Q6MmqF32Policy::new();
    assert!(p.route(0, true).is_err());
    for m in 1..=32 {
        assert_eq!(p.route(m, true).unwrap(), Q6MmqF32Route::Mmq);
    }
    for m in [33, 2048, u32::MAX] {
        assert_eq!(p.route(m, true).unwrap(), Q6MmqF32Route::Strict);
    }
    assert_eq!(p.route(1, false).unwrap(), Q6MmqF32Route::Strict);
    let valid = block();
    assert!(Q6MmqF32Policy::eligible_weight(Some(&valid), 5120, 248320));
    assert!(!Q6MmqF32Policy::eligible_weight(Some(&valid), 255, 7));
    assert!(!Q6MmqF32Policy::eligible_weight(Some(&valid), 256, 0));
    assert!(!Q6MmqF32Policy::eligible_weight(None, 256, 7));
    let mut wrong = valid.clone();
    wrong.bytes_per_block = 209;
    assert!(!Q6MmqF32Policy::eligible_weight(Some(&wrong), 256, 7));
    wrong = valid;
    wrong.logical_values_per_block = 32;
    assert!(!Q6MmqF32Policy::eligible_weight(Some(&wrong), 256, 7));
    let mut tampered = p;
    tampered.maximum_physical_rows = 33;
    assert!(tampered.validate().is_err());
    assert!(tampered.route(1, true).is_err());
}

#[test]
fn q6_head_f32_contract_preserves_ports_and_declares_retained_resources() {
    let old = last_token_dense_linear_f32_contract().unwrap();
    let new = last_token_dense_linear_q6_mmq_f32_contract().unwrap();
    let a = old.descriptor();
    let b = new.descriptor();
    assert_eq!(a.inputs, b.inputs);
    assert_eq!(a.outputs, b.outputs);
    assert_eq!(a.attributes, b.attributes);
    assert_eq!(b.resources.scratch, ResourcePresenceRequirement::Required);
    assert_eq!(
        b.resources.persistent,
        ResourcePresenceRequirement::Required
    );
    assert_eq!(b.resources.binding, ResourcePresenceRequirement::Forbidden);
    assert_ne!(a.id, b.id);
    assert_eq!(b.oracle, OracleSpec::OperationDefined);
    let s = Q6MmqF32Policy::new().semantics();
    assert_eq!(
        (s.input_type, s.output_type),
        (ElementType::F32, ElementType::F32)
    );
    assert_eq!(s.canonical_nan_bits, 0x7fc00000);
    assert_eq!(s.retained_weight_flag_bytes_per_leaf, 4);
}
