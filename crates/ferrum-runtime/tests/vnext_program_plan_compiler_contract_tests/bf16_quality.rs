use super::*;

fn canonical_bytes(value: &Value) -> Vec<u8> {
    // The workspace enables serde_json/preserve_order. Mutating an artifact
    // must not turn insertion order into its wire format or vector digest.
    let mut value = value.clone();
    value.sort_all_objects();
    serde_json::to_vec(&value).unwrap()
}

fn fixture() -> (WeightMaterializerDescriptor, Value) {
    // BF16 1, minimum subnormal, maximum finite, and negative zero. These
    // encodings cannot be substituted with binary16 and retain the same values.
    let actual = [0x3f80_u16, 0x0001, 0x7f7f, 0x8000];
    let reference = actual.map(|bits| u32::from(bits) << 16);
    let reference_digest = little_endian_u32_sha256(&reference);
    let mut vector = test_quality_vector_payload();
    vector["activation_contract"]["dtype"] = json!("BF16");
    vector["source_contract"]["values_dtype"] = json!("BF16");
    vector["fixture_id"] = json!("synthetic-bf16-numeric-vector-v1");
    for case in vector["cases"].as_array_mut().unwrap() {
        case["reference_f32le_sha256"] = json!(reference_digest);
    }
    let quality = ApproximateWeightQualityContract::new(
        sha('b'),
        format!("{:x}", Sha256::digest(canonical_bytes(&vector))),
        4,
        CanonicalRational::new(1, 20).unwrap(),
        0,
        0,
    )
    .unwrap();
    let descriptor =
        PaddedDenseMaterializer::with_fidelity('8', WeightMaterializationFidelity::Approximate)
            .descriptor
            .with_approximate_quality_contract(quality)
            .unwrap();
    let mut artifact: Value =
        serde_json::from_slice(&test_numeric_quality_artifact(&descriptor)).unwrap();
    artifact["schema_id"] = json!(NUMERIC_WEIGHT_QUALITY_BF16_ARTIFACT_SCHEMA_ID);
    artifact["authority"]["version"] = json!({"major":2,"minor":0});
    artifact["authority"]["implementation_fingerprint"] =
        json!(numeric_bf16_weight_quality_authority_implementation_fingerprint().unwrap());
    artifact["quality_vector_payload"] = vector;
    for case in artifact["cases"].as_array_mut().unwrap() {
        let case = case.as_object_mut().unwrap();
        case.remove("actual_f16_bits");
        case.remove("actual_f16le_sha256");
        case.insert("actual_bf16_bits".into(), json!(actual));
        case.insert(
            "actual_bf16le_sha256".into(),
            json!(little_endian_u16_sha256(&actual)),
        );
        case.insert("reference_f32_bits".into(), json!(reference));
        case.insert("reference_f32le_sha256".into(), json!(reference_digest));
    }
    (descriptor, artifact)
}

fn select(
    descriptor: WeightMaterializerDescriptor,
    artifact: &Value,
) -> Result<TrustedExecutionWeightPlan, VNextError> {
    let family = TestRegistry::new().prepare();
    let id = descriptor.id().clone();
    let registry =
        WeightMaterializerRegistry::new(vec![Box::new(PaddedDenseMaterializer { descriptor })])
            .unwrap();
    let catalog = registry.augment_catalog(catalog()).unwrap();
    registry.select_with_numeric_quality_artifact(
        &family,
        &catalog,
        &id,
        &canonical_bytes(artifact),
    )
}

#[test]
fn bf16_quality_artifact_has_distinct_authority_and_live_schema_receipt() {
    let (descriptor, artifact) = fixture();
    let selected = select(descriptor, &artifact).unwrap();
    let approval = selected.plan().approximate_quality_approval().unwrap();
    assert_eq!(approval.authority_version(), ContractVersion::new(2, 0));
    assert_eq!(approval.relative_l2_max_observed().numerator(), 0);
    assert_eq!(approval.nan_count(), 0);
    assert_eq!(approval.inf_count(), 0);
    let wire = serde_json::to_value(selected.plan()).unwrap();
    let decoded: ExecutionWeightPlan = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(serde_json::to_value(decoded).unwrap(), wire);
    assert_ne!(
        numeric_bf16_weight_quality_authority_implementation_fingerprint().unwrap(),
        numeric_weight_quality_authority_implementation_fingerprint().unwrap()
    );
}

#[test]
fn bf16_quality_rejects_cross_encoding_mixed_fields_and_false_counts() {
    let (descriptor, artifact) = fixture();
    let mut wrong_schema = artifact.clone();
    wrong_schema["schema_id"] = json!(NUMERIC_WEIGHT_QUALITY_ARTIFACT_SCHEMA_ID);
    assert!(select(descriptor.clone(), &wrong_schema).is_err());
    let mut mixed = artifact.clone();
    let case = mixed["cases"][0].as_object_mut().unwrap();
    let bits = case.remove("actual_bf16_bits").unwrap();
    let digest = case.remove("actual_bf16le_sha256").unwrap();
    case.insert("actual_f16_bits".into(), bits);
    case.insert("actual_f16le_sha256".into(), digest);
    assert!(select(descriptor.clone(), &mixed).is_err());
    let mut extra = artifact.clone();
    extra["cases"][0]["actual_f16_bits"] = json!([0x3c00]);
    assert!(select(descriptor.clone(), &extra).is_err());
    for nonfinite in [0x7f80_u16, 0x7fc0] {
        let mut bad = artifact.clone();
        let actual = [nonfinite, 0x0001, 0x7f7f, 0x8000];
        bad["cases"][0]["actual_bf16_bits"] = json!(actual);
        bad["cases"][0]["actual_bf16le_sha256"] = json!(little_endian_u16_sha256(&actual));
        assert!(select(descriptor.clone(), &bad).is_err());
    }
    let mut wrong_finite = artifact;
    let actual = [0x3f80_u16, 0x0001, 0, 0x8000];
    wrong_finite["cases"][0]["actual_bf16_bits"] = json!(actual);
    wrong_finite["cases"][0]["actual_bf16le_sha256"] = json!(little_endian_u16_sha256(&actual));
    assert!(select(descriptor, &wrong_finite).is_err());
}
