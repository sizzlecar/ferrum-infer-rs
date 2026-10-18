use super::*;

const THIRD_PARTY: &[u8] = br#"{"architectures":["ThirdPartyDecoder"]}"#;

struct SourceFiles {
    _root: tempfile::TempDir,
    semantic: PathBuf,
    tokenizer: PathBuf,
    weights: PathBuf,
}

impl SourceFiles {
    fn new(semantic_config: &[u8], physical_config: Option<&[u8]>) -> Self {
        let root = tempfile::tempdir().unwrap();
        let semantic = root.path().join("semantic");
        let tokenizer = root.path().join("tokenizer");
        let weights = root.path().join("weights");
        for directory in [&semantic, &tokenizer, &weights] {
            std::fs::create_dir_all(directory).unwrap();
        }
        std::fs::write(semantic.join("config.json"), semantic_config).unwrap();
        std::fs::write(tokenizer.join("tokenizer.json"), br#"{"version":"1.0"}"#).unwrap();
        if let Some(config) = physical_config {
            std::fs::write(weights.join("config.json"), config).unwrap();
        }
        // These tests exercise the source lease, not tensor deserialization.
        std::fs::write(weights.join("model.safetensors"), b"weight-payload").unwrap();
        Self {
            _root: root,
            semantic,
            tokenizer,
            weights,
        }
    }

    fn original_sources(&self) -> OriginalModelSources {
        let local = |path: &Path| OriginalModelSource {
            kind: ModelSourceKind::LocalDirectory,
            location: path.display().to_string(),
            requested_revision: None,
        };
        OriginalModelSources {
            semantic: local(&self.semantic),
            tokenizer: local(&self.tokenizer),
            weights: local(&self.weights),
        }
    }

    fn open(&self) -> Result<ProductionModelSourceBundle> {
        ProductionModelSourceBundle::open(
            &self.semantic,
            &self.tokenizer,
            ProductionWeightArtifact::safetensors_directory(&self.weights),
            self.original_sources(),
        )
    }
}

#[test]
fn matching_third_party_architectures_do_not_require_a_builtin_registration() {
    let physical =
        br#"{"architectures":["ThirdPartyDecoder"],"quantization_config":{"quant_method":"gptq"}}"#;
    let files = SourceFiles::new(THIRD_PARTY, Some(physical));
    let bundle = files.open().unwrap();
    assert_eq!(bundle.config_json(), THIRD_PARTY);
    assert_eq!(bundle.weight_config_json(), Some(physical.as_slice()));
}

#[test]
fn different_architectures_are_rejected_before_weight_fingerprinting() {
    let files = SourceFiles::new(THIRD_PARTY, Some(br#"{"architectures":["OtherDecoder"]}"#));
    // The identity mismatch must be reported before any shard is required.
    std::fs::remove_file(files.weights.join("model.safetensors")).unwrap();
    let error = files.open().unwrap_err().to_string();
    assert!(
        error.contains("does not match semantic architecture"),
        "{error}"
    );
    assert!(error.contains("ThirdPartyDecoder"), "{error}");
    assert!(error.contains("OtherDecoder"), "{error}");
}

#[test]
fn every_present_architectures_declaration_must_be_one_valid_identity() {
    let invalid = [
        serde_json::Value::Null,
        serde_json::json!("ThirdPartyDecoder"),
        serde_json::json!([]),
        serde_json::json!(["ThirdPartyDecoder", "OtherDecoder"]),
        serde_json::json!([3]),
        serde_json::json!([""]),
        serde_json::json!([" "]),
    ];
    for declaration in invalid {
        let config = serde_json::to_vec(&serde_json::json!({
            "architectures": declaration,
        }))
        .unwrap();
        let semantic = SourceFiles::new(&config, None);
        assert!(semantic.open().is_err(), "accepted semantic {declaration}");
        let physical = SourceFiles::new(THIRD_PARTY, Some(&config));
        assert!(physical.open().is_err(), "accepted physical {declaration}");
    }
}

#[test]
fn physical_architecture_requires_a_semantic_declaration() {
    let files = SourceFiles::new(br#"{}"#, Some(THIRD_PARTY));
    let error = files.open().unwrap_err().to_string();
    assert!(
        error.contains("requires a matching semantic architectures declaration"),
        "{error}"
    );
}

#[test]
fn absent_physical_config_and_quantization_only_config_remain_valid() {
    let quant_only = br#"{"quantization_config":{"quant_method":"gptq"}}"#;
    for semantic in [THIRD_PARTY, br#"{}"#.as_slice()] {
        for physical in [None, Some(quant_only.as_slice())] {
            let files = SourceFiles::new(semantic, physical);
            let bundle = files.open().unwrap();
            assert_eq!(bundle.weight_config_json(), physical);
            assert_eq!(
                bundle
                    .fingerprint(ModelArtifactSourceRole::Weights, "config.json")
                    .is_some(),
                physical.is_some()
            );
        }
    }
}

#[test]
fn physical_config_fingerprint_uses_retained_bytes_after_file_changes() {
    let files = SourceFiles::new(THIRD_PARTY, Some(THIRD_PARTY));
    let bundle = files.open().unwrap();
    let fingerprint = bundle
        .fingerprint(ModelArtifactSourceRole::Weights, "config.json")
        .unwrap();
    assert_eq!(fingerprint.size_bytes, THIRD_PARTY.len() as u64);
    assert_eq!(
        fingerprint.sha256,
        format!("{:x}", Sha256::digest(THIRD_PARTY))
    );

    std::fs::write(files.weights.join("config.json"), b"changed after read").unwrap();
    // Exercise the production fingerprinting step on the retained read. It
    // must not reopen config.json, even while it enumerates the real directory.
    let files_after_change =
        fingerprint_weight_artifact(bundle.weights(), bundle.weight_config_json()).unwrap();
    assert_eq!(
        files_after_change
            .iter()
            .find(|file| file.relative_path == "config.json"),
        Some(fingerprint)
    );
    assert_eq!(bundle.weight_config_json(), Some(THIRD_PARTY));
    assert_eq!(
        files_after_change
            .iter()
            .filter(|file| file.relative_path == "config.json")
            .count(),
        1
    );
}

#[test]
fn colocated_roles_reuse_the_preflight_read_after_config_replacement() {
    let mut files = SourceFiles::new(THIRD_PARTY, None);
    std::fs::write(files.semantic.join("model.safetensors"), b"weight-payload").unwrap();
    files.weights = files.semantic.join(".");
    let bundle = ProductionModelSourceBundle::open_with_semantic_preflight(
        &files.semantic,
        &files.tokenizer,
        ProductionWeightArtifact::safetensors_directory(&files.weights),
        files.original_sources(),
        |bytes| {
            assert_eq!(bytes, THIRD_PARTY);
            std::fs::write(
                files.semantic.join("config.json"),
                br#"{"architectures":["ChangedAfterPreflight"]}"#,
            )
            .unwrap();
            Ok(())
        },
    )
    .unwrap();

    assert_eq!(bundle.weight_config_json(), Some(THIRD_PARTY));
    assert!(Arc::ptr_eq(
        &bundle.config_json,
        bundle.weight_config_json.as_ref().unwrap()
    ));
    assert_eq!(
        bundle.fingerprint(ModelArtifactSourceRole::Semantic, "config.json"),
        bundle.fingerprint(ModelArtifactSourceRole::Weights, "config.json")
    );
    assert_eq!(
        std::fs::read(files.semantic.join("config.json")).unwrap(),
        br#"{"architectures":["ChangedAfterPreflight"]}"#
    );
}
