use super::*;
use crate::observability_product::{
    ProductJournalSession, ProductObservabilityConfig, ProductSessionMetadata, ProfileDetailArg,
};

fn config() -> ResolvedFerrumConfig {
    startup_auto_config(
        HardwareCapabilities::unknown(),
        Some(ModelCapabilities::unknown()),
        ferrum_types::ExecutionResourceAuthority::PlanRuntime,
        None,
        None,
        None,
        None,
        RuntimeConfigSnapshot::default(),
    )
    .unwrap()
}

#[test]
fn product_session_metadata_rejects_invalid_explicit_values() {
    let resolved = config();
    for (hash, flags) in [
        (Some("not-sha256"), None),
        (None, Some("[]")),
        (None, Some("{")),
    ] {
        let result = ProductSessionMetadata::new(
            None,
            hash.map(str::to_owned),
            None,
            None,
            flags.map(str::to_owned),
            &resolved,
            "actual-model",
        );
        assert!(matches!(result, Err(FerrumError::Config { .. })));
    }
}

#[tokio::test]
async fn product_session_metadata_uses_standard_schema_without_relabeling_actual_model() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("profile.jsonl");
    let product = ProductObservabilityConfig::new(
        ferrum_types::ProfileEntrypoint::Serve,
        "actual-model",
        Some(&path),
        ProfileDetailArg::Off,
        None,
        None,
        None,
        1.0,
    );
    let mut engine = ferrum_types::EngineConfig::default();
    engine.runtime.profile_jsonl = Some(path.clone());
    let journals = ProductJournalSession::prepare(&product, &engine);
    let metadata = ProductSessionMetadata::new(
        Some("commit".into()),
        Some("sha256:declared".into()),
        Some("user-label".into()),
        Some(3),
        Some("{\"user_flag\":true}".into()),
        &config(),
        "actual-model",
    )
    .unwrap();
    metadata.write(&journals, &product).unwrap();
    journals.finish().await.unwrap();
    let records: Vec<ferrum_types::FerrumProfileEvent> = std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(records.len(), 1);
    let event = &records[0];
    event.validate().unwrap();
    assert_eq!(event.phase, "product_session_metadata");
    assert_eq!(event.model.as_deref(), Some("actual-model"));
    assert_eq!(event.attributes["profile_detail"], "off");
    assert_eq!(
        event.attributes["declared_metadata"],
        serde_json::json!({
            "commit_sha":"commit", "env_hash":"sha256:declared", "model":"user-label", "concurrency":3,
            "runtime_flags":{"user_flag":true}
        })
    );
}
