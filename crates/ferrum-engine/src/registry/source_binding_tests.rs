use super::*;
use std::path::PathBuf;

struct DefinedFixture {
    defined: Arc<DefinedProductionModel>,
    directory: PathBuf,
}

impl DefinedFixture {
    fn new() -> Self {
        let directory = tests::write_qwen35_fixture_model_dir();
        std::fs::write(directory.join("tokenizer.json"), br#"{"version":"1.0"}"#).unwrap();
        std::fs::write(directory.join("tokenizer_config.json"), br#"{"chat_template":"{% for message in messages %}{{ message['content'] }}{% endfor %}","eos_token_id":2,"unk_token":"<unk>"}"#).unwrap();
        let defined =
            Arc::new(ferrum_models::vnext::qwen35::define_from_model_dir(&directory).unwrap());
        Self { defined, directory }
    }

    fn cleanup(self) {
        let Self { defined, directory } = self;
        drop(defined);
        std::fs::remove_dir_all(directory).unwrap();
    }
}

#[test]
fn component_config_recovers_the_exact_defined_source_lease() {
    let fixture = DefinedFixture::new();
    let config = ComponentConfig::from_engine_config_and_product_model(
        &EngineConfig::default(),
        None,
        Some(fixture.defined.clone()),
    );
    assert!(Arc::ptr_eq(
        config.model_sources.as_ref().unwrap(),
        fixture.defined.sources(),
    ));
    assert!(Arc::ptr_eq(
        config.defined_model.as_ref().unwrap(),
        &fixture.defined,
    ));
    drop(config);
    fixture.cleanup();
}

#[tokio::test]
async fn llm_factory_rejects_missing_or_distinct_defined_source_lease_before_loading() {
    let fixture = DefinedFixture::new();
    let mut config = ComponentConfig::from_engine_config_and_product_model(
        &EngineConfig::default(),
        None,
        Some(fixture.defined.clone()),
    );
    // ComponentConfig's public fields also permit direct factory callers.
    // A missing lease must not fall through to this nonexistent legacy path.
    config.engine_config.runtime.model_path = Some(
        fixture
            .directory
            .join("missing-model")
            .display()
            .to_string(),
    );
    config.model_sources = None;
    let error = LlmExecutorFactory
        .create(&config)
        .await
        .err()
        .expect("a defined model without its source lease must fail");
    assert!(
        error.to_string().contains("do not share one source lease"),
        "unexpected missing-lease error: {error}"
    );

    // Equal fingerprints are insufficient for this already-defined package:
    // its factories must reuse the same immutable source lease.
    let distinct = Arc::new(fixture.defined.sources().as_ref().clone());
    assert_eq!(
        distinct.resolved_sources(),
        fixture.defined.sources().resolved_sources(),
    );
    assert!(!Arc::ptr_eq(&distinct, fixture.defined.sources()));
    config.model_sources = Some(distinct);
    let error = LlmExecutorFactory
        .create(&config)
        .await
        .err()
        .expect("an independently owned source bundle must fail the defined lease contract");
    assert!(
        error.to_string().contains("do not share one source lease"),
        "unexpected distinct-lease error: {error}"
    );
    drop(config);
    fixture.cleanup();
}
