use super::*;
use clap::Parser;

#[derive(Parser)]
struct TestCli {
    #[command(flatten)]
    serve: ServeCommand,
}

#[test]
fn explicit_key_overrides_server_configuration_and_exports_redact() {
    let command = TestCli::try_parse_from(["ferrum", "--api-key", "explicit-key"]).unwrap();
    let key = resolve_api_key(
        command.serve.api_key,
        Some(ferrum_server::ApiKey::new("configured-key").unwrap()),
    )
    .unwrap();
    assert_eq!(
        key,
        Some(ferrum_server::ApiKey::new("explicit-key").unwrap())
    );
    let mut config = CliConfig::default();
    config.server.api_key = key;
    assert!(!serde_json::to_string(&config)
        .unwrap()
        .contains("explicit-key"));
    assert!(!format!("{config:?}").contains("explicit-key"));
    assert_eq!(
        resolve_api_key(None, config.server.api_key.clone()).unwrap(),
        config.server.api_key
    );
    assert_eq!(resolve_api_key(None, None).unwrap(), None);
}

#[tokio::test]
async fn malformed_key_is_rejected_before_text_model_resolution() {
    let command = TestCli::try_parse_from([
        "ferrum",
        "--api-key",
        "secret with whitespace",
        "--model",
        "missing-model.gguf",
        "--backend",
        "cpu",
    ])
    .unwrap()
    .serve;
    let error = execute(command, CliConfig::default()).await.unwrap_err();
    assert!(matches!(error.cause(), FerrumError::Config { .. }));
    assert!(!error.to_string().contains("secret with whitespace"));
}

#[tokio::test]
async fn malformed_configured_key_is_rejected_before_image_source_or_device() {
    let key: ferrum_server::ApiKey = serde_json::from_str("\"\"").unwrap();
    let directory = tempfile::tempdir().unwrap();
    let source = super::super::image_product::ImageProductSource::ExplicitConfig(
        directory.path().join("missing-image.json"),
    );
    let error = super::super::image_product::execute_serve_source_with_api_key(
        source,
        "127.0.0.1".into(),
        0,
        vec![],
        Some(key),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("API key"));
    assert!(!error.to_string().contains("requires the cuda feature"));
}
