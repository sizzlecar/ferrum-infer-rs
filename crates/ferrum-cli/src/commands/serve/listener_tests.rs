use super::*;
use clap::Parser;

#[derive(Parser)]
struct TestCli {
    #[command(flatten)]
    serve: ServeCommand,
}

#[tokio::test]
async fn serve_rejects_occupied_port_before_resolving_or_loading_model() {
    let occupied = BoundHttpListener::bind("127.0.0.1", 0).await.unwrap();
    let directory = tempfile::tempdir().unwrap();
    let missing_model = directory.path().join("missing-model.gguf");
    let command = TestCli::try_parse_from([
        "ferrum",
        "--model",
        missing_model.to_str().unwrap(),
        "--backend",
        "cpu",
    ])
    .unwrap()
    .serve;
    let mut config = CliConfig::default();
    config.server.host = "127.0.0.1".to_owned();
    config.server.port = occupied.local_addr().port();
    let error = tokio::time::timeout(Duration::from_secs(5), execute(command, config))
        .await
        .unwrap()
        .unwrap_err();
    assert!(
        matches!(error.cause(), FerrumError::Network { .. }),
        "{error}"
    );
    assert!(error
        .to_string()
        .contains(&occupied.local_addr().to_string()));
    assert!(
        error.cleanup().is_none(),
        "failed bind must not create a native owner"
    );
    assert!(!missing_model.exists());
}
