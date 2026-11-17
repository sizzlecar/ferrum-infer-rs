use super::*;

fn parse_image_serve_cli(key: &str) -> serve::ServeCliCommand {
    let definition = Cli::command();
    let matches = definition
        .clone()
        .try_get_matches_from([
            "ferrum",
            "serve",
            "--image-config",
            "missing-image.json",
            "--api-key",
            key,
        ])
        .unwrap();
    image_product::validate_cli_matches(&definition, &matches).unwrap();
    let serve = matches.subcommand_matches("serve").unwrap();
    assert_eq!(
        serve.get_one::<String>("api_key").map(String::as_str),
        Some(key)
    );
    assert_eq!(
        serve
            .get_one::<std::path::PathBuf>("image_config")
            .map(std::path::PathBuf::as_path),
        Some(std::path::Path::new("missing-image.json"))
    );
    let Commands::Serve(command) = Cli::from_arg_matches(&matches).unwrap().command else {
        panic!("expected serve command");
    };
    command
}

#[test]
fn image_serve_cli_accepts_explicit_api_key() {
    parse_image_serve_cli("explicit-key");
}

#[tokio::test]
async fn image_serve_cli_rejects_malformed_key_without_echo_or_model_io() {
    let command = parse_image_serve_cli("secret with whitespace");
    let config: CliConfig =
        toml::from_str("[server]\napi_key = \"valid-configured-key\"\n").unwrap();
    let error = serve::execute_cli(command, config, true)
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("API key"), "{error}");
    assert!(!error.contains("secret with whitespace"));
    assert!(!error.contains("valid-configured-key"));
}
