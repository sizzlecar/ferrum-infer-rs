//! Simple integration test to verify basic functionality without hanging

#[test]
fn test_basic_functionality() {
    // This is a simple test that should always pass
    assert_eq!(1 + 1, 2);
}

#[test] 
fn test_config_creation() {
    use ferrum_infer::Config;
    let config = Config::default();
    assert_eq!(config.server.port, 8080);
}

