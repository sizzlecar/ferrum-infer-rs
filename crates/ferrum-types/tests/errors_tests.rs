use ferrum_types::*;

#[test]
fn error_constructors_and_classes() {
    let e = FerrumError::config("bad");
    // Config is not classified as client/server specifically; must be non-retryable
    assert!(!e.is_client_error());
    assert!(!e.is_retryable());

    let e = FerrumError::timeout("t");
    assert!(e.is_server_error());
    assert!(e.is_retryable());
}

#[test]
fn error_conversions() {
    let io_err: FerrumError = std::io::Error::new(std::io::ErrorKind::Other, "io").into();
    match io_err {
        FerrumError::IO { .. } => {}
        _ => panic!("wrong kind"),
    }

    let ser_err: FerrumError = serde_json::from_str::<serde_json::Value>("{")
        .unwrap_err()
        .into();
    match ser_err {
        FerrumError::Serialization { .. } => {}
        _ => panic!("wrong kind"),
    }
}
