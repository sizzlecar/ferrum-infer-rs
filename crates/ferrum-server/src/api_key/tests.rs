use super::*;
use crate::{axum_server::AppState, AxumServer, BoundHttpListener, ServerConfig};
use axum::{body::Body, routing::post};
use std::{sync::Arc, time::Duration};
use tower::ServiceExt;

#[test]
fn credentials_validate_without_echoing_and_configuration_exports_redact() {
    for value in [
        "",
        " ",
        "key with space",
        "key\nline",
        "=padding",
        "key=middle",
    ] {
        let error = ApiKey::new(value).unwrap_err().to_string();
        assert!(!error.contains("key with space"));
        assert!(!error.contains("key\nline"));
    }
    let key = ApiKey::new("private-key_+/==").unwrap();
    let config = ServerConfig {
        api_key: Some(key),
        ..Default::default()
    };
    for exported in [
        format!("{config:?}"),
        serde_json::to_string(&config).unwrap(),
    ] {
        assert!(!exported.contains("private-key_+/=="));
        assert!(exported.contains("REDACTED"));
    }
    let empty: ApiKey = serde_json::from_str("\"\"").unwrap();
    assert!(protect(Router::new(), Some(&empty)).is_err());
}

#[tokio::test]
async fn unauthorized_upload_is_rejected_before_polling_body_or_running_handler() {
    let router = Router::new().route(
        "/v1/images/edits",
        post(|| async {
            panic!("unauthorized handler must not run");
            #[allow(unreachable_code)]
            "unreachable"
        }),
    );
    let router = protect(router, Some(&ApiKey::new("expected-key").unwrap())).unwrap();
    let body = Body::from_stream(futures::stream::poll_fn(
        |_| -> std::task::Poll<Option<Result<axum::body::Bytes, std::io::Error>>> {
            panic!("unauthorized upload body must not be polled")
        },
    ));
    let response = router
        .oneshot(Request::post("/v1/images/edits").body(body).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn actual_http_routes_require_bearer_and_keep_health_and_assets_public() {
    let server = Arc::new(AxumServer::from_state(AppState::default()));
    let listener = BoundHttpListener::bind("127.0.0.1", 0).await.unwrap();
    let address = listener.local_addr();
    let config = ServerConfig {
        api_key: Some(ApiKey::new("test-server-key").unwrap()),
        ..Default::default()
    };
    let (ready, started) = tokio::sync::oneshot::channel();
    let serving = Arc::clone(&server);
    let task = tokio::spawn(async move {
        serving
            .start_with_listener(&config, listener, |_| {
                let _ = ready.send(());
            })
            .await
    });
    tokio::time::timeout(Duration::from_secs(5), started)
        .await
        .unwrap()
        .unwrap();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let base = format!("http://{address}");
    for path in ["/v1/models", "/metrics", "/playground/config"] {
        let response = client.get(format!("{base}{path}")).send().await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{path}");
        assert_eq!(response.headers()[header::WWW_AUTHENTICATE], "Bearer");
        let body: serde_json::Value = response.json().await.unwrap();
        assert_eq!(body["error"]["code"], "invalid_api_key");
    }
    for path in [
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
        "/v1/images/generations",
        "/v1/images/edits",
    ] {
        let response = client
            .post(format!("{base}{path}"))
            .body("not valid JSON or multipart")
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{path}");
    }
    for credential in [
        "Bearer wrong-key",
        "Basic test-server-key",
        "Bearer ",
        "Bearer test-server-key other",
    ] {
        let response = client
            .get(format!("{base}/v1/models"))
            .header(header::AUTHORIZATION, credential)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }
    let response = client
        .get(format!("{base}/v1/models?api_key=test-server-key"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let response = client
        .get(format!("{base}/v1/models"))
        .header(header::AUTHORIZATION, "Bearer test-server-key")
        .header(header::AUTHORIZATION, "Bearer test-server-key")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let response = client
        .get(format!("{base}/v1/models"))
        .header(header::AUTHORIZATION, "bEaReR test-server-key")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    for path in [
        "/health",
        "/playground",
        "/playground/app.js",
        "/playground/style.css",
    ] {
        let response = client.get(format!("{base}{path}")).send().await.unwrap();
        assert_ne!(response.status(), StatusCode::UNAUTHORIZED, "{path}");
    }
    let preflight = client
        .request(
            reqwest::Method::OPTIONS,
            format!("{base}/v1/chat/completions"),
        )
        .header(header::ORIGIN, "https://client.example")
        .header(header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
        .header(
            header::ACCESS_CONTROL_REQUEST_HEADERS,
            "authorization,content-type",
        )
        .send()
        .await
        .unwrap();
    assert!(preflight.status().is_success());
    server.finish_stop().await.unwrap();
    task.await.unwrap().unwrap();
}

#[tokio::test]
async fn absent_key_preserves_open_server_behavior() {
    let router = Router::new().route("/v1/models", axum::routing::get(|| async { "available" }));
    let router = protect(router, None).unwrap();
    let response = router
        .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}
