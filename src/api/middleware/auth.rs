//! Authentication middleware for API key validation
//!
//! This middleware validates API keys when configured and provides
//! authentication for the inference engine endpoints.

use actix_web::{
    dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform},
    Error, HttpMessage, HttpResponse,
};
use futures_util::future::LocalBoxFuture;
use std::{
    future::{ready, Ready},
    rc::Rc,
};

/// API Key authentication middleware
pub struct ApiKeyAuth {
    api_key: Option<String>,
}

impl ApiKeyAuth {
    pub fn new(api_key: Option<String>) -> Self {
        Self { api_key }
    }
}

impl<S, B> Transform<S, ServiceRequest> for ApiKeyAuth
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Transform = ApiKeyAuthMiddleware<S>;
    type InitError = ();
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(ApiKeyAuthMiddleware {
            service: Rc::new(service),
            api_key: self.api_key.clone(),
        }))
    }
}

pub struct ApiKeyAuthMiddleware<S> {
    service: Rc<S>,
    api_key: Option<String>,
}

impl<S, B> Service<ServiceRequest> for ApiKeyAuthMiddleware<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future = LocalBoxFuture<'static, Result<Self::Response, Self::Error>>;

    forward_ready!(service);

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let service = Rc::clone(&self.service);
        let api_key = self.api_key.clone();

        Box::pin(async move {
            // Skip authentication for health endpoints
            let path = req.path();
            if path == "/health" || path == "/ping" || path == "/metrics" {
                return service.call(req).await;
            }

            // If no API key is configured, allow all requests
            let required_key = match api_key {
                Some(key) => key,
                None => return service.call(req).await,
            };

            // Check for API key in Authorization header
            let auth_header = req
                .headers()
                .get("Authorization")
                .and_then(|h| h.to_str().ok())
                .and_then(|h| h.strip_prefix("Bearer "));

            // Check for API key in query parameter
            let query_key = req.query_string().split('&').find_map(|pair| {
                let mut parts = pair.splitn(2, '=');
                if parts.next() == Some("api_key") {
                    parts.next()
                } else {
                    None
                }
            });

            let provided_key = auth_header.or(query_key);

            match provided_key {
                Some(key) if key == required_key => {
                    // API key is valid, proceed with the request
                    service.call(req).await
                }
                _ => {
                    // API key is missing or invalid
                    let response = HttpResponse::Unauthorized().json(serde_json::json!({
                        "error": {
                            "message": "Invalid or missing API key",
                            "type": "authentication_error",
                            "code": "INVALID_API_KEY"
                        }
                    }));

                    let (http_req, _) = req.into_parts();
                    let service_response = ServiceResponse::new(http_req, response);
                    Ok(service_response)
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test, web, App, HttpResponse};

    async fn dummy_handler() -> Result<HttpResponse, Error> {
        Ok(HttpResponse::Ok().json(serde_json::json!({"status": "success"})))
    }

    #[actix_web::test]
    async fn test_no_api_key_configured_allows_all_requests() {
        let app = test::init_service(
            App::new()
                .wrap(ApiKeyAuth::new(None))
                .route("/test", web::get().to(dummy_handler)),
        )
        .await;

        let req = test::TestRequest::get().uri("/test").to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    #[actix_web::test]
    async fn test_health_endpoints_bypass_auth() {
        let app = test::init_service(
            App::new()
                .wrap(ApiKeyAuth::new(Some("secret-key".to_string())))
                .route("/health", web::get().to(dummy_handler))
                .route("/ping", web::get().to(dummy_handler))
                .route("/metrics", web::get().to(dummy_handler)),
        )
        .await;

        // Test /health endpoint
        let req = test::TestRequest::get().uri("/health").to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());

        // Test /ping endpoint
        let req = test::TestRequest::get().uri("/ping").to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());

        // Test /metrics endpoint
        let req = test::TestRequest::get().uri("/metrics").to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    #[actix_web::test]
    async fn test_valid_api_key_in_authorization_header() {
        let app = test::init_service(
            App::new()
                .wrap(ApiKeyAuth::new(Some("secret-key".to_string())))
                .route("/test", web::get().to(dummy_handler)),
        )
        .await;

        let req = test::TestRequest::get()
            .uri("/test")
            .insert_header(("Authorization", "Bearer secret-key"))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    #[actix_web::test]
    async fn test_valid_api_key_in_query_parameter() {
        let app = test::init_service(
            App::new()
                .wrap(ApiKeyAuth::new(Some("secret-key".to_string())))
                .route("/test", web::get().to(dummy_handler)),
        )
        .await;

        let req = test::TestRequest::get()
            .uri("/test?api_key=secret-key")
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    #[actix_web::test]
    async fn test_invalid_api_key_returns_unauthorized() {
        let app = test::init_service(
            App::new()
                .wrap(ApiKeyAuth::new(Some("secret-key".to_string())))
                .route("/test", web::get().to(dummy_handler)),
        )
        .await;

        let req = test::TestRequest::get()
            .uri("/test")
            .insert_header(("Authorization", "Bearer wrong-key"))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 401);
    }

    #[actix_web::test]
    async fn test_missing_api_key_returns_unauthorized() {
        let app = test::init_service(
            App::new()
                .wrap(ApiKeyAuth::new(Some("secret-key".to_string())))
                .route("/test", web::get().to(dummy_handler)),
        )
        .await;

        let req = test::TestRequest::get().uri("/test").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 401);
    }

    #[actix_web::test]
    async fn test_malformed_authorization_header() {
        let app = test::init_service(
            App::new()
                .wrap(ApiKeyAuth::new(Some("secret-key".to_string())))
                .route("/test", web::get().to(dummy_handler)),
        )
        .await;

        let req = test::TestRequest::get()
            .uri("/test")
            .insert_header(("Authorization", "secret-key")) // Missing "Bearer " prefix
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 401);
    }

    #[actix_web::test]
    async fn test_api_key_priority_header_over_query() {
        let app = test::init_service(
            App::new()
                .wrap(ApiKeyAuth::new(Some("secret-key".to_string())))
                .route("/test", web::get().to(dummy_handler)),
        )
        .await;

        // Header has correct key, query has wrong key
        let req = test::TestRequest::get()
            .uri("/test?api_key=wrong-key")
            .insert_header(("Authorization", "Bearer secret-key"))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    #[actix_web::test]
    async fn test_error_response_format() {
        let app = test::init_service(
            App::new()
                .wrap(ApiKeyAuth::new(Some("secret-key".to_string())))
                .route("/test", web::get().to(dummy_handler)),
        )
        .await;

        let req = test::TestRequest::get().uri("/test").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 401);

        let body = test::read_body(resp).await;
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["error"]["message"], "Invalid or missing API key");
        assert_eq!(json["error"]["type"], "authentication_error");
        assert_eq!(json["error"]["code"], "INVALID_API_KEY");
    }
}
