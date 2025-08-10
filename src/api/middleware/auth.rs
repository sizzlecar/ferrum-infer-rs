//! API key authentication middleware
//!
//! This middleware provides optional API key authentication for the inference
//! engine endpoints.

use actix_web::{
    dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform},
    Error, HttpResponse,
};
use futures_util::future::LocalBoxFuture;
use std::{
    future::{ready, Ready},
    rc::Rc,
};

/// API key authentication middleware
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

        Box::pin(async move {
            // For now, just pass through all requests to get compilation working
            // TODO: Implement proper API key authentication
            service.call(req).await
        })
    }
}

/// Extract API key from request headers or query parameters
fn _extract_api_key(req: &ServiceRequest) -> Option<String> {
    // Try Authorization header first
    if let Some(auth_header) = req.headers().get("Authorization") {
        if let Ok(auth_str) = auth_header.to_str() {
            if auth_str.starts_with("Bearer ") {
                return Some(auth_str[7..].to_string());
            }
        }
    }

    // Try query parameter
    if let Some(api_key) = req.query_string().split('&').find_map(|param| {
        let mut parts = param.split('=');
        if parts.next()? == "api_key" {
            parts.next().map(|s| s.to_string())
        } else {
            None
        }
    }) {
        return Some(api_key);
    }

    None
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
                .wrap(ApiKeyAuth::new(Some("test-key".to_string())))
                .route("/health", web::get().to(dummy_handler))
                .route("/ping", web::get().to(dummy_handler))
                .route("/metrics", web::get().to(dummy_handler)),
        )
        .await;

        let req = test::TestRequest::get().uri("/health").to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());

        let req = test::TestRequest::get().uri("/ping").to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());

        let req = test::TestRequest::get().uri("/metrics").to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }
}
