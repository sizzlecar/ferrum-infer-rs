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
            let query_key = req
                .query_string()
                .split('&')
                .find_map(|pair| {
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