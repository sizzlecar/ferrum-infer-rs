//! Request logging middleware
//!
//! This middleware logs incoming HTTP requests and responses for monitoring
//! and debugging purposes.

use actix_web::{
    dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform},
    Error, HttpMessage,
};
use futures_util::future::LocalBoxFuture;
use std::{
    future::{ready, Ready},
    rc::Rc,
    time::Instant,
};
use tracing::{info, warn};

/// Request logging middleware
pub struct RequestLogging;

impl<S, B> Transform<S, ServiceRequest> for RequestLogging
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Transform = RequestLoggingMiddleware<S>;
    type InitError = ();
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(RequestLoggingMiddleware {
            service: Rc::new(service),
        }))
    }
}

pub struct RequestLoggingMiddleware<S> {
    service: Rc<S>,
}

impl<S, B> Service<ServiceRequest> for RequestLoggingMiddleware<S>
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
        let start_time = Instant::now();
        
        // Extract request information
        let method = req.method().to_string();
        let path = req.path().to_string();
        let query = req.query_string().to_string();
        let user_agent = req
            .headers()
            .get("User-Agent")
            .and_then(|h| h.to_str().ok())
            .unwrap_or("unknown")
            .to_string();
        let remote_addr = req
            .connection_info()
            .peer_addr()
            .unwrap_or("unknown")
            .to_string();

        Box::pin(async move {
            let response = service.call(req).await;
            let duration = start_time.elapsed();

            match &response {
                Ok(service_response) => {
                    let status = service_response.status();
                    let duration_ms = duration.as_millis();

                    if status.is_success() {
                        info!(
                            method = %method,
                            path = %path,
                            query = %query,
                            status = %status,
                            duration_ms = %duration_ms,
                            remote_addr = %remote_addr,
                            user_agent = %user_agent,
                            "HTTP request completed successfully"
                        );
                    } else if status.is_client_error() {
                        warn!(
                            method = %method,
                            path = %path,
                            query = %query,
                            status = %status,
                            duration_ms = %duration_ms,
                            remote_addr = %remote_addr,
                            user_agent = %user_agent,
                            "HTTP request failed with client error"
                        );
                    } else {
                        warn!(
                            method = %method,
                            path = %path,
                            query = %query,
                            status = %status,
                            duration_ms = %duration_ms,
                            remote_addr = %remote_addr,
                            user_agent = %user_agent,
                            "HTTP request failed with server error"
                        );
                    }
                }
                Err(error) => {
                    let duration_ms = duration.as_millis();
                    warn!(
                        method = %method,
                        path = %path,
                        query = %query,
                        duration_ms = %duration_ms,
                        remote_addr = %remote_addr,
                        user_agent = %user_agent,
                        error = %error,
                        "HTTP request failed with error"
                    );
                }
            }

            response
        })
    }
}