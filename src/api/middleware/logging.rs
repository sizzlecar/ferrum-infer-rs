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

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test, web, App, HttpResponse};

    async fn success_handler() -> Result<HttpResponse, Error> {
        Ok(HttpResponse::Ok().json(serde_json::json!({"status": "success"})))
    }

    async fn client_error_handler() -> Result<HttpResponse, Error> {
        Ok(HttpResponse::BadRequest().json(serde_json::json!({"error": "bad request"})))
    }

    async fn server_error_handler() -> Result<HttpResponse, Error> {
        Ok(HttpResponse::InternalServerError().json(serde_json::json!({"error": "server error"})))
    }

    async fn error_handler() -> Result<HttpResponse, Error> {
        Err(actix_web::error::ErrorInternalServerError("Service error"))
    }

    #[actix_web::test]
    async fn test_logging_middleware_with_success_response() {
        let app = test::init_service(
            App::new()
                .wrap(RequestLogging)
                .route("/success", web::get().to(success_handler)),
        )
        .await;

        let req = test::TestRequest::get()
            .uri("/success")
            .insert_header(("User-Agent", "test-agent/1.0"))
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    #[actix_web::test]
    async fn test_logging_middleware_with_client_error() {
        let app = test::init_service(
            App::new()
                .wrap(RequestLogging)
                .route("/client-error", web::get().to(client_error_handler)),
        )
        .await;

        let req = test::TestRequest::get()
            .uri("/client-error")
            .insert_header(("User-Agent", "test-agent/1.0"))
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 400);
    }

    #[actix_web::test]
    async fn test_logging_middleware_with_server_error() {
        let app = test::init_service(
            App::new()
                .wrap(RequestLogging)
                .route("/server-error", web::get().to(server_error_handler)),
        )
        .await;

        let req = test::TestRequest::get()
            .uri("/server-error")
            .insert_header(("User-Agent", "test-agent/1.0"))
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 500);
    }

    #[actix_web::test]
    async fn test_logging_middleware_with_handler_error() {
        let app = test::init_service(
            App::new()
                .wrap(RequestLogging)
                .route("/error", web::get().to(error_handler)),
        )
        .await;

        let req = test::TestRequest::get()
            .uri("/error")
            .insert_header(("User-Agent", "test-agent/1.0"))
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_server_error());
    }

    #[actix_web::test]
    async fn test_logging_middleware_with_query_parameters() {
        let app = test::init_service(
            App::new()
                .wrap(RequestLogging)
                .route("/test", web::get().to(success_handler)),
        )
        .await;

        let req = test::TestRequest::get()
            .uri("/test?param1=value1&param2=value2")
            .insert_header(("User-Agent", "test-agent/1.0"))
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    #[actix_web::test]
    async fn test_logging_middleware_without_user_agent() {
        let app = test::init_service(
            App::new()
                .wrap(RequestLogging)
                .route("/test", web::get().to(success_handler)),
        )
        .await;

        let req = test::TestRequest::get().uri("/test").to_request();

        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    #[actix_web::test]
    async fn test_logging_middleware_with_different_methods() {
        let app = test::init_service(
            App::new()
                .wrap(RequestLogging)
                .route("/test", web::post().to(success_handler))
                .route("/test", web::put().to(success_handler))
                .route("/test", web::delete().to(success_handler)),
        )
        .await;

        // Test POST
        let req = test::TestRequest::post().uri("/test").to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());

        // Test PUT
        let req = test::TestRequest::put().uri("/test").to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());

        // Test DELETE
        let req = test::TestRequest::delete().uri("/test").to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    #[actix_web::test]
    async fn test_logging_middleware_timing() {
        use std::time::Duration;
        use tokio::time::sleep;

        async fn slow_handler() -> Result<HttpResponse, Error> {
            sleep(Duration::from_millis(100)).await;
            Ok(HttpResponse::Ok().json(serde_json::json!({"status": "success"})))
        }

        let app = test::init_service(
            App::new()
                .wrap(RequestLogging)
                .route("/slow", web::get().to(slow_handler)),
        )
        .await;

        let req = test::TestRequest::get().uri("/slow").to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
        // The middleware should log the duration, which should be >= 100ms
    }
}
