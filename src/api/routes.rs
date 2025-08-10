//! Route configuration for the API endpoints
//!
//! This module configures all the HTTP routes and their corresponding handlers.

use super::handlers;
use actix_web::{web, HttpResponse};

/// Configure all API routes
pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg
        // OpenAI-compatible endpoints
        .route("/v1/chat/completions", web::post().to(handlers::chat_completions))
        .route("/v1/completions", web::post().to(handlers::completions))
        .route("/v1/models", web::get().to(handlers::list_models))
        .route("/v1/engines", web::get().to(handlers::list_engines)) // Deprecated
        
        // Health and monitoring endpoints
        .route("/health", web::get().to(handlers::health_check))
        .route("/ping", web::get().to(ping))
        .route("/metrics", web::get().to(handlers::metrics))
        
        // Catch-all for unmatched routes
        .default_service(web::route().to(handlers::not_found));
}

/// Simple ping endpoint for basic connectivity testing
async fn ping() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({
        "message": "pong",
        "timestamp": chrono::Utc::now().timestamp(),
        "version": env!("CARGO_PKG_VERSION")
    }))
}