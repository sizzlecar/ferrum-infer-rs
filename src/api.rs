//! REST API layer with OpenAI-compatible endpoints
//!
//! This module provides HTTP endpoints compatible with OpenAI's API, allowing
//! the inference engine to be used as a drop-in replacement for OpenAI services.

pub mod handlers;
pub mod middleware;
pub mod routes;
pub mod types;

use crate::config::Config;
use crate::error::Result;
use crate::inference::InferenceEngine;
use actix_cors::Cors;
use actix_web::{web, App, HttpServer};
use std::sync::Arc;
use tracing::info;

/// API server state shared across handlers
#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<InferenceEngine>,
    pub config: Config,
}

/// Start the API server
pub async fn start_server(config: Config, engine: Arc<InferenceEngine>) -> Result<()> {
    let bind_address = config.server_address();
    info!("Starting API server on {}", bind_address);

    let app_state = AppState {
        engine: Arc::clone(&engine),
        config: config.clone(),
    };

    HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header()
            .max_age(3600);

        App::new()
            .app_data(web::Data::new(app_state.clone()))
            .wrap(cors)
            .wrap(middleware::logging::RequestLogging)
            .wrap(middleware::auth::ApiKeyAuth::new(config.server.api_key.clone()))
            .configure(routes::configure_routes)
    })
    .bind(&bind_address)?
    .run()
    .await?;

    Ok(())
}

pub use handlers::*;
pub use types::*;