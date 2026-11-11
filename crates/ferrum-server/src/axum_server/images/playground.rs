//! A same-origin image client. Inference still uses the original image routes,
//! queue, cancellation and response admission.

use super::{AppState, ImageGenerationRequest, ImageHttpService, ServerError};
use axum::{
    extract::State,
    http::header,
    response::{IntoResponse, Response},
    Json,
};
use ferrum_models::qwen_image21::image_input::ImageInputLimits;
use ferrum_output::png::PngDimensions;
use serde::Serialize;

#[derive(Serialize)]
struct PlaygroundConfig<'a> {
    model: &'a str,
    default_width: u32,
    default_height: u32,
    default_seed: u64,
    supports_edits: bool,
    maximum_reference_images: usize,
    maximum_reference_image_bytes: usize,
    sizes: Vec<PlaygroundSize>,
    /// The browser's in-flight request limit, not a server batch capability.
    maximum_concurrency: u8,
}

#[derive(Serialize)]
struct PlaygroundSize {
    width: u32,
    height: u32,
    label: String,
}

fn images(state: &AppState) -> Result<&ImageHttpService, ServerError> {
    state
        .images
        .as_ref()
        .ok_or_else(|| ServerError::ServiceUnavailable("image generator not loaded".into()))
}

fn configuration(images: &ImageHttpService) -> PlaygroundConfig<'_> {
    let mut sizes = Vec::<PlaygroundSize>::new();
    let mut add_size = |width, height| {
        let request = ImageGenerationRequest {
            prompt: String::new(),
            width,
            height,
            seed: images.config.default_seed,
            n: 1,
        };
        let codec_accepts = PngDimensions::new(width.into(), height.into())
            .is_ok_and(|dimensions| images.config.png_limits.validate(dimensions).is_ok());
        if images.service.validate_request(&request).is_ok()
            && codec_accepts
            && !sizes
                .iter()
                .any(|size| size.width == width && size.height == height)
        {
            sizes.push(PlaygroundSize {
                width,
                height,
                label: format!("{width} × {height}"),
            });
        }
    };
    add_size(images.config.default_width, images.config.default_height);
    for edge in [256, 512, 768, 1024, 1536, 2048] {
        add_size(edge, edge);
        let short = edge * 3 / 4;
        add_size(edge, short);
        add_size(short, edge);
    }
    PlaygroundConfig {
        model: images.model_name(),
        default_width: images.config.default_width,
        default_height: images.config.default_height,
        default_seed: images.config.default_seed,
        supports_edits: images.service.supports_edits(),
        maximum_reference_images: images.service.maximum_reference_images(),
        maximum_reference_image_bytes: ImageInputLimits::default().maximum_encoded_bytes(),
        sizes,
        maximum_concurrency: 2,
    }
}

pub(in crate::axum_server) async fn config(
    State(state): State<AppState>,
) -> Result<Response, ServerError> {
    Ok((
        [(header::CACHE_CONTROL, "no-store")],
        Json(configuration(images(&state)?)),
    )
        .into_response())
}

fn asset(
    state: &AppState,
    content_type: &'static str,
    body: &'static str,
) -> Result<Response, ServerError> {
    images(state)?;
    Ok((
        [
            (header::CONTENT_TYPE, content_type),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        body,
    )
        .into_response())
}

pub(in crate::axum_server) async fn page(
    State(state): State<AppState>,
) -> Result<Response, ServerError> {
    asset(
        &state,
        "text/html; charset=utf-8",
        include_str!("playground/index.html"),
    )
}

pub(in crate::axum_server) async fn javascript(
    State(state): State<AppState>,
) -> Result<Response, ServerError> {
    asset(
        &state,
        "text/javascript; charset=utf-8",
        include_str!("playground/app.js"),
    )
}

pub(in crate::axum_server) async fn stylesheet(
    State(state): State<AppState>,
) -> Result<Response, ServerError> {
    asset(
        &state,
        "text/css; charset=utf-8",
        include_str!("playground/style.css"),
    )
}

#[cfg(test)]
mod tests;
