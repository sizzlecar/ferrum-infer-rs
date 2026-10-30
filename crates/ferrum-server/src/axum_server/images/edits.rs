//! Single-image OpenAI multipart adapter. Transport/parser buffers are bounded
//! by EDIT_BODY_LIMIT. Uploaded PNG storage grows under Host admission and is
//! compacted to its actual encoded length before entering the worker queue.

use super::*;
use axum::extract::{
    multipart::{Field, MultipartRejection},
    Multipart,
};
use ferrum_engine::image_generation::ImageEditRequest;
use ferrum_models::qwen_image21::image_input::ImageInputLimits;
use ferrum_runtime::{HostMemoryAccount, HostMemoryClass, ManagedHostBytes};
use std::collections::BTreeSet;

pub(in crate::axum_server) const EDIT_BODY_LIMIT: usize =
    ImageInputLimits::DEFAULT_MAXIMUM_ENCODED_BYTES + 2 * 1024 * 1024;
const PROMPT_LIMIT: usize = 1024 * 1024;

pub(in crate::axum_server) async fn edits_handler(
    State(state): State<AppState>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Response, ServerError> {
    let images = state
        .images
        .as_ref()
        .ok_or_else(|| ServerError::ServiceUnavailable("image generator not loaded".into()))?;
    if !images.service.supports_edits() {
        return Err(ServerError::unsupported_feature(
            "loaded image pipeline does not support editing",
            None,
        ));
    }
    let multipart =
        multipart.map_err(|error| invalid(format!("invalid multipart request: {error}")))?;
    // Reserve the same output-session capacity used by generation before
    // retaining an upload or joining the worker queue.
    let session = images
        .output
        .try_open_session()
        .map_err(|error| ServerError::CapacityExceeded(error.to_string()))?;
    let (wire, image) = parse(multipart, images.service.host_account()).await?;
    let request = ImageEditRequest {
        generation: wire.convert(images.model_name(), &images.config)?,
        image,
    };
    images
        .service
        .validate_edit_request(&request)
        .map_err(service_error)?;
    let response = admit_response(images, session, &request.generation)?;
    let started = Instant::now();
    let (result, grant) = images
        .service
        .try_submit_edit_with_output(request, response)
        .map_err(service_error)?
        .wait_with_output()
        .await
        .map_err(service_error)?;
    let elapsed = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX);
    let admission = grant.into_inner::<response::Admission>().map_err(|_| {
        ServerError::InternalError(
            "image edit response grant belongs to another output gate".into(),
        )
    })?;
    encode_result(admission, result, elapsed).await
}

async fn parse(
    mut multipart: Multipart,
    host: &HostMemoryAccount,
) -> Result<(ImageRequest, ManagedHostBytes), ServerError> {
    let mut seen = BTreeSet::new();
    let mut model = None;
    let mut prompt = None;
    let mut n = 1;
    let mut size = None;
    let mut response_format = b64_json();
    let mut seed = None;
    let mut image = None;
    while let Some(mut field) = multipart.next_field().await.map_err(invalid)? {
        let name = field
            .name()
            .filter(|name| name.len() <= 64)
            .ok_or_else(|| invalid("multipart field requires a bounded name"))?
            .to_owned();
        // image and image[] are aliases for the single accepted file. No
        // filename is interpreted as a filesystem path or model identity.
        let key = if name == "image[]" { "image" } else { &name };
        if !matches!(
            key,
            "model" | "prompt" | "n" | "size" | "response_format" | "seed" | "image"
        ) {
            return Err(ServerError::unsupported_feature(
                format!("unsupported image edit field: {name}"),
                None,
            ));
        }
        if !seen.insert(key.to_owned()) {
            return Err(invalid(
                "duplicate field or multiple reference images are unsupported",
            ));
        }
        if key == "image" {
            if field
                .content_type()
                .is_some_and(|mime| mime != "image/png" && mime != "application/octet-stream")
            {
                return Err(ServerError::unsupported_feature(
                    "image input must be a static RGB/RGBA8 PNG",
                    Some("image"),
                ));
            }
            image = Some(read_image(&mut field, host).await?);
        } else {
            if field.file_name().is_some() {
                return Err(invalid("text field cannot be a file"));
            }
            let value = read_text(field, if key == "prompt" { PROMPT_LIMIT } else { 1024 }).await?;
            match key {
                "model" => model = Some(value),
                "prompt" => prompt = Some(value),
                "size" => size = Some(value),
                "response_format" => response_format = value,
                "n" => n = decimal(&value, "n")?,
                "seed" => seed = Some(decimal(&value, "seed")?),
                _ => unreachable!(),
            }
        }
    }
    Ok((
        ImageRequest {
            model: model.ok_or_else(|| invalid("missing model"))?,
            prompt: prompt.ok_or_else(|| invalid("missing prompt"))?,
            n,
            size,
            response_format,
            seed,
        },
        image.ok_or_else(|| invalid("missing image"))?,
    ))
}

async fn read_image(
    field: &mut Field<'_>,
    host: &HostMemoryAccount,
) -> Result<ManagedHostBytes, ServerError> {
    let maximum = ImageInputLimits::default().maximum_encoded_bytes();
    let mut backing: Option<ferrum_runtime::ManagedHostBuffer> = None;
    let mut count = 0usize;
    while let Some(chunk) = field.chunk().await.map_err(invalid)? {
        let end = count
            .checked_add(chunk.len())
            .filter(|&length| length <= maximum)
            .ok_or_else(|| invalid(format!("encoded image exceeds {maximum} bytes")))?;
        let previous_capacity = backing.as_ref().map_or(0, |buffer| buffer.len());
        if end > previous_capacity {
            let capacity = previous_capacity
                .checked_mul(2)
                .unwrap_or(maximum)
                .max(end)
                .min(maximum);
            // Admit the new allocation while the old one is still charged:
            // copying, compaction and failures must preserve the real peak.
            let mut next = host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, capacity)
                .map_err(|error| ServerError::CapacityExceeded(error.to_string()))?;
            if let Some(previous) = &backing {
                next.as_mut()[..count].copy_from_slice(&previous.as_ref()[..count]);
            }
            backing = Some(next);
        }
        if let Some(buffer) = &mut backing {
            buffer.as_mut()[count..end].copy_from_slice(&chunk);
        }
        count = end;
    }
    let backing = backing.ok_or_else(|| invalid("image input is empty"))?;
    if !backing.as_ref()[..count].starts_with(b"\x89PNG\r\n\x1a\n") {
        return Err(ServerError::unsupported_feature(
            "image input is not PNG",
            Some("image"),
        ));
    }
    if count == backing.len() {
        return Ok(backing.freeze());
    }
    let mut exact = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, count)
        .map_err(|error| ServerError::CapacityExceeded(error.to_string()))?;
    exact.as_mut().copy_from_slice(&backing.as_ref()[..count]);
    Ok(exact.freeze())
}

async fn read_text(mut field: Field<'_>, maximum: usize) -> Result<String, ServerError> {
    // Prompt/control strings are separately bounded request metadata, not image
    // pixel payload. UTF-8 is checked after chunks are joined (a codepoint may
    // straddle transport chunks).
    let mut bytes = Vec::new();
    while let Some(chunk) = field.chunk().await.map_err(invalid)? {
        if bytes
            .len()
            .checked_add(chunk.len())
            .is_none_or(|length| length > maximum)
        {
            return Err(invalid("multipart text field exceeds its byte limit"));
        }
        bytes.extend_from_slice(&chunk);
    }
    String::from_utf8(bytes).map_err(invalid)
}
fn decimal<T: std::str::FromStr>(value: &str, field: &str) -> Result<T, ServerError> {
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
        return Err(invalid(format!(
            "{field} must be an unsigned decimal integer"
        )));
    }
    value
        .parse()
        .map_err(|_| invalid(format!("{field} is out of range")))
}
fn invalid(error: impl std::fmt::Display) -> ServerError {
    ServerError::invalid_request(error.to_string(), None)
}

#[cfg(test)]
mod tests;
