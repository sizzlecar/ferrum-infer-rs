use super::*;
use axum::{extract::Request, http::header};
use base64::{engine::general_purpose::STANDARD, Engine};
use ferrum_engine::video_generation::{
    VideoFrameRate, VideoGenerationRequest, VideoReference, VideoReferenceKind,
};
use ferrum_models::audio_input::WaveAudioPlan;
use ferrum_models::qwen_image21::image_input::{ImageInputError, ManagedRgba8Image};
use ferrum_runtime::{HostMemoryAccount, HostMemoryClass, ManagedHostBuffer, ManagedHostBytes};
use futures::StreamExt;
use serde::Deserialize;
use std::borrow::Cow;
use tokio::sync::OwnedSemaphorePermit;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire<'a> {
    #[serde(borrow)]
    model: Cow<'a, str>,
    #[serde(borrow)]
    prompt: Cow<'a, str>,
    seed: u64,
    width: u32,
    height: u32,
    num_frames: u32,
    fps: u32,
    #[serde(default, borrow)]
    image_reference: Option<Vec<Reference<'a>>>,
    #[serde(default, borrow)]
    references: Option<Vec<MediaReference<'a>>>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Reference<'a> {
    #[serde(borrow)]
    image_url: Cow<'a, str>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MediaReference<'a> {
    #[serde(rename = "type")]
    kind: VideoReferenceKind,
    #[serde(borrow)]
    data_url: Cow<'a, str>,
}

pub(super) async fn read(
    request: Request,
    videos: &VideoHttpService,
    slot: Arc<OwnedSemaphorePermit>,
) -> Result<(VideoGenerationRequest, Arc<OwnedSemaphorePermit>), ServerError> {
    Ingress {
        host: videos.service.host_account().clone(),
        config: videos.config.clone(),
        model: videos.model_name.clone(),
        slot,
        closing: videos.shutdown_signal.subscribe(),
    }
    .read(request)
    .await
}

/// Owns admission through upload and blocking media validation. Cancelling or
/// timing out an upload drops both its original slot and partial Host backing.
struct Ingress {
    host: HostMemoryAccount,
    config: VideoHttpConfig,
    model: Arc<str>,
    slot: Arc<OwnedSemaphorePermit>,
    closing: tokio::sync::watch::Receiver<bool>,
}
impl Ingress {
    async fn read(
        self,
        request: Request,
    ) -> Result<(VideoGenerationRequest, Arc<OwnedSemaphorePermit>), ServerError> {
        let Self {
            host,
            config,
            model,
            slot,
            closing,
        } = self;
        let body = read_body(
            request,
            &host,
            config.maximum_body_bytes,
            config.upload_idle_timeout,
            config.upload_timeout,
            closing,
        )
        .await?;
        // Actual PNG validation is CPU work. The original ingress slot and Host
        // backing remain in this task even if the HTTP caller disconnects.
        tokio::task::spawn_blocking(move || {
            let input = parse(body, &host, &config, &model)?;
            Ok((input, slot))
        })
        .await
        .map_err(internal)?
    }
}

async fn read_body(
    request: Request,
    host: &HostMemoryAccount,
    maximum: usize,
    idle_timeout: std::time::Duration,
    timeout: std::time::Duration,
    mut closing: tokio::sync::watch::Receiver<bool>,
) -> Result<ManagedHostBytes, ServerError> {
    let started = tokio::time::Instant::now();
    let deadline = started + timeout;
    let mut idle_deadline = started + idle_timeout;
    if request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.split(';').next())
        .map(str::trim)
        != Some("application/json")
    {
        return Err(invalid("video submission requires application/json"));
    }
    if let Some(length) = request.headers().get(header::CONTENT_LENGTH) {
        let length = length
            .to_str()
            .map_err(invalid)?
            .parse::<u64>()
            .map_err(invalid)?;
        if length > maximum as u64 {
            return Err(invalid("video request body exceeds configured limit"));
        }
    }
    let mut stream = request.into_body().into_data_stream();
    let mut backing: Option<ManagedHostBuffer> = None;
    let mut used = 0usize;
    loop {
        if *closing.borrow() {
            return Err(ServerError::ServiceUnavailable(
                "video service is shutting down".into(),
            ));
        }
        let next = tokio::select! {
            // A continuously ready body must not starve shutdown or the total
            // deadline. Empty chunks do not extend the idle deadline.
            biased;
            changed = closing.changed() => {
                let _ = changed;
                return Err(ServerError::ServiceUnavailable("video service is shutting down".into()));
            },
            _ = tokio::time::sleep_until(deadline) => {
                return Err(ServerError::RequestTimeout("video upload exceeded its total time limit".into()));
            },
            _ = tokio::time::sleep_until(idle_deadline) => {
                return Err(ServerError::RequestTimeout("video upload stopped receiving data".into()));
            },
            next = stream.next() => next,
        };
        let Some(chunk) = next else {
            break;
        };
        let chunk = chunk.map_err(invalid)?;
        if chunk.is_empty() {
            continue;
        }
        idle_deadline = tokio::time::Instant::now() + idle_timeout;
        let end = used
            .checked_add(chunk.len())
            .filter(|n| *n <= maximum)
            .ok_or_else(|| invalid("video request body exceeds configured limit"))?;
        if backing.as_ref().map_or(0, |b| b.len()) < end {
            let length = backing
                .as_ref()
                .map_or(4096, |b| b.len().saturating_mul(2))
                .max(end)
                .min(maximum);
            let mut next = host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, length)
                .map_err(capacity)?;
            if let Some(old) = backing.as_ref() {
                next.as_mut()[..used].copy_from_slice(&old.as_ref()[..used]);
            }
            backing = Some(next);
        }
        backing.as_mut().expect("admitted body").as_mut()[used..end].copy_from_slice(&chunk);
        used = end;
    }
    backing
        .ok_or_else(|| invalid("empty video request"))?
        .freeze()
        .slice(0..used)
        .map_err(internal)
}

fn parse(
    body: ManagedHostBytes,
    host: &HostMemoryAccount,
    config: &VideoHttpConfig,
    model: &str,
) -> Result<VideoGenerationRequest, ServerError> {
    let wire: Wire<'_> = serde_json::from_slice(body.bytes()).map_err(invalid)?;
    if wire.model != model {
        return Err(invalid("unknown video model"));
    }
    if wire.prompt.trim().is_empty() || wire.prompt.len() > config.maximum_prompt_bytes {
        return Err(invalid("video prompt is empty or exceeds configured limit"));
    }
    if wire.width == 0 || wire.height == 0 || wire.num_frames == 0 || wire.fps == 0 {
        return Err(invalid("video dimensions, frames and fps must be positive"));
    }
    let media = match (wire.references, wire.image_reference) {
        (Some(media), None) => media,
        (None, Some(images)) => images
            .into_iter()
            .map(|image| MediaReference {
                kind: VideoReferenceKind::Image,
                data_url: image.image_url,
            })
            .collect(),
        _ => {
            return Err(invalid(
                "supply exactly one ordered references or image_reference list",
            ))
        }
    };
    config
        .references
        .validate_kinds(media.iter().map(|r| r.kind))
        .map_err(invalid)?;
    let mut references = Vec::with_capacity(media.len());
    for reference in media {
        let (encoded, byte_limit) = match reference.kind {
            VideoReferenceKind::Image => (
                reference.data_url.strip_prefix("data:image/png;base64,"),
                config.image_limits.maximum_encoded_bytes(),
            ),
            VideoReferenceKind::Audio => (
                reference
                    .data_url
                    .strip_prefix("data:audio/wav;base64,")
                    .or_else(|| reference.data_url.strip_prefix("data:audio/x-wav;base64,")),
                config.audio_limits.maximum_encoded_bytes,
            ),
            VideoReferenceKind::Video => (
                [
                    "data:video/mp4;base64,",
                    "data:video/webm;base64,",
                    "data:video/quicktime;base64,",
                    "data:video/x-matroska;base64,",
                ]
                .into_iter()
                .find_map(|prefix| reference.data_url.strip_prefix(prefix)),
                config.image_limits.maximum_encoded_bytes(),
            ),
        };
        let encoded = encoded.ok_or_else(|| invalid("reference requires a matching PNG, video container or WAVE base64 data URL; remote URLs and paths are not fetched"))?;
        let maximum = base64::decoded_len_estimate(encoded.len());
        // This estimate includes at most two padding bytes. The decoded extent
        // is checked separately; even temporary backing has original Host credit.
        if maximum == 0 || maximum > byte_limit.saturating_add(2) {
            return Err(invalid("encoded reference exceeds configured limit"));
        }
        let mut decoded = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, maximum)
            .map_err(capacity)?;
        let length = STANDARD
            .decode_slice(encoded, decoded.as_mut())
            .map_err(invalid)?;
        if length == 0 || length > byte_limit {
            return Err(invalid("encoded reference exceeds configured limit"));
        }
        let encoded = decoded.freeze().slice(0..length).map_err(internal)?;
        // No image-shaped bytes are accepted on signature alone. The managed
        // decoder checks framing/CRC/static RGB(A)8 and admitted pixel bounds.
        // Pixels are discarded here; the worker later owns its actual model
        // preprocessing. No extra decoded image is retained in the queue.
        match reference.kind {
            VideoReferenceKind::Image => {
                ManagedRgba8Image::decode_png(host, &encoded, config.image_limits).map_err(
                    |e| match e {
                        ImageInputError::Host(e) => capacity(e),
                        other => invalid(other),
                    },
                )?;
            }
            VideoReferenceKind::Audio => {
                let plan =
                    WaveAudioPlan::parse(encoded.clone(), config.audio_limits).map_err(invalid)?;
                let maximum_samples = u64::from(wire.num_frames) * u64::from(plan.sampling_rate())
                    / u64::from(wire.fps);
                plan.validate_samples(usize::try_from(maximum_samples).map_err(invalid)?)
                    .map_err(invalid)?;
            }
            // The shared worker validates container geometry before admission;
            // the HTTP task does not launch a second codec or retain raw frames.
            VideoReferenceKind::Video => {}
        }
        references.push(VideoReference::from_encoded(reference.kind, encoded));
    }
    Ok(VideoGenerationRequest {
        parameters: VideoGenerationParameters {
            prompt: wire.prompt.into_owned(),
            width: wire.width,
            height: wire.height,
            num_frames: wire.num_frames,
            frame_rate: VideoFrameRate {
                numerator: wire.fps,
                denominator: 1,
            },
            seed: wire.seed,
        },
        references,
    })
}

#[cfg(test)]
mod tests;
