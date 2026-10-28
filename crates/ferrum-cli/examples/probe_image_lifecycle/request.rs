use super::*;
use bytes::Bytes;
use ferrum_bench_core::images::{
    validate_image_response, validate_reference_png, ImageHttpRequest, ImageRequestMode,
    ImageSample, ValidatedImage,
};
use std::collections::BTreeMap;

pub(super) struct Prepared {
    pub dataset_sha256: String,
    pub requests: Vec<Input>,
    pub references: Vec<ValidatedImage>,
}
#[derive(Clone)]
pub(super) struct Input {
    pub sample: ImageSample,
    reference: Option<Bytes>,
}

pub(super) fn prepare(args: &Arguments, mut dataset: ImageDataset) -> anyhow::Result<Prepared> {
    let mut references = Vec::<ValidatedImage>::new();
    let mut files = BTreeMap::<PathBuf, (Bytes, usize)>::new();
    let mut total = 0usize;
    let mut requests = Vec::new();
    for sample in &mut dataset.samples[..args.requests] {
        let mode = sample.mode.as_mut().unwrap_or(&mut dataset.mode);
        let reference = match mode {
            ImageRequestMode::Generate => None,
            ImageRequestMode::Edit { image } => {
                let path = if image.path.is_absolute() {
                    image.path.clone()
                } else {
                    args.dataset
                        .parent()
                        .unwrap_or(Path::new("."))
                        .join(&image.path)
                };
                if !files.contains_key(&path) {
                    let bytes = read_bounded(&path, args.maximum_reference_bytes - total)?;
                    let mut limits = args.limits();
                    limits.maximum_png_bytes = args.maximum_reference_bytes;
                    let validated = validate_reference_png(&bytes, &limits)?;
                    total = total
                        .checked_add(bytes.len())
                        .context("reference byte total overflow")?;
                    let index = references.len();
                    references.push(validated);
                    files.insert(path.clone(), (Bytes::from(bytes), index));
                }
                let (bytes, index) = &files[&path];
                let actual = &references[*index].png_sha256;
                ensure!(
                    image
                        .encoded_sha256
                        .as_ref()
                        .is_none_or(|expected| expected == actual),
                    "reference PNG identity differs from dataset"
                );
                image.encoded_sha256 = Some(actual.clone());
                Some(bytes.clone())
            }
        };
        requests.push(Input {
            sample: sample.clone(),
            reference,
        });
    }
    Ok(Prepared {
        dataset_sha256: dataset.fingerprint()?,
        requests,
        references,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Phase {
    Perturbation,
    Recovery,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Outcome {
    Validated,
    ClientFutureDropped,
    ClientBodyDropped,
    HttpRejected,
    HttpError,
    TransportError,
    TimedOut,
    InvalidResponse,
    ClientTaskError,
}
#[derive(Debug, Serialize)]
pub(super) struct Record {
    pub sample_id: String,
    pub seed: u64,
    pub width: u32,
    pub height: u32,
    pub phase: Phase,
    pub request_index: usize,
    pub started_ms: f64,
    pub headers_received_ms: Option<f64>,
    pub body_hold_started_ms: Option<f64>,
    pub body_hold_finished_ms: Option<f64>,
    pub body_dropped_ms: Option<f64>,
    pub finished_ms: f64,
    pub http_status: Option<u16>,
    pub response_bytes: usize,
    pub outcome: Outcome,
    pub image: Option<ValidatedImage>,
    pub error: Option<String>,
}

async fn bounded_body(
    mut response: reqwest::Response,
    maximum: usize,
) -> Result<Vec<u8>, (Outcome, String)> {
    if response
        .content_length()
        .is_some_and(|length| length > maximum as u64)
    {
        return Err((
            Outcome::InvalidResponse,
            "response byte limit exceeded".into(),
        ));
    }
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| (Outcome::TransportError, format!("body transport: {error}")))?
    {
        if chunk.len() > maximum - body.len() {
            return Err((
                Outcome::InvalidResponse,
                "response byte limit exceeded".into(),
            ));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn build(
    client: &reqwest::Client,
    args: &Arguments,
    input: &Input,
) -> anyhow::Result<reqwest::RequestBuilder> {
    let Some(reference) = &input.reference else {
        return Ok(client
            .post(&args.endpoint)
            .json(&ImageHttpRequest::new(&args.model, &input.sample)));
    };
    let sample = &input.sample;
    let part = reqwest::multipart::Part::stream_with_length(
        reqwest::Body::from(reference.clone()),
        reference.len() as u64,
    )
    .file_name("reference.png")
    .mime_str("image/png")?;
    let mut form = reqwest::multipart::Form::new()
        .text("model", args.model.clone())
        .text("prompt", sample.prompt.clone())
        .text("seed", sample.seed.to_string())
        .text("size", format!("{}x{}", sample.width, sample.height))
        .text("n", "1")
        .text("response_format", "b64_json")
        .part("image", part);
    if let Some(steps) = sample.num_inference_steps {
        form = form.text("num_inference_steps", steps.to_string());
    }
    Ok(client
        .post(
            args.edit_endpoint
                .as_ref()
                .context("edit endpoint missing")?,
        )
        .multipart(form))
}

pub(super) async fn execute(
    client: &reqwest::Client,
    args: &Arguments,
    input: &Input,
    index: usize,
    phase: Phase,
    origin: Instant,
) -> Record {
    let mut record = Record {
        sample_id: input.sample.id.clone(),
        seed: input.sample.seed,
        width: input.sample.width,
        height: input.sample.height,
        phase,
        request_index: index,
        started_ms: origin.elapsed().as_secs_f64() * 1000.0,
        headers_received_ms: None,
        body_hold_started_ms: None,
        body_hold_finished_ms: None,
        body_dropped_ms: None,
        finished_ms: 0.0,
        http_status: None,
        response_bytes: 0,
        outcome: Outcome::ClientTaskError,
        image: None,
        error: None,
    };
    let request = match build(client, args, input) {
        Ok(request) => request,
        Err(error) => {
            record.error = Some(error.to_string());
            record.finished_ms = origin.elapsed().as_secs_f64() * 1000.0;
            return record;
        }
    };
    let result = {
        let operation = async {
            let response = request
                .send()
                .await
                .map_err(|error| (Outcome::TransportError, error.to_string()))?;
            record.headers_received_ms = Some(origin.elapsed().as_secs_f64() * 1000.0);
            let status = response.status();
            record.http_status = Some(status.as_u16());
            if !status.is_success() {
                return Err((
                    if status.as_u16() == 429 {
                        Outcome::HttpRejected
                    } else {
                        Outcome::HttpError
                    },
                    format!("HTTP {}", status.as_u16()),
                ));
            }
            if response
                .content_length()
                .is_some_and(|length| length > args.maximum_response_bytes as u64)
            {
                return Err((
                    Outcome::InvalidResponse,
                    "response byte limit exceeded".into(),
                ));
            }
            if phase == Phase::Perturbation && matches!(args.mode, Mode::HoldBody | Mode::DropBody)
            {
                record.body_hold_started_ms = Some(origin.elapsed().as_secs_f64() * 1000.0);
                // No response.chunk/body future is polled here. HTTP/TCP buffers
                // may still receive data; this does not prove server backpressure.
                tokio::time::sleep(Duration::from_millis(args.hold_body_ms)).await;
                record.body_hold_finished_ms = Some(origin.elapsed().as_secs_f64() * 1000.0);
            }
            if phase == Phase::Perturbation && args.mode == Mode::DropBody {
                drop(response);
                record.body_dropped_ms = Some(origin.elapsed().as_secs_f64() * 1000.0);
                return Err((Outcome::ClientBodyDropped, "client dropped unread response after complete headers; server cleanup requires independent health observation".into()));
            }
            let body = bounded_body(response, args.maximum_response_bytes).await?;
            record.response_bytes = body.len();
            let width = input.sample.width;
            let height = input.sample.height;
            let limits = args.limits();
            let image = tokio::task::spawn_blocking(move || {
                validate_image_response(&body, width, height, &limits)
            })
            .await
            .map_err(|error| (Outcome::ClientTaskError, error.to_string()))?
            .map_err(|error| (Outcome::InvalidResponse, error.to_string()))?;
            Ok(image)
        };
        if phase == Phase::Perturbation && args.mode == Mode::Cancel {
            match tokio::time::timeout(Duration::from_millis(args.cancel_after_ms), operation).await {
                Ok(value) => value,
                Err(_) => Err((Outcome::ClientFutureDropped, "client HTTP future dropped at declared deadline; server cancellation is unproven".into())),
            }
        } else {
            match tokio::time::timeout(Duration::from_millis(args.request_timeout_ms), operation)
                .await
            {
                Ok(value) => value,
                Err(_) => Err((Outcome::TimedOut, "request deadline expired".into())),
            }
        }
    };
    match result {
        Ok(image) => {
            record.image = Some(image);
            record.outcome = Outcome::Validated;
        }
        Err((outcome, error)) => {
            record.outcome = outcome;
            record.error = Some(error.chars().take(1024).collect());
        }
    }
    record.finished_ms = origin.elapsed().as_secs_f64() * 1000.0;
    record
}

pub(super) async fn health_body(
    response: reqwest::Response,
    maximum: usize,
) -> Result<Vec<u8>, (Outcome, String)> {
    bounded_body(response, maximum).await
}
