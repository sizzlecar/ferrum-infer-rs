use super::*;
use ferrum_bench_core::images::{
    validate_image_response, ImageFailureKind, ImageHttpRequest, ImageSample,
};
use futures::{stream::FuturesUnordered, StreamExt};
use std::time::Instant;

/// Endpoints run sequentially with an identical ordered dataset. Active HTTP
/// requests AND pending decoder jobs share the same fixed concurrency bound.
/// No requests are retried: retrying would silently change the workload.
pub async fn run_benchmark(
    config: ImageBenchmarkConfig,
    dataset: ImageDataset,
) -> anyhow::Result<ImageBenchmarkReport> {
    config.validate(&dataset)?;
    let started_unix_ms = super::started_unix_ms()?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(config.timeout_seconds))
        .connect_timeout(Duration::from_secs(config.timeout_seconds.min(30)))
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .build()
        .context("build image benchmark HTTP client")?;
    let mut endpoint_reports = Vec::new();
    for (endpoint_index, endpoint) in config.endpoints.iter().enumerate() {
        let mut records = Vec::new();
        let mut repetitions = Vec::new();
        let mut measured_seconds = 0.0;
        for phase in [ImageRequestPhase::Warmup, ImageRequestPhase::Measured] {
            let count = if phase == ImageRequestPhase::Warmup {
                config.warmup_repetitions
            } else {
                config.repetitions
            };
            for repetition in 0..count {
                let (run, mut requests) =
                    run_round(&client, endpoint, &config, &dataset, phase, repetition).await;
                if phase == ImageRequestPhase::Measured {
                    measured_seconds += run.metrics.elapsed_seconds;
                }
                repetitions.push(run);
                records.append(&mut requests);
            }
        }
        let measured: Vec<_> = records
            .iter()
            .filter(|r| r.phase == ImageRequestPhase::Measured)
            .cloned()
            .collect();
        let metrics =
            ImageMetrics::from_records(measured_seconds, &measured, config.latency_slo_p99_ms);
        endpoint_reports.push(ImageEndpointReport {
            endpoint_index,
            repetitions,
            measured: metrics,
            requests: records,
        });
    }
    Ok(ImageBenchmarkReport {
        schema_version: 1,
        client_version: env!("CARGO_PKG_VERSION").into(),
        started_unix_ms,
        dataset_sha256: dataset.fingerprint()?,
        dataset_file_sha256: None,
        configuration: config,
        dataset,
        endpoints: endpoint_reports,
        evidence: ImageEvidenceScope::default(),
    })
}

async fn run_round(
    client: &reqwest::Client,
    endpoint: &ImageBenchmarkEndpoint,
    config: &ImageBenchmarkConfig,
    dataset: &ImageDataset,
    phase: ImageRequestPhase,
    repetition: usize,
) -> (ImageBenchmarkRepetition, Vec<ImageRequestRecord>) {
    let start = Instant::now();
    let mut next = 0usize;
    let mut active = FuturesUnordered::new();
    let mut records = Vec::with_capacity(dataset.samples.len());
    let mut http_span = 0.0_f64;
    while next < dataset.samples.len() || !active.is_empty() {
        while next < dataset.samples.len() && active.len() < config.concurrency {
            active.push(run_request(
                client,
                endpoint,
                config,
                &dataset.samples[next],
                next,
                phase,
                repetition,
                start,
            ));
            next += 1;
        }
        if let Some((record, body_complete_seconds)) = active.next().await {
            http_span = http_span.max(body_complete_seconds);
            records.push(record);
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    records.sort_by_key(|record| record.sample_index);
    let metrics = ImageMetrics::from_records(
        elapsed,
        &records,
        (phase == ImageRequestPhase::Measured)
            .then_some(config.latency_slo_p99_ms)
            .flatten(),
    );
    let rate = (http_span > 0.0).then_some(metrics.succeeded as f64 / http_span);
    (
        ImageBenchmarkRepetition {
            phase,
            repetition,
            metrics,
            http_completion_span_seconds: http_span,
            successful_images_per_http_completion_second: rate,
        },
        records,
    )
}

async fn run_request(
    client: &reqwest::Client,
    endpoint: &ImageBenchmarkEndpoint,
    config: &ImageBenchmarkConfig,
    sample: &ImageSample,
    sample_index: usize,
    phase: ImageRequestPhase,
    repetition: usize,
    round_start: Instant,
) -> (ImageRequestRecord, f64) {
    let start = Instant::now();
    let mut record = ImageRequestRecord {
        sample_index,
        sample_id: sample.id.clone(),
        phase,
        repetition,
        dispatch_index: sample_index,
        http_status: None,
        e2e_ms: 0.0,
        validation_ms: 0.0,
        response_bytes: 0,
        image: None,
        failure: None,
        error: None,
    };
    let received = receive(
        client,
        endpoint,
        sample,
        config.limits.maximum_response_bytes,
        &mut record,
    )
    .await;
    record.e2e_ms = start.elapsed().as_secs_f64() * 1000.0;
    let body_complete_seconds = round_start.elapsed().as_secs_f64();
    match received {
        Err((kind, message)) => {
            record.failure = Some(kind);
            record.error = Some(message.into());
        }
        Ok(body) => {
            let width = sample.width;
            let height = sample.height;
            let limits = config.limits.clone();
            let validation_start = Instant::now();
            // CPU decode cannot stall the async reactor. This job retains its
            // active-request slot until joined; no unbounded blocking queue.
            let result = tokio::task::spawn_blocking(move || {
                validate_image_response(&body, width, height, &limits)
            })
            .await;
            record.validation_ms = validation_start.elapsed().as_secs_f64() * 1000.0;
            match result {
                Ok(Ok(image)) => record.image = Some(image),
                Ok(Err(error)) => {
                    record.failure = Some(ImageFailureKind::InvalidResponse);
                    record.error = Some(error.to_string());
                }
                Err(_) => {
                    record.failure = Some(ImageFailureKind::ClientTask);
                    record.error = Some("PNG validation task failed".into());
                }
            }
        }
    }
    (record, body_complete_seconds)
}

async fn receive(
    client: &reqwest::Client,
    endpoint: &ImageBenchmarkEndpoint,
    sample: &ImageSample,
    maximum_bytes: usize,
    record: &mut ImageRequestRecord,
) -> Result<Vec<u8>, (ImageFailureKind, &'static str)> {
    let mut response = client
        .post(&endpoint.url)
        .json(&ImageHttpRequest::new(&endpoint.model, sample))
        .send()
        .await
        .map_err(transport_error)?;
    record.http_status = Some(response.status().as_u16());
    if !response.status().is_success() {
        return Err((
            ImageFailureKind::HttpStatus,
            "server returned a non-success HTTP status",
        ));
    }
    if response
        .content_length()
        .is_some_and(|length| length > maximum_bytes as u64)
    {
        return Err((
            ImageFailureKind::ResponseLimit,
            "HTTP content length exceeds response limit",
        ));
    }
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(transport_error)? {
        let length = body
            .len()
            .checked_add(chunk.len())
            .filter(|length| *length <= maximum_bytes)
            .ok_or((
                ImageFailureKind::ResponseLimit,
                "HTTP body exceeds response limit",
            ))?;
        body.try_reserve(chunk.len()).map_err(|_| {
            (
                ImageFailureKind::ResponseLimit,
                "cannot reserve HTTP response buffer",
            )
        })?;
        body.extend_from_slice(&chunk);
        record.response_bytes = length;
    }
    Ok(body)
}

fn transport_error(error: reqwest::Error) -> (ImageFailureKind, &'static str) {
    if error.is_timeout() {
        (ImageFailureKind::Timeout, "HTTP request timed out")
    } else {
        (ImageFailureKind::Transport, "HTTP transport failed")
    }
}
