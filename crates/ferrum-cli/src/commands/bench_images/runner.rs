use super::capture::{capture_validated_png, CaptureSink};
use super::*;
use ferrum_bench_core::images::{
    validate_image_response_with_png, ImageFailureKind, ImageHttpRequest, ImageSample,
};
use futures::{stream::FuturesUnordered, StreamExt};
use std::{sync::Arc, time::Instant};

/// Endpoints run sequentially with an identical ordered dataset. Active HTTP
/// requests AND pending decoder/capture jobs share the same fixed concurrency bound.
/// No requests are retried: retrying would silently change the workload.
pub async fn run_benchmark(
    config: ImageBenchmarkConfig,
    dataset: ImageDataset,
) -> anyhow::Result<ImageBenchmarkReport> {
    config.validate(&dataset)?;
    let capture = config
        .capture
        .as_ref()
        .map(CaptureSink::create)
        .transpose()?;
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
        let mut captures = capture.as_ref().map(|_| Vec::new());
        let mut repetitions = Vec::new();
        let mut measured_seconds = 0.0;
        for phase in [ImageRequestPhase::Warmup, ImageRequestPhase::Measured] {
            let count = if phase == ImageRequestPhase::Warmup {
                config.warmup_repetitions
            } else {
                config.repetitions
            };
            for repetition in 0..count {
                let (run, mut requests, mut artifacts) = run_round(
                    &client,
                    endpoint_index,
                    endpoint,
                    &config,
                    &dataset,
                    phase,
                    repetition,
                    capture.as_ref(),
                )
                .await;
                if phase == ImageRequestPhase::Measured {
                    measured_seconds += run.metrics.elapsed_seconds;
                }
                repetitions.push(run);
                records.append(&mut requests);
                if let Some(captures) = &mut captures {
                    captures.append(&mut artifacts);
                }
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
            captures,
        });
    }
    let capture_summary = capture.as_ref().map(|sink| sink.summary(&endpoint_reports));
    let mut evidence = ImageEvidenceScope::default();
    if capture_summary.is_some() {
        evidence.throughput_scope = "successful validated images divided by measured wall time including errors, local validation and PNG capture IO; warmup excluded; capture retains request slots and changes subsequent dispatch pacing";
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
        evidence,
        capture: capture_summary,
    })
}

async fn run_round(
    client: &reqwest::Client,
    endpoint_index: usize,
    endpoint: &ImageBenchmarkEndpoint,
    config: &ImageBenchmarkConfig,
    dataset: &ImageDataset,
    phase: ImageRequestPhase,
    repetition: usize,
    capture: Option<&Arc<CaptureSink>>,
) -> (
    ImageBenchmarkRepetition,
    Vec<ImageRequestRecord>,
    Vec<PngCaptureRecord>,
) {
    let start = Instant::now();
    let mut next = 0usize;
    let mut active = FuturesUnordered::new();
    let mut records = Vec::with_capacity(dataset.samples.len());
    let mut captures = Vec::new();
    let mut http_span = 0.0_f64;
    while next < dataset.samples.len() || !active.is_empty() {
        while next < dataset.samples.len() && active.len() < config.concurrency {
            active.push(run_request(
                client,
                endpoint_index,
                endpoint,
                config,
                &dataset.samples[next],
                next,
                phase,
                repetition,
                start,
                capture.cloned(),
            ));
            next += 1;
        }
        if let Some((record, body_complete_seconds, captured)) = active.next().await {
            http_span = http_span.max(body_complete_seconds);
            records.push(record);
            if let Some(captured) = captured {
                captures.push(captured);
            }
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    records.sort_by_key(|record| record.sample_index);
    captures.sort_by_key(|record| record.sample_index);
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
        captures,
    )
}

async fn run_request(
    client: &reqwest::Client,
    endpoint_index: usize,
    endpoint: &ImageBenchmarkEndpoint,
    config: &ImageBenchmarkConfig,
    sample: &ImageSample,
    sample_index: usize,
    phase: ImageRequestPhase,
    repetition: usize,
    round_start: Instant,
    capture: Option<Arc<CaptureSink>>,
) -> (ImageRequestRecord, f64, Option<PngCaptureRecord>) {
    let capture_enabled = capture.is_some();
    let mut capture_result = PngCaptureResult::NotProduced;
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
            // Decode and optional filesystem IO cannot stall the reactor. This
            // job retains its slot until joined: no unbounded writer queue.
            let result = tokio::task::spawn_blocking(move || {
                let validated = validate_image_response_with_png(&body, width, height, &limits);
                let validation_ms = validation_start.elapsed().as_secs_f64() * 1000.0;
                // No need to retain JSON/base64 while the original PNG is saved.
                drop(body);
                let result = validated.map(|response| match capture.as_ref() {
                    Some(sink) => {
                        let (image, captured) = capture_validated_png(response, |image, png| {
                            sink.save(endpoint_index, phase, repetition, sample_index, image, png)
                        });
                        (image, Some(captured))
                    }
                    None => {
                        let (image, _png) = response.into_parts();
                        (image, None)
                    }
                });
                (result, validation_ms)
            })
            .await;
            // Preserve the original capture-disabled cutoff after worker join.
            // It also records elapsed work/queue time if validation panics.
            record.validation_ms = validation_start.elapsed().as_secs_f64() * 1000.0;
            match result {
                Ok((Ok((image, captured)), validation_ms)) => {
                    if capture_enabled {
                        record.validation_ms = validation_ms;
                    }
                    record.image = Some(image);
                    if let Some(captured) = captured {
                        capture_result = captured;
                    }
                }
                Ok((Err(error), validation_ms)) => {
                    if capture_enabled {
                        record.validation_ms = validation_ms;
                    }
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
    let captured = capture_enabled.then(|| PngCaptureRecord {
        sample_index,
        sample_id: sample.id.clone(),
        phase,
        repetition,
        result: capture_result,
    });
    (record, body_complete_seconds, captured)
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
