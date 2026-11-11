use super::capture::{capture_validated_png, CaptureSink};
use super::edit::{PreparedReference, PreparedReferences};
use super::protocol::{PreparedProtocol, ReceivedPayload, RunControl};
use super::*;
use ferrum_bench_core::images::{ImageFailureKind, ImageHttpRequest, ImageSample};
use futures::{stream::FuturesUnordered, StreamExt};
use std::{sync::Arc, time::Instant};

/// Endpoints run sequentially with an identical ordered dataset. Active HTTP
/// requests AND pending decoder/capture jobs share the same fixed concurrency bound.
/// No requests are retried: retrying would silently change the workload.
pub async fn run_benchmark(
    config: ImageBenchmarkConfig,
    dataset: ImageDataset,
) -> anyhow::Result<ImageBenchmarkReport> {
    run_benchmark_with_control(config, dataset, RunControl::new()).await
}

pub(super) async fn run_benchmark_with_control(
    config: ImageBenchmarkConfig,
    mut dataset: ImageDataset,
    control: RunControl,
) -> anyhow::Result<ImageBenchmarkReport> {
    config.validate(&dataset)?;
    let references = PreparedReferences::load(&mut dataset, &config)?;
    // Read and validate every immutable workflow before any endpoint is called.
    let protocols = config
        .endpoints
        .iter()
        .map(|endpoint| PreparedProtocol::prepare(endpoint, &config, &dataset, &references))
        .collect::<anyhow::Result<Vec<_>>>()?;
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
        let resource_sampler = config
            .resource_sampling
            .as_ref()
            .map(|config| resources::EndpointSampler::new(config, endpoint_index));
        let mut records = Vec::new();
        let mut arrivals = Vec::new();
        let mut protocol_requests = Vec::new();
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
                let resource_window = if let Some(sampler) = &resource_sampler {
                    if control.reason().is_some() {
                        None
                    } else {
                        Some(resources::RunningResourceWindow::begin(Arc::clone(sampler)).await)
                    }
                } else {
                    None
                };
                let (mut run, mut requests, mut artifacts, mut scheduled, mut protocol_run) =
                    run_round(
                        &client,
                        endpoint_index,
                        endpoint,
                        &config,
                        &dataset,
                        phase,
                        repetition,
                        capture.as_ref(),
                        &references,
                        &protocols[endpoint_index],
                        &control,
                    )
                    .await;
                if let Some(window) = resource_window {
                    run.resources = Some(window.finish().await);
                }
                if phase == ImageRequestPhase::Measured {
                    measured_seconds += run.metrics.elapsed_seconds;
                }
                repetitions.push(run);
                records.append(&mut requests);
                arrivals.append(&mut scheduled);
                protocol_requests.append(&mut protocol_run);
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
        let measured_workloads = ferrum_bench_core::images::image_workload_metrics(
            measured_seconds,
            &measured,
            &dataset,
            config.latency_slo_p99_ms,
        )?;
        endpoint_reports.push(ImageEndpointReport {
            endpoint_index,
            protocol: protocols[endpoint_index].identity(),
            protocol_requests,
            repetitions,
            measured: metrics,
            measured_workloads,
            requests: records,
            arrivals,
            captures,
        });
    }
    let capture_summary = capture.as_ref().map(|sink| sink.summary(&endpoint_reports));
    let mut evidence = ImageEvidenceScope::default();
    if config
        .endpoints
        .iter()
        .any(|endpoint| endpoint.protocol.is_comfy())
    {
        evidence.latency_scope = "request start through protocol-specific complete image body reception, excludes local PNG validation/capture and bounded cancellation cleanup; Comfy includes each edit upload, acknowledgement, queue wait and polling granularity; see endpoint protocol identity";
    }
    if config.resource_sampling.is_some() {
        evidence.resource_peak_scope = Some("sampled lower bounds across measured windows only, excludes warmup; aggregate peaks withheld if any measured sampling window is incomplete; per-window observed values and errors remain available; successful sampling never proves an instantaneous true peak");
        let windows = endpoint_reports
            .iter()
            .flat_map(|endpoint| &endpoint.repetitions)
            .filter(|run| run.phase == ImageRequestPhase::Measured)
            .filter_map(|run| run.resources.as_ref())
            .collect::<Vec<_>>();
        let complete = control.reason().is_none()
            && !windows.is_empty()
            && windows.iter().all(|window| window.complete);
        evidence.resource_sampling_complete = Some(complete);
        if complete {
            evidence.peak_host_memory_bytes = windows
                .iter()
                .filter_map(|window| {
                    window
                        .host
                        .as_ref()
                        .and_then(|sample| sample.observed_peak_bytes)
                })
                .max();
            evidence.peak_device_memory_bytes = windows
                .iter()
                .filter_map(|window| {
                    window
                        .device
                        .as_ref()
                        .and_then(|sample| sample.observed_peak_bytes)
                })
                .max();
        }
    }
    if capture_summary.is_some() {
        evidence.throughput_scope = "successful validated images divided by measured wall time including errors, local validation and PNG capture IO; warmup excluded; capture retains request slots and changes subsequent dispatch pacing";
    }
    if config.arrival.is_open_loop() {
        evidence.dispatch_order = "absolute planned arrivals in dataset order; client in-flight saturation produces explicit unsent failures instead of deferred requests; actual dispatch lag is recorded; network/server arrival order is not guaranteed";
    }
    let reference_image = (dataset.schema_version < 3 && !dataset.mode.is_generate())
        .then(|| references.entries[0].identity.clone());
    Ok(ImageBenchmarkReport {
        complete: control.reason().is_none(),
        stop_reason: control.reason(),
        schema_version: if dataset.schema_version == 3 {
            3
        } else if dataset.mode.is_generate() {
            1
        } else {
            2
        },
        client_version: env!("CARGO_PKG_VERSION").into(),
        started_unix_ms,
        dataset_sha256: dataset.fingerprint()?,
        dataset_file_sha256: None,
        configuration: config,
        dataset,
        reference_image,
        reference_images: references
            .entries
            .iter()
            .map(|reference| reference.identity.clone())
            .collect(),
        sample_reference_indices: references.sample_indices,
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
    references: &PreparedReferences,
    protocol: &PreparedProtocol,
    control: &RunControl,
) -> (
    ImageBenchmarkRepetition,
    Vec<ImageRequestRecord>,
    Vec<PngCaptureRecord>,
    Vec<ImageArrivalRecord>,
    Vec<ComfyRequestEvidence>,
) {
    let start = Instant::now();
    let mut next = 0usize;
    let mut active = FuturesUnordered::new();
    let mut records = Vec::with_capacity(dataset.samples.len());
    let mut captures = Vec::new();
    let mut arrivals = Vec::with_capacity(dataset.samples.len());
    let mut protocol_records = Vec::new();
    let mut http_span = 0.0_f64;
    let bound = config.maximum_in_flight.unwrap_or(config.concurrency);
    while next < dataset.samples.len() || !active.is_empty() {
        let completed = if control.reason().is_some() {
            while next < dataset.samples.len() {
                let sample = &dataset.samples[next];
                let mut record = request_record(sample, next, phase, repetition);
                record.failure = Some(ImageFailureKind::ClientStopped);
                record.error = Some(
                    "planned request not sent: benchmark cancelled or remote state unknown".into(),
                );
                records.push(record);
                arrivals.push(ImageArrivalRecord {
                    sample_index: next,
                    phase,
                    repetition,
                    scheduled_offset_ms: config
                        .arrival
                        .offset(next)
                        .map(|d| d.as_secs_f64() * 1000.0),
                    dispatched_offset_ms: None,
                    dispatch_lag_ms: None,
                });
                if capture.is_some() {
                    captures.push(PngCaptureRecord {
                        sample_index: next,
                        sample_id: sample.id.clone(),
                        phase,
                        repetition,
                        result: PngCaptureResult::NotProduced,
                    });
                }
                next += 1;
            }
            active.next().await
        } else if config.arrival.is_open_loop() && next < dataset.samples.len() {
            let due = start
                + config
                    .arrival
                    .offset(next)
                    .expect("validated arrival clock");
            tokio::select! {
                // First release slots whose complete body/validation/capture is
                // already ready. Never wait for a busy slot past a due arrival.
                biased;
                completed = active.next(), if !active.is_empty() => completed,
                _ = tokio::time::sleep_until(due.into()) => {
                    let elapsed = start.elapsed();
                    while next < dataset.samples.len()
                        && config.arrival.offset(next).unwrap() <= elapsed
                    {
                        let offset = config.arrival.offset(next).unwrap();
                        let sample = &dataset.samples[next];
                        if active.len() < bound {
                            active.push(run_request(client, endpoint_index, endpoint, config,
                                sample, next, phase, repetition, start, Some(offset),
                                capture.cloned(), references.for_sample(next), protocol, control));
                        } else {
                            let mut record = request_record(sample, next, phase, repetition);
                            record.failure = Some(ImageFailureKind::ClientCapacity);
                            record.error = Some("planned arrival not sent: client in-flight limit reached".into());
                            records.push(record);
                            arrivals.push(ImageArrivalRecord {
                                sample_index: next, phase, repetition,
                                scheduled_offset_ms: Some(offset.as_secs_f64() * 1000.0),
                                dispatched_offset_ms: None,
                                dispatch_lag_ms: None,
                            });
                            if capture.is_some() {
                                captures.push(PngCaptureRecord {
                                    sample_index: next, sample_id: sample.id.clone(), phase, repetition,
                                    result: PngCaptureResult::NotProduced,
                                });
                            }
                        }
                        next += 1;
                    }
                    None
                }
            }
        } else {
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
                    None,
                    capture.cloned(),
                    references.for_sample(next),
                    protocol,
                    control,
                ));
                next += 1;
            }
            active.next().await
        };
        if let Some((record, body_complete_seconds, captured, arrival, protocol_record)) = completed
        {
            http_span = http_span.max(body_complete_seconds);
            records.push(record);
            arrivals.push(arrival);
            if let Some(protocol_record) = protocol_record {
                protocol_records.push(protocol_record);
            }
            if let Some(captured) = captured {
                captures.push(captured);
            }
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    records.sort_by_key(|record| record.sample_index);
    captures.sort_by_key(|record| record.sample_index);
    arrivals.sort_by_key(|record| record.sample_index);
    protocol_records.sort_by_key(|record| record.sample_index);
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
            resources: None,
        },
        records,
        captures,
        arrivals,
        protocol_records,
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
    scheduled_offset: Option<Duration>,
    capture: Option<Arc<CaptureSink>>,
    reference: Option<&PreparedReference>,
    protocol: &PreparedProtocol,
    control: &RunControl,
) -> (
    ImageRequestRecord,
    f64,
    Option<PngCaptureRecord>,
    ImageArrivalRecord,
    Option<ComfyRequestEvidence>,
) {
    let capture_enabled = capture.is_some();
    let mut capture_result = PngCaptureResult::NotProduced;
    let start = Instant::now();
    if control.reason().is_some() {
        let mut record = request_record(sample, sample_index, phase, repetition);
        record.failure = Some(ImageFailureKind::ClientStopped);
        record.error = Some("request not sent: benchmark stopped before dispatch".into());
        return (
            record,
            0.0,
            capture_enabled.then(|| PngCaptureRecord {
                sample_index,
                sample_id: sample.id.clone(),
                phase,
                repetition,
                result: PngCaptureResult::NotProduced,
            }),
            ImageArrivalRecord {
                sample_index,
                phase,
                repetition,
                scheduled_offset_ms: scheduled_offset.map(|d| d.as_secs_f64() * 1000.0),
                dispatched_offset_ms: None,
                dispatch_lag_ms: None,
            },
            None,
        );
    }
    let dispatched = start.duration_since(round_start);
    let arrival = ImageArrivalRecord {
        sample_index,
        phase,
        repetition,
        scheduled_offset_ms: scheduled_offset.map(|offset| offset.as_secs_f64() * 1000.0),
        dispatched_offset_ms: Some(dispatched.as_secs_f64() * 1000.0),
        dispatch_lag_ms: scheduled_offset
            .map(|offset| dispatched.saturating_sub(offset).as_secs_f64() * 1000.0),
    };
    let mut record = request_record(sample, sample_index, phase, repetition);
    // Mint before polling any protocol operation. Timeout, cancellation and
    // missing HTTP responses retain this ID; repetitions never reuse it.
    record.client_attempt_id = Some(uuid::Uuid::new_v4());
    let (received, mut protocol_record) = if matches!(protocol, PreparedProtocol::Comfy(_)) {
        let (received, evidence) = protocol
            .receive_comfy(client, sample, config, reference, &mut record, control)
            .await;
        (received, Some(evidence))
    } else {
        let received = tokio::select! {
            biased;
            _ = control.stopped() => Err((ImageFailureKind::ClientTask, "benchmark cancelled")),
            received = receive(
        client,
        endpoint,
        sample,
        config.limits.maximum_response_bytes,
        &mut record,
        reference,
            ) => received.map(ReceivedPayload::OpenAiJson),
        };
        (received, None)
    };
    record.e2e_ms = (start.elapsed().as_secs_f64() * 1000.0
        - protocol_record
            .as_ref()
            .map_or(0.0, |record| record.cleanup_ms + record.audit_tail_ms))
    .max(0.0);
    let body_complete_seconds = dispatched.as_secs_f64() + record.e2e_ms / 1000.0;
    if let Some(protocol_record) = &mut protocol_record {
        protocol_record.sample_index = sample_index;
        protocol_record.phase = phase;
        protocol_record.repetition = repetition;
    }
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
                let validated = body.validate(width, height, &limits);
                let validation_ms = validation_start.elapsed().as_secs_f64() * 1000.0;
                // No need to retain JSON/base64 while the original PNG is saved.
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
    (
        record,
        body_complete_seconds,
        captured,
        arrival,
        protocol_record,
    )
}

fn request_record(
    sample: &ImageSample,
    sample_index: usize,
    phase: ImageRequestPhase,
    repetition: usize,
) -> ImageRequestRecord {
    ImageRequestRecord {
        sample_index,
        sample_id: sample.id.clone(),
        phase,
        repetition,
        dispatch_index: sample_index,
        client_attempt_id: None,
        server_request_id: None,
        trace_session_uuid: None,
        http_status: None,
        e2e_ms: 0.0,
        validation_ms: 0.0,
        response_bytes: 0,
        image: None,
        failure: None,
        error: None,
    }
}

async fn receive(
    client: &reqwest::Client,
    endpoint: &ImageBenchmarkEndpoint,
    sample: &ImageSample,
    maximum_bytes: usize,
    record: &mut ImageRequestRecord,
    reference: Option<&PreparedReference>,
) -> Result<Vec<u8>, (ImageFailureKind, &'static str)> {
    let url = if reference.is_some() {
        endpoint.edit_url.as_deref().unwrap_or(&endpoint.url)
    } else {
        &endpoint.url
    };
    let request = client.post(url);
    let request = match record.client_attempt_id {
        Some(attempt) => request.header(
            ferrum_types::image_trace::IMAGE_ATTEMPT_ID_HEADER,
            attempt.to_string(),
        ),
        None => request,
    };
    let request = match reference {
        Some(reference) => request.multipart(reference.form(endpoint, sample).map_err(|_| {
            (
                ImageFailureKind::ClientTask,
                "cannot build PNG multipart request",
            )
        })?),
        None => request.json(&ImageHttpRequest::new(&endpoint.model, sample)),
    };
    let mut response = request.send().await.map_err(transport_error)?;
    record.http_status = Some(response.status().as_u16());
    read_correlation_headers(response.headers(), record)?;
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

fn read_correlation_headers(
    headers: &reqwest::header::HeaderMap,
    record: &mut ImageRequestRecord,
) -> Result<(), (ImageFailureKind, &'static str)> {
    use ferrum_types::image_trace::{
        parse_image_attempt_id, IMAGE_ATTEMPT_ID_HEADER, IMAGE_REQUEST_ID_HEADER,
        IMAGE_TRACE_SESSION_HEADER,
    };
    let parse = |name: &'static str| {
        let all = headers.get_all(name);
        let mut values = all.iter();
        let value = values.next();
        if values.next().is_some() {
            return Err("duplicate image response correlation header");
        }
        let value = value
            .map(|value| {
                value
                    .to_str()
                    .map_err(|_| "invalid image response correlation header")
            })
            .transpose()?;
        parse_image_attempt_id(value).map_err(|_| "invalid image response correlation UUID")
    };
    // Read each independently so valid response IDs survive a later malformed
    // header or a non-success response. No body/prompt/image data is retained.
    let request = parse(IMAGE_REQUEST_ID_HEADER);
    let trace = parse(IMAGE_TRACE_SESSION_HEADER);
    let echo = parse(IMAGE_ATTEMPT_ID_HEADER);
    record.server_request_id = request.as_ref().ok().copied().flatten();
    record.trace_session_uuid = trace.as_ref().ok().copied().flatten();
    let invalid = |message| (ImageFailureKind::InvalidResponse, message);
    request.map_err(invalid)?;
    trace.map_err(invalid)?;
    if let Some(echo) = echo.map_err(invalid)? {
        if Some(echo) != record.client_attempt_id {
            return Err(invalid(
                "image response attempt UUID does not match dispatch",
            ));
        }
    }
    Ok(())
}
