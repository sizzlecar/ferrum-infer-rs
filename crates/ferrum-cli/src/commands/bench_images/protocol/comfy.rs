//! Pinned Comfy prompt/history/view contract. Cancellation never uses the
//! process-global interrupt, queue clear, history clear or model-free APIs.

use super::*;
use ferrum_bench_core::images::ImageSample;
use serde_json::{json, Value};
use std::time::Instant;

struct Job {
    id: String,
    may_be_submitted: bool,
    acknowledged: bool,
    terminal: bool,
    last_status: Option<u16>,
}

pub(super) async fn receive(
    prepared: &PreparedComfy,
    client: &reqwest::Client,
    sample: &ImageSample,
    config: &ImageBenchmarkConfig,
    reference: Option<&super::super::edit::PreparedReference>,
    record: &mut ImageRequestRecord,
    control: &RunControl,
) -> (
    Result<ReceivedPayload, RequestFailure>,
    ComfyRequestEvidence,
) {
    let protocol_start = Instant::now();
    let id = uuid::Uuid::new_v4().to_string();
    let mut job = Job {
        id: id.clone(),
        may_be_submitted: false,
        acknowledged: false,
        terminal: false,
        last_status: None,
    };
    let mut evidence = ComfyRequestEvidence {
        sample_index: 0,
        phase: ImageRequestPhase::Measured,
        repetition: 0,
        prompt_id: id,
        submitted_workflow_sha256: None,
        history_execution_graph_sha256: None,
        submission_acknowledged: false,
        upload_ms: 0.0,
        submission_ms: 0.0,
        history_ms: 0.0,
        image_download_ms: 0.0,
        polls: 0,
        poll_interval_ms: prepared.limits.poll_interval_ms,
        protocol_elapsed_ms: 0.0,
        body_complete_elapsed_ms: None,
        audit_tail_ms: 0.0,
        execution_audit: None,
        cleanup_ms: 0.0,
        metadata_bytes: 0,
        downloaded_png_bytes: 0,
        uploaded_png_bytes: 0,
        cleanup: ComfyCleanupOutcome::NotSubmitted,
    };
    let mut audit = prepared
        .execution_audit
        .as_ref()
        .map(|_| ComfyExecutionAuditEvidence::default());
    let result = {
        let exchange = exchange_with_audit(
            prepared,
            client,
            sample,
            reference,
            config
                .limits
                .maximum_png_bytes
                .min(config.limits.maximum_response_bytes),
            &mut job,
            &mut evidence,
            &mut audit,
            protocol_start,
        );
        tokio::pin!(exchange);
        tokio::select! {
            biased;
            _ = control.stopped() => Err((ImageFailureKind::ClientTask, "benchmark cancelled or stopped after unknown remote state")),
            result = tokio::time::timeout(Duration::from_secs(config.timeout_seconds), &mut exchange) => {
                result.unwrap_or(Err((ImageFailureKind::Timeout, "Comfy request exceeded its complete protocol deadline")))
            }
        }
    };
    evidence.protocol_elapsed_ms = protocol_start.elapsed().as_secs_f64() * 1000.0;
    evidence.audit_tail_ms = evidence
        .body_complete_elapsed_ms
        .map_or(0.0, |body| (evidence.protocol_elapsed_ms - body).max(0.0));
    if let Some(audit) = &mut audit {
        audit.history_graph_verified = evidence.history_execution_graph_sha256.is_some();
        audit.complete =
            result.is_ok() && audit.history_graph_verified && audit.execution_success_observed;
    }
    evidence.execution_audit = audit;
    record.http_status = job.last_status;
    evidence.submission_acknowledged = job.acknowledged;
    if job.terminal {
        evidence.cleanup = ComfyCleanupOutcome::Terminal;
    } else if job.may_be_submitted {
        let cleanup_start = Instant::now();
        evidence.cleanup = match tokio::time::timeout(
            Duration::from_millis(prepared.limits.cleanup_timeout_ms),
            cancel_owned(prepared, client, &job),
        )
        .await
        {
            Ok(Ok(())) => ComfyCleanupOutcome::CancelledQuiescent,
            _ => {
                control.stop(BenchmarkStopReason::RemoteStateUnknown);
                ComfyCleanupOutcome::Unknown
            }
        };
        evidence.cleanup_ms = cleanup_start.elapsed().as_secs_f64() * 1000.0;
    }
    record.response_bytes = evidence.downloaded_png_bytes;
    (result, evidence)
}

async fn exchange_with_audit(
    prepared: &PreparedComfy,
    client: &reqwest::Client,
    sample: &ImageSample,
    reference: Option<&super::super::edit::PreparedReference>,
    maximum_png_bytes: usize,
    job: &mut Job,
    evidence: &mut ComfyRequestEvidence,
    audit: &mut Option<ComfyExecutionAuditEvidence>,
    protocol_start: Instant,
) -> Result<ReceivedPayload, RequestFailure> {
    if let (Some(config), Some(audit)) = (&prepared.execution_audit, audit) {
        let workflow = if reference.is_some() {
            prepared.edit.as_ref()
        } else {
            prepared.generation.as_ref()
        }
        .ok_or(invalid("prepared audited Comfy workflow is absent"))?;
        audit.required_nodes = workflow.required_execution_nodes().to_vec();
        let id = job.id.clone();
        // No POST occurs before the original connection is established. Both
        // futures are borrowed and dropped together; no detached audit task.
        let socket = execution_audit::connect(&prepared.origin, &id, config, audit).await?;
        let (payload, ()) = tokio::try_join!(
            exchange(
                prepared,
                client,
                sample,
                reference,
                maximum_png_bytes,
                job,
                evidence,
                protocol_start
            ),
            execution_audit::observe(socket, &id, config, workflow, audit, protocol_start),
        )?;
        Ok(payload)
    } else {
        exchange(
            prepared,
            client,
            sample,
            reference,
            maximum_png_bytes,
            job,
            evidence,
            protocol_start,
        )
        .await
    }
}

async fn exchange(
    prepared: &PreparedComfy,
    client: &reqwest::Client,
    sample: &ImageSample,
    reference: Option<&super::super::edit::PreparedReference>,
    maximum_png_bytes: usize,
    job: &mut Job,
    evidence: &mut ComfyRequestEvidence,
    protocol_start: Instant,
) -> Result<ReceivedPayload, RequestFailure> {
    let workflow = if reference.is_some() {
        prepared.edit.as_ref()
    } else {
        prepared.generation.as_ref()
    }
    .ok_or((
        ImageFailureKind::ClientTask,
        "prepared Comfy workflow is absent",
    ))?;
    let uploaded = if let Some(reference) = reference {
        let start = Instant::now();
        let bytes = reference.encoded_png();
        evidence.uploaded_png_bytes = bytes.len();
        let part = reqwest::multipart::Part::stream_with_length(
            reqwest::Body::from(bytes),
            reference.identity.png_bytes as u64,
        )
        .file_name("reference.png")
        .mime_str("image/png")
        .map_err(|_| {
            (
                ImageFailureKind::ClientTask,
                "cannot build Comfy PNG upload",
            )
        })?;
        let folder = format!("ferrum-bench/{}", job.id);
        let form = reqwest::multipart::Form::new()
            .text("type", "input")
            .text("overwrite", "false")
            .text("subfolder", folder.clone())
            .part("image", part);
        let response = client
            .post(url(prepared, "upload/image")?)
            .multipart(form)
            .send()
            .await
            .map_err(transport)?;
        let upload = json_body(
            response,
            &prepared.limits,
            &mut evidence.metadata_bytes,
            &mut job.last_status,
        )
        .await?;
        evidence.upload_ms = start.elapsed().as_secs_f64() * 1000.0;
        let name = upload
            .get("name")
            .and_then(Value::as_str)
            .ok_or(invalid("invalid Comfy upload name"))?;
        let subfolder = upload
            .get("subfolder")
            .and_then(Value::as_str)
            .ok_or(invalid("invalid Comfy upload subfolder"))?;
        if upload.get("type").and_then(Value::as_str) != Some("input")
            || subfolder != folder
            || !filename(name)
        {
            return Err(invalid(
                "Comfy upload identity differs from the owned namespace",
            ));
        }
        Some(format!("{subfolder}/{name}"))
    } else {
        None
    };
    let graph = workflow.bind(sample, uploaded.as_deref(), &job.id)?;
    let expected_execution_graph = prepared
        .execution_audit
        .as_ref()
        .map(|_| execution_audit::execution_graph_sha(&graph))
        .transpose()?;
    let canonical = serde_json::to_vec(&graph).map_err(|_| {
        (
            ImageFailureKind::ClientTask,
            "cannot serialize prepared Comfy graph",
        )
    })?;
    evidence.submitted_workflow_sha256 = Some(sha256(&canonical));
    drop(canonical);
    let start = Instant::now();
    // Register the UUID before sending: a lost acknowledgement is never retried.
    job.may_be_submitted = true;
    let response = client
        .post(url(prepared, "prompt")?)
        .json(&json!({"prompt_id":job.id,"client_id":job.id,"prompt":graph}))
        .send()
        .await
        .map_err(transport)?;
    drop(graph);
    let status = response.status();
    job.last_status = Some(status.as_u16());
    if status == reqwest::StatusCode::BAD_REQUEST {
        // The pinned server rejects validation before enqueue. A proxy/body
        // mismatch is not accepted as this proof.
        let body = read_body(response, prepared.limits.maximum_metadata_response_bytes).await?;
        account_metadata(&prepared.limits, &mut evidence.metadata_bytes, body.len())?;
        let rejected: Value =
            serde_json::from_slice(&body).map_err(|_| invalid("invalid Comfy rejection JSON"))?;
        if rejected.get("error").is_some_and(Value::is_object)
            && rejected.get("node_errors").is_some_and(Value::is_object)
            && rejected.get("prompt_id").is_none()
        {
            job.terminal = true;
        }
        return Err((
            ImageFailureKind::HttpStatus,
            "Comfy rejected the submitted workflow",
        ));
    }
    let ack = json_body(
        response,
        &prepared.limits,
        &mut evidence.metadata_bytes,
        &mut job.last_status,
    )
    .await?;
    evidence.submission_ms = start.elapsed().as_secs_f64() * 1000.0;
    if ack.get("prompt_id").and_then(Value::as_str) != Some(job.id.as_str()) {
        return Err(invalid(
            "Comfy acknowledgement does not match the submitted UUID",
        ));
    }
    job.acknowledged = true;
    if !ack
        .get("node_errors")
        .and_then(Value::as_object)
        .is_some_and(|errors| errors.is_empty())
    {
        return Err(invalid("Comfy acknowledged a partially invalid workflow"));
    }
    let history_start = Instant::now();
    let image = loop {
        if evidence.polls == prepared.limits.maximum_polls {
            return Err((
                ImageFailureKind::ResponseLimit,
                "Comfy history poll capacity exhausted",
            ));
        }
        evidence.polls += 1;
        let response = client
            .get(url(prepared, &format!("history/{}", job.id))?)
            .send()
            .await
            .map_err(transport)?;
        let body = json_body(
            response,
            &prepared.limits,
            &mut evidence.metadata_bytes,
            &mut job.last_status,
        )
        .await?;
        let Some(entry) = history_entry(&body, &job.id)? else {
            tokio::time::sleep(Duration::from_millis(prepared.limits.poll_interval_ms)).await;
            continue;
        };
        let prompt = entry
            .get("prompt")
            .and_then(Value::as_array)
            .ok_or(invalid("Comfy history omitted prompt identity"))?;
        if prompt.get(1).and_then(Value::as_str) != Some(job.id.as_str())
            || prompt
                .get(3)
                .and_then(|data| data.get("client_id"))
                .and_then(Value::as_str)
                != Some(job.id.as_str())
        {
            return Err(invalid(
                "Comfy history prompt identity differs from the request",
            ));
        }
        let status = entry
            .get("status")
            .ok_or(invalid("Comfy history omitted terminal status"))?;
        match status.get("status_str").and_then(Value::as_str) {
            Some("error") => {
                job.terminal = true;
                return Err(invalid("Comfy execution failed"));
            }
            Some("success") if status.get("completed").and_then(Value::as_bool) == Some(true) => {
                job.terminal = true;
            }
            _ => return Err(invalid("Comfy history has no complete success status")),
        }
        if let Some(expected) = &expected_execution_graph {
            let actual = execution_audit::execution_graph_sha(
                prompt
                    .get(2)
                    .ok_or(invalid("Comfy history omitted original execution graph"))?,
            )?;
            if &actual != expected {
                return Err(invalid(
                    "Comfy history execution graph differs from the submitted graph",
                ));
            }
            evidence.history_execution_graph_sha256 = Some(actual);
        }
        let images = entry
            .get("outputs")
            .and_then(|outputs| outputs.get(workflow.output_node()))
            .and_then(|output| output.get("images"))
            .and_then(Value::as_array)
            .ok_or(invalid("Comfy declared output node has no image list"))?;
        if images.len() != 1 {
            return Err(invalid(
                "Comfy declared output must contain exactly one image",
            ));
        }
        break images[0].clone();
    };
    evidence.history_ms = history_start.elapsed().as_secs_f64() * 1000.0;
    let name = image
        .get("filename")
        .and_then(Value::as_str)
        .ok_or(invalid("Comfy output has no filename"))?;
    let folder = image
        .get("subfolder")
        .and_then(Value::as_str)
        .ok_or(invalid("Comfy output has no subfolder"))?;
    if !filename(name)
        || !relative_folder(folder)
        || image.get("type").and_then(Value::as_str) != Some("output")
    {
        return Err(invalid(
            "Comfy output is not a relative output PNG descriptor",
        ));
    }
    let start = Instant::now();
    let mut view = url(prepared, "view")?;
    view.query_pairs_mut()
        .append_pair("filename", name)
        .append_pair("subfolder", folder)
        .append_pair("type", "output");
    let response = client.get(view).send().await.map_err(transport)?;
    job.last_status = Some(response.status().as_u16());
    if !response.status().is_success() {
        return Err((
            ImageFailureKind::HttpStatus,
            "Comfy PNG download returned a non-success status",
        ));
    }
    let bytes = read_body_counted(
        response,
        maximum_png_bytes,
        &mut evidence.downloaded_png_bytes,
    )
    .await?;
    evidence.image_download_ms = start.elapsed().as_secs_f64() * 1000.0;
    evidence.body_complete_elapsed_ms = Some(protocol_start.elapsed().as_secs_f64() * 1000.0);
    Ok(ReceivedPayload::Png(bytes))
}

async fn cancel_owned(
    prepared: &PreparedComfy,
    client: &reqwest::Client,
    job: &Job,
) -> Result<(), RequestFailure> {
    let mut used = 0;
    let mut status = None;
    let response = client
        .post(url(prepared, &format!("api/jobs/{}/cancel", job.id))?)
        .json(&json!({}))
        .send()
        .await
        .map_err(transport)?;
    let body = json_body(response, &prepared.limits, &mut used, &mut status).await?;
    let cancelled = body
        .get("cancelled")
        .and_then(Value::as_bool)
        .ok_or(invalid("invalid targeted Comfy cancellation response"))?;
    loop {
        let response = client
            .get(url(prepared, &format!("history/{}", job.id))?)
            .send()
            .await
            .map_err(transport)?;
        let body = json_body(response, &prepared.limits, &mut used, &mut status).await?;
        if let Some(entry) = history_entry(&body, &job.id)? {
            let status = entry.get("status");
            if status
                .and_then(|s| s.get("status_str"))
                .and_then(Value::as_str)
                == Some("error")
                || (status
                    .and_then(|s| s.get("status_str"))
                    .and_then(Value::as_str)
                    == Some("success")
                    && status
                        .and_then(|s| s.get("completed"))
                        .and_then(Value::as_bool)
                        == Some(true))
            {
                return Ok(());
            }
        }
        // A successful scoped cancel plus absence proves pending dequeue (or
        // retirement after an interrupt). False before an unacknowledged POST
        // does not: the original validation may still enqueue this UUID later.
        let response = client
            .get(url(prepared, &format!("api/jobs/{}", job.id))?)
            .send()
            .await
            .map_err(transport)?;
        if response.status() == reqwest::StatusCode::NOT_FOUND && (cancelled || job.acknowledged) {
            return Ok(());
        }
        if !cancelled && !job.acknowledged {
            return Err(invalid("Comfy submission retirement is unknown"));
        }
        tokio::time::sleep(Duration::from_millis(prepared.limits.poll_interval_ms)).await;
    }
}

fn history_entry<'a>(body: &'a Value, id: &str) -> Result<Option<&'a Value>, RequestFailure> {
    let entries = body
        .as_object()
        .ok_or(invalid("invalid Comfy history JSON"))?;
    if entries.is_empty() {
        return Ok(None);
    }
    if entries.len() != 1 || !entries.contains_key(id) {
        return Err(invalid("Comfy history contains a foreign UUID"));
    }
    Ok(entries.get(id))
}

fn url(prepared: &PreparedComfy, path: &str) -> Result<reqwest::Url, RequestFailure> {
    prepared.origin.join(path).map_err(|_| {
        (
            ImageFailureKind::ClientTask,
            "cannot construct Comfy protocol URL",
        )
    })
}

fn filename(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 1024
        && value != "."
        && value != ".."
        && !value
            .chars()
            .any(|c| matches!(c, '/' | '\\' | ':' | '[' | ']') || c.is_control())
}
fn relative_folder(value: &str) -> bool {
    value.len() <= 4096 && (value.is_empty() || value.split('/').all(filename))
}
fn invalid(message: &'static str) -> RequestFailure {
    (ImageFailureKind::InvalidResponse, message)
}
fn transport(error: reqwest::Error) -> RequestFailure {
    if error.is_timeout() {
        (ImageFailureKind::Timeout, "Comfy HTTP operation timed out")
    } else {
        (ImageFailureKind::Transport, "Comfy HTTP transport failed")
    }
}

fn account_metadata(
    limits: &ComfyProtocolLimits,
    used: &mut usize,
    count: usize,
) -> Result<(), RequestFailure> {
    *used = used
        .checked_add(count)
        .filter(|total| *total <= limits.maximum_metadata_bytes_per_request)
        .ok_or((
            ImageFailureKind::ResponseLimit,
            "Comfy aggregate metadata limit exceeded",
        ))?;
    Ok(())
}
async fn json_body(
    response: reqwest::Response,
    limits: &ComfyProtocolLimits,
    used: &mut usize,
    status: &mut Option<u16>,
) -> Result<Value, RequestFailure> {
    *status = Some(response.status().as_u16());
    if !response.status().is_success() {
        return Err((
            ImageFailureKind::HttpStatus,
            "Comfy metadata operation returned a non-success status",
        ));
    }
    let body = read_body(response, limits.maximum_metadata_response_bytes).await?;
    account_metadata(limits, used, body.len())?;
    serde_json::from_slice(&body).map_err(|_| invalid("invalid Comfy metadata JSON"))
}
async fn read_body(response: reqwest::Response, maximum: usize) -> Result<Vec<u8>, RequestFailure> {
    let mut received = 0;
    read_body_counted(response, maximum, &mut received).await
}

async fn read_body_counted(
    mut response: reqwest::Response,
    maximum: usize,
    received: &mut usize,
) -> Result<Vec<u8>, RequestFailure> {
    if response
        .content_length()
        .is_some_and(|length| length > maximum as u64)
    {
        return Err((
            ImageFailureKind::ResponseLimit,
            "Comfy HTTP content length exceeds its byte limit",
        ));
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(transport)? {
        bytes
            .len()
            .checked_add(chunk.len())
            .filter(|length| *length <= maximum)
            .ok_or((
                ImageFailureKind::ResponseLimit,
                "Comfy HTTP body exceeds its byte limit",
            ))?;
        bytes.try_reserve(chunk.len()).map_err(|_| {
            (
                ImageFailureKind::ResponseLimit,
                "cannot reserve Comfy HTTP buffer",
            )
        })?;
        bytes.extend_from_slice(&chunk);
        *received = bytes.len();
    }
    Ok(bytes)
}
