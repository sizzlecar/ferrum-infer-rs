//! Bounded evidence from the request's own Comfy WebSocket. No server cache mutation.

use super::*;
use futures::{SinkExt, StreamExt};
use serde_json::{Map, Value};
use std::{
    collections::BTreeSet,
    time::{Duration, Instant},
};
use tokio_tungstenite::{
    connect_async_with_config,
    tungstenite::{protocol::WebSocketConfig, Message},
    MaybeTlsStream, WebSocketStream,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComfyExecutionAuditConfig {
    #[serde(default)]
    pub limits: ComfyExecutionAuditLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ComfyExecutionAuditLimits {
    /// Applies to both a reassembled message and each individual frame.
    pub maximum_message_bytes: usize,
    /// All received messages, including previews and foreign/status events.
    pub maximum_total_bytes: u64,
    pub maximum_messages: u32,
}

impl Default for ComfyExecutionAuditLimits {
    fn default() -> Self {
        Self {
            maximum_message_bytes: 2 * 1024 * 1024,
            maximum_total_bytes: 32 * 1024 * 1024,
            maximum_messages: 100_000,
        }
    }
}

impl ComfyExecutionAuditConfig {
    pub(super) fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            self.limits.maximum_message_bytes > 0
                && self.limits.maximum_message_bytes < isize::MAX as usize
                && self.limits.maximum_total_bytes >= self.limits.maximum_message_bytes as u64
                && self.limits.maximum_messages > 0,
            "Comfy execution audit byte/message limits must be positive and consistent"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ComfyExecutionAuditEvidence {
    pub required_nodes: Vec<String>,
    pub handshake_ms: f64,
    pub messages: u32,
    pub message_bytes: u64,
    pub execution_start_observed: bool,
    pub execution_cached_observed: bool,
    pub execution_success_observed: bool,
    /// Observed exact required node entry, not display-node or prefix matching.
    pub executed_required_nodes: BTreeSet<String>,
    /// Original-graph cache snapshot; this does not cover later dynamic caches.
    pub cached_nodes: BTreeSet<String>,
    pub event_timing_scope: ComfyEventTimingScope,
    /// Receive order, including repeated/dynamic nodes, bound to this prompt.
    /// Adjacent offsets include transport, scheduling and cache-pressure work.
    /// `executed` is optional UI-output evidence, not a universal node end.
    /// No interval here proves exclusive CPU time or GPU execution duration.
    pub events: Vec<ComfyExecutionEvent>,
    pub history_graph_verified: bool,
    pub complete: bool,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ComfyEventTimingScope {
    #[default]
    ClientReceiveMonotonicProtocolElapsed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ComfyExecutionEventKind {
    ExecutionStart,
    ExecutionCached,
    Executing,
    Executed,
    ExecutionSuccess,
    ExecutionError,
    ExecutionInterrupted,
}

#[derive(Debug, Clone, Serialize)]
pub struct ComfyExecutionEvent {
    pub kind: ComfyExecutionEventKind,
    /// One-based receive ordinal, including preceding ignored/status messages.
    pub message_ordinal: u32,
    /// Same origin as the enclosing request's protocol/body-complete timings.
    pub client_received_elapsed_ns: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_node: Option<String>,
    /// Unmodified optional server wall-clock timestamp; never used as our clock.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_timestamp_ms: Option<u64>,
}

type Socket = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

pub(super) async fn connect(
    origin: &reqwest::Url,
    id: &str,
    config: &ComfyExecutionAuditConfig,
    evidence: &mut ComfyExecutionAuditEvidence,
) -> Result<Socket, RequestFailure> {
    let start = Instant::now();
    let mut endpoint = origin
        .join("ws")
        .map_err(|_| invalid("invalid Comfy WebSocket URL"))?;
    let scheme = if endpoint.scheme() == "https" {
        "wss"
    } else {
        "ws"
    };
    endpoint
        .set_scheme(scheme)
        .map_err(|_| invalid("invalid Comfy WebSocket scheme"))?;
    endpoint.query_pairs_mut().append_pair("clientId", id);
    let options = WebSocketConfig {
        write_buffer_size: 0,
        max_write_buffer_size: 4096,
        max_message_size: Some(config.limits.maximum_message_bytes),
        max_frame_size: Some(config.limits.maximum_message_bytes),
        ..Default::default()
    };
    let result = connect_async_with_config(endpoint.as_str(), Some(options), true).await;
    evidence.handshake_ms = start.elapsed().as_secs_f64() * 1000.0;
    result.map(|(socket, _)| socket).map_err(socket_error)
}

pub(super) async fn observe(
    mut socket: Socket,
    id: &str,
    config: &ComfyExecutionAuditConfig,
    workflow: &PreparedWorkflow,
    evidence: &mut ComfyExecutionAuditEvidence,
    protocol_start: Instant,
) -> Result<(), RequestFailure> {
    loop {
        let message = socket
            .next()
            .await
            .ok_or(invalid(
                "Comfy audit disconnected before execution evidence completed",
            ))?
            .map_err(socket_error)?;
        let received_offset = protocol_start.elapsed();
        evidence.messages = evidence.messages.checked_add(1).ok_or(limit())?;
        evidence.message_bytes = evidence
            .message_bytes
            .checked_add(message.len() as u64)
            .ok_or(limit())?;
        if evidence.messages > config.limits.maximum_messages
            || evidence.message_bytes > config.limits.maximum_total_bytes
        {
            return Err(limit());
        }
        match message {
            Message::Text(text) => {
                let event: Value = serde_json::from_str(&text)
                    .map_err(|_| invalid("Comfy audit event is not JSON"))?;
                if consume(&event, id, workflow, evidence, received_offset)? {
                    return Ok(());
                }
            }
            Message::Ping(_) => socket.flush().await.map_err(socket_error)?,
            Message::Close(_) => {
                return Err(invalid(
                    "Comfy audit closed before execution evidence completed",
                ))
            }
            // Previews are neither evidence of execution nor the final PNG.
            Message::Binary(_) | Message::Pong(_) | Message::Frame(_) => {}
        }
    }
}

pub(super) fn consume(
    event: &Value,
    id: &str,
    workflow: &PreparedWorkflow,
    evidence: &mut ComfyExecutionAuditEvidence,
    received_offset: Duration,
) -> Result<bool, RequestFailure> {
    let kind = event
        .get("type")
        .and_then(Value::as_str)
        .ok_or(invalid("Comfy audit event omitted its type"))?;
    let data = event
        .get("data")
        .and_then(Value::as_object)
        .ok_or(invalid("Comfy audit event omitted its data"))?;
    let prompt = data.get("prompt_id").and_then(Value::as_str);
    if prompt.is_some_and(|prompt| prompt != id) {
        return Ok(false);
    }
    if prompt.is_none() {
        // Initial status and reconnect executing carry no prompt ID. Neither
        // can satisfy this audit. Known execution lifecycle messages must bind.
        if matches!(
            kind,
            "execution_start"
                | "execution_cached"
                | "execution_success"
                | "execution_error"
                | "execution_interrupted"
        ) {
            return Err(invalid(
                "Comfy audit lifecycle event omitted prompt identity",
            ));
        }
        return Ok(false);
    }
    // observe() charges every frame/message before parsing or retaining data.
    // At most one compact event is retained per charged message. Its strings
    // are subsets of that message; UI/preview payloads are never retained.
    let event_kind = match kind {
        "execution_start" => Some(ComfyExecutionEventKind::ExecutionStart),
        "execution_cached" => Some(ComfyExecutionEventKind::ExecutionCached),
        "executing" => Some(ComfyExecutionEventKind::Executing),
        "executed" => Some(ComfyExecutionEventKind::Executed),
        "execution_success" => Some(ComfyExecutionEventKind::ExecutionSuccess),
        "execution_error" => Some(ComfyExecutionEventKind::ExecutionError),
        "execution_interrupted" => Some(ComfyExecutionEventKind::ExecutionInterrupted),
        _ => None,
    };
    if let Some(kind) = event_kind {
        evidence.events.try_reserve(1).map_err(|_| limit())?;
        evidence.events.push(ComfyExecutionEvent {
            kind,
            message_ordinal: evidence.messages,
            client_received_elapsed_ns: u64::try_from(received_offset.as_nanos())
                .unwrap_or(u64::MAX),
            node: data.get("node").and_then(Value::as_str).map(str::to_owned),
            display_node: data
                .get("display_node")
                .and_then(Value::as_str)
                .map(str::to_owned),
            server_timestamp_ms: data.get("timestamp").and_then(Value::as_u64),
        });
    }
    match kind {
        "execution_start" => {
            if evidence.execution_start_observed {
                return Err(invalid("Comfy audit duplicated execution start"));
            }
            evidence.execution_start_observed = true;
        }
        "execution_cached" => {
            if !evidence.execution_start_observed || evidence.execution_cached_observed {
                return Err(invalid(
                    "Comfy audit cache event is missing its unique execution start",
                ));
            }
            let nodes = data
                .get("nodes")
                .and_then(Value::as_array)
                .ok_or(invalid("Comfy audit cache event omitted nodes"))?;
            for node in nodes {
                let node = node
                    .as_str()
                    .ok_or(invalid("Comfy audit cache node is not an ID"))?;
                if !workflow.contains_node(node) || !evidence.cached_nodes.insert(node.to_owned()) {
                    return Err(invalid(
                        "Comfy audit cache snapshot contains an unknown or duplicate node",
                    ));
                }
            }
            evidence.execution_cached_observed = true;
            if workflow
                .required_execution_nodes()
                .iter()
                .any(|node| evidence.cached_nodes.contains(node))
            {
                return Err(invalid("Comfy audit required execution node was cached"));
            }
        }
        "executing" => {
            let Some(node) = data.get("node").and_then(Value::as_str) else {
                if data.get("node") == Some(&Value::Null) {
                    return Ok(false);
                }
                return Err(invalid("Comfy audit executing event omitted node"));
            };
            if !evidence.execution_start_observed || !evidence.execution_cached_observed {
                return Err(invalid(
                    "Comfy audit node execution preceded its lifecycle evidence",
                ));
            }
            if workflow
                .required_execution_nodes()
                .iter()
                .any(|required| required == node)
            {
                if evidence.cached_nodes.contains(node) {
                    return Err(invalid("Comfy audit execution contradicts cache snapshot"));
                }
                evidence.executed_required_nodes.insert(node.to_owned());
            }
        }
        "execution_success" => {
            evidence.execution_success_observed = true;
            if !evidence.execution_start_observed
                || !evidence.execution_cached_observed
                || workflow
                    .required_execution_nodes()
                    .iter()
                    .any(|node| !evidence.executed_required_nodes.contains(node))
            {
                return Err(invalid(
                    "Comfy audit success omitted required execution evidence",
                ));
            }
            return Ok(true);
        }
        "execution_error" | "execution_interrupted" => {
            return Err(invalid("Comfy audited execution failed or was interrupted"))
        }
        _ => {}
    }
    Ok(false)
}

/// Comfy can append `is_changed` and other server metadata. Bind all original
/// node IDs, classes and inputs, without treating metadata as client input.
pub(super) fn execution_graph_sha(graph: &Value) -> Result<String, RequestFailure> {
    let graph = graph
        .as_object()
        .ok_or(invalid("Comfy execution graph is not an object"))?;
    let mut normalized = Map::new();
    for (id, node) in graph {
        let class = node
            .get("class_type")
            .and_then(Value::as_str)
            .ok_or(invalid("Comfy execution graph omitted node class"))?;
        let inputs = node
            .get("inputs")
            .and_then(Value::as_object)
            .ok_or(invalid("Comfy execution graph omitted node inputs"))?;
        normalized.insert(
            id.clone(),
            serde_json::json!({"class_type":class,"inputs":inputs}),
        );
    }
    serde_json::to_vec(&normalized)
        .map(|bytes| sha256(&bytes))
        .map_err(|_| invalid("Comfy execution graph cannot be serialized"))
}

fn invalid(message: &'static str) -> RequestFailure {
    (ImageFailureKind::InvalidResponse, message)
}
fn limit() -> RequestFailure {
    (
        ImageFailureKind::ResponseLimit,
        "Comfy execution audit exceeded its declared message/byte capacity",
    )
}
fn socket_error(error: tokio_tungstenite::tungstenite::Error) -> RequestFailure {
    if matches!(error, tokio_tungstenite::tungstenite::Error::Capacity(_)) {
        limit()
    } else {
        (
            ImageFailureKind::Transport,
            "Comfy execution audit WebSocket transport failed",
        )
    }
}
