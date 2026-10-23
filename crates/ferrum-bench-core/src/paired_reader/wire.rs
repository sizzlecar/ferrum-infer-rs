use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
pub(super) struct Config {
    pub model: String,
    pub protocol: String,
    pub deadline_ms: u64,
    pub temperature: f64,
    pub request_policy: Value,
    pub provenance: Value,
    pub requests: Vec<Input>,
}
#[derive(Deserialize)]
pub(super) struct Input {
    pub scenario: String,
    pub prompt: String,
    pub max_tokens: u64,
    pub offset_ms: u64,
    pub source: Value,
}
#[derive(Deserialize)]
pub(super) struct Summary {
    pub schema: String,
    pub epoch_unix_ms: f64,
    pub model: String,
    pub protocol: String,
    pub request_policy: Value,
    pub provenance: Value,
    pub request_deadline_ms: u64,
    pub overall: Value,
}
#[derive(Deserialize)]
pub(super) struct Record {
    pub index: usize,
    pub scenario: String,
    pub source: Value,
    pub requested_max_tokens: u64,
    pub scheduled_ms: u64,
    pub start_ms: f64,
    pub finish_ms: f64,
    pub first_visible_ms: Option<f64>,
    pub last_visible_ms: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub ttft_from_scheduled_ms: Option<f64>,
    pub tpot_last_visible_usage_ms: Option<f64>,
    pub success: bool,
    pub timed_out: bool,
    pub error: Option<String>,
    pub http_status: Option<u16>,
    pub content_type: Option<String>,
    pub usage: Option<Value>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub visible_event_ms: Option<Vec<Value>>,
    pub visible_event_utf8_bytes: Option<Vec<Value>>,
    pub visible_event_transport_chunk: Option<Vec<Value>>,
    pub itl_sse_text_event_ms: Option<Vec<Value>>,
    pub transport_chunks: Option<Value>,
    pub all_sse_events: Option<Value>,
    pub text: Option<Value>,
    pub finish_reason: Option<String>,
    pub done: bool,
    // Text is bounded by the input line and used only for byte consistency;
    // it is never retained in the analysis DTO or semantically graded.
}
