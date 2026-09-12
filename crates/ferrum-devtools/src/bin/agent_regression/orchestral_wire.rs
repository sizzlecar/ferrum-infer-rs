//! Bind public Orchestral exchanges to actual OpenAI-compatible SSE and replay.
//! The first slice certifies one fresh, uncompacted Run. Unsupported rewrites
//! remain unproven; public delivery is still separate from task validation.
use super::{config::OrchestralToolResultFormat, orchestral_evidence, proxy::RequestRecord};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

#[derive(Debug, Default, Serialize)]
pub(crate) struct Evidence {
    pub tool_result_format: OrchestralToolResultFormat,
    pub receipts: Vec<Receipt>,
    pub terminal: Option<TerminalReceipt>,
    pub unproven: Vec<String>,
}

impl Evidence {
    pub fn complete(&self) -> bool {
        self.terminal.is_some() && self.unproven.is_empty()
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct Receipt {
    pub session_seq: u64,
    /// Orchestral logical model request identity, distinct from HTTP/SSE IDs.
    pub model_request_id: String,
    pub origin_request_index: u32,
    pub following_request_index: u32,
    pub origin_response_id: String,
    pub canonical_call_ids: Vec<String>,
    pub native_call_ids: Vec<String>,
    pub result_messages_sha256: String,
    pub source_sse: PathBuf,
    pub source_sse_sha256: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct TerminalReceipt {
    pub request_index: u32,
    pub response_id: String,
    pub source_sse: PathBuf,
    pub source_sse_sha256: String,
    pub output_sha256: String,
    pub successful_requests_bound: usize,
}

pub(crate) fn bind(
    public: &orchestral_evidence::Evidence,
    records: &[&RequestRecord],
    raw_dir: &Path,
    tool_result_format: OrchestralToolResultFormat,
) -> Evidence {
    let mut evidence = Evidence {
        tool_result_format,
        ..Evidence::default()
    };
    if let Err(error) = bind_inner(public, records, raw_dir, &mut evidence) {
        evidence.unproven.push(format!("{error:#}"));
    }
    evidence
}

fn bind_inner(
    public: &orchestral_evidence::Evidence,
    records: &[&RequestRecord],
    raw_dir: &Path,
    evidence: &mut Evidence,
) -> Result<()> {
    ensure!(
        public.complete(),
        "public Run/Session lifecycle is incomplete"
    );
    ensure!(
        public.compaction_events == 0,
        "compacted history requires separate causal evidence"
    );
    let input = public.input.as_deref().context("missing public input")?;
    let output = public.output.as_deref().context("missing public output")?;
    let task = &records.first().context("no HTTP requests")?.task_id;
    ensure!(
        !task.is_empty() && !task.contains(['/', '\\']),
        "invalid task artifact identity"
    );
    let mut indexes = BTreeSet::new();
    let mut responses = Vec::new();
    for record in records {
        ensure!(&record.task_id == task, "mixed task HTTP evidence");
        ensure!(
            indexes.insert(record.request_index),
            "duplicate HTTP request index"
        );
        if record.http_status != Some(200) || record.error.is_some() || !record.saw_done {
            continue;
        }
        let path = raw_dir.join(format!("{task}-{}.response.sse", record.request_index));
        let response = read_sse(&path).with_context(|| format!("read {}", path.display()))?;
        ensure!(
            record.ended_ns.is_some(),
            "successful request has no completion timestamp"
        );
        ensure!(
            record
                .finish_reasons
                .iter()
                .any(|reason| reason == &response.finish),
            "raw/proxy finish reasons disagree"
        );
        responses.push(BoundResponse {
            record,
            history: normalize_history(&record.messages)?,
            response,
            path,
        });
    }
    responses.sort_by_key(|response| response.record.request_index);
    let first = responses.first().context("no successful HTTP response")?;
    let (last, leading) = first
        .history
        .split_last()
        .context("empty initial history")?;
    ensure!(
        last == &json!({"role":"user", "content":input}),
        "initial wire user differs from public input"
    );
    ensure!(
        leading.iter().all(|message| message["role"] == "system"),
        "initial history is not a fresh headless Run"
    );
    let mut expected = first.history.clone();
    let mut seen_requests = BTreeSet::new();
    let mut model_ids = BTreeSet::new();
    for exchange in &public.tool_exchanges {
        ensure!(
            model_ids.insert(&exchange.request_id),
            "ambiguous repeated logical model request"
        );
        let origin = unique_history(&responses, &expected)?;
        ensure!(
            seen_requests.insert(origin.record.request_index),
            "HTTP source reused for multiple exchanges"
        );
        ensure!(
            matches!(origin.response.finish.as_str(), "tool_calls" | "stop"),
            "tool source did not finish normally"
        );
        let (assistant, results) =
            project_exchange(exchange, &origin.response, evidence.tool_result_format)?;
        expected.push(assistant);
        expected.extend(results.clone());
        let following = unique_history(&responses, &expected)?;
        ensure!(
            following.record.request_index > origin.record.request_index
                && following.record.submitted_ns >= origin.record.ended_ns.unwrap(),
            "tool replay does not follow completed source response"
        );
        evidence.receipts.push(Receipt {
            session_seq: exchange.session_seq,
            model_request_id: exchange.request_id.clone(),
            origin_request_index: origin.record.request_index,
            following_request_index: following.record.request_index,
            origin_response_id: origin.response.id.clone(),
            canonical_call_ids: exchange
                .calls
                .iter()
                .map(|call| call.call_id.clone())
                .collect(),
            native_call_ids: origin
                .response
                .calls
                .iter()
                .map(|call| call.id.clone())
                .collect(),
            result_messages_sha256: sha(&serde_json::to_vec(&results)?),
            source_sse: origin.path.clone(),
            source_sse_sha256: origin.response.sha256.clone(),
        });
    }
    let terminal = unique_history(&responses, &expected)?;
    ensure!(
        seen_requests.insert(terminal.record.request_index),
        "terminal reuses a tool response"
    );
    ensure!(
        terminal.response.finish == "stop" && terminal.response.calls.is_empty(),
        "terminal HTTP response is not a plain stop"
    );
    ensure!(
        terminal.response.text == output,
        "terminal raw SSE differs from public delivered output"
    );
    ensure!(
        seen_requests.len() == responses.len(),
        "successful HTTP requests remain outside the public causal chain"
    );
    evidence.terminal = Some(TerminalReceipt {
        request_index: terminal.record.request_index,
        response_id: terminal.response.id.clone(),
        source_sse: terminal.path.clone(),
        source_sse_sha256: terminal.response.sha256.clone(),
        output_sha256: sha(output.as_bytes()),
        successful_requests_bound: seen_requests.len(),
    });
    Ok(())
}

struct BoundResponse<'a> {
    record: &'a RequestRecord,
    history: Vec<Value>,
    response: Response,
    path: PathBuf,
}

fn unique_history<'a, 'r>(
    responses: &'a [BoundResponse<'r>],
    expected: &[Value],
) -> Result<&'a BoundResponse<'r>> {
    let mut matches = responses
        .iter()
        .filter(|response| response.history == expected);
    let result = matches
        .next()
        .context("no successful request carries the exact expected history")?;
    ensure!(
        matches.next().is_none(),
        "multiple successful origins carry identical history"
    );
    Ok(result)
}

fn normalize_history(messages: &[Value]) -> Result<Vec<Value>> {
    messages
        .iter()
        .map(|message| {
            let mut message = message.clone();
            if let Some(calls) = message.get_mut("tool_calls") {
                for call in calls
                    .as_array_mut()
                    .context("wire tool_calls is not an array")?
                {
                    let arguments = call["function"]["arguments"]
                        .as_str()
                        .context("wire arguments are not JSON text")?;
                    call["function"]["arguments"] =
                        serde_json::from_str(arguments).context("malformed wire arguments")?;
                }
            }
            Ok(message)
        })
        .collect()
}

fn project_exchange(
    exchange: &orchestral_evidence::ToolExchange,
    response: &Response,
    format: OrchestralToolResultFormat,
) -> Result<(Value, Vec<Value>)> {
    ensure!(
        exchange.assistant["role"] == "assistant" && exchange.tool["role"] == "tool",
        "public exchange roles changed"
    );
    let mut texts = Vec::new();
    let mut reasoning = None;
    for block in exchange.assistant["content"]
        .as_array()
        .context("missing public assistant content")?
    {
        match block["type"].as_str() {
            Some("text") => texts.push(
                block["text"]
                    .as_str()
                    .context("invalid public text")?
                    .to_owned(),
            ),
            Some("json" | "data") => texts.push(
                block
                    .get("value")
                    .context("missing structured public content")?
                    .to_string(),
            ),
            Some("tool_call") => {}
            Some("continuation") => {
                ensure!(
                    block["namespace"] == "openai-compatible/reasoning-content/v1"
                        && reasoning.is_none(),
                    "unknown or duplicate public continuation"
                );
                reasoning = Some(
                    block["value"]
                        .as_str()
                        .context("invalid public reasoning")?,
                );
            }
            _ => anyhow::bail!("unsupported public assistant content"),
        }
    }
    ensure!(
        texts.join("\n") == response.text,
        "raw visible assistant content differs from public exchange"
    );
    ensure!(
        reasoning == response.reasoning.as_deref(),
        "raw/public message continuation differs"
    );
    ensure!(
        !exchange.calls.is_empty() && exchange.calls.len() == response.calls.len(),
        "raw/public tool cardinality differs"
    );
    let mut calls = Vec::new();
    let mut results = BTreeMap::new();
    for (public, raw) in exchange.calls.iter().zip(&response.calls) {
        ensure!(
            public.native_call_id.as_deref() == Some(raw.id.as_str()),
            "raw/public native tool identity differs"
        );
        let canonical = canonical_id(&exchange.request_id, &raw.id)?;
        ensure!(
            public.call_id == canonical,
            "canonical Tool ID is not scoped to the logical model request"
        );
        ensure!(
            public.name == raw.name && public.arguments == raw.arguments,
            "raw/public exact tool name or JSON arguments differ"
        );
        calls.push(json!({"id":raw.id,"type":"function","function":{"name":raw.name,"arguments":raw.arguments}}));
        ensure!(
            results
                .insert(
                    public.call_id.as_str(),
                    json!({
                        "role":"tool", "tool_call_id":raw.id,
                        "content":tool_result_content(format, &public.result, public.is_error)?,
                    })
                )
                .is_none(),
            "duplicate canonical tool result"
        );
    }
    // Preserve actual Tool-role order, which need not equal call-start order.
    let mut ordered = Vec::new();
    for block in exchange.tool["content"]
        .as_array()
        .context("missing public tool content")?
    {
        ensure!(
            block["type"] == "tool_result",
            "unsupported public tool content"
        );
        let id = block["call_id"]
            .as_str()
            .context("missing public result identity")?;
        ordered.push(results.remove(id).context("unpaired public tool result")?);
    }
    ensure!(results.is_empty(), "public tool result was not replayed");
    let mut assistant = json!({
        "role":"assistant", "content":if texts.is_empty() {Value::Null} else {Value::String(texts.join("\n"))},
        "tool_calls":calls,
    });
    if let Some(reasoning) = reasoning {
        assistant["reasoning_content"] = reasoning.into();
    }
    Ok((assistant, ordered))
}

fn tool_result_content(
    format: OrchestralToolResultFormat,
    result: &Value,
    is_error: bool,
) -> Result<Value> {
    let envelope = json!({"result":result,"is_error":is_error});
    match format {
        OrchestralToolResultFormat::Json => Ok(envelope.to_string().into()),
        OrchestralToolResultFormat::Yaml => serde_yaml::to_string(&envelope)
            .map(Value::String)
            .context("serialize declared YAML tool result"),
        OrchestralToolResultFormat::TextParts => Ok(tool_text_parts::content(result, is_error)),
    }
}

#[path = "orchestral_wire/tool_text_parts.rs"]
mod tool_text_parts;

fn canonical_id(model_request_id: &str, native_id: &str) -> Result<String> {
    Ok(format!(
        "openai-{}",
        sha(&serde_json::to_vec(&(model_request_id, native_id))?)
    ))
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Debug)]
struct Response {
    id: String,
    text: String,
    reasoning: Option<String>,
    calls: Vec<Call>,
    finish: String,
    sha256: String,
}

#[derive(Debug)]
struct Call {
    id: String,
    name: String,
    arguments: Value,
}

#[derive(Default)]
struct CallFragments {
    id: Option<String>,
    name: String,
    arguments: String,
}

#[derive(Default)]
struct Frames {
    id: Option<String>,
    text: String,
    reasoning: Option<String>,
    calls: BTreeMap<u64, CallFragments>,
    finish: Option<String>,
    done: bool,
}

impl Frames {
    fn accept(&mut self, data: &str) -> Result<()> {
        ensure!(!self.done, "SSE data follows DONE");
        if data.trim() == "[DONE]" {
            self.done = true;
            return Ok(());
        }
        let value: Value = serde_json::from_str(data).context("malformed raw SSE JSON")?;
        ensure!(value.get("error").is_none(), "raw SSE error");
        if let Some(id) = value.get("id").and_then(Value::as_str) {
            ensure!(
                !id.is_empty() && self.id.as_deref().is_none_or(|previous| previous == id),
                "missing or changed SSE response identity"
            );
            self.id = Some(id.to_owned());
        }
        let choices = value["choices"].as_array().context("missing SSE choices")?;
        ensure!(
            choices.len() <= 1,
            "multiple response choices are not certified"
        );
        if let Some(choice) = choices.first() {
            ensure!(choice["index"] == 0, "nonzero response choice");
            let delta = &choice["delta"];
            let text = delta.get("content").and_then(Value::as_str).unwrap_or("");
            let calls = delta.get("tool_calls").filter(|calls| !calls.is_null());
            let mut reasoning = None;
            for field in ["reasoning_content", "reasoning", "reasoning_text"] {
                if let Some(value) = delta.get(field).filter(|value| !value.is_null()) {
                    let fragment = value.as_str().context("non-text SSE reasoning")?;
                    ensure!(
                        reasoning.is_none_or(|previous: &str| previous.is_empty()
                            || fragment.is_empty()
                            || previous == fragment),
                        "conflicting SSE reasoning aliases"
                    );
                    if reasoning.is_none() || !fragment.is_empty() {
                        reasoning = Some(fragment);
                    }
                }
            }
            ensure!(
                self.finish.is_none()
                    || (text.is_empty() && calls.is_none() && reasoning.is_none()),
                "content follows finish reason"
            );
            self.text.push_str(text);
            if let Some(fragment) = reasoning {
                self.reasoning
                    .get_or_insert_with(String::new)
                    .push_str(fragment);
            }
            if let Some(calls) = calls {
                for fragment in calls.as_array().context("SSE tool_calls is not an array")? {
                    let index = fragment["index"]
                        .as_u64()
                        .context("SSE Tool delta has no index")?;
                    let call = self.calls.entry(index).or_default();
                    if let Some(id) = fragment
                        .get("id")
                        .and_then(Value::as_str)
                        .filter(|id| !id.is_empty())
                    {
                        ensure!(
                            call.id.as_deref().is_none_or(|previous| previous == id),
                            "native Tool ID changed while streaming"
                        );
                        call.id = Some(id.to_owned());
                    }
                    if let Some(name) = fragment.pointer("/function/name").and_then(Value::as_str) {
                        call.name.push_str(name);
                    }
                    if let Some(arguments) = fragment
                        .pointer("/function/arguments")
                        .and_then(Value::as_str)
                        .filter(|text| !text.is_empty())
                    {
                        ensure!(
                            call.id.is_some() && !call.name.is_empty(),
                            "Tool arguments preceded identity/name"
                        );
                        call.arguments.push_str(arguments);
                    }
                }
            }
            if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                ensure!(
                    self.finish
                        .as_deref()
                        .is_none_or(|previous| previous == reason),
                    "conflicting SSE finish reasons"
                );
                self.finish = Some(reason.to_owned());
            }
        }
        Ok(())
    }

    fn finish(self, sha256: String) -> Result<Response> {
        ensure!(self.done, "raw SSE has no DONE");
        let mut ids = BTreeSet::new();
        let calls = self
            .calls
            .into_values()
            .map(|call| {
                let id = call.id.context("raw Tool omitted identity")?;
                ensure!(
                    ids.insert(id.clone()) && !call.name.is_empty(),
                    "duplicate or incomplete native Tool"
                );
                Ok(Call {
                    id,
                    name: call.name,
                    arguments: serde_json::from_str(&call.arguments)
                        .context("raw Tool arguments are malformed JSON")?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Response {
            id: self.id.context("raw SSE omitted response identity")?,
            text: self.text,
            reasoning: self.reasoning,
            calls,
            finish: self.finish.context("raw SSE omitted finish reason")?,
            sha256,
        })
    }
}

fn read_sse(path: &Path) -> Result<Response> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut frames = Frames::default();
    let mut data = Vec::new();
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        hasher.update(line.as_bytes());
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            if !data.is_empty() {
                frames.accept(&data.join("\n"))?;
                data.clear();
            }
        } else if let Some(value) = line.strip_prefix("data:") {
            data.push(value.strip_prefix(' ').unwrap_or(value).to_owned());
        }
    }
    if !data.is_empty() {
        frames.accept(&data.join("\n"))?;
    }
    frames.finish(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    mod reasoning_tests {
        include!("orchestral_wire/reasoning_tests.rs");
    }

    struct Fixture {
        dir: tempfile::TempDir,
        public: orchestral_evidence::Evidence,
        records: Vec<RequestRecord>,
    }

    impl Fixture {
        fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            let inline = |text: &str| json!({"body":{"kind":"inline","value":text}});
            let input = "Repair code";
            let output = "Changed and checked.";
            let events = [
                json!({"type":"run_accepted","session_id":"session","spec_digest":"spec"}),
                json!({"type":"run_started"}),
                json!({"type":"output_committed","content":[inline(output)]}),
                json!({"type":"delivery_committed","delivery":{"run_id":"run","spec_digest":"spec","final_response":inline(output)}}),
            ];
            let run = json!({"schema_version":1,"run":{
                "registration":{
                    "request":{"run":{"spec":{"protocol_version":{"major":1,"minor":0},"session_id":"session","run_id":"run","input":[inline(input)]},"spec_digest":"spec"}},
                    "execution":{"session_id":"session","run_id":"run","spec_digest":"spec"}
                },
                "records":events.into_iter().enumerate().map(|(i,payload)|json!({"event":{"event_id":format!("event-{i}"),"run_id":"run","run_seq":i+1,"payload":payload}})).collect::<Vec<_>>()
            }});
            let mut session = vec![json!({"type":"run_input_committed","message":{
                "role":"user","content":[{"type":"text","text":input}],
            }})];
            let mut history = vec![
                json!({"role":"system","content":"Host policy"}),
                json!({"role":"user","content":input}),
            ];
            let mut records = Vec::new();
            for index in 0..2 {
                let request_id = format!("model-cli-run-{}", index + 1);
                let call_id = canonical_id(&request_id, "call_0").unwrap();
                let arguments = json!({"path":"src/lib.rs"});
                let text = format!("Inspecting {index}: λ.\n");
                let result = json!({"text":format!("fn answer() {{\n\t{index}\n}}\n")});
                session.push(json!({"type":"tool_exchange_committed","request_id":request_id,
                    "assistant":{"role":"assistant","content":[
                        {"type":"text","text":text},
                        {"type":"tool_call","call_id":call_id,"name":"file_read","arguments":arguments,"extensions":{"openai/tool_call_id":"call_0"}},
                    ]},
                    "tool":{"role":"tool","content":[{"type":"tool_result","call_id":call_id,"result":result,"is_error":false}]},
                    "usage":{"input_tokens":30,"output_tokens":8},
                }));
                records.push(record(index, history.clone(), "tool_calls"));
                write_tool(dir.path(), index, &text, &arguments);
                history.push(json!({"role":"assistant","content":text,"tool_calls":[{
                    "id":"call_0","type":"function","function":{"name":"file_read","arguments":arguments.to_string()},
                }]}));
                history.push(json!({"role":"tool","tool_call_id":"call_0","content":json!({"result":result,"is_error":false}).to_string()}));
            }
            session.push(
                json!({"type":"run_output_committed","request_id":"model-cli-run-3",
                    "message":{"role":"assistant","content":[{"type":"text","text":output}]},
                    "usage":{"input_tokens":45,"output_tokens":4},
                }),
            );
            let session = session.into_iter().enumerate().map(|(i,payload)|json!({
                "session_seq":i+1,"event_id":format!("session-{i}"),"session_id":"session","run_id":"run","payload":payload,
            })).collect::<Vec<_>>();
            fs::write(
                dir.path().join("run-test.json"),
                serde_json::to_vec(&run).unwrap(),
            )
            .unwrap();
            fs::write(
                dir.path().join("session-test.json"),
                serde_json::to_vec(&session).unwrap(),
            )
            .unwrap();
            records.push(record(2, history, "stop"));
            write_frames(
                dir.path(),
                2,
                &[
                    json!({"id":"http-response-2","choices":[{"index":0,"delta":{"content":output},"finish_reason":"stop"}]}),
                ],
            );
            let public = orchestral_evidence::read(dir.path(), "session");
            assert!(public.complete(), "{:?}", public.errors);
            Self {
                dir,
                public,
                records,
            }
        }

        fn bind(&self) -> Evidence {
            self.bind_as(OrchestralToolResultFormat::Json)
        }

        fn bind_as(&self, format: OrchestralToolResultFormat) -> Evidence {
            bind(
                &self.public,
                &self.records.iter().collect::<Vec<_>>(),
                self.dir.path(),
                format,
            )
        }

        fn yaml() -> Self {
            let mut fixture = Self::new();
            for record in &mut fixture.records {
                for message in &mut record.messages {
                    if message["role"] == "tool" {
                        let envelope: Value =
                            serde_json::from_str(message["content"].as_str().unwrap()).unwrap();
                        message["content"] = serde_yaml::to_string(&envelope).unwrap().into();
                    }
                }
            }
            fixture
        }

        fn text_parts() -> Self {
            let mut fixture = Self::new();
            for record in &mut fixture.records {
                for message in &mut record.messages {
                    if message["role"] == "tool" {
                        let envelope: Value =
                            serde_json::from_str(message["content"].as_str().unwrap()).unwrap();
                        message["content"] = tool_text_parts::content(
                            &envelope["result"],
                            envelope["is_error"].as_bool().unwrap(),
                        );
                    }
                }
            }
            fixture
        }
    }

    fn record(index: u32, messages: Vec<Value>, finish: &str) -> RequestRecord {
        RequestRecord {
            task_id: "coding".into(),
            request_index: index,
            submitted_ns: u64::from(index) * 100,
            ended_ns: Some(u64::from(index) * 100 + 50),
            server_request_id: Some(format!("header-request-{index}")),
            http_status: Some(200),
            saw_done: true,
            finish_reasons: vec![finish.into()],
            messages,
            ..RequestRecord::default()
        }
    }

    fn write_tool(dir: &Path, index: u32, text: &str, arguments: &Value) {
        let arguments = arguments.to_string();
        let (first, second) = arguments.split_at(arguments.len() / 2);
        let frames = [
            json!({"id":format!("http-response-{index}"),"choices":[{"index":0,"delta":{"content":text,"tool_calls":[{"index":0,"id":"call_0","type":"function","function":{"name":"file_read","arguments":first}}]},"finish_reason":null}]}),
            json!({"id":format!("http-response-{index}"),"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":second}}]},"finish_reason":"tool_calls"}]}),
        ];
        write_frames(dir, index, &frames);
    }

    fn write_frames(dir: &Path, index: u32, frames: &[Value]) {
        let mut raw = String::from(": transport comment\r\n\r\n");
        for frame in frames {
            raw.push_str(&format!("data: {frame}\r\n\r\n"));
        }
        raw.push_str("data: [DONE]\r\n\r\n");
        fs::write(dir.join(format!("coding-{index}.response.sse")), raw).unwrap();
    }

    #[test]
    fn raw_sse_binds_reused_native_ids_to_distinct_logical_requests_and_exact_results() {
        let fixture = Fixture::new();
        let evidence = fixture.bind();
        assert!(evidence.complete(), "{:?}", evidence.unproven);
        assert_eq!(evidence.receipts.len(), 2);
        assert_eq!(evidence.receipts[0].origin_response_id, "http-response-0");
        assert_eq!(evidence.receipts[0].model_request_id, "model-cli-run-1");
        assert_eq!(
            evidence.receipts[0].native_call_ids,
            evidence.receipts[1].native_call_ids
        );
        assert_ne!(
            evidence.receipts[0].canonical_call_ids,
            evidence.receipts[1].canonical_call_ids
        );
        assert_eq!(
            evidence.receipts[0].following_request_index,
            evidence.receipts[1].origin_request_index
        );
        assert_eq!(
            evidence
                .terminal
                .as_ref()
                .unwrap()
                .successful_requests_bound,
            3
        );
        let bytes = fs::read(&evidence.receipts[0].source_sse).unwrap();
        assert_eq!(evidence.receipts[0].source_sse_sha256, sha(&bytes));
    }

    #[test]
    fn correct_following_history_cannot_hide_changed_raw_arguments_or_visible_prose() {
        for change_arguments in [true, false] {
            let fixture = Fixture::new();
            let text = if change_arguments {
                "Inspecting 0: λ.\n"
            } else {
                "different visible content"
            };
            let arguments = if change_arguments {
                json!({"path":"different.rs"})
            } else {
                json!({"path":"src/lib.rs"})
            };
            write_tool(fixture.dir.path(), 0, text, &arguments);
            let evidence = fixture.bind();
            assert!(!evidence.complete());
            assert!(
                evidence.unproven[0].contains("raw/public")
                    || evidence.unproven[0].contains("raw visible")
            );
        }
    }

    #[test]
    fn exact_replay_rejects_changed_tool_result_or_input_but_accepts_json_formatting() {
        let mut fixture = Fixture::new();
        fixture.records[1].messages[2]["tool_calls"][0]["function"]["arguments"] =
            json!("{ \"path\" : \"src/lib.rs\" }");
        assert!(fixture.bind().complete());
        fixture.records[1].messages[3]["content"] =
            json!({"result":{"text":"different\n"},"is_error":false})
                .to_string()
                .into();
        assert!(!fixture.bind().complete());
        let mut fixture = Fixture::new();
        fixture.records[0].messages[1]["content"] = json!("Different task");
        assert!(!fixture.bind().complete());
    }

    #[test]
    fn failed_source_duplicate_origin_and_reversed_causality_are_unproven() {
        let mut fixture = Fixture::new();
        fixture.records[0].http_status = Some(500);
        assert!(!fixture.bind().complete());
        let mut fixture = Fixture::new();
        let mut duplicate = fixture.records[0].clone();
        duplicate.request_index = 9;
        fs::copy(
            fixture.dir.path().join("coding-0.response.sse"),
            fixture.dir.path().join("coding-9.response.sse"),
        )
        .unwrap();
        fixture.records.push(duplicate);
        let evidence = fixture.bind();
        assert!(!evidence.complete());
        assert!(evidence.unproven[0].contains("multiple successful origins"));
        let mut fixture = Fixture::new();
        fixture.records[1].submitted_ns = 1;
        assert!(!fixture.bind().complete());
    }

    #[test]
    fn missing_raw_finish_or_changed_stream_identity_cannot_be_inferred_from_public_delivery() {
        for suffix in ["data: [DONE]\r\n\r\n", "\"http-response-0\""] {
            let fixture = Fixture::new();
            let path = fixture.dir.path().join("coding-0.response.sse");
            let original = fs::read_to_string(&path).unwrap();
            let changed = if suffix.starts_with("data:") {
                original.replace(suffix, "")
            } else {
                original.replacen(suffix, "\"foreign-response\"", 1)
            };
            fs::write(path, changed).unwrap();
            assert!(!fixture.bind().complete());
        }
    }

    #[test]
    fn compaction_and_wrong_canonical_scope_are_explicitly_unproven() {
        let mut fixture = Fixture::new();
        fixture.public.compaction_events = 1;
        let evidence = fixture.bind();
        assert!(!evidence.complete());
        assert!(evidence.unproven[0].contains("compacted history"));
        let mut fixture = Fixture::new();
        fixture.public.tool_exchanges[0].request_id = "http-response-0".into();
        let evidence = fixture.bind();
        assert!(!evidence.complete());
        assert!(evidence.unproven[0].contains("logical model request"));
    }

    #[test]
    fn yaml_tool_envelope_preserves_code_strings_errors_and_scalar_types() {
        let code = "fn main() {\n    println!(\"<tool_call>\\\\path\");\n}\n";
        let result = json!({"content":code,"empty":"","crlf":"first\r\nsecond\r\n",
            "controls":"\u{0}\t\u{feff}","numeric_text":"001","bool_text":"false","none":null});
        let yaml = tool_result_content(OrchestralToolResultFormat::Yaml, &result, true).unwrap();
        let yaml = yaml.as_str().unwrap();
        assert!(yaml.contains("content: |\n"));
        assert!(yaml.contains("    println!(\"<tool_call>\\\\path\");"));
        assert_eq!(
            serde_yaml::from_str::<Value>(yaml).unwrap(),
            json!({"result":result,"is_error":true})
        );
        assert_eq!(
            tool_result_content(OrchestralToolResultFormat::Json, &result, true).unwrap(),
            json!({"result":result,"is_error":true}).to_string()
        );
    }

    #[test]
    fn declared_yaml_binds_exact_history_without_format_detection_or_normalization() {
        let fixture = Fixture::yaml();
        let evidence = fixture.bind_as(OrchestralToolResultFormat::Yaml);
        assert!(evidence.complete(), "{:?}", evidence.unproven);
        assert_eq!(
            evidence.tool_result_format,
            OrchestralToolResultFormat::Yaml
        );
        assert_eq!(evidence.receipts.len(), 2);
        assert!(
            !fixture.bind().complete(),
            "JSON declaration must not accept YAML wire"
        );
        assert!(!Fixture::new()
            .bind_as(OrchestralToolResultFormat::Yaml)
            .complete());
        let mut changed = Fixture::yaml();
        let original = changed.records[1].messages[3]["content"]
            .as_str()
            .unwrap()
            .to_owned();
        let with_blank_line = format!("{original}\n");
        assert_eq!(
            serde_yaml::from_str::<Value>(&original).unwrap(),
            serde_yaml::from_str::<Value>(&with_blank_line).unwrap()
        );
        changed.records[1].messages[3]["content"] = with_blank_line.into();
        assert!(
            !changed.bind_as(OrchestralToolResultFormat::Yaml).complete(),
            "semantic YAML equality is insufficient"
        );
        let mut changed = Fixture::yaml();
        changed.public.tool_exchanges[0].calls[0].is_error = true;
        assert!(!changed.bind_as(OrchestralToolResultFormat::Yaml).complete());
    }

    #[test]
    fn text_parts_binding_requires_exact_parts_metadata_and_declared_format() {
        let fixture = Fixture::text_parts();
        let evidence = fixture.bind_as(OrchestralToolResultFormat::TextParts);
        assert!(evidence.complete(), "{:?}", evidence.unproven);
        assert!(!fixture.bind().complete());
        assert!(!fixture.bind_as(OrchestralToolResultFormat::Yaml).complete());
        assert!(!Fixture::new()
            .bind_as(OrchestralToolResultFormat::TextParts)
            .complete());

        let mut reordered = Fixture::text_parts();
        reordered.records[1].messages[3]["content"]
            .as_array_mut()
            .unwrap()
            .reverse();
        assert!(!reordered
            .bind_as(OrchestralToolResultFormat::TextParts)
            .complete());

        let mut flattened = Fixture::text_parts();
        let text = flattened.records[1].messages[3]["content"]
            .as_array()
            .unwrap()
            .iter()
            .map(|part| part["text"].as_str().unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        flattened.records[1].messages[3]["content"] = text.into();
        assert!(!flattened
            .bind_as(OrchestralToolResultFormat::TextParts)
            .complete());

        let mut modified = Fixture::text_parts();
        let text = modified.records[1].messages[3]["content"][1]["text"]
            .as_str()
            .unwrap();
        modified.records[1].messages[3]["content"][1]["text"] = format!("{text}\n").into();
        assert!(!modified
            .bind_as(OrchestralToolResultFormat::TextParts)
            .complete());

        let mut changed_status = Fixture::text_parts();
        changed_status.public.tool_exchanges[0].calls[0].is_error = true;
        assert!(!changed_status
            .bind_as(OrchestralToolResultFormat::TextParts)
            .complete());
    }

    #[test]
    fn yaml_matching_history_does_not_relax_delivery_finish_metadata_or_causality() {
        let mut cancelled = Fixture::yaml();
        cancelled.public.delivered = false;
        cancelled.public.terminal = Some("run_cancelled".into());
        assert!(!cancelled
            .bind_as(OrchestralToolResultFormat::Yaml)
            .complete());
        let mut missing = Fixture::yaml();
        missing.public.tool_exchanges[0].calls[0].native_call_id = None;
        assert!(!missing.bind_as(OrchestralToolResultFormat::Yaml).complete());
        let mut incomplete = Fixture::yaml();
        incomplete.records[0].saw_done = false;
        assert!(!incomplete
            .bind_as(OrchestralToolResultFormat::Yaml)
            .complete());
        let mut premature = Fixture::yaml();
        premature.records[1].submitted_ns = 1;
        assert!(!premature
            .bind_as(OrchestralToolResultFormat::Yaml)
            .complete());
        let mut length = Fixture::yaml();
        length.records[0].finish_reasons = vec!["length".into()];
        let path = length.dir.path().join("coding-0.response.sse");
        let raw = fs::read_to_string(&path).unwrap().replace(
            "\"finish_reason\":\"tool_calls\"",
            "\"finish_reason\":\"length\"",
        );
        fs::write(path, raw).unwrap();
        assert!(!length.bind_as(OrchestralToolResultFormat::Yaml).complete());
    }
}
