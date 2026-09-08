//! Reuse the actual model regression wire parsers and semantic answer oracle.
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::time::Duration;

/// Construct the request only after its future is polled inside a Tokio runtime.
pub async fn health(
    client: &reqwest::Client,
    url: &str,
    timeout: Duration,
) -> reqwest::Result<Value> {
    client
        .get(format!("{url}/health"))
        .timeout(timeout)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await
}

#[path = "../model_regression/protocol.rs"]
#[allow(dead_code, unused_imports)]
mod protocol;

pub fn chat(text: &str) -> Result<Value> {
    let result = protocol::sync(text)?;
    ensure!(
        result.finish == "stop",
        "upgrade inference did not finish naturally"
    );
    protocol::answer(result.content(), "42")?;
    Ok(json!({"message":result.message,"finish_reason":result.finish,"usage":result.usage}))
}

pub fn run(text: &str, model: &str) -> Result<Value> {
    let records = protocol::run_records(text)?;
    let ready = records
        .iter()
        .find(|r| r["event"] == "ready")
        .context("missing ready")?;
    let backend = ready["backend"].as_str().context("missing run backend")?;
    ensure!(
        backend
            .strip_prefix("CUDA(")
            .and_then(|s| s.strip_suffix(')'))
            .is_some_and(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())),
        "new run did not use CUDA"
    );
    ensure!(
        ready["requested_model"] == model,
        "new run loaded another model"
    );
    let assistants: Vec<_> = records
        .iter()
        .filter(|r| r["event"] == "assistant")
        .collect();
    ensure!(
        assistants.len() == 1,
        "new one-shot run omitted or repeated its response"
    );
    let answer = assistants[0];
    ensure!(
        matches!(answer["finish_reason"].as_str(), Some("eos" | "stop")),
        "new run did not finish naturally"
    );
    let mut wire = json!({"choices":[{"index":0,"message":{"role":"assistant","content":answer["content"],"reasoning":answer["reasoning"],"tool_calls":answer["tool_calls"]},"finish_reason":"stop"}],"usage":answer["usage"]});
    if let Some(alias) = answer.get("reasoning_content") {
        wire["choices"][0]["message"]["reasoning_content"] = alias.clone();
    }
    chat(&wire.to_string())?;
    Ok(json!({"ready":ready,"assistant":answer}))
}

pub fn require_overlap(observations: &[Value], start: &str, end: &str) -> Result<()> {
    let start = chrono::DateTime::parse_from_rfc3339(start)?;
    let end = chrono::DateTime::parse_from_rfc3339(end)?;
    ensure!(end >= start, "installer clock moved backwards");
    let mut overlaps = false;
    for observation in observations {
        let began = chrono::DateTime::parse_from_rfc3339(
            observation["started_at"]
                .as_str()
                .context("missing request start")?,
        )?;
        let finished = chrono::DateTime::parse_from_rfc3339(
            observation["finished_at"]
                .as_str()
                .context("missing request end")?,
        )?;
        ensure!(finished >= began, "request clock moved backwards");
        overlaps |= began < end && finished > start;
    }
    ensure!(
        overlaps,
        "no successful inference overlapped the actual installer process"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn health_request_can_start_outside_the_runtime_context() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        // Match the synchronous lifecycle caller: construct the future before block_on.
        let request = health(&client, &url, Duration::from_secs(2));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let actual = runtime.block_on(async {
            let listener = tokio::net::TcpListener::from_std(listener).unwrap();
            let server = async {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut buffer = [0; 1024];
                let n = stream.read(&mut buffer).await.unwrap();
                assert!(buffer[..n].starts_with(b"GET /health HTTP/1.1\r\n"));
                stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 20\r\nConnection: close\r\n\r\n{\"status\":\"healthy\"}").await.unwrap();
            };
            tokio::time::timeout(Duration::from_secs(5), async {
                let ((), response) = tokio::join!(server, request);
                response.unwrap()
            })
            .await
            .unwrap()
        });
        assert_eq!(actual, json!({"status":"healthy"}));
    }
    #[test]
    fn live_inference_requires_semantics_usage_and_natural_completion() {
        let valid = json!({"choices":[{"index":0,"message":{"role":"assistant","content":"42"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}});
        assert!(chat(&valid.to_string()).is_ok());
        for changed in [json!("41"), json!("")] {
            let mut bad = valid.clone();
            bad["choices"][0]["message"]["content"] = changed;
            assert!(chat(&bad.to_string()).is_err());
        }
        let mut bad = valid.clone();
        bad["usage"]["total_tokens"] = json!(0);
        assert!(chat(&bad.to_string()).is_err());
        let mut bad = valid;
        bad["choices"][0]["finish_reason"] = json!("length");
        assert!(chat(&bad.to_string()).is_err());
    }
    #[test]
    fn old_requests_outside_installer_interval_do_not_prove_live_upgrade() {
        let request = |start, end| json!({"started_at":start,"finished_at":end});
        let start = "2026-09-08T09:00:10Z";
        let end = "2026-09-08T09:00:20Z";
        assert!(require_overlap(
            &[
                request("2026-09-08T09:00:00Z", start),
                request(end, "2026-09-08T09:00:21Z")
            ],
            start,
            end
        )
        .is_err());
        assert!(require_overlap(
            &[request("2026-09-08T09:00:09Z", "2026-09-08T09:00:11Z")],
            start,
            end
        )
        .is_ok());
        assert!(require_overlap(&[request(end, start)], start, end).is_err());
    }
    #[test]
    fn new_run_requires_real_cuda_identity_and_canonical_assistant_record() {
        let ready = json!({"event":"ready","backend":"CUDA(0)","requested_model":"local-model"});
        let answer = json!({"event":"assistant","content":"42","finish_reason":"eos","usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}});
        let transcript = |ready: &Value, answer: &Value| {
            format!("{ready}\n{answer}\n{}\n", json!({"event":"exit"}))
        };
        assert!(run(&transcript(&ready, &answer), "local-model").is_ok());
        let mut wrong = ready.clone();
        wrong["backend"] = json!("CPU");
        assert!(run(&transcript(&wrong, &answer), "local-model").is_err());
        assert!(run(&transcript(&ready, &answer), "another-model").is_err());
        let mut malformed = answer.clone();
        malformed["reasoning_content"] = Value::Null;
        assert!(run(&transcript(&ready, &malformed), "local-model").is_err());
        let mut malformed = answer;
        malformed["tool_calls"] = json!([{}]);
        assert!(run(&transcript(&ready, &malformed), "local-model").is_err());
    }
}
