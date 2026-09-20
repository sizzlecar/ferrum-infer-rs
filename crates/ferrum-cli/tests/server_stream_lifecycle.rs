//! Live resource-lifetime regression against a dedicated `ferrum serve`.
//!
//! Start the backend/model under test with prefix/session caching disabled,
//! at least three sequences, context >= 4096, and the desired split/mixed
//! execution policy. Set FERRUM_LIVE_TEST_URL to its HTTP origin and run this
//! ignored test. It exercises real disconnects; scheduler/device traces are
//! still required to prove the exact point of cancellation within a GPU wave.

use futures::future::join_all;
use reqwest::{Client, Response};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

struct Stream {
    response: Response,
    pending: Vec<u8>,
    request_id: Option<String>,
    content_seen: bool,
    done: bool,
    output_tokens: Option<u64>,
    finish_reason: Option<String>,
}

impl Stream {
    async fn start(client: &Client, origin: &str, model: &str, prompt: &str, limit: u32) -> Self {
        let response = client
            .post(format!("{origin}/v1/chat/completions"))
            .json(&json!({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": true,
                "stream_options": {"include_usage": true},
                "temperature": 0,
                "top_p": 1,
                "repetition_penalty": 1,
                "seed": 42,
                "max_tokens": limit,
                "ignore_eos": true,
                "chat_template_kwargs": {"enable_thinking": false}
            }))
            .send()
            .await
            .expect("start live generation");
        assert!(response.status().is_success(), "HTTP {}", response.status());
        Self {
            response,
            pending: Vec::new(),
            request_id: None,
            content_seen: false,
            done: false,
            output_tokens: None,
            finish_reason: None,
        }
    }

    async fn advance(&mut self) -> bool {
        let Some(bytes) = self.response.chunk().await.expect("read SSE body") else {
            assert!(self.pending.iter().all(u8::is_ascii_whitespace));
            return false;
        };
        self.pending.extend_from_slice(&bytes);
        while let Some(end) = self.pending.iter().position(|byte| *byte == b'\n') {
            let line = self.pending.drain(..=end).collect::<Vec<_>>();
            let line = std::str::from_utf8(&line).expect("UTF-8 SSE line").trim();
            let Some(data) = line.strip_prefix("data:").map(str::trim) else {
                continue;
            };
            assert!(!self.done, "data after stream terminal");
            if data == "[DONE]" {
                self.done = true;
                continue;
            }
            let event: Value = serde_json::from_str(data).expect("JSON SSE event");
            assert!(event.get("error").is_none(), "stream error: {event}");
            if let Some(id) = event["id"].as_str() {
                if let Some(previous) = &self.request_id {
                    assert_eq!(previous, id, "stream request identity changed");
                }
                self.request_id = Some(id.to_owned());
            }
            self.content_seen |= event["choices"][0]["delta"]["content"]
                .as_str()
                .is_some_and(|text| !text.is_empty());
            if let Some(tokens) = event["usage"]["completion_tokens"].as_u64() {
                assert!(self.output_tokens.is_none(), "duplicate terminal usage");
                self.output_tokens = Some(tokens);
            }
            if let Some(reason) = event["choices"][0]["finish_reason"].as_str() {
                assert!(self.finish_reason.is_none(), "duplicate finish reason");
                self.finish_reason = Some(reason.to_owned());
            }
        }
        true
    }

    async fn first_content(&mut self) {
        while !self.content_seen {
            assert!(self.advance().await, "EOF before generated content");
            assert!(
                !self.done,
                "request completed before live cancellation point"
            );
        }
        assert!(!self.done, "request already terminal at cancellation point");
    }

    async fn finish(mut self, expected_tokens: u64) {
        while self.advance().await {}
        assert!(self.done, "missing stream terminal");
        assert!(self.content_seen, "empty generated text");
        assert!(self.request_id.is_some(), "missing request identity");
        assert_eq!(self.finish_reason.as_deref(), Some("length"));
        assert_eq!(self.output_tokens, Some(expected_tokens));
        eprintln!(
            "completed request {:?}: {expected_tokens} usage tokens",
            self.request_id
        );
    }
}

async fn health(client: &Client, origin: &str) -> Value {
    client
        .get(format!("{origin}/health"))
        .send()
        .await
        .expect("health response")
        .error_for_status()
        .expect("healthy HTTP status")
        .json()
        .await
        .expect("health JSON")
}

async fn assert_drained(client: &Client, origin: &str) -> Value {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let state = health(client, origin).await;
        assert_eq!(state["status"], "healthy");
        let pools = state["cache"]["prefix_cache"]["dynamic_pools"]["pools"]
            .as_array()
            .expect("typed runtime pool status");
        assert!(!pools.is_empty(), "real plan exposes its resource pools");
        let mut transient_drained = true;
        for pool in pools {
            assert_eq!(pool["poisoned"], false, "{pool}");
            assert_eq!(pool["quarantined_chunks"], 0, "{pool}");
            for scope in [
                "request",
                "sequence",
                "checkpoint",
                "step",
                "invocation",
                "initial_sequence_bundle",
            ] {
                transient_drained &= pool["live_occupancy"]["transient"][scope]["physical_bytes"]
                    .as_u64()
                    .expect("typed transient physical byte count")
                    == 0;
            }
        }
        if transient_drained
            && state["engine"]["active_requests"].as_u64() == Some(0)
            && state["engine"]["queued_requests"].as_u64() == Some(0)
            && state["admission"]["active_sequences"].as_u64() == Some(0)
            && state["admission"]["queue_depth"].as_u64() == Some(0)
        {
            return state;
        }
        assert!(Instant::now() < deadline, "requests did not drain: {state}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires a dedicated live model server; set FERRUM_LIVE_TEST_URL"]
async fn disconnected_decode_releases_resources_without_losing_peers() {
    tokio::time::timeout(Duration::from_secs(240), async {
        let origin = std::env::var("FERRUM_LIVE_TEST_URL")
            .expect("FERRUM_LIVE_TEST_URL must name the dedicated test server")
            .trim_end_matches('/')
            .to_owned();
        let client = Client::builder()
            .no_proxy()
            .http1_only()
            .timeout(Duration::from_secs(180))
            .build()
            .unwrap();
        let before = assert_drained(&client, &origin).await;
        let models: Value = client
            .get(format!("{origin}/v1/models"))
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .await
            .unwrap();
        let model = models["data"][0]["id"].as_str().expect("served model");
        let (mut cancelled, mut survivor) = tokio::join!(
            Stream::start(
                &client,
                &origin,
                model,
                "Continue counting integers starting at one.",
                1024
            ),
            Stream::start(
                &client,
                &origin,
                model,
                "Continue counting integers starting at ten.",
                128
            )
        );
        tokio::join!(cancelled.first_content(), survivor.first_content());
        let prompt = format!(
            "Read this list then repeat its colors: {}",
            "red green blue. ".repeat(256)
        );
        let newcomer = Stream::start(&client, &origin, model, &prompt, 32).await;
        eprintln!("disconnecting live request {:?}", cancelled.request_id);
        // Drop the HTTP/1 body while it still has unconsumed generation.
        // The peer and a newly admitted long prefill remain independent.
        drop(cancelled);
        tokio::join!(survivor.finish(128), newcomer.finish(32));
        assert_drained(&client, &origin).await;
        let followup_count = 3;
        let followups = (0..followup_count).map(|_| async {
            Stream::start(&client, &origin, model, "List the primary colors.", 8)
                .await
                .finish(8)
                .await;
        });
        join_all(followups).await;
        let after = assert_drained(&client, &origin).await;
        let completed = |state: &Value| {
            state["admission"]["completed_requests_total"]
                .as_u64()
                .expect("completed request count")
        };
        assert_eq!(
            completed(&after) - completed(&before),
            2 + followup_count,
            "the disconnected request must not silently run to normal completion"
        );
        assert_eq!(
            after["admission"]["failed_requests_total"],
            before["admission"]["failed_requests_total"],
            "disconnect must cancel its request without failing generation"
        );
    })
    .await
    .expect("bounded live resource lifecycle regression");
}
