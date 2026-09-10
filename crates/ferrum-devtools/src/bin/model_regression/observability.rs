use super::*;
use ferrum_bench_core::release_regression::{
    model_observability::{verify, ObservabilityEvidence, ObservedRequest},
    Backend, Entrypoint,
};
use ferrum_types::FerrumProfileEvent;
use std::{fs, path::Path, time::Instant};

fn read_events(path: &Path) -> Result<Vec<FerrumProfileEvent>> {
    fs::read_to_string(path)
        .with_context(|| format!("read actual journal {}", path.display()))?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).context("parse actual scheduler journal event"))
        .collect()
}

fn backend(args: &Args) -> Result<Backend> {
    serde_json::from_value(json!(args.backend)).context("parse expected backend")
}

pub(crate) async fn run_observability(args: &Args) -> Result<Value> {
    let run = run_chat(
        args,
        "run-observability",
        Input::Prompt(ARITHMETIC_PROMPT),
        None,
    )
    .await?;
    ensure!(
        run.assistants.len() == 1,
        "diagnostic run did not produce its single requested answer"
    );
    let assistant = &run.assistants[0];
    let observation = ObservabilityEvidence {
        requests: vec![ObservedRequest {
            entrypoint: Entrypoint::Run,
            request_id: assistant["request_id"]
                .as_str()
                .context("missing output request identity")?
                .into(),
            content: assistant["content"]
                .as_str()
                .context("missing output content")?
                .into(),
            finish_reason: assistant["finish_reason"]
                .as_str()
                .context("missing finish reason")?
                .into(),
            usage: assistant["usage"].clone(),
        }],
        events: read_events(&args.report_dir.join("run-observability.scheduler.jsonl"))?,
    };
    verify(
        &observation,
        backend(args)?,
        &[Entrypoint::Run],
        args.max_tokens,
    )
    .map_err(anyhow::Error::msg)?;
    Ok(
        json!({"ready": run.ready, "source_identity": identity::source_evidence(args, "run-observability")?, "observability": observation}),
    )
}

pub(crate) async fn serve_observability(server: &Server<'_>) -> Result<Value> {
    let body = request(
        server,
        vec![json!({"role": "user", "content": ARITHMETIC_PROMPT})],
    );
    let mut requests = Vec::new();
    for (name, entrypoint, stream) in [
        ("observability-sync", Entrypoint::ServeSync, false),
        ("observability-stream", Entrypoint::ServeStream, true),
    ] {
        let answer = chat(server, name, body.clone(), stream).await?;
        requests.push(ObservedRequest {
            entrypoint,
            request_id: answer
                .response_id
                .context("HTTP output is missing its request identity")?,
            content: answer.message["content"]
                .as_str()
                .context("missing HTTP content")?
                .into(),
            finish_reason: answer.finish,
            usage: answer.usage,
        });
    }
    // HTTP completion can precede the asynchronous journal's flush. Wait for
    // these actual requests to close; do not assume a sleep proves persistence.
    let path = server.args.report_dir.join("serve.scheduler.jsonl");
    let start = Instant::now();
    let events = loop {
        let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        if text.ends_with('\n') {
            let events = read_events(&path)?;
            if requests.iter().all(|request| {
                events.iter().any(|event| {
                    event.request_id == request.request_id && event.phase == "engine_request_close"
                })
            }) {
                break events;
            }
        }
        ensure!(
            start.elapsed() < Duration::from_secs(server.args.request_timeout_secs),
            "request closure was not persisted before the observation deadline"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    let observation = ObservabilityEvidence { requests, events };
    verify(
        &observation,
        backend(server.args)?,
        &[Entrypoint::ServeSync, Entrypoint::ServeStream],
        server.args.max_tokens,
    )
    .map_err(anyhow::Error::msg)?;
    Ok(json!({"observability": observation}))
}
