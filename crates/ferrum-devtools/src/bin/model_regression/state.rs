use super::*;
use ferrum_bench_core::release_regression::model_state::{
    self, RunStep, ServeStateEvidence, StateExchange,
};

pub(crate) async fn run_state(args: &Args) -> Result<Value> {
    let mut stdin = String::new();
    for step in model_state::run_steps() {
        stdin.push_str(&match step {
            RunStep::Turn { prompt, .. } => prompt,
            RunStep::Reset => "/clear".into(),
        });
        stdin.push('\n');
    }
    stdin.push_str("/bye\n");
    let run = run_chat(args, "run-state", Input::Repl(&stdin), None).await?;
    let protocol = serde_json::from_value(run.ready["reasoning_protocol"].clone())?;
    let evidence = json!({"ready": run.ready, "source_identity": identity::source_evidence(args, "run-state")?, "records": run.records});
    model_state::verify_run(
        evidence["records"]
            .as_array()
            .context("run state records")?,
        protocol,
        args.max_tokens,
    )
    .map_err(anyhow::Error::msg)
    .map_err(|error| super::super::case_failure(evidence.clone(), error))?;
    Ok(evidence)
}

async fn exchange(
    server: &Server<'_>,
    name: &str,
    messages: Vec<Value>,
    stream: bool,
) -> Result<StateExchange> {
    let result = chat(server, name, request(server, messages.clone()), stream).await?;
    Ok(StateExchange {
        observation: observation(&result),
        request_id: result
            .response_id
            .context("state exchange has no request identity")?,
        messages,
        stream,
    })
}

pub(crate) async fn serve_state(server: &Server<'_>) -> Result<Value> {
    let mut evidence = ServeStateEvidence {
        writes: Vec::new(),
        recall_rounds: Vec::new(),
        fresh: Vec::new(),
    };
    let result = async {
        let mut histories = Vec::new();
        for (index, code) in model_state::CODES.iter().enumerate() {
            let mut history = vec![json!({"role": "user", "content": model_state::remember(code)})];
            let write = exchange(
                server,
                &format!("state-write-{index}"),
                history.clone(),
                index == 1,
            )
            .await?;
            history.push(write.observation["message"].clone());
            evidence.writes.push(write);
            histories.push(history);
        }
        for round in 0..2 {
            for history in &mut histories {
                history.push(json!({"role": "user", "content": model_state::RECALL}));
            }
            // Launch both requests together even when capacity admits only one;
            // successful serialization does not claim simultaneous GPU execution.
            let names = [
                format!("state-recall-{round}-0"),
                format!("state-recall-{round}-1"),
            ];
            let (a, b) = tokio::join!(
                exchange(server, &names[0], histories[0].clone(), round == 1),
                exchange(server, &names[1], histories[1].clone(), round == 0),
            );
            let replies = vec![a?, b?];
            for (history, reply) in histories.iter_mut().zip(&replies) {
                history.push(reply.observation["message"].clone());
            }
            evidence.recall_rounds.push(replies);
        }
        for index in 0..2 {
            evidence.fresh.push(
                exchange(
                    server,
                    &format!("state-fresh-{index}"),
                    vec![json!({"role": "user", "content": model_state::EMPTY_RECALL})],
                    index == 1,
                )
                .await?,
            );
        }
        let protocol = serde_json::from_value(server.health["reasoning_protocol"].clone())?;
        model_state::verify_serve(&evidence, protocol, server.args.max_tokens)
            .map_err(anyhow::Error::msg)
    }
    .await;
    let evidence = json!({"state": evidence});
    result.map_err(|error| super::super::case_failure(evidence.clone(), error))?;
    Ok(evidence)
}
