//! The input length is explicit; observed tokenizer usage establishes coverage.
//! This probes fact retrieval, repeated input, and an appended history suffix.
use super::*;
use ferrum_bench_core::release_regression::model_tasks::verify_reasoning_absence_observation;
use std::fmt::Write;

const TARGET_KEY: &str = "target-key";
const TARGET_VALUE: &str = "violet-583";

#[derive(Debug, serde::Serialize)]
struct FactProbe {
    records: u32,
    target_record: u32,
    expected: &'static str,
    prompt: String,
}

impl FactProbe {
    fn new(records: u32) -> Self {
        let target_record = records / 2;
        let mut prompt = String::from("Read these reference records and find the requested value. Each record has a unique key.\n");
        for index in 0..records {
            if index == target_record {
                writeln!(
                    prompt,
                    "Record {index:06}: key={TARGET_KEY}; value={TARGET_VALUE}."
                )
                .unwrap();
            } else {
                writeln!(
                    prompt,
                    "Record {index:06}: key=archive-{index:06}; value=filler-{index:06}."
                )
                .unwrap();
            }
        }
        write!(prompt, "\nWhat is the value for key {TARGET_KEY}? Copy only its exact value, without explanation.").unwrap();
        Self {
            records,
            target_record,
            expected: TARGET_VALUE,
            prompt,
        }
    }
}

fn validate_answer(observed: &Value, expected: &str, args: &Args) -> Result<()> {
    verify_reasoning_absence_observation(observed, expected).map_err(anyhow::Error::msg)?;
    let usage = &observed["usage"];
    let prompt = usage["prompt_tokens"]
        .as_u64()
        .context("missing prompt usage")?;
    let completion = usage["completion_tokens"]
        .as_u64()
        .context("missing completion usage")?;
    ensure!(
        completion > 0
            && completion <= u64::from(args.max_tokens)
            && prompt.checked_add(completion) == usage["total_tokens"].as_u64(),
        "invalid long-context usage: {usage}"
    );
    let minimum = args
        .long_context_min_prompt_tokens
        .context("long-context minimum is required")?;
    ensure!(
        prompt >= u64::from(minimum),
        "long-context input had {prompt} prompt tokens, below the required {minimum}"
    );
    Ok(())
}

pub(crate) async fn run_long_context(args: &Args) -> Result<Value> {
    let probe = FactProbe::new(args.long_context_records.context("long-context records")?);
    let run = capture_run(
        args,
        "run-long-context",
        Input::Prompt(&probe.prompt),
        None,
        RunCaptureMode::State,
    )
    .await?;
    let evidence = json!({"probe": probe, "ready": run.ready, "answers": run.assistants,
        "source_identity": identity::source_evidence(args, "run-long-context")?});
    let result = (|| {
        let answers = evidence["answers"]
            .as_array()
            .context("missing long-context answers")?;
        ensure!(answers.len() == 1, "expected one long-context run answer");
        let answer = &answers[0];
        let mut observation = json!({"message": {"role": "assistant", "content": answer["content"],
            "reasoning": answer["reasoning"], "tool_calls": answer["tool_calls"]},
            "finish_reason": answer["finish_reason"], "usage": answer["usage"]});
        if let Some(alias) = answer.get("reasoning_content") {
            observation["message"]["reasoning_content"] = alias.clone();
        }
        validate_answer(&observation, TARGET_VALUE, args)
    })();
    result.map_err(|error| super::super::case_failure(evidence.clone(), error))?;
    Ok(evidence)
}

pub(crate) async fn serve_long_context(server: &Server<'_>) -> Result<Value> {
    let probe = FactProbe::new(
        server
            .args
            .long_context_records
            .context("long-context records")?,
    );
    let health_before = server.health_snapshot("long-context.before-health").await?;
    let mut evidence = json!({"probe": probe, "observations": {}, "health_before": health_before});
    let result = async {
        let mut body = request(server, vec![json!({"role": "user", "content": probe.prompt})]);
        body["chat_template_kwargs"] = json!({"enable_thinking": false});
        let mut first_message = Value::Null;
        for (name, stream) in [("long-context-sync", false), ("long-context-stream", true)] {
            let result = chat(server, name, body.clone(), stream).await?;
            let observed = observation(&result);
            evidence["observations"][name] = observed.clone();
            validate_answer(&observed, TARGET_VALUE, server.args)?;
            if !stream { first_message = result.message; }
        }
        let history = body["messages"].as_array_mut().context("long-context messages")?;
        history.push(first_message);
        history.push(json!({"role": "user", "content": "Copy that same value again, exactly. Reply with only the value."}));
        let continuation = chat(server, "long-context-continuation", body, false).await?;
        let observed = observation(&continuation);
        evidence["observations"]["long-context-continuation"] = observed.clone();
        validate_answer(&observed, TARGET_VALUE, server.args)
    }.await;
    let health_after = server.health_snapshot("long-context.after-health").await;
    match &health_after {
        Ok(health) => evidence["health_after"] = health.clone(),
        Err(error) => evidence["health_after_error"] = json!(format!("{error:#}")),
    }
    result.map_err(|error| super::super::case_failure(evidence.clone(), error))?;
    health_after.map_err(|error| super::super::case_failure(evidence.clone(), error))?;
    Ok(evidence)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn generated_fact_has_one_target_amid_parameterized_irrelevant_records() {
        for records in [1, 7, 128] {
            let probe = FactProbe::new(records);
            assert_eq!(
                probe
                    .prompt
                    .lines()
                    .filter(|line| line.starts_with("Record "))
                    .count(),
                records as usize
            );
            assert_eq!(probe.prompt.matches(TARGET_VALUE).count(), 1);
            assert_eq!(probe.target_record, records / 2);
        }
    }

    #[test]
    fn fact_oracle_checks_actual_usage_natural_completion_and_exact_identifier() {
        let args = Args::try_parse_from([
            "model-regression",
            "--ferrum-bin",
            "fixture",
            "--model",
            "fixture-model",
            "--backend",
            "metal",
            "--report-dir",
            "fixture-report",
            "--context-tokens",
            "4096",
            "--max-num-seqs",
            "1",
            "--long-context-records",
            "128",
            "--long-context-min-prompt-tokens",
            "1024",
        ])
        .unwrap();
        let observed = json!({"message": {"role": "assistant", "content": TARGET_VALUE}, "finish_reason": "stop",
            "usage": {"prompt_tokens": 1024, "completion_tokens": 5, "total_tokens": 1029}});
        validate_answer(&observed, TARGET_VALUE, &args).unwrap();
        for (field, value) in [
            ("content", json!("Violet-583")),
            ("content", json!("violet-584")),
            ("content", json!("583")),
        ] {
            let mut changed = observed.clone();
            changed["message"][field] = value;
            assert!(validate_answer(&changed, TARGET_VALUE, &args).is_err());
        }
        let mut short = observed.clone();
        short["usage"] = json!({"prompt_tokens": 16, "completion_tokens": 5, "total_tokens": 21});
        assert!(validate_answer(&short, TARGET_VALUE, &args).is_err());
        let mut truncated = observed;
        truncated["finish_reason"] = json!("length");
        assert!(validate_answer(&truncated, TARGET_VALUE, &args).is_err());
    }
}
