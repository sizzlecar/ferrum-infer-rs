//! Report-only standard-answer reference on a frozen Ferrum capture's actual input IDs.
//! This does not prepare samples, modify the paired experiment, or establish a quality gate.
use anyhow::{ensure, Context, Result};
use clap::{Parser, Subcommand};
use futures::{stream, StreamExt};
use serde::{de::DeserializeOwned, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

#[path = "reference_quality/capture.rs"]
mod capture;
#[path = "reference_quality/input.rs"]
mod input;
#[path = "reference_quality/score.rs"]
mod score;
#[cfg(test)]
#[path = "reference_quality/tests.rs"]
mod tests;

#[derive(Parser)]
#[command(about = "Collect a report-only llama.cpp reference using frozen actual prompt token IDs")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Pin the guard without client arguments, which contain this pin.
    GuardPin {
        #[arg(long)]
        config: PathBuf,
    },
    Run(Args),
}
#[derive(clap::Args)]
struct Args {
    #[arg(long)]
    plan: PathBuf,
    #[arg(long)]
    plan_sha256: String,
    /// Completed strict quality-gate capture, containing every sample and its actual prompt dump.
    #[arg(long)]
    strict: PathBuf,
    #[arg(long)]
    answers: PathBuf,
    /// Independently pinned lifecycle guard; this tool neither launches nor attests the server.
    #[arg(long)]
    guard_config: PathBuf,
    #[arg(long)]
    guard_runtime_sha256: String,
    #[arg(long)]
    reference_url: reqwest::Url,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long)]
    execute: bool,
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn read_bounded(path: &Path, maximum: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    fs::File::open(path)
        .with_context(|| path.display().to_string())?
        .take(maximum + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= maximum,
        "file exceeds read bound: {}",
        path.display()
    );
    Ok(bytes)
}
fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    Ok(serde_json::from_slice(&read_bounded(
        path,
        64 * 1024 * 1024,
    )?)?)
}
fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(bytes)?;
    Ok(())
}
fn save(path: &Path, value: &impl Serialize) -> Result<()> {
    write_new(
        path,
        &[serde_json::to_vec_pretty(value)?, b"\n".to_vec()].concat(),
    )
}

async fn run(a: Args) -> Result<()> {
    let source = input::load(&a.plan, &a.plan_sha256, &a.strict, &a.answers)?;
    let guard_bytes = read_bounded(&a.guard_config, 16 * 1024 * 1024)?;
    let guard: Value = serde_json::from_slice(&guard_bytes)?;
    ensure!(
        input::guard_runtime_pin(&guard)? == a.guard_runtime_sha256,
        "reference guard runtime SHA mismatch"
    );
    let collector_sha = sha(&read_bounded(&std::env::current_exe()?, 256 * 1024 * 1024)?);
    let endpoint =
        input::validate_reference(&guard, &source.plan, &a.reference_url, &collector_sha)?;
    fs::create_dir(&a.output_dir)?;
    write_new(
        &a.output_dir.join("source-frozen-plan.json"),
        &source.plan_bytes,
    )?;
    write_new(&a.output_dir.join("reference-guard.json"), &guard_bytes)?;
    save(
        &a.output_dir.join("manifest.json"),
        &json!({
            "schema_version":1,"scope":"standard-answer-reference-only","report_only":true,
            "execute":a.execute,"source_plan_sha256":a.plan_sha256,
            "strict_capture":a.strict,"source_manifest_sha256":source.manifest_sha256,
            "reference_guard_runtime_sha256":a.guard_runtime_sha256,
            "reference_guard_sha256":sha(&guard_bytes),"collector_binary_sha256":collector_sha,
            "weights_sha256":source.plan.execution.weights_sha256,"endpoint":endpoint,
            "sample_count":source.samples.len(),"policy":source.plan.policy,
            "input_policy":"exact actual strict prompt IDs; no retemplating, retokenization, or teacher-forced outputs",
            "runtime_attestation":"external pinned guard responsibility; HTTP success alone does not attest runtime or model",
            "source_paired_gate_changed":false,"answer_key_used_for_requests":false
        }),
    )?;
    let mut requests = Vec::new();
    for (index, sample) in source.samples.iter().enumerate() {
        let directory = a.output_dir.join(format!("sample-{index:03}"));
        fs::create_dir(&directory)?;
        save(&directory.join("source-case.json"), &sample.case)?;
        save(
            &directory.join("source-prompt.json"),
            &json!({
                "source_request_id":sample.request_id,"token_ids":sample.prompt,
                "source_prompt_dump_sha256":sample.prompt_dump_sha256,
                "source_response_sha256":sample.response_sha256,"retokenized":false
            }),
        )?;
        let request = capture::request(&source.plan.policy, sample);
        save(&directory.join("request.json"), &request)?;
        let bytes = serde_json::to_vec(&request)?;
        write_new(&directory.join("request.body"), &bytes)?;
        requests.push((index, directory, bytes, sample));
    }
    if !a.execute {
        return Ok(());
    }
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(source.plan.policy.timeout_secs))
        .build()?;
    let policy = &source.plan.policy;
    let mut outcomes: Vec<_> = stream::iter(requests).map(|(index, directory, bytes, sample)| {
        let client = &client;
        let endpoint = &endpoint;
        async move {
            let captured = capture::collect(client, endpoint, bytes, &directory, sample, policy).await;
            let row = match captured {
                Ok(observation) => json!({"sample_index":index,"id":sample.case.id,
                    "source_record_index":sample.case.source_record_index,"stratum":sample.case.stratum,
                    "status":"valid_scored","correct":score::grade(&sample.case.task,
                        &observation.text,&observation.finish,&sample.answer),
                    "finish_reason":observation.finish,"stop_type":observation.stop_type,
                    "completion_tokens":observation.tokens.len(),
                    "reference_prompt_ids_confirmed":observation.prompt_confirmed}),
                Err(error) => json!({"sample_index":index,"id":sample.case.id,
                    "source_record_index":sample.case.source_record_index,"stratum":sample.case.stratum,
                    "status":"invalid_not_scored","error":format!("{error:#}")}),
            };
            save(&directory.join("result.json"), &row)?;
            Ok::<_,anyhow::Error>((index,row))
        }
    }).buffer_unordered(policy.concurrency).collect().await;
    let mut rows = Vec::with_capacity(outcomes.len());
    for outcome in outcomes.drain(..) {
        rows.push(outcome?);
    }
    rows.sort_by_key(|(index, _)| *index);
    let rows: Vec<Value> = rows.into_iter().map(|(_, row)| row).collect();
    let invalid = rows
        .iter()
        .filter(|r| r["status"] != "valid_scored")
        .count();
    let correct = rows.iter().filter(|r| r["correct"] == true).count();
    save(&a.output_dir.join("results.json"), &rows)?;
    save(
        &a.output_dir.join("summary.json"),
        &json!({
            "scope":"standard-answer-reference-only","report_only":true,"gate_pass":null,
            "status":if invalid == 0 {"reference_complete"} else {"reference_incomplete"},
            "source_plan_sha256":a.plan_sha256,"sample_count":rows.len(),"invalid_samples":invalid,
            "correct":correct,"accuracy":if invalid==0 {Some(correct as f64/rows.len() as f64)}else{None},
            "length_truncations":rows.iter().filter(|r|r["finish_reason"]=="length").count(),
            "reference_prompt_ids_confirmed":rows.iter().filter(|r|r["reference_prompt_ids_confirmed"]==true).count(),
            "policy":"valid length truncations are wrong; invalid evidence is retained, never dropped or scored wrong; no bootstrap or noninferiority conclusion",
            "source_paired_gate_changed":false
        }),
    )?;
    ensure!(
        invalid == 0,
        "{invalid} invalid reference samples; original paired gate unchanged"
    );
    Ok(())
}

#[tokio::main]
async fn main() -> std::process::ExitCode {
    let result = match Cli::parse().command {
        Command::GuardPin { config } => read_json(&config).and_then(|guard| {
            println!("{}", input::guard_runtime_pin(&guard)?);
            Ok(())
        }),
        Command::Run(args) => run(args).await,
    };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error:#}");
            std::process::ExitCode::FAILURE
        }
    }
}
