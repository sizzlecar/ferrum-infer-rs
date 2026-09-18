//! Compare a complete Ferrum teacher capture with a native llama.cpp server.
//! The reference returns pre-sampling log probabilities, not raw logits; this
//! diagnostic compares full-vocabulary distributions on the exact same history.
use anyhow::{ensure, Context, Result};
use clap::Parser;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Duration,
};

#[path = "reference_teacher_compare/capture.rs"]
mod capture;
#[path = "reference_teacher_compare/distribution.rs"]
mod distribution;
#[cfg(test)]
#[path = "reference_teacher_compare/tests.rs"]
mod tests;

#[derive(Parser)]
#[command(
    about = "Compare exact teacher histories against a native llama.cpp /completion endpoint"
)]
struct Args {
    /// Complete schema-4 Ferrum checkpoint directory, including teacher-prompt.json.
    #[arg(long)]
    candidate_dir: PathBuf,
    /// Native llama.cpp server base URL. Model loading remains explicit.
    #[arg(long)]
    reference_url: reqwest::Url,
    /// JSON recording reference model hash, runtime revision/build, and launch settings.
    #[arg(long)]
    reference_identity_file: PathBuf,
    /// New directory for reference distributions and comparison evidence.
    #[arg(long)]
    output_dir: PathBuf,
    /// Reuse the reference's matching prefix across teacher decisions.
    #[arg(long, default_value_t = false)]
    cache_prompt: bool,
    #[arg(long, default_value_t = 600)]
    request_timeout_seconds: u64,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .with_context(|| format!("reserve {}", path.display()))?;
    file.write_all(bytes)?;
    Ok(())
}

fn reference_request(history: &[u32], vocabulary: usize, cache_prompt: bool) -> Value {
    json!({"prompt":history,"n_predict":1,"stream":false,"temperature":-1.0,
        "top_k":0,"top_p":1.0,"min_p":0.0,"repeat_penalty":1.0,
        "presence_penalty":0.0,"frequency_penalty":0.0,
        "n_probs":vocabulary,"post_sampling_probs":false,"return_tokens":true,
        "cache_prompt":cache_prompt,"seed":0})
}

async fn compare(args: &Args) -> Result<Value> {
    ensure!(
        args.request_timeout_seconds > 0,
        "request timeout must be positive"
    );
    ensure!(
        matches!(args.reference_url.scheme(), "http" | "https"),
        "reference endpoint must use HTTP(S)"
    );
    ensure!(
        args.reference_url.username().is_empty() && args.reference_url.password().is_none(),
        "reference URL must not contain credentials"
    );
    let capture = capture::Capture::read(&args.candidate_dir)?;
    let identity_bytes = fs::read(&args.reference_identity_file)?;
    let identity: Value = serde_json::from_slice(&identity_bytes)?;
    ensure!(
        identity.is_object() && identity.as_object().is_some_and(|map| !map.is_empty()),
        "reference identity must be a nonempty JSON object"
    );
    write_new(
        &args.output_dir.join("reference-identity.json"),
        &identity_bytes,
    )?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(args.request_timeout_seconds))
        .build()?;
    let endpoint = args.reference_url.join("completion")?;
    let mut history = capture.prompt.clone();
    let mut results = Vec::new();
    for (index, wave) in capture.waves.iter().enumerate() {
        let request = reference_request(&history, wave.logits.len(), args.cache_prompt);
        let response = client
            .post(endpoint.clone())
            .json(&request)
            .send()
            .await?
            .error_for_status()?
            .bytes()
            .await?;
        let body: Value = serde_json::from_slice(&response)?;
        let reference =
            distribution::reference_log_probabilities(&body, wave.logits.len(), history.len())?;
        let metrics = distribution::compare(&reference, &wave.logits, wave.token);
        let raw: Vec<_> = reference
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let raw_file = format!("reference-log-probabilities-{index:04}.f64le");
        write_new(&args.output_dir.join(&raw_file), &raw)?;
        let record = json!({"teacher_token_index":index,"teacher_token_id":wave.token,
            "history_token_count":history.len(),"history_token_ids_sha256":capture::token_digest(&history),
            "candidate_wave_sha256":wave.manifest_sha256,"candidate_logits_sha256":wave.raw_sha256,
            "reference_request":request,"reference_response_sha256":sha256(&response),
            "reference_timings":body.get("timings"),"reference_distribution":{
                "file":raw_file,"encoding":"f64-le","element_count":reference.len(),
                "zero_probability_token_count":reference.iter().filter(|p|p.exp()==0.0).count(),
                "sha256":sha256(&raw),"representation":"pre-sampling-log-probabilities"},
            "comparison":metrics.as_ref().ok(),
            "comparison_error":metrics.as_ref().err().map(|error|format!("{error:#}"))});
        write_new(
            &args.output_dir.join(format!("decision-{index:04}.json")),
            &serde_json::to_vec_pretty(&record)?,
        )?;
        metrics.with_context(|| format!("teacher decision {index} is not measurable; complete reference evidence was retained"))?;
        results.push(record);
        history.push(wave.token);
        eprintln!(
            "compared teacher decision {}/{}",
            index + 1,
            capture.waves.len()
        );
    }
    let aggregate = distribution::aggregate(&results)?;
    Ok(
        json!({"schema_version":1,"scope":"external-runtime-teacher-distribution-diagnostic",
        "release_approved":false,"weight_identity_independently_verified":false,
        "reference_identity":identity,"reference_identity_sha256":sha256(&identity_bytes),
        "reference_url":args.reference_url,"reference_cache_prompt":args.cache_prompt,
        "candidate_plan_sha256":capture.plan_sha256,"teacher_token_ids_sha256":capture.teacher_sha256,
        "prompt_token_ids_sha256":capture::token_digest(&capture.prompt),
        "reference_representation":"native endpoint pre-sampling log probabilities; F32 softmax then log",
        "candidate_representation":"captured raw F32 logits; F64 log-softmax",
        "raw_logit_error_measured":false,"wave_count":results.len(),"decisions":results,"aggregate":aggregate}),
    )
}

#[tokio::main]
async fn main() -> std::process::ExitCode {
    let args = Args::parse();
    if let Err(error) = fs::create_dir(&args.output_dir) {
        eprintln!(
            "reserve output directory {}: {error}",
            args.output_dir.display()
        );
        return std::process::ExitCode::FAILURE;
    }
    let result = compare(&args).await;
    let report = result.as_ref().cloned().unwrap_or_else(|error| {
        json!({"schema_version":1,
        "scope":"external-runtime-teacher-distribution-diagnostic","release_approved":false,
        "error":format!("{error:#}")})
    });
    if let Err(error) = write_new(
        &args.output_dir.join("report.json"),
        &serde_json::to_vec_pretty(&report).expect("JSON evidence"),
    ) {
        eprintln!("write report: {error:#}");
        return std::process::ExitCode::FAILURE;
    }
    match result {
        Ok(_) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("reference teacher comparison: {error:#}");
            std::process::ExitCode::FAILURE
        }
    }
}
