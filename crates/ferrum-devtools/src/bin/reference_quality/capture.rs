use super::*;
use input::{Policy, Sample};

pub struct Observation {
    pub text: String,
    pub tokens: Vec<u32>,
    pub finish: String,
    pub stop_type: String,
    pub prompt_confirmed: bool,
}
pub fn request(policy: &Policy, sample: &Sample) -> Value {
    // Native numeric prompt arrays bypass template rendering and tokenization.
    // Unlike reference_teacher_compare, generate the complete answer freely.
    json!({"prompt":sample.prompt,"n_predict":policy.max_tokens(&sample.case.task),
        "stream":false,"temperature":0,"top_p":1,"repeat_penalty":1,
        "presence_penalty":0,"frequency_penalty":0,"seed":policy.generation_seed,
        "ignore_eos":false,"stop":[],"return_tokens":true,"cache_prompt":false})
}
pub fn observation(body: &Value, sample: &Sample, policy: &Policy) -> Result<Observation> {
    ensure!(
        body.get("error").is_none_or(Value::is_null)
            && body["stop"] == true
            && body["truncated"] == false,
        "reference failed, did not finish, or shifted/truncated context"
    );
    ensure!(
        body["tokens_evaluated"].as_u64() == Some(sample.prompt.len() as u64),
        "reference prompt usage differs from the supplied complete actual IDs"
    );
    let tokens = input::ids(&body["tokens"])?;
    let ceiling = policy.max_tokens(&sample.case.task) as usize;
    ensure!(
        !tokens.is_empty()
            && tokens.len() <= ceiling
            && body["tokens_predicted"].as_u64() == Some(tokens.len() as u64),
        "reference output IDs/usage differ or exceed the frozen ceiling"
    );
    let stop_type = body["stop_type"]
        .as_str()
        .context("reference stop type missing")?;
    let finish = match stop_type {
        "eos" => "stop",
        "word" => anyhow::bail!(
            "reference stopped on an unrequested stop word; frozen stop policy is empty"
        ),
        "limit" => {
            ensure!(
                tokens.len() == ceiling,
                "reference reached a different token limit"
            );
            "length"
        }
        _ => anyhow::bail!("unsupported reference stop type {stop_type}"),
    };
    let prompt_confirmed = if let Some(observed) = body.get("prompt_token_ids") {
        ensure!(
            input::ids(observed)? == sample.prompt,
            "reference admitted prompt IDs differ from supplied IDs"
        );
        true
    } else {
        false
    };
    Ok(Observation {
        text: body["content"]
            .as_str()
            .context("reference visible text missing")?
            .to_owned(),
        tokens,
        finish: finish.to_owned(),
        stop_type: stop_type.to_owned(),
        prompt_confirmed,
    })
}
pub async fn collect(
    client: &reqwest::Client,
    endpoint: &str,
    bytes: Vec<u8>,
    directory: &Path,
    sample: &Sample,
    policy: &Policy,
) -> Result<Observation> {
    ensure!(
        serde_json::from_slice::<Value>(&bytes)? == request(policy, sample),
        "reference request changed frozen IDs/policy"
    );
    let request_sha = sha(&bytes);
    let mut response = client
        .post(endpoint)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(bytes)
        .send()
        .await?;
    let status = response.status();
    let headers: std::collections::BTreeMap<_, _> = response
        .headers()
        .iter()
        .filter(|(name, _)| {
            matches!(
                name.as_str(),
                "content-type" | "x-request-id" | "request-id"
            )
        })
        .map(|(name, value)| {
            (
                name.to_string(),
                value.to_str().unwrap_or("<non-UTF8>").to_owned(),
            )
        })
        .collect();
    save(
        &directory.join("http.json"),
        &json!({"status":status.as_u16(),"headers":headers}),
    )?;
    let mut raw = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join("response.body"))?;
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        ensure!(
            bytes.len().saturating_add(chunk.len()) <= 8 * 1024 * 1024,
            "response exceeds 8 MiB; partial raw response retained"
        );
        raw.write_all(&chunk)?;
        bytes.extend_from_slice(&chunk);
    }
    raw.flush()?;
    ensure!(
        status.is_success(),
        "reference HTTP failure {}; raw body retained",
        status.as_u16()
    );
    let body: Value = serde_json::from_slice(&bytes).context("reference response is not JSON")?;
    // Preserve token evidence even when its count or format fails validation.
    save(
        &directory.join("tokens.json"),
        &json!({"source":"native_response.tokens",
        "token_ids":body.get("tokens"),"tokens_predicted":body.get("tokens_predicted"),
        "terminal_tokens_removed":false,"retokenized":false}),
    )?;
    if let Some(text) = body["content"].as_str() {
        write_new(&directory.join("visible-text.txt"), text.as_bytes())?;
    }
    save(
        &directory.join("prompt-tokens.json"),
        &json!({
            "source":"submitted_actual_strict_prompt_ids","source_request_id":sample.request_id,
            "submitted_token_ids":sample.prompt,"request_sha256":request_sha,
            "response_sha256":sha(&bytes),"tokens_evaluated":body.get("tokens_evaluated"),
            "response_prompt_token_ids":body.get("prompt_token_ids"),
            "retokenized":false,"scope":"supplied IDs are source-server observations; reference-side observation exists only if response_prompt_token_ids is present"
        }),
    )?;
    observation(&body, sample, policy)
}
