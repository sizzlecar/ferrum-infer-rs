use super::*;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    Math,
    Choice,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub id: String,
    pub stratum: String,
    pub source_record_index: u64,
    pub task: Task,
    pub prompt: String,
    pub prompt_sha256: String,
    pub public_input_tokens: usize,
}
#[derive(Deserialize)]
pub struct FilePin {
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Deserialize, Serialize)]
pub struct Policy {
    pub context_tokens: usize,
    pub template_reserve_tokens: usize,
    pub math_max_tokens: u32,
    pub choice_max_tokens: u32,
    pub generation_seed: u64,
    pub concurrency: usize,
    pub timeout_secs: u64,
}
impl Policy {
    pub fn max_tokens(&self, task: &Task) -> u32 {
        match task {
            Task::Math => self.math_max_tokens,
            Task::Choice => self.choice_max_tokens,
        }
    }
    fn fits(&self, count: usize, ceiling: u32) -> bool {
        count
            .checked_add(ceiling as usize)
            .is_some_and(|n| n <= self.context_tokens)
    }
}
#[derive(Deserialize)]
pub struct Execution {
    pub weights_sha256: String,
    pub model_alias: String,
    pub server_capacity: usize,
    pub strict_runtime_sha256: String,
    pub collector_binary_sha256: String,
}
/// Read the existing schema without adding a reference arm to it. Unused source,
/// tokenizer and bootstrap fields remain covered by the exact original plan SHA.
#[derive(Deserialize)]
pub struct Plan {
    schema: u32,
    frozen: bool,
    sampling: String,
    scoring: String,
    pub sample_count: usize,
    pub cases: FilePin,
    pub answers_sha256: String,
    pub policy: Policy,
    pub execution: Execution,
}
pub struct Sample {
    pub case: Case,
    pub prompt: Vec<u32>,
    pub request_id: String,
    pub prompt_dump_sha256: String,
    pub response_sha256: String,
    pub answer: String,
}
pub struct Source {
    pub plan: Plan,
    pub plan_bytes: Vec<u8>,
    pub manifest_sha256: String,
    pub samples: Vec<Sample>,
}
pub fn ids(value: &Value) -> Result<Vec<u32>> {
    value
        .as_array()
        .context("token IDs missing")?
        .iter()
        .map(|n| {
            u32::try_from(
                n.as_u64()
                    .context("token ID is not a nonnegative integer")?,
            )
            .context("token ID exceeds u32")
        })
        .collect()
}
pub fn hash_valid(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}
fn source_request(plan: &Plan, case: &Case) -> Value {
    json!({"model":plan.execution.model_alias,"messages":[{"role":"user","content":case.prompt}],
        "stream":false,"max_tokens":plan.policy.max_tokens(&case.task),"temperature":0,"top_p":1,
        "repetition_penalty":1,"repeat_penalty":1,"seed":plan.policy.generation_seed,
        "ignore_eos":false,"stop":[],"chat_template_kwargs":{"enable_thinking":false},
        "return_tokens":true,"verbose":true})
}
fn sample(
    root: &Path,
    index: usize,
    case: Case,
    plan: &Plan,
    final_result: &Value,
    answer: String,
) -> Result<Sample> {
    let root = root.join(format!("sample-{index:03}"));
    ensure!(
        read_json::<Case>(&root.join("source-case.json"))? == case,
        "source case mismatch at {index}"
    );
    let request: Value = read_json(&root.join("request.body"))?;
    ensure!(
        request == source_request(plan, &case)
            && request == read_json::<Value>(&root.join("request.json"))?,
        "source request differs from frozen prompt/policy at {index}"
    );
    let result: Value = read_json(&root.join("result.json"))?;
    ensure!(
        result == *final_result
            && result["success"] == true
            && result["sample_index"].as_u64() == Some(index as u64)
            && result["source_record_index"].as_u64() == Some(case.source_record_index),
        "source result mapping mismatch"
    );
    let response_bytes = read_bounded(&root.join("response.body"), 8 * 1024 * 1024)?;
    let response: Value = serde_json::from_slice(&response_bytes)?;
    let request_id = response["id"]
        .as_str()
        .filter(|s| !s.is_empty())
        .context("missing source response identity")?;
    ensure!(
        result["response_id"] == request_id && result["evidence"]["request_id"] == request_id,
        "source response identity mismatch"
    );
    let http: Value = read_json(&root.join("http.json"))?;
    ensure!(
        http["status"]
            .as_u64()
            .is_some_and(|s| (200..300).contains(&s)),
        "source HTTP failed"
    );
    let dump_bytes = read_bounded(&root.join("server-prompt-dump.json"), 4 * 1024 * 1024)?;
    let dump: Value = serde_json::from_slice(&dump_bytes)?;
    let prompt = ids(&dump["token_ids"])?;
    let saved: Value = read_json(&root.join("prompt-tokens.json"))?;
    ensure!(
        dump["request_id"] == request_id
            && dump.get("unavailable_reason").is_some_and(Value::is_null)
            && dump["token_count"].as_u64() == Some(prompt.len() as u64)
            && saved["source"] == "server_dump/prompt_token_ids.json"
            && saved["status"] == "captured"
            && saved["request_id"] == request_id
            && ids(&saved["token_ids"])? == prompt
            && response["usage"]["prompt_tokens"].as_u64() == Some(prompt.len() as u64),
        "source actual prompt evidence mismatch"
    );
    ensure!(
        !prompt.is_empty()
            && case
                .public_input_tokens
                .checked_add(plan.policy.template_reserve_tokens)
                .is_some_and(|n| prompt.len() <= n)
            && plan
                .policy
                .fits(prompt.len(), plan.policy.max_tokens(&case.task)),
        "source actual prompt exceeds frozen context/template budget"
    );
    Ok(Sample {
        case,
        prompt,
        request_id: request_id.to_owned(),
        prompt_dump_sha256: sha(&dump_bytes),
        response_sha256: sha(&response_bytes),
        answer,
    })
}
pub fn load(path: &Path, expected_sha: &str, strict: &Path, answers: &Path) -> Result<Source> {
    let plan_bytes = read_bounded(path, 64 * 1024 * 1024)?;
    ensure!(
        hash_valid(expected_sha) && sha(&plan_bytes) == expected_sha,
        "frozen plan SHA mismatch"
    );
    let plan: Plan = serde_json::from_slice(&plan_bytes)?;
    ensure!(
        plan.schema == 1 && plan.frozen && plan.sample_count > 0,
        "unsupported or unfrozen source plan"
    );
    ensure!(plan.sampling=="sha256_rank_without_replacement_v1; eligible_subject_proportional_largest_remainder; pooled_dispatch_hash_shuffle"
        && plan.scoring=="final_marker_exact_rational_v1; trimmed_uppercase_letter_v1; length_is_wrong; protocol_invalid_blocks_gate",
        "unsupported source sampling/scoring contract");
    ensure!(
        plan.policy.concurrency > 0
            && plan.policy.concurrency <= plan.execution.server_capacity
            && plan.policy.timeout_secs > 0
            && plan.policy.timeout_secs <= 3600
            && plan.policy.math_max_tokens >= 512
            && plan.policy.choice_max_tokens > 0,
        "invalid frozen policy"
    );
    let manifest_bytes = read_bounded(&strict.join("manifest.json"), 64 * 1024 * 1024)?;
    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    let original: Value = serde_json::from_slice(&plan_bytes)?;
    ensure!(
        manifest["schema"] == 1
            && manifest["arm"] == "strict"
            && manifest["execute"] == true
            && manifest["plan_sha256"] == expected_sha
            && manifest["plan"] == original
            && manifest["concurrency"].as_u64() == Some(plan.policy.concurrency as u64)
            && manifest["runtime_sha256"] == plan.execution.strict_runtime_sha256
            && manifest["collector_binary_sha256"] == plan.execution.collector_binary_sha256,
        "strict manifest differs from original frozen execution"
    );
    let case_bytes = read_bounded(&plan.cases.path, 64 * 1024 * 1024)?;
    ensure!(
        sha(&case_bytes) == plan.cases.sha256,
        "frozen cases SHA mismatch"
    );
    let cases: Vec<Case> = serde_json::from_slice(&case_bytes)?;
    let answer_bytes = read_bounded(answers, 64 * 1024 * 1024)?;
    ensure!(
        sha(&answer_bytes) == plan.answers_sha256,
        "frozen answers SHA mismatch"
    );
    #[derive(Deserialize)]
    struct Answer {
        id: String,
        answer: String,
    }
    let mut key = BTreeMap::new();
    for a in serde_json::from_slice::<Vec<Answer>>(&answer_bytes)? {
        ensure!(key.insert(a.id, a.answer).is_none(), "duplicate answer ID");
    }
    let results: Vec<Value> = read_json(&strict.join("results.json"))?;
    ensure!(
        cases.len() == plan.sample_count
            && key.len() == cases.len()
            && results.len() == cases.len(),
        "incomplete frozen cases, answers, or strict results"
    );
    let mut case_ids = BTreeSet::new();
    let mut response_ids = BTreeSet::new();
    let mut samples = Vec::new();
    for (index, case) in cases.into_iter().enumerate() {
        ensure!(
            case_ids.insert(case.id.clone())
                && !case.prompt.is_empty()
                && sha(case.prompt.as_bytes()) == case.prompt_sha256,
            "duplicate case or altered public prompt"
        );
        let answer = key.remove(&case.id).context("answer selection differs")?;
        let item = sample(strict, index, case, &plan, &results[index], answer)?;
        ensure!(
            response_ids.insert(item.request_id.clone()),
            "duplicate source response ID"
        );
        samples.push(item);
    }
    Ok(Source {
        plan,
        plan_bytes,
        manifest_sha256: sha(&manifest_bytes),
        samples,
    })
}

fn arg<'a>(cell: &'a Value, key: &str) -> Result<&'a str> {
    let args = cell["args"]
        .as_array()
        .context("guard server args missing")?;
    let positions: Vec<_> = args
        .iter()
        .enumerate()
        .filter(|(_, v)| v.as_str() == Some(key))
        .map(|(i, _)| i)
        .collect();
    ensure!(positions.len() == 1, "expected one guard argument {key}");
    args.get(positions[0] + 1)
        .and_then(Value::as_str)
        .context("guard argument value missing")
}
/// Server, environment, executable and asset pins stay bound. Client arguments
/// contain the resulting hash, so including them would create a hash cycle.
pub fn guard_runtime_pin(guard: &Value) -> Result<String> {
    let mut canonical = guard.clone();
    let cells = canonical["serving_cells"]
        .as_array_mut()
        .context("guard cells missing")?;
    ensure!(cells.len() == 1, "reference guard must contain one cell");
    for cell in cells {
        cell.as_object_mut()
            .context("guard cell is not an object")?
            .remove("client_args");
    }
    // Remain canonical even when another workspace crate enables preserve_order.
    canonical.sort_all_objects();
    Ok(sha(&serde_json::to_vec(&canonical)?))
}
pub fn validate_reference(
    guard: &Value,
    plan: &Plan,
    url: &reqwest::Url,
    collector_sha: &str,
) -> Result<String> {
    ensure!(
        url.scheme() == "http"
            && matches!(
                url.host_str(),
                Some("127.0.0.1" | "localhost" | "[::1]" | "::1")
            )
            && url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none()
            && matches!(url.path(), "" | "/"),
        "reference requires an HTTP loopback base URL"
    );
    let cells = guard["serving_cells"]
        .as_array()
        .context("guard cells missing")?;
    ensure!(cells.len() == 1, "reference guard must contain one cell");
    let c = &cells[0];
    ensure!(
        c["client_sha256"] == collector_sha && c["binary_sha256"].as_str().is_some_and(hash_valid),
        "reference executable pins missing or collector differs"
    );
    let slots = arg(c, "--parallel")?.parse::<usize>()?;
    let context = arg(c, "--ctx-size")?.parse::<usize>()?;
    ensure!(
        slots == plan.execution.server_capacity
            && slots >= plan.policy.concurrency
            && slots.checked_mul(plan.policy.context_tokens) == Some(context),
        "reference capacity/context differs from frozen execution"
    );
    if c["args"]
        .as_array()
        .unwrap()
        .iter()
        .any(|v| v == "--kv-unified-per-slot")
    {
        ensure!(
            arg(c, "--kv-unified-per-slot")?.parse::<usize>()? == plan.policy.context_tokens,
            "reference per-slot context differs"
        );
    }
    let port = arg(c, "--port")?.parse::<u16>()?;
    ensure!(
        port > 0
            && url.port_or_known_default() == Some(port)
            && c["port"].as_u64() == Some(u64::from(port))
            && arg(c, "--host")? == url.host_str().unwrap(),
        "reference endpoint differs from guard"
    );
    let model = arg(c, "--model")?;
    let pins = guard["input_files"]
        .as_array()
        .context("reference input pins missing")?;
    let model_pins: Vec<_> = pins.iter().filter(|v| v["path"] == model).collect();
    ensure!(
        hash_valid(&plan.execution.weights_sha256)
            && model_pins.len() == 1
            && model_pins[0]["sha256"] == plan.execution.weights_sha256,
        "reference does not pin the same GGUF bytes"
    );
    Ok(url.join("completion")?.to_string())
}
