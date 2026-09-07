//! Match selected real-model tasks to the existing runner's terminal results.
//!
//! The runner parses the actual product protocols. This module detects missing
//! execution and mismatched inputs, and rechecks the recorded behavior
//! observations, including available raw stop prefixes. Declared targets and
//! binary digests are not numerical evidence or release approval.
pub use super::model_stop_evidence::verify_stop_raw_observations;
use super::types::{Backend, ExecutionTarget, ModelProfile};
use ferrum_types::ModelReasoningProtocol;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

/// Workload text; stop boundaries come from actual observed output channels.
/// A reasoning stop never substitutes for the independently required final stop.
pub const DEFAULT_STOP_PROMPT: &str = "Write a short original paragraph about rain falling on a quiet street. Use your own wording and provide only the paragraph.";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelCheck {
    Basic,
    Stop,
    Structured,
    Tools,
    Reasoning,
    Length,
}

impl fmt::Display for ModelCheck {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Basic => "basic",
            Self::Stop => "stop",
            Self::Structured => "structured",
            Self::Tools => "tools",
            Self::Reasoning => "reasoning",
            Self::Length => "length",
        })
    }
}

impl FromStr for ModelCheck {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "basic" => Ok(Self::Basic),
            "stop" => Ok(Self::Stop),
            "structured" => Ok(Self::Structured),
            "tools" => Ok(Self::Tools),
            "reasoning" => Ok(Self::Reasoning),
            "length" => Ok(Self::Length),
            _ => Err(format!("unknown model check {value:?}")),
        }
    }
}

impl ModelCheck {
    fn cases(self) -> &'static [&'static str] {
        match self {
            Self::Basic => &["run-basic", "serve-basic"],
            Self::Stop => &["run-stop", "serve-stop"],
            Self::Structured => &["serve-structured"],
            Self::Tools => &["serve-tools"],
            Self::Reasoning => &["run-reasoning", "serve-reasoning"],
            Self::Length => &["run-length", "serve-length"],
        }
    }
}

/// Explicit capacity for a functional workload, separate from product defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRunCapacity {
    pub context_tokens: u32,
    pub max_num_seqs: u32,
}

/// Functional correctness workload policy; Quick Start retains product defaults.
pub const DEFAULT_FUNCTIONAL_CAPACITY: ModelRunCapacity = ModelRunCapacity {
    context_tokens: 2048,
    max_num_seqs: 1,
};

impl ModelRunCapacity {
    pub fn validate(self, max_tokens: u32) -> Result<(), String> {
        if self.max_num_seqs == 0 || max_tokens == 0 || max_tokens >= self.context_tokens {
            return Err("functional capacity must have positive concurrency and room for prompt plus output".into());
        }
        Ok(())
    }

    pub fn verify_health(self, health: &Value) -> Result<(), String> {
        let actual = &health["auto_config"];
        for field in ["selected_max_model_len", "selected_kv_capacity"] {
            if !actual[field]
                .as_u64()
                .is_some_and(|n| n == u64::from(self.context_tokens))
            {
                return Err(format!(
                    "runtime {field} differs from the declared functional context"
                ));
            }
        }
        if !actual["selected_max_sequences"]
            .as_u64()
            .is_some_and(|n| n > 0 && n <= u64::from(self.max_num_seqs))
        {
            return Err(
                "runtime concurrency exceeds or does not expose the functional capacity ceiling"
                    .into(),
            );
        }
        Ok(())
    }
}

/// Inputs fixed before the runner starts. The target is a declared execution
/// class; readiness observations independently check the selected backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedModelRun {
    pub profile: ModelProfile,
    pub binary_sha256: String,
    pub version: String,
    pub checks: Vec<ModelCheck>,
    pub disable_thinking: bool,
    pub use_default_backend: bool,
    pub max_tokens: u32,
    #[serde(default)]
    pub runtime_capacity: Option<ModelRunCapacity>,
    pub reasoning_alias_replay: bool,
    pub stop_prompt: String,
}

fn require(errors: &mut Vec<String>, condition: bool, message: impl Into<String>) {
    if !condition {
        errors.push(message.into());
    }
}

fn finish(errors: Vec<String>) -> Result<(), Vec<String>> {
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn nonblank(value: &str) -> bool {
    !value.trim().is_empty() && value == value.trim()
}

fn sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn digest_matches(value: &Value, expected: &str) -> bool {
    value
        .as_str()
        .is_some_and(|value| sha256(value) && value.eq_ignore_ascii_case(expected))
}

pub(super) fn backend_name(backend: Backend) -> &'static str {
    match backend {
        Backend::Cpu => "cpu",
        Backend::Metal => "metal",
        Backend::Cuda => "cuda",
    }
}

pub(super) fn run_backend(value: &Value) -> Option<Backend> {
    match value.as_str()? {
        "CPU" => Some(Backend::Cpu),
        "Metal" => Some(Backend::Metal),
        value => value
            .strip_prefix("CUDA(")
            .and_then(|value| value.strip_suffix(')'))
            .filter(|index| !index.is_empty() && index.bytes().all(|byte| byte.is_ascii_digit()))
            .and_then(|index| index.parse::<usize>().ok())
            .map(|_| Backend::Cuda),
    }
}

fn expected_errors(expected: &ExpectedModelRun) -> Vec<String> {
    let mut errors = Vec::new();
    for (field, value) in [
        ("profile.id", expected.profile.id.as_str()),
        ("profile.model", expected.profile.model.as_str()),
        (
            "target.architecture",
            expected.profile.target.architecture.as_str(),
        ),
        (
            "target.precision",
            expected.profile.target.precision.as_str(),
        ),
        (
            "target.execution_path",
            expected.profile.target.execution_path.as_str(),
        ),
    ] {
        require(
            &mut errors,
            nonblank(value),
            format!("expected {field} must be nonblank and unpadded"),
        );
    }
    require(
        &mut errors,
        sha256(&expected.binary_sha256),
        "expected binary_sha256 must be a SHA-256 digest",
    );
    require(
        &mut errors,
        semver::Version::parse(&expected.version).is_ok(),
        "expected version must be a semantic version",
    );
    require(
        &mut errors,
        expected.max_tokens > 0,
        "expected max_tokens must be positive",
    );
    require(
        &mut errors,
        !expected.stop_prompt.trim().is_empty(),
        "expected stop_prompt must be nonempty",
    );
    require(
        &mut errors,
        !expected.checks.is_empty(),
        "expected model checks must not be empty",
    );
    if let Some(capacity) = expected.runtime_capacity {
        if let Err(error) = capacity.validate(expected.max_tokens) {
            errors.push(error);
        }
    }
    let checks: BTreeSet<_> = expected.checks.iter().copied().collect();
    require(
        &mut errors,
        !checks.contains(&ModelCheck::Reasoning)
            || expected.profile.reasoning_protocol.supports_reasoning(),
        "reasoning positive check requires a known reasoning-capable profile",
    );
    require(
        &mut errors,
        checks.len() == expected.checks.len(),
        "expected model checks contain duplicates",
    );
    require(
        &mut errors,
        !expected.reasoning_alias_replay || checks.contains(&ModelCheck::Tools),
        "reasoning_alias_replay requires the tools check",
    );
    errors
}

/// Validate inputs before spawning the staged binary. Paths and descriptive
/// source/precision labels remain compatible with the existing runner; they do
/// not replace the selected profile or declared target.
pub fn verify_model_options(
    expected: &ExpectedModelRun,
    options: &Value,
) -> Result<(), Vec<String>> {
    let mut errors = expected_errors(expected);
    for (field, value) in [
        ("profile_id", expected.profile.id.as_str()),
        ("model", expected.profile.model.as_str()),
        ("backend", backend_name(expected.profile.target.backend)),
        ("stop_prompt", expected.stop_prompt.as_str()),
    ] {
        require(
            &mut errors,
            options[field].as_str() == Some(value),
            format!("options.{field} differs from the expected task"),
        );
    }
    for (field, value) in [
        ("disable_thinking", expected.disable_thinking),
        ("use_default_backend", expected.use_default_backend),
        ("reasoning_alias_replay", expected.reasoning_alias_replay),
    ] {
        require(
            &mut errors,
            options[field].as_bool() == Some(value),
            format!("options.{field} differs from the expected task"),
        );
    }
    require(
        &mut errors,
        options["max_tokens"].as_u64() == Some(u64::from(expected.max_tokens)),
        "options.max_tokens differs from the expected task",
    );
    for (field, value) in [
        (
            "context_tokens",
            expected.runtime_capacity.map(|c| c.context_tokens),
        ),
        (
            "max_num_seqs",
            expected.runtime_capacity.map(|c| c.max_num_seqs),
        ),
    ] {
        require(
            &mut errors,
            options[field] == serde_json::json!(value),
            format!("options.{field} differs from the expected runtime capacity"),
        );
    }
    match serde_json::from_value::<Vec<ModelCheck>>(options["checks"].clone()) {
        Ok(checks) => {
            let actual: BTreeSet<_> = checks.iter().copied().collect();
            let required: BTreeSet<_> = expected.checks.iter().copied().collect();
            require(
                &mut errors,
                checks.len() == actual.len(),
                "options.checks contains duplicates",
            );
            require(
                &mut errors,
                actual == required,
                "options.checks differs from the expected task",
            );
        }
        Err(error) => errors.push(format!("invalid options.checks: {error}")),
    }
    finish(errors)
}

/// Consume schema 2 produced by model_regression's actual case/oracle chain.
/// A terminal report cannot hide an omitted case or an independently failed case.
pub fn verify_model_report(expected: &ExpectedModelRun, report: &Value) -> Result<(), Vec<String>> {
    let mut errors = verify_model_options(expected, &report["options"])
        .err()
        .unwrap_or_default();
    require(
        &mut errors,
        report["schema_version"].as_u64() == Some(2),
        "model report must use schema_version 2",
    );
    require(
        &mut errors,
        report["profile_id"].as_str() == Some(expected.profile.id.as_str()),
        "report profile_id differs from the expected task",
    );
    let declared = serde_json::from_value::<ExecutionTarget>(report["target"].clone());
    require(
        &mut errors,
        declared
            .as_ref()
            .is_ok_and(|target| target == &expected.profile.target),
        "report declared target is missing or differs from the expected task",
    );
    require(
        &mut errors,
        report["status"] == "passed",
        "model report is not terminal passed",
    );
    require(
        &mut errors,
        digest_matches(&report["binary_sha256"], &expected.binary_sha256),
        "report binary_sha256 differs from the staged binary",
    );
    let environment_policy = if expected.use_default_backend {
        "remove_inherited_ferrum_overrides"
    } else {
        "inherit"
    };
    require(
        &mut errors,
        report["environment_policy"].as_str() == Some(environment_policy),
        "report environment policy differs from the expected invocation mode",
    );
    require(
        &mut errors,
        report["sampling"]["temperature"].as_f64() == Some(0.0)
            && report["sampling"]["seed"].as_u64() == Some(7)
            && report["sampling"]["max_tokens"].as_u64() == Some(u64::from(expected.max_tokens)),
        "report sampling differs from the runner's controlled inputs",
    );
    if let Some(unexecuted) = report.get("unexecuted_serve_checks") {
        require(
            &mut errors,
            unexecuted.as_array().is_some_and(Vec::is_empty),
            "model report contains unexecuted serve checks",
        );
    }
    if let Some(issues) = report.get("task_verification_errors") {
        require(
            &mut errors,
            issues.as_array().is_some_and(Vec::is_empty),
            "model report contains task verification errors",
        );
    }
    let mut required: BTreeSet<&str> = ["binary-version", "serve-startup", "binary-unchanged"]
        .into_iter()
        .collect();
    for check in &expected.checks {
        required.extend(check.cases().iter().copied());
    }
    let Some(cases) = report["cases"].as_array() else {
        errors.push("model report is missing its cases array".into());
        return Err(errors);
    };
    let mut recorded = BTreeMap::new();
    for case in cases {
        let Some(name) = case["case"].as_str() else {
            errors.push("model report contains a case without a name".into());
            continue;
        };
        require(
            &mut errors,
            required.contains(name),
            format!("unexpected model case {name}"),
        );
        require(
            &mut errors,
            recorded.insert(name, case).is_none(),
            format!("duplicate model case {name}"),
        );
        require(
            &mut errors,
            case["status"] == "passed",
            format!("model case {name} is not passed"),
        );
        require(
            &mut errors,
            case["error"].is_null(),
            format!("model case {name} contains an error"),
        );
        require(
            &mut errors,
            case["evidence"].is_object(),
            format!("model case {name} is missing oracle evidence"),
        );
    }
    for name in required {
        require(
            &mut errors,
            recorded.contains_key(name),
            format!("missing required model case {name}"),
        );
    }
    for name in ["run-basic", "serve-startup"] {
        if let Some(case) = recorded.get(name) {
            if let Err(error) = super::model_sources::verify_pinned_source(
                &expected.profile.model,
                &case["evidence"]["source_identity"],
            ) {
                errors.push(format!("{name}: {error}"));
            }
        }
    }
    if let Some(case) = recorded.get("binary-version") {
        let words: Vec<_> = case["evidence"]["version"]
            .as_str()
            .unwrap_or("")
            .split_whitespace()
            .collect();
        require(
            &mut errors,
            words == ["ferrum", expected.version.as_str()],
            "binary-version differs from the expected Ferrum version",
        );
    }
    if let Some(case) = recorded.get("binary-unchanged") {
        require(
            &mut errors,
            digest_matches(&case["evidence"]["sha256"], &expected.binary_sha256),
            "binary-unchanged does not match the staged binary",
        );
    }
    if let Some(case) = recorded.get("serve-startup") {
        let health = &case["evidence"];
        if let Some(capacity) = expected.runtime_capacity {
            if let Err(error) = capacity.verify_health(health) {
                errors.push(error);
            }
        }
        if expected.checks.contains(&ModelCheck::Reasoning)
            || (expected.profile.reasoning_protocol == ModelReasoningProtocol::None
                && expected.checks.contains(&ModelCheck::Basic))
        {
            if let Err(error) =
                verify_reasoning_identity(expected.profile.reasoning_protocol, health)
            {
                errors.push(format!("serve-startup: {error}"));
            }
        }
        require(
            &mut errors,
            health["status"] == "healthy",
            "serve-startup reports unhealthy or missing health status",
        );
        require(
            &mut errors,
            health["auto_config"]["hardware_capabilities"]["backend"].as_str()
                == Some(backend_name(expected.profile.target.backend)),
            "serve-startup backend is unobservable or differs from the expected backend",
        );
        require(
            &mut errors,
            health["version"].as_str() == Some(expected.version.as_str()),
            "serve-startup version differs from the expected version",
        );
    }
    for name in ["run-basic", "run-stop", "run-reasoning", "run-length"] {
        if let Some(case) = recorded.get(name) {
            let ready = &case["evidence"]["ready"];
            if name == "run-reasoning"
                || (name == "run-basic"
                    && expected.profile.reasoning_protocol == ModelReasoningProtocol::None)
            {
                if let Err(error) =
                    verify_reasoning_identity(expected.profile.reasoning_protocol, ready)
                {
                    errors.push(format!("{name}: {error}"));
                }
            }
            require(
                &mut errors,
                ready["event"] == "ready",
                format!("{name} is missing its ready observation"),
            );
            require(
                &mut errors,
                ready["requested_model"].as_str() == Some(expected.profile.model.as_str()),
                format!("{name} requested_model differs from the expected model"),
            );
            require(
                &mut errors,
                run_backend(&ready["backend"]) == Some(expected.profile.target.backend),
                format!("{name} backend is unobservable or differs from the expected backend"),
            );
        }
    }
    for name in ["run-stop", "serve-stop"] {
        if let Some(case) = recorded.get(name) {
            if let Err(error) = super::model_stop_evidence::verify_case(
                expected,
                &case["evidence"],
                name == "run-stop",
            ) {
                errors.push(format!("{name}: {error}"));
            }
        }
    }
    if let Some(case) = recorded.get("run-length") {
        let ready = &case["evidence"]["baseline_ready"];
        require(
            &mut errors,
            ready["event"] == "ready"
                && ready["requested_model"].as_str() == Some(expected.profile.model.as_str())
                && run_backend(&ready["backend"]) == Some(expected.profile.target.backend),
            "run-length baseline readiness differs from the selected model/backend",
        );
    }
    for name in ["run-basic", "serve-basic"] {
        if let Some(case) = recorded.get(name) {
            let verify = if name == "run-basic" {
                super::model_basic::verify_basic_run
            } else {
                super::model_basic::verify_basic_serve
            };
            if let Err(error) = verify(
                &case["evidence"],
                expected.profile.reasoning_protocol,
                expected.max_tokens,
            ) {
                errors.push(format!("{name}: {error}"));
            }
        }
    }
    for name in ["run-reasoning", "serve-reasoning"] {
        if let Some(case) = recorded.get(name) {
            let evidence = &case["evidence"];
            require(
                &mut errors,
                evidence["enable_thinking"] == true,
                format!("{name} did not explicitly enable thinking"),
            );
            require(
                &mut errors,
                evidence["max_tokens"].as_u64() == Some(u64::from(expected.max_tokens)),
                format!("{name} reasoning budget differs from task"),
            );
            let modes: &[&str] = if name.starts_with("run-") {
                &["output"]
            } else {
                &["sync", "stream"]
            };
            for mode in modes {
                require(
                    &mut errors,
                    evidence[*mode]["usage"]["completion_tokens"]
                        .as_u64()
                        .is_some_and(|n| n <= u64::from(expected.max_tokens)),
                    format!("{name}/{mode} exceeds task token budget"),
                );
                if let Err(error) = verify_reasoning_observation(&evidence[*mode]) {
                    errors.push(format!("{name}/{mode}: {error}"));
                }
            }
        }
    }
    for name in ["run-length", "serve-length"] {
        if let Some(case) = recorded.get(name) {
            let evidence = &case["evidence"];
            require(
                &mut errors,
                evidence["baseline_max_tokens"].as_u64() == Some(u64::from(expected.max_tokens))
                    && evidence["baseline"]["usage"]["completion_tokens"]
                        .as_u64()
                        .is_some_and(|n| n <= u64::from(expected.max_tokens)),
                format!("{name} baseline budget differs from task"),
            );
            let budget = evidence["budget"]
                .as_u64()
                .and_then(|value| u32::try_from(value).ok());
            require(
                &mut errors,
                evidence["enable_thinking"] == false,
                format!("{name} is missing its explicit truncation configuration"),
            );
            let modes: &[&str] = if name.starts_with("run-") {
                &["output"]
            } else {
                &["sync", "stream"]
            };
            for mode in modes {
                let result = budget
                    .ok_or_else(|| "missing truncation budget".to_owned())
                    .and_then(|budget| {
                        verify_length_observations(&evidence["baseline"], &evidence[*mode], budget)
                    });
                if let Err(error) = result {
                    errors.push(format!("{name}/{mode}: {error}"));
                }
            }
        }
    }
    if let Some(case) = recorded.get("serve-tools") {
        if let Err(error) = super::model_tool::verify_tool_case(
            &case["evidence"],
            expected.max_tokens,
            expected.reasoning_alias_replay,
        ) {
            errors.push(format!("serve-tools: {error}"));
        }
    }
    finish(errors)
}

/// Each expected profile must have exactly one matching terminal report. This
/// entrypoint is for model-runtime verification; an empty model schedule belongs
/// to the caller's explicit no-model-work path, not a vacuous runtime success.
pub fn verify_model_reports(
    expected: &[ExpectedModelRun],
    reports: &[Value],
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    require(
        &mut errors,
        !expected.is_empty(),
        "expected model runs must not be empty",
    );
    let mut tasks = BTreeMap::new();
    for task in expected {
        require(
            &mut errors,
            tasks.insert(task.profile.id.as_str(), task).is_none(),
            format!("duplicate expected profile {}", task.profile.id),
        );
    }
    let mut observed = BTreeSet::new();
    for report in reports {
        let Some(id) = report["profile_id"].as_str() else {
            errors.push("model report is missing profile_id".into());
            continue;
        };
        require(
            &mut errors,
            observed.insert(id),
            format!("duplicate report for profile {id}"),
        );
        match tasks.get(id) {
            Some(task) => {
                if let Err(reasons) = verify_model_report(task, report) {
                    errors.extend(
                        reasons
                            .into_iter()
                            .map(|reason| format!("profile {id}: {reason}")),
                    );
                }
            }
            None => errors.push(format!("unexpected report for profile {id}")),
        }
    }
    for id in tasks.keys() {
        require(
            &mut errors,
            observed.contains(id),
            format!("missing report for profile {id}"),
        );
    }
    finish(errors)
}

#[cfg(test)]
#[path = "model_tasks_tests.rs"]
mod tests;

/// Small observation oracles shared by the actual runner and its report consumer.
/// Wire framing is checked by the runner before these observations are produced.
/// These functions do not turn a report into a proof of arbitrary model behavior.
pub fn probe_answer_matches(text: &str, expected: &str) -> bool {
    fn trim_markup(text: &str) -> &str {
        text.trim()
            .trim_matches(|c| matches!(c, '`' | '*' | '"' | '\''))
            .trim()
    }
    let text = trim_markup(text);
    // A trailing sentence period is cosmetic; a leading decimal point changes
    // the numeric answer and must never be discarded.
    trim_markup(text.strip_suffix('.').unwrap_or(text)) == expected
}

pub(super) fn boundary_observation(observation: &Value) -> Result<(&str, &str, u64), String> {
    let message = &observation["message"];
    let content = message["content"]
        .as_str()
        .ok_or("missing visible content")?;
    let reasoning = match message.get("reasoning") {
        None | Some(Value::Null) => "",
        Some(value) => value.as_str().ok_or("invalid reasoning type")?,
    };
    if message["role"] != "assistant"
        || message.get("reasoning_content").is_some()
        || !(message["tool_calls"].is_null()
            || message["tool_calls"].as_array().is_some_and(Vec::is_empty))
    {
        return Err("invalid canonical assistant text observation".into());
    }
    // These probes never ask for literal control syntax. Raw JSONL evidence may
    // retain it; parsed content/reasoning must not expose full or partial frames.
    for text in [content, reasoning] {
        if ["<|", "|>", "<think", "</think", "<channel|"]
            .iter()
            .any(|marker| text.contains(marker))
        {
            return Err("parsed text leaked protocol framing".into());
        }
    }
    if content.contains(['<', '>']) {
        return Err("parsed visible probe text leaked a partial control delimiter".into());
    }
    let usage = &observation["usage"];
    let prompt = usage["prompt_tokens"]
        .as_u64()
        .ok_or("missing prompt usage")?;
    let completion = usage["completion_tokens"]
        .as_u64()
        .ok_or("missing completion usage")?;
    let total = prompt
        .checked_add(completion)
        .ok_or("token usage overflows")?;
    if prompt == 0 || completion == 0 || Some(total) != usage["total_tokens"].as_u64() {
        return Err("invalid boundary observation token usage".into());
    }
    Ok((content, reasoning, completion))
}

/// Compare a prepared declaration with the actual loaded template/protocol.
pub fn verify_reasoning_identity(
    expected: ModelReasoningProtocol,
    observation: &Value,
) -> Result<(), String> {
    let actual =
        serde_json::from_value::<ModelReasoningProtocol>(observation["reasoning_protocol"].clone())
            .map_err(|_| "missing or invalid actual reasoning capability".to_string())?;
    if expected == ModelReasoningProtocol::Unknown || actual != expected {
        return Err(format!(
            "loaded reasoning capability {actual:?} differs from known declaration {expected:?}"
        ));
    }
    Ok(())
}

/// Absence is a separate assertion, never a successful enabled-thinking probe.
pub fn verify_reasoning_absence_observation(
    observation: &Value,
    answer: &str,
) -> Result<(), String> {
    let (content, reasoning, _) = boundary_observation(observation)?;
    if !matches!(observation["finish_reason"].as_str(), Some("stop" | "eos"))
        || !reasoning.is_empty()
        || content.trim() != answer
    {
        return Err(
            "non-thinking observation has reasoning, an incorrect answer or an unnatural finish"
                .into(),
        );
    }
    Ok(())
}

/// Enabled-thinking integration requires an actual distinct thought and a clean
/// final answer; nullable/empty reasoning is explicitly unsupported, not success.
pub fn verify_reasoning_observation(observation: &Value) -> Result<(), String> {
    let (content, reasoning, _) = boundary_observation(observation)?;
    if !matches!(observation["finish_reason"].as_str(), Some("stop" | "eos")) {
        return Err("reasoning probe did not finish naturally".into());
    }
    if reasoning.trim().is_empty() || reasoning.trim() == "42" {
        return Err("unsupported reasoning probe: no distinct nonempty thought observed".into());
    }
    if content.trim() != "42" {
        return Err(
            "reasoning leaked into visible output or final arithmetic answer was wrong".into(),
        );
    }
    Ok(())
}

/// Leave room before a naturally observed ending, rather than blindly cutting
/// the first protocol header token. The replay must still prove a proper visible
/// prefix: token counts alone cannot establish where a model's header ends.
pub fn length_probe_budget(natural_completion_tokens: u64) -> Result<u32, String> {
    natural_completion_tokens
        .checked_sub(2)
        .filter(|budget| *budget > 0)
        .and_then(|budget| u32::try_from(budget).ok())
        .ok_or_else(|| {
            "unsupported length probe: natural baseline too short or budget unrepresentable".into()
        })
}

pub fn verify_length_observations(
    baseline: &Value,
    truncated: &Value,
    budget: u32,
) -> Result<(), String> {
    let (full, _, natural_tokens) = boundary_observation(baseline)?;
    let (partial, _, actual_tokens) = boundary_observation(truncated)?;
    if !matches!(baseline["finish_reason"].as_str(), Some("stop" | "eos")) {
        return Err("length baseline did not finish naturally".into());
    }
    if length_probe_budget(natural_tokens)? != budget
        || truncated["finish_reason"] != "length"
        || actual_tokens != u64::from(budget)
    {
        return Err(
            "length finish or observed completion usage differs from the derived budget".into(),
        );
    }
    let full = full.trim();
    let partial = partial.trim();
    if partial.is_empty() || partial.len() >= full.len() || !full.starts_with(partial) {
        return Err(
            "unsupported length probe: no safe nonempty proper visible baseline prefix observed"
                .into(),
        );
    }
    Ok(())
}
