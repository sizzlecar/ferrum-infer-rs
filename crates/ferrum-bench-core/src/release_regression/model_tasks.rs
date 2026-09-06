//! Match selected real-model tasks to the existing runner's terminal results.
//!
//! The runner owns the semantic oracles. This module detects missing execution,
//! failed checks and mismatched inputs; it neither replays raw responses nor
//! treats a declared target or a binary digest as numerical evidence. Success
//! here covers only the requested model runs, not installation or release approval.
use super::types::{Backend, ExecutionTarget, ModelProfile};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelCheck {
    Basic,
    Stop,
    Structured,
    Tools,
}

impl fmt::Display for ModelCheck {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Basic => "basic",
            Self::Stop => "stop",
            Self::Structured => "structured",
            Self::Tools => "tools",
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
        }
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

fn backend_name(backend: Backend) -> &'static str {
    match backend {
        Backend::Cpu => "cpu",
        Backend::Metal => "metal",
        Backend::Cuda => "cuda",
    }
}

fn run_backend(value: &Value) -> Option<Backend> {
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
    let checks: BTreeSet<_> = expected.checks.iter().copied().collect();
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
    for name in ["run-basic", "run-stop"] {
        if let Some(case) = recorded.get(name) {
            let ready = &case["evidence"]["ready"];
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
    if let Some(case) = recorded.get("serve-tools") {
        require(
            &mut errors,
            case["evidence"]["reasoning_alias_replayed"].as_bool()
                == Some(expected.reasoning_alias_replay),
            "serve-tools did not execute the expected reasoning alias replay mode",
        );
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
