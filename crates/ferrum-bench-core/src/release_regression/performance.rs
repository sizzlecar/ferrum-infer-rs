//! Registration and path-independent inputs for the implemented same-host
//! legacy Metal HTTP latency comparison. A scheduled task is not a measured pass.
use super::types::*;
use ferrum_types::ModelOutputProtocol;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const PERFORMANCE_CHECK_PREFIX: &str = "release-performance.metal-legacy-http";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Workload {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub measured_requests: u32,
    pub warmup_requests: u32,
    pub repeats: u32,
    pub seed: u64,
    pub max_model_len: u32,
}
impl Workload {
    pub fn validate(&self) -> Result<(), String> {
        if self.input_tokens == 0
            || self.output_tokens < 2
            || self.measured_requests == 0
            || self.warmup_requests == 0
            || self.repeats < 3
            || self
                .input_tokens
                .checked_add(self.output_tokens)
                .is_none_or(|n| n >= self.max_model_len)
            || self
                .measured_requests
                .checked_add(self.warmup_requests)
                .and_then(|n| n.checked_mul(self.repeats))
                .is_none()
        {
            return Err("invalid performance workload, repetitions or context headroom".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub ttft_max_relative_increase: f64,
    pub tpot_max_relative_increase: f64,
}
impl Limits {
    pub fn validate(&self) -> Result<(), String> {
        if [
            self.ttft_max_relative_increase,
            self.tpot_max_relative_increase,
        ]
        .into_iter()
        .any(|limit| !limit.is_finite() || limit <= 0.0)
        {
            return Err("performance limits must be finite and positive".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerformancePolicy {
    pub workload: Workload,
    pub limits: Limits,
    pub runtime_memory_budget_bytes: u64,
    pub startup_timeout_secs: u64,
    pub request_timeout_secs: u64,
    pub task_timeout_secs: u64,
}
impl PerformancePolicy {
    pub fn validate(&self) -> Result<(), String> {
        self.workload.validate()?;
        self.limits.validate()?;
        if self.runtime_memory_budget_bytes == 0
            || self.startup_timeout_secs == 0
            || self.request_timeout_secs == 0
            || self.task_timeout_secs == 0
        {
            return Err("performance memory ceiling and time budgets must be positive".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileDigest {
    pub bytes: u64,
    pub sha256: String,
}
impl FileDigest {
    pub fn validate(&self) -> Result<(), String> {
        if self.bytes == 0 || !valid_hash(&self.sha256) {
            return Err(
                "performance source requires nonempty files and valid SHA-256 digests".into(),
            );
        }
        Ok(())
    }
}

/// Relative metadata names and digests; no worker-local path is needed for replay.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceIdentity {
    pub gguf: FileDigest,
    pub sidecars: BTreeMap<String, FileDigest>,
}
impl SourceIdentity {
    pub fn validate(&self) -> Result<(), String> {
        self.gguf.validate()?;
        for (name, digest) in &self.sidecars {
            if ![
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "chat_template.json",
                "chat_template.jinja",
                "generation_config.json",
            ]
            .contains(&name.as_str())
            {
                return Err(format!(
                    "unsupported performance source metadata name: {name}"
                ));
            }
            digest.validate()?;
        }
        if !self.sidecars.contains_key("tokenizer.json")
            || !self.sidecars.contains_key("tokenizer_config.json")
        {
            return Err(
                "performance source must pin tokenizer.json and tokenizer_config.json".into(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedPerformanceRun {
    pub schema_version: u32,
    pub profile: ModelProfile,
    pub baseline_version: String,
    pub baseline_sha256: String,
    pub candidate_version: String,
    pub candidate_sha256: String,
    pub client_version: String,
    pub client_sha256: String,
    pub policy: PerformancePolicy,
    pub source: SourceIdentity,
    pub obligations: Vec<usize>,
}
impl ExpectedPerformanceRun {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 || !supported_profile(&self.profile) {
            return Err(
                "performance task requires an available declared legacy Metal GGUF/Text profile"
                    .into(),
            );
        }
        for version in [
            &self.baseline_version,
            &self.candidate_version,
            &self.client_version,
        ] {
            let parsed =
                semver::Version::parse(version).map_err(|e| format!("performance version: {e}"))?;
            if parsed.to_string() != *version || !parsed.pre.is_empty() || !parsed.build.is_empty()
            {
                return Err("performance binary versions must be formal semantic versions".into());
            }
        }
        if [
            &self.baseline_sha256,
            &self.candidate_sha256,
            &self.client_sha256,
        ]
        .into_iter()
        .any(|hash| !valid_hash(hash))
        {
            return Err("performance binary SHA-256 digest is invalid".into());
        }
        if self.obligations.is_empty()
            || self.obligations.iter().collect::<BTreeSet<_>>().len() != self.obligations.len()
        {
            return Err("performance task must bind distinct plan obligations".into());
        }
        self.policy.validate()?;
        self.source.validate()
    }
}
fn valid_hash(hash: &str) -> bool {
    hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit())
}
fn concrete(value: &str) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && !value.eq_ignore_ascii_case("model-native")
        && !value
            .split(|c: char| !c.is_ascii_alphanumeric())
            .any(|part| {
                [
                    "unknown",
                    "unresolved",
                    "unspecified",
                    "placeholder",
                    "tbd",
                    "todo",
                    "pending",
                ]
                .iter()
                .any(|unknown| part.eq_ignore_ascii_case(unknown))
            })
}
/// This is the implemented runner's reach, not a claim that these targets ran.
pub fn supports_target(target: &ExecutionTarget) -> bool {
    target.backend == Backend::Metal
        && target.execution_path == "legacy-model-executor"
        && target.protocol == ModelOutputProtocol::Text
        && concrete(&target.architecture)
        && concrete(&target.precision)
        && target
            .precision
            .strip_prefix("gguf-")
            .is_some_and(|suffix| !suffix.is_empty())
}
fn supported_profile(profile: &ModelProfile) -> bool {
    profile.available
        && !profile.id.trim().is_empty()
        && !profile.model.trim().is_empty()
        && supports_target(&profile.target)
}

fn descriptor(target: &ExecutionTarget) -> Option<CheckDescriptor> {
    if !supports_target(target) {
        return None;
    }
    // The digest only makes a stable unique name for this complete target. It
    // is not an execution result or a source/performance correctness criterion.
    let identity =
        serde_json::to_vec(target).expect("execution target contains serializable fields");
    Some(CheckDescriptor {
        id: format!("{PERFORMANCE_CHECK_PREFIX}.{:x}", Sha256::digest(identity)),
        behavior: Behavior::Performance,
        layer: EvidenceLayer::Performance,
        entrypoints: vec![Entrypoint::ServeStream],
        target: Some(target.clone()),
    })
}

pub fn performance_check_descriptors(targets: &[ExecutionTarget]) -> Vec<CheckDescriptor> {
    targets
        .iter()
        .filter_map(descriptor)
        .map(|check| (check.id.clone(), check))
        .collect::<BTreeMap<_, _>>()
        .into_values()
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerformanceRunRequirements {
    pub profile: ModelProfile,
    pub obligations: Vec<usize>,
}
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerformanceTaskSchedule {
    pub runs: Vec<PerformanceRunRequirements>,
    pub unsupported_obligations: Vec<usize>,
}

/// Transfer only known, uniquely assigned SSE performance obligations to the
/// separate runner. It cannot substitute Basic or remove existing plan gaps.
pub fn performance_task_schedule(plan: &Plan) -> PerformanceTaskSchedule {
    let mut runs = BTreeMap::<String, PerformanceRunRequirements>::new();
    let mut unsupported_obligations = Vec::new();
    for (index, obligation) in plan.obligations.iter().enumerate() {
        if obligation.layer != EvidenceLayer::Performance {
            continue;
        }
        let mut owners = plan
            .selected
            .iter()
            .filter(|owner| owner.obligations.contains(&index));
        let usable = owners.next().filter(|owner| {
            owners.next().is_none()
                && obligation.behavior == Behavior::Performance
                && supported_profile(&owner.profile)
                && super::model_schedule::scope_matches(&obligation.scope, &owner.profile)
                && obligation.entrypoints == [Entrypoint::ServeStream]
                && descriptor(&owner.profile.target)
                    .is_some_and(|check| obligation.checkers.contains(&check.id))
                && runs
                    .get(&owner.profile.id)
                    .is_none_or(|run| run.profile == owner.profile)
        });
        let Some(owner) = usable else {
            unsupported_obligations.push(index);
            continue;
        };
        runs.entry(owner.profile.id.clone())
            .or_insert_with(|| PerformanceRunRequirements {
                profile: owner.profile.clone(),
                obligations: Vec::new(),
            })
            .obligations
            .push(index);
    }
    PerformanceTaskSchedule {
        runs: runs.into_values().collect(),
        unsupported_obligations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn profile() -> ModelProfile {
        ModelProfile {
            gguf: None,
            reasoning_protocol: ferrum_types::ModelReasoningProtocol::None,
            id: "declared-dense".into(),
            model: "fixture:gguf-q4".into(),
            available: true,
            estimate: None,
            target: ExecutionTarget {
                architecture: "dense".into(),
                protocol: ModelOutputProtocol::Text,
                precision: "gguf-q4_k_m".into(),
                backend: Backend::Metal,
                execution_path: "legacy-model-executor".into(),
            },
        }
    }
    fn expected() -> ExpectedPerformanceRun {
        ExpectedPerformanceRun {
            schema_version: 1,
            profile: profile(),
            baseline_version: "1.2.3".into(),
            candidate_version: "1.2.4".into(),
            client_version: "1.2.4".into(),
            baseline_sha256: "a".repeat(64),
            candidate_sha256: "b".repeat(64),
            client_sha256: "c".repeat(64),
            policy: PerformancePolicy {
                workload: Workload {
                    input_tokens: 32,
                    output_tokens: 16,
                    measured_requests: 3,
                    warmup_requests: 1,
                    repeats: 3,
                    seed: 7,
                    max_model_len: 128,
                },
                limits: Limits {
                    ttft_max_relative_increase: 0.1,
                    tpot_max_relative_increase: 0.1,
                },
                runtime_memory_budget_bytes: 1024,
                startup_timeout_secs: 60,
                request_timeout_secs: 30,
                task_timeout_secs: 600,
            },
            source: SourceIdentity {
                gguf: FileDigest {
                    bytes: 1024,
                    sha256: "d".repeat(64),
                },
                sidecars: [
                    (
                        "tokenizer.json".into(),
                        FileDigest {
                            bytes: 32,
                            sha256: "e".repeat(64),
                        },
                    ),
                    (
                        "tokenizer_config.json".into(),
                        FileDigest {
                            bytes: 32,
                            sha256: "f".repeat(64),
                        },
                    ),
                ]
                .into(),
            },
            obligations: vec![2, 5],
        }
    }
    #[test]
    fn expected_task_has_path_independent_pins_and_no_completed_evidence() {
        let expected = expected();
        expected.validate().unwrap();
        let bytes = serde_json::to_vec(&expected).unwrap();
        let replay: ExpectedPerformanceRun = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(replay, expected);
        replay.validate().unwrap();
    }
    #[test]
    fn descriptor_only_registers_actual_metal_legacy_gguf_text_inventory() {
        let base = profile().target;
        let mut candidates = vec![base.clone()];
        for target in [
            ExecutionTarget {
                backend: Backend::Cuda,
                ..base.clone()
            },
            ExecutionTarget {
                execution_path: "production-plan-runtime".into(),
                ..base.clone()
            },
            ExecutionTarget {
                protocol: ModelOutputProtocol::HarmonyGptOss,
                ..base.clone()
            },
            ExecutionTarget {
                precision: "bf16".into(),
                ..base.clone()
            },
            ExecutionTarget {
                precision: "gguf-unknown".into(),
                ..base.clone()
            },
            ExecutionTarget {
                precision: "gguf-".into(),
                ..base.clone()
            },
            ExecutionTarget {
                architecture: "unknown".into(),
                ..base.clone()
            },
        ] {
            assert!(!supports_target(&target));
            candidates.push(target);
        }
        let result = performance_check_descriptors(&candidates);
        assert_eq!(result, performance_check_descriptors(&[base.clone(), base]));
        assert!(performance_check_descriptors(&[]).is_empty());
        assert_eq!(result[0].behavior, Behavior::Performance);
        assert_eq!(result[0].entrypoints, [Entrypoint::ServeStream]);
    }
    #[test]
    fn task_rejects_invalid_policy_binary_pins_and_unsupported_targets_before_execution() {
        let mut invalid = Vec::new();
        let mut value = expected();
        value.policy.workload.warmup_requests = 0;
        invalid.push(value);
        let mut value = expected();
        value.policy.workload.repeats = 2;
        invalid.push(value);
        let mut value = expected();
        value.policy.workload.max_model_len = 48;
        invalid.push(value);
        let mut value = expected();
        value.policy.workload.measured_requests = u32::MAX;
        invalid.push(value);
        let mut value = expected();
        value.policy.runtime_memory_budget_bytes = 0;
        invalid.push(value);
        let mut value = expected();
        value.policy.request_timeout_secs = 0;
        invalid.push(value);
        for limit in [0.0, -0.1, f64::NAN, f64::INFINITY] {
            let mut value = expected();
            value.policy.limits.ttft_max_relative_increase = limit;
            invalid.push(value);
        }
        let mut value = expected();
        value.candidate_sha256 = "not-a-digest".into();
        invalid.push(value);
        let mut value = expected();
        value.baseline_version = "1.2.3-rc.1".into();
        invalid.push(value);
        let mut value = expected();
        value.obligations.clear();
        invalid.push(value);
        let mut value = expected();
        value.obligations = vec![2, 2];
        invalid.push(value);
        let mut value = expected();
        value.profile.target.backend = Backend::Cuda;
        invalid.push(value);
        for value in invalid {
            assert!(value.validate().is_err(), "accepted {value:?}");
        }
    }
    #[test]
    fn source_requires_complete_registered_metadata_without_worker_local_paths() {
        let mut source = expected().source;
        source.sidecars.remove("tokenizer.json");
        assert!(source.validate().is_err());
        let mut source = expected().source;
        source
            .sidecars
            .insert("../tokenizer.json".into(), source.gguf.clone());
        assert!(source.validate().is_err());
        let mut source = expected().source;
        source.gguf.bytes = 0;
        assert!(source.validate().is_err());
        let mut source = expected().source;
        source.gguf.sha256 = "g".repeat(64);
        assert!(source.validate().is_err());
    }
}
