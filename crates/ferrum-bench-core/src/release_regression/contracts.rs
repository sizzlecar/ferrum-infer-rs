//! CPU contract checks execute registered Rust assertions from Cargo artifacts.
//! Registration protects against a removed/renamed/ignored test silently becoming
//! an empty filtered run. Exit status, not libtest PASS text, decides execution.
//! These are deterministic boundary samples, not GPU arithmetic or real-model proof.
use super::types::{Behavior, CheckDescriptor, Entrypoint, EvidenceLayer};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContractTest {
    pub package: String,
    pub target: String,
    pub kind: String,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContractGroup {
    pub id: String,
    pub behavior: Behavior,
    pub entrypoints: Vec<Entrypoint>,
    pub tests: Vec<ContractTest>,
}

fn lib(package: &str, name: &str) -> ContractTest {
    ContractTest {
        package: package.into(),
        target: package.replace('-', "_"),
        kind: "lib".into(),
        name: name.into(),
    }
}
fn integration(package: &str, target: &str, name: &str) -> ContractTest {
    ContractTest {
        package: package.into(),
        target: target.into(),
        kind: "test".into(),
        name: name.into(),
    }
}

/// Each group names actual semantic assertions. Adding a new behavior requires
/// its real test bindings; nearby tests do not imply unreviewed branch coverage.
/// Protocol-specific probes below cover Text/Harmony/Gemma where listed. They
/// do not certify every model template, architecture, accelerator or future protocol.
pub fn contract_groups() -> Vec<ContractGroup> {
    use Behavior::*;
    let download = |name: &str| {
        lib(
            "ferrum-models",
            &format!("hf_download::download_tests::{name}"),
        )
    };
    let engine = |name: &str| {
        lib(
            "ferrum-engine",
            &format!("continuous_engine::tests::{name}"),
        )
    };
    let server = |name: &str| lib("ferrum-server", &format!("axum_server::tests::{name}"));
    let tiny = |name: &str| integration("ferrum-engine", "tiny_stack", name);
    let scheduler = |name: &str| lib("ferrum-scheduler", &format!("vnext::tests::{name}"));
    let kv = |name: &str| lib("ferrum-kv", &format!("managers::paged::tests::{name}"));
    let all = vec![
        Entrypoint::Run,
        Entrypoint::ServeSync,
        Entrypoint::ServeStream,
    ];
    let http = vec![Entrypoint::ServeSync, Entrypoint::ServeStream];
    let specifications = vec![
        ("source-closure", SourceClosure, all.clone(), vec![download("indexed_fresh_download_fetches_only_referenced_shards_and_sidecars"), download("fresh_download_preserves_standalone_chat_template_in_source_bundle")]),
        ("source-revision", SourceRevision, all.clone(), vec![download("indexed_download_uses_resolved_snapshot_when_main_moves")]),
        ("download-recovery", DownloadRecovery, all.clone(), vec![download("indexed_transfer_failure_preserves_ref_and_retry_completes"), download("failed_template_download_does_not_publish_ref_and_can_retry")]),
        ("cache-completeness", CacheCompleteness, all.clone(), vec![download("indexed_invalid_index_stops_before_weights_and_preserves_ref")]),
        ("template-history", TemplateHistory, all.clone(), vec![integration("ferrum-server", "chat_template_golden", "chat_template_goldens_match_transformers"), tiny("tiny_stack_multi_turn_five_rounds")]),
        ("protocol-framing", ProtocolFraming, all.clone(), vec![tiny("tiny_stack_stream_chunk_contract"), integration("ferrum-cli", "download_jsonl", "cold_and_blob_cached_safetensors_download_preserves_jsonl_stdout"), integration("ferrum-cli", "download_jsonl", "cold_and_blob_cached_gguf_alias_download_preserves_jsonl_stdout"), server("engine_stop_contract::wire_stop_halts_the_engine_before_the_natural_terminal")]),
        ("reasoning-boundaries", ReasoningBoundaries, all.clone(), vec![lib("ferrum-types", "reasoning::tests::prompt_opened_thinking_keeps_closed_boundary_across_stream_prefixes"), lib("ferrum-types", "reasoning::gemma::tests::every_chunk_split_preserves_visible_and_reasoning_prefixes"), lib("ferrum-types", "reasoning::gemma::tests::incomplete_headers_and_closing_markers_never_become_content"), lib("ferrum-types", "harmony::tests::parses_analysis_then_final_response")]),
        ("user-stop", UserStop, all.clone(), vec![engine("stop_boundary_tests::stop_boundary_flushes_and_selects_the_earliest_match"), server("engine_stop_contract::wire_stop_halts_the_engine_before_the_natural_terminal")]),
        ("natural-end", NaturalEnd, all.clone(), vec![engine("sequence_stop_reason_distinguishes_user_stop_from_natural_eos"), server("engine_stop_contract::natural_eos_cannot_impersonate_a_matched_wire_stop"), tiny("tiny_stack_eos_terminates")]),
        ("length-limit", LengthLimit, all.clone(), vec![tiny("tiny_stack_repetition_runaway_guard"), lib("ferrum-types", "harmony::tests::user_or_length_truncation_keeps_tool_calls_and_control_markers_fail_closed"), engine("stop_boundary_tests::stop_boundary_flushes_and_selects_the_earliest_match")]),
        // These production-engine fixtures choose between competing logits and
        // inspect the selected token as the next decode input, not just valid JSON.
        ("structured-sampling", StructuredSampling, http.clone(), vec![server("engine_stop_contract::structured::wire_schema_changes_actual_text_sampling"), server("engine_stop_contract::structured::wire_schema_preserves_harmony_framing_before_constraining_payload")]),
        ("structured-validity", StructuredValidity, http.clone(), vec![server("engine_stop_contract::structured::wire_schema_changes_actual_text_sampling"), server("engine_stop_contract::structured::wire_schema_preserves_harmony_framing_before_constraining_payload")]),
        ("tool-selection", ToolSelection, http.clone(), vec![server("engine_stop_contract::tools::harmony_function_handoff_rejects_a_different_named_choice"), server("engine_stop_contract::tools::harmony_final_json_cannot_impersonate_a_required_tool_call"), server("engine_stop_contract::tools::harmony_tool_choice_none_rejects_native_call_sync"), server("engine_stop_contract::tools::harmony_tool_choice_none_rejects_native_call_sse")]),
        ("tool-handoff", ToolHandoff, http.clone(), vec![server("engine_stop_contract::tools::harmony_function_handoff_reaches_sync_and_sse")]),
        // Both routes preserve returned call IDs, arguments and tool results in
        // the next engine request. The template case includes out-of-order
        // results for two calls to the same function; no model semantics are inferred.
        ("tool-continuation", ToolContinuation, http, vec![server("route_tool_request_reaches_engine_structured_boundary"), server("route_tool_request_prefers_model_chat_template")]),
        ("scheduling-progress", SchedulingProgress, all.clone(), vec![scheduler("deferred_head_does_not_block_an_eligible_smaller_request"), scheduler("release_epoch_wakes_and_admits_a_deferred_request"), scheduler("unchanged_evidence_suppresses_blind_retries")]),
        ("cancellation", Cancellation, all.clone(), vec![scheduler("cancellation_returns_the_exact_waiting_request"), engine("plan_runtime_capacity_wait_wakes_and_cancels_when_stream_is_dropped"), engine("plan_runtime_capacity_wait_wakes_and_cancels_when_sync_future_is_aborted")]),
        ("capacity-admission", CapacityAdmission, all.clone(), vec![scheduler("permanent_rejection_and_fault_leave_no_waiting_ownership"), kv("failed_allocate_rolls_back_partial_blocks"), kv("failed_extend_rolls_back_partial_blocks_and_handle_table"), engine("explicit_request_budget_accepts_exact_capacity_and_rejects_one_token_over"), lib("ferrum-cli", "commands::run::tests::no_context_shift_preserves_history_at_capacity_and_rejects_overflow")]),
        // CPU tiny forward and real paged-manager rollback. The tiny engine's
        // allocation handles are mocks; this does not certify GPU cache arithmetic.
        ("kv-isolation", KvIsolation, all.clone(), vec![tiny("tiny_stack_concurrent_sessions_isolated")]),
        ("kv-release", KvRelease, all.clone(), vec![kv("failed_allocate_rolls_back_partial_blocks"), kv("failed_extend_rolls_back_partial_blocks_and_handle_table"), engine("process_batch_unified_capacity_defer_releases_existing_kv")]),
        // Supported preemption discards physical KV and replays the preserved
        // token history. Pair its scheduler transition with real CPU KV/logit
        // equivalence; this does not certify GPU swapping or persistence.
        ("kv-resume", KvResume, all, vec![engine("plan_runtime_batch_decode_capacity_deferral_recomputes_a_blocked_progress_victim"), engine("cpu_kv_recompute_matches_uninterrupted_logits_and_preserves_peer")]),
    ];
    specifications
        .into_iter()
        .map(|(id, behavior, entrypoints, tests)| ContractGroup {
            id: format!("cpu-contract.{id}"),
            behavior,
            entrypoints,
            tests,
        })
        .collect()
}

pub fn contract_check_descriptors() -> Vec<CheckDescriptor> {
    contract_groups()
        .into_iter()
        .map(|group| CheckDescriptor {
            id: group.id,
            behavior: group.behavior,
            layer: EvidenceLayer::Contract,
            entrypoints: group.entrypoints,
            target: None,
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HarnessArtifact {
    pub package: String,
    pub package_id: String,
    pub target: String,
    pub kind: String,
    pub executable: PathBuf,
    pub manifest_path: PathBuf,
}

/// Read Cargo's machine output, including its terminal build result. Package
/// names come from the artifact's manifest, not target-name guesses or filenames.
pub fn parse_compiler_artifacts(text: &str) -> Result<Vec<HarnessArtifact>, String> {
    let mut artifacts = Vec::new();
    let mut build_finished = None;
    for (line_index, line) in text
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
    {
        let message: Value = serde_json::from_str(line)
            .map_err(|error| format!("Cargo JSON line {}: {error}", line_index + 1))?;
        if build_finished.is_some() {
            return Err("Cargo messages follow build-finished".into());
        }
        match message["reason"].as_str() {
            Some("build-finished") => {
                build_finished = message["success"].as_bool();
                if build_finished != Some(true) {
                    return Err("Cargo build did not succeed".into());
                }
            }
            Some("compiler-artifact")
                if message["profile"]["test"] == true && !message["executable"].is_null() =>
            {
                let field = |key: &str| {
                    message[key]
                        .as_str()
                        .filter(|value| !value.is_empty())
                        .ok_or_else(|| format!("Cargo artifact is missing {key}"))
                };
                let manifest_path = PathBuf::from(field("manifest_path")?);
                let document = fs::read_to_string(&manifest_path)
                    .map_err(|error| {
                        format!(
                            "read artifact manifest {}: {error}",
                            manifest_path.display()
                        )
                    })?
                    .parse::<toml_edit::DocumentMut>()
                    .map_err(|error| format!("parse artifact manifest: {error}"))?;
                let package = document
                    .get("package")
                    .and_then(|item| item.get("name"))
                    .and_then(toml_edit::Item::as_str)
                    .ok_or("artifact manifest is missing package.name")?;
                let target = message["target"]["name"]
                    .as_str()
                    .filter(|name| !name.is_empty())
                    .ok_or("artifact target is missing name")?;
                let kinds = message["target"]["kind"]
                    .as_array()
                    .ok_or("artifact target is missing kind")?;
                let kind = kinds
                    .first()
                    .and_then(Value::as_str)
                    .filter(|_| kinds.len() == 1)
                    .ok_or("test harness must have one target kind")?;
                let artifact = HarnessArtifact {
                    package: package.into(),
                    package_id: field("package_id")?.into(),
                    target: target.into(),
                    kind: kind.into(),
                    executable: PathBuf::from(field("executable")?),
                    manifest_path,
                };
                if !artifact.executable.is_absolute() || !artifact.manifest_path.is_absolute() {
                    return Err("Cargo harness paths must be absolute".into());
                }
                if !artifacts.contains(&artifact) {
                    artifacts.push(artifact);
                }
            }
            Some(_) => {}
            None => return Err("Cargo message is missing reason".into()),
        }
    }
    if build_finished != Some(true) {
        return Err("Cargo messages are missing a successful build-finished".into());
    }
    Ok(artifacts)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandStatus {
    Passed,
    Failed,
    TimedOut,
    StartFailed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommandObservation {
    pub status: CommandStatus,
    pub exit_code: Option<i32>,
    pub elapsed_ms: u64,
    pub stdout: PathBuf,
    pub stderr: PathBuf,
    pub error: Option<String>,
}
impl CommandObservation {
    fn passed(&self) -> bool {
        self.status == CommandStatus::Passed && self.exit_code == Some(0) && self.error.is_none()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContractTestResult {
    pub binding: ContractTest,
    pub artifact: Option<HarnessArtifact>,
    pub registered: bool,
    pub listing: Option<CommandObservation>,
    pub execution: Option<CommandObservation>,
    pub error: Option<String>,
}
impl ContractTestResult {
    fn passed(&self) -> bool {
        self.registered
            && self.error.is_none()
            && self.artifact.is_some()
            && self
                .listing
                .as_ref()
                .is_some_and(CommandObservation::passed)
            && self
                .execution
                .as_ref()
                .is_some_and(CommandObservation::passed)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContractStatus {
    Passed,
    Failed,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContractGroupResult {
    pub id: String,
    pub behavior: Behavior,
    pub status: ContractStatus,
    pub tests: Vec<ContractTestResult>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContractReport {
    pub schema_version: u32,
    pub status: ContractStatus,
    pub groups: Vec<ContractGroupResult>,
}

pub struct ContractRunOptions {
    pub log_dir: PathBuf,
    /// Applied separately to harness discovery and each exact test invocation.
    pub timeout: Duration,
}

fn invoke(
    artifact: &HarnessArtifact,
    args: &[&str],
    stem: &Path,
    timeout: Duration,
    configure: &impl Fn(&mut Command),
) -> CommandObservation {
    let started = Instant::now();
    let mut observation = CommandObservation {
        status: CommandStatus::StartFailed,
        exit_code: None,
        elapsed_ms: 0,
        stdout: stem.with_extension("stdout.txt"),
        stderr: stem.with_extension("stderr.txt"),
        error: None,
    };
    let result = (|| -> Result<(), String> {
        let mut command = Command::new(&artifact.executable);
        command
            .args(args)
            .current_dir(
                artifact
                    .manifest_path
                    .parent()
                    .ok_or("manifest has no parent")?,
            )
            .stdin(Stdio::null())
            .stdout(File::create(&observation.stdout).map_err(|error| error.to_string())?)
            .stderr(File::create(&observation.stderr).map_err(|error| error.to_string())?);
        configure(&mut command);
        let mut child = command
            .spawn()
            .map_err(|error| format!("start harness: {error}"))?;
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    observation.exit_code = status.code();
                    observation.status = if status.success() {
                        CommandStatus::Passed
                    } else {
                        CommandStatus::Failed
                    };
                    break;
                }
                Ok(None) if started.elapsed() < timeout => {
                    std::thread::sleep(Duration::from_millis(5))
                }
                Ok(None) => {
                    observation.status = CommandStatus::TimedOut;
                    let _ = child.kill();
                    let _ = child.wait();
                    observation.error = Some("harness deadline exceeded".into());
                    break;
                }
                Err(error) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!("wait for harness: {error}"));
                }
            }
        }
        Ok(())
    })();
    if let Err(error) = result {
        observation.error = Some(error);
    }
    observation.elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    observation
}

pub fn run_contract_checks(
    artifacts: &[HarnessArtifact],
    groups: &[ContractGroup],
    options: &ContractRunOptions,
) -> Result<ContractReport, String> {
    run_checks(artifacts, groups, options, &|_| {})
}

fn run_checks(
    artifacts: &[HarnessArtifact],
    groups: &[ContractGroup],
    options: &ContractRunOptions,
    configure: &impl Fn(&mut Command),
) -> Result<ContractReport, String> {
    if groups.is_empty() || options.timeout.is_zero() {
        return Err("contract groups and positive timeout are required".into());
    }
    let mut ids = BTreeSet::new();
    for group in groups {
        if group.id.is_empty() || !ids.insert(&group.id) || group.tests.is_empty() {
            return Err("contract groups need unique IDs and nonempty test bindings".into());
        }
        let mut unique = BTreeSet::new();
        for test in &group.tests {
            if !unique.insert(test)
                || [&test.package, &test.target, &test.kind, &test.name]
                    .iter()
                    .any(|text| text.trim().is_empty())
            {
                return Err("contract tests need unique, nonempty bindings".into());
            }
        }
    }
    fs::create_dir(&options.log_dir).map_err(|error| {
        format!(
            "create new contract log directory {}: {error}",
            options.log_dir.display()
        )
    })?;
    // Shared registrations and assertions execute once per invocation, even when
    // their result is relevant to more than one behavioral group.
    let mut listings = BTreeMap::<PathBuf, (CommandObservation, BTreeSet<String>)>::new();
    let mut executions = BTreeMap::<ContractTest, ContractTestResult>::new();
    let mut output = Vec::new();
    for group in groups {
        let mut tests = Vec::new();
        for binding in &group.tests {
            if let Some(previous) = executions.get(binding) {
                tests.push(previous.clone());
                continue;
            }
            let mut test = ContractTestResult {
                binding: binding.clone(),
                artifact: None,
                registered: false,
                listing: None,
                execution: None,
                error: None,
            };
            let candidates: Vec<_> = artifacts
                .iter()
                .filter(|artifact| {
                    artifact.package == binding.package
                        && artifact.target == binding.target
                        && artifact.kind == binding.kind
                })
                .collect();
            if candidates.len() != 1 {
                test.error = Some(format!(
                    "expected one Cargo test harness for {}/{}/{}, found {}",
                    binding.package,
                    binding.kind,
                    binding.target,
                    candidates.len()
                ));
            } else {
                let artifact = candidates[0];
                test.artifact = Some(artifact.clone());
                if !listings.contains_key(&artifact.executable) {
                    let mut listing = invoke(
                        artifact,
                        &["--list", "--format", "terse"],
                        &options.log_dir.join(format!("list-{}", listings.len())),
                        options.timeout,
                        configure,
                    );
                    let mut names = BTreeSet::new();
                    if listing.passed() {
                        match fs::read_to_string(&listing.stdout) {
                            Ok(text) => {
                                names.extend(
                                    text.lines()
                                        .filter_map(|line| line.strip_suffix(": test"))
                                        .map(str::to_owned),
                                );
                            }
                            Err(error) => {
                                listing.status = CommandStatus::Failed;
                                listing.error = Some(format!("read harness registration: {error}"));
                            }
                        }
                    }
                    listings.insert(artifact.executable.clone(), (listing, names));
                }
                let (listing, names) = &listings[&artifact.executable];
                test.listing = Some(listing.clone());
                test.registered = listing.passed() && names.contains(&binding.name);
                if test.registered {
                    test.execution = Some(invoke(
                        artifact,
                        &["--exact", &binding.name, "--include-ignored", "--nocapture"],
                        &options.log_dir.join(format!("test-{}", executions.len())),
                        options.timeout,
                        configure,
                    ));
                } else {
                    test.error = Some(format!(
                        "required test {} was not registered by its successful harness listing",
                        binding.name
                    ));
                }
            }
            executions.insert(binding.clone(), test.clone());
            tests.push(test);
        }
        let status = if tests.iter().all(ContractTestResult::passed) {
            ContractStatus::Passed
        } else {
            ContractStatus::Failed
        };
        output.push(ContractGroupResult {
            id: group.id.clone(),
            behavior: group.behavior,
            status,
            tests,
        });
    }
    let status = if output
        .iter()
        .all(|group| group.status == ContractStatus::Passed)
    {
        ContractStatus::Passed
    } else {
        ContractStatus::Failed
    };
    Ok(ContractReport {
        schema_version: 1,
        status,
        groups: output,
    })
}

/// A consuming release gate supplies the registry groups it actually requires.
/// Missing groups/tests and contradictory passed summaries remain failures.
pub fn verify_contract_report(
    expected: &[ContractGroup],
    report: &ContractReport,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if expected.is_empty() || report.schema_version != 1 || report.status != ContractStatus::Passed
    {
        errors.push("contract report is not a nonempty terminal schema 1 pass".into());
    }
    let required: BTreeMap<_, _> = expected.iter().map(|group| (&group.id, group)).collect();
    if required.len() != expected.len() {
        errors.push("duplicate expected contract group".into());
    }
    let mut seen = BTreeSet::new();
    for group in &report.groups {
        if !seen.insert(&group.id) {
            errors.push(format!("duplicate contract group {}", group.id));
        }
        let Some(required) = required.get(&group.id) else {
            errors.push(format!("unexpected contract group {}", group.id));
            continue;
        };
        let bindings: BTreeSet<_> = group.tests.iter().map(|test| &test.binding).collect();
        let expected_bindings: BTreeSet<_> = required.tests.iter().collect();
        if group.behavior != required.behavior
            || group.status != ContractStatus::Passed
            || bindings != expected_bindings
            || bindings.len() != group.tests.len()
            || bindings.is_empty()
        {
            errors.push(format!(
                "contract group {} differs from required successful bindings",
                group.id
            ));
        }
        for test in &group.tests {
            let matching_artifact = test.artifact.as_ref().is_some_and(|artifact| {
                artifact.package == test.binding.package
                    && artifact.target == test.binding.target
                    && artifact.kind == test.binding.kind
            });
            if !test.passed() || !matching_artifact {
                errors.push(format!(
                    "contract assertion {} did not execute successfully",
                    test.binding.name
                ));
            }
        }
    }
    for id in required.keys() {
        if !seen.contains(id) {
            errors.push(format!("missing contract group {id}"));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[cfg(test)]
#[path = "contracts_tests.rs"]
mod tests;
