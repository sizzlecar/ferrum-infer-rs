//! Locked, independently runnable native source-build plans.

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use ferrum_native_ops::{
    CudaNativeBuildUnit, NativeBuildArtifactCache, NativeBuildArtifactLookup,
    NativeBuildArtifactSpec,
};
use ferrum_types::{is_sha256_digest, NativeOperatorSourcePackage};
use serde::{Deserialize, Serialize};

use super::{
    read_json, require_file, sha256_bytes, sha256_file, symbol_slug, validate_relative_path,
    write_json, NativeOperatorBuilderError, Result,
};

pub const NATIVE_OPERATOR_SOURCE_DEFINITION_SCHEMA_VERSION: u32 = 2;
pub const NATIVE_OPERATOR_SOURCE_BUILD_PLAN_SCHEMA_VERSION: u32 = 2;
pub const NATIVE_OPERATOR_SOURCE_BUILD_RECEIPT_SCHEMA_VERSION: u32 = 2;
pub const NATIVE_OPERATOR_SOURCE_OBJECT_BUILD_CONTRACT_VERSION: u32 = 1;
pub const MAX_NVCC_THREADS: u32 = 8;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorSourceDefinition {
    pub schema_version: u32,
    pub operator: String,
    pub source_package_kind: String,
    pub source_package_revision: String,
    pub upstream_sources: Vec<NativeOperatorUpstreamSource>,
    pub translation_units: Vec<String>,
    pub headers: Vec<String>,
    pub include_dirs: Vec<String>,
    pub defines: Vec<String>,
    pub nvcc_policy: NativeOperatorNvccPolicy,
    pub architecture: NativeOperatorCudaArchitecture,
    pub archive_file: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorUpstreamSource {
    pub repository: String,
    pub revision: String,
    pub license: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorSourceBuildPlan {
    pub schema_version: u32,
    pub operator: String,
    pub source_package: NativeOperatorSourcePackage,
    pub upstream_sources: Vec<NativeOperatorUpstreamSource>,
    pub translation_units: Vec<NativeOperatorSourceFileLock>,
    pub headers: Vec<NativeOperatorSourceFileLock>,
    pub include_dirs: Vec<String>,
    pub defines: Vec<String>,
    pub nvcc_policy: NativeOperatorNvccPolicy,
    pub architecture: NativeOperatorCudaArchitecture,
    pub archive_file: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NativeOperatorSourceFileLock {
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeOperatorCudaArchitecture {
    DeviceComputeCapability,
    Compute80Ptx,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeOperatorCppStandard {
    Cpp17,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeOperatorOptimization {
    O3,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorNvccPolicy {
    pub cpp_standard: NativeOperatorCppStandard,
    pub optimization: NativeOperatorOptimization,
    pub use_fast_math: bool,
    pub relaxed_constexpr: bool,
    pub extended_lambda: bool,
    pub host_position_independent_code: bool,
    pub host_default_visibility: bool,
}

#[derive(Debug, Clone)]
pub struct NativeOperatorSourceBuildRequest {
    pub plan_path: PathBuf,
    pub source_root: PathBuf,
    pub output_dir: PathBuf,
    pub compute_capability: String,
    pub builder_sha: String,
    pub nvcc_path: PathBuf,
    pub ccbin_path: PathBuf,
    pub ar_path: PathBuf,
    pub nvcc_threads: u32,
    pub object_cache_dir: PathBuf,
    pub plan_only: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorSourceBuildReceipt {
    pub schema_version: u32,
    pub status: NativeOperatorSourceBuildStatus,
    pub operator: String,
    pub plan_only: bool,
    pub plan_sha256: String,
    pub source_package: NativeOperatorSourcePackage,
    pub builder_sha: String,
    pub compute_capability: String,
    pub architecture_argument: String,
    pub nvcc_threads: u32,
    pub object_cache_root: String,
    pub toolchain: Option<NativeOperatorSourceBuildToolchain>,
    pub effective_environment: BTreeMap<String, String>,
    pub inputs_sha256: String,
    pub commands: Vec<NativeOperatorSourceBuildCommand>,
    pub compiled_translation_units: Vec<String>,
    pub cache_hit_translation_units: Vec<String>,
    pub archive_file: Option<String>,
    pub archive_sha256: Option<String>,
    pub started_unix_ms: u64,
    pub elapsed_ms: u64,
    pub failure_class: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeOperatorSourceBuildStatus {
    Plan,
    Pass,
    Reject,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorSourceBuildToolchain {
    pub nvcc: NativeOperatorToolIdentity,
    pub host_compiler: NativeOperatorToolIdentity,
    pub archiver: NativeOperatorToolIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorToolIdentity {
    pub path: String,
    pub sha256: String,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorSourceBuildCommand {
    pub translation_unit: Option<String>,
    pub working_directory: String,
    pub argv: Vec<String>,
    pub object_file: Option<String>,
    pub stdout_log: String,
    pub stderr_log: String,
    pub object_cache_key: Option<String>,
    pub object_cache_status: Option<NativeOperatorSourceObjectCacheStatus>,
    pub object_cache_entry: Option<String>,
    pub object_sha256: Option<String>,
    pub compiler_executed: bool,
    pub elapsed_ms: Option<u64>,
    pub return_code: Option<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeOperatorSourceObjectCacheStatus {
    Plan,
    Pending,
    Hit,
    Miss,
    Published,
    Rejected,
}

#[derive(Debug, Serialize)]
struct NativeOperatorSourceInventoryIdentity<'a> {
    operator: &'a str,
    upstream_sources: &'a [NativeOperatorUpstreamSource],
    translation_units: &'a [NativeOperatorSourceFileLock],
    headers: &'a [NativeOperatorSourceFileLock],
}

#[derive(Debug, Serialize)]
struct NativeOperatorBuildInputIdentity<'a> {
    plan_sha256: &'a str,
    source_package_sha256: &'a str,
    builder_contract_version: u32,
    architecture_argument: &'a str,
    effective_environment: &'a BTreeMap<String, String>,
    toolchain: Option<&'a NativeOperatorSourceBuildToolchain>,
}

#[derive(Debug, Serialize)]
struct NativeOperatorObjectInputIdentity<'a> {
    schema_version: u32,
    operator: &'a str,
    translation_unit: &'a NativeOperatorSourceFileLock,
    headers: &'a [NativeOperatorSourceFileLock],
    include_dirs: &'a [String],
    defines: &'a [String],
    nvcc_policy: &'a NativeOperatorNvccPolicy,
    architecture_argument: &'a str,
    builder_contract_version: u32,
    effective_environment: &'a BTreeMap<String, String>,
    toolchain: &'a NativeOperatorSourceBuildToolchain,
}

pub fn lock_native_operator_source_definition(
    definition_path: &Path,
    source_root: &Path,
    output_plan_path: &Path,
) -> Result<NativeOperatorSourceBuildPlan> {
    require_file(definition_path)?;
    if output_plan_path.exists() {
        return Err(NativeOperatorBuilderError::OutputExists(
            output_plan_path.to_path_buf(),
        ));
    }
    let definition: NativeOperatorSourceDefinition = read_json(definition_path)?;
    validate_definition(&definition)?;
    let canonical_root =
        source_root
            .canonicalize()
            .map_err(|source| NativeOperatorBuilderError::Io {
                path: source_root.to_path_buf(),
                source,
            })?;
    if !canonical_root.is_dir() {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "source_root is not a directory: {}",
            canonical_root.display()
        )));
    }
    for include_dir in &definition.include_dirs {
        let path = canonical_root.join(include_dir);
        let canonical = path
            .canonicalize()
            .map_err(|source| NativeOperatorBuilderError::Io {
                path: path.clone(),
                source,
            })?;
        if !canonical.starts_with(&canonical_root) || !canonical.is_dir() {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "include directory escapes source root or does not exist: {}",
                path.display()
            )));
        }
    }
    let translation_units = lock_source_files(&canonical_root, &definition.translation_units)?;
    let headers = lock_source_files(&canonical_root, &definition.headers)?;
    let inventory_identity = NativeOperatorSourceInventoryIdentity {
        operator: &definition.operator,
        upstream_sources: &definition.upstream_sources,
        translation_units: &translation_units,
        headers: &headers,
    };
    let source_package_sha256 =
        sha256_bytes(&serde_json::to_vec(&inventory_identity).map_err(|source| {
            NativeOperatorBuilderError::Json {
                path: output_plan_path.to_path_buf(),
                source,
            }
        })?);
    let plan = NativeOperatorSourceBuildPlan {
        schema_version: NATIVE_OPERATOR_SOURCE_BUILD_PLAN_SCHEMA_VERSION,
        operator: definition.operator,
        source_package: NativeOperatorSourcePackage {
            kind: definition.source_package_kind,
            revision: definition.source_package_revision,
            sha256: source_package_sha256,
        },
        upstream_sources: definition.upstream_sources,
        translation_units,
        headers,
        include_dirs: definition.include_dirs,
        defines: definition.defines,
        nvcc_policy: definition.nvcc_policy,
        architecture: definition.architecture,
        archive_file: definition.archive_file,
    };
    validate_plan(&plan)?;
    if let Some(parent) = output_plan_path.parent() {
        fs::create_dir_all(parent).map_err(|source| NativeOperatorBuilderError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    write_json(output_plan_path, &plan)?;
    Ok(plan)
}

pub fn run_native_operator_source_build(
    request: &NativeOperatorSourceBuildRequest,
) -> Result<NativeOperatorSourceBuildReceipt> {
    require_file(&request.plan_path)?;
    if request.output_dir.exists() {
        return Err(NativeOperatorBuilderError::OutputExists(
            request.output_dir.clone(),
        ));
    }
    if !request.output_dir.is_absolute() {
        return Err(NativeOperatorBuilderError::Invalid(
            "source build output_dir must be absolute".to_string(),
        ));
    }
    if !request.object_cache_dir.is_absolute() {
        return Err(NativeOperatorBuilderError::Invalid(
            "source build object_cache_dir must be absolute".to_string(),
        ));
    }
    if [&request.nvcc_path, &request.ccbin_path, &request.ar_path]
        .iter()
        .any(|path| !path.is_absolute())
    {
        return Err(NativeOperatorBuilderError::Invalid(
            "source build tool paths must be absolute".to_string(),
        ));
    }
    validate_compute_capability(&request.compute_capability)?;
    if request.nvcc_threads == 0 || request.nvcc_threads > MAX_NVCC_THREADS {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "nvcc_threads must be in [1,{MAX_NVCC_THREADS}]"
        )));
    }
    if !is_git_oid(&request.builder_sha) {
        return Err(NativeOperatorBuilderError::Invalid(
            "builder_sha must be a lowercase 40- or 64-hex git object id".to_string(),
        ));
    }
    let plan: NativeOperatorSourceBuildPlan = read_json(&request.plan_path)?;
    validate_plan(&plan)?;
    let plan_sha256 = sha256_file(&request.plan_path)?;
    let canonical_root =
        request
            .source_root
            .canonicalize()
            .map_err(|source| NativeOperatorBuilderError::Io {
                path: request.source_root.clone(),
                source,
            })?;
    validate_locked_source_tree(&canonical_root, &plan)?;
    let architecture_argument =
        architecture_argument(plan.architecture, &request.compute_capability);
    let started_unix_ms = unix_ms();
    let started = Instant::now();
    fs::create_dir_all(&request.output_dir).map_err(|source| NativeOperatorBuilderError::Io {
        path: request.output_dir.clone(),
        source,
    })?;
    let receipt_path = request.output_dir.join("source-build.receipt.json");
    let logs_dir = request.output_dir.join("logs");
    let objects_dir = request.output_dir.join("objects");
    fs::create_dir_all(&logs_dir).map_err(|source| NativeOperatorBuilderError::Io {
        path: logs_dir.clone(),
        source,
    })?;
    if !request.plan_only {
        fs::create_dir_all(&objects_dir).map_err(|source| NativeOperatorBuilderError::Io {
            path: objects_dir.clone(),
            source,
        })?;
    }

    let (toolchain, toolchain_failure) = if request.plan_only {
        (None, None)
    } else {
        match resolve_toolchain(request) {
            Ok(toolchain) => (Some(toolchain), None),
            Err(error) => (None, Some(error.to_string())),
        }
    };
    let effective_environment = effective_build_environment(request, toolchain.as_ref())?;
    let inputs_sha256 = build_inputs_sha256(
        &plan_sha256,
        &plan.source_package.sha256,
        &architecture_argument,
        &effective_environment,
        toolchain.as_ref(),
        &receipt_path,
    )?;
    let mut commands = build_commands(
        request,
        &plan,
        &canonical_root,
        &architecture_argument,
        &objects_dir,
        &logs_dir,
        toolchain.as_ref(),
        &effective_environment,
    );
    let initial_log = if request.plan_only {
        b"plan-only: command was not executed\n".as_slice()
    } else {
        b"pending: command has not executed\n".as_slice()
    };
    for command in &commands {
        write_command_stream(
            &request.output_dir.join(&command.stdout_log),
            "stdout",
            &command.argv,
            initial_log,
        )?;
        write_command_stream(
            &request.output_dir.join(&command.stderr_log),
            "stderr",
            &command.argv,
            initial_log,
        )?;
    }
    let mut receipt = NativeOperatorSourceBuildReceipt {
        schema_version: NATIVE_OPERATOR_SOURCE_BUILD_RECEIPT_SCHEMA_VERSION,
        status: if request.plan_only {
            NativeOperatorSourceBuildStatus::Plan
        } else {
            NativeOperatorSourceBuildStatus::Reject
        },
        operator: plan.operator.clone(),
        plan_only: request.plan_only,
        plan_sha256,
        source_package: plan.source_package.clone(),
        builder_sha: request.builder_sha.clone(),
        compute_capability: request.compute_capability.clone(),
        architecture_argument,
        nvcc_threads: request.nvcc_threads,
        object_cache_root: request.object_cache_dir.display().to_string(),
        toolchain,
        effective_environment,
        inputs_sha256,
        commands: commands.clone(),
        compiled_translation_units: Vec::new(),
        cache_hit_translation_units: Vec::new(),
        archive_file: None,
        archive_sha256: None,
        started_unix_ms,
        elapsed_ms: 0,
        failure_class: None,
    };
    if request.plan_only {
        receipt.elapsed_ms = millis(started.elapsed());
        write_json(&receipt_path, &receipt)?;
        return Ok(receipt);
    }
    if let Some(error) = toolchain_failure {
        receipt.elapsed_ms = millis(started.elapsed());
        return reject_source_build(
            &receipt_path,
            &mut receipt,
            format!("toolchain_preflight_failed:{error}"),
        );
    }
    let object_cache = match NativeBuildArtifactCache::new(&request.object_cache_dir) {
        Ok(cache) => cache,
        Err(error) => {
            receipt.elapsed_ms = millis(started.elapsed());
            return reject_source_build(
                &receipt_path,
                &mut receipt,
                format!("object_cache_init_failed:{error}"),
            );
        }
    };
    let object_specs = match build_object_cache_specs(
        &plan,
        &receipt.architecture_argument,
        receipt
            .toolchain
            .as_ref()
            .expect("actual source build has a resolved toolchain"),
        &receipt.effective_environment,
    ) {
        Ok(specs) => specs,
        Err(error) => {
            receipt.elapsed_ms = millis(started.elapsed());
            return reject_source_build(
                &receipt_path,
                &mut receipt,
                format!("object_cache_spec_failed:{error}"),
            );
        }
    };
    for (command, spec) in commands.iter_mut().zip(object_specs.iter()) {
        command.object_cache_key = Some(spec.input_signature_sha256().to_string());
    }
    receipt.commands = commands.clone();
    receipt.failure_class = Some("build_incomplete".to_string());
    write_json(&receipt_path, &receipt)?;

    for index in 0..plan.translation_units.len() {
        let translation_unit = &plan.translation_units[index];
        let object_path = PathBuf::from(
            commands[index]
                .object_file
                .as_deref()
                .expect("translation-unit command has an object file"),
        );
        let command_started = Instant::now();
        match object_cache.restore(&object_specs[index], &object_path) {
            Ok(NativeBuildArtifactLookup::Hit(cache_receipt)) => {
                commands[index].object_cache_status =
                    Some(NativeOperatorSourceObjectCacheStatus::Hit);
                commands[index].object_cache_entry =
                    Some(cache_receipt.cache_entry.display().to_string());
                commands[index].object_sha256 = Some(cache_receipt.artifact_sha256);
                commands[index].elapsed_ms = Some(millis(command_started.elapsed()));
                append_command_stream(
                    &request.output_dir.join(&commands[index].stdout_log),
                    b"object-cache-hit: compiler was not executed\n",
                )?;
                receipt
                    .cache_hit_translation_units
                    .push(translation_unit.path.clone());
                receipt.commands = commands.clone();
                receipt.elapsed_ms = millis(started.elapsed());
                write_json(&receipt_path, &receipt)?;
                continue;
            }
            Ok(NativeBuildArtifactLookup::Miss { reason }) => {
                commands[index].object_cache_status =
                    Some(NativeOperatorSourceObjectCacheStatus::Miss);
                append_command_stream(
                    &request.output_dir.join(&commands[index].stdout_log),
                    format!("object-cache-miss: {reason}\n").as_bytes(),
                )?;
            }
            Err(error) => {
                commands[index].object_cache_status =
                    Some(NativeOperatorSourceObjectCacheStatus::Rejected);
                receipt.commands = commands.clone();
                receipt.elapsed_ms = millis(started.elapsed());
                return reject_source_build(
                    &receipt_path,
                    &mut receipt,
                    format!("object_cache_restore_failed:{index}:{error}"),
                );
            }
        }
        let stdout_path = request.output_dir.join(&commands[index].stdout_log);
        let stderr_path = request.output_dir.join(&commands[index].stderr_log);
        commands[index].compiler_executed = true;
        let status = match run_logged_command(
            &commands[index].argv,
            &stdout_path,
            &stderr_path,
            &commands[index].working_directory,
            &receipt.effective_environment,
        ) {
            Ok(status) => status,
            Err(error) => {
                append_command_stream(&stderr_path, format!("spawn failed: {error}\n").as_bytes())?;
                receipt.commands = commands.clone();
                receipt.elapsed_ms = millis(started.elapsed());
                return reject_source_build(
                    &receipt_path,
                    &mut receipt,
                    format!("nvcc_translation_unit_{index}_spawn_failed"),
                );
            }
        };
        commands[index].elapsed_ms = Some(millis(command_started.elapsed()));
        commands[index].return_code = status.code();
        receipt
            .compiled_translation_units
            .push(translation_unit.path.clone());
        receipt.commands = commands.clone();
        receipt.elapsed_ms = millis(started.elapsed());
        if !status.success() {
            return reject_source_build(
                &receipt_path,
                &mut receipt,
                format!("nvcc_translation_unit_{index}_failed"),
            );
        }
        if let Err(error) = require_file(&object_path) {
            commands[index].object_cache_status =
                Some(NativeOperatorSourceObjectCacheStatus::Rejected);
            receipt.commands = commands.clone();
            receipt.elapsed_ms = millis(started.elapsed());
            return reject_source_build(
                &receipt_path,
                &mut receipt,
                format!("compiled_object_missing:{index}:{error}"),
            );
        }
        let cache_receipt = match object_cache.publish(&object_specs[index], &object_path) {
            Ok(cache_receipt) => cache_receipt,
            Err(error) => {
                commands[index].object_cache_status =
                    Some(NativeOperatorSourceObjectCacheStatus::Rejected);
                receipt.commands = commands.clone();
                receipt.elapsed_ms = millis(started.elapsed());
                return reject_source_build(
                    &receipt_path,
                    &mut receipt,
                    format!("object_cache_publish_failed:{index}:{error}"),
                );
            }
        };
        commands[index].object_cache_status =
            Some(NativeOperatorSourceObjectCacheStatus::Published);
        commands[index].object_cache_entry = Some(cache_receipt.cache_entry.display().to_string());
        commands[index].object_sha256 = Some(cache_receipt.artifact_sha256);
        receipt.commands = commands.clone();
        write_json(&receipt_path, &receipt)?;
    }

    let archive_index = plan.translation_units.len();
    let archive_command = commands
        .get_mut(archive_index)
        .expect("archive command follows translation units");
    let archive_started = Instant::now();
    let archive_stdout_path = request.output_dir.join(&archive_command.stdout_log);
    let archive_stderr_path = request.output_dir.join(&archive_command.stderr_log);
    let archive_status = match run_logged_command(
        &archive_command.argv,
        &archive_stdout_path,
        &archive_stderr_path,
        &archive_command.working_directory,
        &receipt.effective_environment,
    ) {
        Ok(status) => status,
        Err(error) => {
            append_command_stream(
                &archive_stderr_path,
                format!("spawn failed: {error}\n").as_bytes(),
            )?;
            receipt.commands = commands.clone();
            receipt.elapsed_ms = millis(started.elapsed());
            return reject_source_build(
                &receipt_path,
                &mut receipt,
                "archive_spawn_failed".to_string(),
            );
        }
    };
    archive_command.elapsed_ms = Some(millis(archive_started.elapsed()));
    archive_command.return_code = archive_status.code();
    receipt.commands = commands;
    receipt.elapsed_ms = millis(started.elapsed());
    if !archive_status.success() {
        return reject_source_build(&receipt_path, &mut receipt, "archive_failed".to_string());
    }

    let archive_path = request.output_dir.join(&plan.archive_file);
    if let Err(error) = require_file(&archive_path) {
        return reject_source_build(
            &receipt_path,
            &mut receipt,
            format!("archive_output_missing:{error}"),
        );
    }
    let archive_sha256 = match sha256_file(&archive_path) {
        Ok(sha256) => sha256,
        Err(error) => {
            return reject_source_build(
                &receipt_path,
                &mut receipt,
                format!("archive_output_hash_failed:{error}"),
            );
        }
    };
    receipt.archive_sha256 = Some(archive_sha256);
    receipt.archive_file = Some(plan.archive_file);
    receipt.status = NativeOperatorSourceBuildStatus::Pass;
    receipt.failure_class = None;
    receipt.elapsed_ms = millis(started.elapsed());
    write_json(&receipt_path, &receipt)?;
    Ok(receipt)
}

fn validate_definition(definition: &NativeOperatorSourceDefinition) -> Result<()> {
    if definition.schema_version != NATIVE_OPERATOR_SOURCE_DEFINITION_SCHEMA_VERSION {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "source definition schema_version must be {NATIVE_OPERATOR_SOURCE_DEFINITION_SCHEMA_VERSION}"
        )));
    }
    validate_common(
        &definition.operator,
        &definition.upstream_sources,
        &definition.translation_units,
        &definition.headers,
        &definition.include_dirs,
        &definition.defines,
        &definition.archive_file,
    )?;
    if definition.source_package_kind.trim().is_empty()
        || definition.source_package_revision.trim().is_empty()
    {
        return Err(NativeOperatorBuilderError::Invalid(
            "source package kind and revision must be non-empty".to_string(),
        ));
    }
    Ok(())
}

fn validate_plan(plan: &NativeOperatorSourceBuildPlan) -> Result<()> {
    if plan.schema_version != NATIVE_OPERATOR_SOURCE_BUILD_PLAN_SCHEMA_VERSION {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "source build plan schema_version must be {NATIVE_OPERATOR_SOURCE_BUILD_PLAN_SCHEMA_VERSION}"
        )));
    }
    let translation_units = plan
        .translation_units
        .iter()
        .map(|file| file.path.clone())
        .collect::<Vec<_>>();
    let headers = plan
        .headers
        .iter()
        .map(|file| file.path.clone())
        .collect::<Vec<_>>();
    validate_common(
        &plan.operator,
        &plan.upstream_sources,
        &translation_units,
        &headers,
        &plan.include_dirs,
        &plan.defines,
        &plan.archive_file,
    )?;
    if plan.source_package.kind.trim().is_empty()
        || plan.source_package.revision.trim().is_empty()
        || !is_sha256_digest(&plan.source_package.sha256)
    {
        return Err(NativeOperatorBuilderError::Invalid(
            "source_package kind/revision must be non-empty and sha256 must be lowercase"
                .to_string(),
        ));
    }
    for file in plan.translation_units.iter().chain(plan.headers.iter()) {
        if !is_sha256_digest(&file.sha256) {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "{} has an invalid sha256",
                file.path
            )));
        }
    }
    let identity = NativeOperatorSourceInventoryIdentity {
        operator: &plan.operator,
        upstream_sources: &plan.upstream_sources,
        translation_units: &plan.translation_units,
        headers: &plan.headers,
    };
    let actual = sha256_bytes(&serde_json::to_vec(&identity).map_err(|source| {
        NativeOperatorBuilderError::Json {
            path: PathBuf::from("<source-build-plan>"),
            source,
        }
    })?);
    if actual != plan.source_package.sha256 {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "source_package.sha256 differs from locked file inventory: expected={} actual={actual}",
            plan.source_package.sha256
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_common(
    operator: &str,
    upstream_sources: &[NativeOperatorUpstreamSource],
    translation_units: &[String],
    headers: &[String],
    include_dirs: &[String],
    defines: &[String],
    archive_file: &str,
) -> Result<()> {
    symbol_slug(operator)?;
    if CudaNativeBuildUnit::from_artifact_operator(operator).is_none() {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "source build operator is not a registered CUDA build unit: {operator}"
        )));
    }
    if upstream_sources.is_empty() {
        return Err(NativeOperatorBuilderError::Invalid(
            "upstream_sources must be non-empty".to_string(),
        ));
    }
    if upstream_sources.windows(2).any(|pair| {
        (&pair[0].repository, &pair[0].revision) >= (&pair[1].repository, &pair[1].revision)
    }) {
        return Err(NativeOperatorBuilderError::Invalid(
            "upstream_sources must be sorted and unique by repository/revision".to_string(),
        ));
    }
    for upstream in upstream_sources {
        if upstream.repository.trim().is_empty()
            || upstream.revision.trim().is_empty()
            || upstream.license.trim().is_empty()
        {
            return Err(NativeOperatorBuilderError::Invalid(
                "upstream source repository, revision, and license must be non-empty".to_string(),
            ));
        }
    }
    require_sorted_paths("translation_units", translation_units)?;
    require_sorted_optional_paths("headers", headers)?;
    if translation_units.iter().any(|path| !path.ends_with(".cu")) {
        return Err(NativeOperatorBuilderError::Invalid(
            "translation_units must use .cu paths".to_string(),
        ));
    }
    let mut all_files = translation_units
        .iter()
        .chain(headers.iter())
        .collect::<Vec<_>>();
    all_files.sort();
    if all_files.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(NativeOperatorBuilderError::Invalid(
            "translation_units and headers overlap".to_string(),
        ));
    }
    require_sorted_optional_paths("include_dirs", include_dirs)?;
    if defines.windows(2).any(|pair| pair[0] >= pair[1])
        || defines
            .iter()
            .any(|define| define.is_empty() || define.chars().any(char::is_whitespace))
    {
        return Err(NativeOperatorBuilderError::Invalid(
            "defines must be sorted, unique, non-empty single arguments".to_string(),
        ));
    }
    validate_relative_path(archive_file)?;
    if Path::new(archive_file).parent() != Some(Path::new(""))
        || !archive_file.starts_with("lib")
        || !archive_file.ends_with(".a")
    {
        return Err(NativeOperatorBuilderError::Invalid(
            "archive_file must be a lib*.a filename without directories".to_string(),
        ));
    }
    Ok(())
}

fn require_sorted_paths(field: &str, paths: &[String]) -> Result<()> {
    if paths.is_empty() {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{field} must be sorted, unique, non-empty normalized relative paths"
        )));
    }
    require_sorted_optional_paths(field, paths)
}

fn require_sorted_optional_paths(field: &str, paths: &[String]) -> Result<()> {
    if paths.windows(2).any(|pair| pair[0] >= pair[1])
        || paths
            .iter()
            .any(|path| validate_relative_path(path).is_err())
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{field} must be sorted, unique normalized relative paths"
        )));
    }
    Ok(())
}

fn lock_source_files(root: &Path, paths: &[String]) -> Result<Vec<NativeOperatorSourceFileLock>> {
    paths
        .iter()
        .map(|relative| {
            let path = locked_source_file(root, relative)?;
            Ok(NativeOperatorSourceFileLock {
                path: relative.clone(),
                sha256: sha256_file(&path)?,
            })
        })
        .collect()
}

fn validate_locked_source_tree(root: &Path, plan: &NativeOperatorSourceBuildPlan) -> Result<()> {
    if !root.is_dir() {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "source_root is not a directory: {}",
            root.display()
        )));
    }
    for include_dir in &plan.include_dirs {
        let path = root.join(include_dir);
        let canonical = path
            .canonicalize()
            .map_err(|source| NativeOperatorBuilderError::Io {
                path: path.clone(),
                source,
            })?;
        if !canonical.starts_with(root) || !canonical.is_dir() {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "locked include directory escapes source root or is missing: {include_dir}"
            )));
        }
    }
    for locked in plan.translation_units.iter().chain(plan.headers.iter()) {
        let path = locked_source_file(root, &locked.path)?;
        let actual = sha256_file(&path)?;
        if actual != locked.sha256 {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "locked source drift: path={} expected={} actual={actual}",
                locked.path, locked.sha256
            )));
        }
    }
    Ok(())
}

fn locked_source_file(root: &Path, relative: &str) -> Result<PathBuf> {
    validate_relative_path(relative)?;
    let path = root.join(relative);
    let metadata =
        fs::symlink_metadata(&path).map_err(|source| NativeOperatorBuilderError::Io {
            path: path.clone(),
            source,
        })?;
    if metadata.file_type().is_symlink() {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "locked source files must not be symlinks: {relative}"
        )));
    }
    let canonical = path
        .canonicalize()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: path.clone(),
            source,
        })?;
    if !canonical.starts_with(root) || !canonical.is_file() {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "locked source file escapes source root or is not a file: {relative}"
        )));
    }
    Ok(canonical)
}

fn resolve_toolchain(
    request: &NativeOperatorSourceBuildRequest,
) -> Result<NativeOperatorSourceBuildToolchain> {
    Ok(NativeOperatorSourceBuildToolchain {
        nvcc: tool_identity(&request.nvcc_path)?,
        host_compiler: tool_identity(&request.ccbin_path)?,
        archiver: tool_identity(&request.ar_path)?,
    })
}

fn tool_identity(path: &Path) -> Result<NativeOperatorToolIdentity> {
    require_file(path)?;
    let canonical = path
        .canonicalize()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let output = Command::new(&canonical)
        .arg("--version")
        .env_clear()
        .env("LANG", "C")
        .env("LC_ALL", "C")
        .env("TZ", "UTC")
        .output()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: canonical.clone(),
            source,
        })?;
    let version = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
    .trim()
    .chars()
    .take(4000)
    .collect::<String>();
    if version.is_empty() {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "tool produced no version identity: {}",
            canonical.display()
        )));
    }
    Ok(NativeOperatorToolIdentity {
        path: canonical.display().to_string(),
        sha256: sha256_file(&canonical)?,
        version,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_inputs_sha256(
    plan_sha256: &str,
    source_package_sha256: &str,
    architecture_argument: &str,
    effective_environment: &BTreeMap<String, String>,
    toolchain: Option<&NativeOperatorSourceBuildToolchain>,
    receipt_path: &Path,
) -> Result<String> {
    let identity = NativeOperatorBuildInputIdentity {
        plan_sha256,
        source_package_sha256,
        builder_contract_version: NATIVE_OPERATOR_SOURCE_OBJECT_BUILD_CONTRACT_VERSION,
        architecture_argument,
        effective_environment,
        toolchain,
    };
    let bytes =
        serde_json::to_vec(&identity).map_err(|source| NativeOperatorBuilderError::Json {
            path: receipt_path.to_path_buf(),
            source,
        })?;
    Ok(sha256_bytes(&bytes))
}

fn effective_build_environment(
    request: &NativeOperatorSourceBuildRequest,
    toolchain: Option<&NativeOperatorSourceBuildToolchain>,
) -> Result<BTreeMap<String, String>> {
    let tool_paths = if let Some(toolchain) = toolchain {
        [
            toolchain.nvcc.path.as_str(),
            toolchain.host_compiler.path.as_str(),
            toolchain.archiver.path.as_str(),
        ]
    } else {
        [
            request.nvcc_path.to_str().unwrap_or(""),
            request.ccbin_path.to_str().unwrap_or(""),
            request.ar_path.to_str().unwrap_or(""),
        ]
    };
    let mut path_entries = tool_paths
        .iter()
        .filter_map(|path| Path::new(path).parent())
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>();
    path_entries.extend(["/bin".to_string(), "/usr/bin".to_string()]);
    path_entries.sort();
    path_entries.dedup();
    if path_entries.iter().any(|path| path.is_empty()) {
        return Err(NativeOperatorBuilderError::Invalid(
            "source build tool paths must have parent directories".to_string(),
        ));
    }
    let mut environment = BTreeMap::new();
    environment.insert("LANG".to_string(), "C".to_string());
    environment.insert("LC_ALL".to_string(), "C".to_string());
    environment.insert("PATH".to_string(), path_entries.join(":"));
    environment.insert("SOURCE_DATE_EPOCH".to_string(), "0".to_string());
    environment.insert("TMPDIR".to_string(), "/tmp".to_string());
    environment.insert("TZ".to_string(), "UTC".to_string());
    environment.insert("ZERO_AR_DATE".to_string(), "1".to_string());
    Ok(environment)
}

fn build_object_cache_specs(
    plan: &NativeOperatorSourceBuildPlan,
    architecture_argument: &str,
    toolchain: &NativeOperatorSourceBuildToolchain,
    effective_environment: &BTreeMap<String, String>,
) -> Result<Vec<NativeBuildArtifactSpec>> {
    plan.translation_units
        .iter()
        .enumerate()
        .map(|(index, translation_unit)| {
            let identity = NativeOperatorObjectInputIdentity {
                schema_version: 1,
                operator: &plan.operator,
                translation_unit,
                headers: &plan.headers,
                include_dirs: &plan.include_dirs,
                defines: &plan.defines,
                nvcc_policy: &plan.nvcc_policy,
                architecture_argument,
                builder_contract_version: NATIVE_OPERATOR_SOURCE_OBJECT_BUILD_CONTRACT_VERSION,
                effective_environment,
                toolchain,
            };
            let input_signature = serde_json::to_string(&identity).map_err(|source| {
                NativeOperatorBuilderError::Json {
                    path: PathBuf::from("<object-cache-input>"),
                    source,
                }
            })?;
            NativeBuildArtifactSpec::new(
                format!("{}.object.{index:02}", plan.operator),
                object_file_name(index, translation_unit),
                input_signature,
            )
            .map_err(NativeOperatorBuilderError::from)
        })
        .collect()
}

fn object_file_name(index: usize, translation_unit: &NativeOperatorSourceFileLock) -> String {
    let stem = Path::new(&translation_unit.path)
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("translation_unit");
    format!(
        "{index:02}_{}_{}.o",
        safe_component(stem),
        &translation_unit.sha256[..8]
    )
}

fn nvcc_policy_flags(policy: &NativeOperatorNvccPolicy) -> Vec<String> {
    let mut flags = vec![
        match policy.cpp_standard {
            NativeOperatorCppStandard::Cpp17 => "-std=c++17",
        }
        .to_string(),
        match policy.optimization {
            NativeOperatorOptimization::O3 => "-O3",
        }
        .to_string(),
    ];
    if policy.use_fast_math {
        flags.push("--use_fast_math".to_string());
    }
    if policy.relaxed_constexpr {
        flags.push("--expt-relaxed-constexpr".to_string());
    }
    if policy.extended_lambda {
        flags.push("--expt-extended-lambda".to_string());
    }
    if policy.host_position_independent_code {
        flags.extend(["-Xcompiler".to_string(), "-fPIC".to_string()]);
    }
    if policy.host_default_visibility {
        flags.extend(["-Xcompiler".to_string(), "-fvisibility=default".to_string()]);
    }
    flags
}

fn build_commands(
    request: &NativeOperatorSourceBuildRequest,
    plan: &NativeOperatorSourceBuildPlan,
    source_root: &Path,
    architecture_argument: &str,
    objects_dir: &Path,
    logs_dir: &Path,
    toolchain: Option<&NativeOperatorSourceBuildToolchain>,
    _effective_environment: &BTreeMap<String, String>,
) -> Vec<NativeOperatorSourceBuildCommand> {
    let nvcc_path = toolchain
        .map(|toolchain| toolchain.nvcc.path.as_str())
        .unwrap_or_else(|| request.nvcc_path.to_str().unwrap_or("<non-utf8-nvcc>"));
    let ccbin_path = toolchain
        .map(|toolchain| toolchain.host_compiler.path.as_str())
        .unwrap_or_else(|| request.ccbin_path.to_str().unwrap_or("<non-utf8-ccbin>"));
    let ar_path = toolchain
        .map(|toolchain| toolchain.archiver.path.as_str())
        .unwrap_or_else(|| request.ar_path.to_str().unwrap_or("<non-utf8-ar>"));
    let mut commands = Vec::with_capacity(plan.translation_units.len() + 1);
    let mut object_paths = Vec::with_capacity(plan.translation_units.len());
    for (index, translation_unit) in plan.translation_units.iter().enumerate() {
        let stem = Path::new(&translation_unit.path)
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("translation_unit");
        let object_name = object_file_name(index, translation_unit);
        let object_path = objects_dir.join(object_name);
        object_paths.push(object_path.clone());
        let mut argv = vec![
            nvcc_path.to_string(),
            "-c".to_string(),
            translation_unit.path.clone(),
            "-o".to_string(),
            object_path.display().to_string(),
            architecture_argument.to_string(),
            "-ccbin".to_string(),
            ccbin_path.to_string(),
        ];
        argv.extend(plan.include_dirs.iter().map(|path| format!("-I{path}")));
        argv.extend(plan.defines.iter().map(|define| format!("-D{define}")));
        argv.extend(nvcc_policy_flags(&plan.nvcc_policy));
        argv.push("--threads".to_string());
        argv.push(request.nvcc_threads.to_string());
        commands.push(NativeOperatorSourceBuildCommand {
            translation_unit: Some(translation_unit.path.clone()),
            working_directory: source_root.display().to_string(),
            argv,
            object_file: Some(object_path.display().to_string()),
            stdout_log: relative_log(logs_dir, &format!("{index:02}-{stem}.stdout.log")),
            stderr_log: relative_log(logs_dir, &format!("{index:02}-{stem}.stderr.log")),
            object_cache_key: None,
            object_cache_status: Some(if request.plan_only {
                NativeOperatorSourceObjectCacheStatus::Plan
            } else {
                NativeOperatorSourceObjectCacheStatus::Pending
            }),
            object_cache_entry: None,
            object_sha256: None,
            compiler_executed: false,
            elapsed_ms: None,
            return_code: None,
        });
    }
    let archive_path = request.output_dir.join(&plan.archive_file);
    let mut archive_argv = vec![
        ar_path.to_string(),
        "rcs".to_string(),
        archive_path.display().to_string(),
    ];
    archive_argv.extend(object_paths.iter().map(|path| path.display().to_string()));
    commands.push(NativeOperatorSourceBuildCommand {
        translation_unit: None,
        working_directory: source_root.display().to_string(),
        argv: archive_argv,
        object_file: None,
        stdout_log: relative_log(logs_dir, "archive.stdout.log"),
        stderr_log: relative_log(logs_dir, "archive.stderr.log"),
        object_cache_key: None,
        object_cache_status: None,
        object_cache_entry: None,
        object_sha256: None,
        compiler_executed: false,
        elapsed_ms: None,
        return_code: None,
    });
    commands
}

fn run_logged_command(
    argv: &[String],
    stdout_path: &Path,
    stderr_path: &Path,
    working_directory: &str,
    effective_environment: &BTreeMap<String, String>,
) -> Result<ExitStatus> {
    let (program, args) = argv.split_first().ok_or_else(|| {
        NativeOperatorBuilderError::Invalid("source build command is empty".to_string())
    })?;
    let stdout = append_command_file(stdout_path)?;
    let stderr = append_command_file(stderr_path)?;
    Command::new(program)
        .args(args)
        .current_dir(working_directory)
        .env_clear()
        .envs(effective_environment)
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .status()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: PathBuf::from(program),
            source,
        })
}

fn append_command_file(path: &Path) -> Result<fs::File> {
    let mut file = OpenOptions::new()
        .append(true)
        .open(path)
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    file.write_all(b"execution-start\n")
        .and_then(|()| file.flush())
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(file)
}

fn write_command_stream(path: &Path, stream: &str, argv: &[String], payload: &[u8]) -> Result<()> {
    let command =
        serde_json::to_string(argv).map_err(|source| NativeOperatorBuilderError::Json {
            path: path.to_path_buf(),
            source,
        })?;
    let mut bytes = format!("stream={stream}\nargv={command}\n").into_bytes();
    bytes.extend_from_slice(payload);
    if !payload.ends_with(b"\n") {
        bytes.push(b'\n');
    }
    fs::write(path, bytes).map_err(|source| NativeOperatorBuilderError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn append_command_stream(path: &Path, payload: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .append(true)
        .open(path)
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    file.write_all(payload)
        .and_then(|()| {
            if payload.ends_with(b"\n") {
                Ok(())
            } else {
                file.write_all(b"\n")
            }
        })
        .and_then(|()| file.flush())
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: path.to_path_buf(),
            source,
        })
}

fn reject_source_build<T>(
    receipt_path: &Path,
    receipt: &mut NativeOperatorSourceBuildReceipt,
    reason: String,
) -> Result<T> {
    receipt.status = NativeOperatorSourceBuildStatus::Reject;
    receipt.failure_class = Some(reason.clone());
    write_json(receipt_path, receipt)?;
    Err(NativeOperatorBuilderError::SourceBuildRejected {
        receipt_path: receipt_path.to_path_buf(),
        reason,
    })
}

fn architecture_argument(
    architecture: NativeOperatorCudaArchitecture,
    compute_capability: &str,
) -> String {
    match architecture {
        NativeOperatorCudaArchitecture::DeviceComputeCapability => {
            format!("-arch={compute_capability}")
        }
        NativeOperatorCudaArchitecture::Compute80Ptx => "-arch=compute_80".to_string(),
    }
}

fn validate_compute_capability(value: &str) -> Result<()> {
    if value.len() >= 5
        && value.starts_with("sm_")
        && value[3..].bytes().all(|byte| byte.is_ascii_digit())
    {
        Ok(())
    } else {
        Err(NativeOperatorBuilderError::Invalid(
            "compute_capability must use sm_<digits> form".to_string(),
        ))
    }
}

fn is_git_oid(value: &str) -> bool {
    matches!(value.len(), 40 | 64)
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn safe_component(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn relative_log(logs_dir: &Path, file: &str) -> String {
    Path::new("logs")
        .join(logs_dir.join(file).file_name().expect("log file name"))
        .to_string_lossy()
        .replace('\\', "/")
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

fn millis(duration: std::time::Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use super::*;

    fn definition() -> NativeOperatorSourceDefinition {
        NativeOperatorSourceDefinition {
            schema_version: NATIVE_OPERATOR_SOURCE_DEFINITION_SCHEMA_VERSION,
            operator: CudaNativeBuildUnit::Marlin.artifact_operator().to_string(),
            source_package_kind: "ferrum-native-source-bundle".to_string(),
            source_package_revision: "fixture".to_string(),
            upstream_sources: vec![NativeOperatorUpstreamSource {
                repository: "https://example.invalid/marlin.git".to_string(),
                revision: "abc123".to_string(),
                license: "Apache-2.0".to_string(),
            }],
            translation_units: vec!["kernels/marlin.cu".to_string()],
            headers: vec!["kernels/marlin.h".to_string()],
            include_dirs: vec!["kernels".to_string()],
            defines: vec!["FERRUM_FIXTURE=1".to_string()],
            nvcc_policy: NativeOperatorNvccPolicy {
                cpp_standard: NativeOperatorCppStandard::Cpp17,
                optimization: NativeOperatorOptimization::O3,
                use_fast_math: false,
                relaxed_constexpr: false,
                extended_lambda: false,
                host_position_independent_code: true,
                host_default_visibility: false,
            },
            architecture: NativeOperatorCudaArchitecture::Compute80Ptx,
            archive_file: "libmarlin.a".to_string(),
        }
    }

    fn write_fixture(root: &Path) -> (PathBuf, PathBuf) {
        let source_root = root.join("source");
        fs::create_dir_all(source_root.join("kernels")).unwrap();
        fs::write(
            source_root.join("kernels/marlin.cu"),
            "int marlin_cuda(void) { return 0; }\n",
        )
        .unwrap();
        fs::write(source_root.join("kernels/marlin.h"), "#define MARLIN 1\n").unwrap();
        let definition_path = root.join("source-definition.json");
        write_json(&definition_path, &definition()).unwrap();
        (source_root, definition_path)
    }

    fn write_fake_nvcc(root: &Path) -> (PathBuf, PathBuf) {
        let path = root.join("fake-nvcc");
        let counter = root.join("fake-nvcc-compile-count");
        fs::write(
            &path,
            format!(
                "#!/bin/sh\n\
             if [ \"$1\" = \"--version\" ]; then echo 'fake nvcc 12.4'; exit 0; fi\n\
             src=''\n\
             out=''\n\
             while [ \"$#\" -gt 0 ]; do\n\
               case \"$1\" in\n\
                 -c) src=\"$2\"; shift 2 ;;\n\
                 -o) out=\"$2\"; shift 2 ;;\n\
                 *) shift ;;\n\
               esac\n\
             done\n\
             build_dir=$(dirname \"$(dirname \"$out\")\")\n\
             receipt=\"$build_dir/source-build.receipt.json\"\n\
             test -s \"$receipt\"\n\
             grep -q '\"status\": \"reject\"' \"$receipt\"\n\
             grep -q '\"failure_class\": \"build_incomplete\"' \"$receipt\"\n\
             printf 'compile\\n' >> '{}'\n\
             exec /usr/bin/cc -x c -c \"$src\" -o \"$out\"\n",
                counter.display()
            ),
        )
        .unwrap();
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&path, permissions).unwrap();
        (path, counter)
    }

    #[test]
    fn locks_self_contained_translation_unit_without_auxiliary_inputs() {
        let root = tempfile::tempdir().unwrap();
        let (source_root, _) = write_fixture(root.path());
        let mut definition = definition();
        definition.headers.clear();
        definition.include_dirs.clear();
        let definition_path = root.path().join("self-contained-definition.json");
        write_json(&definition_path, &definition).unwrap();
        let plan_path = root.path().join("source-build.plan.json");

        let plan =
            lock_native_operator_source_definition(&definition_path, &source_root, &plan_path)
                .unwrap();

        assert!(plan.headers.is_empty());
        assert!(plan.include_dirs.is_empty());
        assert_eq!(plan.translation_units.len(), 1);
    }

    #[test]
    fn locks_files_and_rejects_source_drift_before_rendering_commands() {
        let root = tempfile::tempdir().unwrap();
        let (source_root, definition_path) = write_fixture(root.path());
        let plan_path = root.path().join("source-build.plan.json");
        lock_native_operator_source_definition(&definition_path, &source_root, &plan_path).unwrap();
        fs::write(source_root.join("kernels/marlin.h"), "#define MARLIN 2\n").unwrap();

        let error = run_native_operator_source_build(&NativeOperatorSourceBuildRequest {
            plan_path,
            source_root,
            output_dir: root.path().join("build"),
            compute_capability: "sm_89".to_string(),
            builder_sha: "7".repeat(40),
            nvcc_path: PathBuf::from("/missing/nvcc"),
            ccbin_path: PathBuf::from("/missing/c++"),
            ar_path: PathBuf::from("/missing/ar"),
            nvcc_threads: 4,
            object_cache_dir: root.path().join("object-cache"),
            plan_only: true,
        })
        .unwrap_err();

        assert!(error.to_string().contains("locked source drift"));
        assert!(!root.path().join("build").exists());
    }

    #[test]
    fn plan_only_records_exact_commands_without_requiring_cuda_tools() {
        let root = tempfile::tempdir().unwrap();
        let (source_root, definition_path) = write_fixture(root.path());
        let plan_path = root.path().join("source-build.plan.json");
        lock_native_operator_source_definition(&definition_path, &source_root, &plan_path).unwrap();

        let receipt = run_native_operator_source_build(&NativeOperatorSourceBuildRequest {
            plan_path,
            source_root,
            output_dir: root.path().join("plan"),
            compute_capability: "sm_89".to_string(),
            builder_sha: "7".repeat(40),
            nvcc_path: PathBuf::from("/missing/nvcc"),
            ccbin_path: PathBuf::from("/missing/c++"),
            ar_path: PathBuf::from("/missing/ar"),
            nvcc_threads: 4,
            object_cache_dir: root.path().join("object-cache"),
            plan_only: true,
        })
        .unwrap();

        assert_eq!(receipt.status, NativeOperatorSourceBuildStatus::Plan);
        assert_eq!(receipt.architecture_argument, "-arch=compute_80");
        assert_eq!(receipt.commands.len(), 2);
        assert!(receipt.commands[0]
            .argv
            .windows(2)
            .any(|pair| pair == ["--threads", "4"]));
        assert!(receipt.toolchain.is_none());
        assert!(receipt.commands.iter().all(|command| {
            [
                root.path().join("plan").join(&command.stdout_log),
                root.path().join("plan").join(&command.stderr_log),
            ]
            .iter()
            .all(|path| fs::metadata(path).is_ok_and(|metadata| metadata.len() > 0))
        }));
    }

    #[test]
    fn missing_toolchain_writes_reject_receipt_before_any_compiler_spawn() {
        let root = tempfile::tempdir().unwrap();
        let (source_root, definition_path) = write_fixture(root.path());
        let plan_path = root.path().join("source-build.plan.json");
        lock_native_operator_source_definition(&definition_path, &source_root, &plan_path).unwrap();
        let output_dir = root.path().join("toolchain-reject");

        let error = run_native_operator_source_build(&NativeOperatorSourceBuildRequest {
            plan_path,
            source_root,
            output_dir: output_dir.clone(),
            compute_capability: "sm_89".to_string(),
            builder_sha: "7".repeat(40),
            nvcc_path: PathBuf::from("/missing/nvcc"),
            ccbin_path: PathBuf::from("/missing/c++"),
            ar_path: PathBuf::from("/missing/ar"),
            nvcc_threads: 4,
            object_cache_dir: root.path().join("object-cache"),
            plan_only: false,
        })
        .unwrap_err();

        assert!(matches!(
            error,
            NativeOperatorBuilderError::SourceBuildRejected { .. }
        ));
        let receipt: NativeOperatorSourceBuildReceipt =
            read_json(&output_dir.join("source-build.receipt.json")).unwrap();
        assert_eq!(receipt.status, NativeOperatorSourceBuildStatus::Reject);
        assert!(receipt
            .failure_class
            .as_deref()
            .is_some_and(|failure| failure.starts_with("toolchain_preflight_failed:")));
        assert!(receipt.commands.iter().all(|command| {
            [
                output_dir.join(&command.stdout_log),
                output_dir.join(&command.stderr_log),
            ]
            .iter()
            .all(|path| fs::metadata(path).is_ok_and(|metadata| metadata.len() > 0))
        }));
    }

    #[test]
    fn bounded_fixture_build_writes_pass_receipt_and_archive_hash() {
        let root = tempfile::tempdir().unwrap();
        let (source_root, definition_path) = write_fixture(root.path());
        let plan_path = root.path().join("source-build.plan.json");
        lock_native_operator_source_definition(&definition_path, &source_root, &plan_path).unwrap();
        let output_dir = root.path().join("build");
        let (fake_nvcc, compile_counter) = write_fake_nvcc(root.path());
        let object_cache_dir = root.path().join("object-cache");

        let receipt = run_native_operator_source_build(&NativeOperatorSourceBuildRequest {
            plan_path: plan_path.clone(),
            source_root: source_root.clone(),
            output_dir: output_dir.clone(),
            compute_capability: "sm_89".to_string(),
            builder_sha: "7".repeat(40),
            nvcc_path: fake_nvcc.clone(),
            ccbin_path: PathBuf::from("/usr/bin/cc"),
            ar_path: PathBuf::from("/usr/bin/ar"),
            nvcc_threads: 2,
            object_cache_dir: object_cache_dir.clone(),
            plan_only: false,
        })
        .unwrap();

        assert_eq!(receipt.status, NativeOperatorSourceBuildStatus::Pass);
        assert!(receipt.toolchain.is_some());
        assert!(is_sha256_digest(receipt.archive_sha256.as_deref().unwrap()));
        assert!(output_dir.join("libmarlin.a").is_file());
        assert!(output_dir.join("source-build.receipt.json").is_file());
        assert_eq!(receipt.compiled_translation_units, ["kernels/marlin.cu"]);
        assert!(receipt.cache_hit_translation_units.is_empty());
        assert!(receipt.commands.iter().all(|command| {
            [
                output_dir.join(&command.stdout_log),
                output_dir.join(&command.stderr_log),
            ]
            .iter()
            .all(|path| {
                fs::read_to_string(path).is_ok_and(|content| content.contains("execution-start"))
            })
        }));

        let cached_output_dir = root.path().join("cached-build");
        let cached_receipt = run_native_operator_source_build(&NativeOperatorSourceBuildRequest {
            plan_path,
            source_root,
            output_dir: cached_output_dir.clone(),
            compute_capability: "sm_89".to_string(),
            builder_sha: "8".repeat(40),
            nvcc_path: fake_nvcc,
            ccbin_path: PathBuf::from("/usr/bin/cc"),
            ar_path: PathBuf::from("/usr/bin/ar"),
            nvcc_threads: 8,
            object_cache_dir,
            plan_only: false,
        })
        .unwrap();

        assert!(cached_receipt.compiled_translation_units.is_empty());
        assert_eq!(
            cached_receipt.cache_hit_translation_units,
            ["kernels/marlin.cu"]
        );
        assert!(!cached_receipt.commands[0].compiler_executed);
        assert_eq!(
            cached_receipt.commands[0].object_cache_status,
            Some(NativeOperatorSourceObjectCacheStatus::Hit)
        );
        assert_eq!(
            fs::read_to_string(compile_counter).unwrap().lines().count(),
            1
        );
        assert_eq!(
            receipt.archive_sha256, cached_receipt.archive_sha256,
            "worker-count changes must not change the object or archive"
        );
        assert_eq!(
            receipt.inputs_sha256, cached_receipt.inputs_sha256,
            "worker-count and provenance commit changes are not output-content inputs"
        );
    }

    #[test]
    fn changing_one_translation_unit_recompiles_only_that_unit() {
        let root = tempfile::tempdir().unwrap();
        let (source_root, _) = write_fixture(root.path());
        fs::write(
            source_root.join("kernels/other.cu"),
            "int other_cuda(void) { return 1; }\n",
        )
        .unwrap();
        let mut source_definition = definition();
        source_definition.translation_units = vec![
            "kernels/marlin.cu".to_string(),
            "kernels/other.cu".to_string(),
        ];
        let definition_path = root.path().join("two-tu-definition.json");
        write_json(&definition_path, &source_definition).unwrap();
        let first_plan_path = root.path().join("first.plan.json");
        lock_native_operator_source_definition(&definition_path, &source_root, &first_plan_path)
            .unwrap();
        let (fake_nvcc, compile_counter) = write_fake_nvcc(root.path());
        let object_cache_dir = root.path().join("object-cache");

        let first_receipt = run_native_operator_source_build(&NativeOperatorSourceBuildRequest {
            plan_path: first_plan_path,
            source_root: source_root.clone(),
            output_dir: root.path().join("first-build"),
            compute_capability: "sm_89".to_string(),
            builder_sha: "7".repeat(40),
            nvcc_path: fake_nvcc.clone(),
            ccbin_path: PathBuf::from("/usr/bin/cc"),
            ar_path: PathBuf::from("/usr/bin/ar"),
            nvcc_threads: 2,
            object_cache_dir: object_cache_dir.clone(),
            plan_only: false,
        })
        .unwrap();
        assert_eq!(first_receipt.compiled_translation_units.len(), 2);
        assert!(first_receipt.cache_hit_translation_units.is_empty());

        fs::write(
            source_root.join("kernels/other.cu"),
            "int other_cuda(void) { return 2; }\n",
        )
        .unwrap();
        let second_plan_path = root.path().join("second.plan.json");
        lock_native_operator_source_definition(&definition_path, &source_root, &second_plan_path)
            .unwrap();
        let second_receipt = run_native_operator_source_build(&NativeOperatorSourceBuildRequest {
            plan_path: second_plan_path,
            source_root,
            output_dir: root.path().join("second-build"),
            compute_capability: "sm_89".to_string(),
            builder_sha: "8".repeat(40),
            nvcc_path: fake_nvcc,
            ccbin_path: PathBuf::from("/usr/bin/cc"),
            ar_path: PathBuf::from("/usr/bin/ar"),
            nvcc_threads: 4,
            object_cache_dir,
            plan_only: false,
        })
        .unwrap();

        assert_eq!(
            second_receipt.compiled_translation_units,
            ["kernels/other.cu"]
        );
        assert_eq!(
            second_receipt.cache_hit_translation_units,
            ["kernels/marlin.cu"]
        );
        assert_eq!(
            fs::read_to_string(compile_counter).unwrap().lines().count(),
            3
        );
    }
}
