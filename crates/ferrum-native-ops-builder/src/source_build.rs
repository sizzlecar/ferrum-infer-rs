//! Locked, independently runnable native source-build plans.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use ferrum_native_ops::{
    CudaNativeBuildUnit, NativeBuildArtifactCache, NativeBuildArtifactLookup,
    NativeBuildArtifactSpec,
};
use ferrum_types::{is_sha256_digest, NativeOperatorBackend, NativeOperatorSourcePackage};
use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;

use super::{
    read_json, require_file, sha256_bytes, sha256_file, symbol_slug, validate_relative_path,
    write_json, NativeOperatorBuilderError, NativeOperatorEvidenceFile, Result,
};

pub const NATIVE_OPERATOR_SOURCE_DEFINITION_SCHEMA_VERSION: u32 = 3;
pub const NATIVE_OPERATOR_SOURCE_BUILD_PLAN_SCHEMA_VERSION: u32 = 3;
pub const NATIVE_OPERATOR_SOURCE_BUILD_RECEIPT_SCHEMA_VERSION: u32 = 7;
pub const NATIVE_OPERATOR_SOURCE_OBJECT_BUILD_CONTRACT_VERSION: u32 = 7;
pub const NATIVE_OPERATOR_CUDA_TOOLKIT_MANIFEST_SCHEMA_VERSION: u32 = 1;
pub const NATIVE_OPERATOR_HOST_TOOLCHAIN_MANIFEST_SCHEMA_VERSION: u32 = 2;
pub const NATIVE_OPERATOR_OBJECT_DEPENDENCY_PROOF_SCHEMA_VERSION: u32 = 3;
const REQUIRED_CUDA_TOOLKIT_FILES: [&str; 6] = [
    "bin/bin2c",
    "bin/cudafe++",
    "bin/fatbinary",
    "bin/nvcc",
    "bin/nvlink",
    "bin/ptxas",
];
const REQUIRED_CUDA_TOOLKIT_SCOPES: [&str; 4] =
    ["bin/crt", "include", "nvvm/bin", "nvvm/libdevice"];
const HOST_TOOLCHAIN_PROGRAMS: [&str; 5] = ["as", "cc1", "cc1plus", "collect2", "ld"];
const MAX_HOST_TOOLCHAIN_FILES: usize = 250_000;
const MAX_DEPFILE_BYTES: usize = 16 * 1024 * 1024;
const MAX_DEPFILE_DEPENDENCIES: usize = 250_000;
const MAX_DEPFILE_WORD_BYTES: usize = 16 * 1024;
const MAX_DEPENDENCY_PROOF_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorSourceDefinition {
    pub schema_version: u32,
    pub operator: String,
    pub source_package_kind: String,
    pub source_package_revision: String,
    pub upstream_sources: Vec<NativeOperatorUpstreamSource>,
    pub translation_units: Vec<String>,
    pub headers: Vec<String>,
    pub dependency_closures: Vec<NativeOperatorTranslationUnitDependencies>,
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
pub struct NativeOperatorTranslationUnitDependencies {
    pub translation_unit: String,
    pub headers: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorSourceBuildPlan {
    pub schema_version: u32,
    pub operator: String,
    pub source_package: NativeOperatorSourcePackage,
    pub upstream_sources: Vec<NativeOperatorUpstreamSource>,
    pub translation_units: Vec<NativeOperatorSourceFileLock>,
    pub headers: Vec<NativeOperatorSourceFileLock>,
    pub dependency_closures: Vec<NativeOperatorTranslationUnitDependencyLock>,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorTranslationUnitDependencyLock {
    pub translation_unit: String,
    pub headers: Vec<NativeOperatorSourceFileLock>,
    pub closure_sha256: String,
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
    pub cuda_toolkit_root: PathBuf,
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
    pub static_identity: NativeOperatorSourceBuildStaticToolchain,
    pub miss_probe: Option<NativeOperatorSourceBuildToolchainProbe>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorSourceBuildStaticToolchain {
    pub backend: NativeOperatorBackend,
    pub compiler_driver: NativeOperatorSourceCompilerDriver,
    pub cuda_toolkit: NativeOperatorCudaToolkitIdentity,
    pub host_toolchain: NativeOperatorHostToolchainIdentity,
    pub archiver: NativeOperatorToolFileIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeOperatorSourceCompilerDriver {
    CudaNvcc,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorCudaToolkitIdentity {
    pub canonical_root: String,
    pub invocation_root: String,
    pub release_version: String,
    pub nvcc: NativeOperatorToolFileIdentity,
    pub manifest: NativeOperatorEvidenceFile,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorToolFileIdentity {
    pub path: String,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorHostToolchainIdentity {
    pub compiler: NativeOperatorToolFileIdentity,
    pub compiler_version: String,
    pub target: String,
    pub manifest: NativeOperatorEvidenceFile,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorHostToolchainManifest {
    pub schema_version: u32,
    pub compiler: NativeOperatorToolFileIdentity,
    pub compiler_version: String,
    pub target: String,
    pub executable_inputs: Vec<NativeOperatorToolFileIdentity>,
    pub include_roots: Vec<String>,
    pub include_probe_sha256: String,
    pub driver_probe_sha256: String,
    pub discovery_roots: Vec<String>,
    pub files: Vec<NativeOperatorHostToolchainFileIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorHostToolchainFileIdentity {
    pub logical_path: String,
    pub resolved_path: String,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorSourceBuildToolchainProbe {
    pub nvcc_version: String,
    pub host_compiler_version: String,
    pub host_target: String,
    pub archiver_version: String,
    pub probed_for_misses: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorCudaToolkitManifest {
    pub schema_version: u32,
    pub canonical_root: String,
    pub entries: Vec<NativeOperatorCudaToolkitFileIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorCudaToolkitFileIdentity {
    pub logical_path: String,
    pub resolved_path: String,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorToolIdentity {
    pub path: String,
    pub sha256: String,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorObjectIdentity {
    pub format: NativeOperatorObjectFormat,
    pub class_bits: u8,
    pub endianness: NativeOperatorObjectEndianness,
    pub machine: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeOperatorObjectFormat {
    Elf,
    MachO,
    Coff,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeOperatorObjectEndianness {
    Little,
    Big,
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
    pub object_size_bytes: Option<u64>,
    pub object_identity: Option<NativeOperatorObjectIdentity>,
    pub dependency_closure_sha256: Option<String>,
    pub dependency_validation: Option<NativeOperatorDependencyValidation>,
    pub compiler_depfile: Option<String>,
    pub compiler_depfile_sha256: Option<String>,
    pub depfile: Option<String>,
    pub depfile_sha256: Option<String>,
    pub depfile_producer_working_directory: Option<String>,
    pub depfile_producer_object_file: Option<String>,
    pub depfile_bindings: Vec<NativeOperatorDepfileDependencyBinding>,
    pub observed_dependencies: Vec<NativeOperatorObservedDependency>,
    pub compiler_executed: bool,
    pub elapsed_ms: Option<u64>,
    pub return_code: Option<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeOperatorDependencyValidation {
    Plan,
    Pending,
    CacheProof,
    Depfile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeOperatorDependencyDomain {
    Source,
    BackendToolchain,
    HostToolchain,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NativeOperatorObservedDependency {
    pub domain: NativeOperatorDependencyDomain,
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorDepfileDependencyBinding {
    pub producer_path: String,
    pub portable_path: String,
    pub dependency: NativeOperatorObservedDependency,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorObjectDependencyProof {
    pub schema_version: u32,
    pub object_cache_key: String,
    pub object_sha256: String,
    pub dependency_closure_sha256: String,
    pub dependency_set_sha256: String,
    pub compiler_depfile_sha256: String,
    pub depfile_sha256: String,
    pub producer_working_directory: String,
    pub producer_object_file: String,
    pub depfile_bindings: Vec<NativeOperatorDepfileDependencyBinding>,
    pub observed_dependencies: Vec<NativeOperatorObservedDependency>,
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
struct NativeOperatorDependencyClosureIdentity<'a> {
    translation_unit: &'a NativeOperatorSourceFileLock,
    headers: &'a [NativeOperatorSourceFileLock],
}

#[derive(Debug, Serialize)]
struct NativeOperatorBuildInputIdentity<'a> {
    plan_sha256: &'a str,
    source_package_sha256: &'a str,
    builder_contract_version: u32,
    architecture_argument: &'a str,
    effective_environment: &'a BTreeMap<String, String>,
    toolchain: Option<&'a NativeOperatorSourceBuildStaticToolchain>,
}

#[derive(Debug, Serialize)]
struct NativeOperatorObjectInputIdentity<'a> {
    schema_version: u32,
    operator: &'a str,
    translation_unit: &'a NativeOperatorSourceFileLock,
    dependency_closure_sha256: &'a str,
    headers: &'a [NativeOperatorSourceFileLock],
    include_dirs: &'a [String],
    defines: &'a [String],
    nvcc_policy: &'a NativeOperatorNvccPolicy,
    architecture_argument: &'a str,
    builder_contract_version: u32,
    effective_environment: &'a BTreeMap<String, String>,
    toolchain: &'a NativeOperatorSourceBuildStaticToolchain,
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
    let translation_unit_by_path = translation_units
        .iter()
        .map(|locked| (locked.path.as_str(), locked))
        .collect::<BTreeMap<_, _>>();
    let header_by_path = headers
        .iter()
        .map(|locked| (locked.path.as_str(), locked))
        .collect::<BTreeMap<_, _>>();
    let dependency_closures = definition
        .dependency_closures
        .iter()
        .map(|closure| {
            let translation_unit = translation_unit_by_path
                .get(closure.translation_unit.as_str())
                .copied()
                .ok_or_else(|| {
                    NativeOperatorBuilderError::Invalid(format!(
                        "dependency closure references unknown translation unit: {}",
                        closure.translation_unit
                    ))
                })?;
            let locked_headers = closure
                .headers
                .iter()
                .map(|path| {
                    header_by_path
                        .get(path.as_str())
                        .copied()
                        .cloned()
                        .ok_or_else(|| {
                            NativeOperatorBuilderError::Invalid(format!(
                                "dependency closure for {} references unknown header: {path}",
                                closure.translation_unit
                            ))
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            let closure_sha256 =
                dependency_closure_sha256(translation_unit, &locked_headers, output_plan_path)?;
            Ok(NativeOperatorTranslationUnitDependencyLock {
                translation_unit: closure.translation_unit.clone(),
                headers: locked_headers,
                closure_sha256,
            })
        })
        .collect::<Result<Vec<_>>>()?;
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
        dependency_closures,
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
    if [
        &request.nvcc_path,
        &request.ccbin_path,
        &request.ar_path,
        &request.cuda_toolkit_root,
    ]
    .iter()
    .any(|path| !path.is_absolute())
    {
        return Err(NativeOperatorBuilderError::Invalid(
            "source build tool paths and cuda_toolkit_root must be absolute".to_string(),
        ));
    }
    validate_compute_capability(&request.compute_capability)?;
    if request.nvcc_threads == 0 {
        return Err(NativeOperatorBuilderError::Invalid(
            "nvcc_threads must be greater than zero".to_string(),
        ));
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
    let depfiles_dir = request.output_dir.join("depfiles");
    fs::create_dir_all(&logs_dir).map_err(|source| NativeOperatorBuilderError::Io {
        path: logs_dir.clone(),
        source,
    })?;
    if !request.plan_only {
        fs::create_dir_all(&objects_dir).map_err(|source| NativeOperatorBuilderError::Io {
            path: objects_dir.clone(),
            source,
        })?;
        fs::create_dir_all(&depfiles_dir).map_err(|source| NativeOperatorBuilderError::Io {
            path: depfiles_dir.clone(),
            source,
        })?;
    }

    let (toolchain, toolchain_failure) = if request.plan_only {
        (None, None)
    } else {
        match resolve_static_toolchain(request) {
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
        toolchain
            .as_ref()
            .map(|toolchain| &toolchain.static_identity),
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
    let toolchain_dependency_scope = match load_toolchain_dependency_scope(
        &request.output_dir,
        &receipt
            .toolchain
            .as_ref()
            .expect("actual source build has a resolved toolchain")
            .static_identity,
    ) {
        Ok(scope) => scope,
        Err(error) => {
            receipt.elapsed_ms = millis(started.elapsed());
            return reject_source_build(
                &receipt_path,
                &mut receipt,
                format!("toolchain_dependency_scope_failed:{error}"),
            );
        }
    };
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
            .map(|toolchain| &toolchain.static_identity)
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

    let mut expected_object_identity: Option<NativeOperatorObjectIdentity> = None;
    let mut miss_indices = Vec::new();
    for index in 0..plan.translation_units.len() {
        let translation_unit = &plan.translation_units[index];
        let object_path = PathBuf::from(
            commands[index]
                .object_file
                .as_deref()
                .expect("translation-unit command has an object file"),
        );
        let lookup_started = Instant::now();
        match object_cache.restore(&object_specs[index], &object_path) {
            Ok(NativeBuildArtifactLookup::Hit(cache_receipt)) => {
                let compiler_depfile_relative = commands[index]
                    .compiler_depfile
                    .as_deref()
                    .expect("translation-unit command has compiler depfile output")
                    .to_string();
                let compiler_depfile_path = request.output_dir.join(&compiler_depfile_relative);
                let depfile_relative = commands[index]
                    .depfile
                    .as_deref()
                    .expect("translation-unit command has depfile output")
                    .to_string();
                let depfile_path = request.output_dir.join(&depfile_relative);
                let dependency_proof = match restore_object_dependency_proof(
                    &cache_receipt.cache_entry,
                    object_specs[index].input_signature_sha256(),
                    &cache_receipt.artifact_sha256,
                    &plan.dependency_closures[index],
                    translation_unit,
                    &object_path,
                    &compiler_depfile_path,
                    &depfile_path,
                    &toolchain_dependency_scope,
                ) {
                    Ok(Some(proof)) => proof,
                    Ok(None) => {
                        fs::remove_file(&object_path).map_err(|source| {
                            NativeOperatorBuilderError::Io {
                                path: object_path.clone(),
                                source,
                            }
                        })?;
                        commands[index].object_cache_status =
                            Some(NativeOperatorSourceObjectCacheStatus::Miss);
                        append_command_stream(
                            &request.output_dir.join(&commands[index].stdout_log),
                            b"object-cache-miss: dependency-proof-absent\n",
                        )?;
                        miss_indices.push(index);
                        continue;
                    }
                    Err(error) => {
                        commands[index].object_cache_status =
                            Some(NativeOperatorSourceObjectCacheStatus::Rejected);
                        receipt.commands = commands.clone();
                        receipt.elapsed_ms = millis(started.elapsed());
                        return reject_source_build(
                            &receipt_path,
                            &mut receipt,
                            format!("cached_dependency_proof_failed:{index}:{error}"),
                        );
                    }
                };
                commands[index].object_cache_status =
                    Some(NativeOperatorSourceObjectCacheStatus::Hit);
                commands[index].object_cache_entry =
                    Some(cache_receipt.cache_entry.display().to_string());
                commands[index].object_sha256 = Some(cache_receipt.artifact_sha256);
                let object_size_bytes = match native_object_size(&object_path) {
                    Ok(size_bytes) => size_bytes,
                    Err(error) => {
                        commands[index].object_cache_status =
                            Some(NativeOperatorSourceObjectCacheStatus::Rejected);
                        receipt.commands = commands.clone();
                        receipt.elapsed_ms = millis(started.elapsed());
                        return reject_source_build(
                            &receipt_path,
                            &mut receipt,
                            format!("cached_object_size_failed:{index}:{error}"),
                        );
                    }
                };
                commands[index].object_size_bytes = Some(object_size_bytes);
                let object_identity = match native_object_identity_file(&object_path) {
                    Ok(identity) => identity,
                    Err(error) => {
                        commands[index].object_cache_status =
                            Some(NativeOperatorSourceObjectCacheStatus::Rejected);
                        receipt.commands = commands.clone();
                        receipt.elapsed_ms = millis(started.elapsed());
                        return reject_source_build(
                            &receipt_path,
                            &mut receipt,
                            format!("cached_object_identity_failed:{index}:{error}"),
                        );
                    }
                };
                if expected_object_identity
                    .as_ref()
                    .is_some_and(|expected| expected != &object_identity)
                {
                    commands[index].object_cache_status =
                        Some(NativeOperatorSourceObjectCacheStatus::Rejected);
                    receipt.commands = commands.clone();
                    receipt.elapsed_ms = millis(started.elapsed());
                    return reject_source_build(
                        &receipt_path,
                        &mut receipt,
                        format!("cached_object_target_mismatch:{index}"),
                    );
                }
                expected_object_identity.get_or_insert_with(|| object_identity.clone());
                commands[index].object_identity = Some(object_identity);
                commands[index].dependency_validation =
                    Some(NativeOperatorDependencyValidation::CacheProof);
                commands[index].compiler_depfile_sha256 =
                    Some(dependency_proof.compiler_depfile_sha256);
                commands[index].depfile_sha256 = Some(dependency_proof.depfile_sha256);
                commands[index].depfile_producer_working_directory =
                    Some(dependency_proof.producer_working_directory);
                commands[index].depfile_producer_object_file =
                    Some(dependency_proof.producer_object_file);
                commands[index].depfile_bindings = dependency_proof.depfile_bindings;
                commands[index].observed_dependencies = dependency_proof.observed_dependencies;
                commands[index].elapsed_ms = Some(millis(lookup_started.elapsed()));
                append_command_stream(
                    &request.output_dir.join(&commands[index].stdout_log),
                    b"object-cache-hit: compiler and toolchain probes were not executed\n",
                )?;
                receipt
                    .cache_hit_translation_units
                    .push(translation_unit.path.clone());
            }
            Ok(NativeBuildArtifactLookup::Miss { reason }) => {
                commands[index].object_cache_status =
                    Some(NativeOperatorSourceObjectCacheStatus::Miss);
                append_command_stream(
                    &request.output_dir.join(&commands[index].stdout_log),
                    format!("object-cache-miss: {reason}\n").as_bytes(),
                )?;
                miss_indices.push(index);
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
    }
    receipt.commands = commands.clone();
    receipt.elapsed_ms = millis(started.elapsed());
    write_json(&receipt_path, &receipt)?;

    if !miss_indices.is_empty() {
        let missed_translation_units = miss_indices
            .iter()
            .map(|index| plan.translation_units[*index].path.clone())
            .collect::<Vec<_>>();
        let static_identity = receipt
            .toolchain
            .as_ref()
            .expect("actual source build has a resolved toolchain")
            .static_identity
            .clone();
        let probe = match probe_source_toolchain(&static_identity, missed_translation_units) {
            Ok(probe) => probe,
            Err(error) => {
                receipt.commands = commands.clone();
                receipt.elapsed_ms = millis(started.elapsed());
                return reject_source_build(
                    &receipt_path,
                    &mut receipt,
                    format!("toolchain_miss_probe_failed:{error}"),
                );
            }
        };
        receipt
            .toolchain
            .as_mut()
            .expect("actual source build has a resolved toolchain")
            .miss_probe = Some(probe);
        write_json(&receipt_path, &receipt)?;
    }

    for index in miss_indices {
        let translation_unit = &plan.translation_units[index];
        let object_path = PathBuf::from(
            commands[index]
                .object_file
                .as_deref()
                .expect("translation-unit command has an object file"),
        );
        let command_started = Instant::now();
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
        let compiler_depfile_relative = commands[index]
            .compiler_depfile
            .as_deref()
            .expect("cache-miss command has a compiler depfile")
            .to_string();
        let compiler_depfile_path = request.output_dir.join(&compiler_depfile_relative);
        let depfile_relative = commands[index]
            .depfile
            .as_deref()
            .expect("cache-miss command has a portable depfile")
            .to_string();
        let depfile_path = request.output_dir.join(&depfile_relative);
        let validated_depfile = match validate_translation_unit_depfile(
            &compiler_depfile_path,
            &depfile_path,
            &object_path,
            &canonical_root,
            translation_unit,
            &plan.dependency_closures[index],
            &toolchain_dependency_scope,
        ) {
            Ok(dependencies) => dependencies,
            Err(error) => {
                commands[index].object_cache_status =
                    Some(NativeOperatorSourceObjectCacheStatus::Rejected);
                receipt.commands = commands.clone();
                receipt.elapsed_ms = millis(started.elapsed());
                return reject_source_build(
                    &receipt_path,
                    &mut receipt,
                    format!("dependency_validation_failed:{index}:{error}"),
                );
            }
        };
        commands[index].compiler_depfile_sha256 = Some(validated_depfile.compiler_sha256.clone());
        commands[index].depfile_sha256 = Some(validated_depfile.portable_sha256.clone());
        commands[index].depfile_producer_working_directory =
            Some(commands[index].working_directory.clone());
        commands[index].depfile_producer_object_file = commands[index].object_file.clone();
        commands[index].depfile_bindings = validated_depfile.bindings.clone();
        commands[index].observed_dependencies = validated_depfile.observed_dependencies.clone();
        commands[index].dependency_validation = Some(NativeOperatorDependencyValidation::Depfile);
        let object_identity = match native_object_identity_file(&object_path) {
            Ok(identity) => identity,
            Err(error) => {
                commands[index].object_cache_status =
                    Some(NativeOperatorSourceObjectCacheStatus::Rejected);
                receipt.commands = commands.clone();
                receipt.elapsed_ms = millis(started.elapsed());
                return reject_source_build(
                    &receipt_path,
                    &mut receipt,
                    format!("compiled_object_identity_failed:{index}:{error}"),
                );
            }
        };
        if expected_object_identity
            .as_ref()
            .is_some_and(|expected| expected != &object_identity)
        {
            commands[index].object_cache_status =
                Some(NativeOperatorSourceObjectCacheStatus::Rejected);
            receipt.commands = commands.clone();
            receipt.elapsed_ms = millis(started.elapsed());
            return reject_source_build(
                &receipt_path,
                &mut receipt,
                format!("compiled_object_target_mismatch:{index}"),
            );
        }
        expected_object_identity.get_or_insert_with(|| object_identity.clone());
        commands[index].object_identity = Some(object_identity);
        let object_size_bytes = match native_object_size(&object_path) {
            Ok(size_bytes) => size_bytes,
            Err(error) => {
                commands[index].object_cache_status =
                    Some(NativeOperatorSourceObjectCacheStatus::Rejected);
                receipt.commands = commands.clone();
                receipt.elapsed_ms = millis(started.elapsed());
                return reject_source_build(
                    &receipt_path,
                    &mut receipt,
                    format!("compiled_object_size_failed:{index}:{error}"),
                );
            }
        };
        commands[index].object_size_bytes = Some(object_size_bytes);
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
        if let Err(error) = publish_object_dependency_proof(
            &cache_receipt.cache_entry,
            object_specs[index].input_signature_sha256(),
            &cache_receipt.artifact_sha256,
            translation_unit,
            &plan.dependency_closures[index],
            &validated_depfile.compiler_raw,
            &validated_depfile.compiler_sha256,
            &validated_depfile.portable_raw,
            &validated_depfile.portable_sha256,
            commands[index]
                .depfile_producer_working_directory
                .as_deref()
                .expect("compiled depfile records its producer working directory"),
            commands[index]
                .depfile_producer_object_file
                .as_deref()
                .expect("compiled depfile records its producer object"),
            &commands[index].depfile_bindings,
            &commands[index].observed_dependencies,
            &toolchain_dependency_scope,
        ) {
            commands[index].object_cache_status =
                Some(NativeOperatorSourceObjectCacheStatus::Rejected);
            receipt.commands = commands.clone();
            receipt.elapsed_ms = millis(started.elapsed());
            return reject_source_build(
                &receipt_path,
                &mut receipt,
                format!("object_dependency_proof_publish_failed:{index}:{error}"),
            );
        }
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
    let static_identity = receipt
        .toolchain
        .as_ref()
        .expect("actual source build has a resolved toolchain")
        .static_identity
        .clone();
    if let Err(error) = validate_tool_file_unchanged(&static_identity.archiver) {
        return reject_source_build(
            &receipt_path,
            &mut receipt,
            format!("archiver_changed_during_build:{error}"),
        );
    }
    if receipt
        .toolchain
        .as_ref()
        .is_some_and(|toolchain| toolchain.miss_probe.is_some())
    {
        if let Err(error) = validate_host_toolchain_unchanged(
            &static_identity.host_toolchain,
            &request.output_dir,
            &receipt.effective_environment,
        )
        .and_then(|()| validate_tool_file_unchanged(&static_identity.cuda_toolkit.nvcc))
        .and_then(|()| validate_cuda_toolkit_unchanged(&static_identity.cuda_toolkit))
        {
            return reject_source_build(
                &receipt_path,
                &mut receipt,
                format!("compiler_toolchain_changed_during_build:{error}"),
            );
        }
    }
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
    validate_definition_dependency_closures(
        &definition.translation_units,
        &definition.headers,
        &definition.dependency_closures,
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
    validate_plan_dependency_closures(plan)?;
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

fn validate_definition_dependency_closures(
    translation_units: &[String],
    headers: &[String],
    closures: &[NativeOperatorTranslationUnitDependencies],
) -> Result<()> {
    let closure_translation_units = closures
        .iter()
        .map(|closure| closure.translation_unit.as_str())
        .collect::<Vec<_>>();
    let expected_translation_units = translation_units
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    if closure_translation_units != expected_translation_units {
        return Err(NativeOperatorBuilderError::Invalid(
            "dependency_closures must contain exactly one row per translation unit in translation_units order"
                .to_string(),
        ));
    }
    let declared_headers = headers.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let mut attached_headers = BTreeSet::new();
    for closure in closures {
        require_sorted_optional_paths(
            &format!("dependency_closures[{}].headers", closure.translation_unit),
            &closure.headers,
        )?;
        for header in &closure.headers {
            if !declared_headers.contains(header.as_str()) {
                return Err(NativeOperatorBuilderError::Invalid(format!(
                    "dependency closure for {} references undeclared header: {header}",
                    closure.translation_unit
                )));
            }
            attached_headers.insert(header.as_str());
        }
    }
    if attached_headers != declared_headers {
        let missing = declared_headers
            .difference(&attached_headers)
            .copied()
            .collect::<Vec<_>>()
            .join(",");
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "every declared header must belong to at least one dependency closure; unattached={missing}"
        )));
    }
    Ok(())
}

fn validate_plan_dependency_closures(plan: &NativeOperatorSourceBuildPlan) -> Result<()> {
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
    let closures = plan
        .dependency_closures
        .iter()
        .map(|closure| NativeOperatorTranslationUnitDependencies {
            translation_unit: closure.translation_unit.clone(),
            headers: closure
                .headers
                .iter()
                .map(|header| header.path.clone())
                .collect(),
        })
        .collect::<Vec<_>>();
    validate_definition_dependency_closures(&translation_units, &headers, &closures)?;

    let translation_unit_by_path = plan
        .translation_units
        .iter()
        .map(|locked| (locked.path.as_str(), locked))
        .collect::<BTreeMap<_, _>>();
    let header_by_path = plan
        .headers
        .iter()
        .map(|locked| (locked.path.as_str(), locked))
        .collect::<BTreeMap<_, _>>();
    for closure in &plan.dependency_closures {
        let translation_unit = translation_unit_by_path
            .get(closure.translation_unit.as_str())
            .copied()
            .expect("definition-shaped closure already validated");
        for header in &closure.headers {
            let global = header_by_path
                .get(header.path.as_str())
                .copied()
                .expect("definition-shaped closure already validated");
            if global != header {
                return Err(NativeOperatorBuilderError::Invalid(format!(
                    "dependency closure header lock differs from the global lock: {}",
                    header.path
                )));
            }
        }
        if !is_sha256_digest(&closure.closure_sha256) {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "dependency closure for {} has an invalid sha256",
                closure.translation_unit
            )));
        }
        let actual = dependency_closure_sha256(
            translation_unit,
            &closure.headers,
            Path::new("<source-build-plan>"),
        )?;
        if actual != closure.closure_sha256 {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "dependency closure hash mismatch for {}: expected={} actual={actual}",
                closure.translation_unit, closure.closure_sha256
            )));
        }
    }
    Ok(())
}

fn dependency_closure_sha256(
    translation_unit: &NativeOperatorSourceFileLock,
    headers: &[NativeOperatorSourceFileLock],
    context: &Path,
) -> Result<String> {
    let identity = NativeOperatorDependencyClosureIdentity {
        translation_unit,
        headers,
    };
    let bytes =
        serde_json::to_vec(&identity).map_err(|source| NativeOperatorBuilderError::Json {
            path: context.to_path_buf(),
            source,
        })?;
    Ok(sha256_bytes(&bytes))
}

pub(crate) fn verify_source_build_receipt_against_plan(
    receipt: &NativeOperatorSourceBuildReceipt,
    receipt_root: &Path,
    plan_path: &Path,
    source_root: &Path,
) -> Result<NativeOperatorSourceBuildPlan> {
    let plan = verify_source_build_receipt_against_plan_portable(receipt, plan_path)?;
    verify_source_build_evidence(receipt, receipt_root, &plan)?;
    let canonical_source_root =
        source_root
            .canonicalize()
            .map_err(|source| NativeOperatorBuilderError::Io {
                path: source_root.to_path_buf(),
                source,
            })?;
    validate_locked_source_tree(&canonical_source_root, &plan)?;
    Ok(plan)
}

pub(crate) fn verify_source_build_receipt_against_plan_portable(
    receipt: &NativeOperatorSourceBuildReceipt,
    plan_path: &Path,
) -> Result<NativeOperatorSourceBuildPlan> {
    require_file(plan_path)?;
    let plan: NativeOperatorSourceBuildPlan = read_json(plan_path)?;
    validate_plan(&plan)?;
    let plan_sha256 = sha256_file(plan_path)?;
    if receipt.plan_sha256 != plan_sha256 {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} source-build receipt plan_sha256 mismatch: expected={plan_sha256} actual={}",
            receipt.operator, receipt.plan_sha256
        )));
    }
    if receipt.operator != plan.operator || receipt.source_package != plan.source_package {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} source-build receipt does not match its locked plan identity",
            receipt.operator
        )));
    }

    let expected_architecture =
        architecture_argument(plan.architecture, &receipt.compute_capability);
    if receipt.architecture_argument != expected_architecture {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} source-build architecture argument differs from its plan: expected={expected_architecture} actual={}",
            receipt.operator, receipt.architecture_argument
        )));
    }
    let toolchain = receipt.toolchain.as_ref().ok_or_else(|| {
        NativeOperatorBuilderError::Invalid(format!(
            "{} source-build receipt is missing toolchain provenance",
            receipt.operator
        ))
    })?;
    validate_static_toolchain_identity(&receipt.operator, &toolchain.static_identity)?;
    let static_identity = &toolchain.static_identity;
    let expected_environment = effective_environment_for_tool_paths([
        static_identity.cuda_toolkit.nvcc.path.as_str(),
        static_identity.host_toolchain.compiler.path.as_str(),
        static_identity.archiver.path.as_str(),
    ])?;
    if receipt.effective_environment != expected_environment {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} source-build effective environment differs from the deterministic policy",
            receipt.operator
        )));
    }
    let expected_inputs_sha256 = build_inputs_sha256(
        &plan_sha256,
        &plan.source_package.sha256,
        &expected_architecture,
        &expected_environment,
        Some(static_identity),
        plan_path,
    )?;
    if receipt.inputs_sha256 != expected_inputs_sha256 {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} source-build inputs_sha256 mismatch: expected={expected_inputs_sha256} actual={}",
            receipt.operator, receipt.inputs_sha256
        )));
    }
    let object_specs = build_object_cache_specs(
        &plan,
        &expected_architecture,
        static_identity,
        &expected_environment,
    )?;
    if receipt.commands.len() != plan.translation_units.len() + 1 {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} source-build command count differs from its plan",
            receipt.operator
        )));
    }
    for (index, ((translation_unit, object_spec), command)) in plan
        .translation_units
        .iter()
        .zip(object_specs.iter())
        .zip(receipt.commands.iter())
        .enumerate()
    {
        let expected_object_file = object_file_name(index, translation_unit);
        let closure = &plan.dependency_closures[index];
        if command.translation_unit.as_deref() != Some(translation_unit.path.as_str())
            || command.object_cache_key.as_deref() != Some(object_spec.input_signature_sha256())
            || command.dependency_closure_sha256.as_deref() != Some(closure.closure_sha256.as_str())
            || command
                .object_file
                .as_deref()
                .and_then(|path| Path::new(path).file_name())
                != Some(std::ffi::OsStr::new(&expected_object_file))
        {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "{} source-build object identity for {} differs from its plan",
                receipt.operator, translation_unit.path
            )));
        }
        let stem = Path::new(&translation_unit.path)
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("translation_unit");
        let expected_depfile = format!("{index:08}-{stem}.d");
        let expected_compiler_depfile = format!("{index:08}-{stem}.compiler.raw.d");
        let mut expected_argv = vec![
            static_identity.cuda_toolkit.nvcc.path.clone(),
            "-c".to_string(),
            translation_unit.path.clone(),
            "-o".to_string(),
            expected_object_file.clone(),
            expected_architecture.clone(),
            "-ccbin".to_string(),
            static_identity.host_toolchain.compiler.path.clone(),
            "-MMD".to_string(),
            "-MF".to_string(),
            expected_compiler_depfile.clone(),
            "-MT".to_string(),
            expected_object_file.clone(),
        ];
        expected_argv.extend(plan.include_dirs.iter().map(|path| format!("-I{path}")));
        expected_argv.extend(plan.defines.iter().map(|define| format!("-D{define}")));
        expected_argv.extend(nvcc_policy_flags(&plan.nvcc_policy));
        expected_argv.push("--threads".to_string());
        expected_argv.push(receipt.nvcc_threads.to_string());
        let mut actual_argv = command.argv.clone();
        if let Some(output) = actual_argv.get_mut(4) {
            *output = Path::new(output)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
        }
        for argument in [10_usize, 12] {
            if let Some(path) = actual_argv.get_mut(argument) {
                *path = Path::new(path)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned();
            }
        }
        if actual_argv != expected_argv {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "{} source-build argv for {} differs from its locked plan",
                receipt.operator, translation_unit.path
            )));
        }
        validate_observed_dependencies(
            &format!("{}:{}", receipt.operator, translation_unit.path),
            &command.observed_dependencies,
        )?;
        let expected_source = expected_source_dependencies(translation_unit, closure);
        let observed_source = command
            .observed_dependencies
            .iter()
            .filter(|dependency| dependency.domain == NativeOperatorDependencyDomain::Source)
            .cloned()
            .collect::<BTreeSet<_>>();
        if observed_source != expected_source {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "{} source-build source dependency evidence for {} differs from its locked closure",
                receipt.operator, translation_unit.path
            )));
        }
        let expected_depfile_relative = format!("depfiles/{expected_depfile}");
        let expected_compiler_depfile_relative = format!("depfiles/{expected_compiler_depfile}");
        let binding_dependencies = validate_depfile_bindings_basic(
            &format!("{}:{}", receipt.operator, translation_unit.path),
            &command.depfile_bindings,
        )?;
        match command.object_cache_status {
            Some(NativeOperatorSourceObjectCacheStatus::Published)
                if command.dependency_validation
                    == Some(NativeOperatorDependencyValidation::Depfile)
                    && command.compiler_depfile.as_deref()
                        == Some(expected_compiler_depfile_relative.as_str())
                    && command
                        .compiler_depfile_sha256
                        .as_deref()
                        .is_some_and(is_sha256_digest)
                    && command.depfile.as_deref() == Some(expected_depfile_relative.as_str())
                    && command
                        .depfile_sha256
                        .as_deref()
                        .is_some_and(is_sha256_digest)
                    && command.depfile_producer_working_directory.as_deref()
                        == Some(command.working_directory.as_str())
                    && command.depfile_producer_object_file.as_deref()
                        == command.object_file.as_deref()
                    && binding_dependencies == command.observed_dependencies => {}
            Some(NativeOperatorSourceObjectCacheStatus::Hit)
                if command.dependency_validation
                    == Some(NativeOperatorDependencyValidation::CacheProof)
                    && command.compiler_depfile.as_deref()
                        == Some(expected_compiler_depfile_relative.as_str())
                    && command
                        .compiler_depfile_sha256
                        .as_deref()
                        .is_some_and(is_sha256_digest)
                    && command.depfile.as_deref() == Some(expected_depfile_relative.as_str())
                    && command
                        .depfile_sha256
                        .as_deref()
                        .is_some_and(is_sha256_digest)
                    && command
                        .depfile_producer_working_directory
                        .as_deref()
                        .is_some_and(|path| Path::new(path).is_absolute())
                    && command
                        .depfile_producer_object_file
                        .as_deref()
                        .is_some_and(|path| {
                            Path::new(path).is_absolute()
                                && Path::new(path).file_name()
                                    == Some(std::ffi::OsStr::new(&expected_object_file))
                        })
                    && binding_dependencies == command.observed_dependencies => {}
            _ => {
                return Err(NativeOperatorBuilderError::Invalid(format!(
                    "{} source-build dependency evidence for {} differs from its locked plan",
                    receipt.operator, translation_unit.path
                )))
            }
        }
    }
    match &toolchain.miss_probe {
        Some(probe)
            if probe.probed_for_misses == receipt.compiled_translation_units
                && !receipt.compiled_translation_units.is_empty()
                && !probe.nvcc_version.trim().is_empty()
                && probe.host_compiler_version
                    == static_identity.host_toolchain.compiler_version
                && !probe.archiver_version.trim().is_empty()
                && probe.host_target == static_identity.host_toolchain.target => {}
        None if receipt.compiled_translation_units.is_empty() => {}
        _ => {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "{} source-build miss-only toolchain probe differs from compiled translation units",
                receipt.operator
            )))
        }
    }
    let archive_command = receipt.commands.last().expect("command count checked");
    let actual_archive_argv = archive_command
        .argv
        .iter()
        .enumerate()
        .map(|(index, value)| {
            if index >= 2 {
                Path::new(value)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned()
            } else {
                value.clone()
            }
        })
        .collect::<Vec<_>>();
    let mut expected_archive_argv = vec![
        static_identity.archiver.path.clone(),
        "rcs".to_string(),
        plan.archive_file.clone(),
    ];
    expected_archive_argv.extend(
        plan.translation_units
            .iter()
            .enumerate()
            .map(|(index, translation_unit)| object_file_name(index, translation_unit)),
    );
    if actual_archive_argv != expected_archive_argv
        || receipt.archive_file.as_deref() != Some(plan.archive_file.as_str())
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} source-build archive command differs from its locked plan",
            receipt.operator
        )));
    }
    Ok(plan)
}

pub(crate) fn verify_source_build_evidence(
    receipt: &NativeOperatorSourceBuildReceipt,
    receipt_root: &Path,
    plan: &NativeOperatorSourceBuildPlan,
) -> Result<()> {
    let toolchain = receipt.toolchain.as_ref().ok_or_else(|| {
        NativeOperatorBuilderError::Invalid(format!(
            "{} source-build receipt is missing toolchain provenance",
            receipt.operator
        ))
    })?;
    let manifest_evidence = &toolchain.static_identity.cuda_toolkit.manifest;
    let manifest_path =
        resolve_source_build_evidence_file(receipt_root, &receipt.operator, manifest_evidence)?;
    let manifest: NativeOperatorCudaToolkitManifest = read_json(&manifest_path)?;
    validate_cuda_toolkit_manifest(
        &receipt.operator,
        &toolchain.static_identity.cuda_toolkit,
        &manifest,
    )?;
    let host_identity = &toolchain.static_identity.host_toolchain;
    let host_manifest_path = resolve_source_build_evidence_file(
        receipt_root,
        &receipt.operator,
        &host_identity.manifest,
    )?;
    let host_manifest: NativeOperatorHostToolchainManifest = read_json(&host_manifest_path)?;
    validate_host_toolchain_manifest(&receipt.operator, &host_manifest)?;
    if host_manifest.compiler != host_identity.compiler
        || host_manifest.compiler_version != host_identity.compiler_version
        || host_manifest.target != host_identity.target
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} host toolchain manifest differs from its receipt identity",
            receipt.operator
        )));
    }
    let toolchain_dependency_scope =
        toolchain_dependency_scope(&toolchain.static_identity, &manifest, &host_manifest)?;
    for (index, command) in receipt
        .commands
        .iter()
        .take(plan.translation_units.len())
        .enumerate()
    {
        if !matches!(
            command.dependency_validation,
            Some(
                NativeOperatorDependencyValidation::Depfile
                    | NativeOperatorDependencyValidation::CacheProof
            )
        ) {
            continue;
        }
        let compiler_relative = command.compiler_depfile.as_deref().ok_or_else(|| {
            NativeOperatorBuilderError::Invalid(format!(
                "{} compiled source-build command is missing compiler depfile evidence",
                receipt.operator
            ))
        })?;
        let compiler_sha256 = command.compiler_depfile_sha256.as_deref().ok_or_else(|| {
            NativeOperatorBuilderError::Invalid(format!(
                "{} compiled source-build command is missing compiler depfile sha256",
                receipt.operator
            ))
        })?;
        let compiler_path = resolve_source_build_relative_file(receipt_root, compiler_relative)?;
        let compiler_bytes = read_bounded_regular_file(
            &compiler_path,
            MAX_DEPFILE_BYTES,
            "source-build compiler depfile",
        )?;
        let compiler_actual = sha256_bytes(&compiler_bytes);
        if compiler_actual != compiler_sha256 {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "{} source-build compiler depfile sha256 mismatch: path={compiler_relative} expected={compiler_sha256} actual={compiler_actual}",
                receipt.operator
            )));
        }
        let compiler_raw = std::str::from_utf8(&compiler_bytes).map_err(|_| {
            NativeOperatorBuilderError::Invalid(format!(
                "{} source-build compiler depfile is not UTF-8: {compiler_relative}",
                receipt.operator
            ))
        })?;
        let relative = command.depfile.as_deref().ok_or_else(|| {
            NativeOperatorBuilderError::Invalid(format!(
                "{} compiled source-build command is missing portable depfile evidence",
                receipt.operator
            ))
        })?;
        let sha256 = command.depfile_sha256.as_deref().ok_or_else(|| {
            NativeOperatorBuilderError::Invalid(format!(
                "{} compiled source-build command is missing depfile sha256",
                receipt.operator
            ))
        })?;
        let path = resolve_source_build_relative_file(receipt_root, relative)?;
        let bytes =
            read_bounded_regular_file(&path, MAX_DEPFILE_BYTES, "source-build portable depfile")?;
        let actual = sha256_bytes(&bytes);
        if actual != sha256 {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "{} source-build depfile sha256 mismatch: path={relative} expected={sha256} actual={actual}",
                receipt.operator
            )));
        }
        let raw = std::str::from_utf8(&bytes).map_err(|_| {
            NativeOperatorBuilderError::Invalid(format!(
                "{} source-build portable depfile is not UTF-8: {relative}",
                receipt.operator
            ))
        })?;
        let producer_object_file =
            command
                .depfile_producer_object_file
                .as_deref()
                .ok_or_else(|| {
                    NativeOperatorBuilderError::Invalid(format!(
                        "{} compiled source-build command is missing depfile producer object",
                        receipt.operator
                    ))
                })?;
        let producer_working_directory = command
            .depfile_producer_working_directory
            .as_deref()
            .ok_or_else(|| {
                NativeOperatorBuilderError::Invalid(format!(
                    "{} compiled source-build command is missing depfile producer working directory",
                    receipt.operator
                ))
            })?;
        command.object_file.as_deref().ok_or_else(|| {
            NativeOperatorBuilderError::Invalid(format!(
                "{} source-build command is missing object path for depfile validation",
                receipt.operator
            ))
        })?;
        let observed = validate_portable_depfile_pair(
            compiler_raw,
            &compiler_path,
            raw,
            &path,
            producer_object_file,
            producer_working_directory,
            &plan.translation_units[index],
            &plan.dependency_closures[index],
            &command.depfile_bindings,
            &toolchain_dependency_scope,
        )?;
        if observed != command.observed_dependencies {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "{} source-build depfile semantics differ from receipt evidence: path={relative}",
                receipt.operator
            )));
        }
    }
    Ok(())
}

fn resolve_source_build_evidence_file(
    root: &Path,
    operator: &str,
    evidence: &NativeOperatorEvidenceFile,
) -> Result<PathBuf> {
    if !is_sha256_digest(&evidence.sha256) || evidence.size_bytes == 0 {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{operator} source-build evidence identity is incomplete: {}",
            evidence.path
        )));
    }
    let path = resolve_source_build_relative_file(root, &evidence.path)?;
    let size_bytes = fs::metadata(&path)
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: path.clone(),
            source,
        })?
        .len();
    let sha256 = sha256_file(&path)?;
    if size_bytes != evidence.size_bytes || sha256 != evidence.sha256 {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{operator} source-build evidence mismatch: path={} expected_size={} actual_size={size_bytes} expected_sha256={} actual_sha256={sha256}",
            evidence.path, evidence.size_bytes, evidence.sha256
        )));
    }
    Ok(path)
}

fn resolve_source_build_relative_file(root: &Path, relative: &str) -> Result<PathBuf> {
    validate_relative_path(relative)?;
    let canonical_root = root
        .canonicalize()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: root.to_path_buf(),
            source,
        })?;
    let path = canonical_root.join(relative);
    let canonical = path
        .canonicalize()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: path.clone(),
            source,
        })?;
    if !canonical.starts_with(&canonical_root) || !canonical.is_file() {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "source-build evidence escapes its root or is not a file: {relative}"
        )));
    }
    Ok(canonical)
}

fn validate_cuda_toolkit_manifest(
    operator: &str,
    identity: &NativeOperatorCudaToolkitIdentity,
    manifest: &NativeOperatorCudaToolkitManifest,
) -> Result<()> {
    if manifest.schema_version != NATIVE_OPERATOR_CUDA_TOOLKIT_MANIFEST_SCHEMA_VERSION
        || manifest.canonical_root != identity.canonical_root
        || manifest.entries.is_empty()
        || manifest
            .entries
            .windows(2)
            .any(|pair| pair[0].logical_path >= pair[1].logical_path)
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{operator} cuda toolkit manifest header/order is invalid"
        )));
    }
    let mut scopes = BTreeSet::new();
    let mut required_files = REQUIRED_CUDA_TOOLKIT_FILES
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut selected_nvcc = None;
    for entry in &manifest.entries {
        validate_relative_path(&entry.logical_path)?;
        validate_relative_path(&entry.resolved_path)?;
        if !is_sha256_digest(&entry.sha256) {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "{operator} cuda toolkit manifest entry is incomplete: {}",
                entry.logical_path
            )));
        }
        for scope in ["bin/crt/", "include/", "nvvm/bin/", "nvvm/libdevice/"] {
            if entry.logical_path.starts_with(scope) {
                scopes.insert(scope);
            }
        }
        required_files.remove(entry.logical_path.as_str());
        if Path::new(&identity.canonical_root).join(&entry.resolved_path)
            == Path::new(&identity.nvcc.path)
        {
            selected_nvcc = Some(entry);
        }
    }
    if scopes.len() != REQUIRED_CUDA_TOOLKIT_SCOPES.len() || !required_files.is_empty() {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{operator} cuda toolkit manifest does not cover every required compiler input; missing={}",
            required_files.into_iter().collect::<Vec<_>>().join(",")
        )));
    }
    if selected_nvcc.is_none_or(|entry| {
        entry.sha256 != identity.nvcc.sha256 || entry.size_bytes != identity.nvcc.size_bytes
    }) {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{operator} cuda toolkit manifest does not bind the selected nvcc"
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

struct NativeOperatorToolchainDependencyScope {
    by_absolute_path: BTreeMap<String, NativeOperatorObservedDependency>,
}

fn load_toolchain_dependency_scope(
    receipt_root: &Path,
    toolchain: &NativeOperatorSourceBuildStaticToolchain,
) -> Result<NativeOperatorToolchainDependencyScope> {
    let cuda_manifest_path = resolve_source_build_evidence_file(
        receipt_root,
        "<dependency-scope>",
        &toolchain.cuda_toolkit.manifest,
    )?;
    let cuda_manifest: NativeOperatorCudaToolkitManifest = read_json(&cuda_manifest_path)?;
    validate_cuda_toolkit_manifest(
        "<dependency-scope>",
        &toolchain.cuda_toolkit,
        &cuda_manifest,
    )?;
    let host_manifest_path = resolve_source_build_evidence_file(
        receipt_root,
        "<dependency-scope>",
        &toolchain.host_toolchain.manifest,
    )?;
    let host_manifest: NativeOperatorHostToolchainManifest = read_json(&host_manifest_path)?;
    validate_host_toolchain_manifest("<dependency-scope>", &host_manifest)?;
    toolchain_dependency_scope(toolchain, &cuda_manifest, &host_manifest)
}

fn toolchain_dependency_scope(
    toolchain: &NativeOperatorSourceBuildStaticToolchain,
    cuda_manifest: &NativeOperatorCudaToolkitManifest,
    host_manifest: &NativeOperatorHostToolchainManifest,
) -> Result<NativeOperatorToolchainDependencyScope> {
    let mut by_absolute_path = BTreeMap::new();
    let cuda_roots = [
        Path::new(&toolchain.cuda_toolkit.canonical_root),
        Path::new(&toolchain.cuda_toolkit.invocation_root),
    ];
    for entry in &cuda_manifest.entries {
        let dependency = NativeOperatorObservedDependency {
            domain: NativeOperatorDependencyDomain::BackendToolchain,
            path: entry.logical_path.clone(),
            sha256: entry.sha256.clone(),
        };
        for root in cuda_roots {
            for relative in [&entry.logical_path, &entry.resolved_path] {
                insert_toolchain_dependency(
                    &mut by_absolute_path,
                    root.join(relative).display().to_string(),
                    dependency.clone(),
                )?;
            }
        }
    }
    for entry in &host_manifest.files {
        let dependency = NativeOperatorObservedDependency {
            domain: NativeOperatorDependencyDomain::HostToolchain,
            path: entry.resolved_path.clone(),
            sha256: entry.sha256.clone(),
        };
        for absolute in [&entry.logical_path, &entry.resolved_path] {
            insert_toolchain_dependency(
                &mut by_absolute_path,
                absolute.clone(),
                dependency.clone(),
            )?;
        }
    }
    Ok(NativeOperatorToolchainDependencyScope { by_absolute_path })
}

fn insert_toolchain_dependency(
    dependencies: &mut BTreeMap<String, NativeOperatorObservedDependency>,
    absolute_path: String,
    dependency: NativeOperatorObservedDependency,
) -> Result<()> {
    validate_normalized_absolute_path(&absolute_path, "toolchain dependency path")?;
    if let Some(existing) = dependencies.get(&absolute_path) {
        if existing != &dependency {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "toolchain manifests ambiguously own dependency path: {absolute_path}"
            )));
        }
        return Ok(());
    }
    dependencies.insert(absolute_path, dependency);
    Ok(())
}

fn expected_source_dependencies(
    translation_unit: &NativeOperatorSourceFileLock,
    closure: &NativeOperatorTranslationUnitDependencyLock,
) -> BTreeSet<NativeOperatorObservedDependency> {
    std::iter::once(translation_unit)
        .chain(closure.headers.iter())
        .map(|locked| NativeOperatorObservedDependency {
            domain: NativeOperatorDependencyDomain::Source,
            path: locked.path.clone(),
            sha256: locked.sha256.clone(),
        })
        .collect()
}

fn validate_observed_dependency(
    context: &str,
    dependency: &NativeOperatorObservedDependency,
) -> Result<()> {
    let path_is_valid = match dependency.domain {
        NativeOperatorDependencyDomain::Source
        | NativeOperatorDependencyDomain::BackendToolchain => {
            validate_relative_path(&dependency.path).is_ok()
        }
        NativeOperatorDependencyDomain::HostToolchain => Path::new(&dependency.path).is_absolute(),
    };
    if !path_is_valid || !is_sha256_digest(&dependency.sha256) {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{context} observed dependency identity is invalid: {:?}:{}",
            dependency.domain, dependency.path
        )));
    }
    Ok(())
}

fn validate_observed_dependencies(
    context: &str,
    dependencies: &[NativeOperatorObservedDependency],
) -> Result<()> {
    if dependencies.is_empty()
        || dependencies.windows(2).any(|pair| pair[0] >= pair[1])
        || dependencies
            .iter()
            .any(|dependency| validate_observed_dependency(context, dependency).is_err())
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{context} observed dependencies are empty, unsorted, duplicated, or invalid"
        )));
    }
    Ok(())
}

fn observed_dependency_set_sha256(
    dependencies: &[NativeOperatorObservedDependency],
) -> Result<String> {
    validate_observed_dependencies("<dependency-set>", dependencies)?;
    let bytes =
        serde_json::to_vec(dependencies).map_err(|source| NativeOperatorBuilderError::Json {
            path: PathBuf::from("<dependency-set>"),
            source,
        })?;
    Ok(sha256_bytes(&bytes))
}

fn validate_translation_unit_depfile(
    compiler_depfile_path: &Path,
    portable_depfile_path: &Path,
    object_path: &Path,
    source_root: &Path,
    translation_unit: &NativeOperatorSourceFileLock,
    closure: &NativeOperatorTranslationUnitDependencyLock,
    toolchain_scope: &NativeOperatorToolchainDependencyScope,
) -> Result<ValidatedTranslationUnitDepfile> {
    if closure.translation_unit != translation_unit.path {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "dependency closure translation unit mismatch: expected={} actual={}",
            translation_unit.path, closure.translation_unit
        )));
    }
    require_file(compiler_depfile_path)?;
    let raw =
        read_bounded_regular_file(compiler_depfile_path, MAX_DEPFILE_BYTES, "compiler depfile")?;
    let raw_text = std::str::from_utf8(&raw).map_err(|_| {
        NativeOperatorBuilderError::Invalid(format!(
            "compiler depfile is not valid UTF-8: {}",
            compiler_depfile_path.display()
        ))
    })?;
    let (target, dependencies) = parse_make_depfile(raw_text, compiler_depfile_path)?;
    let expected_target = object_path.display().to_string();
    validate_normalized_absolute_path(&expected_target, "compiler depfile object target")?;
    if target != expected_target {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "compiler depfile target differs from its exact -MT object: expected={expected_target} actual={target}"
        )));
    }

    let mut observed = BTreeSet::new();
    let mut portable_paths = BTreeMap::new();
    let mut bindings = Vec::with_capacity(dependencies.len());
    for dependency in dependencies {
        let candidate = if Path::new(&dependency).is_absolute() {
            PathBuf::from(&dependency)
        } else {
            source_root.join(&dependency)
        };
        let canonical =
            candidate
                .canonicalize()
                .map_err(|source| NativeOperatorBuilderError::Io {
                    path: candidate.clone(),
                    source,
                })?;
        if !canonical.is_file() {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "compiler depfile dependency is not a file: {}",
                canonical.display()
            )));
        }
        let (identity, portable_path) = if canonical.starts_with(source_root) {
            let relative = canonical.strip_prefix(source_root).map_err(|_| {
                NativeOperatorBuilderError::Invalid(format!(
                    "compiler depfile dependency escapes the source root: {}",
                    canonical.display()
                ))
            })?;
            let identity = NativeOperatorObservedDependency {
                domain: NativeOperatorDependencyDomain::Source,
                path: path_with_forward_slashes(relative)?,
                sha256: sha256_file(&canonical)?,
            };
            let portable_path = identity.path.clone();
            (identity, portable_path)
        } else {
            let candidate_path = candidate.display().to_string();
            let canonical_path = canonical.display().to_string();
            let identity = toolchain_scope
                .by_absolute_path
                .get(&candidate_path)
                .or_else(|| toolchain_scope.by_absolute_path.get(&canonical_path))
                .ok_or_else(|| {
                    NativeOperatorBuilderError::Invalid(format!(
                        "compiler depfile dependency is outside the locked source and toolchain manifests: {}",
                        canonical.display()
                    ))
                })?
                .clone();
            let actual = sha256_file(&canonical)?;
            if actual != identity.sha256 {
                return Err(NativeOperatorBuilderError::Invalid(format!(
                    "toolchain dependency changed while compiling: path={} expected={} actual={actual}",
                    canonical.display(),
                    identity.sha256
                )));
            }
            validate_normalized_absolute_path(
                &canonical_path,
                "compiled depfile canonical toolchain dependency",
            )?;
            (identity, canonical_path)
        };
        validate_observed_dependency("<compiled-depfile>", &identity)?;
        if !observed.insert(identity.clone()) {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "compiler depfile contains a duplicate dependency identity: {:?}:{}",
                identity.domain, identity.path
            )));
        }
        portable_paths.insert(identity.clone(), portable_path.clone());
        bindings.push(NativeOperatorDepfileDependencyBinding {
            producer_path: dependency,
            portable_path,
            dependency: identity,
        });
    }

    let expected = expected_source_dependencies(translation_unit, closure);
    let observed_source = observed
        .iter()
        .filter(|dependency| dependency.domain == NativeOperatorDependencyDomain::Source)
        .cloned()
        .collect::<BTreeSet<_>>();
    if observed_source != expected {
        let missing = expected
            .difference(&observed_source)
            .cloned()
            .collect::<Vec<_>>();
        let undeclared = observed_source
            .difference(&expected)
            .cloned()
            .collect::<Vec<_>>();
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "depfile differs from the declared source dependency closure: missing={missing:?} undeclared={undeclared:?}"
        )));
    }

    for locked in std::iter::once(translation_unit).chain(closure.headers.iter()) {
        let path = locked_source_file(source_root, &locked.path)?;
        let actual = sha256_file(&path)?;
        if actual != locked.sha256 {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "dependency changed while compiling: path={} expected={} actual={actual}",
                locked.path, locked.sha256
            )));
        }
    }
    let observed_dependencies = observed.into_iter().collect::<Vec<_>>();
    let portable_dependencies = observed_dependencies
        .iter()
        .map(|identity| {
            portable_paths.get(identity).cloned().ok_or_else(|| {
                NativeOperatorBuilderError::Invalid(format!(
                    "validated dependency has no portable depfile path: {:?}:{}",
                    identity.domain, identity.path
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let portable_raw =
        serialize_portable_depfile(&object_path.display().to_string(), &portable_dependencies)?;
    let portable_text = std::str::from_utf8(&portable_raw).expect("portable depfile is UTF-8");
    let verified_dependencies = validate_portable_depfile_pair(
        raw_text,
        compiler_depfile_path,
        portable_text,
        portable_depfile_path,
        &object_path.display().to_string(),
        &source_root.display().to_string(),
        translation_unit,
        closure,
        &bindings,
        toolchain_scope,
    )?;
    if verified_dependencies != observed_dependencies {
        return Err(NativeOperatorBuilderError::Invalid(
            "producer depfile verification changed the typed dependency set".to_string(),
        ));
    }
    atomic_write_bytes(portable_depfile_path, &portable_raw)?;
    Ok(ValidatedTranslationUnitDepfile {
        compiler_sha256: sha256_bytes(&raw),
        compiler_raw: raw,
        portable_sha256: sha256_bytes(&portable_raw),
        portable_raw,
        bindings,
        observed_dependencies,
    })
}

struct ValidatedTranslationUnitDepfile {
    compiler_raw: Vec<u8>,
    compiler_sha256: String,
    portable_raw: Vec<u8>,
    portable_sha256: String,
    bindings: Vec<NativeOperatorDepfileDependencyBinding>,
    observed_dependencies: Vec<NativeOperatorObservedDependency>,
}

fn serialize_portable_depfile(target: &str, dependencies: &[String]) -> Result<Vec<u8>> {
    validate_normalized_absolute_path(target, "portable depfile target")?;
    if dependencies.is_empty() || dependencies.len() > MAX_DEPFILE_DEPENDENCIES {
        return Err(NativeOperatorBuilderError::Invalid(
            "portable depfile dependency count is invalid".to_string(),
        ));
    }
    let mut result = escape_make_word(target)?;
    result.push(':');
    for dependency in dependencies {
        if Path::new(dependency).is_absolute() {
            validate_normalized_absolute_path(dependency, "portable depfile dependency")?;
        } else {
            validate_relative_path(dependency)?;
        }
        result.push(' ');
        result.push_str(&escape_make_word(dependency)?);
    }
    result.push('\n');
    if result.len() > MAX_DEPFILE_BYTES {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "portable depfile exceeds {MAX_DEPFILE_BYTES} bytes"
        )));
    }
    let (parsed_target, parsed_dependencies) =
        parse_make_depfile(&result, Path::new("<portable-depfile>"))?;
    if parsed_target != target || parsed_dependencies != dependencies {
        return Err(NativeOperatorBuilderError::Invalid(
            "portable depfile serialization did not round-trip".to_string(),
        ));
    }
    Ok(result.into_bytes())
}

fn escape_make_word(value: &str) -> Result<String> {
    if value.is_empty()
        || value.len() > MAX_DEPFILE_WORD_BYTES
        || value
            .chars()
            .any(|character| matches!(character, '\0' | '\n' | '\r'))
    {
        return Err(NativeOperatorBuilderError::Invalid(
            "make depfile word must be non-empty and single-line".to_string(),
        ));
    }
    let mut result = String::with_capacity(value.len());
    for character in value.chars() {
        if matches!(character, '\\' | ' ' | '\t' | ':' | '#' | '$') {
            result.push('\\');
        }
        result.push(character);
    }
    Ok(result)
}

fn publish_object_dependency_proof(
    cache_entry: &Path,
    object_cache_key: &str,
    object_sha256: &str,
    translation_unit: &NativeOperatorSourceFileLock,
    closure: &NativeOperatorTranslationUnitDependencyLock,
    compiler_depfile_raw: &[u8],
    expected_compiler_depfile_sha256: &str,
    depfile_raw: &[u8],
    expected_depfile_sha256: &str,
    producer_working_directory: &str,
    producer_object_file: &str,
    depfile_bindings: &[NativeOperatorDepfileDependencyBinding],
    observed_dependencies: &[NativeOperatorObservedDependency],
    toolchain_scope: &NativeOperatorToolchainDependencyScope,
) -> Result<()> {
    let compiler_depfile_sha256 = sha256_bytes(compiler_depfile_raw);
    if compiler_depfile_sha256 != expected_compiler_depfile_sha256 {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "compiler depfile bytes differ from their recorded sha256: expected={expected_compiler_depfile_sha256} actual={compiler_depfile_sha256}"
        )));
    }
    let depfile_sha256 = sha256_bytes(depfile_raw);
    if depfile_sha256 != expected_depfile_sha256 {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "validated depfile bytes differ from their recorded sha256: expected={expected_depfile_sha256} actual={depfile_sha256}"
        )));
    }
    let proof = NativeOperatorObjectDependencyProof {
        schema_version: NATIVE_OPERATOR_OBJECT_DEPENDENCY_PROOF_SCHEMA_VERSION,
        object_cache_key: object_cache_key.to_string(),
        object_sha256: object_sha256.to_string(),
        dependency_closure_sha256: closure.closure_sha256.clone(),
        dependency_set_sha256: observed_dependency_set_sha256(observed_dependencies)?,
        compiler_depfile_sha256,
        depfile_sha256: depfile_sha256.clone(),
        producer_working_directory: producer_working_directory.to_string(),
        producer_object_file: producer_object_file.to_string(),
        depfile_bindings: depfile_bindings.to_vec(),
        observed_dependencies: observed_dependencies.to_vec(),
    };
    validate_object_dependency_proof(
        "<object-cache-publish>",
        &proof,
        object_cache_key,
        object_sha256,
        translation_unit,
        closure,
    )?;
    let proof_dir = cache_entry.join("dependency-proof");
    if proof_dir.exists() {
        return validate_existing_dependency_proof(
            &proof_dir,
            object_cache_key,
            object_sha256,
            translation_unit,
            closure,
            producer_object_file,
            toolchain_scope,
        );
    }

    let staging = tempfile::Builder::new()
        .prefix(".dependency-proof-")
        .tempdir_in(cache_entry)
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: cache_entry.to_path_buf(),
            source,
        })?;
    let staging_path = staging.path().to_path_buf();
    atomic_write_bytes(
        &staging_path.join("compiler-dependency.raw.d"),
        compiler_depfile_raw,
    )?;
    atomic_write_bytes(&staging_path.join("dependency.d"), depfile_raw)?;
    write_json(&staging_path.join("proof.json"), &proof)?;
    sync_directory(&staging_path)?;
    let staging_path = staging.keep();
    match fs::rename(&staging_path, &proof_dir) {
        Ok(()) => {
            sync_directory(cache_entry)?;
            validate_existing_dependency_proof(
                &proof_dir,
                object_cache_key,
                object_sha256,
                translation_unit,
                closure,
                producer_object_file,
                toolchain_scope,
            )
        }
        Err(_source) if proof_dir.exists() => {
            fs::remove_dir_all(&staging_path).map_err(|cleanup_source| {
                NativeOperatorBuilderError::Io {
                    path: staging_path.clone(),
                    source: cleanup_source,
                }
            })?;
            validate_existing_dependency_proof(
                &proof_dir,
                object_cache_key,
                object_sha256,
                translation_unit,
                closure,
                producer_object_file,
                toolchain_scope,
            )
        }
        Err(source) => {
            let _ = fs::remove_dir_all(&staging_path);
            Err(NativeOperatorBuilderError::Io {
                path: proof_dir,
                source,
            })
        }
    }
}

fn validate_dependency_proof_directory(proof_dir: &Path) -> Result<()> {
    let metadata =
        fs::symlink_metadata(proof_dir).map_err(|source| NativeOperatorBuilderError::Io {
            path: proof_dir.to_path_buf(),
            source,
        })?;
    if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "object dependency proof is not a real directory: {}",
            proof_dir.display()
        )));
    }
    let mut entries = fs::read_dir(proof_dir)
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: proof_dir.to_path_buf(),
            source,
        })?
        .map(|entry| {
            entry
                .map(|entry| entry.file_name())
                .map_err(|source| NativeOperatorBuilderError::Io {
                    path: proof_dir.to_path_buf(),
                    source,
                })
        })
        .collect::<Result<Vec<_>>>()?;
    entries.sort();
    let expected = [
        std::ffi::OsString::from("compiler-dependency.raw.d"),
        std::ffi::OsString::from("dependency.d"),
        std::ffi::OsString::from("proof.json"),
    ];
    if entries != expected {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "object dependency proof directory is incomplete or contains extra files: {}",
            proof_dir.display()
        )));
    }
    for name in expected {
        let path = proof_dir.join(name);
        let metadata =
            fs::symlink_metadata(&path).map_err(|source| NativeOperatorBuilderError::Io {
                path: path.clone(),
                source,
            })?;
        if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "object dependency proof member is not a regular file: {}",
                path.display()
            )));
        }
    }
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    let directory = File::open(path).map_err(|source| NativeOperatorBuilderError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    directory
        .sync_all()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: path.to_path_buf(),
            source,
        })
}

struct ValidatedCachedDependencyProof {
    proof: NativeOperatorObjectDependencyProof,
    compiler_raw: Vec<u8>,
    portable_raw: Vec<u8>,
}

#[allow(clippy::too_many_arguments)]
fn load_validated_dependency_proof(
    proof_dir: &Path,
    object_cache_key: &str,
    object_sha256: &str,
    translation_unit: &NativeOperatorSourceFileLock,
    closure: &NativeOperatorTranslationUnitDependencyLock,
    toolchain_scope: &NativeOperatorToolchainDependencyScope,
) -> Result<ValidatedCachedDependencyProof> {
    validate_dependency_proof_directory(proof_dir)?;
    let proof_path = proof_dir.join("proof.json");
    let proof_bytes = read_bounded_regular_file(
        &proof_path,
        MAX_DEPENDENCY_PROOF_BYTES,
        "object dependency proof",
    )?;
    let proof: NativeOperatorObjectDependencyProof =
        serde_json::from_slice(&proof_bytes).map_err(|source| {
            NativeOperatorBuilderError::Json {
                path: proof_path.clone(),
                source,
            }
        })?;
    validate_object_dependency_proof(
        "<object-cache-proof>",
        &proof,
        object_cache_key,
        object_sha256,
        translation_unit,
        closure,
    )?;

    let compiler_depfile_path = proof_dir.join("compiler-dependency.raw.d");
    let compiler_raw = read_bounded_regular_file(
        &compiler_depfile_path,
        MAX_DEPFILE_BYTES,
        "cached compiler depfile",
    )?;
    if sha256_bytes(&compiler_raw) != proof.compiler_depfile_sha256 {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "cached object compiler depfile hash mismatch: {}",
            compiler_depfile_path.display()
        )));
    }
    let depfile_path = proof_dir.join("dependency.d");
    let portable_raw =
        read_bounded_regular_file(&depfile_path, MAX_DEPFILE_BYTES, "cached portable depfile")?;
    if sha256_bytes(&portable_raw) != proof.depfile_sha256 {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "cached object dependency depfile hash mismatch: {}",
            depfile_path.display()
        )));
    }
    let compiler_text = std::str::from_utf8(&compiler_raw).map_err(|_| {
        NativeOperatorBuilderError::Invalid(format!(
            "cached object compiler depfile is not UTF-8: {}",
            compiler_depfile_path.display()
        ))
    })?;
    let portable_text = std::str::from_utf8(&portable_raw).map_err(|_| {
        NativeOperatorBuilderError::Invalid(format!(
            "cached object dependency depfile is not UTF-8: {}",
            depfile_path.display()
        ))
    })?;
    let observed = validate_portable_depfile_pair(
        compiler_text,
        &compiler_depfile_path,
        portable_text,
        &depfile_path,
        &proof.producer_object_file,
        &proof.producer_working_directory,
        translation_unit,
        closure,
        &proof.depfile_bindings,
        toolchain_scope,
    )?;
    if observed != proof.observed_dependencies {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "cached object dependency proof observed list differs from depfile: {}",
            proof_path.display()
        )));
    }
    Ok(ValidatedCachedDependencyProof {
        proof,
        compiler_raw,
        portable_raw,
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_existing_dependency_proof(
    proof_dir: &Path,
    object_cache_key: &str,
    object_sha256: &str,
    translation_unit: &NativeOperatorSourceFileLock,
    closure: &NativeOperatorTranslationUnitDependencyLock,
    producer_object_file: &str,
    toolchain_scope: &NativeOperatorToolchainDependencyScope,
) -> Result<()> {
    let validated = load_validated_dependency_proof(
        proof_dir,
        object_cache_key,
        object_sha256,
        translation_unit,
        closure,
        toolchain_scope,
    )?;
    if Path::new(&validated.proof.producer_object_file).file_name()
        != Path::new(producer_object_file).file_name()
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "published dependency proof object name differs from the current object: {}",
            proof_dir.display()
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn restore_object_dependency_proof(
    cache_entry: &Path,
    object_cache_key: &str,
    object_sha256: &str,
    closure: &NativeOperatorTranslationUnitDependencyLock,
    translation_unit: &NativeOperatorSourceFileLock,
    object_path: &Path,
    output_compiler_depfile: &Path,
    output_depfile: &Path,
    toolchain_scope: &NativeOperatorToolchainDependencyScope,
) -> Result<Option<NativeOperatorObjectDependencyProof>> {
    let proof_dir = cache_entry.join("dependency-proof");
    if !proof_dir.exists() {
        return Ok(None);
    }
    let validated = load_validated_dependency_proof(
        &proof_dir,
        object_cache_key,
        object_sha256,
        translation_unit,
        closure,
        toolchain_scope,
    )?;
    if Path::new(&validated.proof.producer_object_file).file_name() != object_path.file_name() {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "cached dependency proof object name differs from restored object: {}",
            proof_dir.display()
        )));
    }
    atomic_write_bytes(output_compiler_depfile, &validated.compiler_raw)?;
    atomic_write_bytes(output_depfile, &validated.portable_raw)?;
    Ok(Some(validated.proof))
}

fn validate_object_dependency_proof(
    context: &str,
    proof: &NativeOperatorObjectDependencyProof,
    object_cache_key: &str,
    object_sha256: &str,
    translation_unit: &NativeOperatorSourceFileLock,
    closure: &NativeOperatorTranslationUnitDependencyLock,
) -> Result<()> {
    let dependency_set_sha256 = observed_dependency_set_sha256(&proof.observed_dependencies)?;
    let binding_dependencies = validate_depfile_bindings_basic(context, &proof.depfile_bindings)?;
    let expected_source = expected_source_dependencies(translation_unit, closure);
    let observed_source = proof
        .observed_dependencies
        .iter()
        .filter(|dependency| dependency.domain == NativeOperatorDependencyDomain::Source)
        .cloned()
        .collect::<BTreeSet<_>>();
    validate_normalized_absolute_path(
        &proof.producer_working_directory,
        &format!("{context} producer_working_directory"),
    )?;
    validate_normalized_absolute_path(
        &proof.producer_object_file,
        &format!("{context} producer_object_file"),
    )?;
    if proof.schema_version != NATIVE_OPERATOR_OBJECT_DEPENDENCY_PROOF_SCHEMA_VERSION
        || proof.object_cache_key != object_cache_key
        || proof.object_sha256 != object_sha256
        || proof.dependency_closure_sha256 != closure.closure_sha256
        || proof.dependency_set_sha256 != dependency_set_sha256
        || !is_sha256_digest(&proof.object_cache_key)
        || !is_sha256_digest(&proof.object_sha256)
        || !is_sha256_digest(&proof.dependency_closure_sha256)
        || !is_sha256_digest(&proof.dependency_set_sha256)
        || !is_sha256_digest(&proof.compiler_depfile_sha256)
        || !is_sha256_digest(&proof.depfile_sha256)
        || binding_dependencies != proof.observed_dependencies
        || observed_source != expected_source
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{context} object dependency proof is invalid"
        )));
    }
    Ok(())
}

fn validate_depfile_bindings_basic(
    context: &str,
    bindings: &[NativeOperatorDepfileDependencyBinding],
) -> Result<Vec<NativeOperatorObservedDependency>> {
    if bindings.is_empty() || bindings.len() > MAX_DEPFILE_DEPENDENCIES {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{context} depfile bindings are empty or exceed {MAX_DEPFILE_DEPENDENCIES}"
        )));
    }
    let mut dependencies = BTreeSet::new();
    for binding in bindings {
        if binding.producer_path.is_empty()
            || binding.producer_path.len() > MAX_DEPFILE_WORD_BYTES
            || binding
                .producer_path
                .chars()
                .any(|character| matches!(character, '\0' | '\n' | '\r'))
            || binding.portable_path.is_empty()
            || binding.portable_path.len() > MAX_DEPFILE_WORD_BYTES
        {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "{context} depfile binding path is invalid"
            )));
        }
        validate_observed_dependency(context, &binding.dependency)?;
        match binding.dependency.domain {
            NativeOperatorDependencyDomain::Source => {
                validate_relative_path(&binding.portable_path)?;
                if binding.portable_path != binding.dependency.path {
                    return Err(NativeOperatorBuilderError::Invalid(format!(
                        "{context} source depfile binding does not use its locked path"
                    )));
                }
            }
            NativeOperatorDependencyDomain::BackendToolchain
            | NativeOperatorDependencyDomain::HostToolchain => {
                validate_normalized_absolute_path(
                    &binding.portable_path,
                    &format!("{context} toolchain depfile binding"),
                )?;
            }
        }
        if !dependencies.insert(binding.dependency.clone()) {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "{context} depfile bindings duplicate a typed dependency"
            )));
        }
    }
    Ok(dependencies.into_iter().collect())
}

#[allow(clippy::too_many_arguments)]
fn validate_portable_depfile_pair(
    compiler_raw: &str,
    compiler_depfile_path: &Path,
    portable_raw: &str,
    depfile_path: &Path,
    object_path: &str,
    working_directory: &str,
    translation_unit: &NativeOperatorSourceFileLock,
    closure: &NativeOperatorTranslationUnitDependencyLock,
    bindings: &[NativeOperatorDepfileDependencyBinding],
    toolchain_scope: &NativeOperatorToolchainDependencyScope,
) -> Result<Vec<NativeOperatorObservedDependency>> {
    let (compiler_target, compiler_dependencies) =
        parse_make_depfile(compiler_raw, compiler_depfile_path)?;
    let (target, dependencies) = parse_make_depfile(portable_raw, depfile_path)?;
    validate_normalized_absolute_path(object_path, "portable depfile producer object")?;
    if compiler_target != object_path || target != object_path {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "compiler or portable depfile target differs from object file: depfile={} compiler_target={compiler_target} portable_target={target}",
            depfile_path.display()
        )));
    }
    validate_normalized_absolute_path(working_directory, "portable depfile working directory")?;
    let working_directory = Path::new(working_directory);
    let expected = expected_source_dependencies(translation_unit, closure);
    let observed = validate_depfile_bindings_basic("<portable-depfile>", bindings)?;
    let expected_compiler_dependencies = bindings
        .iter()
        .map(|binding| binding.producer_path.as_str())
        .collect::<Vec<_>>();
    if compiler_dependencies
        != expected_compiler_dependencies
            .iter()
            .map(|value| (*value).to_string())
            .collect::<Vec<_>>()
    {
        return Err(NativeOperatorBuilderError::Invalid(
            "compiler depfile bytes differ from their ordered typed bindings".to_string(),
        ));
    }
    let portable_by_dependency = bindings
        .iter()
        .map(|binding| (binding.dependency.clone(), binding.portable_path.clone()))
        .collect::<BTreeMap<_, _>>();
    let expected_portable_dependencies = observed
        .iter()
        .map(|dependency| {
            portable_by_dependency
                .get(dependency)
                .cloned()
                .expect("binding dependencies were collected from the same rows")
        })
        .collect::<Vec<_>>();
    let canonical_portable =
        serialize_portable_depfile(object_path, &expected_portable_dependencies)?;
    if dependencies != expected_portable_dependencies
        || portable_raw.as_bytes() != canonical_portable.as_slice()
    {
        return Err(NativeOperatorBuilderError::Invalid(
            "portable depfile bytes differ from their canonical typed bindings".to_string(),
        ));
    }
    for binding in bindings {
        match binding.dependency.domain {
            NativeOperatorDependencyDomain::Source => {
                let producer = Path::new(&binding.producer_path);
                let relative = if producer.is_absolute() {
                    producer.strip_prefix(working_directory).map_err(|_| {
                        NativeOperatorBuilderError::Invalid(format!(
                            "source compiler depfile path escapes its recorded working directory: {}",
                            binding.producer_path
                        ))
                    })?
                } else {
                    producer
                };
                if normalize_portable_relative_path(relative)? != binding.dependency.path {
                    return Err(NativeOperatorBuilderError::Invalid(format!(
                        "source compiler depfile path differs from its locked identity: {}",
                        binding.producer_path
                    )));
                }
            }
            NativeOperatorDependencyDomain::BackendToolchain
            | NativeOperatorDependencyDomain::HostToolchain => {
                let normalized_producer = normalize_absolute_posix_path_lexically(
                    &binding.producer_path,
                    "toolchain compiler depfile binding",
                )?;
                if toolchain_scope.by_absolute_path.get(&normalized_producer)
                    != Some(&binding.dependency)
                    || toolchain_scope.by_absolute_path.get(&binding.portable_path)
                        != Some(&binding.dependency)
                {
                    return Err(NativeOperatorBuilderError::Invalid(format!(
                        "toolchain depfile binding is outside its typed manifest: producer={} portable={}",
                        binding.producer_path, binding.portable_path
                    )));
                }
            }
        }
    }
    let observed_source = observed
        .iter()
        .filter(|dependency| dependency.domain == NativeOperatorDependencyDomain::Source)
        .cloned()
        .collect::<BTreeSet<_>>();
    if observed_source != expected {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "portable depfile differs from locked source dependency closure: expected={expected:?} observed={observed_source:?}"
        )));
    }
    validate_observed_dependencies("<portable-depfile>", &observed)?;
    for dependency in &observed {
        if !bindings
            .iter()
            .any(|binding| &binding.dependency == dependency)
        {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "portable depfile is missing a binding for {:?}:{}",
                dependency.domain, dependency.path
            )));
        }
    }
    Ok(observed)
}

fn normalize_absolute_posix_path_lexically(value: &str, label: &str) -> Result<String> {
    if !value.starts_with('/')
        || value.contains('\\')
        || value
            .chars()
            .any(|character| matches!(character, '\0' | '\n' | '\r'))
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{label} must be an absolute POSIX path: {value}"
        )));
    }
    let mut components = Vec::new();
    for component in value.split('/') {
        match component {
            "" | "." => {}
            ".." => {
                if components.pop().is_none() {
                    return Err(NativeOperatorBuilderError::Invalid(format!(
                        "{label} escapes the filesystem root: {value}"
                    )));
                }
            }
            _ => components.push(component),
        }
    }
    if components.is_empty() {
        Ok("/".to_string())
    } else {
        Ok(format!("/{}", components.join("/")))
    }
}

fn validate_normalized_absolute_path(value: &str, label: &str) -> Result<()> {
    if value == "/" {
        return Ok(());
    }
    if !value.starts_with('/')
        || value.contains('\\')
        || value.ends_with('/')
        || value[1..]
            .split('/')
            .any(|component| component.is_empty() || component == "." || component == "..")
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{label} must be a normalized absolute POSIX path: {value}"
        )));
    }
    Ok(())
}

fn normalize_portable_relative_path(path: &Path) -> Result<String> {
    let raw = path.to_str().ok_or_else(|| {
        NativeOperatorBuilderError::Invalid(format!(
            "depfile path is not valid UTF-8: {}",
            path.display()
        ))
    })?;
    if raw.contains('\\')
        || raw
            .chars()
            .any(|character| matches!(character, '\0' | '\n' | '\r'))
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "depfile path is not a relative POSIX path: {}",
            path.display()
        )));
    }
    let mut components = Vec::new();
    for component in path.components() {
        match component {
            std::path::Component::Normal(value) => {
                components.push(value.to_str().ok_or_else(|| {
                    NativeOperatorBuilderError::Invalid(format!(
                        "depfile path is not valid UTF-8: {}",
                        path.display()
                    ))
                })?)
            }
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if components.pop().is_none() {
                    return Err(NativeOperatorBuilderError::Invalid(format!(
                        "depfile path escapes its working directory: {}",
                        path.display()
                    )));
                }
            }
            _ => {
                return Err(NativeOperatorBuilderError::Invalid(format!(
                    "depfile path is not normalized beneath its working directory: {}",
                    path.display()
                )))
            }
        }
    }
    let normalized = components.join("/");
    validate_relative_path(&normalized)?;
    Ok(normalized)
}

fn read_bounded_regular_file(path: &Path, max_bytes: usize, label: &str) -> Result<Vec<u8>> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_NOFOLLOW);
    let file = options
        .open(path)
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let metadata = file
        .metadata()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    if !metadata.is_file() || metadata.len() > max_bytes as u64 {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{label} is not a regular file or exceeds {max_bytes} bytes: {}",
            path.display()
        )));
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(max_bytes as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    if bytes.len() > max_bytes {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{label} grew beyond {max_bytes} bytes while reading: {}",
            path.display()
        )));
    }
    Ok(bytes)
}

fn atomic_write_bytes(destination: &Path, bytes: &[u8]) -> Result<()> {
    let parent = destination.parent().ok_or_else(|| {
        NativeOperatorBuilderError::Invalid(format!(
            "atomic write destination has no parent: {}",
            destination.display()
        ))
    })?;
    fs::create_dir_all(parent).map_err(|source| NativeOperatorBuilderError::Io {
        path: parent.to_path_buf(),
        source,
    })?;
    let mut temporary =
        NamedTempFile::new_in(parent).map_err(|source| NativeOperatorBuilderError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    temporary
        .write_all(bytes)
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: temporary.path().to_path_buf(),
            source,
        })?;
    temporary
        .as_file_mut()
        .flush()
        .and_then(|()| temporary.as_file().sync_all())
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: temporary.path().to_path_buf(),
            source,
        })?;
    temporary
        .persist(destination)
        .map_err(|error| NativeOperatorBuilderError::Io {
            path: destination.to_path_buf(),
            source: error.error,
        })?;
    Ok(())
}

fn parse_make_depfile(raw: &str, path: &Path) -> Result<(String, Vec<String>)> {
    if raw.len() > MAX_DEPFILE_BYTES || raw.trim().is_empty() || raw.as_bytes().contains(&0) {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "depfile is too large, empty, or contains NUL: {}",
            path.display()
        )));
    }
    let normalized = raw.replace("\\\r\n", "").replace("\\\n", "");
    let normalized = normalized.trim_end_matches(&['\r', '\n'][..]);
    if normalized.contains('\n') || normalized.contains('\r') {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "depfile contains more than one make rule: {}",
            path.display()
        )));
    }
    let mut escaped = false;
    let mut delimiter = None;
    for (index, character) in normalized.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match character {
            '\\' => escaped = true,
            ':' => {
                delimiter = Some(index);
                break;
            }
            '\n' | '\r' => {
                return Err(NativeOperatorBuilderError::Invalid(format!(
                    "depfile contains multiple or unterminated rules: {}",
                    path.display()
                )))
            }
            _ => {}
        }
    }
    let delimiter = delimiter.ok_or_else(|| {
        NativeOperatorBuilderError::Invalid(format!(
            "depfile has no make target delimiter: {}",
            path.display()
        ))
    })?;
    let targets = parse_make_words(&normalized[..delimiter], path, 1)?;
    let dependencies =
        parse_make_words(&normalized[delimiter + 1..], path, MAX_DEPFILE_DEPENDENCIES)?;
    if targets.len() != 1 || dependencies.is_empty() {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "depfile target/dependency count or word size is invalid: {}",
            path.display()
        )));
    }
    Ok((targets[0].clone(), dependencies))
}

fn parse_make_words(value: &str, path: &Path, max_words: usize) -> Result<Vec<String>> {
    let mut words = Vec::new();
    let mut word = String::new();
    let mut escaped = false;
    for character in value.chars() {
        if escaped {
            word.push(character);
            if word.len() > MAX_DEPFILE_WORD_BYTES {
                return Err(NativeOperatorBuilderError::Invalid(format!(
                    "depfile word exceeds {MAX_DEPFILE_WORD_BYTES} bytes: {}",
                    path.display()
                )));
            }
            escaped = false;
            continue;
        }
        match character {
            '\\' => escaped = true,
            '\n' | '\r' => unreachable!("newlines rejected before make-word parsing"),
            character if character.is_whitespace() => {
                if !word.is_empty() {
                    if words.len() >= max_words {
                        return Err(NativeOperatorBuilderError::Invalid(format!(
                            "depfile exceeds its word-count limit: {}",
                            path.display()
                        )));
                    }
                    words.push(std::mem::take(&mut word));
                }
            }
            _ => {
                word.push(character);
                if word.len() > MAX_DEPFILE_WORD_BYTES {
                    return Err(NativeOperatorBuilderError::Invalid(format!(
                        "depfile word exceeds {MAX_DEPFILE_WORD_BYTES} bytes: {}",
                        path.display()
                    )));
                }
            }
        }
    }
    if escaped {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "depfile ends with an incomplete escape: {}",
            path.display()
        )));
    }
    if !word.is_empty() {
        if words.len() >= max_words {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "depfile exceeds its word-count limit: {}",
                path.display()
            )));
        }
        words.push(word);
    }
    Ok(words)
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

fn resolve_static_toolchain(
    request: &NativeOperatorSourceBuildRequest,
) -> Result<NativeOperatorSourceBuildToolchain> {
    let invocation_root_path = if request.cuda_toolkit_root.is_absolute() {
        request.cuda_toolkit_root.clone()
    } else {
        std::env::current_dir()
            .map_err(|source| NativeOperatorBuilderError::Io {
                path: PathBuf::from("."),
                source,
            })?
            .join(&request.cuda_toolkit_root)
    };
    let invocation_root = invocation_root_path.to_str().ok_or_else(|| {
        NativeOperatorBuilderError::Invalid(format!(
            "cuda_toolkit_root is not valid UTF-8: {}",
            invocation_root_path.display()
        ))
    })?;
    validate_normalized_absolute_path(invocation_root, "cuda_toolkit_root")?;
    let canonical_root = request.cuda_toolkit_root.canonicalize().map_err(|source| {
        NativeOperatorBuilderError::Io {
            path: request.cuda_toolkit_root.clone(),
            source,
        }
    })?;
    if !canonical_root.is_dir() {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "cuda_toolkit_root is not a directory: {}",
            canonical_root.display()
        )));
    }
    let canonical_root_string = canonical_root.to_str().ok_or_else(|| {
        NativeOperatorBuilderError::Invalid(format!(
            "canonical cuda_toolkit_root is not valid UTF-8: {}",
            canonical_root.display()
        ))
    })?;
    validate_normalized_absolute_path(canonical_root_string, "canonical cuda_toolkit_root")?;
    let nvcc = tool_file_identity(&request.nvcc_path)?;
    let canonical_nvcc = Path::new(&nvcc.path);
    if !canonical_nvcc.starts_with(&canonical_root) {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "nvcc must resolve inside cuda_toolkit_root: nvcc={} root={}",
            canonical_nvcc.display(),
            canonical_root.display()
        )));
    }
    let manifest = build_cuda_toolkit_manifest(&canonical_root)?;
    if !manifest.entries.iter().any(|entry| {
        canonical_root.join(&entry.resolved_path) == canonical_nvcc
            && entry.sha256 == nvcc.sha256
            && entry.size_bytes == nvcc.size_bytes
    }) {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "cuda toolkit manifest does not contain the selected nvcc: {}",
            canonical_nvcc.display()
        )));
    }
    let manifest_relative = "toolchain/cuda-static-manifest.json";
    let manifest_path = request.output_dir.join(manifest_relative);
    let manifest_parent = manifest_path
        .parent()
        .expect("cuda toolkit manifest has a parent directory");
    fs::create_dir_all(manifest_parent).map_err(|source| NativeOperatorBuilderError::Io {
        path: manifest_parent.to_path_buf(),
        source,
    })?;
    write_json(&manifest_path, &manifest)?;
    let manifest_size = fs::metadata(&manifest_path)
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: manifest_path.clone(),
            source,
        })?
        .len();
    let manifest_evidence = NativeOperatorEvidenceFile {
        path: manifest_relative.to_string(),
        sha256: sha256_file(&manifest_path)?,
        size_bytes: manifest_size,
    };
    let host_toolchain = resolve_host_toolchain(request)?;
    Ok(NativeOperatorSourceBuildToolchain {
        static_identity: NativeOperatorSourceBuildStaticToolchain {
            backend: NativeOperatorBackend::Cuda,
            compiler_driver: NativeOperatorSourceCompilerDriver::CudaNvcc,
            cuda_toolkit: NativeOperatorCudaToolkitIdentity {
                canonical_root: canonical_root_string.to_string(),
                invocation_root: invocation_root.to_string(),
                release_version: cuda_toolkit_release_version(&canonical_root)?,
                nvcc,
                manifest: manifest_evidence,
            },
            host_toolchain,
            archiver: tool_file_identity(&request.ar_path)?,
        },
        miss_probe: None,
    })
}

fn resolve_host_toolchain(
    request: &NativeOperatorSourceBuildRequest,
) -> Result<NativeOperatorHostToolchainIdentity> {
    let compiler = tool_file_identity(&request.ccbin_path)?;
    let cache_key = sha256_bytes(format!("{}\n{}\n", compiler.path, compiler.sha256).as_bytes());
    let cache_dir = request
        .object_cache_dir
        .join(".host-toolchains")
        .join(cache_key);
    fs::create_dir_all(&cache_dir).map_err(|source| NativeOperatorBuilderError::Io {
        path: cache_dir.clone(),
        source,
    })?;
    let cached_path = cache_dir.join("manifest.json");
    let environment = effective_environment_for_tool_paths([
        request.nvcc_path.to_str().unwrap_or(""),
        compiler.path.as_str(),
        request.ar_path.to_str().unwrap_or(""),
    ])?;
    let cached = cached_path
        .is_file()
        .then(|| read_json::<NativeOperatorHostToolchainManifest>(&cached_path))
        .transpose()
        .ok()
        .flatten()
        .filter(|manifest| {
            validate_host_toolchain_manifest("<host-toolchain-cache>", manifest).is_ok()
        });
    let manifest = match cached {
        Some(cached)
            if host_toolchain_manifest_matches_current(&cached, &compiler, &environment)? =>
        {
            cached
        }
        _ => {
            let probed = probe_host_toolchain_manifest(&compiler, &environment)?;
            write_json(&cached_path, &probed)?;
            probed
        }
    };

    let relative = "toolchain/host-static-manifest.json";
    let output_path = request.output_dir.join(relative);
    write_json(&output_path, &manifest)?;
    let size_bytes = fs::metadata(&output_path)
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: output_path.clone(),
            source,
        })?
        .len();
    Ok(NativeOperatorHostToolchainIdentity {
        compiler: manifest.compiler.clone(),
        compiler_version: manifest.compiler_version.clone(),
        target: manifest.target.clone(),
        manifest: NativeOperatorEvidenceFile {
            path: relative.to_string(),
            sha256: sha256_file(&output_path)?,
            size_bytes,
        },
    })
}

fn probe_host_toolchain_manifest(
    compiler: &NativeOperatorToolFileIdentity,
    environment: &BTreeMap<String, String>,
) -> Result<NativeOperatorHostToolchainManifest> {
    let compiler_path = Path::new(&compiler.path);
    let compiler_version = host_compiler_output(compiler_path, &["--version"], b"", environment)?;
    let target = host_compiler_output(compiler_path, &["-dumpmachine"], b"", environment)?;
    if target.is_empty() || target.len() > 256 || target.chars().any(char::is_whitespace) {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "host compiler produced an invalid target: {}",
            compiler.path
        )));
    }

    let discovery_probe = probe_host_compiler_discovery(compiler_path, environment)?;

    let mut executable_inputs = BTreeMap::new();
    executable_inputs.insert(compiler.path.clone(), compiler.clone());
    for program in HOST_TOOLCHAIN_PROGRAMS {
        let value = host_compiler_output(
            compiler_path,
            &[&format!("-print-prog-name={program}")],
            b"",
            environment,
        )?;
        if let Some(path) = resolve_host_program(&value, environment)? {
            let identity = tool_file_identity(&path)?;
            executable_inputs.insert(identity.path.clone(), identity);
        }
    }

    let mut discovery_roots = executable_inputs
        .values()
        .filter_map(|identity| Path::new(&identity.path).parent())
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>();
    discovery_roots.sort();
    discovery_roots.dedup();
    let scope_roots = discovery_probe
        .include_roots
        .iter()
        .chain(discovery_roots.iter())
        .cloned()
        .collect::<Vec<_>>();
    let files = collect_host_toolchain_scope_files(&scope_roots)?;
    let manifest = NativeOperatorHostToolchainManifest {
        schema_version: NATIVE_OPERATOR_HOST_TOOLCHAIN_MANIFEST_SCHEMA_VERSION,
        compiler: compiler.clone(),
        compiler_version,
        target,
        executable_inputs: executable_inputs.into_values().collect(),
        include_roots: discovery_probe.include_roots,
        include_probe_sha256: discovery_probe.include_probe_sha256,
        driver_probe_sha256: discovery_probe.driver_probe_sha256,
        discovery_roots,
        files,
    };
    validate_host_toolchain_manifest("<host-toolchain-probe>", &manifest)?;
    Ok(manifest)
}

fn rebuild_host_toolchain_manifest(
    recorded: &NativeOperatorHostToolchainManifest,
) -> Result<NativeOperatorHostToolchainManifest> {
    validate_host_toolchain_manifest("<host-toolchain-recorded>", recorded)?;
    let compiler = tool_file_identity(Path::new(&recorded.compiler.path))?;
    let executable_inputs = recorded
        .executable_inputs
        .iter()
        .map(|identity| tool_file_identity(Path::new(&identity.path)))
        .collect::<Result<Vec<_>>>()?;
    let scope_roots = recorded
        .include_roots
        .iter()
        .chain(recorded.discovery_roots.iter())
        .cloned()
        .collect::<Vec<_>>();
    let files = collect_host_toolchain_scope_files(&scope_roots)?;
    let current = NativeOperatorHostToolchainManifest {
        schema_version: NATIVE_OPERATOR_HOST_TOOLCHAIN_MANIFEST_SCHEMA_VERSION,
        compiler,
        compiler_version: recorded.compiler_version.clone(),
        target: recorded.target.clone(),
        executable_inputs,
        include_roots: recorded.include_roots.clone(),
        include_probe_sha256: recorded.include_probe_sha256.clone(),
        driver_probe_sha256: recorded.driver_probe_sha256.clone(),
        discovery_roots: recorded.discovery_roots.clone(),
        files,
    };
    validate_host_toolchain_manifest("<host-toolchain-current>", &current)?;
    Ok(current)
}

struct HostCompilerDiscoveryProbe {
    include_roots: Vec<String>,
    include_probe_sha256: String,
    driver_probe_sha256: String,
}

fn probe_host_compiler_discovery(
    compiler: &Path,
    environment: &BTreeMap<String, String>,
) -> Result<HostCompilerDiscoveryProbe> {
    let include_probe = host_compiler_raw_output(
        compiler,
        &["-E", "-x", "c++", "-", "-v"],
        b"\n",
        environment,
    )?;
    let include_roots = parse_host_compiler_include_roots(
        &String::from_utf8_lossy(&include_probe.stderr),
        compiler,
    )?;
    let driver_probe = host_compiler_raw_output(
        compiler,
        &["-###", "-pipe", "-x", "c++", "-c", "-", "-o", "/dev/null"],
        b"\n",
        environment,
    )?;
    Ok(HostCompilerDiscoveryProbe {
        include_roots,
        include_probe_sha256: compiler_probe_sha256(&include_probe),
        driver_probe_sha256: compiler_probe_sha256(&driver_probe),
    })
}

fn compiler_probe_sha256(output: &std::process::Output) -> String {
    let mut identity = Vec::with_capacity(output.stdout.len() + output.stderr.len() + 16);
    identity.extend_from_slice(&(output.stdout.len() as u64).to_le_bytes());
    identity.extend_from_slice(&output.stdout);
    identity.extend_from_slice(&(output.stderr.len() as u64).to_le_bytes());
    identity.extend_from_slice(&output.stderr);
    sha256_bytes(&identity)
}

fn host_toolchain_manifest_matches_current(
    recorded: &NativeOperatorHostToolchainManifest,
    compiler: &NativeOperatorToolFileIdentity,
    environment: &BTreeMap<String, String>,
) -> Result<bool> {
    if &recorded.compiler != compiler || rebuild_host_toolchain_manifest(recorded)? != *recorded {
        return Ok(false);
    }
    let discovery = probe_host_compiler_discovery(Path::new(&compiler.path), environment)?;
    Ok(discovery.include_roots == recorded.include_roots
        && discovery.include_probe_sha256 == recorded.include_probe_sha256
        && discovery.driver_probe_sha256 == recorded.driver_probe_sha256)
}

fn host_compiler_output(
    compiler: &Path,
    args: &[&str],
    stdin: &[u8],
    environment: &BTreeMap<String, String>,
) -> Result<String> {
    let output = host_compiler_raw_output(compiler, args, stdin, environment)?;
    let value = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
    .trim()
    .chars()
    .take(16_384)
    .collect::<String>();
    if value.is_empty() {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "host compiler produced no output for {:?}: {}",
            args,
            compiler.display()
        )));
    }
    Ok(value)
}

fn host_compiler_raw_output(
    compiler: &Path,
    args: &[&str],
    stdin: &[u8],
    environment: &BTreeMap<String, String>,
) -> Result<std::process::Output> {
    let mut child = Command::new(compiler)
        .args(args)
        .env_clear()
        .envs(environment)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: compiler.to_path_buf(),
            source,
        })?;
    let stdin_result = {
        let mut child_stdin = child.stdin.take().expect("host compiler stdin is piped");
        let result = child_stdin.write_all(stdin);
        drop(child_stdin);
        result
    };
    let output = child
        .wait_with_output()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: compiler.to_path_buf(),
            source,
        })?;
    if let Err(source) = stdin_result {
        // A successful compiler probe may close stdin before the parent gets
        // scheduled to write the tiny probe payload. The child exit status is
        // authoritative in that case; treating EPIPE as a tool-access failure
        // makes concurrent probes flaky and can also leave the child unreaped.
        if source.kind() != std::io::ErrorKind::BrokenPipe {
            return Err(NativeOperatorBuilderError::Io {
                path: compiler.to_path_buf(),
                source,
            });
        }
    }
    if !output.status.success() {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "host compiler probe failed for {:?}: path={} status={}",
            args,
            compiler.display(),
            output.status
        )));
    }
    Ok(output)
}

fn parse_host_compiler_include_roots(stderr: &str, compiler: &Path) -> Result<Vec<String>> {
    let mut in_search_list = false;
    let mut roots = Vec::new();
    let mut seen = BTreeSet::new();
    for line in stderr.lines() {
        let trimmed = line.trim();
        if trimmed == "#include <...> search starts here:" {
            in_search_list = true;
            continue;
        }
        if in_search_list && trimmed == "End of search list." {
            break;
        }
        if !in_search_list || trimmed.is_empty() {
            continue;
        }
        let path = trimmed
            .strip_suffix(" (framework directory)")
            .unwrap_or(trimmed);
        validate_normalized_absolute_path(path, "host compiler include search entry")?;
        let canonical =
            Path::new(path)
                .canonicalize()
                .map_err(|source| NativeOperatorBuilderError::Io {
                    path: PathBuf::from(path),
                    source,
                })?;
        if !canonical.is_dir() {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "host compiler include search entry is not a directory: {}",
                canonical.display()
            )));
        }
        if seen.insert(path.to_string()) {
            roots.push(path.to_string());
        }
    }
    if roots.is_empty() {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "host compiler include search probe produced no roots: {}",
            compiler.display()
        )));
    }
    Ok(roots)
}

fn resolve_host_program(
    value: &str,
    environment: &BTreeMap<String, String>,
) -> Result<Option<PathBuf>> {
    if value.is_empty()
        || value
            .chars()
            .any(|character| matches!(character, '\n' | '\r'))
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "host compiler returned an invalid program path: {value:?}"
        )));
    }
    let candidate = Path::new(value);
    if candidate.is_absolute() {
        return Ok(candidate.is_file().then(|| candidate.to_path_buf()));
    }
    let path = environment.get("PATH").ok_or_else(|| {
        NativeOperatorBuilderError::Invalid(
            "host compiler probe environment has no PATH".to_string(),
        )
    })?;
    Ok(std::env::split_paths(std::ffi::OsStr::new(path))
        .map(|directory| directory.join(candidate))
        .find(|candidate| candidate.is_file()))
}

fn collect_host_toolchain_scope_files(
    roots: &[String],
) -> Result<Vec<NativeOperatorHostToolchainFileIdentity>> {
    let mut files = BTreeMap::new();
    for root in roots {
        collect_host_toolchain_files(Path::new(root), &mut BTreeSet::new(), &mut files)?;
    }
    if files.len() > MAX_HOST_TOOLCHAIN_FILES {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "host toolchain manifest must contain at most {MAX_HOST_TOOLCHAIN_FILES} sorted unique files"
        )));
    }
    Ok(files.into_values().collect())
}

fn collect_host_toolchain_files(
    directory: &Path,
    active_directories: &mut BTreeSet<PathBuf>,
    files: &mut BTreeMap<String, NativeOperatorHostToolchainFileIdentity>,
) -> Result<()> {
    let resolved_directory =
        directory
            .canonicalize()
            .map_err(|source| NativeOperatorBuilderError::Io {
                path: directory.to_path_buf(),
                source,
            })?;
    if !resolved_directory.is_dir() {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "host toolchain scope is not a directory: {}",
            directory.display()
        )));
    }
    if !active_directories.insert(resolved_directory.clone()) {
        return Ok(());
    }
    let mut children = fs::read_dir(directory)
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: directory.to_path_buf(),
            source,
        })?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: directory.to_path_buf(),
            source,
        })?;
    children.sort_by_key(|entry| entry.file_name());
    for child in children {
        let logical = child.path();
        let resolved = logical
            .canonicalize()
            .map_err(|source| NativeOperatorBuilderError::Io {
                path: logical.clone(),
                source,
            })?;
        if resolved.is_dir() {
            collect_host_toolchain_files(&logical, active_directories, files)?;
        } else if resolved.is_file() {
            let logical_path = logical.display().to_string();
            if !files.contains_key(&logical_path) && files.len() >= MAX_HOST_TOOLCHAIN_FILES {
                return Err(NativeOperatorBuilderError::Invalid(format!(
                    "host toolchain manifest exceeds {MAX_HOST_TOOLCHAIN_FILES} files"
                )));
            }
            let size_bytes = fs::metadata(&resolved)
                .map_err(|source| NativeOperatorBuilderError::Io {
                    path: resolved.clone(),
                    source,
                })?
                .len();
            let identity = NativeOperatorHostToolchainFileIdentity {
                logical_path,
                resolved_path: resolved.display().to_string(),
                sha256: sha256_file(&resolved)?,
                size_bytes,
            };
            if let Some(existing) = files.insert(identity.logical_path.clone(), identity.clone()) {
                if existing != identity {
                    return Err(NativeOperatorBuilderError::Invalid(format!(
                        "host toolchain scope resolved inconsistently: {}",
                        identity.logical_path
                    )));
                }
            }
        } else {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "host toolchain scope contains a non-file entry: {}",
                logical.display()
            )));
        }
    }
    active_directories.remove(&resolved_directory);
    Ok(())
}

fn validate_host_toolchain_manifest(
    context: &str,
    manifest: &NativeOperatorHostToolchainManifest,
) -> Result<()> {
    if manifest.schema_version != NATIVE_OPERATOR_HOST_TOOLCHAIN_MANIFEST_SCHEMA_VERSION
        || manifest.compiler_version.trim().is_empty()
        || manifest.target.trim().is_empty()
        || manifest.target.len() > 256
        || manifest.target.chars().any(char::is_whitespace)
        || manifest.executable_inputs.is_empty()
        || manifest.include_roots.is_empty()
        || !is_sha256_digest(&manifest.include_probe_sha256)
        || !is_sha256_digest(&manifest.driver_probe_sha256)
        || manifest.discovery_roots.is_empty()
        || manifest.files.is_empty()
        || manifest
            .executable_inputs
            .windows(2)
            .any(|pair| pair[0].path >= pair[1].path)
        || manifest.include_roots.iter().collect::<BTreeSet<_>>().len()
            != manifest.include_roots.len()
        || manifest
            .discovery_roots
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || manifest
            .files
            .windows(2)
            .any(|pair| pair[0].logical_path >= pair[1].logical_path)
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{context} host toolchain manifest header/order is invalid"
        )));
    }
    for tool in std::iter::once(&manifest.compiler).chain(manifest.executable_inputs.iter()) {
        validate_normalized_absolute_path(
            &tool.path,
            &format!("{context} host toolchain executable path"),
        )?;
        if !is_sha256_digest(&tool.sha256) || tool.size_bytes == 0 {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "{context} host toolchain executable identity is invalid: {}",
                tool.path
            )));
        }
    }
    for root in manifest
        .include_roots
        .iter()
        .chain(manifest.discovery_roots.iter())
    {
        validate_normalized_absolute_path(root, &format!("{context} host toolchain scope root"))?;
    }
    if !manifest
        .executable_inputs
        .iter()
        .any(|tool| tool == &manifest.compiler)
        || manifest.executable_inputs.iter().any(|tool| {
            Path::new(&tool.path).parent().map_or(true, |parent| {
                !manifest
                    .discovery_roots
                    .iter()
                    .any(|root| Path::new(root) == parent)
            })
        })
        || manifest.discovery_roots.iter().any(|root| {
            !manifest
                .executable_inputs
                .iter()
                .any(|tool| Path::new(&tool.path).parent() == Some(Path::new(root)))
        })
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{context} host toolchain manifest does not bind its compiler/search roots"
        )));
    }
    for file in &manifest.files {
        validate_normalized_absolute_path(
            &file.logical_path,
            &format!("{context} host toolchain logical path"),
        )?;
        validate_normalized_absolute_path(
            &file.resolved_path,
            &format!("{context} host toolchain resolved path"),
        )?;
        if !is_sha256_digest(&file.sha256)
            || !manifest
                .include_roots
                .iter()
                .chain(manifest.discovery_roots.iter())
                .any(|root| Path::new(&file.logical_path).starts_with(root))
        {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "{context} host toolchain file identity is invalid: {}",
                file.logical_path
            )));
        }
    }
    Ok(())
}

fn build_cuda_toolkit_manifest(root: &Path) -> Result<NativeOperatorCudaToolkitManifest> {
    let canonical_root = root
        .canonicalize()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: root.to_path_buf(),
            source,
        })?;
    let root = canonical_root.as_path();
    let mut entries = Vec::new();
    for relative in REQUIRED_CUDA_TOOLKIT_FILES {
        collect_cuda_toolkit_single_file(root, relative, &mut entries)?;
    }
    for optional in ["bin/cudafe", "bin/nvcc.profile"] {
        if root.join(optional).exists() {
            collect_cuda_toolkit_single_file(root, optional, &mut entries)?;
        }
    }
    for scope in REQUIRED_CUDA_TOOLKIT_SCOPES {
        let scope_path = root.join(scope);
        if !scope_path.is_dir() {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "cuda toolkit compiler scope is missing: {scope}"
            )));
        }
        collect_cuda_toolkit_files(root, &scope_path, &mut BTreeSet::new(), &mut entries)?;
    }
    entries.sort_by(|left, right| left.logical_path.cmp(&right.logical_path));
    if entries.is_empty()
        || entries
            .windows(2)
            .any(|pair| pair[0].logical_path >= pair[1].logical_path)
    {
        return Err(NativeOperatorBuilderError::Invalid(
            "cuda toolkit manifest entries must be non-empty, sorted, and unique".to_string(),
        ));
    }
    Ok(NativeOperatorCudaToolkitManifest {
        schema_version: NATIVE_OPERATOR_CUDA_TOOLKIT_MANIFEST_SCHEMA_VERSION,
        canonical_root: root.display().to_string(),
        entries,
    })
}

fn collect_cuda_toolkit_single_file(
    root: &Path,
    relative: &str,
    entries: &mut Vec<NativeOperatorCudaToolkitFileIdentity>,
) -> Result<()> {
    validate_relative_path(relative)?;
    let logical = root.join(relative);
    let resolved = logical
        .canonicalize()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: logical.clone(),
            source,
        })?;
    if !resolved.starts_with(root) || !resolved.is_file() {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "cuda toolkit compiler input escapes root or is not a file: {relative}"
        )));
    }
    entries.push(cuda_toolkit_file_identity(root, &logical, &resolved)?);
    Ok(())
}

fn cuda_toolkit_release_version(root: &Path) -> Result<String> {
    let cuda_header = root.join("include/cuda.h");
    require_file(&cuda_header)?;
    let contents =
        fs::read_to_string(&cuda_header).map_err(|source| NativeOperatorBuilderError::Io {
            path: cuda_header.clone(),
            source,
        })?;
    let encoded = contents
        .lines()
        .find_map(|line| {
            let mut fields = line.split_whitespace();
            match (fields.next(), fields.next(), fields.next(), fields.next()) {
                (Some("#define"), Some("CUDA_VERSION"), Some(value), None) => {
                    value.parse::<u32>().ok()
                }
                _ => None,
            }
        })
        .filter(|value| *value >= 1000)
        .ok_or_else(|| {
            NativeOperatorBuilderError::Invalid(format!(
                "cuda toolkit include/cuda.h has no valid CUDA_VERSION: {}",
                cuda_header.display()
            ))
        })?;
    Ok(format!(
        "{}.{}.{}",
        encoded / 1000,
        (encoded % 1000) / 10,
        encoded % 10
    ))
}

fn collect_cuda_toolkit_files(
    root: &Path,
    directory: &Path,
    active_directories: &mut BTreeSet<PathBuf>,
    entries: &mut Vec<NativeOperatorCudaToolkitFileIdentity>,
) -> Result<()> {
    let resolved_directory =
        directory
            .canonicalize()
            .map_err(|source| NativeOperatorBuilderError::Io {
                path: directory.to_path_buf(),
                source,
            })?;
    if !resolved_directory.starts_with(root) || !resolved_directory.is_dir() {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "cuda toolkit compiler directory escapes its canonical root: {}",
            directory.display()
        )));
    }
    if !active_directories.insert(resolved_directory.clone()) {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "cuda toolkit compiler directory contains a symlink cycle: {}",
            directory.display()
        )));
    }
    let mut children = fs::read_dir(directory)
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: directory.to_path_buf(),
            source,
        })?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: directory.to_path_buf(),
            source,
        })?;
    children.sort_by_key(|entry| entry.file_name());
    for child in children {
        let logical_path = child.path();
        let resolved =
            logical_path
                .canonicalize()
                .map_err(|source| NativeOperatorBuilderError::Io {
                    path: logical_path.clone(),
                    source,
                })?;
        if !resolved.starts_with(root) {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "cuda toolkit symlink escapes its canonical root: {}",
                logical_path.display()
            )));
        }
        if resolved.is_dir() {
            collect_cuda_toolkit_files(root, &logical_path, active_directories, entries)?;
        } else if resolved.is_file() {
            entries.push(cuda_toolkit_file_identity(root, &logical_path, &resolved)?);
        } else {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "cuda toolkit compiler scope contains a non-file entry: {}",
                logical_path.display()
            )));
        }
    }
    active_directories.remove(&resolved_directory);
    Ok(())
}

fn cuda_toolkit_file_identity(
    root: &Path,
    logical: &Path,
    resolved: &Path,
) -> Result<NativeOperatorCudaToolkitFileIdentity> {
    let logical_path = logical.strip_prefix(root).map_err(|_| {
        NativeOperatorBuilderError::Invalid(format!(
            "cuda toolkit logical path escapes root: {}",
            logical.display()
        ))
    })?;
    let resolved_path = resolved.strip_prefix(root).map_err(|_| {
        NativeOperatorBuilderError::Invalid(format!(
            "cuda toolkit resolved path escapes root: {}",
            resolved.display()
        ))
    })?;
    let size_bytes = fs::metadata(resolved)
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: resolved.to_path_buf(),
            source,
        })?
        .len();
    Ok(NativeOperatorCudaToolkitFileIdentity {
        logical_path: path_with_forward_slashes(logical_path)?,
        resolved_path: path_with_forward_slashes(resolved_path)?,
        sha256: sha256_file(resolved)?,
        size_bytes,
    })
}

fn path_with_forward_slashes(path: &Path) -> Result<String> {
    let components = path
        .components()
        .map(|component| match component {
            std::path::Component::Normal(value) => {
                value.to_str().map(str::to_string).ok_or_else(|| {
                    NativeOperatorBuilderError::Invalid(format!(
                        "native build path is not valid UTF-8: {}",
                        path.display()
                    ))
                })
            }
            _ => Err(NativeOperatorBuilderError::Invalid(format!(
                "native build path is not normalized and relative: {}",
                path.display()
            ))),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(components.join("/"))
}

fn tool_file_identity(path: &Path) -> Result<NativeOperatorToolFileIdentity> {
    require_file(path)?;
    let canonical = path
        .canonicalize()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let size_bytes = fs::metadata(&canonical)
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: canonical.clone(),
            source,
        })?
        .len();
    if size_bytes == 0 {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "source build tool is empty: {}",
            canonical.display()
        )));
    }
    Ok(NativeOperatorToolFileIdentity {
        path: canonical.display().to_string(),
        sha256: sha256_file(&canonical)?,
        size_bytes,
    })
}

fn probe_source_toolchain(
    static_identity: &NativeOperatorSourceBuildStaticToolchain,
    missed_translation_units: Vec<String>,
) -> Result<NativeOperatorSourceBuildToolchainProbe> {
    Ok(NativeOperatorSourceBuildToolchainProbe {
        nvcc_version: tool_version(Path::new(&static_identity.cuda_toolkit.nvcc.path))?,
        host_compiler_version: static_identity.host_toolchain.compiler_version.clone(),
        host_target: static_identity.host_toolchain.target.clone(),
        archiver_version: tool_version(Path::new(&static_identity.archiver.path))?,
        probed_for_misses: missed_translation_units,
    })
}

fn validate_static_toolchain_identity(
    operator: &str,
    toolchain: &NativeOperatorSourceBuildStaticToolchain,
) -> Result<()> {
    if toolchain.backend != NativeOperatorBackend::Cuda
        || toolchain.compiler_driver != NativeOperatorSourceCompilerDriver::CudaNvcc
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{operator} source-build toolchain must use the CUDA nvcc driver"
        )));
    }
    validate_normalized_absolute_path(
        &toolchain.cuda_toolkit.canonical_root,
        &format!("{operator} cuda toolkit canonical_root"),
    )?;
    validate_normalized_absolute_path(
        &toolchain.cuda_toolkit.invocation_root,
        &format!("{operator} cuda toolkit invocation_root"),
    )?;
    if toolchain.cuda_toolkit.release_version.trim().is_empty()
        || toolchain
            .cuda_toolkit
            .release_version
            .chars()
            .any(|character| !(character.is_ascii_digit() || character == '.'))
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{operator} cuda toolkit release_version is invalid"
        )));
    }
    for (name, tool) in [
        ("nvcc", &toolchain.cuda_toolkit.nvcc),
        ("host_compiler", &toolchain.host_toolchain.compiler),
        ("archiver", &toolchain.archiver),
    ] {
        if validate_normalized_absolute_path(
            &tool.path,
            &format!("{operator} source-build static {name} path"),
        )
        .is_err()
            || !is_sha256_digest(&tool.sha256)
            || tool.size_bytes == 0
        {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "{operator} source-build static {name} identity is incomplete"
            )));
        }
    }
    if !Path::new(&toolchain.cuda_toolkit.nvcc.path)
        .starts_with(&toolchain.cuda_toolkit.canonical_root)
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{operator} nvcc identity escapes cuda toolkit root"
        )));
    }
    let manifest = &toolchain.cuda_toolkit.manifest;
    if manifest.path != "toolchain/cuda-static-manifest.json"
        || !is_sha256_digest(&manifest.sha256)
        || manifest.size_bytes == 0
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{operator} cuda toolkit manifest evidence is incomplete"
        )));
    }
    let host = &toolchain.host_toolchain;
    if host.compiler_version.trim().is_empty()
        || host.target.trim().is_empty()
        || host.target.len() > 256
        || host.target.chars().any(char::is_whitespace)
        || host.manifest.path != "toolchain/host-static-manifest.json"
        || !is_sha256_digest(&host.manifest.sha256)
        || host.manifest.size_bytes == 0
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{operator} host toolchain manifest evidence is incomplete"
        )));
    }
    Ok(())
}

fn validate_tool_file_unchanged(identity: &NativeOperatorToolFileIdentity) -> Result<()> {
    let current = tool_file_identity(Path::new(&identity.path))?;
    if &current != identity {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "tool file changed after static identity was recorded: {}",
            identity.path
        )));
    }
    Ok(())
}

fn validate_cuda_toolkit_unchanged(identity: &NativeOperatorCudaToolkitIdentity) -> Result<()> {
    let manifest_path = Path::new(&identity.canonical_root);
    let current = build_cuda_toolkit_manifest(manifest_path)?;
    validate_cuda_toolkit_manifest("<source-build-finalize>", identity, &current)?;
    let recorded_path = Path::new(&identity.canonical_root);
    if current.canonical_root != recorded_path.display().to_string() {
        return Err(NativeOperatorBuilderError::Invalid(
            "cuda toolkit canonical root changed during source build".to_string(),
        ));
    }
    let recorded_manifest = identity.manifest.sha256.as_str();
    let serialized_with_newline = {
        let mut bytes = serde_json::to_vec_pretty(&current).map_err(|source| {
            NativeOperatorBuilderError::Json {
                path: PathBuf::from("<cuda-static-manifest>"),
                source,
            }
        })?;
        bytes.push(b'\n');
        sha256_bytes(&bytes)
    };
    if serialized_with_newline != recorded_manifest {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "cuda toolkit manifest changed during source build: expected={recorded_manifest} actual={serialized_with_newline}"
        )));
    }
    Ok(())
}

fn validate_host_toolchain_unchanged(
    identity: &NativeOperatorHostToolchainIdentity,
    receipt_root: &Path,
    environment: &BTreeMap<String, String>,
) -> Result<()> {
    let manifest_path = resolve_source_build_evidence_file(
        receipt_root,
        "<source-build-finalize>",
        &identity.manifest,
    )?;
    let recorded: NativeOperatorHostToolchainManifest = read_json(&manifest_path)?;
    validate_host_toolchain_manifest("<source-build-finalize>", &recorded)?;
    if recorded.compiler != identity.compiler
        || recorded.compiler_version != identity.compiler_version
        || recorded.target != identity.target
    {
        return Err(NativeOperatorBuilderError::Invalid(
            "host toolchain identity differs from its manifest".to_string(),
        ));
    }
    if !host_toolchain_manifest_matches_current(&recorded, &identity.compiler, environment)? {
        return Err(NativeOperatorBuilderError::Invalid(
            "host toolchain files or driver configuration changed during source build".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn compiler_target(path: &Path) -> Result<String> {
    require_file(path)?;
    let canonical = path
        .canonicalize()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let output = Command::new(&canonical)
        .arg("-dumpmachine")
        .env_clear()
        .env("LANG", "C")
        .env("LC_ALL", "C")
        .env("TZ", "UTC")
        .output()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: canonical.clone(),
            source,
        })?;
    let target = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if !output.status.success()
        || target.is_empty()
        || target.len() > 256
        || target.chars().any(char::is_whitespace)
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "compiler produced no valid target identity: {}",
            canonical.display()
        )));
    }
    Ok(target)
}

pub(crate) fn native_object_identity_file(path: &Path) -> Result<NativeOperatorObjectIdentity> {
    let bytes = fs::read(path).map_err(|source| NativeOperatorBuilderError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    native_object_identity_bytes(&bytes, &path.display().to_string())
}

fn native_object_size(path: &Path) -> Result<u64> {
    let size_bytes = fs::metadata(path)
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: path.to_path_buf(),
            source,
        })?
        .len();
    if size_bytes == 0 {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "native object is empty: {}",
            path.display()
        )));
    }
    Ok(size_bytes)
}

pub(crate) fn native_object_identity_bytes(
    bytes: &[u8],
    context: &str,
) -> Result<NativeOperatorObjectIdentity> {
    if bytes.len() >= 20 && bytes.starts_with(b"\x7fELF") {
        let class_bits = match bytes[4] {
            1 => 32,
            2 => 64,
            value => {
                return Err(NativeOperatorBuilderError::Invalid(format!(
                    "unsupported ELF class in {context}: {value}"
                )))
            }
        };
        let endianness = match bytes[5] {
            1 => NativeOperatorObjectEndianness::Little,
            2 => NativeOperatorObjectEndianness::Big,
            value => {
                return Err(NativeOperatorBuilderError::Invalid(format!(
                    "unsupported ELF endianness in {context}: {value}"
                )))
            }
        };
        let machine = u32::from(read_u16(&bytes[18..20], endianness));
        let identity = NativeOperatorObjectIdentity {
            format: NativeOperatorObjectFormat::Elf,
            class_bits,
            endianness,
            machine,
        };
        validate_native_object_identity(&identity, context)?;
        return Ok(identity);
    }

    if bytes.len() >= 8 {
        let (class_bits, endianness) = match &bytes[..4] {
            [0xce, 0xfa, 0xed, 0xfe] => (32, NativeOperatorObjectEndianness::Little),
            [0xfe, 0xed, 0xfa, 0xce] => (32, NativeOperatorObjectEndianness::Big),
            [0xcf, 0xfa, 0xed, 0xfe] => (64, NativeOperatorObjectEndianness::Little),
            [0xfe, 0xed, 0xfa, 0xcf] => (64, NativeOperatorObjectEndianness::Big),
            _ => (0, NativeOperatorObjectEndianness::Little),
        };
        if class_bits != 0 {
            let machine = read_u32(&bytes[4..8], endianness);
            let identity = NativeOperatorObjectIdentity {
                format: NativeOperatorObjectFormat::MachO,
                class_bits,
                endianness,
                machine,
            };
            validate_native_object_identity(&identity, context)?;
            return Ok(identity);
        }
    }

    if bytes.len() >= 20 {
        let machine = u16::from_le_bytes([bytes[0], bytes[1]]);
        if matches!(machine, 0x014c | 0x01c0 | 0x01c4 | 0x8664 | 0xaa64) {
            let identity = NativeOperatorObjectIdentity {
                format: NativeOperatorObjectFormat::Coff,
                class_bits: if matches!(machine, 0x8664 | 0xaa64) {
                    64
                } else {
                    32
                },
                endianness: NativeOperatorObjectEndianness::Little,
                machine: u32::from(machine),
            };
            validate_native_object_identity(&identity, context)?;
            return Ok(identity);
        }
    }

    Err(NativeOperatorBuilderError::Invalid(format!(
        "native object has no supported ELF, Mach-O, or COFF header: {context}"
    )))
}

pub(crate) fn validate_native_object_identity(
    identity: &NativeOperatorObjectIdentity,
    context: &str,
) -> Result<()> {
    if !matches!(identity.class_bits, 32 | 64) || identity.machine == 0 {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "native object identity is incomplete for {context}: {identity:?}"
        )));
    }
    if identity.format == NativeOperatorObjectFormat::Coff
        && identity.endianness != NativeOperatorObjectEndianness::Little
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "COFF object must be little-endian for {context}"
        )));
    }
    Ok(())
}

fn read_u16(bytes: &[u8], endianness: NativeOperatorObjectEndianness) -> u16 {
    let bytes = [bytes[0], bytes[1]];
    match endianness {
        NativeOperatorObjectEndianness::Little => u16::from_le_bytes(bytes),
        NativeOperatorObjectEndianness::Big => u16::from_be_bytes(bytes),
    }
}

fn read_u32(bytes: &[u8], endianness: NativeOperatorObjectEndianness) -> u32 {
    let bytes = [bytes[0], bytes[1], bytes[2], bytes[3]];
    match endianness {
        NativeOperatorObjectEndianness::Little => u32::from_le_bytes(bytes),
        NativeOperatorObjectEndianness::Big => u32::from_be_bytes(bytes),
    }
}

pub(crate) fn tool_identity(path: &Path) -> Result<NativeOperatorToolIdentity> {
    require_file(path)?;
    let canonical = path
        .canonicalize()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(NativeOperatorToolIdentity {
        path: canonical.display().to_string(),
        sha256: sha256_file(&canonical)?,
        version: tool_version(&canonical)?,
    })
}

fn tool_version(path: &Path) -> Result<String> {
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
    Ok(version)
}

#[allow(clippy::too_many_arguments)]
fn build_inputs_sha256(
    plan_sha256: &str,
    source_package_sha256: &str,
    architecture_argument: &str,
    effective_environment: &BTreeMap<String, String>,
    toolchain: Option<&NativeOperatorSourceBuildStaticToolchain>,
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
            toolchain.static_identity.cuda_toolkit.nvcc.path.as_str(),
            toolchain
                .static_identity
                .host_toolchain
                .compiler
                .path
                .as_str(),
            toolchain.static_identity.archiver.path.as_str(),
        ]
    } else {
        [
            request.nvcc_path.to_str().unwrap_or(""),
            request.ccbin_path.to_str().unwrap_or(""),
            request.ar_path.to_str().unwrap_or(""),
        ]
    };
    effective_environment_for_tool_paths(tool_paths)
}

fn effective_environment_for_tool_paths(tool_paths: [&str; 3]) -> Result<BTreeMap<String, String>> {
    let mut path_entries = tool_paths
        .iter()
        .filter_map(|path| Path::new(path).parent())
        .map(Path::to_path_buf)
        .collect::<Vec<_>>();
    path_entries.extend([PathBuf::from("/bin"), PathBuf::from("/usr/bin")]);
    path_entries.sort();
    path_entries.dedup();
    if path_entries.iter().any(|path| path.as_os_str().is_empty()) {
        return Err(NativeOperatorBuilderError::Invalid(
            "source build tool paths must have parent directories".to_string(),
        ));
    }
    let path = std::env::join_paths(&path_entries)
        .map_err(|error| {
            NativeOperatorBuilderError::Invalid(format!(
                "source build tool PATH cannot be represented: {error}"
            ))
        })?
        .into_string()
        .map_err(|_| {
            NativeOperatorBuilderError::Invalid(
                "source build tool PATH is not valid UTF-8".to_string(),
            )
        })?;
    let mut environment = BTreeMap::new();
    environment.insert("LANG".to_string(), "C".to_string());
    environment.insert("LC_ALL".to_string(), "C".to_string());
    environment.insert("PATH".to_string(), path);
    environment.insert("SOURCE_DATE_EPOCH".to_string(), "0".to_string());
    environment.insert("TMPDIR".to_string(), "/tmp".to_string());
    environment.insert("TZ".to_string(), "UTC".to_string());
    environment.insert("ZERO_AR_DATE".to_string(), "1".to_string());
    Ok(environment)
}

fn build_object_cache_specs(
    plan: &NativeOperatorSourceBuildPlan,
    architecture_argument: &str,
    toolchain: &NativeOperatorSourceBuildStaticToolchain,
    effective_environment: &BTreeMap<String, String>,
) -> Result<Vec<NativeBuildArtifactSpec>> {
    plan.translation_units
        .iter()
        .enumerate()
        .map(|(index, translation_unit)| {
            let closure = plan.dependency_closures.get(index).ok_or_else(|| {
                NativeOperatorBuilderError::Invalid(format!(
                    "missing dependency closure for {}",
                    translation_unit.path
                ))
            })?;
            if closure.translation_unit != translation_unit.path {
                return Err(NativeOperatorBuilderError::Invalid(format!(
                    "dependency closure order differs from translation units: expected={} actual={}",
                    translation_unit.path, closure.translation_unit
                )));
            }
            let identity = NativeOperatorObjectInputIdentity {
                schema_version: NATIVE_OPERATOR_SOURCE_OBJECT_BUILD_CONTRACT_VERSION,
                operator: &plan.operator,
                translation_unit,
                dependency_closure_sha256: &closure.closure_sha256,
                headers: &closure.headers,
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
        "{index:08}_{}_{}.o",
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
        .map(|toolchain| toolchain.static_identity.cuda_toolkit.nvcc.path.as_str())
        .unwrap_or_else(|| request.nvcc_path.to_str().unwrap_or("<non-utf8-nvcc>"));
    let ccbin_path = toolchain
        .map(|toolchain| {
            toolchain
                .static_identity
                .host_toolchain
                .compiler
                .path
                .as_str()
        })
        .unwrap_or_else(|| request.ccbin_path.to_str().unwrap_or("<non-utf8-ccbin>"));
    let ar_path = toolchain
        .map(|toolchain| toolchain.static_identity.archiver.path.as_str())
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
        let depfile_name = format!("{index:08}-{stem}.d");
        let depfile_relative = format!("depfiles/{depfile_name}");
        let compiler_depfile_relative = format!("depfiles/{index:08}-{stem}.compiler.raw.d");
        let compiler_depfile_path = request.output_dir.join(&compiler_depfile_relative);
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
            "-MMD".to_string(),
            "-MF".to_string(),
            compiler_depfile_path.display().to_string(),
            "-MT".to_string(),
            object_path.display().to_string(),
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
            object_size_bytes: None,
            object_identity: None,
            dependency_closure_sha256: plan
                .dependency_closures
                .get(index)
                .map(|closure| closure.closure_sha256.clone()),
            dependency_validation: Some(if request.plan_only {
                NativeOperatorDependencyValidation::Plan
            } else {
                NativeOperatorDependencyValidation::Pending
            }),
            compiler_depfile: Some(compiler_depfile_relative),
            compiler_depfile_sha256: None,
            depfile: Some(depfile_relative),
            depfile_sha256: None,
            depfile_producer_working_directory: None,
            depfile_producer_object_file: None,
            depfile_bindings: Vec::new(),
            observed_dependencies: Vec::new(),
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
        object_size_bytes: None,
        object_identity: None,
        dependency_closure_sha256: None,
        dependency_validation: None,
        compiler_depfile: None,
        compiler_depfile_sha256: None,
        depfile: None,
        depfile_sha256: None,
        depfile_producer_working_directory: None,
        depfile_producer_object_file: None,
        depfile_bindings: Vec::new(),
        observed_dependencies: Vec::new(),
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
    use std::os::unix::fs::{symlink, PermissionsExt};

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
            dependency_closures: vec![NativeOperatorTranslationUnitDependencies {
                translation_unit: "kernels/marlin.cu".to_string(),
                headers: vec!["kernels/marlin.h".to_string()],
            }],
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
        fs::create_dir_all(source_root.join("kernels/core")).unwrap();
        fs::write(
            source_root.join("kernels/marlin.cu"),
            "#include \"marlin.h\"\n\
             int marlin_cuda(void) { return MARLIN - 1; }\n\
             int marlin_cuda_moe(void) { return 0; }\n",
        )
        .unwrap();
        fs::write(source_root.join("kernels/marlin.h"), "#define MARLIN 1\n").unwrap();
        let definition_path = root.join("source-definition.json");
        write_json(&definition_path, &definition()).unwrap();
        (source_root, definition_path)
    }

    struct FakeCudaToolkit {
        root: PathBuf,
        nvcc: PathBuf,
        ccbin: PathBuf,
        empty_host_include_root: PathBuf,
        compile_counter: PathBuf,
        invocation_counter: PathBuf,
        host_compiler_invocation_counter: PathBuf,
        host_driver_config: PathBuf,
    }

    #[derive(Clone, Copy)]
    enum FakeDepfileMode {
        Valid,
        MissingDeclaredHeader,
        UndeclaredExternal,
    }

    fn write_fake_nvcc(root: &Path) -> FakeCudaToolkit {
        write_fake_nvcc_with_mode(root, FakeDepfileMode::Valid)
    }

    fn write_fake_nvcc_with_mode(root: &Path, mode: FakeDepfileMode) -> FakeCudaToolkit {
        let toolkit_root = root.join("fake-cuda");
        for directory in ["bin/crt", "include", "nvvm/bin", "nvvm/libdevice"] {
            fs::create_dir_all(toolkit_root.join(directory)).unwrap();
        }
        for (relative, contents) in [
            ("bin/bin2c", "fake bin2c\n"),
            ("bin/crt/link.stub", "fake link stub\n"),
            ("bin/cudafe++", "fake cudafe\n"),
            ("bin/ptxas", "fake ptxas\n"),
            ("bin/fatbinary", "fake fatbinary\n"),
            ("bin/nvlink", "fake nvlink\n"),
            ("include/cuda.h", "#define CUDA_VERSION 12040\n"),
            ("nvvm/bin/cicc", "fake cicc\n"),
            ("nvvm/libdevice/libdevice.10.bc", "fake libdevice\n"),
        ] {
            fs::write(toolkit_root.join(relative), contents).unwrap();
        }
        let path = toolkit_root.join("bin/nvcc");
        let counter = root.join("fake-nvcc-compile-count");
        let invocation_counter = root.join("fake-nvcc-invocation-count");
        let host_root = root.join("fake-host-toolchain");
        let compile_tail = match mode {
            FakeDepfileMode::Valid => format!(
                "/usr/bin/cc -x c -c \"$src\" -o \"$out\" || exit $?\n\
                 declared_header=''\n\
                 case \"$src\" in */marlin.cu) declared_header=' kernels/core/../marlin.h' ;; esac\n\
                 printf '%s: %s%s %s %s\\n' \"$dep_target\" \"$src\" \"$declared_header\" '{}' '{}' > \"$depfile\"\n",
                toolkit_root.join("bin/../include/cuda.h").display(),
                host_root.join("include/stddef.h").display(),
            ),
            FakeDepfileMode::MissingDeclaredHeader => {
                "/usr/bin/cc -x c -MMD -MF \"$depfile\" -MT \"$dep_target\" -c \"$src\" -o \"$out\" || exit $?\n\
                 printf '%s: %s\\n' \"$dep_target\" \"$src\" > \"$depfile\"\n"
                    .to_string()
            }
            FakeDepfileMode::UndeclaredExternal => {
                "/usr/bin/cc -x c -MMD -MF \"$depfile\" -MT \"$dep_target\" -c \"$src\" -o \"$out\" || exit $?\n\
                 printf '%s: %s kernels/marlin.h /etc/hosts\\n' \"$dep_target\" \"$src\" > \"$depfile\"\n"
                    .to_string()
            }
        };
        fs::write(
            &path,
            format!(
                "#!/bin/sh\n\
             printf 'invoke:%s\\n' \"$*\" >> '{}'\n\
             if [ \"$1\" = \"--version\" ]; then echo 'fake nvcc 12.4'; exit 0; fi\n\
             src=''\n\
             out=''\n\
             depfile=''\n\
             dep_target=''\n\
             while [ \"$#\" -gt 0 ]; do\n\
               case \"$1\" in\n\
                 -c) src=\"$2\"; shift 2 ;;\n\
                 -o) out=\"$2\"; shift 2 ;;\n\
                 -MF) depfile=\"$2\"; shift 2 ;;\n\
                 -MT) dep_target=\"$2\"; shift 2 ;;\n\
                 *) shift ;;\n\
               esac\n\
             done\n\
             build_dir=$(dirname \"$(dirname \"$out\")\")\n\
             receipt=\"$build_dir/source-build.receipt.json\"\n\
             test -s \"$receipt\"\n\
             grep -q '\"status\": \"reject\"' \"$receipt\"\n\
             grep -q '\"failure_class\": \"build_incomplete\"' \"$receipt\"\n\
             printf 'compile\\n' >> '{}'\n\
             {}",
                invocation_counter.display(),
                counter.display(),
                compile_tail
            ),
        )
        .unwrap();
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&path, permissions).unwrap();

        fs::create_dir_all(host_root.join("bin")).unwrap();
        fs::create_dir_all(host_root.join("include")).unwrap();
        let empty_host_include_root = host_root.join("empty-include");
        fs::create_dir_all(&empty_host_include_root).unwrap();
        fs::write(
            host_root.join("include/stddef.h"),
            "#define FAKE_SIZE_T 1\n",
        )
        .unwrap();
        for program in HOST_TOOLCHAIN_PROGRAMS {
            fs::write(
                host_root.join("bin").join(program),
                format!("fake host tool {program}\n"),
            )
            .unwrap();
        }
        fs::write(
            host_root.join("bin/driver.specs"),
            "fake host driver configuration\n",
        )
        .unwrap();
        let ccbin = host_root.join("bin/c++");
        let host_compiler_invocation_counter = root.join("fake-host-compiler-invocation-count");
        let host_driver_config = root.join("fake-host-driver.conf");
        fs::write(&host_driver_config, "external driver option v1\n").unwrap();
        fs::write(
            &ccbin,
            format!(
                "#!/bin/sh\n\
                 printf 'invoke:%s\\n' \"$*\" >> '{}'\n\
                 case \"$1\" in\n\
                   --version) echo 'fake host compiler 1.0'; exit 0 ;;\n\
                   -dumpmachine) echo 'x86_64-ferrum-linux-gnu'; exit 0 ;;\n\
                   -E) echo '#include <...> search starts here:' >&2; echo ' {}' >&2; echo ' {}' >&2; echo 'End of search list.' >&2; exit 0 ;;\n\
                   -###) test \"$2\" = '-pipe' || exit 3; echo 'fake cc1plus -O2 -x c++' >&2; cat '{}' >&2; exit 0 ;;\n\
                   -print-prog-name=*) name=${{1#*=}}; echo '{}/bin/'\"$name\"; exit 0 ;;\n\
                 esac\n\
                 exit 2\n",
                host_compiler_invocation_counter.display(),
                host_root.join("include").display(),
                empty_host_include_root.display(),
                host_driver_config.display(),
                host_root.display(),
            ),
        )
        .unwrap();
        let mut permissions = fs::metadata(&ccbin).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&ccbin, permissions).unwrap();
        FakeCudaToolkit {
            root: toolkit_root,
            nvcc: path,
            ccbin,
            empty_host_include_root,
            compile_counter: counter,
            invocation_counter,
            host_compiler_invocation_counter,
            host_driver_config,
        }
    }

    #[test]
    fn object_file_names_preserve_lexical_order_past_one_hundred_units() {
        let translation_unit = NativeOperatorSourceFileLock {
            path: "kernels/unit.cu".to_string(),
            sha256: "a".repeat(64),
        };

        assert!(object_file_name(99, &translation_unit) < object_file_name(100, &translation_unit));
    }

    #[test]
    fn portable_depfile_serialization_round_trips_restricted_make_words() {
        let target = "/tmp/build path/object:name#$value.o";
        let dependencies = vec![
            "kernels/header name.h".to_string(),
            "/tmp/tool chain/header:name#$value.h".to_string(),
        ];

        let raw = serialize_portable_depfile(target, &dependencies).unwrap();
        let text = std::str::from_utf8(&raw).unwrap();
        let (parsed_target, parsed_dependencies) =
            parse_make_depfile(text, Path::new("<round-trip>")).unwrap();

        assert_eq!(parsed_target, target);
        assert_eq!(parsed_dependencies, dependencies);
        assert!(text.contains("\\ "));
        assert!(text.contains("\\:"));
        assert!(text.contains("\\#"));
        assert!(text.contains("\\$"));
    }

    #[test]
    fn source_depfile_paths_lexically_normalize_without_escaping_working_directory() {
        assert_eq!(
            normalize_portable_relative_path(Path::new(
                "kernels/vllm_marlin_moe/core/../vllm_torch_shim.h"
            ))
            .unwrap(),
            "kernels/vllm_marlin_moe/vllm_torch_shim.h"
        );
        for path in ["../outside.h", "kernels/../../outside.h"] {
            let error = normalize_portable_relative_path(Path::new(path)).unwrap_err();
            assert!(error.to_string().contains("escapes its working directory"));
        }
        let error = normalize_portable_relative_path(Path::new("kernels\\outside.h")).unwrap_err();
        assert!(error.to_string().contains("relative POSIX path"));
    }

    #[test]
    fn host_include_probe_preserves_search_order_and_deduplicates_first_occurrence() {
        let root = tempfile::tempdir().unwrap();
        let first = root.path().join("first include");
        let second = root.path().join("second include");
        fs::create_dir_all(&first).unwrap();
        fs::create_dir_all(&second).unwrap();
        let raw = format!(
            "#include <...> search starts here:\n {}\n {}\n {}\nEnd of search list.\n",
            second.display(),
            first.display(),
            second.display()
        );

        let roots = parse_host_compiler_include_roots(&raw, Path::new("/fake/compiler")).unwrap();

        assert_eq!(
            roots,
            [second.display().to_string(), first.display().to_string(),]
        );
    }

    #[test]
    fn successful_host_probe_may_close_stdin_before_parent_finishes_writing() {
        let root = tempfile::tempdir().unwrap();
        let compiler = root.path().join("successful-probe");
        fs::write(
            &compiler,
            "#!/bin/sh\nexec 0<&-\nprintf 'probe complete\\n'\nexit 0\n",
        )
        .unwrap();
        let mut permissions = fs::metadata(&compiler).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&compiler, permissions).unwrap();
        let environment = BTreeMap::from([
            ("LANG".to_string(), "C".to_string()),
            ("LC_ALL".to_string(), "C".to_string()),
            ("TZ".to_string(), "UTC".to_string()),
        ]);

        let output =
            host_compiler_raw_output(&compiler, &[], &vec![b'x'; 1024 * 1024], &environment)
                .unwrap();

        assert!(output.status.success());
        assert_eq!(output.stdout, b"probe complete\n");
    }

    #[test]
    fn host_program_resolution_and_effective_path_support_spaces() {
        let root = tempfile::tempdir().unwrap();
        let tool_dir = root.path().join("tool chain");
        fs::create_dir_all(&tool_dir).unwrap();
        let helper = tool_dir.join("cc helper");
        fs::write(&helper, "fixture\n").unwrap();
        let path = std::env::join_paths([tool_dir.clone()])
            .unwrap()
            .into_string()
            .unwrap();
        let environment = BTreeMap::from([("PATH".to_string(), path)]);

        assert_eq!(
            resolve_host_program("cc helper", &environment).unwrap(),
            Some(helper)
        );

        let nvcc = tool_dir.join("nvcc");
        let ccbin = tool_dir.join("c++");
        let ar = tool_dir.join("ar");
        let effective = effective_environment_for_tool_paths([
            nvcc.to_str().unwrap(),
            ccbin.to_str().unwrap(),
            ar.to_str().unwrap(),
        ])
        .unwrap();
        assert!(
            std::env::split_paths(std::ffi::OsStr::new(&effective["PATH"]))
                .any(|entry| entry == tool_dir)
        );
    }

    #[test]
    fn toolchain_dependency_scope_rejects_cross_domain_aliases() {
        let absolute = "/toolchain/include/shared.h".to_string();
        let mut dependencies = BTreeMap::new();
        insert_toolchain_dependency(
            &mut dependencies,
            absolute.clone(),
            NativeOperatorObservedDependency {
                domain: NativeOperatorDependencyDomain::BackendToolchain,
                path: "include/shared.h".to_string(),
                sha256: "a".repeat(64),
            },
        )
        .unwrap();

        let error = insert_toolchain_dependency(
            &mut dependencies,
            absolute,
            NativeOperatorObservedDependency {
                domain: NativeOperatorDependencyDomain::HostToolchain,
                path: "/toolchain/include/shared.h".to_string(),
                sha256: "a".repeat(64),
            },
        )
        .unwrap_err();

        assert!(error
            .to_string()
            .contains("toolchain manifests ambiguously own dependency path"));
    }

    #[test]
    fn locks_self_contained_translation_unit_without_auxiliary_inputs() {
        let root = tempfile::tempdir().unwrap();
        let (source_root, _) = write_fixture(root.path());
        let mut definition = definition();
        definition.headers.clear();
        definition.dependency_closures[0].headers.clear();
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
            cuda_toolkit_root: PathBuf::from("/missing/cuda"),
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
            cuda_toolkit_root: PathBuf::from("/missing/cuda"),
            ccbin_path: PathBuf::from("/missing/c++"),
            ar_path: PathBuf::from("/missing/ar"),
            nvcc_threads: 256,
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
            .any(|pair| pair == ["--threads", "256"]));
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
            cuda_toolkit_root: PathBuf::from("/missing/cuda"),
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
    fn rejects_incomplete_or_undeclared_depfiles_before_cache_publish() {
        for (name, mode) in [
            ("missing-header", FakeDepfileMode::MissingDeclaredHeader),
            ("undeclared-external", FakeDepfileMode::UndeclaredExternal),
        ] {
            let root = tempfile::tempdir().unwrap();
            let (source_root, definition_path) = write_fixture(root.path());
            let plan_path = root.path().join("source-build.plan.json");
            lock_native_operator_source_definition(&definition_path, &source_root, &plan_path)
                .unwrap();
            let fake_cuda = write_fake_nvcc_with_mode(root.path(), mode);
            let output_dir = root.path().join(name);
            let object_cache_dir = root.path().join("object-cache");

            let error = run_native_operator_source_build(&NativeOperatorSourceBuildRequest {
                plan_path,
                source_root,
                output_dir: output_dir.clone(),
                compute_capability: "sm_89".to_string(),
                builder_sha: "7".repeat(40),
                nvcc_path: fake_cuda.nvcc,
                cuda_toolkit_root: fake_cuda.root,
                ccbin_path: fake_cuda.ccbin.clone(),
                ar_path: PathBuf::from("/usr/bin/ar"),
                nvcc_threads: 2,
                object_cache_dir: object_cache_dir.clone(),
                plan_only: false,
            })
            .unwrap_err();

            assert!(matches!(
                error,
                NativeOperatorBuilderError::SourceBuildRejected { .. }
            ));
            let receipt: NativeOperatorSourceBuildReceipt =
                read_json(&output_dir.join("source-build.receipt.json")).unwrap();
            assert!(receipt
                .failure_class
                .as_deref()
                .is_some_and(|failure| failure.starts_with("dependency_validation_failed:")));
            assert_eq!(
                receipt.commands[0].object_cache_status,
                Some(NativeOperatorSourceObjectCacheStatus::Rejected)
            );
            assert!(receipt.commands[0].object_cache_entry.is_none());
            assert!(
                fs::read_dir(object_cache_dir).unwrap().all(|entry| {
                    entry
                        .is_ok_and(|entry| entry.file_name() == ".host-toolchains")
                }),
                "invalid dependency evidence may cache toolchain inventory but must not publish an object"
            );
        }
    }

    #[test]
    fn rejects_tampered_cached_depfile_proof_before_compiler_start() {
        let root = tempfile::tempdir().unwrap();
        let (source_root, definition_path) = write_fixture(root.path());
        let plan_path = root.path().join("source-build.plan.json");
        lock_native_operator_source_definition(&definition_path, &source_root, &plan_path).unwrap();
        let fake_cuda = write_fake_nvcc(root.path());
        let object_cache_dir = root.path().join("object-cache");
        let request = |name: &str| NativeOperatorSourceBuildRequest {
            plan_path: plan_path.clone(),
            source_root: source_root.clone(),
            output_dir: root.path().join(name),
            compute_capability: "sm_89".to_string(),
            builder_sha: "7".repeat(40),
            nvcc_path: fake_cuda.nvcc.clone(),
            cuda_toolkit_root: fake_cuda.root.clone(),
            ccbin_path: fake_cuda.ccbin.clone(),
            ar_path: PathBuf::from("/usr/bin/ar"),
            nvcc_threads: 2,
            object_cache_dir: object_cache_dir.clone(),
            plan_only: false,
        };

        let cold = run_native_operator_source_build(&request("cold")).unwrap();
        let cache_entry = PathBuf::from(
            cold.commands[0]
                .object_cache_entry
                .as_deref()
                .expect("published object records its cache entry"),
        );
        let proof_dir = cache_entry.join("dependency-proof");
        fs::write(
            proof_dir.join("dependency.d"),
            "forged.o: kernels/marlin.cu\n",
        )
        .unwrap();

        let error = run_native_operator_source_build(&request("tampered")).unwrap_err();
        assert!(matches!(
            error,
            NativeOperatorBuilderError::SourceBuildRejected { .. }
        ));
        let receipt: NativeOperatorSourceBuildReceipt =
            read_json(&root.path().join("tampered/source-build.receipt.json")).unwrap();
        assert!(receipt
            .failure_class
            .as_deref()
            .is_some_and(|failure| failure.starts_with("cached_dependency_proof_failed:")));
        assert_eq!(
            fs::read_to_string(&fake_cuda.compile_counter)
                .unwrap()
                .lines()
                .count(),
            1,
            "tampered cache proof must reject before another compiler starts"
        );
    }

    #[test]
    fn rejects_tampered_cached_compiler_depfile_before_compiler_start() {
        let root = tempfile::tempdir().unwrap();
        let (source_root, definition_path) = write_fixture(root.path());
        let plan_path = root.path().join("source-build.plan.json");
        lock_native_operator_source_definition(&definition_path, &source_root, &plan_path).unwrap();
        let fake_cuda = write_fake_nvcc(root.path());
        let object_cache_dir = root.path().join("object-cache");
        let request = |name: &str| NativeOperatorSourceBuildRequest {
            plan_path: plan_path.clone(),
            source_root: source_root.clone(),
            output_dir: root.path().join(name),
            compute_capability: "sm_89".to_string(),
            builder_sha: "7".repeat(40),
            nvcc_path: fake_cuda.nvcc.clone(),
            cuda_toolkit_root: fake_cuda.root.clone(),
            ccbin_path: fake_cuda.ccbin.clone(),
            ar_path: PathBuf::from("/usr/bin/ar"),
            nvcc_threads: 2,
            object_cache_dir: object_cache_dir.clone(),
            plan_only: false,
        };

        let cold = run_native_operator_source_build(&request("cold")).unwrap();
        let cache_entry = PathBuf::from(
            cold.commands[0]
                .object_cache_entry
                .as_deref()
                .expect("published object records its cache entry"),
        );
        let proof_path = cache_entry.join("dependency-proof/proof.json");
        let compiler_depfile_path = cache_entry.join("dependency-proof/compiler-dependency.raw.d");
        let mut proof: NativeOperatorObjectDependencyProof = read_json(&proof_path).unwrap();
        let backend_binding = proof
            .depfile_bindings
            .iter_mut()
            .find(|binding| {
                binding.dependency.domain == NativeOperatorDependencyDomain::BackendToolchain
            })
            .expect("fixture depfile contains a backend toolchain dependency");
        let original_producer = backend_binding.producer_path.clone();
        let forged_producer =
            original_producer.replace("/bin/../include/cuda.h", "/forged/include/cuda.h");
        assert_ne!(forged_producer, original_producer);
        backend_binding.producer_path = forged_producer.clone();
        let compiler_raw = fs::read_to_string(&compiler_depfile_path)
            .unwrap()
            .replace(&original_producer, &forged_producer);
        proof.compiler_depfile_sha256 = sha256_bytes(compiler_raw.as_bytes());
        fs::write(&compiler_depfile_path, compiler_raw).unwrap();
        write_json(&proof_path, &proof).unwrap();

        let error = run_native_operator_source_build(&request("tampered")).unwrap_err();
        assert!(matches!(
            error,
            NativeOperatorBuilderError::SourceBuildRejected { .. }
        ));
        let receipt: NativeOperatorSourceBuildReceipt =
            read_json(&root.path().join("tampered/source-build.receipt.json")).unwrap();
        assert!(receipt
            .failure_class
            .as_deref()
            .is_some_and(|failure| failure.starts_with("cached_dependency_proof_failed:")));
        assert_eq!(
            fs::read_to_string(&fake_cuda.compile_counter)
                .unwrap()
                .lines()
                .count(),
            1,
            "tampered compiler depfile must reject before another compiler starts"
        );
    }

    #[test]
    fn rejects_partial_cached_dependency_proof_before_compiler_start() {
        let root = tempfile::tempdir().unwrap();
        let (source_root, definition_path) = write_fixture(root.path());
        let plan_path = root.path().join("source-build.plan.json");
        lock_native_operator_source_definition(&definition_path, &source_root, &plan_path).unwrap();
        let fake_cuda = write_fake_nvcc(root.path());
        let object_cache_dir = root.path().join("object-cache");
        let request = |name: &str| NativeOperatorSourceBuildRequest {
            plan_path: plan_path.clone(),
            source_root: source_root.clone(),
            output_dir: root.path().join(name),
            compute_capability: "sm_89".to_string(),
            builder_sha: "7".repeat(40),
            nvcc_path: fake_cuda.nvcc.clone(),
            cuda_toolkit_root: fake_cuda.root.clone(),
            ccbin_path: fake_cuda.ccbin.clone(),
            ar_path: PathBuf::from("/usr/bin/ar"),
            nvcc_threads: 2,
            object_cache_dir: object_cache_dir.clone(),
            plan_only: false,
        };

        let cold = run_native_operator_source_build(&request("cold")).unwrap();
        let cache_entry = PathBuf::from(
            cold.commands[0]
                .object_cache_entry
                .as_deref()
                .expect("published object records its cache entry"),
        );
        fs::remove_file(cache_entry.join("dependency-proof/proof.json")).unwrap();

        let error = run_native_operator_source_build(&request("partial")).unwrap_err();
        assert!(matches!(
            error,
            NativeOperatorBuilderError::SourceBuildRejected { .. }
        ));
        let receipt: NativeOperatorSourceBuildReceipt =
            read_json(&root.path().join("partial/source-build.receipt.json")).unwrap();
        assert!(receipt
            .failure_class
            .as_deref()
            .is_some_and(|failure| failure.starts_with("cached_dependency_proof_failed:")));
        assert_eq!(
            fs::read_to_string(&fake_cuda.compile_counter)
                .unwrap()
                .lines()
                .count(),
            1,
            "partial proof publication must reject before another compiler starts"
        );
    }

    #[test]
    fn rejects_tampered_cached_dependency_identity_before_compiler_start() {
        let root = tempfile::tempdir().unwrap();
        let (source_root, definition_path) = write_fixture(root.path());
        let plan_path = root.path().join("source-build.plan.json");
        lock_native_operator_source_definition(&definition_path, &source_root, &plan_path).unwrap();
        let fake_cuda = write_fake_nvcc(root.path());
        let object_cache_dir = root.path().join("object-cache");
        let request = |name: &str| NativeOperatorSourceBuildRequest {
            plan_path: plan_path.clone(),
            source_root: source_root.clone(),
            output_dir: root.path().join(name),
            compute_capability: "sm_89".to_string(),
            builder_sha: "7".repeat(40),
            nvcc_path: fake_cuda.nvcc.clone(),
            cuda_toolkit_root: fake_cuda.root.clone(),
            ccbin_path: fake_cuda.ccbin.clone(),
            ar_path: PathBuf::from("/usr/bin/ar"),
            nvcc_threads: 2,
            object_cache_dir: object_cache_dir.clone(),
            plan_only: false,
        };

        let cold = run_native_operator_source_build(&request("cold")).unwrap();
        let cache_entry = PathBuf::from(
            cold.commands[0]
                .object_cache_entry
                .as_deref()
                .expect("published object records its cache entry"),
        );
        let proof_path = cache_entry.join("dependency-proof/proof.json");
        let mut proof: NativeOperatorObjectDependencyProof = read_json(&proof_path).unwrap();
        let backend_dependency = proof
            .observed_dependencies
            .iter_mut()
            .find(|dependency| {
                dependency.domain == NativeOperatorDependencyDomain::BackendToolchain
            })
            .expect("fixture depfile contains a backend toolchain dependency");
        backend_dependency.sha256 = "b".repeat(64);
        proof.dependency_set_sha256 =
            observed_dependency_set_sha256(&proof.observed_dependencies).unwrap();
        write_json(&proof_path, &proof).unwrap();

        let error = run_native_operator_source_build(&request("tampered")).unwrap_err();
        assert!(matches!(
            error,
            NativeOperatorBuilderError::SourceBuildRejected { .. }
        ));
        let receipt: NativeOperatorSourceBuildReceipt =
            read_json(&root.path().join("tampered/source-build.receipt.json")).unwrap();
        assert!(receipt
            .failure_class
            .as_deref()
            .is_some_and(|failure| failure.starts_with("cached_dependency_proof_failed:")));
        assert_eq!(
            fs::read_to_string(&fake_cuda.compile_counter)
                .unwrap()
                .lines()
                .count(),
            1,
            "tampered typed dependency identity must reject before another compiler starts"
        );
    }

    #[test]
    fn compiler_inputs_invalidate_object_cache_without_hidden_probe_hits() {
        let root = tempfile::tempdir().unwrap();
        let (source_root, definition_path) = write_fixture(root.path());
        let plan_path = root.path().join("source-build.plan.json");
        lock_native_operator_source_definition(&definition_path, &source_root, &plan_path).unwrap();
        let fake_cuda = write_fake_nvcc(root.path());
        let object_cache_dir = root.path().join("object-cache");
        let run_build = |name: &str| {
            run_native_operator_source_build(&NativeOperatorSourceBuildRequest {
                plan_path: plan_path.clone(),
                source_root: source_root.clone(),
                output_dir: root.path().join(name),
                compute_capability: "sm_89".to_string(),
                builder_sha: "7".repeat(40),
                nvcc_path: fake_cuda.nvcc.clone(),
                cuda_toolkit_root: fake_cuda.root.clone(),
                ccbin_path: fake_cuda.ccbin.clone(),
                ar_path: PathBuf::from("/usr/bin/ar"),
                nvcc_threads: 2,
                object_cache_dir: object_cache_dir.clone(),
                plan_only: false,
            })
            .unwrap()
        };

        assert_eq!(run_build("cold").compiled_translation_units.len(), 1);
        assert_eq!(run_build("hit").cache_hit_translation_units.len(), 1);
        for (index, relative) in [
            "bin/ptxas",
            "include/cuda.h",
            "nvvm/libdevice/libdevice.10.bc",
        ]
        .iter()
        .enumerate()
        {
            let path = fake_cuda.root.join(relative);
            let mut contents = fs::read_to_string(&path).unwrap();
            contents.push_str(&format!("mutation-{index}\n"));
            fs::write(path, contents).unwrap();
            let receipt = run_build(&format!("mutation-{index}"));
            assert_eq!(receipt.compiled_translation_units, ["kernels/marlin.cu"]);
            assert!(receipt.cache_hit_translation_units.is_empty());
        }
        let host_root = fake_cuda
            .ccbin
            .parent()
            .and_then(Path::parent)
            .unwrap()
            .to_path_buf();
        for (index, relative) in ["include/stddef.h", "bin/cc1plus", "bin/driver.specs"]
            .iter()
            .enumerate()
        {
            let path = host_root.join(relative);
            let mut contents = fs::read_to_string(&path).unwrap();
            contents.push_str(&format!("host-mutation-{index}\n"));
            fs::write(path, contents).unwrap();
            let receipt = run_build(&format!("host-mutation-{index}"));
            assert_eq!(receipt.compiled_translation_units, ["kernels/marlin.cu"]);
            assert!(receipt.cache_hit_translation_units.is_empty());
        }
        fs::write(&fake_cuda.host_driver_config, "external driver option v2\n").unwrap();
        let external_config_receipt = run_build("external-driver-config-mutation");
        assert_eq!(
            external_config_receipt.compiled_translation_units,
            ["kernels/marlin.cu"]
        );
        assert!(external_config_receipt
            .cache_hit_translation_units
            .is_empty());
        assert_eq!(
            fs::read_to_string(&fake_cuda.compile_counter)
                .unwrap()
                .lines()
                .count(),
            8
        );
        assert_eq!(
            fs::read_to_string(&fake_cuda.invocation_counter)
                .unwrap()
                .lines()
                .count(),
            16,
            "each cache miss invokes one nvcc version probe and one compile; the full hit invokes zero"
        );
    }

    #[test]
    fn empty_host_include_root_is_locked_and_new_header_invalidates_object_cache() {
        let root = tempfile::tempdir().unwrap();
        let (source_root, definition_path) = write_fixture(root.path());
        let plan_path = root.path().join("source-build.plan.json");
        lock_native_operator_source_definition(&definition_path, &source_root, &plan_path).unwrap();
        let fake_cuda = write_fake_nvcc(root.path());
        let object_cache_dir = root.path().join("object-cache");
        let run_build = |name: &str| {
            run_native_operator_source_build(&NativeOperatorSourceBuildRequest {
                plan_path: plan_path.clone(),
                source_root: source_root.clone(),
                output_dir: root.path().join(name),
                compute_capability: "sm_89".to_string(),
                builder_sha: "7".repeat(40),
                nvcc_path: fake_cuda.nvcc.clone(),
                cuda_toolkit_root: fake_cuda.root.clone(),
                ccbin_path: fake_cuda.ccbin.clone(),
                ar_path: PathBuf::from("/usr/bin/ar"),
                nvcc_threads: 2,
                object_cache_dir: object_cache_dir.clone(),
                plan_only: false,
            })
            .unwrap()
        };

        assert_eq!(
            fs::read_dir(&fake_cuda.empty_host_include_root)
                .unwrap()
                .count(),
            0
        );
        let cold = run_build("cold-empty-root");
        assert_eq!(cold.compiled_translation_units, ["kernels/marlin.cu"]);
        let cold_manifest: NativeOperatorHostToolchainManifest = read_json(
            &root
                .path()
                .join("cold-empty-root/toolchain/host-static-manifest.json"),
        )
        .unwrap();
        let empty_root = fake_cuda.empty_host_include_root.display().to_string();
        assert!(cold_manifest.include_roots.contains(&empty_root));
        assert!(!cold_manifest
            .files
            .iter()
            .any(|file| Path::new(&file.logical_path).starts_with(&empty_root)));

        let hit = run_build("unchanged-empty-root");
        assert!(hit.compiled_translation_units.is_empty());
        assert_eq!(hit.cache_hit_translation_units, ["kernels/marlin.cu"]);
        assert!(!hit.commands[0].compiler_executed);

        let late_header = fake_cuda.empty_host_include_root.join("late-header.h");
        fs::write(&late_header, "#define LATE_HEADER 1\n").unwrap();
        let changed = run_build("populated-root");
        assert_eq!(changed.compiled_translation_units, ["kernels/marlin.cu"]);
        assert!(changed.cache_hit_translation_units.is_empty());
        let changed_manifest: NativeOperatorHostToolchainManifest = read_json(
            &root
                .path()
                .join("populated-root/toolchain/host-static-manifest.json"),
        )
        .unwrap();
        assert!(changed_manifest.files.iter().any(|file| {
            file.logical_path == late_header.display().to_string()
                && file.resolved_path == late_header.canonicalize().unwrap().display().to_string()
        }));
        assert_eq!(
            fs::read_to_string(&fake_cuda.compile_counter)
                .unwrap()
                .lines()
                .count(),
            2
        );
        assert_eq!(
            fs::read_to_string(&fake_cuda.invocation_counter)
                .unwrap()
                .lines()
                .count(),
            4,
            "the unchanged empty root is an nvcc-free hit; adding a header forces one probe and compile"
        );
    }

    #[test]
    fn cuda_toolkit_manifest_accepts_internal_symlink_directories_and_rejects_escapes() {
        let root = tempfile::tempdir().unwrap();
        let fake_cuda = write_fake_nvcc(root.path());
        let internal_target = fake_cuda.root.join("targets/headers");
        fs::create_dir_all(&internal_target).unwrap();
        fs::write(internal_target.join("linked.h"), "#define LINKED 1\n").unwrap();
        let internal_link = fake_cuda.root.join("include/linked");
        symlink(&internal_target, &internal_link).unwrap();

        let manifest = build_cuda_toolkit_manifest(&fake_cuda.root).unwrap();
        assert!(manifest.entries.iter().any(|entry| {
            entry.logical_path == "include/linked/linked.h"
                && entry.resolved_path == "targets/headers/linked.h"
        }));

        fs::remove_file(&internal_link).unwrap();
        symlink("/etc", &internal_link).unwrap();
        let error = build_cuda_toolkit_manifest(&fake_cuda.root).unwrap_err();
        assert!(error.to_string().contains("escapes its canonical root"));
    }

    #[test]
    fn bounded_fixture_build_writes_pass_receipt_and_archive_hash() {
        let root = tempfile::tempdir().unwrap();
        let (source_root, definition_path) = write_fixture(root.path());
        let plan_path = root.path().join("source-build.plan.json");
        lock_native_operator_source_definition(&definition_path, &source_root, &plan_path).unwrap();
        let output_dir = root.path().join("build");
        let fake_cuda = write_fake_nvcc(root.path());
        let object_cache_dir = root.path().join("object-cache");

        let receipt = run_native_operator_source_build(&NativeOperatorSourceBuildRequest {
            plan_path: plan_path.clone(),
            source_root: source_root.clone(),
            output_dir: output_dir.clone(),
            compute_capability: "sm_89".to_string(),
            builder_sha: "7".repeat(40),
            nvcc_path: fake_cuda.nvcc.clone(),
            cuda_toolkit_root: fake_cuda.root.clone(),
            ccbin_path: fake_cuda.ccbin.clone(),
            ar_path: PathBuf::from("/usr/bin/ar"),
            nvcc_threads: 2,
            object_cache_dir: object_cache_dir.clone(),
            plan_only: false,
        })
        .unwrap();

        assert_eq!(receipt.status, NativeOperatorSourceBuildStatus::Pass);
        let static_toolchain = &receipt
            .toolchain
            .as_ref()
            .expect("completed build records toolchain identity")
            .static_identity;
        assert_eq!(static_toolchain.backend, NativeOperatorBackend::Cuda);
        assert_eq!(
            static_toolchain.compiler_driver,
            NativeOperatorSourceCompilerDriver::CudaNvcc
        );
        assert!(is_sha256_digest(receipt.archive_sha256.as_deref().unwrap()));
        assert!(output_dir.join("libmarlin.a").is_file());
        assert!(output_dir.join("source-build.receipt.json").is_file());
        assert_eq!(receipt.compiled_translation_units, ["kernels/marlin.cu"]);
        assert!(receipt.cache_hit_translation_units.is_empty());
        assert_eq!(
            receipt.commands[0]
                .observed_dependencies
                .iter()
                .map(|dependency| dependency.domain)
                .collect::<Vec<_>>(),
            [
                NativeOperatorDependencyDomain::Source,
                NativeOperatorDependencyDomain::Source,
                NativeOperatorDependencyDomain::BackendToolchain,
                NativeOperatorDependencyDomain::HostToolchain,
            ]
        );
        assert!(receipt.commands[0]
            .observed_dependencies
            .iter()
            .all(|dependency| is_sha256_digest(&dependency.sha256)));
        let compiler_depfile = fs::read_to_string(
            output_dir.join(
                receipt.commands[0]
                    .compiler_depfile
                    .as_deref()
                    .expect("cold build records its compiler depfile"),
            ),
        )
        .unwrap();
        let portable_depfile = fs::read_to_string(
            output_dir.join(
                receipt.commands[0]
                    .depfile
                    .as_deref()
                    .expect("cold build records its portable depfile"),
            ),
        )
        .unwrap();
        assert!(compiler_depfile.contains("/bin/../include/cuda.h"));
        assert!(!portable_depfile.contains("/../"));
        let plan: NativeOperatorSourceBuildPlan = read_json(&plan_path).unwrap();
        let toolchain_scope =
            load_toolchain_dependency_scope(&output_dir, static_toolchain).unwrap();
        let cache_entry = PathBuf::from(
            receipt.commands[0]
                .object_cache_entry
                .as_deref()
                .expect("cold build records its cache entry"),
        );
        let object_name = Path::new(
            receipt.commands[0]
                .object_file
                .as_deref()
                .expect("cold build records its object"),
        )
        .file_name()
        .unwrap()
        .to_str()
        .unwrap();
        validate_existing_dependency_proof(
            &cache_entry.join("dependency-proof"),
            receipt.commands[0].object_cache_key.as_deref().unwrap(),
            receipt.commands[0].object_sha256.as_deref().unwrap(),
            &plan.translation_units[0],
            &plan.dependency_closures[0],
            &format!("/another-worktree/objects/{object_name}"),
            &toolchain_scope,
        )
        .expect("a concurrent cache winner from another worktree remains valid");
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
        let host_probe_count = fs::read_to_string(&fake_cuda.host_compiler_invocation_counter)
            .unwrap()
            .lines()
            .count();

        let cached_output_dir = root.path().join("cached-build");
        let cached_receipt = run_native_operator_source_build(&NativeOperatorSourceBuildRequest {
            plan_path: plan_path.clone(),
            source_root: source_root.clone(),
            output_dir: cached_output_dir.clone(),
            compute_capability: "sm_89".to_string(),
            builder_sha: "8".repeat(40),
            nvcc_path: fake_cuda.nvcc.clone(),
            cuda_toolkit_root: fake_cuda.root.clone(),
            ccbin_path: fake_cuda.ccbin.clone(),
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
            cached_receipt.commands[0].dependency_validation,
            Some(NativeOperatorDependencyValidation::CacheProof)
        );
        assert_eq!(
            cached_receipt.commands[0].observed_dependencies,
            receipt.commands[0].observed_dependencies
        );
        assert_ne!(
            cached_receipt.commands[0].object_file,
            cached_receipt.commands[0].depfile_producer_object_file,
            "portable cache proof must preserve the producer object while restoring to a new output root"
        );
        assert!(cached_receipt.commands[0]
            .depfile
            .as_deref()
            .is_some_and(|relative| cached_output_dir.join(relative).is_file()));
        assert_eq!(
            fs::read_to_string(&fake_cuda.compile_counter)
                .unwrap()
                .lines()
                .count(),
            1
        );
        assert_eq!(
            fs::read_to_string(&fake_cuda.invocation_counter)
                .unwrap()
                .lines()
                .count(),
            2,
            "cold build invokes one miss-only version probe plus one compile; full cache hit invokes neither"
        );
        assert_eq!(
            fs::read_to_string(&fake_cuda.host_compiler_invocation_counter)
                .unwrap()
                .lines()
                .count(),
            host_probe_count + 2,
            "full cache hit runs only the bounded include/driver configuration probes"
        );
        assert_eq!(
            receipt.archive_sha256, cached_receipt.archive_sha256,
            "worker-count changes must not change the object or archive"
        );
        assert_eq!(
            receipt.inputs_sha256, cached_receipt.inputs_sha256,
            "worker-count and provenance commit changes are not output-content inputs"
        );

        fs::write(source_root.join("LICENSE"), "fixture license\n").unwrap();
        let package_spec = crate::NativeOperatorPackageSpec {
            schema_version: crate::NATIVE_OPERATOR_PACKAGE_SPEC_SCHEMA_VERSION,
            operator: CudaNativeBuildUnit::Marlin.artifact_operator().to_string(),
            operator_abi_version: "1".to_string(),
            backend: ferrum_types::NativeOperatorBackend::Cuda,
            compute_capabilities: vec!["sm_89".to_string()],
            operation_bindings: vec![ferrum_types::NativeOperatorBinding {
                operation_id: "operation.dense_linear".to_string(),
                operation_contract_version: ferrum_types::NativeOperatorContractVersion::new(1, 0),
                provider_id: "provider.cuda.dense_linear.f16.marlin".to_string(),
                provider_version: ferrum_types::NativeOperatorContractVersion::new(1, 0),
                provider_implementation_fingerprint: "a".repeat(64),
                entrypoints: CudaNativeBuildUnit::Marlin
                    .required_exports()
                    .iter()
                    .map(|value| (*value).to_string())
                    .collect(),
            }],
            required_exports: CudaNativeBuildUnit::Marlin
                .required_exports()
                .iter()
                .map(|value| (*value).to_string())
                .collect(),
            license_files: vec![crate::NativeOperatorLicenseInput {
                source_path: "LICENSE".to_string(),
                output_path: "licenses/LICENSE".to_string(),
            }],
            cuda_toolkit: Some("12.4".to_string()),
            cuda_runtime_min: Some("12.4".to_string()),
            system_libraries: vec![
                ferrum_native_ops::NativeOperatorSystemLibrary::CudaRuntime,
                ferrum_native_ops::NativeOperatorSystemLibrary::StdCxx,
            ],
        };
        let package_spec_path = root.path().join("package-spec.json");
        write_json(&package_spec_path, &package_spec).unwrap();
        let catalog_path = root.path().join("operation-catalog.json");
        let abi_path = root.path().join("native-abi.json");
        write_json(
            &catalog_path,
            &ferrum_types::NativeOperatorProviderCatalog {
                schema_version: ferrum_types::NATIVE_OPERATOR_PROVIDER_CATALOG_SCHEMA_VERSION,
                backend: ferrum_types::NativeOperatorBackend::Cuda,
                providers: vec![ferrum_types::NativeOperatorProviderCatalogRow {
                    operation_id: "operation.dense_linear".to_string(),
                    operation_contract_version: ferrum_types::NativeOperatorContractVersion::new(
                        1, 0,
                    ),
                    operation_fingerprint: "b".repeat(64),
                    provider_id: "provider.cuda.dense_linear.f16.marlin".to_string(),
                    provider_version: ferrum_types::NativeOperatorContractVersion::new(1, 0),
                    provider_implementation_fingerprint: "a".repeat(64),
                }],
            },
        )
        .unwrap();
        write_json(
            &abi_path,
            &ferrum_types::NativeOperatorAbiContract {
                schema_version: ferrum_types::NATIVE_OPERATOR_ABI_CONTRACT_SCHEMA_VERSION,
                ferrum_native_abi_version: ferrum_types::FERRUM_NATIVE_OPERATOR_ABI_VERSION
                    .to_string(),
                descriptor_struct: "FerrumNativeOperatorDescriptorV2".to_string(),
                descriptor_symbol_policy: "operator_namespaced".to_string(),
                descriptor_fields: [
                    ("struct_size", "uint32_t"),
                    ("ferrum_native_abi_version", "uint32_t"),
                    ("operator_name", "const char *"),
                    ("operator_abi_version", "const char *"),
                    ("g03_catalog_sha256", "const char *"),
                    ("abi_contract_sha256", "const char *"),
                ]
                .into_iter()
                .map(|(name, c_type)| ferrum_types::NativeOperatorAbiField {
                    name: name.to_string(),
                    c_type: c_type.to_string(),
                })
                .collect(),
            },
        )
        .unwrap();
        let package_output = root.path().join("package");

        let package_receipt =
            crate::package_native_operator(&crate::NativeOperatorPackageRequest {
                spec_path: package_spec_path,
                source_root: source_root.clone(),
                license_root: source_root.clone(),
                source_build_receipt_path: output_dir.join("source-build.receipt.json"),
                source_build_plan_path: plan_path.clone(),
                g03_catalog_path: catalog_path,
                abi_contract_path: abi_path,
                output_dir: package_output.clone(),
                cc: PathBuf::from("/usr/bin/cc"),
                ar: PathBuf::from("/usr/bin/ar"),
            })
            .unwrap();

        assert_eq!(
            package_receipt.source_build_plan.sha256,
            receipt.plan_sha256
        );
        assert_eq!(
            package_receipt.source_archive_sha256,
            receipt.archive_sha256.unwrap()
        );
        assert!(package_output.join("package.receipt.json").is_file());

        let cached_package_spec_path = root.path().join("cached-package-spec.json");
        write_json(&cached_package_spec_path, &package_spec).unwrap();
        let cached_package_output = root.path().join("cached-package");
        crate::package_native_operator(&crate::NativeOperatorPackageRequest {
            spec_path: cached_package_spec_path,
            source_root: source_root.clone(),
            license_root: source_root,
            source_build_receipt_path: cached_output_dir.join("source-build.receipt.json"),
            source_build_plan_path: plan_path,
            g03_catalog_path: root.path().join("operation-catalog.json"),
            abi_contract_path: root.path().join("native-abi.json"),
            output_dir: cached_package_output.clone(),
            cc: PathBuf::from("/usr/bin/cc"),
            ar: PathBuf::from("/usr/bin/ar"),
        })
        .unwrap();
        let cached_manifest: ferrum_types::NativeOperatorManifest =
            read_json(&cached_package_output.join("native_operator_manifest.json")).unwrap();
        assert_eq!(
            cached_manifest.build_summary.nvcc_version.as_deref(),
            Some("cuda-toolkit-static 12.4.0")
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
        source_definition.dependency_closures = vec![
            NativeOperatorTranslationUnitDependencies {
                translation_unit: "kernels/marlin.cu".to_string(),
                headers: vec!["kernels/marlin.h".to_string()],
            },
            NativeOperatorTranslationUnitDependencies {
                translation_unit: "kernels/other.cu".to_string(),
                headers: Vec::new(),
            },
        ];
        let definition_path = root.path().join("two-tu-definition.json");
        write_json(&definition_path, &source_definition).unwrap();
        let first_plan_path = root.path().join("first.plan.json");
        lock_native_operator_source_definition(&definition_path, &source_root, &first_plan_path)
            .unwrap();
        let fake_cuda = write_fake_nvcc(root.path());
        let object_cache_dir = root.path().join("object-cache");

        let first_receipt = run_native_operator_source_build(&NativeOperatorSourceBuildRequest {
            plan_path: first_plan_path,
            source_root: source_root.clone(),
            output_dir: root.path().join("first-build"),
            compute_capability: "sm_89".to_string(),
            builder_sha: "7".repeat(40),
            nvcc_path: fake_cuda.nvcc.clone(),
            cuda_toolkit_root: fake_cuda.root.clone(),
            ccbin_path: fake_cuda.ccbin.clone(),
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
            source_root: source_root.clone(),
            output_dir: root.path().join("second-build"),
            compute_capability: "sm_89".to_string(),
            builder_sha: "8".repeat(40),
            nvcc_path: fake_cuda.nvcc.clone(),
            cuda_toolkit_root: fake_cuda.root.clone(),
            ccbin_path: fake_cuda.ccbin.clone(),
            ar_path: PathBuf::from("/usr/bin/ar"),
            nvcc_threads: 4,
            object_cache_dir: object_cache_dir.clone(),
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
            fs::read_to_string(&fake_cuda.compile_counter)
                .unwrap()
                .lines()
                .count(),
            3
        );

        fs::write(source_root.join("kernels/marlin.h"), "#define MARLIN 2\n").unwrap();
        let third_plan_path = root.path().join("third.plan.json");
        lock_native_operator_source_definition(&definition_path, &source_root, &third_plan_path)
            .unwrap();
        let third_receipt = run_native_operator_source_build(&NativeOperatorSourceBuildRequest {
            plan_path: third_plan_path,
            source_root,
            output_dir: root.path().join("third-build"),
            compute_capability: "sm_89".to_string(),
            builder_sha: "9".repeat(40),
            nvcc_path: fake_cuda.nvcc,
            cuda_toolkit_root: fake_cuda.root,
            ccbin_path: fake_cuda.ccbin.clone(),
            ar_path: PathBuf::from("/usr/bin/ar"),
            nvcc_threads: 4,
            object_cache_dir,
            plan_only: false,
        })
        .unwrap();
        assert_eq!(
            third_receipt.compiled_translation_units,
            ["kernels/marlin.cu"]
        );
        assert_eq!(
            third_receipt.cache_hit_translation_units,
            ["kernels/other.cu"]
        );
        assert_eq!(
            fs::read_to_string(fake_cuda.compile_counter)
                .unwrap()
                .lines()
                .count(),
            4,
            "private header drift must recompile only its owning translation unit"
        );
    }
}
