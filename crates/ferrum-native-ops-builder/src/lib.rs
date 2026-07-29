//! Isolated packaging and set assembly for source-built native operators.

pub mod source_build;

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs;
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use ferrum_native_ops::{
    CudaNativeBuildUnit, NativeOperatorArtifactLock, NativeOperatorArtifactSetLock,
    NativeOperatorEvidenceFile, NativeOperatorResolveRequest, NativeOperatorResolver,
    NativeOperatorSystemLibrary, NATIVE_OPERATOR_ARTIFACT_SET_SCHEMA_VERSION,
};
use ferrum_types::{
    is_sha256_digest, NativeOperatorBackend, NativeOperatorBinding, NativeOperatorBuildSummary,
    NativeOperatorLinkage, NativeOperatorManifest, FERRUM_NATIVE_OPERATOR_ABI_VERSION,
    NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tempfile::{Builder as TempBuilder, NamedTempFile};
use thiserror::Error;

pub const NATIVE_OPERATOR_PACKAGE_SPEC_SCHEMA_VERSION: u32 = 2;
pub const NATIVE_OPERATOR_PACKAGE_RECEIPT_SCHEMA_VERSION: u32 = 2;

pub use source_build::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeOperatorPackageSpec {
    pub schema_version: u32,
    pub operator: String,
    pub operator_abi_version: String,
    pub backend: NativeOperatorBackend,
    pub compute_capabilities: Vec<String>,
    pub operation_bindings: Vec<NativeOperatorBinding>,
    pub required_exports: Vec<String>,
    pub license_files: Vec<NativeOperatorLicenseInput>,
    pub cuda_toolkit: Option<String>,
    pub cuda_runtime_min: Option<String>,
    pub system_libraries: Vec<NativeOperatorSystemLibrary>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorLicenseInput {
    pub source_path: String,
    pub output_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeOperatorPackageReceipt {
    pub schema_version: u32,
    pub operator: String,
    pub source_build_receipt: NativeOperatorEvidenceFile,
    pub source_build_plan: NativeOperatorEvidenceFile,
    pub source_build_logs: Vec<NativeOperatorEvidenceFile>,
    pub source_archive_sha256: String,
    pub source_archive_members: Vec<NativeOperatorArchiveMemberEvidence>,
    pub source_archive_verification: NativeOperatorEvidenceFile,
    pub manifest_file: String,
    pub artifact_file: String,
    pub manifest_sha256: String,
    pub binary_sha256: String,
    pub g03_catalog_sha256: String,
    pub abi_contract_sha256: String,
    pub descriptor_export: String,
    pub system_libraries: Vec<NativeOperatorSystemLibrary>,
    pub package_toolchain: NativeOperatorPackageToolchain,
    pub package_environment: BTreeMap<String, String>,
    pub package_commands: Vec<NativeOperatorPackageCommand>,
    pub package_build_logs: Vec<NativeOperatorEvidenceFile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorPackageToolchain {
    pub descriptor_compiler: NativeOperatorToolIdentity,
    pub archiver: NativeOperatorToolIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorPackageCommand {
    pub argv: Vec<String>,
    pub working_directory: String,
    pub stdout_log: String,
    pub stderr_log: String,
    pub return_code: i32,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorArchiveMemberEvidence {
    pub member: String,
    pub sha256: String,
}

#[derive(Debug, Clone)]
pub struct NativeOperatorPackageRequest {
    pub spec_path: PathBuf,
    pub source_root: PathBuf,
    pub source_build_receipt_path: PathBuf,
    pub source_build_plan_path: PathBuf,
    pub g03_catalog_path: PathBuf,
    pub abi_contract_path: PathBuf,
    pub output_dir: PathBuf,
    pub cc: PathBuf,
    pub ar: PathBuf,
}

#[derive(Debug, Clone)]
pub struct NativeOperatorSetRequest {
    pub receipt_paths: Vec<PathBuf>,
    pub output_lock_path: PathBuf,
    pub compute_capability: String,
}

#[derive(Debug, Error)]
pub enum NativeOperatorBuilderError {
    #[error("invalid native operator package request: {0}")]
    Invalid(String),
    #[error("path does not exist or is not a file: {0}")]
    MissingFile(PathBuf),
    #[error("output already exists: {0}")]
    OutputExists(PathBuf),
    #[error("failed to access {path}: {source}")]
    Io { path: PathBuf, source: io::Error },
    #[error("failed to parse JSON {path}: {source}")]
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("native operator tool failed: tool={tool} status={status} stderr={stderr}")]
    Tool {
        tool: String,
        status: String,
        stderr: String,
    },
    #[error("native operator artifact validation failed: {0}")]
    Resolve(#[from] ferrum_native_ops::NativeOperatorResolveError),
    #[error("native operator artifact-set validation failed: {0}")]
    ArtifactSet(#[from] ferrum_native_ops::NativeOperatorArtifactSetError),
    #[error("native build artifact cache failed: {0}")]
    BuildCache(#[from] ferrum_native_ops::NativeBuildArtifactCacheError),
    #[error("native operator source build rejected: receipt={receipt_path} reason={reason}")]
    SourceBuildRejected {
        receipt_path: PathBuf,
        reason: String,
    },
}

pub type Result<T> = std::result::Result<T, NativeOperatorBuilderError>;

pub fn package_native_operator(
    request: &NativeOperatorPackageRequest,
) -> Result<NativeOperatorPackageReceipt> {
    require_file(&request.spec_path)?;
    require_file(&request.source_build_receipt_path)?;
    require_file(&request.source_build_plan_path)?;
    require_file(&request.g03_catalog_path)?;
    require_file(&request.abi_contract_path)?;
    if !request.cc.is_absolute() || !request.ar.is_absolute() {
        return Err(NativeOperatorBuilderError::Invalid(
            "package compiler and archiver paths must be absolute".to_string(),
        ));
    }
    if request.output_dir.exists() {
        return Err(NativeOperatorBuilderError::OutputExists(
            request.output_dir.clone(),
        ));
    }
    let spec: NativeOperatorPackageSpec = read_json(&request.spec_path)?;
    validate_package_spec(&spec)?;
    let source_root =
        request
            .source_root
            .canonicalize()
            .map_err(|source| NativeOperatorBuilderError::Io {
                path: request.source_root.clone(),
                source,
            })?;
    if !source_root.is_dir() {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "source_root is not a directory: {}",
            source_root.display()
        )));
    }
    let source_build = load_source_build_for_package(
        &request.source_build_receipt_path,
        &request.source_build_plan_path,
        &spec,
        &source_root,
    )?;
    let package_toolchain = NativeOperatorPackageToolchain {
        descriptor_compiler: tool_identity(&request.cc)?,
        archiver: tool_identity(&request.ar)?,
    };
    let package_environment = package_build_environment(&package_toolchain)?;

    let g03_catalog_sha256 = sha256_file(&request.g03_catalog_path)?;
    let abi_contract_sha256 = sha256_file(&request.abi_contract_path)?;
    let identity_suffix = &sha256_bytes(spec.operator.as_bytes())[..12];
    let symbol_slug = symbol_slug(&spec.operator)?;
    let descriptor_export = format!("ferrum_native_{symbol_slug}_{identity_suffix}_descriptor_v2");
    let artifact_file = format!("libferrum_native_{symbol_slug}_{identity_suffix}.a");
    let manifest_file = "native_operator_manifest.json".to_string();
    let receipt_file = "package.receipt.json";

    let output_parent = request
        .output_dir
        .parent()
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(output_parent).map_err(|source| NativeOperatorBuilderError::Io {
        path: output_parent.to_path_buf(),
        source,
    })?;
    let staging = TempBuilder::new()
        .prefix(".ferrum-native-package-")
        .tempdir_in(output_parent)
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: output_parent.to_path_buf(),
            source,
        })?;
    let source_build_receipt = copy_evidence_file(
        &request.source_build_receipt_path,
        staging.path(),
        "provenance/source-build.receipt.json",
    )?;
    if source_build_receipt.sha256 != source_build.receipt_sha256 {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} source-build receipt changed while packaging",
            spec.operator
        )));
    }
    let source_build_plan = copy_evidence_file(
        &source_build.plan_path,
        staging.path(),
        "provenance/source-build.plan.json",
    )?;
    if source_build_plan.sha256 != source_build.receipt.plan_sha256 {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} source-build plan changed while packaging",
            spec.operator
        )));
    }
    let source_build_logs = copy_source_build_logs(
        &request.source_build_receipt_path,
        &source_build.receipt,
        staging.path(),
    )?;
    let artifact_path = staging.path().join(&artifact_file);
    fs::copy(&source_build.archive_path, &artifact_path).map_err(|source| {
        NativeOperatorBuilderError::Io {
            path: artifact_path.clone(),
            source,
        }
    })?;
    let copied_source_archive_sha256 = sha256_file(&artifact_path)?;
    if copied_source_archive_sha256 != source_build.source_archive_sha256 {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} source archive changed while packaging: expected={} actual={copied_source_archive_sha256}",
            spec.operator, source_build.source_archive_sha256
        )));
    }
    let (source_archive_members, source_archive_verification) = verify_source_archive_members(
        &artifact_path,
        &source_build.receipt,
        &package_toolchain.archiver.path,
        staging.path(),
        &package_environment,
    )?;

    let descriptor_source = staging.path().join("descriptor.c");
    let descriptor_object = staging.path().join("descriptor.o");
    fs::write(
        &descriptor_source,
        render_descriptor_source(
            &descriptor_export,
            &spec.operator,
            &spec.operator_abi_version,
            &g03_catalog_sha256,
            &abi_contract_sha256,
        ),
    )
    .map_err(|source| NativeOperatorBuilderError::Io {
        path: descriptor_source.clone(),
        source,
    })?;
    let descriptor_compile = run_package_command(
        &package_toolchain.descriptor_compiler.path,
        vec![
            "-std=c11".to_string(),
            "-O2".to_string(),
            "-fno-ident".to_string(),
            "-fvisibility=hidden".to_string(),
            "-c".to_string(),
            "descriptor.c".to_string(),
            "-o".to_string(),
            "descriptor.o".to_string(),
        ],
        staging.path(),
        "build-logs/descriptor-compile.stdout.log",
        "build-logs/descriptor-compile.stderr.log",
        &package_environment,
    )?;
    let descriptor_archive = run_package_command(
        &package_toolchain.archiver.path,
        vec![
            "rcs".to_string(),
            artifact_file.clone(),
            "descriptor.o".to_string(),
        ],
        staging.path(),
        "build-logs/descriptor-archive.stdout.log",
        "build-logs/descriptor-archive.stderr.log",
        &package_environment,
    )?;
    let package_commands = vec![descriptor_compile, descriptor_archive];
    let mut package_build_logs = vec![source_archive_verification.clone()];
    package_build_logs.extend(
        package_commands
            .iter()
            .flat_map(|command| [&command.stdout_log, &command.stderr_log])
            .map(|relative| evidence_file_at(staging.path(), relative))
            .collect::<Result<Vec<_>>>()?,
    );
    package_build_logs.sort_by(|left, right| left.path.cmp(&right.path));

    let mut license_files = Vec::with_capacity(spec.license_files.len());
    for license in &spec.license_files {
        let source = resolve_relative_file(&source_root, &license.source_path)?;
        let destination = staging.path().join(&license.output_path);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).map_err(|source| NativeOperatorBuilderError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        fs::copy(&source, &destination).map_err(|source| NativeOperatorBuilderError::Io {
            path: destination,
            source,
        })?;
        license_files.push(license.output_path.clone());
    }

    let mut exports = spec.required_exports.clone();
    exports.push(descriptor_export.clone());
    exports.sort();
    let binary_sha256 = sha256_file(&artifact_path)?;
    let manifest = NativeOperatorManifest {
        schema_version: NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION,
        operator: spec.operator.clone(),
        operator_abi_version: spec.operator_abi_version.clone(),
        ferrum_native_abi_version: FERRUM_NATIVE_OPERATOR_ABI_VERSION.to_string(),
        backend: spec.backend,
        cuda_toolkit: spec.cuda_toolkit.clone(),
        cuda_runtime_min: spec.cuda_runtime_min.clone(),
        compute_capabilities: spec.compute_capabilities.clone(),
        source_package: source_build.receipt.source_package.clone(),
        inputs_sha256: source_build.receipt.inputs_sha256.clone(),
        binary_sha256: binary_sha256.clone(),
        linkage: NativeOperatorLinkage::Static,
        g03_catalog_sha256: Some(g03_catalog_sha256.clone()),
        abi_contract_sha256: Some(abi_contract_sha256.clone()),
        descriptor_export: Some(descriptor_export.clone()),
        operation_bindings: spec.operation_bindings.clone(),
        exports: exports.clone(),
        license_files,
        build_summary: source_build.build_summary,
    };
    manifest
        .validate()
        .map_err(NativeOperatorBuilderError::Invalid)?;
    let manifest_path = staging.path().join(&manifest_file);
    write_json(&manifest_path, &manifest)?;
    NativeOperatorResolver.resolve(
        &NativeOperatorResolveRequest::new(
            spec.operator.clone(),
            spec.backend,
            &manifest_path,
            &artifact_path,
        )
        .with_compute_capability(source_build.receipt.compute_capability.clone())
        .with_operator_abi_version(spec.operator_abi_version.clone())
        .with_ferrum_native_abi_version(FERRUM_NATIVE_OPERATOR_ABI_VERSION)
        .with_g03_catalog_sha256(g03_catalog_sha256.clone())
        .with_abi_contract_sha256(abi_contract_sha256.clone())
        .with_descriptor_export(descriptor_export.clone())
        .with_required_exports(exports)
        .with_operation_bindings(spec.operation_bindings.clone()),
    )?;

    let manifest_sha256 = sha256_file(&manifest_path)?;
    let receipt = NativeOperatorPackageReceipt {
        schema_version: NATIVE_OPERATOR_PACKAGE_RECEIPT_SCHEMA_VERSION,
        operator: spec.operator,
        source_build_receipt,
        source_build_plan,
        source_build_logs,
        source_archive_sha256: source_build.source_archive_sha256,
        source_archive_members,
        source_archive_verification,
        manifest_file,
        artifact_file,
        manifest_sha256,
        binary_sha256,
        g03_catalog_sha256,
        abi_contract_sha256,
        descriptor_export,
        system_libraries: spec.system_libraries,
        package_toolchain,
        package_environment,
        package_commands,
        package_build_logs,
    };
    write_json(&staging.path().join(receipt_file), &receipt)?;
    fs::remove_file(&descriptor_source).map_err(|source| NativeOperatorBuilderError::Io {
        path: descriptor_source,
        source,
    })?;
    fs::remove_file(&descriptor_object).map_err(|source| NativeOperatorBuilderError::Io {
        path: descriptor_object,
        source,
    })?;

    let staging_path = staging.keep();
    fs::rename(&staging_path, &request.output_dir).map_err(|source| {
        NativeOperatorBuilderError::Io {
            path: request.output_dir.clone(),
            source,
        }
    })?;
    Ok(receipt)
}

struct ValidatedSourceBuildForPackage {
    receipt: NativeOperatorSourceBuildReceipt,
    receipt_sha256: String,
    plan_path: PathBuf,
    archive_path: PathBuf,
    source_archive_sha256: String,
    build_summary: NativeOperatorBuildSummary,
}

fn load_source_build_for_package(
    receipt_path: &Path,
    plan_path: &Path,
    spec: &NativeOperatorPackageSpec,
    source_root: &Path,
) -> Result<ValidatedSourceBuildForPackage> {
    let bytes = fs::read(receipt_path).map_err(|source| NativeOperatorBuilderError::Io {
        path: receipt_path.to_path_buf(),
        source,
    })?;
    let receipt_sha256 = sha256_bytes(&bytes);
    let receipt: NativeOperatorSourceBuildReceipt =
        serde_json::from_slice(&bytes).map_err(|source| NativeOperatorBuilderError::Json {
            path: receipt_path.to_path_buf(),
            source,
        })?;
    validate_source_build_for_package(&receipt, spec)?;
    verify_source_build_receipt_against_plan(&receipt, plan_path, source_root)?;

    let archive_file = receipt
        .archive_file
        .as_deref()
        .expect("validated PASS receipt has archive_file");
    let receipt_root = receipt_path.parent().unwrap_or_else(|| Path::new("."));
    let archive_path = resolve_relative_file(receipt_root, archive_file)?;
    let source_archive_sha256 = sha256_file(&archive_path)?;
    let expected_archive_sha256 = receipt
        .archive_sha256
        .as_deref()
        .expect("validated PASS receipt has archive_sha256");
    if source_archive_sha256 != expected_archive_sha256 {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} source-build archive sha256 differs from its receipt: expected={expected_archive_sha256} actual={source_archive_sha256}",
            receipt.operator
        )));
    }

    let toolchain = receipt
        .toolchain
        .as_ref()
        .expect("validated PASS receipt has toolchain");
    let build_summary = NativeOperatorBuildSummary {
        builder_sha: receipt.builder_sha.clone(),
        elapsed_ms: receipt.elapsed_ms,
        nvcc_version: Some(normalize_tool_version(&toolchain.nvcc.version)),
        host_compiler: format!(
            "path={};sha256={};version={}",
            toolchain.host_compiler.path,
            toolchain.host_compiler.sha256,
            normalize_tool_version(&toolchain.host_compiler.version)
        ),
    };

    Ok(ValidatedSourceBuildForPackage {
        receipt,
        receipt_sha256,
        plan_path: plan_path.to_path_buf(),
        archive_path,
        source_archive_sha256,
        build_summary,
    })
}

fn validate_source_build_for_package(
    receipt: &NativeOperatorSourceBuildReceipt,
    spec: &NativeOperatorPackageSpec,
) -> Result<()> {
    if receipt.schema_version != NATIVE_OPERATOR_SOURCE_BUILD_RECEIPT_SCHEMA_VERSION {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "source-build receipt schema_version must be {NATIVE_OPERATOR_SOURCE_BUILD_RECEIPT_SCHEMA_VERSION}"
        )));
    }
    if receipt.status != NativeOperatorSourceBuildStatus::Pass
        || receipt.plan_only
        || receipt.failure_class.is_some()
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "package requires a terminal PASS source-build receipt: status={:?} plan_only={} failure_class={:?}",
            receipt.status, receipt.plan_only, receipt.failure_class
        )));
    }
    if receipt.operator != spec.operator {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "source-build operator differs from package spec: expected={} actual={}",
            spec.operator, receipt.operator
        )));
    }
    if spec.compute_capabilities.as_slice() != std::slice::from_ref(&receipt.compute_capability) {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} package compute_capabilities must exactly equal the source-build target [{}]",
            receipt.operator, receipt.compute_capability
        )));
    }
    for (field, digest) in [
        ("plan_sha256", receipt.plan_sha256.as_str()),
        (
            "source_package.sha256",
            receipt.source_package.sha256.as_str(),
        ),
        ("inputs_sha256", receipt.inputs_sha256.as_str()),
    ] {
        if !is_sha256_digest(digest) {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "{} source-build receipt {field} is not a lowercase sha256 digest",
                receipt.operator
            )));
        }
    }
    if receipt.source_package.kind.trim().is_empty()
        || receipt.source_package.revision.trim().is_empty()
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} source-build source_package kind and revision must be non-empty",
            receipt.operator
        )));
    }
    if !is_git_object_id(&receipt.builder_sha) {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} source-build builder_sha must be a lowercase 40- or 64-hex git object id",
            receipt.operator
        )));
    }
    if receipt.nvcc_threads == 0 || receipt.nvcc_threads > MAX_NVCC_THREADS {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} source-build nvcc_threads must be in [1,{MAX_NVCC_THREADS}]",
            receipt.operator
        )));
    }
    if receipt.architecture_argument.trim().is_empty()
        || receipt.object_cache_root.trim().is_empty()
        || receipt.effective_environment.is_empty()
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} source-build architecture, object cache, and effective environment must be recorded",
            receipt.operator
        )));
    }

    let archive_file = receipt.archive_file.as_deref().ok_or_else(|| {
        NativeOperatorBuilderError::Invalid(format!(
            "{} PASS source-build receipt is missing archive_file",
            receipt.operator
        ))
    })?;
    validate_relative_path(archive_file)?;
    if Path::new(archive_file).parent() != Some(Path::new(""))
        || Path::new(archive_file).extension() != Some(OsStr::new("a"))
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} source-build archive_file must be a .a filename without directories",
            receipt.operator
        )));
    }
    if !receipt
        .archive_sha256
        .as_deref()
        .is_some_and(is_sha256_digest)
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} PASS source-build receipt is missing a valid archive_sha256",
            receipt.operator
        )));
    }

    let toolchain = receipt.toolchain.as_ref().ok_or_else(|| {
        NativeOperatorBuilderError::Invalid(format!(
            "{} PASS source-build receipt is missing toolchain provenance",
            receipt.operator
        ))
    })?;
    for (name, tool) in [
        ("nvcc", &toolchain.nvcc),
        ("host_compiler", &toolchain.host_compiler),
        ("archiver", &toolchain.archiver),
    ] {
        if tool.path.trim().is_empty()
            || tool.version.trim().is_empty()
            || !is_sha256_digest(&tool.sha256)
        {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "{} source-build {name} identity is incomplete",
                receipt.operator
            )));
        }
    }

    if receipt.commands.len() < 2 {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} PASS source-build receipt must contain translation-unit and archive commands",
            receipt.operator
        )));
    }
    let (archive_command, translation_unit_commands) = receipt
        .commands
        .split_last()
        .expect("commands length checked above");
    let expected_working_directory = translation_unit_commands[0].working_directory.clone();
    if expected_working_directory.trim().is_empty()
        || !Path::new(&expected_working_directory).is_absolute()
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} source-build working directory must be a recorded absolute path",
            receipt.operator
        )));
    }
    let mut observed_compiled = Vec::new();
    let mut observed_cache_hits = Vec::new();
    let mut previous_translation_unit: Option<&str> = None;
    for command in translation_unit_commands {
        let translation_unit = command.translation_unit.as_deref().ok_or_else(|| {
            NativeOperatorBuilderError::Invalid(format!(
                "{} source-build translation-unit command is missing its source path",
                receipt.operator
            ))
        })?;
        validate_relative_path(translation_unit)?;
        if previous_translation_unit.is_some_and(|previous| previous >= translation_unit) {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "{} source-build translation-unit commands must be sorted and unique",
                receipt.operator
            )));
        }
        previous_translation_unit = Some(translation_unit);
        validate_source_build_command_common(receipt, command, &expected_working_directory)?;
        if command.object_file.as_deref().is_none_or(str::is_empty)
            || !command
                .object_cache_key
                .as_deref()
                .is_some_and(is_sha256_digest)
            || !command
                .object_sha256
                .as_deref()
                .is_some_and(is_sha256_digest)
            || command
                .object_cache_entry
                .as_deref()
                .is_none_or(str::is_empty)
            || command.elapsed_ms.is_none()
        {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "{} source-build command for {translation_unit} has incomplete object evidence",
                receipt.operator
            )));
        }
        match command.object_cache_status {
            Some(NativeOperatorSourceObjectCacheStatus::Hit)
                if !command.compiler_executed && command.return_code.is_none() =>
            {
                observed_cache_hits.push(translation_unit.to_string());
            }
            Some(NativeOperatorSourceObjectCacheStatus::Published)
                if command.compiler_executed && command.return_code == Some(0) =>
            {
                observed_compiled.push(translation_unit.to_string());
            }
            _ => {
                return Err(NativeOperatorBuilderError::Invalid(format!(
                    "{} source-build command for {translation_unit} is not terminal",
                    receipt.operator
                )));
            }
        }
    }
    validate_source_build_command_common(receipt, archive_command, &expected_working_directory)?;
    if archive_command.translation_unit.is_some()
        || archive_command.object_file.is_some()
        || archive_command.object_cache_status.is_some()
        || archive_command.object_cache_key.is_some()
        || archive_command.object_sha256.is_some()
        || archive_command.return_code != Some(0)
        || archive_command.elapsed_ms.is_none()
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} source-build archive command is not terminal",
            receipt.operator
        )));
    }
    if observed_compiled != receipt.compiled_translation_units
        || observed_cache_hits != receipt.cache_hit_translation_units
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} source-build compiled/cache-hit summaries do not match command evidence",
            receipt.operator
        )));
    }
    Ok(())
}

fn validate_source_build_command_common(
    receipt: &NativeOperatorSourceBuildReceipt,
    command: &NativeOperatorSourceBuildCommand,
    expected_working_directory: &str,
) -> Result<()> {
    if command.working_directory != expected_working_directory || command.argv.is_empty() {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} source-build command has an unexpected working directory or empty argv",
            receipt.operator
        )));
    }
    validate_relative_path(&command.stdout_log)?;
    validate_relative_path(&command.stderr_log)?;
    Ok(())
}

fn is_git_object_id(value: &str) -> bool {
    matches!(value.len(), 40 | 64)
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn normalize_tool_version(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn copy_source_build_logs(
    receipt_path: &Path,
    receipt: &NativeOperatorSourceBuildReceipt,
    package_root: &Path,
) -> Result<Vec<NativeOperatorEvidenceFile>> {
    let source_build_root = receipt_path.parent().unwrap_or_else(|| Path::new("."));
    let mut logs = receipt
        .commands
        .iter()
        .flat_map(|command| [&command.stdout_log, &command.stderr_log])
        .cloned()
        .collect::<Vec<_>>();
    logs.sort();
    logs.dedup();
    if logs.is_empty() {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} source-build receipt contains no command logs",
            receipt.operator
        )));
    }
    logs.into_iter()
        .map(|relative| {
            let source = resolve_relative_file(source_build_root, &relative)?;
            let evidence =
                copy_evidence_file(&source, package_root, &format!("provenance/{relative}"))?;
            if evidence.size_bytes == 0 {
                return Err(NativeOperatorBuilderError::Invalid(format!(
                    "{} source-build log is empty: {relative}",
                    receipt.operator
                )));
            }
            Ok(evidence)
        })
        .collect()
}

fn copy_evidence_file(
    source: &Path,
    package_root: &Path,
    output_relative: &str,
) -> Result<NativeOperatorEvidenceFile> {
    require_file(source)?;
    validate_relative_path(output_relative)?;
    let destination = package_root.join(output_relative);
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|source| NativeOperatorBuilderError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    fs::copy(source, &destination).map_err(|source| NativeOperatorBuilderError::Io {
        path: destination.clone(),
        source,
    })?;
    let size_bytes = fs::metadata(&destination)
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: destination.clone(),
            source,
        })?
        .len();
    if size_bytes == 0 {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "native operator evidence file is empty: {}",
            source.display()
        )));
    }
    Ok(NativeOperatorEvidenceFile {
        path: output_relative.to_string(),
        sha256: sha256_file(&destination)?,
        size_bytes,
    })
}

fn evidence_file_at(package_root: &Path, relative: &str) -> Result<NativeOperatorEvidenceFile> {
    let path = resolve_relative_file(package_root, relative)?;
    let size_bytes = fs::metadata(&path)
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: path.clone(),
            source,
        })?
        .len();
    if size_bytes == 0 {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "package build log is empty: {relative}"
        )));
    }
    Ok(NativeOperatorEvidenceFile {
        path: relative.to_string(),
        sha256: sha256_file(&path)?,
        size_bytes,
    })
}

fn verify_source_archive_members(
    archive_path: &Path,
    receipt: &NativeOperatorSourceBuildReceipt,
    archiver: &str,
    package_root: &Path,
    environment: &BTreeMap<String, String>,
) -> Result<(
    Vec<NativeOperatorArchiveMemberEvidence>,
    NativeOperatorEvidenceFile,
)> {
    let archive_file = archive_path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            NativeOperatorBuilderError::Invalid(format!(
                "source archive has no UTF-8 file name: {}",
                archive_path.display()
            ))
        })?;
    let translation_unit_commands = receipt
        .commands
        .get(..receipt.commands.len().saturating_sub(1))
        .ok_or_else(|| {
            NativeOperatorBuilderError::Invalid(format!(
                "{} source-build receipt has no archive command",
                receipt.operator
            ))
        })?;
    let expected = translation_unit_commands
        .iter()
        .map(|command| {
            let object_file = command.object_file.as_deref().ok_or_else(|| {
                NativeOperatorBuilderError::Invalid(format!(
                    "{} source-build command is missing object_file",
                    receipt.operator
                ))
            })?;
            let member = Path::new(object_file)
                .file_name()
                .and_then(|value| value.to_str())
                .ok_or_else(|| {
                    NativeOperatorBuilderError::Invalid(format!(
                        "{} source-build object has no UTF-8 file name: {object_file}",
                        receipt.operator
                    ))
                })?
                .to_string();
            validate_relative_path(&member)?;
            let sha256 = command.object_sha256.clone().ok_or_else(|| {
                NativeOperatorBuilderError::Invalid(format!(
                    "{} source-build command is missing object_sha256: {member}",
                    receipt.operator
                ))
            })?;
            Ok((member, sha256))
        })
        .collect::<Result<Vec<_>>>()?;
    if expected.is_empty()
        || expected
            .windows(2)
            .any(|pair| pair[0].0.as_str() >= pair[1].0.as_str())
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} source-build archive members must be sorted, unique, and non-empty",
            receipt.operator
        )));
    }

    let list_output = Command::new(archiver)
        .args(["t", archive_file])
        .current_dir(package_root)
        .env_clear()
        .envs(environment)
        .output()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: PathBuf::from(archiver),
            source,
        })?;
    if !list_output.status.success() {
        return Err(NativeOperatorBuilderError::Tool {
            tool: archiver.to_string(),
            status: list_output.status.to_string(),
            stderr: String::from_utf8_lossy(&list_output.stderr)
                .trim()
                .chars()
                .take(2000)
                .collect(),
        });
    }
    let raw_listed = std::str::from_utf8(&list_output.stdout)
        .map_err(|_| {
            NativeOperatorBuilderError::Invalid(format!(
                "{} source archive member list is not UTF-8",
                receipt.operator
            ))
        })?
        .lines()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    let listed = raw_listed
        .iter()
        .filter(|member| !is_archive_metadata_member(member))
        .cloned()
        .collect::<Vec<_>>();
    let expected_names = expected
        .iter()
        .map(|(member, _)| member.clone())
        .collect::<Vec<_>>();
    if listed != expected_names {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} source archive members differ from source-build receipt: expected={expected_names:?} actual={listed:?}",
            receipt.operator
        )));
    }

    let mut evidence = Vec::with_capacity(expected.len());
    let mut log = format!(
        "operator={}\narchive={archive_file}\narchive_sha256={}\nargv={} t {archive_file}\nraw_members={}\nmembers={}\n",
        receipt.operator,
        sha256_file(archive_path)?,
        archiver,
        raw_listed.join(","),
        expected_names.join(",")
    );
    for (member, expected_sha256) in expected {
        let output = Command::new(archiver)
            .args(["p", archive_file, &member])
            .current_dir(package_root)
            .env_clear()
            .envs(environment)
            .output()
            .map_err(|source| NativeOperatorBuilderError::Io {
                path: PathBuf::from(archiver),
                source,
            })?;
        if !output.status.success() {
            return Err(NativeOperatorBuilderError::Tool {
                tool: archiver.to_string(),
                status: output.status.to_string(),
                stderr: String::from_utf8_lossy(&output.stderr)
                    .trim()
                    .chars()
                    .take(2000)
                    .collect(),
            });
        }
        let actual_sha256 = sha256_bytes(&output.stdout);
        if actual_sha256 != expected_sha256 {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "{} source archive member hash mismatch: member={member} expected={expected_sha256} actual={actual_sha256}",
                receipt.operator
            )));
        }
        log.push_str(&format!(
            "argv={archiver} p {archive_file} {member}\nmember={member} sha256={actual_sha256} size_bytes={}\n",
            output.stdout.len()
        ));
        evidence.push(NativeOperatorArchiveMemberEvidence {
            member,
            sha256: actual_sha256,
        });
    }
    let verification_log = "build-logs/source-archive-verify.log";
    let verification_path = package_root.join(verification_log);
    if let Some(parent) = verification_path.parent() {
        fs::create_dir_all(parent).map_err(|source| NativeOperatorBuilderError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    fs::write(&verification_path, log).map_err(|source| NativeOperatorBuilderError::Io {
        path: verification_path,
        source,
    })?;
    Ok((evidence, evidence_file_at(package_root, verification_log)?))
}

fn is_archive_metadata_member(member: &str) -> bool {
    matches!(
        member,
        "/" | "//" | "__.SYMDEF" | "__.SYMDEF SORTED" | "__.SYMDEF_64" | "__.SYMDEF_64 SORTED"
    )
}

fn package_build_environment(
    toolchain: &NativeOperatorPackageToolchain,
) -> Result<BTreeMap<String, String>> {
    let mut path_entries = [
        toolchain.descriptor_compiler.path.as_str(),
        toolchain.archiver.path.as_str(),
    ]
    .iter()
    .filter_map(|path| Path::new(path).parent())
    .map(|path| path.display().to_string())
    .collect::<Vec<_>>();
    path_entries.extend(["/bin".to_string(), "/usr/bin".to_string()]);
    path_entries.sort();
    path_entries.dedup();
    if path_entries.iter().any(|path| path.is_empty()) {
        return Err(NativeOperatorBuilderError::Invalid(
            "package tool paths must have parent directories".to_string(),
        ));
    }
    Ok(BTreeMap::from([
        ("LANG".to_string(), "C".to_string()),
        ("LC_ALL".to_string(), "C".to_string()),
        ("PATH".to_string(), path_entries.join(":")),
        ("SOURCE_DATE_EPOCH".to_string(), "0".to_string()),
        ("TMPDIR".to_string(), "/tmp".to_string()),
        ("TZ".to_string(), "UTC".to_string()),
        ("ZERO_AR_DATE".to_string(), "1".to_string()),
    ]))
}

fn run_package_command(
    program: &str,
    args: Vec<String>,
    working_directory: &Path,
    stdout_log: &str,
    stderr_log: &str,
    environment: &BTreeMap<String, String>,
) -> Result<NativeOperatorPackageCommand> {
    validate_relative_path(stdout_log)?;
    validate_relative_path(stderr_log)?;
    for relative in [stdout_log, stderr_log] {
        let parent = working_directory
            .join(relative)
            .parent()
            .expect("validated log path has parent")
            .to_path_buf();
        fs::create_dir_all(&parent).map_err(|source| NativeOperatorBuilderError::Io {
            path: parent,
            source,
        })?;
    }
    let argv = std::iter::once(program.to_string())
        .chain(args.iter().cloned())
        .collect::<Vec<_>>();
    let started = Instant::now();
    let output = Command::new(program)
        .args(&args)
        .current_dir(working_directory)
        .env_clear()
        .envs(environment)
        .output()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: PathBuf::from(program),
            source,
        })?;
    let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    let write_log = |relative: &str, stream: &str, bytes: &[u8]| -> Result<()> {
        let path = working_directory.join(relative);
        let mut content = format!(
            "stream={stream}\nworking_directory=.\nargv={}\n",
            argv.join(" ")
        )
        .into_bytes();
        content.extend_from_slice(bytes);
        if !content.ends_with(b"\n") {
            content.push(b'\n');
        }
        fs::write(&path, content).map_err(|source| NativeOperatorBuilderError::Io { path, source })
    };
    write_log(stdout_log, "stdout", &output.stdout)?;
    write_log(stderr_log, "stderr", &output.stderr)?;
    let return_code = output.status.code().unwrap_or(-1);
    if !output.status.success() {
        return Err(NativeOperatorBuilderError::Tool {
            tool: program.to_string(),
            status: output.status.to_string(),
            stderr: String::from_utf8_lossy(&output.stderr)
                .trim()
                .chars()
                .take(2000)
                .collect(),
        });
    }
    Ok(NativeOperatorPackageCommand {
        argv,
        working_directory: ".".to_string(),
        stdout_log: stdout_log.to_string(),
        stderr_log: stderr_log.to_string(),
        return_code,
        elapsed_ms,
    })
}

pub fn assemble_native_operator_set(
    request: &NativeOperatorSetRequest,
) -> Result<NativeOperatorArtifactSetLock> {
    if request.receipt_paths.is_empty() {
        return Err(NativeOperatorBuilderError::Invalid(
            "artifact set requires at least one package receipt".to_string(),
        ));
    }
    if request.output_lock_path.exists() {
        return Err(NativeOperatorBuilderError::OutputExists(
            request.output_lock_path.clone(),
        ));
    }
    if !request.compute_capability.starts_with("sm_") {
        return Err(NativeOperatorBuilderError::Invalid(
            "compute_capability must use sm_xx form".to_string(),
        ));
    }
    let root = request
        .output_lock_path
        .parent()
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(root).map_err(|source| NativeOperatorBuilderError::Io {
        path: root.to_path_buf(),
        source,
    })?;
    let canonical_root = root
        .canonicalize()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: root.to_path_buf(),
            source,
        })?;

    let mut expected_catalog: Option<String> = None;
    let mut artifacts = Vec::with_capacity(request.receipt_paths.len());
    for receipt_path in &request.receipt_paths {
        require_file(receipt_path)?;
        let receipt: NativeOperatorPackageReceipt = read_json(receipt_path)?;
        validate_package_receipt(&receipt)?;
        match expected_catalog.as_deref() {
            Some(expected) if expected != receipt.g03_catalog_sha256 => {
                return Err(NativeOperatorBuilderError::Invalid(format!(
                    "package {} catalog differs from artifact set: expected={expected} actual={}",
                    receipt.operator, receipt.g03_catalog_sha256
                )))
            }
            None => expected_catalog = Some(receipt.g03_catalog_sha256.clone()),
            _ => {}
        }
        let package_root = receipt_path.parent().unwrap_or_else(|| Path::new("."));
        let package_receipt = resolve_file_evidence(receipt_path, &canonical_root)?;
        let manifest_path = resolve_relative_file(package_root, &receipt.manifest_file)?;
        let artifact_path = resolve_relative_file(package_root, &receipt.artifact_file)?;
        let source_build_receipt =
            resolve_package_evidence(package_root, &canonical_root, &receipt.source_build_receipt)?;
        let source_build_plan =
            resolve_package_evidence(package_root, &canonical_root, &receipt.source_build_plan)?;
        let source_build_logs = receipt
            .source_build_logs
            .iter()
            .map(|evidence| resolve_package_evidence(package_root, &canonical_root, evidence))
            .collect::<Result<Vec<_>>>()?;
        let package_build_logs = receipt
            .package_build_logs
            .iter()
            .map(|evidence| resolve_package_evidence(package_root, &canonical_root, evidence))
            .collect::<Result<Vec<_>>>()?;
        if sha256_file(&manifest_path)? != receipt.manifest_sha256 {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "{} manifest sha256 differs from its package receipt",
                receipt.operator
            )));
        }
        if sha256_file(&artifact_path)? != receipt.binary_sha256 {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "{} artifact sha256 differs from its package receipt",
                receipt.operator
            )));
        }
        let manifest: NativeOperatorManifest = read_json(&manifest_path)?;
        manifest
            .validate()
            .map_err(NativeOperatorBuilderError::Invalid)?;
        if manifest.operator != receipt.operator
            || manifest.binary_sha256 != receipt.binary_sha256
            || manifest.g03_catalog_sha256.as_deref() != Some(&receipt.g03_catalog_sha256)
            || manifest.abi_contract_sha256.as_deref() != Some(&receipt.abi_contract_sha256)
            || manifest.descriptor_export.as_deref() != Some(&receipt.descriptor_export)
        {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "{} package receipt does not match its manifest",
                receipt.operator
            )));
        }
        NativeOperatorResolver.resolve(
            &NativeOperatorResolveRequest::new(
                manifest.operator.clone(),
                manifest.backend,
                &manifest_path,
                &artifact_path,
            )
            .with_compute_capability(request.compute_capability.clone())
            .with_operator_abi_version(manifest.operator_abi_version.clone())
            .with_ferrum_native_abi_version(manifest.ferrum_native_abi_version.clone())
            .with_g03_catalog_sha256(receipt.g03_catalog_sha256.clone())
            .with_abi_contract_sha256(receipt.abi_contract_sha256.clone())
            .with_descriptor_export(receipt.descriptor_export.clone())
            .with_required_exports(manifest.exports.clone())
            .with_operation_bindings(manifest.operation_bindings.clone()),
        )?;

        artifacts.push(NativeOperatorArtifactLock {
            operator: manifest.operator,
            backend: manifest.backend,
            manifest_path: relative_path(&canonical_root, &manifest_path)?,
            artifact_path: relative_path(&canonical_root, &artifact_path)?,
            operator_abi_version: manifest.operator_abi_version,
            ferrum_native_abi_version: manifest.ferrum_native_abi_version,
            source_package_sha256: manifest.source_package.sha256,
            inputs_sha256: manifest.inputs_sha256,
            source_build_receipt,
            source_build_plan,
            source_build_logs,
            source_archive_sha256: receipt.source_archive_sha256,
            package_receipt,
            package_build_logs,
            binary_sha256: receipt.binary_sha256,
            abi_contract_sha256: receipt.abi_contract_sha256,
            descriptor_export: receipt.descriptor_export,
            required_exports: manifest.exports,
            operation_bindings: manifest.operation_bindings,
            system_libraries: receipt.system_libraries,
        });
    }
    artifacts.sort_by(|left, right| left.operator.cmp(&right.operator));
    if artifacts
        .windows(2)
        .any(|pair| pair[0].operator == pair[1].operator)
    {
        return Err(NativeOperatorBuilderError::Invalid(
            "artifact set contains duplicate operators".to_string(),
        ));
    }
    let lock = NativeOperatorArtifactSetLock {
        schema_version: NATIVE_OPERATOR_ARTIFACT_SET_SCHEMA_VERSION,
        g03_catalog_sha256: expected_catalog.expect("non-empty receipts"),
        artifacts,
    };

    let mut temporary =
        NamedTempFile::new_in(root).map_err(|source| NativeOperatorBuilderError::Io {
            path: root.to_path_buf(),
            source,
        })?;
    serde_json::to_writer_pretty(&mut temporary, &lock).map_err(|source| {
        NativeOperatorBuilderError::Json {
            path: temporary.path().to_path_buf(),
            source,
        }
    })?;
    temporary
        .write_all(b"\n")
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: temporary.path().to_path_buf(),
            source,
        })?;
    temporary
        .flush()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: temporary.path().to_path_buf(),
            source,
        })?;
    NativeOperatorArtifactSetLock::load_and_resolve(
        temporary.path(),
        Some(&request.compute_capability),
    )?;
    temporary
        .persist(&request.output_lock_path)
        .map_err(|error| NativeOperatorBuilderError::Io {
            path: request.output_lock_path.clone(),
            source: error.error,
        })?;
    Ok(lock)
}

fn validate_package_spec(spec: &NativeOperatorPackageSpec) -> Result<()> {
    if spec.schema_version != NATIVE_OPERATOR_PACKAGE_SPEC_SCHEMA_VERSION {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "package spec schema_version must be {NATIVE_OPERATOR_PACKAGE_SPEC_SCHEMA_VERSION}"
        )));
    }
    if spec.backend != NativeOperatorBackend::Cuda {
        return Err(NativeOperatorBuilderError::Invalid(
            "source-build packager currently accepts CUDA artifacts only".to_string(),
        ));
    }
    symbol_slug(&spec.operator)?;
    if spec.operator_abi_version.trim().is_empty() {
        return Err(NativeOperatorBuilderError::Invalid(
            "operator_abi_version must be non-empty".to_string(),
        ));
    }
    require_sorted_unique_non_empty("compute_capabilities", &spec.compute_capabilities)?;
    if spec
        .compute_capabilities
        .iter()
        .any(|capability| !capability.starts_with("sm_"))
    {
        return Err(NativeOperatorBuilderError::Invalid(
            "compute_capabilities must use sm_xx form".to_string(),
        ));
    }
    require_sorted_unique_non_empty("required_exports", &spec.required_exports)?;
    if let Some(unit) = CudaNativeBuildUnit::from_artifact_operator(&spec.operator) {
        for required in unit.required_exports() {
            if !spec
                .required_exports
                .iter()
                .any(|export| export == required)
            {
                return Err(NativeOperatorBuilderError::Invalid(format!(
                    "{} package spec is missing build-unit export {required}",
                    unit.as_str()
                )));
            }
        }
    }
    if spec.license_files.is_empty() {
        return Err(NativeOperatorBuilderError::Invalid(
            "license_files must be non-empty".to_string(),
        ));
    }
    let mut previous_license: Option<&str> = None;
    for license in &spec.license_files {
        validate_relative_path(&license.source_path)?;
        validate_relative_path(&license.output_path)?;
        if previous_license.is_some_and(|previous| previous >= license.output_path.as_str()) {
            return Err(NativeOperatorBuilderError::Invalid(
                "license_files must be sorted and unique by output_path".to_string(),
            ));
        }
        previous_license = Some(&license.output_path);
    }
    if spec
        .system_libraries
        .windows(2)
        .any(|pair| pair[0] >= pair[1])
    {
        return Err(NativeOperatorBuilderError::Invalid(
            "system_libraries must be sorted and unique".to_string(),
        ));
    }
    Ok(())
}

fn validate_package_receipt(receipt: &NativeOperatorPackageReceipt) -> Result<()> {
    if receipt.schema_version != NATIVE_OPERATOR_PACKAGE_RECEIPT_SCHEMA_VERSION {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} package receipt schema_version must be {}",
            receipt.operator, NATIVE_OPERATOR_PACKAGE_RECEIPT_SCHEMA_VERSION
        )));
    }
    if receipt.operator.trim().is_empty() || receipt.descriptor_export.trim().is_empty() {
        return Err(NativeOperatorBuilderError::Invalid(
            "package receipt operator and descriptor_export must be non-empty".to_string(),
        ));
    }
    validate_relative_path(&receipt.manifest_file)?;
    validate_relative_path(&receipt.artifact_file)?;
    validate_evidence_record("source_build_receipt", &receipt.source_build_receipt)?;
    validate_evidence_record("source_build_plan", &receipt.source_build_plan)?;
    if receipt.source_build_logs.is_empty()
        || receipt
            .source_build_logs
            .windows(2)
            .any(|pair| pair[0].path >= pair[1].path)
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} package receipt source_build_logs must be sorted, unique, and non-empty",
            receipt.operator
        )));
    }
    for evidence in &receipt.source_build_logs {
        validate_evidence_record("source_build_log", evidence)?;
    }
    validate_evidence_record(
        "source_archive_verification",
        &receipt.source_archive_verification,
    )?;
    if receipt.source_archive_members.is_empty()
        || receipt
            .source_archive_members
            .windows(2)
            .any(|pair| pair[0].member >= pair[1].member)
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} package receipt source_archive_members must be sorted, unique, and non-empty",
            receipt.operator
        )));
    }
    for member in &receipt.source_archive_members {
        validate_relative_path(&member.member)?;
        if Path::new(&member.member).parent() != Some(Path::new(""))
            || !is_sha256_digest(&member.sha256)
        {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "{} package receipt source archive member is invalid: {}",
                receipt.operator, member.member
            )));
        }
    }
    for (field, digest) in [
        ("source_archive_sha256", &receipt.source_archive_sha256),
        ("manifest_sha256", &receipt.manifest_sha256),
        ("binary_sha256", &receipt.binary_sha256),
        ("g03_catalog_sha256", &receipt.g03_catalog_sha256),
        ("abi_contract_sha256", &receipt.abi_contract_sha256),
    ] {
        if !is_sha256_digest(digest) {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "{} package receipt {field} is not a sha256 digest",
                receipt.operator
            )));
        }
    }
    if receipt
        .system_libraries
        .windows(2)
        .any(|pair| pair[0] >= pair[1])
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} package receipt system_libraries must be sorted and unique",
            receipt.operator
        )));
    }
    for (name, tool) in [
        (
            "descriptor_compiler",
            &receipt.package_toolchain.descriptor_compiler,
        ),
        ("archiver", &receipt.package_toolchain.archiver),
    ] {
        if !Path::new(&tool.path).is_absolute()
            || tool.version.trim().is_empty()
            || !is_sha256_digest(&tool.sha256)
        {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "{} package receipt {name} identity is incomplete",
                receipt.operator
            )));
        }
    }
    let expected_environment = package_build_environment(&receipt.package_toolchain)?;
    if receipt.package_environment != expected_environment {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} package environment differs from deterministic policy",
            receipt.operator
        )));
    }
    let expected_commands = [
        (
            vec![
                receipt.package_toolchain.descriptor_compiler.path.clone(),
                "-std=c11".to_string(),
                "-O2".to_string(),
                "-fno-ident".to_string(),
                "-fvisibility=hidden".to_string(),
                "-c".to_string(),
                "descriptor.c".to_string(),
                "-o".to_string(),
                "descriptor.o".to_string(),
            ],
            "build-logs/descriptor-compile.stdout.log",
            "build-logs/descriptor-compile.stderr.log",
        ),
        (
            vec![
                receipt.package_toolchain.archiver.path.clone(),
                "rcs".to_string(),
                receipt.artifact_file.clone(),
                "descriptor.o".to_string(),
            ],
            "build-logs/descriptor-archive.stdout.log",
            "build-logs/descriptor-archive.stderr.log",
        ),
    ];
    if receipt.package_commands.len() != expected_commands.len() {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} package receipt must contain exactly {} package commands",
            receipt.operator,
            expected_commands.len()
        )));
    }
    for (command, (expected_argv, expected_stdout, expected_stderr)) in
        receipt.package_commands.iter().zip(expected_commands)
    {
        validate_relative_path(&command.stdout_log)?;
        validate_relative_path(&command.stderr_log)?;
        if command.argv != expected_argv
            || command.working_directory != "."
            || command.stdout_log != expected_stdout
            || command.stderr_log != expected_stderr
            || command.return_code != 0
        {
            return Err(NativeOperatorBuilderError::Invalid(format!(
                "{} package command differs from deterministic policy: {:?}",
                receipt.operator, command.argv
            )));
        }
    }
    if receipt.package_build_logs.is_empty()
        || receipt
            .package_build_logs
            .windows(2)
            .any(|pair| pair[0].path >= pair[1].path)
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} package_build_logs must be sorted, unique, and non-empty",
            receipt.operator
        )));
    }
    for evidence in &receipt.package_build_logs {
        validate_evidence_record("package_build_log", evidence)?;
    }
    let mut expected_log_paths = vec![receipt.source_archive_verification.path.clone()];
    expected_log_paths.extend(
        receipt
            .package_commands
            .iter()
            .flat_map(|command| [command.stdout_log.clone(), command.stderr_log.clone()]),
    );
    expected_log_paths.sort();
    let actual_log_paths = receipt
        .package_build_logs
        .iter()
        .map(|evidence| evidence.path.clone())
        .collect::<Vec<_>>();
    if actual_log_paths != expected_log_paths
        || !receipt
            .package_build_logs
            .contains(&receipt.source_archive_verification)
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{} package build-log evidence does not match recorded commands",
            receipt.operator
        )));
    }
    Ok(())
}

fn validate_evidence_record(field: &str, evidence: &NativeOperatorEvidenceFile) -> Result<()> {
    validate_relative_path(&evidence.path)?;
    if !is_sha256_digest(&evidence.sha256) || evidence.size_bytes == 0 {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{field} must record a non-empty file and lowercase sha256"
        )));
    }
    Ok(())
}

fn resolve_package_evidence(
    package_root: &Path,
    artifact_set_root: &Path,
    evidence: &NativeOperatorEvidenceFile,
) -> Result<NativeOperatorEvidenceFile> {
    validate_evidence_record("package evidence", evidence)?;
    let path = resolve_relative_file(package_root, &evidence.path)?;
    let actual_size = fs::metadata(&path)
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: path.clone(),
            source,
        })?
        .len();
    let actual_sha256 = sha256_file(&path)?;
    if actual_sha256 != evidence.sha256 || actual_size != evidence.size_bytes {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "package evidence differs from its receipt: {}",
            evidence.path
        )));
    }
    Ok(NativeOperatorEvidenceFile {
        path: relative_path(artifact_set_root, &path)?,
        sha256: actual_sha256,
        size_bytes: actual_size,
    })
}

fn resolve_file_evidence(
    path: &Path,
    artifact_set_root: &Path,
) -> Result<NativeOperatorEvidenceFile> {
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
            "native operator evidence file is empty: {}",
            canonical.display()
        )));
    }
    Ok(NativeOperatorEvidenceFile {
        path: relative_path(artifact_set_root, &canonical)?,
        sha256: sha256_file(&canonical)?,
        size_bytes,
    })
}

fn render_descriptor_source(
    descriptor_export: &str,
    operator: &str,
    operator_abi_version: &str,
    g03_catalog_sha256: &str,
    abi_contract_sha256: &str,
) -> String {
    format!(
        "#include <stdint.h>\n\
         typedef struct {{\n\
           uint32_t struct_size;\n\
           uint32_t ferrum_native_abi_version;\n\
           const char *operator_name;\n\
           const char *operator_abi_version;\n\
           const char *g03_catalog_sha256;\n\
           const char *abi_contract_sha256;\n\
         }} FerrumNativeOperatorDescriptorV2;\n\
         static const FerrumNativeOperatorDescriptorV2 descriptor = {{\n\
           sizeof(FerrumNativeOperatorDescriptorV2),\n\
           {FERRUM_NATIVE_OPERATOR_ABI_VERSION},\n\
           \"{}\",\n\
           \"{}\",\n\
           \"{g03_catalog_sha256}\",\n\
           \"{abi_contract_sha256}\"\n\
         }};\n\
         __attribute__((visibility(\"default\")))\n\
         const FerrumNativeOperatorDescriptorV2 *{descriptor_export}(void) {{\n\
           return &descriptor;\n\
         }}\n",
        c_string(operator),
        c_string(operator_abi_version),
    )
}

fn c_string(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

fn symbol_slug(operator: &str) -> Result<String> {
    if operator.is_empty()
        || !operator
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(NativeOperatorBuilderError::Invalid(
            "operator must use ASCII alphanumeric, dot, underscore, or hyphen characters"
                .to_string(),
        ));
    }
    Ok(operator
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '_'
            }
        })
        .collect())
}

fn require_sorted_unique_non_empty(field: &str, values: &[String]) -> Result<()> {
    if values.is_empty()
        || values.iter().any(|value| value.trim().is_empty())
        || values.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "{field} must be sorted, unique, and non-empty"
        )));
    }
    Ok(())
}

fn validate_relative_path(path: &str) -> Result<()> {
    let path = Path::new(path);
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "path must be a non-empty normalized relative path: {}",
            path.display()
        )));
    }
    Ok(())
}

fn resolve_relative_file(root: &Path, relative: &str) -> Result<PathBuf> {
    validate_relative_path(relative)?;
    let path = root.join(relative);
    let canonical = path
        .canonicalize()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: path.clone(),
            source,
        })?;
    let canonical_root = root
        .canonicalize()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: root.to_path_buf(),
            source,
        })?;
    if !canonical.starts_with(&canonical_root) || !canonical.is_file() {
        return Err(NativeOperatorBuilderError::Invalid(format!(
            "relative file escapes its root or is not a file: {relative}"
        )));
    }
    Ok(canonical)
}

fn relative_path(root: &Path, path: &Path) -> Result<String> {
    let canonical = path
        .canonicalize()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let relative = canonical.strip_prefix(root).map_err(|_| {
        NativeOperatorBuilderError::Invalid(format!(
            "artifact-set input {} is outside {}",
            canonical.display(),
            root.display()
        ))
    })?;
    let value = relative.to_string_lossy().replace('\\', "/");
    validate_relative_path(&value)?;
    Ok(value)
}

fn require_file(path: &Path) -> Result<()> {
    if path.is_file() {
        Ok(())
    } else {
        Err(NativeOperatorBuilderError::MissingFile(path.to_path_buf()))
    }
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).map_err(|source| NativeOperatorBuilderError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| NativeOperatorBuilderError::Json {
        path: path.to_path_buf(),
        source,
    })
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut bytes =
        serde_json::to_vec_pretty(value).map_err(|source| NativeOperatorBuilderError::Json {
            path: path.to_path_buf(),
            source,
        })?;
    bytes.push(b'\n');
    let parent = path.parent().ok_or_else(|| {
        NativeOperatorBuilderError::Invalid(format!(
            "JSON output has no parent directory: {}",
            path.display()
        ))
    })?;
    let mut temporary =
        NamedTempFile::new_in(parent).map_err(|source| NativeOperatorBuilderError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    temporary
        .as_file_mut()
        .write_all(&bytes)
        .and_then(|()| temporary.as_file_mut().flush())
        .and_then(|()| temporary.as_file().sync_all())
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: temporary.path().to_path_buf(),
            source,
        })?;
    temporary
        .persist(path)
        .map_err(|error| NativeOperatorBuilderError::Io {
            path: path.to_path_buf(),
            source: error.error,
        })?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).map_err(|source| NativeOperatorBuilderError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(sha256_bytes(&bytes))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_native_ops::NativeOperatorArtifactSetLock;
    use std::os::unix::fs::PermissionsExt;

    fn digest(character: char) -> String {
        std::iter::repeat(character).take(64).collect()
    }

    fn run_source_build_fixture(
        root: &Path,
        source_root: &Path,
        operator: &str,
        name: &str,
        exports: &[&str],
    ) -> (PathBuf, PathBuf, PathBuf) {
        let source = exports
            .iter()
            .enumerate()
            .map(|(index, export)| format!("int {export}(void) {{ return {index}; }}\n"))
            .collect::<String>();
        fs::write(source_root.join("fixture.cu"), source).unwrap();
        let definition = NativeOperatorSourceDefinition {
            schema_version: NATIVE_OPERATOR_SOURCE_DEFINITION_SCHEMA_VERSION,
            operator: operator.to_string(),
            source_package_kind: "fixture-source".to_string(),
            source_package_revision: "fixture".to_string(),
            upstream_sources: vec![NativeOperatorUpstreamSource {
                repository: "https://example.invalid/native-op.git".to_string(),
                revision: "fixture".to_string(),
                license: "Apache-2.0".to_string(),
            }],
            translation_units: vec!["fixture.cu".to_string()],
            headers: Vec::new(),
            include_dirs: Vec::new(),
            defines: Vec::new(),
            nvcc_policy: NativeOperatorNvccPolicy {
                cpp_standard: NativeOperatorCppStandard::Cpp17,
                optimization: NativeOperatorOptimization::O3,
                use_fast_math: false,
                relaxed_constexpr: false,
                extended_lambda: false,
                host_position_independent_code: true,
                host_default_visibility: false,
            },
            architecture: NativeOperatorCudaArchitecture::DeviceComputeCapability,
            archive_file: format!("lib{name}.a"),
        };
        let definition_path = root.join(format!("{name}.source-definition.json"));
        let plan_path = root.join(format!("{name}.source-build.plan.json"));
        write_json(&definition_path, &definition).unwrap();
        lock_native_operator_source_definition(&definition_path, source_root, &plan_path).unwrap();

        let fake_nvcc = root.join(format!("fake-nvcc-{name}"));
        fs::write(
            &fake_nvcc,
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
             exec /usr/bin/cc -x c -c \"$src\" -o \"$out\"\n",
        )
        .unwrap();
        let mut permissions = fs::metadata(&fake_nvcc).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&fake_nvcc, permissions).unwrap();

        let output_dir = root.join("source-builds").join(name);
        run_native_operator_source_build(&NativeOperatorSourceBuildRequest {
            plan_path: plan_path.clone(),
            source_root: source_root.to_path_buf(),
            output_dir: output_dir.clone(),
            compute_capability: "sm_89".to_string(),
            builder_sha: "7".repeat(40),
            nvcc_path: fake_nvcc,
            ccbin_path: PathBuf::from("/usr/bin/cc"),
            ar_path: PathBuf::from("/usr/bin/ar"),
            nvcc_threads: 2,
            object_cache_dir: root.join("object-cache"),
            plan_only: false,
        })
        .unwrap();
        (
            output_dir.join("source-build.receipt.json"),
            plan_path,
            output_dir.join(format!("lib{name}.a")),
        )
    }

    fn package_spec(
        operator: &str,
        operation_id: &str,
        provider_id: &str,
        exports: &[&str],
    ) -> NativeOperatorPackageSpec {
        NativeOperatorPackageSpec {
            schema_version: NATIVE_OPERATOR_PACKAGE_SPEC_SCHEMA_VERSION,
            operator: operator.to_string(),
            operator_abi_version: "1".to_string(),
            backend: NativeOperatorBackend::Cuda,
            compute_capabilities: vec!["sm_89".to_string()],
            operation_bindings: vec![NativeOperatorBinding {
                operation_id: operation_id.to_string(),
                operation_contract_version: 1,
                provider_id: provider_id.to_string(),
                provider_version: 1,
                provider_implementation_fingerprint: digest('c'),
                entrypoints: exports.iter().map(|value| (*value).to_string()).collect(),
            }],
            required_exports: exports.iter().map(|value| (*value).to_string()).collect(),
            license_files: vec![NativeOperatorLicenseInput {
                source_path: "LICENSE".to_string(),
                output_path: "licenses/LICENSE".to_string(),
            }],
            cuda_toolkit: Some("12.4".to_string()),
            cuda_runtime_min: Some("12.4".to_string()),
            system_libraries: vec![
                NativeOperatorSystemLibrary::CudaRuntime,
                NativeOperatorSystemLibrary::StdCxx,
            ],
        }
    }

    fn fixture_files(root: &Path) -> (PathBuf, PathBuf, PathBuf) {
        let source_root = root.join("source");
        fs::create_dir_all(&source_root).unwrap();
        fs::write(source_root.join("LICENSE"), "fixture license\n").unwrap();
        let catalog_path = root.join("operation-catalog.json");
        let abi_path = root.join("native-abi.json");
        fs::write(&catalog_path, "{\"schema\":1}\n").unwrap();
        fs::write(&abi_path, "{\"ferrum_native_abi\":2}\n").unwrap();
        (source_root, catalog_path, abi_path)
    }

    fn package_fixture(
        root: &Path,
        operator: &str,
        operation_id: &str,
        provider_id: &str,
        exports: &[&str],
        output_name: &str,
    ) -> Result<(PathBuf, NativeOperatorPackageReceipt)> {
        let (source_root, catalog_path, abi_path) = fixture_files(root);
        let (source_build_receipt, source_build_plan, _) =
            run_source_build_fixture(root, &source_root, operator, output_name, exports);
        let spec = package_spec(operator, operation_id, provider_id, exports);
        let spec_path = root.join(format!("{output_name}.package.json"));
        write_json(&spec_path, &spec)?;
        let output_dir = root.join("packages").join(output_name);
        let receipt = package_native_operator(&NativeOperatorPackageRequest {
            spec_path,
            source_root,
            source_build_receipt_path: source_build_receipt,
            source_build_plan_path: source_build_plan,
            g03_catalog_path: catalog_path,
            abi_contract_path: abi_path,
            output_dir: output_dir.clone(),
            cc: PathBuf::from("/usr/bin/cc"),
            ar: PathBuf::from("/usr/bin/ar"),
        })?;
        Ok((output_dir, receipt))
    }

    #[test]
    fn packages_archive_with_namespaced_descriptor_and_verified_manifest() {
        let root = tempfile::tempdir().unwrap();
        let exports = ["marlin_cuda", "marlin_cuda_moe"];
        let (output_dir, receipt) = package_fixture(
            root.path(),
            CudaNativeBuildUnit::Marlin.artifact_operator(),
            "operation.dense_linear",
            "provider.cuda.dense_linear.f16.marlin",
            &exports,
            "marlin",
        )
        .unwrap();

        assert!(output_dir.join("package.receipt.json").is_file());
        assert!(output_dir.join("licenses/LICENSE").is_file());
        assert!(!output_dir.join("descriptor.c").exists());
        assert!(!output_dir.join("descriptor.o").exists());
        let manifest: NativeOperatorManifest =
            read_json(&output_dir.join(&receipt.manifest_file)).unwrap();
        assert_eq!(
            manifest.descriptor_export.as_deref(),
            Some(receipt.descriptor_export.as_str())
        );
        assert!(manifest
            .exports
            .iter()
            .any(|export| export == &receipt.descriptor_export));
        assert!(manifest
            .exports
            .iter()
            .any(|export| export == "marlin_cuda"));
        assert_eq!(
            sha256_file(&output_dir.join(&receipt.artifact_file)).unwrap(),
            receipt.binary_sha256
        );
        assert!(is_sha256_digest(&manifest.source_package.sha256));
        assert!(is_sha256_digest(&manifest.inputs_sha256));
        assert_eq!(manifest.build_summary.builder_sha, "7".repeat(40));
        assert_eq!(
            manifest.build_summary.nvcc_version.as_deref(),
            Some("fake nvcc 12.4")
        );
        assert!(is_sha256_digest(&receipt.source_build_receipt.sha256));
        assert!(is_sha256_digest(&receipt.source_build_plan.sha256));
        assert!(!receipt.source_build_logs.is_empty());
        assert!(is_sha256_digest(&receipt.source_archive_sha256));
        assert_eq!(receipt.source_archive_members.len(), 1);
        assert!(receipt
            .package_build_logs
            .contains(&receipt.source_archive_verification));
        assert_eq!(receipt.package_commands.len(), 2);
    }

    #[test]
    fn package_spec_rejects_legacy_self_reported_build_provenance() {
        let mut value = serde_json::to_value(package_spec(
            CudaNativeBuildUnit::Marlin.artifact_operator(),
            "operation.dense_linear",
            "provider.cuda.dense_linear.f16.marlin",
            CudaNativeBuildUnit::Marlin.required_exports(),
        ))
        .unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("inputs_sha256".to_string(), serde_json::json!(digest('9')));

        assert!(serde_json::from_value::<NativeOperatorPackageSpec>(value).is_err());
    }

    #[test]
    fn accepts_source_build_receipt_after_checkout_relocation() {
        let root = tempfile::tempdir().unwrap();
        let (source_root, catalog_path, abi_path) = fixture_files(root.path());
        let exports = CudaNativeBuildUnit::Marlin.required_exports();
        let (source_build_receipt, source_build_plan, _) = run_source_build_fixture(
            root.path(),
            &source_root,
            CudaNativeBuildUnit::Marlin.artifact_operator(),
            "relocated",
            exports,
        );
        let mut source_build: NativeOperatorSourceBuildReceipt =
            read_json(&source_build_receipt).unwrap();
        for command in &mut source_build.commands {
            command.working_directory = "/workspace/remote/ferrum-infer-rs".to_string();
        }
        write_json(&source_build_receipt, &source_build).unwrap();
        let spec = package_spec(
            CudaNativeBuildUnit::Marlin.artifact_operator(),
            "operation.dense_linear",
            "provider.cuda.dense_linear.f16.marlin",
            exports,
        );
        let spec_path = root.path().join("marlin.package.json");
        write_json(&spec_path, &spec).unwrap();
        let output_dir = root.path().join("packages/marlin");

        let receipt = package_native_operator(&NativeOperatorPackageRequest {
            spec_path,
            source_root,
            source_build_receipt_path: source_build_receipt,
            source_build_plan_path: source_build_plan,
            g03_catalog_path: catalog_path,
            abi_contract_path: abi_path,
            output_dir: output_dir.clone(),
            cc: PathBuf::from("/usr/bin/cc"),
            ar: PathBuf::from("/usr/bin/ar"),
        })
        .unwrap();

        assert!(is_sha256_digest(&receipt.source_build_receipt.sha256));
        assert!(output_dir.join("package.receipt.json").is_file());
    }

    #[test]
    fn rejects_package_that_overclaims_source_build_compute_capabilities() {
        let root = tempfile::tempdir().unwrap();
        let (source_root, _, _) = fixture_files(root.path());
        let exports = CudaNativeBuildUnit::Marlin.required_exports();
        let (source_build_receipt, _, _) = run_source_build_fixture(
            root.path(),
            &source_root,
            CudaNativeBuildUnit::Marlin.artifact_operator(),
            "overclaimed-sm",
            exports,
        );
        let source_build: NativeOperatorSourceBuildReceipt =
            read_json(&source_build_receipt).unwrap();
        let mut spec = package_spec(
            CudaNativeBuildUnit::Marlin.artifact_operator(),
            "operation.dense_linear",
            "provider.cuda.dense_linear.f16.marlin",
            exports,
        );
        spec.compute_capabilities.push("sm_90".to_string());

        let error = validate_source_build_for_package(&source_build, &spec).unwrap_err();

        assert!(error.to_string().contains("must exactly equal"));
    }

    #[test]
    fn rejects_non_pass_source_build_without_publishing_output() {
        let root = tempfile::tempdir().unwrap();
        let (source_root, catalog_path, abi_path) = fixture_files(root.path());
        let exports = CudaNativeBuildUnit::Marlin.required_exports();
        let (source_build_receipt, source_build_plan, _) = run_source_build_fixture(
            root.path(),
            &source_root,
            CudaNativeBuildUnit::Marlin.artifact_operator(),
            "marlin-plan",
            exports,
        );
        let mut source_build: NativeOperatorSourceBuildReceipt =
            read_json(&source_build_receipt).unwrap();
        source_build.status = NativeOperatorSourceBuildStatus::Plan;
        source_build.plan_only = true;
        write_json(&source_build_receipt, &source_build).unwrap();
        let spec = package_spec(
            CudaNativeBuildUnit::Marlin.artifact_operator(),
            "operation.dense_linear",
            "provider.cuda.dense_linear.f16.marlin",
            exports,
        );
        let spec_path = root.path().join("marlin.package.json");
        write_json(&spec_path, &spec).unwrap();
        let output_dir = root.path().join("packages/marlin");

        let error = package_native_operator(&NativeOperatorPackageRequest {
            spec_path,
            source_root,
            source_build_receipt_path: source_build_receipt,
            source_build_plan_path: source_build_plan,
            g03_catalog_path: catalog_path,
            abi_contract_path: abi_path,
            output_dir: output_dir.clone(),
            cc: PathBuf::from("/usr/bin/cc"),
            ar: PathBuf::from("/usr/bin/ar"),
        })
        .unwrap_err();

        assert!(error.to_string().contains("terminal PASS"));
        assert!(!output_dir.exists());
    }

    #[test]
    fn rejects_source_build_for_another_operator_without_publishing_output() {
        let root = tempfile::tempdir().unwrap();
        let (source_root, catalog_path, abi_path) = fixture_files(root.path());
        let exports = CudaNativeBuildUnit::Marlin.required_exports();
        let (source_build_receipt, source_build_plan, _) = run_source_build_fixture(
            root.path(),
            &source_root,
            CudaNativeBuildUnit::VllmMarlin.artifact_operator(),
            "wrong-operator",
            exports,
        );
        let spec = package_spec(
            CudaNativeBuildUnit::Marlin.artifact_operator(),
            "operation.dense_linear",
            "provider.cuda.dense_linear.f16.marlin",
            exports,
        );
        let spec_path = root.path().join("marlin.package.json");
        write_json(&spec_path, &spec).unwrap();
        let output_dir = root.path().join("packages/marlin");

        let error = package_native_operator(&NativeOperatorPackageRequest {
            spec_path,
            source_root,
            source_build_receipt_path: source_build_receipt,
            source_build_plan_path: source_build_plan,
            g03_catalog_path: catalog_path,
            abi_contract_path: abi_path,
            output_dir: output_dir.clone(),
            cc: PathBuf::from("/usr/bin/cc"),
            ar: PathBuf::from("/usr/bin/ar"),
        })
        .unwrap_err();

        assert!(error.to_string().contains("operator differs"));
        assert!(!output_dir.exists());
    }

    #[test]
    fn rejects_archive_tampered_after_source_build_without_publishing_output() {
        let root = tempfile::tempdir().unwrap();
        let (source_root, catalog_path, abi_path) = fixture_files(root.path());
        let exports = CudaNativeBuildUnit::Marlin.required_exports();
        let (source_build_receipt, source_build_plan, archive) = run_source_build_fixture(
            root.path(),
            &source_root,
            CudaNativeBuildUnit::Marlin.artifact_operator(),
            "tampered",
            exports,
        );
        fs::write(&archive, b"tampered after source build\n").unwrap();
        let spec = package_spec(
            CudaNativeBuildUnit::Marlin.artifact_operator(),
            "operation.dense_linear",
            "provider.cuda.dense_linear.f16.marlin",
            exports,
        );
        let spec_path = root.path().join("marlin.package.json");
        write_json(&spec_path, &spec).unwrap();
        let output_dir = root.path().join("packages/marlin");

        let error = package_native_operator(&NativeOperatorPackageRequest {
            spec_path,
            source_root,
            source_build_receipt_path: source_build_receipt,
            source_build_plan_path: source_build_plan,
            g03_catalog_path: catalog_path,
            abi_contract_path: abi_path,
            output_dir: output_dir.clone(),
            cc: PathBuf::from("/usr/bin/cc"),
            ar: PathBuf::from("/usr/bin/ar"),
        })
        .unwrap_err();

        assert!(error.to_string().contains("archive sha256 differs"));
        assert!(!output_dir.exists());
    }

    #[test]
    fn rejects_source_archive_member_hash_mismatch_with_updated_archive_pin() {
        let root = tempfile::tempdir().unwrap();
        let (source_root, catalog_path, abi_path) = fixture_files(root.path());
        let exports = CudaNativeBuildUnit::Marlin.required_exports();
        let (source_build_receipt, source_build_plan, archive) = run_source_build_fixture(
            root.path(),
            &source_root,
            CudaNativeBuildUnit::Marlin.artifact_operator(),
            "forged-member",
            exports,
        );
        let mut source_build: NativeOperatorSourceBuildReceipt =
            read_json(&source_build_receipt).unwrap();
        let object_path = PathBuf::from(
            source_build.commands[0]
                .object_file
                .as_deref()
                .expect("fixture object path"),
        );
        let replacement_source = root.path().join("replacement.c");
        fs::write(
            &replacement_source,
            "int marlin_cuda(void) { return 91; }\nint marlin_cuda_moe(void) { return 92; }\n",
        )
        .unwrap();
        assert!(Command::new("/usr/bin/cc")
            .args(["-c"])
            .arg(&replacement_source)
            .arg("-o")
            .arg(&object_path)
            .status()
            .unwrap()
            .success());
        fs::remove_file(&archive).unwrap();
        assert!(Command::new("/usr/bin/ar")
            .arg("rcs")
            .arg(&archive)
            .arg(&object_path)
            .status()
            .unwrap()
            .success());
        source_build.archive_sha256 = Some(sha256_file(&archive).unwrap());
        write_json(&source_build_receipt, &source_build).unwrap();

        let spec = package_spec(
            CudaNativeBuildUnit::Marlin.artifact_operator(),
            "operation.dense_linear",
            "provider.cuda.dense_linear.f16.marlin",
            exports,
        );
        let spec_path = root.path().join("marlin.package.json");
        write_json(&spec_path, &spec).unwrap();
        let output_dir = root.path().join("packages/marlin");

        let error = package_native_operator(&NativeOperatorPackageRequest {
            spec_path,
            source_root,
            source_build_receipt_path: source_build_receipt,
            source_build_plan_path: source_build_plan,
            g03_catalog_path: catalog_path,
            abi_contract_path: abi_path,
            output_dir: output_dir.clone(),
            cc: PathBuf::from("/usr/bin/cc"),
            ar: PathBuf::from("/usr/bin/ar"),
        })
        .unwrap_err();

        assert!(error.to_string().contains("member hash mismatch"));
        assert!(!output_dir.exists());
    }

    #[test]
    fn rejects_archive_missing_a_build_unit_export_without_publishing_output() {
        let root = tempfile::tempdir().unwrap();
        let (source_root, catalog_path, abi_path) = fixture_files(root.path());
        let (source_build_receipt, source_build_plan, _) = run_source_build_fixture(
            root.path(),
            &source_root,
            CudaNativeBuildUnit::Marlin.artifact_operator(),
            "marlin-incomplete",
            &["marlin_cuda"],
        );
        let spec = package_spec(
            CudaNativeBuildUnit::Marlin.artifact_operator(),
            "operation.dense_linear",
            "provider.cuda.dense_linear.f16.marlin",
            &["marlin_cuda", "marlin_cuda_moe"],
        );
        let spec_path = root.path().join("marlin.package.json");
        write_json(&spec_path, &spec).unwrap();
        let output_dir = root.path().join("packages/marlin");

        let error = package_native_operator(&NativeOperatorPackageRequest {
            spec_path,
            source_root,
            source_build_receipt_path: source_build_receipt,
            source_build_plan_path: source_build_plan,
            g03_catalog_path: catalog_path,
            abi_contract_path: abi_path,
            output_dir: output_dir.clone(),
            cc: PathBuf::from("/usr/bin/cc"),
            ar: PathBuf::from("/usr/bin/ar"),
        })
        .unwrap_err();

        assert!(matches!(
            error,
            NativeOperatorBuilderError::Resolve(
                ferrum_native_ops::NativeOperatorResolveError::ArtifactMissingExports { .. }
            )
        ));
        assert!(!output_dir.exists());
    }

    #[test]
    fn identical_inputs_produce_identical_archive_and_manifest_hashes() {
        let root = tempfile::tempdir().unwrap();
        let (source_root, catalog_path, abi_path) = fixture_files(root.path());
        let exports = ["marlin_cuda", "marlin_cuda_moe"];
        let (source_build_receipt, source_build_plan, _) = run_source_build_fixture(
            root.path(),
            &source_root,
            CudaNativeBuildUnit::Marlin.artifact_operator(),
            "marlin-deterministic",
            &exports,
        );
        let spec = package_spec(
            CudaNativeBuildUnit::Marlin.artifact_operator(),
            "operation.dense_linear",
            "provider.cuda.dense_linear.f16.marlin",
            &exports,
        );
        let spec_path = root.path().join("marlin.package.json");
        write_json(&spec_path, &spec).unwrap();
        let package = |name: &str| {
            package_native_operator(&NativeOperatorPackageRequest {
                spec_path: spec_path.clone(),
                source_root: source_root.clone(),
                source_build_receipt_path: source_build_receipt.clone(),
                source_build_plan_path: source_build_plan.clone(),
                g03_catalog_path: catalog_path.clone(),
                abi_contract_path: abi_path.clone(),
                output_dir: root.path().join("packages").join(name),
                cc: PathBuf::from("/usr/bin/cc"),
                ar: PathBuf::from("/usr/bin/ar"),
            })
            .unwrap()
        };

        let first = package("first");
        let second = package("second");

        assert_eq!(first.binary_sha256, second.binary_sha256);
        assert_eq!(first.manifest_sha256, second.manifest_sha256);
        assert_eq!(first.descriptor_export, second.descriptor_export);
    }

    #[test]
    fn assembles_multiple_verified_packages_into_one_resolvable_lock() {
        let root = tempfile::tempdir().unwrap();
        let marlin_exports = CudaNativeBuildUnit::Marlin.required_exports();
        let (alpha_dir, _) = package_fixture(
            root.path(),
            CudaNativeBuildUnit::Marlin.artifact_operator(),
            "operation.dense_linear",
            "provider.cuda.dense_linear.f16.marlin",
            marlin_exports,
            "alpha",
        )
        .unwrap();
        let vllm_marlin_exports = CudaNativeBuildUnit::VllmMarlin.required_exports();
        let (beta_dir, _) = package_fixture(
            root.path(),
            CudaNativeBuildUnit::VllmMarlin.artifact_operator(),
            "operation.quantized_linear",
            "provider.cuda.quantized_linear.gptq_marlin",
            vllm_marlin_exports,
            "beta",
        )
        .unwrap();
        let lock_path = root.path().join("packages/native-operators.lock.json");

        let lock = assemble_native_operator_set(&NativeOperatorSetRequest {
            receipt_paths: vec![
                alpha_dir.join("package.receipt.json"),
                beta_dir.join("package.receipt.json"),
            ],
            output_lock_path: lock_path.clone(),
            compute_capability: "sm_89".to_string(),
        })
        .unwrap();

        assert_eq!(lock.artifacts.len(), 2);
        let resolved =
            NativeOperatorArtifactSetLock::load_and_resolve(&lock_path, Some("sm_89")).unwrap();
        assert_eq!(resolved.artifacts.len(), 2);
        assert_eq!(
            resolved.artifacts[0].resolved.manifest.operator,
            CudaNativeBuildUnit::Marlin.artifact_operator()
        );
        assert_eq!(
            resolved.artifacts[1].resolved.manifest.operator,
            CudaNativeBuildUnit::VllmMarlin.artifact_operator()
        );
    }
}
