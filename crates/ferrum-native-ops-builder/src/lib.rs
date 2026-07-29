//! Isolated packaging and set assembly for source-built native operators.

use std::ffi::OsStr;
use std::fs;
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use ferrum_native_ops::{
    CudaNativeBuildUnit, NativeOperatorArtifactLock, NativeOperatorArtifactSetLock,
    NativeOperatorResolveRequest, NativeOperatorResolver, NativeOperatorSystemLibrary,
    NATIVE_OPERATOR_ARTIFACT_SET_SCHEMA_VERSION,
};
use ferrum_types::{
    is_sha256_digest, NativeOperatorBackend, NativeOperatorBinding, NativeOperatorBuildSummary,
    NativeOperatorLinkage, NativeOperatorManifest, NativeOperatorSourcePackage,
    FERRUM_NATIVE_OPERATOR_ABI_VERSION, NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tempfile::{Builder as TempBuilder, NamedTempFile};
use thiserror::Error;

pub const NATIVE_OPERATOR_PACKAGE_SPEC_SCHEMA_VERSION: u32 = 1;
pub const NATIVE_OPERATOR_PACKAGE_RECEIPT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorPackageSpec {
    pub schema_version: u32,
    pub operator: String,
    pub operator_abi_version: String,
    pub backend: NativeOperatorBackend,
    pub compute_capabilities: Vec<String>,
    pub source_package: NativeOperatorSourcePackage,
    pub inputs_sha256: String,
    pub operation_bindings: Vec<NativeOperatorBinding>,
    pub required_exports: Vec<String>,
    pub license_files: Vec<NativeOperatorLicenseInput>,
    pub cuda_toolkit: Option<String>,
    pub cuda_runtime_min: Option<String>,
    pub build_summary: NativeOperatorBuildSummary,
    pub system_libraries: Vec<NativeOperatorSystemLibrary>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorLicenseInput {
    pub source_path: String,
    pub output_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorPackageReceipt {
    pub schema_version: u32,
    pub operator: String,
    pub manifest_file: String,
    pub artifact_file: String,
    pub manifest_sha256: String,
    pub binary_sha256: String,
    pub g03_catalog_sha256: String,
    pub abi_contract_sha256: String,
    pub descriptor_export: String,
    pub system_libraries: Vec<NativeOperatorSystemLibrary>,
}

#[derive(Debug, Clone)]
pub struct NativeOperatorPackageRequest {
    pub spec_path: PathBuf,
    pub source_root: PathBuf,
    pub input_archive: PathBuf,
    pub g03_catalog_path: PathBuf,
    pub abi_contract_path: PathBuf,
    pub output_dir: PathBuf,
    pub cc: String,
    pub ar: String,
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
}

pub type Result<T> = std::result::Result<T, NativeOperatorBuilderError>;

pub fn package_native_operator(
    request: &NativeOperatorPackageRequest,
) -> Result<NativeOperatorPackageReceipt> {
    require_file(&request.spec_path)?;
    require_file(&request.input_archive)?;
    require_file(&request.g03_catalog_path)?;
    require_file(&request.abi_contract_path)?;
    if request.output_dir.exists() {
        return Err(NativeOperatorBuilderError::OutputExists(
            request.output_dir.clone(),
        ));
    }
    if request.input_archive.extension() != Some(OsStr::new("a")) {
        return Err(NativeOperatorBuilderError::Invalid(
            "package input must be a static .a archive".to_string(),
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
    let artifact_path = staging.path().join(&artifact_file);
    fs::copy(&request.input_archive, &artifact_path).map_err(|source| {
        NativeOperatorBuilderError::Io {
            path: artifact_path.clone(),
            source,
        }
    })?;

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
    run_tool(
        &request.cc,
        [
            OsStr::new("-std=c11"),
            OsStr::new("-O2"),
            OsStr::new("-fno-ident"),
            OsStr::new("-fvisibility=hidden"),
            OsStr::new("-c"),
            descriptor_source.as_os_str(),
            OsStr::new("-o"),
            descriptor_object.as_os_str(),
        ],
    )?;
    run_tool(
        &request.ar,
        [
            OsStr::new("rcs"),
            artifact_path.as_os_str(),
            descriptor_object.as_os_str(),
        ],
    )?;

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
        source_package: spec.source_package.clone(),
        inputs_sha256: spec.inputs_sha256.clone(),
        binary_sha256: binary_sha256.clone(),
        linkage: NativeOperatorLinkage::Static,
        g03_catalog_sha256: Some(g03_catalog_sha256.clone()),
        abi_contract_sha256: Some(abi_contract_sha256.clone()),
        descriptor_export: Some(descriptor_export.clone()),
        operation_bindings: spec.operation_bindings.clone(),
        exports: exports.clone(),
        license_files,
        build_summary: spec.build_summary.clone(),
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
        .with_compute_capability(spec.compute_capabilities[0].clone())
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
        manifest_file,
        artifact_file,
        manifest_sha256,
        binary_sha256,
        g03_catalog_sha256,
        abi_contract_sha256,
        descriptor_export,
        system_libraries: spec.system_libraries,
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
        let manifest_path = resolve_relative_file(package_root, &receipt.manifest_file)?;
        let artifact_path = resolve_relative_file(package_root, &receipt.artifact_file)?;
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
    if !is_sha256_digest(&spec.inputs_sha256) {
        return Err(NativeOperatorBuilderError::Invalid(
            "inputs_sha256 must be a lowercase sha256 digest".to_string(),
        ));
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
    validate_relative_path(&receipt.manifest_file)?;
    validate_relative_path(&receipt.artifact_file)?;
    for (field, digest) in [
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
    Ok(())
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
    fs::write(path, bytes).map_err(|source| NativeOperatorBuilderError::Io {
        path: path.to_path_buf(),
        source,
    })
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

fn run_tool<I, S>(tool: &str, args: I) -> Result<()>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let output = Command::new(tool)
        .args(args)
        .env("ZERO_AR_DATE", "1")
        .output()
        .map_err(|source| NativeOperatorBuilderError::Io {
            path: PathBuf::from(tool),
            source,
        })?;
    if !output.status.success() {
        return Err(NativeOperatorBuilderError::Tool {
            tool: tool.to_string(),
            status: output.status.to_string(),
            stderr: String::from_utf8_lossy(&output.stderr)
                .trim()
                .chars()
                .take(2000)
                .collect(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_native_ops::NativeOperatorArtifactSetLock;

    fn digest(character: char) -> String {
        std::iter::repeat(character).take(64).collect()
    }

    fn write_archive(root: &Path, name: &str, exports: &[&str]) -> PathBuf {
        let source_path = root.join(format!("{name}.c"));
        let object_path = root.join(format!("{name}.o"));
        let archive_path = root.join(format!("lib{name}.a"));
        let source = exports
            .iter()
            .enumerate()
            .map(|(index, export)| format!("int {export}(void) {{ return {index}; }}\n"))
            .collect::<String>();
        fs::write(&source_path, source).unwrap();
        run_tool(
            "cc",
            [
                OsStr::new("-c"),
                source_path.as_os_str(),
                OsStr::new("-o"),
                object_path.as_os_str(),
            ],
        )
        .unwrap();
        run_tool(
            "ar",
            [
                OsStr::new("rcs"),
                archive_path.as_os_str(),
                object_path.as_os_str(),
            ],
        )
        .unwrap();
        archive_path
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
            source_package: NativeOperatorSourcePackage {
                kind: "locked-source-archive".to_string(),
                revision: "fixture".to_string(),
                sha256: digest('a'),
            },
            inputs_sha256: digest('b'),
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
            build_summary: NativeOperatorBuildSummary {
                builder_sha: "7".repeat(40),
                elapsed_ms: 10,
                nvcc_version: Some("12.4".to_string()),
                host_compiler: "cc".to_string(),
            },
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
        let archive = write_archive(root, output_name, exports);
        let spec = package_spec(operator, operation_id, provider_id, exports);
        let spec_path = root.join(format!("{output_name}.package.json"));
        write_json(&spec_path, &spec)?;
        let output_dir = root.join("packages").join(output_name);
        let receipt = package_native_operator(&NativeOperatorPackageRequest {
            spec_path,
            source_root,
            input_archive: archive,
            g03_catalog_path: catalog_path,
            abi_contract_path: abi_path,
            output_dir: output_dir.clone(),
            cc: "cc".to_string(),
            ar: "ar".to_string(),
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
    }

    #[test]
    fn rejects_archive_missing_a_build_unit_export_without_publishing_output() {
        let root = tempfile::tempdir().unwrap();
        let (source_root, catalog_path, abi_path) = fixture_files(root.path());
        let archive = write_archive(root.path(), "marlin-incomplete", &["marlin_cuda"]);
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
            input_archive: archive,
            g03_catalog_path: catalog_path,
            abi_contract_path: abi_path,
            output_dir: output_dir.clone(),
            cc: "cc".to_string(),
            ar: "ar".to_string(),
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
        let archive = write_archive(root.path(), "marlin-deterministic", &exports);
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
                input_archive: archive.clone(),
                g03_catalog_path: catalog_path.clone(),
                abi_contract_path: abi_path.clone(),
                output_dir: root.path().join("packages").join(name),
                cc: "cc".to_string(),
                ar: "ar".to_string(),
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
        let (alpha_dir, _) = package_fixture(
            root.path(),
            "ferrum.cuda.fixture_alpha",
            "operation.fixture_alpha",
            "provider.cuda.fixture_alpha",
            &["ferrum_fixture_alpha"],
            "alpha",
        )
        .unwrap();
        let (beta_dir, _) = package_fixture(
            root.path(),
            "ferrum.cuda.fixture_beta",
            "operation.fixture_beta",
            "provider.cuda.fixture_beta",
            &["ferrum_fixture_beta"],
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
            "ferrum.cuda.fixture_alpha"
        );
        assert_eq!(
            resolved.artifacts[1].resolved.manifest.operator,
            "ferrum.cuda.fixture_beta"
        );
    }
}
