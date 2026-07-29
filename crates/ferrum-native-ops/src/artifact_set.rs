//! Deterministic, fail-closed resolution for a set of native operator artifacts.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

use ferrum_types::{
    is_sha256_digest, NativeOperatorBackend, NativeOperatorBinding, NativeOperatorLinkage,
    FERRUM_NATIVE_OPERATOR_ABI_VERSION, NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    NativeOperatorResolveError, NativeOperatorResolveRequest, NativeOperatorResolver,
    ResolvedNativeOperator,
};

pub const NATIVE_OPERATOR_ARTIFACT_SET_SCHEMA_VERSION: u32 = 3;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorArtifactSetLock {
    pub schema_version: u32,
    pub g03_catalog_sha256: String,
    pub artifacts: Vec<NativeOperatorArtifactLock>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorArtifactLock {
    pub operator: String,
    pub backend: NativeOperatorBackend,
    pub manifest_path: String,
    pub manifest: NativeOperatorEvidenceFile,
    pub artifact_path: String,
    pub operator_abi_version: String,
    pub ferrum_native_abi_version: String,
    pub source_package_sha256: String,
    pub inputs_sha256: String,
    pub package_spec: NativeOperatorEvidenceFile,
    pub source_build_receipt: NativeOperatorEvidenceFile,
    pub source_build_plan: NativeOperatorEvidenceFile,
    pub source_build_logs: Vec<NativeOperatorEvidenceFile>,
    pub source_archive_sha256: String,
    pub package_receipt: NativeOperatorEvidenceFile,
    pub package_build_logs: Vec<NativeOperatorEvidenceFile>,
    pub license_files: Vec<NativeOperatorEvidenceFile>,
    pub binary_sha256: String,
    pub abi_contract_sha256: String,
    pub descriptor_export: String,
    pub required_exports: Vec<String>,
    pub operation_bindings: Vec<NativeOperatorBinding>,
    #[serde(default)]
    pub system_libraries: Vec<NativeOperatorSystemLibrary>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NativeOperatorEvidenceFile {
    pub path: String,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Debug)]
pub struct ResolvedNativeOperatorArtifactSet {
    pub lock_path: PathBuf,
    pub g03_catalog_sha256: String,
    pub artifacts: Vec<ResolvedNativeOperatorArtifact>,
}

#[derive(Debug)]
pub struct ResolvedNativeOperatorArtifact {
    pub lock: NativeOperatorArtifactLock,
    pub resolved: ResolvedNativeOperator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeOperatorSystemLibrary {
    CudaDriver,
    CudaRuntime,
    Cublas,
    CublasLt,
    StdCxx,
}

#[derive(Debug, Error)]
pub enum NativeOperatorArtifactSetError {
    #[error("native operator artifact-set lock does not exist: {0}")]
    LockMissing(PathBuf),
    #[error("failed to read native operator artifact-set lock {path}: {source}")]
    LockRead { path: PathBuf, source: io::Error },
    #[error("failed to parse native operator artifact-set lock {path}: {source}")]
    LockJson {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("invalid native operator artifact-set lock: {0}")]
    LockInvalid(String),
    #[error("native operator artifact-set path escapes its lock directory: {0}")]
    PathEscape(String),
    #[error("native operator artifact resolution failed for {operator}: {source}")]
    ArtifactResolve {
        operator: String,
        #[source]
        source: NativeOperatorResolveError,
    },
    #[error(
        "native operator artifact-set pin mismatch for {operator}.{field}: expected={expected} actual={actual}"
    )]
    PinMismatch {
        operator: String,
        field: String,
        expected: String,
        actual: String,
    },
    #[error(
        "native operator artifact-set static symbol collision: symbol={symbol} first={first} second={second}"
    )]
    StaticSymbolCollision {
        symbol: String,
        first: String,
        second: String,
    },
    #[error(
        "native operator artifact-set link-name collision: link_name={link_name} first={first} second={second}"
    )]
    LinkNameCollision {
        link_name: String,
        first: String,
        second: String,
    },
    #[error(
        "native operator artifact-set operation/provider collision: operation={operation_id} provider={provider_id} first={first} second={second}"
    )]
    OperationProviderCollision {
        operation_id: String,
        provider_id: String,
        first: String,
        second: String,
    },
}

impl NativeOperatorArtifactSetLock {
    pub fn load_and_resolve(
        lock_path: impl AsRef<Path>,
        compute_capability: Option<&str>,
    ) -> Result<ResolvedNativeOperatorArtifactSet, NativeOperatorArtifactSetError> {
        let lock_path = lock_path.as_ref();
        if !lock_path.is_file() {
            return Err(NativeOperatorArtifactSetError::LockMissing(
                lock_path.to_path_buf(),
            ));
        }
        let raw = fs::read_to_string(lock_path).map_err(|source| {
            NativeOperatorArtifactSetError::LockRead {
                path: lock_path.to_path_buf(),
                source,
            }
        })?;
        let lock: Self = serde_json::from_str(&raw).map_err(|source| {
            NativeOperatorArtifactSetError::LockJson {
                path: lock_path.to_path_buf(),
                source,
            }
        })?;
        lock.resolve(lock_path, compute_capability)
    }

    pub fn resolve(
        &self,
        lock_path: impl AsRef<Path>,
        compute_capability: Option<&str>,
    ) -> Result<ResolvedNativeOperatorArtifactSet, NativeOperatorArtifactSetError> {
        self.validate()?;
        let lock_path = lock_path.as_ref();
        let root = lock_path.parent().unwrap_or_else(|| Path::new("."));
        let canonical_root =
            fs::canonicalize(root).map_err(|source| NativeOperatorArtifactSetError::LockRead {
                path: root.to_path_buf(),
                source,
            })?;

        let mut resolved_artifacts = Vec::with_capacity(self.artifacts.len());
        let mut link_names = BTreeMap::<String, String>::new();
        let mut strong_symbols = BTreeMap::<String, String>::new();
        let mut operation_providers = BTreeMap::<(String, String), String>::new();

        for artifact_lock in &self.artifacts {
            let manifest_path = resolve_locked_path(
                &canonical_root,
                &artifact_lock.manifest_path,
                &artifact_lock.operator,
            )?;
            let artifact_path = resolve_locked_path(
                &canonical_root,
                &artifact_lock.artifact_path,
                &artifact_lock.operator,
            )?;
            verify_evidence_file(
                &canonical_root,
                &artifact_lock.operator,
                "manifest",
                &artifact_lock.manifest,
            )?;
            verify_evidence_file(
                &canonical_root,
                &artifact_lock.operator,
                "package_spec",
                &artifact_lock.package_spec,
            )?;
            verify_evidence_file(
                &canonical_root,
                &artifact_lock.operator,
                "source_build_receipt",
                &artifact_lock.source_build_receipt,
            )?;
            verify_evidence_file(
                &canonical_root,
                &artifact_lock.operator,
                "source_build_plan",
                &artifact_lock.source_build_plan,
            )?;
            for evidence in &artifact_lock.source_build_logs {
                verify_evidence_file(
                    &canonical_root,
                    &artifact_lock.operator,
                    "source_build_log",
                    evidence,
                )?;
            }
            verify_evidence_file(
                &canonical_root,
                &artifact_lock.operator,
                "package_receipt",
                &artifact_lock.package_receipt,
            )?;
            for evidence in &artifact_lock.package_build_logs {
                verify_evidence_file(
                    &canonical_root,
                    &artifact_lock.operator,
                    "package_build_log",
                    evidence,
                )?;
            }
            for evidence in &artifact_lock.license_files {
                verify_evidence_file(
                    &canonical_root,
                    &artifact_lock.operator,
                    "license_file",
                    evidence,
                )?;
            }
            let mut request = NativeOperatorResolveRequest::new(
                artifact_lock.operator.clone(),
                artifact_lock.backend,
                manifest_path,
                artifact_path,
            )
            .with_operator_abi_version(artifact_lock.operator_abi_version.clone())
            .with_ferrum_native_abi_version(artifact_lock.ferrum_native_abi_version.clone())
            .with_g03_catalog_sha256(self.g03_catalog_sha256.clone())
            .with_abi_contract_sha256(artifact_lock.abi_contract_sha256.clone())
            .with_descriptor_export(artifact_lock.descriptor_export.clone())
            .with_required_exports(artifact_lock.required_exports.clone())
            .with_operation_bindings(artifact_lock.operation_bindings.clone());
            if let Some(compute_capability) = compute_capability {
                request = request.with_compute_capability(compute_capability);
            }
            let resolved = NativeOperatorResolver.resolve(&request).map_err(|source| {
                NativeOperatorArtifactSetError::ArtifactResolve {
                    operator: artifact_lock.operator.clone(),
                    source,
                }
            })?;
            if resolved.manifest.schema_version != NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION {
                return Err(NativeOperatorArtifactSetError::LockInvalid(format!(
                    "{} uses legacy manifest schema {}; artifact sets require schema {}",
                    artifact_lock.operator,
                    resolved.manifest.schema_version,
                    NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION
                )));
            }
            require_pin(
                artifact_lock,
                "source_package_sha256",
                &artifact_lock.source_package_sha256,
                &resolved.manifest.source_package.sha256,
            )?;
            require_pin(
                artifact_lock,
                "inputs_sha256",
                &artifact_lock.inputs_sha256,
                &resolved.manifest.inputs_sha256,
            )?;
            require_pin(
                artifact_lock,
                "binary_sha256",
                &artifact_lock.binary_sha256,
                &resolved.artifact_sha256,
            )?;

            let link_name =
                native_artifact_link_name(&resolved.artifact_path, resolved.manifest.linkage)
                    .map_err(NativeOperatorArtifactSetError::LockInvalid)?;
            if let Some(first) =
                link_names.insert(link_name.clone(), artifact_lock.operator.clone())
            {
                return Err(NativeOperatorArtifactSetError::LinkNameCollision {
                    link_name,
                    first,
                    second: artifact_lock.operator.clone(),
                });
            }
            if resolved.manifest.linkage == NativeOperatorLinkage::Static {
                for symbol in &resolved.binary_validation.strong_defined_symbols {
                    if let Some(first) =
                        strong_symbols.insert(symbol.clone(), artifact_lock.operator.clone())
                    {
                        if first != artifact_lock.operator {
                            return Err(NativeOperatorArtifactSetError::StaticSymbolCollision {
                                symbol: symbol.clone(),
                                first,
                                second: artifact_lock.operator.clone(),
                            });
                        }
                    }
                }
            }
            for binding in &resolved.manifest.operation_bindings {
                let key = (binding.operation_id.clone(), binding.provider_id.clone());
                if let Some(first) =
                    operation_providers.insert(key.clone(), artifact_lock.operator.clone())
                {
                    return Err(NativeOperatorArtifactSetError::OperationProviderCollision {
                        operation_id: key.0,
                        provider_id: key.1,
                        first,
                        second: artifact_lock.operator.clone(),
                    });
                }
            }
            resolved_artifacts.push(ResolvedNativeOperatorArtifact {
                lock: artifact_lock.clone(),
                resolved,
            });
        }

        Ok(ResolvedNativeOperatorArtifactSet {
            lock_path: lock_path.to_path_buf(),
            g03_catalog_sha256: self.g03_catalog_sha256.clone(),
            artifacts: resolved_artifacts,
        })
    }

    fn validate(&self) -> Result<(), NativeOperatorArtifactSetError> {
        if self.schema_version != NATIVE_OPERATOR_ARTIFACT_SET_SCHEMA_VERSION {
            return Err(NativeOperatorArtifactSetError::LockInvalid(format!(
                "schema_version must be {NATIVE_OPERATOR_ARTIFACT_SET_SCHEMA_VERSION}"
            )));
        }
        if !is_sha256_digest(&self.g03_catalog_sha256) {
            return Err(NativeOperatorArtifactSetError::LockInvalid(
                "g03_catalog_sha256 must be a lowercase hex sha256 digest".to_string(),
            ));
        }
        if self.artifacts.is_empty() {
            return Err(NativeOperatorArtifactSetError::LockInvalid(
                "artifacts must be non-empty".to_string(),
            ));
        }
        let mut previous: Option<&str> = None;
        for artifact in &self.artifacts {
            if artifact.operator.trim().is_empty() {
                return Err(NativeOperatorArtifactSetError::LockInvalid(
                    "artifact operator must be non-empty".to_string(),
                ));
            }
            if artifact.ferrum_native_abi_version != FERRUM_NATIVE_OPERATOR_ABI_VERSION {
                return Err(NativeOperatorArtifactSetError::LockInvalid(format!(
                    "{}.ferrum_native_abi_version must be {}",
                    artifact.operator, FERRUM_NATIVE_OPERATOR_ABI_VERSION
                )));
            }
            if previous.is_some_and(|value| value >= artifact.operator.as_str()) {
                return Err(NativeOperatorArtifactSetError::LockInvalid(
                    "artifacts must be sorted and unique by operator".to_string(),
                ));
            }
            previous = Some(&artifact.operator);
            for (field, digest) in [
                ("source_package_sha256", &artifact.source_package_sha256),
                ("inputs_sha256", &artifact.inputs_sha256),
                ("source_archive_sha256", &artifact.source_archive_sha256),
                ("binary_sha256", &artifact.binary_sha256),
                ("abi_contract_sha256", &artifact.abi_contract_sha256),
            ] {
                if !is_sha256_digest(digest) {
                    return Err(NativeOperatorArtifactSetError::LockInvalid(format!(
                        "{}.{field} must be a lowercase hex sha256 digest",
                        artifact.operator
                    )));
                }
            }
            validate_evidence_file(&artifact.operator, "manifest", &artifact.manifest)?;
            if artifact.manifest.path != artifact.manifest_path {
                return Err(NativeOperatorArtifactSetError::LockInvalid(format!(
                    "{}.manifest evidence path must equal manifest_path",
                    artifact.operator
                )));
            }
            validate_evidence_file(&artifact.operator, "package_spec", &artifact.package_spec)?;
            validate_evidence_file(
                &artifact.operator,
                "source_build_receipt",
                &artifact.source_build_receipt,
            )?;
            validate_evidence_file(
                &artifact.operator,
                "source_build_plan",
                &artifact.source_build_plan,
            )?;
            if artifact.source_build_logs.is_empty()
                || artifact
                    .source_build_logs
                    .windows(2)
                    .any(|pair| pair[0].path >= pair[1].path)
            {
                return Err(NativeOperatorArtifactSetError::LockInvalid(format!(
                    "{}.source_build_logs must be sorted, unique, and non-empty",
                    artifact.operator
                )));
            }
            for evidence in &artifact.source_build_logs {
                validate_evidence_file(&artifact.operator, "source_build_log", evidence)?;
            }
            validate_evidence_file(
                &artifact.operator,
                "package_receipt",
                &artifact.package_receipt,
            )?;
            if artifact.package_build_logs.is_empty()
                || artifact
                    .package_build_logs
                    .windows(2)
                    .any(|pair| pair[0].path >= pair[1].path)
            {
                return Err(NativeOperatorArtifactSetError::LockInvalid(format!(
                    "{}.package_build_logs must be sorted, unique, and non-empty",
                    artifact.operator
                )));
            }
            for evidence in &artifact.package_build_logs {
                validate_evidence_file(&artifact.operator, "package_build_log", evidence)?;
            }
            if artifact.license_files.is_empty()
                || artifact
                    .license_files
                    .windows(2)
                    .any(|pair| pair[0].path >= pair[1].path)
            {
                return Err(NativeOperatorArtifactSetError::LockInvalid(format!(
                    "{}.license_files must be sorted, unique, and non-empty",
                    artifact.operator
                )));
            }
            for evidence in &artifact.license_files {
                validate_evidence_file(&artifact.operator, "license_file", evidence)?;
            }
            if artifact.required_exports.is_empty() {
                return Err(NativeOperatorArtifactSetError::LockInvalid(format!(
                    "{}.required_exports must be non-empty",
                    artifact.operator
                )));
            }
            if artifact
                .required_exports
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            {
                return Err(NativeOperatorArtifactSetError::LockInvalid(format!(
                    "{}.required_exports must be sorted and unique",
                    artifact.operator
                )));
            }
            if artifact.operation_bindings.is_empty() {
                return Err(NativeOperatorArtifactSetError::LockInvalid(format!(
                    "{}.operation_bindings must be non-empty",
                    artifact.operator
                )));
            }
            if artifact
                .system_libraries
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            {
                return Err(NativeOperatorArtifactSetError::LockInvalid(format!(
                    "{}.system_libraries must be sorted and unique",
                    artifact.operator
                )));
            }
            validate_relative_path(&artifact.manifest_path)?;
            validate_relative_path(&artifact.artifact_path)?;
        }
        Ok(())
    }
}

pub fn native_artifact_link_name(
    path: &Path,
    linkage: NativeOperatorLinkage,
) -> Result<String, String> {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("native artifact has no UTF-8 file name: {}", path.display()))?;
    let link_name = match linkage {
        NativeOperatorLinkage::Static => name
            .strip_prefix("lib")
            .and_then(|value| value.strip_suffix(".a")),
        NativeOperatorLinkage::Dynamic => {
            if let Some(value) = name
                .strip_prefix("lib")
                .and_then(|value| value.strip_suffix(".dylib"))
            {
                Some(value)
            } else {
                name.strip_prefix("lib")
                    .and_then(|value| value.split_once(".so"))
                    .map(|(value, _)| value)
            }
        }
    }
    .filter(|value| !value.is_empty())
    .ok_or_else(|| {
        format!(
            "native artifact file name does not match {:?} linkage: {}",
            linkage,
            path.display()
        )
    })?;
    Ok(link_name.to_string())
}

fn validate_relative_path(path: &str) -> Result<(), NativeOperatorArtifactSetError> {
    let path = Path::new(path);
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::RootDir))
    {
        return Err(NativeOperatorArtifactSetError::PathEscape(
            path.display().to_string(),
        ));
    }
    Ok(())
}

fn resolve_locked_path(
    root: &Path,
    relative: &str,
    operator: &str,
) -> Result<PathBuf, NativeOperatorArtifactSetError> {
    validate_relative_path(relative)?;
    let path = root.join(relative);
    let canonical =
        fs::canonicalize(&path).map_err(|source| NativeOperatorArtifactSetError::LockRead {
            path: path.clone(),
            source,
        })?;
    if !canonical.starts_with(root) {
        return Err(NativeOperatorArtifactSetError::PathEscape(format!(
            "{operator}:{relative}"
        )));
    }
    Ok(canonical)
}

fn validate_evidence_file(
    operator: &str,
    field: &str,
    evidence: &NativeOperatorEvidenceFile,
) -> Result<(), NativeOperatorArtifactSetError> {
    validate_relative_path(&evidence.path)?;
    if !is_sha256_digest(&evidence.sha256) || evidence.size_bytes == 0 {
        return Err(NativeOperatorArtifactSetError::LockInvalid(format!(
            "{operator}.{field} must record a non-empty file and lowercase sha256"
        )));
    }
    Ok(())
}

fn verify_evidence_file(
    root: &Path,
    operator: &str,
    field: &str,
    evidence: &NativeOperatorEvidenceFile,
) -> Result<(), NativeOperatorArtifactSetError> {
    let path = resolve_locked_path(root, &evidence.path, operator)?;
    let bytes = fs::read(&path).map_err(|source| NativeOperatorArtifactSetError::LockRead {
        path: path.clone(),
        source,
    })?;
    let actual_sha256 = format!("{:x}", Sha256::digest(&bytes));
    let actual_size = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if actual_sha256 != evidence.sha256 {
        return Err(NativeOperatorArtifactSetError::PinMismatch {
            operator: operator.to_string(),
            field: format!("{field}.sha256"),
            expected: evidence.sha256.clone(),
            actual: actual_sha256,
        });
    }
    if actual_size != evidence.size_bytes {
        return Err(NativeOperatorArtifactSetError::PinMismatch {
            operator: operator.to_string(),
            field: format!("{field}.size_bytes"),
            expected: evidence.size_bytes.to_string(),
            actual: actual_size.to_string(),
        });
    }
    Ok(())
}

fn require_pin(
    artifact: &NativeOperatorArtifactLock,
    field: &str,
    expected: &str,
    actual: &str,
) -> Result<(), NativeOperatorArtifactSetError> {
    if expected == actual {
        Ok(())
    } else {
        Err(NativeOperatorArtifactSetError::PinMismatch {
            operator: artifact.operator.clone(),
            field: field.to_string(),
            expected: expected.to_string(),
            actual: actual.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use ferrum_types::{
        NativeOperatorBuildSummary, NativeOperatorManifest, NativeOperatorSourcePackage,
        FERRUM_NATIVE_OPERATOR_ABI_VERSION,
    };
    use sha2::{Digest, Sha256};

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TestDir(PathBuf);

    impl TestDir {
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn temp_dir(name: &str) -> TestDir {
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "ferrum-native-artifact-set-{name}-{}-{counter}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        TestDir(path)
    }

    fn digest_bytes(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn digest(ch: char) -> String {
        std::iter::repeat(ch).take(64).collect()
    }

    fn write_artifact(
        root: &Path,
        operator: &str,
        operation_id: &str,
        provider_id: &str,
        extra_strong_symbol: Option<&str>,
    ) -> NativeOperatorArtifactLock {
        let dir = root.join(operator);
        fs::create_dir_all(&dir).unwrap();
        let descriptor = format!("ferrum_native_{operator}_descriptor_v2");
        let execute = format!("ferrum_native_{operator}_execute_v1");
        let mut source = format!(
            "int {execute}(void) {{ return 0; }}\n\
             const char *{descriptor}(void) {{ return \"{operator}\"; }}\n"
        );
        if let Some(symbol) = extra_strong_symbol {
            source.push_str(&format!("int {symbol}(void) {{ return 1; }}\n"));
        }
        let source_path = dir.join("operator.c");
        let object_path = dir.join("operator.o");
        let artifact_path = dir.join(format!("libferrum_native_{operator}.a"));
        fs::write(&source_path, source).unwrap();
        assert!(Command::new("cc")
            .args(["-c"])
            .arg(&source_path)
            .arg("-o")
            .arg(&object_path)
            .status()
            .unwrap()
            .success());
        assert!(Command::new("ar")
            .arg("rcs")
            .arg(&artifact_path)
            .arg(&object_path)
            .status()
            .unwrap()
            .success());

        let binary_sha256 = digest_bytes(&fs::read(&artifact_path).unwrap());
        let source_package_sha256 = digest(if operator == "alpha" { 'a' } else { 'b' });
        let inputs_sha256 = digest(if operator == "alpha" { 'c' } else { 'd' });
        let abi_contract_sha256 = digest(if operator == "alpha" { 'e' } else { 'f' });
        let provider_fingerprint = digest(if operator == "alpha" { '1' } else { '2' });
        let mut exports = vec![descriptor.clone(), execute.clone()];
        if let Some(symbol) = extra_strong_symbol {
            exports.push(symbol.to_string());
            exports.sort();
        }
        let operation_bindings = vec![NativeOperatorBinding {
            operation_id: operation_id.to_string(),
            operation_contract_version: 1,
            provider_id: provider_id.to_string(),
            provider_version: 1,
            provider_implementation_fingerprint: provider_fingerprint,
            entrypoints: vec![execute],
        }];
        let manifest = NativeOperatorManifest {
            schema_version: NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION,
            operator: operator.to_string(),
            operator_abi_version: "1".to_string(),
            ferrum_native_abi_version: FERRUM_NATIVE_OPERATOR_ABI_VERSION.to_string(),
            backend: NativeOperatorBackend::Cuda,
            cuda_toolkit: Some("12.4".to_string()),
            cuda_runtime_min: Some("12.4".to_string()),
            compute_capabilities: vec!["sm_89".to_string()],
            source_package: NativeOperatorSourcePackage {
                kind: "external_archive".to_string(),
                revision: "fixture".to_string(),
                sha256: source_package_sha256.clone(),
            },
            inputs_sha256: inputs_sha256.clone(),
            binary_sha256: binary_sha256.clone(),
            linkage: NativeOperatorLinkage::Static,
            g03_catalog_sha256: Some(digest('9')),
            abi_contract_sha256: Some(abi_contract_sha256.clone()),
            descriptor_export: Some(descriptor.clone()),
            operation_bindings: operation_bindings.clone(),
            exports: exports.clone(),
            license_files: vec!["LICENSE".to_string()],
            build_summary: NativeOperatorBuildSummary {
                builder_sha: digest('7'),
                elapsed_ms: 1,
                nvcc_version: Some("12.4".to_string()),
                host_compiler: "cc".to_string(),
            },
        };
        let manifest_path = dir.join("native_operator_manifest.json");
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();
        let receipt_path = dir.join("source-build.receipt.json");
        let plan_path = dir.join("source-build.plan.json");
        let log_path = dir.join("source-build.log");
        let package_spec_path = dir.join("package.spec.json");
        let package_receipt_path = dir.join("package.receipt.json");
        let package_log_path = dir.join("package-build.log");
        let license_path = dir.join("LICENSE");
        fs::write(&receipt_path, "{\"status\":\"pass\"}\n").unwrap();
        fs::write(&plan_path, "{\"schema_version\":2}\n").unwrap();
        fs::write(&log_path, "source build complete\n").unwrap();
        fs::write(&package_spec_path, "{\"schema_version\":2}\n").unwrap();
        fs::write(&package_receipt_path, "{\"status\":\"pass\"}\n").unwrap();
        fs::write(&package_log_path, "package build complete\n").unwrap();
        fs::write(&license_path, "fixture license\n").unwrap();
        let evidence = |path: &Path| NativeOperatorEvidenceFile {
            path: format!("{operator}/{}", path.file_name().unwrap().to_string_lossy()),
            sha256: digest_bytes(&fs::read(path).unwrap()),
            size_bytes: fs::metadata(path).unwrap().len(),
        };
        NativeOperatorArtifactLock {
            operator: operator.to_string(),
            backend: NativeOperatorBackend::Cuda,
            manifest_path: format!("{operator}/native_operator_manifest.json"),
            manifest: evidence(&manifest_path),
            artifact_path: format!("{operator}/libferrum_native_{operator}.a"),
            operator_abi_version: "1".to_string(),
            ferrum_native_abi_version: FERRUM_NATIVE_OPERATOR_ABI_VERSION.to_string(),
            source_package_sha256,
            inputs_sha256,
            package_spec: evidence(&package_spec_path),
            source_build_receipt: evidence(&receipt_path),
            source_build_plan: evidence(&plan_path),
            source_build_logs: vec![evidence(&log_path)],
            source_archive_sha256: digest('8'),
            package_receipt: evidence(&package_receipt_path),
            package_build_logs: vec![evidence(&package_log_path)],
            license_files: vec![evidence(&license_path)],
            binary_sha256,
            abi_contract_sha256,
            descriptor_export: descriptor,
            required_exports: exports,
            operation_bindings,
            system_libraries: vec![
                NativeOperatorSystemLibrary::CudaRuntime,
                NativeOperatorSystemLibrary::StdCxx,
            ],
        }
    }

    #[test]
    fn resolves_multiple_schema_v3_artifacts_in_deterministic_order() {
        let dir = temp_dir("pass");
        let alpha = write_artifact(
            dir.path(),
            "alpha",
            "operation.alpha",
            "provider.cuda.alpha",
            None,
        );
        let beta = write_artifact(
            dir.path(),
            "beta",
            "operation.beta",
            "provider.cuda.beta",
            None,
        );
        let lock = NativeOperatorArtifactSetLock {
            schema_version: NATIVE_OPERATOR_ARTIFACT_SET_SCHEMA_VERSION,
            g03_catalog_sha256: digest('9'),
            artifacts: vec![alpha, beta],
        };
        let lock_path = dir.path().join("native-operators.lock.json");
        fs::write(&lock_path, serde_json::to_string_pretty(&lock).unwrap()).unwrap();

        let resolved =
            NativeOperatorArtifactSetLock::load_and_resolve(&lock_path, Some("sm_89")).unwrap();
        assert_eq!(resolved.artifacts.len(), 2);
        assert_eq!(resolved.artifacts[0].resolved.manifest.operator, "alpha");
        assert_eq!(resolved.artifacts[1].resolved.manifest.operator, "beta");
    }

    #[test]
    fn rejects_cross_artifact_strong_symbol_collision() {
        let dir = temp_dir("symbol-collision");
        let alpha = write_artifact(
            dir.path(),
            "alpha",
            "operation.alpha",
            "provider.cuda.alpha",
            Some("ferrum_native_shared_collision"),
        );
        let beta = write_artifact(
            dir.path(),
            "beta",
            "operation.beta",
            "provider.cuda.beta",
            Some("ferrum_native_shared_collision"),
        );
        let lock = NativeOperatorArtifactSetLock {
            schema_version: NATIVE_OPERATOR_ARTIFACT_SET_SCHEMA_VERSION,
            g03_catalog_sha256: digest('9'),
            artifacts: vec![alpha, beta],
        };
        let lock_path = dir.path().join("native-operators.lock.json");
        fs::write(&lock_path, serde_json::to_string_pretty(&lock).unwrap()).unwrap();

        let error =
            NativeOperatorArtifactSetLock::load_and_resolve(&lock_path, Some("sm_89")).unwrap_err();
        assert!(matches!(
            error,
            NativeOperatorArtifactSetError::StaticSymbolCollision { .. }
        ));
    }

    #[test]
    fn rejects_tampered_package_provenance_files() {
        for field in [
            "manifest",
            "package_spec",
            "package_receipt",
            "package_build_log",
            "license_file",
        ] {
            let dir = temp_dir(field);
            let artifact = write_artifact(
                dir.path(),
                "alpha",
                "operation.alpha",
                "provider.cuda.alpha",
                None,
            );
            let evidence = match field {
                "manifest" => &artifact.manifest,
                "package_spec" => &artifact.package_spec,
                "package_receipt" => &artifact.package_receipt,
                "package_build_log" => &artifact.package_build_logs[0],
                "license_file" => &artifact.license_files[0],
                _ => unreachable!(),
            };
            let evidence_path = dir.path().join(&evidence.path);
            fs::write(&evidence_path, format!("tampered {field}\n")).unwrap();
            let lock = NativeOperatorArtifactSetLock {
                schema_version: NATIVE_OPERATOR_ARTIFACT_SET_SCHEMA_VERSION,
                g03_catalog_sha256: digest('9'),
                artifacts: vec![artifact],
            };
            let lock_path = dir.path().join("native-operators.lock.json");
            fs::write(&lock_path, serde_json::to_string_pretty(&lock).unwrap()).unwrap();

            let error = NativeOperatorArtifactSetLock::load_and_resolve(&lock_path, Some("sm_89"))
                .unwrap_err();
            assert!(matches!(
                error,
                NativeOperatorArtifactSetError::PinMismatch { .. }
            ));
        }
    }
}
