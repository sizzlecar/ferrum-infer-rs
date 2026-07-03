//! Fail-closed resolver for native operator artifacts.

use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::Command;

use ferrum_types::{NativeOperatorBackend, NativeOperatorLinkage, NativeOperatorManifest};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::abi::{
    FERRUM_NATIVE_ABI_VERSION, FERRUM_NATIVE_OP_DESCRIPTOR_SYMBOL, FERRUM_NATIVE_OP_INIT_SYMBOL,
};
use crate::manifest::load_manifest;

pub type Result<T> = std::result::Result<T, NativeOperatorResolveError>;

#[derive(Debug, Error)]
pub enum NativeOperatorResolveError {
    #[error("native operator manifest does not exist: {0}")]
    ManifestMissing(PathBuf),
    #[error("native operator artifact does not exist: {0}")]
    ArtifactMissing(PathBuf),
    #[error("native operator artifact must not be a Python wheel: {0}")]
    PythonWheelArtifact(PathBuf),
    #[error("failed to read native operator manifest {path}: {source}")]
    ManifestRead { path: PathBuf, source: io::Error },
    #[error("failed to parse native operator manifest {path}: {source}")]
    ManifestJson {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("invalid native operator manifest: {0}")]
    ManifestInvalid(String),
    #[error("native operator mismatch: expected {expected}, got {actual}")]
    OperatorMismatch { expected: String, actual: String },
    #[error("native operator backend mismatch: expected {expected:?}, got {actual:?}")]
    BackendMismatch {
        expected: NativeOperatorBackend,
        actual: NativeOperatorBackend,
    },
    #[error("native operator ABI mismatch: expected {expected}, got {actual}")]
    AbiMismatch { expected: String, actual: String },
    #[error("native operator compute capability mismatch: expected {expected}")]
    ComputeCapabilityMismatch { expected: String },
    #[error("failed to read native operator artifact {path}: {source}")]
    ArtifactRead { path: PathBuf, source: io::Error },
    #[error("native operator artifact sha256 mismatch: expected {expected}, got {actual}")]
    ArtifactSha256Mismatch { expected: String, actual: String },
    #[error("native operator artifact suffix mismatch for {path}: linkage={linkage:?}")]
    ArtifactSuffixMismatch {
        path: PathBuf,
        linkage: NativeOperatorLinkage,
    },
    #[error("native operator static archive is empty: {0}")]
    ArtifactArchiveEmpty(PathBuf),
    #[error(
        "native operator artifact tool failed: tool={tool} path={path} status={status} stderr={stderr}"
    )]
    ArtifactToolFailed {
        tool: String,
        path: PathBuf,
        status: String,
        stderr: String,
    },
    #[error("native operator artifact missing required exports in {path}: {missing:?}")]
    ArtifactMissingExports { path: PathBuf, missing: Vec<String> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeOperatorResolveRequest {
    pub operator: String,
    pub backend: NativeOperatorBackend,
    pub compute_capability: Option<String>,
    pub manifest_path: PathBuf,
    pub artifact_path: PathBuf,
    pub ferrum_native_abi_version: String,
}

impl NativeOperatorResolveRequest {
    pub fn new(
        operator: impl Into<String>,
        backend: NativeOperatorBackend,
        manifest_path: impl Into<PathBuf>,
        artifact_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            operator: operator.into(),
            backend,
            compute_capability: None,
            manifest_path: manifest_path.into(),
            artifact_path: artifact_path.into(),
            ferrum_native_abi_version: FERRUM_NATIVE_ABI_VERSION.to_string(),
        }
    }

    pub fn with_compute_capability(mut self, compute_capability: impl Into<String>) -> Self {
        self.compute_capability = Some(compute_capability.into());
        self
    }

    pub fn with_ferrum_native_abi_version(mut self, version: impl Into<String>) -> Self {
        self.ferrum_native_abi_version = version.into();
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedNativeOperator {
    pub manifest: NativeOperatorManifest,
    pub manifest_path: PathBuf,
    pub artifact_path: PathBuf,
    pub artifact_sha256: String,
    pub binary_validation: NativeOperatorBinaryValidation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeOperatorArtifactFormat {
    StaticArchive,
    DynamicLibrary,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeOperatorBinaryValidation {
    pub format: NativeOperatorArtifactFormat,
    pub archive_members: Vec<String>,
    pub required_exports: Vec<String>,
    pub matched_exports: Vec<String>,
}

#[derive(Debug, Default, Clone)]
pub struct NativeOperatorResolver;

impl NativeOperatorResolver {
    pub fn resolve(
        &self,
        request: &NativeOperatorResolveRequest,
    ) -> Result<ResolvedNativeOperator> {
        if !request.manifest_path.is_file() {
            return Err(NativeOperatorResolveError::ManifestMissing(
                request.manifest_path.clone(),
            ));
        }
        if request
            .artifact_path
            .extension()
            .is_some_and(|extension| extension == "whl")
        {
            return Err(NativeOperatorResolveError::PythonWheelArtifact(
                request.artifact_path.clone(),
            ));
        }
        if !request.artifact_path.is_file() {
            return Err(NativeOperatorResolveError::ArtifactMissing(
                request.artifact_path.clone(),
            ));
        }

        let manifest = load_manifest(&request.manifest_path)?;
        if manifest.operator != request.operator {
            return Err(NativeOperatorResolveError::OperatorMismatch {
                expected: request.operator.clone(),
                actual: manifest.operator.clone(),
            });
        }
        if manifest.backend != request.backend {
            return Err(NativeOperatorResolveError::BackendMismatch {
                expected: request.backend,
                actual: manifest.backend,
            });
        }
        if manifest.ferrum_native_abi_version != request.ferrum_native_abi_version {
            return Err(NativeOperatorResolveError::AbiMismatch {
                expected: request.ferrum_native_abi_version.clone(),
                actual: manifest.ferrum_native_abi_version.clone(),
            });
        }
        if !manifest
            .exports
            .iter()
            .any(|export| export == FERRUM_NATIVE_OP_INIT_SYMBOL)
        {
            return Err(NativeOperatorResolveError::ManifestInvalid(format!(
                "exports must include {FERRUM_NATIVE_OP_INIT_SYMBOL}"
            )));
        }
        if !manifest
            .exports
            .iter()
            .any(|export| export == FERRUM_NATIVE_OP_DESCRIPTOR_SYMBOL)
        {
            return Err(NativeOperatorResolveError::ManifestInvalid(format!(
                "exports must include {FERRUM_NATIVE_OP_DESCRIPTOR_SYMBOL}"
            )));
        }
        if let Some(expected) = &request.compute_capability {
            if !manifest
                .compute_capabilities
                .iter()
                .any(|capability| capability == expected)
            {
                return Err(NativeOperatorResolveError::ComputeCapabilityMismatch {
                    expected: expected.clone(),
                });
            }
        }

        let artifact_sha256 = file_sha256(&request.artifact_path)?;
        if artifact_sha256 != manifest.binary_sha256 {
            return Err(NativeOperatorResolveError::ArtifactSha256Mismatch {
                expected: manifest.binary_sha256.clone(),
                actual: artifact_sha256,
            });
        }
        let binary_validation =
            validate_binary_artifact(&request.artifact_path, manifest.linkage, &manifest.exports)?;
        Ok(ResolvedNativeOperator {
            manifest,
            manifest_path: request.manifest_path.clone(),
            artifact_path: request.artifact_path.clone(),
            artifact_sha256,
            binary_validation,
        })
    }
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut file =
        fs::File::open(path).map_err(|source| NativeOperatorResolveError::ArtifactRead {
            path: path.to_path_buf(),
            source,
        })?;
    let mut hasher = Sha256::new();
    let mut buf = [0_u8; 16 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|source| NativeOperatorResolveError::ArtifactRead {
                path: path.to_path_buf(),
                source,
            })?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_binary_artifact(
    path: &Path,
    linkage: NativeOperatorLinkage,
    exports: &[String],
) -> Result<NativeOperatorBinaryValidation> {
    let (format, archive_members) = match linkage {
        NativeOperatorLinkage::Static => {
            if path.extension().and_then(|extension| extension.to_str()) != Some("a") {
                return Err(NativeOperatorResolveError::ArtifactSuffixMismatch {
                    path: path.to_path_buf(),
                    linkage,
                });
            }
            let output = run_artifact_tool("ar", &["t"], path)?;
            let members = output
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>();
            if members.is_empty() {
                return Err(NativeOperatorResolveError::ArtifactArchiveEmpty(
                    path.to_path_buf(),
                ));
            }
            (NativeOperatorArtifactFormat::StaticArchive, members)
        }
        NativeOperatorLinkage::Dynamic => {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("");
            let suffix_ok = path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension == "dylib" || extension == "so")
                || name.contains(".so.");
            if !suffix_ok {
                return Err(NativeOperatorResolveError::ArtifactSuffixMismatch {
                    path: path.to_path_buf(),
                    linkage,
                });
            }
            (NativeOperatorArtifactFormat::DynamicLibrary, Vec::new())
        }
    };

    let nm_output = run_artifact_tool("nm", &["-g"], path)?;
    let defined_symbols = collect_defined_symbols(&nm_output);
    let missing = exports
        .iter()
        .filter(|export| !defined_symbols.contains(export.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(NativeOperatorResolveError::ArtifactMissingExports {
            path: path.to_path_buf(),
            missing,
        });
    }

    Ok(NativeOperatorBinaryValidation {
        format,
        archive_members,
        required_exports: exports.to_vec(),
        matched_exports: exports.to_vec(),
    })
}

fn run_artifact_tool(program: &str, args: &[&str], path: &Path) -> Result<String> {
    let output = Command::new(program)
        .args(args)
        .arg(path)
        .output()
        .map_err(|source| NativeOperatorResolveError::ArtifactRead {
            path: path.to_path_buf(),
            source,
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr)
            .trim()
            .chars()
            .take(1000)
            .collect::<String>();
        return Err(NativeOperatorResolveError::ArtifactToolFailed {
            tool: program.to_string(),
            path: path.to_path_buf(),
            status: output.status.to_string(),
            stderr,
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn collect_defined_symbols(nm_output: &str) -> std::collections::BTreeSet<String> {
    let mut symbols = std::collections::BTreeSet::new();
    for raw_line in nm_output.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.ends_with(':') {
            continue;
        }
        let parts = line.split_whitespace().collect::<Vec<_>>();
        if parts.len() < 2 {
            continue;
        }
        let mut symbol_type = None;
        let mut symbol = None;
        if parts.len() >= 3 && parts[parts.len() - 2].len() == 1 {
            symbol_type = parts.get(parts.len() - 2).copied();
            symbol = parts.last().copied();
        } else if parts[0].len() == 1 {
            symbol_type = parts.first().copied();
            symbol = parts.last().copied();
        }
        let (Some(symbol_type), Some(symbol)) = (symbol_type, symbol) else {
            continue;
        };
        if symbol_type.eq_ignore_ascii_case("u") {
            continue;
        }
        symbols.insert(symbol.to_string());
        if let Some(stripped) = symbol.strip_prefix('_') {
            symbols.insert(stripped.to_string());
        }
    }
    symbols
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_types::{
        NativeOperatorBuildSummary, NativeOperatorLinkage, NativeOperatorSourcePackage,
        NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION,
    };
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn digest_bytes(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn digest(ch: char) -> String {
        std::iter::repeat(ch).take(64).collect()
    }

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

    struct TestFixture {
        _dir: TestDir,
        manifest: PathBuf,
        artifact: PathBuf,
    }

    fn temp_dir(name: &str) -> TestDir {
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "ferrum-native-ops-{name}-{}-{counter}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        TestDir(dir)
    }

    fn write_manifest(
        path: &Path,
        binary_sha256: String,
        abi: &str,
        caps: Vec<String>,
        exports: Vec<String>,
    ) {
        let manifest = NativeOperatorManifest {
            schema_version: NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION,
            operator: "dummy".to_string(),
            operator_abi_version: "1".to_string(),
            ferrum_native_abi_version: abi.to_string(),
            backend: NativeOperatorBackend::Cuda,
            cuda_toolkit: Some("12.4".to_string()),
            cuda_runtime_min: Some("12.4".to_string()),
            compute_capabilities: caps,
            source_package: NativeOperatorSourcePackage {
                kind: "external_archive".to_string(),
                revision: "fixture".to_string(),
                sha256: digest('a'),
            },
            inputs_sha256: digest('b'),
            binary_sha256,
            linkage: NativeOperatorLinkage::Static,
            exports,
            license_files: vec!["LICENSE".to_string()],
            build_summary: NativeOperatorBuildSummary {
                builder_sha: "fixture-builder".to_string(),
                elapsed_ms: 1,
                nvcc_version: Some("12.4".to_string()),
                host_compiler: "clang".to_string(),
            },
        };
        fs::write(path, serde_json::to_string_pretty(&manifest).unwrap()).unwrap();
    }

    fn required_exports() -> Vec<String> {
        vec![
            FERRUM_NATIVE_OP_INIT_SYMBOL.to_string(),
            FERRUM_NATIVE_OP_DESCRIPTOR_SYMBOL.to_string(),
        ]
    }

    fn write_static_archive(dir: &Path, include_descriptor: bool) -> PathBuf {
        let source = dir.join("native_op.c");
        let mut source_text = String::from("int ferrum_native_op_init(void) { return 0; }\n");
        if include_descriptor {
            source_text
                .push_str("const char *ferrum_native_op_descriptor(void) { return \"dummy\"; }\n");
        }
        fs::write(&source, source_text).unwrap();
        let object = dir.join("native_op.o");
        let archive = dir.join("libferrum_native_dummy.a");
        let cc_status = Command::new("cc")
            .arg("-c")
            .arg(&source)
            .arg("-o")
            .arg(&object)
            .status()
            .unwrap();
        assert!(cc_status.success());
        let ar_status = Command::new("ar")
            .arg("rcs")
            .arg(&archive)
            .arg(&object)
            .status()
            .unwrap();
        assert!(ar_status.success());
        archive
    }

    fn fixture() -> TestFixture {
        let dir = temp_dir("resolver");
        let artifact = write_static_archive(dir.path(), true);
        let bytes = fs::read(&artifact).unwrap();
        let manifest = dir.path().join("native_operator_manifest.json");
        write_manifest(
            &manifest,
            digest_bytes(&bytes),
            FERRUM_NATIVE_ABI_VERSION,
            vec!["sm_89".to_string()],
            required_exports(),
        );
        TestFixture {
            _dir: dir,
            manifest,
            artifact,
        }
    }

    fn request(manifest: &Path, artifact: &Path) -> NativeOperatorResolveRequest {
        NativeOperatorResolveRequest::new("dummy", NativeOperatorBackend::Cuda, manifest, artifact)
            .with_compute_capability("sm_89")
    }

    #[test]
    fn resolves_matching_manifest_and_artifact() {
        let fixture = fixture();
        let resolved = NativeOperatorResolver
            .resolve(&request(&fixture.manifest, &fixture.artifact))
            .unwrap();
        assert_eq!(resolved.manifest.operator, "dummy");
        assert_eq!(resolved.artifact_path, fixture.artifact);
        assert_eq!(
            resolved.binary_validation.format,
            NativeOperatorArtifactFormat::StaticArchive
        );
        assert_eq!(
            resolved.binary_validation.required_exports,
            required_exports()
        );
    }

    #[test]
    fn fails_closed_for_missing_manifest() {
        let fixture = fixture();
        let err = NativeOperatorResolver
            .resolve(&request(
                &fixture.manifest.with_file_name("missing.json"),
                &fixture.artifact,
            ))
            .unwrap_err();
        assert!(matches!(
            err,
            NativeOperatorResolveError::ManifestMissing(_)
        ));
    }

    #[test]
    fn fails_closed_for_hash_mismatch() {
        let fixture = fixture();
        fs::write(&fixture.artifact, b"changed").unwrap();
        let err = NativeOperatorResolver
            .resolve(&request(&fixture.manifest, &fixture.artifact))
            .unwrap_err();
        assert!(
            matches!(
                err,
                NativeOperatorResolveError::ArtifactSha256Mismatch { .. }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn fails_closed_for_abi_mismatch() {
        let fixture = fixture();

        let abi_manifest = fixture._dir.path().join("abi_mismatch.json");
        write_manifest(
            &abi_manifest,
            digest_bytes(&fs::read(&fixture.artifact).unwrap()),
            "999",
            vec!["sm_89".to_string()],
            required_exports(),
        );
        let err = NativeOperatorResolver
            .resolve(&request(&abi_manifest, &fixture.artifact))
            .unwrap_err();
        assert!(
            matches!(err, NativeOperatorResolveError::AbiMismatch { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn fails_closed_for_compute_capability_mismatch() {
        let fixture = fixture();

        let cap_manifest = fixture._dir.path().join("cap_mismatch.json");
        write_manifest(
            &cap_manifest,
            digest_bytes(&fs::read(&fixture.artifact).unwrap()),
            FERRUM_NATIVE_ABI_VERSION,
            vec!["sm_80".to_string()],
            required_exports(),
        );
        let err = NativeOperatorResolver
            .resolve(&request(&cap_manifest, &fixture.artifact))
            .unwrap_err();
        assert!(
            matches!(
                err,
                NativeOperatorResolveError::ComputeCapabilityMismatch { .. }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn rejects_python_wheel_artifacts() {
        let fixture = fixture();
        let wheel = fixture._dir.path().join("native_op.whl");
        fs::write(&wheel, b"not allowed").unwrap();
        let err = NativeOperatorResolver
            .resolve(&request(&fixture.manifest, &wheel))
            .unwrap_err();
        assert!(matches!(
            err,
            NativeOperatorResolveError::PythonWheelArtifact(_)
        ));
    }

    #[test]
    fn rejects_text_file_even_when_hash_matches() {
        let dir = temp_dir("text-artifact");
        let artifact = dir.path().join("libferrum_native_dummy.a");
        let bytes = b"not an archive";
        fs::write(&artifact, bytes).unwrap();
        let manifest = dir.path().join("native_operator_manifest.json");
        write_manifest(
            &manifest,
            digest_bytes(bytes),
            FERRUM_NATIVE_ABI_VERSION,
            vec!["sm_89".to_string()],
            required_exports(),
        );

        let err = NativeOperatorResolver
            .resolve(&request(&manifest, &artifact))
            .unwrap_err();
        assert!(
            matches!(err, NativeOperatorResolveError::ArtifactToolFailed { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn rejects_archive_missing_declared_export() {
        let dir = temp_dir("missing-export");
        let artifact = write_static_archive(dir.path(), false);
        let bytes = fs::read(&artifact).unwrap();
        let manifest = dir.path().join("native_operator_manifest.json");
        write_manifest(
            &manifest,
            digest_bytes(&bytes),
            FERRUM_NATIVE_ABI_VERSION,
            vec!["sm_89".to_string()],
            required_exports(),
        );

        let err = NativeOperatorResolver
            .resolve(&request(&manifest, &artifact))
            .unwrap_err();
        assert!(
            matches!(
                err,
                NativeOperatorResolveError::ArtifactMissingExports { .. }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn rejects_manifest_without_descriptor_export() {
        let fixture = fixture();
        let manifest = fixture._dir.path().join("missing_descriptor.json");
        write_manifest(
            &manifest,
            digest_bytes(&fs::read(&fixture.artifact).unwrap()),
            FERRUM_NATIVE_ABI_VERSION,
            vec!["sm_89".to_string()],
            vec![FERRUM_NATIVE_OP_INIT_SYMBOL.to_string()],
        );

        let err = NativeOperatorResolver
            .resolve(&request(&manifest, &fixture.artifact))
            .unwrap_err();
        assert!(
            matches!(err, NativeOperatorResolveError::ManifestInvalid(_)),
            "{err:?}"
        );
    }
}
