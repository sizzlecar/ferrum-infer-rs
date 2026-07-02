//! Fail-closed resolver for native operator artifacts.

use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use ferrum_types::{NativeOperatorBackend, NativeOperatorManifest};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::abi::{FERRUM_NATIVE_ABI_VERSION, FERRUM_NATIVE_OP_INIT_SYMBOL};
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
        Ok(ResolvedNativeOperator {
            manifest,
            manifest_path: request.manifest_path.clone(),
            artifact_path: request.artifact_path.clone(),
            artifact_sha256,
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

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_types::{
        NativeOperatorBuildSummary, NativeOperatorLinkage, NativeOperatorSourcePackage,
        NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION,
    };
    use std::time::{SystemTime, UNIX_EPOCH};

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
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("ferrum-native-ops-{name}-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        TestDir(dir)
    }

    fn write_manifest(path: &Path, binary_sha256: String, abi: &str, caps: Vec<String>) {
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
            exports: vec![FERRUM_NATIVE_OP_INIT_SYMBOL.to_string()],
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

    fn fixture() -> TestFixture {
        let dir = temp_dir("resolver");
        let artifact = dir.path().join("libferrum_native_dummy.a");
        let bytes = b"dummy native operator artifact";
        fs::write(&artifact, bytes).unwrap();
        let manifest = dir.path().join("native_operator_manifest.json");
        write_manifest(
            &manifest,
            digest_bytes(bytes),
            FERRUM_NATIVE_ABI_VERSION,
            vec!["sm_89".to_string()],
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
            digest_bytes(b"dummy native operator artifact"),
            "999",
            vec!["sm_89".to_string()],
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
            digest_bytes(b"dummy native operator artifact"),
            FERRUM_NATIVE_ABI_VERSION,
            vec!["sm_80".to_string()],
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
}
