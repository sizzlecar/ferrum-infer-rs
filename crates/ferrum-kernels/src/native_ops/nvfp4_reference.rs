//! Explicit qualification artifact identity. This never selects a production
//! implementation or loads a library during manifest validation.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::PathBuf};

pub const NVFP4_REFERENCE_PROVIDER_ID: &str = "provider.cuda.diagnostic.flashinfer_nvfp4.bf16.v1";
pub const NVFP4_REFERENCE_PROFILE: &str = "cuda_rcp_approx_ftz_v1";
const MAX_MANIFEST_BYTES: u64 = 256 * 1024;
const MAX_LIBRARY_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Nvfp4ReferenceSourceFile {
    pub path: String,
    pub sha256: String,
}

/// Written by the external qualification build. The caller pins this entire
/// document as well as the resulting shared library. Source files include the
/// compiler-observed transitive header closure; no vendored copy is necessary.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Nvfp4ReferenceManifest {
    pub abi_version: u32,
    pub profile: String,
    pub library_sha256: String,
    pub wrapper_sha256: String,
    pub source_identity: String,
    pub source_files: Vec<Nvfp4ReferenceSourceFile>,
    pub nvcc_sha256: String,
    pub host_compiler_sha256: String,
    pub nvcc_version: String,
    pub host_compiler_version: String,
    pub compiler_arguments: Vec<String>,
    pub use_fast_math: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CudaNvfp4ReferenceConfig {
    pub library: PathBuf,
    pub library_sha256: String,
    pub manifest: PathBuf,
    pub manifest_sha256: String,
    /// Explicit conservative module/runtime allowance, charged before dlopen.
    pub opaque_memory_bytes: u64,
}

#[derive(Debug)]
pub(crate) struct VerifiedNvfp4Reference {
    pub library: PathBuf,
    pub identity: String,
}

pub fn nvfp4_reference_wrapper_sha256() -> String {
    format!(
        "{:x}",
        Sha256::digest(include_bytes!(
            "../../native/diagnostics/flashinfer_nvfp4.cu"
        ))
    )
}

impl CudaNvfp4ReferenceConfig {
    pub(crate) fn verify(&self) -> Result<VerifiedNvfp4Reference, String> {
        if self.opaque_memory_bytes == 0
            || !sha(&self.library_sha256)
            || !sha(&self.manifest_sha256)
        {
            return Err("reference requires explicit positive capacity and SHA256 pins".into());
        }
        let mut file =
            File::open(&self.manifest).map_err(|e| format!("reference manifest: {e}"))?;
        if !file.metadata().map_err(|e| e.to_string())?.is_file() {
            return Err("reference manifest must be a regular file".into());
        }
        let mut bytes = Vec::new();
        file.by_ref()
            .take(MAX_MANIFEST_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        if bytes.len() as u64 > MAX_MANIFEST_BYTES
            || format!("{:x}", Sha256::digest(&bytes)) != self.manifest_sha256
        {
            return Err("reference manifest size or SHA256 differs from its pin".into());
        }
        let manifest: Nvfp4ReferenceManifest = serde_json::from_slice(&bytes)
            .map_err(|e| format!("reference manifest schema: {e}"))?;
        if manifest.abi_version != 1
            || manifest.profile != NVFP4_REFERENCE_PROFILE
            || manifest.library_sha256 != self.library_sha256
            || manifest.wrapper_sha256 != nvfp4_reference_wrapper_sha256()
            || !manifest.use_fast_math
            || !sha(&manifest.nvcc_sha256)
            || !sha(&manifest.host_compiler_sha256)
            || !text(&manifest.source_identity)
            || !text(&manifest.nvcc_version)
            || !text(&manifest.host_compiler_version)
            || manifest.source_files.is_empty()
            || manifest.source_files.len() > 2048
            || manifest
                .source_files
                .iter()
                .any(|f| !text(&f.path) || !sha(&f.sha256))
            || manifest.compiler_arguments.is_empty()
            || manifest.compiler_arguments.len() > 256
            || manifest.compiler_arguments.iter().any(|a| !text(a))
        {
            return Err(
                "reference manifest has another ABI/profile/wrapper or incomplete provenance"
                    .into(),
            );
        }
        let library = self
            .library
            .canonicalize()
            .map_err(|e| format!("reference library: {e}"))?;
        let mut file = File::open(&library).map_err(|e| e.to_string())?;
        let metadata = file.metadata().map_err(|e| e.to_string())?;
        if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_LIBRARY_BYTES {
            return Err("reference library must be a bounded nonempty regular file".into());
        }
        let mut hash = Sha256::new();
        let mut count = 0u64;
        let mut buffer = [0u8; 16384];
        loop {
            let read = file.read(&mut buffer).map_err(|e| e.to_string())?;
            if read == 0 {
                break;
            }
            count += read as u64;
            if count > MAX_LIBRARY_BYTES {
                return Err("reference library grew beyond its bound".into());
            }
            hash.update(&buffer[..read]);
        }
        if count != metadata.len() || format!("{:x}", hash.finalize()) != self.library_sha256 {
            return Err("reference library differs from its SHA256 pin".into());
        }
        Ok(VerifiedNvfp4Reference {
            library,
            identity: format!(
                "{:x}",
                Sha256::digest(format!(
                    "{}:{}:{}",
                    self.library_sha256, self.manifest_sha256, self.opaque_memory_bytes
                ))
            ),
        })
    }
}

fn sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn text(value: &str) -> bool {
    !value.is_empty() && value.len() <= 4096 && !value.contains('\0')
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn artifact_validation_rejects_changes_before_loading_any_code() {
        let dir =
            std::env::temp_dir().join(format!("ferrum-nvfp4-reference-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let library = dir.join("library.fixture");
        std::fs::write(&library, b"not executable; validation never loads it").unwrap();
        let lib_sha = format!("{:x}", Sha256::digest(std::fs::read(&library).unwrap()));
        let manifest = Nvfp4ReferenceManifest {
            abi_version: 1,
            profile: NVFP4_REFERENCE_PROFILE.into(),
            library_sha256: lib_sha.clone(),
            wrapper_sha256: nvfp4_reference_wrapper_sha256(),
            source_identity: "pinned independent fixture".into(),
            source_files: vec![Nvfp4ReferenceSourceFile {
                path: "upstream/helper.cuh".into(),
                sha256: "1".repeat(64),
            }],
            nvcc_sha256: "2".repeat(64),
            host_compiler_sha256: "3".repeat(64),
            nvcc_version: "fixture".into(),
            host_compiler_version: "fixture".into(),
            compiler_arguments: vec!["--use_fast_math".into()],
            use_fast_math: true,
        };
        let path = dir.join("manifest.json");
        let bytes = serde_json::to_vec(&manifest).unwrap();
        std::fs::write(&path, &bytes).unwrap();
        let mut config = CudaNvfp4ReferenceConfig {
            library: library.clone(),
            library_sha256: lib_sha,
            manifest: path,
            manifest_sha256: format!("{:x}", Sha256::digest(&bytes)),
            opaque_memory_bytes: 4096,
        };
        assert!(config.verify().is_ok());
        std::fs::write(&library, b"changed").unwrap();
        assert!(config.verify().unwrap_err().contains("library differs"));
        config.opaque_memory_bytes = 0;
        assert!(config.verify().is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
