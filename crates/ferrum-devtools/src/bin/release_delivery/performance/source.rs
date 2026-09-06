//! Pin only the files consumed by this performance cell; never copy model weights.
use super::remaining;
use ferrum_bench_core::release_regression::{
    performance::{FileDigest, SourceIdentity},
    Backend, ModelProfile,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::{Component, Path, PathBuf},
};
use tokio::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FilePin {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SourceManifest {
    pub schema_version: u32,
    pub profile: ModelProfile,
    pub gguf: FilePin,
    pub tokenizer_dir: PathBuf,
    pub sidecars: Vec<FilePin>,
}
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct Bundle {
    pub gguf: PathBuf,
    pub tokenizer_dir: PathBuf,
}
pub(super) const SIDECARS: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.json",
    "chat_template.jinja",
    "generation_config.json",
];
pub(super) fn verify_file(
    path: &Path,
    bytes: Option<u64>,
    expected: &str,
    deadline: Instant,
) -> Result<String, String> {
    if expected.len() != 64 || !expected.bytes().all(|v| v.is_ascii_hexdigit()) {
        return Err("invalid registered SHA-256".into());
    }
    let actual = digest_file(path, deadline)?;
    if bytes.is_some_and(|n| n != actual.bytes) || !actual.sha256.eq_ignore_ascii_case(expected) {
        return Err(format!(
            "pinned file digest/size mismatch: {}",
            path.display()
        ));
    }
    Ok(actual.sha256)
}
pub(super) fn digest_file(path: &Path, deadline: Instant) -> Result<FileDigest, String> {
    let canonical = fs::canonicalize(path)
        .map_err(|e| format!("resolve pinned file {}: {e}", path.display()))?;
    let mut file = fs::File::open(&canonical)
        .map_err(|e| format!("open pinned file {}: {e}", path.display()))?;
    let metadata = file.metadata().map_err(|e| e.to_string())?;
    if !metadata.is_file() || metadata.len() == 0 {
        return Err(format!("pinned file must be nonempty: {}", path.display()));
    }
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    let mut actual = 0u64;
    loop {
        remaining(deadline)?;
        let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        actual = actual
            .checked_add(count as u64)
            .ok_or("file size overflow")?;
        digest.update(&buffer[..count]);
    }
    if actual != metadata.len() {
        return Err(format!("pinned file changed size: {}", path.display()));
    }
    Ok(FileDigest {
        bytes: actual,
        sha256: format!("{:x}", digest.finalize()),
    })
}

fn sidecar_name(path: &Path) -> Result<&str, String> {
    let mut parts = path.components();
    let name = match parts.next() {
        Some(Component::Normal(value)) => value.to_str(),
        _ => None,
    };
    match name {
        Some(name) if parts.next().is_none() && SIDECARS.contains(&name) => Ok(name),
        _ => Err(format!("unregistered metadata name: {}", path.display())),
    }
}
impl SourceManifest {
    pub(super) fn identity(&self) -> Result<SourceIdentity, String> {
        let gguf = FileDigest {
            bytes: self.gguf.bytes,
            sha256: self.gguf.sha256.to_ascii_lowercase(),
        };
        let mut sidecars = BTreeMap::new();
        for pin in &self.sidecars {
            if sidecars
                .insert(
                    sidecar_name(&pin.path)?.to_owned(),
                    FileDigest {
                        bytes: pin.bytes,
                        sha256: pin.sha256.to_ascii_lowercase(),
                    },
                )
                .is_some()
            {
                return Err("duplicate pinned metadata file".into());
            }
        }
        Ok(SourceIdentity { gguf, sidecars })
    }

    pub(super) fn validate(
        &self,
        expected: &ModelProfile,
        deadline: Instant,
    ) -> Result<(), String> {
        if self.schema_version != 1
            || &self.profile != expected
            || !expected.available
            || expected.target.backend != Backend::Metal
            || expected.target.execution_path != "legacy-model-executor"
            || !expected.target.precision.starts_with("gguf-")
            || expected.id.trim().is_empty()
            || expected.target.architecture.trim().is_empty()
        {
            return Err(
                "performance source differs from the registered legacy Metal GGUF profile".into(),
            );
        }
        if !self.gguf.path.is_absolute()
            || self.gguf.path.extension().and_then(|v| v.to_str()) != Some("gguf")
            || !self.tokenizer_dir.is_absolute()
            || !self.tokenizer_dir.is_dir()
            || self.gguf.bytes == 0
        {
            return Err("performance source requires an existing absolute local GGUF and tokenizer directory; aliases are not accepted".into());
        }
        verify_file(
            &self.gguf.path,
            Some(self.gguf.bytes),
            &self.gguf.sha256,
            deadline,
        )?;
        let mut names = BTreeSet::new();
        for pin in &self.sidecars {
            let name = sidecar_name(&pin.path)?;
            if !names.insert(name) {
                return Err("duplicate pinned metadata file".into());
            }
            verify_file(
                &self.tokenizer_dir.join(&pin.path),
                Some(pin.bytes),
                &pin.sha256,
                deadline,
            )?;
        }
        if !names.contains("tokenizer.json") || !names.contains("tokenizer_config.json") {
            return Err("pin tokenizer.json and tokenizer_config.json for benchmark/model tokenizer consistency".into());
        }
        Ok(())
    }
    #[cfg(unix)]
    pub(super) fn prepare(&self, directory: &Path, deadline: Instant) -> Result<Bundle, String> {
        fs::create_dir(directory).map_err(|e| e.to_string())?;
        let weight = fs::canonicalize(&self.gguf.path).map_err(|e| e.to_string())?;
        let bundle = Bundle {
            gguf: directory.join("model.gguf"),
            tokenizer_dir: directory.to_owned(),
        };
        std::os::unix::fs::symlink(&weight, &bundle.gguf).map_err(|e| e.to_string())?;
        for pin in &self.sidecars {
            sidecar_name(&pin.path)?;
            remaining(deadline)?;
            fs::copy(
                fs::canonicalize(self.tokenizer_dir.join(&pin.path)).map_err(|e| e.to_string())?,
                directory.join(&pin.path),
            )
            .map_err(|e| e.to_string())?;
        }
        self.verify_bundle(&bundle, deadline)?;
        Ok(bundle)
    }
    #[cfg(not(unix))]
    pub(super) fn prepare(&self, _directory: &Path, _deadline: Instant) -> Result<Bundle, String> {
        Err("same-host Metal performance requires a Unix worker".into())
    }
    pub(super) fn verify_bundle(
        &self,
        bundle: &Bundle,
        deadline: Instant,
    ) -> Result<SourceIdentity, String> {
        let expected: BTreeSet<_> = std::iter::once("model.gguf".to_string())
            .chain(
                self.sidecars
                    .iter()
                    .map(|p| p.path.to_string_lossy().into_owned()),
            )
            .collect();
        let actual = fs::read_dir(&bundle.tokenizer_dir)
            .map_err(|e| e.to_string())?
            .map(|entry| {
                entry
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<BTreeSet<_>, _>>()?;
        if expected != actual {
            return Err("performance source bundle contains missing/unregistered files".into());
        }
        let gguf = FileDigest {
            bytes: self.gguf.bytes,
            sha256: verify_file(
                &bundle.gguf,
                Some(self.gguf.bytes),
                &self.gguf.sha256,
                deadline,
            )?,
        };
        let mut sidecars = BTreeMap::new();
        for pin in &self.sidecars {
            sidecars.insert(
                sidecar_name(&pin.path)?.to_owned(),
                FileDigest {
                    bytes: pin.bytes,
                    sha256: verify_file(
                        &bundle.tokenizer_dir.join(&pin.path),
                        Some(pin.bytes),
                        &pin.sha256,
                        deadline,
                    )?,
                },
            );
        }
        Ok(SourceIdentity { gguf, sidecars })
    }
}
