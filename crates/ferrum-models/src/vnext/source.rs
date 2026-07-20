use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ferrum_interfaces::vnext::{
    FileFingerprint, ModelArtifactSourceRole, ModelSourceKind, OriginalModelSource,
    OriginalModelSources, ResolvedModelSource, ResolvedModelSources,
};
use ferrum_types::{FerrumError, Result};
use sha2::{Digest, Sha256};

const SEMANTIC_FILES: &[&str] = &["config.json"];
const TOKENIZER_REQUIRED_FILES: &[&str] = &["tokenizer.json"];
const TOKENIZER_OPTIONAL_FILES: &[&str] = &[
    "tokenizer_config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "chat_template.json",
    "chat_template.jinja",
];

/// Exact physical weight artifact selected by product source resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProductionWeightArtifact {
    SafetensorsDirectory(PathBuf),
    GgufFile(PathBuf),
}

impl ProductionWeightArtifact {
    pub fn safetensors_directory(path: impl Into<PathBuf>) -> Self {
        Self::SafetensorsDirectory(path.into())
    }

    pub fn gguf_file(path: impl Into<PathBuf>) -> Self {
        Self::GgufFile(path.into())
    }

    pub fn path(&self) -> &Path {
        match self {
            Self::SafetensorsDirectory(path) | Self::GgufFile(path) => path,
        }
    }

    pub fn is_gguf(&self) -> bool {
        matches!(self, Self::GgufFile(_))
    }
}

/// Non-serialized product source lease shared by tokenizer, model preparation,
/// plan resolution, and the executor. Source roles remain distinct even when
/// all three happen to resolve to one Hugging Face snapshot.
#[derive(Clone)]
pub struct ProductionModelSourceBundle {
    semantic_root: PathBuf,
    tokenizer_root: PathBuf,
    weights: ProductionWeightArtifact,
    original_sources: OriginalModelSources,
    resolved_sources: ResolvedModelSources,
    config_json: Arc<[u8]>,
    tokenizer_json: Arc<[u8]>,
    tokenizer_config_json: Option<Arc<[u8]>>,
    generation_config_json: Option<Arc<[u8]>>,
    chat_template_json: Option<Arc<[u8]>>,
    chat_template_jinja: Option<Arc<[u8]>>,
}

impl fmt::Debug for ProductionModelSourceBundle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProductionModelSourceBundle")
            .field("semantic_root", &self.semantic_root)
            .field("tokenizer_root", &self.tokenizer_root)
            .field("weights", &self.weights)
            .field("original_sources", &self.original_sources)
            .field("resolved_sources", &self.resolved_sources)
            .field("config_json_bytes", &self.config_json.len())
            .field("tokenizer_json_bytes", &self.tokenizer_json.len())
            .field(
                "tokenizer_config_json_bytes",
                &self.tokenizer_config_json.as_ref().map(|bytes| bytes.len()),
            )
            .field(
                "generation_config_json_bytes",
                &self
                    .generation_config_json
                    .as_ref()
                    .map(|bytes| bytes.len()),
            )
            .field(
                "chat_template_json_bytes",
                &self.chat_template_json.as_ref().map(|bytes| bytes.len()),
            )
            .field(
                "chat_template_jinja_bytes",
                &self.chat_template_jinja.as_ref().map(|bytes| bytes.len()),
            )
            .finish()
    }
}

impl ProductionModelSourceBundle {
    pub fn open(
        semantic_root: impl AsRef<Path>,
        tokenizer_root: impl AsRef<Path>,
        weights: ProductionWeightArtifact,
        original_sources: OriginalModelSources,
    ) -> Result<Self> {
        let semantic_root = canonical_directory(semantic_root.as_ref(), "semantic source")?;
        let tokenizer_root = canonical_directory(tokenizer_root.as_ref(), "tokenizer source")?;
        let weights = normalize_weight_artifact(weights)?;

        let semantic_files = fingerprint_named_files(&semantic_root, SEMANTIC_FILES, &[])?;
        let tokenizer_files = fingerprint_named_files(
            &tokenizer_root,
            TOKENIZER_REQUIRED_FILES,
            TOKENIZER_OPTIONAL_FILES,
        )?;
        let weight_files = fingerprint_weight_artifact(&weights)?;

        let config_json = read_required_file(&semantic_root.join("config.json"))?;
        let tokenizer_json = read_required_file(&tokenizer_root.join("tokenizer.json"))?;
        let tokenizer_config_json =
            read_optional_file(&tokenizer_root.join("tokenizer_config.json"))?;
        let generation_config_json =
            read_optional_file(&tokenizer_root.join("generation_config.json"))?;
        let chat_template_json = read_optional_file(&tokenizer_root.join("chat_template.json"))?;
        let chat_template_jinja = read_optional_file(&tokenizer_root.join("chat_template.jinja"))?;

        let resolved_sources = ResolvedModelSources {
            semantic: resolved_source(&original_sources.semantic, &semantic_root, semantic_files)?,
            tokenizer: resolved_source(
                &original_sources.tokenizer,
                &tokenizer_root,
                tokenizer_files,
            )?,
            weights: resolved_source(&original_sources.weights, weights.path(), weight_files)?,
        };

        Ok(Self {
            semantic_root,
            tokenizer_root,
            weights,
            original_sources,
            resolved_sources,
            config_json,
            tokenizer_json,
            tokenizer_config_json,
            generation_config_json,
            chat_template_json,
            chat_template_jinja,
        })
    }

    /// Compatibility constructor for existing co-located safetensors callers.
    /// Product entrypoints should construct explicit role-specific sources.
    pub fn open_colocated_safetensors(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let location = model_dir.display().to_string();
        let original = OriginalModelSource {
            kind: ModelSourceKind::LocalDirectory,
            location,
            requested_revision: None,
        };
        Self::open(
            model_dir,
            model_dir,
            ProductionWeightArtifact::safetensors_directory(model_dir),
            OriginalModelSources {
                semantic: original.clone(),
                tokenizer: original.clone(),
                weights: original,
            },
        )
    }

    pub fn semantic_root(&self) -> &Path {
        &self.semantic_root
    }

    pub fn tokenizer_root(&self) -> &Path {
        &self.tokenizer_root
    }

    pub fn tokenizer_file(&self) -> PathBuf {
        self.tokenizer_root.join("tokenizer.json")
    }

    pub fn weights(&self) -> &ProductionWeightArtifact {
        &self.weights
    }

    pub fn original_sources(&self) -> &OriginalModelSources {
        &self.original_sources
    }

    pub fn resolved_sources(&self) -> &ResolvedModelSources {
        &self.resolved_sources
    }

    pub fn config_json(&self) -> &[u8] {
        &self.config_json
    }

    pub fn tokenizer_json(&self) -> &[u8] {
        &self.tokenizer_json
    }

    pub fn tokenizer_config_json(&self) -> Option<&[u8]> {
        self.tokenizer_config_json.as_deref()
    }

    pub fn generation_config_json(&self) -> Option<&[u8]> {
        self.generation_config_json.as_deref()
    }

    pub fn chat_template_json(&self) -> Option<&[u8]> {
        self.chat_template_json.as_deref()
    }

    pub fn chat_template_jinja(&self) -> Option<&[u8]> {
        self.chat_template_jinja.as_deref()
    }

    pub fn fingerprint(
        &self,
        role: ModelArtifactSourceRole,
        relative_path: &str,
    ) -> Option<&FileFingerprint> {
        self.resolved_sources
            .for_role(role)
            .files
            .iter()
            .find(|file| file.relative_path == relative_path)
    }
}

fn canonical_directory(path: &Path, kind: &str) -> Result<PathBuf> {
    if !path.is_dir() {
        return Err(FerrumError::model(format!(
            "{kind} is not a directory: {}",
            path.display()
        )));
    }
    path.canonicalize()
        .map_err(|error| FerrumError::model(format!("canonicalize {}: {error}", path.display())))
}

fn normalize_weight_artifact(
    artifact: ProductionWeightArtifact,
) -> Result<ProductionWeightArtifact> {
    match artifact {
        ProductionWeightArtifact::SafetensorsDirectory(path) => {
            Ok(ProductionWeightArtifact::SafetensorsDirectory(
                canonical_directory(&path, "safetensors weight source")?,
            ))
        }
        ProductionWeightArtifact::GgufFile(path) => {
            if !path.is_file() {
                return Err(FerrumError::model(format!(
                    "GGUF weight source is not a file: {}",
                    path.display()
                )));
            }
            let file_name = path.file_name().ok_or_else(|| {
                FerrumError::model(format!("GGUF source has no file name: {}", path.display()))
            })?;
            let parent = path.parent().ok_or_else(|| {
                FerrumError::model(format!("GGUF source has no parent: {}", path.display()))
            })?;
            let parent = parent.canonicalize().map_err(|error| {
                FerrumError::model(format!("canonicalize {}: {error}", parent.display()))
            })?;
            Ok(ProductionWeightArtifact::GgufFile(parent.join(file_name)))
        }
    }
}

fn fingerprint_named_files(
    root: &Path,
    required: &[&str],
    optional: &[&str],
) -> Result<Vec<FileFingerprint>> {
    let mut files = Vec::with_capacity(required.len() + optional.len());
    for relative_path in required {
        files.push(fingerprint_file(root, relative_path)?);
    }
    for relative_path in optional {
        if root.join(relative_path).is_file() {
            files.push(fingerprint_file(root, relative_path)?);
        }
    }
    files.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    Ok(files)
}

fn fingerprint_weight_artifact(
    artifact: &ProductionWeightArtifact,
) -> Result<Vec<FileFingerprint>> {
    match artifact {
        ProductionWeightArtifact::GgufFile(path) => {
            let file_name = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| FerrumError::model("GGUF source file name is not UTF-8"))?;
            fingerprint_file(
                path.parent()
                    .ok_or_else(|| FerrumError::model("GGUF source has no parent"))?,
                file_name,
            )
            .map(|file| vec![file])
        }
        ProductionWeightArtifact::SafetensorsDirectory(root) => {
            let mut relative_paths = std::fs::read_dir(root)
                .map_err(|error| {
                    FerrumError::model(format!("read weight directory {}: {error}", root.display()))
                })?
                .filter_map(|entry| entry.ok())
                .filter_map(|entry| {
                    let path = entry.path();
                    let name = path.file_name()?.to_str()?.to_owned();
                    (path.is_file()
                        && (name.ends_with(".safetensors")
                            || name == "model.safetensors.index.json"))
                        .then_some(name)
                })
                .collect::<Vec<_>>();
            relative_paths.sort();
            if !relative_paths
                .iter()
                .any(|path| path.ends_with(".safetensors"))
            {
                return Err(FerrumError::model(format!(
                    "safetensors source contains no shard: {}",
                    root.display()
                )));
            }
            relative_paths
                .iter()
                .map(|path| fingerprint_file(root, path))
                .collect()
        }
    }
}

fn fingerprint_file(root: &Path, relative_path: &str) -> Result<FileFingerprint> {
    if !portable_relative_path(relative_path) {
        return Err(FerrumError::model(format!(
            "source manifest path is not portable: {relative_path:?}"
        )));
    }
    let path = root.join(relative_path);
    let metadata = std::fs::metadata(&path)
        .map_err(|error| FerrumError::model(format!("stat {}: {error}", path.display())))?;
    if !metadata.is_file() || metadata.len() == 0 {
        return Err(FerrumError::model(format!(
            "source manifest file is missing or empty: {}",
            path.display()
        )));
    }
    let sha256 = match trusted_hf_blob_sha256(&path) {
        Some(sha256) => sha256,
        None => hash_file(&path)?,
    };
    Ok(FileFingerprint {
        relative_path: relative_path.to_owned(),
        size_bytes: metadata.len(),
        sha256,
    })
}

fn trusted_hf_blob_sha256(path: &Path) -> Option<String> {
    let mut entry = path.to_path_buf();
    for _ in 0..8 {
        if let Some(sha256) = trusted_hf_snapshot_entry_sha256(&entry) {
            return Some(sha256);
        }
        let target = std::fs::read_link(&entry).ok()?;
        entry = if target.is_absolute() {
            target
        } else {
            entry.parent()?.join(target)
        };
    }
    None
}

fn trusted_hf_snapshot_entry_sha256(entry: &Path) -> Option<String> {
    let revision = entry.parent()?;
    let snapshots = revision.parent()?;
    (snapshots.file_name()?.to_str()? == "snapshots").then_some(())?;
    let repository = snapshots.parent()?;
    repository
        .file_name()?
        .to_str()?
        .starts_with("models--")
        .then_some(())?;

    let target = std::fs::read_link(entry).ok()?;
    let target = if target.is_absolute() {
        target
    } else {
        revision.join(target)
    };
    let blob = target.canonicalize().ok()?;
    let blobs = blob.parent()?;
    (blobs.file_name()?.to_str()? == "blobs").then_some(())?;
    (blobs.parent()?.canonicalize().ok()? == repository.canonicalize().ok()?).then_some(())?;
    sha256_file_name(&blob)
}

fn sha256_file_name(path: &Path) -> Option<String> {
    let digest = path.file_name()?.to_str()?;
    (digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .then(|| digest.to_ascii_lowercase())
}

fn hash_file(path: &Path) -> Result<String> {
    let file = File::open(path)
        .map_err(|error| FerrumError::model(format!("open {}: {error}", path.display())))?;
    let mut reader = BufReader::with_capacity(4 * 1024 * 1024, file);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 4 * 1024 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| FerrumError::model(format!("read {}: {error}", path.display())))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn resolved_source(
    original: &OriginalModelSource,
    path: &Path,
    files: Vec<FileFingerprint>,
) -> Result<ResolvedModelSource> {
    let manifest_revision = manifest_revision(&files)?;
    let (canonical_location, resolved_revision) = match original.kind {
        ModelSourceKind::Repository => (
            original.location.clone(),
            huggingface_snapshot_revision(path).ok_or_else(|| {
                FerrumError::model(format!(
                    "repository source {} did not resolve below snapshots/<revision>: {}",
                    original.location,
                    path.display()
                ))
            })?,
        ),
        ModelSourceKind::LocalDirectory | ModelSourceKind::LocalFile => {
            (path.display().to_string(), manifest_revision)
        }
        ModelSourceKind::ReleaseArtifact => {
            return Err(FerrumError::unsupported(
                "release artifact source bundles are not implemented",
            ))
        }
    };
    Ok(ResolvedModelSource {
        canonical_location,
        resolved_revision,
        files,
    })
}

fn huggingface_snapshot_revision(path: &Path) -> Option<String> {
    let root = if path.is_file() { path.parent()? } else { path };
    if root.parent()?.file_name()?.to_str()? != "snapshots" {
        return None;
    }
    root.file_name()?.to_str().map(str::to_owned)
}

fn manifest_revision(files: &[FileFingerprint]) -> Result<String> {
    let bytes = serde_json::to_vec(files)
        .map_err(|error| FerrumError::internal(format!("serialize source manifest: {error}")))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn read_required_file(path: &Path) -> Result<Arc<[u8]>> {
    let bytes = std::fs::read(path)
        .map_err(|error| FerrumError::model(format!("read {}: {error}", path.display())))?;
    if bytes.is_empty() {
        return Err(FerrumError::model(format!(
            "required source file is empty: {}",
            path.display()
        )));
    }
    Ok(bytes.into())
}

fn read_optional_file(path: &Path) -> Result<Option<Arc<[u8]>>> {
    if !path.is_file() {
        return Ok(None);
    }
    read_required_file(path).map(Some)
}

fn portable_relative_path(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && !path.ends_with('/')
        && !path.contains('\\')
        && path
            .split('/')
            .all(|component| !matches!(component, "" | "." | ".."))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn original(kind: ModelSourceKind, location: &str) -> OriginalModelSource {
        OriginalModelSource {
            kind,
            location: location.to_owned(),
            requested_revision: None,
        }
    }

    #[test]
    fn preserves_three_distinct_roots_and_same_named_files() {
        let root = tempfile::tempdir().unwrap();
        let semantic = root.path().join("semantic");
        let tokenizer = root.path().join("tokenizer");
        let weights = root.path().join("weights");
        std::fs::create_dir_all(&semantic).unwrap();
        std::fs::create_dir_all(&tokenizer).unwrap();
        std::fs::create_dir_all(&weights).unwrap();
        std::fs::write(
            semantic.join("config.json"),
            br#"{"architectures":["Fixture"]}"#,
        )
        .unwrap();
        std::fs::write(tokenizer.join("tokenizer.json"), br#"{"version":"1.0"}"#).unwrap();
        std::fs::write(
            tokenizer.join("tokenizer_config.json"),
            br#"{"chat_template":"fixture"}"#,
        )
        .unwrap();
        std::fs::write(weights.join("model.safetensors"), b"fixture-weights").unwrap();

        let bundle = ProductionModelSourceBundle::open(
            &semantic,
            &tokenizer,
            ProductionWeightArtifact::safetensors_directory(&weights),
            OriginalModelSources {
                semantic: original(ModelSourceKind::LocalDirectory, "semantic-input"),
                tokenizer: original(ModelSourceKind::LocalDirectory, "tokenizer-input"),
                weights: original(ModelSourceKind::LocalDirectory, "weights-input"),
            },
        )
        .unwrap();

        assert_ne!(bundle.semantic_root(), bundle.tokenizer_root());
        assert_ne!(bundle.tokenizer_root(), bundle.weights().path());
        assert_eq!(
            bundle
                .fingerprint(ModelArtifactSourceRole::Semantic, "config.json")
                .unwrap()
                .size_bytes,
            29
        );
        assert!(bundle
            .fingerprint(ModelArtifactSourceRole::Tokenizer, "tokenizer.json")
            .is_some());
        assert!(bundle
            .fingerprint(ModelArtifactSourceRole::Weights, "model.safetensors")
            .is_some());
    }

    #[cfg(unix)]
    #[test]
    fn trusts_huggingface_lfs_blob_identity_without_rehashing_name() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let repository = root.path().join("models--fixture--weights");
        let blobs = repository.join("blobs");
        let snapshot = repository.join("snapshots").join("revision");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(&snapshot).unwrap();
        let digest = "a".repeat(64);
        std::fs::write(blobs.join(&digest), b"weight-bytes").unwrap();
        symlink(
            Path::new("../../blobs").join(&digest),
            snapshot.join("model.safetensors"),
        )
        .unwrap();

        let fingerprint = fingerprint_file(&snapshot, "model.safetensors").unwrap();
        assert_eq!(fingerprint.sha256, digest);
        assert_eq!(fingerprint.size_bytes, 12);
    }

    #[cfg(unix)]
    #[test]
    fn trusts_huggingface_blob_identity_through_product_package_symlink() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let repository = root.path().join("models--fixture--Qwen3.5-4B-GGUF");
        let blobs = repository.join("blobs");
        let snapshot = repository.join("snapshots").join("revision");
        let package = root.path().join("product-package");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::create_dir_all(&package).unwrap();
        let digest = "b".repeat(64);
        std::fs::write(blobs.join(&digest), b"weight-bytes").unwrap();
        let snapshot_weight = snapshot.join("model.gguf");
        symlink(Path::new("../../blobs").join(&digest), &snapshot_weight).unwrap();
        symlink(&snapshot_weight, package.join("model.gguf")).unwrap();

        let fingerprint = fingerprint_file(&package, "model.gguf").unwrap();
        assert_eq!(fingerprint.sha256, digest);
        assert_eq!(fingerprint.size_bytes, 12);

        let direct_blob_link = package.join("direct-blob.gguf");
        symlink(blobs.join(&digest), &direct_blob_link).unwrap();
        let direct_blob_fingerprint = fingerprint_file(&package, "direct-blob.gguf").unwrap();
        assert_eq!(
            direct_blob_fingerprint.sha256,
            format!("{:x}", Sha256::digest(b"weight-bytes"))
        );
        assert_ne!(direct_blob_fingerprint.sha256, digest);
    }
}
