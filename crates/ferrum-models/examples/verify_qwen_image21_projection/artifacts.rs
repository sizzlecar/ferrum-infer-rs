use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > limit as u64 {
        bail!(
            "{} is not a nonempty regular file within {limit} bytes",
            path.display()
        );
    }
    let mut bytes = Vec::new();
    file.take(
        u64::try_from(limit)?
            .checked_add(1)
            .context("read bound overflows")?,
    )
    .read_to_end(&mut bytes)?;
    if bytes.is_empty() || bytes.len() > limit {
        bail!("{} changed beyond its read limit", path.display());
    }
    Ok(bytes)
}

pub struct Artifacts {
    directory: PathBuf,
}

impl Artifacts {
    pub fn create(path: &Path) -> Result<Self> {
        let name = path
            .file_name()
            .context("artifact directory needs a final directory name")?;
        if !matches!(
            Path::new(name).components().next(),
            Some(Component::Normal(_))
        ) {
            bail!("artifact directory needs a normal final component");
        }
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let parent = parent
            .canonicalize()
            .context("artifact parent must already exist")?;
        if !parent.is_dir() {
            bail!("artifact parent is not a directory");
        }
        if parent
            .ancestors()
            .any(|ancestor| ancestor.join(".git").exists())
        {
            bail!("artifact directory must be outside Git worktrees");
        }
        let directory = parent.join(name);
        std::fs::create_dir(&directory).context("artifact directory must be new")?;
        Ok(Self { directory })
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    fn file(&self, name: &str) -> Result<File> {
        if Path::new(name).components().count() != 1
            || !matches!(
                Path::new(name).components().next(),
                Some(Component::Normal(_))
            )
        {
            bail!("artifact name must be one normal path component");
        }
        Ok(OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(self.directory.join(name))?)
    }

    pub fn bytes(&self, name: &str, bytes: &[u8]) -> Result<()> {
        let mut file = self.file(name)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        Ok(())
    }

    pub fn json(&self, name: &str, value: &impl Serialize) -> Result<()> {
        let mut file = self.file(name)?;
        serde_json::to_writer_pretty(&mut file, value)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        Ok(())
    }

    pub fn single_weight_archive(
        &self,
        weight: &[u8],
        outputs: usize,
        inputs: usize,
        digest: &str,
    ) -> Result<()> {
        // Validate with the existing format library; stream its small framing
        // and the borrowed payload into an exclusively created file. The
        // library's serialize_to_file uses File::create (overwrites), whereas
        // serialize would unnecessarily duplicate this entire payload.
        let _view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::BF16,
            vec![outputs, inputs],
            weight,
        )?;
        let header = BTreeMap::from([
            (
                "__metadata__",
                json!({
                    "source_kind": "caller-provided raw tensor fragment",
                    "source_sha256": digest,
                    "source_authenticity": "not independently established",
                    "tensor": "img_in.weight"
                }),
            ),
            (
                "img_in.weight",
                json!({"dtype":"BF16", "shape":[outputs, inputs], "data_offsets":[0, weight.len()]}),
            ),
        ]);
        let mut header = serde_json::to_vec(&header)?;
        let padding = (8 - header.len() % 8) % 8;
        header.resize(header.len() + padding, b' ');
        let mut file = self.file("model.safetensors")?;
        file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
        file.write_all(&header)?;
        file.write_all(weight)?;
        file.sync_all()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archive_roundtrips_bf16_payload_and_refuses_overwrite() {
        let parent = tempfile::tempdir().unwrap();
        let artifacts = Artifacts::create(&parent.path().join("run")).unwrap();
        let bytes = [0x80, 0x3f, 0x00, 0xc0];
        artifacts
            .single_weight_archive(&bytes, 1, 2, &sha256(&bytes))
            .unwrap();
        let archive = ferrum_quantization::SafetensorsArchive::open(artifacts.directory()).unwrap();
        let tensor = archive.tensor("img_in.weight").unwrap();
        assert_eq!(tensor.shape(), &[1, 2]);
        assert_eq!(tensor.bytes(), &bytes);
        assert_eq!(tensor.dtype(), safetensors::Dtype::BF16);
        assert!(artifacts
            .single_weight_archive(&bytes, 1, 2, &sha256(&bytes))
            .is_err());
        assert!(Artifacts::create(artifacts.directory()).is_err());
    }

    #[test]
    fn bounded_reads_reject_empty_and_oversized_files() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("input");
        std::fs::write(&path, []).unwrap();
        assert!(read_bounded(&path, 3).is_err());
        std::fs::write(&path, [1, 2, 3, 4]).unwrap();
        assert!(read_bounded(&path, 3).is_err());
        assert_eq!(read_bounded(&path, 4).unwrap(), vec![1, 2, 3, 4]);
    }
}
