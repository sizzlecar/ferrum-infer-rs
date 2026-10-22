use super::artifacts;
use anyhow::{ensure, Context, Result};
use ferrum_quantization::SafetensorsArchive;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Component, Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Tensor {
    pub value_id: String,
    pub file: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub bytes: u64,
    pub sha256: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Weight {
    pub tensor: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub bytes: u64,
    pub sha256: String,
}
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Identity {
    pub canonical_config: Value,
    pub config_file_sha256: String,
    pub inputs: Vec<Tensor>,
    pub weights: Vec<Weight>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: String,
    pub backend: String,
    pub identity: Identity,
    pub outputs: Vec<Tensor>,
}
pub struct Artifact {
    directory: PathBuf,
    pub manifest: Manifest,
    pub manifest_sha256: String,
}
impl Artifact {
    pub fn open(directory: &Path, backend: &str) -> Result<Self> {
        let directory = directory.canonicalize()?;
        let bytes = artifacts::read_bounded(&directory.join("manifest.json"), 4 * 1024 * 1024)?;
        let manifest: Manifest = serde_json::from_slice(&bytes)?;
        ensure!(
            manifest.schema == "ferrum.qwen_image21.block0-artifacts.v1"
                && manifest.backend == backend,
            "unexpected artifact schema/backend"
        );
        let this = Self {
            directory,
            manifest,
            manifest_sha256: artifacts::sha256(&bytes),
        };
        let mut seen = std::collections::BTreeSet::new();
        for t in &this.manifest.outputs {
            ensure!(seen.insert(&t.value_id), "duplicate output identity");
        }
        seen.clear();
        for t in &this.manifest.identity.inputs {
            ensure!(seen.insert(&t.value_id), "duplicate input identity");
            this.read_tensor(t)?;
        }
        Ok(this)
    }
    fn read_tensor(&self, t: &Tensor) -> Result<Vec<u8>> {
        let path = Path::new(&t.file);
        ensure!(
            path.components().count() == 1
                && matches!(path.components().next(), Some(Component::Normal(_))),
            "tensor filename must be a basename"
        );
        let path = self.directory.join(path).canonicalize()?;
        ensure!(
            path.starts_with(&self.directory),
            "tensor symlink leaves artifact directory"
        );
        let element = match t.dtype.as_str() {
            "BF16" => 2_u64,
            "F32" => 4,
            _ => anyhow::bail!("unsupported tensor dtype"),
        };
        let bytes = t
            .shape
            .iter()
            .try_fold(element, |n, d| n.checked_mul(*d))
            .context("shape byte overflow")?;
        ensure!(
            !t.shape.is_empty()
                && !t.shape.contains(&0)
                && bytes == t.bytes
                && bytes <= 16 * 1024 * 1024,
            "diagnostic tensor extent mismatch/limit"
        );
        let value = artifacts::read_bounded(&path, usize::try_from(bytes)?)?;
        ensure!(
            value.len() as u64 == bytes && artifacts::sha256(&value) == t.sha256,
            "artifact bytes/SHA mismatch for {}",
            t.value_id
        );
        Ok(value)
    }
    pub fn output(&self, name: &str) -> Result<(Tensor, Vec<u8>)> {
        let t = self
            .manifest
            .outputs
            .iter()
            .find(|t| t.value_id == name)
            .context("missing diagnostic output port")?;
        ensure!(
            t.dtype == "BF16" && t.shape.len() == 2,
            "dense replay needs rank-two BF16"
        );
        Ok((t.clone(), self.read_tensor(t)?))
    }
    pub fn output_metadata(&self, name: &str) -> Result<&Tensor> {
        self.manifest
            .outputs
            .iter()
            .find(|t| t.value_id == name)
            .context("missing diagnostic output metadata")
    }
    pub fn weight<'a>(
        &self,
        source: &'a SafetensorsArchive,
        name: &str,
        k: usize,
        n: usize,
    ) -> Result<&'a [u8]> {
        let records = self
            .manifest
            .identity
            .weights
            .iter()
            .filter(|w| w.tensor == name)
            .collect::<Vec<_>>();
        ensure!(records.len() == 1, "weight identity must be unique");
        let w = records[0];
        let t = source.tensor(name)?;
        ensure!(
            w.dtype == "BF16"
                && w.shape == [n as u64, k as u64]
                && t.dtype() == safetensors::Dtype::BF16
                && t.shape() == w.shape
                && t.bytes().len() as u64 == w.bytes
                && artifacts::sha256(t.bytes()) == w.sha256,
            "weight metadata/SHA mismatch: {name}"
        );
        // tensor() returns a borrowed view; bytes are owned by source mmap.
        Ok(t.bytes())
    }
}
