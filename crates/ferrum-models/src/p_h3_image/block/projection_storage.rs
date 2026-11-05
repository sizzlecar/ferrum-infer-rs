//! Per-projection physical storage declared by the original checkpoint.
//! No layer number or model name selects a quantization format.
use anyhow::{bail, Result};
use ferrum_quantization::comfy_convrot::{
    ComfyCanonicalWeightSource, ComfyQuantizationMetadata, ComfyQuantizedMatrix,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PH3ImageProjectionStorage {
    CodebookW4A8,
    Int8W8A8,
}
impl PH3ImageProjectionStorage {
    pub fn from_metadata(m: &ComfyQuantizationMetadata) -> Result<Self> {
        match m {
            ComfyQuantizationMetadata::CodebookW4A8 {
                group_size: 16,
                convrot: true,
                convrot_groupsize: 256,
            } => Ok(Self::CodebookW4A8),
            ComfyQuantizationMetadata::Int8Tensorwise {
                convrot: true,
                convrot_groupsize: 256,
            } => Ok(Self::Int8W8A8),
            _ => bail!("H3 projection requires canonical W4 group16/H256 or W8 H256 metadata"),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PH3ImageBlockProjectionStorage {
    qkv: PH3ImageProjectionStorage,
    attention_output: PH3ImageProjectionStorage,
    mlp_up_gate: PH3ImageProjectionStorage,
    mlp_down: PH3ImageProjectionStorage,
}
impl Default for PH3ImageBlockProjectionStorage {
    fn default() -> Self {
        Self::uniform(PH3ImageProjectionStorage::CodebookW4A8)
    }
}
impl PH3ImageBlockProjectionStorage {
    pub const fn new(
        qkv: PH3ImageProjectionStorage,
        attention_output: PH3ImageProjectionStorage,
        mlp_up_gate: PH3ImageProjectionStorage,
        mlp_down: PH3ImageProjectionStorage,
    ) -> Self {
        Self {
            qkv,
            attention_output,
            mlp_up_gate,
            mlp_down,
        }
    }
    pub const fn uniform(storage: PH3ImageProjectionStorage) -> Self {
        Self::new(storage, storage, storage, storage)
    }
    pub fn from_checkpoint(source: &ComfyCanonicalWeightSource, block_index: u64) -> Result<Self> {
        let get = |module: &str| -> Result<PH3ImageProjectionStorage> {
            let matrix = ComfyQuantizedMatrix::from_archive(
                source.archive(),
                &format!("blocks.{block_index}.{module}"),
            )?;
            PH3ImageProjectionStorage::from_metadata(matrix.metadata())
        };
        Ok(Self::new(
            get("attn.qkv_proj")?,
            get("attn.out_proj")?,
            get("mlp.fc1")?,
            get("mlp.fc2")?,
        ))
    }
    pub const fn all_w4(self) -> bool {
        matches!(self.qkv, PH3ImageProjectionStorage::CodebookW4A8)
            && matches!(
                self.attention_output,
                PH3ImageProjectionStorage::CodebookW4A8
            )
            && matches!(self.mlp_up_gate, PH3ImageProjectionStorage::CodebookW4A8)
            && matches!(self.mlp_down, PH3ImageProjectionStorage::CodebookW4A8)
    }
    pub fn modules(self) -> [(&'static str, PH3ImageProjectionStorage); 4] {
        [
            ("attn.qkv_proj", self.qkv),
            ("attn.out_proj", self.attention_output),
            ("mlp.fc1", self.mlp_up_gate),
            ("mlp.fc2", self.mlp_down),
        ]
    }
    pub(super) fn module(self, name: &str) -> Result<PH3ImageProjectionStorage> {
        self.modules()
            .into_iter()
            .find(|(module, _)| *module == name)
            .map(|(_, storage)| storage)
            .ok_or_else(|| anyhow::anyhow!("unknown H3 projection module {name}"))
    }
}
