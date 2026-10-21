//! Qwen-Image 2.1 Transformer configuration and checkpoint structure preflight.
//!
//! This module consumes small JSON metadata only. Detached safetensors headers
//! establish declared structure, not payload authenticity or executable model
//! support. It produces the same runtime `WeightSchema` a future model program
//! will consume, without claiming to implement that program or a weight source.

mod config;
mod headers;
mod weights;

pub use config::QwenImage21Config;
pub use headers::{
    HeaderEvidence, PayloadVerification, QwenImage21Structure, ShardStructure, TensorStructure,
};
pub use weights::{ExpectedTensor, ExpectedWeights};

use ferrum_types::{FerrumError, Result};

pub const TRANSFORMER_CLASS: &str = "QwenImage21Transformer2DModel";
pub const FAMILY_ID: &str = "family.qwen_image21.transformer";

/// Metadata parsing and inventory limits, not a bound on process RSS or future
/// device allocations. Parser scratch and container overhead are additional.
#[derive(Debug, Clone, Copy)]
pub struct StructureLimits {
    metadata_file_bytes: u64,
    total_metadata_bytes: u64,
    tensors: usize,
    shards: usize,
}

impl StructureLimits {
    pub fn new(
        metadata_file_bytes: u64,
        total_metadata_bytes: u64,
        tensors: usize,
        shards: usize,
    ) -> Result<Self> {
        if metadata_file_bytes == 0
            || total_metadata_bytes == 0
            || metadata_file_bytes >= isize::MAX as u64
            || total_metadata_bytes >= isize::MAX as u64
            || tensors == 0
            || shards == 0
        {
            return Err(invalid(
                "structure limits must be positive and leave room for a host read sentinel",
            ));
        }
        Ok(Self {
            metadata_file_bytes,
            total_metadata_bytes,
            tensors,
            shards,
        })
    }
    pub const fn metadata_file_bytes(self) -> u64 {
        self.metadata_file_bytes
    }
    pub const fn total_metadata_bytes(self) -> u64 {
        self.total_metadata_bytes
    }
    pub const fn maximum_tensors(self) -> usize {
        self.tensors
    }
    pub const fn maximum_shards(self) -> usize {
        self.shards
    }

    fn check_metadata(self, label: &str, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() || bytes.len() as u64 > self.metadata_file_bytes {
            return Err(invalid(format!(
                "{label} is empty or exceeds the metadata file limit"
            )));
        }
        Ok(())
    }
}

impl Default for StructureLimits {
    fn default() -> Self {
        Self {
            metadata_file_bytes: 4 * 1024 * 1024,
            total_metadata_bytes: 32 * 1024 * 1024,
            tensors: 65536,
            shards: 64,
        }
    }
}

fn invalid(message: impl Into<String>) -> FerrumError {
    FerrumError::model(message)
}

#[cfg(test)]
mod tests;
