//! Checked physical ABI for two representations of the same RN-F16 projection.
//! This declares storage, not source conversion approval or execution authority.
use super::*;

pub const RN_F16_FRAGMENT_ABI_V1: u32 = 0x524e4631;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RnF16FragmentSourceFormatV1 {
    Q4K,
    Q5K,
    Q6K,
}

/// Immutable, checked shape/byte plan. The packet tiles 16 output rows and
/// 32 input columns. It reconstructs RN-even F16 operands from source codes
/// and F32 scale metadata; these bytes are not native GGUF K256 blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RnF16FragmentPlanV1 {
    source_format: RnF16FragmentSourceFormatV1,
    n: u64,
    k: u64,
    source_row_bytes: u64,
    source_bytes: u64,
    dense_bytes: u64,
    packed_bytes: u64,
    packed_dimensions: [u64; 2],
}

impl RnF16FragmentPlanV1 {
    pub fn new(
        source_format: RnF16FragmentSourceFormatV1,
        n: u64,
        k: u64,
    ) -> Result<Self, VNextError> {
        let invalid = || {
            VNextError::InvalidExecutionPlan {
            reason: "RN-F16 fragment requires bounded nonzero N/K, K256 alignment and checked exact byte spans".into(),
        }
        };
        if n == 0
            || k == 0
            || n > u64::from(u32::MAX)
            || k > u64::from(u32::MAX)
            || !k.is_multiple_of(256)
        {
            return Err(invalid());
        }
        let source_block_bytes = match source_format {
            RnF16FragmentSourceFormatV1::Q4K => 144,
            RnF16FragmentSourceFormatV1::Q5K => 176,
            RnF16FragmentSourceFormatV1::Q6K => 210,
        };
        let packet_bytes = match source_format {
            RnF16FragmentSourceFormatV1::Q4K => 384,
            RnF16FragmentSourceFormatV1::Q5K => 448,
            RnF16FragmentSourceFormatV1::Q6K => 512,
        };
        let source_row_bytes = (k / 256)
            .checked_mul(source_block_bytes)
            .ok_or_else(invalid)?;
        let source_bytes = source_row_bytes.checked_mul(n).ok_or_else(invalid)?;
        let dense_bytes = n
            .checked_mul(k)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(invalid)?;
        let packed_dimensions = [n.div_ceil(16), k / 32];
        let packed_bytes = packed_dimensions[0]
            .checked_mul(packed_dimensions[1])
            .and_then(|v| v.checked_mul(packet_bytes))
            .ok_or_else(invalid)?;
        // Both representations are admitted together; their sum must fit too.
        dense_bytes.checked_add(packed_bytes).ok_or_else(invalid)?;
        Ok(Self {
            source_format,
            n,
            k,
            source_row_bytes,
            source_bytes,
            dense_bytes,
            packed_bytes,
            packed_dimensions,
        })
    }

    /// Logical matrix or stacked gate/up; flatten leading axes without changing
    /// their declared order. Do not round each source part to its own N tile.
    pub fn from_dimensions(
        source_format: RnF16FragmentSourceFormatV1,
        dimensions: &[u64],
    ) -> Result<Self, VNextError> {
        let invalid = || VNextError::InvalidExecutionPlan {
            reason: "RN-F16 fragment logical dimensions must be a nonempty rank-2/3 projection"
                .into(),
        };
        if !(2..=3).contains(&dimensions.len()) || dimensions.contains(&0) {
            return Err(invalid());
        }
        let n = dimensions[..dimensions.len() - 1]
            .iter()
            .try_fold(1_u64, |n, d| n.checked_mul(*d))
            .ok_or_else(invalid)?;
        Self::new(source_format, n, dimensions[dimensions.len() - 1])
    }
    pub fn n(&self) -> u64 {
        self.n
    }
    pub fn k(&self) -> u64 {
        self.k
    }
    pub fn source_format(&self) -> RnF16FragmentSourceFormatV1 {
        self.source_format
    }
    pub fn source_row_bytes(&self) -> u64 {
        self.source_row_bytes
    }
    pub fn source_bytes(&self) -> u64 {
        self.source_bytes
    }
    pub fn dense_bytes(&self) -> u64 {
        self.dense_bytes
    }
    pub fn packed_bytes(&self) -> u64 {
        self.packed_bytes
    }
    pub fn packed_dimensions(&self) -> [u64; 2] {
        self.packed_dimensions
    }
    pub fn packing_abi(&self) -> u32 {
        RN_F16_FRAGMENT_ABI_V1
    }
    pub fn source_block_spec(&self) -> BlockQuantizationSpec {
        let (format, bytes) = match self.source_format {
            RnF16FragmentSourceFormatV1::Q4K => ("quantization.gguf.q4-k", 144),
            RnF16FragmentSourceFormatV1::Q5K => ("quantization.gguf.q5-k", 176),
            RnF16FragmentSourceFormatV1::Q6K => ("quantization.gguf.q6-k", 210),
        };
        BlockQuantizationSpec {
            format_id: QuantizationFormatId::new(format).expect("fixed format ID"),
            logical_values_per_block: 256,
            bytes_per_block: bytes,
        }
    }
    pub fn packed_encoding(&self) -> WeightEncoding {
        let (format, bytes) = match self.source_format {
            RnF16FragmentSourceFormatV1::Q4K => ("quantization.gguf-rn-f16-fragment.q4-k.v1", 384),
            RnF16FragmentSourceFormatV1::Q5K => ("quantization.gguf-rn-f16-fragment.q5-k.v1", 448),
            RnF16FragmentSourceFormatV1::Q6K => ("quantization.gguf-rn-f16-fragment.q6-k.v1", 512),
        };
        WeightEncoding::BlockQuantized(BlockQuantizationSpec {
            format_id: QuantizationFormatId::new(format).expect("fixed format ID"),
            logical_values_per_block: 512,
            bytes_per_block: bytes,
        })
    }
}

#[cfg(test)]
mod tests;
