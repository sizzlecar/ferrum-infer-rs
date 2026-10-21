//! Physical KV payload geometry declared by the selected allocator.
//!
//! A requested precision/marker is not a storage declaration. These layouts
//! count K, V and required scale arrays only: allocation overhead, sequence
//! tables, weights and execution scratch are separate resources.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheStorage {
    F32,
    F16,
    /// One signed byte per value, with separate F16 K/V scales for each
    /// token/head. No head-dimension padding is used by this layout.
    Int8PerTokenHeadF16Scales,
}

impl KvCacheStorage {
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Int8PerTokenHeadF16Scales => "int8_per_token_head_f16_scales",
        }
    }

    /// Exact payload bytes for one token across the given local layers.
    /// Missing/invalid geometry and arithmetic overflow are not estimates.
    pub fn bytes_per_token(self, layers: usize, kv_heads: usize, head_dim: usize) -> Option<u64> {
        if layers == 0 || kv_heads == 0 || head_dim == 0 {
            return None;
        }
        let per_head = match self {
            Self::F32 => u64::try_from(head_dim).ok()?.checked_mul(4)?,
            Self::F16 => u64::try_from(head_dim).ok()?.checked_mul(2)?,
            Self::Int8PerTokenHeadF16Scales => u64::try_from(head_dim).ok()?.checked_add(2)?,
        };
        per_head
            .checked_mul(2)? // K and V each have their own values/scales.
            .checked_mul(u64::try_from(kv_heads).ok()?)?
            .checked_mul(u64::try_from(layers).ok()?)
    }
}

#[cfg(test)]
mod tests {
    use super::KvCacheStorage;

    #[test]
    fn physical_kv_storage_counts_actual_values_and_independent_scales() {
        assert_eq!(
            KvCacheStorage::F32.bytes_per_token(28, 8, 128),
            Some(229_376)
        );
        assert_eq!(
            KvCacheStorage::F16.bytes_per_token(28, 8, 128),
            Some(114_688)
        );
        // An odd head dimension proves scales are per token/head, not per
        // value, and no unimplemented vector padding enters the count.
        assert_eq!(
            KvCacheStorage::Int8PerTokenHeadF16Scales.bytes_per_token(3, 5, 7),
            Some(3 * 5 * 2 * (7 + 2))
        );
    }

    #[test]
    fn physical_kv_storage_rejects_missing_geometry_and_overflow() {
        for storage in [
            KvCacheStorage::F32,
            KvCacheStorage::F16,
            KvCacheStorage::Int8PerTokenHeadF16Scales,
        ] {
            assert_eq!(storage.bytes_per_token(0, 8, 128), None);
            assert_eq!(storage.bytes_per_token(28, 0, 128), None);
            assert_eq!(storage.bytes_per_token(28, 8, 0), None);
            assert_eq!(
                storage.bytes_per_token(usize::MAX, usize::MAX, usize::MAX),
                None
            );
        }
    }
}
