//! Block-level tensor storage for PagedAttention KV cache.
//!
//! Each physical block holds K/V data for `block_size` token slots across
//! all transformer layers.  Data is stored as flat `Vec<f32>` per layer,
//! laid out as `[slot][head][dim]` for both keys and values separately.

use ferrum_types::{FerrumError, Result};

/// Configuration describing the model dimensions needed for storage sizing.
#[derive(Debug, Clone, Copy)]
pub struct BlockStorageConfig {
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of KV heads (may differ from query heads in GQA).
    pub num_kv_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Tokens per block.
    pub block_size: usize,
}

impl BlockStorageConfig {
    /// Floats needed for one token's keys (or values) in one layer.
    #[inline]
    pub fn token_kv_size(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// Floats needed for one layer's key (or value) buffer in a full block.
    #[inline]
    pub fn layer_buffer_size(&self) -> usize {
        self.block_size * self.token_kv_size()
    }
}

/// Tensor storage backing a single physical block.
///
/// Layout per layer: two flat buffers (`keys` and `values`), each of length
/// `block_size * num_kv_heads * head_dim`.  Within a buffer the data is
/// row-major `[slot][head][dim]`.
pub struct BlockStorage {
    config: BlockStorageConfig,
    /// `keys[layer]` — flat f32 buffer for key vectors.
    keys: Vec<Vec<f32>>,
    /// `values[layer]` — flat f32 buffer for value vectors.
    values: Vec<Vec<f32>>,
}

impl std::fmt::Debug for BlockStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockStorage")
            .field("num_layers", &self.config.num_layers)
            .field("block_size", &self.config.block_size)
            .finish()
    }
}

impl BlockStorage {
    /// Allocate zeroed storage for one block.
    pub fn new(config: BlockStorageConfig) -> Self {
        let buf_size = config.layer_buffer_size();
        let keys = (0..config.num_layers)
            .map(|_| vec![0.0f32; buf_size])
            .collect();
        let values = (0..config.num_layers)
            .map(|_| vec![0.0f32; buf_size])
            .collect();
        Self {
            config,
            keys,
            values,
        }
    }

    pub fn config(&self) -> &BlockStorageConfig {
        &self.config
    }

    /// Write one token's key and value vectors for a given layer and slot.
    ///
    /// `key` and `value` must each have length `num_kv_heads * head_dim`.
    pub fn write_slot(
        &mut self,
        layer: usize,
        slot: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<()> {
        let tok_size = self.config.token_kv_size();
        if key.len() != tok_size || value.len() != tok_size {
            return Err(FerrumError::invalid_parameter(format!(
                "KV vector length mismatch: expected {}, got key={} value={}",
                tok_size,
                key.len(),
                value.len()
            )));
        }
        if layer >= self.config.num_layers {
            return Err(FerrumError::invalid_parameter(format!(
                "Layer {} out of range (max {})",
                layer, self.config.num_layers
            )));
        }
        if slot >= self.config.block_size {
            return Err(FerrumError::invalid_parameter(format!(
                "Slot {} out of range (block_size {})",
                slot, self.config.block_size
            )));
        }

        let offset = slot * tok_size;
        self.keys[layer][offset..offset + tok_size].copy_from_slice(key);
        self.values[layer][offset..offset + tok_size].copy_from_slice(value);
        Ok(())
    }

    /// Read one token's key and value vectors for a given layer and slot.
    ///
    /// Returns `(key, value)` each of length `num_kv_heads * head_dim`.
    pub fn read_slot(&self, layer: usize, slot: usize) -> Result<(&[f32], &[f32])> {
        if layer >= self.config.num_layers {
            return Err(FerrumError::invalid_parameter(format!(
                "Layer {} out of range (max {})",
                layer, self.config.num_layers
            )));
        }
        if slot >= self.config.block_size {
            return Err(FerrumError::invalid_parameter(format!(
                "Slot {} out of range (block_size {})",
                slot, self.config.block_size
            )));
        }

        let tok_size = self.config.token_kv_size();
        let offset = slot * tok_size;
        Ok((
            &self.keys[layer][offset..offset + tok_size],
            &self.values[layer][offset..offset + tok_size],
        ))
    }

    /// Read contiguous key and value buffers for a range of slots in one layer.
    ///
    /// Returns `(keys, values)` each of length `num_slots * num_kv_heads * head_dim`.
    pub fn read_slots(
        &self,
        layer: usize,
        start_slot: usize,
        num_slots: usize,
    ) -> Result<(&[f32], &[f32])> {
        if layer >= self.config.num_layers {
            return Err(FerrumError::invalid_parameter("Layer out of range"));
        }
        let end_slot = start_slot + num_slots;
        if end_slot > self.config.block_size {
            return Err(FerrumError::invalid_parameter("Slot range out of bounds"));
        }

        let tok_size = self.config.token_kv_size();
        let start = start_slot * tok_size;
        let end = end_slot * tok_size;
        Ok((&self.keys[layer][start..end], &self.values[layer][start..end]))
    }

    /// Deep-copy all data from another storage (for COW).
    pub fn copy_from(&mut self, other: &BlockStorage) -> Result<()> {
        if self.config.num_layers != other.config.num_layers
            || self.config.layer_buffer_size() != other.config.layer_buffer_size()
        {
            return Err(FerrumError::invalid_parameter(
                "BlockStorage config mismatch for copy",
            ));
        }
        for layer in 0..self.config.num_layers {
            self.keys[layer].copy_from_slice(&other.keys[layer]);
            self.values[layer].copy_from_slice(&other.values[layer]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> BlockStorageConfig {
        BlockStorageConfig {
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 8,
            block_size: 4,
        }
    }

    #[test]
    fn write_and_read_slot() {
        let config = test_config();
        let mut storage = BlockStorage::new(config);
        let tok_size = config.token_kv_size(); // 4 * 8 = 32

        let key: Vec<f32> = (0..tok_size).map(|i| i as f32).collect();
        let value: Vec<f32> = (0..tok_size).map(|i| (i as f32) + 100.0).collect();

        storage.write_slot(0, 2, &key, &value).unwrap();

        let (k, v) = storage.read_slot(0, 2).unwrap();
        assert_eq!(k, &key[..]);
        assert_eq!(v, &value[..]);

        // Other slots should still be zero
        let (k0, v0) = storage.read_slot(0, 0).unwrap();
        assert!(k0.iter().all(|&x| x == 0.0));
        assert!(v0.iter().all(|&x| x == 0.0));

        // Other layer should still be zero
        let (k1, _) = storage.read_slot(1, 2).unwrap();
        assert!(k1.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn read_slots_contiguous() {
        let config = test_config();
        let mut storage = BlockStorage::new(config);
        let tok_size = config.token_kv_size();

        // Write slots 0 and 1
        for slot in 0..2 {
            let key: Vec<f32> = (0..tok_size).map(|i| (slot * 100 + i) as f32).collect();
            let val: Vec<f32> = (0..tok_size).map(|i| (slot * 100 + i + 50) as f32).collect();
            storage.write_slot(0, slot, &key, &val).unwrap();
        }

        let (keys, vals) = storage.read_slots(0, 0, 2).unwrap();
        assert_eq!(keys.len(), 2 * tok_size);
        assert_eq!(vals.len(), 2 * tok_size);
        // First slot key starts at 0.0
        assert_eq!(keys[0], 0.0);
        // Second slot key starts at 100.0
        assert_eq!(keys[tok_size], 100.0);
    }

    #[test]
    fn out_of_range_errors() {
        let config = test_config();
        let mut storage = BlockStorage::new(config);
        let tok_size = config.token_kv_size();
        let key = vec![0.0; tok_size];
        let val = vec![0.0; tok_size];

        assert!(storage.write_slot(5, 0, &key, &val).is_err()); // bad layer
        assert!(storage.write_slot(0, 10, &key, &val).is_err()); // bad slot
        assert!(storage.write_slot(0, 0, &[1.0], &val).is_err()); // bad key len
    }

    #[test]
    fn copy_from_duplicates_data() {
        let config = test_config();
        let mut src = BlockStorage::new(config);
        let tok_size = config.token_kv_size();

        let key: Vec<f32> = (0..tok_size).map(|i| i as f32).collect();
        let val: Vec<f32> = (0..tok_size).map(|i| (i as f32) + 1.0).collect();
        src.write_slot(1, 3, &key, &val).unwrap();

        let mut dst = BlockStorage::new(config);
        dst.copy_from(&src).unwrap();

        let (k, v) = dst.read_slot(1, 3).unwrap();
        assert_eq!(k, &key[..]);
        assert_eq!(v, &val[..]);
    }
}
