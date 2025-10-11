//! Compression support for KV cache

use ferrum_types::{DataType, FerrumError, Result};
use std::sync::Arc;

/// Compression strategy trait
pub trait CompressionStrategy: Send + Sync + std::fmt::Debug {
    /// Compress data
    fn compress(&self, data: &[u8], original_dtype: DataType) -> Result<CompressedData>;

    /// Decompress data
    fn decompress(&self, compressed: &CompressedData) -> Result<Vec<u8>>;

    /// Get compression ratio estimate
    fn compression_ratio(&self) -> f32;

    /// Get strategy name
    fn name(&self) -> &str;
}

/// Compressed data container
#[derive(Debug, Clone)]
pub struct CompressedData {
    /// Compressed bytes
    pub data: Vec<u8>,
    /// Original data type
    pub original_dtype: DataType,
    /// Original size in bytes
    pub original_size: usize,
    /// Compression algorithm used
    pub algorithm: String,
    /// Compression parameters
    pub params: CompressionParams,
}

/// Compression parameters
#[derive(Debug, Clone)]
pub struct CompressionParams {
    /// Quantization bits (for quantization-based compression)
    pub quantization_bits: Option<u8>,
    /// Block size for block-wise compression
    pub block_size: Option<usize>,
    /// Custom parameters
    pub custom: std::collections::HashMap<String, String>,
}

impl Default for CompressionParams {
    fn default() -> Self {
        Self {
            quantization_bits: None,
            block_size: None,
            custom: std::collections::HashMap::new(),
        }
    }
}

/// No-op compression (passthrough)
#[derive(Debug, Clone, Default)]
pub struct NoCompression;

impl CompressionStrategy for NoCompression {
    fn compress(&self, data: &[u8], original_dtype: DataType) -> Result<CompressedData> {
        Ok(CompressedData {
            data: data.to_vec(),
            original_dtype,
            original_size: data.len(),
            algorithm: "none".to_string(),
            params: CompressionParams::default(),
        })
    }

    fn decompress(&self, compressed: &CompressedData) -> Result<Vec<u8>> {
        Ok(compressed.data.clone())
    }

    fn compression_ratio(&self) -> f32 {
        1.0 // No compression
    }

    fn name(&self) -> &str {
        "none"
    }
}

/// INT4 quantization compression
#[derive(Debug, Clone)]
pub struct Int4Compression {
    /// Quantization parameters
    params: CompressionParams,
}

impl Int4Compression {
    /// Create new INT4 compression
    pub fn new() -> Self {
        let mut params = CompressionParams::default();
        params.quantization_bits = Some(4);

        Self { params }
    }
}

impl Default for Int4Compression {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressionStrategy for Int4Compression {
    fn compress(&self, data: &[u8], original_dtype: DataType) -> Result<CompressedData> {
        match original_dtype {
            DataType::FP16 | DataType::FP32 => {
                // Simplified INT4 quantization
                // In practice, this would involve proper float->int4 quantization
                let compressed_size = (data.len() + 1) / 2; // 4 bits per element
                let mut compressed = vec![0u8; compressed_size];

                // Placeholder compression (just truncate for demo)
                for (i, chunk) in data.chunks(2).enumerate() {
                    if i < compressed.len() {
                        if chunk.len() == 2 {
                            compressed[i] = (chunk[0] & 0xF0) | ((chunk[1] & 0xF0) >> 4);
                        } else {
                            compressed[i] = chunk[0] & 0xF0;
                        }
                    }
                }

                Ok(CompressedData {
                    data: compressed,
                    original_dtype,
                    original_size: data.len(),
                    algorithm: "int4".to_string(),
                    params: self.params.clone(),
                })
            }
            _ => Err(FerrumError::invalid_parameter(format!(
                "INT4 compression not supported for {:?}",
                original_dtype
            ))),
        }
    }

    fn decompress(&self, compressed: &CompressedData) -> Result<Vec<u8>> {
        if compressed.algorithm != "int4" {
            return Err(FerrumError::invalid_parameter(
                "Expected INT4 compressed data",
            ));
        }

        // Simplified decompression (reverse of compression)
        let mut decompressed = Vec::with_capacity(compressed.original_size);

        for &byte in &compressed.data {
            decompressed.push(byte & 0xF0);
            if decompressed.len() < compressed.original_size {
                decompressed.push((byte & 0x0F) << 4);
            }
        }

        // Trim to original size
        decompressed.truncate(compressed.original_size);

        Ok(decompressed)
    }

    fn compression_ratio(&self) -> f32 {
        2.0 // 4 bits instead of 8, so 2:1 compression
    }

    fn name(&self) -> &str {
        "int4"
    }
}

/// FP8 compression
#[derive(Debug, Clone)]
pub struct Fp8Compression {
    params: CompressionParams,
}

impl Fp8Compression {
    /// Create new FP8 compression
    pub fn new() -> Self {
        let mut params = CompressionParams::default();
        params.quantization_bits = Some(8);

        Self { params }
    }
}

impl Default for Fp8Compression {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressionStrategy for Fp8Compression {
    fn compress(&self, data: &[u8], original_dtype: DataType) -> Result<CompressedData> {
        match original_dtype {
            DataType::FP32 => {
                // Convert F32 to FP8 (simplified - would need proper FP8 format)
                // For now, just take every 4th byte as a placeholder
                let compressed: Vec<u8> = data.iter().step_by(4).cloned().collect();

                Ok(CompressedData {
                    data: compressed,
                    original_dtype,
                    original_size: data.len(),
                    algorithm: "fp8".to_string(),
                    params: self.params.clone(),
                })
            }
            DataType::FP16 => {
                // Convert F16 to FP8 (simplified)
                let compressed: Vec<u8> = data.iter().step_by(2).cloned().collect();

                Ok(CompressedData {
                    data: compressed,
                    original_dtype,
                    original_size: data.len(),
                    algorithm: "fp8".to_string(),
                    params: self.params.clone(),
                })
            }
            _ => Err(FerrumError::invalid_parameter(format!(
                "FP8 compression not supported for {:?}",
                original_dtype
            ))),
        }
    }

    fn decompress(&self, compressed: &CompressedData) -> Result<Vec<u8>> {
        if compressed.algorithm != "fp8" {
            return Err(FerrumError::invalid_parameter(
                "Expected FP8 compressed data",
            ));
        }

        match compressed.original_dtype {
            DataType::FP32 => {
                // Expand back to F32 (placeholder)
                let mut decompressed = Vec::with_capacity(compressed.original_size);
                for &byte in &compressed.data {
                    decompressed.push(byte);
                    decompressed.push(0);
                    decompressed.push(0);
                    decompressed.push(0);
                }
                decompressed.truncate(compressed.original_size);
                Ok(decompressed)
            }
            DataType::FP16 => {
                // Expand back to F16 (placeholder)
                let mut decompressed = Vec::with_capacity(compressed.original_size);
                for &byte in &compressed.data {
                    decompressed.push(byte);
                    decompressed.push(0);
                }
                decompressed.truncate(compressed.original_size);
                Ok(decompressed)
            }
            _ => Err(FerrumError::invalid_parameter(format!(
                "Cannot decompress FP8 to {:?}",
                compressed.original_dtype
            ))),
        }
    }

    fn compression_ratio(&self) -> f32 {
        match self.params.quantization_bits.unwrap_or(8) {
            8 => 2.0, // F16 -> FP8: 2:1, F32 -> FP8: 4:1, average ~2:1
            _ => 1.5,
        }
    }

    fn name(&self) -> &str {
        "fp8"
    }
}

/// Compression manager
#[derive(Debug)]
pub struct CompressionManager {
    /// Available compression strategies
    strategies: std::collections::HashMap<String, Arc<dyn CompressionStrategy>>,
    /// Default strategy name
    default_strategy: String,
}

impl CompressionManager {
    /// Create new compression manager
    pub fn new() -> Self {
        let mut strategies: std::collections::HashMap<String, Arc<dyn CompressionStrategy>> =
            std::collections::HashMap::new();

        // Register built-in strategies
        strategies.insert("none".to_string(), Arc::new(NoCompression));
        strategies.insert("int4".to_string(), Arc::new(Int4Compression::new()));
        strategies.insert("fp8".to_string(), Arc::new(Fp8Compression::new()));

        Self {
            strategies,
            default_strategy: "none".to_string(),
        }
    }

    /// Register compression strategy
    pub fn register_strategy<S>(&mut self, name: String, strategy: S)
    where
        S: CompressionStrategy + 'static,
    {
        self.strategies.insert(name, Arc::new(strategy));
    }

    /// Set default strategy
    pub fn set_default_strategy(&mut self, name: &str) -> Result<()> {
        if self.strategies.contains_key(name) {
            self.default_strategy = name.to_string();
            Ok(())
        } else {
            Err(FerrumError::not_found(format!(
                "Compression strategy not found: {}",
                name
            )))
        }
    }

    /// Get strategy by name
    pub fn get_strategy(&self, name: &str) -> Result<Arc<dyn CompressionStrategy>> {
        self.strategies.get(name).cloned().ok_or_else(|| {
            FerrumError::not_found(format!("Compression strategy not found: {}", name))
        })
    }

    /// Get default strategy
    pub fn default_strategy(&self) -> Result<Arc<dyn CompressionStrategy>> {
        self.get_strategy(&self.default_strategy)
    }

    /// List available strategies
    pub fn available_strategies(&self) -> Vec<String> {
        self.strategies.keys().cloned().collect()
    }

    /// Compress data using specified strategy
    pub fn compress(
        &self,
        data: &[u8],
        original_dtype: DataType,
        strategy_name: Option<&str>,
    ) -> Result<CompressedData> {
        let strategy_name = strategy_name.unwrap_or(&self.default_strategy);
        let strategy = self.get_strategy(strategy_name)?;
        strategy.compress(data, original_dtype)
    }

    /// Decompress data
    pub fn decompress(&self, compressed: &CompressedData) -> Result<Vec<u8>> {
        let strategy = self.get_strategy(&compressed.algorithm)?;
        strategy.decompress(compressed)
    }
}

impl Default for CompressionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_compression() {
        let compression = NoCompression;
        let data = vec![1, 2, 3, 4];

        let compressed = compression.compress(&data, DataType::FP32).unwrap();
        assert_eq!(compressed.data, data);
        assert_eq!(compressed.algorithm, "none");

        let decompressed = compression.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_int4_compression() {
        let compression = Int4Compression::new();
        let data = vec![0xAB, 0xCD, 0xEF, 0x12];

        let compressed = compression.compress(&data, DataType::FP16).unwrap();
        assert_eq!(compressed.algorithm, "int4");
        assert!(compressed.data.len() <= data.len()); // Should be smaller or equal

        let decompressed = compression.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), data.len());
    }

    #[test]
    fn test_compression_manager() {
        let manager = CompressionManager::new();

        let strategies = manager.available_strategies();
        assert!(strategies.contains(&"none".to_string()));
        assert!(strategies.contains(&"int4".to_string()));
        assert!(strategies.contains(&"fp8".to_string()));

        let strategy = manager.default_strategy().unwrap();
        assert_eq!(strategy.name(), "none");
    }

    #[test]
    fn test_manager_compress_decompress() {
        let manager = CompressionManager::new();
        let data = vec![1, 2, 3, 4];

        let compressed = manager
            .compress(&data, DataType::FP32, Some("none"))
            .unwrap();
        let decompressed = manager.decompress(&compressed).unwrap();

        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_unsupported_compression() {
        let compression = Int4Compression::new();
        let data = vec![1, 2, 3, 4];

        // INT4 doesn't support integer types in our implementation
        let result = compression.compress(&data, DataType::INT32);
        assert!(result.is_err());
    }
}
