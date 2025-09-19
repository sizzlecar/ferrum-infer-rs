//! SafeTensors weight loader implementation

use ferrum_interfaces::{WeightLoader, WeightSpec};
use ferrum_types::{Result, DataType, FerrumError};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};
use async_trait::async_trait;

/// SafeTensors format weight loader
#[derive(Debug, Clone)]
pub struct SafeTensorsLoader {
    /// Loaded tensor metadata
    metadata: parking_lot::RwLock<HashMap<String, TensorMetadata>>,
    /// File paths for tensor data
    file_paths: parking_lot::RwLock<Vec<PathBuf>>,
}

/// Tensor metadata
#[derive(Debug, Clone)]
struct TensorMetadata {
    /// Shape of the tensor
    shape: Vec<usize>,
    /// Data type
    dtype: DataType,
    /// File index (for multi-file models)
    file_index: usize,
    /// Byte offset in file
    offset: usize,
    /// Size in bytes
    size: usize,
}

impl SafeTensorsLoader {
    /// Create new SafeTensors loader
    pub fn new() -> Self {
        Self {
            metadata: parking_lot::RwLock::new(HashMap::new()),
            file_paths: parking_lot::RwLock::new(Vec::new()),
        }
    }

    /// Load index file to discover tensor locations
    pub async fn load_index<P: AsRef<Path>>(&self, model_path: P) -> Result<()> {
        let model_path = model_path.as_ref();
        debug!("Loading SafeTensors index from: {:?}", model_path);

        // Look for index file
        let index_path = if model_path.is_file() && model_path.extension().map_or(false, |ext| ext == "safetensors") {
            // Single file
            self.file_paths.write().push(model_path.to_path_buf());
            return self.load_single_file(model_path).await;
        } else {
            // Multi-file model with index
            model_path.join("model.safetensors.index.json")
        };

        if index_path.exists() {
            self.load_multi_file_index(&index_path).await
        } else {
            // Try single file
            let single_file = model_path.join("model.safetensors");
            if single_file.exists() {
                self.file_paths.write().push(single_file.clone());
                self.load_single_file(&single_file).await
            } else {
                Err(FerrumError::not_found(
                    format!("No SafeTensors files found in {:?}", model_path)
                ))
            }
        }
    }

    /// Load single SafeTensors file
    async fn load_single_file<P: AsRef<Path>>(&self, file_path: P) -> Result<()> {
        let file_path = file_path.as_ref();
        info!("Loading single SafeTensors file: {:?}", file_path);

        // Read file header to get tensor metadata
        let file_data = tokio::fs::read(file_path).await
            .map_err(|e| FerrumError::io(format!("Failed to read SafeTensors file: {}", e)))?;

        if file_data.len() < 8 {
            return Err(FerrumError::invalid_format("SafeTensors file too small"));
        }

        // Parse header length (first 8 bytes, little endian)
        let header_len = u64::from_le_bytes([
            file_data[0], file_data[1], file_data[2], file_data[3],
            file_data[4], file_data[5], file_data[6], file_data[7],
        ]) as usize;

        if file_data.len() < 8 + header_len {
            return Err(FerrumError::invalid_format("Invalid SafeTensors header length"));
        }

        // Parse JSON metadata
        let header_data = &file_data[8..8 + header_len];
        let metadata: serde_json::Value = serde_json::from_slice(header_data)
            .map_err(|e| FerrumError::invalid_format(format!("Invalid SafeTensors JSON: {}", e)))?;

        let mut tensor_metadata = self.metadata.write();
        let data_offset = 8 + header_len;

        // Parse tensor info from metadata
        if let Some(obj) = metadata.as_object() {
            for (name, info) in obj {
                if name == "__metadata__" {
                    continue; // Skip metadata section
                }

                if let Some(tensor_info) = info.as_object() {
                    let tensor_meta = self.parse_tensor_info(tensor_info, 0, data_offset)?;
                    tensor_metadata.insert(name.clone(), tensor_meta);
                }
            }
        }

        info!("Loaded metadata for {} tensors", tensor_metadata.len());
        Ok(())
    }

    /// Load multi-file index
    async fn load_multi_file_index<P: AsRef<Path>>(&self, index_path: P) -> Result<()> {
        let index_path = index_path.as_ref();
        info!("Loading SafeTensors multi-file index: {:?}", index_path);

        let index_data = tokio::fs::read_to_string(index_path).await
            .map_err(|e| FerrumError::io(format!("Failed to read index file: {}", e)))?;

        let index: serde_json::Value = serde_json::from_str(&index_data)
            .map_err(|e| FerrumError::invalid_format(format!("Invalid index JSON: {}", e)))?;

        // Parse weight map
        if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
            let model_dir = index_path.parent().unwrap();
            let mut file_paths = self.file_paths.write();
            let mut file_map = HashMap::new();

            // Build file index
            for file_name in weight_map.values().filter_map(|v| v.as_str()) {
                if !file_map.contains_key(file_name) {
                    let file_index = file_paths.len();
                    file_paths.push(model_dir.join(file_name));
                    file_map.insert(file_name.to_string(), file_index);
                }
            }

            // Load metadata from each file
            let mut tensor_metadata = self.metadata.write();
            for (tensor_name, file_name) in weight_map {
                let file_name = file_name.as_str().unwrap();
                let file_index = file_map[file_name];
                let file_path = &file_paths[file_index];

                // Load metadata for this tensor (simplified - would need actual file parsing)
                if let Ok(meta) = self.load_tensor_metadata_from_file(file_path, tensor_name).await {
                    let mut meta = meta;
                    meta.file_index = file_index;
                    tensor_metadata.insert(tensor_name.clone(), meta);
                }
            }

            info!("Loaded index for {} files, {} tensors", file_paths.len(), tensor_metadata.len());
        }

        Ok(())
    }

    /// Parse tensor info from JSON
    fn parse_tensor_info(
        &self,
        tensor_info: &serde_json::Map<String, serde_json::Value>,
        file_index: usize,
        base_offset: usize,
    ) -> Result<TensorMetadata> {
        let dtype = tensor_info.get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FerrumError::invalid_format("Missing tensor dtype"))?;

        let shape = tensor_info.get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| FerrumError::invalid_format("Missing tensor shape"))?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect::<Vec<_>>();

        let data_offsets = tensor_info.get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| FerrumError::invalid_format("Missing tensor data_offsets"))?;

        let start_offset = data_offsets[0].as_u64().unwrap_or(0) as usize;
        let end_offset = data_offsets[1].as_u64().unwrap_or(0) as usize;

        let dtype_parsed = match dtype {
            "F32" | "float32" => DataType::F32,
            "F16" | "float16" => DataType::F16,
            "BF16" | "bfloat16" => DataType::BF16,
            "I32" | "int32" => DataType::I32,
            "I64" | "int64" => DataType::I64,
            "U8" | "uint8" => DataType::U8,
            _ => return Err(FerrumError::invalid_format(format!("Unsupported dtype: {}", dtype))),
        };

        Ok(TensorMetadata {
            shape,
            dtype: dtype_parsed,
            file_index,
            offset: base_offset + start_offset,
            size: end_offset - start_offset,
        })
    }

    /// Load tensor metadata from a specific file (placeholder)
    async fn load_tensor_metadata_from_file<P: AsRef<Path>>(
        &self,
        _file_path: P,
        _tensor_name: &str,
    ) -> Result<TensorMetadata> {
        // Placeholder - would need to parse the actual file
        Ok(TensorMetadata {
            shape: vec![1],
            dtype: DataType::F32,
            file_index: 0,
            offset: 0,
            size: 4,
        })
    }
}

impl Default for SafeTensorsLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl WeightLoader for SafeTensorsLoader {
    async fn load_tensor(&self, spec: &WeightSpec, dst: &mut dyn std::any::Any) -> Result<()> {
        debug!("Loading tensor: {}", spec.name);

        let metadata = self.metadata.read();
        let tensor_meta = metadata.get(&spec.name)
            .ok_or_else(|| FerrumError::not_found(format!("Tensor not found: {}", spec.name)))?;

        // Validate shape compatibility
        if let Some(expected_shape) = &spec.shape {
            if tensor_meta.shape != *expected_shape {
                return Err(FerrumError::invalid_format(
                    format!("Shape mismatch for {}: expected {:?}, got {:?}",
                        spec.name, expected_shape, tensor_meta.shape)
                ));
            }
        }

        // Validate data type
        if let Some(expected_dtype) = spec.dtype {
            if tensor_meta.dtype != expected_dtype {
                warn!("Data type mismatch for {}: expected {:?}, got {:?}",
                      spec.name, expected_dtype, tensor_meta.dtype);
            }
        }

        // Load tensor data from file
        let file_paths = self.file_paths.read();
        let file_path = &file_paths[tensor_meta.file_index];

        let file = tokio::fs::File::open(file_path).await
            .map_err(|e| FerrumError::io(format!("Failed to open weight file: {}", e)))?;

        // Read tensor data at offset
        use tokio::io::{AsyncReadExt, AsyncSeekExt};
        let mut file = file;
        file.seek(std::io::SeekFrom::Start(tensor_meta.offset as u64)).await
            .map_err(|e| FerrumError::io(format!("Failed to seek in weight file: {}", e)))?;

        let mut buffer = vec![0u8; tensor_meta.size];
        file.read_exact(&mut buffer).await
            .map_err(|e| FerrumError::io(format!("Failed to read tensor data: {}", e)))?;

        // In a real implementation, this would copy the data to the destination buffer
        // The destination would be a device-specific buffer that can accept the raw bytes
        debug!("Successfully loaded tensor: {} ({} bytes)", spec.name, buffer.len());

        Ok(())
    }

    fn supported_formats(&self) -> Vec<String> {
        vec!["safetensors".to_string()]
    }

    fn name(&self) -> &str {
        "safetensors"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loader_creation() {
        let loader = SafeTensorsLoader::new();
        assert_eq!(loader.name(), "safetensors");
        assert_eq!(loader.supported_formats(), vec!["safetensors"]);
    }

    #[test]
    fn test_tensor_info_parsing() {
        let loader = SafeTensorsLoader::new();
        
        let mut tensor_info = serde_json::Map::new();
        tensor_info.insert("dtype".to_string(), serde_json::Value::String("F32".to_string()));
        tensor_info.insert("shape".to_string(), serde_json::json!([2, 3]));
        tensor_info.insert("data_offsets".to_string(), serde_json::json!([0, 24]));

        let metadata = loader.parse_tensor_info(&tensor_info, 0, 1000).unwrap();
        
        assert_eq!(metadata.dtype, DataType::F32);
        assert_eq!(metadata.shape, vec![2, 3]);
        assert_eq!(metadata.size, 24);
        assert_eq!(metadata.offset, 1000);
    }

    #[test]
    fn test_unsupported_dtype() {
        let loader = SafeTensorsLoader::new();
        
        let mut tensor_info = serde_json::Map::new();
        tensor_info.insert("dtype".to_string(), serde_json::Value::String("UNSUPPORTED".to_string()));
        tensor_info.insert("shape".to_string(), serde_json::json!([2, 3]));
        tensor_info.insert("data_offsets".to_string(), serde_json::json!([0, 24]));

        let result = loader.parse_tensor_info(&tensor_info, 0, 0);
        assert!(result.is_err());
    }
}
