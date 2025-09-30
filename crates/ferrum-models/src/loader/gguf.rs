//! GGUF weight loader implementation

use async_trait::async_trait;
use ferrum_interfaces::{WeightLoader, WeightSpec};
use ferrum_types::{DataType, FerrumError, Result};
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info, warn};

/// GGUF format weight loader
#[derive(Debug, Clone)]
pub struct GGUFLoader {
    /// Loaded tensor metadata
    metadata: parking_lot::RwLock<HashMap<String, GGUFTensorInfo>>,
    /// Model metadata
    model_metadata: parking_lot::RwLock<HashMap<String, GGUFValue>>,
    /// File path
    file_path: parking_lot::RwLock<Option<std::path::PathBuf>>,
}

/// GGUF tensor information
#[derive(Debug, Clone)]
struct GGUFTensorInfo {
    /// Tensor name
    name: String,
    /// Number of dimensions
    n_dims: u32,
    /// Shape/dimensions
    dimensions: Vec<u64>,
    /// Data type
    ggml_type: GGMLType,
    /// Offset in file
    offset: u64,
}

/// GGUF metadata value
#[derive(Debug, Clone)]
enum GGUFValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GGUFValue>),
}

/// GGML data types (subset)
#[derive(Debug, Clone, Copy, PartialEq)]
enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
    I64 = 19,
    BF16 = 20,
}

impl GGMLType {
    /// Create from type ID
    fn from_type_id(type_id: u32) -> Result<Self> {
        match type_id {
            0 => Ok(GGMLType::F32),
            1 => Ok(GGMLType::F16),
            2 => Ok(GGMLType::Q4_0),
            3 => Ok(GGMLType::Q4_1),
            6 => Ok(GGMLType::Q5_0),
            7 => Ok(GGMLType::Q5_1),
            8 => Ok(GGMLType::Q8_0),
            9 => Ok(GGMLType::Q8_1),
            10 => Ok(GGMLType::Q2_K),
            11 => Ok(GGMLType::Q3_K),
            12 => Ok(GGMLType::Q4_K),
            13 => Ok(GGMLType::Q5_K),
            14 => Ok(GGMLType::Q6_K),
            15 => Ok(GGMLType::Q8_K),
            16 => Ok(GGMLType::I8),
            17 => Ok(GGMLType::I16),
            18 => Ok(GGMLType::I32),
            19 => Ok(GGMLType::I64),
            20 => Ok(GGMLType::BF16),
            _ => Err(FerrumError::invalid_format(format!(
                "Unknown GGML type: {}",
                type_id
            ))),
        }
    }

    /// Convert to Ferrum DataType
    fn to_ferrum_dtype(&self) -> DataType {
        match self {
            GGMLType::F32 => DataType::F32,
            GGMLType::F16 => DataType::F16,
            GGMLType::BF16 => DataType::BF16,
            GGMLType::I8 => DataType::I8,
            GGMLType::I16 => DataType::I16,
            GGMLType::I32 => DataType::I32,
            GGMLType::I64 => DataType::I64,
            // Quantized types default to F16 for now
            _ => DataType::F16,
        }
    }

    /// Get size in bytes per element
    fn bytes_per_element(&self) -> usize {
        match self {
            GGMLType::F32 | GGMLType::I32 => 4,
            GGMLType::F16 | GGMLType::BF16 | GGMLType::I16 => 2,
            GGMLType::I8 => 1,
            GGMLType::I64 => 8,
            // Quantized types have variable sizes
            GGMLType::Q4_0 | GGMLType::Q4_1 => 18, // 32 elements in 18 bytes
            GGMLType::Q5_0 | GGMLType::Q5_1 => 22, // 32 elements in 22 bytes
            GGMLType::Q8_0 | GGMLType::Q8_1 => 34, // 32 elements in 34 bytes
            _ => 2,                                // Default for K-quants
        }
    }

    /// Check if type is quantized
    fn is_quantized(&self) -> bool {
        !matches!(
            self,
            GGMLType::F32
                | GGMLType::F16
                | GGMLType::BF16
                | GGMLType::I8
                | GGMLType::I16
                | GGMLType::I32
                | GGMLType::I64
        )
    }
}

impl GGUFLoader {
    /// Create new GGUF loader
    pub fn new() -> Self {
        Self {
            metadata: parking_lot::RwLock::new(HashMap::new()),
            model_metadata: parking_lot::RwLock::new(HashMap::new()),
            file_path: parking_lot::RwLock::new(None),
        }
    }

    /// Load GGUF file
    pub async fn load_file<P: AsRef<Path>>(&self, file_path: P) -> Result<()> {
        let file_path = file_path.as_ref();
        info!("Loading GGUF file: {:?}", file_path);

        *self.file_path.write() = Some(file_path.to_path_buf());

        let file_data = tokio::fs::read(file_path)
            .await
            .map_err(|e| FerrumError::io(format!("Failed to read GGUF file: {}", e)))?;

        self.parse_gguf(&file_data).await
    }

    /// Parse GGUF format
    async fn parse_gguf(&self, data: &[u8]) -> Result<()> {
        if data.len() < 20 {
            return Err(FerrumError::invalid_format("GGUF file too small"));
        }

        let mut offset = 0;

        // Check magic number
        let magic = &data[offset..offset + 4];
        if magic != b"GGUF" {
            return Err(FerrumError::invalid_format("Invalid GGUF magic number"));
        }
        offset += 4;

        // Read version
        let version = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        if version != 3 {
            warn!("GGUF version {} may not be fully supported", version);
        }
        offset += 4;

        // Read tensor count
        let tensor_count = u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        offset += 8;

        // Read metadata count
        let metadata_count = u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        offset += 8;

        debug!(
            "GGUF: version={}, tensors={}, metadata={}",
            version, tensor_count, metadata_count
        );

        // Parse metadata
        let mut model_metadata = self.model_metadata.write();
        for _ in 0..metadata_count {
            let (key, value, new_offset) = self.parse_metadata_kv(&data, offset)?;
            model_metadata.insert(key, value);
            offset = new_offset;
        }

        // Parse tensor info
        let mut tensor_metadata = self.metadata.write();
        for _ in 0..tensor_count {
            let (tensor_info, new_offset) = self.parse_tensor_info(&data, offset)?;
            tensor_metadata.insert(tensor_info.name.clone(), tensor_info);
            offset = new_offset;
        }

        // The tensor data starts after alignment padding
        let alignment = 32; // GGUF typically uses 32-byte alignment
        let aligned_offset = (offset + alignment - 1) & !(alignment - 1);
        let data_offset = aligned_offset;

        // Update tensor offsets to be absolute file offsets
        for tensor_info in tensor_metadata.values_mut() {
            tensor_info.offset += data_offset as u64;
        }

        info!(
            "Parsed GGUF: {} metadata entries, {} tensors",
            metadata_count, tensor_count
        );
        Ok(())
    }

    /// Parse metadata key-value pair
    fn parse_metadata_kv(&self, data: &[u8], offset: usize) -> Result<(String, GGUFValue, usize)> {
        let mut pos = offset;

        // Read key string
        let (key, new_pos) = self.read_string(data, pos)?;
        pos = new_pos;

        // Read value type
        if pos + 4 > data.len() {
            return Err(FerrumError::invalid_format(
                "Unexpected end of GGUF metadata",
            ));
        }
        let value_type =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        pos += 4;

        // Read value based on type
        let (value, new_pos) = self.read_metadata_value(data, pos, value_type)?;

        Ok((key, value, new_pos))
    }

    /// Read string from GGUF
    fn read_string(&self, data: &[u8], offset: usize) -> Result<(String, usize)> {
        if offset + 8 > data.len() {
            return Err(FerrumError::invalid_format("Cannot read string length"));
        }

        let length = u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]) as usize;

        let string_start = offset + 8;
        if string_start + length > data.len() {
            return Err(FerrumError::invalid_format("String extends beyond file"));
        }

        let string_data = &data[string_start..string_start + length];
        let string = String::from_utf8_lossy(string_data).to_string();

        Ok((string, string_start + length))
    }

    /// Read metadata value
    fn read_metadata_value(
        &self,
        data: &[u8],
        offset: usize,
        value_type: u32,
    ) -> Result<(GGUFValue, usize)> {
        match value_type {
            0 => {
                // UINT8
                if offset + 1 > data.len() {
                    return Err(FerrumError::invalid_format("Cannot read U8"));
                }
                Ok((GGUFValue::U8(data[offset]), offset + 1))
            }
            1 => {
                // INT8
                if offset + 1 > data.len() {
                    return Err(FerrumError::invalid_format("Cannot read I8"));
                }
                Ok((GGUFValue::I8(data[offset] as i8), offset + 1))
            }
            4 => {
                // UINT32
                if offset + 4 > data.len() {
                    return Err(FerrumError::invalid_format("Cannot read U32"));
                }
                let value = u32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]);
                Ok((GGUFValue::U32(value), offset + 4))
            }
            6 => {
                // FLOAT32
                if offset + 4 > data.len() {
                    return Err(FerrumError::invalid_format("Cannot read F32"));
                }
                let bytes = [
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ];
                let value = f32::from_le_bytes(bytes);
                Ok((GGUFValue::F32(value), offset + 4))
            }
            8 => {
                // STRING
                let (string, new_offset) = self.read_string(data, offset)?;
                Ok((GGUFValue::String(string), new_offset))
            }
            9 => {
                // BOOL
                if offset + 1 > data.len() {
                    return Err(FerrumError::invalid_format("Cannot read bool"));
                }
                Ok((GGUFValue::Bool(data[offset] != 0), offset + 1))
            }
            _ => Err(FerrumError::invalid_format(format!(
                "Unsupported metadata type: {}",
                value_type
            ))),
        }
    }

    /// Parse tensor info
    fn parse_tensor_info(&self, data: &[u8], offset: usize) -> Result<(GGUFTensorInfo, usize)> {
        let mut pos = offset;

        // Read tensor name
        let (name, new_pos) = self.read_string(data, pos)?;
        pos = new_pos;

        // Read number of dimensions
        if pos + 4 > data.len() {
            return Err(FerrumError::invalid_format("Cannot read tensor n_dims"));
        }
        let n_dims = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        pos += 4;

        // Read dimensions
        let mut dimensions = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            if pos + 8 > data.len() {
                return Err(FerrumError::invalid_format("Cannot read tensor dimension"));
            }
            let dim = u64::from_le_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
                data[pos + 4],
                data[pos + 5],
                data[pos + 6],
                data[pos + 7],
            ]);
            dimensions.push(dim);
            pos += 8;
        }

        // Read GGML type
        if pos + 4 > data.len() {
            return Err(FerrumError::invalid_format("Cannot read tensor type"));
        }
        let type_id = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        let ggml_type = GGMLType::from_type_id(type_id)?;
        pos += 4;

        // Read tensor data offset
        if pos + 8 > data.len() {
            return Err(FerrumError::invalid_format("Cannot read tensor offset"));
        }
        let offset = u64::from_le_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]);
        pos += 8;

        let tensor_info = GGUFTensorInfo {
            name,
            n_dims,
            dimensions,
            ggml_type,
            offset,
        };

        Ok((tensor_info, pos))
    }

    /// Get model metadata value
    pub fn get_metadata(&self, key: &str) -> Option<GGUFValue> {
        self.model_metadata.read().get(key).cloned()
    }

    /// List tensor names
    pub fn tensor_names(&self) -> Vec<String> {
        self.metadata.read().keys().cloned().collect()
    }
}

impl Default for GGUFLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl WeightLoader for GGUFLoader {
    async fn load_tensor(&self, spec: &WeightSpec, dst: &mut dyn std::any::Any) -> Result<()> {
        debug!("Loading GGUF tensor: {}", spec.name);

        let metadata = self.metadata.read();
        let tensor_info = metadata
            .get(&spec.name)
            .ok_or_else(|| FerrumError::not_found(format!("Tensor not found: {}", spec.name)))?;

        // Validate shape if specified
        if let Some(expected_shape) = &spec.shape {
            let tensor_shape: Vec<usize> =
                tensor_info.dimensions.iter().map(|&d| d as usize).collect();
            if tensor_shape != *expected_shape {
                return Err(FerrumError::invalid_format(format!(
                    "Shape mismatch for {}: expected {:?}, got {:?}",
                    spec.name, expected_shape, tensor_shape
                )));
            }
        }

        // Check data type compatibility
        let ferrum_dtype = tensor_info.ggml_type.to_ferrum_dtype();
        if let Some(expected_dtype) = spec.dtype {
            if ferrum_dtype != expected_dtype && !tensor_info.ggml_type.is_quantized() {
                warn!(
                    "Data type mismatch for {}: expected {:?}, got {:?} (GGML: {:?})",
                    spec.name, expected_dtype, ferrum_dtype, tensor_info.ggml_type
                );
            }
        }

        // Load tensor data
        let file_path = self.file_path.read().as_ref().unwrap().clone();
        let file = tokio::fs::File::open(&file_path)
            .await
            .map_err(|e| FerrumError::io(format!("Failed to open GGUF file: {}", e)))?;

        use tokio::io::{AsyncReadExt, AsyncSeekExt};
        let mut file = file;
        file.seek(std::io::SeekFrom::Start(tensor_info.offset))
            .await
            .map_err(|e| FerrumError::io(format!("Failed to seek in GGUF file: {}", e)))?;

        // Calculate tensor size
        let element_count: u64 = tensor_info.dimensions.iter().product();
        let bytes_per_elem = tensor_info.ggml_type.bytes_per_element();

        let tensor_size = if tensor_info.ggml_type.is_quantized() {
            // For quantized types, calculate based on quantization scheme
            match tensor_info.ggml_type {
                GGMLType::Q4_0 | GGMLType::Q4_1 => (element_count / 32) * bytes_per_elem as u64,
                GGMLType::Q5_0 | GGMLType::Q5_1 => (element_count / 32) * bytes_per_elem as u64,
                GGMLType::Q8_0 | GGMLType::Q8_1 => (element_count / 32) * bytes_per_elem as u64,
                _ => element_count * bytes_per_elem as u64, // K-quants and others
            }
        } else {
            element_count * bytes_per_elem as u64
        };

        let mut buffer = vec![0u8; tensor_size as usize];
        file.read_exact(&mut buffer)
            .await
            .map_err(|e| FerrumError::io(format!("Failed to read GGUF tensor data: {}", e)))?;

        debug!(
            "Successfully loaded GGUF tensor: {} ({} bytes, type: {:?})",
            spec.name,
            buffer.len(),
            tensor_info.ggml_type
        );

        Ok(())
    }

    fn supported_formats(&self) -> Vec<String> {
        vec!["gguf".to_string()]
    }

    fn name(&self) -> &str {
        "gguf"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggml_type_conversion() {
        assert_eq!(GGMLType::F32.to_ferrum_dtype(), DataType::F32);
        assert_eq!(GGMLType::F16.to_ferrum_dtype(), DataType::F16);
        assert_eq!(GGMLType::BF16.to_ferrum_dtype(), DataType::BF16);

        // Quantized types should map to F16
        assert_eq!(GGMLType::Q4_0.to_ferrum_dtype(), DataType::F16);
        assert!(GGMLType::Q4_0.is_quantized());
        assert!(!GGMLType::F32.is_quantized());
    }

    #[test]
    fn test_ggml_type_from_id() {
        assert_eq!(GGMLType::from_type_id(0).unwrap(), GGMLType::F32);
        assert_eq!(GGMLType::from_type_id(1).unwrap(), GGMLType::F16);
        assert_eq!(GGMLType::from_type_id(2).unwrap(), GGMLType::Q4_0);

        assert!(GGMLType::from_type_id(999).is_err());
    }

    #[test]
    fn test_loader_creation() {
        let loader = GGUFLoader::new();
        assert_eq!(loader.name(), "gguf");
        assert_eq!(loader.supported_formats(), vec!["gguf"]);
    }

    #[test]
    fn test_bytes_per_element() {
        assert_eq!(GGMLType::F32.bytes_per_element(), 4);
        assert_eq!(GGMLType::F16.bytes_per_element(), 2);
        assert_eq!(GGMLType::Q4_0.bytes_per_element(), 18); // 32 elements in 18 bytes
    }
}
