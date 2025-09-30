//! Backend abstraction split into compute and weight loading concerns
//!
//! This module separates the previous "fat" Backend trait into focused
//! interfaces: ComputeBackend for tensor operations and WeightLoader for
//! model weight management.

use crate::{TensorFactory, TensorOps, TensorRef};
use async_trait::async_trait;
use ferrum_types::{DataType, Device, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Compute backend for tensor operations and kernel execution
#[async_trait]
pub trait ComputeBackend: Send + Sync {
    /// Get backend name/identifier
    fn name(&self) -> &str;

    /// Get backend capabilities
    fn capabilities(&self) -> BackendCapabilities;

    /// Get tensor operations interface
    fn tensor_ops(&self) -> &dyn TensorOps;

    /// Get tensor factory for creating tensors
    fn tensor_factory(&self) -> &dyn TensorFactory;

    /// Get memory manager for this backend
    fn memory_manager(&self) -> &dyn crate::DeviceMemoryManager;

    /// Get kernel executor (if backend supports custom kernels)
    fn kernel_executor(&self) -> Option<&dyn KernelExecutor>;

    /// Initialize backend with device
    async fn initialize(&mut self, device: &Device) -> Result<()>;

    /// Check if backend supports specific device
    fn supports_device(&self, device: &Device) -> bool;

    /// Get backend version
    fn version(&self) -> String;

    /// Synchronize all pending operations
    async fn synchronize(&self, device: &Device) -> Result<()>;

    /// Get backend status
    fn status(&self) -> BackendStatus;

    /// Shutdown backend gracefully
    async fn shutdown(&mut self) -> Result<()>;
}

/// Weight loading interface for model parameter management
#[async_trait]
pub trait WeightLoader: Send + Sync {
    /// Load tensor from weight specification
    async fn load_tensor(&self, spec: &TensorSpec) -> Result<TensorRef>;

    /// Load multiple tensors at once (batch loading)
    async fn load_tensors(&self, specs: &[TensorSpec]) -> Result<Vec<TensorRef>>;

    /// Check if weight source is available
    async fn is_available(&self, source: &WeightSource) -> bool;

    /// Get metadata about weight source
    async fn get_metadata(&self, source: &WeightSource) -> Result<WeightMetadata>;

    /// Preload weights into cache/memory
    async fn preload(&self, source: &WeightSource) -> Result<()>;

    /// Get loader capabilities
    fn capabilities(&self) -> WeightLoaderCapabilities;
}

/// Tensor specification for weight loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    /// Name/identifier of the tensor
    pub name: String,
    /// Expected tensor shape
    pub shape: Vec<usize>,
    /// Target data type
    pub dtype: DataType,
    /// Target device
    pub device: Device,
    /// Weight source location
    pub source: WeightSource,
    /// Optional transformations to apply
    pub transformations: Vec<TensorTransformation>,
}

/// Weight source specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WeightSource {
    /// Local file path
    File {
        path: String,
        /// Tensor name within file (for formats like safetensors)
        tensor_name: Option<String>,
    },
    /// URL for download
    Url {
        url: String,
        headers: HashMap<String, String>,
    },
    /// Hugging Face Hub
    HuggingFace {
        repo_id: String,
        filename: String,
        revision: Option<String>,
        cache_dir: Option<String>,
    },
    /// Raw bytes in memory
    Memory { data: Vec<u8>, format: WeightFormat },
    /// S3-compatible storage
    S3 {
        bucket: String,
        key: String,
        region: Option<String>,
        endpoint: Option<String>,
    },
}

/// Weight file formats
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum WeightFormat {
    /// PyTorch tensor format
    PyTorch,
    /// Safetensors format
    SafeTensors,
    /// NumPy array format
    Numpy,
    /// Raw binary data
    Raw,
    /// ONNX format
    Onnx,
    /// Custom format
    Custom(u32),
}

/// Weight metadata information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightMetadata {
    /// Available tensor names and their shapes
    pub tensors: HashMap<String, Vec<usize>>,
    /// File format
    pub format: WeightFormat,
    /// Total size in bytes
    pub total_size_bytes: u64,
    /// Data types used
    pub dtypes: Vec<DataType>,
    /// Additional metadata
    pub extra: HashMap<String, serde_json::Value>,
}

/// Transformations that can be applied to loaded tensors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorTransformation {
    /// Transpose dimensions
    Transpose { dim0: usize, dim1: usize },
    /// Reshape to new shape
    Reshape { shape: Vec<usize> },
    /// Convert data type
    Cast { dtype: DataType },
    /// Quantize tensor
    Quantize { config: QuantizationConfig },
    /// Apply scaling
    Scale { factor: f32 },
    /// Slice tensor
    Slice {
        dim: usize,
        start: Option<usize>,
        end: Option<usize>,
    },
}

/// Quantization configuration for weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationConfig {
    /// INT8 uniform quantization
    INT8 { symmetric: bool },
    /// INT4 grouped quantization  
    INT4 { group_size: usize },
    /// FP8 quantization
    FP8 { e4m3: bool },
    /// GPTQ quantization
    GPTQ {
        bits: u8,
        group_size: usize,
        desc_act: bool,
    },
    /// AWQ quantization
    AWQ { bits: u8, zero_point: bool },
}

/// Backend capabilities description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendCapabilities {
    /// Supported data types
    pub supported_dtypes: Vec<DataType>,
    /// Supported devices
    pub supported_devices: Vec<Device>,
    /// Maximum tensor dimensions supported
    pub max_tensor_dims: usize,
    /// Whether backend supports FP16 operations
    pub supports_fp16: bool,
    /// Whether backend supports BF16 operations
    pub supports_bf16: bool,
    /// Whether backend supports INT8 quantization
    pub supports_int8: bool,
    /// Whether backend supports flash attention
    pub supports_flash_attention: bool,
    /// Whether backend supports paged attention
    pub supports_paged_attention: bool,
    /// Whether backend supports tensor parallelism
    pub supports_tensor_parallelism: bool,
    /// Whether backend supports pipeline parallelism
    pub supports_pipeline_parallelism: bool,
    /// Maximum batch size supported
    pub max_batch_size: usize,
    /// Maximum sequence length supported
    pub max_sequence_length: usize,
    /// Memory alignment requirements
    pub memory_alignment: usize,
    /// Whether backend supports custom kernels
    pub supports_custom_kernels: bool,
    /// Whether backend supports CUDA graphs
    pub supports_cuda_graphs: bool,
    /// Additional capabilities
    pub extra_capabilities: HashMap<String, serde_json::Value>,
}

impl BackendCapabilities {
    /// Check if capabilities meet requirements
    pub fn meets_requirements(&self, requirements: &BackendRequirements) -> bool {
        // Check devices
        if !requirements
            .required_devices
            .iter()
            .all(|dev| self.supported_devices.contains(dev))
        {
            return false;
        }

        // Check dtypes
        if !requirements
            .required_dtypes
            .iter()
            .all(|dtype| self.supported_dtypes.contains(dtype))
        {
            return false;
        }

        // Check batch size
        if requirements.min_batch_size > self.max_batch_size {
            return false;
        }

        // Check sequence length
        if requirements.min_sequence_length > self.max_sequence_length {
            return false;
        }

        true
    }
}

/// Requirements for backend selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendRequirements {
    /// Required devices
    pub required_devices: Vec<Device>,
    /// Required data types
    pub required_dtypes: Vec<DataType>,
    /// Minimum batch size needed
    pub min_batch_size: usize,
    /// Minimum sequence length needed
    pub min_sequence_length: usize,
    /// Whether flash attention is required
    pub requires_flash_attention: bool,
    /// Whether paged attention is required
    pub requires_paged_attention: bool,
    /// Additional requirements
    pub extra_requirements: HashMap<String, serde_json::Value>,
}

/// Weight loader capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightLoaderCapabilities {
    /// Supported weight formats
    pub supported_formats: Vec<WeightFormat>,
    /// Supported weight sources
    pub supported_sources: Vec<WeightSourceType>,
    /// Maximum single tensor size in bytes
    pub max_tensor_size: u64,
    /// Whether loader supports streaming/chunked loading
    pub supports_streaming: bool,
    /// Whether loader supports concurrent loading
    pub supports_concurrent: bool,
    /// Supported transformations
    pub supported_transformations: Vec<TransformationType>,
}

/// Weight source types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum WeightSourceType {
    File,
    Url,
    HuggingFace,
    Memory,
    S3,
}

/// Transformation types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TransformationType {
    Transpose,
    Reshape,
    Cast,
    Quantize,
    Scale,
    Slice,
}

/// Backend status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendStatus {
    /// Whether backend is initialized
    pub is_initialized: bool,
    /// Whether backend is ready for operations
    pub is_ready: bool,
    /// Currently active devices
    pub active_devices: Vec<Device>,
    /// Memory usage per device
    pub memory_usage: HashMap<Device, u64>,
    /// Number of operations completed
    pub operations_completed: u64,
    /// Last error (if any)
    pub last_error: Option<String>,
    /// Backend-specific status information
    pub backend_specific: HashMap<String, serde_json::Value>,
}

/// Kernel executor for custom GPU kernels
#[async_trait]
pub trait KernelExecutor: Send + Sync {
    /// Load kernel from source code
    async fn load_kernel(&self, source: &str, name: &str, device: &Device) -> Result<KernelHandle>;

    /// Execute kernel with arguments
    async fn execute_kernel(
        &self,
        handle: KernelHandle,
        grid_size: (u32, u32, u32),
        block_size: (u32, u32, u32),
        args: &[KernelArg],
    ) -> Result<()>;

    /// Get kernel information
    fn get_kernel_info(&self, handle: KernelHandle) -> Option<KernelInfo>;

    /// Unload kernel
    async fn unload_kernel(&self, handle: KernelHandle) -> Result<()>;
}

/// Handle for loaded kernel
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelHandle(pub u64);

/// Kernel argument types
#[derive(Debug, Clone)]
pub enum KernelArg {
    /// Tensor reference
    Tensor(TensorRef),
    /// Raw memory buffer
    Buffer { ptr: *const u8, size: usize },
    /// Scalar value
    Scalar(ScalarValue),
    /// Local/shared memory allocation
    LocalMemory(usize),
}

/// Scalar values for kernel arguments
#[derive(Debug, Clone)]
pub enum ScalarValue {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
    Bool(bool),
}

/// Kernel information and metadata
#[derive(Debug, Clone)]
pub struct KernelInfo {
    /// Kernel name
    pub name: String,
    /// Maximum threads per block
    pub max_threads_per_block: u32,
    /// Shared memory size required
    pub shared_memory_size: usize,
    /// Register count per thread
    pub registers_per_thread: u32,
    /// Preferred block size
    pub preferred_block_size: (u32, u32, u32),
}

/// Backend factory for creating backend instances
#[async_trait]
pub trait BackendFactory: Send + Sync {
    /// Create compute backend
    async fn create_compute_backend(
        &self,
        config: &BackendConfig,
    ) -> Result<Box<dyn ComputeBackend>>;

    /// Create weight loader
    async fn create_weight_loader(
        &self,
        config: &WeightLoaderConfig,
    ) -> Result<Box<dyn WeightLoader>>;

    /// Get supported backend types
    fn supported_backend_types(&self) -> Vec<BackendType>;

    /// Validate backend configuration
    fn validate_config(&self, config: &BackendConfig) -> Result<()>;
}

/// Backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    /// Backend type
    pub backend_type: BackendType,
    /// Target device
    pub device: Device,
    /// Optimization level (0-3)
    pub optimization_level: u8,
    /// Enable debugging
    pub enable_debug: bool,
    /// Memory configuration
    pub memory_config: BackendMemoryConfig,
    /// Backend-specific options
    pub backend_options: HashMap<String, serde_json::Value>,
}

/// Weight loader configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightLoaderConfig {
    /// Enable caching
    pub enable_caching: bool,
    /// Cache directory
    pub cache_dir: Option<String>,
    /// Maximum cache size in bytes
    pub max_cache_size: Option<u64>,
    /// Number of concurrent downloads
    pub max_concurrent_downloads: usize,
    /// Connection timeout for downloads
    pub download_timeout_seconds: u64,
    /// Enable integrity checks
    pub enable_integrity_checks: bool,
    /// Custom headers for HTTP requests
    pub default_headers: HashMap<String, String>,
}

/// Memory configuration for backends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendMemoryConfig {
    /// Memory pool size in bytes (None for auto)
    pub pool_size: Option<u64>,
    /// Memory alignment in bytes
    pub alignment: usize,
    /// Enable memory pooling
    pub enable_pooling: bool,
    /// Memory growth strategy
    pub growth_strategy: MemoryGrowthStrategy,
}

/// Memory growth strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MemoryGrowthStrategy {
    /// Pre-allocate all memory upfront
    Static,
    /// Grow memory as needed
    Dynamic,
    /// Pre-allocate with incremental growth
    Incremental,
}

/// Backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendType {
    /// Candle framework
    Candle,
    /// ONNX Runtime
    OnnxRuntime,
    /// TensorRT
    TensorRT,
    /// Custom Metal implementation
    Metal,
    /// Custom CPU implementation
    CPU,
    /// Custom backend
    Custom,
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            BackendType::Candle => "candle",
            BackendType::OnnxRuntime => "onnx_runtime",
            BackendType::TensorRT => "tensorrt",
            BackendType::Metal => "metal",
            BackendType::CPU => "cpu",
            BackendType::Custom => "custom",
        };
        write!(f, "{}", name)
    }
}

/// Backend registry for managing multiple backends
pub trait BackendRegistry: Send + Sync {
    /// Register compute backend
    fn register_compute_backend(
        &mut self,
        name: &str,
        backend: Box<dyn ComputeBackend>,
    ) -> Result<()>;

    /// Register weight loader
    fn register_weight_loader(&mut self, name: &str, loader: Box<dyn WeightLoader>) -> Result<()>;

    /// Get compute backend by name
    fn get_compute_backend(&self, name: &str) -> Option<&dyn ComputeBackend>;

    /// Get weight loader by name
    fn get_weight_loader(&self, name: &str) -> Option<&dyn WeightLoader>;

    /// Find best compute backend for requirements
    fn find_best_compute_backend(
        &self,
        requirements: &BackendRequirements,
    ) -> Option<&dyn ComputeBackend>;

    /// List all registered backend names
    fn list_backend_names(&self) -> (Vec<String>, Vec<String>); // (compute, weight)
}
