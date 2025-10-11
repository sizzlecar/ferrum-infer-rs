//! Model builder interface for constructing model executors
//!
//! This module provides interfaces for building model executors from
//! configurations and weight sources, separating model construction
//! from backend implementation.

use crate::{ComputeBackend, ModelExecutor, WeightLoader};
use async_trait::async_trait;
use ferrum_types::{ModelConfig, ModelInfo, ModelSource, Result};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

/// Model builder for constructing model executors
#[async_trait]
pub trait ModelBuilder: Send + Sync {
    /// Build model executor from configuration
    async fn build_model(
        &self,
        config: &ModelConfig,
        compute_backend: Arc<dyn ComputeBackend>,
        weight_loader: Arc<dyn WeightLoader>,
    ) -> Result<Box<dyn ModelExecutor>>;

    /// Build model executor from model source
    async fn build_from_source(
        &self,
        source: &ModelSource,
        compute_backend: Arc<dyn ComputeBackend>,
        weight_loader: Arc<dyn WeightLoader>,
        build_options: &BuildOptions,
    ) -> Result<Box<dyn ModelExecutor>>;

    /// Validate model configuration
    fn validate_config(&self, config: &ModelConfig) -> Result<Vec<ValidationIssue>>;

    /// Get supported model types
    fn supported_model_types(&self) -> Vec<ferrum_types::ModelType>;

    /// Get estimated build time
    async fn estimate_build_time(&self, config: &ModelConfig) -> Result<BuildTimeEstimate>;

    /// Get builder information
    fn builder_info(&self) -> BuilderInfo;
}

/// Build options for model construction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildOptions {
    /// Enable model validation after build
    pub enable_validation: bool,
    /// Enable model optimization
    pub enable_optimization: bool,
    /// Optimization level (0-3)
    pub optimization_level: u8,
    /// Enable model quantization
    pub enable_quantization: bool,
    /// Quantization configuration
    pub quantization_config: Option<ferrum_types::QuantizationConfig>,
    /// Enable model compression
    pub enable_compression: bool,
    /// Build timeout in seconds
    pub build_timeout_seconds: Option<u64>,
    /// Additional build options
    pub additional_options: HashMap<String, serde_json::Value>,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            enable_validation: true,
            enable_optimization: true,
            optimization_level: 2,
            enable_quantization: false,
            quantization_config: None,
            enable_compression: false,
            build_timeout_seconds: Some(3600), // 1 hour
            additional_options: HashMap::new(),
        }
    }
}

/// Validation issue found during configuration validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    /// Issue severity
    pub severity: ValidationSeverity,
    /// Issue category
    pub category: String,
    /// Issue description
    pub description: String,
    /// Suggested fix
    pub suggested_fix: Option<String>,
    /// Configuration path where issue was found
    pub config_path: String,
}

/// Validation issue severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ValidationSeverity {
    /// Warning that doesn't prevent build
    Warning,
    /// Error that prevents build
    Error,
    /// Critical error that indicates serious misconfiguration
    Critical,
}

/// Build time estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildTimeEstimate {
    /// Estimated minimum build time
    pub min_time_seconds: u64,
    /// Estimated maximum build time
    pub max_time_seconds: u64,
    /// Most likely build time
    pub expected_time_seconds: u64,
    /// Breakdown of build time by phase
    pub time_breakdown: BuildTimeBreakdown,
    /// Factors affecting build time
    pub factors: Vec<BuildTimeFactor>,
}

/// Build time breakdown by phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildTimeBreakdown {
    /// Time for weight loading
    pub weight_loading_seconds: u64,
    /// Time for model initialization
    pub model_init_seconds: u64,
    /// Time for optimization
    pub optimization_seconds: u64,
    /// Time for validation
    pub validation_seconds: u64,
    /// Other overhead time
    pub overhead_seconds: u64,
}

/// Factor affecting build time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildTimeFactor {
    /// Factor name
    pub factor: String,
    /// Impact on build time (multiplier)
    pub impact: f32,
    /// Description
    pub description: String,
}

/// Builder information and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuilderInfo {
    /// Builder name
    pub name: String,
    /// Builder version
    pub version: String,
    /// Supported model architectures
    pub supported_architectures: Vec<ModelArchitecture>,
    /// Supported weight formats
    pub supported_weight_formats: Vec<crate::backend::WeightFormat>,
    /// Supported optimization techniques
    pub supported_optimizations: Vec<OptimizationTechnique>,
    /// Builder capabilities
    pub capabilities: BuilderCapabilities,
}

/// Model architecture types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArchitecture {
    /// Architecture name
    pub name: String,
    /// Architecture family
    pub family: ModelArchitectureFamily,
    /// Supported variants
    pub variants: Vec<String>,
    /// Required features
    pub required_features: Vec<String>,
}

/// Model architecture families
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelArchitectureFamily {
    /// Transformer-based models
    Transformer,
    /// Convolutional neural networks
    CNN,
    /// Recurrent neural networks
    RNN,
    /// Graph neural networks
    GNN,
    /// Diffusion models
    Diffusion,
    /// Custom architecture
    Custom,
}

/// Optimization techniques
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OptimizationTechnique {
    /// Operator fusion
    OperatorFusion,
    /// Constant folding
    ConstantFolding,
    /// Dead code elimination
    DeadCodeElimination,
    /// Memory layout optimization
    MemoryLayoutOptimization,
    /// Kernel selection optimization
    KernelSelection,
    /// Quantization
    Quantization,
    /// Pruning
    Pruning,
    /// Knowledge distillation
    Distillation,
}

/// Builder capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuilderCapabilities {
    /// Maximum model size supported (parameters)
    pub max_model_size: Option<u64>,
    /// Supports dynamic shapes
    pub supports_dynamic_shapes: bool,
    /// Supports custom operations
    pub supports_custom_ops: bool,
    /// Supports mixed precision
    pub supports_mixed_precision: bool,
    /// Supports model parallelism
    pub supports_model_parallelism: bool,
    /// Parallel build support
    pub supports_parallel_build: bool,
    /// Incremental build support
    pub supports_incremental_build: bool,
}

/// Advanced model builder with additional capabilities
#[async_trait]
pub trait AdvancedModelBuilder: ModelBuilder {
    /// Build model with custom layers
    async fn build_with_custom_layers(
        &self,
        config: &ModelConfig,
        custom_layers: Vec<Box<dyn CustomLayer>>,
        compute_backend: Arc<dyn ComputeBackend>,
        weight_loader: Arc<dyn WeightLoader>,
    ) -> Result<Box<dyn ModelExecutor>>;

    /// Build model incrementally (for large models)
    async fn build_incremental(
        &self,
        config: &ModelConfig,
        compute_backend: Arc<dyn ComputeBackend>,
        weight_loader: Arc<dyn WeightLoader>,
        progress_callback: Box<dyn Fn(BuildProgress) + Send + Sync>,
    ) -> Result<Box<dyn ModelExecutor>>;

    /// Build model with custom optimization pipeline
    async fn build_with_optimization(
        &self,
        config: &ModelConfig,
        optimization_pipeline: Vec<Box<dyn OptimizationPass>>,
        compute_backend: Arc<dyn ComputeBackend>,
        weight_loader: Arc<dyn WeightLoader>,
    ) -> Result<Box<dyn ModelExecutor>>;

    /// Export model definition for debugging
    async fn export_model_definition(&self, config: &ModelConfig) -> Result<ModelIR>;

    /// Import model definition for custom builds
    async fn import_model_definition(
        &self,
        definition: &ModelIR,
        compute_backend: Arc<dyn ComputeBackend>,
        weight_loader: Arc<dyn WeightLoader>,
    ) -> Result<Box<dyn ModelExecutor>>;
}

/// Custom layer interface for advanced builders
pub trait CustomLayer: Send + Sync {
    /// Get layer name
    fn name(&self) -> &str;

    /// Get layer type
    fn layer_type(&self) -> &str;

    /// Get input shape requirements
    fn input_shape(&self) -> Vec<i64>; // -1 for dynamic dimensions

    /// Get output shape
    fn output_shape(&self, input_shape: &[i64]) -> Vec<i64>;

    /// Initialize layer parameters
    fn initialize_parameters(&self) -> Result<HashMap<String, crate::TensorRef>>;

    /// Get layer configuration
    fn config(&self) -> serde_json::Value;
}

/// Build progress information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildProgress {
    /// Current build phase
    pub phase: BuildPhase,
    /// Progress percentage (0.0 - 1.0)
    pub progress: f32,
    /// Current operation description
    pub current_operation: String,
    /// Elapsed time in seconds
    pub elapsed_seconds: u64,
    /// Estimated remaining time in seconds
    pub remaining_seconds: Option<u64>,
    /// Phase-specific details
    pub phase_details: HashMap<String, serde_json::Value>,
}

/// Build phases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuildPhase {
    /// Configuration validation
    Validation,
    /// Weight loading
    WeightLoading,
    /// Model initialization
    ModelInitialization,
    /// Layer construction
    LayerConstruction,
    /// Parameter binding
    ParameterBinding,
    /// Model optimization
    Optimization,
    /// Final validation
    FinalValidation,
    /// Cleanup and finalization
    Finalization,
}

/// Optimization pass for custom optimization pipelines
pub trait OptimizationPass: Send + Sync {
    /// Get optimization pass name
    fn name(&self) -> &str;

    /// Apply optimization to model definition
    fn apply(&self, definition: &mut ModelIR) -> Result<OptimizationResult>;

    /// Check if optimization is applicable
    fn is_applicable(&self, definition: &ModelIR) -> bool;

    /// Get optimization dependencies (must run before this)
    fn dependencies(&self) -> Vec<String>;
}

/// Optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Whether optimization was applied
    pub applied: bool,
    /// Optimization statistics
    pub stats: OptimizationStats,
    /// Warnings or issues
    pub warnings: Vec<String>,
}

/// Optimization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStats {
    /// Parameters eliminated
    pub parameters_eliminated: u64,
    /// Operations eliminated
    pub operations_eliminated: usize,
    /// Memory saved (bytes)
    pub memory_saved: u64,
    /// Estimated speedup
    pub estimated_speedup: f32,
}

/// Model IR (Intermediate Representation) for export/import
/// 
/// Note: This is different from ferrum_models::ModelDefinition which is used
/// for parsing HuggingFace config.json files. This type represents a complete
/// model definition including computational graph for model building/export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelIR {
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Model architecture definition
    pub architecture: ArchitectureDefinition,
    /// Parameter specifications
    pub parameters: Vec<ParameterSpec>,
    /// Layer definitions
    pub layers: Vec<LayerDefinition>,
    /// Model graph/connectivity
    pub graph: GraphDefinition,
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Model type
    pub model_type: ferrum_types::ModelType,
    /// Architecture family
    pub architecture_family: ModelArchitectureFamily,
    /// Model description
    pub description: Option<String>,
    /// Author information
    pub author: Option<String>,
    /// License information
    pub license: Option<String>,
    /// Additional metadata
    pub additional: HashMap<String, serde_json::Value>,
}

/// Architecture definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureDefinition {
    /// Architecture name
    pub name: String,
    /// Model dimensions
    pub dimensions: ModelDimensions,
    /// Architecture-specific configuration
    pub config: HashMap<String, serde_json::Value>,
}

/// Model dimensions and hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDimensions {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden/embedding dimension
    pub hidden_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of key-value heads (for GQA/MQA)
    pub num_kv_heads: Option<usize>,
    /// Intermediate/FFN dimension
    pub intermediate_size: Option<usize>,
    /// Maximum sequence length
    pub max_sequence_length: usize,
}

/// Parameter specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSpec {
    /// Parameter name
    pub name: String,
    /// Parameter shape
    pub shape: Vec<i64>,
    /// Data type
    pub dtype: ferrum_types::DataType,
    /// Whether parameter is trainable
    pub trainable: bool,
    /// Initialization strategy
    pub initialization: InitializationStrategy,
    /// Additional parameter metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Parameter initialization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InitializationStrategy {
    /// Zero initialization
    Zeros,
    /// One initialization
    Ones,
    /// Uniform random initialization
    Uniform { min: f32, max: f32 },
    /// Normal/Gaussian initialization
    Normal { mean: f32, std: f32 },
    /// Xavier/Glorot initialization
    Xavier,
    /// Kaiming/He initialization
    Kaiming,
    /// Custom initialization
    Custom(String),
}

/// Layer definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerDefinition {
    /// Layer name
    pub name: String,
    /// Layer type
    pub layer_type: String,
    /// Input specifications
    pub inputs: Vec<TensorSpec>,
    /// Output specifications
    pub outputs: Vec<TensorSpec>,
    /// Layer parameters
    pub parameters: Vec<String>, // Parameter names
    /// Layer configuration
    pub config: HashMap<String, serde_json::Value>,
}

/// Tensor specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    /// Tensor name
    pub name: String,
    /// Tensor shape (-1 for dynamic dimensions)
    pub shape: Vec<i64>,
    /// Data type
    pub dtype: ferrum_types::DataType,
}

/// Model graph definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphDefinition {
    /// Input nodes
    pub inputs: Vec<String>,
    /// Output nodes
    pub outputs: Vec<String>,
    /// Graph nodes (layers)
    pub nodes: Vec<GraphNode>,
    /// Graph edges (connections)
    pub edges: Vec<GraphEdge>,
}

/// Graph node representing a layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Node ID
    pub id: String,
    /// Layer name
    pub layer_name: String,
    /// Node metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Graph edge representing a connection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Source output index
    pub source_output: Option<usize>,
    /// Target input index
    pub target_input: Option<usize>,
}

/// Model builder factory
#[async_trait]
pub trait ModelBuilderFactory: Send + Sync {
    /// Create standard model builder
    async fn create_builder(&self) -> Result<Box<dyn ModelBuilder>>;

    /// Create advanced model builder
    async fn create_advanced_builder(&self) -> Result<Box<dyn AdvancedModelBuilder>>;

    /// Get supported model types
    fn supported_types(&self) -> Vec<ferrum_types::ModelType>;

    /// Create builder for specific model type
    async fn create_builder_for_type(
        &self,
        model_type: ferrum_types::ModelType,
    ) -> Result<Box<dyn ModelBuilder>>;
}

/// Model registry for managing built models
pub trait ModelRegistry: Send + Sync {
    /// Register model executor
    fn register_model(
        &mut self,
        model_id: &ferrum_types::ModelId,
        executor: Box<dyn ModelExecutor>,
    ) -> Result<()>;

    /// Get model executor
    fn get_model(&self, model_id: &ferrum_types::ModelId) -> Option<&dyn ModelExecutor>;

    /// Remove model executor
    fn remove_model(&mut self, model_id: &ferrum_types::ModelId) -> Option<Box<dyn ModelExecutor>>;

    /// List registered models
    fn list_models(&self) -> Vec<ferrum_types::ModelId>;

    /// Get model information
    fn get_model_info(&self, model_id: &ferrum_types::ModelId) -> Option<&ModelInfo>;

    /// Check if model exists
    fn contains_model(&self, model_id: &ferrum_types::ModelId) -> bool;
}
