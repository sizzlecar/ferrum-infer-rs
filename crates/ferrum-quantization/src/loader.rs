//! `WeightLoader` trait — unified interface for loading tensor/linear weights
//! into a specific backend.
//!
//! Implementations (landing in Phase B):
//!   - `SafeTensorsLoader` — reads `.safetensors` files, returns `DenseLinear`
//!     unless `quantize_config.json` indicates GPTQ/AWQ, in which case it
//!     returns `GptqLinear` / `AwqLinear`.
//!   - `GgufLoader` — reads `.gguf` files, returns `GgufLinear`.
//!
//! The trait is generic over `B: Backend` so the loader can materialise
//! tensors directly into backend-native buffers (zero-copy on Apple Silicon
//! shared memory, dtoh/htod for CUDA, etc.).

use ferrum_kernels::backend::Backend;
use ferrum_types::Result;

use crate::config::QuantConfig;
use crate::traits::Linear;

pub trait WeightLoader<B: Backend>: Send + Sync {
    /// Load a single tensor by fully qualified name
    /// (e.g. `"model.embed_tokens.weight"`).
    fn load_tensor(&self, name: &str) -> Result<B::Buffer>;

    /// Load a projection as a `Linear<B>`. The concrete implementation
    /// (DenseLinear / GptqLinear / AwqLinear / GgufLinear) depends on the
    /// loader's file format and quant config.
    ///
    /// `name` is the module path without the `.weight` suffix, e.g.
    /// `"model.layers.0.self_attn.qkv_proj"`.
    fn load_linear(&self, name: &str) -> Result<Box<dyn Linear<B>>>;

    /// Whether a tensor with this name exists in the source.
    fn has_tensor(&self, name: &str) -> bool;

    /// Quantization metadata (parsed from `quantize_config.json` or a GGUF header).
    /// `None` means the source is dense.
    fn quant_config(&self) -> Option<&QuantConfig>;
}
