//! Weight-format abstraction for Ferrum models.
//!
//! Separates "what is the weight matrix like" (dense f32, GPTQ int4, AWQ,
//! GGUF, ...) from "what device does the math" (Backend) and "how does the
//! model wire things together" (model code).
//!
//! Usage in model code:
//! ```ignore
//! let qkv: ferrum_kernels::LoadedLinear<B> = loader.load_linear(ctx, "model.layers.0.self_attn.qkv_proj")?;
//! qkv.forward(ctx, &input, &mut out, m);
//! ```
//!
//! The `Linear` trait dispatches to the appropriate backend kernel
//! (`B::gemm` for Dense, `B::gemm_gptq` for GPTQ, etc.) without the model
//! having to branch on quantization type.

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod bf16_whole_tensor_fp8;
pub mod block_fp8_safetensors_source;
pub mod compressed_tensors_marlin_source;
pub mod dense;
pub mod gguf;
pub mod gptq;
pub mod gptq_marlin_source;
pub mod linear_adapter;
pub mod loader;
pub mod lora;
pub mod mxfp4_safetensors_source;
pub mod native_safetensors;
pub mod quant_linear;
pub mod safetensors_archive;
pub mod tensor_archive;
pub mod traits;

pub use bf16_whole_tensor_fp8::Bf16WholeTensorFp8Materializer;
pub use block_fp8_safetensors_source::{
    BlockFp8SafetensorsSource, BLOCK_FP8_E4M3_SOURCE_FORMAT_ID,
};
pub use compressed_tensors_marlin_source::{
    CompressedTensorsMarlinSafetensorsSource, COMPRESSED_TENSORS_MARLIN_INT4_FORMAT_ID,
    COMPRESSED_TENSORS_MARLIN_INT4_SYMMETRIC_FORMAT_ID,
};
pub use dense::DenseLinear;
pub use gguf::{GgufFile, GgufLinear, GgufLoader, GgufWeightComponentSource};
pub use gptq::{GptqLinear, StackedExpertLinear};
pub use gptq_marlin_source::{GptqMarlinSafetensorsSource, GPTQ_MARLIN_INT4_FORMAT_ID};
pub use loader::{PrefixedLoader, WeightLoader};
pub use lora::LoraLinearRef;
pub use mxfp4_safetensors_source::{Mxfp4SafetensorsSource, MXFP4_E2M1_E8M0_SOURCE_FORMAT_ID};
pub use native_safetensors::{NativeSafetensorsLoadOptions, NativeSafetensorsLoader};
pub use quant_linear::QuantLinear;
pub use safetensors_archive::{SafetensorsArchive, SafetensorsReadAheadPolicy, SafetensorsTensor};
pub use tensor_archive::{ArchiveTensor, ManagedDenseArchiveSource, TensorArchive};
pub use traits::Linear;

// Quant config types — populated from safetensors metadata or GGUF header.
pub mod config;
pub use config::{QuantConfig, QuantMethod};
