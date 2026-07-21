//! Weight-format abstraction for Ferrum models.
//!
//! Separates "what is the weight matrix like" (dense f32, GPTQ int4, AWQ,
//! GGUF, ...) from "what device does the math" (Backend) and "how does the
//! model wire things together" (model code).
//!
//! Usage in model code:
//! ```ignore
//! let qkv: Box<dyn Linear<B>> = loader.load_linear("model.layers.0.self_attn.qkv_proj")?;
//! qkv.forward(ctx, &input, &mut out, m);
//! ```
//!
//! The `Linear` trait dispatches to the appropriate backend kernel
//! (`B::gemm` for Dense, `B::gemm_gptq` for GPTQ, etc.) without the model
//! having to branch on quantization type.

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod dense;
pub mod gguf;
pub mod gptq;
pub mod gptq_marlin_source;
pub mod loader;
pub mod lora;
pub mod native_safetensors;
pub mod quant_linear;
pub mod safetensors_archive;
pub mod traits;

pub use dense::DenseLinear;
pub use gguf::{GgufFile, GgufLinear, GgufLoader, GgufWeightComponentSource};
pub use gptq::{GptqLinear, StackedExpertLinear};
pub use gptq_marlin_source::{GptqMarlinSafetensorsSource, GPTQ_MARLIN_INT4_FORMAT_ID};
pub use loader::{PrefixedLoader, WeightLoader};
pub use lora::LoraLinearRef;
pub use native_safetensors::NativeSafetensorsLoader;
pub use quant_linear::QuantLinear;
pub use safetensors_archive::{SafetensorsArchive, SafetensorsTensor};
pub use traits::Linear;

// Quant config types — populated from safetensors metadata or GGUF header.
pub mod config;
pub use config::{QuantConfig, QuantMethod};
