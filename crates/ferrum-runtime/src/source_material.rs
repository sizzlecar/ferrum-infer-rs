//! Crate-owned source inputs for native provider declaration fingerprints.
//!
//! Consumers obtain these bytes through their Rust dependency. They must not
//! assume a sibling workspace checkout exists when compiling a published crate.

pub const COMFY_CONVROT_W4A8_BF16: &[u8] =
    include_bytes!("standard_operations/comfy_convrot_w4a8_bf16.rs");
pub const COMFY_CONVROT_W8A8_BF16: &[u8] =
    include_bytes!("standard_operations/comfy_convrot_w8a8_bf16.rs");
pub const NVFP4_MATMUL_BF16: &[u8] = include_bytes!("standard_operations/nvfp4_matmul_bf16.rs");
pub const NVFP4_QUANTIZE_BF16: &[u8] = include_bytes!("standard_operations/nvfp4_quantize_bf16.rs");
pub const SPARSE_FP4_ATTENTION: &[u8] =
    include_bytes!("standard_operations/sparse_fp4_attention.rs");
pub const SPARSE_FP4_QKV_QUANTIZE: &[u8] =
    include_bytes!("standard_operations/sparse_fp4_qkv_quantize.rs");
