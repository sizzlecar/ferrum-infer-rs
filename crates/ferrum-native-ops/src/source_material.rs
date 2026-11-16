//! Crate-owned ABI source inputs for native provider declaration fingerprints.
//!
//! Exporting source bytes keeps provenance available in both workspace builds
//! and isolated registry packages without depending on a consumer's layout.

pub const CUTLASS_INT8_FUSED_OUTPUT: &[u8] = include_bytes!("cutlass_int8_fused_output.rs");
pub const CUTLASS_NVFP4: &[u8] = include_bytes!("cutlass_nvfp4.rs");
pub const CUTLASS_NVFP4_QUANTIZE: &[u8] = include_bytes!("cutlass_nvfp4/quantize.rs");
pub const SAGE_VSA_FP4: &[u8] = include_bytes!("sage_vsa_fp4.rs");
pub const SAGE_VSA_FP4_LISTS: &[u8] = include_bytes!("sage_vsa_fp4/lists.rs");
pub const SAGE_VSA_FP4_SELECTION: &[u8] = include_bytes!("sage_vsa_fp4/selection.rs");
pub const SAGE_VSA_FP4_KERNEL_UPLOAD: &[u8] =
    include_bytes!("sage_vsa_fp4/selection/kernel_upload.rs");
pub const SAGE_VSA_FP4_QUANTIZE: &[u8] = include_bytes!("sage_vsa_fp4/quantize.rs");
