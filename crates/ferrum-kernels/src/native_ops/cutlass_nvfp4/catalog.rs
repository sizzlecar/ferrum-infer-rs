//! CPU source declaration; it does not initialize CUDA or qualify an artifact.
use ferrum_runtime::{nvfp4_matmul_bf16_contract, nvfp4_quantize_bf16_contract, OperationContract};
use ferrum_types::{
    NativeOperatorBackend, NativeOperatorContractVersion, NativeOperatorProviderCatalog,
    NativeOperatorProviderCatalogRow, NATIVE_OPERATOR_PROVIDER_CATALOG_SCHEMA_VERSION,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const CAPABILITY_ID: &str = "capability.cuda.nvfp4_matmul.sm120.k128";
pub const PROVIDER_ID: &str = "provider.cuda.nvfp4_matmul.bf16.cutlass.sm120.k128";
pub const QUANTIZE_CAPABILITY_ID: &str = "capability.cuda.nvfp4_quantize.bf16.rcp-approx-ftz.sm120";
pub const QUANTIZE_PROVIDER_ID: &str = "provider.cuda.nvfp4_quantize.bf16.rcp-approx-ftz.sm120";
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CutlassNvfp4Config {
    pub opaque_memory_bytes: u64,
    pub opaque_memory_profile: String,
}
impl CutlassNvfp4Config {
    pub fn validate(&self) -> Result<(), String> {
        if self.opaque_memory_bytes == 0
            || self.opaque_memory_profile.trim().is_empty()
            || self.opaque_memory_profile.len() > 256
            || !self.opaque_memory_profile.is_ascii()
            || self
                .opaque_memory_profile
                .bytes()
                .any(|b| b.is_ascii_control())
        {
            return Err(
                "NVFP4 requires positive opaque allowance and bounded ASCII memory profile".into(),
            );
        }
        Ok(())
    }
}
pub fn source_declaration_fingerprint() -> String {
    digest(&[
        b"ferrum.cutlass-nvfp4.source.v1",
        super::CUTLASS_NVFP4_NUMERICAL_IDENTITY.as_bytes(),
        include_bytes!("catalog.rs"),
        include_bytes!("../cutlass_nvfp4.rs"),
        include_bytes!("quantize.rs"),
        include_bytes!("../../backend/cuda/vnext_ops/tensor_nvfp4_bf16.rs"),
        include_bytes!("../../backend/cuda/vnext_ops/tensor_nvfp4_bf16/quantize.rs"),
        include_bytes!("../../backend/cuda/vnext_ops/tensor_storage.rs"),
        include_bytes!("../../backend/cuda/vnext_runtime/tensor_eager/native_module_owner.rs"),
        include_bytes!("../../backend/cuda/vnext_runtime/tensor_eager/native_module_capacity.rs"),
        include_bytes!("../../../native/cutlass-nvfp4/abi.h"),
        include_bytes!("../../../native/cutlass-nvfp4/gemm.cuh"),
        include_bytes!("../../../native/cutlass-nvfp4/boundary.cu"),
        include_bytes!("../../../native/cutlass-nvfp4/boundary_checks.cuh"),
        include_bytes!("../../../native/cutlass-nvfp4/quantize.cu"),
        include_bytes!("../../../native/cutlass-nvfp4/source.seed.json"),
        include_bytes!("../../../../ferrum-native-ops/src/cutlass_nvfp4.rs"),
        include_bytes!("../../../../ferrum-native-ops/src/cutlass_nvfp4/quantize.rs"),
        include_bytes!("../../../../ferrum-runtime/src/standard_operations/nvfp4_matmul_bf16.rs"),
        include_bytes!("../../../../ferrum-runtime/src/standard_operations/nvfp4_quantize_bf16.rs"),
    ])
}
pub fn provider_catalog_declaration(
    config: &CutlassNvfp4Config,
) -> Result<NativeOperatorProviderCatalog, String> {
    config.validate()?;
    let mut providers = Vec::new();
    for (contract, provider_id) in [
        (nvfp4_matmul_bf16_contract(), PROVIDER_ID),
        (nvfp4_quantize_bf16_contract(), QUANTIZE_PROVIDER_ID),
    ] {
        let contract = contract.map_err(|e| e.to_string())?;
        let d = contract.descriptor();
        providers.push(NativeOperatorProviderCatalogRow {
            operation_id: d.id.to_string(),
            operation_contract_version: NativeOperatorContractVersion::new(
                d.version.major,
                d.version.minor,
            ),
            operation_fingerprint: d.fingerprint().map_err(|e| e.to_string())?,
            provider_id: provider_id.into(),
            provider_version: NativeOperatorContractVersion::new(1, 0),
            provider_implementation_fingerprint: digest(&[
                source_declaration_fingerprint().as_bytes(),
                &serde_json::to_vec(config).map_err(|e| e.to_string())?,
                provider_id.as_bytes(),
            ]),
        });
    }
    let catalog = NativeOperatorProviderCatalog {
        schema_version: NATIVE_OPERATOR_PROVIDER_CATALOG_SCHEMA_VERSION,
        backend: NativeOperatorBackend::Cuda,
        providers,
    };
    catalog.validate()?;
    Ok(catalog)
}
fn digest(parts: &[&[u8]]) -> String {
    let mut hash = Sha256::new();
    for part in parts {
        hash.update((part.len() as u64).to_le_bytes());
        hash.update(part);
    }
    format!("{:x}", hash.finalize())
}
