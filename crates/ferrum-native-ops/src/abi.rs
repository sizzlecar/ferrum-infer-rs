//! Stable native operator ABI constants and C descriptor shapes.

pub const LEGACY_FERRUM_NATIVE_ABI_VERSION: &str =
    ferrum_types::LEGACY_FERRUM_NATIVE_OPERATOR_ABI_VERSION;
pub const FERRUM_NATIVE_ABI_VERSION: &str = ferrum_types::FERRUM_NATIVE_OPERATOR_ABI_VERSION;
pub const LEGACY_FERRUM_NATIVE_OP_INIT_SYMBOL: &str = "ferrum_native_op_init";
pub const LEGACY_FERRUM_NATIVE_OP_DESCRIPTOR_SYMBOL: &str = "ferrum_native_op_descriptor";

// Compatibility aliases for schema-v1 consumers. New artifacts must declare a
// unique descriptor symbol in their schema-v2 manifest.
pub const FERRUM_NATIVE_OP_INIT_SYMBOL: &str = "ferrum_native_op_init";
pub const FERRUM_NATIVE_OP_DESCRIPTOR_SYMBOL: &str = "ferrum_native_op_descriptor";

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LegacyFerrumNativeOperatorDescriptor {
    pub abi_version: u32,
    pub operator_name: *const std::ffi::c_char,
    pub operator_abi_version: *const std::ffi::c_char,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FerrumNativeOperatorDescriptorV2 {
    pub struct_size: u32,
    pub ferrum_native_abi_version: u32,
    pub operator_name: *const std::ffi::c_char,
    pub operator_abi_version: *const std::ffi::c_char,
    pub g03_catalog_sha256: *const std::ffi::c_char,
    pub abi_contract_sha256: *const std::ffi::c_char,
}

pub type FerrumNativeOperatorDescriptor = LegacyFerrumNativeOperatorDescriptor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeOperatorAbi {
    pub ferrum_native_abi_version: &'static str,
}

impl NativeOperatorAbi {
    pub const fn current() -> Self {
        Self {
            ferrum_native_abi_version: FERRUM_NATIVE_ABI_VERSION,
        }
    }

    pub const fn legacy_v1() -> LegacyNativeOperatorAbi {
        LegacyNativeOperatorAbi {
            ferrum_native_abi_version: LEGACY_FERRUM_NATIVE_ABI_VERSION,
            init_symbol: LEGACY_FERRUM_NATIVE_OP_INIT_SYMBOL,
            descriptor_symbol: LEGACY_FERRUM_NATIVE_OP_DESCRIPTOR_SYMBOL,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LegacyNativeOperatorAbi {
    pub ferrum_native_abi_version: &'static str,
    pub init_symbol: &'static str,
    pub descriptor_symbol: &'static str,
}
