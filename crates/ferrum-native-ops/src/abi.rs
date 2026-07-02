//! Stable native operator ABI constants and C descriptor shape.

pub const FERRUM_NATIVE_ABI_VERSION: &str = "1";
pub const FERRUM_NATIVE_OP_INIT_SYMBOL: &str = "ferrum_native_op_init";
pub const FERRUM_NATIVE_OP_DESCRIPTOR_SYMBOL: &str = "ferrum_native_op_descriptor";

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FerrumNativeOperatorDescriptor {
    pub abi_version: u32,
    pub operator_name: *const std::ffi::c_char,
    pub operator_abi_version: *const std::ffi::c_char,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeOperatorAbi {
    pub ferrum_native_abi_version: &'static str,
    pub init_symbol: &'static str,
    pub descriptor_symbol: &'static str,
}

impl NativeOperatorAbi {
    pub const fn current() -> Self {
        Self {
            ferrum_native_abi_version: FERRUM_NATIVE_ABI_VERSION,
            init_symbol: FERRUM_NATIVE_OP_INIT_SYMBOL,
            descriptor_symbol: FERRUM_NATIVE_OP_DESCRIPTOR_SYMBOL,
        }
    }
}
