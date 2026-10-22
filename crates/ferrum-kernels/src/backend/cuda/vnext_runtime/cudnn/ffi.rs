//! Narrow cuDNN 9 ABI. No allocator or autotuning API is exposed.
//!
//! Integer enum arguments deliberately use c_int: foreign status values must
//! not create an invalid Rust enum discriminant.

use std::ffi::{c_int, c_void};
use std::path::Path;

use libloading::Library;

pub(super) type Handle = *mut c_void;
pub(super) type Descriptor = *mut c_void;
pub(super) const SUCCESS: c_int = 0;
pub(super) const NCHW: c_int = 0;
pub(super) const FLOAT: c_int = 0;
pub(super) const BF16: c_int = 9;
pub(super) const CROSS_CORRELATION: c_int = 1;
pub(super) const TENSOR_OP_MATH: c_int = 1;
pub(super) const DETERMINISTIC: c_int = 1;
pub(super) const ALGORITHMS: usize = 8;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ForwardPerformance {
    pub algo: c_int,
    pub status: c_int,
    pub time: f32,
    pub memory: usize,
    pub determinism: c_int,
    pub math_type: c_int,
    pub reserved: [c_int; 3],
}

pub(super) struct Api {
    pub version: unsafe extern "C" fn() -> usize,
    pub create: unsafe extern "C" fn(*mut Handle) -> c_int,
    pub destroy: unsafe extern "C" fn(Handle) -> c_int,
    pub set_stream: unsafe extern "C" fn(Handle, *mut c_void) -> c_int,
    pub create_tensor: unsafe extern "C" fn(*mut Descriptor) -> c_int,
    pub destroy_tensor: unsafe extern "C" fn(Descriptor) -> c_int,
    pub set_tensor:
        unsafe extern "C" fn(Descriptor, c_int, c_int, c_int, c_int, c_int, c_int) -> c_int,
    pub create_filter: unsafe extern "C" fn(*mut Descriptor) -> c_int,
    pub destroy_filter: unsafe extern "C" fn(Descriptor) -> c_int,
    pub set_filter:
        unsafe extern "C" fn(Descriptor, c_int, c_int, c_int, c_int, c_int, c_int) -> c_int,
    pub create_convolution: unsafe extern "C" fn(*mut Descriptor) -> c_int,
    pub destroy_convolution: unsafe extern "C" fn(Descriptor) -> c_int,
    pub set_convolution: unsafe extern "C" fn(
        Descriptor,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
    ) -> c_int,
    pub set_groups: unsafe extern "C" fn(Descriptor, c_int) -> c_int,
    pub set_math: unsafe extern "C" fn(Descriptor, c_int) -> c_int,
    pub algorithms: unsafe extern "C" fn(
        Handle,
        Descriptor,
        Descriptor,
        Descriptor,
        Descriptor,
        c_int,
        *mut c_int,
        *mut ForwardPerformance,
    ) -> c_int,
    pub workspace: unsafe extern "C" fn(
        Handle,
        Descriptor,
        Descriptor,
        Descriptor,
        Descriptor,
        c_int,
        *mut usize,
    ) -> c_int,
    pub forward: unsafe extern "C" fn(
        Handle,
        *const c_void,
        Descriptor,
        *const c_void,
        Descriptor,
        *const c_void,
        Descriptor,
        c_int,
        *mut c_void,
        usize,
        *const c_void,
        Descriptor,
        *mut c_void,
    ) -> c_int,
    // Symbols are copied only while this library is retained. It must outlive
    // every native handle and all submitted commands using these functions.
    _library: Library,
}

impl Api {
    /// Loading native code is permitted only after the caller reserves its
    /// opaque capacity. The supplied deployment library is trusted and immutable.
    pub(super) unsafe fn load(path: &Path) -> Result<Self, String> {
        // After successful dlopen, a missing symbol/panic does not certify
        // native-initializer cleanup. Preserve the library with the caller's
        // fail-closed capacity/context retention on those paths.
        let mut library = std::mem::ManuallyDrop::new(
            unsafe { Library::new(path) }.map_err(|e| format!("cuDNN library load: {e}"))?,
        );
        macro_rules! symbol {
            ($name:literal) => {
                *unsafe { library.get(concat!($name, "\0").as_bytes()) }
                    .map_err(|e| format!("cuDNN symbol {}: {e}", $name))?
            };
        }
        Ok(Self {
            version: symbol!("cudnnGetVersion"),
            create: symbol!("cudnnCreate"),
            destroy: symbol!("cudnnDestroy"),
            set_stream: symbol!("cudnnSetStream"),
            create_tensor: symbol!("cudnnCreateTensorDescriptor"),
            destroy_tensor: symbol!("cudnnDestroyTensorDescriptor"),
            set_tensor: symbol!("cudnnSetTensor4dDescriptor"),
            create_filter: symbol!("cudnnCreateFilterDescriptor"),
            destroy_filter: symbol!("cudnnDestroyFilterDescriptor"),
            set_filter: symbol!("cudnnSetFilter4dDescriptor"),
            create_convolution: symbol!("cudnnCreateConvolutionDescriptor"),
            destroy_convolution: symbol!("cudnnDestroyConvolutionDescriptor"),
            set_convolution: symbol!("cudnnSetConvolution2dDescriptor"),
            set_groups: symbol!("cudnnSetConvolutionGroupCount"),
            set_math: symbol!("cudnnSetConvolutionMathType"),
            algorithms: symbol!("cudnnGetConvolutionForwardAlgorithm_v7"),
            workspace: symbol!("cudnnGetConvolutionForwardWorkspaceSize"),
            forward: symbol!("cudnnConvolutionForward"),
            _library: unsafe { std::mem::ManuallyDrop::take(&mut library) },
        })
    }
}

pub(super) fn check(status: c_int, operation: &str) -> Result<(), String> {
    if status == SUCCESS {
        Ok(())
    } else {
        Err(format!("cuDNN {operation} failed with status {status}"))
    }
}
