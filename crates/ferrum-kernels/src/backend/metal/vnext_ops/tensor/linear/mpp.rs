//! Optional MSL 4 BF16 GEMM in the admitted tensor provider bundle.
//!
//! The provider retains the pipeline under its existing budget owner, includes
//! both sources in its fingerprint, and dispatches only `supports_shape` shapes.
//! BlockLinear keeps its admitted decode scratch and encoder dependency.
#![allow(
    unexpected_cfgs,
    reason = "objc 0.2 macros expand their legacy cargo-clippy feature cfg in the calling crate"
)]

use metal::objc::{msg_send, sel, sel_impl};
use metal::{CompileOptions, ComputePipelineState, DeviceRef, MTLGPUFamily};
use std::ffi::CStr;

pub(in crate::backend::metal::vnext_ops::tensor) const THREADS: u64 = 128;
pub(in crate::backend::metal::vnext_ops::tensor) const TILE_ROWS: u32 = 32;
pub(in crate::backend::metal::vnext_ops::tensor) const TILE_COLUMNS: u32 = 64;
pub(in crate::backend::metal::vnext_ops::tensor) const NUMERICAL_IDENTITY: &str =
    "metal.mpp.bf16-bf16-f32.m32n64.dynamic-k;relaxed=false;integer-BF16-epilogue.v1";
pub(in crate::backend::metal::vnext_ops::tensor) const FINGERPRINT_SOURCE: &str =
    concat!(include_str!("mpp.rs"), include_str!("mpp.metal"));

pub(in crate::backend::metal::vnext_ops::tensor) enum MppPipelineAvailability {
    Unsupported { reason: String },
    Ready(ComputePipelineState),
}

/// Only capability absence returns `Unsupported`. A supported device with a
/// shader, function, or PSO creation error returns `Err`, including Metal's text.
/// This compiles a separate library and allocates no application device buffers.
pub(in crate::backend::metal::vnext_ops::tensor) fn compile(
    device: &DeviceRef,
) -> Result<MppPipelineAvailability, String> {
    let version = macos_version()?;
    if version < (26, 1) {
        return Ok(MppPipelineAvailability::Unsupported {
            reason: format!(
                "MPP BF16 tensors require macOS 26.1 or newer; found {}.{}",
                version.0, version.1
            ),
        });
    }
    if !device.supports_family(MTLGPUFamily::Apple7) {
        return Ok(MppPipelineAvailability::Unsupported {
            reason: "MPP candidate requires an Apple GPU supporting the Apple7 feature family"
                .into(),
        });
    }

    let options = CompileOptions::new();
    options.set_fast_math_enabled(false);
    // metal-rs 0.31's Rust enum ends at MSL 3.1. Objective-C declares this
    // property as an NSUInteger-backed MTLLanguageVersion; send that exact ABI
    // instead of constructing an invalid Rust enum discriminant. Never call the
    // typed `language_version()` getter on these options for the same reason.
    let msl4: metal::NSUInteger = 0x40000;
    unsafe {
        let _: () = msg_send![&*options, setLanguageVersion: msl4];
        let actual: metal::NSUInteger = msg_send![&*options, languageVersion];
        if actual != msl4 {
            return Err(format!(
                "MPP compile options rejected MSL 4: requested {msl4:#x}, received {actual:#x}"
            ));
        }
    }
    let source = concat!(
        include_str!("../bf16.metal"),
        "\n",
        include_str!("mpp.metal")
    );
    let library = device
        .new_library_with_source(source, &options)
        .map_err(|error| format!("MPP BF16 MSL 4 library compilation failed: {error}"))?;
    let function = library
        .get_function("ferrum_tensor_linear_bf16_mpp", None)
        .map_err(|error| format!("MPP BF16 kernel lookup failed: {error}"))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| format!("MPP BF16 compute pipeline creation failed: {error}"))?;
    if pipeline.thread_execution_width() != 32 {
        return Ok(MppPipelineAvailability::Unsupported {
            reason: format!(
                "MPP candidate requires four 32-thread SIMD groups; device reports width {}",
                pipeline.thread_execution_width()
            ),
        });
    }
    if pipeline.max_total_threads_per_threadgroup() < THREADS {
        return Ok(MppPipelineAvailability::Unsupported {
            reason: format!(
                "MPP candidate requires {THREADS} threads; pipeline permits {}",
                pipeline.max_total_threads_per_threadgroup()
            ),
        });
    }
    let required = pipeline.static_threadgroup_memory_length();
    let available = device.max_threadgroup_memory_length();
    if required > available {
        return Ok(MppPipelineAvailability::Unsupported {
            reason: format!(
                "MPP candidate requires {required} bytes of static threadgroup memory; device permits {available}"
            ),
        });
    }
    Ok(MppPipelineAvailability::Ready(pipeline))
}

/// Full output tiles only. MPP tensor dimensions/strides use `int`; conservatively
/// keep every flattened operand and output within its positive indexing range.
/// The caller keeps the existing GEMM for all other shapes, including empty ones.
pub(in crate::backend::metal::vnext_ops::tensor) fn supports_shape(
    rows: u32,
    inputs: u32,
    outputs: u32,
) -> bool {
    let maximum = i32::MAX as u64;
    rows > 0
        && inputs > 0
        && outputs > 0
        && rows % TILE_ROWS == 0
        && outputs % TILE_COLUMNS == 0
        && u64::from(rows) * u64::from(inputs) <= maximum
        && u64::from(outputs) * u64::from(inputs) <= maximum
        && u64::from(rows) * u64::from(outputs) <= maximum
}

fn macos_version() -> Result<(u32, u32), String> {
    let mut bytes = [0_u8; 64];
    let mut length = bytes.len();
    // Read-only OS metadata; no process invocation, filesystem access, or GPU
    // work. sysctl supplies a NUL-terminated product version on macOS.
    let status = unsafe {
        libc::sysctlbyname(
            c"kern.osproductversion".as_ptr(),
            bytes.as_mut_ptr().cast(),
            &mut length,
            std::ptr::null_mut(),
            0,
        )
    };
    if status != 0 {
        return Err(format!(
            "cannot determine macOS version for MPP: {}",
            std::io::Error::last_os_error()
        ));
    }
    let version = bytes
        .get(..length)
        .and_then(|value| CStr::from_bytes_with_nul(value).ok())
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| "invalid kern.osproductversion value for MPP".to_owned())?;
    let mut components = version.split('.');
    let major = components.next().and_then(|value| value.parse().ok());
    let minor = components.next().and_then(|value| value.parse().ok());
    major
        .zip(minor)
        .ok_or_else(|| format!("cannot parse macOS major/minor version for MPP from {version:?}"))
}

#[cfg(test)]
mod tests {
    use super::supports_shape;

    #[test]
    fn full_tiles_allow_dynamic_k_and_reject_unsafe_tensor_extents() {
        assert!(supports_shape(32, 1, 64));
        assert!(supports_shape(64, 129, 192));
        assert!(!supports_shape(31, 129, 64));
        assert!(!supports_shape(32, 129, 65));
        assert!(!supports_shape(0, 1, 64));
        assert!(!supports_shape(32, 0, 64));
        assert!(!supports_shape(32, 1, 0));
        assert!(!supports_shape(32, u32::MAX, 64));
        assert!(!supports_shape(65_536, 32, 65_536));
        assert!(supports_shape(32, 33_554_431, 64));
        assert!(!supports_shape(32, 33_554_432, 64));
    }
}
