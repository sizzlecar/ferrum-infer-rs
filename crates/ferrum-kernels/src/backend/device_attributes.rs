//! Device attributes read from the selected CUDA driver device at startup.

#[cfg(any(feature = "cuda", test))]
use ferrum_types::{FerrumError, Result};

/// Physical attributes, independent of GPU names and command-line utilities.
/// These describe the device, not the CUDA toolkit or runtime version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaDeviceAttributes {
    pub compute_capability_major: u32,
    pub compute_capability_minor: u32,
    pub multiprocessor_count: u32,
}

#[cfg(any(feature = "cuda", test))]
impl CudaDeviceAttributes {
    fn from_raw(major: i32, minor: i32, multiprocessors: i32) -> Result<Self> {
        if major <= 0 || minor < 0 || multiprocessors <= 0 {
            return Err(FerrumError::device(format!(
                "invalid CUDA device attributes: compute capability {major}.{minor}, multiprocessors {multiprocessors}"
            )));
        }
        Ok(Self {
            compute_capability_major: major as u32,
            compute_capability_minor: minor as u32,
            multiprocessor_count: multiprocessors as u32,
        })
    }
}

/// Query the selected driver-visible ordinal without loading operators or
/// allocating model resources. The driver applies CUDA_VISIBLE_DEVICES; no
/// process-wide GPU-name or nvidia-smi index mapping is involved.
#[cfg(feature = "cuda")]
pub fn probe_cuda_device_attributes(ordinal: usize) -> Result<CudaDeviceAttributes> {
    let context = cudarc::driver::CudaContext::new(ordinal)
        .map_err(|error| FerrumError::device(format!("probe CUDA {ordinal}: {error}")))?;
    cuda_context_device_attributes(&context)
}

#[cfg(feature = "cuda")]
pub(super) fn cuda_context_device_attributes(
    context: &cudarc::driver::CudaContext,
) -> Result<CudaDeviceAttributes> {
    use cudarc::driver::sys::CUdevice_attribute::*;

    let attribute = |key| {
        context.attribute(key).map_err(|error| {
            FerrumError::device(format!("read CUDA device attribute {key:?}: {error}"))
        })
    };
    CudaDeviceAttributes::from_raw(
        attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
        attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
        attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_device_attributes_preserve_major_minor_and_sm_count() {
        let attributes = CudaDeviceAttributes::from_raw(12, 0, 32).unwrap();
        assert_eq!(attributes.compute_capability_major, 12);
        assert_eq!(attributes.compute_capability_minor, 0);
        assert_eq!(attributes.multiprocessor_count, 32);
    }

    #[test]
    fn cuda_device_attributes_reject_invalid_driver_values() {
        for (major, minor, multiprocessors) in
            [(0, 0, 32), (-1, 0, 32), (8, -1, 32), (8, 0, 0), (8, 0, -1)]
        {
            assert!(CudaDeviceAttributes::from_raw(major, minor, multiprocessors).is_err());
        }
    }
}
