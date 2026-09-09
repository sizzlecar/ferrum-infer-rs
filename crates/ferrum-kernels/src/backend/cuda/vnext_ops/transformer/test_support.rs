//! Live, offset device allocations with checked canaries for raw CUDA launchers.
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DeviceRepr};
use std::{fmt::Debug, sync::Arc};

pub(super) struct Guarded<T> {
    device: CudaSlice<T>,
    original: Vec<T>,
}

impl<T: DeviceRepr + Clone + PartialEq + Debug> Guarded<T> {
    pub(super) fn new(stream: &Arc<CudaStream>, values: &[T], guard: T) -> Self {
        let mut original = vec![guard.clone(); 8];
        original.extend_from_slice(values);
        original.extend(vec![guard; 8]);
        Self {
            device: stream.clone_htod(&original).unwrap(),
            original,
        }
    }

    pub(super) fn pointer(&self, stream: &Arc<CudaStream>) -> u64 {
        self.device.device_ptr(stream).0 + (8 * std::mem::size_of::<T>()) as u64
    }

    pub(super) fn read(&self, stream: &Arc<CudaStream>) -> Vec<T> {
        // The fixture retains buffers on one stream until every raw launch finishes.
        stream.synchronize().unwrap();
        let result = stream.clone_dtoh(&self.device).unwrap();
        let end = result.len() - 8;
        assert_eq!(&result[..8], &self.original[..8], "leading guard modified");
        assert_eq!(
            &result[end..],
            &self.original[end..],
            "trailing guard modified"
        );
        result[8..end].to_vec()
    }

    pub(super) fn assert_unchanged(&self, stream: &Arc<CudaStream>) {
        assert_eq!(self.read(stream), self.original[8..self.original.len() - 8]);
    }
}
