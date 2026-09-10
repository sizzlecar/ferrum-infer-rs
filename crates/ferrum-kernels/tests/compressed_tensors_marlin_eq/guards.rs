use cudarc::driver::{CudaSlice, CudaStream, CudaViewMut, DevicePtr, DeviceRepr};
use std::{fmt::Debug, sync::Arc};

/// Guard both sides while preserving the launcher's vector-load alignment.
pub struct Guarded<T> {
    storage: CudaSlice<T>,
    guard: T,
    padding: usize,
    count: usize,
}

impl<T: DeviceRepr + Copy + PartialEq + Debug> Guarded<T> {
    pub fn new(stream: &Arc<CudaStream>, values: &[T], guard: T) -> Self {
        let padding = 128 / std::mem::size_of::<T>();
        let mut host = vec![guard; values.len() + 2 * padding];
        host[padding..padding + values.len()].copy_from_slice(values);
        Self {
            storage: stream.clone_htod(&host).expect("upload guarded buffer"),
            guard,
            padding,
            count: values.len(),
        }
    }

    pub fn view_mut(&mut self) -> CudaViewMut<'_, T> {
        self.storage
            .slice_mut(self.padding..self.padding + self.count)
    }

    pub fn pointer(&self, stream: &Arc<CudaStream>) -> u64 {
        self.storage.device_ptr(stream).0 + (self.padding * std::mem::size_of::<T>()) as u64
    }

    pub fn read(&self, stream: &Arc<CudaStream>) -> Vec<T> {
        let host = stream
            .clone_dtoh(&self.storage)
            .expect("read guarded buffer");
        assert!(
            host[..self.padding]
                .iter()
                .chain(&host[self.padding + self.count..])
                .all(|value| *value == self.guard),
            "Marlin wrote outside its output or workspace region"
        );
        host[self.padding..self.padding + self.count].to_vec()
    }
}
