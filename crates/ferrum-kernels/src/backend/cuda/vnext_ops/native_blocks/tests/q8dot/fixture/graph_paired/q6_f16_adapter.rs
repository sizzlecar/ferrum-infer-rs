//! Diagnostic-only F16 boundary around the existing Q6 D4/MMQ mathematics.
//! No provider, numerical profile or production selector uses these exports.
use super::{balanced_route, measure, Captured};
use crate::backend::cuda::vnext_ops::native_blocks::CudaNativeBlockKernels;
use crate::gguf_blocks::{fixtures::oracle_blocks, q6_mmq_oracle as oracle, GgufBlockFormat};
use cudarc::driver::{
    sys, CudaContext, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use half::f16;
use std::{ffi::c_void, mem::size_of, sync::Arc};

mod abi;
mod case;
mod correctness;
mod paired;
use abi::{Plan, Request};
use case::Case;

fn context() -> (Arc<CudaContext>, Arc<CudaStream>, CudaNativeBlockKernels) {
    let context = CudaContext::new(0).expect("actual CUDA required");
    // Every fixture owns its buffers until captured graphs are destroyed.
    unsafe { context.disable_event_tracking() };
    let stream = context.new_stream().unwrap();
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    (context, stream, kernels)
}

struct Buffer {
    data: CudaSlice<u8>,
    bytes: usize,
}
impl Buffer {
    fn new(stream: &Arc<CudaStream>, bytes: usize) -> Self {
        Self {
            data: stream.clone_htod(&vec![0x35; bytes + 64]).unwrap(),
            bytes,
        }
    }
    fn ptr(&self, stream: &Arc<CudaStream>) -> *mut c_void {
        let (address, guard) = self.data.device_ptr(stream);
        drop(guard);
        (address + 32) as *mut c_void
    }
    fn write(&mut self, stream: &Arc<CudaStream>, bytes: &[u8]) {
        assert_eq!(bytes.len(), self.bytes);
        stream
            .memcpy_htod(bytes, &mut self.data.slice_mut(32..32 + self.bytes))
            .unwrap();
    }
    fn read(&self, stream: &Arc<CudaStream>) -> Vec<u8> {
        stream
            .clone_dtoh(&self.data.slice(32..32 + self.bytes))
            .unwrap()
    }
    fn guards(&self, stream: &Arc<CudaStream>) {
        assert!(stream
            .clone_dtoh(&self.data.slice(..32))
            .unwrap()
            .iter()
            .all(|b| *b == 0x35));
        assert!(stream
            .clone_dtoh(&self.data.slice(32 + self.bytes..))
            .unwrap()
            .iter()
            .all(|b| *b == 0x35));
    }
}
fn half_bytes(values: &[f16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|x| x.to_bits().to_le_bytes())
        .collect()
}
fn float_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|x| x.to_bits().to_le_bytes())
        .collect()
}
fn halves(bytes: &[u8]) -> Vec<f16> {
    bytes
        .chunks_exact(2)
        .map(|x| f16::from_bits(u16::from_le_bytes(x.try_into().unwrap())))
        .collect()
}
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect()
}
