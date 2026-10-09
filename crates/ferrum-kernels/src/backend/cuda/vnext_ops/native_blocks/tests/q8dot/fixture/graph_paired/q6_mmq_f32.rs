//! Qualification of the independently locked production Q6 F32 artifact.
use super::{balanced_route, measure, Captured};
use crate::backend::cuda::vnext_ops::native_blocks::CudaNativeBlockKernels;
use crate::gguf_blocks::{fixtures::oracle_blocks, q6_mmq_oracle as oracle, GgufBlockFormat};
use crate::native_ops::upstream_linear::{Device, DeviceSpan};
use crate::native_ops::upstream_q6_f32_linear::PreparedQ6F32Linear;
use cudarc::driver::{
    sys, CudaContext, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use ferrum_native_ops::upstream_q6_f32_linear::{
    UpstreamQ6F32PlanV1 as Plan, UpstreamQ6F32RequestV1 as Request,
};
use half::f16;
use std::{ffi::c_void, mem::size_of, sync::Arc};
mod case;
mod correctness;
mod paired;
use case::Case;

// Symbols are supplied exclusively by the validated production artifact set.
unsafe extern "C" {
    fn ferrum_upstream_q6_f32_plan_v1(r: *const Request, p: *mut Plan) -> i32;
    fn ferrum_upstream_q6_f32_dot_v1(
        p: *const Plan,
        w: *const c_void,
        packed: *const c_void,
        raw: *mut c_void,
        fixup: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
}

fn context() -> (Arc<CudaContext>, Arc<CudaStream>, CudaNativeBlockKernels) {
    let c = CudaContext::new(0).expect("actual CUDA required");
    // Fixture owns all allocations through graph destruction and synchronizes
    // before every read/drop; avoid incidental tracked events inside capture.
    unsafe { c.disable_event_tracking() };
    let s = c.new_stream().unwrap();
    let k = CudaNativeBlockKernels::load(&c).unwrap();
    (c, s, k)
}
fn request(c: &CudaContext, m: u32, k: u32, n: u32) -> Request {
    use sys::CUdevice_attribute::*;
    let a = |key| c.attribute(key).unwrap() as u32;
    Request {
        abi: 1,
        size: size_of::<Request>() as u32,
        format: 14,
        layout: 0,
        rows: m,
        inputs: k,
        outputs: n,
        cc: a(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) * 100
            + a(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR) * 10,
        sm_count: a(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        reserved: 0,
        shared_limit: a(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN) as u64,
    }
}
fn plan(c: &CudaContext, m: u32, k: u32, n: u32) -> PreparedQ6F32Linear {
    let r = request(c, m, k, n);
    r.validate().unwrap();
    let prepared = PreparedQ6F32Linear::new(
        m,
        k,
        n,
        Device {
            architecture: r.cc,
            multiprocessors: r.sm_count,
            maximum_dynamic_shared_bytes: r.shared_limit,
        },
    )
    .unwrap();
    let p = prepared.geometry();
    p.validate_identity(&r).unwrap();
    assert_eq!(p.request, r);
    assert_eq!(p.weight_bytes, u64::from(n) * u64::from(k / 256) * 210);
    assert_eq!(p.output_bytes, u64::from(m) * u64::from(n) * 4);
    prepared
}
struct Buffer {
    data: CudaSlice<u8>,
    bytes: usize,
}
impl Buffer {
    fn new(s: &Arc<CudaStream>, bytes: usize) -> Self {
        Self {
            data: s.clone_htod(&vec![0x35; bytes + 64]).unwrap(),
            bytes,
        }
    }
    fn ptr(&self, s: &Arc<CudaStream>) -> *mut c_void {
        let (p, guard) = self.data.device_ptr(s);
        drop(guard);
        (p + 32) as *mut c_void
    }
    fn span(&self, s: &Arc<CudaStream>) -> DeviceSpan {
        DeviceSpan {
            address: self.ptr(s) as u64,
            bytes: self.bytes as u64,
        }
    }
    fn write(&mut self, s: &Arc<CudaStream>, b: &[u8]) {
        assert_eq!(b.len(), self.bytes);
        s.memcpy_htod(b, &mut self.data.slice_mut(32..32 + self.bytes))
            .unwrap();
    }
    fn read(&self, s: &Arc<CudaStream>) -> Vec<u8> {
        s.clone_dtoh(&self.data.slice(32..32 + self.bytes)).unwrap()
    }
    fn guards(&self, s: &Arc<CudaStream>) {
        let prefix = s.clone_dtoh(&self.data.slice(..32)).unwrap();
        let suffix = s.clone_dtoh(&self.data.slice(32 + self.bytes..)).unwrap();
        assert!(prefix.iter().chain(&suffix).all(|v| *v == 0x35));
    }
}
fn bytes(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}
fn floats(x: &[u8]) -> Vec<f32> {
    x.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
