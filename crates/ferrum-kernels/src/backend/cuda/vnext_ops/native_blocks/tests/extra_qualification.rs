//! Qualification of the actual locked production extra operator, using the
//! independent K oracle. No direct/test-only linking or provider registration.
use super::*;
use crate::gguf_blocks::extra_upstream_oracle as oracle;
use crate::native_ops::upstream_linear::{
    Algorithm, Arithmetic, Device, DeviceSpan, ExtraFormat, Layout, PreparedUpstreamLinear,
};
use cudarc::driver::{sys, CudaSlice, DevicePtr};
use ferrum_native_ops::upstream_linear::{UpstreamLinearPlanV1, UpstreamLinearRequestV1};
use std::{ffi::c_void, mem::size_of};
mod all_rows;
mod case;
mod correctness;
mod prefill;
use case::{Case, Route};

const FORMATS: [GgufBlockFormat; 3] = [
    GgufBlockFormat::Q3K,
    GgufBlockFormat::Iq3S,
    GgufBlockFormat::Iq4Nl,
];

fn context() -> (Arc<CudaContext>, Arc<CudaStream>, CudaNativeBlockKernels) {
    let context = CudaContext::new(0).expect("extra-format native experiment requires CUDA");
    // All buffers/graphs are owned for the whole experiment, single stream,
    // synchronized before host reads and destruction. No implicit event work.
    unsafe { context.disable_event_tracking() };
    let stream = context.new_stream().unwrap();
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    (context, stream, kernels)
}
fn request(
    context: &CudaContext,
    f: GgufBlockFormat,
    m: u32,
    k: u32,
    n: u32,
) -> UpstreamLinearRequestV1 {
    use sys::CUdevice_attribute::*;
    let get = |a| context.attribute(a).unwrap() as u32;
    // Construct the test ABI explicitly. The original-family Request::validate must still reject these formats;
    // production new_extra owns the separate closed format domain.
    UpstreamLinearRequestV1 {
        abi: 1,
        size: size_of::<UpstreamLinearRequestV1>() as u32,
        format: f.ggml_type_id(),
        layout: 0,
        rows: m,
        inputs: k,
        outputs: n,
        cc: get(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) * 100
            + get(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR) * 10,
        sm_count: get(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        reserved: 0,
        shared_limit: get(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN) as u64,
    }
}
fn extra_format(f: GgufBlockFormat) -> ExtraFormat {
    match f {
        GgufBlockFormat::Q3K => ExtraFormat::Q3K,
        GgufBlockFormat::Iq3S => ExtraFormat::Iq3S,
        GgufBlockFormat::Iq4Nl => ExtraFormat::Iq4Nl,
        _ => unreachable!(),
    }
}
fn plan(
    stream: &Arc<CudaStream>,
    f: GgufBlockFormat,
    m: u32,
    k: u32,
    n: u32,
    algorithm: u32,
) -> PreparedUpstreamLinear {
    let r = request(stream.context(), f, m, k, n);
    assert!(r.validate().is_err(), "old original format domain changed");
    let p = PreparedUpstreamLinear::new_extra(
        if algorithm == 1 {
            Algorithm::Mmq
        } else {
            Algorithm::Mmvq
        },
        Arithmetic::MarkerV2,
        extra_format(f),
        Layout::Columns,
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
    assert_eq!(
        p.operator(),
        ferrum_native_ops::upstream_extra_linear::UPSTREAM_EXTRA_LINEAR_OPERATOR
    );
    assert_eq!(p.arithmetic(), Arithmetic::MarkerV2);
    let raw = p.geometry();
    assert_eq!(raw.request, r);
    assert_eq!(raw.algorithm, algorithm);
    assert_eq!(raw.pack_abi, if algorithm == 1 { 1 } else { 3 });
    assert_eq!(
        raw.weight_bytes,
        raw.padded_outputs as u64 * (k as u64 / f.block_values() as u64) * f.block_bytes() as u64
    );
    assert_eq!(raw.output_bytes, m as u64 * n as u64 * 4);
    p
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
    fn pointer(&self, s: &Arc<CudaStream>) -> *mut c_void {
        let (p, g) = self.data.device_ptr(s);
        drop(g);
        (p + 32) as *mut c_void
    }
    fn span(&self, s: &Arc<CudaStream>) -> DeviceSpan {
        DeviceSpan {
            address: if self.bytes == 0 {
                0
            } else {
                self.pointer(s) as u64
            },
            bytes: self.bytes as u64,
        }
    }
    fn write(&mut self, s: &Arc<CudaStream>, bytes: &[u8]) {
        assert!(bytes.len() <= self.bytes);
        let mut view = self.data.slice_mut(32..32 + bytes.len());
        s.memcpy_htod(bytes, &mut view).unwrap();
    }
    fn poison(&mut self, s: &Arc<CudaStream>) {
        self.write(s, &vec![0x35; self.bytes]);
    }
    fn read(&self, s: &Arc<CudaStream>) -> Vec<u8> {
        s.clone_dtoh(&self.data.slice(32..32 + self.bytes)).unwrap()
    }
    fn guards(&self, s: &Arc<CudaStream>) {
        let b = s.clone_dtoh(&self.data).unwrap();
        assert!(b[..32]
            .iter()
            .chain(&b[32 + self.bytes..])
            .all(|v| *v == 0x35));
    }
}
fn half_bytes(v: &[f16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}
fn half_read(v: &[u8]) -> Vec<u16> {
    v.chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn float_read(v: &[u8]) -> Vec<f32> {
    v.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
