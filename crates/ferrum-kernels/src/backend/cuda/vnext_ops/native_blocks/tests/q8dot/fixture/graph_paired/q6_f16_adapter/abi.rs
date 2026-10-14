//! The F16 diagnostic ABI has a distinct boundary tag from the F32 head ABI.
use super::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub(super) struct Request {
    pub abi: u32,
    pub size: u32,
    pub format: u32,
    pub layout: u32,
    pub rows: u32,
    pub inputs: u32,
    pub outputs: u32,
    pub cc: u32,
    pub sm_count: u32,
    pub reserved: u32,
    pub shared_limit: u64,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub(super) struct Plan {
    pub request: Request,
    pub abi: u32,
    pub size: u32,
    pub algorithm: u32,
    pub pack_abi: u32,
    pub padded_inputs: u32,
    pub padded_outputs: u32,
    pub guard_blocks: u32,
    pub j: u32,
    pub i: u32,
    pub nthreads: u32,
    pub shared_bytes: u32,
    pub blocks: u32,
    pub tiles_y: u32,
    pub fixup: u32,
    pub ncols: u32,
    pub channels: u32,
    pub nwarps: u32,
    pub rows_per_block: u32,
    pub small_k: u32,
    pub reserved: u32,
    pub weight_bytes: u64,
    pub converted_bytes: u64,
    pub packed_bytes: u64,
    pub output_bytes: u64,
    pub fixup_bytes: u64,
}
const _: () = {
    assert!(size_of::<Request>() == 48);
    assert!(size_of::<Plan>() == 168);
    assert!(std::mem::offset_of!(Plan, weight_bytes) == 128);
};

unsafe extern "C" {
    pub(super) fn ferrum_upstream_q6_f16_plan_v1(request: *const Request, plan: *mut Plan) -> i32;
    pub(super) fn ferrum_upstream_q6_f16_pack_v1(
        plan: *const Plan,
        input: *const c_void,
        stride: u32,
        converted: *mut c_void,
        packed: *mut c_void,
        rows: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    pub(super) fn ferrum_upstream_q6_f16_dot_v1(
        plan: *const Plan,
        weights: *const c_void,
        packed: *const c_void,
        raw: *mut c_void,
        fixup: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    pub(super) fn ferrum_upstream_q6_f16_check_weights_v1(
        plan: *const Plan,
        weights: *const c_void,
        flag: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    pub(super) fn ferrum_upstream_q6_f16_cast_v1(
        plan: *const Plan,
        raw: *const c_void,
        output: *mut c_void,
        stride: u32,
        rows: *const c_void,
        flag: *const c_void,
        stream: *mut c_void,
    ) -> i32;
}

pub(super) fn request(context: &CudaContext, m: u32, k: u32, n: u32) -> Request {
    use sys::CUdevice_attribute::*;
    let a = |key| context.attribute(key).unwrap() as u32;
    Request {
        abi: 2,
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
pub(super) fn plan(context: &CudaContext, m: usize, k: usize, n: usize) -> Plan {
    let r = request(
        context,
        m.try_into().unwrap(),
        k.try_into().unwrap(),
        n.try_into().unwrap(),
    );
    let mut p = Plan::default();
    assert_eq!(unsafe { ferrum_upstream_q6_f16_plan_v1(&r, &mut p) }, 0);
    assert_eq!(p.request, r);
    assert_eq!(
        (p.abi, p.size, p.algorithm, p.pack_abi),
        (2, size_of::<Plan>() as u32, 1, 1)
    );
    assert_eq!(p.weight_bytes, (n * (k / 256) * 210) as u64);
    assert_eq!(p.converted_bytes, (m * k.div_ceil(512) * 512 * 4) as u64);
    assert_eq!(p.output_bytes, (m * n * 4) as u64);
    assert_eq!(p.padded_inputs as usize, k.div_ceil(512) * 512);
    assert_eq!(
        p.packed_bytes,
        (m * p.padded_inputs as usize * 9 / 8 + p.guard_blocks as usize * 144) as u64
    );
    // Independently bound the unpredicated K128 tile loader's final copy.
    let j = p.j as usize;
    let copy_bytes = (j * 36).div_ceil(p.nthreads as usize) * p.nthreads as usize * 4;
    let end = ((k / 128 - 1) * m + (m.div_ceil(j) - 1) * j) * 144 + copy_bytes;
    assert!(p.packed_bytes >= end as u64);
    p
}
