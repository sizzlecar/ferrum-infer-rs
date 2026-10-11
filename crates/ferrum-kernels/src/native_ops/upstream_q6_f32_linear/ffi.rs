use ferrum_native_ops::upstream_q6_f32_linear::{
    UpstreamQ6F32PlanV1 as Plan, UpstreamQ6F32RequestV1 as Request,
};
use std::ffi::c_void;
unsafe extern "C" {
    pub fn ferrum_upstream_q6_f32_plan_v1(request: *const Request, plan: *mut Plan) -> i32;
    pub fn ferrum_upstream_q6_f32_pack_v1(
        plan: *const Plan,
        input: *const c_void,
        stride: u32,
        converted: *mut c_void,
        packed: *mut c_void,
        rows: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    pub fn ferrum_upstream_q6_f32_dot_v1(
        plan: *const Plan,
        weights: *const c_void,
        packed: *const c_void,
        raw: *mut c_void,
        fixup: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    pub fn ferrum_upstream_q6_f32_check_weights_v1(
        plan: *const Plan,
        weights: *const c_void,
        flag: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    pub fn ferrum_upstream_q6_f32_publish_v1(
        plan: *const Plan,
        raw: *const c_void,
        output: *mut c_void,
        stride: u32,
        rows: *const c_void,
        flag: *const c_void,
        stream: *mut c_void,
    ) -> i32;
}
