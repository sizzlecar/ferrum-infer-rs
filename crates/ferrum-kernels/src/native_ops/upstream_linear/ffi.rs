use super::*;
unsafe extern "C" {
    pub(super) fn ferrum_upstream_mmq_prefill_plan_v1(
        r: *const UpstreamLinearRequestV1,
        p: *mut UpstreamLinearPlanV1,
    ) -> i32;
    pub(super) fn ferrum_upstream_mmq_plan_v1(
        r: *const UpstreamLinearRequestV1,
        p: *mut UpstreamLinearPlanV1,
    ) -> i32;
    pub(super) fn ferrum_upstream_mmq_pack_v1(
        p: *const UpstreamLinearPlanV1,
        x: *const c_void,
        stride: u32,
        c: *mut c_void,
        q: *mut c_void,
        s: *mut c_void,
    ) -> i32;
    pub(super) fn ferrum_upstream_mmq_dot_v1(
        p: *const UpstreamLinearPlanV1,
        w: *const c_void,
        q: *const c_void,
        o: *mut c_void,
        f: *mut c_void,
        s: *mut c_void,
    ) -> i32;
    pub(super) fn ferrum_upstream_mmq_cast_v1(
        p: *const UpstreamLinearPlanV1,
        x: *const c_void,
        o: *mut c_void,
        stride: u32,
        s: *mut c_void,
    ) -> i32;
    pub(super) fn ferrum_upstream_mmq_pack_v2(
        p: *const UpstreamLinearPlanV1,
        x: *const c_void,
        stride: u32,
        c: *mut c_void,
        q: *mut c_void,
        flags: *mut c_void,
        s: *mut c_void,
    ) -> i32;
    pub(super) fn ferrum_upstream_mmq_cast_v2(
        p: *const UpstreamLinearPlanV1,
        x: *const c_void,
        o: *mut c_void,
        stride: u32,
        row_flags: *const c_void,
        weight_flag: *const c_void,
        s: *mut c_void,
    ) -> i32;
    pub(super) fn ferrum_upstream_mmq_check_weights_v2(
        p: *const UpstreamLinearPlanV1,
        w: *const c_void,
        flag: *mut c_void,
        s: *mut c_void,
    ) -> i32;
    pub(super) fn ferrum_upstream_mmvq_plan_v1(
        r: *const UpstreamLinearRequestV1,
        p: *mut UpstreamLinearPlanV1,
    ) -> i32;
    pub(super) fn ferrum_upstream_mmvq_pack_v1(
        p: *const UpstreamLinearPlanV1,
        x: *const c_void,
        stride: u32,
        c: *mut c_void,
        q: *mut c_void,
        s: *mut c_void,
    ) -> i32;
    pub(super) fn ferrum_upstream_mmvq_dot_v1(
        p: *const UpstreamLinearPlanV1,
        w: *const c_void,
        q: *const c_void,
        o: *mut c_void,
        f: *mut c_void,
        s: *mut c_void,
    ) -> i32;
    pub(super) fn ferrum_upstream_mmvq_cast_v1(
        p: *const UpstreamLinearPlanV1,
        x: *const c_void,
        o: *mut c_void,
        stride: u32,
        s: *mut c_void,
    ) -> i32;
    pub(super) fn ferrum_upstream_mmvq_pack_v2(
        p: *const UpstreamLinearPlanV1,
        x: *const c_void,
        stride: u32,
        c: *mut c_void,
        q: *mut c_void,
        flags: *mut c_void,
        s: *mut c_void,
    ) -> i32;
    pub(super) fn ferrum_upstream_mmvq_cast_v2(
        p: *const UpstreamLinearPlanV1,
        x: *const c_void,
        o: *mut c_void,
        stride: u32,
        row_flags: *const c_void,
        weight_flag: *const c_void,
        s: *mut c_void,
    ) -> i32;
    pub(super) fn ferrum_upstream_mmvq_check_weights_v2(
        p: *const UpstreamLinearPlanV1,
        w: *const c_void,
        flag: *mut c_void,
        s: *mut c_void,
    ) -> i32;
}
