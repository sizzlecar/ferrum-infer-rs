use super::*;
unsafe extern "C" {
    #[cfg(ferrum_native_mmq_pdl)]
    pub(super) fn ferrum_upstream_mmq_programmatic_support_v1(
        plan: *const UpstreamLinearPlanV1,
        supported: *mut u32,
    ) -> i32;
    #[cfg(ferrum_native_mmq_pdl)]
    pub(super) fn ferrum_upstream_mmq_programmatic_pack_f16_small_v2(
        plan: *const UpstreamLinearPlanV1,
        input: *const c_void,
        stride: u32,
        converted: *mut c_void,
        packed: *mut c_void,
        rows: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(ferrum_native_mmq_pdl)]
    pub(super) fn ferrum_upstream_mmq_programmatic_dot_v1(
        plan: *const UpstreamLinearPlanV1,
        weights: *const c_void,
        packed: *const c_void,
        output: *mut c_void,
        fixup: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(ferrum_native_mmq_pdl)]
    pub(super) fn ferrum_upstream_mmq_programmatic_cast_v2(
        plan: *const UpstreamLinearPlanV1,
        input: *const c_void,
        output: *mut c_void,
        stride: u32,
        rows: *const c_void,
        weights: *const c_void,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(ferrum_native_mmvq_pdl)]
    pub(super) fn ferrum_upstream_mmvq_pdl_support_v1(
        plan: *const UpstreamLinearPlanV1,
        dot_kind: u32,
        supported: *mut u32,
    ) -> i32;
    #[cfg(ferrum_native_mmvq_pdl)]
    pub(super) fn ferrum_upstream_mmvq_pdl_pack_v2(
        plan: *const UpstreamLinearPlanV1,
        input: *const c_void,
        stride: u32,
        converted: *mut c_void,
        packed: *mut c_void,
        rows: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(ferrum_native_mmvq_pdl)]
    pub(super) fn ferrum_upstream_mmvq_pdl_dot_v1(
        plan: *const UpstreamLinearPlanV1,
        weights: *const c_void,
        packed: *const c_void,
        output: *mut c_void,
        fixup: *mut c_void,
        stream: *mut c_void,
        dot_kind: u32,
    ) -> i32;
    #[cfg(ferrum_native_mmvq_pdl)]
    pub(super) fn ferrum_upstream_mmvq_pdl_cast_v2(
        plan: *const UpstreamLinearPlanV1,
        input: *const c_void,
        output: *mut c_void,
        stride: u32,
        rows: *const c_void,
        weights: *const c_void,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(feature = "cuda-q4-shared-weight-decode")]
    pub(super) fn ferrum_upstream_mmvq_q4_shared_weight_decode_m4_dot_v1(
        plan: *const UpstreamLinearPlanV1,
        weights: *const c_void,
        packed: *const c_void,
        output: *mut c_void,
        fixup: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(feature = "cuda-q5-shared-weight-decode")]
    pub(super) fn ferrum_upstream_mmvq_q5_shared_weight_decode_m4_dot_v1(
        plan: *const UpstreamLinearPlanV1,
        weights: *const c_void,
        packed: *const c_void,
        output: *mut c_void,
        fixup: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(feature = "cuda-q4-shared-weight-decode")]
    pub(super) fn ferrum_upstream_mmvq_q4_shared_weight_decode_dot_v1(
        plan: *const UpstreamLinearPlanV1,
        weights: *const c_void,
        packed: *const c_void,
        output: *mut c_void,
        fixup: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(feature = "cuda-q5-shared-weight-decode")]
    pub(super) fn ferrum_upstream_mmvq_q5_shared_weight_decode_dot_v1(
        plan: *const UpstreamLinearPlanV1,
        weights: *const c_void,
        packed: *const c_void,
        output: *mut c_void,
        fixup: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
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
