//! A prepared family selects one complete ABI. No symbol-level fallback.
use super::*;
type P = UpstreamLinearPlanV1;
type V = c_void;
pub(super) struct Api {
    pub check_weights: unsafe extern "C" fn(*const P, *const V, *mut V, *mut V) -> i32,
    pub pack_v1: unsafe extern "C" fn(*const P, *const V, u32, *mut V, *mut V, *mut V) -> i32,
    pub pack_v2:
        unsafe extern "C" fn(*const P, *const V, u32, *mut V, *mut V, *mut V, *mut V) -> i32,
    pub dot: unsafe extern "C" fn(*const P, *const V, *const V, *mut V, *mut V, *mut V) -> i32,
    pub cast_v1: unsafe extern "C" fn(*const P, *const V, *mut V, u32, *mut V) -> i32,
    pub cast_v2:
        unsafe extern "C" fn(*const P, *const V, *mut V, u32, *const V, *const V, *mut V) -> i32,
}
static ORIGINAL_MMQ: Api = Api {
    check_weights: ffi::ferrum_upstream_mmq_check_weights_v2,
    pack_v1: ffi::ferrum_upstream_mmq_pack_v1,
    pack_v2: ffi::ferrum_upstream_mmq_pack_v2,
    dot: ffi::ferrum_upstream_mmq_dot_v1,
    cast_v1: ffi::ferrum_upstream_mmq_cast_v1,
    cast_v2: ffi::ferrum_upstream_mmq_cast_v2,
};
static ORIGINAL_MMVQ: Api = Api {
    check_weights: ffi::ferrum_upstream_mmvq_check_weights_v2,
    pack_v1: ffi::ferrum_upstream_mmvq_pack_v1,
    pack_v2: ffi::ferrum_upstream_mmvq_pack_v2,
    dot: ffi::ferrum_upstream_mmvq_dot_v1,
    cast_v1: ffi::ferrum_upstream_mmvq_cast_v1,
    cast_v2: ffi::ferrum_upstream_mmvq_cast_v2,
};
#[cfg(feature = "cuda-upstream-extra-linear")]
static EXTRA_MMQ: Api = Api {
    check_weights: extra_ffi::ferrum_upstream_extra_mmq_check_weights_v2,
    pack_v1: extra_ffi::ferrum_upstream_extra_mmq_pack_v1,
    pack_v2: extra_ffi::ferrum_upstream_extra_mmq_pack_v2,
    dot: extra_ffi::ferrum_upstream_extra_mmq_dot_v1,
    cast_v1: extra_ffi::ferrum_upstream_extra_mmq_cast_v1,
    cast_v2: extra_ffi::ferrum_upstream_extra_mmq_cast_v2,
};
#[cfg(feature = "cuda-upstream-extra-linear")]
static EXTRA_MMVQ: Api = Api {
    check_weights: extra_ffi::ferrum_upstream_extra_mmvq_check_weights_v2,
    pack_v1: extra_ffi::ferrum_upstream_extra_mmvq_pack_v1,
    pack_v2: extra_ffi::ferrum_upstream_extra_mmvq_pack_v2,
    dot: extra_ffi::ferrum_upstream_extra_mmvq_dot_v1,
    cast_v1: extra_ffi::ferrum_upstream_extra_mmvq_cast_v1,
    cast_v2: extra_ffi::ferrum_upstream_extra_mmvq_cast_v2,
};
pub(super) fn api(family: Family, algorithm: u32) -> Result<&'static Api, Error> {
    match (family, algorithm) {
        (Family::Original, 1) => Ok(&ORIGINAL_MMQ),
        (Family::Original, 2) => Ok(&ORIGINAL_MMVQ),
        #[cfg(feature = "cuda-upstream-extra-linear")]
        (Family::Extra, 1) => Ok(&EXTRA_MMQ),
        #[cfg(feature = "cuda-upstream-extra-linear")]
        (Family::Extra, 2) => Ok(&EXTRA_MMVQ),
        #[cfg(not(feature = "cuda-upstream-extra-linear"))]
        (Family::Extra, _) => Err(Error::Unavailable),
        _ => Err(Error::AbiMismatch),
    }
}
