//! Optional CUDA profiler correlation for vNext reusable executable launches.
//!
//! NVTX is loaded only for full device attribution. Normal product execution
//! neither resolves the library nor enters an NVTX range.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::sync::OnceLock;

type NvtxRangePush = unsafe extern "C" fn(*const c_char) -> c_int;
type NvtxRangePop = unsafe extern "C" fn() -> c_int;

struct NvtxApi {
    _handle: *mut c_void,
    range_push: NvtxRangePush,
    range_pop: NvtxRangePop,
}

unsafe impl Send for NvtxApi {}
unsafe impl Sync for NvtxApi {}

static NVTX_API: OnceLock<Option<NvtxApi>> = OnceLock::new();

fn nvtx_api() -> Option<&'static NvtxApi> {
    NVTX_API
        .get_or_init(|| {
            let api = load_nvtx_api();
            if api.is_none() {
                tracing::debug!(
                    "CUDA reusable executable NVTX correlation is unavailable; internal timing remains enabled"
                );
            }
            api
        })
        .as_ref()
}

#[cfg(unix)]
fn load_nvtx_api() -> Option<NvtxApi> {
    for library in [
        c"libnvToolsExt.so.1".as_ptr(),
        c"libnvToolsExt.so".as_ptr(),
        c"/usr/local/cuda/lib64/libnvToolsExt.so.1".as_ptr(),
        c"/usr/local/cuda/lib64/libnvToolsExt.so".as_ptr(),
        c"/usr/local/cuda/targets/x86_64-linux/lib/libnvToolsExt.so.1".as_ptr(),
        c"/usr/local/cuda/targets/x86_64-linux/lib/libnvToolsExt.so".as_ptr(),
    ] {
        let handle = unsafe { libc::dlopen(library, libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if handle.is_null() {
            continue;
        }
        let range_push = unsafe { libc::dlsym(handle, c"nvtxRangePushA".as_ptr()) };
        let range_pop = unsafe { libc::dlsym(handle, c"nvtxRangePop".as_ptr()) };
        if range_push.is_null() || range_pop.is_null() {
            unsafe {
                libc::dlclose(handle);
            }
            continue;
        }
        return Some(NvtxApi {
            _handle: handle,
            range_push: unsafe { std::mem::transmute::<*mut c_void, NvtxRangePush>(range_push) },
            range_pop: unsafe { std::mem::transmute::<*mut c_void, NvtxRangePop>(range_pop) },
        });
    }
    None
}

#[cfg(not(unix))]
fn load_nvtx_api() -> Option<NvtxApi> {
    None
}

pub(super) fn prepare() {
    let _ = nvtx_api();
}

pub(super) fn correlate_replay_launch<T>(label: &CStr, launch: impl FnOnce() -> T) -> T {
    let Some(api) = nvtx_api() else {
        return launch();
    };
    correlate_replay_launch_with_api(api, label, launch)
}

fn correlate_replay_launch_with_api<T>(
    api: &NvtxApi,
    label: &CStr,
    launch: impl FnOnce() -> T,
) -> T {
    let push_status = unsafe { (api.range_push)(label.as_ptr()) };
    let result = launch();
    let pop_status = unsafe { (api.range_pop)() };
    if push_status < 0 || pop_status < 0 {
        tracing::debug!(
            push_status,
            pop_status,
            "CUDA reusable executable NVTX correlation reported an unbalanced range"
        );
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU8, Ordering};

    static CALL_ORDER: AtomicU8 = AtomicU8::new(0);

    unsafe extern "C" fn rejected_push(_label: *const c_char) -> c_int {
        assert_eq!(CALL_ORDER.fetch_add(1, Ordering::SeqCst), 0);
        -1
    }

    unsafe extern "C" fn required_pop() -> c_int {
        assert_eq!(CALL_ORDER.fetch_add(1, Ordering::SeqCst), 2);
        0
    }

    #[test]
    fn replay_correlation_always_balances_a_recorded_push() {
        CALL_ORDER.store(0, Ordering::SeqCst);
        let api = NvtxApi {
            _handle: std::ptr::null_mut(),
            range_push: rejected_push,
            range_pop: required_pop,
        };
        let result = correlate_replay_launch_with_api(&api, c"ferrum.cuda.replay/test", || {
            assert_eq!(CALL_ORDER.fetch_add(1, Ordering::SeqCst), 1);
            7
        });
        assert_eq!(result, 7);
        assert_eq!(CALL_ORDER.load(Ordering::SeqCst), 3);
    }
}
