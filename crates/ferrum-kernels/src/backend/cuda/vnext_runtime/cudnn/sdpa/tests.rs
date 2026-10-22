use super::*;

fn signature() -> CudnnSdpaSignature {
    CudnnSdpaSignature {
        batch: 2,
        heads: 32,
        query_length: 1024,
        key_value_length: 1045,
        head_dim: 128,
        query_batch_stride: 1045 * 4096,
        key_value_batch_stride: 1045 * 4096,
        output_batch_stride: 1045 * 4096,
        scale_bits: (1.0_f32 / 128.0_f32.sqrt()).to_bits(),
    }
}
fn bindings() -> CudnnSdpaBindings {
    CudnnSdpaBindings {
        q: 0x1000_0000,
        k: 0x2000_0000,
        v: 0x3000_0000,
        o: 0x4000_0000,
        seq_len_q: 0x5000_0000,
        seq_len_kv: 0x5000_0100,
        workspace: 0x6000_0000,
        workspace_bytes: 256,
    }
}

#[test]
fn sliced_bshd_geometry_preserves_original_batch_stride_without_payload_allocation() {
    let s = signature();
    s.validate().unwrap();
    assert_eq!(
        s.extent(s.query_length, s.query_batch_stride).unwrap(),
        (1045 + 1024) * 4096 * 2
    );
    let mut bad = s;
    bad.query_batch_stride = 1024 * 4096 - 1;
    assert!(bad.validate().is_err());
    bad = s;
    bad.head_dim = 64;
    assert!(bad.validate().is_err());
    bad = s;
    bad.scale_bits = f32::NAN.to_bits();
    assert!(bad.validate().is_err());
    bad = s;
    bad.key_value_length = 0;
    assert!(bad.validate().is_err());
    bad = s;
    bad.output_batch_stride = i64::MAX;
    assert!(bad.validate().is_err());
}

#[test]
fn bindings_reject_alias_misalignment_overflow_and_a_different_workspace_quote() {
    let s = signature();
    let b = bindings();
    validate_bindings(s, 256, b).unwrap();
    let mut alias = b;
    alias.o = alias.q;
    assert!(validate_bindings(s, 256, alias).is_err());
    alias = b;
    alias.workspace = alias.k;
    assert!(validate_bindings(s, 256, alias).is_err());
    let mut bad = b;
    bad.seq_len_q += 1;
    assert!(validate_bindings(s, 256, bad).is_err());
    bad = b;
    bad.workspace_bytes = 255;
    assert!(validate_bindings(s, 256, bad).is_err());
    bad = b;
    bad.q = u64::MAX - 15;
    assert!(validate_bindings(s, 256, bad).is_err());
    let mut shared = b;
    shared.v = shared.k;
    validate_bindings(s, 256, shared).unwrap(); // read-only alias is harmless here
}

#[test]
fn binding_limit_rejects_second_live_ticket_and_recovers_after_drop() {
    let limit = Arc::new(BindingLimit {
        live: AtomicUsize::new(0),
        maximum: 1,
    });
    let first = limit.acquire().unwrap();
    assert!(limit.acquire().is_err());
    drop(first);
    let second = limit.acquire().unwrap();
    assert_eq!(limit.live.load(Ordering::Acquire), 1);
    drop(second);
    assert_eq!(limit.live.load(Ordering::Acquire), 0);
}

#[test]
fn cache_policy_rejects_zero_bounds_and_unrepresentable_workspace() {
    let p = CudnnSdpaPlanPolicy {
        maximum_plans: 8,
        maximum_live_bindings: 256,
        opaque_memory_bytes: 1 << 20,
        workspace_limit_bytes: 0,
    };
    p.validate().unwrap();
    assert!(CudnnSdpaPlanPolicy {
        maximum_plans: 0,
        ..p
    }
    .validate()
    .is_err());
    assert!(CudnnSdpaPlanPolicy {
        maximum_live_bindings: 0,
        ..p
    }
    .validate()
    .is_err());
    assert!(CudnnSdpaPlanPolicy {
        opaque_memory_bytes: 0,
        ..p
    }
    .validate()
    .is_err());
    assert!(CudnnSdpaPlanPolicy {
        workspace_limit_bytes: u64::MAX,
        ..p
    }
    .validate()
    .is_err());
}

// A narrow C-ABI fixture exercises descriptor ownership and workspace/engine
// decisions. It neither loads CUDA nor pretends to execute an attention kernel.
mod native_abi {
    use super::super::super::ffi::{Descriptor, GraphApi, Handle};
    use super::*;
    use std::ffi::{c_int, c_void};
    use std::sync::Mutex;

    static SERIAL: Mutex<()> = Mutex::new(());
    static STATE: Mutex<State> = Mutex::new(State {
        created: 0,
        destroyed: 0,
        workspace: 256,
        nondeterministic: false,
        fail_finalize: false,
        fail_destroy: false,
        execute_calls: 0,
    });
    struct State {
        created: usize,
        destroyed: usize,
        workspace: i64,
        nondeterministic: bool,
        fail_finalize: bool,
        fail_destroy: bool,
        execute_calls: usize,
    }
    struct FakeDescriptor {
        kind: c_int,
    }
    unsafe extern "C" fn create(kind: c_int, out: *mut Descriptor) -> c_int {
        let d = Box::into_raw(Box::new(FakeDescriptor { kind }));
        unsafe {
            *out = d.cast();
        }
        STATE.lock().unwrap().created += 1;
        0
    }
    unsafe extern "C" fn destroy(d: Descriptor) -> c_int {
        let mut state = STATE.lock().unwrap();
        if state.fail_destroy {
            return 1;
        }
        unsafe {
            drop(Box::from_raw(d.cast::<FakeDescriptor>()));
        }
        state.destroyed += 1;
        0
    }
    unsafe extern "C" fn set(_: Descriptor, _: c_int, _: c_int, _: i64, _: *const c_void) -> c_int {
        0
    }
    unsafe extern "C" fn finalize(d: Descriptor) -> c_int {
        if unsafe { (*d.cast::<FakeDescriptor>()).kind } == 5 && STATE.lock().unwrap().fail_finalize
        {
            1
        } else {
            0
        }
    }
    unsafe extern "C" fn get(
        _: Descriptor,
        name: c_int,
        _: c_int,
        requested: i64,
        count: *mut i64,
        out: *mut c_void,
    ) -> c_int {
        let state = STATE.lock().unwrap();
        match name {
            202 | 300 => unsafe {
                *count = 1;
            },
            402 | 1301 => unsafe {
                *count = 1;
                if requested > 0 {
                    *out.cast::<i64>() = if name == 402 { state.workspace } else { 17 };
                }
            },
            1303 => unsafe {
                *count = i64::from(state.nondeterministic);
                if requested > 0 && state.nondeterministic {
                    *out.cast::<c_int>() = 4;
                }
            },
            302 => unsafe {
                *count = 0;
            },
            _ => return 1,
        }
        0
    }
    unsafe extern "C" fn execute(_: Handle, _: Descriptor, _: Descriptor) -> c_int {
        STATE.lock().unwrap().execute_calls += 1;
        0
    }
    fn api() -> GraphApi {
        GraphApi {
            create,
            destroy,
            set,
            get,
            finalize,
            execute,
        }
    }
    fn reset(workspace: i64, nondeterministic: bool) {
        *STATE.lock().unwrap() = State {
            created: 0,
            destroyed: 0,
            workspace,
            nondeterministic,
            fail_finalize: false,
            fail_destroy: false,
            execute_calls: 0,
        };
    }
    fn no_native_execution_and_all_descriptors_released() {
        let s = STATE.lock().unwrap();
        assert_eq!(s.execute_calls, 0);
        assert_eq!(s.created, s.destroyed);
    }

    #[test]
    fn native_preparation_quotes_actual_workspace_and_retains_descriptors_until_retirement() {
        let _serial = SERIAL.lock().unwrap();
        reset(256, false);
        let mut p = match native::NativePlan::build(&api(), std::ptr::null_mut(), signature(), 512)
        {
            Ok(p) => p,
            Err(e) => panic!("{}", e.message),
        };
        assert_eq!(p.workspace_bytes, 256);
        assert!(p.identity.contains("engine=17"));
        {
            let s = STATE.lock().unwrap();
            assert!(s.created > s.destroyed);
            assert_eq!(s.execute_calls, 0);
        }
        assert!(p.destroy(&api()));
        no_native_execution_and_all_descriptors_released();
    }
    #[test]
    fn over_budget_and_nondeterministic_engines_fail_without_execution_and_clean_up() {
        let _serial = SERIAL.lock().unwrap();
        reset(257, false);
        let e = match native::NativePlan::build(&api(), std::ptr::null_mut(), signature(), 256) {
            Err(e) => e,
            Ok(_) => panic!("oversized workspace accepted"),
        };
        assert!(e.cleanup_known);
        no_native_execution_and_all_descriptors_released();
        reset(0, true);
        let e = match native::NativePlan::build(&api(), std::ptr::null_mut(), signature(), 256) {
            Err(e) => e,
            Ok(_) => panic!("nondeterministic engine accepted"),
        };
        assert!(e.message.contains("nondeterministic"));
        assert!(e.cleanup_known);
        no_native_execution_and_all_descriptors_released();
    }
    #[test]
    fn native_finalize_error_preserves_cleanup_status() {
        let _serial = SERIAL.lock().unwrap();
        reset(256, false);
        STATE.lock().unwrap().fail_finalize = true;
        let e = match native::NativePlan::build(&api(), std::ptr::null_mut(), signature(), 256) {
            Err(e) => e,
            Ok(_) => panic!("failed finalize accepted"),
        };
        assert!(e.cleanup_known);
        no_native_execution_and_all_descriptors_released();
    }

    #[test]
    fn failed_descriptor_retirement_keeps_handles_for_a_later_safe_cleanup() {
        let _serial = SERIAL.lock().unwrap();
        reset(256, false);
        let mut p = match native::NativePlan::build(&api(), std::ptr::null_mut(), signature(), 512)
        {
            Ok(p) => p,
            Err(e) => panic!("{}", e.message),
        };
        STATE.lock().unwrap().fail_destroy = true;
        assert!(!p.destroy(&api()));
        {
            let state = STATE.lock().unwrap();
            assert_eq!(state.destroyed, 0);
            assert!(state.created > 0);
        }
        STATE.lock().unwrap().fail_destroy = false;
        assert!(p.destroy(&api()));
        no_native_execution_and_all_descriptors_released();
    }
}
