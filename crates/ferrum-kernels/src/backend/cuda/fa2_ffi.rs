//! Runtime vLLM FlashAttention-2 shim.
//!
//! This is an opt-in bridge for benchmarking against the exact FA2 paged-KV
//! runner that vLLM uses. The heavy Python/Torch extension dependencies stay
//! outside Ferrum's normal link path: setting `FERRUM_FA2_DIRECT_FFI_SHIM`
//! to a small C ABI `.so` enables the path, and this module resolves it with
//! `dlopen`.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use ferrum_types::{FerrumError, Result};
use half::f16;

pub(super) type Fa2PagedVarlenFn = unsafe extern "C" fn(
    q: *const c_void,
    k: *const c_void,
    v: *const c_void,
    out: *mut c_void,
    lse: *mut c_void,
    cu_seqlens_q: *const c_void,
    seq_lens: *const c_void,
    block_tables: *const c_void,
    num_seqs: c_int,
    total_q_tokens: c_int,
    max_q_len: c_int,
    max_kv_len: c_int,
    num_heads: c_int,
    num_kv_heads: c_int,
    head_dim: c_int,
    block_size: c_int,
    max_blocks_per_seq: c_int,
    stream: *mut c_void,
    err_buf: *mut c_char,
    err_buf_len: usize,
) -> c_int;

struct Fa2Shim {
    _handle: *mut c_void,
    paged_varlen: Fa2PagedVarlenFn,
}

unsafe impl Send for Fa2Shim {}
unsafe impl Sync for Fa2Shim {}

static FA2_SHIM: std::sync::OnceLock<Result<Fa2Shim>> = std::sync::OnceLock::new();

#[link(name = "dl")]
extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

const RTLD_NOW: c_int = 2;
const RTLD_LOCAL: c_int = 0;

fn dl_error_string() -> String {
    unsafe {
        let err = dlerror();
        if err.is_null() {
            "unknown dlerror".to_string()
        } else {
            CStr::from_ptr(err).to_string_lossy().into_owned()
        }
    }
}

fn fa2_direct_ffi_shim_path_from_env() -> Option<String> {
    std::env::vars()
        .find(|(name, _)| name == "FERRUM_FA2_DIRECT_FFI_SHIM")
        .map(|(_, value)| value)
}

fn load_fa2_shim() -> Result<Fa2Shim> {
    let path = fa2_direct_ffi_shim_path_from_env().ok_or_else(|| {
        FerrumError::unsupported(
            "FA2 direct FFI requires FERRUM_FA2_DIRECT_FFI_SHIM=/path/libferrum_fa2_shim.so",
        )
    })?;
    let c_path = CString::new(path.clone()).map_err(|_| {
        FerrumError::model(format!(
            "FERRUM_FA2_DIRECT_FFI_SHIM contains an interior NUL: {path:?}"
        ))
    })?;
    let handle = unsafe { dlopen(c_path.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
    if handle.is_null() {
        return Err(FerrumError::model(format!(
            "dlopen({path}) failed: {}",
            dl_error_string()
        )));
    }

    let sym_name = CString::new("ferrum_fa2_paged_varlen_fwd").unwrap();
    let sym = unsafe { dlsym(handle, sym_name.as_ptr()) };
    if sym.is_null() {
        return Err(FerrumError::model(format!(
            "dlsym(ferrum_fa2_paged_varlen_fwd) failed: {}",
            dl_error_string()
        )));
    }
    let paged_varlen = unsafe { std::mem::transmute::<*mut c_void, Fa2PagedVarlenFn>(sym) };
    Ok(Fa2Shim {
        _handle: handle,
        paged_varlen,
    })
}

fn fa2_shim() -> Result<&'static Fa2Shim> {
    match FA2_SHIM.get_or_init(load_fa2_shim) {
        Ok(shim) => Ok(shim),
        Err(err) => Err(FerrumError::model(format!(
            "FA2 direct FFI shim unavailable: {err}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn call_paged_varlen_fn(
    paged_varlen: Fa2PagedVarlenFn,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<f16>,
    k_pool: &CudaSlice<f16>,
    v_pool: &CudaSlice<f16>,
    out: &mut CudaSlice<f16>,
    lse: &mut CudaSlice<f32>,
    cu_seqlens_q: &CudaSlice<u32>,
    seq_lens: &CudaSlice<u32>,
    block_tables: &CudaSlice<u32>,
    num_seqs: usize,
    total_q_tokens: usize,
    max_q_len: usize,
    max_kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    max_blocks_per_seq: usize,
) -> Result<()> {
    if num_seqs == 0 || total_q_tokens == 0 {
        return Ok(());
    }
    if head_dim != 128 || block_size != 16 {
        return Err(FerrumError::unsupported(format!(
            "FA2 direct FFI only supports head_dim=128 block_size=16, got head_dim={head_dim} block_size={block_size}"
        )));
    }
    if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return Err(FerrumError::model(format!(
            "invalid GQA shape for FA2 direct FFI: heads={num_heads} kv_heads={num_kv_heads}"
        )));
    }
    if lse.len() < num_heads * total_q_tokens {
        return Err(FerrumError::model(format!(
            "FA2 LSE scratch too small: have {} need {}",
            lse.len(),
            num_heads * total_q_tokens
        )));
    }

    let (q_ptr, _qg) = q.device_ptr(stream);
    let (k_ptr, _kg) = k_pool.device_ptr(stream);
    let (v_ptr, _vg) = v_pool.device_ptr(stream);
    let (out_ptr, _og) = out.device_ptr_mut(stream);
    let (lse_ptr, _lg) = lse.device_ptr_mut(stream);
    let (cuq_ptr, _cg) = cu_seqlens_q.device_ptr(stream);
    let (seq_ptr, _sg) = seq_lens.device_ptr(stream);
    let (bt_ptr, _bg) = block_tables.device_ptr(stream);
    let raw_stream = stream.cu_stream() as *mut c_void;
    let mut err_buf = vec![0i8; 512];
    let ret = paged_varlen(
        q_ptr as *const c_void,
        k_ptr as *const c_void,
        v_ptr as *const c_void,
        out_ptr as *mut c_void,
        lse_ptr as *mut c_void,
        cuq_ptr as *const c_void,
        seq_ptr as *const c_void,
        bt_ptr as *const c_void,
        num_seqs as c_int,
        total_q_tokens as c_int,
        max_q_len as c_int,
        max_kv_len as c_int,
        num_heads as c_int,
        num_kv_heads as c_int,
        head_dim as c_int,
        block_size as c_int,
        max_blocks_per_seq as c_int,
        raw_stream,
        err_buf.as_mut_ptr(),
        err_buf.len(),
    );
    if ret != 0 {
        let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        return Err(FerrumError::model(format!(
            "ferrum_fa2_paged_varlen_fwd failed ret={ret}: {msg}"
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn paged_varlen_attention_fa2_ffi(
    stream: &Arc<CudaStream>,
    q: &CudaSlice<f16>,
    k_pool: &CudaSlice<f16>,
    v_pool: &CudaSlice<f16>,
    out: &mut CudaSlice<f16>,
    lse: &mut CudaSlice<f32>,
    cu_seqlens_q: &CudaSlice<u32>,
    seq_lens: &CudaSlice<u32>,
    block_tables: &CudaSlice<u32>,
    num_seqs: usize,
    total_q_tokens: usize,
    max_q_len: usize,
    max_kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    max_blocks_per_seq: usize,
) -> Result<()> {
    let shim = fa2_shim()?;
    unsafe {
        call_paged_varlen_fn(
            shim.paged_varlen,
            stream,
            q,
            k_pool,
            v_pool,
            out,
            lse,
            cu_seqlens_q,
            seq_lens,
            block_tables,
            num_seqs,
            total_q_tokens,
            max_q_len,
            max_kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size,
            max_blocks_per_seq,
        )
    }
}
