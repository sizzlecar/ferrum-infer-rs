//! Metal backend — one unified backend with dtype-tagged buffers.
//!
//! Precision is carried as a runtime tag on [`MetalBuf`] rather than a
//! separate backend type. Ops branch on buffer dtype where it matters
//! (`gemm`, `embedding_lookup`); the rest assert F32 and go through the
//! existing f32 shaders.
//!
//! The default policy:
//!   - Activations (`alloc`, `from_slice`, scratch) are F32 — precision-
//!     sensitive paths (norm / rope / attention) stay unchanged.
//!   - Weights loaded through the original context go F16 for tensors past a
//!     size threshold when its typed weight policy selects F16; otherwise
//!     they stay F32.
//!
//! The same shape generalises: an INT8 / GPTQ variant would add another
//! `Dtype` case + the corresponding shader, without introducing a new
//! `Backend` type. Hardware (Metal / CUDA / CPU) and precision remain
//! orthogonal, matching how the `Backend::GptqStore` + `gemm_gptq`
//! trait plumbing already handles INT4 on CUDA.
//!
//! Command buffer lifecycle mirrors llama.cpp: one retained buffer per
//! forward pass that accumulates encodes; `sync()` commits and waits.
//!
//! ## Layout (Audit #8 split)
//!
//! - `mod.rs` (this file) — shared state, `MetalBuf`/`MetalContext`,
//!   `impl Backend for MetalBackend`, empty stubs for `BackendGraph`,
//!   `BackendCollective`, `BackendQuantMarlin`, and `BackendKvDtype<KvFp16>`.
//! - [`quant`] — `MetalQuantStore` + `BackendQuantGguf` + GGUF k-quant
//!   matmul dispatcher + MoE expert-stack builders.
//! - [`paged`] — `BackendPagedKv` (paged-KV decode + split_qkv into pool).
//! - [`moe`]   — `BackendMoeFused` (route/topk, silu/mul, weighted-sum).

mod moe;
mod paged;
pub(crate) mod quant;
mod residency;
mod retained_memory;
pub mod vnext_ops;
pub mod vnext_runtime;

#[cfg(test)]
mod copy_slice_tests;
#[cfg(test)]
mod host_access_tests;
#[cfg(test)]
mod instance_config_tests;
#[cfg(test)]
mod instance_queue_tests;
#[cfg(test)]
mod mmap_lifetime_tests;

// Raw encoders and device pipelines stay behind the context-bound Backend
// and Linear contracts. The typed runtime keeps its own explicit providers.
pub(crate) mod k_quant_gemm;
pub(crate) mod moe_post_ops;
pub(crate) mod moe_post_ops_batched;
pub(crate) mod moe_router;
pub(crate) mod q4_k;
pub(crate) mod q4_k_gemm;
pub(crate) mod q4_k_gemv;
pub(crate) mod q4_k_gemv_v2;
pub(crate) mod q4_k_moe_id_gate_up_silu;
pub(crate) mod q4_k_moe_id_gate_up_silu_batched;
pub(crate) mod q4_k_moe_id_gemm;
pub(crate) mod q4_k_moe_id_gemv;
pub(crate) mod q4_k_moe_id_gemv_batched;
pub(crate) mod q5_k_gemv;
pub(crate) mod q6_k_gemm;
pub(crate) mod q6_k_gemv;
pub(crate) mod q6_k_moe_id_gemm;
pub(crate) mod q6_k_moe_id_gemv;
pub(crate) mod q6_k_moe_id_gemv_batched;

pub(crate) use quant::{
    dispatch_gemv_moe_id, dispatch_gemv_moe_id_offset, metal_gemm_quant_dispatch, MetalQuantStore,
};

use super::{AttnConfig, Backend, SrcDtype};
use crate::attention::metal::pipelines::MetalPipelines;
use crate::attention::AttentionParams;
use ferrum_runtime::RetainedHostMemoryRegion;
use ferrum_types::{BackendContextConfig, FerrumError, MetalBackendConfig, Result};
use half::{bf16, f16};
use metal::{Device, MTLResourceOptions, MTLSize};
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock, Weak};

// ── Shared Metal state ────────────────────────────────────────────────

/// One registered host-memory region eligible for zero-copy Metal binding.
///
/// The exact registered range and its stable owner come from `memory`.
/// Each native no-copy buffer also retains this host memory independently
/// of its registration or Rust weight, through its deallocator block.
///
/// Unlike the early "one big buffer" attempt, we do **not** create a
/// single MTLBuffer for the whole region here — that approach worked but
/// regressed decode tok/s ~30× on M1 Max because Apple's GPU residency
/// logic on large buffers (16 GiB) is very expensive per dispatch.
/// Instead, `quant::buffer_for_quant_bytes` creates a small per-tensor
/// MTLBuffer via `newBufferWithBytesNoCopy` covering only the
/// page-aligned region around each tensor — same memory footprint, but
/// many small buffers fit Apple's GPU residency model the way llama.cpp
/// observed.
struct MetalMmapEntry {
    residency: Option<Mutex<residency::MetalResidencySet>>,
    // Rust drops fields in declaration order. Keep this after `residency` so
    // the mmap remains valid until the set releases its Metal allocations.
    memory: RetainedHostMemoryRegion,
}

/// Keeps residency requested while a quantized weight exists. Native buffers
/// independently retain the host memory until their final native release.
pub struct MetalMmapLease {
    _entry: Arc<MetalMmapEntry>,
}

impl MetalMmapLease {
    fn new(entry: Arc<MetalMmapEntry>) -> Self {
        Self { _entry: entry }
    }
}

/// Construction-time handle for one registered GGUF mmap.
#[must_use = "keep this registration alive while constructing Metal quant stores"]
pub struct MetalMmapRegistration {
    entry: Arc<MetalMmapEntry>,
}

impl MetalMmapRegistration {
    /// Publish all registered zero-copy allocations and request residency.
    pub fn commit_residency(&self) -> Result<()> {
        let Some(residency) = self.entry.residency.as_ref() else {
            return Ok(());
        };
        let residency = residency
            .lock()
            .map_err(|e| FerrumError::model(format!("Metal residency set poisoned: {e}")))?;
        let stats = residency.commit_and_request();
        tracing::info!(
            allocation_count = stats.allocation_count,
            allocated_size_bytes = stats.allocated_size,
            "Metal GGUF residency set committed"
        );
        Ok(())
    }
}

pub(crate) struct MetalState {
    pub(crate) pipes: Arc<MetalPipelines>,
    /// Weak index of mmap regions eligible for zero-copy Metal buffers.
    /// Per-weight leases, rather than this registry, own each live mapping.
    mmaps: Mutex<Vec<Weak<MetalMmapEntry>>>,
}
// Default-device pipelines and the weak mmap index may be shared. Execution
// queues, completion state and instance policy never live in this catalog.
static METAL_STATE: OnceLock<MetalState> = OnceLock::new();
pub(crate) fn st() -> &'static MetalState {
    METAL_STATE.get_or_init(|| MetalState {
        pipes: Arc::new(MetalPipelines::new(&Device::system_default().unwrap())),
        mmaps: Mutex::new(Vec::new()),
    })
}

/// Register a host-memory region (typically the full mmap of a GGUF file)
/// so subsequent `load_quant*` calls whose input slice lives inside this
/// range can use the shared zero-copy `MTLBuffer` instead of allocating a
/// fresh device-resident copy.
///
/// The stable memory owner proves the registered range and keeps it mapped.
/// The returned registration is held during construction; each zero-copy
/// quant store then owns a residency lease. Each native buffer separately
/// retains the host memory, including native command and raw-buffer clones.
///
/// Constraints (`newBufferWithBytesNoCopy`):
///   * the native buffer's base pointer must be page-aligned
///   * the wrapped length must be a multiple of the page size
///
/// Loading proves that the entire aligned native view is within the stable
/// owner's readable range. A tensor without such a view uses the copy path;
/// registration does not grant reads beyond the owner's declared bytes.
///
/// Returns a construction handle on success. Duplicate registration of the
/// same live mapping shares the existing entry. Diagnostics use the caller's
/// explicit options even when another instance registered that mapping first.
pub fn register_gguf_mmap(
    memory: RetainedHostMemoryRegion,
    options: &MetalBackendConfig,
) -> Result<MetalMmapRegistration> {
    let slice = memory.bytes();
    let base_addr = slice.as_ptr() as usize;
    let trace = options.mmap_trace;
    if trace {
        eprintln!(
            "[mmap] register file at 0x{base_addr:x} len={} ({:.2} GB)",
            slice.len(),
            slice.len() as f64 / 1e9
        );
    }
    let state = st();
    let mut guard = state
        .mmaps
        .lock()
        .map_err(|e| FerrumError::model(format!("register_gguf_mmap: registry poisoned: {e}")))?;
    guard.retain(|entry| entry.strong_count() > 0);
    if let Some(entry) = guard.iter().filter_map(Weak::upgrade).find(|entry| {
        let registered = entry.memory.bytes();
        registered.as_ptr() as usize == base_addr && registered.len() == slice.len()
    }) {
        return Ok(MetalMmapRegistration { entry });
    }
    let residency = residency::MetalResidencySet::new(state.pipes.device())?.map(Mutex::new);
    if residency.is_none() {
        tracing::info!("Metal residency sets unavailable; using implicit resource residency");
    }
    let entry = Arc::new(MetalMmapEntry { residency, memory });
    guard.push(Arc::downgrade(&entry));
    Ok(MetalMmapRegistration { entry })
}

fn registered_mmap_entry(bytes: &[u8]) -> Option<Arc<MetalMmapEntry>> {
    if bytes.is_empty() {
        return None;
    }
    let ptr = bytes.as_ptr() as usize;
    let len = bytes.len();
    let end = match ptr.checked_add(len) {
        Some(e) => e,
        None => return None,
    };
    let mut guard = match st().mmaps.lock() {
        Ok(g) => g,
        Err(_) => return None,
    };
    guard.retain(|entry| entry.strong_count() > 0);
    for entry in guard.iter().filter_map(Weak::upgrade) {
        let registered = entry.memory.bytes();
        let entry_start = registered.as_ptr() as usize;
        let entry_end = match entry_start.checked_add(registered.len()) {
            Some(e) => e,
            None => continue,
        };
        if ptr >= entry_start && end <= entry_end {
            return Some(entry);
        }
    }
    None
}

/// Add a zero-copy buffer to the residency set owned by its registered mmap.
/// The returned lease ties the mapping and residency lifetime to the weight.
fn add_registered_mmap_buffer(
    entry: Arc<MetalMmapEntry>,
    buffer: &metal::BufferRef,
) -> Result<MetalMmapLease> {
    if let Some(residency) = entry.residency.as_ref() {
        residency
            .lock()
            .map_err(|e| FerrumError::model(format!("Metal residency set poisoned: {e}")))?
            .add_allocation(buffer);
    }
    Ok(MetalMmapLease::new(entry))
}

// ── Dtype tag + tagged buffer ─────────────────────────────────────────

/// Element storage type for a [`MetalBuf`]. Same shape generalises to INT8
/// / bf16 etc. when their shaders land — just add a variant + wire it in
/// the op dispatches.
// Phase A: Dtype moved to `crate::backend::dtype::Dtype` (shared across
// CPU/CUDA/Metal). Re-export here so existing `metal::Dtype` users keep
// working without touching them.
pub use super::dtype::Dtype;

/// Metal device buffer with a runtime dtype tag and logical element count.
///
/// Two buffers of identical raw bytes but different `dtype` are treated as
/// different types for shader selection. `n` is the number of logical
/// elements; `raw.length()` is `n * dtype.bytes_per_elem()`.
pub struct MetalBuf {
    pub(crate) raw: metal::Buffer,
    pub(crate) dtype: Dtype,
    pub(crate) n: usize,
}

impl MetalBuf {
    pub fn raw(&self) -> &metal::Buffer {
        &self.raw
    }
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }
    pub fn len(&self) -> usize {
        self.n
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
    pub fn is_f16(&self) -> bool {
        matches!(self.dtype, Dtype::F16)
    }

    #[inline]
    pub(crate) fn expect_f32<'a>(&'a self, what: &str) -> &'a metal::Buffer {
        debug_assert!(
            matches!(self.dtype, Dtype::F32),
            "{what}: expected F32 buffer, got {:?}",
            self.dtype
        );
        &self.raw
    }
    #[inline]
    pub(crate) fn expect_f32_mut<'a>(&'a mut self, what: &str) -> &'a mut metal::Buffer {
        debug_assert!(
            matches!(self.dtype, Dtype::F32),
            "{what}: expected F32 buffer, got {:?}",
            self.dtype
        );
        &mut self.raw
    }
}

// Safety: the retained native Metal allocation supports sharing its object
// across threads. Retention is separate from content readiness: host access
// and dependent work on another context still require producer completion.
unsafe impl Send for MetalBuf {}
unsafe impl Sync for MetalBuf {}

// ── Context ───────────────────────────────────────────────────────────

/// Metal context — one in-flight command buffer that accumulates encodes
/// across multiple `Backend` method calls. `sync()` commits + waits and
/// drops the handle so the next call creates a fresh buffer. The context owns
/// its queue and immutable policy, and may share device-bound pipelines.
pub struct MetalContext {
    config: BackendContextConfig,
    pipes: Arc<MetalPipelines>,
    queue: metal::CommandQueue,
    #[cfg(test)]
    last_attention_kernel: Option<crate::attention::metal::pipelines::MetalAttentionKernel>,
    // Own the Objective-C objects. The command queue's lifetime does not make
    // an autoreleased command buffer or encoder a Rust 'static reference.
    cmd: Option<metal::CommandBuffer>,
    encoder: Option<metal::ComputeCommandEncoder>,
    // Kept across an unsuccessful observation. A retry observes this exact
    // command buffer; it must not commit a replacement or forget its status.
    submitted: Option<metal::CommandBuffer>,
    last_completion: Option<ObservedMetalCommand>,
    #[cfg(test)]
    completion_observation_override: Option<ObservedMetalCommand>,
}

struct ObservedMetalCommand {
    status: metal::MTLCommandBufferStatus,
    code: Option<i64>,
    detail: Option<String>,
}

impl MetalContext {
    /// Permit ordinary unsubmitted encoding/preparation, but never reuse an
    /// unresolved submission or forget a terminal failure from this invocation.
    fn require_reusable_state(&self) -> Result<()> {
        if self.submitted.is_some() {
            return Err(FerrumError::backend(
                "Metal context has an unresolved submission",
            ));
        }
        if let Some(observed) = &self.last_completion {
            classify_command_buffer_completion(
                observed.status,
                observed.code,
                observed.detail.as_deref(),
            )
            .into_result()?;
        }
        Ok(())
    }

    fn cmd(&mut self) -> &metal::CommandBufferRef {
        self.require_reusable_state()
            .unwrap_or_else(|error| std::panic::panic_any(error));
        if self.cmd.is_none() {
            self.last_completion = None;
            self.cmd = Some(self.queue.new_command_buffer().to_owned());
        }
        self.cmd.as_ref().expect("command buffer was initialized")
    }

    /// Keep one encoder open across adjacent compute operations, preserving
    /// the existing command order and batching behavior.
    pub(crate) fn compute_encoder(&mut self) -> &metal::ComputeCommandEncoderRef {
        self.require_reusable_state()
            .unwrap_or_else(|error| std::panic::panic_any(error));
        if self.encoder.is_none() {
            let encoder = self.cmd().new_compute_command_encoder().to_owned();
            self.encoder = Some(encoder);
        }
        self.encoder.as_ref().expect("encoder was initialized")
    }

    /// Borrow the pipeline catalog and encoder from the same original owner.
    /// Field splitting avoids cloning shared catalog/policy handles per op.
    pub(crate) fn encoder_parts(
        &mut self,
    ) -> (
        &MetalPipelines,
        &metal::ComputeCommandEncoderRef,
        &ferrum_types::MetalBackendConfig,
    ) {
        self.compute_encoder();
        (
            &self.pipes,
            self.encoder.as_ref().expect("encoder was initialized"),
            &self.config.metal,
        )
    }

    /// End the sticky compute encoder before callers use another encoder on
    /// this invocation's original command buffer. This does not submit it.
    pub(crate) fn command_parts(
        &mut self,
    ) -> (
        &MetalPipelines,
        &metal::CommandBufferRef,
        &ferrum_types::MetalBackendConfig,
    ) {
        self.require_reusable_state()
            .unwrap_or_else(|error| std::panic::panic_any(error));
        self.compute_encoder_end();
        self.cmd();
        (
            &self.pipes,
            self.cmd.as_ref().expect("command buffer was initialized"),
            &self.config.metal,
        )
    }

    fn compute_encoder_end(&mut self) {
        if let Some(enc) = self.encoder.take() {
            enc.end_encoding();
        }
    }

    fn submit_and_wait(&mut self) -> Option<&metal::CommandBufferRef> {
        if self.submitted.is_none() {
            self.compute_encoder_end();
            self.submitted = self.cmd.take();
            self.submitted.as_ref()?.commit();
        }
        let cmd = self.submitted.as_ref()?;
        cmd.wait_until_completed();
        Some(cmd)
    }

    fn synchronize_for_host_access(&mut self) -> Result<()> {
        // Preserve the pristine empty no-op, not a prior failed observation.
        // A retry may prove an unresolved submission terminal; until then its
        // shared buffers cannot be accessed by the host.
        if MetalBackend::preparation_is_host_only(self) {
            return Ok(());
        }
        self.observe_completion().into_result()
    }

    fn observe_completion(&mut self) -> crate::backend::BackendCompletion {
        use crate::backend::BackendCompletion;
        if self.cmd.is_none() && self.submitted.is_none() {
            return match &self.last_completion {
                Some(observed) => classify_command_buffer_completion(
                    observed.status,
                    observed.code,
                    observed.detail.as_deref(),
                ),
                None => BackendCompletion::Unresolved(FerrumError::backend(
                    "Metal checked synchronization has no pending command buffer to submit",
                )),
            };
        }
        let cmd = self.submit_and_wait().expect("pending command is retained");
        let status = cmd.status();
        let (code, detail) = if status == metal::MTLCommandBufferStatus::Completed {
            (None, None)
        } else {
            command_buffer_error(cmd)
        };
        // Test-only observation fault, after a real native wait. It does not
        // manufacture a GPU failure or change the command's native status.
        #[cfg(test)]
        let (status, code, detail) = self
            .completion_observation_override
            .take()
            .map(|observed| (observed.status, observed.code, observed.detail))
            .unwrap_or((status, code, detail));
        let completion = classify_command_buffer_completion(status, code, detail.as_deref());
        if completion.is_terminal() {
            self.last_completion = Some(ObservedMetalCommand {
                status,
                code,
                detail,
            });
            self.submitted = None;
        }
        completion
    }

    fn flush_checked(&mut self) -> Result<()> {
        // Operator validation requires a real submission in this call. The
        // lifecycle observer can separately reuse an earlier terminal fact
        // when no subsequent command has been encoded into this context.
        if self.cmd.is_none() && self.submitted.is_none() {
            return Err(FerrumError::backend(
                "Metal checked synchronization has no pending command buffer to submit",
            ));
        }
        self.observe_completion().into_result()
    }
}

/// Read the NSError while the waited command buffer is still alive. The metal
/// crate exposes status but does not wrap MTLCommandBuffer.error.
#[allow(
    unexpected_cfgs,
    reason = "objc 0.2 macros expand their legacy cargo-clippy feature cfg in the calling crate"
)]
fn command_buffer_error(cmd: &metal::CommandBufferRef) -> (Option<i64>, Option<String>) {
    use metal::objc::runtime::Object;
    use metal::objc::{msg_send, sel, sel_impl};
    use std::ffi::CStr;

    unsafe {
        let error: *mut Object = msg_send![cmd, error];
        if error.is_null() {
            return (None, None);
        }
        let code: metal::NSInteger = msg_send![error, code];
        let description: *mut Object = msg_send![error, localizedDescription];
        let detail = if description.is_null() {
            None
        } else {
            let utf8: *const std::os::raw::c_char = msg_send![description, UTF8String];
            (!utf8.is_null()).then(|| CStr::from_ptr(utf8).to_string_lossy().into_owned())
        };
        (Some(code as i64), detail)
    }
}

fn classify_command_buffer_completion(
    status: metal::MTLCommandBufferStatus,
    code: Option<i64>,
    detail: Option<&str>,
) -> crate::backend::BackendCompletion {
    use crate::backend::BackendCompletion;
    if status == metal::MTLCommandBufferStatus::Completed {
        return BackendCompletion::Succeeded;
    }
    let mut message =
        format!("Metal command buffer did not complete successfully: status {status:?}");
    if let Some(code) = code {
        message.push_str(&format!(", driver error code {code}"));
    }
    if let Some(detail) = detail {
        message.push_str(": ");
        message.push_str(detail);
    }
    let error = FerrumError::backend(message);
    if status == metal::MTLCommandBufferStatus::Error {
        BackendCompletion::Failed(error)
    } else {
        BackendCompletion::Unresolved(error)
    }
}

#[cfg(test)]
fn validate_command_buffer_completion(
    status: metal::MTLCommandBufferStatus,
    code: Option<i64>,
    detail: Option<&str>,
) -> Result<()> {
    classify_command_buffer_completion(status, code, detail).into_result()
}

#[cfg(test)]
mod checked_sync_tests {
    use super::*;
    use metal::MTLCommandBufferStatus as Status;

    #[test]
    fn only_completed_command_buffers_are_successful() {
        assert!(validate_command_buffer_completion(Status::Completed, None, None).is_ok());
        for status in [
            Status::NotEnqueued,
            Status::Enqueued,
            Status::Committed,
            Status::Scheduled,
            Status::Error,
        ] {
            assert!(validate_command_buffer_completion(status, None, None).is_err());
        }
    }

    #[test]
    fn failed_command_buffer_preserves_driver_diagnostics() {
        let error = validate_command_buffer_completion(
            Status::Error,
            Some(8),
            Some("Insufficient memory to execute command buffer"),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("Error"));
        assert!(error.contains("driver error code 8"));
        assert!(error.contains("Insufficient memory"));
    }

    #[test]
    fn terminal_device_error_is_distinct_from_unknown_lifetime() {
        use crate::backend::BackendCompletion;
        assert!(matches!(
            classify_command_buffer_completion(Status::Error, Some(8), Some("device error")),
            BackendCompletion::Failed(_)
        ));
        for status in [
            Status::NotEnqueued,
            Status::Enqueued,
            Status::Committed,
            Status::Scheduled,
        ] {
            assert!(matches!(
                classify_command_buffer_completion(status, None, None),
                BackendCompletion::Unresolved(_)
            ));
        }
    }

    #[test]
    #[ignore = "requires an actual Metal device"]
    fn owned_command_buffer_survives_context_transfer_and_is_not_resubmitted() {
        let mut buffer = MetalBackend::from_slice(&[1.0, -2.0, 3.0, -4.0]);
        let mut ctx =
            MetalBackend::new_context(&BackendContextConfig::default()).expect("Metal context");
        MetalBackend::scale_inplace(&mut ctx, &mut buffer, 2.0, 4);
        let original = &**ctx.cmd.as_ref().expect("actual encoded command")
            as *const metal::CommandBufferRef as usize;
        // Move the real owner to the thread that observes completion. No
        // queue-lifetime or autorelease-pool assumption keeps the object alive.
        std::thread::spawn(move || {
            let command = ctx.submit_and_wait().expect("actual submitted command");
            assert_eq!(command as *const metal::CommandBufferRef as usize, original);
            assert_eq!(command.status(), Status::Completed);
            assert!(ctx.observe_completion().is_terminal());
            assert!(ctx.submitted.is_none());
            assert_eq!(MetalBackend::to_vec(&buffer, 4), vec![2.0, -4.0, 6.0, -8.0]);
            // An empty repeat does not manufacture another completion or
            // commit the already completed command buffer a second time.
            assert!(ctx.flush_checked().is_err());
            assert_eq!(MetalBackend::to_vec(&buffer, 4), vec![2.0, -4.0, 6.0, -8.0]);
        })
        .join()
        .expect("completion observer did not panic");
    }

    #[test]
    #[ignore = "requires the original Metal context and its native queue"]
    fn missing_submission_is_not_success() {
        let mut ctx = MetalBackend::new_context(&BackendContextConfig::default()).unwrap();
        let error = MetalBackend::sync_checked(&mut ctx).unwrap_err();
        assert!(error.to_string().contains("no pending command buffer"));
        // The legacy API and subsequent Drop still permit an empty no-op.
        MetalBackend::sync(&mut ctx);
        assert!(ctx.cmd.is_none());
        assert!(ctx.encoder.is_none());
    }
}

impl Drop for MetalContext {
    fn drop(&mut self) {
        // Destruction must not reuse the throwing host-access path, especially
        // during an encoding unwind. Model retirement owns the decision to
        // release a whole model; this only observes the original native work.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if self.cmd.is_some() || self.submitted.is_some() {
                let _ = self.observe_completion();
            }
        }));
    }
}

/// Element-count threshold above which a weight tensor goes to F16 storage
/// (when the context selects F16 weights). Tiny tensors (norm weights, biases) stay
/// F32 so the existing f32 shaders continue to see f32 inputs.
///
/// 1M elements = 4 MB as f32 / 2 MB as f16. Anything smaller saves little
/// memory and would force the f32-only shaders to sprout f16 variants.
const F16_MIN_ELEMS: usize = 1_048_576;

// ── Buffer allocation helpers (context-free) ─────────────────────────

fn alloc_f32_raw(n: usize) -> metal::Buffer {
    st().pipes.buffer_empty(n)
}

fn buffer_from_f32_slice(data: &[f32]) -> metal::Buffer {
    st().pipes.buffer_from_data(data)
}

fn buffer_from_f16_bytes(bytes: &[u8]) -> metal::Buffer {
    debug_assert_eq!(bytes.len() % 2, 0);
    st().pipes.device().new_buffer_with_data(
        bytes.as_ptr() as *const c_void,
        bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

/// Pack a slice of f32 into a fresh f16 metal::Buffer.
fn buffer_f16_from_f32(data: &[f32]) -> metal::Buffer {
    let n = data.len();
    let mut f16_bytes = vec![0u8; n * 2];
    for i in 0..n {
        let h = f16::from_f32(data[i]).to_le_bytes();
        f16_bytes[i * 2] = h[0];
        f16_bytes[i * 2 + 1] = h[1];
    }
    buffer_from_f16_bytes(&f16_bytes)
}

// ── Backend impl ──────────────────────────────────────────────────────

pub struct MetalBackend;

impl MetalBackend {
    /// Submit pending work through the ordinary production command path, wait,
    /// and require successful Metal completion before the caller reads results.
    /// Requires pending work: an empty context returns an error, so a missing
    /// submission cannot satisfy required operator validation. The legacy
    /// Backend::sync and Drop APIs still permit an empty no-op.
    pub fn sync_checked(ctx: &mut MetalContext) -> Result<()> {
        ctx.flush_checked()
    }
}

impl Backend for MetalBackend {
    type Buffer = MetalBuf;
    type Context = MetalContext;
    type Timer = crate::backend::timer::MetalTimer;
    fn make_timer() -> Self::Timer {
        crate::backend::timer::MetalTimer::new()
    }
    // type GptqStore: removed in Phase C step 4e. Metal has no Marlin
    // GPTQ path (GGUF is the quant story on Metal). Adding GPTQ later
    // means impl-ing MarlinExpertStack<MetalBackend> + load_gptq_stacked.

    fn new_context(config: &BackendContextConfig) -> Result<Self::Context> {
        config.validate()?;
        if config.device_ordinal != 0 {
            return Err(FerrumError::unsupported(
                "Metal's shared device boundary does not support device ordinal selection",
            ));
        }
        let pipes = Arc::clone(&st().pipes);
        let queue = pipes.device().new_command_queue();
        Ok(MetalContext {
            config: config.clone(),
            pipes,
            queue,
            #[cfg(test)]
            last_attention_kernel: None,
            cmd: None,
            encoder: None,
            submitted: None,
            last_completion: None,
            #[cfg(test)]
            completion_observation_override: None,
        })
    }

    fn context_config(ctx: &Self::Context) -> &BackendContextConfig {
        &ctx.config
    }

    fn supports_retained_execution() -> bool {
        true
    }

    fn sync(ctx: &mut Self::Context) {
        ctx.synchronize_for_host_access()
            .unwrap_or_else(|error| std::panic::panic_any(error));
    }

    fn begin_execution(ctx: &mut Self::Context) -> Result<()> {
        if ctx.cmd.is_some() || ctx.encoder.is_some() || ctx.submitted.is_some() {
            return Err(FerrumError::backend(
                "Metal cannot begin an invocation before the original command is terminal",
            ));
        }
        // This is an invocation boundary, not resource retirement. Only the
        // previous terminal status is reset; queue and allocations stay owned.
        ctx.last_completion = None;
        Ok(())
    }

    fn retire_context(ctx: &mut Self::Context) -> Result<()> {
        if Self::preparation_is_host_only(ctx) {
            return Ok(());
        }
        match ctx.observe_completion() {
            crate::backend::BackendCompletion::Succeeded
            | crate::backend::BackendCompletion::Failed(_) => Ok(()),
            crate::backend::BackendCompletion::Unresolved(error) => Err(error),
        }
    }

    fn try_sync(ctx: &mut Self::Context) -> Result<()> {
        ctx.flush_checked()
    }

    fn observe_completion(ctx: &mut Self::Context) -> crate::backend::BackendCompletion {
        ctx.observe_completion()
    }

    fn preparation_is_host_only(ctx: &Self::Context) -> bool {
        // Metal allocation and write_typed use shared host memory. Any real
        // encode, unresolved command or previous completion invalidates this
        // pristine-only proof; ordinary empty checked_sync remains an error.
        ctx.cmd.is_none()
            && ctx.encoder.is_none()
            && ctx.submitted.is_none()
            && ctx.last_completion.is_none()
    }
    fn sync_before_host_readback(ctx: &mut Self::Context) {
        Self::sync(ctx);
    }
    fn activation_elem_size_bytes() -> usize {
        std::mem::size_of::<f32>()
    }

    // ── Q4_K_M ────────────────────────────────────────────────────────

    /// Phase D step 2+3: unified typed uploader. Replaces from_slice_i32 +
    /// the legacy `from_slice` (which is kept as f32-default convenience).
    fn from_slice_typed<T: crate::backend::HostDtype>(data: &[T]) -> Self::Buffer {
        let bytes = data.len() * std::mem::size_of::<T>();
        let raw = st().pipes.device().new_buffer_with_data(
            data.as_ptr() as *const c_void,
            bytes as u64,
            MTLResourceOptions::StorageModeShared,
        );
        MetalBuf {
            raw,
            dtype: T::DTYPE,
            n: data.len(),
        }
    }

    /// Phase D step 2+3: unified typed in-place write. Replaces
    /// write_i32_into + write_f32_into. Unified memory on Apple
    /// Silicon means CPU writes the buffer's contents pointer
    /// directly — no blit encoder.
    fn write_typed<T: crate::backend::HostDtype>(
        ctx: &mut Self::Context,
        buf: &mut Self::Buffer,
        data: &[T],
    ) {
        ctx.require_reusable_state()
            .unwrap_or_else(|error| std::panic::panic_any(error));
        debug_assert_eq!(
            buf.dtype,
            T::DTYPE,
            "Metal write_typed: buf.dtype {:?} != T::DTYPE {:?}",
            buf.dtype,
            T::DTYPE
        );
        let dst = buf.raw.contents() as *mut T;
        let n = data.len().min(buf.n);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, n);
        }
    }

    // ── GEMM — dispatches on B-weight dtype ──────────────────────────

    fn gemm(
        ctx: &mut Self::Context,
        a: &Self::Buffer,
        b: &Self::Buffer,
        out: &mut Self::Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) {
        let a_buf = a.expect_f32("gemm a");
        let out_buf = out.expect_f32_mut("gemm out");
        let (pipes, enc, _) = ctx.encoder_parts();
        match b.dtype {
            Dtype::F16 => {
                // f16 weights — route through the f32-activation / f16-weight kernels.
                if m == 1 {
                    pipes.gemv_enc_f16w(enc, a_buf, &b.raw, out_buf, n, k);
                } else {
                    pipes.gemm_v2_f16w(enc, a_buf, &b.raw, out_buf, m, n, k);
                }
            }
            Dtype::F32 => {
                if m == 1 {
                    // GEMV with K-reduction via simd_sum — good for lm_head
                    // (N = vocab = 152k for Qwen3).
                    pipes.gemv_enc(enc, a_buf, &b.raw, out_buf, n, k);
                } else {
                    // GPU simdgroup-matrix tiled GEMM. Replaces an earlier
                    // cblas_sgemm fallback that serialised against GPU work.
                    pipes.gemm_v2(enc, a_buf, &b.raw, out_buf, m, n, k);
                }
            }
            // Phase A: integer dtypes are reserved for the upcoming typed-
            // buffer migration (alloc_u32 / write_u32 / from_slice_i32 →
            // typed-buffer alloc + write). No caller creates an integer
            // MetalBuf yet, so reaching this arm = caller bug.
            other => panic!(
                "MetalBackend::gemm: b.dtype = {} unsupported (only F16 / F32)",
                other.name()
            ),
        }
    }

    // ── Norm / attention / fused ops — all f32 ───────────────────────

    fn rms_norm(
        ctx: &mut Self::Context,
        x: &Self::Buffer,
        w: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    ) {
        let x = x.expect_f32("rms_norm x");
        let w = w.expect_f32("rms_norm w");
        let out = out.expect_f32_mut("rms_norm out");
        let (pipes, enc, _) = ctx.encoder_parts();
        pipes.rms_norm_enc(enc, x, w, out, tokens, dim, eps);
    }

    fn fused_add_rms_norm(
        ctx: &mut Self::Context,
        residual: &mut Self::Buffer,
        x: &Self::Buffer,
        w: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    ) {
        let residual = residual.expect_f32_mut("fused_add_rms_norm residual");
        let x = x.expect_f32("fused_add_rms_norm x");
        let w = w.expect_f32("fused_add_rms_norm w");
        let out = out.expect_f32_mut("fused_add_rms_norm out");
        let (pipes, enc, _) = ctx.encoder_parts();
        pipes.fused_residual_norm_enc(
            enc, residual, x, None, w, residual, out, tokens, dim, eps, 0,
        );
    }

    fn flash_attention(
        ctx: &mut Self::Context,
        q: &Self::Buffer,
        k: &Self::Buffer,
        v: &Self::Buffer,
        out: &mut Self::Buffer,
        batch: usize,
        q_len: usize,
        kv_len: usize,
        pos_offset: usize,
        cfg: &AttnConfig,
    ) {
        let p = AttentionParams {
            batch,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            q_len,
            kv_len,
            head_dim: cfg.head_dim,
            causal: cfg.causal,
            pos_offset,
            sliding_window: cfg.sliding_window,
        };
        let q = q.expect_f32("flash_attention q");
        let k = k.expect_f32("flash_attention k");
        let v = v.expect_f32("flash_attention v");
        let out = out.expect_f32_mut("flash_attention out");
        // flash_attn_v2 opens its own compute encoder internally; close
        // the sticky one first so we don't have two open at once on the
        // same cmd buffer.
        let (pipes, cmd, options) = ctx.command_parts();
        let kernel = pipes.flash_attn_v2(cmd, q, k, v, out, &p, cfg.kv_seq_stride, options);
        #[cfg(test)]
        {
            ctx.last_attention_kernel = Some(kernel);
        }
        #[cfg(not(test))]
        let _ = kernel;
    }

    fn copy_slice(
        ctx: &mut Self::Context,
        src: &Self::Buffer,
        src_offset: usize,
        dst: &mut Self::Buffer,
        dst_offset: usize,
        len: usize,
    ) {
        assert_eq!(src.dtype, dst.dtype, "Metal copy_slice dtype mismatch");
        let src_end = src_offset
            .checked_add(len)
            .expect("Metal copy_slice source extent overflow");
        let dst_end = dst_offset
            .checked_add(len)
            .expect("Metal copy_slice destination extent overflow");
        assert!(src_end <= src.n, "Metal copy_slice source out of bounds");
        assert!(
            dst_end <= dst.n,
            "Metal copy_slice destination out of bounds"
        );
        let element_bytes = src.dtype.bytes_per_elem();
        let bytes = |elements: usize| {
            u64::try_from(
                elements
                    .checked_mul(element_bytes)
                    .expect("Metal copy_slice byte extent overflow"),
            )
            .expect("Metal copy_slice byte extent exceeds Metal address range")
        };
        assert!(
            bytes(src_end) <= src.raw.length(),
            "Metal copy_slice source exceeds allocation"
        );
        assert!(
            bytes(dst_end) <= dst.raw.length(),
            "Metal copy_slice destination exceeds allocation"
        );
        let src_offset_bytes = bytes(src_offset);
        let dst_offset_bytes = bytes(dst_offset);
        let len_bytes = bytes(len);
        if len == 0 {
            return;
        }
        // Blit encoder stays in the same command buffer as neighbouring
        // compute encoders, keeping the single-command-buffer invariant.
        // Close the sticky compute encoder first — Metal forbids two
        // active encoders on one cmd buffer.
        let (_, cmd, _) = ctx.command_parts();
        let blit = cmd.new_blit_command_encoder();
        blit.copy_from_buffer(
            &src.raw,
            src_offset_bytes,
            &dst.raw,
            dst_offset_bytes,
            len_bytes,
        );
        blit.end_encoding();
    }

    fn embedding_lookup(
        ctx: &mut Self::Context,
        table: &Self::Buffer,
        ids: &[u32],
        out: &mut Self::Buffer,
        dim: usize,
    ) {
        // CPU-side gather into the f32 activation output. `ids.len()` is
        // small (one per batch item); flush any pending GPU work first.
        let out = out.expect_f32_mut("embedding_lookup out");
        ctx.synchronize_for_host_access()
            .unwrap_or_else(|error| std::panic::panic_any(error));
        unsafe {
            let o = std::slice::from_raw_parts_mut(out.contents() as *mut f32, ids.len() * dim);
            match table.dtype {
                Dtype::F32 => {
                    let t = std::slice::from_raw_parts(
                        table.raw.contents() as *const f32,
                        table.raw.length() as usize / 4,
                    );
                    for (i, &id) in ids.iter().enumerate() {
                        let s = id as usize * dim;
                        o[i * dim..(i + 1) * dim].copy_from_slice(&t[s..s + dim]);
                    }
                }
                Dtype::F16 => {
                    let t = std::slice::from_raw_parts(
                        table.raw.contents() as *const f16,
                        table.raw.length() as usize / 2,
                    );
                    for (i, &id) in ids.iter().enumerate() {
                        let s = id as usize * dim;
                        for j in 0..dim {
                            o[i * dim + j] = t[s + j].to_f32();
                        }
                    }
                }
                other => panic!(
                    "MetalBackend::embedding_lookup: table.dtype = {} unsupported",
                    other.name()
                ),
            }
        }
    }

    fn split_qkv(
        ctx: &mut Self::Context,
        qkv: &Self::Buffer,
        q: &mut Self::Buffer,
        k: &mut Self::Buffer,
        v: &mut Self::Buffer,
        tokens: usize,
        q_dim: usize,
        kv_dim: usize,
    ) {
        let qkv = qkv.expect_f32("split_qkv qkv");
        let q = q.expect_f32_mut("split_qkv q");
        let k = k.expect_f32_mut("split_qkv k");
        let v = v.expect_f32_mut("split_qkv v");
        let (pipes, enc, _) = ctx.encoder_parts();
        pipes.split_qkv_enc(enc, qkv, q, k, v, tokens, q_dim, kv_dim);
    }

    fn fused_silu_mul_split(
        ctx: &mut Self::Context,
        gu: &Self::Buffer,
        out: &mut Self::Buffer,
        tokens: usize,
        im: usize,
    ) {
        let gu = gu.expect_f32("fused_silu_mul_split gate_up");
        let out = out.expect_f32_mut("fused_silu_mul_split out");
        let (pipes, enc, _) = ctx.encoder_parts();
        pipes.silu_mul_split_enc(enc, gu, out, tokens, im);
    }

    fn fused_gelu_tanh_mul_split(
        ctx: &mut Self::Context,
        gu: &Self::Buffer,
        out: &mut Self::Buffer,
        tokens: usize,
        im: usize,
    ) {
        let gu = gu.expect_f32("fused_gelu_tanh_mul_split gate_up");
        let out = out.expect_f32_mut("fused_gelu_tanh_mul_split out");
        let (pipes, enc, _) = ctx.encoder_parts();
        pipes.gelu_tanh_mul_split_enc(enc, gu, out, tokens, im);
    }

    fn scale_inplace(ctx: &mut Self::Context, buf: &mut Self::Buffer, scale: f32, len: usize) {
        // mul_scale with scale_len=1 broadcasts a scalar; in/out aliasing is
        // safe for the 1:1 elementwise kernel.
        let scale_buf = Self::from_slice(&[scale]);
        let scale_mb = scale_buf.expect_f32("scale_inplace scale");
        let buf = buf.expect_f32_mut("scale_inplace buf");
        let (pipes, enc, _) = ctx.encoder_parts();
        pipes.mul_scale_enc(enc, buf, scale_mb, buf, len, 1);
    }

    fn qk_norm_rope(
        ctx: &mut Self::Context,
        input: &Self::Buffer,
        norm_w: &Self::Buffer,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        output: &mut Self::Buffer,
        tokens: usize,
        heads: usize,
        head_dim: usize,
        pos_offset: usize,
        eps: f32,
        mode: i32,
    ) {
        let input = input.expect_f32("qk_norm_rope input");
        let norm_w = norm_w.expect_f32("qk_norm_rope norm_w");
        let cos = cos.expect_f32("qk_norm_rope cos");
        let sin = sin.expect_f32("qk_norm_rope sin");
        let output = output.expect_f32_mut("qk_norm_rope output");
        let (pipes, enc, _) = ctx.encoder_parts();
        pipes.qk_norm_rope(
            enc, input, norm_w, cos, sin, output, tokens, heads, head_dim, pos_offset, eps, mode,
        );
    }

    fn split_qkv_norm_rope(
        ctx: &mut Self::Context,
        qkv: &Self::Buffer,
        q_norm_w: &Self::Buffer,
        k_norm_w: &Self::Buffer,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        q_out: &mut Self::Buffer,
        k_out: &mut Self::Buffer,
        v_out: &mut Self::Buffer,
        tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        pos_offset: usize,
        eps: f32,
        qk_mode: i32,
    ) -> Result<()> {
        let qkv = qkv.expect_f32("split_qkv_norm_rope qkv");
        let q_norm_w = q_norm_w.expect_f32("split_qkv_norm_rope q_norm_w");
        let k_norm_w = k_norm_w.expect_f32("split_qkv_norm_rope k_norm_w");
        let cos = cos.expect_f32("split_qkv_norm_rope cos");
        let sin = sin.expect_f32("split_qkv_norm_rope sin");
        let q_out = q_out.expect_f32_mut("split_qkv_norm_rope q_out");
        let k_out = k_out.expect_f32_mut("split_qkv_norm_rope k_out");
        let v_out = v_out.expect_f32_mut("split_qkv_norm_rope v_out");
        let (pipes, enc, _) = ctx.encoder_parts();
        pipes.split_qkv_norm_rope(
            enc, qkv, q_norm_w, k_norm_w, cos, sin, q_out, k_out, v_out, tokens, q_heads, kv_heads,
            head_dim, pos_offset, eps, qk_mode,
        );
        Ok(())
    }

    fn split_qkv_norm_rope_into_cache(
        ctx: &mut Self::Context,
        qkv: &Self::Buffer,
        q_norm_w: &Self::Buffer,
        k_norm_w: &Self::Buffer,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        q_out: &mut Self::Buffer,
        cache_k: &mut Self::Buffer,
        cache_v: &mut Self::Buffer,
        tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        pos_offset: usize,
        eps: f32,
        qk_mode: i32,
        cache_len: usize,
        cache_capacity: usize,
    ) -> Result<()> {
        let qkv = qkv.expect_f32("split_qkv_norm_rope_kvc qkv");
        let q_norm_w = q_norm_w.expect_f32("split_qkv_norm_rope_kvc q_norm_w");
        let k_norm_w = k_norm_w.expect_f32("split_qkv_norm_rope_kvc k_norm_w");
        let cos = cos.expect_f32("split_qkv_norm_rope_kvc cos");
        let sin = sin.expect_f32("split_qkv_norm_rope_kvc sin");
        let q_out = q_out.expect_f32_mut("split_qkv_norm_rope_kvc q_out");
        let cache_k = cache_k.expect_f32_mut("split_qkv_norm_rope_kvc cache_k");
        let cache_v = cache_v.expect_f32_mut("split_qkv_norm_rope_kvc cache_v");
        let (pipes, enc, _) = ctx.encoder_parts();
        pipes.split_qkv_norm_rope_into_cache(
            enc,
            qkv,
            q_norm_w,
            k_norm_w,
            cos,
            sin,
            q_out,
            cache_k,
            cache_v,
            tokens,
            q_heads,
            kv_heads,
            head_dim,
            pos_offset,
            eps,
            qk_mode,
            cache_len,
            cache_capacity,
        );
        Ok(())
    }

    /// Phase D step 2+3 unified typed allocator. Replaces alloc_u32.
    fn alloc_typed(dtype: Dtype, n: usize) -> Self::Buffer {
        let bytes = (n * dtype.bytes_per_elem()) as u64;
        let raw = st()
            .pipes
            .device()
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        MetalBuf { raw, dtype, n }
    }

    fn kv_cache_append_head_major(
        ctx: &mut Self::Context,
        cache_k: &mut Self::Buffer,
        cache_v: &mut Self::Buffer,
        cache_len: usize,
        cache_capacity: usize,
        new_k_head_major: &Self::Buffer,
        new_v_head_major: &Self::Buffer,
        new_tokens: usize,
        nkv: usize,
        hd: usize,
    ) {
        debug_assert!(cache_len + new_tokens <= cache_capacity);
        let cache_k = cache_k.expect_f32_mut("kv_cache_append cache_k");
        let cache_v = cache_v.expect_f32_mut("kv_cache_append cache_v");
        let new_k_head_major = new_k_head_major.expect_f32("kv_cache_append new_k");
        let new_v_head_major = new_v_head_major.expect_f32("kv_cache_append new_v");
        let (pipes, enc, _) = ctx.encoder_parts();
        pipes.kv_cache_append(
            enc,
            new_k_head_major,
            cache_k,
            nkv,
            hd,
            cache_len,
            new_tokens,
            cache_capacity,
        );
        pipes.kv_cache_append(
            enc,
            new_v_head_major,
            cache_v,
            nkv,
            hd,
            cache_len,
            new_tokens,
            cache_capacity,
        );
    }

    fn transpose_head_to_token(
        ctx: &mut Self::Context,
        src: &Self::Buffer,
        dst: &mut Self::Buffer,
        tokens: usize,
        heads: usize,
        dim: usize,
    ) {
        let src = src.expect_f32("transpose_head_to_token src");
        let dst = dst.expect_f32_mut("transpose_head_to_token dst");
        let (pipes, enc, _) = ctx.encoder_parts();
        pipes.transpose_out(enc, src, dst, tokens, heads, dim);
    }

    fn add_bias(
        ctx: &mut Self::Context,
        data: &mut Self::Buffer,
        bias: &Self::Buffer,
        rows: usize,
        cols: usize,
    ) {
        let data = data.expect_f32_mut("add_bias data");
        let bias = bias.expect_f32("add_bias bias");
        let (pipes, enc, _) = ctx.encoder_parts();
        pipes.add_bias_enc(enc, data, bias, rows, cols);
    }

    fn layer_norm(
        ctx: &mut Self::Context,
        x: &Self::Buffer,
        gamma: &Self::Buffer,
        beta: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    ) {
        let x = x.expect_f32("layer_norm x");
        let gamma = gamma.expect_f32("layer_norm gamma");
        let beta = beta.expect_f32("layer_norm beta");
        let out = out.expect_f32_mut("layer_norm out");
        let (pipes, enc, _) = ctx.encoder_parts();
        pipes.layer_norm_enc(enc, x, gamma, beta, out, tokens, dim, eps);
    }

    fn gelu(ctx: &mut Self::Context, x: &Self::Buffer, out: &mut Self::Buffer, len: usize) {
        let x = x.expect_f32("gelu x");
        let out = out.expect_f32_mut("gelu out");
        let (pipes, enc, _) = ctx.encoder_parts();
        pipes.gelu_enc(enc, x, out, len);
    }

    fn add_inplace(ctx: &mut Self::Context, r: &mut Self::Buffer, x: &Self::Buffer, len: usize) {
        let r = r.expect_f32_mut("add_inplace r");
        let x = x.expect_f32("add_inplace x");
        let (pipes, enc, _) = ctx.encoder_parts();
        pipes.add_enc(enc, r, x, r, len);
    }

    fn scaled_add_inplace(
        ctx: &mut Self::Context,
        dst: &mut Self::Buffer,
        src: &Self::Buffer,
        scale: f32,
        len: usize,
    ) {
        let dst_buf = dst.expect_f32_mut("scaled_add_inplace dst");
        let src_buf = src.expect_f32("scaled_add_inplace src");
        let (pipes, enc, _) = ctx.encoder_parts();
        pipes.scaled_add_inplace_enc(enc, dst_buf, src_buf, scale, len);
    }

    // ── Buffer management ────────────────────────────────────────────

    fn alloc(len: usize) -> Self::Buffer {
        // Scratch / output buffers are always f32 — the precision-sensitive
        // compute paths expect it.
        MetalBuf {
            raw: alloc_f32_raw(len),
            dtype: Dtype::F32,
            n: len,
        }
    }

    fn to_vec(buf: &Self::Buffer, len: usize) -> Vec<f32> {
        match buf.dtype {
            Dtype::F32 => {
                let ptr = buf.raw.contents() as *const f32;
                unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
            }
            Dtype::F16 => {
                let ptr = buf.raw.contents() as *const f16;
                let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                slice.iter().map(|h| h.to_f32()).collect()
            }
            other => panic!(
                "MetalBackend::to_vec: buf.dtype = {} unsupported (expected F32 or F16)",
                other.name()
            ),
        }
    }

    fn to_vec_in_context(ctx: &mut Self::Context, buf: &Self::Buffer, len: usize) -> Vec<f32> {
        ctx.require_reusable_state()
            .unwrap_or_else(|error| std::panic::panic_any(error));
        Self::to_vec(buf, len)
    }

    fn argmax_rows_f16(
        ctx: &mut Self::Context,
        logits: &Self::Buffer,
        m: usize,
        n: usize,
    ) -> Result<Vec<u32>> {
        ctx.require_reusable_state()?;
        if !matches!(logits.dtype, Dtype::F32) {
            ctx.synchronize_for_host_access()?;
            let host = Self::to_vec(logits, m * n);
            let mut out = Vec::with_capacity(m);
            for row in 0..m {
                let slice = &host[row * n..(row + 1) * n];
                let mut max_idx = 0usize;
                let mut max_val = f32::NEG_INFINITY;
                for (i, &v) in slice.iter().enumerate() {
                    if v > max_val {
                        max_val = v;
                        max_idx = i;
                    }
                }
                out.push(max_idx as u32);
            }
            return Ok(out);
        }

        #[repr(C)]
        struct ArgmaxParams {
            n: i32,
        }

        let out = ctx.pipes.device().new_buffer(
            (m * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let params = ArgmaxParams { n: n as i32 };
        let (pipes, enc, _) = ctx.encoder_parts();
        enc.set_compute_pipeline_state(pipes.pipeline("argmax_rows_f32"));
        enc.set_buffer(0, Some(&logits.raw), 0);
        enc.set_buffer(1, Some(&out), 0);
        enc.set_bytes(
            2,
            std::mem::size_of::<ArgmaxParams>() as u64,
            &params as *const _ as *const c_void,
        );
        enc.dispatch_thread_groups(MTLSize::new(1, m as u64, 1), MTLSize::new(256, 1, 1));
        ctx.synchronize_for_host_access()?;
        let ptr = out.contents() as *const u32;
        let tokens = unsafe { std::slice::from_raw_parts(ptr, m).to_vec() };
        Ok(tokens)
    }

    fn from_slice(data: &[f32]) -> Self::Buffer {
        // Activations, cos/sin tables, temporary scratch — all f32.
        MetalBuf {
            raw: buffer_from_f32_slice(data),
            dtype: Dtype::F32,
            n: data.len(),
        }
    }

    /// Context-free compatibility construction uses the fixed F32 default.
    /// Model weight loading uses `from_weight_bytes_in_context` below.
    fn from_weight_bytes(raw: &[u8], src_dtype: SrcDtype) -> Self::Buffer {
        weight_bytes_with_options(raw, src_dtype, &MetalBackendConfig::default())
    }

    fn from_weight_bytes_in_context(
        ctx: &mut Self::Context,
        raw: &[u8],
        src_dtype: SrcDtype,
    ) -> Self::Buffer {
        weight_bytes_with_options(raw, src_dtype, &ctx.config.metal)
    }
}

fn weight_bytes_with_options(
    raw: &[u8],
    src_dtype: SrcDtype,
    options: &MetalBackendConfig,
) -> MetalBuf {
    let n = raw.len() / src_dtype.bytes_per_elem();
    let want_f16 = options.prefer_f16_weights && n >= F16_MIN_ELEMS;

    if !want_f16 {
        // Default behaviour: materialise as f32, matches pre-refactor
        // MetalBackend byte-for-byte.
        let data = src_dtype.to_f32_vec(raw);
        return MetalBuf {
            raw: buffer_from_f32_slice(&data),
            dtype: Dtype::F32,
            n,
        };
    }

    // f16 storage path — go directly from raw bytes where possible to
    // avoid the transient 2× RAM spike.
    match src_dtype {
        SrcDtype::F16 => MetalBuf {
            raw: buffer_from_f16_bytes(raw),
            dtype: Dtype::F16,
            n,
        },
        SrcDtype::BF16 => {
            // bf16 → f16 via f32. Loses magnitude range (bf16 has a
            // broader exponent) but gains mantissa precision, which for
            // typical weight magnitudes (|w| < 32) is a net upgrade.
            let mut f16_bytes = vec![0u8; n * 2];
            for i in 0..n {
                let v = bf16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]).to_f32();
                let h = f16::from_f32(v).to_le_bytes();
                f16_bytes[i * 2] = h[0];
                f16_bytes[i * 2 + 1] = h[1];
            }
            MetalBuf {
                raw: buffer_from_f16_bytes(&f16_bytes),
                dtype: Dtype::F16,
                n,
            }
        }
        SrcDtype::F32 => {
            // f32 → f16 downcast. Halves storage.
            let data = src_dtype.to_f32_vec(raw);
            MetalBuf {
                raw: buffer_f16_from_f32(&data),
                dtype: Dtype::F16,
                n,
            }
        }
    }
}

// Metal has no graph-capture analogue; inherit BackendGraph defaults.
impl crate::backend::BackendGraph for MetalBackend {}

// Metal has no multi-GPU collectives; inherit BackendCollective single-rank defaults.
impl crate::backend::BackendCollective for MetalBackend {}

// Metal does not ship Marlin INT4 kernels; inherit unsupported defaults.
impl crate::backend::BackendQuantMarlin for MetalBackend {}

// Metal: existing KV cache path is FP16.
impl crate::backend::BackendKvDtype<crate::backend::KvFp16> for MetalBackend {
    type KvBuffer = <Self as crate::backend::Backend>::Buffer;
    type KvScales = ();
}
