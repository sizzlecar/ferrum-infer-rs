//! Explicitly budgeted, eager-only cuDNN 9 convolution owner.
//!
//! A fixed handle and four descriptors are shared under one mutex. No find/
//! autotune API is called, and native workspace always comes from an admitted
//! operation scratch view. Opaque allowances are declarations, not measurements.
//! Optional SDPA graphs use a separately reserved, bounded plan/binding cache;
//! their commands retain this same library/context owner through completion.

use std::ffi::c_void;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaStream};
use ferrum_runtime::DeviceCapacityRetention;
use sha2::{Digest, Sha256};

use super::{CudaDeviceRuntime, CudaDeviceRuntimeError};

mod dependencies;
mod ffi;
mod sdpa;
use ffi::{check, Api, Descriptor, ForwardPerformance, Handle};
pub use sdpa::CudnnSdpaPlanPolicy;
pub(crate) use sdpa::{
    CudaCudnnSdpaCache, CudnnSdpaBindings, CudnnSdpaExecution, CudnnSdpaPlan, CudnnSdpaSignature,
};

/// Deployment configuration for a separately charged cuDNN provider owner.
/// The library and its sibling cuDNN 9 libraries must be trusted, immutable
/// deployment files. This profile does not certify their opaque memory peak.
#[derive(Clone, Debug)]
pub struct CudaCudnnConfig {
    pub library_path: PathBuf,
    /// Trusted, immutable native dependency files, loaded in this order before
    /// cuDNN. Empty preserves the existing deployment-loader behavior. Explicit
    /// dependencies currently require the Linux GNU loader; no environment or
    /// global symbol visibility is changed. Their memory belongs to this same
    /// declared opaque allowance, which is not a measured upper bound.
    pub dependency_libraries: Vec<PathBuf>,
    pub opaque_memory_bytes: u64,
    pub workspace_limit_bytes: u64,
    pub opaque_memory_profile: String,
}

impl CudaCudnnConfig {
    pub(crate) fn validate(&self) -> Result<(), CudaDeviceRuntimeError> {
        if !self.library_path.is_absolute()
            || self.opaque_memory_bytes == 0
            || self.workspace_limit_bytes > isize::MAX as u64
            || self.opaque_memory_profile.trim().is_empty()
            || self.opaque_memory_profile.len() > 256
            || self
                .opaque_memory_profile
                .bytes()
                .any(|b| b.is_ascii_control())
        {
            return Err(CudaDeviceRuntimeError::contract(
                "cuDNN requires an absolute library path, positive opaque allowance, bounded workspace and versioned profile",
            ));
        }
        dependencies::validate_paths(&self.dependency_libraries)
            .map_err(CudaDeviceRuntimeError::contract)?;
        Ok(())
    }
}

/// All dimensions and byte ranges are checked by the provider before this
/// descriptor reaches a submitted command. Input may be explicit-pad scratch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CudnnConv2dDescriptor {
    pub input: [i32; 4],
    pub filter: [i32; 4],
    pub output: [i32; 4],
    pub padding: [i32; 2],
    pub stride: [i32; 2],
    pub dilation: [i32; 2],
    pub groups: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ConvolutionAlgorithm {
    algo: i32,
    math_type: i32,
    required_bytes: usize,
}
impl ConvolutionAlgorithm {
    fn validate(self, cap: usize) -> Result<(), String> {
        if !(0..ffi::ALGORITHMS as i32).contains(&self.algo)
            || !matches!(self.math_type, 0 | ffi::TENSOR_OP_MATH)
            || self.required_bytes > cap
        {
            return Err(
                "cuDNN convolution selection exceeds admitted algorithm/math/workspace policy"
                    .into(),
            );
        }
        Ok(())
    }

    fn validate_workspace(self, workspace: u64, bytes: u64) -> Result<(), String> {
        if bytes != self.required_bytes as u64 || (bytes == 0) != (workspace == 0) {
            return Err("cuDNN workspace differs from prepared actual requirement".into());
        }
        Ok(())
    }
}

/// Immutable metadata prepared before resource admission. The compiled plan and
/// each encoded command retain this exact selection and its native owner.
/// There is no separate convolution cache or limit on retained shape records.
pub(crate) struct PreparedConv {
    owner: Arc<CudaCudnnOwner>,
    shape: CudnnConv2dDescriptor,
    selected: ConvolutionAlgorithm,
    fingerprint: String,
}

impl PreparedConv {
    pub(crate) fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub(crate) fn workspace_bytes(&self) -> u64 {
        self.selected.required_bytes as u64
    }

    pub(crate) fn validate_invocation(
        &self,
        owner: &Arc<CudaCudnnOwner>,
        shape: CudnnConv2dDescriptor,
        fingerprint: Option<&str>,
    ) -> Result<(), String> {
        if !Arc::ptr_eq(&self.owner, owner)
            || self.shape != shape
            || fingerprint != Some(self.fingerprint())
        {
            return Err("cuDNN convolution owner/shape/prepared identity mismatch".into());
        }
        self.selected
            .validate(self.owner.workspace_limit_bytes as usize)
    }

    /// Submission only consumes the admitted selection; it never queries an
    /// algorithm or workspace size. All bindings remain command-owned.
    pub(crate) fn convolution(
        &self,
        stream: &CudaStream,
        input: u64,
        filter: u64,
        output: u64,
        workspace: u64,
        workspace_bytes: u64,
    ) -> Result<(), CudaDeviceRuntimeError> {
        self.selected
            .validate_workspace(workspace, workspace_bytes)
            .map_err(CudaDeviceRuntimeError::contract)?;
        let mut guard = self
            .owner
            .inner
            .lock()
            .map_err(|_| CudaDeviceRuntimeError::contract("cuDNN owner is poisoned"))?;
        let native = guard
            .as_mut()
            .ok_or_else(|| CudaDeviceRuntimeError::contract("cuDNN owner is closed"))?;
        if native.failed {
            return Err(CudaDeviceRuntimeError::contract(
                "cuDNN owner requires quarantine",
            ));
        }
        native
            .context
            .bind_to_thread()
            .map_err(|e| CudaDeviceRuntimeError::driver("cuDNN execution context bind", e))?;
        unsafe {
            native.convolution(
                stream,
                self.shape,
                self.selected,
                input,
                filter,
                output,
                workspace,
            )
        }
        .map_err(CudaDeviceRuntimeError::contract)
    }
}

/// Native errors and unwinds retain quarantine. A fully successful preparation
/// or forward call is the only path that restores the reusable owner state.
fn guarded_convolution<T>(
    failed: &mut bool,
    operation: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    if *failed {
        return Err("cuDNN owner requires quarantine".into());
    }
    *failed = true;
    let result = operation()?;
    *failed = false;
    Ok(result)
}

/// A successful native query can still have no selection inside the policy
/// cap. Reject that prepared plan without quarantining the reusable owner.
/// Native query failures and unwinds remain inside the guarded boundary.
fn prepare_convolution_selection(
    failed: &mut bool,
    workspace_cap: usize,
    query: impl FnOnce() -> Result<Option<ConvolutionAlgorithm>, String>,
) -> Result<ConvolutionAlgorithm, String> {
    let selected = guarded_convolution(failed, query)?
        .ok_or("no deterministic cuDNN convolution fits configured workspace policy")?;
    selected.validate(workspace_cap)?;
    Ok(selected)
}

pub(crate) struct CudaCudnnOwner {
    inner: Mutex<Option<NativeOwner>>,
    fingerprint: String,
    workspace_limit_bytes: u64,
}

struct NativeOwner {
    handle: Handle,
    input: Descriptor,
    output: Descriptor,
    filter: Descriptor,
    convolution: Descriptor,
    failed: bool,
    // Native objects are explicitly destroyed before any of these fields.
    api: Api,
    context: Arc<CudaContext>,
    _opaque: DeviceCapacityRetention,
    _bootstrap: DeviceCapacityRetention,
}

// Access is serialized by CudaCudnnOwner.inner; every access binds the original
// CUDA context to the calling thread. No raw handle escapes this module.
unsafe impl Send for NativeOwner {}

impl CudaCudnnOwner {
    pub(crate) fn new(
        runtime: &CudaDeviceRuntime,
        config: &CudaCudnnConfig,
    ) -> Result<Arc<Self>, CudaDeviceRuntimeError> {
        config.validate()?;
        // Hash streams bounded stack chunks, never a library-sized payload Vec.
        // The complete sibling bundle is pinned as deployment provenance;
        // callers must not replace it or preload a conflicting cuDNN instance.
        let library_identity = library_fingerprint(config)?;
        let dependencies = dependencies::Selection::prepare(&config.dependency_libraries)
            .map_err(CudaDeviceRuntimeError::contract)?;
        dependencies
            .validate_loader_support()
            .map_err(CudaDeviceRuntimeError::contract)?;
        let bootstrap = runtime.tensor_bootstrap_retention()?;
        let opaque = runtime.reserve_tensor_opaque(config.opaque_memory_bytes)?;
        let context = Arc::clone(runtime.context());
        context
            .bind_to_thread()
            .map_err(|e| CudaDeviceRuntimeError::driver("cuDNN context bind", e))?;
        let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            Api::load(&config.library_path, &dependencies)
        }));
        let api = match loaded {
            Ok(Ok(api)) => api,
            Ok(Err(error)) => {
                // A native library initializer may have partially run.
                std::mem::forget((opaque, bootstrap, context));
                return Err(CudaDeviceRuntimeError::contract(error));
            }
            Err(payload) => {
                std::mem::forget((opaque, bootstrap, context));
                std::panic::resume_unwind(payload);
            }
        };
        let version = unsafe { (api.version)() };
        if !(90_000..100_000).contains(&version) {
            std::mem::forget((api, opaque, bootstrap, context));
            return Err(CudaDeviceRuntimeError::contract(format!(
                "cuDNN 9 is required, got {version}"
            )));
        }
        let mut native = NativeOwner {
            handle: std::ptr::null_mut(),
            input: std::ptr::null_mut(),
            output: std::ptr::null_mut(),
            filter: std::ptr::null_mut(),
            convolution: std::ptr::null_mut(),
            failed: false,
            api,
            context,
            _opaque: opaque,
            _bootstrap: bootstrap,
        };
        let created = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            check((native.api.create)(&mut native.handle), "create handle")?;
            check(
                (native.api.create_tensor)(&mut native.input),
                "create input descriptor",
            )?;
            check(
                (native.api.create_tensor)(&mut native.output),
                "create output descriptor",
            )?;
            check(
                (native.api.create_filter)(&mut native.filter),
                "create filter descriptor",
            )?;
            check(
                (native.api.create_convolution)(&mut native.convolution),
                "create convolution descriptor",
            )
        }));
        match created {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                std::mem::forget(native);
                return Err(CudaDeviceRuntimeError::contract(error));
            }
            Err(payload) => {
                std::mem::forget(native);
                std::panic::resume_unwind(payload);
            }
        }
        let mut digest = Sha256::new();
        digest.update(library_identity);
        digest.update(dependencies.fingerprint());
        digest.update(version.to_le_bytes());
        digest.update(config.opaque_memory_bytes.to_le_bytes());
        digest.update(config.workspace_limit_bytes.to_le_bytes());
        digest.update(config.opaque_memory_profile.as_bytes());
        digest.update(include_bytes!("cudnn.rs"));
        digest.update(include_bytes!("cudnn/ffi.rs"));
        digest.update(include_bytes!("cudnn/dependencies.rs"));
        digest.update(include_bytes!("cudnn/dependencies/linux.rs"));
        digest.update(include_bytes!("cudnn/sdpa.rs"));
        digest.update(include_bytes!("cudnn/sdpa/native.rs"));
        Ok(Arc::new(Self {
            inner: Mutex::new(Some(native)),
            fingerprint: format!("{:x}", digest.finalize()),
            workspace_limit_bytes: config.workspace_limit_bytes,
        }))
    }

    pub(crate) fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
    pub(crate) fn workspace_limit_bytes(&self) -> u64 {
        self.workspace_limit_bytes
    }

    /// Query a deterministic algorithm and its actual workspace before the
    /// provider publishes its resource estimate. No payload or device workspace
    /// is allocated here, and the selection is not reconstructed during encode.
    pub(crate) fn prepare_convolution(
        self: &Arc<Self>,
        shape: CudnnConv2dDescriptor,
    ) -> Result<Arc<PreparedConv>, CudaDeviceRuntimeError> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| CudaDeviceRuntimeError::contract("cuDNN owner is poisoned"))?;
        let native = guard
            .as_mut()
            .ok_or_else(|| CudaDeviceRuntimeError::contract("cuDNN owner is closed"))?;
        if native.failed {
            return Err(CudaDeviceRuntimeError::contract(
                "cuDNN owner requires quarantine",
            ));
        }
        native
            .context
            .bind_to_thread()
            .map_err(|e| CudaDeviceRuntimeError::driver("cuDNN preparation context bind", e))?;
        let selected =
            unsafe { native.prepare_convolution(shape, self.workspace_limit_bytes() as usize) }
                .map_err(CudaDeviceRuntimeError::contract)?;
        Ok(Arc::new(PreparedConv {
            owner: Arc::clone(self),
            shape,
            selected,
            fingerprint: convolution_fingerprint(
                self.fingerprint(),
                shape,
                self.workspace_limit_bytes(),
                selected,
            ),
        }))
    }
}

#[derive(Clone, Copy)]
struct ConvolutionDescriptors {
    input: Descriptor,
    output: Descriptor,
    filter: Descriptor,
    convolution: Descriptor,
}

impl ConvolutionDescriptors {
    unsafe fn configure(self, a: &Api, s: CudnnConv2dDescriptor) -> Result<(), String> {
        check(
            (a.set_tensor)(
                self.input,
                ffi::NCHW,
                ffi::BF16,
                s.input[0],
                s.input[1],
                s.input[2],
                s.input[3],
            ),
            "set input",
        )?;
        check(
            (a.set_tensor)(
                self.output,
                ffi::NCHW,
                ffi::BF16,
                s.output[0],
                s.output[1],
                s.output[2],
                s.output[3],
            ),
            "set output",
        )?;
        check(
            (a.set_filter)(
                self.filter,
                ffi::BF16,
                ffi::NCHW,
                s.filter[0],
                s.filter[1],
                s.filter[2],
                s.filter[3],
            ),
            "set filter",
        )?;
        check(
            (a.set_convolution)(
                self.convolution,
                s.padding[0],
                s.padding[1],
                s.stride[0],
                s.stride[1],
                s.dilation[0],
                s.dilation[1],
                ffi::CROSS_CORRELATION,
                ffi::FLOAT,
            ),
            "set convolution",
        )?;
        check((a.set_groups)(self.convolution, s.groups), "set groups")
    }
}

impl NativeOwner {
    fn convolution_descriptors(&self) -> ConvolutionDescriptors {
        ConvolutionDescriptors {
            input: self.input,
            output: self.output,
            filter: self.filter,
            convolution: self.convolution,
        }
    }

    unsafe fn prepare_convolution(
        &mut self,
        shape: CudnnConv2dDescriptor,
        workspace_cap: usize,
    ) -> Result<ConvolutionAlgorithm, String> {
        let d = self.convolution_descriptors();
        let a = &self.api;
        let handle = self.handle;
        let started = Instant::now();
        let selected = prepare_convolution_selection(&mut self.failed, workspace_cap, || unsafe {
            check(
                (a.set_stream)(handle, std::ptr::null_mut()),
                "clear stream for convolution preparation",
            )?;
            d.configure(a, shape)?;
            check(
                (a.set_math)(d.convolution, ffi::TENSOR_OP_MATH),
                "set math policy",
            )?;
            let mut candidates = [ForwardPerformance::default(); ffi::ALGORITHMS];
            let mut count = 0;
            check(
                (a.algorithms)(
                    handle,
                    d.input,
                    d.filter,
                    d.convolution,
                    d.output,
                    ffi::ALGORITHMS as i32,
                    &mut count,
                    candidates.as_mut_ptr(),
                ),
                "query convolution algorithms",
            )?;
            let count = usize::try_from(count)
                .ok()
                .filter(|&n| n > 0 && n <= candidates.len())
                .ok_or("cuDNN returned an invalid algorithm count")?;
            let Some(candidate) = choose_algorithm(&candidates[..count], workspace_cap) else {
                return Ok(None);
            };
            check(
                (a.set_math)(d.convolution, candidate.math_type),
                "set selected math type",
            )?;
            let mut required = 0;
            check(
                (a.workspace)(
                    handle,
                    d.input,
                    d.filter,
                    d.convolution,
                    d.output,
                    candidate.algo,
                    &mut required,
                ),
                "query workspace",
            )?;
            Ok(Some(ConvolutionAlgorithm {
                algo: candidate.algo,
                math_type: candidate.math_type,
                required_bytes: required,
            }))
        })?;
        tracing::debug!(
            shape = ?shape,
            workspace_cap_bytes = workspace_cap,
            algorithm = selected.algo,
            math_type = selected.math_type,
            required_workspace_bytes = selected.required_bytes,
            selection_query_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
            "cuDNN convolution prepared selection"
        );
        Ok(selected)
    }

    unsafe fn convolution(
        &mut self,
        stream: &CudaStream,
        shape: CudnnConv2dDescriptor,
        selected: ConvolutionAlgorithm,
        input: u64,
        filter: u64,
        output: u64,
        workspace: u64,
    ) -> Result<(), String> {
        let d = self.convolution_descriptors();
        let a = &self.api;
        let handle = self.handle;
        guarded_convolution(&mut self.failed, || unsafe {
            check(
                (a.set_stream)(handle, stream.cu_stream() as *mut c_void),
                "set stream",
            )?;
            d.configure(a, shape)?;
            check(
                (a.set_math)(d.convolution, selected.math_type),
                "set selected math type",
            )?;
            let alpha = 1.0_f32;
            let beta = 0.0_f32;
            check(
                (a.forward)(
                    handle,
                    (&alpha as *const f32).cast(),
                    d.input,
                    input as *const c_void,
                    d.filter,
                    filter as *const c_void,
                    d.convolution,
                    selected.algo,
                    workspace as *mut c_void,
                    selected.required_bytes,
                    (&beta as *const f32).cast(),
                    d.output,
                    output as *mut c_void,
                ),
                "convolution forward",
            )?;
            check((a.set_stream)(handle, std::ptr::null_mut()), "clear stream")
        })
    }

    fn destroy(&mut self) -> bool {
        if self.failed || self.context.bind_to_thread().is_err() {
            return false;
        }
        unsafe {
            for (pointer, destroy) in [
                (&mut self.convolution, self.api.destroy_convolution),
                (&mut self.filter, self.api.destroy_filter),
                (&mut self.output, self.api.destroy_tensor),
                (&mut self.input, self.api.destroy_tensor),
            ] {
                if !pointer.is_null() && destroy(*pointer) != ffi::SUCCESS {
                    return false;
                }
                *pointer = std::ptr::null_mut();
            }
            if !self.handle.is_null() && (self.api.destroy)(self.handle) != ffi::SUCCESS {
                return false;
            }
            self.handle = std::ptr::null_mut();
        }
        true
    }
}

impl Drop for CudaCudnnOwner {
    fn drop(&mut self) {
        let owner = self
            .inner
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(mut native) = owner {
            if !native.destroy() {
                std::mem::forget(native);
            }
        }
    }
}

fn choose_algorithm(candidates: &[ForwardPerformance], cap: usize) -> Option<ForwardPerformance> {
    candidates.iter().copied().find(|c| {
        c.status == ffi::SUCCESS
            && c.determinism == ffi::DETERMINISTIC
            && c.memory <= cap
            && (0..ffi::ALGORITHMS as i32).contains(&c.algo)
            && matches!(c.math_type, 0 | ffi::TENSOR_OP_MATH)
    })
}

fn convolution_fingerprint(
    owner: &str,
    shape: CudnnConv2dDescriptor,
    workspace_cap: u64,
    selected: ConvolutionAlgorithm,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"cuda.cudnn.conv2d.prepared-actual-workspace.v1");
    digest.update(owner.as_bytes());
    for dimensions in [shape.input, shape.filter, shape.output] {
        for value in dimensions {
            digest.update(value.to_le_bytes());
        }
    }
    for dimensions in [shape.padding, shape.stride, shape.dilation] {
        for value in dimensions {
            digest.update(value.to_le_bytes());
        }
    }
    digest.update(shape.groups.to_le_bytes());
    digest.update(workspace_cap.to_le_bytes());
    digest.update(selected.algo.to_le_bytes());
    digest.update(selected.math_type.to_le_bytes());
    digest.update((selected.required_bytes as u64).to_le_bytes());
    format!("{:x}", digest.finalize())
}

fn library_fingerprint(config: &CudaCudnnConfig) -> Result<[u8; 32], CudaDeviceRuntimeError> {
    let directory = config
        .library_path
        .parent()
        .ok_or_else(|| CudaDeviceRuntimeError::contract("cuDNN library has no parent"))?;
    let mut digest = Sha256::new();
    let paths = [
        config.library_path.clone(),
        directory.join("libcudnn_cnn.so.9"),
        directory.join("libcudnn_ops.so.9"),
        directory.join("libcudnn_graph.so.9"),
        directory.join("libcudnn_heuristic.so.9"),
        directory.join("libcudnn_engines_precompiled.so.9"),
        directory.join("libcudnn_engines_runtime_compiled.so.9"),
    ];
    let mut chunk = [0_u8; 64 * 1024];
    for path in paths {
        let mut file = File::open(&path).map_err(|e| {
            CudaDeviceRuntimeError::contract(format!("cuDNN identity {}: {e}", path.display()))
        })?;
        let length = file
            .metadata()
            .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))?
            .len();
        digest.update(length.to_le_bytes());
        loop {
            let count = file
                .read(&mut chunk)
                .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))?;
            if count == 0 {
                break;
            }
            digest.update(&chunk[..count]);
        }
    }
    Ok(digest.finalize().into())
}

#[cfg(test)]
mod tests;
