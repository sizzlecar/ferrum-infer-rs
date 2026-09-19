//! Explicit, test-only access to a caller-supplied Prism release's Metal backend.
//!
//! C signatures were checked against Prism commit
//! 9a9394a895b96003ca842a6041cb28ac49a108f7. Tensor/graph internals stay opaque;
//! no reference kernel source or library is bundled into Ferrum. The caller is
//! responsible for supplying trusted libraries and recording their provenance.

use std::cell::Cell;
use std::ffi::{c_char, c_int, c_void, CStr};
use std::marker::PhantomData;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::rc::Rc;
use std::time::{Duration, Instant};

use libloading::Library;

macro_rules! opaque {
    ($($name:ident),+ $(,)?) => {$(
        #[repr(C)]
        struct $name { _private: [u8; 0] }
    )+};
}
opaque!(
    Context,
    Tensor,
    Graph,
    Backend,
    BackendBuffer,
    BackendDevice
);

#[repr(C)]
struct InitParams {
    mem_size: usize,
    mem_buffer: *mut c_void,
    no_alloc: bool,
    // Initialize the C struct's trailing padding as well as its actual fields.
    padding: [u8; std::mem::size_of::<usize>() - 1],
}

struct Api {
    init: unsafe extern "C" fn(InitParams) -> *mut Context,
    free: unsafe extern "C" fn(*mut Context),
    tensor_overhead: unsafe extern "C" fn() -> usize,
    graph_overhead: unsafe extern "C" fn(usize, bool) -> usize,
    new_tensor_2d: unsafe extern "C" fn(*mut Context, c_int, i64, i64) -> *mut Tensor,
    mul_mat: unsafe extern "C" fn(*mut Context, *mut Tensor, *mut Tensor) -> *mut Tensor,
    new_graph: unsafe extern "C" fn(*mut Context, usize, bool) -> *mut Graph,
    build_forward: unsafe extern "C" fn(*mut Graph, *mut Tensor),
    graph_nodes: unsafe extern "C" fn(*mut Graph) -> c_int,
    nbytes: unsafe extern "C" fn(*const Tensor) -> usize,
    type_name: unsafe extern "C" fn(c_int) -> *const c_char,
    block_size: unsafe extern "C" fn(c_int) -> i64,
    type_size: unsafe extern "C" fn(c_int) -> usize,
    row_size: unsafe extern "C" fn(c_int, i64) -> usize,
    version: unsafe extern "C" fn() -> *const c_char,
    commit: unsafe extern "C" fn() -> *const c_char,
    metal_init: unsafe extern "C" fn() -> *mut Backend,
    is_metal: unsafe extern "C" fn(*mut Backend) -> bool,
    backend_name: unsafe extern "C" fn(*mut Backend) -> *const c_char,
    backend_device: unsafe extern "C" fn(*mut Backend) -> *mut BackendDevice,
    device_description: unsafe extern "C" fn(*mut BackendDevice) -> *const c_char,
    backend_free: unsafe extern "C" fn(*mut Backend),
    synchronize: unsafe extern "C" fn(*mut Backend),
    supports_op: unsafe extern "C" fn(*mut Backend, *const Tensor) -> bool,
    alloc_tensors: unsafe extern "C" fn(*mut Context, *mut Backend) -> *mut BackendBuffer,
    buffer_free: unsafe extern "C" fn(*mut BackendBuffer),
    tensor_set: unsafe extern "C" fn(*mut Tensor, *const c_void, usize, usize),
    tensor_get: unsafe extern "C" fn(*const Tensor, *mut c_void, usize, usize),
    tensor_memset: unsafe extern "C" fn(*mut Tensor, u8, usize, usize),
    graph_compute: unsafe extern "C" fn(*mut Backend, *mut Graph) -> c_int,
}

impl Api {
    fn load(base: &Library, metal: &Library) -> Result<Self, String> {
        macro_rules! symbol {
            ($library:ident, $name:literal) => {{
                // SAFETY: The explicit trusted release must implement these C
                // signatures. Function pointers remain valid while Reference
                // owns both libraries; all graph owners borrow Reference.
                unsafe {
                    *$library
                        .get(concat!($name, "\0").as_bytes())
                        .map_err(|error| {
                            format!("Prism {} missing {}: {error}", stringify!($library), $name)
                        })?
                }
            }};
        }
        // Reject a vanilla GGML library before querying Prism's extended enum.
        // This extension symbol is checked, not invoked or retained.
        let _: unsafe extern "C" fn(*const c_void, *mut f32, i64) =
            symbol!(base, "dequantize_row_pq2_0");
        Ok(Self {
            init: symbol!(base, "ggml_init"),
            free: symbol!(base, "ggml_free"),
            tensor_overhead: symbol!(base, "ggml_tensor_overhead"),
            graph_overhead: symbol!(base, "ggml_graph_overhead_custom"),
            new_tensor_2d: symbol!(base, "ggml_new_tensor_2d"),
            mul_mat: symbol!(base, "ggml_mul_mat"),
            new_graph: symbol!(base, "ggml_new_graph_custom"),
            build_forward: symbol!(base, "ggml_build_forward_expand"),
            graph_nodes: symbol!(base, "ggml_graph_n_nodes"),
            nbytes: symbol!(base, "ggml_nbytes"),
            type_name: symbol!(base, "ggml_type_name"),
            block_size: symbol!(base, "ggml_blck_size"),
            type_size: symbol!(base, "ggml_type_size"),
            row_size: symbol!(base, "ggml_row_size"),
            version: symbol!(base, "ggml_version"),
            commit: symbol!(base, "ggml_commit"),
            metal_init: symbol!(metal, "ggml_backend_metal_init"),
            is_metal: symbol!(metal, "ggml_backend_is_metal"),
            backend_name: symbol!(base, "ggml_backend_name"),
            backend_device: symbol!(base, "ggml_backend_get_device"),
            device_description: symbol!(base, "ggml_backend_dev_description"),
            backend_free: symbol!(base, "ggml_backend_free"),
            synchronize: symbol!(base, "ggml_backend_synchronize"),
            supports_op: symbol!(base, "ggml_backend_supports_op"),
            alloc_tensors: symbol!(base, "ggml_backend_alloc_ctx_tensors"),
            buffer_free: symbol!(base, "ggml_backend_buffer_free"),
            tensor_set: symbol!(base, "ggml_backend_tensor_set"),
            tensor_get: symbol!(base, "ggml_backend_tensor_get"),
            tensor_memset: symbol!(base, "ggml_backend_tensor_memset"),
            graph_compute: symbol!(base, "ggml_backend_graph_compute"),
        })
    }
}

pub(super) struct Reference {
    api: Api,
    backend: NonNull<Backend>,
    name: String,
    device: String,
    version: String,
    commit: String,
    paths: [PathBuf; 2],
    // Drop the Metal library before its base dependency, after backend cleanup.
    _metal: Library,
    _base: Library,
    // A graph backend is used serially by this test; do not infer thread safety
    // from libloading's handles or accidentally allow concurrent graph calls.
    _single_thread: PhantomData<Rc<()>>,
}

impl Reference {
    pub(super) fn load(dir: &Path) -> Result<Self, String> {
        let dir = dir
            .canonicalize()
            .map_err(|error| format!("resolve Prism directory {}: {error}", dir.display()))?;
        let paths = [
            dir.join("libggml-base.dylib"),
            dir.join("libggml-metal.dylib"),
        ];
        // SAFETY: These explicit paths are developer-supplied trusted native
        // test inputs. No PATH lookup, download, fallback or production loading.
        let base = unsafe { Library::new(&paths[0]) }
            .map_err(|error| format!("load Prism {}: {error}", paths[0].display()))?;
        let metal = unsafe { Library::new(&paths[1]) }
            .map_err(|error| format!("load Prism {}: {error}", paths[1].display()))?;
        let api = Api::load(&base, &metal)?;
        // These IDs belong to the checked reference ABI, not Ferrum's enums.
        // Check the library's names and physical layouts before making tensors.
        for (id, name, block, bytes) in [(142, "pq2_0", 128, 34), (0, "f32", 1, 4)] {
            // SAFETY: Extension-symbol validation above selects the Prism ABI;
            // the C functions return immutable metadata for valid type IDs.
            let (actual_name, actual_block, actual_bytes) = unsafe {
                (
                    ffi_string((api.type_name)(id), "type name")?,
                    (api.block_size)(id),
                    (api.type_size)(id),
                )
            };
            if actual_name != name || actual_block != block || actual_bytes != bytes {
                return Err(format!(
                    "Prism type {id} ABI mismatch: {actual_name}, block={actual_block}, bytes={actual_bytes}; expected {name}/{block}/{bytes}"
                ));
            }
        }
        // SAFETY: No backend arguments are required; a non-null result is an
        // owned backend which this Reference releases before unloading code.
        let backend = required(unsafe { (api.metal_init)() }, "initialize Metal backend")?;
        let mut reference = Self {
            api,
            backend,
            name: String::new(),
            device: String::new(),
            version: String::new(),
            commit: String::new(),
            paths,
            _metal: metal,
            _base: base,
            _single_thread: PhantomData,
        };
        // SAFETY: Backend and both libraries are live. Failure after this point
        // drops Reference, so backend initialization never leaks on an error.
        unsafe {
            let api = &reference.api;
            if !(api.is_metal)(backend.as_ptr()) {
                return Err("Prism returned a non-Metal backend".into());
            }
            reference.name = ffi_string((api.backend_name)(backend.as_ptr()), "backend name")?;
            let device = required((api.backend_device)(backend.as_ptr()), "get backend device")?;
            reference.device = ffi_string((api.device_description)(device.as_ptr()), "device")?;
            reference.version = ffi_string((api.version)(), "version")?;
            reference.commit = ffi_string((api.commit)(), "commit")?;
        }
        Ok(reference)
    }

    pub(super) fn backend_name(&self) -> &str {
        &self.name
    }

    pub(super) fn device_description(&self) -> &str {
        &self.device
    }

    pub(super) fn version(&self) -> &str {
        &self.version
    }

    pub(super) fn commit(&self) -> &str {
        &self.commit
    }

    pub(super) fn library_paths(&self) -> &[PathBuf; 2] {
        &self.paths
    }

    pub(super) fn matrix(
        &self,
        rows: u32,
        width: u32,
        outputs: u32,
        weights: &[u8],
        input: &[f32],
    ) -> Result<MatrixGraph<'_>, String> {
        if rows == 0 || width == 0 || outputs == 0 || !width.is_multiple_of(128) {
            return Err("Prism matrix requires positive dimensions and complete PQ2 blocks".into());
        }
        if [rows, width, outputs]
            .into_iter()
            .any(|v| v > i32::MAX as u32)
        {
            return Err("Prism Metal matrix dimensions exceed its signed 32-bit ABI".into());
        }
        let weight_bytes = product(&[outputs as usize, width as usize / 128, 34])?;
        let input_values = product(&[rows as usize, width as usize])?;
        let output_values = product(&[rows as usize, outputs as usize])?;
        let input_bytes = product(&[input_values, size_of::<f32>()])?;
        let output_bytes = product(&[output_values, size_of::<f32>()])?;
        if weights.len() != weight_bytes || input.len() != input_values {
            return Err(format!(
                "Prism payload lengths mismatch: weights {}/{weight_bytes} bytes, input {}/{input_values} values",
                weights.len(), input.len()
            ));
        }
        let api = &self.api;
        // Two leaves and one MUL_MAT node.
        const GRAPH_CAPACITY: usize = 3;
        // SAFETY: Pure metadata calls with valid dimensions and checked IDs.
        let metadata_bytes = unsafe {
            if (api.row_size)(142, i64::from(width)) != weight_bytes / outputs as usize {
                return Err("Prism PQ2 row-size ABI mismatch".into());
            }
            (api.tensor_overhead)()
                .checked_mul(3)
                .and_then(|bytes| bytes.checked_add((api.graph_overhead)(GRAPH_CAPACITY, false)))
                .ok_or("Prism metadata allocation size overflow")?
        };
        // SAFETY: no_alloc=true allocates only metadata in the context; the
        // backend owns data storage. The initialized by-value struct is C ABI.
        let context = required(
            unsafe {
                (api.init)(InitParams {
                    mem_size: metadata_bytes,
                    mem_buffer: std::ptr::null_mut(),
                    no_alloc: true,
                    padding: [0; std::mem::size_of::<usize>() - 1],
                })
            },
            "allocate graph context",
        )?;
        let mut allocation = GraphAllocation {
            reference: self,
            context,
            buffer: None,
        };
        // SAFETY: Context is live and owns every returned tensor/graph. Checked
        // dimensions match input payloads; no tensor internals are dereferenced.
        let (graph, output) = unsafe {
            let weight = required(
                (api.new_tensor_2d)(context.as_ptr(), 142, i64::from(width), i64::from(outputs)),
                "create PQ2 weight tensor",
            )?;
            let activation = required(
                (api.new_tensor_2d)(context.as_ptr(), 0, i64::from(width), i64::from(rows)),
                "create F32 input tensor",
            )?;
            let output = required(
                (api.mul_mat)(context.as_ptr(), weight.as_ptr(), activation.as_ptr()),
                "create matrix product",
            )?;
            for (tensor, bytes) in [
                (weight, weight_bytes),
                (activation, input_bytes),
                (output, output_bytes),
            ] {
                if (api.nbytes)(tensor.as_ptr()) != bytes {
                    return Err("Prism tensor byte size differs from the validated ABI".into());
                }
            }
            if !(api.supports_op)(self.backend.as_ptr(), output.as_ptr()) {
                return Err("Prism Metal backend does not support this PQ2 matrix product".into());
            }
            let graph = required(
                (api.new_graph)(context.as_ptr(), GRAPH_CAPACITY, false),
                "create compute graph",
            )?;
            (api.build_forward)(graph.as_ptr(), output.as_ptr());
            if (api.graph_nodes)(graph.as_ptr()) != 1 {
                return Err(
                    "Prism reference graph must contain exactly one matrix operation".into(),
                );
            }
            allocation.buffer = Some(required(
                (api.alloc_tensors)(context.as_ptr(), self.backend.as_ptr()),
                "allocate Metal tensor storage",
            )?);
            (api.tensor_set)(weight.as_ptr(), weights.as_ptr().cast(), 0, weight_bytes);
            (api.tensor_set)(activation.as_ptr(), input.as_ptr().cast(), 0, input_bytes);
            (api.synchronize)(self.backend.as_ptr());
            (graph, output)
        };
        Ok(MatrixGraph {
            allocation,
            graph,
            output,
            output_values,
            has_result: Cell::new(false),
        })
    }
}

impl Drop for Reference {
    fn drop(&mut self) {
        // SAFETY: Graphs borrow Reference and have already released their
        // storage. Libraries remain loaded until after this Drop completes.
        unsafe {
            (self.api.synchronize)(self.backend.as_ptr());
            (self.api.backend_free)(self.backend.as_ptr());
        }
    }
}

struct GraphAllocation<'a> {
    reference: &'a Reference,
    context: NonNull<Context>,
    buffer: Option<NonNull<BackendBuffer>>,
}

impl Drop for GraphAllocation<'_> {
    fn drop(&mut self) {
        // SAFETY: Ownership is unique, including partial construction failures.
        // Complete pending work before freeing data, then tensor/graph metadata.
        unsafe {
            let api = &self.reference.api;
            (api.synchronize)(self.reference.backend.as_ptr());
            if let Some(buffer) = self.buffer {
                (api.buffer_free)(buffer.as_ptr());
            }
            (api.free)(self.context.as_ptr());
        }
    }
}

pub(super) struct MatrixGraph<'a> {
    allocation: GraphAllocation<'a>,
    graph: NonNull<Graph>,
    output: NonNull<Tensor>,
    output_values: usize,
    has_result: Cell<bool>,
}

impl MatrixGraph<'_> {
    pub(super) fn run_timed(&self) -> Result<Duration, String> {
        let reference = self.allocation.reference;
        self.has_result.set(false);
        // SAFETY: The live output allocation has the validated byte length.
        // All-ones bytes are F32 NaNs, exposing unwritten/stale output values.
        // Poisoning and its synchronization stay outside the timed interval.
        unsafe {
            (reference.api.tensor_memset)(
                self.output.as_ptr(),
                0xff,
                0,
                self.output_values * size_of::<f32>(),
            );
            (reference.api.synchronize)(reference.backend.as_ptr());
        }
        let started = Instant::now();
        // SAFETY: Backend, graph and data storage are live. This public call
        // synchronizes internally, so the timer includes encoding/submission
        // and completion, but excludes setup and output readback.
        let status = unsafe {
            (reference.api.graph_compute)(reference.backend.as_ptr(), self.graph.as_ptr())
        };
        let elapsed = started.elapsed();
        if status != 0 {
            return Err(format!(
                "Prism Metal graph compute failed with ggml_status={status}"
            ));
        }
        self.has_result.set(true);
        Ok(elapsed)
    }

    pub(super) fn read_output(&self) -> Result<Vec<f32>, String> {
        if !self.has_result.get() {
            return Err("Prism output requires a successful synchronized execution first".into());
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(self.output_values)
            .map_err(|error| format!("allocate Prism output readback: {error}"))?;
        values.resize(self.output_values, 0.0_f32);
        // SAFETY: Output size was checked against ggml_nbytes. Readback writes
        // exactly the initialized Vec's storage; this synchronous API does not
        // retain the host pointer after returning.
        unsafe {
            let reference = self.allocation.reference;
            (reference.api.tensor_get)(
                self.output.as_ptr(),
                values.as_mut_ptr().cast(),
                0,
                std::mem::size_of_val(values.as_slice()),
            );
        }
        Ok(values)
    }
}

fn required<T>(pointer: *mut T, operation: &str) -> Result<NonNull<T>, String> {
    NonNull::new(pointer).ok_or_else(|| format!("Prism failed to {operation}"))
}

fn product(values: &[usize]) -> Result<usize, String> {
    values.iter().try_fold(1_usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| "Prism matrix byte count overflow".into())
    })
}

unsafe fn ffi_string(pointer: *const c_char, field: &str) -> Result<String, String> {
    if pointer.is_null() {
        return Err(format!("Prism returned a null {field}"));
    }
    // SAFETY: Caller guarantees a valid, library-owned NUL-terminated C string.
    unsafe { CStr::from_ptr(pointer) }
        .to_str()
        .map(str::to_owned)
        .map_err(|error| format!("Prism {field} is not UTF-8: {error}"))
}
