//! Optional native decode uses the existing checked metadata and KV ownership.
//! All device payloads live in the provider's declared invocation workspace.

use super::*;
use crate::native_ops::flashinfer_xqa::{
    self as native, Bindings, DeviceSpan, PreparedXqa, Request,
};
use cudarc::driver::sys::CUdevice_attribute;

const SINK_BYTES: u64 = 64 * 4;
const SEMAPHORE_BYTES_PER_SEQUENCE: u64 = 8 * 4;
const SCRATCH_BYTES_PER_SUBSEQUENCE: u64 = 1280;
const ALIGNMENT: u64 = native::FLASHINFER_XQA_SCRATCH_ALIGNMENT;

#[derive(Clone)]
pub(super) struct XqaSupport {
    context: Arc<cudarc::driver::CudaContext>,
    cast_sinks: CudaFunction,
    ordinal: u32,
    multiprocessors: u32,
    maximum_shared_bytes: u64,
}

impl XqaSupport {
    pub(super) fn discover(
        runtime: &CudaDeviceRuntime,
    ) -> Result<Option<Self>, CudaDeviceRuntimeError> {
        if !native::compiled() {
            return Ok(None);
        }
        let attribute = |name| {
            runtime.context().attribute(name).map_err(|error| {
                CudaDeviceRuntimeError::driver("GPT-OSS XQA device capability", error)
            })
        };
        let major = attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        if (major, minor) != (12, 0) {
            return Ok(None);
        }
        let multiprocessors =
            attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
        let maximum_shared_bytes =
            attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)?;
        if multiprocessors <= 0 || maximum_shared_bytes <= 0 {
            return Err(CudaDeviceRuntimeError::contract(
                "GPT-OSS XQA device limits are invalid",
            ));
        }
        let module = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::SANDWICH_NORM.to_owned()))
            .map_err(|error| {
                CudaDeviceRuntimeError::driver("GPT-OSS XQA sink conversion module", error)
            })?;
        Ok(Some(Self {
            context: runtime.context().clone(),
            cast_sinks: load_function(
                &module,
                "cast_f16_to_f32_slice",
                "GPT-OSS XQA sink conversion",
            )?,
            ordinal: runtime.descriptor().ordinal,
            multiprocessors: multiprocessors as u32,
            maximum_shared_bytes: maximum_shared_bytes as u64,
        }))
    }

    pub(super) fn fingerprint_material(&self) -> Vec<u8> {
        let mut result = Vec::new();
        result.extend(self.ordinal.to_le_bytes());
        result.extend(self.multiprocessors.to_le_bytes());
        result.extend(self.maximum_shared_bytes.to_le_bytes());
        result
    }

    pub(super) fn eligible(shape: GptOssAttentionShape) -> bool {
        shape.query_heads == 64 && shape.kv_heads == 8 && shape.head_dim == 64
    }

    // Native split > 1 implies B*8*split <= SMcount. This static bound is
    // independent of wave size; the prepared plan still checks exact extents.
    fn scratch_bound(&self) -> u64 {
        u64::from(self.multiprocessors) * SCRATCH_BYTES_PER_SUBSEQUENCE
    }

    pub(super) fn workspace_formula(&self) -> (u64, u64, u64) {
        (
            SINK_BYTES + ALIGNMENT - 1 + self.scratch_bound(),
            SEMAPHORE_BYTES_PER_SEQUENCE,
            ALIGNMENT,
        )
    }

    pub(super) fn workspace_bytes(&self, base: u64, sequences: usize) -> Result<u64, String> {
        let (fixed, per_sequence, _) = self.workspace_formula();
        base.checked_add(fixed)
            .and_then(|bytes| bytes.checked_add(per_sequence.checked_mul(sequences as u64)?))
            .ok_or_else(|| "GPT-OSS XQA workspace extent overflows".to_owned())
    }

    pub(super) fn prepare(
        &self,
        shape: GptOssAttentionShape,
        layout: ScratchLayout,
        binding: BindingLayout,
        batch: usize,
    ) -> Result<Option<XqaDecode>, String> {
        if !Self::eligible(shape) || batch > 65535 {
            return Ok(None);
        }
        let request = Request {
            abi: native::FLASHINFER_XQA_ABI,
            size: std::mem::size_of::<Request>() as u32,
            batch: u32::try_from(batch).map_err(|_| "GPT-OSS XQA batch overflows")?,
            max_pages_per_sequence: u32::try_from(shape.maximum_pages()?)
                .map_err(|_| "GPT-OSS XQA pages overflow")?,
            binding_stride_bytes: u32::try_from(binding.slot_bytes)
                .map_err(|_| "GPT-OSS XQA binding stride overflows")?,
            sliding_window: u32::try_from(shape.sliding_window_tokens)
                .map_err(|_| "GPT-OSS XQA window overflows")?,
            device_ordinal: self.ordinal,
            architecture: 1200,
            multiprocessors: self.multiprocessors,
            reserved: 0,
            maximum_dynamic_shared_bytes: self.maximum_shared_bytes,
        };
        // Preparation queries native shared-memory requirements outside graph
        // capture. Unsupported device geometry can use the existing provider;
        // an artifact/ABI/runtime failure must remain visible.
        self.context
            .bind_to_thread()
            .map_err(|error| format!("GPT-OSS XQA preparation context: {error}"))?;
        let prepared = match PreparedXqa::new(request) {
            Ok(prepared) => prepared,
            Err(native::Error::UnsupportedGeometry) => return Ok(None),
            Err(error) => return Err(format!("GPT-OSS XQA preparation: {error}")),
        };
        let workspace =
            XqaWorkspace::new(layout.required_bytes, prepared.plan(), self.scratch_bound())?;
        if workspace.required_bytes > self.workspace_bytes(layout.required_bytes, batch)? {
            return Err("GPT-OSS XQA workspace exceeds the declared resource estimate".to_owned());
        }
        Ok(Some(XqaDecode {
            prepared,
            cast_sinks: self.cast_sinks.clone(),
            workspace,
        }))
    }
}

#[derive(Debug, Clone, Copy)]
struct XqaWorkspace {
    sinks: u64,
    semaphores: u64,
    scratch: u64,
    required_bytes: u64,
}

impl XqaWorkspace {
    fn new(base: u64, plan: &native::Plan, scratch_bound: u64) -> Result<Self, String> {
        plan.validate().map_err(|error| error.to_string())?;
        if base % SCRATCH_ALIGNMENT != 0 || plan.scratch_bytes > scratch_bound {
            return Err("GPT-OSS XQA workspace does not fit the static bound".to_owned());
        }
        let semaphores = base
            .checked_add(SINK_BYTES)
            .ok_or("XQA sink offset overflows")?;
        let semaphore_end = semaphores
            .checked_add(plan.semaphore_bytes)
            .ok_or("XQA semaphore offset overflows")?;
        let scratch = semaphore_end
            .checked_add(ALIGNMENT - 1)
            .ok_or("XQA scratch alignment overflows")?
            / ALIGNMENT
            * ALIGNMENT;
        let required_bytes = scratch
            .checked_add(plan.scratch_bytes)
            .ok_or("XQA scratch extent overflows")?;
        Ok(Self {
            sinks: base,
            semaphores,
            scratch,
            required_bytes,
        })
    }
}

#[derive(Clone)]
pub(super) struct XqaDecode {
    prepared: PreparedXqa,
    cast_sinks: CudaFunction,
    workspace: XqaWorkspace,
}

impl XqaDecode {
    pub(super) fn transfer_count(&self) -> u64 {
        u64::from(self.prepared.plan().subsequences > 1)
    }

    pub(super) fn fingerprint_material(&self) -> Vec<u8> {
        let mut result = self.prepared.fingerprint_material();
        for value in [
            self.workspace.sinks,
            self.workspace.semaphores,
            self.workspace.scratch,
            self.workspace.required_bytes,
        ] {
            result.extend(value.to_le_bytes());
        }
        result
    }

    fn bindings(
        &self,
        scratch: &CudaBufferRegion,
        metadata: &CudaBufferRegion,
        layout: ScratchLayout,
    ) -> Result<Bindings, CudaDeviceRuntimeError> {
        if scratch.length_bytes() < self.workspace.required_bytes
            || scratch.device_ptr() % ALIGNMENT != 0
        {
            return Err(CudaDeviceRuntimeError::contract(
                "GPT-OSS XQA workspace extent/alignment disagrees with admission",
            ));
        }
        let plan = self.prepared.plan();
        let metadata_bytes = plan
            .request
            .metadata_bytes()
            .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))?;
        if metadata.length_bytes() < metadata_bytes {
            return Err(CudaDeviceRuntimeError::contract(
                "GPT-OSS XQA metadata extent exceeds binding workspace",
            ));
        }
        let span = |offset, bytes| -> Result<DeviceSpan, CudaDeviceRuntimeError> {
            Ok(DeviceSpan {
                address: scratch_pointer(scratch.device_ptr(), offset)?,
                bytes,
            })
        };
        let bindings = Bindings {
            query: span(layout.query, plan.request.query_bytes())?,
            sinks_f32: span(self.workspace.sinks, SINK_BYTES)?,
            packed_metadata: DeviceSpan {
                address: metadata.device_ptr(),
                bytes: metadata_bytes,
            },
            output: span(layout.context, plan.request.query_bytes())?,
            semaphores: span(self.workspace.semaphores, plan.semaphore_bytes)?,
            scratch: if plan.scratch_bytes == 0 {
                DeviceSpan::default()
            } else {
                span(self.workspace.scratch, plan.scratch_bytes)?
            },
        };
        self.prepared
            .validate(&bindings)
            .map_err(|e| CudaDeviceRuntimeError::contract(format!("GPT-OSS XQA bindings: {e}")))?;
        Ok(bindings)
    }

    pub(super) fn enqueue(
        &self,
        stream: &CudaStream,
        scratch: &CudaBufferRegion,
        metadata: &CudaBufferRegion,
        sinks: &CudaBufferRegion,
        layout: ScratchLayout,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let bindings = self.bindings(scratch, metadata, layout)?;
        if sinks.length_bytes() != 64 * 2 {
            return Err(CudaDeviceRuntimeError::contract(
                "GPT-OSS XQA sink weight extent differs from F16[64]",
            ));
        }
        let source = sinks.device_ptr();
        let destination = bindings.sinks_f32.address;
        let zero = 0_i32;
        let count = 64_i32;
        let mut args = stream.launch_builder(&self.cast_sinks);
        args.arg(&source)
            .arg(&destination)
            .arg(&zero)
            .arg(&zero)
            .arg(&count);
        // SAFETY: both exact spans are retained by the compute command; the
        // conversion preserves the original F16-rounded learned sink values.
        unsafe { args.launch(LaunchConfig::for_num_elems(64)) }.map_err(|error| {
            CudaDeviceRuntimeError::driver("GPT-OSS XQA sink conversion", error)
        })?;
        // SAFETY: existing per-wave admission validates all metadata and page
        // extents. The command retains workspace/weights and page fence leases
        // until completion, including failures after partial enqueue.
        unsafe { self.prepared.run(&bindings, stream.cu_stream().cast()) }.map_err(|error| {
            CudaDeviceRuntimeError::contract(format!("GPT-OSS XQA execution: {error}"))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_bound_covers_all_split_sizes_and_alignment() {
        for multiprocessors in [1, 84, 170, 192] {
            for batch in [1, 2, 8, 16, 32, 256, 65535] {
                let request = Request {
                    abi: native::FLASHINFER_XQA_ABI,
                    size: std::mem::size_of::<Request>() as u32,
                    batch,
                    max_pages_per_sequence: 4096,
                    binding_stride_bytes: 32784,
                    sliding_window: 128,
                    device_ordinal: 0,
                    architecture: 1200,
                    multiprocessors,
                    reserved: 0,
                    maximum_dynamic_shared_bytes: 100000,
                };
                let plan = native::Plan::checked(request, 65536).unwrap();
                let bound = u64::from(multiprocessors) * SCRATCH_BYTES_PER_SUBSEQUENCE;
                for base in [0, 16, 1008, 1024, 1040] {
                    let layout = XqaWorkspace::new(base, &plan, bound).unwrap();
                    let estimate = base + SINK_BYTES + ALIGNMENT - 1
                        + bound
                        + u64::from(batch) * SEMAPHORE_BYTES_PER_SEQUENCE;
                    assert!(layout.required_bytes <= estimate);
                    assert_eq!(layout.scratch % ALIGNMENT, 0);
                    assert!(layout.scratch >= layout.semaphores + plan.semaphore_bytes);
                }
            }
        }
    }
}
