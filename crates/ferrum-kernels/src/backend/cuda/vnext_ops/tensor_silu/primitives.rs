//! Generic tensor primitives. Affine projection uses admitted FP32 scratch
//! and the lane-owned cuBLAS handle; every other payload stays in its binding.
use super::*;
use cudarc::cublas::{result::gemm_ex, sys};
use ferrum_runtime::{
    DynamicStorageRequirement, PhysicalWeightLayout, ProviderWorkspaceRequirement,
    ProviderWorkspaceReusePolicy, ProviderWorkspaceScope, ProviderWorkspaceSizeFormula,
    TensorPrimitiveBf16, TensorPrimitiveBf16Geometry, WeightFormatId,
};
use std::ffi::c_void;

pub(super) const FUNCTION: &str = "ferrum_vnext_tensor_primitive_bf16";
pub const NUMERICAL_IDENTITY: &str = "bf16-tensor-primitives-v1;affine=gemmex-fp32-output-plus-fp32-bias-then-bf16;patch=gemmex-fp32-output-to-bf16-before-bias-add-and-final-bf16;zero-blas-workspace;norm=fp32-two-pass-256-thread-reduction-affine-then-bf16;rope=fp32-half-rotation-then-bf16;gelu=libdevice-erff;attention=padded-power2-dot-sequential-online-softmax-independent-segments;reshape=bitcopy;scatter=fp32-add-then-bf16;no-target-parity-claim";

pub fn profile(kind: TensorPrimitiveBf16) -> String {
    format!("cuda.tensor_primitive.{}.bf16.v1", kind.name())
}

pub(super) fn descriptor(
    device: &DeviceDescriptor,
    kind: TensorPrimitiveBf16,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let capability = CapabilityId::new(kind.capability_id()).map_err(contract_error)?;
    if !device.capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA lacks selected tensor primitive capability",
        ));
    }
    let contract = kind.contract().map_err(contract_error)?;
    let provider = format!("provider.cuda.tensor_primitive.{}.bf16", kind.name());
    let estimator = format!(
        "resource-estimator.cuda.tensor_primitive.{}.bf16",
        kind.name()
    );
    let version = super::super::tensor_dense_bf16::cublas_version()?;
    let implementation = implementation_fingerprint(&[
        numerical_fingerprint().as_bytes(),
        NUMERICAL_IDENTITY.as_bytes(),
        profile(kind).as_bytes(),
        provider.as_bytes(),
        format!("cublas={version}").as_bytes(),
        include_bytes!("primitives.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/vnext_tensor_primitives.cuh"
        )),
    ]);
    let estimator_fingerprint =
        implementation_fingerprint(&[estimator.as_bytes(), implementation.as_bytes()]);
    OperationProviderDescriptor::new(
        ProviderId::new(provider).map_err(contract_error)?,
        contract.descriptor().id.clone(),
        contract
            .descriptor()
            .fingerprint()
            .map_err(contract_error)?,
        implementation,
        ProviderExecutionSemantics::bitwise_eager_only(),
        contract.descriptor().version,
        device.id.clone(),
        BTreeSet::from([capability]),
        BTreeSet::from([
            WeightFormatId::new(super::super::DENSE_SAFETENSORS_FORMAT_ID)
                .map_err(contract_error)?,
        ]),
        BTreeSet::new(),
        super::super::transformer::contiguous_bindings(kind.input_count()),
        estimator,
        ContractVersion::new(1, 0),
        estimator_fingerprint,
    )
    .map_err(contract_error)
    .map(|d| d.with_fixed_tensor_work(true))
}

pub(super) fn provider(
    kind: TensorPrimitiveBf16,
    descriptor: OperationProviderDescriptor,
    kernels: &Arc<RetainedFunctions>,
) -> Box<dyn OperationProvider<CudaDeviceRuntime>> {
    Box::new(Provider {
        kind,
        descriptor,
        kernels: Arc::clone(kernels),
    })
}
struct Provider {
    kind: TensorPrimitiveBf16,
    descriptor: OperationProviderDescriptor,
    kernels: Arc<RetainedFunctions>,
}
impl OperationResourceEstimator for Provider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }
    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        if request.operation().id != *self.descriptor.operation_id()
            || request.operation().fingerprint()? != self.descriptor.operation_fingerprint()
        {
            return Err(invalid_plan(
                "tensor primitive estimator received another operation",
            ));
        }
        let geometry = checked_geometry(self.kind, request.values(), request.attributes())
            .map_err(invalid_plan)?;
        let bytes = scratch_bytes(&geometry).map_err(invalid_plan)?;
        let scratch = if bytes == 0 {
            None
        } else {
            Some(ProviderWorkspaceRequirement::from_formula(
                ProviderWorkspaceSizeFormula::fixed(bytes)?,
                ALIGNMENT,
                ProviderWorkspaceScope::Invocation,
                ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
                DynamicStorageRequirement::contiguous(),
            )?)
        };
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            ALIGNMENT,
            scratch,
            None,
        ))
    }
}
impl OperationProvider<CudaDeviceRuntime> for Provider {
    fn reusable_execution_topology(
        &self,
        _: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        Ok(ReusableExecutionTopology::EagerBoundary)
    }
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        Err(provider_failure(
            invocation.participants()[0].identity().clone(),
            "cuda.tensor_primitive",
            "requires fixed tensors".into(),
        ))
    }
    fn encode_tensor_selected(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<CudaDeviceCommand>>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        self.encode(invocation)
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|e| provider_failure(identity, "cuda.tensor_primitive", e))
    }
}
impl Provider {
    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [participant] = invocation.participants() else {
            return Err("tensor primitive requires one participant".into());
        };
        if invocation.operation().id.as_str() != self.kind.operation_id()
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("tensor primitive requires its fixed tensor contract".into());
        }
        let geometry =
            checked_geometry(self.kind, participant.bindings(), participant.attributes())?;
        let native = Native::new(&geometry);
        let bytes = scratch_bytes(&geometry)?;
        let mut regions = Vec::with_capacity(self.kind.input_count() as usize + 2);
        for ordinal in 0..self.kind.input_count() {
            regions.push(retained_typed_region(
                participant,
                binding(participant.bindings(), ResolvedValueRole::Input, ordinal)?,
                geometry.input_bytes[ordinal as usize],
                input_dtype(self.kind, ordinal),
            )?);
        }
        let output = retained_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            geometry.output_bytes,
        )?;
        for input in &regions {
            nonoverlapping_ranges(
                input.device_ptr(),
                input.length_bytes(),
                output.device_ptr(),
                output.length_bytes(),
            )?;
        }
        regions.push(output);
        if bytes > 0 {
            let view = participant
                .scratch_view()
                .ok_or("FP32 affine scratch was not admitted")?;
            let translated = view.translate(0, bytes).map_err(|e| e.to_string())?;
            let mut pieces = translated.iter();
            let piece = pieces.next().ok_or("affine scratch is empty")?;
            if pieces.next().is_some() {
                return Err("affine scratch must be contiguous".into());
            }
            let (buffer, range, retention) = piece.buffer_and_physical_range();
            let region = buffer
                .retained_region(range, retention)
                .map_err(|e| e.to_string())?;
            if region.element_type() != ElementType::U8
                || region.length_bytes() != bytes
                || region.device_ptr() % ALIGNMENT != 0
            {
                return Err("affine scratch dtype, extent or alignment mismatch".into());
            }
            for input in &regions {
                nonoverlapping_ranges(
                    input.device_ptr(),
                    input.length_bytes(),
                    region.device_ptr(),
                    region.length_bytes(),
                )?;
            }
            regions.push(region);
        } else if participant.scratch_view().is_some() {
            return Err("unexpected scratch for tensor primitive".into());
        }
        let count = geometry.elements();
        let operations = arithmetic(&geometry)?;
        let kernels = Arc::clone(&self.kernels);
        let kind = self.kind;
        CudaDeviceCommand::operation_with_blas(FUNCTION, regions, move |stream, blas, regions| {
            let x = regions[0].device_ptr();
            let a = if kind.input_count() > 1 {
                regions[1].device_ptr()
            } else {
                0
            };
            let b = if kind.input_count() > 2 {
                regions[2].device_ptr()
            } else {
                0
            };
            let y = regions[kind.input_count() as usize].device_ptr();
            let scratch = if bytes > 0 {
                regions.last().unwrap().device_ptr()
            } else {
                0
            };
            if bytes > 0 {
                affine_gemm(stream, blas, x, a, scratch, &geometry)?;
            }
            let threads = if kind == TensorPrimitiveBf16::SegmentedAttention {
                geometry.head_dim.next_power_of_two() as u32
            } else {
                THREADS
            };
            let tasks = match kind {
                TensorPrimitiveBf16::AffineLayerNorm => geometry.rows,
                TensorPrimitiveBf16::SegmentedAttention => {
                    geometry.rows * (geometry.columns / geometry.head_dim)
                }
                _ => count.div_ceil(u64::from(threads)),
            };
            let mut launch = stream.launch_builder(kernels.primitive_function());
            launch
                .arg(&x)
                .arg(&a)
                .arg(&b)
                .arg(&y)
                .arg(&scratch)
                .arg(&native);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (tasks.min(MAX_BLOCKS) as u32, 1, 1),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map(|_| ())
            .map_err(|e| CudaDeviceRuntimeError::driver("tensor primitive launch", e))
        })
        .and_then(|c| c.with_tensor_work(count, operations, if bytes > 0 { 2 } else { 1 }, 0))
        .map_err(|e| e.to_string())
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Native {
    rows: u64,
    columns: u64,
    head_dim: u64,
    starts: [u64; 32],
    lengths: [u64; 32],
    scalar: f32,
    kind: u32,
    count: u32,
    padding: u32,
}
// SAFETY: initialized fixed-width scalars and arrays match FerrumTensorPrimitive.
unsafe impl cudarc::driver::DeviceRepr for Native {}
const _: [(); 552] = [(); std::mem::size_of::<Native>()];
impl Native {
    fn new(g: &TensorPrimitiveBf16Geometry) -> Self {
        let mut p = Self {
            rows: g.rows,
            columns: g.columns,
            head_dim: g.head_dim,
            starts: [0; 32],
            lengths: [0; 32],
            scalar: g.scalar,
            kind: TensorPrimitiveBf16::ALL
                .iter()
                .position(|&k| k == g.kind)
                .unwrap() as u32,
            count: g.lengths.len() as u32,
            padding: 0,
        };
        p.starts[..g.starts.len()].copy_from_slice(&g.starts);
        p.lengths[..g.lengths.len()].copy_from_slice(&g.lengths);
        p
    }
}
fn checked_geometry(
    kind: TensorPrimitiveBf16,
    values: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<TensorPrimitiveBf16Geometry, String> {
    let g = TensorPrimitiveBf16Geometry::new(kind, values, attrs).map_err(|e| e.to_string())?;
    for value in values {
        if value
            .weight()
            .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        {
            return Err("tensor primitive requires dense unquantized storage".into());
        }
    }
    for dim in [g.rows, g.columns, g.inner] {
        if dim > i32::MAX as u64 {
            return Err("CUDA tensor primitive dimensions exceed i32".into());
        }
    }
    arithmetic(&g)?;
    Ok(g)
}
fn scratch_bytes(g: &TensorPrimitiveBf16Geometry) -> Result<u64, String> {
    if matches!(
        g.kind,
        TensorPrimitiveBf16::AffineLinear | TensorPrimitiveBf16::PatchProjection
    ) {
        g.output_bytes
            .checked_mul(2)
            .filter(|&n| n <= isize::MAX as u64)
            .ok_or_else(|| "affine scratch exceeds address space".into())
    } else {
        Ok(0)
    }
}
fn input_dtype(kind: TensorPrimitiveBf16, ordinal: u32) -> ElementType {
    if kind == TensorPrimitiveBf16::HalfRotary && ordinal > 0 {
        ElementType::F32
    } else {
        ElementType::Bf16
    }
}
fn arithmetic(g: &TensorPrimitiveBf16Geometry) -> Result<u64, String> {
    // Nominal finite-path scalar work, not hardware FLOPs/instruction counts.
    // Count sqrt/erf/exp as one logical call and BF16 conversions/copies as zero.
    // A shared scalar (e.g. one row's reciprocal) is counted once, even when
    // CUDA threads compute replicas. Reduction padding follows this provider.
    let work = (|| -> Option<u64> {
        Some(match g.kind {
            TensorPrimitiveBf16::AffineLinear | TensorPrimitiveBf16::PatchProjection => g
                .elements()
                .checked_mul(g.inner.checked_mul(2)?.checked_add(1)?)?,
            TensorPrimitiveBf16::AffineLayerNorm => {
                // C mean adds + 255 tree adds + divide; 3C centered-square
                // work + 255 tree adds + divide; epsilon/sqrt/reciprocal;
                // 4C normalize/affine work. The native launch uses 256 lanes.
                let reduction = u64::from(THREADS).checked_sub(1)?.checked_mul(2)?;
                g.rows.checked_mul(
                    g.columns
                        .checked_mul(8)?
                        .checked_add(reduction)?
                        .checked_add(5)?,
                )?
            }
            TensorPrimitiveBf16::GeluErf => g.elements().checked_mul(5)?,
            TensorPrimitiveBf16::HalfRotary => (g.elements() / 2).checked_mul(7)?,
            TensorPrimitiveBf16::RowScatterAdd => {
                let updated = g.lengths.iter().try_fold(0_u64, |n, &l| n.checked_add(l))?;
                updated.checked_mul(g.columns)?
            }
            TensorPrimitiveBf16::ReshapeCopy => 0,
            TensorPrimitiveBf16::SegmentedAttention => {
                let heads = g.columns / g.head_dim;
                // One key: D products, (padded D - 1) tree adds, scale.
                let dot = g
                    .head_dim
                    .checked_add(g.head_dim.checked_next_power_of_two()?)?;
                let update = g.head_dim.checked_mul(3)?.checked_add(6)?;
                g.lengths.iter().try_fold(0_u64, |total, &length| {
                    let per_query = length
                        .checked_mul(dot)?
                        .checked_add(length.checked_sub(1)?.checked_mul(update)?)?
                        .checked_add(g.head_dim.checked_add(1)?)?;
                    total.checked_add(per_query.checked_mul(length)?.checked_mul(heads)?)
                })?
            }
        })
    })();
    work.ok_or_else(|| "tensor primitive work overflows".into())
}
fn affine_gemm(
    stream: &cudarc::driver::CudaStream,
    blas: &cudarc::cublas::CudaBlas,
    x: u64,
    w: u64,
    output: u64,
    g: &TensorPrimitiveBf16Geometry,
) -> Result<(), CudaDeviceRuntimeError> {
    let handle = *blas.handle();
    let alpha = 1.0_f32;
    let beta = 0.0_f32;
    unsafe {
        sys::cublasSetStream_v2(handle, stream.cu_stream() as _)
            .result()
            .map_err(|e| CudaDeviceRuntimeError::blas("affine stream", e))?;
        sys::cublasSetWorkspace_v2(handle, std::ptr::null_mut(), 0)
            .result()
            .map_err(|e| CudaDeviceRuntimeError::blas("affine zero workspace", e))?;
        sys::cublasSetPointerMode_v2(handle, sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST)
            .result()
            .map_err(|e| CudaDeviceRuntimeError::blas("affine host scalars", e))?;
        sys::cublasSetMathMode(
            handle,
            sys::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
        )
        .result()
        .map_err(|e| CudaDeviceRuntimeError::blas("affine FP32 reduction", e))?;
        sys::cublasSetAtomicsMode(handle, sys::cublasAtomicsMode_t::CUBLAS_ATOMICS_NOT_ALLOWED)
            .result()
            .map_err(|e| CudaDeviceRuntimeError::blas("affine atomics", e))?;
        gemm_ex(
            handle,
            sys::cublasOperation_t::CUBLAS_OP_T,
            sys::cublasOperation_t::CUBLAS_OP_N,
            g.columns as i32,
            g.rows as i32,
            g.inner as i32,
            &alpha as *const f32 as *const c_void,
            w as *const c_void,
            sys::cudaDataType_t::CUDA_R_16BF,
            g.inner as i32,
            x as *const c_void,
            sys::cudaDataType_t::CUDA_R_16BF,
            g.inner as i32,
            &beta as *const f32 as *const c_void,
            output as *mut c_void,
            sys::cudaDataType_t::CUDA_R_32F,
            g.columns as i32,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    }
    .map_err(|e| CudaDeviceRuntimeError::blas("affine BF16 to FP32 GEMM", e))
}

#[cfg(test)]
mod tests;
