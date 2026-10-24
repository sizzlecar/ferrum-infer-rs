//! Fixed-tensor BF16 GEMM using an explicitly configured, lane-owned cuBLAS
//! handle. The provider allocates no matrices, workspace, handles or modules.
//!
//! CUDA 12.8.1 cuBLAS sections 2.4.8 and 2.2.10 specify that workspace size
//! zero prevents use of the default pool, and DISALLOW_REDUCED_PRECISION_REDUCTION
//! keeps mixed-precision reductions in the compute type. An unsupported zero
//! workspace execution fails; it never retries with an implicit workspace.
//! https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html#cublassetworkspace

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::c_void;

use cudarc::cublas::{result::gemm_ex, sys, CudaBlas};
use cudarc::driver::CudaStream;
use ferrum_runtime::{
    dense_linear_bf16_contract, AttributeId, BatchedOperationInvocation, CapabilityId,
    ContractVersion, DeviceCapacityRetention, DeviceRuntime, ElementType, EncodedDeviceOperation,
    InvocationWorkProjection, NodeWorkContract, OperationContract, OperationFailure,
    OperationInvocation, OperationProvider, OperationProviderDescriptor, OperationResourceEstimate,
    OperationResourceEstimateRequest, OperationResourceEstimator, PhysicalWeightLayout,
    ProviderExecutionSemantics, ProviderId, ResolvedTensorLayout, ResolvedTensorSpec,
    ResolvedValueBinding, ResolvedValueRole, ReusableExecutionTopology,
    ReusableExecutionTopologyRequest, SemanticValue, TensorOperationInvocation, VNextError,
    WeightFormatId, DENSE_LINEAR_BF16_CAPABILITY_ID, DENSE_LINEAR_BF16_OPERATION_ID,
};

use super::super::vnext_runtime::{
    CudaBufferRegion, CudaDeviceBuffer, CudaDeviceCommand, CudaDeviceRuntime,
    CudaDeviceRuntimeError,
};
use super::{binding, contract_error, implementation_fingerprint, provider_failure};

pub const NUMERICAL_PROFILE_ID: &str = "cuda.cublas.gemmex.bf16_f32.no_workspace.v1";
pub const NUMERICAL_PROFILE_IDENTITY: &str = concat!(
    "gemm-ex;A=B=C=CUDA_R_16BF;compute=CUBLAS_COMPUTE_32F;",
    "math=DEFAULT|DISALLOW_REDUCED_PRECISION_REDUCTION;algo=GEMM_DEFAULT;",
    "atomics=NOT_ALLOWED;alpha=host-f32-one;beta=host-f32-zero;workspace=null-0;",
    "OI-row-major;OP_T,OP_N;no-bias;no-replay;operation-defined-error-bound;",
    "no-Reference-reduction-or-bitwise-equivalence"
);
const PROVIDER_ID: &str = "provider.cuda.dense_linear.bf16.cublas.no_workspace";
const ESTIMATOR_ID: &str = "resource-estimator.cuda.dense_linear.bf16.cublas.no_workspace";
const ALIGNMENT: u64 = 16;
const ALPHA: f32 = 1.0;
const BETA: f32 = 0.0;

pub(super) fn source_fingerprint() -> String {
    implementation_fingerprint(&[
        NUMERICAL_PROFILE_ID.as_bytes(),
        NUMERICAL_PROFILE_IDENTITY.as_bytes(),
        include_bytes!("tensor_dense_bf16.rs"),
    ])
}

pub(super) fn cublas_version() -> Result<i32, CudaDeviceRuntimeError> {
    // cuBLAS 12.8.1 section 2.4.3 explicitly permits a null handle here.
    let mut version = 0;
    unsafe { sys::cublasGetVersion_v2(std::ptr::null_mut(), &mut version) }
        .result()
        .map_err(|error| CudaDeviceRuntimeError::blas("cuBLAS version query", error))?;
    if version <= 0 {
        return Err(CudaDeviceRuntimeError::contract(
            "cuBLAS returned an invalid version",
        ));
    }
    Ok(version)
}

pub(super) struct CudaTensorDenseBf16Provider {
    descriptor: OperationProviderDescriptor,
    bootstrap: DeviceCapacityRetention,
}

impl CudaTensorDenseBf16Provider {
    pub(super) fn new(runtime: &CudaDeviceRuntime) -> Result<Self, CudaDeviceRuntimeError> {
        let bootstrap = runtime.tensor_bootstrap_retention()?;
        let capability =
            CapabilityId::new(DENSE_LINEAR_BF16_CAPABILITY_ID).map_err(contract_error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA tensor runtime lacks BF16 Dense capability",
            ));
        }
        let contract = dense_linear_bf16_contract().map_err(contract_error)?;
        // NVIDIA permits GetVersion with a null handle. This reads the loaded
        // library identity without creating an unbudgeted second handle.
        let version = cublas_version()?;
        let implementation = implementation_fingerprint(&[
            source_fingerprint().as_bytes(),
            format!("cublas.version={version}").as_bytes(),
        ]);
        let estimator =
            implementation_fingerprint(&[ESTIMATOR_ID.as_bytes(), implementation.as_bytes()]);
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new(PROVIDER_ID).map_err(contract_error)?,
            contract.descriptor().id.clone(),
            contract
                .descriptor()
                .fingerprint()
                .map_err(contract_error)?,
            implementation,
            ProviderExecutionSemantics::bitwise_eager_only(),
            contract.descriptor().version,
            runtime.descriptor().id.clone(),
            BTreeSet::from([capability]),
            BTreeSet::from([
                WeightFormatId::new(super::DENSE_SAFETENSORS_FORMAT_ID).map_err(contract_error)?,
                WeightFormatId::new("weight-format.gguf.native-block").map_err(contract_error)?,
            ]),
            BTreeSet::new(),
            super::transformer::contiguous_bindings(2),
            ESTIMATOR_ID,
            ContractVersion::new(1, 0),
            estimator,
        )
        .map_err(contract_error)?
        .with_fixed_tensor_work(true);
        Ok(Self {
            descriptor,
            bootstrap,
        })
    }
}

impl OperationResourceEstimator for CudaTensorDenseBf16Provider {
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
                "CUDA BF16 Dense estimator received another operation",
            ));
        }
        geometry(request.values(), request.attributes()).map_err(invalid_plan)?;
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            ALIGNMENT,
            None,
            None,
        ))
    }
}

impl OperationProvider<CudaDeviceRuntime> for CudaTensorDenseBf16Provider {
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
            "cuda.dense.bf16.encode",
            "CUDA BF16 Dense supports fixed tensors only".into(),
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
            .map_err(|message| provider_failure(identity, "cuda.dense.bf16.encode", message))
    }
}

impl CudaTensorDenseBf16Provider {
    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [participant] = invocation.participants() else {
            return Err("CUDA BF16 Dense requires exactly one tensor participant".into());
        };
        if invocation.operation().id.as_str() != DENSE_LINEAR_BF16_OPERATION_ID
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("CUDA BF16 Dense requires its fixed tensor contract".into());
        }
        let shape = geometry(participant.bindings(), participant.attributes())?;
        let input = retained_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
            shape.input_bytes,
        )?;
        let weight = retained_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 1)?,
            shape.weight_bytes,
        )?;
        let output = retained_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            shape.output_bytes,
        )?;
        // Read/read sharing is valid, including disjoint or identical ranges
        // in one allocation. Only the write/read intersections are forbidden.
        disjoint(
            input.device_ptr(),
            shape.input_bytes,
            output.device_ptr(),
            shape.output_bytes,
        )?;
        disjoint(
            weight.device_ptr(),
            shape.weight_bytes,
            output.device_ptr(),
            shape.output_bytes,
        )?;
        let bootstrap = self.bootstrap.clone();
        CudaDeviceCommand::operation_with_blas(
            "vnext_dense_linear_bf16_cublas_no_workspace",
            vec![input, weight, output],
            move |stream, blas, regions| {
                let _bootstrap = &bootstrap;
                launch(
                    stream,
                    blas,
                    &regions[0],
                    regions[1].device_ptr(),
                    &regions[2],
                    shape,
                )
            },
        )
        .and_then(|command| {
            command.with_tensor_work(shape.output_elements, shape.arithmetic_operations, 1, 0)
        })
        .map_err(|error| error.to_string())
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Shape {
    pub(super) rows: i32,
    pub(super) inputs: i32,
    pub(super) outputs: i32,
    pub(super) input_bytes: u64,
    pub(super) weight_bytes: u64,
    pub(super) output_bytes: u64,
    pub(super) output_elements: u64,
    pub(super) arithmetic_operations: u64,
}

impl Shape {
    fn new(rows: u64, inputs: u64, outputs: u64) -> Result<Self, String> {
        let dimension = |value| {
            i32::try_from(value)
                .ok()
                .filter(|&v| v > 0)
                .ok_or_else(|| "CUDA BF16 Dense dimensions must be positive i32".to_owned())
        };
        let rows_i32 = dimension(rows)?;
        let inputs_i32 = dimension(inputs)?;
        let outputs_i32 = dimension(outputs)?;
        let bytes = |a: u64, b: u64| {
            a.checked_mul(b)
                .and_then(|v| v.checked_mul(2))
                .filter(|&v| v <= isize::MAX as u64)
                .ok_or_else(|| "CUDA BF16 Dense byte length exceeds address space".to_owned())
        };
        let output_elements = rows
            .checked_mul(outputs)
            .ok_or("CUDA Dense output count overflows")?;
        let arithmetic_operations = output_elements
            .checked_mul(inputs)
            .and_then(|v| v.checked_mul(2))
            .ok_or("CUDA Dense arithmetic work overflows")?;
        Ok(Self {
            rows: rows_i32,
            inputs: inputs_i32,
            outputs: outputs_i32,
            input_bytes: bytes(rows, inputs)?,
            weight_bytes: bytes(outputs, inputs)?,
            output_bytes: bytes(rows, outputs)?,
            output_elements,
            arithmetic_operations,
        })
    }
}

fn geometry(
    bindings: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Shape, String> {
    if bindings.len() != 3 || attributes.len() != 2 {
        return Err("CUDA BF16 Dense requires exactly three ports and two shape attributes".into());
    }
    let inputs = super::unsigned_attribute(attributes, "in_features")?;
    let outputs = super::unsigned_attribute(attributes, "out_features")?;
    let input = binding(bindings, ResolvedValueRole::Input, 0)?;
    let weight = binding(bindings, ResolvedValueRole::Input, 1)?;
    let output = binding(bindings, ResolvedValueRole::Output, 0)?;
    if weight.weight().is_some_and(|weight| {
        !matches!(weight.physical_layout(), PhysicalWeightLayout::Dense { .. })
    }) {
        return Err("CUDA BF16 Dense cannot consume quantized or packed weight storage".into());
    }
    validate_specs(
        input.tensor(),
        weight.tensor(),
        output.tensor(),
        inputs,
        outputs,
    )
}

pub(super) fn validate_specs(
    input: &ResolvedTensorSpec,
    weight: &ResolvedTensorSpec,
    output: &ResolvedTensorSpec,
    inputs: u64,
    outputs: u64,
) -> Result<Shape, String> {
    let [rows, width] = input.dimensions() else {
        return Err("CUDA BF16 Dense input must be rank two".into());
    };
    if *width != inputs
        || weight.dimensions() != [outputs, inputs]
        || output.dimensions() != [*rows, outputs]
        || [input, weight, output].iter().any(|tensor| {
            tensor.element_type() != ElementType::Bf16
                || tensor.layout() != &ResolvedTensorLayout::Contiguous
        })
    {
        return Err("CUDA BF16 Dense requires contiguous BF16 X[R,K], W[O,K], Y[R,O]".into());
    }
    Shape::new(*rows, inputs, outputs)
}

pub(super) fn retained_region<'a, W: InvocationWorkProjection<'a>>(
    participant: &OperationInvocation<'a, CudaDeviceBuffer, W>,
    binding: &ResolvedValueBinding,
    expected_bytes: u64,
) -> Result<CudaBufferRegion, String> {
    let [component] = binding.storage().components() else {
        return Err("CUDA BF16 Dense requires one physical storage component".into());
    };
    if component.element_type() != ElementType::Bf16 || component.length_bytes() != expected_bytes {
        return Err("CUDA BF16 Dense physical component differs from its tensor".into());
    }
    let view = participant
        .views()
        .iter()
        .find(|view| view.resource_id() == component.resource_id())
        .ok_or("CUDA BF16 Dense lacks its resource view")?;
    let translated = view
        .translate(component.offset_bytes(), expected_bytes)
        .map_err(|e| e.to_string())?;
    let mut physical = translated.iter();
    let physical_region = physical
        .next()
        .ok_or("CUDA BF16 Dense translated to no physical region")?;
    if physical.next().is_some() {
        return Err("CUDA BF16 Dense requires a contiguous device region".into());
    }
    let (buffer, range, retention) = physical_region.buffer_and_physical_range();
    let retained = buffer
        .retained_region(range, retention)
        .map_err(|e| e.to_string())?;
    if retained.element_type() != ElementType::Bf16
        || retained.length_bytes() != expected_bytes
        || retained.device_ptr() % ALIGNMENT != 0
    {
        return Err(
            "CUDA BF16 Dense physical slice differs from its 16-byte-aligned contract".into(),
        );
    }
    Ok(retained)
}

pub(super) fn disjoint(
    read: u64,
    read_bytes: u64,
    write: u64,
    write_bytes: u64,
) -> Result<(), String> {
    let read_end = read
        .checked_add(read_bytes)
        .ok_or("CUDA Dense read address overflows")?;
    let write_end = write
        .checked_add(write_bytes)
        .ok_or("CUDA Dense write address overflows")?;
    if read < write_end && write < read_end {
        return Err("CUDA Dense output overlaps a read-only input".into());
    }
    Ok(())
}

pub(super) fn launch(
    stream: &CudaStream,
    blas: &CudaBlas,
    input: &CudaBufferRegion,
    weight: u64,
    output: &CudaBufferRegion,
    shape: Shape,
) -> Result<(), CudaDeviceRuntimeError> {
    // Lane handles are not shared between streams. Each launch sets every
    // numerical/resource policy explicitly. SetStream resets workspace, so
    // the zero-workspace selection MUST follow it, even on the same stream.
    let handle = *blas.handle();
    unsafe {
        sys::cublasSetStream_v2(handle, stream.cu_stream() as _)
            .result()
            .map_err(|e| CudaDeviceRuntimeError::blas("BF16 Dense stream selection", e))?;
        sys::cublasSetWorkspace_v2(handle, std::ptr::null_mut(), 0)
            .result()
            .map_err(|e| CudaDeviceRuntimeError::blas("BF16 Dense zero workspace", e))?;
        sys::cublasSetPointerMode_v2(handle, sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST)
            .result()
            .map_err(|e| CudaDeviceRuntimeError::blas("BF16 Dense host scalars", e))?;
        // DEFAULT_MATH is zero, so this enum value is precisely DEFAULT | flag.
        sys::cublasSetMathMode(
            handle,
            sys::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
        )
        .result()
        .map_err(|e| CudaDeviceRuntimeError::blas("BF16 Dense FP32 reduction", e))?;
        sys::cublasSetAtomicsMode(handle, sys::cublasAtomicsMode_t::CUBLAS_ATOMICS_NOT_ALLOWED)
            .result()
            .map_err(|e| CudaDeviceRuntimeError::blas("BF16 Dense atomics policy", e))?;
        // Column-major view of row-major Y = X W^T:
        // Y^T[O,R] = W[O,K] * X^T[K,R]. The OI weight is transposed by BLAS.
        gemm_ex(
            handle,
            sys::cublasOperation_t::CUBLAS_OP_T,
            sys::cublasOperation_t::CUBLAS_OP_N,
            shape.outputs,
            shape.rows,
            shape.inputs,
            &ALPHA as *const f32 as *const c_void,
            weight as *const c_void,
            sys::cudaDataType_t::CUDA_R_16BF,
            shape.inputs,
            input.device_ptr() as *const c_void,
            sys::cudaDataType_t::CUDA_R_16BF,
            shape.inputs,
            &BETA as *const f32 as *const c_void,
            output.device_ptr() as *mut c_void,
            sys::cudaDataType_t::CUDA_R_16BF,
            shape.outputs,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    }
    .map_err(|error| CudaDeviceRuntimeError::blas("BF16 Dense zero-workspace GEMM", error))
}

fn invalid_plan(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(shape: &[u64], element_type: ElementType) -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(
            shape.to_vec(),
            element_type,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap()
    }

    #[test]
    fn rectangular_dense_geometry_uses_oi_weights_and_output_element_work() {
        let shape = validate_specs(
            &tensor(&[3, 5], ElementType::Bf16),
            &tensor(&[7, 5], ElementType::Bf16),
            &tensor(&[3, 7], ElementType::Bf16),
            5,
            7,
        )
        .unwrap();
        assert_eq!((shape.rows, shape.inputs, shape.outputs), (3, 5, 7));
        assert_eq!(
            (shape.input_bytes, shape.weight_bytes, shape.output_bytes),
            (30, 70, 42)
        );
        assert_eq!(
            (shape.output_elements, shape.arithmetic_operations),
            (21, 210)
        );
        assert!(validate_specs(
            &tensor(&[3, 5], ElementType::Bf16),
            &tensor(&[5, 7], ElementType::Bf16),
            &tensor(&[3, 7], ElementType::Bf16),
            5,
            7
        )
        .is_err());
        assert!(validate_specs(
            &tensor(&[3, 5], ElementType::F16),
            &tensor(&[7, 5], ElementType::Bf16),
            &tensor(&[3, 7], ElementType::Bf16),
            5,
            7
        )
        .is_err());
        assert!(validate_specs(
            &tensor(&[3, 5], ElementType::Bf16),
            &tensor(&[7, 5], ElementType::Bf16),
            &tensor(&[7, 3], ElementType::Bf16),
            5,
            7
        )
        .is_err());
    }

    #[test]
    fn dimensions_and_work_fail_before_any_native_call() {
        assert!(Shape::new(0, 64, 4096).is_err());
        assert!(Shape::new(1, 0, 4096).is_err());
        assert!(Shape::new(1, 64, 0).is_err());
        assert!(Shape::new(i32::MAX as u64 + 1, 1, 1).is_err());
        assert!(Shape::new(i32::MAX as u64, i32::MAX as u64, i32::MAX as u64).is_err());
        let projection = Shape::new(4, 64, 4096).unwrap();
        assert_eq!(projection.weight_bytes, 512 * 1024);
        assert_eq!(projection.output_bytes, 32 * 1024);
    }

    #[test]
    fn output_alias_and_address_overflow_fail_closed() {
        assert!(disjoint(16, 16, 32, 32).is_ok());
        assert!(disjoint(32, 32, 16, 16).is_ok());
        assert!(disjoint(16, 16, 16, 16).is_err());
        assert!(disjoint(16, 32, 32, 16).is_err());
        assert!(disjoint(u64::MAX - 7, 8, 16, 8).is_err());
        assert!(disjoint(16, 8, u64::MAX - 7, 8).is_err());
    }
}
