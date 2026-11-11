//! Fixed-tensor BF16/F32 GEMM using an explicitly configured, lane-owned cuBLAS
//! handle. Standalone providers allocate no modules. Explicit selected F32
//! affine/patch providers retain a borrowed handle to the already admitted
//! shared module; no matrices, workspace or additional handles are allocated.
//!
//! CUDA 12.8.1 cuBLAS sections 2.4.8 and 2.2.10 specify that workspace size
//! zero prevents use of the default pool, and DISALLOW_REDUCED_PRECISION_REDUCTION
//! keeps mixed-precision reductions in the compute type. An unsupported zero
//! workspace execution fails; it never retries with an implicit workspace.
//! https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html#cublassetworkspace

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::c_void;

use cudarc::cublas::{
    result::{gemm_ex, sgemm},
    sys, CudaBlas,
};
use cudarc::driver::CudaStream;
use ferrum_runtime::{
    affine_linear_f32_contract, dense_linear_bf16_contract, dense_linear_f32_contract,
    patch_projection_f32_contract, AffineLinearF32Geometry, AttributeId,
    BatchedOperationInvocation, CapabilityId, ContractVersion, DeviceCapacityRetention,
    DeviceDescriptor, DeviceRuntime, ElementType, EncodedDeviceOperation, InvocationWorkProjection,
    NodeWorkContract, OperationContract, OperationFailure, OperationInvocation, OperationProvider,
    OperationProviderDescriptor, OperationResourceEstimate, OperationResourceEstimateRequest,
    OperationResourceEstimator, PatchProjectionF32Geometry, PhysicalWeightLayout,
    ProviderExecutionSemantics, ProviderId, ResolvedTensorLayout, ResolvedTensorSpec,
    ResolvedValueBinding, ResolvedValueRole, ReusableExecutionTopology,
    ReusableExecutionTopologyRequest, SemanticValue, TensorOperationInvocation, VNextError,
    WeightFormatId, AFFINE_LINEAR_F32_CAPABILITY_ID, AFFINE_LINEAR_F32_OPERATION_ID,
    DENSE_LINEAR_BF16_CAPABILITY_ID, DENSE_LINEAR_BF16_OPERATION_ID,
    DENSE_LINEAR_F32_CAPABILITY_ID, DENSE_LINEAR_F32_OPERATION_ID,
    PATCH_PROJECTION_F32_CAPABILITY_ID, PATCH_PROJECTION_F32_OPERATION_ID,
};

use super::super::vnext_runtime::{
    CudaBufferRegion, CudaDeviceBuffer, CudaDeviceCommand, CudaDeviceRuntime,
    CudaDeviceRuntimeError,
};
use super::tensor_silu::RetainedF32RowBias;
use super::{binding, contract_error, implementation_fingerprint, provider_failure};

pub const NUMERICAL_PROFILE_ID: &str = "cuda.cublas.gemmex.bf16_f32.no_workspace.v1";
pub const NUMERICAL_PROFILE_IDENTITY: &str = concat!(
    "gemm-ex;A=B=C=CUDA_R_16BF;compute=CUBLAS_COMPUTE_32F;",
    "math=DEFAULT|DISALLOW_REDUCED_PRECISION_REDUCTION;algo=GEMM_DEFAULT;",
    "atomics=NOT_ALLOWED;alpha=host-f32-one;beta=host-f32-zero;workspace=null-0;",
    "OI-row-major;OP_T,OP_N;no-bias;no-replay;operation-defined-error-bound;",
    "no-Reference-reduction-or-bitwise-equivalence"
);
pub const F32_NUMERICAL_PROFILE_ID: &str = "cuda.cublas.sgemm.f32.pedantic.no_workspace.v1";
pub const F32_NUMERICAL_PROFILE_IDENTITY: &str = concat!(
    "sgemm;A=B=C=f32;math=PEDANTIC;no-TF32-or-16-bit-operands;",
    "atomics=NOT_ALLOWED;alpha=host-f32-one;beta=host-f32-zero;workspace=null-0;",
    "OI-row-major;OP_T,OP_N;no-bias;no-replay;operation-defined-error-bound;",
    "no-Reference-reduction-or-bitwise-equivalence"
);
pub const F32_AFFINE_NUMERICAL_PROFILE_ID: &str =
    "cuda.cublas.sgemm_saxpy.f32.pedantic.no_workspace.v1";
pub const F32_AFFINE_NUMERICAL_PROFILE_IDENTITY: &str = concat!(
    "sgemm-f32-dot-store;then-saxpy-f32-alpha-one-per-row;F32-bias;math=PEDANTIC;",
    "no-TF32-or-16-bit-operands;atomics=NOT_ALLOWED;host-f32-scalars;workspace=null-0;",
    "OI-row-major;no-replay;no-Reference-or-Torch-bitwise-equivalence"
);
pub const F32_PATCH_NUMERICAL_PROFILE_ID: &str =
    "cuda.cublas.patch_projection.f32.original_rank5.sgemm_saxpy.pedantic.v1";
pub const F32_AFFINE_ROW_BIAS_NUMERICAL_PROFILE_ID: &str =
    "cuda.cublas.sgemm_then_row_bias.f32.pedantic.rn.gradual.v1";
pub const F32_PATCH_ROW_BIAS_NUMERICAL_PROFILE_ID: &str =
    "cuda.cublas.patch_projection.f32.original_rank5.sgemm_then_row_bias.rn.gradual.v1";
pub const F32_ROW_BIAS_NUMERICAL_PROFILE_IDENTITY: &str = concat!(
    "sgemm-f32-dot-store;then-one-row-major-broadcast-__fadd_rn;F32-bias;",
    "bias-ftz=false;NaN-class-only;math=PEDANTIC;no-TF32-or-16-bit-operands;",
    "atomics=NOT_ALLOWED;host-f32-scalars;workspace=null-0;OI-row-major;",
    "one-original-admitted-module;no-replay;no-Reference-or-Torch-bitwise-equivalence"
);

#[derive(Clone, Copy)]
pub(super) enum RowBiasProjection {
    Affine,
    Patch,
}
impl RowBiasProjection {
    fn scalar(self) -> Scalar {
        match self {
            Self::Affine => Scalar::F32Affine,
            Self::Patch => Scalar::F32Patch,
        }
    }
}

#[derive(Clone)]
enum ProviderRetention {
    LaneBootstrap(DeviceCapacityRetention),
    // The original module permit retains TensorEagerState and its bootstrap
    // allowance. Cloning this handle does not grant a second module/BLAS lane.
    SharedRowBias(RetainedF32RowBias),
}

const PROVIDER_ID: &str = "provider.cuda.dense_linear.bf16.cublas.no_workspace";
const ESTIMATOR_ID: &str = "resource-estimator.cuda.dense_linear.bf16.cublas.no_workspace";
const ALIGNMENT: u64 = 16;
const ALPHA: f32 = 1.0;
const BETA: f32 = 0.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scalar {
    Bf16,
    F32,
    F32Affine,
    F32Patch,
}
impl Scalar {
    fn element_type(self) -> ElementType {
        match self {
            Self::Bf16 => ElementType::Bf16,
            Self::F32 | Self::F32Affine | Self::F32Patch => ElementType::F32,
        }
    }
    fn operation(self) -> &'static str {
        match self {
            Self::Bf16 => DENSE_LINEAR_BF16_OPERATION_ID,
            Self::F32 => DENSE_LINEAR_F32_OPERATION_ID,
            Self::F32Affine => AFFINE_LINEAR_F32_OPERATION_ID,
            Self::F32Patch => PATCH_PROJECTION_F32_OPERATION_ID,
        }
    }
    fn has_bias(self) -> bool {
        matches!(self, Self::F32Affine | Self::F32Patch)
    }
}

pub(super) fn source_fingerprint() -> String {
    implementation_fingerprint(&[
        NUMERICAL_PROFILE_ID.as_bytes(),
        NUMERICAL_PROFILE_IDENTITY.as_bytes(),
        F32_NUMERICAL_PROFILE_ID.as_bytes(),
        F32_NUMERICAL_PROFILE_IDENTITY.as_bytes(),
        F32_AFFINE_NUMERICAL_PROFILE_ID.as_bytes(),
        F32_AFFINE_NUMERICAL_PROFILE_IDENTITY.as_bytes(),
        F32_PATCH_NUMERICAL_PROFILE_ID.as_bytes(),
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
    retention: ProviderRetention,
    scalar: Scalar,
}

impl CudaTensorDenseBf16Provider {
    pub(super) fn new(runtime: &CudaDeviceRuntime) -> Result<Self, CudaDeviceRuntimeError> {
        Self::new_for(runtime, Scalar::Bf16)
    }

    pub(super) fn new_f32(runtime: &CudaDeviceRuntime) -> Result<Self, CudaDeviceRuntimeError> {
        Self::new_for(runtime, Scalar::F32)
    }
    pub(super) fn new_affine_f32(
        runtime: &CudaDeviceRuntime,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::new_for(runtime, Scalar::F32Affine)
    }
    pub(super) fn new_patch_f32(
        runtime: &CudaDeviceRuntime,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::new_for(runtime, Scalar::F32Patch)
    }

    fn new_for(
        runtime: &CudaDeviceRuntime,
        scalar: Scalar,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let bootstrap = runtime.tensor_bootstrap_retention()?;
        Ok(Self {
            descriptor: descriptor_for(runtime.descriptor(), scalar, false)?,
            retention: ProviderRetention::LaneBootstrap(bootstrap),
            scalar,
        })
    }
}

/// Selection metadata is prepared before acquiring the shared native module.
/// The null-handle cuBLAS query is software identity, not device acquisition.
pub(super) fn row_bias_descriptor(
    device: &DeviceDescriptor,
    kind: RowBiasProjection,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    descriptor_for(device, kind.scalar(), true)
}

pub(super) fn row_bias_provider(
    descriptor: OperationProviderDescriptor,
    bias: RetainedF32RowBias,
    kind: RowBiasProjection,
) -> Box<dyn OperationProvider<CudaDeviceRuntime>> {
    Box::new(CudaTensorDenseBf16Provider {
        descriptor,
        retention: ProviderRetention::SharedRowBias(bias),
        scalar: kind.scalar(),
    })
}

fn descriptor_for(
    device: &DeviceDescriptor,
    scalar: Scalar,
    shared_row_bias: bool,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let (contract, capability, provider_id, estimator_id) = match scalar {
        Scalar::Bf16 => (
            dense_linear_bf16_contract(),
            DENSE_LINEAR_BF16_CAPABILITY_ID,
            PROVIDER_ID,
            ESTIMATOR_ID,
        ),
        Scalar::F32 => (
            dense_linear_f32_contract(),
            DENSE_LINEAR_F32_CAPABILITY_ID,
            "provider.cuda.dense_linear.f32.cublas.pedantic.no_workspace",
            "resource-estimator.cuda.dense_linear.f32.cublas.pedantic.no_workspace",
        ),
        Scalar::F32Affine => (
            affine_linear_f32_contract(),
            AFFINE_LINEAR_F32_CAPABILITY_ID,
            "provider.cuda.affine_linear.f32.cublas.pedantic.no_workspace",
            "resource-estimator.cuda.affine_linear.f32.cublas.pedantic.no_workspace",
        ),
        Scalar::F32Patch => (
            patch_projection_f32_contract(),
            PATCH_PROJECTION_F32_CAPABILITY_ID,
            "provider.cuda.patch_projection.f32.cublas.pedantic.no_workspace",
            "resource-estimator.cuda.patch_projection.f32.cublas.pedantic.no_workspace",
        ),
    };
    let (provider_id, estimator_id) = if shared_row_bias {
        match scalar {
            Scalar::F32Affine => (
                "provider.cuda.affine_linear.f32.cublas.pedantic.row_bias",
                "resource-estimator.cuda.affine_linear.f32.cublas.pedantic.row_bias",
            ),
            Scalar::F32Patch => (
                "provider.cuda.patch_projection.f32.cublas.pedantic.row_bias",
                "resource-estimator.cuda.patch_projection.f32.cublas.pedantic.row_bias",
            ),
            _ => {
                return Err(CudaDeviceRuntimeError::contract(
                    "row bias requires affine or patch",
                ))
            }
        }
    } else {
        (provider_id, estimator_id)
    };
    let capability = CapabilityId::new(capability).map_err(contract_error)?;
    if !device.capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA tensor runtime lacks the selected Dense capability",
        ));
    }
    let contract = contract.map_err(contract_error)?;
    // NVIDIA permits GetVersion with a null handle. This reads the loaded
    // library identity without creating an unbudgeted second handle.
    let version = cublas_version()?;
    let base_implementation = implementation_fingerprint(&[
        source_fingerprint().as_bytes(),
        format!("cublas.version={version}").as_bytes(),
        provider_id.as_bytes(),
    ]);
    let implementation = if shared_row_bias {
        implementation_fingerprint(&[
            base_implementation.as_bytes(),
            match scalar {
                Scalar::F32Affine => F32_AFFINE_ROW_BIAS_NUMERICAL_PROFILE_ID,
                Scalar::F32Patch => F32_PATCH_ROW_BIAS_NUMERICAL_PROFILE_ID,
                _ => unreachable!("checked projection"),
            }
            .as_bytes(),
            F32_ROW_BIAS_NUMERICAL_PROFILE_IDENTITY.as_bytes(),
            RetainedF32RowBias::fingerprint().as_bytes(),
        ])
    } else {
        base_implementation
    };
    let estimator =
        implementation_fingerprint(&[estimator_id.as_bytes(), implementation.as_bytes()]);
    let mut weight_formats = BTreeSet::from([
        WeightFormatId::new(super::DENSE_SAFETENSORS_FORMAT_ID).map_err(contract_error)?,
        WeightFormatId::new("weight-format.gguf.native-block").map_err(contract_error)?,
    ]);
    // Canonical mixed archives still supply this provider exact dense F32
    // components. Quantized matrices remain exclusive to their raw ops.
    if matches!(scalar, Scalar::F32Affine) {
        weight_formats.insert(
            WeightFormatId::new("weight-format.comfy.convrot.canonical.v1")
                .map_err(contract_error)?,
        );
    }
    if scalar == Scalar::F32Patch {
        weight_formats.retain(|f| f.as_str() == super::DENSE_SAFETENSORS_FORMAT_ID);
    }
    let descriptor = OperationProviderDescriptor::new(
        ProviderId::new(provider_id).map_err(contract_error)?,
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
        weight_formats,
        BTreeSet::new(),
        super::transformer::contiguous_bindings(if scalar.has_bias() { 3 } else { 2 }),
        estimator_id,
        ContractVersion::new(1, 0),
        estimator,
    )
    .map_err(contract_error)?
    .with_fixed_tensor_work(true);
    Ok(descriptor)
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
                "CUDA tensor Dense estimator received another operation",
            ));
        }
        geometry(request.values(), request.attributes(), self.scalar).map_err(invalid_plan)?;
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
            "cuda.tensor_dense.encode",
            "CUDA tensor Dense supports fixed tensors only".into(),
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
            .map_err(|message| provider_failure(identity, "cuda.tensor_dense.encode", message))
    }
}

impl CudaTensorDenseBf16Provider {
    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [participant] = invocation.participants() else {
            return Err("CUDA tensor Dense requires exactly one tensor participant".into());
        };
        if invocation.operation().id.as_str() != self.scalar.operation()
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("CUDA tensor Dense requires its fixed tensor contract".into());
        }
        let shape = geometry(
            participant.bindings(),
            participant.attributes(),
            self.scalar,
        )?;
        let input = retained_region_for(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
            shape.input_bytes,
            self.scalar,
        )?;
        let weight = retained_region_for(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 1)?,
            shape.weight_bytes,
            self.scalar,
        )?;
        let output = retained_region_for(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            shape.output_bytes,
            self.scalar,
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
        let bias = if self.scalar.has_bias() {
            let bias = retained_region_for(
                participant,
                binding(participant.bindings(), ResolvedValueRole::Input, 2)?,
                shape.outputs as u64 * 4,
                self.scalar,
            )?;
            disjoint(
                bias.device_ptr(),
                bias.length_bytes(),
                output.device_ptr(),
                shape.output_bytes,
            )?;
            Some(bias)
        } else {
            None
        };
        let mut regions = vec![input, weight, output];
        if let Some(bias) = bias {
            regions.push(bias);
        }
        self.command(shape, regions)
    }

    fn command(
        &self,
        shape: Shape,
        regions: Vec<CudaBufferRegion>,
    ) -> Result<CudaDeviceCommand, String> {
        let retention = self.retention.clone();
        let row_bias = matches!(&retention, ProviderRetention::SharedRowBias(_));
        let scalar = self.scalar;
        if let ProviderRetention::SharedRowBias(bias) = &retention {
            bias.validate_regions(&regions)?;
        }
        CudaDeviceCommand::operation_with_blas(
            match scalar {
                Scalar::Bf16 => "vnext_dense_linear_bf16_cublas_no_workspace",
                Scalar::F32 => "vnext_dense_linear_f32_cublas_pedantic_no_workspace",
                Scalar::F32Affine if row_bias => "vnext_affine_linear_f32_cublas_pedantic_row_bias",
                Scalar::F32Affine => "vnext_affine_linear_f32_cublas_pedantic_no_workspace",
                Scalar::F32Patch if row_bias => {
                    "vnext_patch_projection_f32_cublas_pedantic_row_bias"
                }
                Scalar::F32Patch => "vnext_patch_projection_f32_cublas_pedantic_no_workspace",
            },
            regions,
            move |stream, blas, regions| {
                // Keep the original capacity/module authority through the
                // command, submission fence and uncertain-completion custody.
                let _retention = &retention;
                launch_for(
                    stream,
                    blas,
                    &regions[0],
                    regions[1].device_ptr(),
                    &regions[2],
                    shape,
                    scalar,
                )?;
                if scalar.has_bias() {
                    match &retention {
                        ProviderRetention::SharedRowBias(bias) => bias.launch(
                            stream,
                            &regions[3],
                            &regions[2],
                            shape.rows as u64,
                            shape.outputs as u64,
                        )?,
                        ProviderRetention::LaneBootstrap(_bootstrap) => {
                            launch_bias(blas, &regions[3], &regions[2], shape)?;
                        }
                    }
                }
                Ok(())
            },
        )
        .and_then(|command| {
            command.with_tensor_work(
                shape.output_elements,
                shape.arithmetic_operations,
                if row_bias {
                    2
                } else if scalar.has_bias() {
                    shape.rows as u64 + 1
                } else {
                    1
                },
                0,
            )
        })
        .map_err(|error| error.to_string())
    }
}

/// Test-only backend custody fixture. Uses the same provider and command
/// encoder as production; the caller supplies actual separately charged ranges.
#[cfg(test)]
pub(super) fn row_bias_retention_fixture(
    runtime: &CudaDeviceRuntime,
    bias: RetainedF32RowBias,
    regions: Vec<CudaBufferRegion>,
) -> Result<
    (
        Box<dyn OperationProvider<CudaDeviceRuntime>>,
        CudaDeviceCommand,
    ),
    String,
> {
    let provider = CudaTensorDenseBf16Provider {
        descriptor: row_bias_descriptor(runtime.descriptor(), RowBiasProjection::Affine)
            .map_err(|e| e.to_string())?,
        retention: ProviderRetention::SharedRowBias(bias),
        scalar: Scalar::F32Affine,
    };
    let shape = Shape::new_for(1, 1, 1, Scalar::F32Affine)?;
    let command = provider.command(shape, regions)?;
    Ok((Box::new(provider), command))
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
        Self::new_for(rows, inputs, outputs, Scalar::Bf16)
    }
    fn new_for(rows: u64, inputs: u64, outputs: u64, scalar: Scalar) -> Result<Self, String> {
        let dimension = |value| {
            i32::try_from(value)
                .ok()
                .filter(|&v| v > 0)
                .ok_or_else(|| "CUDA tensor Dense dimensions must be positive i32".to_owned())
        };
        let rows_i32 = dimension(rows)?;
        let inputs_i32 = dimension(inputs)?;
        let outputs_i32 = dimension(outputs)?;
        let bytes = |a: u64, b: u64| {
            a.checked_mul(b)
                .and_then(|v| v.checked_mul(scalar.element_type().size_bytes()))
                .filter(|&v| v <= isize::MAX as u64)
                .ok_or_else(|| "CUDA tensor Dense byte length exceeds address space".to_owned())
        };
        let output_elements = rows
            .checked_mul(outputs)
            .ok_or("CUDA Dense output count overflows")?;
        let arithmetic_operations = output_elements
            .checked_mul(inputs)
            .and_then(|v| v.checked_mul(2))
            .and_then(|value| {
                if scalar.has_bias() {
                    value.checked_add(output_elements)
                } else {
                    Some(value)
                }
            })
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
    scalar: Scalar,
) -> Result<Shape, String> {
    if bindings.len() != (if scalar.has_bias() { 4 } else { 3 })
        || attributes.len() != if scalar == Scalar::F32Patch { 0 } else { 2 }
    {
        return Err(
            "CUDA tensor projection requires its exact ports and two shape attributes".into(),
        );
    }
    let input = binding(bindings, ResolvedValueRole::Input, 0)?;
    let weight = binding(bindings, ResolvedValueRole::Input, 1)?;
    let output = binding(bindings, ResolvedValueRole::Output, 0)?;
    if weight.weight().is_some_and(|weight| {
        !matches!(weight.physical_layout(), PhysicalWeightLayout::Dense { .. })
    }) {
        return Err("CUDA tensor Dense cannot consume quantized or packed weight storage".into());
    }
    if scalar == Scalar::F32Patch {
        let bias = binding(bindings, ResolvedValueRole::Input, 2)?;
        if bias
            .weight()
            .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        {
            return Err("CUDA patch bias must retain dense physical storage".into());
        }
        let g = PatchProjectionF32Geometry::new(
            input.tensor(),
            weight.tensor(),
            bias.tensor(),
            output.tensor(),
        )
        .map_err(|e| e.to_string())?
        .affine();
        return Shape::new_for(g.rows(), g.inputs(), g.outputs(), scalar);
    }
    let inputs = super::unsigned_attribute(attributes, "in_features")?;
    let outputs = super::unsigned_attribute(attributes, "out_features")?;
    if scalar.has_bias() {
        let bias = binding(bindings, ResolvedValueRole::Input, 2)?;
        if bias.weight().is_some_and(|weight| {
            !matches!(weight.physical_layout(), PhysicalWeightLayout::Dense { .. })
        }) {
            return Err("CUDA F32 affine requires exact dense F32 bias storage".into());
        }
        AffineLinearF32Geometry::new(
            input.tensor(),
            weight.tensor(),
            bias.tensor(),
            output.tensor(),
            inputs,
            outputs,
        )
        .map_err(|error| error.to_string())?;
    }
    validate_specs_for(
        input.tensor(),
        weight.tensor(),
        output.tensor(),
        inputs,
        outputs,
        scalar,
    )
}

pub(super) fn validate_specs(
    input: &ResolvedTensorSpec,
    weight: &ResolvedTensorSpec,
    output: &ResolvedTensorSpec,
    inputs: u64,
    outputs: u64,
) -> Result<Shape, String> {
    validate_specs_for(input, weight, output, inputs, outputs, Scalar::Bf16)
}

fn validate_specs_for(
    input: &ResolvedTensorSpec,
    weight: &ResolvedTensorSpec,
    output: &ResolvedTensorSpec,
    inputs: u64,
    outputs: u64,
    scalar: Scalar,
) -> Result<Shape, String> {
    let [rows, width] = input.dimensions() else {
        return Err("CUDA tensor Dense input must be rank two".into());
    };
    if *width != inputs
        || weight.dimensions() != [outputs, inputs]
        || output.dimensions() != [*rows, outputs]
        || [input, weight, output].iter().any(|tensor| {
            tensor.element_type() != scalar.element_type()
                || tensor.layout() != &ResolvedTensorLayout::Contiguous
        })
    {
        return Err(
            "CUDA tensor Dense requires its declared contiguous scalar X[R,K], W[O,K], Y[R,O]"
                .into(),
        );
    }
    Shape::new_for(*rows, inputs, outputs, scalar)
}

pub(super) fn retained_region<'a, W: InvocationWorkProjection<'a>>(
    participant: &OperationInvocation<'a, CudaDeviceBuffer, W>,
    binding: &ResolvedValueBinding,
    expected_bytes: u64,
) -> Result<CudaBufferRegion, String> {
    retained_region_for(participant, binding, expected_bytes, Scalar::Bf16)
}

fn retained_region_for<'a, W: InvocationWorkProjection<'a>>(
    participant: &OperationInvocation<'a, CudaDeviceBuffer, W>,
    binding: &ResolvedValueBinding,
    expected_bytes: u64,
    scalar: Scalar,
) -> Result<CudaBufferRegion, String> {
    let [component] = binding.storage().components() else {
        return Err("CUDA tensor Dense requires one physical storage component".into());
    };
    if component.element_type() != scalar.element_type()
        || component.length_bytes() != expected_bytes
    {
        return Err("CUDA tensor Dense physical component differs from its tensor".into());
    }
    let view = participant
        .views()
        .iter()
        .find(|view| view.resource_id() == component.resource_id())
        .ok_or("CUDA tensor Dense lacks its resource view")?;
    let translated = view
        .translate(component.offset_bytes(), expected_bytes)
        .map_err(|e| e.to_string())?;
    let mut physical = translated.iter();
    let physical_region = physical
        .next()
        .ok_or("CUDA tensor Dense translated to no physical region")?;
    if physical.next().is_some() {
        return Err("CUDA tensor Dense requires a contiguous device region".into());
    }
    let (buffer, range, retention) = physical_region.buffer_and_physical_range();
    let retained = buffer
        .retained_region(range, retention)
        .map_err(|e| e.to_string())?;
    if retained.element_type() != scalar.element_type()
        || retained.length_bytes() != expected_bytes
        || retained.device_ptr() % ALIGNMENT != 0
    {
        return Err(
            "CUDA tensor Dense physical slice differs from its 16-byte-aligned contract".into(),
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
    launch_for(stream, blas, input, weight, output, shape, Scalar::Bf16)
}

fn launch_for(
    stream: &CudaStream,
    blas: &CudaBlas,
    input: &CudaBufferRegion,
    weight: u64,
    output: &CudaBufferRegion,
    shape: Shape,
    scalar: Scalar,
) -> Result<(), CudaDeviceRuntimeError> {
    // Lane handles are not shared between streams. Each launch sets every
    // numerical/resource policy explicitly. SetStream resets workspace, so
    // the zero-workspace selection MUST follow it, even on the same stream.
    let handle = *blas.handle();
    unsafe {
        sys::cublasSetStream_v2(handle, stream.cu_stream() as _)
            .result()
            .map_err(|e| CudaDeviceRuntimeError::blas("Tensor Dense stream selection", e))?;
        sys::cublasSetWorkspace_v2(handle, std::ptr::null_mut(), 0)
            .result()
            .map_err(|e| CudaDeviceRuntimeError::blas("Tensor Dense zero workspace", e))?;
        sys::cublasSetPointerMode_v2(handle, sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST)
            .result()
            .map_err(|e| CudaDeviceRuntimeError::blas("Tensor Dense host scalars", e))?;
        // DEFAULT_MATH is zero, so this enum value is precisely DEFAULT | flag.
        sys::cublasSetMathMode(
            handle,
            match scalar {
                Scalar::Bf16 => sys::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
                Scalar::F32 | Scalar::F32Affine | Scalar::F32Patch => {
                    sys::cublasMath_t::CUBLAS_PEDANTIC_MATH
                }
            },
        )
        .result()
        .map_err(|e| CudaDeviceRuntimeError::blas("Tensor Dense FP32 reduction", e))?;
        sys::cublasSetAtomicsMode(handle, sys::cublasAtomicsMode_t::CUBLAS_ATOMICS_NOT_ALLOWED)
            .result()
            .map_err(|e| CudaDeviceRuntimeError::blas("Tensor Dense atomics policy", e))?;
        // Column-major view of row-major Y = X W^T:
        // Y^T[O,R] = W[O,K] * X^T[K,R]. The OI weight is transposed by BLAS.
        if scalar != Scalar::Bf16 {
            // PEDANTIC_MATH on SGEMM uses the prescribed single precision;
            // it does not enable TF32 or 16-bit tensor-core substitution.
            sgemm(
                handle,
                sys::cublasOperation_t::CUBLAS_OP_T,
                sys::cublasOperation_t::CUBLAS_OP_N,
                shape.outputs,
                shape.rows,
                shape.inputs,
                &ALPHA,
                weight as *const f32,
                shape.inputs,
                input.device_ptr() as *const f32,
                shape.inputs,
                &BETA,
                output.device_ptr() as *mut f32,
                shape.outputs,
            )
        } else {
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
    }
    .map_err(|error| CudaDeviceRuntimeError::blas("Tensor Dense zero-workspace GEMM", error))
}

fn invalid_plan(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

/// The preceding SGEMM configures this same lane handle, including its stream,
/// host scalar pointer mode, pedantic arithmetic and zero workspace. With
/// alpha=1, SAXPY's product is exact and the only new rounding is F32(dot+bias).
/// Each row is an explicit native dispatch; no module or broadcast buffer exists.
fn launch_bias(
    blas: &CudaBlas,
    bias: &CudaBufferRegion,
    output: &CudaBufferRegion,
    shape: Shape,
) -> Result<(), CudaDeviceRuntimeError> {
    for row in 0..shape.rows as u64 {
        let destination = output.device_ptr() + row * shape.outputs as u64 * 4;
        unsafe {
            sys::cublasSaxpy_v2(
                *blas.handle(),
                shape.outputs,
                &ALPHA,
                bias.device_ptr() as *const f32,
                1,
                destination as *mut f32,
                1,
            )
        }
        .result()
        .map_err(|error| CudaDeviceRuntimeError::blas("F32 affine row bias SAXPY", error))?;
    }
    Ok(())
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
    fn f32_geometry_uses_four_byte_operands_and_rejects_downcast_ports() {
        let shape = validate_specs_for(
            &tensor(&[3, 5], ElementType::F32),
            &tensor(&[7, 5], ElementType::F32),
            &tensor(&[3, 7], ElementType::F32),
            5,
            7,
            Scalar::F32,
        )
        .unwrap();
        assert_eq!((shape.rows, shape.inputs, shape.outputs), (3, 5, 7));
        assert_eq!(
            (shape.input_bytes, shape.weight_bytes, shape.output_bytes),
            (60, 140, 84)
        );
        assert!(validate_specs_for(
            &tensor(&[3, 5], ElementType::Bf16),
            &tensor(&[7, 5], ElementType::F32),
            &tensor(&[3, 7], ElementType::F32),
            5,
            7,
            Scalar::F32,
        )
        .is_err());
        assert!(validate_specs_for(
            &tensor(&[3, 5], ElementType::F32),
            &tensor(&[5, 7], ElementType::F32),
            &tensor(&[3, 7], ElementType::F32),
            5,
            7,
            Scalar::F32,
        )
        .is_err());
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
