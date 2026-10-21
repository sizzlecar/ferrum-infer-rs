//! Eager, weightless, fixed-tensor CUDA BF16 SiLU.

use std::collections::{BTreeMap, BTreeSet};
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::Arc;

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ferrum_runtime::{
    silu_bf16_contract, AttributeId, BatchedOperationInvocation, Bf16SameShapeGeometry,
    CapabilityId, ContractVersion, DeviceRuntime, ElementType, EncodedDeviceOperation,
    NodeWorkContract, OperationContract, OperationFailure, OperationProvider,
    OperationProviderDescriptor, OperationResourceEstimate, OperationResourceEstimateRequest,
    OperationResourceEstimator, ProviderExecutionSemantics, ProviderId, ResolvedValueBinding,
    ResolvedValueRole, ReusableExecutionTopology, ReusableExecutionTopologyRequest, SemanticValue,
    TensorOperationInvocation, VNextError, SILU_BF16_CAPABILITY_ID, SILU_BF16_OPERATION_ID,
};

use super::super::vnext_runtime::{
    CudaBufferRegion, CudaDeviceBuffer, CudaDeviceCommand, CudaDeviceRuntime,
    CudaDeviceRuntimeError, CudaTensorProviderRetention,
};
use super::{binding, contract_error, implementation_fingerprint, provider_failure};

pub const NUMERICAL_PROFILE_ID: &str = "cuda.libdevice.precise.silu.bf16.v1";
pub const NUMERICAL_PROFILE_IDENTITY: &str =
    "libdevice-expf;fadd-rn;fdiv-rn;integer-bf16-rne;ftz=false;prec-div=true;prec-sqrt=true;fmad=false;no-fast-math;nan-class-only";
const FUNCTION: &str = "ferrum_vnext_silu_bf16";
const PROVIDER: &str = "provider.cuda.silu.bf16.libdevice";
const ESTIMATOR: &str = "resource-estimator.cuda.silu.bf16.libdevice";
const THREADS: u32 = 256;
const MAX_BLOCKS: u64 = 65_535;

pub(super) fn numerical_fingerprint() -> String {
    implementation_fingerprint(&[
        NUMERICAL_PROFILE_ID.as_bytes(),
        NUMERICAL_PROFILE_IDENTITY.as_bytes(),
        include_bytes!("tensor_silu.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/vnext_bf16_silu.cu"
        )),
        include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/build.rs")),
        crate::ptx::VNEXT_BF16_SILU.as_bytes(),
    ])
}

pub(super) struct CudaTensorSiluProvider {
    descriptor: OperationProviderDescriptor,
    kernel: Arc<RetainedFunction>,
}

struct RetainedFunction {
    // Field order drops the CUDA function/module before its budget claim.
    function: CudaFunction,
    // The function owns its module/context; its budget authority must live at
    // least as long, including encoded commands that outlive this provider.
    _provider: CudaTensorProviderRetention,
}

impl CudaTensorSiluProvider {
    pub(super) fn new(runtime: &CudaDeviceRuntime) -> Result<Self, CudaDeviceRuntimeError> {
        let capability = CapabilityId::new(SILU_BF16_CAPABILITY_ID).map_err(contract_error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA runtime lacks BF16 SiLU capability",
            ));
        }
        let contract = silu_bf16_contract().map_err(contract_error)?;
        let implementation = numerical_fingerprint();
        let estimator =
            implementation_fingerprint(&[ESTIMATOR.as_bytes(), implementation.as_bytes()]);
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new(PROVIDER).map_err(contract_error)?,
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
            BTreeSet::new(),
            BTreeSet::new(),
            super::transformer::contiguous_bindings(1),
            ESTIMATOR,
            ContractVersion::new(1, 0),
            estimator,
        )
        .map_err(contract_error)?
        .with_fixed_tensor_work(true);
        // One native provider bundle may be live per TensorEager runtime.
        // A failed/partial native load cannot certify cleanup, so it retains
        // the permit and bootstrap allowance instead of enabling another load.
        let provider = runtime.reserve_tensor_provider()?;
        let loaded = catch_unwind(AssertUnwindSafe(|| {
            let module = runtime
                .context()
                .load_module(Ptx::from_src(crate::ptx::VNEXT_BF16_SILU.to_owned()))
                .map_err(|error| CudaDeviceRuntimeError::driver("BF16 SiLU module load", error))?;
            module
                .load_function(FUNCTION)
                .map_err(|error| CudaDeviceRuntimeError::driver("BF16 SiLU function load", error))
        }));
        let function = match loaded {
            Ok(Ok(function)) => function,
            Ok(Err(error)) => {
                std::mem::forget(provider);
                return Err(error);
            }
            Err(payload) => {
                std::mem::forget(provider);
                resume_unwind(payload);
            }
        };
        Ok(Self {
            descriptor,
            kernel: Arc::new(RetainedFunction {
                function,
                _provider: provider,
            }),
        })
    }
}

impl OperationResourceEstimator for CudaTensorSiluProvider {
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
                "CUDA SiLU estimator received another operation",
            ));
        }
        geometry(request.values(), request.attributes()).map_err(invalid_plan)?;
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            16,
            None,
            None,
        ))
    }
}

impl OperationProvider<CudaDeviceRuntime> for CudaTensorSiluProvider {
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
            "cuda.silu.bf16.encode",
            "CUDA BF16 SiLU supports fixed tensors only".into(),
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
            .map_err(|message| provider_failure(identity, "cuda.silu.bf16.encode", message))
    }
}

impl CudaTensorSiluProvider {
    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [participant] = invocation.participants() else {
            return Err("CUDA BF16 SiLU requires one tensor participant".into());
        };
        if invocation.operation().id.as_str() != SILU_BF16_OPERATION_ID
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("CUDA BF16 SiLU requires its fixed tensor contract".into());
        }
        let elements = geometry(participant.bindings(), participant.attributes())?;
        let byte_len = elements
            .checked_mul(2)
            .ok_or("CUDA SiLU byte length overflows")?;
        let region = |role| -> Result<CudaBufferRegion, String> {
            let binding = binding(participant.bindings(), role, 0)?;
            let [component] = binding.storage().components() else {
                return Err("CUDA SiLU requires one storage component per value".into());
            };
            if component.element_type() != ElementType::Bf16 || component.length_bytes() != byte_len
            {
                return Err("CUDA SiLU storage differs from its BF16 tensor".into());
            }
            let view = participant
                .views()
                .iter()
                .find(|view| view.resource_id() == component.resource_id())
                .ok_or("CUDA SiLU value has no resource view")?;
            let translated = view
                .translate(component.offset_bytes(), byte_len)
                .map_err(|error| error.to_string())?;
            let mut physical = translated.iter();
            let physical_region = physical
                .next()
                .ok_or("CUDA SiLU translated to no physical region")?;
            if physical.next().is_some() {
                return Err("CUDA SiLU requires contiguous physical storage".into());
            }
            let (buffer, range, retention) = physical_region.buffer_and_physical_range();
            let region = buffer
                .retained_region(range, retention)
                .map_err(|error| error.to_string())?;
            if region.element_type() != ElementType::Bf16
                || region.length_bytes() != byte_len
                || region.device_ptr() % 2 != 0
            {
                return Err(
                    "CUDA SiLU physical region is not the expected aligned BF16 slice".into(),
                );
            }
            Ok(region)
        };
        let input = region(ResolvedValueRole::Input)?;
        let output = region(ResolvedValueRole::Output)?;
        nonoverlapping_ranges(input.device_ptr(), output.device_ptr(), byte_len)?;
        let kernel = Arc::clone(&self.kernel);
        let blocks = elements.div_ceil(u64::from(THREADS)).min(MAX_BLOCKS) as u32;
        CudaDeviceCommand::operation(FUNCTION, vec![input, output], move |stream, regions| {
            let input = regions[0].device_ptr();
            let output = regions[1].device_ptr();
            let mut builder = stream.launch_builder(&kernel.function);
            builder.arg(&input).arg(&output).arg(&elements);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map(|_| ())
            .map_err(|error| CudaDeviceRuntimeError::driver("BF16 SiLU launch", error))
        })
        .and_then(|command| command.with_tensor_work(elements, elements * 4, 1, 0))
        .map_err(|error| error.to_string())
    }
}

fn geometry(
    bindings: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<u64, String> {
    if bindings.len() != 2 || !attributes.is_empty() {
        return Err("CUDA BF16 SiLU requires exactly two ports and no attributes".into());
    }
    let geometry = Bf16SameShapeGeometry::new(
        binding(bindings, ResolvedValueRole::Input, 0)?.tensor(),
        binding(bindings, ResolvedValueRole::Output, 0)?.tensor(),
    )
    .map_err(|error| error.to_string())?;
    let elements = geometry
        .rows()
        .checked_mul(geometry.width())
        .ok_or("CUDA SiLU element count overflows")?;
    elements
        .checked_mul(4)
        .ok_or("CUDA SiLU logical arithmetic work overflows")?;
    let bytes = elements
        .checked_mul(2)
        .ok_or("CUDA SiLU byte length overflows")?;
    if bytes > isize::MAX as u64 {
        return Err("CUDA SiLU tensor exceeds the host address space".into());
    }
    Ok(elements)
}

fn nonoverlapping_ranges(input: u64, output: u64, length: u64) -> Result<(), String> {
    let input_end = input
        .checked_add(length)
        .ok_or("CUDA SiLU input address overflows")?;
    let output_end = output
        .checked_add(length)
        .ok_or("CUDA SiLU output address overflows")?;
    if input < output_end && output < input_end {
        return Err("CUDA SiLU output overlaps its read-only input".into());
    }
    Ok(())
}

fn invalid_plan(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::nonoverlapping_ranges;

    #[test]
    fn physical_alias_check_accepts_adjacent_regions_and_rejects_overlap_and_overflow() {
        assert!(nonoverlapping_ranges(16, 32, 16).is_ok());
        assert!(nonoverlapping_ranges(32, 16, 16).is_ok());
        assert!(nonoverlapping_ranges(16, 16, 16).is_err());
        assert!(nonoverlapping_ranges(16, 30, 16).is_err());
        assert!(nonoverlapping_ranges(u64::MAX - 4, 16, 8).is_err());
    }
}
