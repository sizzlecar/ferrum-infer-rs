//! cuDNN backward-data implements the adjoint of zero-padded F32 correlation.
//! Its plan, native workspace and bias launch use the original CNN authorities.
use super::*;
use crate::backend::cuda::vnext_runtime::{
    CudaCudnnOwner, CudnnConv2dDescriptor, CudnnConvScalar, PreparedConv,
};
use ferrum_runtime::{
    conv_transpose2d_f32_contract, ConvTranspose2dF32Geometry, DynamicStorageRequirement,
    PhysicalWeightLayout, ProviderWorkspaceRequirement, ProviderWorkspaceReusePolicy,
    ProviderWorkspaceScope, ProviderWorkspaceSizeFormula, WeightFormatId,
    CONV_TRANSPOSE2D_F32_CAPABILITY_ID, CONV_TRANSPOSE2D_F32_OPERATION_ID,
};

pub const NUMERICAL_PROFILE_ID: &str =
    "cuda.cudnn9.conv_transpose2d.f32.fma_only.iohw.separate_bias.v1";

pub(super) fn require_capability(
    device: &DeviceDescriptor,
) -> Result<CapabilityId, CudaDeviceRuntimeError> {
    let capability =
        CapabilityId::new(CONV_TRANSPOSE2D_F32_CAPABILITY_ID).map_err(contract_error)?;
    if !device.capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA lacks F32 transposed convolution capability",
        ));
    }
    Ok(capability)
}

pub(super) fn provider(
    runtime: &CudaDeviceRuntime,
    kernels: Arc<RetainedFunctions>,
    owner: Arc<CudaCudnnOwner>,
) -> Result<Box<dyn OperationProvider<CudaDeviceRuntime>>, CudaDeviceRuntimeError> {
    let cap = require_capability(runtime.descriptor())?;
    let contract = conv_transpose2d_f32_contract().map_err(contract_error)?;
    let fingerprint = implementation_fingerprint(&[
        NUMERICAL_PROFILE_ID.as_bytes(),
        numerical_fingerprint().as_bytes(),
        owner.fingerprint().as_bytes(),
        include_bytes!("conv_transpose2d_f32.rs"),
        include_bytes!("conv2d_f32.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/vnext_cnn_f32.cuh"
        )),
    ]);
    let estimator = "resource-estimator.cuda.conv_transpose2d.f32.cudnn9_fma";
    let descriptor = OperationProviderDescriptor::new(
        ProviderId::new("provider.cuda.conv_transpose2d.f32.cudnn9_fma").map_err(contract_error)?,
        contract.descriptor().id.clone(),
        contract
            .descriptor()
            .fingerprint()
            .map_err(contract_error)?,
        fingerprint.clone(),
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        runtime.descriptor().id.clone(),
        BTreeSet::from([cap]),
        BTreeSet::from([
            WeightFormatId::new(super::super::DENSE_SAFETENSORS_FORMAT_ID)
                .map_err(contract_error)?,
        ]),
        BTreeSet::new(),
        super::super::transformer::contiguous_bindings(3),
        estimator,
        ContractVersion::new(1, 0),
        implementation_fingerprint(&[estimator.as_bytes(), fingerprint.as_bytes()]),
    )
    .map_err(contract_error)?
    .with_fixed_tensor_work(true);
    Ok(Box::new(Provider {
        descriptor,
        kernels,
        owner,
    }))
}

struct Provider {
    descriptor: OperationProviderDescriptor,
    kernels: Arc<RetainedFunctions>,
    owner: Arc<CudaCudnnOwner>,
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
                "transposed convolution estimator received another operation",
            ));
        }
        let shape = geometry(request.values(), request.attributes()).map_err(invalid_plan)?;
        let prepared = self
            .owner
            .prepare_transposed_convolution(shape.native)
            .map_err(|e| invalid_plan(e.to_string()))?;
        let workspace = prepared.workspace_bytes();
        if workspace > isize::MAX as u64 {
            return Err(invalid_plan(
                "transposed convolution workspace exceeds address space",
            ));
        }
        let scratch = if workspace == 0 {
            None
        } else {
            Some(ProviderWorkspaceRequirement::from_formula(
                ProviderWorkspaceSizeFormula::fixed(workspace)?,
                conv2d_f32::ALIGNMENT,
                ProviderWorkspaceScope::Invocation,
                ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
                DynamicStorageRequirement::contiguous(),
            )?)
        };
        OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            conv2d_f32::ALIGNMENT,
            scratch,
            None,
        )
        .with_prepared_implementation_fingerprint(prepared.fingerprint().to_owned())?
        .with_prepared_implementation_retention(prepared)
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
            "cuda.conv_transpose2d.f32.encode",
            "requires fixed tensor invocation".into(),
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
            .map_err(|e| provider_failure(identity, "cuda.conv_transpose2d.f32.encode", e))
    }
}

impl Provider {
    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [participant] = invocation.participants() else {
            return Err("F32 transposed convolution requires one participant".into());
        };
        if invocation.operation().id.as_str() != CONV_TRANSPOSE2D_F32_OPERATION_ID
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("requires selected fixed F32 transposed convolution contract".into());
        }
        let prepared = invocation
            .prepared_implementation::<PreparedConv>()
            .ok_or("transposed convolution has no retained prepared implementation")?;
        let shape = geometry(participant.bindings(), participant.attributes())?;
        prepared.validate_transposed_invocation(
            &self.owner,
            shape.native,
            invocation.prepared_implementation_fingerprint(),
        )?;
        let g = shape.geometry;
        let workspace_bytes = prepared.workspace_bytes();
        let regions = conv2d_f32::retain_regions(
            participant,
            [
                g.input_elements(0).ok_or("missing input extent")?,
                g.input_elements(1).ok_or("missing weight extent")?,
                g.input_elements(2).ok_or("missing bias extent")?,
            ],
            g.elements(),
            workspace_bytes,
        )?;
        let kernels = Arc::clone(&self.kernels);
        CudaDeviceCommand::operation(
            "vnext_conv_transpose2d_f32_cudnn9_fma",
            regions,
            move |stream, regions| {
                let output = regions[3].device_ptr();
                let workspace = if workspace_bytes == 0 {
                    0
                } else {
                    regions[4].device_ptr()
                };
                prepared.transposed_convolution(
                    stream,
                    regions[0].device_ptr(),
                    regions[1].device_ptr(),
                    output,
                    workspace,
                    workspace_bytes,
                )?;
                let bias = regions[2].device_ptr();
                let elements = g.elements();
                let channels = g.output()[1];
                let spatial = g.output()[2] * g.output()[3];
                let mut launch = stream.launch_builder(kernels.cnn_f32_function(1));
                launch
                    .arg(&output)
                    .arg(&bias)
                    .arg(&elements)
                    .arg(&channels)
                    .arg(&spatial);
                unsafe { launch.launch(conv2d_f32::flat(elements)) }
                    .map(|_| ())
                    .map_err(|e| {
                        CudaDeviceRuntimeError::driver("transposed convolution separate bias", e)
                    })
            },
        )
        .and_then(|command| {
            command.with_tensor_work(g.elements(), g.logical_arithmetic_operations(), 2, 0)
        })
        .map_err(|e| e.to_string())
    }
}

struct Shape {
    native: CudnnConv2dDescriptor,
    geometry: ConvTranspose2dF32Geometry,
}

fn geometry(
    bindings: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Shape, String> {
    if bindings.len() != 4
        || bindings.iter().any(|value| {
            value.weight().is_some_and(|weight| {
                !matches!(weight.physical_layout(), PhysicalWeightLayout::Dense { .. })
            })
        })
    {
        return Err("requires exact Dense F32 transposed convolution ports".into());
    }
    let g = ConvTranspose2dF32Geometry::new(
        binding(bindings, ResolvedValueRole::Input, 0)?.tensor(),
        binding(bindings, ResolvedValueRole::Input, 1)?.tensor(),
        binding(bindings, ResolvedValueRole::Input, 2)?.tensor(),
        binding(bindings, ResolvedValueRole::Output, 0)?.tensor(),
        attributes,
    )
    .map_err(|e| e.to_string())?;
    let dimension = |value: u64| {
        i32::try_from(value)
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| "cuDNN dimensions require positive i32".to_owned())
    };
    let tensor = |values: [u64; 4]| -> Result<[i32; 4], String> {
        values
            .into_iter()
            .try_fold(1u64, |count, value| count.checked_mul(value))
            .filter(|count| *count <= i32::MAX as u64)
            .ok_or("cuDNN tensor numel exceeds i32")?;
        Ok([
            dimension(values[0])?,
            dimension(values[1])?,
            dimension(values[2])?,
            dimension(values[3])?,
        ])
    };
    let padding = g.padding().map(i32::try_from);
    // The ordinary forward descriptor describes the dual problem: DX is its
    // input, DY its output, and IOHW is its OIHW. Output padding only selects DX.
    let native = CudnnConv2dDescriptor {
        scalar: CudnnConvScalar::F32Fma,
        input: tensor(g.output())?,
        filter: tensor(g.weight())?,
        output: tensor(g.input())?,
        padding: [
            padding[0].map_err(|e| e.to_string())?,
            padding[1].map_err(|e| e.to_string())?,
        ],
        stride: [dimension(g.stride()[0])?, dimension(g.stride()[1])?],
        dilation: [dimension(g.dilation()[0])?, dimension(g.dilation()[1])?],
        groups: dimension(g.groups())?,
    };
    Ok(Shape {
        native,
        geometry: g,
    })
}
