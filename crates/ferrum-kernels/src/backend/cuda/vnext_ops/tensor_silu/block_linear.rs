//! Native packed matrices, one core-admitted BF16 decode buffer, lane GEMM.
use super::super::{tensor_dense_bf16 as dense, tensor_storage};
use super::*;
use crate::gguf_blocks::GgufBlockFormat;
use ferrum_runtime::{
    dense_linear_block_bf16_contract, DynamicStorageRequirement, PhysicalStorageLayout,
    PhysicalWeightLayout, PhysicalWeightPadding, ProviderWorkspaceRequirement,
    ProviderWorkspaceReusePolicy, ProviderWorkspaceScope, ProviderWorkspaceSizeFormula,
    QuantizationFormatId, ResolvedStorageComponent, WeightComponentRole, WeightEncoding,
    WeightFormatId, DENSE_LINEAR_BLOCK_BF16_CAPABILITY_ID, DENSE_LINEAR_BLOCK_BF16_OPERATION_ID,
};

const PROVIDER: &str = "provider.cuda.dense_linear.native_block.bf16";
const ESTIMATOR: &str = "resource-estimator.cuda.dense_linear.native_block.bf16";
const FORMATS: [GgufBlockFormat; 4] = [
    GgufBlockFormat::Q8_0,
    GgufBlockFormat::Q4K,
    GgufBlockFormat::Q5K,
    GgufBlockFormat::Q6K,
];
pub const NUMERICAL_PROFILE: &str = "cuda.native_block.decode_f32_rn_bf16.cublas_f32.v1";

pub(super) fn descriptor(
    device: &DeviceDescriptor,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let capability =
        CapabilityId::new(DENSE_LINEAR_BLOCK_BF16_CAPABILITY_ID).map_err(contract_error)?;
    if !device.capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA runtime lacks fixed-tensor native block capability",
        ));
    }
    let contract = dense_linear_block_bf16_contract().map_err(contract_error)?;
    let implementation = implementation_fingerprint(&[
        NUMERICAL_PROFILE.as_bytes(),
        numerical_fingerprint().as_bytes(),
        include_bytes!("block_linear.rs"),
        include_bytes!("../tensor_storage.rs"),
        include_bytes!("../tensor_dense_bf16.rs"),
        format!("cublas.version={}", dense::cublas_version()?).as_bytes(),
    ]);
    let estimator = implementation_fingerprint(&[ESTIMATOR.as_bytes(), implementation.as_bytes()]);
    OperationProviderDescriptor::new(
        ProviderId::new(PROVIDER).map_err(contract_error)?,
        contract.descriptor().id.clone(),
        contract
            .descriptor()
            .fingerprint()
            .map_err(contract_error)?,
        implementation,
        ferrum_runtime::ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        device.id.clone(),
        BTreeSet::from([capability]),
        BTreeSet::from([
            WeightFormatId::new("weight-format.gguf.native-block").map_err(contract_error)?
        ]),
        FORMATS
            .into_iter()
            .map(|format| QuantizationFormatId::new(format.format_id()))
            .collect::<Result<_, _>>()
            .map_err(contract_error)?,
        super::super::transformer::contiguous_bindings(2),
        ESTIMATOR,
        ContractVersion::new(1, 0),
        estimator,
    )
    .map(|descriptor| descriptor.with_fixed_tensor_work(true))
    .map_err(contract_error)
}

pub(super) fn provider(
    descriptor: OperationProviderDescriptor,
    kernels: &Arc<RetainedFunctions>,
) -> Box<dyn OperationProvider<CudaDeviceRuntime>> {
    Box::new(Provider {
        descriptor,
        kernels: Arc::clone(kernels),
    })
}
struct Provider {
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
            return Err(invalid("native block estimator received another operation"));
        }
        let (shape, _) = geometry(request.values(), request.attributes()).map_err(invalid)?;
        let scratch = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::fixed(shape.weight_bytes)?,
            16,
            ProviderWorkspaceScope::Invocation,
            ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
            DynamicStorageRequirement::contiguous(),
        )?;
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            16,
            Some(scratch),
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
            "cuda.native_block.encode",
            "native block BF16 Linear requires fixed tensors".into(),
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
            .map_err(|message| provider_failure(identity, "cuda.native_block.encode", message))
    }
}
impl Provider {
    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [participant] = invocation.participants() else {
            return Err("native block Linear requires one tensor participant".into());
        };
        if invocation.operation().id.as_str() != DENSE_LINEAR_BLOCK_BF16_OPERATION_ID
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("native block Linear requires its fixed tensor contract".into());
        }
        let (shape, format) = geometry(participant.bindings(), participant.attributes())?;
        let input = dense::retained_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
            shape.input_bytes,
        )?;
        let weight_binding = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
        let packed =
            tensor_storage::component(participant, packed_component(weight_binding)?.0, 16)?;
        let output = dense::retained_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            shape.output_bytes,
        )?;
        let scratch = tensor_storage::scratch(participant, shape.weight_bytes, 16)?;
        dense::disjoint(
            input.device_ptr(),
            input.length_bytes(),
            output.device_ptr(),
            output.length_bytes(),
        )?;
        dense::disjoint(
            packed.device_ptr(),
            packed.length_bytes(),
            output.device_ptr(),
            output.length_bytes(),
        )?;
        for read in [&input, &packed, &output] {
            dense::disjoint(
                read.device_ptr(),
                read.length_bytes(),
                scratch.device_ptr(),
                scratch.length_bytes(),
            )?;
        }
        let kernels = Arc::clone(&self.kernels);
        CudaDeviceCommand::operation_with_blas(
            "vnext_dense_linear_native_block_bf16",
            vec![input, packed, output, scratch],
            move |stream, blas, regions| {
                let count = shape.weight_bytes / 2;
                let format = format.ggml_type_id();
                let (values, bytes) = if format == 8 {
                    (32_u32, 34_u32)
                } else {
                    (
                        256_u32,
                        match format {
                            12 => 144_u32,
                            13 => 176_u32,
                            14 => 210_u32,
                            _ => unreachable!(),
                        },
                    )
                };
                let source = regions[1].device_ptr();
                let decoded = regions[3].device_ptr();
                let launch = LaunchConfig {
                    grid_dim: (count.div_ceil(256).min(65_535) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    stream
                        .launch_builder(kernels.block_decode_function())
                        .arg(&source)
                        .arg(&decoded)
                        .arg(&count)
                        .arg(&format)
                        .arg(&values)
                        .arg(&bytes)
                        .launch(launch)
                }
                .map_err(|e| CudaDeviceRuntimeError::driver("native block BF16 decode", e))?;
                dense::launch(stream, blas, &regions[0], decoded, &regions[2], shape)
            },
        )
        .and_then(|command| {
            command.with_tensor_work(shape.output_elements, shape.arithmetic_operations, 2, 0)
        })
        .map_err(|e| e.to_string())
    }
}

fn geometry(
    bindings: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<(dense::Shape, GgufBlockFormat), String> {
    if bindings.len() != 3 || attributes.len() != 2 {
        return Err("native block Linear requires three ports and two shape attributes".into());
    }
    let input = binding(bindings, ResolvedValueRole::Input, 0)?;
    let weight = binding(bindings, ResolvedValueRole::Input, 1)?;
    let output = binding(bindings, ResolvedValueRole::Output, 0)?;
    let shape = dense::validate_specs(
        input.tensor(),
        weight.tensor(),
        output.tensor(),
        super::super::unsigned_attribute(attributes, "in_features")?,
        super::super::unsigned_attribute(attributes, "out_features")?,
    )?;
    let (_, format) = packed_component(weight)?;
    Ok((shape, format))
}

fn packed_component(
    binding: &ResolvedValueBinding,
) -> Result<(&ResolvedStorageComponent, GgufBlockFormat), String> {
    let weight = binding
        .weight()
        .ok_or("native block weight lacks typed physical layout")?;
    weight
        .validate_logical(binding.tensor().dimensions(), ElementType::Bf16)
        .map_err(|e| e.to_string())?;
    let PhysicalWeightLayout::BlockQuantized {
        blocks,
        block_axis: 1,
        block_padding: PhysicalWeightPadding::Exact,
    } = weight.physical_layout()
    else {
        return Err("native block Linear requires one exact OI block leaf".into());
    };
    if !matches!(
        blocks.storage,
        PhysicalStorageLayout::Contiguous {
            padding: PhysicalWeightPadding::Exact
        }
    ) {
        return Err("native blocks require exact contiguous storage".into());
    }
    let [component] = weight.components() else {
        return Err("native blocks require one physical component".into());
    };
    let WeightEncoding::BlockQuantized(spec) = component.encoding() else {
        return Err("native block component has no block ABI".into());
    };
    let format = GgufBlockFormat::from_spec(spec)?;
    if !FORMATS.contains(&format)
        || component.role() != WeightComponentRole::PackedValues
        || component.component_id() != &blocks.component_id
    {
        return Err("native block Linear does not support this format or component role".into());
    }
    let [stored] = binding.storage().components() else {
        return Err("native blocks require one committed storage range".into());
    };
    if stored.component_id() != Some(component.component_id())
        || stored.element_type() != ElementType::U8
        || stored.length_bytes() != component.physical_bytes().map_err(|e| e.to_string())?
    {
        return Err("native block storage identity, type or byte extent differs".into());
    }
    Ok((stored, format))
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
