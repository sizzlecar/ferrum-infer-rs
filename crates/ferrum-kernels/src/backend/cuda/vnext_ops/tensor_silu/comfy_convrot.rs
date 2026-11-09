//! Canonical W4/W8 static bytes and actual INT8/INT32 cuBLAS GEMM.
//! No BF16 matrix multiplication or whole-model weight expansion is used.
use super::super::{tensor_dense_bf16 as dense, tensor_storage};
use super::*;
use ferrum_runtime::{
    comfy_convrot_swiglu_f32_w4a8_bf16_contract, comfy_convrot_swiglu_f32_w8a8_bf16_contract,
    comfy_convrot_w4a8_bf16_contract, comfy_convrot_w4a8_f32_contract,
    comfy_convrot_w8a8_bf16_contract, comfy_convrot_w8a8_f32_contract, ComfyConvRotW4A8Geometry,
    ComfyConvRotW8A8Geometry, DynamicStorageRequirement, ProviderWorkspaceRequirement,
    ProviderWorkspaceReusePolicy, ProviderWorkspaceScope, ProviderWorkspaceSizeFormula,
    WeightFormatId, COMFY_CONVROT_SWIGLU_F32_W4A8_BF16_CAPABILITY_ID,
    COMFY_CONVROT_SWIGLU_F32_W4A8_BF16_OPERATION_ID,
    COMFY_CONVROT_SWIGLU_F32_W8A8_BF16_CAPABILITY_ID,
    COMFY_CONVROT_SWIGLU_F32_W8A8_BF16_OPERATION_ID, COMFY_CONVROT_W4A8_BF16_CAPABILITY_ID,
    COMFY_CONVROT_W4A8_BF16_OPERATION_ID, COMFY_CONVROT_W4A8_F32_CAPABILITY_ID,
    COMFY_CONVROT_W4A8_F32_OPERATION_ID, COMFY_CONVROT_W8A8_BF16_CAPABILITY_ID,
    COMFY_CONVROT_W8A8_BF16_OPERATION_ID, COMFY_CONVROT_W8A8_F32_CAPABILITY_ID,
    COMFY_CONVROT_W8A8_F32_OPERATION_ID,
};

mod launch;

pub const NUMERICAL_PROFILE: &str = "cuda.comfy.regular_h256.codebook_w4a8.bf16_int32.v1";
pub const INT8_NUMERICAL_PROFILE: &str =
    "cuda.comfy.regular_h256.w8a8.bf16_int32.per_output_scale.v1";
// expf is CUDA libdevice under this precise module's flags. It is intentionally
// independent of upstream's shape-dependent --use_fast_math activation path.
pub const SWIGLU_W4_NUMERICAL_PROFILE: &str =
    "cuda.comfy.swiglu_f32.libdevice_exp.rn_div.no_ftz.regular_h256.w4a8.bf16_int32.v1";
pub const SWIGLU_W8_NUMERICAL_PROFILE: &str =
    "cuda.comfy.swiglu_f32.libdevice_exp.rn_div.no_ftz.regular_h256.w8a8.bf16_int32.v1";
pub const F32_W4_NUMERICAL_PROFILE: &str =
    "cuda.comfy.regular_h256.codebook_w4a8.f32_int32.rn_div.no_ftz.v1";
pub const F32_W8_NUMERICAL_PROFILE: &str =
    "cuda.comfy.regular_h256.w8a8.f32_int32.rn_div.no_ftz.v1";
const PROVIDER: &str = "provider.cuda.comfy_convrot_codebook_w4a8.bf16.cublas_int32";
const ESTIMATOR: &str = "resource-estimator.cuda.comfy_convrot_codebook_w4a8.bf16";
const FORMAT: &str = "weight-format.comfy.convrot.canonical.v1";

#[derive(Clone, Copy)]
enum Storage {
    CodebookW4,
    RotatedI8,
    SwiGluCodebookW4,
    SwiGluRotatedI8,
    F32CodebookW4,
    F32RotatedI8,
}
impl Storage {
    fn codebook(self) -> bool {
        matches!(
            self,
            Self::CodebookW4 | Self::SwiGluCodebookW4 | Self::F32CodebookW4
        )
    }
    fn f32(self) -> bool {
        matches!(self, Self::F32CodebookW4 | Self::F32RotatedI8)
    }
    fn swiglu(self) -> bool {
        matches!(self, Self::SwiGluCodebookW4 | Self::SwiGluRotatedI8)
    }
    fn inputs(self) -> usize {
        if self.codebook() {
            5
        } else {
            3
        }
    }
    fn operation(self) -> &'static str {
        match self {
            Self::CodebookW4 => COMFY_CONVROT_W4A8_BF16_OPERATION_ID,
            Self::RotatedI8 => COMFY_CONVROT_W8A8_BF16_OPERATION_ID,
            Self::SwiGluCodebookW4 => COMFY_CONVROT_SWIGLU_F32_W4A8_BF16_OPERATION_ID,
            Self::SwiGluRotatedI8 => COMFY_CONVROT_SWIGLU_F32_W8A8_BF16_OPERATION_ID,
            Self::F32CodebookW4 => COMFY_CONVROT_W4A8_F32_OPERATION_ID,
            Self::F32RotatedI8 => COMFY_CONVROT_W8A8_F32_OPERATION_ID,
        }
    }
    fn profile(self) -> &'static str {
        match self {
            Self::CodebookW4 => NUMERICAL_PROFILE,
            Self::RotatedI8 => INT8_NUMERICAL_PROFILE,
            Self::SwiGluCodebookW4 => SWIGLU_W4_NUMERICAL_PROFILE,
            Self::SwiGluRotatedI8 => SWIGLU_W8_NUMERICAL_PROFILE,
            Self::F32CodebookW4 => F32_W4_NUMERICAL_PROFILE,
            Self::F32RotatedI8 => F32_W8_NUMERICAL_PROFILE,
        }
    }
    fn capability(self) -> &'static str {
        match self {
            Self::CodebookW4 => COMFY_CONVROT_W4A8_BF16_CAPABILITY_ID,
            Self::RotatedI8 => COMFY_CONVROT_W8A8_BF16_CAPABILITY_ID,
            Self::SwiGluCodebookW4 => COMFY_CONVROT_SWIGLU_F32_W4A8_BF16_CAPABILITY_ID,
            Self::SwiGluRotatedI8 => COMFY_CONVROT_SWIGLU_F32_W8A8_BF16_CAPABILITY_ID,
            Self::F32CodebookW4 => COMFY_CONVROT_W4A8_F32_CAPABILITY_ID,
            Self::F32RotatedI8 => COMFY_CONVROT_W8A8_F32_CAPABILITY_ID,
        }
    }
    fn provider(self) -> &'static str {
        match self {
            Self::CodebookW4 => PROVIDER,
            Self::RotatedI8 => "provider.cuda.comfy_convrot_w8a8.bf16.cublas_int32",
            Self::SwiGluCodebookW4 => {
                "provider.cuda.comfy_convrot_w4a8.swiglu_f32.bf16.cublas_int32"
            }
            Self::SwiGluRotatedI8 => {
                "provider.cuda.comfy_convrot_w8a8.swiglu_f32.bf16.cublas_int32"
            }
            Self::F32CodebookW4 => "provider.cuda.comfy_convrot_codebook_w4a8.f32.cublas_int32",
            Self::F32RotatedI8 => "provider.cuda.comfy_convrot_w8a8.f32.cublas_int32",
        }
    }
    fn estimator(self) -> &'static str {
        match self {
            Self::CodebookW4 => ESTIMATOR,
            Self::RotatedI8 => "resource-estimator.cuda.comfy_convrot_w8a8.bf16",
            Self::SwiGluCodebookW4 => "resource-estimator.cuda.comfy_convrot_w4a8.swiglu_f32.bf16",
            Self::SwiGluRotatedI8 => "resource-estimator.cuda.comfy_convrot_w8a8.swiglu_f32.bf16",
            Self::F32CodebookW4 => "resource-estimator.cuda.comfy_convrot_codebook_w4a8.f32",
            Self::F32RotatedI8 => "resource-estimator.cuda.comfy_convrot_w8a8.f32",
        }
    }
    fn command(self) -> &'static str {
        match self {
            Self::CodebookW4 => "vnext_comfy_convrot_w4a8_bf16",
            Self::RotatedI8 => "vnext_comfy_convrot_w8a8_bf16",
            Self::SwiGluCodebookW4 => "vnext_comfy_swiglu_f32_convrot_w4a8_bf16",
            Self::SwiGluRotatedI8 => "vnext_comfy_swiglu_f32_convrot_w8a8_bf16",
            Self::F32CodebookW4 => "vnext_comfy_convrot_w4a8_f32",
            Self::F32RotatedI8 => "vnext_comfy_convrot_w8a8_f32",
        }
    }
}

/// Common launch geometry; W8 has no decoded weight allocation.
#[derive(Clone, Copy)]
struct Shape {
    rows: u64,
    inputs: u64,
    outputs: u64,
    rotated_activation: u64,
    quantized_activation: u64,
    row_scales: u64,
    decoded_weight: Option<u64>,
    accumulator: u64,
    scratch_bytes: u64,
    arithmetic: u64,
}
impl Shape {
    fn output_elements(self) -> u64 {
        self.rows * self.outputs
    }
    fn arithmetic(self) -> u64 {
        self.arithmetic
    }
}

pub(super) fn descriptor(
    device: &DeviceDescriptor,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    descriptor_for(device, Storage::CodebookW4)
}
pub(super) fn int8_descriptor(
    device: &DeviceDescriptor,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    descriptor_for(device, Storage::RotatedI8)
}
pub(super) fn swiglu_w4_descriptor(
    device: &DeviceDescriptor,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    descriptor_for(device, Storage::SwiGluCodebookW4)
}
pub(super) fn swiglu_w8_descriptor(
    device: &DeviceDescriptor,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    descriptor_for(device, Storage::SwiGluRotatedI8)
}
pub(super) fn f32_w4_descriptor(
    device: &DeviceDescriptor,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    descriptor_for(device, Storage::F32CodebookW4)
}
pub(super) fn f32_w8_descriptor(
    device: &DeviceDescriptor,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    descriptor_for(device, Storage::F32RotatedI8)
}
fn descriptor_for(
    device: &DeviceDescriptor,
    storage: Storage,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let capability = CapabilityId::new(storage.capability()).map_err(contract_error)?;
    if !device.capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA runtime lacks the selected Comfy ConvRot capability",
        ));
    }
    let contract = match storage {
        Storage::CodebookW4 => comfy_convrot_w4a8_bf16_contract(),
        Storage::RotatedI8 => comfy_convrot_w8a8_bf16_contract(),
        Storage::SwiGluCodebookW4 => comfy_convrot_swiglu_f32_w4a8_bf16_contract(),
        Storage::SwiGluRotatedI8 => comfy_convrot_swiglu_f32_w8a8_bf16_contract(),
        Storage::F32CodebookW4 => comfy_convrot_w4a8_f32_contract(),
        Storage::F32RotatedI8 => comfy_convrot_w8a8_f32_contract(),
    }
    .map_err(contract_error)?;
    let implementation = implementation_fingerprint(&[
        storage.profile().as_bytes(),
        numerical_fingerprint().as_bytes(),
        include_bytes!("comfy_convrot.rs"),
        include_bytes!("comfy_convrot/launch.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/vnext_comfy_convrot.cuh"
        )),
        format!("cublas.version={}", dense::cublas_version()?).as_bytes(),
    ]);
    let estimator =
        implementation_fingerprint(&[storage.estimator().as_bytes(), implementation.as_bytes()]);
    OperationProviderDescriptor::new(
        ProviderId::new(storage.provider()).map_err(contract_error)?,
        contract.descriptor().id.clone(),
        contract
            .descriptor()
            .fingerprint()
            .map_err(contract_error)?,
        implementation,
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        device.id.clone(),
        BTreeSet::from([capability]),
        BTreeSet::from([
            WeightFormatId::new(FORMAT).map_err(contract_error)?,
            // A mixed model can store these explicit raw canonical ports
            // beside dense BF16/F32 exclusions in one safetensors schema.
            WeightFormatId::new(super::super::DENSE_SAFETENSORS_FORMAT_ID)
                .map_err(contract_error)?,
        ]),
        BTreeSet::new(),
        super::super::transformer::contiguous_bindings(storage.inputs() as u32),
        storage.estimator(),
        ContractVersion::new(1, 0),
        estimator,
    )
    .map(|value| value.with_fixed_tensor_work(true))
    .map_err(contract_error)
}
pub(super) fn provider(
    descriptor: OperationProviderDescriptor,
    kernels: &Arc<RetainedFunctions>,
) -> Box<dyn OperationProvider<CudaDeviceRuntime>> {
    Box::new(Provider {
        descriptor,
        kernels: Arc::clone(kernels),
        storage: Storage::CodebookW4,
    })
}
pub(super) fn int8_provider(
    descriptor: OperationProviderDescriptor,
    kernels: &Arc<RetainedFunctions>,
) -> Box<dyn OperationProvider<CudaDeviceRuntime>> {
    Box::new(Provider {
        descriptor,
        kernels: Arc::clone(kernels),
        storage: Storage::RotatedI8,
    })
}
pub(super) fn swiglu_w4_provider(
    descriptor: OperationProviderDescriptor,
    kernels: &Arc<RetainedFunctions>,
) -> Box<dyn OperationProvider<CudaDeviceRuntime>> {
    Box::new(Provider {
        descriptor,
        kernels: Arc::clone(kernels),
        storage: Storage::SwiGluCodebookW4,
    })
}
pub(super) fn swiglu_w8_provider(
    descriptor: OperationProviderDescriptor,
    kernels: &Arc<RetainedFunctions>,
) -> Box<dyn OperationProvider<CudaDeviceRuntime>> {
    Box::new(Provider {
        descriptor,
        kernels: Arc::clone(kernels),
        storage: Storage::SwiGluRotatedI8,
    })
}
pub(super) fn f32_w4_provider(
    descriptor: OperationProviderDescriptor,
    kernels: &Arc<RetainedFunctions>,
) -> Box<dyn OperationProvider<CudaDeviceRuntime>> {
    Box::new(Provider {
        descriptor,
        kernels: Arc::clone(kernels),
        storage: Storage::F32CodebookW4,
    })
}
pub(super) fn f32_w8_provider(
    descriptor: OperationProviderDescriptor,
    kernels: &Arc<RetainedFunctions>,
) -> Box<dyn OperationProvider<CudaDeviceRuntime>> {
    Box::new(Provider {
        descriptor,
        kernels: Arc::clone(kernels),
        storage: Storage::F32RotatedI8,
    })
}
struct Provider {
    descriptor: OperationProviderDescriptor,
    kernels: Arc<RetainedFunctions>,
    storage: Storage,
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
            return Err(invalid("estimator received another operation"));
        }
        let shape =
            geometry(self.storage, request.values(), request.attributes()).map_err(invalid)?;
        let scratch = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::fixed(shape.scratch_bytes)?,
            256,
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
            "cuda.comfy_convrot.encode",
            "ConvRot requires fixed tensor execution".into(),
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
            .map_err(|message| provider_failure(identity, "cuda.comfy_convrot.encode", message))
    }
}
impl Provider {
    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [participant] = invocation.participants() else {
            return Err("ConvRot requires one tensor participant".into());
        };
        if invocation.operation().id.as_str() != self.storage.operation()
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("ConvRot requires its selected fixed tensor contract".into());
        }
        let shape = geometry(
            self.storage,
            participant.bindings(),
            participant.attributes(),
        )?;
        let inputs = self.storage.inputs();
        let mut regions = Vec::with_capacity(inputs + 2);
        for (role, ordinal) in (0..inputs)
            .map(|ordinal| (ResolvedValueRole::Input, ordinal))
            .chain(std::iter::once((ResolvedValueRole::Output, 0)))
        {
            let binding = binding(participant.bindings(), role, ordinal as u32)?;
            let [component] = binding.storage().components() else {
                return Err("ConvRot requires one contiguous component per port".into());
            };
            if component.element_type() != binding.tensor().element_type()
                || component.length_bytes()
                    != binding
                        .tensor()
                        .minimum_storage_bytes()
                        .map_err(|error| error.to_string())?
            {
                return Err("ConvRot component byte extent or dtype differs".into());
            }
            regions.push(tensor_storage::component(participant, component, 16)?);
        }
        regions.push(tensor_storage::scratch(
            participant,
            shape.scratch_bytes,
            256,
        )?);
        for read in &regions[..inputs] {
            dense::disjoint(
                read.device_ptr(),
                read.length_bytes(),
                regions[inputs].device_ptr(),
                regions[inputs].length_bytes(),
            )?;
        }
        for live in &regions[..inputs + 1] {
            dense::disjoint(
                live.device_ptr(),
                live.length_bytes(),
                regions[inputs + 1].device_ptr(),
                regions[inputs + 1].length_bytes(),
            )?;
        }
        let kernels = Arc::clone(&self.kernels);
        let storage = self.storage;
        CudaDeviceCommand::operation_with_blas(
            storage.command(),
            regions,
            move |stream, blas, regions| {
                launch::enqueue(stream, blas, &kernels, regions, shape, storage)
            },
        )
        .and_then(|command| {
            command.with_tensor_work(
                shape.output_elements(),
                shape.arithmetic(),
                if storage.codebook() { 5 } else { 4 },
                0,
            )
        })
        .map_err(|error| error.to_string())
    }
}
fn geometry(
    storage: Storage,
    bindings: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Shape, String> {
    if bindings.len() != storage.inputs() + 1 || attributes.len() != 2 {
        return Err(
            "ConvRot requires its canonical inputs, one output and two shape attributes".into(),
        );
    }
    if bindings.iter().any(|binding| {
        binding.weight().is_some_and(|weight| {
            !matches!(
                weight.physical_layout(),
                ferrum_runtime::PhysicalWeightLayout::Dense { .. }
            )
        })
    }) {
        return Err("ConvRot requires exact dense canonical storage ports, without layout transforms or implicit decoding".into());
    }
    let inputs = (0..storage.inputs())
        .map(|ordinal| {
            binding(bindings, ResolvedValueRole::Input, ordinal as u32).map(|value| value.tensor())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let output = binding(bindings, ResolvedValueRole::Output, 0)?.tensor();
    let k = super::super::unsigned_attribute(attributes, "in_features")?;
    let n = super::super::unsigned_attribute(attributes, "out_features")?;
    match storage {
        Storage::CodebookW4 | Storage::SwiGluCodebookW4 | Storage::F32CodebookW4 => {
            let geometry = if storage.f32() {
                ComfyConvRotW4A8Geometry::new_f32(&inputs, output, k, n)
            } else if storage.swiglu() {
                ComfyConvRotW4A8Geometry::new_swiglu_f32(&inputs, output, k, n)
            } else {
                ComfyConvRotW4A8Geometry::new(&inputs, output, k, n)
            }
            .map_err(|error| error.to_string())?;
            let scratch = geometry.scratch();
            Ok(Shape {
                rows: geometry.rows(),
                inputs: k,
                outputs: n,
                rotated_activation: scratch.rotated_activation,
                quantized_activation: scratch.quantized_activation,
                row_scales: scratch.row_scales,
                decoded_weight: Some(scratch.decoded_weight),
                accumulator: scratch.accumulator,
                scratch_bytes: scratch.total_bytes,
                arithmetic: geometry.logical_arithmetic_operations(),
            })
        }
        Storage::RotatedI8 | Storage::SwiGluRotatedI8 | Storage::F32RotatedI8 => {
            let geometry = if storage.f32() {
                ComfyConvRotW8A8Geometry::new_f32(&inputs, output, k, n)
            } else if storage.swiglu() {
                ComfyConvRotW8A8Geometry::new_swiglu_f32(&inputs, output, k, n)
            } else {
                ComfyConvRotW8A8Geometry::new(&inputs, output, k, n)
            }
            .map_err(|error| error.to_string())?;
            let scratch = geometry.scratch();
            Ok(Shape {
                rows: geometry.rows(),
                inputs: k,
                outputs: n,
                rotated_activation: scratch.rotated_activation,
                quantized_activation: scratch.quantized_activation,
                row_scales: scratch.row_scales,
                decoded_weight: None,
                accumulator: scratch.accumulator,
                scratch_bytes: scratch.total_bytes,
                arithmetic: geometry.logical_arithmetic_operations(),
            })
        }
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("CUDA ConvRot: {}", reason.into()),
    }
}
