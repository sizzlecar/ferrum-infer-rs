//! Fixed tensor providers. Native ownership is admitted once per runtime.
use super::super::vnext_runtime::{MetalDeviceCommand, MetalTensorProviderRetention};
use super::*;
use ferrum_runtime::{
    DeviceCommandLogicalWork, EncodedDeviceOperation, NodeWorkContract, OperationResourceEstimator,
    ReusableExecutionTopologyRequest, TensorOperationInvocation,
};
use metal::{CompileOptions, ComputePipelineState, MTLSize};

mod pointwise;
pub use pointwise::MetalTensorOperation;

const SOURCE: &str = include_str!("pointwise.metal");
pub const METAL_TENSOR_NUMERICAL_IDENTITY: &str = concat!(
    "metal-std-f32;fast-math=false;fp-contract=off;bf16-integer-rne;",
    "silu_mul=round-silu-then-mul;modulation=round-one-plus-scale-then-mul;",
    "gated-residual=round-gate-times-update-then-add;nan-class-only;",
    "independent-Metal-profile;no-CUDA-or-Reference-bitwise-claim"
);

pub fn metal_tensor_runtime_config(
    device_id: DeviceId,
) -> Result<MetalDeviceRuntimeConfig, VNextError> {
    Ok(MetalDeviceRuntimeConfig {
        device_id,
        runtime_implementation_fingerprint: fingerprint(),
        capabilities: MetalTensorOperation::ALL
            .into_iter()
            .map(|op| CapabilityId::new(op.capability()))
            .collect::<Result<_, _>>()?,
        dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
            DynamicStorageAllocator::LinearArena,
            DynamicStorageView::Contiguous,
        )?]),
    })
}

fn fingerprint() -> String {
    implementation_fingerprint(&[
        include_bytes!("mod.rs"),
        include_bytes!("pointwise.rs"),
        SOURCE.as_bytes(),
        METAL_TENSOR_NUMERICAL_IDENTITY.as_bytes(),
        include_bytes!("../mod.rs"),
    ])
}

struct Pipelines {
    pointwise: ComputePipelineState,
    // Last field: native pipeline states release before their budget owner.
    _retention: MetalTensorProviderRetention,
}

pub fn metal_tensor_providers(
    runtime: &MetalDeviceRuntime,
    selected: &[MetalTensorOperation],
) -> Result<Vec<Box<dyn OperationProvider<MetalDeviceRuntime>>>, MetalDeviceRuntimeError> {
    if selected.is_empty() || selected.iter().collect::<BTreeSet<_>>().len() != selected.len() {
        return Err(MetalDeviceRuntimeError::contract(
            "Metal tensor provider selection is empty or repeated",
        ));
    }
    // Validate every contract and capability before native creation.
    let descriptors = selected
        .iter()
        .map(|&kind| {
            let contract = kind.contract().map_err(contract_error)?;
            provider_descriptor(
                runtime,
                &contract,
                &format!("provider.metal.{}.bf16", kind.name()),
                kind.capability(),
                &format!("resource-estimator.metal.{}.bf16", kind.name()),
                contiguous_bindings(kind.inputs()),
                &[],
                &[],
                implementation_fingerprint(&[fingerprint().as_bytes(), kind.profile().as_bytes()]),
            )
            .map(|d| d.with_fixed_tensor_work(true))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let retention = runtime.retain_tensor_provider()?;
    let options = CompileOptions::new();
    options.set_fast_math_enabled(false);
    let library = runtime
        .device()
        .new_library_with_source(SOURCE, &options)
        .map_err(MetalDeviceRuntimeError::contract)?;
    let function = library
        .get_function("ferrum_tensor_pointwise_bf16", None)
        .map_err(MetalDeviceRuntimeError::contract)?;
    let pipeline = runtime
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(MetalDeviceRuntimeError::contract)?;
    let pipelines = Arc::new(Pipelines {
        pointwise: pipeline,
        _retention: retention,
    });
    Ok(selected
        .iter()
        .copied()
        .zip(descriptors)
        .map(|(kind, descriptor)| {
            Box::new(PointwiseProvider {
                kind,
                descriptor,
                pipelines: Arc::clone(&pipelines),
            }) as Box<dyn OperationProvider<MetalDeviceRuntime>>
        })
        .collect())
}

struct PointwiseProvider {
    kind: MetalTensorOperation,
    descriptor: OperationProviderDescriptor,
    pipelines: Arc<Pipelines>,
}

impl OperationResourceEstimator for PointwiseProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }
    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        let estimate =
            estimate_without_workspace(&self.descriptor, &request, self.kind.operation())?;
        pointwise::geometry(self.kind, request.values(), request.attributes())
            .map_err(invalid_plan)?;
        Ok(estimate)
    }
}

impl OperationProvider<MetalDeviceRuntime> for PointwiseProvider {
    fn reusable_execution_topology(
        &self,
        _: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        Ok(ReusableExecutionTopology::EagerBoundary)
    }

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<MetalDeviceCommand>, OperationFailure> {
        Err(provider_failure(
            invocation.participants()[0].identity().clone(),
            "metal.tensor.pointwise",
            "Metal pointwise requires fixed tensor work".into(),
        ))
    }

    fn encode_tensor_selected(
        &self,
        invocation: TensorOperationInvocation<'_, MetalDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<MetalDeviceCommand>>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        self.encode(invocation)
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|reason| provider_failure(identity, "metal.tensor.pointwise", reason))
    }
}

impl PointwiseProvider {
    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, MetalDeviceBuffer>,
    ) -> Result<MetalDeviceCommand, String> {
        let [participant] = invocation.participants() else {
            return Err("Metal pointwise requires one fixed tensor participant".into());
        };
        if invocation.operation().id.as_str() != self.kind.operation()
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("Metal pointwise received a different operation or work contract".into());
        }
        let shape =
            pointwise::geometry(self.kind, participant.bindings(), participant.attributes())?;
        let mut regions = Vec::with_capacity(self.kind.inputs() as usize + 1);
        for ordinal in 0..self.kind.inputs() {
            let region = contiguous_region(
                participant,
                binding(participant.bindings(), ResolvedValueRole::Input, ordinal)?,
                ElementType::Bf16,
            )?;
            let expected = if self.kind.broadcasts() && ordinal == 1 {
                shape.parameter_bytes
            } else {
                shape.bytes
            };
            validate_region(&region, expected)?;
            regions.push(region);
        }
        let output = contiguous_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            ElementType::Bf16,
        )?;
        validate_region(&output, shape.bytes)?;
        for input in &regions {
            if input
                .overlaps_shared_region(&output)
                .map_err(|e| e.to_string())?
            {
                return Err("Metal pointwise output overlaps a live input".into());
            }
        }
        regions.push(output);
        self.command(shape, regions)
            .map_err(|error| error.to_string())
    }

    fn command(
        &self,
        shape: pointwise::Shape,
        regions: Vec<MetalBufferRegion>,
    ) -> Result<MetalDeviceCommand, MetalDeviceRuntimeError> {
        let pipelines = Arc::clone(&self.pipelines);
        let kind = self.kind;
        let params = Params {
            elements: shape.elements,
            width: shape.width,
            kind: kind as u32,
            row_broadcast: u32::from(shape.row_broadcast),
        };
        let work = DeviceCommandLogicalWork::tensor(shape.elements, shape.operations)
            .map_err(contract_error)?;
        MetalDeviceCommand::tensor_operation(
            "metal.pointwise.bf16",
            regions,
            work,
            move |encoder, regions| {
                let compute = encoder.compute_encoder();
                compute.set_compute_pipeline_state(&pipelines.pointwise);
                for (slot, index) in [
                    0,
                    if kind.inputs() > 1 { 1 } else { 0 },
                    if kind.inputs() > 2 { 2 } else { 0 },
                    kind.inputs() as usize,
                ]
                .into_iter()
                .enumerate()
                {
                    let region = &regions[index];
                    compute.set_buffer(slot as u64, Some(region.buffer()), region.offset_bytes());
                }
                compute.set_bytes(
                    4,
                    std::mem::size_of::<Params>() as u64,
                    (&params as *const Params).cast(),
                );
                let threads = pipelines
                    .pointwise
                    .max_total_threads_per_threadgroup()
                    .min(256);
                compute.dispatch_threads(
                    MTLSize::new(shape.elements, 1, 1),
                    MTLSize::new(threads, 1, 1),
                );
                Ok(())
            },
        )
    }
}

fn validate_region(region: &MetalBufferRegion, bytes: u64) -> Result<(), String> {
    if region.length_bytes() != bytes || !region.offset_bytes().is_multiple_of(2) {
        return Err("Metal pointwise requires an exact aligned BF16 region".into());
    }
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    elements: u64,
    width: u64,
    kind: u32,
    row_broadcast: u32,
}
