//! Fixed tensor providers. Native ownership is admitted once per runtime.
use super::super::vnext_runtime::{MetalDeviceCommand, MetalTensorProviderRetention};
use super::*;
use ferrum_runtime::{
    DeviceCommandLogicalWork, EncodedDeviceOperation, NodeWorkContract, OperationResourceEstimator,
    ReusableExecutionTopologyRequest, TensorOperationInvocation,
};
use metal::{CompileOptions, ComputePipelineState, MTLSize};

mod linear;
mod pointwise;
pub use pointwise::MetalTensorOperation;

pub const METAL_TENSOR_NUMERICAL_IDENTITY: &str = concat!(
    "metal-std-f32;fast-math=false;fp-contract=off;bf16-integer-rne;",
    "silu_mul=round-silu-then-mul;modulation=round-one-plus-scale-then-mul;",
    "gated-residual=round-gate-times-update-then-add;nan-class-only;",
    "gemm=simdgroup-f32-m32n64k32;OI;BF16-final-boundary;",
    "native-block=decode-f32-then-BF16;admitted-invocation-scratch;",
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
        include_bytes!("bf16.metal"),
        include_bytes!("pointwise.metal"),
        include_bytes!("linear.rs"),
        include_bytes!("linear.metal"),
        include_bytes!("block_decode.metal"),
        native_blocks::FINGERPRINT_SOURCE.as_bytes(),
        METAL_TENSOR_NUMERICAL_IDENTITY.as_bytes(),
        include_bytes!("../mod.rs"),
    ])
}

struct Pipelines {
    pointwise: Option<ComputePipelineState>,
    linear: Option<ComputePipelineState>,
    block_decode: Option<ComputePipelineState>,
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
            let weights: &[&str] = match kind {
                MetalTensorOperation::DenseLinear => {
                    &[DENSE_SAFETENSORS_FORMAT_ID, GGUF_NATIVE_BLOCK_FORMAT_ID]
                }
                MetalTensorOperation::BlockLinear => &[GGUF_NATIVE_BLOCK_FORMAT_ID],
                _ => &[],
            };
            let quantization: Vec<_> = if kind == MetalTensorOperation::BlockLinear {
                linear::FORMATS
                    .iter()
                    .map(|format| format.format_id())
                    .collect()
            } else {
                Vec::new()
            };
            provider_descriptor(
                runtime,
                &contract,
                &format!("provider.metal.{}.bf16", kind.name()),
                kind.capability(),
                &format!("resource-estimator.metal.{}.bf16", kind.name()),
                contiguous_bindings(kind.inputs()),
                weights,
                &quantization,
                implementation_fingerprint(&[fingerprint().as_bytes(), kind.profile().as_bytes()]),
            )
            .map(|d| d.with_fixed_tensor_work(true))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let retention = runtime.retain_tensor_provider()?;
    let has_pointwise = selected.iter().any(|op| !op.is_linear());
    let has_linear = selected.iter().any(|op| op.is_linear());
    let has_blocks = selected.contains(&MetalTensorOperation::BlockLinear);
    let mut source = String::from(include_str!("bf16.metal"));
    if has_blocks {
        source.push_str(&native_blocks::decoder_source());
    }
    if has_pointwise {
        source.push_str(include_str!("pointwise.metal"));
    }
    if has_linear {
        source.push_str(include_str!("linear.metal"));
    }
    if has_blocks {
        source.push_str(include_str!("block_decode.metal"));
    }
    let options = CompileOptions::new();
    options.set_fast_math_enabled(false);
    let library = runtime
        .device()
        .new_library_with_source(&source, &options)
        .map_err(MetalDeviceRuntimeError::contract)?;
    let pipeline = |selected: bool, name: &str| -> Result<_, MetalDeviceRuntimeError> {
        if !selected {
            return Ok(None);
        }
        let function = library
            .get_function(name, None)
            .map_err(MetalDeviceRuntimeError::contract)?;
        runtime
            .device()
            .new_compute_pipeline_state_with_function(&function)
            .map(Some)
            .map_err(MetalDeviceRuntimeError::contract)
    };
    let pointwise = pipeline(has_pointwise, "ferrum_tensor_pointwise_bf16")?;
    let linear = pipeline(has_linear, "ferrum_tensor_linear_bf16")?;
    let block_decode = pipeline(has_blocks, "ferrum_tensor_block_decode_bf16")?;
    if let Some(pipeline) = &linear {
        if pipeline.thread_execution_width() != 32
            || pipeline.max_total_threads_per_threadgroup() < 128
            || pipeline.static_threadgroup_memory_length()
                > runtime.device().max_threadgroup_memory_length()
        {
            return Err(MetalDeviceRuntimeError::contract(
                "Metal BF16 GEMM needs four 32-lane SIMD groups and its declared on-chip storage",
            ));
        }
    }
    let pipelines = Arc::new(Pipelines {
        pointwise,
        linear,
        block_decode,
        _retention: retention,
    });
    Ok(selected
        .iter()
        .copied()
        .zip(descriptors)
        .map(
            |(kind, descriptor)| -> Box<dyn OperationProvider<MetalDeviceRuntime>> {
                if kind.is_linear() {
                    return Box::new(linear::LinearProvider {
                        kind,
                        descriptor,
                        pipelines: Arc::clone(&pipelines),
                    });
                }
                Box::new(PointwiseProvider {
                    kind,
                    descriptor,
                    pipelines: Arc::clone(&pipelines),
                })
            },
        )
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
                let pipeline = pipelines
                    .pointwise
                    .as_ref()
                    .expect("selected pointwise pipeline");
                compute.set_compute_pipeline_state(pipeline);
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
                let threads = pipeline.max_total_threads_per_threadgroup().min(256);
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
