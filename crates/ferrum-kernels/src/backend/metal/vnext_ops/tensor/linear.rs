//! BF16 GEMM and native-block weights with core-admitted decode scratch.
use super::*;
use crate::gguf_blocks::GgufBlockFormat;
use ferrum_runtime::{
    PhysicalStorageLayout, PhysicalWeightLayout, PhysicalWeightPadding,
    ProviderWorkspaceRequirement, ProviderWorkspaceReusePolicy, ProviderWorkspaceScope,
    ProviderWorkspaceSizeFormula, ResolvedStorageComponent, ResolvedTensorSpec,
    WeightComponentRole, WeightEncoding,
};

pub(super) const DENSE_PROFILE: &str = "metal.simdgroup.f32.bf16.v1";
pub(super) const BLOCK_PROFILE: &str = "metal.native_block.decode_f32_rn_bf16.simdgroup_f32.v1";
pub(super) const FORMATS: [GgufBlockFormat; 4] = [
    GgufBlockFormat::Q8_0,
    GgufBlockFormat::Q4K,
    GgufBlockFormat::Q5K,
    GgufBlockFormat::Q6K,
];

pub(super) struct LinearProvider {
    pub kind: MetalTensorOperation,
    pub descriptor: OperationProviderDescriptor,
    pub pipelines: Arc<Pipelines>,
}

impl OperationResourceEstimator for LinearProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }
    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        let estimate =
            estimate_without_workspace(&self.descriptor, &request, self.kind.operation())?;
        let (shape, _) =
            geometry(self.kind, request.values(), request.attributes()).map_err(invalid_plan)?;
        if self.kind == MetalTensorOperation::DenseLinear {
            return Ok(estimate);
        }
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

impl OperationProvider<MetalDeviceRuntime> for LinearProvider {
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
            "metal.tensor.linear",
            "Metal BF16 linear requires fixed tensor work".into(),
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
            .map_err(|reason| provider_failure(identity, "metal.tensor.linear", reason))
    }
}

impl LinearProvider {
    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, MetalDeviceBuffer>,
    ) -> Result<MetalDeviceCommand, String> {
        let [participant] = invocation.participants() else {
            return Err("Metal linear requires one tensor participant".into());
        };
        if invocation.operation().id.as_str() != self.kind.operation()
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("Metal linear operation or work contract differs".into());
        }
        let (shape, format) =
            geometry(self.kind, participant.bindings(), participant.attributes())?;
        let input = contiguous_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
            ElementType::Bf16,
        )?;
        validate_region(&input, shape.input_bytes)?;
        let weight_binding = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
        let weight = contiguous_region(
            participant,
            weight_binding,
            if format.is_some() {
                ElementType::U8
            } else {
                ElementType::Bf16
            },
        )?;
        if format.is_none() {
            validate_region(&weight, shape.weight_bytes)?;
        }
        let output = contiguous_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            ElementType::Bf16,
        )?;
        validate_region(&output, shape.output_bytes)?;
        for read in [&input, &weight] {
            disjoint(read, &output)?;
        }
        let mut regions = vec![input, weight, output];
        if format.is_some() {
            let scratch = contiguous_scratch_region(participant, shape.weight_bytes)?;
            if !scratch.offset_bytes().is_multiple_of(2) {
                return Err("Metal decoded weights require BF16 alignment".into());
            }
            for read in &regions {
                disjoint(read, &scratch)?;
            }
            regions.push(scratch);
        }
        let pipelines = Arc::clone(&self.pipelines);
        let parameters = LinearParams {
            rows: shape.rows,
            inputs: shape.inputs,
            outputs: shape.outputs,
            reserved: 0,
        };
        let work = DeviceCommandLogicalWork::tensor(shape.output_bytes / 2, shape.operations)
            .map_err(|e| e.to_string())?;
        MetalDeviceCommand::tensor_operation(
            if format.is_some() {
                "metal.native_block_linear.bf16"
            } else {
                "metal.dense_linear.bf16"
            },
            regions,
            work,
            move |encoder, regions| {
                let weight_index = if let Some(format) = format {
                    let decode = pipelines
                        .block_decode
                        .as_ref()
                        .expect("selected block decoder");
                    let count = shape.weight_bytes / 2;
                    let grid_width = count.min(1 << 20);
                    let params = BlockDecodeParams {
                        elements: count,
                        grid_width,
                        format: format.ggml_type_id(),
                        values: format.block_values() as u32,
                        bytes: format.block_bytes() as u32,
                        reserved: 0,
                    };
                    let compute = encoder.compute_encoder();
                    compute.set_compute_pipeline_state(decode);
                    set_region(compute, 0, &regions[1]);
                    set_region(compute, 1, &regions[3]);
                    set_params(compute, 2, &params);
                    compute.dispatch_threads(
                        MTLSize::new(grid_width, 1, 1),
                        MTLSize::new(decode.max_total_threads_per_threadgroup().min(256), 1, 1),
                    );
                    // End the decode encoder before the dependent GEMM reads
                    // its tracked, invocation-owned scratch buffer.
                    encoder.end_compute();
                    3
                } else {
                    1
                };
                let compute = encoder.compute_encoder();
                compute.set_compute_pipeline_state(
                    pipelines.linear.as_ref().expect("selected linear pipeline"),
                );
                set_region(compute, 0, &regions[0]);
                set_region(compute, 1, &regions[weight_index]);
                set_region(compute, 2, &regions[2]);
                set_params(compute, 3, &parameters);
                compute.dispatch_thread_groups(
                    MTLSize::new(
                        u64::from(shape.rows).div_ceil(32),
                        u64::from(shape.outputs).div_ceil(64),
                        1,
                    ),
                    MTLSize::new(128, 1, 1),
                );
                Ok(())
            },
        )
        .map_err(|error| error.to_string())
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LinearParams {
    rows: u32,
    inputs: u32,
    outputs: u32,
    reserved: u32,
}
#[repr(C)]
struct BlockDecodeParams {
    elements: u64,
    grid_width: u64,
    format: u32,
    values: u32,
    bytes: u32,
    reserved: u32,
}
#[derive(Clone, Copy)]
struct Shape {
    rows: u32,
    inputs: u32,
    outputs: u32,
    input_bytes: u64,
    weight_bytes: u64,
    output_bytes: u64,
    operations: u64,
}

fn geometry(
    kind: MetalTensorOperation,
    values: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<(Shape, Option<GgufBlockFormat>), String> {
    if values.len() != 3 || attributes.len() != 2 {
        return Err("Metal linear requires three ports and two shape attributes".into());
    }
    let input = binding(values, ResolvedValueRole::Input, 0)?;
    let weight = binding(values, ResolvedValueRole::Input, 1)?;
    let output = binding(values, ResolvedValueRole::Output, 0)?;
    let shape = validate_specs(
        input.tensor(),
        weight.tensor(),
        output.tensor(),
        unsigned_attribute(attributes, "in_features")?,
        unsigned_attribute(attributes, "out_features")?,
    )?;
    let format = match kind {
        MetalTensorOperation::BlockLinear => Some(packed_component(weight)?.1),
        MetalTensorOperation::DenseLinear => {
            if let Some(physical) = weight.weight() {
                physical
                    .validate_logical(weight.tensor().dimensions(), ElementType::Bf16)
                    .map_err(|e| e.to_string())?;
                if !matches!(
                    physical.physical_layout(),
                    PhysicalWeightLayout::Dense { .. }
                ) {
                    return Err("Metal BF16 dense cannot read packed weights".into());
                }
            }
            None
        }
        _ => return Err("Metal linear estimator received a non-matrix operation".into()),
    };
    Ok((shape, format))
}

fn validate_specs(
    input: &ResolvedTensorSpec,
    weight: &ResolvedTensorSpec,
    output: &ResolvedTensorSpec,
    inputs: u64,
    outputs: u64,
) -> Result<Shape, String> {
    let [rows, width] = input.dimensions() else {
        return Err("Metal linear requires rank-two input".into());
    };
    if *width != inputs
        || weight.dimensions() != [outputs, inputs]
        || output.dimensions() != [*rows, outputs]
        || [input, weight, output].iter().any(|spec| {
            spec.element_type() != ElementType::Bf16
                || spec.layout() != &ResolvedTensorLayout::Contiguous
        })
    {
        return Err("Metal linear requires contiguous BF16 X[M,K], W[N,K], Y[M,N]".into());
    }
    let dimension = |n| {
        u32::try_from(n)
            .ok()
            .filter(|n| *n > 0)
            .ok_or("Metal linear dimensions must fit positive u32")
    };
    let bytes = |a: u64, b: u64| {
        a.checked_mul(b)
            .and_then(|v| v.checked_mul(2))
            .filter(|v| *v <= isize::MAX as u64)
            .ok_or("Metal linear extent exceeds address space")
    };
    Ok(Shape {
        rows: dimension(*rows)?,
        inputs: dimension(inputs)?,
        outputs: dimension(outputs)?,
        input_bytes: bytes(*rows, inputs)?,
        weight_bytes: bytes(outputs, inputs)?,
        output_bytes: bytes(*rows, outputs)?,
        operations: rows
            .checked_mul(outputs)
            .and_then(|v| v.checked_mul(inputs))
            .and_then(|v| v.checked_mul(2))
            .ok_or("Metal linear work overflows")?,
    })
}

fn packed_component(
    binding: &ResolvedValueBinding,
) -> Result<(&ResolvedStorageComponent, GgufBlockFormat), String> {
    let weight = binding
        .weight()
        .ok_or("Metal native block weights need a typed physical layout")?;
    weight
        .validate_logical(binding.tensor().dimensions(), ElementType::Bf16)
        .map_err(|e| e.to_string())?;
    let PhysicalWeightLayout::BlockQuantized {
        blocks,
        block_axis: 1,
        block_padding: PhysicalWeightPadding::Exact,
    } = weight.physical_layout()
    else {
        return Err("Metal native block linear requires exact OI blocks".into());
    };
    if !matches!(
        blocks.storage,
        PhysicalStorageLayout::Contiguous {
            padding: PhysicalWeightPadding::Exact
        }
    ) {
        return Err("Metal native block storage must be exact and contiguous".into());
    }
    let [component] = weight.components() else {
        return Err("Metal block weight needs one component".into());
    };
    let WeightEncoding::BlockQuantized(spec) = component.encoding() else {
        return Err("Metal block component lacks block ABI".into());
    };
    let format = GgufBlockFormat::from_spec(spec)?;
    if !FORMATS.contains(&format)
        || component.role() != WeightComponentRole::PackedValues
        || component.component_id() != &blocks.component_id
    {
        return Err("Metal block format or component role is unsupported".into());
    }
    let [stored] = binding.storage().components() else {
        return Err("Metal block weight needs one storage range".into());
    };
    if stored.component_id() != Some(component.component_id())
        || stored.element_type() != ElementType::U8
        || stored.length_bytes() != component.physical_bytes().map_err(|e| e.to_string())?
    {
        return Err("Metal block storage identity/type/length differs".into());
    }
    Ok((stored, format))
}
