//! BF16 GEMM and native-block weights with core-admitted decode scratch.
use super::*;
use crate::gguf_blocks::GgufBlockFormat;
use ferrum_runtime::{
    PhysicalStorageLayout, PhysicalWeightLayout, PhysicalWeightPadding,
    ProviderWorkspaceRequirement, ProviderWorkspaceReusePolicy, ProviderWorkspaceScope,
    ProviderWorkspaceSizeFormula, ResolvedStorageComponent, ResolvedTensorSpec,
    WeightComponentRole, WeightEncoding,
};

pub(super) mod mpp;

pub(super) const DENSE_PROFILE: &str = "metal.mpp_interior_simdgroup_tails.f32.bf16.v3";
pub(super) const BLOCK_PROFILE: &str =
    "metal.native_block.decode_f32_rn_bf16.mpp_interior_simdgroup_tails_f32.v3";
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
        if self.kind != MetalTensorOperation::BlockLinear {
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
        let bias_mode = match self.kind {
            MetalTensorOperation::AffineLinear => 1,
            MetalTensorOperation::PatchProjection => 2,
            _ => 0,
        };
        if bias_mode != 0 {
            let bias = tensor_port_region(
                participant,
                ResolvedValueRole::Input,
                2,
                ElementType::Bf16,
                u64::from(shape.outputs) * 2,
            )?;
            disjoint(&bias, &regions[2])?;
            regions.push(bias);
        }
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
        let use_mpp = self.pipelines.linear_mpp.is_some()
            && mpp::supports_shape(shape.rows, shape.inputs, shape.outputs);
        let use_large_mpp = self.pipelines.linear_mpp_large.is_some()
            && mpp::supports_large_shape(shape.rows, shape.inputs, shape.outputs);
        let pipelines = Arc::clone(&self.pipelines);
        let parameters = LinearParams {
            rows: shape.rows,
            inputs: shape.inputs,
            outputs: shape.outputs,
            bias_mode,
            row_offset: 0,
            column_offset: 0,
        };
        let dispatches = linear_dispatches(shape.rows, shape.outputs, use_mpp, use_large_mpp);
        let work = DeviceCommandLogicalWork::tensor(shape.output_bytes / 2, shape.operations)
            .map_err(|e| e.to_string())?;
        MetalDeviceCommand::tensor_operation(
            match (format.is_some(), use_mpp || use_large_mpp) {
                (true, true) => "metal.native_block_linear.bf16.mpp",
                (true, false) => "metal.native_block_linear.bf16",
                (false, true) => "metal.dense_linear.bf16.mpp",
                (false, false) => "metal.dense_linear.bf16",
            },
            regions,
            work,
            move |encoder, regions| {
                let weight_index = if let Some(format) = format {
                    encoder.begin_compute_subwork("metal.linear.block_decode");
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
                    encoder.record_compute_dispatches(1);
                    // End the decode encoder before the dependent GEMM reads
                    // its tracked, invocation-owned scratch buffer.
                    encoder.end_compute();
                    3
                } else {
                    1
                };
                // These rectangles cover the output exactly once. They share
                // the admitted inputs/decode scratch and write disjoint output
                // ranges. Only explicit timing adds boundaries for attribution.
                for dispatch in dispatches {
                    if dispatch.row_groups == 0 || dispatch.column_groups == 0 {
                        continue;
                    }
                    let pipeline = match dispatch.kernel {
                        LinearKernel::Mpp32 => pipelines
                            .linear_mpp
                            .as_ref()
                            .expect("selected 32-row MPP linear pipeline"),
                        LinearKernel::Mpp64 => pipelines
                            .linear_mpp_large
                            .as_ref()
                            .expect("selected 64-row MPP linear pipeline"),
                        LinearKernel::Simd => {
                            pipelines.linear.as_ref().expect("selected linear pipeline")
                        }
                    };
                    encoder.begin_compute_subwork(match dispatch.kernel {
                        LinearKernel::Mpp32 => "metal.linear.gemm.mpp32",
                        LinearKernel::Mpp64 => "metal.linear.gemm.mpp64",
                        LinearKernel::Simd => "metal.linear.gemm.simd",
                    });
                    let compute = encoder.compute_encoder();
                    set_region(compute, 0, &regions[0]);
                    set_region(compute, 1, &regions[weight_index]);
                    set_region(compute, 2, &regions[2]);
                    // The no-bias mode never reads its typed placeholder.
                    set_region(compute, 4, &regions[if bias_mode == 0 { 0 } else { 3 }]);
                    compute.set_compute_pipeline_state(pipeline);
                    set_params(
                        compute,
                        3,
                        &LinearParams {
                            row_offset: dispatch.row_offset,
                            column_offset: dispatch.column_offset,
                            ..parameters
                        },
                    );
                    compute.dispatch_thread_groups(
                        MTLSize::new(
                            u64::from(dispatch.row_groups),
                            u64::from(dispatch.column_groups),
                            1,
                        ),
                        MTLSize::new(128, 1, 1),
                    );
                    encoder.record_compute_dispatches(1);
                }
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
    bias_mode: u32,
    row_offset: u32,
    column_offset: u32,
}

// These are the actual fallback shader's dimensions, independent of MPP tiles.
const SIMD_TILE_ROWS: u32 = 32;
const SIMD_TILE_COLUMNS: u32 = 64;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum LinearKernel {
    #[default]
    Simd,
    Mpp32,
    Mpp64,
}

#[derive(Clone, Copy, Debug, Default)]
struct LinearDispatch {
    row_offset: u32,
    column_offset: u32,
    row_groups: u32,
    column_groups: u32,
    kernel: LinearKernel,
}

fn linear_dispatches(
    rows: u32,
    outputs: u32,
    use_mpp: bool,
    use_large_mpp: bool,
) -> [LinearDispatch; 4] {
    let mut dispatches = [LinearDispatch::default(); 4];
    if !use_mpp && !use_large_mpp {
        dispatches[0] = LinearDispatch {
            row_groups: rows.div_ceil(SIMD_TILE_ROWS),
            column_groups: outputs.div_ceil(SIMD_TILE_COLUMNS),
            ..LinearDispatch::default()
        };
        return dispatches;
    }
    let column_groups = outputs / mpp::TILE_COLUMNS;
    let large_groups = if use_large_mpp {
        rows / mpp::LARGE_TILE_ROWS
    } else {
        0
    };
    dispatches[0] = LinearDispatch {
        row_groups: large_groups,
        column_groups,
        kernel: LinearKernel::Mpp64,
        ..LinearDispatch::default()
    };
    let large_rows = large_groups * mpp::LARGE_TILE_ROWS;
    let small_groups = if use_mpp {
        (rows - large_rows) / mpp::TILE_ROWS
    } else {
        0
    };
    dispatches[1] = LinearDispatch {
        row_offset: large_rows,
        row_groups: small_groups,
        column_groups,
        kernel: LinearKernel::Mpp32,
        ..LinearDispatch::default()
    };
    let complete_rows = large_rows + small_groups * mpp::TILE_ROWS;
    // The bottom SIMD strip owns the bottom-right corner as well. It can
    // include a full 32-row tile if only the large MPP pipeline is available.
    dispatches[2] = LinearDispatch {
        row_offset: complete_rows,
        row_groups: (rows - complete_rows).div_ceil(SIMD_TILE_ROWS),
        column_groups: outputs.div_ceil(SIMD_TILE_COLUMNS),
        ..LinearDispatch::default()
    };
    dispatches[3] = LinearDispatch {
        column_offset: column_groups * mpp::TILE_COLUMNS,
        row_groups: complete_rows / SIMD_TILE_ROWS,
        column_groups: u32::from(outputs % SIMD_TILE_COLUMNS != 0),
        ..LinearDispatch::default()
    };
    dispatches
}

#[cfg(test)]
mod dispatch_tests {
    use super::*;

    #[test]
    fn interior_and_tail_dispatches_cover_each_output_once() {
        for rows in [0, 1, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129] {
            for outputs in [0, 1, 63, 64, 65, 127, 128, 129] {
                for (small_available, large_available) in
                    [(false, false), (true, false), (false, true), (true, true)]
                {
                    let use_mpp = small_available && mpp::supports_shape(rows, 129, outputs);
                    let use_large_mpp =
                        large_available && mpp::supports_large_shape(rows, 129, outputs);
                    let mut visits = vec![0_u8; (rows * outputs) as usize];
                    let dispatches = linear_dispatches(rows, outputs, use_mpp, use_large_mpp);
                    assert_eq!(
                        dispatches
                            .iter()
                            .any(|dispatch| dispatch.kernel == LinearKernel::Mpp64
                                && dispatch.row_groups > 0
                                && dispatch.column_groups > 0),
                        use_large_mpp,
                        "available large tiles must execute the complete interior"
                    );
                    for dispatch in dispatches {
                        if dispatch.row_groups == 0 || dispatch.column_groups == 0 {
                            continue;
                        }
                        // Independently model the actual shader extents, including
                        // the unchanged 32x64 SIMD fallback, rather than reusing
                        // the dispatch arithmetic to manufacture expected visits.
                        let tile_rows = match dispatch.kernel {
                            LinearKernel::Simd => 32,
                            LinearKernel::Mpp32 => {
                                assert!(use_mpp);
                                32
                            }
                            LinearKernel::Mpp64 => {
                                assert!(use_large_mpp);
                                64
                            }
                        };
                        let row_end = dispatch.row_offset + dispatch.row_groups * tile_rows;
                        let column_end = dispatch.column_offset + dispatch.column_groups * 64;
                        if dispatch.kernel != LinearKernel::Simd {
                            assert!(row_end <= rows && column_end <= outputs);
                        }
                        for row in dispatch.row_offset..row_end.min(rows) {
                            for column in dispatch.column_offset..column_end.min(outputs) {
                                visits[(row * outputs + column) as usize] += 1;
                            }
                        }
                    }
                    assert!(visits.into_iter().all(|count| count == 1), "rows={rows} outputs={outputs} small={small_available} large={large_available}");
                }
            }
        }
    }
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
    if matches!(
        kind,
        MetalTensorOperation::AffineLinear | MetalTensorOperation::PatchProjection
    ) {
        let g = primitives::geometry(kind, values, attributes)?;
        // The original rank-five patch ABI was checked above; only its dense
        // physical reduction axis is flattened for this GEMM's launch shape.
        let weight = ResolvedTensorSpec::new(
            vec![g.columns, g.inner],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .map_err(|e| e.to_string())?;
        let mut shape = validate_specs(
            binding(values, ResolvedValueRole::Input, 0)?.tensor(),
            &weight,
            binding(values, ResolvedValueRole::Output, 0)?.tensor(),
            g.inner,
            g.columns,
        )?;
        shape.operations = shape
            .operations
            .checked_add(g.elements())
            .ok_or("Metal affine work overflows")?;
        return Ok((shape, None));
    }
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
