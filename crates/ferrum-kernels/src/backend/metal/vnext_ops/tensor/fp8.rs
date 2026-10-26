//! Explicit W8A8 projection: packed weights stay resident, invocation-owned
//! scratch holds rowwise E4M3FN activations and F32 scales. No dense weight copy.
use super::*;
use ferrum_runtime::{
    DenseLinearFp8Bf16Geometry, ProviderWorkspaceRequirement, ProviderWorkspaceReusePolicy,
    ProviderWorkspaceScope, ProviderWorkspaceSizeFormula, ResolvedStorageComponent,
};

mod storage;
pub(super) const PROFILE: &str = "metal.simdgroup.f32.w8a8.bf16.rowwise_e4m3fn.v1";

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    rows: u32,
    inputs: u32,
    outputs: u32,
    reserved: u32,
    scale_offset: u64,
}
struct Geometry {
    params: Params,
    scratch: u64,
    work: DenseLinearFp8Bf16Geometry,
}
fn geometry(
    values: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Geometry, String> {
    let x = binding(values, ResolvedValueRole::Input, 0)?;
    let w = binding(values, ResolvedValueRole::Input, 1)?;
    let y = binding(values, ResolvedValueRole::Output, 0)?;
    let work = DenseLinearFp8Bf16Geometry::new(
        x.tensor(),
        w.tensor(),
        y.tensor(),
        unsigned_attribute(attributes, "in_features")?,
        unsigned_attribute(attributes, "out_features")?,
    )
    .map_err(|e| e.to_string())?;
    storage::validate(w)?;
    let activation_bytes = work
        .rows()
        .checked_mul(work.inputs())
        .ok_or("FP8 activation extent overflows")?;
    let scale_offset = activation_bytes
        .checked_add(15)
        .ok_or("FP8 alignment overflows")?
        & !15;
    let scratch = work
        .rows()
        .checked_mul(4)
        .and_then(|n| n.checked_add(scale_offset))
        .filter(|&n| n <= isize::MAX as u64)
        .ok_or("FP8 scratch overflows address space")?;
    Ok(Geometry {
        params: Params {
            rows: checked_u32(work.rows(), "FP8 rows")?,
            inputs: checked_u32(work.inputs(), "FP8 inputs")?,
            outputs: checked_u32(work.outputs(), "FP8 outputs")?,
            reserved: 0,
            scale_offset,
        },
        scratch,
        work,
    })
}

pub(super) fn estimate(
    descriptor: &OperationProviderDescriptor,
    request: &OperationResourceEstimateRequest<'_>,
) -> Result<OperationResourceEstimate, VNextError> {
    estimate_without_workspace(
        descriptor,
        request,
        MetalTensorOperation::Fp8Linear.operation(),
    )?;
    let shape = geometry(request.values(), request.attributes()).map_err(invalid_plan)?;
    Ok(OperationResourceEstimate::new(
        descriptor.resource_estimator_id(),
        descriptor.resource_estimator_version(),
        descriptor.resource_estimator_implementation_fingerprint(),
        request.input_fingerprint(),
        16,
        Some(ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::fixed(shape.scratch)?,
            16,
            ProviderWorkspaceScope::Invocation,
            ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
            DynamicStorageRequirement::contiguous(),
        )?),
        None,
    ))
}

fn component(
    participant: &TensorParticipant<'_>,
    component: &ResolvedStorageComponent,
    alignment: u64,
) -> Result<MetalBufferRegion, String> {
    let view = participant
        .views()
        .iter()
        .find(|v| v.resource_id() == component.resource_id())
        .ok_or("FP8 component has no resource view")?;
    let translated = view
        .translate(component.offset_bytes(), component.length_bytes())
        .map_err(|e| e.to_string())?;
    let mut parts = translated.iter();
    let part = parts.next().ok_or("FP8 component is empty")?;
    if parts.next().is_some() {
        return Err("FP8 requires contiguous physical storage".into());
    }
    let (buffer, range, retention) = part.buffer_and_physical_range();
    let region = buffer
        .retained_region(range, retention)
        .map_err(|e| e.to_string())?;
    if region.element_type() != component.element_type()
        || region.length_bytes() != component.length_bytes()
        || !region.offset_bytes().is_multiple_of(alignment)
    {
        return Err("FP8 physical component extent, dtype or alignment differs".into());
    }
    Ok(region)
}

pub(super) fn encode(
    participant: &TensorParticipant<'_>,
    pipelines: Arc<Pipelines>,
) -> Result<MetalDeviceCommand, String> {
    let shape = geometry(participant.bindings(), participant.attributes())?;
    let weight = storage::validate(binding(
        participant.bindings(),
        ResolvedValueRole::Input,
        1,
    )?)?;
    let input = tensor_port_region(
        participant,
        ResolvedValueRole::Input,
        0,
        ElementType::Bf16,
        u64::from(shape.params.rows) * u64::from(shape.params.inputs) * 2,
    )?;
    let packed = component(participant, weight.packed, 16)?;
    let scale = component(participant, weight.scale, 4)?;
    let output = tensor_port_region(
        participant,
        ResolvedValueRole::Output,
        0,
        ElementType::Bf16,
        shape.work.output_elements() * 2,
    )?;
    let scratch = contiguous_scratch_region(participant, shape.scratch)?;
    if !scratch.offset_bytes().is_multiple_of(16) {
        return Err("FP8 scratch alignment differs".into());
    }
    for read in [&input, &packed, &scale] {
        disjoint(read, &output)?;
        disjoint(read, &scratch)?;
    }
    disjoint(&output, &scratch)?;
    disjoint(&packed, &scale)?;
    let work = DeviceCommandLogicalWork::tensor(
        shape.work.output_elements(),
        shape.work.logical_arithmetic_operations(),
    )
    .map_err(|e| e.to_string())?;
    let p = shape.params;
    MetalDeviceCommand::tensor_operation(
        "metal.dense_linear_fp8.bf16",
        vec![input, packed, scale, output, scratch],
        work,
        move |encoder, regions| {
            let quantize = pipelines
                .fp8_quantize
                .as_ref()
                .expect("selected FP8 quantizer");
            let compute = encoder.compute_encoder();
            compute.set_compute_pipeline_state(quantize);
            set_region(compute, 0, &regions[0]);
            set_region(compute, 1, &regions[4]);
            set_params(compute, 2, &p);
            compute.dispatch_thread_groups(
                MTLSize::new(u64::from(p.rows), 1, 1),
                MTLSize::new(256, 1, 1),
            );
            encoder.end_compute();
            let compute = encoder.compute_encoder();
            compute.set_compute_pipeline_state(
                pipelines.fp8_linear.as_ref().expect("selected FP8 GEMM"),
            );
            for (slot, index) in [4, 1, 3].into_iter().enumerate() {
                set_region(compute, slot as u64, &regions[index]);
            }
            set_params(compute, 3, &p);
            set_region(compute, 4, &regions[2]);
            compute.dispatch_thread_groups(
                MTLSize::new(
                    u64::from(p.rows).div_ceil(32),
                    u64::from(p.outputs).div_ceil(64),
                    1,
                ),
                MTLSize::new(128, 1, 1),
            );
            Ok(())
        },
    )
    .map_err(|e| e.to_string())
}
