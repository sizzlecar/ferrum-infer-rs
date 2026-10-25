//! Explicit immutable U32 seed/offset -> admitted BF16 NCHW noise.
//!
//! The 64-bit seed is the Philox4x32-10 key. Offset counts flat normal draws:
//! counter=[low32(floor((offset+i)/4)), high32(...), 0, 0], with no u64
//! addition wrap; pairs use words [0,1]/[2,3], cos then sin. Uniforms are
//! (high23(word)+0.5)*2^-23, strictly inside (0,1). Box-Muller uses separate
//! FP32 log/sqrt/angle/trig/multiply and a final BF16 RNE. Mapping is independent
//! of shape and launch width, not CUDA/Torch's device-dependent mapping.
//! No state array, hidden allocator, global RNG, state mutation or global scratch.
use super::*;
use ferrum_runtime::PhiloxNormalBf16Geometry;

pub(super) const PROFILE: &str = "metal.philox4x32_10.normal.bf16.flat_u23_box_muller.v1";

pub(super) fn geometry(
    values: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<PhiloxNormalBf16Geometry, String> {
    if values.len() != 2 || !attributes.is_empty() || values.iter().any(|v| v.weight().is_some()) {
        return Err(
            "Metal noise requires one explicit state input, one output and no weights/attributes"
                .into(),
        );
    }
    PhiloxNormalBf16Geometry::new(
        binding(values, ResolvedValueRole::Input, 0)?.tensor(),
        binding(values, ResolvedValueRole::Output, 0)?.tensor(),
    )
    .map_err(|e| e.to_string())
}

#[repr(C)]
struct Params {
    elements: u64,
    grid_width: u64,
}

pub(super) fn encode(
    participant: &TensorParticipant<'_>,
    pipelines: Arc<Pipelines>,
) -> Result<MetalDeviceCommand, String> {
    let g = geometry(participant.bindings(), participant.attributes())?;
    let state = tensor_port_region(
        participant,
        ResolvedValueRole::Input,
        0,
        ElementType::U32,
        16,
    )?;
    let output = tensor_port_region(
        participant,
        ResolvedValueRole::Output,
        0,
        ElementType::Bf16,
        g.output_bytes(),
    )?;
    disjoint(&state, &output)?;
    let params = Params {
        elements: g.element_count(),
        grid_width: (g.element_count() + 3).div_ceil(4).min(65535),
    };
    let work =
        DeviceCommandLogicalWork::tensor(g.element_count(), g.logical_arithmetic_operations())
            .map_err(|e| e.to_string())?;
    MetalDeviceCommand::tensor_operation(
        "metal.philox_normal.bf16",
        vec![state, output],
        work,
        move |encoder, regions| {
            let compute = encoder.compute_encoder();
            let pipeline = pipelines.philox.as_ref().expect("selected Philox pipeline");
            compute.set_compute_pipeline_state(pipeline);
            set_region(compute, 0, &regions[0]);
            set_region(compute, 1, &regions[1]);
            set_params(compute, 2, &params);
            compute.dispatch_threads(
                MTLSize::new(params.grid_width, 1, 1),
                MTLSize::new(pipeline.max_total_threads_per_threadgroup().min(256), 1, 1),
            );
            Ok(())
        },
    )
    .map_err(|e| e.to_string())
}
