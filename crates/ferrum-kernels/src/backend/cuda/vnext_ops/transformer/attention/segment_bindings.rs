//! Closed recurrent-state address rule consumed by the whole-segment encoder.
use super::super::segment_bindings::{self as segment, DynamicRecipe, ValidationRecipe};
use super::*;
use ferrum_interfaces::vnext::{
    PreparedSegmentBindingNode, SegmentBindingDeclaration, SegmentBindingRegionExtent,
    SegmentBindingRegionRequest, SegmentBindingRegionSelector,
};

pub(in crate::backend::cuda::vnext_ops) struct GatedDeltaRecipe {
    participants: usize,
    conv_bytes: u64,
    delta_bytes: u64,
}

pub(super) fn declare(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    validations: Vec<ValidationRecipe>,
) -> Result<Option<SegmentBindingDeclaration>, String> {
    if invocation.program_binding().is_none() {
        return Ok(None);
    }
    let first = &invocation.participants()[0];
    let conv = contiguous_region(
        first,
        binding(first.bindings(), ResolvedValueRole::Input, 8)?,
        ElementType::F16,
    )?;
    let delta = contiguous_region(
        first,
        binding(first.bindings(), ResolvedValueRole::Input, 9)?,
        ElementType::F32,
    )?;
    let layout = StateBindingLayout::new(invocation.participants().len())?;
    let request = |selector, bytes, element_type, alignment_bytes| SegmentBindingRegionRequest {
        selector,
        offset_bytes: 0,
        extent: SegmentBindingRegionExtent::Exact(bytes),
        element_type,
        alignment_bytes,
    };
    let regions = vec![
        request(
            SegmentBindingRegionSelector::Value {
                role: ResolvedValueRole::Input,
                ordinal: 8,
                component: None,
            },
            conv.length_bytes(),
            ElementType::F16,
            2,
        ),
        request(
            SegmentBindingRegionSelector::Value {
                role: ResolvedValueRole::Input,
                ordinal: 9,
                component: None,
            },
            delta.length_bytes(),
            ElementType::F32,
            4,
        ),
        request(
            SegmentBindingRegionSelector::ProgramBinding,
            layout.required_bytes,
            ElementType::U8,
            1,
        ),
    ];
    segment::declaration(
        DynamicRecipe::GatedDelta(GatedDeltaRecipe {
            participants: invocation.participants().len(),
            conv_bytes: conv.length_bytes(),
            delta_bytes: delta.length_bytes(),
        }),
        regions,
        validations,
    )
    .map(Some)
}

pub(in crate::backend::cuda::vnext_ops) fn encode(
    node: &PreparedSegmentBindingNode<'_, CudaDeviceBuffer>,
    recipe: &GatedDeltaRecipe,
) -> Result<EncodedReusableExecutionBindings<CudaDeviceCommand>, String> {
    if node.participant_count() != recipe.participants || recipe.participants == 0 {
        return Err("segment recurrent participant shape differs from cold geometry".into());
    }
    let destination = segment::shared_region(node, 2)?;
    let layout = StateBindingLayout::new(recipe.participants)?;
    let mut writes = Vec::with_capacity(recipe.participants);
    let mut retained = Vec::with_capacity(recipe.participants.saturating_mul(2));
    for participant in 0..recipe.participants {
        let conv = segment::contiguous_region(node, participant, 0)?;
        let delta = segment::contiguous_region(node, participant, 1)?;
        if conv.element_type() != ElementType::F16
            || conv.length_bytes() != recipe.conv_bytes
            || delta.element_type() != ElementType::F32
            || delta.length_bytes() != recipe.delta_bytes
        {
            return Err("segment recurrent state differs from admitted geometry".into());
        }
        writes.push(
            super::super::CudaProgramBindingWrite::new(
                layout.offset(participant)?,
                state_binding_payload(&conv, &delta),
            )
            .map_err(|e| e.to_string())?,
        );
        retained.push(conv);
        retained.push(delta);
    }
    let participants =
        u32::try_from(recipe.participants).map_err(|_| "segment participant count exceeds u32")?;
    let binding = node
        .program_binding()
        .ok_or("segment recurrent binding slot is absent")?
        .clone();
    let command = CudaDeviceCommand::program_binding_patch(
        "vnext_gated_delta_recurrent_attention_bindings",
        binding,
        destination,
        writes,
        retained,
    )
    .and_then(|c| {
        c.with_work_attribution(
            DeviceBatchingForm::ParticipantLoop,
            participants,
            node.work_shape().immediate_tokens(),
            0,
            u64::from(participants),
        )
    })
    .map_err(|e| e.to_string())?;
    Ok(EncodedReusableExecutionBindings::empty().with_program_binding(command))
}

pub(in crate::backend::cuda::vnext_ops) fn prepare_indexed(
    node: &PreparedSegmentBindingNode<'_, CudaDeviceBuffer>,
    recipe: &GatedDeltaRecipe,
    owners: &mut crate::backend::cuda::vnext_runtime::CudaSegmentOwnerBuilder<'_>,
) -> Result<segment::PendingBinding, String> {
    if node.participant_count() != recipe.participants || recipe.participants == 0 {
        return Err("segment recurrent participant shape differs from cold geometry".into());
    }
    let destination = segment::shared_borrowed_region(node, 2)?;
    let layout = StateBindingLayout::new(recipe.participants)?;
    let mut writes = Vec::with_capacity(recipe.participants);
    let mut ranges = Vec::with_capacity(recipe.participants.saturating_mul(2));
    for participant in 0..recipe.participants {
        let conv = segment::indexed_contiguous_region(node, participant, 0, owners)?;
        let delta = segment::indexed_contiguous_region(node, participant, 1, owners)?;
        let conv_view = owners.region(&conv).map_err(|e| e.to_string())?;
        let delta_view = owners.region(&delta).map_err(|e| e.to_string())?;
        if conv_view.element_type() != ElementType::F16
            || conv_view.length_bytes() != recipe.conv_bytes
            || delta_view.element_type() != ElementType::F32
            || delta_view.length_bytes() != recipe.delta_bytes
        {
            return Err("segment recurrent state differs from admitted geometry".into());
        }
        let mut payload = Vec::with_capacity(STATE_BINDING_SLOT_BYTES as usize);
        payload.extend_from_slice(&conv_view.device_ptr().to_le_bytes());
        payload.extend_from_slice(&delta_view.device_ptr().to_le_bytes());
        writes.push(
            super::super::CudaProgramBindingWrite::new(
                layout.offset(participant)?,
                payload.into_boxed_slice(),
            )
            .map_err(|e| e.to_string())?,
        );
        ranges.push(conv);
        ranges.push(delta);
    }
    Ok(segment::PendingBinding {
        operation: "vnext_gated_delta_recurrent_attention_bindings",
        binding: node
            .program_binding()
            .ok_or("segment recurrent binding slot is absent")?
            .clone(),
        destination,
        writes,
        ranges,
        numerical_status: None,
        participants: u32::try_from(recipe.participants)
            .map_err(|_| "segment participant count exceeds u32")?,
        tokens: node.work_shape().immediate_tokens(),
    })
}
