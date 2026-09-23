use super::*;

pub(in crate::executor::vnext_executor) struct ProductUploadLayouts {
    pub(in crate::executor::vnext_executor) mask: HostTransferLayout,
    pub(in crate::executor::vnext_executor) repetition_ids: HostTransferLayout,
    pub(in crate::executor::vnext_executor) offsets: HostTransferLayout,
    pub(in crate::executor::vnext_executor) penalty: HostTransferLayout,
}

pub(in crate::executor::vnext_executor) fn product_upload_layouts(
    io: &VNextIoBinding,
) -> std::result::Result<ProductUploadLayouts, VNextError> {
    let invalid = || VNextError::InvalidExecutionPlan {
        reason: "product upload layout exceeds u64".to_owned(),
    };
    Ok(ProductUploadLayouts {
        mask: HostTransferLayout::new(
            ElementType::U8,
            u64::try_from(io.output_elements).map_err(|_| invalid())?,
        )?,
        repetition_ids: HostTransferLayout::new(
            ElementType::U32,
            u64::try_from(io.repetition_capacity).map_err(|_| invalid())?,
        )?,
        offsets: HostTransferLayout::new(ElementType::U32, 2)?,
        penalty: HostTransferLayout::new(ElementType::F32, 1)?,
    })
}

pub(in crate::executor::vnext_executor) fn product_token_upload_layout(
    offset: u64,
    count: u64,
) -> std::result::Result<(u64, HostTransferLayout), VNextError> {
    let offset = offset
        .checked_mul(ElementType::U32.size_bytes())
        .ok_or_else(|| VNextError::InvalidExecutionPlan {
            reason: "vNext token upload offset overflows u64".to_owned(),
        })?;
    Ok((offset, HostTransferLayout::new(ElementType::U32, count)?))
}

pub(in crate::executor::vnext_executor) fn product_readback_binding(
    io: &VNextIoBinding,
    output: VNextProductOutputMode,
) -> (&NodeId, &ResourceId, u64, HostTransferLayout) {
    match output {
        VNextProductOutputMode::FullLogits => (
            &io.output_node_id,
            &io.output_resource_id,
            io.output_offset_bytes,
            io.output_layout,
        ),
        VNextProductOutputMode::GreedyToken => (
            &io.greedy_token_output_node_id,
            &io.greedy_token_output_resource_id,
            io.greedy_token_output_offset_bytes,
            io.greedy_token_output_layout,
        ),
    }
}

/// Same product input order as dispatch_participant_wave: tokens, full masks,
/// optional repetition IDs, offsets, penalties. This initial route has no
/// repetition IDs; offsets and penalty uploads still occur on every row.
pub(super) fn input_uploads<'a>(
    io: &'a VNextIoBinding,
    rows: &[OperationCostWorkRow],
) -> std::result::Result<Vec<EagerCoreInputUpload<'a>>, U> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(rows.len().checked_mul(4).ok_or(U::Capacity)?)
        .map_err(|_| U::Capacity)?;
    for (index, row) in rows.iter().enumerate() {
        let (logical_offset_bytes, layout) =
            product_token_upload_layout(row.offset, row.count.get())
                .map_err(|_| U::InvalidInput)?;
        result.push(EagerCoreInputUpload {
            node_id: &io.input_node_id,
            input_ordinal: io.input_ordinal,
            participant_index: index,
            logical_offset_bytes,
            layout,
        });
    }
    let ProductUploadLayouts {
        mask,
        offsets,
        penalty,
        ..
    } = product_upload_layouts(io).map_err(|_| U::InvalidInput)?;
    for (node, ordinal, layout) in [
        (
            &io.token_mask_input_node_id,
            io.token_mask_input_ordinal,
            mask,
        ),
        (
            &io.repetition_offsets_input_node_id,
            io.repetition_offsets_input_ordinal,
            offsets,
        ),
        (
            &io.repetition_penalty_input_node_id,
            io.repetition_penalty_input_ordinal,
            penalty,
        ),
    ] {
        for index in 0..rows.len() {
            result.push(EagerCoreInputUpload {
                node_id: node,
                input_ordinal: ordinal,
                participant_index: index,
                logical_offset_bytes: 0,
                layout,
            });
        }
    }
    Ok(result)
}

pub(super) fn readbacks(
    io: &VNextIoBinding,
    rows: usize,
    output: VNextProductOutputMode,
) -> std::result::Result<Vec<EagerCoreReadback<'_>>, U> {
    let (node_id, resource_id, logical_offset_bytes, layout) = product_readback_binding(io, output);
    let mut result = Vec::new();
    result.try_reserve_exact(rows).map_err(|_| U::Capacity)?;
    for index in 0..rows {
        result.push(EagerCoreReadback {
            node_id,
            resource_id,
            participant_index: index,
            logical_offset_bytes,
            layout,
        });
    }
    Ok(result)
}
