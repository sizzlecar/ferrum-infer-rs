//! Last-row projection including gather/scatter. Layout evidence comes from
//! the compiled resource contract; no backing claim or device work is created.
use super::half_head::LastTokenProjectionKind;
use super::*;
use ferrum_interfaces::vnext::{
    DeviceCommandPhase, OperationCostCommand, OperationCostRoute, OperationCostRouteRequest,
};

pub(super) fn packed_eligible(participants: usize, layout: LastTokenPackedScratchLayout) -> bool {
    participants > 1
        && layout.input_row_bytes % METAL_BLIT_ALIGNMENT_BYTES == 0
        && layout.output_row_bytes % METAL_BLIT_ALIGNMENT_BYTES == 0
}

pub(super) fn route(
    request: OperationCostRouteRequest<'_>,
    operation_id: &str,
    dtype: ElementType,
    projection: &LastTokenProjection,
) -> Result<Option<OperationCostRoute>, VNextError> {
    if request.operation_id().as_str() != operation_id {
        return Ok(None);
    }
    let policy = projection.kind();
    let calculate = || -> Result<Option<OperationCostCommand>, String> {
        let hidden = unsigned_attribute(request.attributes(), "hidden_size")?;
        let outputs = unsigned_attribute(request.attributes(), "out_features")?;
        validate_last_token_bindings(request.bindings(), hidden, outputs, dtype)?;
        let Some((metadata, layout)) = cost_route::plain_weight(
            binding(request.bindings(), ResolvedValueRole::Input, 1)?,
            false,
        )?
        else {
            return Ok(None);
        };
        let part = prepare_leaf_encoding(&metadata, &layout, outputs, hidden, 1, 0)?;
        policy.validate_part(part)?;
        let participants = request.rows().len();
        let scratch =
            LastTokenPackedScratchLayout::new(participants as u64, hidden, outputs, dtype)?;
        let input_packed = request
            .binding_uses_packed_batch_coordinates(ResolvedValueRole::Input, 0)
            .map_err(|e| e.to_string())?;
        let shared_input = input_packed
            && participants > 1
            && request.rows().iter().all(|row| row.count.get() == 1);
        // Validate the same token-stride ABI used by contiguous_token_region.
        let input = binding(request.bindings(), ResolvedValueRole::Input, 0)?;
        let [input_component] = input.storage().components() else {
            return Ok(None);
        };
        let canonical = input.tensor().dimensions()[0];
        if input_component.element_type() != dtype
            || input_component.offset_bytes() != 0
            || input_component.length_bytes() % canonical != 0
            || input_component.length_bytes() / canonical != scratch.input_row_bytes
        {
            return Err("last-token cost input differs from canonical token stride".into());
        }
        let output = binding(request.bindings(), ResolvedValueRole::Output, 0)?;
        let [output_component] = output.storage().components() else {
            return Ok(None);
        };
        if output_component.element_type() != dtype {
            return Err("last-token cost output differs from activation dtype".into());
        }
        let packed = packed_eligible(participants, scratch);
        if packed {
            // The real encoder falls back if gather/scatter regions fail its
            // length/alignment test. Without a proof of that choice, decline.
            let aligned = |role, ordinal| -> Result<bool, String> {
                Ok(request
                    .binding_contiguous_base_alignment(role, ordinal)
                    .map_err(|e| e.to_string())?
                    .is_some_and(|alignment| alignment.get() % METAL_BLIT_ALIGNMENT_BYTES == 0))
            };
            if (!shared_input && !aligned(ResolvedValueRole::Input, 0)?)
                || !aligned(ResolvedValueRole::Output, 0)?
                || output_component.offset_bytes() % METAL_BLIT_ALIGNMENT_BYTES != 0
                || output_component.length_bytes() != scratch.output_row_bytes
            {
                return Ok(None);
            }
        }
        // Both source and packed-token last-row offsets are bounded checked
        // arithmetic, including the non-gathered participant-loop path.
        let mut immediate = 0_u64;
        for row in request.rows() {
            let end = if input_packed {
                immediate = immediate
                    .checked_add(row.count.get())
                    .ok_or("last-token packed token range overflows")?;
                immediate
            } else {
                row.offset
                    .checked_add(row.count.get())
                    .ok_or("last-token source token range overflows")?
            };
            end.checked_mul(scratch.input_row_bytes)
                .ok_or("last-token source byte range overflows")?;
        }
        let mut command = command(
            policy,
            dtype,
            part,
            hidden,
            outputs,
            participants,
            request.immediate_tokens(),
            packed,
            shared_input,
        )?;
        let rows = if packed { participants as u64 } else { 1 };
        let launch = linear_launch_typed(
            part,
            0,
            0,
            rows,
            hidden,
            outputs,
            0,
            if packed {
                scratch.output_offset_bytes
            } else {
                0
            },
            dtype,
        )?;
        let launches = vec![launch; if packed { 1 } else { participants }];
        if let Some(evidence) = head_selected::evidence(
            projection,
            &launches,
            request.immediate_tokens(),
            packed.then_some((scratch, participants as u32, shared_input)),
        ) {
            if let Ok(checked) = command.clone().with_statistical_evidence(evidence) {
                command = checked;
            }
        }
        Ok(Some(command))
    };
    calculate()
        .map_err(invalid_plan)?
        .map(|command| OperationCostRoute::new(vec![command]))
        .transpose()
}

#[allow(clippy::too_many_arguments)]
fn command(
    policy: LastTokenProjectionKind,
    dtype: ElementType,
    part: PreparedLinearPart,
    hidden: u64,
    outputs: u64,
    participants: usize,
    tokens: u64,
    packed: bool,
    shared_input: bool,
) -> Result<OperationCostCommand, String> {
    let count = checked_u32(participants as u64, "last-token participants")?;
    if count == 0 || tokens < count as u64 {
        return Err("last-token cost has an empty participant or token set".into());
    }
    let rows = if packed { participants as u64 } else { 1 };
    let launch = linear_launch_typed(part, 0, 0, rows, hidden, outputs, 0, 0, dtype)?;
    policy.validate_numeric_launch(launch)?;
    let dispatches = policy
        .dispatch_count(launch)
        .checked_mul(if packed { 1 } else { participants as u64 })
        .ok_or("last-token cost dispatch count overflows")?;
    let transfers = if packed {
        (participants as u64)
            .checked_mul(if shared_input { 1 } else { 2 })
            .ok_or("last-token cost transfer count overflows")?
    } else {
        0
    };
    OperationCostCommand::new(
        policy.operation_label(dtype),
        DeviceCommandPhase::Compute,
        if packed {
            DeviceBatchingForm::Packed
        } else if participants == 1 {
            DeviceBatchingForm::Scalar
        } else {
            DeviceBatchingForm::ParticipantLoop
        },
        0,
        count,
        tokens,
        dispatches,
        transfers,
    )
    .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn part() -> PreparedLinearPart {
        PreparedLinearPart {
            region: 0,
            format: LinearPhysicalFormat::Q6K,
            out_features: 1024,
            output_offset: 0,
            transform: None,
        }
    }

    #[test]
    fn half_packed_decode_skips_gather_but_keeps_scatter_and_small_cohort_splits() {
        for (count, dispatches) in [(2, 1), (5, 2), (7, 2), (8, 1), (17, 1), (33, 1)] {
            let route = command(
                LastTokenProjectionKind::Half,
                ElementType::F32,
                part(),
                256,
                1024,
                count,
                count as u64,
                true,
                true,
            )
            .unwrap();
            assert_eq!(route.transfer_command_count(), count as u64);
            assert_eq!(route.compute_dispatch_count(), dispatches);
            assert_eq!(route.batching(), DeviceBatchingForm::Packed);
            let gathered = command(
                LastTokenProjectionKind::Half,
                ElementType::F32,
                part(),
                256,
                1024,
                count,
                count as u64 + 2,
                true,
                false,
            )
            .unwrap();
            assert_eq!(gathered.transfer_command_count(), 2 * count as u64);
            assert_eq!(
                gathered.compute_dispatch_count(),
                route.compute_dispatch_count()
            );
        }
    }

    #[test]
    fn half_head_checks_kernel_integer_boundaries_before_submission() {
        assert!(command(
            LastTokenProjectionKind::Half,
            ElementType::F32,
            part(),
            256,
            i32::MAX as u64 + 1,
            2,
            2,
            true,
            true
        )
        .is_err());
        assert!(command(
            LastTokenProjectionKind::Half,
            ElementType::F16,
            part(),
            256,
            1024,
            2,
            2,
            true,
            true
        )
        .is_err());
        assert!(command(
            LastTokenProjectionKind::Strict,
            ElementType::F32,
            part(),
            256,
            1024,
            0,
            0,
            false,
            false
        )
        .is_err());
    }
}

#[cfg(test)]
mod strict_regression {
    use super::*;

    fn part() -> PreparedLinearPart {
        PreparedLinearPart {
            region: 0,
            format: LinearPhysicalFormat::Q6K,
            out_features: 1024,
            output_offset: 0,
            transform: None,
        }
    }

    #[test]
    fn packed_decode_skips_gather_but_keeps_scatter_and_small_cohort_splits() {
        for (count, dispatches) in [(2, 1), (5, 1), (7, 1), (8, 2), (17, 5), (33, 1)] {
            let route = command(
                LastTokenProjectionKind::Strict,
                ElementType::F32,
                part(),
                256,
                1024,
                count,
                count as u64,
                true,
                true,
            )
            .unwrap();
            assert_eq!(route.transfer_command_count(), count as u64);
            assert_eq!(route.compute_dispatch_count(), dispatches);
            assert_eq!(route.batching(), DeviceBatchingForm::Packed);
            let gathered = command(
                LastTokenProjectionKind::Strict,
                ElementType::F32,
                part(),
                256,
                1024,
                count,
                count as u64 + 2,
                true,
                false,
            )
            .unwrap();
            assert_eq!(gathered.transfer_command_count(), 2 * count as u64);
            assert_eq!(
                gathered.compute_dispatch_count(),
                route.compute_dispatch_count()
            );
        }
    }

    #[test]
    fn scalar_and_unaligned_row_loop_have_no_gather_scatter() {
        for count in [1, 3] {
            let route = command(
                LastTokenProjectionKind::Strict,
                ElementType::F16,
                part(),
                256,
                1024,
                count,
                7,
                false,
                false,
            )
            .unwrap();
            assert_eq!(route.compute_dispatch_count(), count as u64);
            assert_eq!(route.transfer_command_count(), 0);
        }
        let layout = LastTokenPackedScratchLayout::new(3, 256, 1025, ElementType::F16).unwrap();
        assert!(!packed_eligible(3, layout));
        let layout = LastTokenPackedScratchLayout::new(3, 256, 1025, ElementType::F32).unwrap();
        assert!(packed_eligible(3, layout));
    }

    #[test]
    fn strict_head_rejects_empty_and_shader_width_overflow() {
        for (outputs, participants, tokens) in [(1024, 0, 0), (u32::MAX as u64 + 1, 2, 2)] {
            assert!(command(
                LastTokenProjectionKind::Strict,
                ElementType::F32,
                part(),
                256,
                outputs,
                participants,
                tokens,
                false,
                false
            )
            .is_err());
        }
    }
}
