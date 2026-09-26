//! Token I/O and selection cost declarations from their actual CUDA selectors.
use super::*;
use ferrum_interfaces::vnext::{
    DeviceCommandPhase, OperationCostCommand, OperationCostRoute, OperationCostRouteRequest,
    PhysicalWeightLayout,
};

pub(super) fn compute(
    operation: &'static str,
    form: DeviceBatchingForm,
    participants: u32,
    tokens: u64,
    dispatches: u64,
) -> Result<OperationCostCommand, VNextError> {
    OperationCostCommand::new(
        operation,
        DeviceCommandPhase::Compute,
        form,
        0,
        participants,
        tokens,
        dispatches,
        0,
    )
}

pub(super) fn argmax_command(
    participants: u32,
    vocabulary: i32,
) -> Result<OperationCostCommand, VNextError> {
    compute(
        "vnext_last_token_masked_argmax",
        if participants == 1 {
            DeviceBatchingForm::Scalar
        } else {
            DeviceBatchingForm::ParticipantLoop
        },
        participants,
        u64::from(participants),
        u64::from(participants) * argmax_dispatches(vocabulary),
    )
}

pub(super) fn apply(
    command: CudaDeviceCommand,
    work: OperationCostCommand,
) -> Result<CudaDeviceCommand, CudaDeviceRuntimeError> {
    command
        .with_work_attribution(
            work.batching(),
            work.participant_count(),
            work.token_count(),
            work.compute_dispatch_count(),
            work.transfer_command_count(),
        )
        .map(|command| command.with_statistical_evidence(work.statistical_evidence().cloned()))
}

pub(super) fn argmax(
    request: OperationCostRouteRequest<'_>,
    precision: ArgmaxPrecision,
    capture: ferrum_types::SloStructuredCostCapture,
) -> Result<Option<OperationCostRoute>, VNextError> {
    let checked = || -> Result<OperationCostCommand, String> {
        if request.operation_id().as_str() != precision.operation() {
            return Err("CUDA argmax cost operation mismatch".into());
        }
        let get = |ordinal| binding(request.bindings(), ResolvedValueRole::Input, ordinal);
        let vocabulary = unsigned_attribute(request.attributes(), "vocab_size")?;
        let repetition_capacity = validate_masked_argmax_signature(
            get(0)?,
            get(1)?,
            get(2)?,
            get(3)?,
            get(4)?,
            binding(request.bindings(), ResolvedValueRole::Output, 0)?,
            vocabulary,
            precision.element(),
        )?;
        masked_argmax_scratch_stride(vocabulary, precision.element())?
            .checked_mul(request.rows().len() as u64)
            .ok_or("CUDA argmax scratch extent overflows")?;
        let vocabulary =
            i32::try_from(vocabulary).map_err(|_| "CUDA argmax vocabulary exceeds i32")?;
        let participants = u32::try_from(request.rows().len())
            .map_err(|_| "CUDA argmax participant count exceeds u32")?;
        let command =
            argmax_command(participants, vocabulary).map_err(|error| error.to_string())?;
        match selection::selected::evidence(
            precision,
            participants,
            (0..participants).map(|_| (vocabulary, repetition_capacity)),
            capture,
        ) {
            Some(evidence) => command
                .with_statistical_evidence(evidence)
                .map_err(|error| format!("{error:?}")),
            None => Ok(command),
        }
    };
    let command = checked().map_err(|reason| VNextError::InvalidExecutionPlan { reason })?;
    Ok(Some(OperationCostRoute::new(vec![command])?))
}

pub(super) fn embedding(
    request: OperationCostRouteRequest<'_>,
    precision: TokenPrecision,
    capture: ferrum_types::SloStructuredCostCapture,
) -> Result<Option<OperationCostRoute>, VNextError> {
    let checked = || -> Result<OperationCostCommand, String> {
        if request.operation_id().as_str() != precision.embedding_operation() {
            return Err("CUDA embedding cost operation mismatch".into());
        }
        let hidden = unsigned_attribute(request.attributes(), "hidden_size")?;
        let vocabulary = unsigned_attribute(request.attributes(), "vocab_size")?;
        let table = binding(request.bindings(), ResolvedValueRole::Input, 1)?;
        validate_signature(
            binding(request.bindings(), ResolvedValueRole::Input, 0)?,
            table,
            binding(request.bindings(), ResolvedValueRole::Output, 0)?,
            vocabulary,
            hidden,
            precision.element(),
        )?;
        let native = precision == TokenPrecision::F32
            || table.weight().is_some_and(|weight| {
                !matches!(weight.physical_layout(), PhysicalWeightLayout::Dense { .. })
            });
        let (operation, dispatches, selected) = if native {
            let weight = table
                .weight()
                .ok_or("CUDA embedding has no physical weight metadata")?;
            let parts = native_blocks::weights::matrix_parts(weight, &[vocabulary, hidden])?;
            let [part] = parts.as_slice() else {
                return Err("CUDA native embedding requires one complete vocabulary table".into());
            };
            (
                "vnext_native_embedding",
                native_io::embedding_dispatches(
                    part,
                    request.rows().iter().map(|row| row.count.get()),
                )?,
                if capture == ferrum_types::SloStructuredCostCapture::Disabled {
                    None
                } else {
                    native_blocks::embedding::selected(
                        part,
                        request.rows().iter().map(|row| row.count.get()),
                        request.immediate_tokens(),
                        precision.element(),
                        native_blocks::hadamard::workspace_bytes_per_token(request.bindings())?
                            .checked_mul(request.immediate_tokens())
                            .ok_or("embedding scratch overflows")?,
                        capture,
                    )
                },
            )
        } else {
            i32::try_from(hidden).map_err(|_| "CUDA embedding hidden size exceeds i32")?;
            u32::try_from(vocabulary).map_err(|_| "CUDA embedding vocabulary exceeds u32")?;
            u32::try_from(hidden.div_ceil(u64::from(THREADS_PER_BLOCK)))
                .map_err(|_| "CUDA embedding launch grid exceeds u32")?;
            (
                "vnext_token_embedding",
                request
                    .rows()
                    .iter()
                    .try_fold(0_u64, |total, row| {
                        total.checked_add(row.count.get().div_ceil(MAXIMUM_TOKENS_PER_LAUNCH))
                    })
                    .ok_or("CUDA embedding dispatch count overflows")?,
                super::embedding::selected_dense(
                    request
                        .rows()
                        .iter()
                        .map(|row| (vocabulary, hidden, row.count.get())),
                    request.immediate_tokens(),
                    capture,
                ),
            )
        };
        let participants = u32::try_from(request.rows().len())
            .map_err(|_| "CUDA embedding participant count exceeds u32")?;
        let command = compute(
            operation,
            DeviceBatchingForm::ParticipantLoop,
            participants,
            request.immediate_tokens(),
            dispatches,
        )
        .map_err(|error| error.to_string())?;
        Ok(match selected {
            Some(selected) => command
                .clone()
                .with_statistical_evidence(selected)
                .unwrap_or(command),
            None => command,
        })
    };
    Ok(Some(OperationCostRoute::new(vec![checked().map_err(
        |reason| VNextError::InvalidExecutionPlan { reason },
    )?])?))
}
