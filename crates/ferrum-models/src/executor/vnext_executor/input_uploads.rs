//! Shared ordinary/resident construction of typed current-participant inputs.
//! This retains the existing token/mask/repetition semantics and upload order.
use super::*;

impl<R: DeviceRuntime> VNextModelExecutor<R> {
    pub(super) fn prepare_participant_uploads(
        &self,
        participants: &[VNextExecutionParticipant<'_, R>],
        kind: VNextExecutionWaveKind,
        output_mode: VNextProductOutputMode,
        token_mask_plans: &[VNextProductTokenMaskSubmissionPlan],
    ) -> Result<Vec<SubmissionWaveInputUpload>> {
        if participants.is_empty() || participants.len() != token_mask_plans.len() {
            return Err(FerrumError::backend(
                "vNext submission wave requires matching participants and token-mask decisions",
            ));
        }
        let timing_enabled = self.host_dispatch_timing_enabled();
        let phase_timing = self.metrics.wave_timing_for(kind);
        let mut uploads = {
            let _timing = self
                .metrics
                .wave_timing
                .token_upload_prepare
                .start_if(timing_enabled);
            let _phase_timing = phase_timing.token_upload_prepare.start_if(timing_enabled);
            participants
                .iter()
                .enumerate()
                .map(|(participant_index, participant)| {
                    let range = participant.span.immediate_token_range();
                    let host_range = Range {
                        start: usize::try_from(range.start).map_err(|_| {
                            FerrumError::backend(
                                "vNext token upload start exceeds host address space",
                            )
                        })?,
                        end: usize::try_from(range.end).map_err(|_| {
                            FerrumError::backend(
                                "vNext token upload end exceeds host address space",
                            )
                        })?,
                    };
                    let host_tokens =
                        participant.tokens.get(host_range.clone()).ok_or_else(|| {
                            FerrumError::backend(format!(
                                "vNext token upload range {host_range:?} exceeds host token length {}",
                                participant.tokens.len()
                            ))
                        })?;
                    let host_bytes = host_tokens
                        .iter()
                        .flat_map(|token| token.to_le_bytes())
                        .collect::<Vec<_>>();
                    let logical_offset_bytes = range
                        .start
                        .checked_mul(ElementType::U32.size_bytes())
                        .ok_or_else(|| {
                            FerrumError::backend("vNext token upload offset overflows u64")
                        })?;
                    let source_layout = HostTransferLayout::new(
                        ElementType::U32,
                        participant.span.immediate_tokens(),
                    )
                    .map_err(|error| FerrumError::backend(error.to_string()))?;
                    let participant_index = u32::try_from(participant_index).map_err(|_| {
                        FerrumError::backend("vNext token upload participant index exceeds u32")
                    })?;
                    SubmissionWaveInputUpload::new(
                        self.io.input_node_id.clone(),
                        participant_index,
                        self.io.input_ordinal,
                        logical_offset_bytes,
                        source_layout,
                        host_bytes,
                    )
                    .map_err(|error| FerrumError::backend(error.to_string()))
                })
                .collect::<Result<Vec<_>>>()
        }?;
        let token_mask_elements = u64::try_from(self.io.output_elements)
            .map_err(|_| FerrumError::backend("vNext token-mask length exceeds u64"))?;
        let token_mask_layout = HostTransferLayout::new(ElementType::U8, token_mask_elements)
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        let token_mask_uploads = participants
            .iter()
            .zip(token_mask_plans)
            .enumerate()
            .map(|(participant_index, (_, plan))| {
                if !plan.upload_required {
                    return Ok(None);
                }
                SubmissionWaveInputUpload::new(
                    self.io.token_mask_input_node_id.clone(),
                    u32::try_from(participant_index).map_err(|_| {
                        FerrumError::backend(
                            "vNext token-mask upload participant index exceeds u32",
                        )
                    })?,
                    self.io.token_mask_input_ordinal,
                    0,
                    token_mask_layout,
                    plan.content.normalized(),
                )
                .map(Some)
                .map_err(|error| FerrumError::backend(error.to_string()))
            })
            .collect::<Result<Vec<_>>>();
        uploads.extend(token_mask_uploads?.into_iter().flatten());
        let repetition_uploads = (|| -> Result<_> {
            let repetition_capacity = u64::try_from(self.io.repetition_capacity)
                .map_err(|_| FerrumError::backend("vNext repetition capacity exceeds u64"))?;
            let repetition_token_id_layout =
                HostTransferLayout::new(ElementType::U32, repetition_capacity)
                    .map_err(|error| FerrumError::backend(error.to_string()))?;
            let repetition_offset_layout = HostTransferLayout::new(ElementType::U32, 2)
                .map_err(|error| FerrumError::backend(error.to_string()))?;
            let repetition_penalty_layout = HostTransferLayout::new(ElementType::F32, 1)
                .map_err(|error| FerrumError::backend(error.to_string()))?;
            let mut token_id_uploads = Vec::with_capacity(participants.len());
            let mut offset_uploads = Vec::with_capacity(participants.len());
            let mut penalty_uploads = Vec::with_capacity(participants.len());
            for (participant_index, participant) in participants.iter().enumerate() {
                let repetition =
                    product_repetition_input(participant.output_role.logits_policy(), output_mode);
                if !repetition.penalty.is_finite() || repetition.penalty <= 0.0 {
                    return Err(FerrumError::backend(
                        "vNext sparse repetition penalty must be finite and positive",
                    ));
                }
                if repetition.token_ids.len() > self.io.repetition_capacity {
                    return Err(FerrumError::backend(format!(
                        "vNext sparse repetition input contains {} ids, capacity is {}",
                        repetition.token_ids.len(),
                        self.io.repetition_capacity
                    )));
                }
                if repetition.token_ids.iter().any(|token| {
                    usize::try_from(*token).map_or(true, |token| token >= self.io.output_elements)
                }) {
                    return Err(FerrumError::backend(
                        "vNext sparse repetition input contains an out-of-vocabulary token",
                    ));
                }
                let participant_index = u32::try_from(participant_index).map_err(|_| {
                    FerrumError::backend("vNext repetition participant index exceeds u32")
                })?;
                let repetition_count = u32::try_from(repetition.token_ids.len()).map_err(|_| {
                    FerrumError::backend("vNext repetition token count exceeds u32")
                })?;
                if repetition_count != 0 {
                    token_id_uploads.push(
                        SubmissionWaveInputUpload::new(
                            self.io.repetition_token_ids_input_node_id.clone(),
                            participant_index,
                            self.io.repetition_token_ids_input_ordinal,
                            0,
                            repetition_token_id_layout,
                            padded_repetition_token_id_bytes(
                                repetition.token_ids,
                                self.io.repetition_capacity,
                            )?,
                        )
                        .map_err(|error| FerrumError::backend(error.to_string()))?,
                    );
                }
                offset_uploads.push(
                    SubmissionWaveInputUpload::new(
                        self.io.repetition_offsets_input_node_id.clone(),
                        participant_index,
                        self.io.repetition_offsets_input_ordinal,
                        0,
                        repetition_offset_layout,
                        [0_u32, repetition_count]
                            .into_iter()
                            .flat_map(u32::to_le_bytes)
                            .collect(),
                    )
                    .map_err(|error| FerrumError::backend(error.to_string()))?,
                );
                penalty_uploads.push(
                    SubmissionWaveInputUpload::new(
                        self.io.repetition_penalty_input_node_id.clone(),
                        participant_index,
                        self.io.repetition_penalty_input_ordinal,
                        0,
                        repetition_penalty_layout,
                        repetition.penalty.to_le_bytes().to_vec(),
                    )
                    .map_err(|error| FerrumError::backend(error.to_string()))?,
                );
            }
            Ok((token_id_uploads, offset_uploads, penalty_uploads))
        })();
        let (token_id_uploads, offset_uploads, penalty_uploads) = repetition_uploads?;
        uploads.extend(token_id_uploads);
        uploads.extend(offset_uploads);
        uploads.extend(penalty_uploads);
        let uploaded_bytes = uploads.iter().fold(0_u64, |total, upload| {
            total.saturating_add(upload.source_layout().byte_len().unwrap_or(0))
        });
        self.metrics
            .uploaded_bytes
            .fetch_add(uploaded_bytes, Ordering::Relaxed);

        Ok(uploads)
    }
}
