use super::*;

impl<R: DeviceRuntime> VNextModelExecutor<R> {
    /// Called with the actual terminal readback receipt before normal step
    /// retirement. The collector consumes it only after retirement succeeds.
    pub(in crate::executor::vnext_executor) fn record_teacher_completion(
        &self,
        participants: &[VNextExecutionParticipant<'_, R>],
        kind: VNextExecutionWaveKind,
        mode: VNextProductOutputMode,
        receipt: &CompletionReadbackBatchReceipt,
    ) -> Result<()> {
        if !self.teacher_capture_active.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut pending = self.teacher_wave_capture.lock();
        let pending = pending.as_mut().ok_or_else(|| {
            FerrumError::internal("teacher capture observed an unplanned physical wave")
        })?;
        if pending.observed.is_some()
            || pending.kind != kind
            || participants.len() != pending.expected.len()
            || !matches!(mode, VNextProductOutputMode::FullLogits)
        {
            return Err(FerrumError::internal(
                "teacher physical wave count, kind, output policy or terminal receipt differs from its exact request",
            ));
        }
        let mut seen = BTreeSet::new();
        let mut observed = Vec::with_capacity(participants.len());
        for (participant_index, participant) in participants.iter().enumerate() {
            let request_id = participant.sequence.request_id();
            let binding = participant.sequence.active_binding.as_ref();
            let nodes = receipt.completion().submission().batch_identity().nodes();
            if nodes.is_empty() {
                return Err(FerrumError::internal(
                    "teacher completion has no physical node identities",
                ));
            }
            for node in nodes {
                if node.participants().len() != participants.len() {
                    return Err(FerrumError::internal(
                        "teacher physical node does not contain the complete owner batch",
                    ));
                }
                let receipt_identity = node.participants()[participant_index].identity().parts();
                if &receipt_identity.run_id != binding.run_id()
                    || &receipt_identity.request_id != binding.request_id()
                    || receipt_identity.active_sequence_fingerprint.as_deref()
                        != Some(binding.fingerprint())
                {
                    return Err(FerrumError::internal(
                        "teacher completion receipt participant differs from its real active sequence binding",
                    ));
                }
            }
            let expected = pending
                .expected
                .iter()
                .find(|expected| &expected.request_id == request_id)
                .ok_or_else(|| {
                    FerrumError::internal("teacher physical wave has an unexpected owner")
                })?;
            if !seen.insert(expected.owner_id.as_str()) {
                return Err(FerrumError::internal(
                    "teacher physical wave repeats an owner",
                ));
            }
            let range = participant.span.immediate_token_range();
            let start = usize::try_from(range.start)
                .map_err(|_| FerrumError::internal("teacher immediate start exceeds usize"))?;
            let end = usize::try_from(range.end)
                .map_err(|_| FerrumError::internal("teacher immediate end exceeds usize"))?;
            let history = participant.tokens.get(..end).ok_or_else(|| {
                FerrumError::internal("teacher physical span exceeds its real token history")
            })?;
            if start != expected.immediate_start
                || end <= start
                || end > expected.history.len()
                || (kind == VNextExecutionWaveKind::Decode && end != expected.history.len())
                || history != &expected.history[..end]
                || expected
                    .cache_id
                    .as_ref()
                    .is_some_and(|cache| cache != &participant.sequence.cache_id)
            {
                return Err(FerrumError::internal(
                    "teacher physical participant changed owner, cache or canonical token history",
                ));
            }
            observed.push(VNextTeacherWaveParticipant {
                owner_id: expected.owner_id.clone(),
                request_id: request_id.to_string(),
                participant_index,
                cache_id: participant.sequence.cache_id.clone(),
                history_tokens: history.len(),
                history_sha256: vnext_teacher_token_digest(history),
                immediate_start: start,
                immediate_end: end,
            });
        }
        let evidence = VNextTeacherWaveEvidence {
            wave_index: pending.wave_index,
            kind: kind.as_str().to_owned(),
            participant_count: observed.len(),
            completion_fingerprint: receipt.completion().fingerprint().to_owned(),
            receipt_fingerprint: receipt.fingerprint().to_owned(),
            completion_receipt: None,
            readbacks: receipt
                .dispositions()
                .iter()
                .map(|disposition| {
                    let CompletionReadbackDisposition::Succeeded(output) = disposition else {
                        return Err(FerrumError::backend("teacher readback did not succeed"));
                    };
                    Ok(VNextTeacherReadbackEvidence {
                        participant_index: output.request().participant_index() as usize,
                        request: serde_json::to_value(output.request()).map_err(|error| {
                            FerrumError::internal(format!(
                                "serialize teacher readback request: {error}"
                            ))
                        })?,
                        byte_count: output.bytes().len(),
                        sha256: output.sha256().to_owned(),
                        raw_artifact: None,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            participants: observed,
        };
        let raw_readbacks = receipt
            .dispositions()
            .iter()
            .map(|disposition| {
                let CompletionReadbackDisposition::Succeeded(output) = disposition else {
                    return Err(FerrumError::backend("teacher readback did not succeed"));
                };
                Ok(output.bytes().to_vec())
            })
            .collect::<Result<Vec<_>>>()?;
        pending.observed = Some(TeacherObservedWave {
            evidence,
            raw_readbacks,
            completion_receipt: serde_json::to_value(receipt.completion()).map_err(|error| {
                FerrumError::internal(format!("serialize teacher completion receipt: {error}"))
            })?,
        });
        Ok(())
    }

    pub(super) fn begin_teacher_wave(
        &self,
        wave_index: usize,
        kind: VNextExecutionWaveKind,
        expected: Vec<TeacherExpectedParticipant>,
    ) -> Result<()> {
        let mut pending = self.teacher_wave_capture.lock();
        if pending.is_some() {
            return Err(FerrumError::internal(
                "teacher wave was not consumed before the next submission",
            ));
        }
        *pending = Some(TeacherPendingWave {
            wave_index,
            kind,
            expected,
            observed: None,
        });
        Ok(())
    }

    pub(super) fn finish_teacher_wave(&self) -> Result<TeacherObservedWave> {
        self.teacher_wave_capture
            .lock()
            .take()
            .and_then(|pending| pending.observed)
            .ok_or_else(|| {
                FerrumError::internal(
                    "teacher execution returned without exactly one observed physical wave",
                )
            })
    }
}
