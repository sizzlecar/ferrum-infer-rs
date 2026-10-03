//! The source8 collector owns declaration and lifecycle; this module alone
//! binds preparation to real sequence state and original output credits.
use super::super::structured::capture_error;
use super::*;

impl CalibrationSession {
    /// A real maintenance observation shares the original FIFO but contributes
    /// no inference call or numerical member. Consume only its exact ACK receipt.
    pub(in crate::continuous_engine::inner::calibration) fn accept_prepared_owner_native_restore(
        &mut self,
        receipt: &crate::continuous_engine::inner::calibration::startup::AcknowledgedProbePrefixRestore,
    ) -> Result<()> {
        let run = self
            .prefix_preparation
            .as_mut()
            .ok_or_else(|| invalid("native restore has no original prefix preparation"))?;
        if !self.prefix_source8
            || self.pending.is_some()
            || self.indeterminate
            || run.failure.is_some()
            || run.pending_offer.is_some()
            || run.pending_wave.is_some()
            || run.records.values().any(|record| record.last_call != 0)
            || receipt
                .maintenance_fifo()
                .is_some_and(|fifo| Some(fifo) != run.last_fifo.checked_add(1))
        {
            return Err(invalid(
                "native restore lost original preparation FIFO order",
            ));
        }
        self.prepared_owner_capture
            .as_mut()
            .ok_or_else(|| invalid("native restore ACK has no original source8 collector"))?
            .native_prefix_restored(receipt)?;
        if let Some(fifo) = receipt.maintenance_fifo() {
            run.last_fifo = fifo;
        }
        Ok(())
    }

    pub(in crate::continuous_engine::inner::calibration) async fn add_declared_prepared_owner_request(
        &mut self,
        request: ferrum_types::InferenceRequest,
        context: InferenceRequestContext,
        contract: Arc<OutputProjectionContract>,
    ) -> Result<CreditedOutputSession> {
        let declaration = self
            .prepared_owner_capture
            .as_ref()
            .filter(|c| c.collecting())
            .ok_or_else(|| invalid("source8 collector is not collecting"))?
            .prefix_declaration_for_next_request();
        let Some((release, slot)) = declaration else {
            return self.add_request_inner(request, context, contract).await;
        };
        let declaration = CalibrationPrefixTokensV1 {
            tokenizer_policy_sha256: slot.tokenizer_policy_sha256,
            token_ids: slot.token_ids,
            release_generated: usize::try_from(release)
                .map_err(|_| invalid("source8 prefix bound overflow"))?,
        };
        self.install_prefix_request(
            request,
            context,
            contract,
            declaration,
            PrefixSource::Source8,
        )
        .await
    }

    pub(in crate::continuous_engine::inner::calibration) fn offer_prepared_owner_prefix(
        &mut self,
    ) -> Result<()> {
        let offered = self
            .prefix_preparation
            .as_ref()
            .and_then(|r| r.pending_offer.as_ref())
            .ok_or_else(|| invalid("source8 preparation lost its original before snapshot"))?;
        let collector = self
            .prepared_owner_capture
            .as_mut()
            .ok_or_else(|| invalid("source8 collector missing"))?;
        collector
            .offer_preparation(offered)
            .map(|_| ())
            .map_err(capture_error)
    }

    pub(in crate::continuous_engine::inner::calibration) fn consume_prepared_owner_prefix_wave(
        &mut self,
        preparing: bool,
    ) -> Result<()> {
        let Some(run) = &mut self.prefix_preparation else {
            if preparing {
                return Err(invalid("source8 preparation run disappeared"));
            }
            return Ok(());
        };
        let evidence = run
            .pending_wave
            .take()
            .ok_or_else(|| invalid("source8 original prefix settlement absent"))?;
        if preparing {
            self.prepared_owner_capture
                .as_mut()
                .ok_or_else(|| invalid("source8 collector missing"))?
                .complete_preparation(&evidence)
                .map_err(capture_error)?;
        } else if let Some(error) = evidence.0.error.or(evidence.0.chain_error) {
            return Err(invalid(error));
        }
        Ok(())
    }

    pub(in crate::continuous_engine::inner::calibration) fn record_prepared_owner_no_submission(
        &mut self,
        report: &CalibrationWaveReport,
    ) -> Result<()> {
        let Some(run) = &mut self.prefix_preparation else {
            return Ok(());
        };
        let proof = report
            .no_submission_proof()
            .ok_or_else(|| invalid("source8 original non-submission proof absent"))?;
        let before = run
            .pending_offer
            .as_ref()
            .ok_or_else(|| invalid("source8 non-submission has no original offer"))?;
        if run.failure.is_some()
            || run.pending_wave.is_some()
            || Some(proof.fifo()) != run.last_fifo.checked_add(1)
            || proof.call_id() <= run.last_call
        {
            return Err(invalid(
                "source8 non-submission lost original call/FIFO order",
            ));
        }
        let sequences = self.engine.inner.sequences.read();
        for row in before {
            let sequence = sequences
                .get(&row.before.request_id)
                .ok_or_else(|| invalid("source8 non-submitted owner disappeared"))?;
            let after = PrefixFrontierV1::capture(sequence)?;
            if after.owner_incarnation != row.before.owner_incarnation
                || after.work_generation != row.before.work_generation
                || after.generated_tokens != row.before.generated_tokens
                || after.kv_tokens != row.before.kv_tokens
                || after.model_cache_id != row.before.model_cache_id
                || after.pending_utf8 != row.before.pending_utf8
                || after.output_accepted_ordinal != row.before.output_accepted_ordinal
                || sequence
                    .calibration_prefix
                    .as_ref()
                    .is_some_and(|p| p.pending_commit.is_some())
            {
                return Err(invalid(
                    "source8 non-submission changed the original owner frontier",
                ));
            }
            let record = run
                .records
                .get_mut(&row.before.request_id)
                .ok_or_else(|| invalid("source8 non-submitted request was not declared"))?;
            record.last_call = proof.call_id();
            record.last_fifo = proof.fifo();
        }
        run.last_call = proof.call_id();
        run.last_fifo = proof.fifo();
        run.pending_offer = None;
        Ok(())
    }

    pub(in crate::continuous_engine::inner::calibration) fn prepared_owner_prefix_release_generated(
        &self,
    ) -> Result<Option<usize>> {
        self.prepared_owner_capture
            .as_ref()
            .ok_or_else(|| invalid("source8 collector absent"))?
            .prefix_release_generated()
            .map(|n| usize::try_from(n).map_err(|_| invalid("source8 release overflow")))
            .transpose()
    }

    pub(in crate::continuous_engine::inner::calibration) fn advance_prepared_owner_prefix_release(
        &mut self,
    ) -> Result<PrefixReleaseProgressV5> {
        let collector = self
            .prepared_owner_capture
            .as_ref()
            .filter(|c| c.collecting())
            .ok_or_else(|| invalid("source8 collector failed or closed"))?;
        if !collector.preparing_prefix() {
            return Ok(PrefixReleaseProgressV5::Inactive);
        }
        let Some(run) = &self.prefix_preparation else {
            return Ok(PrefixReleaseProgressV5::Preparing);
        };
        if run.failure.is_some()
            || run.pending_wave.is_some()
            || run.pending_offer.is_some()
            || self.pending.is_some()
            || self.indeterminate
        {
            return Err(invalid(
                "source8 release requires every original attempt settled",
            ));
        }
        let ids = collector.prefix_release_order();
        if ids.is_empty() {
            return Err(invalid("source8 release has no declared original owners"));
        }
        let sequences = self.engine.inner.sequences.read();
        for id in &ids {
            let record = run
                .records
                .get(id)
                .ok_or_else(|| invalid("source8 prefix owner not installed"))?;
            let sequence = sequences
                .get(id)
                .ok_or_else(|| invalid("source8 owner ended before release"))?;
            if sequence.generated_tokens.len() < record.declaration.release_generated {
                return Ok(PrefixReleaseProgressV5::Preparing);
            }
            if sequence.generated_tokens.len() != record.declaration.release_generated {
                return Err(invalid("source8 exceeded declared preparation frontier"));
            }
        }
        for id in &ids {
            let output = sequences[id]
                .credited_output
                .as_ref()
                .ok_or_else(|| invalid("source8 credited output missing"))?;
            if output.port.applied_ordinal_while_ready() != Some(output.accepted_ordinal) {
                return Ok(PrefixReleaseProgressV5::AwaitingCredit);
            }
        }
        drop(sequences);
        let result: Result<PrefixReleaseProgressV5> = (|| {
            let mut receipts = Vec::with_capacity(ids.len());
            for id in ids {
                let captured = CapturedPrefixReleaseV5(self.release_prefix_preparation_inner(&id)?);
                self.prepared_owner_capture
                    .as_mut()
                    .unwrap()
                    .released(&captured)
                    .map_err(capture_error)?;
                receipts.push(captured.0);
            }
            Ok(PrefixReleaseProgressV5::Released { receipts })
        })();
        if let Err(error) = &result {
            self.prepared_owner_capture
                .as_mut()
                .unwrap()
                .invalidate(error.to_string());
            self.prefix_preparation
                .as_mut()
                .unwrap()
                .failure
                .get_or_insert_with(|| error.to_string());
        }
        result
    }

    pub(in crate::continuous_engine::inner::calibration) fn finish_prepared_owner_prefix(
        &mut self,
    ) -> Result<()> {
        let Some(run) = &self.prefix_preparation else {
            return Ok(());
        };
        if run.failure.is_some()
            || run.pending_wave.is_some()
            || run.pending_offer.is_some()
            || run.records.is_empty()
            || run
                .records
                .values()
                .any(|r| r.released.is_none() || !r.completed_length)
        {
            return Err(invalid(
                "source8 cohort has incomplete preparation, release or original terminal",
            ));
        }
        self.prefix_preparation = None;
        Ok(())
    }

    pub(in crate::continuous_engine::inner::calibration) fn fail_prepared_owner_prefix(
        &mut self,
        reason: &str,
    ) {
        if let Some(run) = &mut self.prefix_preparation {
            run.failure.get_or_insert_with(|| reason.to_owned());
        }
    }
}
