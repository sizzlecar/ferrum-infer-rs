//! Source5 consumes private actual events, never the public diagnostic DTOs.
use super::super::structured::capture_error;
use super::*;

impl CalibrationSession {
    pub fn structured_prefix_release_generated_v5(&self) -> Result<Option<usize>> {
        if !self.prefix_source5 {
            return Ok(None);
        }
        self.structured_group_v2
            .as_ref()
            .ok_or_else(|| invalid("source5 collector is closed"))?
            .prefix_release_generated()
            .map_err(capture_error)?
            .map(|n| usize::try_from(n).map_err(|_| invalid("source5 prefix bound overflow")))
            .transpose()
    }
    pub(in crate::continuous_engine::inner::calibration) async fn add_declared_prefix_source_request_v5(
        &mut self,
        request: ferrum_types::InferenceRequest,
        context: InferenceRequestContext,
        contract: Arc<OutputProjectionContract>,
    ) -> Result<CreditedOutputSession> {
        let declaration = self
            .structured_group_v2
            .as_ref()
            .filter(|g| g.collecting())
            .ok_or_else(|| invalid("source5 group is not collecting"))?
            .prefix_declaration_for_next_request()
            .map_err(capture_error)?;
        let Some((release, slot)) = declaration else {
            return self.add_request_inner(request, context, contract).await;
        };
        let declaration = CalibrationPrefixTokensV1 {
            tokenizer_policy_sha256: slot.tokenizer_policy_sha256,
            token_ids: slot.token_ids,
            release_generated: usize::try_from(release)
                .map_err(|_| invalid("source5 release bound overflow"))?,
        };
        self.install_prefix_request(request, context, contract, declaration, true)
            .await
    }

    pub(in crate::continuous_engine::inner::calibration) fn offer_prefix_source_wave_v5(
        &mut self,
    ) -> Result<()> {
        let offered = self
            .prefix_preparation
            .as_ref()
            .and_then(|r| r.pending_offer.as_ref())
            .ok_or_else(|| invalid("source5 preparation lost its actual before snapshot"))?;
        let group = self
            .structured_group_v2
            .as_mut()
            .ok_or_else(|| invalid("source5 collector missing"))?;
        let result = group.offer_preparation(offered);
        if let Err(error) = &result {
            group.invalidate(error.to_string());
        }
        result.map_err(capture_error)
    }

    pub(super) fn record_prefix_source_wave_v5(&mut self) {
        let Some(run) = &mut self.prefix_preparation else {
            return;
        };
        let Some(evidence) = run.pending_wave.take() else {
            return;
        };
        let Some(group) = &mut self.structured_group_v2 else {
            run.failure
                .get_or_insert_with(|| "source5 collector missing at actual settlement".into());
            return;
        };
        if group.prefix_pending() {
            // The tuple constructor is private to this actual recorder module.
            if let Err(error) = group.complete_preparation(&CapturedPrefixWaveV5(evidence)) {
                run.failure.get_or_insert_with(|| error.to_string());
                group.invalidate(error.to_string());
            }
        } else if let Some(error) = evidence.error.or(evidence.chain_error) {
            // Ordinary suffix events retain the original numeric path, but an
            // incomplete real lifecycle cannot later certify a source5 model.
            group.invalidate(error);
        }
    }

    /// The declared cohort crosses one common preparation frontier. Actor
    /// readiness is observed before removing any slot's private capability.
    pub fn advance_structured_prefix_release_v5(&mut self) -> Result<PrefixReleaseProgressV5> {
        if !self.prefix_source5 {
            return Ok(PrefixReleaseProgressV5::Inactive);
        }
        let group = self
            .structured_group_v2
            .as_ref()
            .ok_or_else(|| invalid("source5 group failed or closed"))?;
        if !group.preparing_prefix().map_err(capture_error)? {
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
                "source5 release requires every original attempt settled",
            ));
        }
        let ids = group.prefix_release_order().map_err(capture_error)?;
        let sequences = self.engine.inner.sequences.read();
        for id in &ids {
            let record = run
                .records
                .get(id)
                .ok_or_else(|| invalid("source5 prefix request was not installed"))?;
            let sequence = sequences
                .get(id)
                .ok_or_else(|| invalid("source5 prefix owner ended before release"))?;
            if sequence.generated_tokens.len() < record.declaration.release_generated {
                return Ok(PrefixReleaseProgressV5::Preparing);
            }
            if sequence.generated_tokens.len() != record.declaration.release_generated {
                return Err(invalid(
                    "source5 exceeded its declared preparation frontier",
                ));
            }
        }
        for id in &ids {
            let sequence = &sequences[id];
            let output = sequence
                .credited_output
                .as_ref()
                .ok_or_else(|| invalid("source5 credited output missing"))?;
            if output.port.applied_ordinal_while_ready() != Some(output.accepted_ordinal) {
                return Ok(PrefixReleaseProgressV5::AwaitingCredit);
            }
        }
        drop(sequences);
        let mut receipts = Vec::with_capacity(ids.len());
        // No await/admission/dispatch is possible between the common readiness
        // check and these private release+write operations.
        for id in ids {
            let receipt = match self.release_prefix_preparation_inner(&id) {
                Ok(receipt) => receipt,
                Err(error) => {
                    self.structured_group_v2
                        .as_mut()
                        .unwrap()
                        .invalidate(error.to_string());
                    self.prefix_preparation
                        .as_mut()
                        .unwrap()
                        .failure
                        .get_or_insert_with(|| error.to_string());
                    return Err(error);
                }
            };
            let captured = CapturedPrefixReleaseV5(receipt);
            let group = self.structured_group_v2.as_mut().unwrap();
            if let Err(error) = group.release_preparation(&captured) {
                group.invalidate(error.to_string());
                return Err(capture_error(error));
            }
            receipts.push(captured.0);
        }
        Ok(PrefixReleaseProgressV5::Released { receipts })
    }

    pub(in crate::continuous_engine::inner::calibration) fn finish_prefix_source_cohort_v5(
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
                "source5 cohort has incomplete preparation/release/full Length",
            ));
        }
        // The permanent source5 marker survives this per-cohort reset. The
        // group additionally verifies every Length row and final FIFO barrier.
        self.prefix_preparation = None;
        Ok(())
    }
}
